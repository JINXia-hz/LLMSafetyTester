"""
weekend_hpo — 周末长期 HPO 实验编排器。

用法：
  python scripts/weekend_hpo.py probe    探活：4 个目标端点 + 生成器/判分 + embedding
  python scripts/weekend_hpo.py smoke    小规模冒烟（l1.jsonl, 2 config × 1 目标, ~10min）
  python scripts/weekend_hpo.py run      全流程：stage1 粗扫 → 自动生成 stage2 细化 → 报告
  python scripts/weekend_hpo.py stage2   仅生成并运行 stage2（stage1 已有结果时）
  python scripts/weekend_hpo.py report   打印两阶段汇总 + 建议 .env 追加行

所有 study 支持断点续跑：中断后重跑同一命令即从 trials.jsonl 恢复。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llmsec.core.config import load_targets  # noqa: E402
from llmsec.core.logging import get_logger  # noqa: E402

logger = get_logger("weekend_hpo")

STAGE1_YAML = ROOT / "experiments" / "weekend_stage1.yaml"
STAGE2_YAML = ROOT / "experiments" / "weekend_stage2.yaml"
STAGE2_TOP_K = 5


# ---------------------------------------------------------------- probe
def cmd_probe() -> int:
    """对全部端点做一次轻量 chat 探活（非流式，小 max_tokens）。"""
    from openai import OpenAI

    from llmsec.core.config import GeneratorConfig

    ok_all = True

    targets = load_targets()
    if not targets:
        logger.error("❌ .env 未配置任何目标（TARGETS 为空）")
        return 1
    # 只对 stage1 实际使用的目标硬失败；其余端点异常仅告警（不影响实验启动）
    import yaml

    used = []
    if STAGE1_YAML.exists():
        used = yaml.safe_load(STAGE1_YAML.read_text(encoding="utf-8")).get("targets") or []
    for name, tc in targets.items():
        try:
            client = OpenAI(api_key=tc.api_key or "EMPTY", base_url=tc.base_url, timeout=60)
            r = client.chat.completions.create(
                model=tc.model, messages=[{"role": "user", "content": "ping"}], max_tokens=8)
            text = (r.choices[0].message.content or "").strip()[:30] if r.choices else "(空)"
            logger.info(f"✅ 目标 {name:<38} {tc.base_url}  model={tc.model}  回复: {text!r}")
        except Exception as e:
            hard = name in used
            ok_all &= not hard
            logger.error(f"{'❌' if hard else '⚠'} 目标 {name:<38} {tc.base_url}  "
                         f"{type(e).__name__}: {e}{'' if hard else '（不在 study 目标中，可忽略）'}")

    g = GeneratorConfig.from_env()
    try:
        client = OpenAI(api_key=g.api_key or "EMPTY", base_url=g.base_url, timeout=90)
        r = client.chat.completions.create(
            model=g.model, messages=[{"role": "user", "content": "ping"}], max_tokens=8)
        text = (r.choices[0].message.content or "").strip()[:30] if r.choices else "(空)"
        logger.info(f"✅ 生成器/判分 {g.model:<30} {g.base_url}  回复: {text!r}")
    except Exception as e:
        ok_all = False
        logger.error(f"❌ 生成器/判分 {g.model:<30} {g.base_url}  {type(e).__name__}: {e}")

    import os

    emb_base = os.getenv("EMBEDDING_API_BASE", "")
    if emb_base:
        try:
            client = OpenAI(api_key=os.getenv("EMBEDDING_API_KEY") or "EMPTY",
                            base_url=emb_base, timeout=30)
            r = client.embeddings.create(model=os.getenv("EMBEDDING_API_MODEL", "bge-m3"),
                                         input="ping")
            dim = len(r.data[0].embedding) if r.data else 0
            logger.info(f"✅ Embedding {os.getenv('EMBEDDING_API_MODEL')}  {emb_base}  dim={dim}")
        except Exception as e:
            # embedding 失败可接受（框架有 TF-IDF 回退链），仅告警
            logger.warning(f"⚠ Embedding 端点异常（将走本地/TF-IDF 回退）: {type(e).__name__}: {e}")

    logger.info("\n探活结果: " + ("全部通过 ✅" if ok_all else "存在失败 ❌（先修复端点再跑实验）"))
    return 0 if ok_all else 1


# ---------------------------------------------------------------- smoke
def cmd_smoke() -> int:
    """迷你 grid 冒烟：验证 runner 子进程 + 指标提取链路（快，用 l1.jsonl）。"""
    from llmsec.experiments.schema import StudyConfig
    from llmsec.experiments.study import run_study

    targets = load_targets()
    tgt = next(iter(targets))
    cfg = StudyConfig.from_dict({
        "name": "weekend_smoke",
        "objective": {"metric": "ci_half", "direction": "minimize", "aggregate": "mean"},
        "strategy": "grid",
        "budget": {"max_trials": 2, "max_wall_minutes": 25, "trial_timeout_minutes": 20},
        "repeats": 1,
        "seed_base": 7,
        "space": {"K_FACTOR": {"type": "int", "low": 16, "high": 32, "step": 16}},
        "fixed": {"input": "l1.jsonl", "max_rounds": 2, "batch_size": 6},
        "targets": [tgt],
        "max_concurrent": 1,
        "config_concurrency": 2,
    })
    summary = run_study(cfg)
    best = summary.get("best") or {}
    mv = best.get("ci_half_mean")
    if mv is None:
        logger.error("❌ 冒烟失败：未提取到 ci_half 指标，检查 runner 日志")
        return 1
    logger.info(f"✅ 冒烟通过：目标 {tgt}，ci_half={mv:.1f}，指标提取链路正常")
    return 0


# ---------------------------------------------------------------- stage2 生成
def _narrow_range(values: list, spec: dict) -> dict:
    """由 top-K config 的取值收窄该维范围（int 保持 step；float 上下各扩 15%）。"""
    vs = sorted(v for v in values if isinstance(v, (int, float)))
    spec = dict(spec)
    if not vs:
        return spec
    if spec.get("type") == "int":
        lo, hi = max(int(spec["low"]), int(math.floor(min(vs)))), min(int(spec["high"]), int(math.ceil(max(vs))))
        step = int(spec.get("step") or 1)
        lo = int(spec["low"]) + ((lo - int(spec["low"])) // step) * step
        if lo > hi:
            lo = hi = int(vs[len(vs) // 2])
        spec["low"], spec["high"] = lo, hi
    else:
        lo, hi = min(vs), max(vs)
        pad = (hi - lo) * 0.15 or abs(hi) * 0.05 or 0.1
        spec["low"] = max(float(spec["low"]), lo - pad)
        spec["high"] = min(float(spec["high"]), hi + pad)
    return spec


def _gen_stage2() -> int:
    """读 stage1 summary.json，生成收窄的 stage2 yaml。"""
    import yaml

    from llmsec.experiments.study import study_dir

    sdir = study_dir("weekend_stage1")
    summary = json.loads((sdir / "summary.json").read_text(encoding="utf-8")) if \
        (sdir / "summary.json").exists() else None
    if not summary or not summary.get("rows"):
        logger.error("❌ stage1 无可用结果，无法生成 stage2（先跑 stage1）")
        return 1

    stage1 = yaml.safe_load(STAGE1_YAML.read_text(encoding="utf-8"))
    top = summary["rows"][:STAGE2_TOP_K]
    space = {}
    for name, spec in stage1["space"].items():
        vals = [r["params"].get(name) for r in top]
        space[name] = _narrow_range(vals, spec)

    stage1_best = top[0]["params"]
    fixed = dict(stage1["fixed"])
    cfg = {
        "name": "weekend_stage2",
        "description": f"周末阶段2细化：stage1 top{STAGE2_TOP_K} 收窄空间，repeats=2 风险厌恶",
        "objective": {"metric": "ci_half", "direction": "minimize",
                      "aggregate": "mean_plus_std"},
        "strategy": "bayesian",
        "budget": {"max_trials": 25, "max_wall_minutes": 720,
                   "trial_timeout_minutes": 75},
        "repeats": 2,
        "seed_base": 2000,
        "space": space,
        "fixed": fixed,
        "targets": stage1["targets"],
        "max_concurrent": 2,
        "config_concurrency": 3,
    }
    STAGE2_YAML.write_text(
        "# 自动生成（scripts/weekend_hpo.py stage2），勿手改——重跑命令会覆盖\n"
        + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    logger.info(f"📝 已生成 {STAGE2_YAML.name}：空间收窄自 stage1 top{STAGE2_TOP_K}，"
                f"stage1 最优={ {k: v for k, v in stage1_best.items()} }")
    return 0


def _run_study_yaml(path: Path) -> int:
    from llmsec.experiments.schema import StudyConfig
    from llmsec.experiments.study import run_study

    run_study(StudyConfig.from_yaml(path))
    return 0


# ---------------------------------------------------------------- report
def _env_lines(best_params: dict) -> list[str]:
    """把最优 search 参数转成 .env 可追加的 LLMSEC_PARAM_* 行。

    CLI 旗标因子（sampler/batch_size/...）不是 params.py 常量，直接写
    LLMSEC_PARAM_SAMPLER 会被当未知名忽略——映射到对应的 DEFAULT_* 常量。
    """
    cli_to_param = {"sampler": "DEFAULT_SAMPLER", "batch_size": "DEFAULT_BATCH_SIZE",
                    "max_rounds": "DEFAULT_MAX_ROUNDS"}
    lines = []
    for k, v in sorted(best_params.items()):
        name = cli_to_param.get(k, k)
        lines.append(f"LLMSEC_PARAM_{name.upper()}={v}")
    return lines


def cmd_report() -> int:
    from llmsec.experiments.schema import StudyConfig
    from llmsec.experiments.study import summarize

    best_overall = None
    best_src = ""
    for name in ("weekend_stage1", "weekend_stage2", "weekend_stage3"):
        from llmsec.experiments.study import study_dir

        sd = study_dir(name)
        if not (sd / "study.yaml").exists():
            continue
        summary = summarize(StudyConfig.from_yaml(sd / "study.yaml"))
        best = summary.get("best")
        if best:
            mv = best.get("ci_half_mean")
            logger.info(f"📊 {name} 最优 ci_half={mv:.2f} ± {best.get('ci_half_std', 0):.2f}"
                        f"  params={best['params']}")
            if best_overall is None or (mv is not None and mv < best_overall[0]):
                best_overall = (mv, best["params"])
                best_src = name
    if best_overall:
        logger.info(f"\n🏆 总体最优（来自 {best_src}）：")
        logger.info("建议追加到 .env：")
        for line in _env_lines(best_overall[1]):
            logger.info(f"   {line}")
    else:
        logger.warning("尚无任何 study 结果")
    return 0


# ---------------------------------------------------------------- confirm
_STAGE3_BEST = {  # stage3 最优（ci_half=6.19）
    "sampler": "infogain", "batch_size": 8, "ADAPTIVE_BATCH_MAX": 8,
    "SAMPLER_INFOGAIN_BETA": 0.2647719057119191, "JUDGE_B_LEVEL_DISCOUNT": 0.9299343092208774,
    "RIDGE_LAMBDA_MAX": 4, "BLEND_PRIOR_K": 18.227585878655034,
    "HDBSCAN_MIN_CLUSTER_DIV": 66,
}
_STAGE2_BASE = {  # stage2 配置（ci_half=8.29；新维度全为当时默认值）
    "sampler": "hybrid", "batch_size": 10, "ADAPTIVE_BATCH_MAX": 12,
    "SAMPLER_INFOGAIN_BETA": 0.3, "JUDGE_B_LEVEL_DISCOUNT": 0.8,
    "RIDGE_LAMBDA_MAX": 4, "BLEND_PRIOR_K": 10.0,
    "HDBSCAN_MIN_CLUSTER_DIV": 40,
}
# 两臂共用的 stage2 调优锁定参数
_LOCKED = {"input": "all_merged.jsonl", "max_rounds": 4,
           "K_FACTOR": 8, "K_DEF_DECAY_N0": 23, "CONV_WINDOW_MIN": 7,
           "SCORE_PERF_TAU": 2.289410576710754, "EMBEDDING_PCA_DIM": 40,
           "TREE_K_MIN": 5, "TREE_K_MAX": 17}


def cmd_confirm() -> int:
    """确认对比：stage3 最优 vs stage2 配置，各 3 seed × 2 目标（新 seed 段 4000+）。"""
    from llmsec.experiments.schema import StudyConfig
    from llmsec.experiments.study import run_study

    import yaml

    targets = yaml.safe_load(STAGE1_YAML.read_text(encoding="utf-8")).get("targets")
    for name, arm in (("weekend_confirm_best", _STAGE3_BEST), ("weekend_confirm_base", _STAGE2_BASE)):
        cfg = StudyConfig.from_dict({
            "name": name,
            "description": f"确认跑：{'stage3最优' if 'best' in name else 'stage2基线'} ×3 seed",
            "objective": {"metric": "ci_half", "direction": "minimize", "aggregate": "mean"},
            "strategy": "grid",
            "budget": {"max_trials": 1, "max_wall_minutes": 120, "trial_timeout_minutes": 75},
            "repeats": 3,
            "seed_base": 4000,
            "space": {},
            "fixed": {**_LOCKED, **arm},
            "targets": targets,
            "max_concurrent": 2,
            "config_concurrency": 1,
        })
        s = run_study(cfg)
        best = s.get("best") or {}
        mv, std = best.get("ci_half_mean"), best.get("ci_half_std", 0)
        logger.info(f"🔬 {name}: ci_half={mv:.2f} ± {std:.2f} (n={best.get('n_success', 0)})")
    return 0


# ---------------------------------------------------------------- main
def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "probe":
        return cmd_probe()
    if cmd == "smoke":
        return cmd_smoke()
    if cmd == "stage2":
        return _gen_stage2() or _run_study_yaml(STAGE2_YAML)
    if cmd == "report":
        return cmd_report()
    if cmd == "confirm":
        return cmd_confirm()
    if cmd == "run":
        rc = _run_study_yaml(STAGE1_YAML)
        if rc:
            return rc
        rc = _gen_stage2()
        if rc:
            return rc
        rc = _run_study_yaml(STAGE2_YAML)
        cmd_report()
        return rc
    logger.info(__doc__)
    return 1 if cmd else 0


if __name__ == "__main__":
    sys.exit(main())
