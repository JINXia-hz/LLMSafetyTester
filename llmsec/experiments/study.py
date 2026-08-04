"""
experiments.study — 实验编排：加载配置 → 顺序执行 trials → 断点续跑 → 聚合。

study 目录：output/experiments/<name>/
  study.yaml     配置副本
  trials.jsonl   append-only trial 记录（断点续跑的真相源）
  best.json      report 时写入的最佳 config
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime
from pathlib import Path

from llmsec.core.config import OUTPUT_DIR
from llmsec.experiments.executor import run_trial
from llmsec.experiments.metrics import aggregate
from llmsec.experiments.schema import StudyConfig
from llmsec.experiments.search import build_search

STUDIES_DIR = OUTPUT_DIR / "experiments"


def study_dir(name: str) -> Path:
    return STUDIES_DIR / name


def _fingerprint(params: dict) -> str:
    return json.dumps(params, sort_keys=True, ensure_ascii=False)


def _load_trials(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def _done_seeds(trials: list[dict], params: dict) -> set[int]:
    fp = _fingerprint(params)
    return {t["seed"] for t in trials
            if t.get("status") == "success" and _fingerprint(t.get("params", {})) == fp}


def run_study(config: StudyConfig) -> dict:
    """顺序执行整个 study，支持断点续跑。返回汇总。"""
    sdir = study_dir(config.name)
    sdir.mkdir(parents=True, exist_ok=True)
    trials_path = sdir / "trials.jsonl"

    completed = _load_trials(trials_path)
    seeds = [config.seed_base + i for i in range(config.repeats)]

    # 已完成的独立 config 数（每个 config 需 repeats 个成功 trial）
    done_fps = {_fingerprint(t["params"]) for t in completed if t.get("status") == "success"}
    configs_done = 0
    for fp in done_fps:
        if len({t["seed"] for t in completed
                if t.get("status") == "success" and _fingerprint(t.get("params", {})) == fp}) >= config.repeats:
            configs_done += 1

    print(f"🧪 study='{config.name}'  strategy={config.strategy}  budget={config.budget_max_trials}  "
          f"repeats={config.repeats}  已完成 config={configs_done}")
    print(f"   目标: {config.objective.direction} {config.objective.metric} ({config.objective.aggregate})")

    engine = build_search(config, _completed_for_engine(completed, config))

    trial_idx = max([t.get("trial", 0) for t in completed], default=0)
    while configs_done < config.budget_max_trials:
        params = engine.ask()
        if params is None:
            print("   搜索空间耗尽（grid）。")
            break
        config_full = {**config.fixed, **params}
        obj_values: list[float] = []
        ran_any = False
        for seed in seeds:
            done = _done_seeds(completed, config_full)
            if seed in done:
                # 复用已有结果（断点续跑）
                t = next(t for t in completed
                         if t["seed"] == seed and _fingerprint(t.get("params", {})) == _fingerprint(config_full))
            else:
                trial_idx += 1
                wd = sdir / f"trial_{trial_idx}_seed{seed}"
                print(f"\n▶ config #{configs_done + 1} seed={seed} → {wd.name}")
                t = run_trial(config_full, seed, wd, config.name, trial_idx)
                with open(trials_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")
                completed.append(t)
                ran_any = True
            mv = (t.get("metrics") or {}).get(config.objective.metric)
            if mv is not None:
                obj_values.append(float(mv))

        if not obj_values:
            print(f"   ⚠ 无有效目标值，跳过该 config 的 tell")
            obj = float("inf") if config.objective.direction == "minimize" else float("-inf")
        else:
            obj = aggregate(obj_values, config.objective.aggregate)
        engine.tell(config_full, obj)
        print(f"   ⇒ {config.objective.metric}={obj:.3f} (seeds={obj_values})")

        # 仅当该 config 全部 seed 成功才计入预算
        if len(_done_seeds(completed, config_full)) >= config.repeats:
            configs_done += 1
        elif not ran_any:
            # 无新运行且未达标——防止死循环
            configs_done += 1

    print(f"\n✅ study 完成：{configs_done} 个 config × {config.repeats} seed")
    return summarize(config)


def _completed_for_engine(trials: list[dict], config: StudyConfig) -> list[dict]:
    """喂给 bayesian 的已完成 (params, objective) 列表（每 config 取代表值）。"""
    by_fp: dict[str, list] = {}
    for t in trials:
        if t.get("status") != "success":
            continue
        fp = _fingerprint(t.get("params", {}))
        mv = (t.get("metrics") or {}).get(config.objective.metric)
        if mv is not None:
            by_fp.setdefault(fp, []).append(float(mv))
    out = []
    for fp, vals in by_fp.items():
        params = json.loads(fp)
        out.append({"params": params, "objective": aggregate(vals, config.objective.aggregate)})
    return out


def summarize(config: StudyConfig) -> dict:
    """聚合所有 trial，按 config 分组，排名。"""
    trials = _load_trials(study_dir(config.name) / "trials.jsonl")
    by_fp: dict[str, list[dict]] = {}
    for t in trials:
        by_fp.setdefault(_fingerprint(t.get("params", {})), []).append(t)

    rows = []
    for fp, ts in by_fp.items():
        succ = [t for t in ts if t.get("status") == "success"]
        if not succ:
            continue
        params = succ[0]["params"]
        row = {"params": params, "n_success": len(succ)}
        # 各指标 mean±std
        metric_names = {k for t in succ for k in (t.get("metrics") or {})}
        for m in metric_names:
            vals = [(t.get("metrics") or {}).get(m) for t in succ]
            vals = [v for v in vals if isinstance(v, (int, float))]
            if vals:
                row[f"{m}_mean"] = statistics.mean(vals)
                row[f"{m}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        rows.append(row)

    rev = config.objective.direction == "maximize"
    rows.sort(key=lambda r: (r.get(f"{config.objective.metric}_mean") if r.get(f"{config.objective.metric}_mean") is not None else (float("-inf") if rev else float("inf")),
                             ), reverse=rev)

    best = rows[0] if rows else None
    summary = {"name": config.name, "objective": config.objective.metric,
               "direction": config.objective.direction, "best": best, "rows": rows}
    (study_dir(config.name) / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary
