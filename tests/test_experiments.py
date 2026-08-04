#!/usr/bin/env python3
"""
回归测试：实验框架（schema 解析 / 搜索引擎 / 指标聚合 / work-dir 隔离）。
"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llmsec.experiments.schema import StudyConfig, resolve_trial
from llmsec.experiments.search import build_search, GridSearch, RandomSearch, BayesianSearch
from llmsec.experiments.metrics import aggregate


def test_schema_and_resolve() -> int:
    cfg = StudyConfig.from_dict({
        "name": "t",
        "objective": {"metric": "conv_rounds", "direction": "minimize"},
        "budget": {"max_trials": 4},
        "strategy": "grid",
        "repeats": 2,
        "seed_base": 7,
        "space": {
            "sampler": {"type": "categorical", "choices": ["hybrid", "gap"]},
            "K_FACTOR": {"type": "int", "low": 16, "high": 32, "step": 16},
            "SCORE_PERF_TAU": {"type": "float", "low": 1.0, "high": 3.0, "log": True},
        },
        "fixed": {"target": "minimax", "input": "x.jsonl", "max_rounds": 5},
    })
    if cfg.repeats != 2 or cfg.seed_base != 7 or cfg.budget_max_trials != 4:
        print("❌ schema 基本字段解析错误"); return 1
    if cfg.space["sampler"].choices != ["hybrid", "gap"]:
        print("❌ categorical 解析错误"); return 1
    # resolve：CLI 因子进 argv，params 因子进 env
    argv, env = resolve_trial({"sampler": "gap", "K_FACTOR": 48, "target": "minimax", "max_rounds": 5})
    if "--sampler" not in argv or "gap" not in argv:
        print("❌ CLI 因子未进 argv:", argv); return 1
    if env.get("LLMSEC_PARAM_K_FACTOR") != "48":
        print("❌ params 因子未进 env:", env); return 1
    if "target" not in " ".join(argv):
        print("❌ target 未进 argv"); return 1
    print("✅ schema + resolve 通过")
    return 0


def test_grid_exhausts() -> int:
    cfg = StudyConfig.from_dict({
        "name": "g", "strategy": "grid", "budget": {"max_trials": 99}, "repeats": 1,
        "space": {
            "sampler": {"choices": ["hybrid", "gap"]},
            "K_FACTOR": {"type": "int", "low": 16, "high": 32, "step": 16},
        },
        "fixed": {},
    })
    g = build_search(cfg)
    combos = []
    while True:
        p = g.ask()
        if p is None:
            break
        combos.append(p)
    if len(combos) != 4:  # 2 × 2
        print(f"❌ grid 应产出 4 组，实际 {len(combos)}: {combos}"); return 1
    print(f"✅ grid 穷举通过 ({len(combos)} 组)")
    return 0


def test_random_in_range() -> int:
    cfg = StudyConfig.from_dict({
        "name": "r", "strategy": "random", "budget": {"max_trials": 3}, "repeats": 1,
        "space": {
            "K_FACTOR": {"type": "int", "low": 16, "high": 64},
            "SCORE_PERF_TAU": {"type": "float", "low": 0.5, "high": 4.0, "log": True},
        },
        "fixed": {},
    })
    r = build_search(cfg)
    for _ in range(5):
        p = r.ask()
        if not (16 <= p["K_FACTOR"] <= 64):
            print("❌ random int 越界:", p); return 1
        if not (0.5 <= p["SCORE_PERF_TAU"] <= 4.0):
            print("❌ random log-float 越界:", p); return 1
    print("✅ random 采样通过")
    return 0


def test_bayesian_ask_tell() -> int:
    cfg = StudyConfig.from_dict({
        "name": "b", "strategy": "bayesian", "budget": {"max_trials": 3}, "repeats": 1,
        "objective": {"metric": "conv_rounds", "direction": "minimize"},
        "space": {"K_FACTOR": {"type": "int", "low": 16, "high": 64}},
        "fixed": {},
    })
    b = build_search(cfg, completed=[])
    vals = []
    for i in range(3):
        p = b.ask()
        if p is None or "K_FACTOR" not in p:
            print("❌ bayesian ask 失败:", p); return 1
        v = 10.0 + (p["K_FACTOR"] - 40) ** 2 / 100  # 假目标：K=40 附近最优
        b.tell(p, v)
        vals.append(v)
    if not vals:
        print("❌ bayesian 未跑完"); return 1
    print(f"✅ bayesian ask/tell 通过 (目标序列 {[round(v,2) for v in vals]})")
    return 0


def test_aggregate() -> int:
    if aggregate([3.0, 5.0], "mean") != 4.0:
        print("❌ mean 聚合错误"); return 1
    # mean_minus_std：[3,5] → mean=4, std≈1.41 → 2.59（比 mean 小，最小化下偏好稳）
    v = aggregate([3.0, 5.0], "mean_minus_std")
    if abs(v - (4.0 - (2 ** 0.5))) > 1e-6:
        print(f"❌ mean_minus_std 聚合错误: {v}"); return 1
    if aggregate([float("inf"), float("inf")], "mean") != float("inf"):
        print("❌ inf 处理错误"); return 1
    print("✅ aggregate 通过")
    return 0


def test_metrics_from_report() -> int:
    from llmsec.experiments.metrics import extract_metrics
    with tempfile.TemporaryDirectory() as d:
        wd = Path(d)
        (wd / "runner_report.json").write_text(json.dumps({
            "elo": {"conv_rounds": 6, "boundary_elo": 1620.0, "ci_half": 18.0,
                    "converged": True, "coverage": 0.55},
            "attack_phase": {"asr": 0.1, "rounds": 8, "total_tested": 20},
            "allergy": {"fpr": 0.2},
        }), encoding="utf-8")
        m = extract_metrics(wd, max_rounds=8)
    if m.get("conv_rounds") != 6 or m.get("defender_elo") != 1620.0:
        print("❌ 指标提取错误:", m); return 1
    print(f"✅ 指标提取通过 (conv_rounds={m['conv_rounds']}, elo={m['defender_elo']})")
    return 0


def test_orchestration_mock() -> int:
    """mock executor 验证 run_study 的循环/聚合/排名/断点续跑（不实跑 runner）。"""
    import llmsec.experiments.study as study_mod

    cfg = StudyConfig.from_dict({
        "name": "orch_test",
        "objective": {"metric": "conv_rounds", "direction": "minimize", "aggregate": "mean"},
        "budget": {"max_trials": 4},
        "strategy": "grid",
        "repeats": 1,
        "seed_base": 0,
        "space": {
            "K_FACTOR": {"type": "int", "low": 16, "high": 32, "step": 16},   # 2 值
            "sampler": {"type": "categorical", "choices": ["hybrid", "gap"]},  # ×2 → 4 config
        },
        "fixed": {"target": "fake", "input": "x.jsonl", "max_rounds": 5},
    })

    call_count = {"n": 0}

    def fake_run_trial(config, seed, work_dir, study_name, trial_idx):
        call_count["n"] += 1
        # 假目标：K=16 比 K=32 收敛快；gap 比 hybrid 慢 → 最低应为 K=16+hybrid
        cr = 3 + (config["K_FACTOR"] - 16) / 16 * 2 + (1 if config["sampler"] == "gap" else 0)
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        (Path(work_dir) / "runner_report.json").write_text(json.dumps({
            "elo": {"conv_rounds": cr, "boundary_elo": 1600.0 + cr, "converged": True},
            "attack_phase": {"asr": 0.1, "rounds": 5, "total_tested": 10},
            "allergy": {},
        }), encoding="utf-8")
        return {"study": study_name, "trial": trial_idx, "seed": seed, "params": config,
                "status": "success", "metrics": {"conv_rounds": cr, "defender_elo": 1600.0 + cr},
                "work_dir": str(work_dir)}

    orig = study_mod.run_trial
    study_mod.run_trial = fake_run_trial
    first_count = second_count = -1
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            study_mod.STUDIES_DIR = Path(d)
            summary = study_mod.run_study(cfg)
            first_count = call_count["n"]

            # 断点续跑：同一 study 目录再跑一次应 0 新调用（4 个 config 已完成）
            call_count["n"] = 0
            study_mod.run_study(cfg)
            second_count = call_count["n"]
    finally:
        study_mod.run_trial = orig

    if first_count != 4:
        print(f"❌ 应跑 4 个 trial，实际 {first_count}"); return 1
    best = summary.get("best")
    if not best or best["params"].get("K_FACTOR") != 16 or best["params"].get("sampler") != "hybrid":
        print(f"❌ 最佳 config 应为 K=16/hybrid，实际: {best and best['params']}"); return 1
    cr = best.get("conv_rounds_mean")
    if cr is None or abs(cr - 3.0) > 1e-6:
        print(f"❌ 最佳 conv_rounds 应为 3.0，实际 {cr}"); return 1
    # 断点续跑：第二次 run_study 应 0 新 trial
    if second_count != 0:
        print(f"❌ 断点续跑应 0 新 trial，实际 {second_count}"); return 1
    print(f"✅ 编排(循环/聚合/排名/续跑)通过 (best=K16/hybrid conv_rounds=3.0)")
    return 0


def main() -> int:
    for t in (test_schema_and_resolve, test_grid_exhausts, test_random_in_range,
              test_bayesian_ask_tell, test_aggregate, test_metrics_from_report,
              test_orchestration_mock):
        if t() != 0:
            return 1
    print("\n✅ 所有实验框架测试通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
