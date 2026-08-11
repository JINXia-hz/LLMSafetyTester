#!/usr/bin/env python3
"""
回归测试：实验框架（schema 解析 / 搜索引擎 / 指标聚合 / work-dir 隔离）。
"""

import json
import tempfile
from pathlib import Path

from llmsec.experiments.metrics import aggregate
from llmsec.experiments.schema import StudyConfig, resolve_trial
from llmsec.experiments.search import build_search


def test_schema_and_resolve():
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
    assert not (cfg.repeats != 2 or cfg.seed_base != 7 or cfg.budget_max_trials != 4), "❌ schema 基本字段解析错误"
    assert not (cfg.space["sampler"].choices != ["hybrid", "gap"]), "❌ categorical 解析错误"
    # resolve：CLI 因子进 argv，params 因子进 env
    argv, env = resolve_trial({"sampler": "gap", "K_FACTOR": 48, "target": "minimax", "max_rounds": 5})
    if "--sampler" not in argv or "gap" not in argv:
        print("❌ CLI 因子未进 argv:", argv); return 1
    if env.get("LLMSEC_PARAM_K_FACTOR") != "48":
        print("❌ params 因子未进 env:", env); return 1
    assert not ("target" not in " ".join(argv)), "❌ target 未进 argv"


def test_grid_exhausts():
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


def test_random_in_range():
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


def test_bayesian_ask_tell():
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
    assert vals, "❌ bayesian 未跑完"


def test_aggregate():
    assert aggregate([3.0, 5.0], "mean") == 4.0
    # mean_plus_std（M-34 修正）：[3,5] → mean=4, std=1.41 → 5.41
    v = aggregate([3.0, 5.0], "mean_plus_std")
    assert abs(v - (4.0 + (2 ** 0.5))) < 1e-6
    assert aggregate([float("inf"), float("inf")], "mean") == float("inf")
def test_metrics_from_report():
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


def test_orchestration_mock():
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

    def fake_run_trial(config, seed, work_dir, study_name, trial_idx, *args):
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

    assert not (first_count != 4), f"❌ 应跑 4 个 trial，实际 {first_count}"
    best = summary.get("best")
    assert not (not best or best["params"].get("K_FACTOR") != 16 or best["params"].get("sampler") != "hybrid"), f"❌ 最佳 config 应为 K=16/hybrid，实际: {best and best['params']}"
    cr = best.get("conv_rounds_mean")
    assert not (cr is None or abs(cr - 3.0) > 1e-6), f"❌ 最佳 conv_rounds 应为 3.0，实际 {cr}"
    # 断点续跑：第二次 run_study 应 0 新 trial
    assert not (second_count != 0), f"❌ 断点续跑应 0 新 trial，实际 {second_count}"


def test_run_trial_strips_task_id_env(monkeypatch, tmp_path):
    """run_trial 子进程 env 不得继承 LLMSEC_TASK_ID（防 trial 污染 HPO 任务进度文件）。"""
    import os

    import llmsec.experiments.executor as ex

    os.environ["LLMSEC_TASK_ID"] = "leak-test"
    captured = {}

    class FakeCP:
        returncode = 0

    def fake_run(*args, **kwargs):
        captured['env'] = kwargs.get('env')
        return FakeCP()

    monkeypatch.setattr(ex.subprocess, "run", fake_run)
    monkeypatch.setattr(ex, "capture_manifest", lambda *a, **k: {})  # 避免 platform/git 子调用被 fake_run 误伤
    monkeypatch.setattr(ex, "extract_metrics", lambda *a, **k: {})
    try:
        ex.run_trial({"input": "attacks/l1.jsonl", "target": "x"}, 1, tmp_path, "st", 0, 30)
    finally:
        os.environ.pop("LLMSEC_TASK_ID", None)

    assert "LLMSEC_TASK_ID" not in (captured.get('env') or {}), \
        "trial 子进程 env 必须剥离 LLMSEC_TASK_ID"
    print('✅ run_trial 剥离 LLMSEC_TASK_ID 通过')


def test_capture_manifest(tmp_path):
    """capture_manifest 落盘 manifest.json 且字段完整、攻击集 sha1 可算。"""
    import json

    from llmsec.core.config import PROJECT_ROOT
    from llmsec.experiments.manifest import capture_manifest

    attack = PROJECT_ROOT / "attacks" / "example.jsonl"
    capture_manifest(tmp_path, ["python", "-m", "llmsec.pipeline.runner"],
                     {"LLMSEC_PARAM_K": "1"}, 7, str(attack), {"input": "attacks/example.jsonl"})
    p = tmp_path / "manifest.json"
    assert p.exists(), "manifest.json 应落盘"
    d = json.loads(p.read_text(encoding="utf-8"))
    for k in ["captured_at", "git", "python", "platform", "seed", "config", "argv",
              "env_override", "attack_set", "attack_set_sha1", "params_snapshot",
              "env_redacted", "library_versions"]:
        assert k in d, f"manifest 缺字段 {k}"
    assert d["seed"] == 7
    assert d["attack_set_sha1"], "example.jsonl 存在 → sha1 应非空（M-路径修复验证）"
    print('✅ capture_manifest 通过')


def test_run_study_no_targets_fails_fast(tmp_path):
    """targets 与 fixed.target 均空 → run_study 立即 ValueError，不再空转误报"空间穷尽"。"""
    import pytest

    import llmsec.experiments.study as study_mod

    cfg = StudyConfig.from_dict({
        "name": "no_target_ut",
        "objective": {"metric": "conv_rounds", "direction": "minimize"},
        "budget": {"max_trials": 3},
        "strategy": "random",
        "repeats": 1,
        "space": {"K_FACTOR": {"type": "int", "low": 16, "high": 32}},
        "fixed": {"input": "x.jsonl"},   # 无 target；config.targets 默认空
    })
    orig_dir = study_mod.STUDIES_DIR
    study_mod.STUDIES_DIR = tmp_path
    try:
        with pytest.raises(ValueError, match="无有效目标"):
            study_mod.run_study(cfg)
    finally:
        study_mod.STUDIES_DIR = orig_dir
    # 不得产生任何 trial 记录
    assert not (tmp_path / "no_target_ut" / "trials.jsonl").exists(), "fail-fast 前不应产生 trials.jsonl"
    print('✅ run_study 空目标 fail-fast 通过')


