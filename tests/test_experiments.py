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
    assert "--sampler" in argv and "gap" in argv, f"CLI 因子未进 argv: {argv}"
    assert env.get("LLMSEC_PARAM_K_FACTOR") == "48", f"params 因子未进 env: {env}"
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
    assert len(combos) == 4, f"grid 应产出 4 组（2×2），实际 {len(combos)}: {combos}"


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
        assert 16 <= p["K_FACTOR"] <= 64, f"random int 越界: {p}"
        assert 0.5 <= p["SCORE_PERF_TAU"] <= 4.0, f"random log-float 越界: {p}"


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
        assert p is not None and "K_FACTOR" in p, f"bayesian ask 失败: {p}"
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
    assert m.get("conv_rounds") == 6 and m.get("defender_elo") == 1620.0,         f"指标提取错误: {m}"


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
        return {"study": study_name, "idx": trial_idx, "seed": seed, "params": config,
                "status": "success", "metrics": {"conv_rounds": cr, "defender_elo": 1600.0 + cr},
                "work_dir": str(work_dir)}

    orig = study_mod.run_trial
    study_mod.run_trial = fake_run_trial
    orig_studies_dir = study_mod.STUDIES_DIR
    first_count = second_count = -1
    summary = {}
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            study_mod.STUDIES_DIR = Path(d)
            summary = study_mod.run_study(cfg) or {}
            first_count = call_count["n"]

            # 断点续跑：同一 study 目录再跑一次应 0 新调用（4 个 config 已完成）
            call_count["n"] = 0
            study_mod.run_study(cfg)
            second_count = call_count["n"]
    finally:
        study_mod.run_trial = orig
        # STUDIES_DIR 必须恢复：此前指向已删除的临时目录且不还原，
        # 同 worker 的后续测试会继承悬空路径
        study_mod.STUDIES_DIR = orig_studies_dir

    assert first_count == 4, f"❌ 应跑 4 个 trial，实际 {first_count}"
    best = summary.get("best")
    assert best and best["params"].get("K_FACTOR") == 16 and best["params"].get("sampler") == "hybrid",         f"❌ 最佳 config 应为 K=16/hybrid，实际: {best and best['params']}"
    cr = best.get("conv_rounds_mean")
    assert cr is not None and abs(cr - 3.0) <= 1e-6, f"❌ 最佳 conv_rounds 应为 3.0，实际 {cr}"
    # 断点续跑：第二次 run_study 应 0 新 trial
    assert second_count == 0, f"❌ 断点续跑应 0 新 trial，实际 {second_count}"


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


# ============================================================
# metrics：产物定位 / 无 report 时 state 回放兜底 / 聚合边界
# ============================================================

def _mk_state(work_dir: Path, elos: list, **over) -> Path:
    """构造一份 ELOTracker 格式的 state.json（多余字段经 over 覆写）。"""
    st = {
        "attacker_ratings": {"atk1": 1500.0, "atk2": 1510.0, "atk3": 1490.0},
        "defender_ratings": {"modelA": elos[-1] if elos else 1600.0},
        "history": [],
        "round_defender_elos": {"modelA": list(elos)},
        "defender_match_count": {"modelA": max(1, len(elos)) * 3},
        "ground_truth": {"atk1": 1500.0, "atk2": 1510.0, "atk3": 1490.0},
        "attacker_stats": {},
        "attacker_pred_std": {},
        "attacker_pred_source": {},
    }
    st.update(over)
    p = work_dir / "state.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(st), encoding="utf-8")
    return p


def test_find_artifact_root_and_glob_layouts(tmp_path):
    """产物定位：根布局直接命中；隔离布局走 */filename glob；都无则 None。"""
    from llmsec.experiments.metrics import _find_artifact

    # 1) 普通模式：产物在 work_dir 根
    (tmp_path / "runner_report.json").write_text("{}", encoding="utf-8")
    assert _find_artifact(tmp_path, "runner_report.json") == tmp_path / "runner_report.json", \
        "❌1 根布局应直接命中"

    # 2) work-dir 隔离模式：产物在 work_dir/<target>/ 下（glob 分支）
    wd = tmp_path / "trial1"
    sub = wd / "modelA"
    sub.mkdir(parents=True)
    (sub / "state.json").write_text("{}", encoding="utf-8")
    assert _find_artifact(wd, "state.json") == sub / "state.json", "❌2 隔离布局应经 glob 命中"

    # 3) 什么都不存在
    assert _find_artifact(wd, "runner_report.json") is None, "❌3 无产物应返回 None"


def test_extract_metrics_state_replay_converged(tmp_path):
    """无 runner_report 时从 state.json 回放：常数轨迹恰在 CONV_WINDOW_MIN 轮收敛。"""
    from llmsec.experiments.metrics import extract_metrics
    from llmsec.params import CONV_WINDOW_MIN

    wd = tmp_path / "trial"
    wd.mkdir()
    _mk_state(wd, [1600.0] * (CONV_WINDOW_MIN + 1))
    m = extract_metrics(wd, max_rounds=CONV_WINDOW_MIN + 1)
    assert m.get("conv_rounds") == CONV_WINDOW_MIN, \
        f"❌1 首个收敛轮应为 {CONV_WINDOW_MIN}，实际 {m.get('conv_rounds')}"
    assert m.get("work_dir") == str(wd), "❌2 work_dir 应原样带回"


def test_extract_metrics_isolated_workdir_layout(tmp_path):
    """work-dir 隔离布局（wd/<target>/state.json，HPO trial 单目标）同样能回放兜底。"""
    from llmsec.experiments.metrics import extract_metrics
    from llmsec.params import CONV_WINDOW_MIN

    wd = tmp_path / "hpo_trial"          # 根下没有 state.json，只有子目录
    _mk_state(wd / "modelA", [1600.0] * (CONV_WINDOW_MIN + 1))
    m = extract_metrics(wd, max_rounds=CONV_WINDOW_MIN + 1)
    assert m.get("conv_rounds") == CONV_WINDOW_MIN, \
        f"❌1 隔离布局回放失败: {m}"


def test_conv_rounds_unconverged_penalty(tmp_path):
    """未收敛（强漂移轨迹）→ 惩罚值 mr + ci/目标：落在 (mr, mr+1) 且严格大于 mr。"""
    from llmsec.experiments.metrics import extract_metrics

    wd = tmp_path
    elos = [1600 + 15 * i + (2.0 if i % 2 else -2.0) for i in range(8)]  # 漂移 15/轮 > 5
    _mk_state(wd, elos)
    m = extract_metrics(wd, max_rounds=8)
    cr = m.get("conv_rounds")
    assert cr is not None and 8 < cr < 9, \
        f"❌1 惩罚值应落在 (8, 9)（mr=8 加上不足一个目标的 ci），实际 {cr}"


def test_conv_rounds_penalty_ci_missing_uses_target(tmp_path):
    """轮次不足的常数轨迹：ci_half 缺失（0 视同 None）→ 惩罚 = mr + 目标/目标 = mr + 1。"""
    from llmsec.experiments.metrics import extract_metrics

    wd = tmp_path
    _mk_state(wd, [1600.0] * 3)          # 3 轮 < CONV_WINDOW_MIN，永不收敛
    m = extract_metrics(wd, max_rounds=3)
    assert m.get("conv_rounds") == 4.0, \
        f"❌1 ci 缺失时惩罚应为 3 + 20/20 = 4.0，实际 {m.get('conv_rounds')}"


def test_conv_rounds_state_without_defender_returns_none(tmp_path):
    """state 无 defender_ratings → 无法定位防御方，conv_rounds 保持缺失（不抛）。"""
    from llmsec.experiments.metrics import extract_metrics

    wd = tmp_path
    _mk_state(wd, [1600.0] * 8, defender_ratings={})
    m = extract_metrics(wd, max_rounds=8)
    assert "conv_rounds" not in m, f"❌1 无 defender 时 conv_rounds 应缺失，实际 {m}"


def test_conv_rounds_bad_state_returns_none(tmp_path):
    """损坏/畸形 state.json → 记 error 后返回 None，绝不上抛（trial 评分缺失而非崩溃）。"""
    from llmsec.experiments.metrics import extract_metrics

    wd = tmp_path
    # 1) 顶层是 JSON 数组：tracker.load 内 data.get 抛 AttributeError → 吞掉
    (wd / "state.json").write_text("[1, 2, 3]", encoding="utf-8")
    m1 = extract_metrics(wd, max_rounds=8)
    assert "conv_rounds" not in m1, "❌1 数组 state 应被吞掉"
    # 2) 非法 JSON：read_json 静默返回 None → 兜底直接跳过
    (wd / "state.json").write_text("{not json", encoding="utf-8")
    m2 = extract_metrics(wd, max_rounds=8)
    assert "conv_rounds" not in m2, "❌2 非法 JSON 应静默跳过"
    # 3) 空目录：无任何产物，只剩 work_dir
    wd2 = tmp_path / "empty"
    wd2.mkdir()
    m3 = extract_metrics(wd2)
    assert m3.get("conv_rounds") is None and m3["work_dir"] == str(wd2), "❌3 空目录应只带回 work_dir"


def test_aggregate_filters_none_inf_and_single_value():
    """aggregate 边界：None/inf 过滤、空集 inf、单值 std=0。"""
    import math

    from llmsec.experiments.metrics import aggregate

    assert math.isinf(aggregate([], "mean")), "❌1 空列表应返回 inf"
    assert math.isinf(aggregate([None, None], "mean")), "❌2 全 None 应返回 inf"
    assert math.isinf(aggregate([float("inf")], "mean_plus_std")), "❌3 全 inf 应返回 inf"
    assert aggregate([2.0], "mean_plus_std") == 2.0, "❌4 单值 std=0，应等于自身"
    assert aggregate([1.0, None, float("inf"), 3.0], "mean") == 2.0, \
        "❌5 应过滤 None/inf 后取均值"


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


