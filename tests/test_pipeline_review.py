"""
流水线审查回归测试（D 组收尾）。

覆盖终审发现的交接问题：
  S1  runner main 调用 run_attack_phase 透传 no_early_stop；--work-dir 模式强制 True
  S2  --phase 2 且 state.json 不存在 → SystemExit 码非 0
  S3  run_allergy_phase 在 tracker.defender_ratings 无该 defender 时早退，
      返回 fpr=None 且不发起任何 call_target
  S5  攻击集含数学探针时 summary 的 jailbreak_tax 含 baseline_accuracy/accuracy_drop；
      measure_math_baseline 抛异常时降级 baseline=None 不炸
  M3  传 r_snapshot 时 ResultsMatrix.load 不被调用（评估期间不读活 R）
  M7  defender_name=None 时不调用 set_active_target
  M8  过敏检测全部 API 失败 → 返回 fpr=None（不伪造 0）
  M4  ELOTracker save/load 往返保留 attacker_pred_source

全部用 monkeypatch/mock，禁止真实 API/网络。
"""
import inspect
from types import SimpleNamespace as NS

import pytest

import llmsec.pipeline.allergy_phase as alp
import llmsec.pipeline.attack_phase as ap
import llmsec.pipeline.runner as rn
from llmsec.core.io import write_jsonl
from llmsec.core.results import ResultsMatrix
from llmsec.evaluation.elo import ELOTracker


# ============================================================
# 公共构造
# ============================================================
def _records(n=12):
    """最小攻击集记录（与 test_runner.py 同构）。"""
    return [{
        "id": f"m{i}", "method": f"m{i}", "prompt": f"attack prompt {i}",
        "expected_answer": 0, "category": "test", "harm_type": "test",
    } for i in range(n)]


def _eval_math(prompt, ea, judge, use_judge=True):
    """带数学探针的 evaluate_single 替身：全部答对（math_score=2）。"""
    return {
        "content": "ok", "math_score": 2, "actual_answer": "0",
        "is_refusal": False, "is_harmful": True, "eval_score": 3.0,
        "jailbreak_tax": 0.5, "status": "fully_compliant",
        "compliance_level": "A", "latency_ms": 100,
        "tokens_prompt": 50, "tokens_completion": 10,
    }


def _run_math_attack(tmp_path, monkeypatch, defender_name="test-def"):
    """跑一轮含数学探针的 run_attack_phase（evaluate_single/R 全部 mock）。"""
    import llmsec.core.config as cfg
    records = _records()
    monkeypatch.setattr(ap, "evaluate_single", _eval_math)
    # 隔离特征缓存到 tmp：fit_features 会原子写 FEATURE_CACHE_FILE，
    # 多 worker(-n auto) 并发抢写全局 output/feature_cache.pkl 在 Windows 上 os.replace 失败。
    monkeypatch.setattr(cfg, "FEATURE_CACHE_FILE", tmp_path / "feature_cache.pkl")
    tracker = ELOTracker()
    tracker.predictor.fit_features(records)
    return ap.run_attack_phase(
        records, judge=None, tracker=tracker,
        batch_size=3, max_rounds=2, attack_file=tmp_path / "attack.jsonl",
        sampler="gap", coordinate_rounds=2,
        state_file=str(tmp_path / "state.json"),
        defender_name=defender_name,
        r_snapshot=ResultsMatrix(),  # 空快照：不读全局 R
    )


def _s5_set_target_spy(monkeypatch, calls: list):
    """只记录 run_attack_phase 帧直发的 set_active_target 调用（种子批 + S5 基线）。

    评估 worker（_eval_one 闭包）每个方法都会调一次 set_active_target 做
    threading.local ambient 目标继承，与帧直发路径无关，须按调用方帧过滤。
    """
    def _spy(name):
        caller = inspect.currentframe().f_back.f_code.co_name
        if caller == "run_attack_phase":
            calls.append(name)
    monkeypatch.setattr(ap, "set_active_target", _spy)


# ============================================================
# S1：runner main 透传 no_early_stop；work_dir 强制 True
# ============================================================
def _patch_main_runtime(monkeypatch, tmp_path, captured):
    """把 main() 的运行时依赖全部替换为 mock，captured 收集 run_attack_phase 的 kwargs。"""
    records = _records()
    attack_file = tmp_path / "attacks.jsonl"
    write_jsonl(str(attack_file), records)

    def _stub_attack_phase(*args, **kwargs):
        captured.update(kwargs)
        return {}
    monkeypatch.setattr(rn, "run_attack_phase", _stub_attack_phase)
    monkeypatch.setattr(rn, "create_judge_client", lambda: None)
    monkeypatch.setattr(rn, "Judge", lambda client: None)
    # runner.py 的 twin_client = OpenAI(...) 在 --phase 2 时先于 state 校验执行；
    # CI 无 GENERATOR_API_KEY 会在构造期抛 OpenAIError。早退路径用不到 twin_client → 置 None。
    monkeypatch.setattr(rn, "OpenAI", lambda **kw: None)
    # 禁写全局 R / 禁重训预筛模型 / 禁生成真实报告
    monkeypatch.setattr(rn, "publish_tracker", lambda tracker, name: None)
    monkeypatch.setattr(ResultsMatrix, "load", classmethod(lambda cls: ResultsMatrix()))
    import llmsec.evaluation.prescreen_ml as pml
    monkeypatch.setattr(pml, "train", lambda: {"trained": False})
    import llmsec.reporting.final_report as fr
    monkeypatch.setattr(fr, "generate_reports", lambda **kw: None)

    import llmsec.targets as tgt
    monkeypatch.setattr(tgt, "available_targets", lambda: {
        "t1": NS(model="t1-model", api_key="k", base_url="http://t1"),
    })

    # main() 的 work_dir 分支会直接改模块属性（不经 monkeypatch），
    # 先登记原值，teardown 时由 monkeypatch 恢复，避免污染其它测试
    import llmsec.core.config as cfg
    import llmsec.core.results as res
    for mod, name in ((res, "RESULTS_FILE"), (cfg, "ELO_CACHE_FILE"),
                      (cfg, "FEATURE_CACHE_FILE"), (cfg, "CLUSTER_RESULT_FILE")):
        monkeypatch.setattr(mod, name, getattr(mod, name))
    return str(attack_file)


def test_s1_no_early_stop_passthrough(tmp_path, monkeypatch):
    """--no-early-stop 应原样透传到 run_attack_phase。"""
    captured = {}
    attack_file = _patch_main_runtime(monkeypatch, tmp_path, captured)
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(rn, "RUNS_DIR", runs)
    monkeypatch.setattr("sys.argv", [
        "runner", "--phase", "1", "--input", attack_file,
        "--targets", "t1", "--no-early-stop",
    ])
    rn.main()
    assert captured.get("no_early_stop") is True


def test_s1_work_dir_forces_no_early_stop(tmp_path, monkeypatch):
    """--work-dir 实验模式即使不带 --no-early-stop 也强制 True（固定预算可比性）。"""
    captured = {}
    attack_file = _patch_main_runtime(monkeypatch, tmp_path, captured)
    monkeypatch.setattr("sys.argv", [
        "runner", "--phase", "1", "--input", attack_file,
        "--targets", "t1", "--work-dir", str(tmp_path / "wd"),
    ])
    rn.main()
    assert captured.get("no_early_stop") is True


# ============================================================
# S2：--phase 2 无 state.json → SystemExit 非 0
# ============================================================
def test_s2_phase2_missing_state_raises(tmp_path, monkeypatch):
    captured = {}
    attack_file = _patch_main_runtime(monkeypatch, tmp_path, captured)
    monkeypatch.setattr("sys.argv", [
        "runner", "--phase", "2", "--input", attack_file,
        "--targets", "t1", "--work-dir", str(tmp_path / "wd"),
    ])
    # R3 修复：worker 线程内 sys.exit 只终止该线程（与串行模式语义不一致），
    # 统一改为 raise RuntimeError——串行模式向上传播，并发模式被 fut.result()
    # 捕获记为该目标失败
    with pytest.raises(RuntimeError, match="state.json"):
        rn.main()


# ============================================================
# S3：无该 defender 的 ELO 数据 → 早退 fpr=None，零 API 调用
# ============================================================
def test_s3_allergy_early_return_no_defender_data(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(alp, "call_target",
                        lambda prompt: calls.append(prompt)
                        or {"error": None, "content": "x", "target_refused": False})
    tracker = ELOTracker()  # defender_ratings 为空
    summary = alp.run_allergy_phase(
        {"m1": {"id": "x1", "prompt": "p", "method": "m1"}},
        twin_client=None, judge=None, tracker=tracker,
        n_window=1, allergy_file=tmp_path / "allergy.json",
        defender_name="def",
    )
    assert summary["fpr"] is None
    assert summary["total_tested"] == 0
    assert not calls, "S3: 无 ELO 数据时不应发起任何 call_target"


# ============================================================
# S5：数学探针 → 基线接线；基线测量异常 → 降级不炸
# ============================================================
def test_s5_math_baseline_wired_into_tax_summary(tmp_path, monkeypatch):
    set_target_calls = []
    _s5_set_target_spy(monkeypatch, set_target_calls)
    monkeypatch.setattr(ap, "measure_math_baseline", lambda: {"accuracy": 0.9})

    summary = _run_math_attack(tmp_path, monkeypatch)

    tax = summary["jailbreak_tax"]
    assert tax["probed"] > 0, "S5: 数学探针记录应进入越狱税统计"
    assert tax["baseline_accuracy"] == 0.9
    # 全部答对 → attack_accuracy=1.0，drop = 0.9 - 1.0
    assert tax["accuracy_drop"] == pytest.approx(-0.1)
    # run_attack_phase 帧直发的 set_active_target 现有两处（均应路由到 test-def）：
    #   1. 种子批发前的 ambient 补设（H1 修复：种子不得回退全局默认客户端）
    #   2. S5 越狱税基线测量前
    assert set_target_calls and all(c == "test-def" for c in set_target_calls), \
        "S5: 种子/基线路径的 set_active_target 都必须路由到 defender"


def test_s5_math_baseline_failure_degrades_gracefully(tmp_path, monkeypatch):
    monkeypatch.setattr(ap, "set_active_target", lambda name: None)

    def _boom():
        raise RuntimeError("baseline API down")
    monkeypatch.setattr(ap, "measure_math_baseline", _boom)

    summary = _run_math_attack(tmp_path, monkeypatch)  # 不炸即过半

    tax = summary["jailbreak_tax"]
    assert tax["probed"] > 0
    assert tax["baseline_accuracy"] is None, "S5: 基线测量失败应降级为无基线输出"
    assert tax["accuracy_drop"] is None


# ============================================================
# M3：传 r_snapshot 时 ResultsMatrix.load 未被调用
# ============================================================
def test_m3_r_snapshot_skips_results_matrix_load(tmp_path, monkeypatch):
    load_calls = []
    monkeypatch.setattr(ResultsMatrix, "load",
                        classmethod(lambda cls: load_calls.append(1) or ResultsMatrix()))
    # 评估后的簇级安全分析（analyze_clusters→build_blend_predictor_summary）会
    # ResultsMatrix.load() 读 R 出报告摘要——主线程事后只读，与 M3 的"评估期间
    # 不读活 R"无关，此处隔离掉只断评估回路
    monkeypatch.setattr(ap, "analyze_clusters", lambda tracker: {})

    records = _records()
    monkeypatch.setattr(ap, "evaluate_single", _eval_math)
    tracker = ELOTracker()
    tracker.predictor.fit_features(records)
    ap.run_attack_phase(
        records, judge=None, tracker=tracker,
        batch_size=3, max_rounds=1, attack_file=tmp_path / "attack.jsonl",
        sampler="gap", coordinate_rounds=1,
        state_file=str(tmp_path / "state.json"),
        defender_name="test-def",
        r_snapshot=ResultsMatrix(),
    )
    assert not load_calls, "M3: 传入 r_snapshot 后评估期间不应再 ResultsMatrix.load()"


# ============================================================
# M7：defender_name=None 时不调用 set_active_target
# ============================================================
def test_m7_no_defender_name_skips_set_active_target(tmp_path, monkeypatch):
    set_target_calls = []
    _s5_set_target_spy(monkeypatch, set_target_calls)
    monkeypatch.setattr(ap, "measure_math_baseline", lambda: {"accuracy": 0.8})

    summary = _run_math_attack(tmp_path, monkeypatch, defender_name=None)

    assert set_target_calls == [], "M7: defender_name=None 时 S5 基线路径不应 set_active_target"
    # S5 块仍应执行（基线照常测量，只是不做目标路由）
    assert summary["jailbreak_tax"]["baseline_accuracy"] == 0.8


# ============================================================
# M8：过敏检测全部 API 失败 → fpr=None
# ============================================================
def test_m8_allergy_all_api_failure_fpr_none(tmp_path, monkeypatch):
    monkeypatch.setattr(alp, "call_target",
                        lambda prompt: {"error": "connection refused", "content": None,
                                        "target_refused": False})
    monkeypatch.setattr(alp, "get_or_create_twin",
                        lambda name, rec, cache, client: f"safe variant of {name}")
    monkeypatch.setattr(alp, "SAFE_TWINS_FILE", tmp_path / "safe_twins.jsonl")
    monkeypatch.setattr(alp, "API_DELAY", 0)

    tracker = ELOTracker()
    tracker.update_round("def", [("m0", 3.0)])
    tracker.update_round("def", [("m1", -2.0)])
    method_records = {f"m{i}": {"id": f"m{i}", "prompt": f"p{i}", "method": f"m{i}"}
                      for i in range(2)}

    summary = alp.run_allergy_phase(
        method_records, twin_client=None, judge=None, tracker=tracker,
        n_window=2, allergy_file=tmp_path / "allergy.json",
        concurrency=0, defender_name="def",
    )
    assert summary["total_tested"] == 0
    assert summary["fpr"] is None, "M8: 全部 API 失败时 fpr 应为 None（未测），不伪造 0"


# ============================================================
# M4：ELOTracker save/load 往返保留 attacker_pred_source
# ============================================================
def test_m4_elo_save_load_roundtrip_pred_source(tmp_path):
    tr = ELOTracker()
    tr.attacker_pred_source = {"m1": "svd_ridge", "m2": "predicted_global"}
    tr.attacker_pred_std = {"m1": 12.5, "m2": 80.0}

    f = tmp_path / "state.json"
    tr.save(f)

    tr2 = ELOTracker().load(str(f))
    assert tr2.attacker_pred_source == {"m1": "svd_ridge", "m2": "predicted_global"}
    assert tr2.attacker_pred_std == {"m1": 12.5, "m2": 80.0}
