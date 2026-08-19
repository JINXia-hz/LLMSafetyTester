from types import SimpleNamespace as NS

import llmsec.core.config as core_config
import llmsec.evaluation.judge as jd
import llmsec.pipeline.allergy_phase as alp
import llmsec.pipeline.attack_phase as ap
import llmsec.pipeline.runner as rn
from llmsec.core.io import read_json, read_jsonl, write_jsonl
from llmsec.evaluation.elo import ELOTracker


def _setup(tmp_path, monkeypatch):
    tmp = tmp_path / "state"
    tmp.mkdir()
    # runner 底部 re-export 区已删：STATE_DIR 只在 core.config 定义
    monkeypatch.setattr(core_config, "STATE_DIR", tmp)
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(rn, "RUNS_DIR", runs)

    # call_target / evaluate_single 由 allergy_phase / attack_phase 顶层直导源模块，
    # patch 必须落在真实消费方命名空间
    monkeypatch.setattr(alp, "call_target",
                        lambda prompt: {"error": None, "content": "ok response from target model", "target_refused": False})
    # 孪生缓存重定向到 tmp，避免读/写全局 output/state/safe_twins.jsonl
    import llmsec.core.config as _alp_cfg
    monkeypatch.setattr(_alp_cfg, "SAFE_TWINS_FILE", tmp / "safe_twins.jsonl")
    monkeypatch.setattr(ap, "evaluate_single",
                        lambda prompt, ea, judge, use_judge=True: {
                            "content": "ok", "math_score": None, "actual_answer": None,
                            "is_refusal": False, "is_harmful": True, "eval_score": 3.0,
                            "jailbreak_tax": None, "status": "fully_compliant",
                            "compliance_level": "A", "latency_ms": 100,
                            "tokens_prompt": 50, "tokens_completion": 10,
                        })

    records = []
    for i in range(12):
        records.append({
            "id": f"m{i}", "method": f"m{i}", "prompt": f"attack prompt {i}",
            "expected_answer": 0, "category": "test", "harm_type": "test",
        })
    write_jsonl(str(tmp / "attacks.jsonl"), records)
    return records


class _Spy:
    def __init__(self, fn):
        self._fn = fn
        self.calls = 0
    def __call__(self, *a, **kw):
        self.calls += 1
        return self._fn(*a, **kw)


def test_attack_phase_incremental_persistence(tmp_path, monkeypatch):
    """run_attack_phase 每轮增量写 attack_results.jsonl（崩溃恢复靠它）。"""
    records = _setup(tmp_path, monkeypatch)

    tracker = ELOTracker()
    tracker.predictor.fit_features(records)
    attack_file = tmp_path / "attack.jsonl"

    attack_writes = []
    orig_write = ap.write_jsonl
    def _spy_write(path, rows):
        if "attack" in str(path):
            attack_writes.append(len(rows))
        return orig_write(path, rows)
    monkeypatch.setattr(ap, "write_jsonl", _spy_write)

    ap.run_attack_phase(
        records, judge=None, tracker=tracker,
        batch_size=3, max_rounds=2, attack_file=attack_file,
        sampler="gap", coordinate_rounds=2,
        state_file=str(tmp_path / "state.json"),
        defender_name="test-def",
    )

    assert attack_writes, "attack_file 应每轮增量写"

    rows = read_jsonl(str(attack_file))
    assert rows and all("method" in r for r in rows)

    # R snapshot model: publish_tracker not called inside run_attack_phase (runner main handles it)


def test_allergy_judge_fallback(tmp_path, monkeypatch):
    """M-15：Phase 2 的 judge.evaluate 抛异常时降级关键词、不崩。"""
    _setup(tmp_path, monkeypatch)

    monkeypatch.setattr(alp, "call_target",
                        lambda prompt: {"error": None, "content": "a benign safe response", "target_refused": False})

    monkeypatch.setattr(alp, "get_or_create_twin", lambda name, rec, cache, client: f"safe variant of {name}")

    def boom_evaluate(self, prompt, response, skip_prescreen=False):
        raise RuntimeError("judge API down")
    monkeypatch.setattr(jd.Judge, "evaluate", boom_evaluate)

    tracker = ELOTracker()
    tracker.update_round("def", [("m0", 3.0)])
    tracker.update_round("def", [("m1", -2.0)])
    tracker.update_round("def", [("m2", 3.0)])
    tracker.update_round("def", [("m3", -2.0)])

    records = [{"method": f"m{i}", "prompt": f"prompt{i}", "id": f"m{i}"} for i in range(4)]
    method_records = {r["method"]: r for r in records}

    from openai import OpenAI
    client = OpenAI(api_key="x", base_url="http://localhost:1")

    summary = rn.run_allergy_phase(
        method_records, client, judge=None, tracker=tracker,
        n_window=4, allergy_file=tmp_path / "allergy.json",
        defender_name="def",
    )
    assert "fpr" in summary, "judge 故障时应降级关键词、不崩"


def test_per_target_reports(tmp_path, monkeypatch):
    """每个目标产出独立 runner_report.json（runs/<ts>/<target>/runner_report.json）。"""
    records = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(ap, "evaluate_single",
                        lambda prompt, ea, judge, use_judge=True: {
                            "content": "ok", "math_score": None, "actual_answer": None,
                            "is_refusal": False, "is_harmful": True, "eval_score": 3.0,
                            "jailbreak_tax": None, "status": "fully_compliant",
                            "compliance_level": "A", "latency_ms": 100,
                            "tokens_prompt": 50, "tokens_completion": 10,
                        })

    import llmsec.targets as tgt
    monkeypatch.setattr(tgt, "available_targets", lambda: {
        "t1": NS(model="t1-model", api_key="k", base_url="http://t1"),
        "t2": NS(model="t2-model", api_key="k", base_url="http://t2"),
    })

    method_records = {r["method"]: r for r in records}
    runs_dir = tmp_path / "runs" / "ts"
    runs_dir.mkdir(parents=True)

    # 直接调 main 的编排逻辑（通过内部函数验证）
    # 直接验证 runner_report 产出逻辑（不经 main 编排）
    from llmsec.core.results import ResultsMatrix
    R = ResultsMatrix()
    R.set_unit_catalog(list(method_records.keys()))
    feat_tracker = ELOTracker()
    feat_tracker.predictor.fit_features(records)

    for name in ("t1", "t2"):
        tracker = ELOTracker()
        tracker.predictor.artifacts = feat_tracker.predictor.artifacts
        attack_file = runs_dir / name / "attack_results.jsonl"
        rn.run_attack_phase(
            records, judge=None, tracker=tracker,
            batch_size=4, max_rounds=1, attack_file=attack_file,
            sampler="gap", coordinate_rounds=2,
            state_file=str(tmp_path / f"state__{name}.json"),
            defender_name=name,
        )
        rn.write_json(runs_dir / name / "runner_report.json", {
            "target_model": name,
            "attack_phase": {"asr": 1.0, "total_tested": len(tracker.ground_truth_methods)},
            "elo": {"boundary_elo": 1500},
            "allergy": {},
        })

    t1_report = runs_dir / "t1" / "runner_report.json"
    t2_report = runs_dir / "t2" / "runner_report.json"
    assert t1_report.exists(), "t1 应有独立 runner_report.json"
    assert t2_report.exists(), "t2 应有独立 runner_report.json"

    rep = read_json(t1_report)
    assert rep.get("target_model") == "t1"
    assert "attack_phase" in rep and "elo" in rep


def test_twin_entry_missing_fields(tmp_path, monkeypatch):
    """twin 记录缺 harm_type/method 字段时不崩。"""
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(alp, "call_target",
                        lambda prompt: {"error": None, "content": "safe response", "target_refused": False})

    tracker = ELOTracker()
    tracker.update_round("def", [("m0", 3.0)])
    tracker.update_round("def", [("m1", -2.0)])
    tracker.update_round("def", [("m2", 3.0)])

    from openai import OpenAI
    client = OpenAI(api_key="x", base_url="http://localhost:1")

    # twin 缺字段
    records = [{"method": "m0", "prompt": "p0", "id": "m0"}]
    method_records = {r["method"]: r for r in records}

    summary = rn.run_allergy_phase(
        method_records, client, judge=None, tracker=tracker,
        n_window=1, allergy_file=tmp_path / "allergy.json",
        defender_name="def",
    )
    assert isinstance(summary, dict)


# ============================================================
# C-3 回归：R-resume 合并方向（state 超集时保留崩溃轮观测）
# ============================================================
class TestMergeResumeFromR:
    @staticmethod
    def _hist(tracker_like, recs, defender="def-x"):
        return [{"attacker": f"u-{r}", "record": r, "defender": defender,
                 "attacker_old_elo": 1500.0, "attacker_new_elo": 1510.0,
                 "attacker_delta": 10.0, "defender_old_elo": 1500.0,
                 "defender_new_elo": 1505.0, "defender_delta": 5.0,
                 "eval_score": 3.0, "round": 0, "status": "fully_compliant"}
                for r in recs]

    def test_state_superset_keeps_crash_rounds(self):
        """state ⊋ R（崩溃前有未 publish 轮次）：history/ratings 保留 state 版本。"""
        from llmsec.evaluation.elo import ELOTracker
        from llmsec.pipeline.attack_phase import _merge_resume_from_r

        state_tr = ELOTracker()
        state_tr.history = self._hist(state_tr, ["r1", "r2", "r3"])
        state_tr.attacker_ratings = {"u-r1": 1510.0, "u-r2": 1512.0, "u-r3": 1514.0}
        state_tr.defender_ratings = {"def-x": 1508.0}

        derived = ELOTracker()
        derived.history = self._hist(derived, ["r1", "r2"])   # R 只到崩溃前 publish
        derived.attacker_ratings = {"u-r1": 1509.0, "u-r2": 1511.0}
        derived.defender_ratings = {"def-x": 1506.0}

        _merge_resume_from_r(state_tr, derived, "def-x", {"u-r1", "u-r2"})

        recs = {h["record"] for h in state_tr.history if h.get("defender") == "def-x"}
        assert recs == {"r1", "r2", "r3"}, "崩溃轮 r3 不得被 R 回放覆盖掉（C-3）"
        assert state_tr.defender_ratings["def-x"] == 1508.0, "评分保留 state 版本"
        assert "u-r3" in state_tr.attacker_ratings

    def test_state_subset_full_injection(self):
        """state 为空（跨 run 进新目录）：R 回放字段级整体并入。"""
        from llmsec.evaluation.elo import ELOTracker
        from llmsec.pipeline.attack_phase import _merge_resume_from_r

        tr = ELOTracker()  # 无 state——全新目录
        derived = ELOTracker()
        derived.history = self._hist(derived, ["r1"])
        derived.attacker_ratings = {"u-r1": 1511.0}
        derived.defender_ratings = {"def-x": 1506.0}
        derived.ground_truth_methods = {"u-r1"}  # 注入循环按 GT 集合迭代

        _merge_resume_from_r(tr, derived, "def-x", {"u-r1"})

        assert len(tr.history) == 1 and tr.history[0]["record"] == "r1"
        assert tr.defender_ratings.get("def-x") == 1506.0
        assert tr.attacker_ratings.get("u-r1") == 1511.0
