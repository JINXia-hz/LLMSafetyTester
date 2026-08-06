"""集成测试：runner 主循环的持久化与韧性（审查 M-11/M-12/M-15/M-36/M-35）。

stub 掉 evaluate_single / call_target / generate_safe_twin（不触网），跑真实的
run_attack_phase / run_allergy_phase / get_or_create_twin / run_multi_target_phase。
"""

from types import SimpleNamespace as NS

import llmsec.core.config as cfg
import llmsec.core.results as res
import llmsec.evaluation.elo_cluster as ec
import llmsec.pipeline.runner as rn
import llmsec.pipeline.allergy_phase as ap
import llmsec.targets as tmod
from llmsec.core.io import read_json, read_jsonl, write_jsonl
from llmsec.evaluation.elo import ELOTracker


class _Spy:
    def __init__(self, real):
        self.real = real
        self.calls = []

    def __call__(self, *a, **k):
        self.calls.append((a, k))
        return self.real(*a, **k)


def _setup(tmp, monkeypatch):
    monkeypatch.setattr(rn, "DEFENDER_NAME", "test-defender")
    monkeypatch.setattr(rn, "SAFE_TWINS_FILE", tmp / "safe_twins.jsonl")
    monkeypatch.setattr(rn, "STATE_DIR", tmp)
    monkeypatch.setattr(res, "RESULTS_FILE", tmp / "results.json")
    monkeypatch.setattr(cfg, "ELO_CACHE_FILE", tmp / "elo_cache.json")


def _fake_eval_success(prompt, expected, judge, use_judge=True):
    return {
        "status": "fully_compliant", "eval_score": 3.0, "content": "here is the harmful answer",
        "is_harmful": True, "is_refusal": False, "math_score": None, "jailbreak_tax": None,
        "compliance_level": "A", "latency_ms": 10,
        "judge_harmfulness": 4, "judge_specificity": 3, "judge_dangerousness": 3,
    }


def test_attack_phase_incremental_persistence(tmp_path, monkeypatch):
    """M-11/M-12：每轮增量写 attack_file + 每轮 publish_tracker。"""
    _setup(tmp_path, monkeypatch)
    attack_file = tmp_path / "attack_results.jsonl"
    pub_spy = _Spy(rn.publish_tracker)
    wj_spy = _Spy(rn.write_jsonl)
    monkeypatch.setattr(rn, "evaluate_single", _fake_eval_success)
    monkeypatch.setattr(rn, "publish_tracker", pub_spy)
    monkeypatch.setattr(rn, "write_jsonl", wj_spy)

    records = [{"id": f"1.{i}.1", "method": f"m{i}", "category": "cat", "harm_type": "fraud",
                "prompt": f"attack prompt number {i}", "expected_answer": 0} for i in range(12)]
    tracker = ELOTracker()
    rn.run_attack_phase(
        records, None, tracker, batch_size=3, max_rounds=2, attack_file=attack_file,
        sampler="gap", sampler_log_file=None, cluster_analysis_file=None,
        skip_final_clustering=True, state_file=str(tmp_path / "state.json"),
    )
    attack_writes = [c for c in wj_spy.calls if c[0] and c[0][0] == attack_file]
    assert attack_writes, "attack_file 应每轮增量写"
    rows = read_jsonl(attack_file)
    assert rows and all("method" in r for r in rows)
    assert pub_spy.calls, "publish_tracker 应每轮同步进 R"


def test_allergy_judge_fallback(tmp_path, monkeypatch):
    """M-15：Phase 2 的 judge.evaluate 抛异常时降级关键词、不崩。"""
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(rn, "call_target",
                        lambda prompt: {"error": None, "content": "a benign safe response", "target_refused": False})
    monkeypatch.setattr(ap, "get_or_create_twin", lambda name, rec, cache, client: f"safe variant of {name}")

    class _BoomJudge:
        def evaluate(self, prompt, response):
            raise RuntimeError("judge API 宕机")

    tracker = ELOTracker()
    tracker.update("m1", "test-defender", 3.0)
    tracker.update("m2", "test-defender", -1.0)
    method_records = {"m1": {"prompt": "p1", "expected_answer": 0}, "m2": {"prompt": "p2", "expected_answer": 0}}
    summary = rn.run_allergy_phase(
        method_records, None, _BoomJudge(), tracker, n_window=2, allergy_file=tmp_path / "allergy.json")
    assert summary.get("fpr") is not None or summary.get("total") is not None


def test_twin_entry_missing_fields(tmp_path, monkeypatch):
    """M-36：get_or_create_twin 对缺 category/harm_type 的记录用 .get 默认，不崩。"""
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(rn, "generate_safe_twin",
                        lambda prompt, client: {"safe_prompt": "a perfectly safe benign request", "replacement": "cake"})
    rec = {"id": "x1", "method": "m_missing", "prompt": "how to do something [MATH:5]"}
    safe = ap.get_or_create_twin("m_missing", rec, {}, None)
    assert safe is not None
    entries = read_jsonl(rn.SAFE_TWINS_FILE)
    assert len(entries) == 1
    e = entries[0]
    assert e.get("category") == "unknown"
    assert e.get("harm_type") == "unknown"
    assert "[MATH:" not in (e.get("original_prompt") or "")


def test_multi_target_canonical_report(tmp_path, monkeypatch):
    """M-35：多目标 run 写 canonical runner_report.json（dashboard 可见）。"""
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(tmod, "available_targets", lambda: {
        "t1": NS(model="mX1", api_key="k", base_url="u"), "t2": NS(model="mX2", api_key="k", base_url="u")})
    monkeypatch.setattr(tmod, "set_active_target", lambda name: None)
    monkeypatch.setattr(ec.ClusterEloPredictor, "fit_features", lambda self, records: setattr(self, "artifacts", {"features": {}}))

    def fake_attack(records, judge, tracker, **kw):
        defender = rn.DEFENDER_NAME
        for i, rec in enumerate(records[:4]):
            tracker.update(rec["method"], defender, 3.0 if i % 2 == 0 else -1.0)
        tracker.record_round_end(defender)
        tracker.save(kw.get("state_file"))
        write_jsonl(kw["attack_file"], [{"method": r["method"], "eval_score": 3.0, "is_harmful": True} for r in records[:4]])
        return {"asr": 0.5, "total_tested": 4}

    monkeypatch.setattr(rn, "run_attack_phase", fake_attack)
    monkeypatch.setattr(rn, "run_allergy_phase", lambda *a, **k: {"fpr": 0.0, "allergic": 0})

    records = [{"id": f"1.{i}.1", "method": f"m{i}", "category": "c", "harm_type": "fraud",
                "prompt": f"prompt {i}", "expected_answer": 0} for i in range(6)]
    method_records = {r["method"]: r for r in records}
    args = NS(targets="t1,t2", phase="all", batch_size=4, max_rounds=1, sampler="gap",
              sampler_alpha=20.0, sampler_beta=5.0, sampler_gamma=10.0, coordinate_rounds=2,
              ridge_refit_threshold=10, refresh_features=False, twin_window=4)
    runs_dir = tmp_path / "runs" / "ts"
    runs_dir.mkdir(parents=True)
    rn.run_multi_target_phase(args, records, method_records, runs_dir, judge=None)

    assert (runs_dir / "multi_target_report.json").exists()
    canon = runs_dir / "runner_report.json"
    assert canon.exists(), "canonical runner_report.json 应写入（M-35，看板可见性）"
    rep = read_json(canon)
    assert rep.get("mode") == "multi_target"
    assert rep.get("target_model") in ("t1", "t2")
    assert "attack_phase" in rep and "elo" in rep and "allergy" in rep
