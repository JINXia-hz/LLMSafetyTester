"""Combined tests: Runner/Pipeline（集成持久化 + 韧性 + P1 回归）。"""

# ===== from test_runner_integration.py =====
from types import SimpleNamespace as NSimport llmsec.core.config as cfgimport llmsec.core.results as resimport llmsec.evaluation.elo_cluster as ecimport llmsec.pipeline.allergy_phase as apimport llmsec.pipeline.runner as rnimport llmsec.targets as tmodfrom llmsec.core.io import read_json, read_jsonl, write_jsonlfrom llmsec.evaluation.elo import ELOTrackerclass _Spy:
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

# ===== from test_runner_resilience.py =====
import jsonimport pytestimport llmsec.clustering.hdb as hdbimport llmsec.evaluation.safe_twin as stfrom llmsec.clustering import run_hdbscan_clusteringdef _make_fake_client(counter):
    class _Completions:
        def create(self, model, messages, **kw):
            counter[0] += 1
            return NS(choices=[NS(message=NS(content="Here is the safe information you requested."))])

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class _Client:
        def __init__(self):
            self.chat = _Chat()

    return _Client()


class _FakeJudge:
    def __init__(self, *a, **kw):
        pass

    def evaluate(self, prompt, response):
        return {"is_refusal": False, "is_harmful": False, "compliance_level": "A"}


def test_fpr_per_model_isolation(tmp_path, monkeypatch):
    """S-3：换模型重跑过敏检测不跳过、汇总按模型隔离。"""
    counter = [0]
    twins = [
        {"original_id": "1", "method": "m1", "harm_type": "fraud", "safe_prompt": "how to bake a cake"},
        {"original_id": "2", "method": "m2", "harm_type": "cybercrime", "safe_prompt": "how to learn python"},
    ]
    twin_file = tmp_path / "twins.jsonl"
    twin_file.write_text("\n".join(json.dumps(t) for t in twins), encoding="utf-8")
    result_file = tmp_path / "allergy.jsonl"
    monkeypatch.setattr(st, "TWIN_FILE", twin_file)
    monkeypatch.setattr(st, "TWIN_RESULT_FILE", result_file)
    monkeypatch.setattr(st, "API_DELAY", 0)
    monkeypatch.setattr(st, "create_openai_client", lambda **kw: _make_fake_client(counter))
    monkeypatch.setattr(st, "create_judge_client", lambda: None)
    monkeypatch.setattr(st, "Judge", _FakeJudge)

    st.TARGET_MODEL = "modelA"
    st.evaluate_allergy()
    assert counter[0] == 2
    rows_a = [r for r in read_jsonl(result_file) if r.get("model") == "modelA"]
    assert len(rows_a) == 2

    st.TARGET_MODEL = "modelB"
    st.evaluate_allergy()
    assert counter[0] == 4  # modelB 重测（不被 modelA 的 done_ids 跳过）
    rows_b = [r for r in read_jsonl(result_file) if r.get("model") == "modelB"]
    assert len(rows_b) == 2
    assert all(r.get("model") == "modelB" for r in rows_b)


def test_hdbscan_single_method_returns_error(tmp_path, monkeypatch):
    """M-30：<2 方法时 run_hdbscan_clustering 返回 error 且不写文件。"""
    pytest.importorskip("hdbscan")  # 可选依赖：CI 未装时跳过（hdbscan 是惰性/可选）
    cr = tmp_path / "cluster_result.pkl"
    monkeypatch.setattr(hdb, "CLUSTER_RESULT_FILE", cr)
    features = {"only_method": {"textual": [0.0]}}
    meta = {"method_names": ["only_method"]}
    report = run_hdbscan_clustering(features, meta, write=True)
    assert report.get("error")
    assert not cr.exists()

# ===== from test_p1_runner.py =====
import argparseimport astimport inspectimport subprocessimport sysfrom pathlib import PathROOT = Path(__file__).resolve().parent.parent
from llmsec.pipeline import runnerdef test_h2_no_toplevel_reexec():
    """H2: re-exec 不在模块顶层；__main__ 块内保留 re-exec 且透传退出码。"""
    assert 'llmsec.pipeline.runner' in sys.modules, 'import llmsec.pipeline.runner 未触发 re-exec 杀进程'
    tree = ast.parse(inspect.getsource(runner))
    top_level_run = [node for node in tree.body if isinstance(node, ast.Expr) and 'subprocess' in ast.dump(node)]
    assert not top_level_run, '模块顶层无 subprocess.run（re-exec 已移出）'
    main_guard = next((node for node in tree.body if isinstance(node, ast.If) and '__main__' in ast.dump(node.test)), None)
    assert main_guard is not None, "存在 if __name__ == '__main__' 块"
    if main_guard is not None:
        body = ast.dump(main_guard)
        assert 'subprocess' in body, '__main__ 块内保留 .venv re-exec'
        assert 'returncode' in body, 're-exec 透传子进程退出码 sys.exit(proc.returncode)'

def test_h3_max_rounds_validation():
    """H3: _positive_int 单测 + round_idx 兜底源码断言 + subprocess 端到端。"""
    assert runner._positive_int('1') == 1, "_positive_int 接受 '1'"
    assert runner._positive_int('10') == 10, "_positive_int 接受 '10'"
    for bad in ('0', '-2', 'abc'):
        try:
            runner._positive_int(bad)
            assert False, f'_positive_int 应拒绝 {bad!r}'
        except argparse.ArgumentTypeError:
            print(f'✅ _positive_int 拒绝 {bad!r}（ArgumentTypeError）')
    src = inspect.getsource(runner.run_attack_phase)
    loop_pos = src.index('for round_idx in range(')
    assert 'round_idx = 0' in src[:loop_pos], 'run_attack_phase 循环前有 round_idx = 0 兜底'
    proc = subprocess.run([sys.executable, '-m', 'llmsec.pipeline.runner', '--max-rounds', '0'], cwd=ROOT, capture_output=True, timeout=120)
    stderr = proc.stderr.decode('utf-8', errors='replace')
    assert proc.returncode != 0, f'--max-rounds 0 退出码非 0（实际 {proc.returncode}）'
    assert 'error' in stderr and '--max-rounds' in stderr, 'stderr 报 argparse 校验错误'

def test_h4_tested_resume_init():
    src = inspect.getsource(runner.run_attack_phase)
    assert 'tested = set(tracker.ground_truth_methods)' in src, 'tested 从 tracker.ground_truth_methods 初始化（源码断言）'

    class _StubTracker:
        ground_truth_methods = {'m1', 'm2'}
    tracker = _StubTracker()
    tested = set(tracker.ground_truth_methods)
    all_methods = ['m1', 'm2', 'm3']
    untested = [m for m in all_methods if m not in tested]
    assert untested == ['m3'], 'stub: 已实测方法 m1/m2 不会被二次选中，仅 m3 待测'

def test_h1_input_default_compat():
    """H1: --input 默认值 attacks/l1.jsonl（旧名兼容映射已随 v1.0 清理移除）。"""
    src = inspect.getsource(runner)
    assert 'default="attacks/l1.jsonl"' in src
    assert '攻击集_L1.jsonl' not in src, '旧名 兼容映射应已移除（v1.0 不向后兼容）'
    from llmsec.core.config import ATTACK_SET_L1_FILE
    assert ATTACK_SET_L1_FILE.name == 'l1.jsonl', 'ATTACK_SET_L1_FILE 指向 l1.jsonl'
