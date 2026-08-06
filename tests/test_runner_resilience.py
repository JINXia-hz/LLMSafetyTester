"""回归测试：runner/safe_twin 的隔离与韧性（审查 S-3/M-30）。

覆盖：
1. S-3：过敏检测结果按模型隔离——换 TARGET_MODEL 重跑时 done_ids 不跳过、汇总不串味。
2. M-30：聚类在 <2 方法时 run_hdbscan_clustering 返回 error 且不写文件。
"""

import json
from types import SimpleNamespace as NS

import llmsec.clustering.hdb as hdb
import llmsec.evaluation.safe_twin as st
from llmsec.clustering import run_hdbscan_clustering
from llmsec.core.io import read_jsonl


def _make_fake_client(counter):
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
    cr = tmp_path / "cluster_result.pkl"
    monkeypatch.setattr(hdb, "CLUSTER_RESULT_FILE", cr)
    features = {"only_method": {"textual": [0.0]}}
    meta = {"method_names": ["only_method"]}
    report = run_hdbscan_clustering(features, meta, write=True)
    assert report.get("error")
    assert not cr.exists()
