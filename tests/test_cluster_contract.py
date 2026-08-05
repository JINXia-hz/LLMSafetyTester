"""回归测试：聚类入口的契约与健壮性（审查 M-31/M-32/M-33）。

覆盖：
1. M-31：cluster_analysis 经 R 派生——analyze_clusters 对空 tracker 不崩。
2. M-32：load_and_extract 的 result_file 路径解析与 cli 一致（相对 OUTPUT_DIR）。
3. M-33：extract_all_features 对缺 method/prompt 字段的自带攻击集不崩。
"""


import llmsec.clustering.features as fm
from llmsec.clustering.features import extract_all_features, load_and_extract
from llmsec.evaluation.cluster_analysis import analyze_clusters
from llmsec.evaluation.elo import ELOTracker


def _jl(rows):
    import json
    return "\n".join(json.dumps(x) for x in rows)


def test_analyze_clusters_handles_empty_tracker():
    """M-31：空 R 派生的空 tracker 不应让 analyze_clusters 崩溃。"""
    tracker = ELOTracker()
    analysis = analyze_clusters(tracker)
    assert isinstance(analysis, dict)


def test_load_and_extract_result_file_resolution(tmp_path, monkeypatch):
    """M-32：result_file 相对路径按 PROJECT_ROOT 解析（与 cli 一致）。"""
    monkeypatch.setattr(fm, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(fm, "OUTPUT_DIR", tmp_path)
    (tmp_path / "attacks").mkdir()
    (tmp_path / "attacks" / "in.jsonl").write_text(
        _jl([{"id": "1", "method": "m1", "prompt": "hello world test"},
             {"id": "2", "method": "m2", "prompt": "another prompt here"}]), encoding="utf-8")
    (tmp_path / "runs" / "ts").mkdir(parents=True)
    (tmp_path / "runs" / "ts" / "attack_results.jsonl").write_text(
        _jl([{"method": "m1", "eval_score": 3.0, "is_harmful": True},
             {"method": "m2", "eval_score": -1.0, "is_harmful": False}]), encoding="utf-8")
    features, meta = load_and_extract(attack_file="attacks/in.jsonl", result_file="runs/ts/attack_results.jsonl")
    assert set(features.keys()) == {"m1", "m2"}
    assert meta.get("has_eval_data") is True


def test_extract_features_missing_fields():
    """M-33：缺 method（跳过）/ 缺 prompt（兜底空串）的自带攻击集不崩。"""
    records = [
        {"id": "1", "method": "good", "prompt": "a normal attack prompt text"},
        {"id": "2", "method": "no_prompt"},  # 缺 prompt
        {"id": "3", "prompt": "missing method field"},  # 缺 method
    ]
    features, meta = extract_all_features(records, eval_results=[])
    assert "good" in features and "no_prompt" in features  # 缺 method 的被跳过
