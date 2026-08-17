"""簇粒度（unit）改造的核心测试：unit 构建/指纹/特征聚合 + R v2 记录级观测 + unit 回放聚合。"""

import json

import numpy as np

from llmsec.core.results import ResultsMatrix
from llmsec.core.units import (
    assemble_units,
    build_unit_features,
    build_unit_proxy_records,
    build_units,
    unit_fingerprint,
)


def _rec(m, i, prompt=None):
    return {"id": f"{m}-r{i}", "method": m, "prompt": prompt or f"{m} 的 prompt {i}",
            "expected_answer": "42", "harm_type": "other"}


def _fixture():
    # 3 方法：A(2 条记录)、B(1 条)、C(1 条)；labels: A+B 同簇，C 另一簇
    methods = ["A", "B", "C"]
    method_records = {m: _rec(m, 0) for m in methods}
    method_pool = {"A": [_rec("A", 0), _rec("A", 1)], "B": [_rec("B", 0)], "C": [_rec("C", 0)]}
    labels = {"A": 0, "B": 0, "C": 1}
    return method_records, method_pool, labels


def test_unit_fingerprint_deterministic():
    fp1 = unit_fingerprint(["A", "B"])
    fp2 = unit_fingerprint(["B", "A"])   # 成员顺序无关
    fp3 = unit_fingerprint(["A", "C"])
    assert fp1 == fp2 and fp1.startswith("c_")
    assert fp1 != fp3, "不同成员集必须不同指纹"
    print("✅ unit 指纹确定性通过")


def test_build_units_basic():
    method_records, method_pool, labels = _fixture()
    units = build_units(labels, method_records, method_pool=method_pool)
    assert len(units) == 2
    u0 = units[unit_fingerprint(["A", "B"])]
    assert u0["size"] == 2 and u0["label"] == 0
    assert len(u0["pool"]) == 3, "pool = A 的 2 条 + B 的 1 条"
    u1 = units[unit_fingerprint(["C"])]
    assert u1["size"] == 1 and len(u1["pool"]) == 1
    print("✅ build_units 分组/pool 通过")


def test_build_units_medoid():
    method_records, method_pool, labels = _fixture()
    coords = {"A": np.array([0.0, 0.0]), "B": np.array([1.0, 0.0]), "C": np.array([9.0, 9.0])}
    units = build_units(labels, method_records, method_pool=method_pool, coords=coords)
    u0 = units[unit_fingerprint(["A", "B"])]
    # 质心 (0.5,0)，A(0,0) 距离 0.5、B(1,0) 距离 0.5 —— 等距取第一个也无妨，
    # 换一个非对称坐标验证真正 medoid
    coords2 = {"A": np.array([0.0, 0.0]), "B": np.array([2.0, 0.0]), "C": np.array([9.0, 9.0])}
    units2 = build_units(labels, method_records, method_pool=method_pool, coords=coords2)
    assert units2[unit_fingerprint(["A", "B"])]["medoid"] == "A", "A 距质心(1,0)更近"
    assert u0["medoid"] in ("A", "B")
    print("✅ medoid 计算通过")


def test_build_unit_features_centroid():
    method_records, method_pool, labels = _fixture()
    units = build_units(labels, method_records, method_pool=method_pool)
    feats = {
        "A": {"textual": np.array([1.0, 3.0]), "embedding": np.array([0.0, 2.0])},
        "B": {"textual": np.array([3.0, 5.0]), "embedding": np.array([2.0, 4.0])},
        "C": {"textual": np.array([9.0, 9.0])},
    }
    uf = build_unit_features(feats, units)
    u0 = uf[unit_fingerprint(["A", "B"])]
    assert np.allclose(u0["textual"], [2.0, 4.0]), "unit 特征 = 成员均值"
    assert np.allclose(u0["embedding"], [1.0, 3.0])
    assert "embedding" not in uf[unit_fingerprint(["C"])], "缺块不补零（保持块缺失语义）"
    print("✅ unit 特征质心聚合通过")


def test_unit_proxy_records():
    method_records, method_pool, labels = _fixture()
    units = build_units(labels, method_records, method_pool=method_pool)
    proxies = build_unit_proxy_records(units)
    for uid, proxy in proxies.items():
        assert proxy["method"] == uid and proxy["id"] == uid, "代理记录键改写为 unit_id"
        assert "prompt" in proxy
    print("✅ unit 代理记录通过")


def test_assemble_units_with_features():
    """assemble_units：带特征 artifacts 时产出 medoid + 命名，不崩。"""
    method_records, method_pool, labels = _fixture()
    feats = {m: {"textual": np.array([float(i), 1.0])} for i, m in enumerate(["A", "B", "C"])}
    artifacts = {"features": feats, "meta": {"method_prompts": {m: "p" for m in feats}}}
    units = assemble_units(labels, method_records, method_pool, artifacts)
    assert len(units) == 2
    assert all(u["name"] for u in units.values()), "命名非空"
    print("✅ assemble_units 通过")


def test_r_v2_unit_aggregation(tmp_path):
    """R v2：行键=记录 id、extra.unit=簇；derive_elo 按 unit 聚合同一簇多条观测。"""
    from llmsec.evaluation.elo import derive_elo

    R = ResultsMatrix()
    # unit c_x 两条观测（不同 prompt 记录），unit c_y 一条
    R.upsert("rec-1", "modelA", 3.0, ts=1, extra={"unit": "c_x", "round": 1})
    R.upsert("rec-2", "modelA", -2.0, ts=2, extra={"unit": "c_x", "round": 1})
    R.upsert("rec-3", "modelA", 1.0, ts=3, extra={"unit": "c_y", "round": 2})

    assert R.tested_records("modelA") == {"rec-1", "rec-2", "rec-3"}
    assert R.tested_units("modelA") == {"c_x", "c_y"}, "tested_units 按 extra.unit 聚合"

    tracker = derive_elo(R, "modelA", unit_catalog=["c_x", "c_y", "c_z"])
    assert set(tracker.ground_truth_methods) == {"c_x", "c_y"}, "GT 键 = unit"
    assert "c_z" in tracker.attacker_ratings, "catalog 注入未测单位初始 Elo"
    # c_x 有两条观测：attacker_stats 累积 2 场
    assert tracker.attacker_stats["c_x"]["n_matches"] == 2

    # 持久化 round-trip
    p = tmp_path / "results.json"
    R.save(p)
    R2 = ResultsMatrix.load(p)
    assert R2.tested_units("modelA") == {"c_x", "c_y"}
    print("✅ R v2 记录级观测 + unit 回放聚合通过")


def test_r_v1_archived(tmp_path):
    """旧 schema（v1 method 键）load 时归档为 results.method-era.bak 并返回空矩阵。"""
    p = tmp_path / "results.json"
    p.write_text(json.dumps({"version": 1, "methods": ["DAN"], "models": ["qwen"],
                             "results": {"DAN": {"qwen": {"eval_score": 3.0}}}}),
                 encoding="utf-8")
    from llmsec.storage import rstore
    R = rstore.matrix_from_legacy_json(p)
    assert R.n_for_model("qwen") == 0, "v1 数据不迁移（废弃重建）"
    assert (tmp_path / "results.method-era.bak").exists(), "v1 文件已归档"
    print("✅ R v1 归档废弃通过")
