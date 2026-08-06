"""回归测试：Elo / 聚类预测器的边界与健壮性（审查 S-2/M-3/M-7/M-4）。

覆盖：
1. S-2：compute_security_boundary 在防御方零有效场次时返回完整键集（不 KeyError）。
2. M-3：ELOTracker.update 接受数字字符串 eval_score（回写 float，不再 TypeError）。
3. M-7：EloPredictorModel 在 GT Elo 全相同（σ²=0）时下限保护（>0，不产零宽 CI）。
4. M-4：ClusterEloPredictor._load_artifacts 优先读 cluster_result.pkl（含 labels）。
"""

import tempfile
from pathlib import Path

import joblib
import numpy as np

from llmsec.evaluation.elo import ELOTracker
from llmsec.evaluation.elo_cluster import ClusterEloPredictor, EloPredictorModel


def test_boundary_no_defender_keys():
    """S-2：无防御方场次时 compute_security_boundary 不崩且键完整。"""
    tracker = ELOTracker()
    b = tracker.compute_security_boundary("ghost-defender")
    required = {"boundary_elo", "defender_elo", "converged", "confidence",
                "methods_above_boundary", "tested_above_boundary", "predicted_above_boundary"}
    assert required.issubset(b.keys()), f"早退 dict 缺键: {required - set(b.keys())}"
    assert b["converged"] is False
    assert b["confidence"] == 0.0
    # 旧代码曾 KeyError 的两个键，现在可直接下标
    _ = b["converged"], b["defender_elo"]


def test_update_accepts_string_score():
    """M-3：update 对数字字符串 eval_score 回写 float，不抛 TypeError。"""
    tracker = ELOTracker()
    tracker.update("DAN", "def", "3.5")  # 字符串分数
    assert tracker.get_attacker_elo("DAN") > 1500
    # 非数字字符串 → 视为 0（不崩；score=0 仍登记）
    tracker.update("X", "def", "not-a-number")
    assert "X" in tracker.attacker_ratings


def test_sigma2_floor_on_constant_gt():
    """M-7：GT Elo 全相同时 σ² 有下限（不产 std=0 的绝对确定预测）。"""
    blocks = ("textual", "embedding", "technique", "intent", "prior")
    features = {
        f"m{i}": {b: np.array([float(i) + k * 0.1], dtype=float) for k, b in enumerate(blocks)}
        for i in range(6)
    }
    gt = {f"m{i}": {"elo": 1500.0} for i in range(6)}  # 全相同 Elo → 残差 0
    m = EloPredictorModel()
    m.fit(features, gt)
    assert m.sigma2 >= 1e-6, f"σ² 应有下限 ≥1e-6（得 {m.sigma2}）"


def test_load_artifacts_prefers_cluster_result():
    """M-4：两文件并存时优先 cluster_result.pkl（含 labels），重启后标签不丢。"""
    import llmsec.core.config as cfg
    orig_cr = cfg.CLUSTER_RESULT_FILE
    orig_fc = cfg.FEATURE_CACHE_FILE
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cfg.CLUSTER_RESULT_FILE = tmp / "cluster_result.pkl"
        cfg.FEATURE_CACHE_FILE = tmp / "feature_cache.pkl"
        # feature_cache：仅 features（无 labels）
        joblib.dump({"features": {"m1": {"t": [0.0]}, "m2": {"t": [1.0]}}}, cfg.FEATURE_CACHE_FILE)
        # cluster_result：features + labels —— 应被优先加载
        joblib.dump(
            {"features": {"m1": {"t": [0.0]}, "m2": {"t": [1.0]}},
             "labels": {"m1": 0, "m2": 1}, "kind": "cluster_result"},
            cfg.CLUSTER_RESULT_FILE,
        )
        try:
            pred = ClusterEloPredictor()
            pred._load_artifacts()
            assert pred.artifacts is not None
            assert pred.artifacts.get("labels"), "应优先加载含 labels 的 cluster_result"
        finally:
            cfg.CLUSTER_RESULT_FILE = orig_cr
            cfg.FEATURE_CACHE_FILE = orig_fc
