#!/usr/bin/env python3
"""
回归测试：BlendPredictor（统一 + 模型双层预测，自适应权重）。

验证 P2：
1. 已测方法 → ground_truth 直返（std=0）。
2. 未测方法 → blend（统一+模型混合，0<w_model<1）。
3. 全新模型（无数据）→ unified_only（w_model=0，全靠统一先验）。
4. 自适应权重：样本越少越偏向统一预测（贝叶斯收缩）。
"""

import sys


import numpy as np

from llmsec.core.results import ResultsMatrix
from llmsec.evaluation.blend_predictor import BlendPredictor
from llmsec.params import BLEND_PRIOR_K
from llmsec.evaluation.elo_cluster import EloPredictorModel

BLOCKS = EloPredictorModel.BLOCK_ORDER


def _feat(methods, seed=0):
    rng = np.random.default_rng(seed)
    return {m: {b: rng.standard_normal(4) for b in BLOCKS} for m in methods}


def test_prediction_paths():
    R = ResultsMatrix()
    for ts in range(1, 11):
        R.upsert("m%d" % ts, "qwen", 3.0 if ts <= 5 else -2.0, ts=ts)
    catalog = ["m%d" % ts for ts in range(1, 11)] + ["untested_x"]
    bp = BlendPredictor().fit(R, _feat(catalog), method_catalog=catalog)

    r1 = bp.predict("m1", "qwen")
    if r1["source"] != "ground_truth" or r1["std"] != 0.0:
        print("❌ 已测方法应 ground_truth/std=0:", r1); return 1
    r2 = bp.predict("untested_x", "qwen")
    if r2["source"] != "blend" or not (0 < r2["w_model"] < 1):
        print("❌ 未测方法应 blend:", r2); return 1
    r3 = bp.predict("m1", "brand_new")
    if r3["w_model"] != 0.0 or r3["source"] != "unified_only":
        print("❌ 全新模型应 unified_only:", r3); return 1


def test_adaptive_weights():
    """样本越少 → w_model 越小（更偏统一）。"""
    R = ResultsMatrix()
    for ts in range(1, 31):
        R.upsert("big_%d" % ts, "big", 3.0 if ts % 2 else -2.0, ts=ts)
    for ts in range(1, 4):
        R.upsert("sml_%d" % ts, "small", 3.0, ts=ts)
    catalog = ["big_%d" % ts for ts in range(1, 31)] + ["sml_%d" % ts for ts in range(1, 4)]
    bp = BlendPredictor().fit(R, _feat(catalog), method_catalog=catalog)
    w_big = bp.blend_weights("big")[0]
    w_small = bp.blend_weights("small")[0]
    if w_small >= w_big:
        print(f"❌ 少样本应更偏统一: w_small={w_small} >= w_big={w_big}"); return 1
    # 公式校验：w_m = n/(n+K)
    import math
    if not math.isclose(w_big, 30 / (30 + BLEND_PRIOR_K), abs_tol=1e-9):
        print(f"❌ w_big 公式不符: {w_big}"); return 1


def test_no_data_fallback():
    """两层都无数据 → fallback 返回初始 Elo + 大 std。"""
    bp = BlendPredictor()  # 未 fit
    r = bp.predict("any", "any")
    if r["source"] != "fallback" or r["elo"] != 1500.0:
        print("❌ 空预测器应 fallback/1500:", r); return 1


