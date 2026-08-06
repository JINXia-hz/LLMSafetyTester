# noqa: E402
"""发现层（D+A：D-optimal 种子→指纹→相似度加权池化）回归测试。

验证：
1. model_fingerprint: 指纹计算、相似度（相关系数）、缺数据兜底、probes 存取。
2. BlendPredictor: sim-加权 unified 在有相似 donor 时与均匀 universal 预测不同；
   首模型/无 donor 回退均匀 universal；无指纹 donor 排除。
"""
import tempfile
from pathlib import Path

import numpy as np

from llmsec.core.results import ResultsMatrix
from llmsec.evaluation.blend_predictor import BlendPredictor
from llmsec.evaluation.elo import ELOTracker
from llmsec.evaluation.elo_cluster import EloPredictorModel
from llmsec.evaluation.model_fingerprint import (
    compute_fingerprint,
    donor_similarities,
    load_probes,
    model_similarity,
    save_probe,
)

BLOCKS = EloPredictorModel.BLOCK_ORDER


def _feat(methods, seed=0):
    rng = np.random.default_rng(seed)
    return {m: {b: rng.standard_normal(4) for b in BLOCKS} for m in methods}


# ---------------- model_fingerprint 单元 ----------------

def test_fingerprint_and_similarity():
    """指纹 = per-seed Elo；相似度 = 相关系数；公共方法不足/零方差 → None。"""
    t = ELOTracker()
    for m, e in [("s1", 1600), ("s2", 1500), ("s3", 1700), ("s4", 1550)]:
        t.attacker_ratings[m] = e
    fp = compute_fingerprint(t, ["s1", "s2", "s3", "s4"])
    assert set(fp.keys()) == {"s1", "s2", "s3", "s4"}
    assert fp["s1"] == 1600.0

    # 完全正相关
    fp_b = {"s1": 3200, "s2": 3000, "s3": 3400, "s4": 3100}
    assert abs(model_similarity(fp, fp_b) - 1.0) < 1e-9

    # 公共方法不足 → None
    fp_c = {"s1": 1600, "s2": 1500}
    assert model_similarity(fp, fp_c) is None

    # 一方零方差 → None
    fp_flat = {"s1": 1500, "s2": 1500, "s3": 1500, "s4": 1500}
    assert model_similarity(fp, fp_flat) is None


def test_probes_roundtrip_and_donor_similarities():
    """probes.json 存取；donor_similarities 排除自身/无指纹；min_sim 裁掉低/负相似。"""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "probes.json"
        save_probe("A", {"s1": 1600, "s2": 1500, "s3": 1700}, ["s1", "s2", "s3"], path=p)
        save_probe("B", {"s1": 1620, "s2": 1490, "s3": 1710}, ["s1", "s2", "s3"], path=p)  # 与 A 高正相关
        save_probe("C", {"s1": 1500, "s2": 1700, "s3": 1450}, ["s1", "s2", "s3"], path=p)  # 与 A 负相关
        probes = load_probes(p)
        assert set(probes.keys()) == {"A", "B", "C"}

        # 默认 min_sim=0：B(正)保留，C(负)裁掉，A 自身排除
        sims = donor_similarities("A", probes)
        assert "A" not in sims
        assert "B" in sims and sims["B"] > 0.9
        assert "C" not in sims  # 负相关被 min_sim=0 裁

        # 放开 min_sim → C(负)也出现
        sims_all = donor_similarities("A", probes, min_sim=-1.0)
        assert "C" in sims_all and sims_all["C"] < 0


# ---------------- BlendPredictor sim-加权 ----------------

def _seed_probes(path, models_fps):
    for mdl, fp in models_fps.items():
        save_probe(mdl, fp, list(fp.keys()), path=path)


def test_blend_sim_weighted_differs_and_first_model_fallback():
    """有相似 donor(A)+不相似 donor(C) 时，B 的 sim-加权 unified 偏向 A、压制 C，
    与均匀 universal(A/C 等权)预测不同；首模型无 donor 回退均匀 universal。"""
    with tempfile.TemporaryDirectory() as d:
        probe_path = Path(d) / "probes.json"
        from llmsec.evaluation import model_fingerprint as mf
        orig = mf.PROBES_FILE
        mf.PROBES_FILE = probe_path
        try:
            rng = np.random.default_rng(1)
            methods = [f"m{i}" for i in range(12)]
            base = {m: rng.normal(1500, 80) for m in methods}
            # A 与 base 同模式；C 与 base 反模式（Elo 取负偏移）→ A、C 行为相反
            elo_a = {m: base[m] for m in methods}
            elo_c = {m: (3000 - base[m]) for m in methods}  # 与 A 负相关
            # B 暖启动：只测 6 个，模式同 A
            R = ResultsMatrix()
            for ts, m in enumerate(methods, 1):
                R.upsert(m, "A", round(elo_a[m], 1), ts=ts)
                R.upsert(m, "C", round(elo_c[m], 1), ts=ts)
            for ts, m in enumerate(methods[:6], 1):
                R.upsert(m, "B", round(base[m] + rng.normal(0, 5), 1), ts=ts)
            feats = _feat(methods + ["untested"], seed=2)

            seeds = methods[:4]
            _seed_probes(probe_path, {
                "A": {s: elo_a[s] for s in seeds},
                "B": {s: base[s] + 5 for s in seeds},   # 与 A 高正相关
                "C": {s: elo_c[s] for s in seeds},       # 与 B 负相关 → 被 min_sim=0 裁
            })

            bp = BlendPredictor().fit(R, feats, method_catalog=methods + ["untested"])
            assert "B" in bp.unified, "B 有相似 donor(A) 却未建 sim-加权 unified"
            r_b = bp.predict("untested", "B")
            assert r_b["source"] != "fallback"

            # sim-加权 unified（B 偏向 A、排除 C）vs 均匀 fallback（A/C 等权平均）应不同
            u_mean_sim, _ = BlendPredictor._predict_one(bp, bp.unified["B"], "untested", feats["untested"])
            u_mean_fb, _ = BlendPredictor._predict_one(bp, bp.unified_fallback, "untested", feats["untested"])
            assert u_mean_sim is not None and u_mean_fb is not None
            assert abs(u_mean_sim - u_mean_fb) > 1e-5, (
                f"sim-加权 unified 与均匀 universal 预测相同（未生效）: {u_mean_sim} vs {u_mean_fb}"
            )
        finally:
            mf.PROBES_FILE = orig


def test_first_model_falls_back_to_uniform():
    """首模型（单模型，无 donor）→ 无 unified_fallback（避免冗余训练），
    predict 用 per-model 层（model_only source）。"""
    with tempfile.TemporaryDirectory() as d:
        probe_path = Path(d) / "probes.json"
        from llmsec.evaluation import model_fingerprint as mf
        orig = mf.PROBES_FILE
        mf.PROBES_FILE = probe_path
        try:
            methods = [f"m{i}" for i in range(10)]
            R = ResultsMatrix()
            for ts, m in enumerate(methods, 1):
                R.upsert(m, "solo", 3.0 if ts % 2 else -2.0, ts=ts)
            feats = _feat(methods + ["untested"])
            # 不写任何指纹 → solo 无 donor
            bp = BlendPredictor().fit(R, feats, method_catalog=methods + ["untested"])
            assert "solo" not in bp.unified, "无 donor 却建了 sim-加权 unified"
            # 单模型 → unified_fallback 跳过（无 pooling 意义，避免与 models[target] 冗余训练）
            assert bp.unified_fallback is None, "单模型不应训练 unified_fallback（无跨模型池化）"
            assert "solo" in bp.models, "per-model 预测器应已训练"
            r = bp.predict("untested", "solo")
            assert r["source"] == "model_only", f"单模型应走 model_only，实际 {r['source']}"
        finally:
            mf.PROBES_FILE = orig
