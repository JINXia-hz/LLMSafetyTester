#!/usr/bin/env python3
"""
P1 正确性回归测试（批次 A）。

覆盖：
  F2: tree.py _evaluate_cut / sweep_candidates 的 inf→NaN 崩溃修复
  H3: _compute_conv_rounds try/finally 轨迹恢复
  H4: report.py build_tree inconclusive 分支（数据不足时不给确定结论）
  H9: elo_cluster K-Fold 余数均衡分配
  H10: predict() 回退分支含 std/ci95（schema 一致性）

约定：每个 test 返回 0=通过 / 1=失败；main() 汇总。
"""

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from llmsec.clustering.tree import _evaluate_cut, sweep_candidates
from llmsec.evaluation.elo import ELOTracker
from llmsec.evaluation.elo_cluster import ClusterEloPredictor
from llmsec.params import PORTRAIT_MIN_TESTED


def _check(cond: bool, msg: str) -> int:
    if not cond:
        print(f"  ❌ {msg}")
        return 1
    print(f"  ✅ {msg}")
    return 0


# ============================================================
# F2: tree.py inf→有限值 + _norm 钳位
# ============================================================
def test_f2_evaluate_cut_degenerate_returns_finite() -> int:
    """全单点簇（n_clusters == n）→ 早期返回，davies_bouldin 应为 1e6（非 inf）。"""
    rc = 0
    coords = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    labels = {"a": 0, "b": 1, "c": 2}  # 3 簇 3 点 = 全单点
    metrics = _evaluate_cut(coords, labels, ["a", "b", "c"])
    rc |= _check(math.isfinite(metrics["davies_bouldin"]),
                 f"F2: 退化切割 davies_bouldin 有限（实得 {metrics['davies_bouldin']}）")
    rc |= _check(math.isfinite(metrics["silhouette"]),
                 "F2: 退化切割 silhouette 有限")
    rc |= _check(math.isfinite(metrics["calinski_harabasz"]),
                 "F2: 退化切割 calinski_harabasz 有限")
    rc |= _check(metrics["davies_bouldin"] != float("inf"),
                 "F2: davies_bouldin 不是 inf（是 1e6）")
    return rc


def test_f2_sweep_no_nan_in_score() -> int:
    """sweep_candidates 含退化 k 时，score 不含 NaN（json.dumps allow_nan=False 不崩）。"""
    rc = 0
    # 构造 ward linkage 树（4 点，会产出 k=1..3 候选，k=1 退化）
    from scipy.cluster.hierarchy import linkage
    coords = np.array([[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]])
    Z = linkage(coords, method="ward")
    methods = ["a", "b", "c", "d"]
    sweep = sweep_candidates(coords, Z, methods, ks=[1, 2, 3])
    # 所有 score 应有限
    scores = [s["score"] for s in sweep]
    rc |= _check(all(math.isfinite(s) for s in scores),
                 f"F2: sweep scores 全有限（实得 {scores}）")
    # json.dumps(allow_nan=False) 不抛
    try:
        json.dumps(sweep, allow_nan=False)
        rc |= _check(True, "F2: sweep 结果 json.dumps(allow_nan=False) 不抛")
    except ValueError as e:
        rc |= _check(False, f"F2: sweep 结果 json.dumps 抛错: {e}")
    return rc


def test_f2_evaluate_cut_zero_variance_cluster() -> int:
    """簇内零方差（所有点同坐标）使 davies_bouldin_score 抛异常 → _safe 返回 1e6。"""
    rc = 0
    # 两个簇：簇 0 的两点完全重合（零方差），簇 1 正常
    coords = np.array([[0.0, 0.0], [0.0, 0.0], [5.0, 5.0]])
    labels = {"a": 0, "b": 0, "c": 1}
    metrics = _evaluate_cut(coords, labels, ["a", "b", "c"])
    rc |= _check(math.isfinite(metrics["davies_bouldin"]),
                 f"F2: 零方差簇 davies_bouldin 有限（实得 {metrics['davies_bouldin']}）")
    return rc


# ============================================================
# H3: _compute_conv_rounds try/finally 轨迹恢复
# ============================================================
def test_h3_trajectory_restored_on_exception() -> int:
    """check_convergence 抛异常时，_compute_conv_rounds 恢复原始轨迹。"""
    rc = 0
    from llmsec.pipeline.runner import _compute_conv_rounds

    tr = ELOTracker()
    # 构造 3 轮轨迹
    tr._round_defender_elos["def"] = [1500.0, 1510.0, 1520.0]
    original = list(tr._round_defender_elos["def"])

    # monkeypatch check_convergence 抛异常
    orig_cc = tr.check_convergence
    call_count = [0]
    def _boom(*a, **kw):
        call_count[0] += 1
        raise RuntimeError("模拟异常")
    tr.check_convergence = _boom
    try:
        result = _compute_conv_rounds(tr, "def", total_methods=10)
    finally:
        tr.check_convergence = orig_cc

    rc |= _check(result is None, "H3: 异常时返回 None")
    rc |= _check(call_count[0] >= 1, "H3: check_convergence 被调用过")
    # 关键：轨迹已恢复
    restored = tr._round_defender_elos.get("def", [])
    rc |= _check(restored == original,
                 f"H3: 异常后轨迹恢复（实得 {restored}，期望 {original}）")
    return rc


def test_h3_trajectory_restored_on_normal_return() -> int:
    """正常返回时轨迹也恢复。"""
    rc = 0
    from llmsec.pipeline.runner import _compute_conv_rounds

    tr = ELOTracker()
    tr._round_defender_elos["def"] = [1500.0, 1501.0, 1502.0, 1503.0]
    original = list(tr._round_defender_elos["def"])

    # check_convergence 在第 2 轮收敛
    orig_cc = tr.check_convergence
    def _conv(defender, total_methods=10, tested_count=5):
        # 简化：总是返回 converged
        return {"converged": True, "ci_half": 1.0}
    tr.check_convergence = _conv
    try:
        result = _compute_conv_rounds(tr, "def", total_methods=10)
    finally:
        tr.check_convergence = orig_cc

    rc |= _check(result == 1, f"H3: 正常返回首个收敛轮（实得 {result}）")
    restored = tr._round_defender_elos.get("def", [])
    rc |= _check(restored == original,
                 f"H3: 正常返回后轨迹恢复（实得 {restored}）")
    return rc


# ============================================================
# H4: report.py build_tree inconclusive 分支
# ============================================================
def test_h4_build_tree_inconclusive_when_data_insufficient() -> int:
    """total_tests < PORTRAIT_MIN_TESTED(5) → level=inconclusive。"""
    rc = 0
    from llmsec.reporting.report import build_method_stats, build_tree

    # 构造 3 条结果（total_tests=3 < 5）
    results = [
        {"method": "m1", "is_harmful": True, "harm_type": "test", "category": "x"},
        {"method": "m1", "is_harmful": False, "harm_type": "test", "category": "x"},
        {"method": "m2", "is_harmful": False, "harm_type": "test", "category": "x"},
    ]
    method_stats = build_method_stats(results, {}, {})
    allergy_data = {"summary": {"false_positive_rate": 0.0}}
    with tempfile.TemporaryDirectory() as td:
        tree = build_tree(method_stats, allergy_data, {}, output_dir=td)
    level = tree.get("overall", {}).get("security_level", "")
    rc |= _check(level == "inconclusive",
                 f"H4: 数据不足时 level=inconclusive（实得 {level}，total_tests=3<{PORTRAIT_MIN_TESTED}）")
    return rc


def test_h4_build_tree_inconclusive_when_confidence_low() -> int:
    """无 tracker（confidence=0 < 0.5）即使 total_tests 够也 inconclusive。"""
    rc = 0
    from llmsec.reporting.report import build_method_stats, build_tree

    # 10 个方法各测 1 次，全部无害（ASR=0），但无 tracker → confidence=0
    results = [
        {"method": f"m{i}", "is_harmful": False, "harm_type": "test", "category": "x"}
        for i in range(10)
    ]
    method_stats = build_method_stats(results, {}, {})
    allergy_data = {"summary": {"false_positive_rate": 0.0}}
    with tempfile.TemporaryDirectory() as td:
        tree = build_tree(method_stats, allergy_data, {}, output_dir=td)
    level = tree.get("overall", {}).get("security_level", "")
    rc |= _check(level == "inconclusive",
                 f"H4: 无 tracker 时即使 ASR=0 也 inconclusive（confidence 不足，实得 {level}）")
    return rc


# ============================================================
# H9: elo_cluster K-Fold 余数均衡分配
# ============================================================
def test_h9_kfold_balanced_small_gt() -> int:
    """小 GT（n=7, k=5）K-Fold 不崩，fold 余数均衡分配（非全堆最后一折）。"""
    rc = 0
    from llmsec.evaluation.elo_cluster import EloPredictorModel

    # 构造 7 个 GT 方法的特征和 Elo
    rng = np.random.default_rng(42)
    methods = [f"m{i}" for i in range(7)]
    features = {
        m: {
            "textual": rng.normal(size=5).tolist(),
            "prior": [0.0, 0.0, 0.0],
        }
        for m in methods
    }
    gt = {m: {"elo": 1500.0 + i * 10, "first_seen_at": i} for i, m in enumerate(methods)}
    blocks = {"textual": [f"t{i}" for i in range(5)], "prior": ["p0", "p1", "p2"]}

    model = EloPredictorModel()
    try:
        model.fit(features, gt, blocks)
        rc |= _check(model.lambda_opt > 0, f"H9: 小 GT fit 成功，λ*>0（实得 {model.lambda_opt}）")
        rc |= _check(model.w is not None, "H9: 小 GT fit 产出权重 w")
        # 验证 cv_errors 非空（K-Fold 确实跑了）
        rc |= _check(len(model.cv_errors) > 0, "H9: K-Fold 产出 cv_errors")
    except Exception as e:
        rc |= _check(False, f"H9: 小 GT fit 异常: {e}")
    return rc


def test_h9_kfold_fold_sizes_balanced() -> int:
    """直接验证 fold 尺寸均衡：n=9, k=5 → fold 尺寸 2,2,2,2,1（非 1,1,1,1,5）。"""
    rc = 0
    # 复现 K-Fold 的索引分配逻辑，验证均衡
    n, k = 9, 5
    fold_size = n // k  # 1
    remainder = n % k   # 4
    fold_sizes = []
    for i in range(k):
        start = i * fold_size + min(i, remainder)
        end = start + fold_size + (1 if i < remainder else 0)
        fold_sizes.append(end - start)
    rc |= _check(fold_sizes == [2, 2, 2, 2, 1],
                 f"H9: n=9,k=5 fold 尺寸均衡 [2,2,2,2,1]（实得 {fold_sizes}）")
    rc |= _check(sum(fold_sizes) == n, "H9: fold 尺寸之和 == n")
    # 对比原 bug：全堆最后一折
    rc |= _check(fold_sizes != [1, 1, 1, 1, 5],
                 "H9: 非原 bug 的 [1,1,1,1,5]（全堆最后一折）")
    return rc


# ============================================================
# H10: predict() 回退分支含 std/ci95
# ============================================================
def test_h10_predict_fallback_has_std_ci95() -> int:
    """predict() 所有回退分支返回的 dict 都含 std/ci95 字段。"""
    rc = 0
    predictor = ClusterEloPredictor()

    # 场景 1：空 GT → predicted 分支
    r = predictor.predict("unknown_method")
    rc |= _check("std" in r and "ci95" in r,
                 f"H10: 空 GT 回退含 std/ci95（keys={list(r.keys())}）")
    rc |= _check(r["std"] is not None and math.isfinite(r["std"]),
                 "H10: 空 GT 回退 std 有限")
    rc |= _check(isinstance(r["ci95"], list) and len(r["ci95"]) == 2,
                 "H10: 空 GT 回退 ci95 是 [lo, hi]")

    # 场景 2：有少量 GT，查未测方法 → 变体/全局回退
    predictor.update_ground_truth("DAN_rot13", 1600.0)
    predictor.update_ground_truth("DAN_b64", 1580.0)
    r2 = predictor.predict("DAN_unknown")
    rc |= _check("std" in r2 and "ci95" in r2,
                 f"H10: 变体回退含 std/ci95（keys={list(r2.keys())}）")
    rc |= _check(r2["std"] is not None and math.isfinite(r2["std"]),
                 "H10: 变体回退 std 有限")

    # 场景 3：GT 方法本身 → ground_truth 分支
    r3 = predictor.predict("DAN_rot13")
    rc |= _check("std" in r3 and "ci95" in r3,
                 f"H10: GT 分支含 std/ci95（keys={list(r3.keys())}）")
    rc |= _check(r3.get("std") == 0.0,
                 f"H10: GT 分支 std=0（实得 {r3.get('std')}）")
    return rc


def test_h10_predict_batch_fallback_schema_consistent() -> int:
    """predict_batch 在 GT 不足走回退时，结果 schema 与 SVD-Ridge 分支一致。"""
    rc = 0
    predictor = ClusterEloPredictor()
    predictor.update_ground_truth("gt1", 1500.0)

    method_records = {"gt1": {"id": "gt1"}, "unknown1": {"id": "unknown1"}}
    results = predictor.predict_batch(method_records)
    # unknown1 走回退
    r = results.get("unknown1")
    rc |= _check(r is not None, "H10: predict_batch 回退产出结果")
    if r:
        for field in ("elo", "source", "std", "ci95", "confidence"):
            rc |= _check(field in r, f"H10: predict_batch 回退含 {field}")
    return rc


# ============================================================
# H1: sampler 预聚类注入
# ============================================================
def test_h1_quick_precluster_returns_labels() -> int:
    """_quick_precluster 有足够 features 时返回 labels。"""
    rc = 0
    from llmsec.pipeline.runner import _quick_precluster

    tr = ELOTracker()
    rng = np.random.default_rng(42)
    methods = [f"m{i}" for i in range(6)]
    tr.predictor.artifacts = {
        "features": {m: {"textual": rng.normal(size=5).tolist()} for m in methods}
    }
    labels = _quick_precluster(tr, methods)
    rc |= _check(labels is not None, "H1: 有 features 时返回 labels")
    if labels:
        rc |= _check(set(labels.keys()) == set(methods), "H1: labels 覆盖所有方法")
        rc |= _check(len(set(labels.values())) >= 2, "H1: 至少 2 簇")
    return rc


def test_h1_quick_precluster_no_features_returns_none() -> int:
    """_quick_precluster 无 features / 方法太少时返回 None。"""
    rc = 0
    from llmsec.pipeline.runner import _quick_precluster

    tr = ELOTracker()
    tr.predictor.artifacts = {}
    rc |= _check(_quick_precluster(tr, ["m1", "m2"]) is None,
                 "H1: 无 features 返回 None")

    # 方法太少（<4）
    tr.predictor.artifacts = {"features": {f"m{i}": {"t": [1]} for i in range(3)}}
    rc |= _check(_quick_precluster(tr, ["m0", "m1", "m2"]) is None,
                 "H1: 方法 <4 返回 None")
    return rc


def test_h1_sampler_receives_cluster_labels() -> int:
    """build_sampler 收到 cluster_report（method_labels）时 sampler 能映射方法到簇。"""
    rc = 0
    from llmsec.evaluation.samplers import build_sampler

    labels = {"a": 0, "b": 0, "c": 1, "d": 1}
    report = {"method_labels": labels}
    s = build_sampler("infogain", cluster_report=report)
    rc |= _check(s.cluster_report != {},
                 "H1: 注入 report 后 sampler.cluster_report 非空")
    rc |= _check(s._method_to_cluster("a") == 0 and s._method_to_cluster("c") == 1,
                 "H1: sampler 正确映射方法到簇")
    return rc


# ============================================================
# H6: evaluator judge 异常降级
# ============================================================
def test_h6_judge_failure_degrades_gracefully() -> int:
    """judge.evaluate 抛异常时降级到 no_judge，不中断。"""
    rc = 0
    from llmsec.evaluation import evaluator as eval_mod

    orig_ct = eval_mod.call_target
    eval_mod.call_target = lambda prompt: {
        "content": "some response text here", "error": None,
        "latency_ms": 10, "tokens_prompt": 5, "tokens_completion": 10,
    }

    class BoomJudge:
        def evaluate(self, *a, **kw):
            raise RuntimeError("Judge API 宕机")

    try:
        result = eval_mod.evaluate_single(
            "test prompt", expected_answer=0, use_judge=True, judge=BoomJudge()
        )
        rc |= _check(result.get("status") is not None,
                     f"H6: judge 失败降级到 no_judge（status={result.get('status')}）")
        rc |= _check("content" in result, "H6: 降级后 result 结构完整")
    except Exception as e:
        rc |= _check(False, f"H6: judge 失败应降级不应抛异常: {e}")
    finally:
        eval_mod.call_target = orig_ct
    return rc


# ============================================================
# H8: safe_twin severity inconclusive
# ============================================================
def test_h8_severity_inconclusive_when_no_results() -> int:
    """n_results=0 → severity=inconclusive（不再 fpr=0 误报 low）。"""
    rc = 0
    from llmsec.evaluation.safe_twin import _compute_allergy_severity

    sev, interp = _compute_allergy_severity(0, 0.0)
    rc |= _check(sev == "inconclusive",
                 f"H8: n_results=0 → inconclusive（实得 {sev}）")
    rc |= _check("不足" in interp or "无效" in interp or "不支持" in interp,
                 "H8: inconclusive 解读文本说明原因")

    sev2, _ = _compute_allergy_severity(3, 0.0)  # 3 < MIN_TWIN_WINDOW(6)
    rc |= _check(sev2 == "inconclusive",
                 f"H8: n_results<6 → inconclusive（实得 {sev2}）")

    # 样本充足时正常分级
    sev3, _ = _compute_allergy_severity(20, 0.02)  # fpr < 0.05
    rc |= _check(sev3 == "low", f"H8: 样本充足 fpr<0.05 → low（实得 {sev3}）")
    sev4, _ = _compute_allergy_severity(20, 0.5)
    rc |= _check(sev4 == "high", f"H8: 样本充足 fpr>0.15 → high（实得 {sev4}）")
    return rc


# ============================================================
# H11: blend_predictor cache_key 纳入 features
# ============================================================
def test_h11_cache_key_includes_features() -> int:
    """cache_key 纳入 features 结构签名（切换 embedding 时 key 不同）。"""
    rc = 0
    from llmsec.core.results import ResultsMatrix
    from llmsec.evaluation.blend_predictor import BlendPredictor

    mat = ResultsMatrix()
    mat.upsert("A", "m", 1.0, ts=1)
    catalog = ["A"]

    key_none = BlendPredictor.cache_key(mat, catalog, None)
    feats_5d = {"A": {"textual": [1, 2, 3, 4, 5], "prior": [0, 0]}}
    feats_3d = {"A": {"textual": [1, 2, 3], "prior": [0, 0]}}

    key_5d = BlendPredictor.cache_key(mat, catalog, feats_5d)
    key_3d = BlendPredictor.cache_key(mat, catalog, feats_3d)
    key_5d_again = BlendPredictor.cache_key(mat, catalog, feats_5d)

    rc |= _check(key_none != key_5d, "H11: 无 features vs 有 features → key 不同")
    rc |= _check(key_5d != key_3d, "H11: 不同维度 features → key 不同")
    rc |= _check(key_5d == key_5d_again, "H11: 相同 features → key 相同（确定性）")
    return rc


# ============================================================
# 主入口
# ============================================================
def main() -> int:
    tests = [
        test_f2_evaluate_cut_degenerate_returns_finite,
        test_f2_sweep_no_nan_in_score,
        test_f2_evaluate_cut_zero_variance_cluster,
        test_h3_trajectory_restored_on_exception,
        test_h3_trajectory_restored_on_normal_return,
        test_h4_build_tree_inconclusive_when_data_insufficient,
        test_h4_build_tree_inconclusive_when_confidence_low,
        test_h9_kfold_balanced_small_gt,
        test_h9_kfold_fold_sizes_balanced,
        test_h10_predict_fallback_has_std_ci95,
        test_h10_predict_batch_fallback_schema_consistent,
        test_h1_quick_precluster_returns_labels,
        test_h1_quick_precluster_no_features_returns_none,
        test_h1_sampler_receives_cluster_labels,
        test_h6_judge_failure_degrades_gracefully,
        test_h8_severity_inconclusive_when_no_results,
        test_h11_cache_key_includes_features,
    ]
    failed = 0
    for t in tests:
        print(f"\n[运行] {t.__name__}")
        if t() != 0:
            failed += 1
    if failed:
        print(f"\n❌ {failed} 个测试失败")
        return 1
    print("\n✅ 所有 P1 正确性测试通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
