#!/usr/bin/env python3
"""
单元测试：马氏白化空间 + 树聚类拐点 auto-k。

验证：
1. 白化后各保留主成分方向的方差 ≈ 1（轻量马氏距离成立）。
2. k0 随 n 按 log 增长。
3. 已知簇数的合成数据上，auto-k 误差 ≤ 2。
4. run_tree_clustering(write=False) 端到端结构正确。
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from llmsec.clustering.space import build_whitened_space
from llmsec.clustering.tree import (
    build_tree,
    candidate_ks,
    cut_tree,
    log_growth_k0,
    run_tree_clustering,
    select_knee,
    sweep_candidates,
)


def _make_blob_features(n_blobs=6, per_blob=15, seed=42):
    """合成 n_blobs 个高斯团簇的特征 dict。"""
    rng = np.random.default_rng(seed)
    features = {}
    methods = []
    centers = rng.normal(0, 6, size=(n_blobs, 20))
    for b in range(n_blobs):
        for i in range(per_blob):
            m = f"attack_b{b}_{i}"
            methods.append(m)
            features[m] = {
                "textual": rng.normal(0, 0.1, size=12),
                "embedding": centers[b] + rng.normal(0, 0.5, size=20),
                "technique": rng.binomial(1, 0.1, size=5).astype(float),
                "intent": rng.normal(0, 0.1, size=3),
            }
    return features, sorted(methods)


def test_whitening_unit_variance() -> int:
    features, methods = _make_blob_features()
    # damp=1（严格马氏）时强信号方向方差应 ≈ 1，噪声方向被抑制
    space = build_whitened_space(features, methods, damp=1.0)
    coords = space["coords"]
    if coords.shape[0] != len(methods) or coords.shape[1] < 2:
        print(f"❌ 白化坐标形状异常: {coords.shape}")
        return 1
    S = space["singular_values"][: space["n_dims"]]
    strong = S**2 > 10 * space["lambda_w"]
    col_var = coords.var(axis=0)
    if strong.sum() < 2 or not np.allclose(col_var[strong], 1.0, atol=0.15):
        print(f"❌ 全白化后强方向方差不为 1: {col_var[strong][:5]}")
        return 1
    weak = S**2 < 0.1 * space["lambda_w"]
    if weak.sum() > 0 and col_var[weak].max() > 0.5:
        print(f"❌ 噪声方向未被抑制: {col_var[weak][:5]}")
        return 1
    # 默认 damp=0.5（轻量马氏）应保留信噪比：最强方向方差 >> 噪声方向方差
    space_d = build_whitened_space(features, methods)
    col_var_d = space_d["coords"].var(axis=0)
    if col_var_d[0] <= col_var_d[-1] * 2:
        print(f"❌ 阻尼白化未保留信噪比: max={col_var_d[0]:.2f} min={col_var_d[-1]:.2f}")
        return 1
    if space["kept_variance"] < 0.9:
        print(f"❌ 保留方差过低: {space['kept_variance']}")
        return 1
    print(f"✅ 白化空间通过 (n_dims={space['n_dims']}, kept={space['kept_variance']:.3f})")
    return 0


def test_log_growth_k0() -> int:
    cases = [(50, 6), (100, 7), (1000, 10), (10000, 14)]
    for n, expected in cases:
        got = log_growth_k0(n)
        if got != expected:
            print(f"❌ log_growth_k0({n}) = {got}, 期望 {expected}")
            return 1
    # n 翻倍 → k0 +1
    if log_growth_k0(200) - log_growth_k0(100) != 1:
        print("❌ k0 未按 log 增长")
        return 1
    ks = candidate_ks(100)
    if not ks or min(ks) < 2 or max(ks) > 99 or ks != sorted(set(ks)):
        print(f"❌ 候选 k 异常: {ks}")
        return 1
    print("✅ log 增长聚类量通过")
    return 0


def test_auto_k_on_blobs() -> int:
    features, methods = _make_blob_features(n_blobs=6)
    space = build_whitened_space(features, methods)
    Z = build_tree(space["coords"])
    sweep = sweep_candidates(space["coords"], Z, methods)
    k_best, top3 = select_knee(sweep)
    if not (4 <= k_best <= 8):
        print(f"❌ auto-k={k_best} 偏离真实簇数 6: sweep={[(s['k'], s['score']) for s in sweep]}")
        return 1
    # 切出的簇应与真实 blob 高度一致：同 blob 方法同簇率
    labels = cut_tree(Z, methods, k_best)
    same = 0
    total = 0
    for i, m1 in enumerate(methods):
        for m2 in methods[i + 1:]:
            b1, b2 = m1.split("_")[1], m2.split("_")[1]
            if b1 == b2:
                total += 1
                if labels[m1] == labels[m2]:
                    same += 1
    purity = same / max(total, 1)
    if purity < 0.9:
        print(f"❌ 同 blob 同簇率过低: {purity:.2%}")
        return 1
    print(f"✅ auto-k 拐点通过 (k*={k_best}, top3={top3}, 同簇率={purity:.1%})")
    return 0


def test_run_tree_clustering_e2e() -> int:
    features, methods = _make_blob_features()
    meta = {
        "method_names": methods,
        "method_prompts": {m: f"prompt for {m}" for m in methods},
        "technique_label_names": ["t0", "t1", "t2", "t3", "t4"],
        "textual_feature_names": [f"tx{i}" for i in range(12)],
        "defense_feature_names": [f"df{i}" for i in range(14)],
    }
    report = run_tree_clustering(features, meta, write=False)
    if report.get("n_clusters", 0) < 2:
        print(f"❌ 端到端聚类失败: {report.get('error')}")
        return 1
    labels = report.get("method_labels", {})
    if len(labels) != len(methods):
        print(f"❌ method_labels 数量不符: {len(labels)}/{len(methods)}")
        return 1
    names = report.get("cluster_names", {})
    if len(names) < report["n_clusters"]:
        print(f"❌ 存在未命名簇: {len(names)}/{report['n_clusters']}")
        return 1
    if not report.get("top_ks") or not report.get("candidate_sweep"):
        print("❌ 缺少 top_ks / candidate_sweep")
        return 1
    print(f"✅ 端到端树聚类通过 (k={report['n_clusters']}, silhouette={report['validation']['silhouette']})")
    return 0


def test_d_optimal_coverage() -> int:
    """冷启动（无 GT）时 D-optimal 种子的特征空间覆盖应优于随机采样。"""
    from llmsec.evaluation.active_learning import greedy_d_optimal

    features, methods = _make_blob_features(n_blobs=6, per_blob=10)
    space = build_whitened_space(features, methods)
    X = space["coords"]
    n_seeds = 8

    idx = greedy_d_optimal(X, n_seeds, lam=1.0)
    if len(idx) != n_seeds:
        print(f"❌ 种子数不符: {len(idx)}/{n_seeds}")
        return 1

    def min_pairwise_dist(indices):
        pts = X[list(indices)]
        d_min = float("inf")
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = float(np.linalg.norm(pts[i] - pts[j]))
                d_min = min(d_min, d)
        return d_min

    d_opt = min_pairwise_dist(idx)
    rng = np.random.default_rng(0)
    d_rand = max(
        min_pairwise_dist(rng.choice(len(methods), n_seeds, replace=False))
        for _ in range(20)
    )
    if d_opt < d_rand:
        print(f"❌ D-optimal 覆盖不如随机最优: {d_opt:.2f} < {d_rand:.2f}")
        return 1

    # 有 GT 时，新选点应与 GT 信息互补（杠杆大于 GT 内部平均杠杆）
    gt_idx = list(range(10))
    M = X[gt_idx].T @ X[gt_idx] + 1.0 * np.eye(X.shape[1])
    M_inv = np.linalg.inv(M)
    from llmsec.evaluation.active_learning import d_optimal_scores
    picked = greedy_d_optimal(X, 3, lam=1.0, X_gt=X[gt_idx])
    s_picked = d_optimal_scores(X[picked], M_inv).mean()
    s_gt = d_optimal_scores(X[gt_idx], M_inv).mean()
    if s_picked <= s_gt:
        print(f"❌ 新选点杠杆未超过 GT 平均: {s_picked:.3f} <= {s_gt:.3f}")
        return 1

    print(f"✅ D-optimal 种子通过 (覆盖 {d_opt:.2f} ≥ 随机最优 {d_rand:.2f}, "
          f"新点杠杆 {s_picked:.2f} > GT均值 {s_gt:.2f})")
    return 0


def main() -> int:
    tests = [
        test_whitening_unit_variance,
        test_log_growth_k0,
        test_auto_k_on_blobs,
        test_run_tree_clustering_e2e,
        test_d_optimal_coverage,
    ]
    for t in tests:
        if t() != 0:
            return 1
    print("\n✅ 所有白化空间/树聚类测试通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
