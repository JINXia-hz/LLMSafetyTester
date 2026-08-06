"""
单元测试：阻尼白化空间 + HDBSCAN 聚类 + 层选择 auto-k + 后验验证。

验证：
1. 白化后强信号方向方差 ≈ 1、噪声方向被抑制（轻量马氏距离成立）。
2. k0 随 n 按 log 增长。
3. 已知簇数的合成数据上，auto-k 误差 ≤ 2（HDBSCAN single-linkage 树）。
4. run_hdbscan_clustering 端到端结构正确（含 ANOVA 簇效验证）。
5. 弱监督特征权重放大相关特征；D-optimal 种子覆盖优于随机。
"""
import numpy as np
import pytest

from llmsec.clustering.space import build_whitened_space
from llmsec.clustering.tree import candidate_ks, cut_tree, log_growth_k0, select_knee, sweep_candidates


def _make_blob_features(n_blobs=6, per_blob=15, seed=42):
    """合成 n_blobs 个高斯团簇的特征 dict。"""
    rng = np.random.default_rng(seed)
    features = {}
    methods = []
    centers = rng.normal(0, 6, size=(n_blobs, 20))
    for b in range(n_blobs):
        for i in range(per_blob):
            m = f'attack_b{b}_{i}'
            methods.append(m)
            features[m] = {'textual': rng.normal(0, 0.1, size=12), 'embedding': centers[b] + rng.normal(0, 0.5, size=20), 'technique': rng.binomial(1, 0.1, size=5).astype(float), 'intent': rng.normal(0, 0.1, size=3)}
    return (features, sorted(methods))

def test_whitening_unit_variance():
    features, methods = _make_blob_features()
    space = build_whitened_space(features, methods, damp=1.0)
    coords = space['coords']
    assert not (coords.shape[0] != len(methods) or coords.shape[1] < 2), f'❌ 白化坐标形状异常: {coords.shape}'
    S = space['singular_values'][:space['n_dims']]
    strong = 10 * space['lambda_w'] < S ** 2
    col_var = coords.var(axis=0)
    assert not (strong.sum() < 2 or not np.allclose(col_var[strong], 1.0, atol=0.15)), f'❌ 全白化后强方向方差不为 1: {col_var[strong][:5]}'
    weak = 0.1 * space['lambda_w'] > S ** 2
    assert not (weak.sum() > 0 and col_var[weak].max() > 0.5), f'❌ 噪声方向未被抑制: {col_var[weak][:5]}'
    space_d = build_whitened_space(features, methods)
    col_var_d = space_d['coords'].var(axis=0)
    assert not col_var_d[0] <= col_var_d[-1] * 2, f'❌ 阻尼白化未保留信噪比: max={col_var_d[0]:.2f} min={col_var_d[-1]:.2f}'
    assert not space['kept_variance'] < 0.9, f"❌ 保留方差过低: {space['kept_variance']}"
    print(f"✅ 白化空间通过 (n_dims={space['n_dims']}, kept={space['kept_variance']:.3f})")

def test_log_growth_k0():
    cases = [(50, 6), (100, 7), (1000, 10), (10000, 14)]
    for n, expected in cases:
        got = log_growth_k0(n)
        assert not got != expected, f'❌ log_growth_k0({n}) = {got}, 期望 {expected}'
    assert not log_growth_k0(200) - log_growth_k0(100) != 1, '❌ k0 未按 log 增长'
    ks = candidate_ks(100)
    assert not (not ks or min(ks) < 2 or max(ks) > 99 or (ks != sorted(set(ks)))), f'❌ 候选 k 异常: {ks}'
    print('✅ log 增长聚类量通过')

def test_auto_k_on_blobs():
    features, methods = _make_blob_features(n_blobs=6)
    space = build_whitened_space(features, methods)
    from scipy.cluster.hierarchy import linkage
    Z = linkage(space['coords'], method='ward')
    sweep = sweep_candidates(space['coords'], Z, methods)
    k_best, top3 = select_knee(sweep)
    assert 4 <= k_best <= 8, f"❌ auto-k={k_best} 偏离真实簇数 6: sweep={[(s['k'], s['score']) for s in sweep]}"
    labels = cut_tree(Z, methods, k_best)
    same = 0
    total = 0
    for i, m1 in enumerate(methods):
        for m2 in methods[i + 1:]:
            b1, b2 = (m1.split('_')[1], m2.split('_')[1])
            if b1 == b2:
                total += 1
                if labels[m1] == labels[m2]:
                    same += 1
    purity = same / max(total, 1)
    assert not purity < 0.85, f'❌ 同 blob 同簇率过低: {purity:.2%}'
    print(f'✅ auto-k 拐点通过 (k*={k_best}, top3={top3}, 同簇率={purity:.1%})')

def test_run_hdbscan_clustering_e2e():
    """HDBSCAN 主管线端到端：团簇 + 反应验证 + 报告结构。"""
    pytest.importorskip("hdbscan")  # 可选依赖：CI 未装时跳过
    from llmsec.clustering.hdb import run_hdbscan_clustering
    from llmsec.clustering.posterior import compute_method_reactions
    features, methods = _make_blob_features()
    meta = {'method_names': methods, 'method_prompts': {m: f'prompt for {m}' for m in methods}, 'technique_label_names': ['t0', 't1', 't2', 't3', 't4'], 'textual_feature_names': [f'tx{i}' for i in range(12)], 'defense_feature_names': [f'df{i}' for i in range(14)]}
    eval_results = [{'method': m, 'eval_score': float(m.split('_')[1][1:]) * 2 - 5} for m in methods]
    reactions = compute_method_reactions(eval_results)
    report = run_hdbscan_clustering(features, meta, reactions=reactions, write=False)
    assert not report.get('n_clusters', 0) < 2, f"❌ 端到端聚类失败: {report.get('error')}"
    labels = report.get('method_labels', {})
    assert not len(labels) != len(methods), f'❌ method_labels 数量不符: {len(labels)}/{len(methods)}'
    from collections import Counter
    max_share = max(Counter(labels.values()).values()) / len(methods)
    assert not max_share >= 0.6, f'❌ 主 labels 出现巨型簇: 最大簇占比 {max_share:.0%}'
    names = report.get('cluster_names', {})
    assert not len(names) < report['n_clusters'], f"❌ 存在未命名簇: {len(names)}/{report['n_clusters']}"
    hdb = report.get('hdbscan')
    assert not (not hdb or 'n_clusters' not in hdb or 'method_labels' not in hdb), '❌ 缺少 hdbscan 密度视图段'
    assert not (not report.get('top_ks') or not report.get('candidate_sweep')), '❌ 缺少 top_ks / candidate_sweep'
    rv = report.get('reaction_validation', {})
    assert rv.get('available'), f"❌ 簇效验证不可用: {rv.get('reason')}"
    assert not (rv['p_anova'] > 0.05 and rv['p_kruskal'] > 0.05), f"❌ 强相关反应下簇效应应显著: p={rv['p_anova']}/{rv['p_kruskal']}"
    print(f"✅ 端到端聚类通过 (k={report['n_clusters']}, 最大簇占比={max_share:.0%}, 密度视图={hdb['n_clusters']}簇+{hdb['n_noise']}噪声, eta²={rv['eta2']})")

def test_posterior_supervision():
    """弱监督：相关特征被放大，无关特征被压低；加权后簇效应增强。"""
    from llmsec.clustering.posterior import learn_supervised_weights, reaction_validation
    rng = np.random.default_rng(7)
    n, d_rel, d_noise = (60, 10, 30)
    y = np.repeat([-2.0, 0.0, 2.0], n // 3)
    X = np.hstack([y[:, None] + rng.normal(0, 0.3, (n, d_rel)), rng.normal(0, 1, (n, d_noise))])
    methods = [f'm{i}' for i in range(n)]
    y_by_method = {m: float(y[i]) for i, m in enumerate(methods)}
    w = learn_supervised_weights(X, methods, y_by_method)
    assert not w.shape[0] != d_rel + d_noise, f'❌ 权重维度错误: {w.shape}'
    assert not w[:d_rel].mean() <= w[d_rel:].mean(), f'❌ 相关特征未被放大: rel={w[:d_rel].mean():.2f} noise={w[d_rel:].mean():.2f}'
    labels = {m: int(np.sign(y[i])) for i, m in enumerate(methods)}
    reactions = {m: {'mean_score': float(y[i]), 'n': 1, 'win_rate': 1.0 if y[i] > 0 else 0.0} for i, m in enumerate(methods)}
    rv = reaction_validation(labels, reactions)
    assert not (not rv.get('available') or not rv.get('effective')), f'❌ 分组反应下簇效应应有效: {rv}'
    y_rand = rng.normal(0, 1, n)
    reactions_rand = {m: {'mean_score': float(y_rand[i]), 'n': 1, 'win_rate': 0.5} for i, m in enumerate(methods)}
    rv_rand = reaction_validation(labels, reactions_rand)
    assert not (rv_rand.get('available') and rv_rand.get('effective') and (rv_rand['p_anova'] < 0.001)), f'❌ 随机反应被误判为有效: {rv_rand}'
    print(f"✅ 弱监督与 ANOVA 通过 (相关特征权重 {w[:d_rel].mean():.2f}× vs 噪声 {w[d_rel:].mean():.2f}×, eta²={rv['eta2']})")

def test_d_optimal_coverage():
    """冷启动（无 GT）时 D-optimal 种子的特征空间覆盖应优于随机采样。"""
    from llmsec.evaluation.active_learning import greedy_d_optimal
    features, methods = _make_blob_features(n_blobs=6, per_blob=10)
    space = build_whitened_space(features, methods)
    X = space['coords']
    n_seeds = 8
    idx = greedy_d_optimal(X, n_seeds, lam=1.0)
    assert not len(idx) != n_seeds, f'❌ 种子数不符: {len(idx)}/{n_seeds}'

    def min_pairwise_dist(indices):
        pts = X[list(indices)]
        d_min = float('inf')
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = float(np.linalg.norm(pts[i] - pts[j]))
                d_min = min(d_min, d)
        return d_min
    d_opt = min_pairwise_dist(idx)
    rng = np.random.default_rng(0)
    d_rand = max(min_pairwise_dist(rng.choice(len(methods), n_seeds, replace=False)) for _ in range(20))
    assert not d_opt < d_rand, f'❌ D-optimal 覆盖不如随机最优: {d_opt:.2f} < {d_rand:.2f}'
    gt_idx = list(range(10))
    M = X[gt_idx].T @ X[gt_idx] + 1.0 * np.eye(X.shape[1])
    M_inv = np.linalg.inv(M)
    from llmsec.evaluation.active_learning import d_optimal_scores
    picked = greedy_d_optimal(X, 3, lam=1.0, X_gt=X[gt_idx])
    s_picked = d_optimal_scores(X[picked], M_inv).mean()
    s_gt = d_optimal_scores(X[gt_idx], M_inv).mean()
    assert not s_picked <= s_gt, f'❌ 新选点杠杆未超过 GT 平均: {s_picked:.3f} <= {s_gt:.3f}'
    print(f'✅ D-optimal 种子通过 (覆盖 {d_opt:.2f} ≥ 随机最优 {d_rand:.2f}, 新点杠杆 {s_picked:.2f} > GT均值 {s_gt:.2f})')

def test_select_knee_real_curve():
    """真实运行暴露的曲线：小 k 端非单调抖动不应让主峰被丢弃（曾误判 k=6，峰值 k=12）。"""
    sweep = [{'k': 6, 'score': 0.4087}, {'k': 7, 'score': 0.4053}, {'k': 8, 'score': 0.3959}, {'k': 9, 'score': 0.4412}, {'k': 10, 'score': 0.5061}, {'k': 12, 'score': 0.585}, {'k': 13, 'score': 0.2519}, {'k': 14, 'score': 0.3715}, {'k': 16, 'score': 0.5737}]
    k, top3 = select_knee(sweep)
    assert not k != 12, f'❌ 真实曲线应选 k=12（主峰），实际 k={k}'
    assert not 12 not in top3, f'❌ top3 应含 12: {top3}'
    print('✅ 真实曲线 auto-k 通过 (k*=12)')

def test_embedding_fallback_chain(monkeypatch):
    """embedding 降级链：API 未配置时本地缓存优先于 HF；env HF_ENDPOINT 覆盖预检列表。"""
    import llmsec.clustering.features as F
    # 降级链为 API → 本地缓存 → HF → TF-IDF；清空 API 配置以聚焦"缓存 vs HF"
    for k in ("EMBEDDING_API_BASE", "EMBEDDING_API_KEY", "EMBEDDING_API_MODEL"):
        monkeypatch.delenv(k, raising=False)
    sentinel = object()
    orig_try = F._try_local_cache
    orig_state = (F._embedding_model, F._embedding_available, F._embedding_source)
    try:
        F._embedding_model, F._embedding_available, F._embedding_source = (None, True, None)
        F._try_local_cache = lambda name: sentinel
        got = F._get_embedding_model()
        assert got is sentinel and F._embedding_source == 'cache', f'API 未配置时本地缓存应优先: source={F._embedding_source}'
    finally:
        F._try_local_cache = orig_try
        F._embedding_model, F._embedding_available, F._embedding_source = orig_state
    monkeypatch.setenv('HF_ENDPOINT', 'https://unreachable.invalid')
    assert F._first_reachable_hf_host() is None
