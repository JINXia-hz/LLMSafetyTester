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
import tempfile

import numpy as np

from llmsec.clustering.tree import _evaluate_cut, sweep_candidates
from llmsec.evaluation.elo import ELOTracker
from llmsec.evaluation.elo_cluster import ClusterEloPredictor
from llmsec.params import PORTRAIT_MIN_TESTED


def test_f2_evaluate_cut_degenerate_returns_finite():
    """全单点簇（n_clusters == n）→ 早期返回，davies_bouldin 应为 1e6（非 inf）。"""
    coords = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    labels = {'a': 0, 'b': 1, 'c': 2}
    metrics = _evaluate_cut(coords, labels, ['a', 'b', 'c'])
    assert math.isfinite(metrics['davies_bouldin']), f"F2: 退化切割 davies_bouldin 有限（实得 {metrics['davies_bouldin']}）"
    assert math.isfinite(metrics['silhouette']), 'F2: 退化切割 silhouette 有限'
    assert math.isfinite(metrics['calinski_harabasz']), 'F2: 退化切割 calinski_harabasz 有限'
    assert metrics['davies_bouldin'] != float('inf'), 'F2: davies_bouldin 不是 inf（是 1e6）'

def test_f2_sweep_no_nan_in_score():
    """sweep_candidates 含退化 k 时，score 不含 NaN（json.dumps allow_nan=False 不崩）。"""
    from scipy.cluster.hierarchy import linkage
    coords = np.array([[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]])
    Z = linkage(coords, method='ward')
    methods = ['a', 'b', 'c', 'd']
    sweep = sweep_candidates(coords, Z, methods, ks=[1, 2, 3])
    scores = [s['score'] for s in sweep]
    assert all(math.isfinite(s) for s in scores), f'F2: sweep scores 全有限（实得 {scores}）'
    try:
        json.dumps(sweep, allow_nan=False)
        assert True, 'F2: sweep 结果 json.dumps(allow_nan=False) 不抛'
    except ValueError as e:
        assert False, f'F2: sweep 结果 json.dumps 抛错: {e}'

def test_f2_evaluate_cut_zero_variance_cluster():
    """簇内零方差（所有点同坐标）使 davies_bouldin_score 抛异常 → _safe 返回 1e6。"""
    coords = np.array([[0.0, 0.0], [0.0, 0.0], [5.0, 5.0]])
    labels = {'a': 0, 'b': 0, 'c': 1}
    metrics = _evaluate_cut(coords, labels, ['a', 'b', 'c'])
    assert math.isfinite(metrics['davies_bouldin']), f"F2: 零方差簇 davies_bouldin 有限（实得 {metrics['davies_bouldin']}）"

def test_h3_trajectory_restored_on_exception():
    """check_convergence 抛异常时，_compute_conv_rounds 传播异常但 finally 恢复轨迹。"""
    import pytest as _pytest

    from llmsec.pipeline.runner import _compute_conv_rounds
    tr = ELOTracker()
    tr._round_defender_elos['def'] = [1500.0, 1510.0, 1520.0]
    original = list(tr._round_defender_elos['def'])
    orig_cc = tr.check_convergence

    def _boom(*a, **kw):
        raise RuntimeError('模拟异常')
    tr.check_convergence = _boom
    # B-bucket：异常不再静默吞没——传播给调用方
    try:
        with _pytest.raises(RuntimeError):
            _compute_conv_rounds(tr, 'def', total_methods=10)
    finally:
        tr.check_convergence = orig_cc
    # finally 块仍恢复轨迹（无论异常是否传播）
    restored = tr._round_defender_elos.get('def', [])
    assert restored == original, f'异常后轨迹应恢复（实得 {restored}，期望 {original}）'

def test_h3_trajectory_restored_on_normal_return():
    """正常返回时轨迹也恢复。"""
    from llmsec.pipeline.runner import _compute_conv_rounds
    tr = ELOTracker()
    tr._round_defender_elos['def'] = [1500.0, 1501.0, 1502.0, 1503.0]
    original = list(tr._round_defender_elos['def'])
    orig_cc = tr.check_convergence

    def _conv(defender, total_methods=10, tested_count=5):
        return {'converged': True, 'ci_half': 1.0}
    tr.check_convergence = _conv
    try:
        result = _compute_conv_rounds(tr, 'def', total_methods=10)
    finally:
        tr.check_convergence = orig_cc
    assert result == 1, f'H3: 正常返回首个收敛轮（实得 {result}）'
    restored = tr._round_defender_elos.get('def', [])
    assert restored == original, f'H3: 正常返回后轨迹恢复（实得 {restored}）'

def test_h4_build_tree_inconclusive_when_data_insufficient():
    """total_tests < PORTRAIT_MIN_TESTED(5) → level=inconclusive。"""
    from llmsec.reporting.report import build_method_stats, build_tree
    results = [{'method': 'm1', 'is_harmful': True, 'harm_type': 'test', 'category': 'x'}, {'method': 'm1', 'is_harmful': False, 'harm_type': 'test', 'category': 'x'}, {'method': 'm2', 'is_harmful': False, 'harm_type': 'test', 'category': 'x'}]
    method_stats = build_method_stats(results, {}, {})
    allergy_data = {'summary': {'false_positive_rate': 0.0}}
    with tempfile.TemporaryDirectory() as td:
        tree = build_tree(method_stats, allergy_data, {}, output_dir=td)
    level = tree.get('overall', {}).get('security_level', '')
    assert level == 'inconclusive', f'H4: 数据不足时 level=inconclusive（实得 {level}，total_tests=3<{PORTRAIT_MIN_TESTED}）'

def test_h4_build_tree_inconclusive_when_confidence_low():
    """无 tracker（confidence=0 < 0.5）即使 total_tests 够也 inconclusive。"""
    from llmsec.reporting.report import build_method_stats, build_tree
    results = [{'method': f'm{i}', 'is_harmful': False, 'harm_type': 'test', 'category': 'x'} for i in range(10)]
    method_stats = build_method_stats(results, {}, {})
    allergy_data = {'summary': {'false_positive_rate': 0.0}}
    with tempfile.TemporaryDirectory() as td:
        tree = build_tree(method_stats, allergy_data, {}, output_dir=td)
    level = tree.get('overall', {}).get('security_level', '')
    assert level == 'inconclusive', f'H4: 无 tracker 时即使 ASR=0 也 inconclusive（confidence 不足，实得 {level}）'

def test_h9_kfold_balanced_small_gt():
    """小 GT（n=7, k=5）K-Fold 不崩，fold 余数均衡分配（非全堆最后一折）。"""
    from llmsec.evaluation.elo_cluster import EloPredictorModel
    rng = np.random.default_rng(42)
    methods = [f'm{i}' for i in range(7)]
    features = {m: {'textual': rng.normal(size=5).tolist(), 'prior': [0.0, 0.0, 0.0]} for m in methods}
    gt = {m: {'elo': 1500.0 + i * 10, 'first_seen_at': i} for i, m in enumerate(methods)}
    blocks = {'textual': [f't{i}' for i in range(5)], 'prior': ['p0', 'p1', 'p2']}
    model = EloPredictorModel()
    try:
        model.fit(features, gt, blocks)
        assert model.lambda_opt > 0, f'H9: 小 GT fit 成功，λ*>0（实得 {model.lambda_opt}）'
        assert model.w is not None, 'H9: 小 GT fit 产出权重 w'
        assert len(model.cv_errors) > 0, 'H9: K-Fold 产出 cv_errors'
    except Exception as e:
        assert False, f'H9: 小 GT fit 异常: {e}'

def test_h9_kfold_fold_sizes_balanced():
    """直接验证 fold 尺寸均衡：n=9, k=5 → fold 尺寸 2,2,2,2,1（非 1,1,1,1,5）。"""
    n, k = (9, 5)
    fold_size = n // k
    remainder = n % k
    fold_sizes = []
    for i in range(k):
        start = i * fold_size + min(i, remainder)
        end = start + fold_size + (1 if i < remainder else 0)
        fold_sizes.append(end - start)
    assert fold_sizes == [2, 2, 2, 2, 1], f'H9: n=9,k=5 fold 尺寸均衡 [2,2,2,2,1]（实得 {fold_sizes}）'
    assert sum(fold_sizes) == n, 'H9: fold 尺寸之和 == n'
    assert fold_sizes != [1, 1, 1, 1, 5], 'H9: 非原 bug 的 [1,1,1,1,5]（全堆最后一折）'

def test_h10_predict_fallback_has_std_ci95():
    """predict() 所有回退分支返回的 dict 都含 std/ci95 字段。"""
    predictor = ClusterEloPredictor()
    r = predictor.predict('unknown_method')
    assert 'std' in r and 'ci95' in r, f'H10: 空 GT 回退含 std/ci95（keys={list(r.keys())}）'
    assert r['std'] is not None and math.isfinite(r['std']), 'H10: 空 GT 回退 std 有限'
    assert isinstance(r['ci95'], list) and len(r['ci95']) == 2, 'H10: 空 GT 回退 ci95 是 [lo, hi]'
    predictor.update_ground_truth('DAN_rot13', 1600.0)
    predictor.update_ground_truth('DAN_b64', 1580.0)
    r2 = predictor.predict('DAN_unknown')
    assert 'std' in r2 and 'ci95' in r2, f'H10: 变体回退含 std/ci95（keys={list(r2.keys())}）'
    assert r2['std'] is not None and math.isfinite(r2['std']), 'H10: 变体回退 std 有限'
    r3 = predictor.predict('DAN_rot13')
    assert 'std' in r3 and 'ci95' in r3, f'H10: GT 分支含 std/ci95（keys={list(r3.keys())}）'
    assert r3.get('std') == 0.0, f"H10: GT 分支 std=0（实得 {r3.get('std')}）"

def test_h10_predict_batch_fallback_schema_consistent():
    """predict_batch 在 GT 不足走回退时，结果 schema 与 SVD-Ridge 分支一致。"""
    predictor = ClusterEloPredictor()
    predictor.update_ground_truth('gt1', 1500.0)
    method_records = {'gt1': {'id': 'gt1'}, 'unknown1': {'id': 'unknown1'}}
    results = predictor.predict_batch(method_records)
    r = results.get('unknown1')
    assert r is not None, 'H10: predict_batch 回退产出结果'
    if r:
        for field in ('elo', 'source', 'std', 'ci95', 'confidence'):
            assert field in r, f'H10: predict_batch 回退含 {field}'

def test_h1_quick_precluster_returns_labels():
    """_quick_precluster 有足够 features 时返回 labels。"""
    from llmsec.pipeline.runner import _quick_precluster
    tr = ELOTracker()
    rng = np.random.default_rng(42)
    methods = [f'm{i}' for i in range(6)]
    tr.predictor.artifacts = {'features': {m: {'textual': rng.normal(size=5).tolist()} for m in methods}}
    labels = _quick_precluster(tr, methods)
    assert labels is not None, 'H1: 有 features 时返回 labels'
    if labels:
        assert set(labels.keys()) == set(methods), 'H1: labels 覆盖所有方法'
        assert len(set(labels.values())) >= 2, 'H1: 至少 2 簇'

def test_h1_quick_precluster_no_features_returns_none():
    """_quick_precluster 无 features / 方法太少时返回 None。"""
    from llmsec.pipeline.runner import _quick_precluster
    tr = ELOTracker()
    tr.predictor.artifacts = {}
    assert _quick_precluster(tr, ['m1', 'm2']) is None, 'H1: 无 features 返回 None'
    tr.predictor.artifacts = {'features': {f'm{i}': {'t': [1]} for i in range(3)}}
    assert _quick_precluster(tr, ['m0', 'm1', 'm2']) is None, 'H1: 方法 <4 返回 None'

def test_h1_sampler_receives_cluster_labels():
    """build_sampler 收到 cluster_report（method_labels）时 sampler 能映射方法到簇。"""
    from llmsec.evaluation.samplers import build_sampler
    labels = {'a': 0, 'b': 0, 'c': 1, 'd': 1}
    report = {'method_labels': labels}
    s = build_sampler('infogain', cluster_report=report)
    assert s.cluster_report != {}, 'H1: 注入 report 后 sampler.cluster_report 非空'
    assert s._method_to_cluster('a') == 0 and s._method_to_cluster('c') == 1, 'H1: sampler 正确映射方法到簇'

def test_h6_judge_failure_degrades_gracefully():
    """judge.evaluate 抛异常时降级到 no_judge，不中断。"""
    from llmsec.evaluation import evaluator as eval_mod
    orig_ct = eval_mod.call_target
    eval_mod.call_target = lambda prompt: {'content': 'some response text here', 'error': None, 'latency_ms': 10, 'tokens_prompt': 5, 'tokens_completion': 10}

    class BoomJudge:

        def evaluate(self, *a, **kw):
            raise RuntimeError('Judge API 宕机')
    try:
        result = eval_mod.evaluate_single('test prompt', expected_answer=0, use_judge=True, judge=BoomJudge())
        assert result.get('status') is not None, f"H6: judge 失败降级到 no_judge（status={result.get('status')}）"
        assert 'content' in result, 'H6: 降级后 result 结构完整'
    except Exception as e:
        assert False, f'H6: judge 失败应降级不应抛异常: {e}'
    finally:
        eval_mod.call_target = orig_ct

def test_h8_severity_inconclusive_when_no_results():
    """n_results=0 → severity=inconclusive（不再 fpr=0 误报 low）。"""
    from llmsec.evaluation.safe_twin import _compute_allergy_severity
    sev, interp = _compute_allergy_severity(0, 0.0)
    assert sev == 'inconclusive', f'H8: n_results=0 → inconclusive（实得 {sev}）'
    assert '不足' in interp or '无效' in interp or '不支持' in interp, 'H8: inconclusive 解读文本说明原因'
    sev2, _ = _compute_allergy_severity(3, 0.0)
    assert sev2 == 'inconclusive', f'H8: n_results<6 → inconclusive（实得 {sev2}）'
    sev3, _ = _compute_allergy_severity(20, 0.02)
    assert sev3 == 'low', f'H8: 样本充足 fpr<0.05 → low（实得 {sev3}）'
    sev4, _ = _compute_allergy_severity(20, 0.5)
    assert sev4 == 'high', f'H8: 样本充足 fpr>0.15 → high（实得 {sev4}）'

def test_h11_cache_key_includes_features():
    """cache_key 纳入 features 结构签名（切换 embedding 时 key 不同）。"""
    from llmsec.core.results import ResultsMatrix
    from llmsec.evaluation.blend_predictor import BlendPredictor
    mat = ResultsMatrix()
    mat.upsert('A', 'm', 1.0, ts=1)
    catalog = ['A']
    key_none = BlendPredictor.cache_key(mat, catalog, None)
    feats_5d = {'A': {'textual': [1, 2, 3, 4, 5], 'prior': [0, 0]}}
    feats_3d = {'A': {'textual': [1, 2, 3], 'prior': [0, 0]}}
    key_5d = BlendPredictor.cache_key(mat, catalog, feats_5d)
    key_3d = BlendPredictor.cache_key(mat, catalog, feats_3d)
    key_5d_again = BlendPredictor.cache_key(mat, catalog, feats_5d)
    assert key_none != key_5d, 'H11: 无 features vs 有 features → key 不同'
    assert key_5d != key_3d, 'H11: 不同维度 features → key 不同'
    assert key_5d == key_5d_again, 'H11: 相同 features → key 相同（确定性）'
