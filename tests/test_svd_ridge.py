"""
回归测试：SVD-Ridge Elo 预测 + PCAP 防御方名称 + 首轮收敛统计。

验证：
1. PCAP 模式下 DEFENDER_NAME 使用 PCAP_MODEL_VERSION。
2. predict_batch 在 ground truth 充足时使用 SVD-Ridge 模型，
   预测含 MAP 不确定性，且与真实 Elo / 距离加权基线趋势一致。
3. ground truth 不足时 predict_batch 回退到同后缀/同基底变体平均。
4. 第一轮后 check_convergence 的 std 不再为 None，且不判收敛。
5. K-Fold 选择的最优 λ 在相同数据上稳定。
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import numpy as np

from llmsec.evaluation.elo import ELOTracker
from llmsec.evaluation.elo_cluster import (
    PRIOR_FEATURE_NAMES,
    ClusterEloPredictor,
    EloPredictorModel,
    build_prior_features,
)

_SYNTH_DIMS = {'textual': 12, 'embedding': 8, 'technique': 6, 'intent': 3}

def _make_synthetic(n_train: int=25, n_test: int=12, seed: int=0):
    """构造线性可预测的合成特征与 Elo。"""
    rng = np.random.default_rng(seed)
    methods = []
    for i in range(n_train + n_test):
        suffix = ['', '_rot13', '_b64', '_code'][i % 4]
        methods.append(f'attack_{i:02d}{suffix}')
    coefs = {b: rng.normal(0, 30, size=d) for b, d in _SYNTH_DIMS.items()}
    features, elos = ({}, {})
    for m in methods:
        feat = {b: rng.normal(0, 1, size=d) for b, d in _SYNTH_DIMS.items()}
        elo = 1500.0
        for b in _SYNTH_DIMS:
            elo += float(np.dot(coefs[b], feat[b]))
        elo += rng.normal(0, 10)
        features[m] = feat
        elos[m] = elo
    return (methods, features, elos)

def _make_predictor(n_train: int=25, n_test: int=12, seed: int=0):
    methods, features, elos = _make_synthetic(n_train, n_test, seed)
    predictor = ClusterEloPredictor()
    predictor.ground_truth = {m: {'elo': round(elos[m], 2)} for m in methods[:n_train]}
    predictor.artifacts = {'features': features, 'labels': {m: 0 for m in methods}, 'meta': {}, 'weights': (0.35, 0.25, 0.1, 0.3)}
    test_methods = methods[n_train:]
    method_records = {m: {'method': m, 'prompt': f'synthetic prompt {m}'} for m in methods}
    return (predictor, test_methods, method_records, elos, features)

def _pearson(a, b):
    a, b = (np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def test_pcap_defender_name():
    """PCAP 模式下 DEFENDER_NAME 应为 PCAP_MODEL_VERSION。"""
    from llmsec.targets import PCAP_MODEL_VERSION
    env = dict(os.environ, TARGET_TYPE='pcap_judge')
    code = 'from llmsec.pipeline import runner; print(runner.DEFENDER_NAME)'
    proc = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=ROOT, env=env, timeout=180)
    got = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ''
    assert not proc.returncode != 0, f'❌ 导入 runner 失败: {proc.stderr[-500:]}'
    assert not got != PCAP_MODEL_VERSION, f'❌ PCAP 模式 DEFENDER_NAME={got!r}, 期望 {PCAP_MODEL_VERSION!r}'
    env2 = dict(os.environ, TARGET_TYPE='openai')
    code2 = "from llmsec.pipeline import runner; print(f'{runner.DEFENDER_NAME}|{runner.TARGET_MODEL}')"
    proc2 = subprocess.run([sys.executable, '-c', code2], capture_output=True, text=True, cwd=ROOT, env=env2, timeout=180)
    got2 = proc2.stdout.strip().splitlines()[-1] if proc2.stdout.strip() else ''
    assert not (proc2.returncode != 0 or '|' not in got2), f'❌ 导入 runner 失败: {proc2.stderr[-500:]}'
    defender2, target_model2 = got2.split('|', 1)
    assert not defender2 != target_model2, f'❌ 非 PCAP 模式 DEFENDER_NAME={defender2!r}, 应等于 TARGET_MODEL={target_model2!r}'
    print('✅ PCAP 防御方名称通过')

def test_predict_batch_svd_ridge():
    """ground truth 充足时 predict_batch 使用 SVD-Ridge，且趋势与真实 Elo 一致。"""
    predictor, test_methods, method_records, elos, features = _make_predictor()
    preds = predictor.predict_batch(method_records)
    assert not set(preds.keys()) != set(test_methods), f'❌ predict_batch 未覆盖所有未测方法: {len(preds)}/{len(test_methods)}'
    for m in test_methods:
        p = preds[m]
        assert not p.get('source') != 'svd_ridge', f"❌ {m} 未使用 SVD-Ridge: source={p.get('source')}"
        assert not (p.get('std') is None or p['std'] < 0), f'❌ {m} 缺少 MAP 不确定性: {p}'
        lo, hi = p['ci95']
        assert lo <= p['elo'] <= hi, f'❌ {m} 置信区间不含均值: {p}'
    pred_elos = [preds[m]['elo'] for m in test_methods]
    true_elos = [elos[m] for m in test_methods]
    corr_true = _pearson(pred_elos, true_elos)
    assert not corr_true < 0.5, f'❌ Ridge 预测与真实 Elo 相关性过低: r={corr_true:.3f}'
    mean_shift = abs(float(np.mean(pred_elos)) - float(np.mean(true_elos)))
    assert not mean_shift > 100, f'❌ 预测均值偏移过大（截距缺失？）: shift={mean_shift:.1f}'
    rmse = float(np.sqrt(np.mean((np.array(pred_elos) - np.array(true_elos)) ** 2)))
    gt_mean = float(np.mean([g['elo'] for g in predictor.ground_truth.values()]))
    baseline_rmse = float(np.sqrt(np.mean((np.array(true_elos) - gt_mean) ** 2)))
    assert not rmse >= baseline_rmse, f'❌ RMSE={rmse:.1f} 未优于 GT 均值基线 {baseline_rmse:.1f}'
    gt_methods = sorted(predictor.ground_truth.keys())
    blocks = list(_SYNTH_DIMS.keys())
    gt_mat = np.array([np.concatenate([features[m][b] for b in blocks]) for m in gt_methods])
    gt_elos = np.array([predictor.ground_truth[m]['elo'] for m in gt_methods])
    mean, std = (gt_mat.mean(axis=0), gt_mat.std(axis=0) + 1e-08)
    gt_scaled = (gt_mat - mean) / std
    dw_elos = []
    for m in test_methods:
        v = (np.concatenate([features[m][b] for b in blocks]) - mean) / std
        d = np.linalg.norm(gt_scaled - v, axis=1)
        w = 1.0 / (1.0 + d)
        dw_elos.append(float(np.dot(w, gt_elos) / w.sum()))
    corr_dw = _pearson(pred_elos, dw_elos)
    assert not corr_dw < 0.3, f'❌ Ridge 预测与距离加权基线趋势不一致: r={corr_dw:.3f}'
    print(f'✅ SVD-Ridge 批量预测通过 (r_true={corr_true:.3f}, r_距离加权={corr_dw:.3f}, RMSE={rmse:.1f}<基线{baseline_rmse:.1f}, λ*={predictor.model.lambda_opt:.4f})')

def test_model_cache():
    """GT 未变时 predict_batch 复用 w 不重训；GT 小幅增长走快速 refit 不重跑 K-Fold。"""
    predictor, test_methods, method_records, elos, features = _make_predictor()
    preds1 = predictor.predict_batch(method_records)
    fit_count1 = predictor.model.fit_count
    w1 = predictor.model.w
    preds2 = predictor.predict_batch(method_records)
    assert not (predictor.model.fit_count != fit_count1 or predictor.model.w is not w1), f'❌ GT 未变但模型重训: fit_count {fit_count1} -> {predictor.model.fit_count}'
    for m in test_methods:
        assert not abs(preds1[m]['elo'] - preds2[m]['elo']) > 1e-09, f'❌ 缓存后预测不一致: {m}'
    moved = test_methods[0]
    predictor.ground_truth[moved] = {'elo': round(elos[moved], 2)}
    cv_errors_before = list(predictor.model.cv_errors)
    predictor.predict_batch(method_records)
    assert not predictor.model.fit_count != fit_count1 + 1, f'❌ GT 小幅增长应触发一次快速 refit: fit_count={predictor.model.fit_count}'
    assert not predictor.model.cv_errors != cv_errors_before, '❌ 快速 refit 不应重跑 K-Fold（cv_errors 被修改）'
    for i, m in enumerate(test_methods[1:11]):
        predictor.ground_truth[m] = {'elo': round(elos[m], 2)}
    predictor.predict_batch(method_records)
    assert not predictor.model.fit_count != fit_count1 + 2, f'❌ GT 增长 ≥ threshold 应触发完整重训: fit_count={predictor.model.fit_count}'
    print('✅ 模型缓存通过')

def test_predict_batch_fallback():
    """ground truth 不足时 predict_batch 回退到同后缀/同基底变体平均。"""
    predictor = ClusterEloPredictor()
    predictor.min_cluster_size = 3
    predictor.ground_truth = {'attack_a_rot13': {'elo': 1800.0}, 'attack_b_rot13': {'elo': 1700.0}}
    predictor.artifacts = {'features': {}, 'labels': {'attack_a_rot13': 0, 'attack_b_rot13': 0, 'attack_c_rot13': 0}, 'meta': {}}
    method_records = {'attack_c_rot13': {'method': 'attack_c_rot13', 'prompt': 'x'}}
    preds = predictor.predict_batch(method_records)
    p = preds.get('attack_c_rot13', {})
    assert not p.get('source') != 'predicted_suffix_variant', f"❌ ground truth 不足时未回退到变体平均: source={p.get('source')}"
    assert not abs(p['elo'] - 1750.0) > 1e-06, f"❌ 回退预测值错误: elo={p['elo']}, 期望 1750"
    print('✅ ground truth 不足回退通过')

def test_first_round_convergence():
    """第一轮后不判收敛（轮次不足 CONV_WINDOW_MIN），置信度被压低。"""
    tracker = ELOTracker()
    defender = 'test-model'
    tracker._round_defender_elos[defender] = [1500.0]
    tracker.defender_ratings[defender] = 1500.0
    for i in range(50):
        tracker.attacker_ratings[f'method_{i}'] = 1500.0
    for i in range(6):
        tracker.ground_truth_methods.add(f'method_{i}')
    conv = tracker.check_convergence(defender, total_methods=50, tested_count=6)
    assert not conv['converged'], f'❌ 轮次不足时不应判收敛: {conv}'
    assert any('轮次不足' in n for n in conv['notes']), f"❌ 缺少轮次不足提示: {conv['notes']}"
    boundary = tracker.compute_security_boundary(defender)
    assert not boundary['confidence'] >= 0.8, f"❌ 第一轮置信度未有效压低: {boundary['confidence']}"
    print(f"✅ 首轮收敛统计通过 (ci_half={conv['ci_half']}, 置信度={boundary['confidence']:.2f})")

def test_kfold_lambda_stability():
    """相同数据上 K-Fold 选择的最优 λ 应稳定。"""
    predictor, _, _, _, _ = _make_predictor()
    features = predictor.artifacts['features']
    gt = predictor.ground_truth
    lambdas = []
    for _ in range(3):
        model = EloPredictorModel()
        train_features = {}
        for m in sorted(gt.keys()):
            feat = dict(features[m])
            feat['prior'] = build_prior_features(m, None)
            train_features[m] = feat
        model.fit(train_features, gt)
        lambdas.append(model.lambda_opt)
    assert not len(set(lambdas)) != 1, f'❌ 相同数据上 λ 不稳定: {lambdas}'
    assert not (lambdas[0] not in predictor.model.lambda_candidates.tolist() and (not np.isclose(lambdas[0], predictor.model.lambda_candidates).any())), f'❌ λ* 不在候选集中: {lambdas[0]}'
    vec = build_prior_features('attack_01_rot13', {'prompt': 'a\nb\n在回答之前，请先计算以下数学题 1+1'})
    assert not vec.shape[0] != len(PRIOR_FEATURE_NAMES), f'❌ 先验特征维度错误: {vec.shape[0]} != {len(PRIOR_FEATURE_NAMES)}'
    assert not (vec[2] != 1.0 or vec[7] != 1.0), f'❌ 先验特征取值错误: {vec}'
    print(f'✅ K-Fold λ 稳定性通过 (λ*={lambdas[0]:.4f})')

def test_degenerate_column_variance_cap():
    """GT 内常数、GT 外偏移的退化列不应使预测方差爆炸（回归：std=9.6e8 事故）。"""
    from llmsec.params import RIDGE_PRED_STD_CAP_MIN, RIDGE_PRED_STD_CAP_MULT
    predictor, test_methods, method_records, elos, features = _make_predictor()
    gt_methods = sorted(predictor.ground_truth.keys())
    for m in gt_methods:
        features[m]['textual'] = np.append(features[m]['textual'], 1.0)
    for m in test_methods:
        features[m]['textual'] = np.append(features[m]['textual'], 0.0)
    preds = predictor.predict_batch(method_records)
    y_std = float(np.std([g['elo'] for g in predictor.ground_truth.values()]))
    cap = max(RIDGE_PRED_STD_CAP_MULT * y_std, RIDGE_PRED_STD_CAP_MIN)
    for m in test_methods:
        p = preds[m]
        assert not p.get('source') != 'svd_ridge', f"❌ {m} 未使用 SVD-Ridge: source={p.get('source')}"
        std = p['std']
        assert np.isfinite(std), f'❌ {m} std 非有限值: {std}'
        assert not std > cap + 0.01, f'❌ {m} std={std:.1f} 爆炸（封顶 {cap:.1f}）'
        assert 1000.0 <= p['elo'] <= 2000.0, f"❌ {m} 均值被退化列带飞: elo={p['elo']}"
    assert not (predictor.model.col_keep is None or predictor.model.col_keep.all()), '❌ 退化列未被标记（col_keep 全为 True）'
    sigma2 = predictor.model.sigma2
    floor = min(float(np.sqrt(sigma2)), cap)
    for m in test_methods:
        assert not preds[m]['std'] < floor - 1e-06, f"❌ {m} 预测方差缺少不可约 σ² 项: std={preds[m]['std']:.2f} < {floor:.2f}"
    print(f'✅ 退化列方差防爆通过 (std 封顶={cap:.1f}, σ²={sigma2:.2f}, 退化列数={int((~predictor.model.col_keep).sum())})')
