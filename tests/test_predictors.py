"""Combined tests: 预测器（SVD-Ridge + BlendPredictor + 发现层指纹/相似度迁移）。"""



# ===== from test_svd_ridge.py =====

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

import numpy as np

from llmsec.clustering.features import PRIOR_FEATURE_NAMES, build_prior_features
from llmsec.evaluation.elo import ELOTracker
from llmsec.evaluation.predictors.cold_start import ColdStartPredictor
from llmsec.evaluation.predictors.svd_ridge import EloPredictorModel

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

    predictor = ColdStartPredictor()

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

    code = ('from llmsec.core.config import TargetConfig, resolve_defender_name; '
            'print(resolve_defender_name(TargetConfig.from_env().model))')

    proc = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=ROOT, env=env, timeout=180)

    got = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ''

    assert not proc.returncode != 0, f'❌ 导入 runner 失败: {proc.stderr[-500:]}'

    assert not got != PCAP_MODEL_VERSION, f'❌ PCAP 模式 DEFENDER_NAME={got!r}, 期望 {PCAP_MODEL_VERSION!r}'

    env2 = dict(os.environ, TARGET_TYPE='openai')

    code2 = ('from llmsec.core.config import TargetConfig, resolve_defender_name; '
             'c = TargetConfig.from_env(); print(f"{resolve_defender_name(c.model)}|{c.model}")')

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

    predictor = ColdStartPredictor()

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



# ===== from test_blend_predictor.py =====

"""

回归测试：BlendPredictor（统一 + 模型双层预测，自适应权重）。



验证 P2：

1. 已测方法 → ground_truth 直返（std=0）。

2. 未测方法 → blend（统一+模型混合，0<w_model<1）。

3. 全新模型（无数据）→ unified_only（w_model=0，全靠统一先验）。

4. 自适应权重：样本越少越偏向统一预测（贝叶斯收缩）。

"""










from llmsec.core.results import ResultsMatrix
from llmsec.evaluation.predictors.blend import BlendPredictor
from llmsec.params import BLEND_PRIOR_K

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







# ===== from test_model_fingerprint.py =====

# noqa: E402

import tempfile
from pathlib import Path

from llmsec.evaluation.predictors.fingerprint import (
    compute_fingerprint,
    donor_similarities,
    load_probes,
    model_similarity,
    save_probe,
)
from llmsec.evaluation.predictors.svd_ridge import EloPredictorModel

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

    with tempfile.TemporaryDirectory() as d:

        probe_path = Path(d) / "probes.json"

        from llmsec.evaluation.predictors import fingerprint as mf

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

    with tempfile.TemporaryDirectory() as d:

        probe_path = Path(d) / "probes.json"

        from llmsec.evaluation.predictors import fingerprint as mf

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

