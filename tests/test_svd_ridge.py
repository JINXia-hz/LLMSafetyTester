#!/usr/bin/env python3
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

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llmsec.evaluation.elo import ELOTracker
from llmsec.evaluation.elo_cluster import (
    ClusterEloPredictor,
    EloPredictorModel,
    PRIOR_FEATURE_NAMES,
    build_prior_features,
)

_SYNTH_DIMS = {"textual": 15, "embedding": 8, "technique": 6, "intent": 3}


def _make_synthetic(n_train: int = 25, n_test: int = 12, seed: int = 0):
    """构造线性可预测的合成特征与 Elo。"""
    rng = np.random.default_rng(seed)
    methods = []
    for i in range(n_train + n_test):
        suffix = ["", "_rot13", "_b64", "_code"][i % 4]
        methods.append(f"attack_{i:02d}{suffix}")

    coefs = {b: rng.normal(0, 30, size=d) for b, d in _SYNTH_DIMS.items()}
    features, elos = {}, {}
    for m in methods:
        feat = {b: rng.normal(0, 1, size=d) for b, d in _SYNTH_DIMS.items()}
        elo = 1500.0
        for b in _SYNTH_DIMS:
            elo += float(np.dot(coefs[b], feat[b]))
        elo += rng.normal(0, 10)
        features[m] = feat
        elos[m] = elo
    return methods, features, elos


def _make_predictor(n_train: int = 25, n_test: int = 12, seed: int = 0):
    methods, features, elos = _make_synthetic(n_train, n_test, seed)
    predictor = ClusterEloPredictor()
    predictor.ground_truth = {
        m: {"elo": round(elos[m], 2)} for m in methods[:n_train]
    }
    predictor.artifacts = {
        "features": features,
        "labels": {m: 0 for m in methods},
        "meta": {},
        "weights": (0.35, 0.25, 0.10, 0.30),
    }
    test_methods = methods[n_train:]
    method_records = {
        m: {"method": m, "prompt": f"synthetic prompt {m}"} for m in methods
    }
    return predictor, test_methods, method_records, elos, features


def _pearson(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def test_pcap_defender_name() -> int:
    """PCAP 模式下 DEFENDER_NAME 应为 PCAP_MODEL_VERSION。"""
    from llmsec.targets import PCAP_MODEL_VERSION

    env = dict(os.environ, TARGET_TYPE="pcap_judge")
    code = "from llmsec.pipeline import runner; print(runner.DEFENDER_NAME)"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=ROOT, env=env, timeout=180,
    )
    got = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    if proc.returncode != 0:
        print(f"❌ 导入 runner 失败: {proc.stderr[-500:]}")
        return 1
    if got != PCAP_MODEL_VERSION:
        print(f"❌ PCAP 模式 DEFENDER_NAME={got!r}, 期望 {PCAP_MODEL_VERSION!r}")
        return 1

    # 非 PCAP 模式：.env 可能重设 TARGET_TYPE，显式覆盖为 openai
    env2 = dict(os.environ, TARGET_TYPE="openai")
    code2 = (
        "from llmsec.pipeline import runner; "
        "print(f'{runner.DEFENDER_NAME}|{runner.TARGET_MODEL}')"
    )
    proc2 = subprocess.run(
        [sys.executable, "-c", code2],
        capture_output=True, text=True, cwd=ROOT, env=env2, timeout=180,
    )
    got2 = proc2.stdout.strip().splitlines()[-1] if proc2.stdout.strip() else ""
    if proc2.returncode != 0 or "|" not in got2:
        print(f"❌ 导入 runner 失败: {proc2.stderr[-500:]}")
        return 1
    defender2, target_model2 = got2.split("|", 1)
    if defender2 != target_model2:
        print(f"❌ 非 PCAP 模式 DEFENDER_NAME={defender2!r}, 应等于 TARGET_MODEL={target_model2!r}")
        return 1
    print("✅ PCAP 防御方名称通过")
    return 0


def test_predict_batch_svd_ridge() -> int:
    """ground truth 充足时 predict_batch 使用 SVD-Ridge，且趋势与真实 Elo 一致。"""
    predictor, test_methods, method_records, elos, features = _make_predictor()

    preds = predictor.predict_batch(method_records)
    if set(preds.keys()) != set(test_methods):
        print(f"❌ predict_batch 未覆盖所有未测方法: {len(preds)}/{len(test_methods)}")
        return 1

    for m in test_methods:
        p = preds[m]
        if p.get("source") != "svd_ridge":
            print(f"❌ {m} 未使用 SVD-Ridge: source={p.get('source')}")
            return 1
        if p.get("std") is None or p["std"] < 0:
            print(f"❌ {m} 缺少 MAP 不确定性: {p}")
            return 1
        lo, hi = p["ci95"]
        if not (lo <= p["elo"] <= hi):
            print(f"❌ {m} 置信区间不含均值: {p}")
            return 1

    pred_elos = [preds[m]["elo"] for m in test_methods]
    true_elos = [elos[m] for m in test_methods]
    corr_true = _pearson(pred_elos, true_elos)
    if corr_true < 0.5:
        print(f"❌ Ridge 预测与真实 Elo 相关性过低: r={corr_true:.3f}")
        return 1

    # 绝对精度：截距（~1500 基准）必须被正确还原，不能整体偏移到 0 附近
    mean_shift = abs(float(np.mean(pred_elos)) - float(np.mean(true_elos)))
    if mean_shift > 100:
        print(f"❌ 预测均值偏移过大（截距缺失？）: shift={mean_shift:.1f}")
        return 1
    rmse = float(np.sqrt(np.mean((np.array(pred_elos) - np.array(true_elos)) ** 2)))
    gt_mean = float(np.mean([g["elo"] for g in predictor.ground_truth.values()]))
    baseline_rmse = float(np.sqrt(np.mean((np.array(true_elos) - gt_mean) ** 2)))
    if rmse >= baseline_rmse:
        print(f"❌ RMSE={rmse:.1f} 未优于 GT 均值基线 {baseline_rmse:.1f}")
        return 1

    # 与距离加权基线趋势一致（欧氏距离倒数加权）
    gt_methods = sorted(predictor.ground_truth.keys())
    blocks = list(_SYNTH_DIMS.keys())
    gt_mat = np.array([np.concatenate([features[m][b] for b in blocks]) for m in gt_methods])
    gt_elos = np.array([predictor.ground_truth[m]["elo"] for m in gt_methods])
    mean, std = gt_mat.mean(axis=0), gt_mat.std(axis=0) + 1e-8
    gt_scaled = (gt_mat - mean) / std
    dw_elos = []
    for m in test_methods:
        v = (np.concatenate([features[m][b] for b in blocks]) - mean) / std
        d = np.linalg.norm(gt_scaled - v, axis=1)
        w = 1.0 / (1.0 + d)
        dw_elos.append(float(np.dot(w, gt_elos) / w.sum()))
    corr_dw = _pearson(pred_elos, dw_elos)
    if corr_dw < 0.3:
        print(f"❌ Ridge 预测与距离加权基线趋势不一致: r={corr_dw:.3f}")
        return 1

    print(f"✅ SVD-Ridge 批量预测通过 (r_true={corr_true:.3f}, r_距离加权={corr_dw:.3f}, "
          f"RMSE={rmse:.1f}<基线{baseline_rmse:.1f}, λ*={predictor.model.lambda_opt:.4f})")
    return 0


def test_model_cache() -> int:
    """GT 未变时 predict_batch 复用 w 不重训；GT 小幅增长走快速 refit 不重跑 K-Fold。"""
    predictor, test_methods, method_records, elos, features = _make_predictor()

    preds1 = predictor.predict_batch(method_records)
    fit_count1 = predictor.model.fit_count
    w1 = predictor.model.w

    # GT 未变 → 复用缓存
    preds2 = predictor.predict_batch(method_records)
    if predictor.model.fit_count != fit_count1 or predictor.model.w is not w1:
        print(f"❌ GT 未变但模型重训: fit_count {fit_count1} -> {predictor.model.fit_count}")
        return 1
    for m in test_methods:
        if abs(preds1[m]["elo"] - preds2[m]["elo"]) > 1e-9:
            print(f"❌ 缓存后预测不一致: {m}")
            return 1

    # GT 小幅增长（+1 < threshold=10）→ 快速 refit，不重跑 K-Fold
    moved = test_methods[0]
    predictor.ground_truth[moved] = {"elo": round(elos[moved], 2)}
    cv_errors_before = list(predictor.model.cv_errors)
    predictor.predict_batch(method_records)
    if predictor.model.fit_count != fit_count1 + 1:
        print(f"❌ GT 小幅增长应触发一次快速 refit: fit_count={predictor.model.fit_count}")
        return 1
    if predictor.model.cv_errors != cv_errors_before:
        print("❌ 快速 refit 不应重跑 K-Fold（cv_errors 被修改）")
        return 1

    # GT 增长 ≥ threshold → 重跑 K-Fold
    for i, m in enumerate(test_methods[1:11]):
        predictor.ground_truth[m] = {"elo": round(elos[m], 2)}
    predictor.predict_batch(method_records)
    if predictor.model.fit_count != fit_count1 + 2:
        print(f"❌ GT 增长 ≥ threshold 应触发完整重训: fit_count={predictor.model.fit_count}")
        return 1

    print("✅ 模型缓存通过")
    return 0


def test_predict_batch_fallback() -> int:
    """ground truth 不足时 predict_batch 回退到同后缀/同基底变体平均。"""
    predictor = ClusterEloPredictor()
    predictor.min_cluster_size = 3
    predictor.ground_truth = {
        "attack_a_rot13": {"elo": 1800.0},
        "attack_b_rot13": {"elo": 1700.0},
    }
    predictor.artifacts = {
        "features": {},
        "labels": {"attack_a_rot13": 0, "attack_b_rot13": 0, "attack_c_rot13": 0},
        "meta": {},
    }
    method_records = {"attack_c_rot13": {"method": "attack_c_rot13", "prompt": "x"}}

    preds = predictor.predict_batch(method_records)
    p = preds.get("attack_c_rot13", {})
    if p.get("source") != "predicted_suffix_variant":
        print(f"❌ ground truth 不足时未回退到变体平均: source={p.get('source')}")
        return 1
    if abs(p["elo"] - 1750.0) > 1e-6:
        print(f"❌ 回退预测值错误: elo={p['elo']}, 期望 1750")
        return 1
    print("✅ ground truth 不足回退通过")
    return 0


def test_first_round_convergence() -> int:
    """第一轮后 check_convergence 的 std 不再为 None，且不判收敛。"""
    tracker = ELOTracker()
    defender = "test-model"
    tracker._round_defender_elos[defender] = [1500.0]
    tracker.defender_ratings[defender] = 1500.0
    for i in range(50):
        tracker.attacker_ratings[f"method_{i}"] = 1500.0
    for i in range(6):
        tracker.ground_truth_methods.add(f"method_{i}")

    conv = tracker.check_convergence(defender, total_methods=50, tested_count=6)
    if conv["std"] is None:
        print(f"❌ 第一轮后 std 仍为 None: {conv}")
        return 1
    if conv["std"] <= 0.0:
        print(f"❌ 单点 std 应按阈值保守处理（不能为 0，否则置信度虚高）: {conv['std']}")
        return 1
    if conv["converged"]:
        print(f"❌ 轮次不足时不应判收敛: {conv}")
        return 1
    if not any("折扣" in n for n in conv["notes"]):
        print(f"❌ 缺少轮次不足折扣提示: {conv['notes']}")
        return 1

    # 置信度应被折扣：不能越过收敛目标 0.8（否则第一轮就假收敛）
    boundary = tracker.compute_security_boundary(defender)
    if boundary["confidence"] >= 0.8:
        print(f"❌ 第一轮置信度未有效折扣: {boundary['confidence']}")
        return 1
    print(f"✅ 首轮收敛统计通过 (std={conv['std']}, 置信度={boundary['confidence']:.2f})")
    return 0


def test_kfold_lambda_stability() -> int:
    """相同数据上 K-Fold 选择的最优 λ 应稳定。"""
    predictor, _, _, _, _ = _make_predictor()
    features = predictor.artifacts["features"]
    gt = predictor.ground_truth

    lambdas = []
    for _ in range(3):
        model = EloPredictorModel()
        train_features = {}
        for m in sorted(gt.keys()):
            feat = dict(features[m])
            feat["prior"] = build_prior_features(m, None)
            train_features[m] = feat
        model.fit(train_features, gt)
        lambdas.append(model.lambda_opt)

    if len(set(lambdas)) != 1:
        print(f"❌ 相同数据上 λ 不稳定: {lambdas}")
        return 1
    if lambdas[0] not in predictor.model.lambda_candidates.tolist() and \
            not np.isclose(lambdas[0], predictor.model.lambda_candidates).any():
        print(f"❌ λ* 不在候选集中: {lambdas[0]}")
        return 1

    # 先验特征向量维度正确
    vec = build_prior_features("attack_01_rot13", {"prompt": "a\nb\n在回答之前，请先计算以下数学题 1+1"})
    if vec.shape[0] != len(PRIOR_FEATURE_NAMES):
        print(f"❌ 先验特征维度错误: {vec.shape[0]} != {len(PRIOR_FEATURE_NAMES)}")
        return 1
    if vec[2] != 1.0 or vec[7] != 1.0:
        print(f"❌ 先验特征取值错误: {vec}")
        return 1

    print(f"✅ K-Fold λ 稳定性通过 (λ*={lambdas[0]:.4f})")
    return 0


def main() -> int:
    tests = [
        test_pcap_defender_name,
        test_predict_batch_svd_ridge,
        test_predict_batch_fallback,
        test_first_round_convergence,
        test_kfold_lambda_stability,
        test_model_cache,
    ]
    for t in tests:
        if t() != 0:
            return 1
    print("\n✅ 所有 SVD-Ridge / PCAP / 收敛回归测试通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
