"""E 组审查修复的回归测试：svd_ridge K-Fold 加权泄漏 与 加权 σ² 自由度。

覆盖：
1. K-Fold 验证折不再乘 √w —— blend 产生的 0 权 donor 行（预测=0/目标=0/误差恒 0）
   不再系统性压低 fold_errors，1-SE 选 λ 不再因此有偏。
2. σ² 的 df_resid 在加权时用有效样本量 (Σw)²/Σw² —— 0 权 donor 行不再按全额 n
   计入自由度而摊薄 σ²（CI 偏窄）。
"""

import numpy as np

from llmsec.evaluation.predictors.svd_ridge import EloPredictorModel


def _make_features(seed_offsets, dim=6):
    """构造简单特征：textual 块 dim 维，其余块留空。"""
    feats = {}
    for name, off in seed_offsets.items():
        rng = np.random.default_rng(abs(hash(name)) % (2**31))
        feats[name] = {"textual": rng.normal(off, 1.0, size=dim)}
    return feats


def _fit(features, elos, sample_weights=None):
    gt = {m: {"elo": e} for m, e in elos.items()}
    model = EloPredictorModel()
    model.fit(features, gt, sample_weights=sample_weights)
    return model


def test_zero_weight_donor_does_not_deflate_cv_error():
    """修复 1：0 权 donor（y 为极端离群）进入验证折时必须贡献真实误差。

    旧实现验证折乘 √w：donor 行预测=0、目标=0、误差恒 0 → fold_errors 被压低。
    修复后 donor 在验证折未加权，其巨大残差应使 CV 误差显著高于无 donor 的基线。
    """
    base_elos = {f"m{i}": 1500.0 + 10.0 * i for i in range(8)}
    donors = {f"donor{i}": 1500.0 for i in range(4)}  # 特征见下
    features = _make_features({**base_elos, **donors})

    model_base = _fit(features, base_elos)
    cv_base = min(model_base.cv_errors)
    assert cv_base > 0

    # 0 权 donor 的 y 离基线 ~2000 Elo：验证折命中时误差应极大
    weighted_elos = dict(base_elos)
    weighted_elos.update({f"donor{i}": 3500.0 for i in range(4)})
    weights = {m: 1.0 for m in base_elos}
    weights.update({f"donor{i}": 0.0 for i in range(4)})
    model_w = _fit(features, weighted_elos, sample_weights=weights)

    cv_weighted = min(model_w.cv_errors)
    # donor 离群残差（~2000² 量级）必须进入 CV 误差；旧实现下该值与基线同量级
    assert cv_weighted > 10 * cv_base, (
        f"0 权 donor 未贡献 CV 误差：weighted={cv_weighted:.2f}, base={cv_base:.2f}"
    )


def test_weighted_sigma2_uses_effective_sample_size():
    """修复 2：加权时 σ² 的自由度用 n_eff=(Σw)²/Σw²，不按全额 n 摊薄。

    用模型内部量重算期望值做精确断言；同时断言旧口径 rss/(n-df) 系统性更小。
    """
    elos = {f"m{i}": 1500.0 + 25.0 * i for i in range(8)}
    donors = {f"donor{i}": 1600.0 + 30.0 * i for i in range(4)}
    all_elos = {**elos, **donors}
    features = _make_features(all_elos)
    weights = {m: 1.0 for m in elos}
    weights.update({d: 0.0 for d in donors})

    model = _fit(features, all_elos, sample_weights=weights)

    # 用模型保存的 (w, x_mean, x_std, col_keep, y_mean) 重算加权残差 RSS
    methods = sorted(all_elos)
    X, _ = EloPredictorModel.features_to_matrix(features, methods)
    Xs = (X - model.x_mean) / model.x_std
    Xs[:, ~model.col_keep] = 0.0
    y = np.array([all_elos[m] for m in methods], dtype=np.float64)
    w = np.array([weights[m] for m in methods], dtype=np.float64)
    sw = np.sqrt(w)
    yc = (y - model.y_mean) * sw
    Xw = Xs * sw[:, None]
    rss = float(np.sum((yc - Xw @ model.w) ** 2))

    df_eff = int(round(model.effective_df))
    n_eff = float(w.sum() ** 2 / np.sum(w**2))
    expected = rss / max(1.0, n_eff - df_eff)
    expected = max(expected, 1e-6)  # M-7 σ² 下限

    assert model.sigma2 == np.float64(expected) or abs(model.sigma2 - expected) < 1e-9, (
        f"σ²={model.sigma2} 与有效样本量口径 {expected} 不一致"
    )
    # 旧口径（全额 n 计入 0 权行）必然低估 σ²
    df_old = max(1, len(methods) - df_eff)
    sigma2_old = max(rss / df_old, 1e-6)
    assert model.sigma2 > sigma2_old, (
        f"σ²={model.sigma2} 未纠正旧口径低估（{sigma2_old}）"
    )


def test_unweighted_sigma2_path_unchanged():
    """修复 2 的护栏：非加权路径行为不变（n_eff=n 时与旧公式一致）。"""
    elos = {f"m{i}": 1500.0 + 25.0 * i for i in range(8)}
    features = _make_features(elos)
    model = _fit(features, elos)  # 无 sample_weights

    methods = sorted(elos)
    X, _ = EloPredictorModel.features_to_matrix(features, methods)
    Xs = (X - model.x_mean) / model.x_std
    Xs[:, ~model.col_keep] = 0.0
    y = np.array([elos[m] for m in methods], dtype=np.float64)
    rss = float(np.sum(((y - model.y_mean) - Xs @ model.w) ** 2))
    expected = max(rss / max(1, len(methods) - int(round(model.effective_df))), 1e-6)
    assert abs(model.sigma2 - expected) < 1e-9
