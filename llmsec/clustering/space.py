#!/usr/bin/env python3
"""
先验特征的马氏白化空间。

第一性原理：聚类与预测的度量空间只使用先验特征（任何方法都可得），
后验特征（defense，未测点全零）一律不进入距离度量。

做法（轻量马氏距离）：
1. 拼接先验特征块（textual + embedding + technique + intent [+ prior]），标准化
2. SVD：X = UΣVᵀ，取累计解释方差 >= variance_ratio 的前 k 个主成分（上限 max_dims）
3. 白化：Z = U_k · √n · σᵢ / sqrt(σᵢ² + λw)
   —— 白化坐标上的欧氏距离 = 原始空间的截断正则化马氏距离

用法:
    from llmsec.clustering.space import build_whitened_space
    space = build_whitened_space(features, methods)
    coords = space["coords"]          # (n, k) 白化坐标
"""

import numpy as np

from llmsec.params import (
    WHITEN_DAMP,
    WHITEN_LAMBDA_W_REL,
    WHITEN_MAX_DIMS,
    WHITEN_VARIANCE_RATIO,
)

# 参与度量的先验特征块；defense（后验）明确排除
PRIOR_BLOCKS = ("textual", "embedding", "technique", "intent", "prior")

# 白化正则地板（相对谱峰）：σᵢ < 0.1·σ₁ 的噪声方向被抑制而非放大，
# 强信号方向仍保持单位方差（截断正则化马氏距离）
LAMBDA_W_REL = WHITEN_LAMBDA_W_REL


def _spectral_knee(S: np.ndarray) -> int:
    """
    奇异值谱拐点：log(σ²) 序列上距首尾弦最远的点。
    拐点之前视为信号方向，之后视为噪声平台；谱太平滑时回退到中位数位置。
    """
    if len(S) < 3:
        return len(S)
    y = np.log(S**2 + 1e-300)
    x = np.arange(len(y), dtype=float)
    chord = y[0] + (y[-1] - y[0]) * x / (len(y) - 1)
    dist = chord - y  # 谱在弦下方 → 距离为正
    knee = int(np.argmax(dist))
    if dist[knee] <= 0:
        # P-1：谱太平滑（无拐点）→ 回退中位数。原实现无日志，簇几何无依据但看起来正常。
        import logging
        logging.getLogger(__name__).warning(
            "奇异值谱无拐点（太平滑/TF-IDF/退化数据），截断维度回退到中位数 %d/%d", len(S) // 2, len(S),
        )
        return len(S) // 2
    return knee + 1


def build_feature_matrix(
    features: dict,
    methods: list[str],
    blocks: tuple[str, ...] = PRIOR_BLOCKS,
) -> np.ndarray:
    """
    把 features dict 拼成特征矩阵（块维度不一致时按块内最大维度零填充）。
    只拼接 methods 中实际存在的块；defense 等后验块不在默认 blocks 中。
    """
    dims = {b: 0 for b in blocks}
    vecs = {}
    for m in methods:
        feat = features.get(m, {})
        v = {}
        for b in blocks:
            vec = np.atleast_1d(np.asarray(feat.get(b, np.zeros(0)), dtype=np.float64))
            v[b] = vec
            dims[b] = max(dims[b], vec.shape[0])
        vecs[m] = v

    rows = []
    for m in methods:
        parts = []
        for b in blocks:
            vec = vecs[m][b]
            if vec.shape[0] < dims[b]:
                vec = np.pad(vec, (0, dims[b] - vec.shape[0]))
            parts.append(vec)
        rows.append(np.concatenate(parts))
    return np.array(rows, dtype=np.float64)


def build_whitened_space(
    features: dict,
    methods: list[str] | None = None,
    variance_ratio: float = WHITEN_VARIANCE_RATIO,
    max_dims: int = WHITEN_MAX_DIMS,
    lambda_w: float | None = None,
    damp: float = WHITEN_DAMP,
    blocks: tuple[str, ...] = PRIOR_BLOCKS,
    feature_weights: np.ndarray | None = None,
) -> dict:
    """
    构建阻尼白化（轻量马氏）空间。

    参数:
        lambda_w: 白化正则地板；None 时取谱拐点处的 σ²（拐点后噪声方向被抑制）
        damp: 白化强度 ∈ [0,1]，默认取 params.WHITEN_DAMP（当前为 0，不白化，纯 PC 得分）。
            实测本数据上白化是负优化：簇分离信号住在高方差方向，
            白化的方向级等权会稀释信号（同配置 damp=0 轮廓系数约为
            damp=0.5 的 2.8 倍）。保留参数仅供实验调参。
            注：特征级量纲修正由 z-score 完成（与白化无关），
            0/1 与连续特征均已归一。

    返回: {
        "methods": [...],
        "coords": (n, k) 阻尼白化坐标,
        "n_dims": k,
        "singular_values": 全部奇异值,
        "explained_variance_ratio": 各主成分解释方差比,
        "kept_variance": 保留维度累计解释方差,
        "x_mean", "x_std": 标准化参数,
        "vt": (k, d) 主成分载荷（供 transform 使用）,
        "lambda_w": 白化正则,
        "damp": 白化强度,
    }
    """
    if methods is None:
        methods = sorted(features.keys())
    n = len(methods)
    X = build_feature_matrix(features, methods, blocks=blocks)

    if n == 0 or X.shape[1] == 0:
        return {
            "methods": methods, "coords": np.zeros((n, 0)), "n_dims": 0,
            "singular_values": np.zeros(0), "explained_variance_ratio": np.zeros(0),
            "kept_variance": 0.0, "x_mean": None, "x_std": None, "vt": None,
            "lambda_w": lambda_w or 0.0, "damp": damp,
        }

    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0) + 1e-8
    X_scaled = (X - x_mean) / x_std

    # 弱监督特征加权必须作用于标准化之后：w·X 的 z-score 为
    # (w·X − w·μ)/(w·σ) = (X−μ)/σ，标准化前加权会被完全抵消（F2 修复）。
    if feature_weights is not None and X.shape[1] == len(feature_weights):
        X_scaled = X_scaled * np.asarray(feature_weights, dtype=np.float64)

    U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)
    var = S**2
    total = float(var.sum())
    ratio = var / total if total > 0 else np.zeros_like(var)
    cumulative = np.cumsum(ratio)

    k_var = int(np.searchsorted(cumulative, variance_ratio) + 1)

    # 谱拐点：既决定 λw（噪声地板），也决定硬截断位置（拐点×2 封顶，
    # 不需要"白化前再 PCA 一次"——那是同一线性操作的重复）
    knee = _spectral_knee(S)
    if lambda_w is None:
        lambda_w = float(S[min(knee - 1, len(S) - 1)] ** 2) if S.size else 0.0
        lambda_w = max(lambda_w, LAMBDA_W_REL * float(S[0] ** 2) if S.size else 0.0, 1e-12)

    k = max(1, min(k_var, knee * 2, max_dims, len(S)))

    # 阻尼白化（轻量马氏）：在原始 PC 得分与全白化之间按 damp 几何插值。
    # damp=1 全白化（各方向方差均等，谱平滑时会放大噪声）；
    # damp=0 不白化（原始 PC 得分）；damp=0.5 为兼顾度量校正与信噪比的折中。
    raw_scale = S[:k] / np.sqrt(n)
    white_scale = np.sqrt(n) * S[:k] / np.sqrt(S[:k] ** 2 + lambda_w)
    if damp == 0.0:
        # 纯 PC 得分（标准主成分投影），跳过低效乘法
        coords = U[:, :k] * raw_scale
    else:
        scale = (raw_scale ** (1 - damp)) * (white_scale ** damp)
        coords = U[:, :k] * scale

    return {
        "methods": methods,
        "coords": coords,
        "n_dims": k,
        "singular_values": S,
        "explained_variance_ratio": ratio,
        "kept_variance": round(float(cumulative[k - 1]), 4),
        "x_mean": x_mean,
        "x_std": x_std,
        "vt": Vt[:k],
        "lambda_w": lambda_w,
        "damp": damp,
        "feature_weights": feature_weights,
    }


def transform_to_space(space: dict, features: dict, methods: list[str]) -> np.ndarray:
    """把新方法投影到已构建的白化空间（用于可视化新增点，不参与聚类定义）。"""
    if space["vt"] is None or space["n_dims"] == 0:
        return np.zeros((len(methods), 0))
    X = build_feature_matrix(features, methods)
    fw = space.get("feature_weights")
    # 对齐训练时的特征维度
    d_train = space["x_mean"].shape[0]
    if X.shape[1] < d_train:
        X = np.pad(X, ((0, 0), (0, d_train - X.shape[1])))
    elif X.shape[1] > d_train:
        X = X[:, :d_train]
    X_scaled = (X - space["x_mean"]) / space["x_std"]
    # 与 build_whitened_space 一致：加权在标准化之后（F2 修复）
    if fw is not None and X.shape[1] == len(fw):
        X_scaled = X_scaled * np.asarray(fw, dtype=np.float64)
    # 新点 PC 得分 = X @ V_kᵀ；按训练时的 damp/λw 做相同的阻尼白化。
    # M-29 修正：训练 coords = U·scale_train（scale_train = (S/√n)^(1-damp)·(√n·S/√(S²+λw))^damp），
    # 而 X@Vᵀ = U·S（PC 得分），故 transform 的 scale 应为 scale_train / S =
    # (1/√n)^(1-damp)·(√n/√(S²+λw))^damp。原实现漏除了 S，每列多一个因子 Sᵢ，
    # 致 transform 与训练 coords 偏差（实测最大 4.31）。
    n_train = max(space["coords"].shape[0], 1)
    S = space["singular_values"][: space["n_dims"]]
    # 旧工件无 damp 字段时兜底与 build_whitened_space 默认一致（不白化）
    damp = space.get("damp", 0.0)
    raw_scale = np.full_like(S, 1.0 / np.sqrt(n_train))  # 1/√n
    white_scale = np.sqrt(n_train) / np.sqrt(S**2 + space["lambda_w"])
    scale = (raw_scale ** (1 - damp)) * (white_scale ** damp)
    return (X_scaled @ space["vt"].T) * scale
