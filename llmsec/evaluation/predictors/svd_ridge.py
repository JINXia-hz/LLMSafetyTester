"""
evaluation.predictors.svd_ridge — 基于 SVD 的 Ridge 回归 Elo 预测模型（纯 ML 模型）。

- 用已测方法的特征向量 X 和真实 Elo y（中心化后）训练模型。
- 对特征矩阵做 SVD：X = UΣV^T，Ridge 解 w(λ) = V (Σ² + λI)^(-1) Σ U^T y。
- 用 K-Fold 交叉验证在正则化路径 λ ∈ logspace(-3, 4, 24) 上选择最优 λ。
- SVD 同时提供主成分视角：解释方差比与有效自由度 df(λ) 由 get_pca_summary 输出。

本文件只含纯 ML 模型（fit/predict/诊断），不含数据编排（ground truth / 特征缓存 /
工件持久化 / 变体兜底）。编排逻辑见 cold_start.py 的 ColdStartPredictor。

从 elo_cluster.py 拆出（M-42）；_features_to_matrix 改为公开 features_to_matrix
（跨文件被编排器调用）。
"""

import numpy as np

from llmsec.core.logging import get_logger
from llmsec.core.seed import get_global_seed as _global_seed
from llmsec.params import (
    RIDGE_DEGENERATE_COL_EPS,
    RIDGE_LAMBDA_COUNT,
    RIDGE_LAMBDA_MAX,
    RIDGE_LAMBDA_MIN,
    RIDGE_N_FOLDS,
)

logger = get_logger(__name__)


class EloPredictorModel:
    """
    基于 SVD 的 Ridge 回归 Elo 预测模型。

    - 用已测方法的特征向量 X 和真实 Elo y（中心化后）训练模型。
    - 对特征矩阵做 SVD：X = UΣV^T，Ridge 解 w(λ) = V (Σ² + λI)^(-1) Σ U^T y。
    - 用 K-Fold 交叉验证在正则化路径 λ ∈ logspace(-3, 4, 24) 上选择最优 λ。
    - SVD 同时提供主成分视角：解释方差比与有效自由度 df(λ) 由 get_pca_summary 输出。
    - 贝叶斯解释：Ridge 等价于高斯先验的 MAP；
      预测均值 E = y_mean + X_test @ w，预测方差 σ²_噪声 · (1 + diag(X_test (X^T X + λI)^(-1) X_test^T))。
      σ²_噪声 用自由度修正的 in-sample 残差 RSS/(n−df_eff) 估计（M1：避免 CV-MSE 双重计数）。
      P6：λ* 顶到候选网格上限（no_signal）时特征无信号、预测≈y_mean，predict 的方差
      下限提到 GT 边际方差 y_std²，下游 confidence 随之降级。
    """

    BLOCK_ORDER = ("textual", "embedding", "technique", "intent", "prior")

    def __init__(self, lambda_candidates=None, n_folds: int = RIDGE_N_FOLDS):
        self.lambda_candidates = (
            np.logspace(RIDGE_LAMBDA_MIN, RIDGE_LAMBDA_MAX, RIDGE_LAMBDA_COUNT)
            if lambda_candidates is None else lambda_candidates
        )
        self.n_folds = n_folds
        self.w: np.ndarray | None = None
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        # 退化列掩码：True=保留。GT 子集内近常数的列（x_std 触地板）在子集外稍偏离
        # 就会产生 ~1e8 的标准化值，使 MAP 方差爆炸；fit/predict 时这些列置零
        self.col_keep: np.ndarray | None = None
        self.y_std: float = 0.0  # GT Elo 标准差（预测 std 封顶用）
        self.y_mean: float = 0.0
        self.lambda_opt: float | None = None
        self.sigma2: float | None = None
        self.cv_mse_at_lambda: float | None = None  # CV-MSE at λ*（诊断用，非 σ²_噪声；M1）
        self.xtx_inv: np.ndarray | None = None
        self.cv_errors: list[float] = []
        self.block_dims: dict[str, int] = {}
        self.feature_names: list[str] = []
        # SVD / 主成分诊断
        self.singular_values: np.ndarray | None = None
        self.n_samples: int = 0
        self.effective_df: float | None = None
        # 训练计数（供缓存与测试断言）
        self.fit_count: int = 0
        # OOS 预测（λ* 下逐折留出，键=ground_truth 方法名）——供 BlendPredictor 估计层间残差协方差（#3）
        self.oos_by_key_: dict[str, float] = {}
        # P6：λ* 顶到候选网格上限 = 方法特征对 Elo 基本无信号、模型退化为均值预测，
        # predict 时方差下限提到 GT 边际方差以降级置信度（残差 σ² 在强正则下趋小，会给出假高置信度）
        self.no_signal: bool = False

    @classmethod
    def features_to_matrix(
        cls,
        features_dict: dict,
        methods: list[str],
        block_dims: dict[str, int] | None = None,
    ) -> tuple[np.ndarray, dict[str, int]]:
        """
        把 features dict 转换为特征矩阵 X（textual + embedding + technique + intent + prior）。

        - block_dims=None（训练）：各块按块内最大维度零填充，返回实际维度。
        - block_dims 给定（预测）：严格按该维度截断/零填充，保证与训练时的 w 对齐。
        """
        blocks: dict[str, list[np.ndarray]] = {b: [] for b in cls.BLOCK_ORDER}
        dims = dict(block_dims) if block_dims is not None else {b: 0 for b in cls.BLOCK_ORDER}
        for m in methods:
            feat = features_dict.get(m, {})
            for b in cls.BLOCK_ORDER:
                vec = np.atleast_1d(
                    np.asarray(feat.get(b, np.zeros(0)), dtype=np.float64)
                )
                blocks[b].append(vec)
                if block_dims is None:
                    dims[b] = max(dims[b], vec.shape[0])

        rows = []
        for i in range(len(methods)):
            parts = []
            for b in cls.BLOCK_ORDER:
                vec = blocks[b][i]
                if vec.shape[0] < dims[b]:
                    vec = np.pad(vec, (0, dims[b] - vec.shape[0]))
                elif vec.shape[0] > dims[b]:
                    vec = vec[: dims[b]]
                parts.append(vec)
            rows.append(np.concatenate(parts))
        return np.array(rows, dtype=np.float64), dims

    def _resolve_feature_names(self, name_blocks: dict | None) -> list[str]:
        """按块顺序生成与 w 对齐的特征名列表，缺失的用通用名补齐。"""
        names = []
        for b in self.BLOCK_ORDER:
            dim = self.block_dims.get(b, 0)
            block_names = list((name_blocks or {}).get(b, []))[:dim]
            block_names += [f"{b}_{i}" for i in range(len(block_names), dim)]
            names.extend(block_names)
        return names

    def fit(
        self,
        features_dict: dict,
        ground_truth: dict,
        feature_name_blocks: dict | None = None,
        lambda_override: float | None = None,
        groups: dict | None = None,
        sample_weights: dict | None = None,
    ) -> "EloPredictorModel":
        """
        用 ground truth 训练 Ridge 回归模型。

        参数:
            features_dict: {method: features}，需包含所有 ground truth 方法
            ground_truth: {method: {"elo": float, ...}}
            feature_name_blocks: 各特征块的特征名（用于特征重要性输出）
            lambda_override: 指定 λ 时跳过 K-Fold 直接 refit w（快速通道，
                用于 ground truth 小幅增长时的增量更新）
            groups: {method_key: group_id}，按组做 GroupKFold（统一预测器场景：
                同方法跨模型样本同组，防 CV 泄漏，#4）。None 时走随机 K-Fold。
            sample_weights: {method: weight}，加权 Ridge 的样本权重（发现层 D+A：
                BlendPredictor 按 donor 相似度加权池化）。对**原始** X/y 行乘 √w，
                下游标准化/CV/SVD 自动成加权版——数学等价 min Σw_i(y_i−x_iβ)²+λ||β||²
                的解 = 标准 Ridge 在 (√w·X, √w·y) 上；predict 时新点不缩放（仅训练加权）。
        """
        methods = sorted(ground_truth.keys())
        if not methods:
            raise ValueError("ground_truth 为空，无法训练")

        X, self.block_dims = self.features_to_matrix(features_dict, methods)
        y = np.array([ground_truth[m]["elo"] for m in methods], dtype=np.float64)
        self.feature_names = self._resolve_feature_names(feature_name_blocks)

        # X 标准化 + y 中心化（截距项：否则零均值的 X_scaled @ w 无法表达 ~1500 的基准 Elo）
        # H2 修复：标准化统计量必须在加权之前用**原始** X/y 计算——原实现先乘 √w 再算
        # mean/std，导致 self.x_mean/x_std 被 √w 衰减，与 predict 端的原始尺度 x* 不一致。
        self.x_mean = X.mean(axis=0)
        raw_std = X.std(axis=0)
        # 退化列：GT 子集内近常数（std 触 1e-8 地板）的列在子集外稍偏离就会产生
        # ~1e8 的标准化值，MAP 方差随之爆炸（均值不受影响：该方向 w=0）。
        # 标准化后置零：w 与杠杆都不再有贡献。已知合法 embedding 列 std ≥ 0.017，阈值安全。
        self.col_keep = raw_std >= RIDGE_DEGENERATE_COL_EPS
        self.x_std = raw_std + 1e-8
        X_scaled = (X - self.x_mean) / self.x_std
        X_scaled[:, ~self.col_keep] = 0.0
        self.y_mean = float(y.mean())
        self.y_std = float(y.std())
        y_c = y - self.y_mean

        # 加权 Ridge（发现层 sample_weights）：对**标准化后**的 X_scaled/y_c 行乘 √w，
        # 数学等价 min Σwᵢ(yᵢ−xᵢβ)²+λ‖β‖² 的解 = 标准 Ridge on (√w·X, √w·y)；
        # predict 时新点不缩放（仅训练加权）。标准化统计量保持原始尺度（H2）。
        # _sw 默认全 1（A2：K-Fold 内部也需按折加权）
        # _w 保留原始权重（σ² 的有效样本量估计用，见下方 df_resid）
        _w = np.ones(len(methods), dtype=np.float64)
        if sample_weights is not None:
            _w = np.array(
                [max(float(sample_weights.get(m, 1.0)), 0.0) for m in methods],
                dtype=np.float64,
            )
        _sw = np.sqrt(_w)
        if sample_weights is not None:
            X_scaled = X_scaled * _sw[:, None]
            y_c = y_c * _sw

        n = len(X_scaled)
        self.n_samples = n
        best_error = None
        self.oos_by_key_ = {}  # 每次 fit 重置；仅完整 K-Fold 路径填充（供 #3 协方差估计）

        if lambda_override is not None:
            # 快速通道：复用既有 λ，不重跑 K-Fold（OOS 不可得，#3 协方差退回 in-sample/零）
            self.lambda_opt = float(lambda_override)
        else:
            # 选 fold 划分：groups 非空 → GroupKFold（同方法跨模型样本同组，防 CV 泄漏，#4）；
            # 否则随机 K-Fold（H-9 余数均衡）
            if groups is not None:
                group_arr = np.asarray([groups.get(m, m) for m in methods])
                unique_groups = list(dict.fromkeys(group_arr.tolist()))
                if len(unique_groups) >= 2:
                    from sklearn.model_selection import GroupKFold

                    k_g = min(self.n_folds, len(unique_groups))
                    splits = list(GroupKFold(n_splits=k_g).split(X_scaled, groups=group_arr))
                else:
                    splits = self._random_kfold_splits(n)
            else:
                splits = self._random_kfold_splits(n)
            k = len(splits)
            if k < 2:
                # 样本太少，直接用中等 λ
                self.lambda_opt = 1.0
                self.cv_errors = []
            else:
                # K-Fold 选 λ（#1 性能优化：每折只算一次 SVD，全部 λ 向量化求解）
                lambda_arr = np.asarray(self.lambda_candidates, dtype=np.float64)
                fold_errors = np.zeros((k, len(lambda_arr)), dtype=np.float64)
                # A2：每折存 fold-local y_mean，OOS 预测还原 Elo 尺度时需加各自折的 y_mean
                fold_cache: list[tuple[np.ndarray, np.ndarray, float]] = []
                for i, (train_idx, test_idx) in enumerate(splits):
                    # A2 修复：per-fold 标准化——用训练折统计量缩放，消除 CV 泄漏
                    # （原实现在全 GT 上算 mean/std → 测试折边际分布泄漏进缩放 → CV 偏乐观）
                    X_tr_raw = X[train_idx]
                    fm = X_tr_raw.mean(axis=0)
                    fs = X_tr_raw.std(axis=0) + 1e-8
                    fk = fs >= RIDGE_DEGENERATE_COL_EPS
                    X_tr = (X_tr_raw - fm) / fs
                    X_tr[:, ~fk] = 0.0
                    X_te = (X[test_idx] - fm) / fs
                    X_te[:, ~fk] = 0.0
                    # 仅训练折乘 √w（与最终模型一致）；验证折保持未加权——
                    # 若验证折也乘 √w，blend 产生的 0 权 donor 行预测=0、目标=0、
                    # 误差恒 0，fold_errors 被系统性压低（1-SE 选 λ 有偏），
                    # 且 fold_cache 的 OOS 预测会带上 √w 缩放、尺度错误
                    X_tr = X_tr * _sw[train_idx, None]
                    # y 用训练折均值中心化（非全 GT 均值）
                    ym = float(y[train_idx].mean())
                    y_tr = (y[train_idx] - ym) * _sw[train_idx]
                    y_te = y[test_idx] - ym

                    U_t, S_t, Vt_t = np.linalg.svd(X_tr, full_matrices=False)
                    Ut_y = U_t.T @ y_tr
                    XVt = X_te @ Vt_t.T
                    scale = S_t[:, None] / (S_t[:, None] ** 2 + lambda_arr[None, :])
                    preds = XVt @ (scale * Ut_y[:, None])
                    fold_errors[i] = np.mean((preds - y_te[:, None]) ** 2, axis=0)
                    fold_cache.append((test_idx, preds, ym))

                avg_errors = fold_errors.mean(axis=0)
                se_errors = fold_errors.std(axis=0) / (k ** 0.5)
                self.cv_errors = avg_errors.tolist()

                # A3 修复：1-SE 规则——在 min(CV)+SE 范围内选最大 λ（更正则、更稳），
                # 原 raw argmin 在 24 网格点 + 小 n 下易挑到"侥幸最低"的偏小 λ
                min_pos = int(np.argmin(avg_errors))
                min_cv = float(avg_errors[min_pos])
                se_at_min = float(se_errors[min_pos]) if np.isfinite(se_errors[min_pos]) else 0.0
                threshold = min_cv + se_at_min
                eligible = np.where(avg_errors <= threshold)[0]
                # 1-SE 仅在 CV 曲线有结构时生效：超过半数 λ eligible 说明曲线过平（无信号或 SE 过大），
                # 1-SE 会盲目推到最大 λ（过正则），此时回退 raw argmin
                if len(eligible) > len(lambda_arr) // 2:
                    best_pos = min_pos
                else:
                    best_pos = int(eligible[-1]) if len(eligible) > 0 else min_pos
                best_error = float(avg_errors[best_pos])
                self.lambda_opt = float(lambda_arr[best_pos])

                # OOS 预测（λ*）：从逐折预测矩阵取 best_pos 列 → #3 协方差估计用
                for test_idx, preds, ym in fold_cache:
                    star = preds[:, best_pos] + ym  # A2：用各自折的 y_mean 还原 Elo 尺度
                    for pos, ti in enumerate(test_idx):
                        self.oos_by_key_[methods[ti]] = float(star[pos])

        # 用最优 λ 在全数据上训练最终模型（截断数值近零奇异值保证稳定）
        U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)
        s_max = S.max() if S.size else 0.0
        keep = max(1e-10 * s_max, 1e-12) < S
        self.singular_values = S
        shrink = np.where(keep, S / (S**2 + self.lambda_opt), 0.0)
        self.w = Vt.T @ (shrink * (U.T @ y_c))

        # 有效自由度 df(λ) = Σ σᵢ²/(σᵢ²+λ)：Ridge 收缩后的有效维度
        self.effective_df = float(np.sum(S**2 / (S**2 + self.lambda_opt))) if S.size else 0.0

        # 残差方差 σ²_噪声（M1 修复）：用自由度修正的 in-sample 残差 RSS/(n−df_eff)
        # 估计**不可约观测噪声**。原实现用 CV-MSE（已含参数不确定性），再在方差公式里乘
        # (1+杠杆) 等于把参数不确定性算了两遍 → CI 系统性偏宽。
        # CV-MSE 保留为 cv_mse_at_lambda 诊断字段。
        self.cv_mse_at_lambda = float(best_error) if (best_error is not None and np.isfinite(best_error)) else None
        if lambda_override is not None and self.sigma2 is not None:
            pass  # 快速通道保留上次 σ²（in-sample 残差在强正则下趋近 0，过于乐观）
        else:
            residuals = y_c - X_scaled @ self.w
            rss = float(np.sum(residuals**2))
            if sample_weights is not None:
                # 加权时用有效样本量 n_eff = (Σw)²/Σw² 替代 n：零权 donor 行
                # 不贡献残差信息，按全额 n 计入会摊薄 σ²、使 CI 系统性偏窄
                n_eff = float(_w.sum() ** 2 / max(float(np.sum(_w**2)), 1e-12))
                df_resid = max(1.0, n_eff - int(round(self.effective_df)))
            else:
                df_resid = max(1, n - int(round(self.effective_df)))
            self.sigma2 = rss / df_resid

        # M-7：σ² 下限——GT Elo 全相同时 σ²=0 → std=0、confidence=1.0（"绝对确定"的
        # 零宽 CI），冷启动早期会误导 infogain 采样的不确定性项。有上限封顶却无下限。
        self.sigma2 = max(self.sigma2, 1e-6)

        # 保存 (X^T X + λI)^(-1) 用于 MAP 方差
        XTX = X_scaled.T @ X_scaled
        self.xtx_inv = np.linalg.inv(XTX + self.lambda_opt * np.eye(XTX.shape[0]))

        # P6：λ* 顶到候选网格上限 = 方法特征对 Elo 基本无信号，模型退化为均值预测。
        # 打标供 predict 降级置信度；快速通道复用顶格 λ 时同样成立（无信号未改善）。
        # 不扩 λ 网格：无信号时更大的 λ 只是更贵地得到均值
        self.no_signal = bool(self.lambda_opt >= float(np.max(self.lambda_candidates)))
        if self.no_signal:
            logger.warning(
                "λ*=%.4g 顶到候选网格上限：方法特征对 Elo 基本无信号，"
                "模型退化为均值预测 (df=%.1f/%d)，预测置信度按边际方差降级",
                self.lambda_opt, self.effective_df, X_scaled.shape[1],
            )

        self.fit_count += 1
        logger.info(
            "EloPredictorModel 训练完成: n=%d, λ*=%.4f, σ²=%.2f, df=%.1f/%d",
            n, self.lambda_opt, self.sigma2, self.effective_df, X_scaled.shape[1],
        )
        return self

    def _random_kfold_splits(self, n: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """随机 K-Fold（H-9 余数均衡：前 r 折各多 1 个，防小 GT 时偏选大 λ）。

        返回 [(train_idx, test_idx), ...]；n 不足 2 折时返回 []。
        """
        k = min(self.n_folds, n)
        if k < 2:
            return []
        indices = np.arange(n)
        rng = np.random.default_rng(_global_seed())
        rng.shuffle(indices)
        fold_size = n // k
        remainder = n % k
        splits = []
        for i in range(k):
            start = i * fold_size + min(i, remainder)
            end = start + fold_size + (1 if i < remainder else 0)
            test_idx = indices[start:end]
            train_idx = np.concatenate([indices[:start], indices[end:]])
            splits.append((train_idx, test_idx))
        return splits

    def predict(self, features_dict: dict, methods: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """
        批量预测 Elo 均值和 MAP 方差。

        返回: (means, variances)，shape 均为 (len(methods),)
        """
        if self.w is None:
            raise ValueError("模型未训练")

        # 严格按训练时的块维度对齐，避免维度不匹配导致静默失败
        X, _ = self.features_to_matrix(features_dict, methods, block_dims=self.block_dims)
        X_scaled = (X - self.x_mean) / self.x_std
        if self.col_keep is not None:
            X_scaled[:, ~self.col_keep] = 0.0

        means = self.y_mean + X_scaled @ self.w
        # 预测方差 = 不可约噪声 σ² + 参数不确定 σ²·x'(XᵀX+λI)⁻¹x
        # P6：λ* 顶格（no_signal）时 w≈0、预测≈y_mean，强正则下残差 σ² 趋小，
        # 不能反映"只会猜均值"的真实不确定性——方差下限提到 GT 边际方差 y_std²，
        # 下游 confidence=1/(1+std/200)（cold_start）与 blend 的 std 权重随之降级
        noise2 = max(self.sigma2, self.y_std ** 2) if self.no_signal else self.sigma2
        variances = noise2 * (
            1.0 + np.sum((X_scaled @ self.xtx_inv) * X_scaled, axis=1)
        )
        return means, variances

    def get_regularization_path(self) -> dict:
        """返回正则化路径信息，用于可视化。"""
        return {
            "lambda_candidates": self.lambda_candidates.tolist(),
            "cv_errors": self.cv_errors,
            "lambda_opt": self.lambda_opt,
            "cv_mse_at_lambda": self.cv_mse_at_lambda,
        }

    def get_pca_summary(self, top_n: int = 20) -> dict:
        """
        返回 SVD 主成分诊断：奇异值谱、解释方差比、累计解释方差、有效自由度。
        用于评估降维效果（df 远小于特征数说明 Ridge/SVD 起到了压缩作用）。
        """
        if self.singular_values is None:
            return {}
        S = self.singular_values
        var = S**2
        total = float(var.sum())
        ratio = var / total if total > 0 else np.zeros_like(var)
        cumulative = np.cumsum(ratio)
        return {
            "n_samples": self.n_samples,
            "n_features": int(len(self.w)) if self.w is not None else 0,
            "singular_values": [round(float(s), 6) for s in S[:top_n]],
            "explained_variance_ratio": [round(float(r), 6) for r in ratio[:top_n]],
            "cumulative_variance_ratio": [round(float(c), 6) for c in cumulative[:top_n]],
            "effective_df": round(self.effective_df, 4) if self.effective_df is not None else None,
            "lambda_opt": self.lambda_opt,
        }

    def get_feature_importance(self, top_n: int = 20) -> list[dict]:
        """按 Ridge 系数×原始 std 降序返回特征重要性（A4 修复）。

        w 是标准化空间的系数，裸 |w| 会系统性高估训练时方差大的特征。
        |w_i| × x_std_i 还原原始尺度，跨特征可比。
        """
        if self.w is None:
            return []
        names = self.feature_names or [f"x_{i}" for i in range(len(self.w))]
        # A4：importance = |w_i| × x_std_i（原始尺度重要性，跨特征可比）
        x_std = self.x_std if self.x_std is not None else np.ones(len(self.w))
        importance = np.abs(self.w) * x_std
        order = np.argsort(-importance)
        return [
            {
                "feature": names[i],
                "coef": round(float(self.w[i]), 4),
                "abs_coef": round(float(abs(self.w[i])), 4),
                "importance": round(float(importance[i]), 4),
            }
            for i in order[:top_n]
        ]
