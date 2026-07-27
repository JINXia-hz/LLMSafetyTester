#!/usr/bin/env python3
"""
基于聚类的 Elo 冷启动预测器。

核心职责：
1. 维护一个只包含真实评估数据的 ground_truth_elo 库。
2. 在 ground_truth 数据增长到一定量时，重新训练聚类模型（动态聚类）。
3. 为新攻击方法预测初始 Elo：找到最近簇，取该簇内 ground truth 方法的平均 Elo。
4. 聚类输入严格只用真实数据，预测值不参与聚类，避免"死数据"污染。

用法：
    from llmsec.evaluation import ClusterEloPredictor

    predictor = ClusterEloPredictor()
    predictor.update_ground_truth("DAN", 1650)
    predictor.fit(attack_records, eval_results, force=True)
    elo_info = predictor.predict("新攻击", record={"prompt": "...", "category": "..."})
"""

import hashlib
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

from llmsec.clustering import (
    CLUSTER_ARTIFACTS_FILE,
    extract_all_features,
    extract_intent_features,
    extract_textual_features,
)
from llmsec.clustering.features import (
    DEFENSE_FEATURE_NAMES,
    INTENT_FEATURE_NAMES,
    TECHNIQUE_LABELS,
    TEXTUAL_FEATURE_NAMES,
)
from llmsec.core.config import INITIAL_ELO
from llmsec.core.logging import get_logger
from llmsec.core.text import MATH_TAX_PATTERN

logger = get_logger(__name__)


class EloPredictorModel:
    """
    基于 SVD 的 Ridge 回归 Elo 预测模型。

    - 用已测方法的特征向量 X 和真实 Elo y（中心化后）训练模型。
    - 对特征矩阵做 SVD：X = UΣV^T，Ridge 解 w(λ) = V (Σ² + λI)^(-1) Σ U^T y。
    - 用 K-Fold 交叉验证在正则化路径 λ ∈ logspace(-3, 4, 24) 上选择最优 λ。
    - SVD 同时提供主成分视角：解释方差比与有效自由度 df(λ) 由 get_pca_summary 输出。
    - 贝叶斯解释：Ridge 等价于高斯先验的 MAP；
      预测均值 E = y_mean + X_test @ w，预测方差 σ² · diag(X_test (X^T X + λI)^(-1) X_test^T)。
    """

    BLOCK_ORDER = ("textual", "embedding", "technique", "intent", "prior")

    def __init__(self, lambda_candidates=None, n_folds: int = 5):
        self.lambda_candidates = (
            np.logspace(-3, 4, 24) if lambda_candidates is None else lambda_candidates
        )
        self.n_folds = n_folds
        self.w: np.ndarray | None = None
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.y_mean: float = 0.0
        self.lambda_opt: float | None = None
        self.sigma2: float | None = None
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

    @classmethod
    def _features_to_matrix(
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
    ) -> "EloPredictorModel":
        """
        用 ground truth 训练 Ridge 回归模型。

        参数:
            features_dict: {method: features}，需包含所有 ground truth 方法
            ground_truth: {method: {"elo": float, ...}}
            feature_name_blocks: 各特征块的特征名（用于特征重要性输出）
            lambda_override: 指定 λ 时跳过 K-Fold 直接 refit w（快速通道，
                用于 ground truth 小幅增长时的增量更新）
        """
        methods = sorted(ground_truth.keys())
        if not methods:
            raise ValueError("ground_truth 为空，无法训练")

        X, self.block_dims = self._features_to_matrix(features_dict, methods)
        y = np.array([ground_truth[m]["elo"] for m in methods], dtype=np.float64)
        self.feature_names = self._resolve_feature_names(feature_name_blocks)

        # X 标准化 + y 中心化（截距项：否则零均值的 X_scaled @ w 无法表达 ~1500 的基准 Elo）
        self.x_mean = X.mean(axis=0)
        self.x_std = X.std(axis=0) + 1e-8
        X_scaled = (X - self.x_mean) / self.x_std
        self.y_mean = float(y.mean())
        y_c = y - self.y_mean

        n = len(X_scaled)
        self.n_samples = n
        best_error = None

        if lambda_override is not None:
            # 快速通道：复用既有 λ，不重跑 K-Fold
            self.lambda_opt = float(lambda_override)
        else:
            k = min(self.n_folds, n)
            if k < 2:
                # 样本太少，直接用中等 λ
                self.lambda_opt = 1.0
                self.cv_errors = []
            else:
                # K-Fold 交叉验证选择 λ
                indices = np.arange(n)
                rng = np.random.default_rng(42)
                rng.shuffle(indices)
                fold_size = n // k

                best_lambda = None
                best_error = float("inf")
                self.cv_errors = []

                for lam in self.lambda_candidates:
                    errors = []
                    for i in range(k):
                        start = i * fold_size
                        end = (i + 1) * fold_size if i < k - 1 else n
                        test_idx = indices[start:end]
                        train_idx = np.concatenate([indices[:start], indices[end:]])

                        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                        y_train, y_test = y_c[train_idx], y_c[test_idx]

                        # 用训练集 SVD 计算 Ridge 解
                        U_train, S_train, Vt_train = np.linalg.svd(X_train, full_matrices=False)
                        w = Vt_train.T @ np.diag(S_train / (S_train**2 + lam)) @ U_train.T @ y_train

                        pred = X_test @ w
                        errors.append(float(np.mean((pred - y_test) ** 2)))

                    avg_error = float(np.mean(errors))
                    self.cv_errors.append(avg_error)
                    if avg_error < best_error:
                        best_error = avg_error
                        best_lambda = lam

                self.lambda_opt = float(best_lambda)

        # 用最优 λ 在全数据上训练最终模型（截断数值近零奇异值保证稳定）
        U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)
        s_max = S.max() if S.size else 0.0
        keep = S > max(1e-10 * s_max, 1e-12)
        self.singular_values = S
        shrink = np.where(keep, S / (S**2 + self.lambda_opt), 0.0)
        self.w = Vt.T @ (shrink * (U.T @ y_c))

        # 残差方差：优先用 K-Fold 在 λ* 上的交叉验证误差（out-of-sample 估计）；
        # 快速通道保留上次 K-Fold 的 σ²（in-sample 残差会趋近于 0，置信区间过于乐观）；
        # 其余情况退回训练集残差
        if best_error is not None and np.isfinite(best_error):
            self.sigma2 = float(best_error)
        elif lambda_override is not None and self.sigma2 is not None:
            pass
        else:
            residuals = y_c - X_scaled @ self.w
            self.sigma2 = float(np.mean(residuals**2))

        # 有效自由度 df(λ) = Σ σᵢ²/(σᵢ²+λ)：Ridge 收缩后的有效维度
        self.effective_df = float(np.sum(S**2 / (S**2 + self.lambda_opt))) if S.size else 0.0

        # 保存 (X^T X + λI)^(-1) 用于 MAP 方差
        XTX = X_scaled.T @ X_scaled
        self.xtx_inv = np.linalg.inv(XTX + self.lambda_opt * np.eye(XTX.shape[0]))

        self.fit_count += 1
        logger.info(
            "EloPredictorModel 训练完成: n=%d, λ*=%.4f, σ²=%.2f, df=%.1f/%d",
            n, self.lambda_opt, self.sigma2, self.effective_df, X_scaled.shape[1],
        )
        return self

    def predict(self, features_dict: dict, methods: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """
        批量预测 Elo 均值和 MAP 方差。

        返回: (means, variances)，shape 均为 (len(methods),)
        """
        if self.w is None:
            raise ValueError("模型未训练")

        # 严格按训练时的块维度对齐，避免维度不匹配导致静默失败
        X, _ = self._features_to_matrix(features_dict, methods, block_dims=self.block_dims)
        X_scaled = (X - self.x_mean) / self.x_std

        means = self.y_mean + X_scaled @ self.w
        variances = self.sigma2 * np.sum((X_scaled @ self.xtx_inv) * X_scaled, axis=1)
        return means, variances

    def get_regularization_path(self) -> dict:
        """返回正则化路径信息，用于可视化。"""
        return {
            "lambda_candidates": self.lambda_candidates.tolist(),
            "cv_errors": self.cv_errors,
            "lambda_opt": self.lambda_opt,
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
        """按 Ridge 系数绝对值降序返回特征重要性。"""
        if self.w is None:
            return []
        names = self.feature_names or [f"x_{i}" for i in range(len(self.w))]
        order = np.argsort(-np.abs(self.w))
        return [
            {
                "feature": names[i],
                "coef": round(float(self.w[i]), 4),
                "abs_coef": round(float(abs(self.w[i])), 4),
            }
            for i in order[:top_n]
        ]


class ClusterEloPredictor:
    """
    新攻击方法的 Elo 冷启动预测器。

    - SVD-Ridge 批量预测为主（predict_batch），同后缀/同基底变体平均为兜底（predict）。
    - 前置树聚类（tree_fit）提供采样分层结构与特征缓存；D-optimality 负责种子选择。
    - 聚类只用先验特征，ground truth 增长不触发重聚类；ridge 模型按 GT 指纹缓存。
    """

    def __init__(
        self,
        threshold: int = 10,
        min_cluster_size: int = 3,
    ):
        self.threshold = threshold
        self.min_cluster_size = min_cluster_size

        # ground truth 库：只记录真实评估过的方法（由 ELOTracker 统一持久化到 state.json）
        self.ground_truth: dict[str, dict] = {}

        self.last_fit_gt_count: int = 0
        self.last_fit_at: str | None = None
        self.artifacts: dict | None = None

        # 机器学习预测模型（SVD-Ridge）
        self.model = EloPredictorModel()
        # 最近一次批量预测结果（含 MAP 不确定性），供聚类安全分析输出
        self.last_predictions: dict[str, dict] = {}
        # 模型缓存：ground truth 未变时复用 w，避免每轮重跑 K-Fold
        self._model_gt_hash: str | None = None
        self._model_cv_gt_count: int = 0  # 上次完整 K-Fold 时的 GT 数

        self._load_artifacts()

    # ============================================================
    # artifacts 持久化（ground truth 已由 ELOTracker 统一保存）
    # ============================================================
    def _load_artifacts(self):
        """从磁盘加载聚类 artifacts（labels/features 覆盖全部方法，含未测方法）。"""
        if CLUSTER_ARTIFACTS_FILE.exists():
            try:
                self.artifacts = joblib.load(CLUSTER_ARTIFACTS_FILE)
                # 丢弃训练期使用的完整 dist_matrix，节省内存
                self.artifacts.pop("dist_matrix", None)
                self.last_fit_gt_count = int(
                    self.artifacts.get("ground_truth_count", self.last_fit_gt_count)
                )
                self.last_fit_at = self.artifacts.get("generated_at", self.last_fit_at)
            except Exception as e:
                logger.warning("加载 cluster artifacts 失败: %s", e)
                self.artifacts = None

    def _save_artifacts(self):
        """保存聚类 artifacts（不含 dist_matrix）。"""
        if self.artifacts is None:
            return
        # 确保不保存完整 dist_matrix
        self.artifacts.pop("dist_matrix", None)
        os.makedirs(os.path.dirname(CLUSTER_ARTIFACTS_FILE) or ".", exist_ok=True)
        joblib.dump(self.artifacts, CLUSTER_ARTIFACTS_FILE)

    # ============================================================
    # ground truth 管理
    # ============================================================
    def update_ground_truth(self, method: str, elo: float):
        """
        把真实评估后的方法及其 Elo 写入 ground truth 库。

        参数:
            method: 攻击方法名
            elo: 该方法的真实 Elo（通常取当前最终 Elo）
        """
        now = datetime.now().isoformat()
        if method in self.ground_truth:
            self.ground_truth[method]["elo"] = round(float(elo), 2)
            self.ground_truth[method]["last_updated_at"] = now
        else:
            self.ground_truth[method] = {
                "elo": round(float(elo), 2),
                "first_seen_at": now,
                "last_updated_at": now,
            }

    def is_ground_truth(self, method: str) -> bool:
        return method in self.ground_truth

    def ground_truth_count(self) -> int:
        return len(self.ground_truth)

    def _ground_truth_hash(self) -> str:
        """ground truth（方法名 + Elo）的指纹，用于判断预测模型是否需要重训。"""
        content = ",".join(
            f"{m}:{self.ground_truth[m]['elo']}" for m in sorted(self.ground_truth)
        )
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    # ============================================================
    # 前置树聚类 / D-optimality 种子 / 最终聚类
    # ============================================================
    def tree_fit(self, attack_records: list[dict]) -> dict | None:
        """
        前置树聚类：提取先验特征 → 马氏白化空间 → Ward 树 + 拐点 auto-k。
        不依赖 ground truth：为采样器提供分层结构，为 ridge / D-optimal 提供特征缓存。

        返回: 树聚类报告 dict；未触发时返回 None。
        """
        if len(attack_records) < 2:
            logger.warning("攻击记录不足，跳过树聚类")
            return None

        from llmsec.clustering import run_tree_clustering

        logger.info("🌲 前置树聚类: 总方法记录 %d 条", len(attack_records))
        features, meta = extract_all_features(attack_records, eval_results=[])
        report = run_tree_clustering(features, meta)

        self.artifacts = joblib.load(CLUSTER_ARTIFACTS_FILE)
        self.artifacts["method_set_hash"] = _compute_method_set_hash(
            sorted(self.artifacts.get("labels", {}).keys())
        )
        self.last_fit_at = datetime.now().isoformat()
        self._save_artifacts()

        logger.info(
            "✅ 前置树聚类完成: k*=%d (top3 %s), silhouette=%.4f",
            report.get("n_clusters", 0),
            report.get("top_ks", []),
            report.get("validation", {}).get("silhouette", 0.0),
        )
        return report

    def select_d_optimal_seeds(
        self,
        method_records: dict[str, dict],
        n: int,
    ) -> list[str]:
        """
        D-optimality 种子选择：贪心选出对预测矩阵信息量最大的 n 个未测方法
        （xᵀ(X_gtᵀX_gt + λI)⁻¹x 最大，Sherman–Morrison 秩1更新）。
        GT 为空时 M = λI，自动退化为最大杠杆点，覆盖特征空间。
        特征不可用时回退随机采样。
        """
        import random

        from llmsec.evaluation.active_learning import greedy_d_optimal

        candidates = [m for m in method_records if m not in self.ground_truth]
        if not candidates:
            return []

        features = self.artifacts.get("features", {}) if self.artifacts else {}
        meta = self.artifacts.get("meta", {}) if self.artifacts else {}

        missing = [m for m in candidates if m not in features]
        extra: dict = {}
        if missing:
            try:
                extra = self._extract_features_for_methods(missing, method_records, meta)
            except Exception as e:
                logger.warning("种子特征提取失败，回退随机采样: %s", e)
                random.shuffle(candidates)
                return candidates[:n]

        def _vec(m: str) -> dict:
            base = features.get(m) or extra.get(m) or {}
            feat = dict(base)
            feat["prior"] = build_prior_features(m, method_records.get(m))
            return feat

        cand_features = {m: _vec(m) for m in candidates}
        X, dims = EloPredictorModel._features_to_matrix(cand_features, candidates)
        mean, std = X.mean(axis=0), X.std(axis=0) + 1e-8
        X_scaled = (X - mean) / std

        gt_methods = [m for m in sorted(self.ground_truth) if m in features]
        X_gt = None
        if gt_methods:
            gt_feats = {m: _vec(m) for m in gt_methods}
            Xg, _ = EloPredictorModel._features_to_matrix(
                gt_feats, gt_methods, block_dims=dims
            )
            X_gt = (Xg - mean) / std

        lam = self.model.lambda_opt or 1.0
        idx = greedy_d_optimal(X_scaled, n, lam=lam, X_gt=X_gt)
        return [candidates[i] for i in idx]

    def predict(
        self,
        method: str,
        record: dict | None = None,
    ) -> dict:
        """
        单方法兜底预测（SVD-Ridge 不可用时的降级链；批量预测请用 predict_batch）。

        预测优先级：
        1. 已是 ground truth → 直接返回真实 Elo
        2. 同后缀变体（如 *_rot13 / *_b64 / *_code / *_story）已有 GT → 变体平均
        3. 同基底变体已有 GT → 变体平均
        4. 全局 ground truth 简单平均
        """
        labels = self.artifacts.get("labels", {}) if self.artifacts else {}
        cid = int(labels.get(method, -1)) if method in labels else -1

        if method in self.ground_truth:
            return {
                "elo": self.ground_truth[method]["elo"],
                "source": "ground_truth",
                "cluster_id": None,
                "confidence": 1.0,
                "based_on_gt_count": self.ground_truth_count(),
            }

        if not self.ground_truth:
            return {
                "elo": float(INITIAL_ELO),
                "source": "predicted",
                "cluster_id": None,
                "confidence": 0.0,
                "based_on_gt_count": 0,
            }

        # ---- 1. 同后缀变体兜底（如 *_rot13 / *_b64 / *_code / *_story） ----
        suffix_gt = self._find_suffix_variant_ground_truth(method)
        if suffix_gt:
            avg = sum(self.ground_truth[m]["elo"] for m in suffix_gt) / len(suffix_gt)
            return {
                "elo": round(avg, 2),
                "source": "predicted_suffix_variant",
                "cluster_id": cid,
                "confidence": round(min(len(suffix_gt) / 3, 1.0), 4),
                "based_on_gt_count": self.ground_truth_count(),
            }

        # ---- 2. 同基底变体兜底 ----
        variant_gt = self._find_variant_ground_truth(method)
        if variant_gt:
            avg = sum(self.ground_truth[m]["elo"] for m in variant_gt) / len(variant_gt)
            return {
                "elo": round(avg, 2),
                "source": "predicted_variant",
                "cluster_id": cid,
                "confidence": round(min(len(variant_gt) / 2, 1.0), 4),
                "based_on_gt_count": self.ground_truth_count(),
            }

        # ---- 3. 全局简单平均 ----
        avg = sum(g["elo"] for g in self.ground_truth.values()) / len(self.ground_truth)
        return {
            "elo": round(avg, 2),
            "source": "predicted_global",
            "cluster_id": cid,
            "confidence": round(min(self.ground_truth_count() / 10, 0.5), 4),
            "based_on_gt_count": self.ground_truth_count(),
        }

    # ============================================================
    # SVD-Ridge 批量预测
    # ============================================================
    def predict_batch(self, method_records: dict[str, dict]) -> dict[str, dict]:
        """
        批量预测未测方法的初始 Elo。

        - ground truth 数 >= min_cluster_size 且特征可用时：
          训练 SVD-Ridge 模型（K-Fold 选 λ），一次前向传播得到预测均值与 MAP 方差。
        - 否则回退到逐方法的同后缀/同基底变体简单平均（predict）。

        返回: {method: {"elo", "source", "std", "ci95", "confidence", ...}}
        """
        methods = [m for m in method_records if m not in self.ground_truth]
        if not methods:
            self.last_predictions = {}
            return {}

        gt_count = self.ground_truth_count()
        features = self.artifacts.get("features", {}) if self.artifacts else {}
        gt_methods = sorted(self.ground_truth.keys())
        use_model = (
            gt_count >= self.min_cluster_size
            and bool(features)
            and all(m in features for m in gt_methods)
        )

        results: dict[str, dict] = {}
        if use_model:
            try:
                results = self._predict_batch_svd_ridge(methods, method_records, features)
            except Exception as e:
                logger.warning("SVD-Ridge 批量预测失败，回退到变体平均: %s", e)

        if not results:
            results = {m: self.predict(m, method_records.get(m)) for m in methods}

        self.last_predictions = results
        return results

    def _predict_batch_svd_ridge(
        self,
        methods: list[str],
        method_records: dict[str, dict],
        features: dict,
    ) -> dict[str, dict]:
        """SVD-Ridge 批量预测主流程：训练 → 预测均值/方差 → 组装结果。"""
        meta = self.artifacts.get("meta", {}) if self.artifacts else {}
        labels = self.artifacts.get("labels", {}) if self.artifacts else {}
        gt_methods = sorted(self.ground_truth.keys())

        # 为缺失特征的未测方法批量提取特征（复用训练时的 vectorizer/PCA 保证同一特征空间）
        missing = [m for m in methods if m not in features]
        extra_features = {}
        if missing:
            extra_features = self._extract_features_for_methods(missing, method_records, meta)

        def _with_prior(method: str) -> dict:
            base = features.get(method) or extra_features.get(method) or {}
            feat = dict(base)
            feat["prior"] = build_prior_features(method, method_records.get(method))
            return feat

        train_features = {m: _with_prior(m) for m in gt_methods}
        test_features = {m: _with_prior(m) for m in methods}

        feature_name_blocks = {
            "textual": meta.get("textual_feature_names", TEXTUAL_FEATURE_NAMES),
            "technique": meta.get("technique_label_names", []),
            "intent": meta.get("intent_feature_names", INTENT_FEATURE_NAMES),
            "prior": PRIOR_FEATURE_NAMES,
        }

        # 模型缓存：GT 未变 → 直接复用 w（纯矩阵预测）；
        # GT 小幅增长 → 用现有 λ* 单次 SVD 快速 refit；
        # GT 增长 ≥ threshold → 重跑 K-Fold 选 λ
        gt_hash = self._ground_truth_hash()
        gt_count = self.ground_truth_count()
        if self.model.w is not None and gt_hash == self._model_gt_hash:
            logger.info("SVD-Ridge 复用缓存模型 (ground truth %d 未变)", gt_count)
        elif (
            self.model.w is not None
            and self.model.lambda_opt is not None
            and 0 < gt_count - self._model_cv_gt_count < self.threshold
        ):
            self.model.fit(
                train_features, self.ground_truth, feature_name_blocks,
                lambda_override=self.model.lambda_opt,
            )
            logger.info("SVD-Ridge 快速 refit (λ*=%.4f 复用, ground truth %d)",
                        self.model.lambda_opt, gt_count)
        else:
            self.model.fit(train_features, self.ground_truth, feature_name_blocks)
            self._model_cv_gt_count = gt_count
        self._model_gt_hash = gt_hash

        means, variances = self.model.predict(test_features, methods)

        results = {}
        for m, mean, var in zip(methods, means, variances):
            std = float(np.sqrt(max(float(var), 0.0)))
            elo = round(float(mean), 2)
            results[m] = {
                "elo": elo,
                "source": "svd_ridge",
                "cluster_id": int(labels.get(m, -1)) if m in labels else -1,
                "std": round(std, 2),
                "ci95": [round(elo - 1.96 * std, 2), round(elo + 1.96 * std, 2)],
                "confidence": round(1.0 / (1.0 + std / 200.0), 4),
                "based_on_gt_count": self.ground_truth_count(),
            }

        logger.info(
            "SVD-Ridge 批量预测: %d 个未测方法 (ground truth %d, λ*=%.4f, σ²=%.2f)",
            len(methods),
            self.ground_truth_count(),
            self.model.lambda_opt,
            self.model.sigma2,
        )
        return results

    def _find_variant_ground_truth(self, method: str) -> list[str]:
        """
        找与 method 同一攻击基底的其它变体（去掉 _rot13/_b64/_code/_story/_N 等后缀）。
        返回这些变体中已有 ground truth 的方法名列表。
        """
        if not self.ground_truth:
            return []
        base = _strip_variant_suffix(method)
        if not base:
            return []
        variants = []
        for gt_method in self.ground_truth.keys():
            if gt_method == method:
                continue
            if _strip_variant_suffix(gt_method) == base:
                variants.append(gt_method)
        return variants

    def _find_suffix_variant_ground_truth(self, method: str, max_members: int = 8) -> list[str]:
        """
        找与 method 同后缀的其它变体（如 *_rot13 / *_b64 / *_code / *_story）。
        返回这些变体中已有 ground truth 的方法名列表，最多返回 max_members 个。
        """
        if not self.ground_truth:
            return []
        suffix = _extract_variant_suffix(method)
        if not suffix:
            return []
        variants = []
        for gt_method in self.ground_truth.keys():
            if gt_method == method:
                continue
            if _extract_variant_suffix(gt_method) == suffix:
                variants.append(gt_method)
        return variants[:max_members]

    def final_fit(
        self,
        attack_records: list[dict],
        eval_results: list[dict],
    ) -> dict | None:
        """
        攻击完成后最终聚类：树聚类(auto-k) + 白化空间递归 DBSCAN 密度视图。
        用全部真实评估数据重建特征空间（后验特征仅用于画像，不进入度量）。

        返回: 最终聚类报告 dict。
        """
        if len(attack_records) < 2:
            logger.warning("攻击记录不足，跳过最终聚类")
            return None

        from llmsec.clustering import run_final_tree_clustering

        logger.info("🏁 最终聚类: 总方法记录 %d 条，评估结果 %d 条", len(attack_records), len(eval_results))
        features, meta = extract_all_features(attack_records, eval_results)
        report = run_final_tree_clustering(features, meta)

        self.artifacts = joblib.load(CLUSTER_ARTIFACTS_FILE)
        self.artifacts["is_final_cluster"] = True
        self.artifacts["ground_truth_count"] = self.ground_truth_count()
        self.artifacts["ground_truth_methods"] = sorted(self.ground_truth.keys())
        self.artifacts["method_set_hash"] = _compute_method_set_hash(
            sorted(self.artifacts.get("labels", {}).keys())
        )
        self.last_fit_gt_count = self.ground_truth_count()
        self.last_fit_at = datetime.now().isoformat()
        self._save_artifacts()

        logger.info(
            "✅ 最终聚类完成: 树 k*=%d, DBSCAN核心簇=%d, 噪声=%d, silhouette=%.4f",
            report.get("n_clusters", 0),
            report.get("dbscan", {}).get("n_core_clusters", 0),
            report.get("dbscan", {}).get("n_noise", 0),
            report.get("validation", {}).get("silhouette", 0.0),
        )
        return report


    # ============================================================
    # 特征提取（预测用）
    # ============================================================
    def _build_technique_vector(self, record: dict, label_names: list[str]) -> np.ndarray:
        """为单条记录构造与 artifacts 中维度一致的技术标签向量。"""
        vec = np.zeros(len(label_names))
        prompt = record.get("prompt", "").lower()

        # 技术标签
        for i, (label, patterns) in enumerate(TECHNIQUE_LABELS.items()):
            if i >= len(label_names):
                break
            for pat in patterns:
                if re.search(pat, prompt):
                    vec[i] = 1.0
                    break

        # harm_type / category
        harm_type = record.get("harm_type", "")
        category = record.get("category", "")
        harm_key = f"harm:{harm_type}" if harm_type else None
        cat_key = f"cat:{category}" if category else None
        if harm_key and harm_key in label_names:
            vec[label_names.index(harm_key)] = 1.0
        if cat_key and cat_key in label_names:
            vec[label_names.index(cat_key)] = 1.0

        return vec

    def _extract_embeddings_batch(
        self,
        prompts: list[str],
        vectorizer,
        pca,
    ) -> np.ndarray:
        """为多个 prompt 批量提取与训练时同维度的 embedding（模型只加载一次）。"""
        from llmsec.core.text import strip_math_tax

        cleaned = [strip_math_tax(p) for p in prompts]

        # TF-IDF 路径
        if vectorizer is not None:
            dense = vectorizer.transform(cleaned).toarray()
            return pca.transform(dense) if pca is not None else dense

        # sentence-transformers 路径
        try:
            from sentence_transformers import SentenceTransformer
            model_name = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            model = SentenceTransformer(model_name)
        except Exception as e:
            logger.warning("批量预测时加载 embedding 模型失败: %s", e)
            dim = pca.n_components if pca is not None else 384
            return np.zeros((len(prompts), dim), dtype=np.float64)

        emb = model.encode(cleaned, show_progress_bar=False, batch_size=32)
        return pca.transform(emb) if pca is not None else emb

    def _extract_features_for_methods(
        self,
        methods: list[str],
        method_records: dict[str, dict],
        meta: dict,
    ) -> dict[str, dict[str, np.ndarray]]:
        """为多个未测方法批量提取与训练时同维度、同尺度的特征块。"""
        prompts = [method_records[m].get("prompt", "") for m in methods]
        vectorizer = meta.get("embedding_artifacts", {}).get("vectorizer")
        pca = meta.get("embedding_artifacts", {}).get("pca")
        embeddings = np.atleast_2d(self._extract_embeddings_batch(prompts, vectorizer, pca))

        textual_names = meta.get("textual_feature_names", TEXTUAL_FEATURE_NAMES)
        technique_names = meta.get("technique_label_names", [])
        defense_names = meta.get("defense_feature_names", DEFENSE_FEATURE_NAMES)

        method_to_idx = {m: i for i, m in enumerate(methods)}
        method_prompts = {m: [method_records[m].get("prompt", "")] for m in methods}
        intent_feats = extract_intent_features(methods, method_prompts, embeddings, method_to_idx)

        result = {}
        for i, m in enumerate(methods):
            rec = method_records[m]
            textual_dict = extract_textual_features(rec.get("prompt", ""))
            textual_vec = np.array(
                [textual_dict.get(k, 0.0) for k in textual_names], dtype=np.float64
            )
            result[m] = {
                "textual": textual_vec,
                "embedding": np.asarray(embeddings[i], dtype=np.float64),
                "technique": self._build_technique_vector(rec, technique_names),
                "intent": intent_feats.get(m, np.zeros(len(INTENT_FEATURE_NAMES))),
                "defense": np.zeros(len(defense_names), dtype=np.float64),
                "cross_model": np.array([], dtype=np.float64),
            }
        return result

    # ============================================================
    # 状态查询
    # ============================================================
    def get_status(self) -> dict:
        """返回当前预测器状态摘要。"""
        n_gt = self.ground_truth_count()
        n_clusters = 0
        n_noise = 0
        cluster_names = {}
        if self.artifacts:
            n_clusters = len(set(self.artifacts.get("labels", {}).values()) - {-1})
            n_noise = sum(1 for v in self.artifacts.get("labels", {}).values() if v == -1)
            cluster_names = self.artifacts.get("cluster_names", {})

        return {
            "ground_truth_count": n_gt,
            "predicted_count": len(self.last_predictions),
            "last_fit_gt_count": self.last_fit_gt_count,
            "last_fit_at": self.last_fit_at,
            "next_fit_at_gt_count": (
                self.last_fit_gt_count + self.threshold if self.last_fit_gt_count else self.min_cluster_size
            ),
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "cluster_names": cluster_names,
        }


# ============================================================
# 模块级辅助函数
# ============================================================
_VARIANT_SUFFIX_RE = re.compile(r"(_rot13|_b64|_base64|_code|_story|_\d+)$", re.IGNORECASE)

# 先验特征：无需真实评估即可从方法名 / prompt 推导，作为 SVD-Ridge 的额外输入
PRIOR_FEATURE_NAMES = [
    "name_char_len",          # 方法名长度
    "name_token_count",       # 方法名分词数
    "suffix_rot13",           # 变体后缀 one-hot
    "suffix_b64",
    "suffix_code",
    "suffix_story",
    "suffix_numeric",
    "has_math_tax",           # prompt 是否含数学越狱税
    "prompt_line_count_log",  # prompt 行数（log1p）
]


def build_prior_features(method: str, record: dict | None = None) -> np.ndarray:
    """
    构造先验特征向量：只依赖方法名与 prompt，不需要任何评估结果。
    变体后缀（rot13/b64/code/story/数字）与数学越狱税是与 Elo 强相关的先验信号。
    """
    suffix = _extract_variant_suffix(method)
    prompt = (record or {}).get("prompt", "") or ""
    tokens = [t for t in re.split(r"[_\s\-]+", method) if t]
    return np.array([
        float(len(method)),
        float(len(tokens)),
        1.0 if suffix == "rot13" else 0.0,
        1.0 if suffix == "b64" else 0.0,
        1.0 if suffix == "code" else 0.0,
        1.0 if suffix == "story" else 0.0,
        1.0 if suffix.isdigit() else 0.0,
        1.0 if MATH_TAX_PATTERN.search(prompt) else 0.0,
        float(math.log1p(prompt.count("\n") + 1)),
    ], dtype=np.float64)


def _strip_variant_suffix(method_name: str) -> str:
    """去掉方法名末尾的变体后缀（如 _rot13/_b64/_code/_story/_0），得到攻击基底名。"""
    return _VARIANT_SUFFIX_RE.sub("", method_name)


def _extract_variant_suffix(method_name: str) -> str:
    """提取方法名末尾的变体后缀（如 rot13 / b64 / code / story / 0），无后缀返回空字符串。"""
    m = _VARIANT_SUFFIX_RE.search(method_name)
    if not m:
        return ""
    suffix = m.group(1).lstrip("_").lower()
    # 统一别名
    if suffix == "base64":
        return "b64"
    return suffix


def _compute_method_set_hash(methods: list[str]) -> str:
    """计算方法集合的指纹 hash，用于判断攻击集是否发生变化。"""
    content = ",".join(sorted(set(methods)))
    return hashlib.md5(content.encode("utf-8")).hexdigest()


# ============================================================
# CLI: 查看状态
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="聚类 Elo 预测器状态")
    parser.add_argument(
        "--status",
        action="store_true",
        help="打印当前 ground truth / 预测 / 聚类状态",
    )
    args = parser.parse_args()

    if args.status:
        predictor = ClusterEloPredictor()
        status = predictor.get_status()
        print("=" * 60)
        print("📊 ClusterEloPredictor 状态")
        print("=" * 60)
        print(f"  ground truth 方法数: {status['ground_truth_count']}")
        print(f"  预测缓存方法数: {status['predicted_count']}")
        print(f"  上次训练 ground truth 数: {status['last_fit_gt_count']}")
        print(f"  上次训练时间: {status['last_fit_at'] or '未训练'}")
        print(f"  下次触发训练需 ≥: {status['next_fit_at_gt_count']} 个 ground truth")
        print(f"  当前簇数: {status['n_clusters']}")
        print(f"  噪声点数: {status['n_noise']}")
        if status["cluster_names"]:
            print("  簇名称:")
            for cid, name in sorted(status["cluster_names"].items(), key=lambda x: int(x[0])):
                print(f"    簇 {cid}: {name}")
        print("=" * 60)
    else:
        parser.print_help()
