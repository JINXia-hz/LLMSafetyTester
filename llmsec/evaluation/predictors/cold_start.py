#!/usr/bin/env python3
"""
evaluation.predictors.cold_start — Elo 冷启动预测器（编排层）。

核心职责：
1. 维护一个只包含真实评估数据的 ground_truth_elo 库。
2. 前置特征缓存（fit_features）供 SVD-Ridge 预测与 D-optimality 种子使用；
   聚类只在整个测试流程结束后进行（final_fit，HDBSCAN），聚类输入严格只用真实数据。
3. 为新攻击方法预测初始 Elo：SVD-Ridge 批量预测为主（predict_batch），
   同后缀/同基底变体平均与全局平均为兜底（predict）。

本文件只含编排器 ColdStartPredictor（GT 库 / 特征缓存 / 工件持久化 / 变体兜底 /
预测分发）。纯 ML 模型见 svd_ridge.py 的 EloPredictorModel。
ClusterEloPredictor → ColdStartPredictor 改名（M-42：原名暗示聚类，实为预测器）。

用法：
    from llmsec.evaluation.predictors.cold_start import ColdStartPredictor

    predictor = ColdStartPredictor()
    predictor.update_ground_truth("DAN", 1650)
    predictor.fit(attack_records, eval_results, force=True)
    elo_info = predictor.predict("新攻击")
"""

import hashlib
import os
import random
import re
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import t as _t_dist

from llmsec.clustering import (
    extract_all_features,
    extract_intent_features,
    extract_textual_features,
    parse_cluster_id,
)
from llmsec.clustering.features import (
    DEFENSE_FEATURE_NAMES,
    INTENT_FEATURE_NAMES,
    PRIOR_FEATURE_NAMES,
    TECHNIQUE_LABELS,
    TEXTUAL_FEATURE_NAMES,
    _extract_variant_suffix,
    _strip_variant_suffix,
    build_prior_features,
)
from llmsec.core import config
from llmsec.core.config import INITIAL_ELO
from llmsec.core.io import save_artifact
from llmsec.core.logging import get_logger
from llmsec.core.seed import get_global_seed as _global_seed
from llmsec.evaluation.predictors.active_learning import greedy_d_optimal
from llmsec.evaluation.predictors.svd_ridge import EloPredictorModel
from llmsec.params import (
    RIDGE_PRED_STD_CAP_MIN,
    RIDGE_PRED_STD_CAP_MULT,
    RIDGE_REFIT_THRESHOLD,
)

# 特征提取代码版本：提取逻辑 / 特征块结构变更时 +1，使旧特征缓存与 ridge w 失效（M-5/M-6）
FEATURE_EXTRACTION_VERSION = 1

# 无模型方差可用时（GT 为空 / 变体兜底 / 全局平均）的保守 std 启发值（Elo 分）：
# 原实现挪用 RIDGE_PRED_STD_CAP_MIN（=200，本是 std 上限的绝对下限）当兜底 std，
# 与 _predict_batch_svd_ridge 里 confidence 硬编码的 std/200 实为同一拍脑袋常数，
# 此处收口为单一模块级常量，取 200 = 半个 ELO_SCALE 量级，经验值
_FALLBACK_STD = 200.0



logger = get_logger(__name__)

def current_feature_config_hash() -> str:
    """当前特征空间配置指纹（md5[:8]）。

    内容 = (embedding 来源/模型, EMBEDDING_PCA_DIM, 特征提取代码版本)。
    任一变化都意味着旧特征缓存与旧 ridge 权重不可复用；
    fit_features 把它写入 artifacts['meta']，runner 用它做特征缓存失效判断（M-6）。
    """
    from llmsec import params
    from llmsec.clustering import features as _feat

    source = _feat._embedding_source  # None=尚未提取过（或 TF-IDF 兜底）
    if source == "api":
        model = os.environ.get("EMBEDDING_API_MODEL", "")
    else:
        model = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    content = (
        f"{source or 'tfidf'}:{model}"
        f"|pca_dim={params.EMBEDDING_PCA_DIM}"
        f"|v{FEATURE_EXTRACTION_VERSION}"
    )
    return hashlib.md5(content.encode("utf-8")).hexdigest()[:8]


# FEATURE_CACHE_FILE / CLUSTER_RESULT_FILE 运行时经 config 动态读取（runner
# work-dir 重绑 _cfg 后即生效，见 pipeline/runner.py 隔离模式）。
def _cluster_files() -> tuple[Path, Path]:
    """返回 (feature_cache_file, cluster_result_file)，动态读 core.config。"""
    return config.FEATURE_CACHE_FILE, config.CLUSTER_RESULT_FILE


class ColdStartPredictor:
    """
    新攻击方法的 Elo 冷启动预测器。

    - SVD-Ridge 批量预测为主（predict_batch），同后缀/同基底变体平均为兜底（predict）。
    - 前置特征缓存（fit_features）供 ridge / D-optimal 使用；聚类只在测试结束后进行（final_fit，HDBSCAN）。
    - 聚类只用先验特征，ground truth 增长不触发重聚类；ridge 模型按 GT 指纹缓存。
    """

    def __init__(
        self,
        ridge_refit_threshold: int = RIDGE_REFIT_THRESHOLD,
        # 命名是历史残留：聚类已移到 final_fit（HDBSCAN 自行定簇），此参数实为
        # SVD-Ridge 训练所需的最小 ground truth 数（< 该值走变体/全局平均兜底）。
        # 不改名——影响面大（runner/HPO/看板多处按现名注入）
        min_cluster_size: int = 3,
    ):
        self.ridge_refit_threshold = ridge_refit_threshold
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
        # 特征空间签名（M-5：方法集合 + 特征维度 + feature_config_hash），
        # 特征重提取后旧 w 不得复用——原缓存只看 GT 指纹，换特征空间后 w 静默错位
        self._model_feature_sig: str | None = None
        self._model_cv_gt_count: int = 0  # 上次完整 K-Fold 时的 GT 数

        self._load_artifacts()

    # ============================================================
    # artifacts 持久化（ground truth 已由 ELOTracker 统一保存）
    # ------------------------------------------------------------
    # 按写者拆分两个文件（修原 cluster_artifacts.pkl 双写者不同 schema 互覆盖）：
    #   feature_cache.pkl — 先验特征缓存，仅 fit_features 经 _save_artifacts 写
    #   cluster_result.pkl — 完整聚类产物，hdb 写、final_fit 增补（见 final_fit）
    # ============================================================
    def _load_artifacts(self):
        """从磁盘恢复特征缓存（features/meta 供 ridge / D-optimal 使用）。

        优先读 cluster_result.pkl（labels 只存在于此文件）；缺失则回退
        feature_cache.pkl 取 features（无 labels，见 M-4）。
        聚类拟合元信息（last_fit_gt_count/last_fit_at）从 cluster_result.pkl 恢复。
        """
        self.artifacts = None
        feature_cache_file, cluster_result_file = _cluster_files()
        # M-4：优先读 cluster_result.pkl——labels 只存在于此文件；feature_cache.pkl 含
        # features 但无 labels。原顺序先读 feature_cache，重启后 cluster_id 恒 -1、
        # get_status() 显示 n_clusters=0。cluster_result 缺失才回退 feature_cache 取 features。
        for p in (cluster_result_file, feature_cache_file):
            if p.exists():
                try:
                    a = joblib.load(p)
                    a.pop("dist_matrix", None)  # 丢弃训练期完整距离矩阵，省内存
                    if "features" in a:
                        self.artifacts = a
                        break
                except Exception as e:
                    logger.warning("加载 %s 失败: %s", p.name, e)

        # 拟合元信息从聚类结果文件恢复（决定是否触发重聚类）
        if cluster_result_file.exists():
            try:
                cr = joblib.load(cluster_result_file)
                self.last_fit_gt_count = int(
                    cr.get("ground_truth_count", self.last_fit_gt_count)
                )
                self.last_fit_at = cr.get("generated_at", self.last_fit_at)
            except Exception:
                pass

    def _save_artifacts(self):
        """保存先验特征缓存到 feature_cache.pkl（仅 feature_cache 形态，原子写）。"""
        if self.artifacts is None:
            return
        self.artifacts.pop("dist_matrix", None)  # 不保存完整距离矩阵
        feature_cache_file, _ = _cluster_files()
        save_artifact(feature_cache_file, self.artifacts)

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

    def ground_truth_count(self) -> int:
        return len(self.ground_truth)

    def _ground_truth_hash(self) -> str:
        """ground truth（方法名 + Elo）的指纹，用于判断预测模型是否需要重训。"""
        content = ",".join(
            f"{m}:{self.ground_truth[m]['elo']}" for m in sorted(self.ground_truth)
        )
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _feature_space_signature(self, features: dict, meta: dict) -> str:
        """特征空间签名（M-5）：方法集合 hash + 各特征块维度 + feature_config_hash。

        特征重提取（embedding 来源 / PCA 维数 / 提取代码版本变化）后签名改变，
        ridge 缓存的 w 不再复用——原缓存只看 GT 指纹，换特征空间后 w 与 X 静默错位。
        """
        method_hash = _compute_method_set_hash(sorted(features))
        sample = next(iter(features.values()), {})
        dims = ",".join(
            f"{b}:{np.atleast_1d(np.asarray(sample.get(b, []))).shape[0]}"
            for b in EloPredictorModel.BLOCK_ORDER
        )
        fch = meta.get("feature_config_hash", "")
        return hashlib.md5(f"{method_hash}|{dims}|{fch}".encode()).hexdigest()

    # ============================================================
    # 前置特征缓存 / D-optimality 种子 / 最终聚类（post-test）
    # ============================================================
    def fit_features(self, attack_records: list[dict]) -> dict | None:
        """
        前置特征缓存（非聚类）：提取先验特征写入 artifacts，
        供 SVD-Ridge 预测与 D-optimality 种子使用。
        聚类只在整个测试流程结束后进行（final_fit）。

        返回: {"method_count": int}；未触发时返回 None。
        """
        if len(attack_records) < 2:
            logger.warning("攻击记录不足，跳过特征缓存")
            return None

        logger.info("🧩 前置特征缓存: 总方法记录 %d 条", len(attack_records))
        features, meta = extract_all_features(attack_records, eval_results=[])
        # M-6：特征配置指纹写入 meta，供 runner 缓存失效判断与 ridge 特征签名使用
        meta["feature_config_hash"] = current_feature_config_hash()

        self.artifacts = {
            "schema_version": 1,
            "kind": "feature_cache",
            "features": features,
            "meta": meta,
            "method_set_hash": _compute_method_set_hash(sorted(features.keys())),
            "generated_at": datetime.now().isoformat(),
        }
        self.last_fit_at = self.artifacts["generated_at"]
        self._save_artifacts()

        logger.info("✅ 特征缓存完成: %d 种方法", len(features))
        return {"method_count": len(features)}

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
                random.Random(_global_seed()).shuffle(candidates)
                return candidates[:n]

        def _vec(m: str) -> dict:
            base = features.get(m) or extra.get(m) or {}
            feat = dict(base)
            feat["prior"] = build_prior_features(m, method_records.get(m))
            return feat

        cand_features = {m: _vec(m) for m in candidates}
        X, dims = EloPredictorModel.features_to_matrix(cand_features, candidates)
        mean, std = X.mean(axis=0), X.std(axis=0) + 1e-8
        X_scaled = (X - mean) / std

        gt_methods = [m for m in sorted(self.ground_truth) if m in features]
        X_gt = None
        if gt_methods:
            gt_feats = {m: _vec(m) for m in gt_methods}
            Xg, _ = EloPredictorModel.features_to_matrix(
                gt_feats, gt_methods, block_dims=dims
            )
            X_gt = (Xg - mean) / std

        lam = self.model.lambda_opt or 1.0
        idx = greedy_d_optimal(X_scaled, n, lam=lam, X_gt=X_gt)
        return [candidates[i] for i in idx]

    def predict(
        self,
        method: str,
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
        cid = parse_cluster_id(labels.get(method, -1))
        gt_count = self.ground_truth_count()

        # H-10 修复：所有分支统一经 _make_result 构造，保证 schema 一致（含 std/ci95）。
        # 原回退分支缺 std/ci95，下游 predict_batch 复用时访问会 KeyError。

        if method in self.ground_truth:
            # GT 真实 Elo，不确定性为 0
            return self._make_result(
                self.ground_truth[method]["elo"], "ground_truth", None, 1.0, gt_count, std=0.0
            )

        if not self.ground_truth:
            return self._make_result(
                float(INITIAL_ELO), "predicted", None, 0.0, 0,
                std=self._fallback_std(),
            )

        # ---- 1. 同后缀变体兜底（如 *_rot13 / *_b64 / *_code / *_story） ----
        suffix_gt = self._find_suffix_variant_ground_truth(method)
        if suffix_gt:
            avg = sum(self.ground_truth[m]["elo"] for m in suffix_gt) / len(suffix_gt)
            conf = round(min(len(suffix_gt) / 3, 1.0), 4)
            return self._make_result(
                round(avg, 2), "predicted_suffix_variant", cid, conf, gt_count,
                std=self._fallback_std(),
            )

        # ---- 2. 同基底变体兜底 ----
        variant_gt = self._find_variant_ground_truth(method)
        if variant_gt:
            avg = sum(self.ground_truth[m]["elo"] for m in variant_gt) / len(variant_gt)
            conf = round(min(len(variant_gt) / 2, 1.0), 4)
            return self._make_result(
                round(avg, 2), "predicted_variant", cid, conf, gt_count,
                std=self._fallback_std(),
            )

        # ---- 3. 全局简单平均 ----
        avg = sum(g["elo"] for g in self.ground_truth.values()) / len(self.ground_truth)
        conf = round(min(gt_count / 10, 0.5), 4)
        return self._make_result(
            round(avg, 2), "predicted_global", cid, conf, gt_count,
            std=self._fallback_std(),
        )

    @staticmethod
    def _make_result(elo, source, cluster_id, confidence, based_on_gt_count, std=None):
        """构造预测结果 dict，保证 schema 一致（含 std/ci95，防下游 KeyError）。"""
        if std is None:
            std = _FALLBACK_STD
        # 兜底分支的 std 是启发值（非模型方差），ci95 用正态 1.96 而非 t 分位数——
        # t 的自由度无从谈起；模型路径（_predict_batch_svd_ridge）才用 t 分位数
        ci95 = [round(elo - 1.96 * std, 2), round(elo + 1.96 * std, 2)]
        return {
            "elo": elo,
            "source": source,
            "cluster_id": cluster_id,
            "std": round(std, 2),
            "ci95": ci95,
            "confidence": confidence,
            "based_on_gt_count": based_on_gt_count,
        }

    @staticmethod
    def _fallback_std() -> float:
        """回退分支的保守 std（无模型方差可用时的高不确定性标记）。"""
        return _FALLBACK_STD

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

        features = self.artifacts.get("features", {}) if self.artifacts else {}
        gt_methods = sorted(self.ground_truth.keys())
        # 过滤掉不在当前特征集中的 GT 方法（跨攻击集 resume 时的 stale GT 污染）：
        # 旧攻击集的方法残留在 ground_truth 里但不在当前特征集，若不过滤会让
        # use_model 永远 False，导致 SVD-Ridge 模型永远不训练（50/58 GT 可用时也被
        # 8 个 stale 方法一票否决）。只拿有特征的 GT 训练，数量够就正常训练。
        gt_in_features = [m for m in gt_methods if m in features]
        use_model = (
            len(gt_in_features) >= self.min_cluster_size
            and bool(features)
        )

        results: dict[str, dict] = {}
        if use_model:
            try:
                results = self._predict_batch_svd_ridge(methods, method_records, features)
            except Exception as e:
                logger.warning("SVD-Ridge 批量预测失败，回退到变体平均: %s", e)

        if not results:
            results = {m: self.predict(m) for m in methods}

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
        # 只用有特征的 GT 方法训练（防 stale GT 污染：跨攻击集 resume 时旧方法
        # 残留在 ground_truth 但不在当前特征集）
        gt_methods = sorted(m for m in self.ground_truth.keys() if m in features)

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
        # 仅用有特征的 GT 方法训练（与上方 gt_methods 过滤一致，防 stale GT 全零特征混入）
        train_gt = {m: self.ground_truth[m] for m in gt_methods}
        test_features = {m: _with_prior(m) for m in methods}

        feature_name_blocks = {
            "textual": meta.get("textual_feature_names", TEXTUAL_FEATURE_NAMES),
            "technique": meta.get("technique_label_names", []),
            "intent": meta.get("intent_feature_names", INTENT_FEATURE_NAMES),
            "prior": PRIOR_FEATURE_NAMES,
        }

        # 模型缓存：GT 未变且特征空间签名未变 → 直接复用 w（纯矩阵预测）；
        # GT 小幅增长 → 用现有 λ* 单次 SVD 快速 refit；
        # GT 增长 ≥ threshold 或特征空间变化（M-5）→ 重跑 K-Fold 选 λ
        gt_hash = self._ground_truth_hash()
        gt_count = self.ground_truth_count()
        feat_sig = self._feature_space_signature(features, meta)
        feature_space_changed = (
            self._model_feature_sig is not None and feat_sig != self._model_feature_sig
        )
        if feature_space_changed:
            logger.info("特征空间已变化（重提取/配置变更），SVD-Ridge 不复用旧 w，重新训练")
        if (
            self.model.w is not None
            and not feature_space_changed
            and gt_hash == self._model_gt_hash
        ):
            logger.info("SVD-Ridge 复用缓存模型 (ground truth %d 未变)", gt_count)
        elif (
            self.model.w is not None
            and not feature_space_changed
            and self.model.lambda_opt is not None
            and 0 < gt_count - self._model_cv_gt_count < self.ridge_refit_threshold
        ):
            self.model.fit(
                train_features, train_gt, feature_name_blocks,
                lambda_override=self.model.lambda_opt,
            )
            logger.info("SVD-Ridge 快速 refit (λ*=%.4f 复用, ground truth %d)",
                        self.model.lambda_opt, gt_count)
        else:
            self.model.fit(train_features, train_gt, feature_name_blocks)
            self._model_cv_gt_count = gt_count
        self._model_gt_hash = gt_hash
        self._model_feature_sig = feat_sig

        means, variances = self.model.predict(test_features, methods)

        results = {}
        # std 封顶：CI 宽于 ±几百 Elo 已无信息量；保护 summary/state/看板/前端所有下游
        y_std = self.model.y_std if self.model.y_std is not None else 0.0
        std_cap = max(RIDGE_PRED_STD_CAP_MULT * y_std, RIDGE_PRED_STD_CAP_MIN)
        # M2 修复：小样本下 CI 用 t 分位数替代 1.96——n=5 时 t₀.₉₇₅(3)≈3.18 比 1.96 宽 62%
        n_gt = len(gt_methods)
        # M-39：用 is None 显式判空——effective_df=0.0 是合法自由度，原 `or 0.0`
        # 语义上虽结果相同，但 is None 更清晰表达"缺失才兜底"的意图
        df_eff = self.model.effective_df if self.model.effective_df is not None else 0.0
        t_q = float(_t_dist.ppf(0.975, max(1, n_gt - int(round(df_eff)))))
        for m, mean, var in zip(methods, means, variances):
            std = min(float(np.sqrt(max(float(var), 0.0))), std_cap)
            elo = round(float(mean), 2)
            results[m] = {
                "elo": elo,
                "source": "svd_ridge",
                "cluster_id": parse_cluster_id(labels.get(m, -1)),
                "std": round(std, 2),
                "ci95": [round(elo - t_q * std, 2), round(elo + t_q * std, 2)],
                # confidence 的 std/200 与兜底 std 同源（_FALLBACK_STD，经验值）：
                # std=200 时 confidence=0.5，随 std 增大单调衰减
                "confidence": round(1.0 / (1.0 + std / _FALLBACK_STD), 4),
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
        攻击完成后最终聚类（post-test）：
        弱监督特征加权（真实 GT 反应）→ 阻尼白化 → HDBSCAN + Ward 树
        → 关键层 auto-k → 全簇命名 → ANOVA/Kruskal 簇效验证。
        后验特征仅用于画像与验证，不进入度量。

        返回: 最终聚类报告 dict。
        """
        if len(attack_records) < 2:
            logger.warning("攻击记录不足，跳过最终聚类")
            return None

        from llmsec.clustering import run_hdbscan_clustering
        from llmsec.clustering.posterior import (
            compute_method_reactions,
            learn_supervised_weights,
        )
        from llmsec.clustering.space import build_feature_matrix

        logger.info("🏁 最终聚类: 总方法记录 %d 条，评估结果 %d 条", len(attack_records), len(eval_results))
        features, meta = extract_all_features(attack_records, eval_results)
        reactions = compute_method_reactions(eval_results)

        # 弱监督特征权重（只用真实 GT 反应，防特征-预测自相关）
        methods = sorted(features.keys())
        X = build_feature_matrix(features, methods)
        y = {m: reactions[m]["mean_score"] for m in methods if m in reactions}
        weights = learn_supervised_weights(X, methods, y)

        report = run_hdbscan_clustering(
            features, meta,
            feature_weights=weights,
            reactions=reactions,
            write=True,
        )

        # M-30：方法数 <2（如 2 条记录同属 1 方法）时 hdb 提前返回 error 且不写文件。
        # 此时不能无条件 joblib.load——新环境 FileNotFoundError，旧环境会把上次运行的
        # 产物当本次结果。优雅返回 None（run_attack_phase 已处理 None 分支）。
        _, cluster_result_file = _cluster_files()
        if not report or report.get("error") or not cluster_result_file.exists():
            logger.warning("聚类未产出（方法数不足或写入失败），跳过 artifacts 加载")
            return None
        self.artifacts = joblib.load(cluster_result_file)
        self.artifacts["schema_version"] = 1
        self.artifacts["kind"] = "cluster_result"
        self.artifacts["is_final_cluster"] = True
        self.artifacts["ground_truth_count"] = self.ground_truth_count()
        self.artifacts["ground_truth_methods"] = sorted(self.ground_truth.keys())
        self.artifacts["method_set_hash"] = _compute_method_set_hash(
            sorted(self.artifacts.get("labels", {}).keys())
        )
        self.last_fit_gt_count = self.ground_truth_count()
        self.last_fit_at = datetime.now().isoformat()
        # 写回聚类结果文件（不走 _save_artifacts——那会把聚类结果错投进 feature_cache）
        save_artifact(cluster_result_file, self.artifacts)

        rv = report.get("reaction_validation", {})
        logger.info(
            "✅ 最终聚类完成: %d 簇, 噪声=%d, k*=%d, silhouette=%.4f, 簇效=%s",
            report.get("n_clusters", 0),
            report.get("n_noise", 0),
            report.get("chosen_k", 0),
            report.get("validation", {}).get("silhouette", 0.0),
            rv.get("verdict", "未验证"),
        )
        return report


    # ============================================================
    # 特征提取（预测用）
    # ============================================================
    def _build_technique_vector(self, record: dict, label_names: list[str]) -> np.ndarray:
        """为单条记录构造与 artifacts 中维度一致的技术标签向量。"""
        vec = np.zeros(len(label_names))
        prompt = record.get("prompt", "")

        # 技术标签（搜原文 + IGNORECASE，lower 后搜会让大写模式永不命中，F3 修复）
        for i, (_label, patterns) in enumerate(TECHNIQUE_LABELS.items()):
            if i >= len(label_names):
                break
            for pat in patterns:
                if re.search(pat, prompt, re.IGNORECASE):
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

        # 语义 embedding 路径：与训练侧共用同一降级链
        # （显式 API → 本地缓存 → HF 镜像），保证训练/预测来源一致
        # C-3 修复：embedding 不可用时 raise 而非返零向量——零向量喂给训练于真 embedding
        # 空间的 Ridge 模型 = 训练/预测空间错位，预测值损坏且仍标 svd_ridge 不可区分。
        # raise 后由 predict_batch 的 except 捕获，整批降级为变体/全局平均（source 可区分）。
        from llmsec.clustering.features import _get_embedding_model
        model = _get_embedding_model()
        if model is None:
            raise RuntimeError("无可用 embedding 通道，拒绝返零向量糊弄（C-3）")

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
            # M-40：归一化 label 为 int 再统计——原 set() - {-1} 在 label 为字符串 "-1"
            # 时集合差不生效，会少计噪声簇 / 多计真实簇
            from llmsec.clustering import parse_cluster_id

            raw_labels = self.artifacts.get("labels", {})
            norm_labels = [parse_cluster_id(v) for v in raw_labels.values()]
            n_clusters = len(set(norm_labels) - {-1})
            n_noise = sum(1 for v in norm_labels if v == -1)
            cluster_names = self.artifacts.get("cluster_names", {})

        return {
            "ground_truth_count": n_gt,
            "predicted_count": len(self.last_predictions),
            "last_fit_gt_count": self.last_fit_gt_count,
            "last_fit_at": self.last_fit_at,
            "next_kfold_at_gt_count": (
                self.last_fit_gt_count + self.ridge_refit_threshold if self.last_fit_gt_count else self.min_cluster_size
            ),
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "cluster_names": cluster_names,
        }


# ============================================================
# 模块级辅助函数
# ============================================================
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
        predictor = ColdStartPredictor()
        status = predictor.get_status()
        logger.info("=" * 60)
        logger.info("📊 ColdStartPredictor 状态")
        logger.info("=" * 60)
        logger.info(f"  ground truth 方法数: {status['ground_truth_count']}")
        logger.info(f"  预测缓存方法数: {status['predicted_count']}")
        logger.info(f"  上次训练 ground truth 数: {status['last_fit_gt_count']}")
        logger.info(f"  上次训练时间: {status['last_fit_at'] or '未训练'}")
        logger.info(f"  下次触发 K-Fold 需 ≥: {status['next_kfold_at_gt_count']} 个 ground truth")
        logger.info(f"  当前簇数: {status['n_clusters']}")
        logger.info(f"  噪声点数: {status['n_noise']}")
        if status["cluster_names"]:
            logger.info("  簇名称:")
            for cid, name in sorted(status["cluster_names"].items(), key=lambda x: int(x[0])):
                logger.info(f"    簇 {cid}: {name}")
        logger.info("=" * 60)
    else:
        parser.print_help()
