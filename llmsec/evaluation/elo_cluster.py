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
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

from llmsec.clustering import (
    CLUSTER_ARTIFACTS_FILE,
    build_composite_distance,
    extract_all_features,
    extract_intent_features,
    extract_text_embeddings,
    extract_textual_features,
    run_clustering_pipeline,
    run_final_clustering,
    run_pre_clustering,
)
from llmsec.clustering.features import (
    DEFENSE_FEATURE_NAMES,
    INTENT_FEATURE_NAMES,
    TECHNIQUE_LABELS,
    TEXTUAL_FEATURE_NAMES,
)
from llmsec.core.config import INITIAL_ELO
from llmsec.core.logging import get_logger

logger = get_logger(__name__)


class ClusterEloPredictor:
    """
    用聚类实现新攻击方法的 Elo 冷启动预测。

    - 只在真实样本数跨过阈值时重训练聚类。
    - 预测新样本时，用最近一次 artifacts 中的 ground truth 模型找最近簇。
    """

    def __init__(
        self,
        threshold: int = 10,
        seed_count: int = 5,
        min_cluster_size: int = 3,
        k_neighbors: int = 3,
        weights: tuple[float, float, float, float] = (0.35, 0.25, 0.10, 0.30),
    ):
        self.threshold = threshold
        self.seed_count = seed_count
        self.min_cluster_size = min_cluster_size
        self.k_neighbors = k_neighbors
        self.weights = weights

        # ground truth 库：只记录真实评估过的方法（由 ELOTracker 统一持久化到 state.json）
        self.ground_truth: dict[str, dict] = {}

        self.last_fit_gt_count: int = 0
        self.last_fit_at: str | None = None
        self.artifacts: dict | None = None

        self._load_artifacts()

    # ============================================================
    # artifacts 持久化（ground truth 已由 ELOTracker 统一保存）
    # ============================================================
    def _load_artifacts(self):
        """从磁盘加载聚类 artifacts（不保存完整 dist_matrix，预测时按需计算局部距离）。"""
        if CLUSTER_ARTIFACTS_FILE.exists():
            try:
                self.artifacts = joblib.load(CLUSTER_ARTIFACTS_FILE)
                # 丢弃训练期使用的完整 dist_matrix，节省内存
                self.artifacts.pop("dist_matrix", None)
                self.last_fit_gt_count = int(
                    self.artifacts.get("ground_truth_count", self.last_fit_gt_count)
                )
                self.last_fit_at = self.artifacts.get("generated_at", self.last_fit_at)
                # 防御：清理 artifacts 中不在当前 ground_truth 的方法，避免缓存污染
                self._sanitize_artifacts()
            except Exception as e:
                logger.warning("加载 cluster artifacts 失败: %s", e)
                self.artifacts = None

    def _sanitize_artifacts(self):
        """确保 artifacts 中的 labels / features 只包含当前 ground_truth 中的方法。"""
        if self.artifacts is None:
            return
        gt_methods = set(self.ground_truth.keys())
        labels = self.artifacts.get("labels", {})
        cleaned_labels = {m: cid for m, cid in labels.items() if m in gt_methods}
        if len(cleaned_labels) != len(labels):
            logger.warning(
                "清理 artifacts: 移除 %d 个不在 ground_truth 中的方法",
                len(labels) - len(cleaned_labels),
            )
            self.artifacts["labels"] = cleaned_labels
        features = self.artifacts.get("features", {})
        cleaned_features = {m: f for m, f in features.items() if m in gt_methods or m == "__all_methods__"}
        if len(cleaned_features) != len(features):
            self.artifacts["features"] = cleaned_features

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

    # ============================================================
    # 动态聚类触发判断
    # ============================================================
    def _should_fit(self, current_gt_count: int) -> bool:
        """
        判断是否需要重训练。

        触发条件：
        - 首次训练（last_fit_gt_count == 0）且样本数 >= min_cluster_size
        - 新增真实样本数 >= threshold
        """
        if current_gt_count < self.min_cluster_size:
            return False
        if self.last_fit_gt_count == 0:
            return True
        return (current_gt_count - self.last_fit_gt_count) >= self.threshold

    # ============================================================
    # 预聚类 / 固定簇预测 / 最终聚类
    # ============================================================
    def pre_fit(self, attack_records: list[dict]) -> dict | None:
        """
        攻击前预聚类：只用攻击本身特征，把方法压到 5~10 个固定簇。
        不依赖 ground truth，用于种子选择与固定簇采样。

        返回: 预聚类报告 dict；未触发时返回 None。
        """
        if len(attack_records) < 2:
            logger.warning("攻击记录不足，跳过预聚类")
            return None

        logger.info("🧊 预聚类: 总方法记录 %d 条", len(attack_records))
        features, meta = extract_all_features(attack_records, eval_results=[])
        # 预聚类更依赖技术标签与意图，降低文本 embedding 权重，避免 rot13/b64 等表面变体主导
        pre_weights = (0.15, 0.45, 0.25, 0.0)
        report = run_pre_clustering(features, meta, weights=pre_weights)

        self.artifacts = joblib.load(CLUSTER_ARTIFACTS_FILE)
        self.artifacts["is_pre_cluster"] = True
        self.artifacts["method_set_hash"] = _compute_method_set_hash(
            sorted(self.artifacts.get("labels", {}).keys())
        )
        self.last_fit_at = datetime.now().isoformat()
        self._save_artifacts()

        logger.info(
            "✅ 预聚类完成: %d 簇, target_k=%d, silhouette=%.4f",
            report.get("n_clusters", 0),
            report.get("target_k", 0),
            report.get("validation", {}).get("silhouette", 0.0),
        )
        return report

    def predict(
        self,
        method: str,
        record: dict | None = None,
    ) -> dict:
        """
        基于固定簇（预聚类结果）内 ground truth 的距离加权 Elo 预测。
        攻击阶段不再改变簇归属，只更新簇内 ground truth 统计。

        预测优先级：
        1. 同攻击基底变体（如 *_rot13 / *_b64 / *_code / *_story）已有 ground truth，优先取变体平均。
        2. 簇内 ground truth 按特征距离倒数加权平均。
        3. 全局 ground truth 按特征距离倒数加权平均。
        """
        if method in self.ground_truth:
            return {
                "elo": self.ground_truth[method]["elo"],
                "source": "ground_truth",
                "cluster_id": None,
                "confidence": 1.0,
                "based_on_gt_count": self.ground_truth_count(),
            }

        if self.artifacts is None or "labels" not in self.artifacts:
            return {
                "elo": float(INITIAL_ELO),
                "source": "predicted",
                "cluster_id": None,
                "confidence": 0.0,
                "based_on_gt_count": self.ground_truth_count(),
            }

        labels = self.artifacts["labels"]

        # ---- 1. 同基底变体兜底 ----
        variant_gt = self._find_variant_ground_truth(method)
        if variant_gt:
            avg = sum(self.ground_truth[m]["elo"] for m in variant_gt) / len(variant_gt)
            return {
                "elo": round(avg, 2),
                "source": "predicted_variant",
                "cluster_id": int(labels.get(method, -1)) if method in labels else -1,
                "confidence": round(min(len(variant_gt) / 2, 1.0), 4),
                "based_on_gt_count": self.ground_truth_count(),
            }

        if method not in labels:
            logger.warning("方法 %s 不在预聚类结果中，回退到全局加权平均", method)
            return self._predict_global_weighted(method, record)

        cluster_id = labels[method]
        cluster_members = [m for m, cid in labels.items() if cid == cluster_id]
        gt_members = [m for m in cluster_members if m in self.ground_truth]

        if not gt_members:
            # 簇内无 ground truth，用全局加权平均
            return self._predict_global_weighted(method, record, cluster_id=cluster_id)

        # ---- 2. 簇内距离加权平均 ----
        predicted_elo, distance_conf = self._weighted_elo_by_distance(method, gt_members, record)
        size_conf = min(len(gt_members) / self.min_cluster_size, 1.0)
        confidence = round(distance_conf * size_conf, 4)

        return {
            "elo": round(predicted_elo, 2),
            "source": "predicted",
            "cluster_id": int(cluster_id),
            "confidence": confidence,
            "based_on_gt_count": self.ground_truth_count(),
        }

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

    def _predict_global_weighted(
        self,
        method: str,
        record: dict | None,
        cluster_id: int = -1,
    ) -> dict:
        """用全局 ground truth 按特征距离倒数加权预测。"""
        gt_members = list(self.ground_truth.keys())
        if not gt_members:
            return {
                "elo": float(INITIAL_ELO),
                "source": "predicted",
                "cluster_id": cluster_id,
                "confidence": 0.0,
                "based_on_gt_count": 0,
            }
        predicted_elo, distance_conf = self._weighted_elo_by_distance(method, gt_members, record)
        return {
            "elo": round(predicted_elo, 2),
            "source": "predicted_global",
            "cluster_id": cluster_id,
            "confidence": round(distance_conf, 4),
            "based_on_gt_count": self.ground_truth_count(),
        }

    def _weighted_elo_by_distance(
        self,
        method: str,
        gt_members: list[str],
        record: dict | None,
    ) -> tuple[float, float]:
        """
        计算 method 到 gt_members 的特征距离，并返回距离倒数加权的 Elo 和距离置信度。
        若无法计算距离，回退到简单平均。
        """
        if not gt_members:
            return float(INITIAL_ELO), 0.0

        # 尝试从 artifacts 中获取 features
        features = self.artifacts.get("features", {}) if self.artifacts else {}
        weights = self.artifacts.get("weights", self.weights) if self.artifacts else self.weights
        meta = self.artifacts.get("meta", {}) if self.artifacts else {}

        target_features = features.get(method)
        if target_features is None and record is not None:
            try:
                target_features = self._extract_features_for_new_method(method, record, meta)
            except Exception as e:
                logger.warning("为 %s 提取特征失败: %s", method, e)

        if target_features is None:
            # 无法获取目标特征，回退简单平均
            avg = sum(self.ground_truth[m]["elo"] for m in gt_members) / len(gt_members)
            return avg, 0.0

        # 构造局部 features dict：目标方法 + ground truth 方法
        local_features = {method: target_features}
        for m in gt_members:
            if m in features:
                local_features[m] = features[m]
            elif m == method:
                continue
            else:
                # 缺少某个 ground truth 的特征，跳过该项（不应发生）
                continue

        local_methods = [method] + [m for m in gt_members if m in local_features and m != method]
        if len(local_methods) < 2:
            avg = sum(self.ground_truth[m]["elo"] for m in gt_members) / len(gt_members)
            return avg, 0.0

        try:
            from llmsec.clustering import build_composite_distance
            dist_matrix, _ = build_composite_distance(local_features, local_methods, weights=weights)
            distances = dist_matrix[0, 1:]  # 目标方法到各 gt 方法的距离
            gt_in_local = local_methods[1:]
        except Exception as e:
            logger.warning("计算 %s 的距离加权失败: %s", method, e)
            avg = sum(self.ground_truth[m]["elo"] for m in gt_members) / len(gt_members)
            return avg, 0.0

        weights_list = []
        elos_list = []
        for i, gt_method in enumerate(gt_in_local):
            d = float(distances[i])
            w = 1.0 / (1.0 + d)
            weights_list.append(w)
            elos_list.append(self.ground_truth[gt_method]["elo"])

        weights_arr = np.array(weights_list)
        predicted_elo = float(np.dot(weights_arr, elos_list) / weights_arr.sum())
        nearest_dist = float(np.min(distances))
        distance_conf = round(1.0 / (1.0 + nearest_dist), 4)
        return predicted_elo, distance_conf

    def final_fit(
        self,
        attack_records: list[dict],
        eval_results: list[dict],
    ) -> dict | None:
        """
        攻击完成后最终聚类：DBSCAN + Agglomerative 两步。
        用全部真实评估数据重新构建特征空间，产出最终簇结构。

        返回: 最终聚类报告 dict。
        """
        if len(attack_records) < 2:
            logger.warning("攻击记录不足，跳过最终聚类")
            return None

        logger.info("🏁 最终聚类: 总方法记录 %d 条，评估结果 %d 条", len(attack_records), len(eval_results))
        features, meta = extract_all_features(attack_records, eval_results)
        report = run_final_clustering(features, meta, weights=self.weights)

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
            "✅ 最终聚类完成: %d 簇, DBSCAN核心簇=%d, 噪声=%d, silhouette=%.4f",
            report.get("n_clusters", 0),
            report.get("dbscan_core_clusters", 0),
            report.get("dbscan_noise", 0),
            report.get("validation", {}).get("silhouette", 0.0),
        )
        return report

    # ============================================================
    # 聚类训练（动态模式，已不在攻击阶段使用）
    # ============================================================
    def fit_dynamic(
        self,
        attack_records: list[dict],
        eval_results: list[dict],
        force: bool = False,
    ) -> dict | None:
        """
        基于 ground truth 方法重新训练聚类模型。

        关键设计：
        - 特征空间用全部 attack_records 构建，保证未测方法与已测方法在同一空间。
        - HDBSCAN 只在 ground truth 方法上训练，避免未测数据污染簇定义。
        - 对仍被标为噪声的 ground truth 方法，用 kNN 挂回最近簇，确保锚点完整。

        参数:
            attack_records: 全部攻击记录
            eval_results: 全部评估结果
            force: 是否强制重训练，忽略阈值

        返回:
            聚类报告 dict；未触发训练时返回 None。
        """
        current_gt_count = self.ground_truth_count()
        if not force and not self._should_fit(current_gt_count):
            return None

        gt_methods = set(self.ground_truth.keys())
        if len(gt_methods) < self.min_cluster_size:
            logger.info(
                "ground truth 方法数 %d < %d，跳过聚类",
                len(gt_methods),
                self.min_cluster_size,
            )
            return None

        if len(attack_records) < self.min_cluster_size:
            logger.warning("攻击记录不足，跳过聚类")
            return None

        logger.info(
            "🔄 重新训练聚类模型: ground truth %d 种方法，总方法记录 %d 条",
            len(gt_methods),
            len(attack_records),
        )

        # 1. 对全部方法构建统一特征空间（含未测方法）
        all_features, meta = extract_all_features(attack_records, eval_results)
        all_methods = sorted(all_features.keys())
        method_to_idx = {m: i for i, m in enumerate(all_methods)}

        # 2. 提取 ground truth 子集，在其上训练 HDBSCAN
        gt_records = [r for r in attack_records if r.get("method") in gt_methods]
        gt_eval_results = [r for r in eval_results if r.get("method") in gt_methods]
        if not gt_records:
            logger.warning("ground truth 方法不在当前攻击集中，跳过动态聚类")
            return None
        gt_features, gt_meta = extract_all_features(gt_records, gt_eval_results)
        gt_method_list = sorted(gt_features.keys())

        report = run_clustering_pipeline(
            gt_features,
            gt_meta,
            method="hdbscan",
            min_cluster_size=min(self.min_cluster_size, len(gt_method_list)),
            weights=self.weights,
            verbose=False,
        )

        # 3. 重新加载 artifacts 并扩展为“全部方法”版本
        self.artifacts = joblib.load(CLUSTER_ARTIFACTS_FILE)
        gt_labels = self.artifacts["labels"]  # {gt_method: cluster_id}

        # 4. 为 ground truth 中的噪声点挂回最近簇
        gt_labels = self._assign_noise_to_nearest_cluster(
            gt_labels, self.artifacts["dist_matrix"], gt_method_list
        )

        # 5. 计算诊断指标
        diagnostics = self._compute_diagnostics(
            gt_labels, self.artifacts["dist_matrix"], gt_method_list
        )
        diagnostics["k_distance_eps"] = self.artifacts.get("hdbscan_params", {}).get(
            "k_distance_eps", 0.0
        )

        # 6. 组装新的 artifacts：包含全部方法特征、gt labels、gt dist_matrix
        self.artifacts["features"] = all_features
        self.artifacts["meta"] = meta
        self.artifacts["labels"] = gt_labels
        self.artifacts["ground_truth_count"] = current_gt_count
        self.artifacts["ground_truth_methods"] = sorted(gt_methods)
        self.artifacts["all_methods"] = all_methods
        self.artifacts["method_to_idx"] = method_to_idx
        self.artifacts["diagnostics"] = diagnostics
        self.artifacts["method_set_hash"] = _compute_method_set_hash(all_methods)
        self.last_fit_gt_count = current_gt_count
        self.last_fit_at = datetime.now().isoformat()

        # 写回 artifacts（不含 dist_matrix）
        self._save_artifacts()

        logger.info(
            "✅ 聚类完成: %d 簇, %d 原始噪声点, %d 挂回噪声点, "
            "noise_ratio=%.2f%%, silhouette=%.4f, k-distance_eps=%.4f, 上次训练时间 %s",
            diagnostics["n_clusters"],
            diagnostics["n_raw_noise"],
            diagnostics["n_reassigned_noise"],
            diagnostics["noise_ratio"] * 100,
            diagnostics["silhouette"],
            diagnostics.get("k_distance_eps", 0.0),
            self.last_fit_at,
        )
        return report

    def _assign_noise_to_nearest_cluster(
        self,
        labels: dict[str, int],
        dist_matrix: np.ndarray,
        methods: list[str],
    ) -> dict[str, int]:
        """
        对 labels 中为 -1 的 ground truth 方法，找到最近非噪声邻居所属簇并挂回。
        保证所有 ground truth 方法都有有效簇归属。
        """
        method_to_idx = {m: i for i, m in enumerate(methods)}
        new_labels = dict(labels)
        n_reassigned = 0

        for method in methods:
            if new_labels[method] != -1:
                continue
            idx = method_to_idx[method]
            distances = dist_matrix[idx].copy()
            distances[idx] = np.inf

            # 按距离排序找第一个非噪声邻居
            sorted_idx = np.argsort(distances)
            for neighbor_idx in sorted_idx:
                neighbor = methods[neighbor_idx]
                neighbor_label = new_labels[neighbor]
                if neighbor_label != -1:
                    new_labels[method] = neighbor_label
                    n_reassigned += 1
                    break

        return new_labels

    def _compute_diagnostics(
        self,
        labels: dict[str, int],
        dist_matrix: np.ndarray,
        methods: list[str],
    ) -> dict:
        """计算聚类诊断指标。"""
        from sklearn.metrics import silhouette_score

        n = len(methods)
        n_noise_raw = sum(1 for v in labels.values() if v == -1)
        # 挂回后不应再出现 -1
        n_noise_final = sum(1 for v in labels.values() if v == -1)
        cluster_ids = sorted(set(labels.values()) - {-1})
        n_clusters = len(cluster_ids)

        silhouette = 0.0
        valid_idx = [i for i, m in enumerate(methods) if labels[m] != -1]
        if len(valid_idx) >= 3 and len(cluster_ids) >= 2:
            try:
                y_valid = [labels[methods[i]] for i in valid_idx]
                d_sub = dist_matrix[np.ix_(valid_idx, valid_idx)]
                silhouette = float(silhouette_score(d_sub, y_valid, metric="precomputed"))
            except Exception:
                pass

        return {
            "n_clusters": n_clusters,
            "n_raw_noise": n_noise_raw,
            "n_reassigned_noise": n_noise_raw - n_noise_final,
            "n_noise_final": n_noise_final,
            "noise_ratio": n_noise_raw / max(1, n),
            "silhouette": round(silhouette, 4),
            "cluster_size_entropy": self._cluster_entropy(labels, cluster_ids),
        }

    def _cluster_entropy(self, labels: dict[str, int], cluster_ids: list[int]) -> float:
        """簇大小分布的香农熵，值越大分布越均匀。"""
        from math import log

        sizes = [sum(1 for v in labels.values() if v == cid) for cid in cluster_ids]
        total = sum(sizes)
        if total == 0:
            return 0.0
        entropy = 0.0
        for s in sizes:
            if s > 0:
                p = s / total
                entropy -= p * log(p)
        return round(entropy, 4)

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

    def _extract_single_embedding(
        self,
        prompt: str,
        vectorizer,
        pca,
    ) -> np.ndarray:
        """
        为单个 prompt 提取与训练时同维度的 embedding。

        注意：不能复用 extract_text_embeddings 对单样本自动降维，
        因为 n=1 时 PCA 目标维度会被算成 0。这里手动 transform。
        """
        from llmsec.core.text import strip_math_tax

        cleaned = strip_math_tax(prompt)

        # TF-IDF 路径
        if vectorizer is not None:
            tfidf = vectorizer.transform([cleaned])
            dense = tfidf.toarray()
            if pca is not None:
                return pca.transform(dense)[0]
            return dense[0]

        # sentence-transformers 路径：加载模型并手动 PCA
        model = None
        try:
            from sentence_transformers import SentenceTransformer
            model_name = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            model = SentenceTransformer(model_name)
        except Exception as e:
            logger.warning("预测时加载 embedding 模型失败: %s", e)
            return np.zeros(
                pca.n_components if pca is not None else 384, dtype=np.float64
            )

        emb = model.encode([cleaned], show_progress_bar=False)
        if pca is not None:
            return pca.transform(emb)[0]
        return emb[0]

    def _extract_features_for_new_method(
        self,
        method: str,
        record: dict,
        meta: dict,
    ) -> dict[str, np.ndarray]:
        """
        为单个新方法提取与训练时同维度、同尺度的特征块。
        """
        prompt = record.get("prompt", "")
        vectorizer = meta.get("embedding_artifacts", {}).get("vectorizer")
        pca = meta.get("embedding_artifacts", {}).get("pca")

        # 1. textual
        textual_dict = extract_textual_features(prompt)
        textual_names = meta.get("textual_feature_names", TEXTUAL_FEATURE_NAMES)
        textual_vec = np.array(
            [textual_dict.get(k, 0.0) for k in textual_names], dtype=np.float64
        )

        # 2. embedding：手动 transform 以保证维度与训练时一致
        embedding_vec = self._extract_single_embedding(prompt, vectorizer, pca)
        embeddings = embedding_vec.reshape(1, -1)

        # 3. technique
        technique_names = meta.get("technique_label_names", [])
        technique_vec = self._build_technique_vector(record, technique_names)

        # 4. intent
        method_to_idx = {method: 0}
        method_prompts = {method: [prompt]}
        intent_feats = extract_intent_features(
            [method], method_prompts, embeddings, method_to_idx
        )
        intent_vec = intent_feats.get(method, np.zeros(len(INTENT_FEATURE_NAMES)))

        # 5. defense：新方法无真实评估，用零向量
        defense_names = meta.get("defense_feature_names", DEFENSE_FEATURE_NAMES)
        defense_vec = np.zeros(len(defense_names), dtype=np.float64)

        return {
            "textual": textual_vec,
            "embedding": embedding_vec,
            "technique": technique_vec,
            "intent": intent_vec,
            "defense": defense_vec,
            "cross_model": np.array([], dtype=np.float64),
        }

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


def _strip_variant_suffix(method_name: str) -> str:
    """去掉方法名末尾的变体后缀（如 _rot13/_b64/_code/_story/_0），得到攻击基底名。"""
    return _VARIANT_SUFFIX_RE.sub("", method_name)


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
