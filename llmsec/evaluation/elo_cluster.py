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
)
from llmsec.clustering.features import (
    DEFENSE_FEATURE_NAMES,
    INTENT_FEATURE_NAMES,
    TECHNIQUE_LABELS,
    TEXTUAL_FEATURE_NAMES,
)
from llmsec.core.config import (
    GROUND_TRUTH_ELO_FILE,
    INITIAL_ELO,
    PREDICTED_ELO_FILE,
)
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

        # ground truth 库：只记录真实评估过的方法
        self.ground_truth: dict[str, dict] = {}
        # 预测缓存：记录最近一次预测结果
        self.predicted: dict[str, dict] = {}

        self.last_fit_gt_count: int = 0
        self.last_fit_at: str | None = None
        self.artifacts: dict | None = None

        self._load()

    # ============================================================
    # 持久化
    # ============================================================
    def _load(self):
        """从磁盘加载 ground truth 与预测缓存。"""
        if GROUND_TRUTH_ELO_FILE.exists():
            try:
                with open(GROUND_TRUTH_ELO_FILE, "r", encoding="utf-8") as f:
                    self.ground_truth = json.load(f)
            except Exception as e:
                logger.warning("加载 ground_truth_elo 失败: %s", e)
                self.ground_truth = {}

        if PREDICTED_ELO_FILE.exists():
            try:
                with open(PREDICTED_ELO_FILE, "r", encoding="utf-8") as f:
                    self.predicted = json.load(f)
            except Exception as e:
                logger.warning("加载 predicted_elo 失败: %s", e)
                self.predicted = {}

        # 若 artifacts 文件存在则预加载（预测时会用到）
        if CLUSTER_ARTIFACTS_FILE.exists():
            try:
                self.artifacts = joblib.load(CLUSTER_ARTIFACTS_FILE)
                self.last_fit_gt_count = int(
                    self.artifacts.get("ground_truth_count", self.last_fit_gt_count)
                )
                self.last_fit_at = self.artifacts.get("generated_at", self.last_fit_at)
            except Exception as e:
                logger.warning("加载 cluster artifacts 失败: %s", e)
                self.artifacts = None

    def save(self):
        """保存 ground truth 与预测缓存。"""
        GROUND_TRUTH_ELO_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(GROUND_TRUTH_ELO_FILE, "w", encoding="utf-8") as f:
            json.dump(self.ground_truth, f, ensure_ascii=False, indent=2)

        PREDICTED_ELO_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(PREDICTED_ELO_FILE, "w", encoding="utf-8") as f:
            json.dump(self.predicted, f, ensure_ascii=False, indent=2)

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
    # 聚类训练
    # ============================================================
    def fit(
        self,
        attack_records: list[dict],
        eval_results: list[dict],
        force: bool = False,
    ) -> dict | None:
        """
        基于 ground truth 方法重新训练聚类模型。

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

        # 过滤出 ground truth 方法的 records / eval_results
        gt_records = [r for r in attack_records if r.get("method") in gt_methods]
        gt_eval_results = [r for r in eval_results if r.get("method") in gt_methods]

        if len(gt_records) < self.min_cluster_size:
            logger.warning("ground truth records 不足，跳过聚类")
            return None

        logger.info(
            "🔄 重新训练聚类模型: ground truth %d 种方法",
            len(gt_methods),
        )

        # 提取特征并聚类
        features, meta = extract_all_features(gt_records, gt_eval_results)
        report = run_clustering_pipeline(
            features,
            meta,
            method="hdbscan",
            min_cluster_size=min(self.min_cluster_size, len(gt_methods)),
            weights=self.weights,
            verbose=False,
        )

        # 重新加载 artifacts（包含最新 labels / dist_matrix 等）
        self.artifacts = joblib.load(CLUSTER_ARTIFACTS_FILE)
        self.artifacts["ground_truth_count"] = current_gt_count
        self.artifacts["ground_truth_methods"] = sorted(gt_methods)
        self.last_fit_gt_count = current_gt_count
        self.last_fit_at = datetime.now().isoformat()

        # 写回 artifacts，便于外部查看
        joblib.dump(self.artifacts, CLUSTER_ARTIFACTS_FILE)

        logger.info(
            "✅ 聚类完成: %d 簇, %d 噪声点, 上次训练时间 %s",
            report.get("n_clusters", 0),
            report.get("n_noise", 0),
            self.last_fit_at,
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
    # 预测
    # ============================================================
    def predict(
        self,
        method: str,
        record: dict | None = None,
    ) -> dict:
        """
        预测单个方法的初始 Elo。

        参数:
            method: 攻击方法名
            record: 该方法的攻击记录（含 prompt / category / harm_type）

        返回:
            {
                "elo": float,
                "source": "ground_truth" | "predicted",
                "cluster_id": int | None,
                "confidence": float,
                "based_on_gt_count": int,
            }
        """
        # 1. 已在 ground truth 中，直接返回真实 Elo
        if method in self.ground_truth:
            return {
                "elo": self.ground_truth[method]["elo"],
                "source": "ground_truth",
                "cluster_id": None,
                "confidence": 1.0,
                "based_on_gt_count": self.ground_truth_count(),
            }

        # 2. 无 artifacts 或 ground truth 不足，回退到 INITIAL_ELO
        if (
            self.artifacts is None
            or self.ground_truth_count() < self.min_cluster_size
        ):
            return {
                "elo": float(INITIAL_ELO),
                "source": "predicted",
                "cluster_id": None,
                "confidence": 0.0,
                "based_on_gt_count": self.ground_truth_count(),
            }

        meta = self.artifacts["meta"]
        labels = self.artifacts["labels"]  # {gt_method: cluster_id}
        gt_features = self.artifacts["features"]
        gt_methods = sorted(labels.keys())

        if not record:
            logger.warning("预测 %s 时未提供 record，回退到 INITIAL_ELO", method)
            return {
                "elo": float(INITIAL_ELO),
                "source": "predicted",
                "cluster_id": None,
                "confidence": 0.0,
                "based_on_gt_count": self.ground_truth_count(),
            }

        # 3. 构造新方法的特征
        new_features = self._extract_features_for_new_method(method, record, meta)

        # 4. 合并 features，计算复合距离
        combined_features = {m: gt_features[m] for m in gt_methods}
        combined_features[method] = new_features
        all_methods = [method] + gt_methods

        dist_matrix, _ = build_composite_distance(
            combined_features, all_methods, weights=self.weights
        )

        # 第一行是新方法到所有 gt 方法的距离
        distances_to_gt = dist_matrix[0, 1:]

        # 5. 找最近邻（跳过噪声 -1，必要时扩大 k 近邻）
        sorted_indices = np.argsort(distances_to_gt)
        chosen_cluster = None
        min_dist = None
        for rank, idx in enumerate(sorted_indices[: self.k_neighbors]):
            candidate = gt_methods[idx]
            cid = labels[candidate]
            if cid != -1:
                chosen_cluster = cid
                min_dist = float(distances_to_gt[idx])
                break

        # 6. 若全是噪声，回退到全局平均真实 Elo
        if chosen_cluster is None:
            avg_elo = sum(v["elo"] for v in self.ground_truth.values()) / len(
                self.ground_truth
            )
            return {
                "elo": round(avg_elo, 2),
                "source": "predicted",
                "cluster_id": -1,
                "confidence": 0.0,
                "based_on_gt_count": self.ground_truth_count(),
            }

        # 7. 取该簇内 ground truth 方法的平均 Elo
        cluster_members = [
            m for m, cid in labels.items() if cid == chosen_cluster
        ]
        cluster_elos = [self.ground_truth[m]["elo"] for m in cluster_members]
        predicted_elo = sum(cluster_elos) / len(cluster_elos)

        # 置信度：距离越近越可信；簇越大越可信
        cluster_size = len(cluster_members)
        distance_conf = 1.0 / (1.0 + min_dist)
        size_conf = min(cluster_size / self.min_cluster_size, 1.0)
        confidence = round(distance_conf * size_conf, 4)

        result = {
            "elo": round(predicted_elo, 2),
            "source": "predicted",
            "cluster_id": int(chosen_cluster),
            "confidence": confidence,
            "based_on_gt_count": self.ground_truth_count(),
        }

        # 8. 缓存预测结果
        self.predicted[method] = {
            **result,
            "predicted_at": datetime.now().isoformat(),
        }

        return result

    def predict_all(
        self,
        method_records: dict[str, dict],
    ) -> dict[str, dict]:
        """
        批量预测多个方法。

        参数:
            method_records: {method_name: record}

        返回:
            {method_name: predict_result}
        """
        results = {}
        for method, record in method_records.items():
            if method in self.ground_truth:
                results[method] = {
                    "elo": self.ground_truth[method]["elo"],
                    "source": "ground_truth",
                    "cluster_id": None,
                    "confidence": 1.0,
                    "based_on_gt_count": self.ground_truth_count(),
                }
            else:
                results[method] = self.predict(method, record)
        return results

    # ============================================================
    # 状态查询
    # ============================================================
    def get_status(self) -> dict:
        """返回当前预测器状态摘要。"""
        n_gt = self.ground_truth_count()
        n_predicted = len(self.predicted)
        n_clusters = 0
        n_noise = 0
        cluster_names = {}
        if self.artifacts:
            report = self.artifacts.get("cluster_profiles", {})
            n_clusters = self.artifacts.get("n_clusters", 0)
            n_noise = self.artifacts.get("n_noise", 0)
            cluster_names = self.artifacts.get("cluster_names", {})

        return {
            "ground_truth_count": n_gt,
            "predicted_count": n_predicted,
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
