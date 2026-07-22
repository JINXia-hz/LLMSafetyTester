#!/usr/bin/env python3
"""
核心聚类模块

接收 features.py 提取的 5 维特征块，构建复合距离矩阵，
用 HDBSCAN / K-Means / 层次聚类进行攻击方法聚类。

关键特性：
- 分块距离：embedding(余弦) + technique(Jaccard) + 连续特征(欧氏/Z-score)
- HDBSCAN 自动选簇（默认），无需预设 K
- 未知类缓冲区（噪声点 cluster=-1）
- 自动命名：TF-IDF 关键词 + 技术标签 + 人工分类
- 验证指标：轮廓系数、Davies-Bouldin、NMI、ARI

输出：
    output/cluster_report.json   — 聚类报告（含命名 + 验证）
    output/cluster_matrix.csv    — 方法×特征矩阵
    output/cluster_features.json — 特征重要性分析
"""

import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime

import joblib
import numpy as np
from scipy.spatial.distance import squareform, pdist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from llmsec.core import (
    CLUSTER_ARTIFACTS_FILE,
    CLUSTER_FEATURES_FILE,
    CLUSTER_MATRIX_FILE,
    CLUSTER_REPORT_FILE,
    OUTPUT_DIR,
    get_logger,
)

logger = get_logger(__name__)


# ============================================================
# 1. 距离计算
# ============================================================
def cosine_distance_matrix(vectors: np.ndarray) -> np.ndarray:
    """计算余弦距离矩阵 (1 - cosine_similarity)。"""
    sim = cosine_similarity(vectors)
    # 避免浮点误差导致的微小负值
    dist = 1.0 - sim
    dist = np.abs(dist)
    np.fill_diagonal(dist, 0.0)
    return dist


def jaccard_distance_matrix(vectors: np.ndarray) -> np.ndarray:
    """
    计算 Jaccard 距离矩阵（适用于稀疏二值特征）。
    d_jaccard(A, B) = 1 - |A ∩ B| / |A ∪ B|
    """
    n = vectors.shape[0]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            intersection = np.sum(vectors[i] * vectors[j])
            union = np.sum(np.clip(vectors[i] + vectors[j], 0, 1))
            jaccard = intersection / union if union > 0 else 1.0
            d = 1.0 - jaccard
            dist[i, j] = d
            dist[j, i] = d
    return dist


def euclidean_distance_matrix(vectors: np.ndarray, standardize: bool = True) -> np.ndarray:
    """
    计算欧氏距离矩阵。可选 Z-score 标准化。
    """
    if standardize and vectors.shape[0] > 1:
        scaler = StandardScaler()
        vectors = scaler.fit_transform(vectors)
    return squareform(pdist(vectors, metric="euclidean"))


# ============================================================
# 2. 复合距离矩阵构建
# ============================================================
def build_composite_distance(
    features: dict,
    methods: list[str],
    weights: tuple[float, float, float, float] = (0.35, 0.25, 0.10, 0.30),
) -> tuple[np.ndarray, dict]:
    """
    构建加权复合距离矩阵。

    权重顺序: (text_embedding, technique, intent, defense)
    各块内部先计算各自距离矩阵，再加权求和。

    返回: (distance_matrix[n,n], block_info)
    """
    n = len(methods)
    w_emb, w_tech, w_int, w_def = weights

    block_distances = {}
    detail = {}

    # ---- 块 1: 文本 embedding (余弦距离) ----
    emb_vectors = np.array([features[m].get("embedding", np.zeros(50)) for m in methods])
    if emb_vectors.shape[1] > 0:
        d_emb = cosine_distance_matrix(emb_vectors)
        # 归一化到 [0, 1]
        d_max = d_emb.max()
        if d_max > 0:
            d_emb = d_emb / d_max
        block_distances["embedding"] = d_emb * w_emb
        detail["embedding"] = {"weight": w_emb, "dim": emb_vectors.shape[1], "method": "cosine"}
    else:
        block_distances["embedding"] = np.zeros((n, n))
        detail["embedding"] = {"weight": 0, "dim": 0, "method": "none"}

    # ---- 块 2: 技术标签 (Jaccard 距离) ----
    tech_vectors = np.array([features[m].get("technique", np.zeros(1)) for m in methods])
    if tech_vectors.shape[1] > 0 and np.any(tech_vectors > 0):
        d_tech = jaccard_distance_matrix(tech_vectors)
        block_distances["technique"] = d_tech * w_tech
        detail["technique"] = {"weight": w_tech, "dim": tech_vectors.shape[1], "method": "jaccard"}
    else:
        block_distances["technique"] = np.zeros((n, n))
        detail["technique"] = {"weight": 0, "dim": 0, "method": "none"}

    # ---- 块 3: 意图与对抗强度 (欧氏距离) ----
    intent_vectors = np.array([features[m].get("intent", np.zeros(3)) for m in methods])
    if intent_vectors.shape[1] > 0 and np.any(intent_vectors > 0):
        d_int = euclidean_distance_matrix(intent_vectors)
        d_max = d_int.max()
        if d_max > 0:
            d_int = d_int / d_max
        block_distances["intent"] = d_int * w_int
        detail["intent"] = {"weight": w_int, "dim": intent_vectors.shape[1], "method": "euclidean"}
    else:
        block_distances["intent"] = np.zeros((n, n))
        detail["intent"] = {"weight": 0, "dim": 0, "method": "none"}

    # ---- 块 4: 防御交互 (欧氏距离) ----
    defense_vectors = np.array([features[m].get("defense", np.zeros(14)) for m in methods])
    if defense_vectors.shape[1] > 0 and np.any(defense_vectors > 0):
        d_def = euclidean_distance_matrix(defense_vectors)
        d_max = d_def.max()
        if d_max > 0:
            d_def = d_def / d_max
        block_distances["defense"] = d_def * w_def
        detail["defense"] = {"weight": w_def, "dim": defense_vectors.shape[1], "method": "euclidean"}
    else:
        block_distances["defense"] = np.zeros((n, n))
        detail["defense"] = {"weight": 0, "dim": 0, "method": "none"}

    # 加权求和
    composite = np.zeros((n, n))
    for block_name, d_mat in block_distances.items():
        composite += d_mat

    # 确保对角线为 0 且对称
    np.fill_diagonal(composite, 0.0)
    composite = (composite + composite.T) / 2.0

    return composite, detail


# ============================================================
# 3. 聚类算法
# ============================================================
def knee_eps(dist_matrix: np.ndarray, k_candidates: list[int] | None = None) -> tuple[int, float]:
    """
    用 k-distance 图找 HDBSCAN 的推荐 min_samples 与 cluster_selection_epsilon。

    对多个 k 候选，计算每个点到第 k 近邻的距离并排序，
    使用 Kneedle 算法找 k-distance 曲线的“肩部”（距离对角线最远的点）。
    该点通常比二阶差分 knee 更大，能让 HDBSCAN 捕获更多簇。

    返回: (recommended_min_samples, recommended_eps)
    """
    if k_candidates is None:
        k_candidates = [2, 3, 4]

    n = dist_matrix.shape[0]
    if n <= max(k_candidates) + 1:
        return 2, 0.2

    best_score = -np.inf
    best_k = 2
    best_eps = 0.2

    for k in k_candidates:
        if k >= n - 1:
            continue
        # 每个点到第 k 近邻的距离（跳过自己，取第 k 小）
        sorted_dists = np.sort(dist_matrix, axis=1)
        k_dists = sorted_dists[:, k]
        k_dists = np.sort(k_dists)

        if len(k_dists) < 3:
            continue

        # Kneedle：归一化后找离对角线 y=x 最远的点
        x = np.linspace(0.0, 1.0, len(k_dists))
        y_min, y_max = k_dists[0], k_dists[-1]
        if y_max - y_min < 1e-9:
            continue
        y = (k_dists - y_min) / (y_max - y_min)
        # 距离对角线 y=x 的垂直距离
        distances = np.abs(y - x)
        knee_idx = int(np.argmax(distances))
        eps = float(k_dists[knee_idx])
        score = float(distances[knee_idx])

        if score > best_score and eps > 1e-6:
            best_score = score
            best_k = k
            best_eps = eps

    return int(best_k), float(best_eps)


def _hdbscan_once(
    dist_matrix: np.ndarray,
    method_names: list[str],
    min_cluster_size: int,
    min_samples: int,
    eps: float,
    selection_method: str = "leaf",
) -> tuple[dict[str, int], int, int]:
    """执行一次 HDBSCAN，返回 (labels, n_clusters, n_noise)。"""
    import hdbscan
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="precomputed",
        cluster_selection_epsilon=eps,
        allow_single_cluster=False,
        cluster_selection_method=selection_method,
    )
    labels = clusterer.fit_predict(dist_matrix)
    n_clusters = len(set(labels) - {-1})
    n_noise = sum(1 for v in labels if v == -1)
    return {name: int(label) for name, label in zip(method_names, labels)}, n_clusters, n_noise


def run_hdbscan(
    dist_matrix: np.ndarray,
    method_names: list[str],
    min_cluster_size: int = 3,
    min_samples: int | None = None,
) -> dict[str, int]:
    """
    用 HDBSCAN 聚类（基于预计算距离矩阵）。
    自动用 k-distance 选择 min_samples 与 cluster_selection_epsilon。
    若结果不理想，依次降低 eps、降低 min_samples 重试；
    仍不足时 fallback 到层次聚类，确保簇数足够。
    返回: {method_name: cluster_id}，噪声点为 -1。
    """
    try:
        import hdbscan
        n = dist_matrix.shape[0]
        # 动态 min_cluster_size：小样本至少 3，大样本允许 2 以捕获小类
        min_cluster_size_lower = 2 if n >= 30 else 3
        effective_min_cluster_size = max(min_cluster_size_lower, min(min_cluster_size, n // 3))
        # 自动选参
        auto_min_samples, eps = knee_eps(dist_matrix)
        effective_min_samples = min_samples if min_samples is not None else auto_min_samples
        # 小样本时限制 min_samples，避免所有点都因邻居不足被判为噪声
        effective_min_samples = min(effective_min_samples, max(2, n // 5))
        effective_min_samples = min(effective_min_samples, n - 1)
        eps = min(eps, dist_matrix.max())

        # 候选 eps：从 knee_eps 开始，逐步下降到最小非零距离附近
        sorted_distances = np.sort(dist_matrix[np.triu_indices_from(dist_matrix, k=1)])
        min_positive_dist = float(sorted_distances[sorted_distances > 0].min()) if np.any(sorted_distances > 0) else 1e-6
        eps_candidates = [eps]
        for factor in [0.75, 0.5, 0.33, 0.25]:
            candidate = max(eps * factor, min_positive_dist * 1.05)
            if candidate < eps_candidates[-1]:
                eps_candidates.append(candidate)

        # 候选 min_samples：从自动值逐步降到 2
        min_samples_candidates = [effective_min_samples]
        for ms in [effective_min_samples - 1, 2]:
            if ms >= 2 and ms not in min_samples_candidates:
                min_samples_candidates.append(ms)

        # 目标簇数：HDBSCAN 必须达到该簇数且噪声不过半才被接受，否则 fallback 到层次聚类
        target_clusters = max(5, int(n ** 0.5))

        last_labels = None
        best_labels = None
        best_n_clusters = 0
        for try_ms in min_samples_candidates:
            for try_eps in eps_candidates:
                logger.info(
                    "HDBSCAN 参数: min_cluster_size=%d, min_samples=%d, cluster_selection_epsilon=%.4f",
                    effective_min_cluster_size,
                    try_ms,
                    try_eps,
                )
                labels, n_clusters, n_noise = _hdbscan_once(
                    dist_matrix, method_names, effective_min_cluster_size, try_ms, try_eps
                )
                last_labels = labels
                # 记录簇数最多的结果，用于后续 fallback 比较
                if n_clusters > best_n_clusters:
                    best_labels = labels
                    best_n_clusters = n_clusters
                # 要求达到目标簇数且噪声不过半才接受；否则继续尝试或 fallback
                if n_clusters >= target_clusters and n_noise < n / 2:
                    return labels
                logger.info(
                    "HDBSCAN eps=%.4f, min_samples=%d 结果不理想 (簇=%d, 噪声=%d)",
                    try_eps, try_ms, n_clusters, n_noise,
                )

        # 若 HDBSCAN 无论如何都太少簇，fallback 到层次聚类
        if best_n_clusters < target_clusters:
            logger.warning(
                "HDBSCAN 最多只分出 %d 簇，fallback 到层次聚类 (target=%d)",
                best_n_clusters, target_clusters,
            )
            return run_hierarchical(dist_matrix, method_names, n_clusters=target_clusters)

        # 全部候选都不行，用最后一个结果
        return last_labels if last_labels is not None else {name: -1 for name in method_names}
    except Exception as e:
        logger.warning("HDBSCAN 失败: %s，回退到 K-Means (K=3)", e)
        return run_kmeans(dist_matrix, method_names, k=3)


def run_kmeans(
    dist_matrix: np.ndarray,
    method_names: list[str],
    k: int = 3,
    random_seed: int = 42,
) -> dict[str, int]:
    """
    用 K-Means 聚类（基于预计算距离矩阵，使用 sklearn）。
    """
    from sklearn.cluster import KMeans
    from sklearn.manifold import MDS

    # 将距离矩阵嵌入到低维空间（MDS）
    n = dist_matrix.shape[0]
    if n <= k:
        return {name: i for i, name in enumerate(method_names)}

    mds = MDS(n_components=min(10, n - 1), dissimilarity="precomputed",
              random_state=random_seed, max_iter=300)
    coords = mds.fit_transform(dist_matrix)

    kmeans = KMeans(n_clusters=k, random_state=random_seed, n_init="auto")
    labels = kmeans.fit_predict(coords)
    return {name: int(label) for name, label in zip(method_names, labels)}


def run_hierarchical(
    dist_matrix: np.ndarray,
    method_names: list[str],
    n_clusters: int = 5,
) -> dict[str, int]:
    """层次聚类（Ward 链接）。"""
    from sklearn.cluster import AgglomerativeClustering
    n = dist_matrix.shape[0]
    k = min(n_clusters, n)
    clusterer = AgglomerativeClustering(
        n_clusters=k, metric="precomputed", linkage="average"
    )
    labels = clusterer.fit_predict(dist_matrix)
    return {name: int(label) for name, label in zip(method_names, labels)}


# ============================================================
# 4. 簇自动命名
# ============================================================
def _is_garbage_token(token: str) -> bool:
    """过滤 base64/rot13 等编码残留、乱码或无意义长串。"""
    if len(token) > 20:
        return True
    # 无元音且长度 >=4 的大概率是缩写/编码残留（保留短停用词）
    if len(token) >= 4 and not re.search(r"[aeiouAEIOU]", token):
        return True
    digits = sum(c.isdigit() for c in token)
    if len(token) >= 5 and digits / len(token) > 0.25:
        return True
    # 纯大小写+数字且长度超过 15 的大概率是编码块
    if len(token) > 15 and re.fullmatch(r"[A-Za-z0-9+/=]+", token):
        return True
    return False


def _extract_tfidf_keywords(
    method_prompts: dict[str, str],
    cluster_members: list[str],
    top_n: int = 5,
) -> list[tuple[str, float]]:
    """对簇内方法提取 TF-IDF 关键词，并剔除编码残留。"""
    texts = [method_prompts[m] for m in cluster_members if m in method_prompts]
    if len(texts) < 2:
        return []

    try:
        vectorizer = TfidfVectorizer(
            max_features=100, stop_words="english",
            ngram_range=(1, 2), max_df=0.8, min_df=1,
        )
        tfidf = vectorizer.fit_transform(texts)
    except Exception:
        return []

    # 取簇内 TF-IDF 均值最高的词，过滤编码残留
    mean_tfidf = tfidf.mean(axis=0).A1
    indices = np.argsort(mean_tfidf)[::-1]
    feature_names = vectorizer.get_feature_names_out()
    keywords = []
    for i in indices:
        kw = feature_names[i]
        if mean_tfidf[i] <= 0.01:
            continue
        if _is_garbage_token(kw):
            continue
        keywords.append((kw, round(float(mean_tfidf[i]), 4)))
        if len(keywords) >= top_n:
            break
    return keywords


def auto_name_clusters(
    labels: dict[str, int],
    features: dict,
    meta: dict,
    method_prompts: dict[str, str],
) -> dict[int, str]:
    """
    为每个簇自动生成名称。
    命名来源：
    1. 簇内最多的 technical label（如 "编码混淆"）
    2. 簇内最多的 harm_type（如 "fraud"）
    3. TF-IDF 关键词 top-2
    """
    methods = meta.get("method_names", list(labels.keys()))
    technique_label_names = meta.get("technique_label_names", [])

    # 按簇分组
    clusters = defaultdict(list)
    for m, cid in labels.items():
        clusters[cid].append(m)

    cluster_names = {}
    for cid, members in clusters.items():
        parts = []

        # 1. 最多技术标签
        if technique_label_names:
            tech_counts = Counter()
            for m in members:
                if m in features and "technique" in features[m]:
                    vec = features[m]["technique"]
                    for i, v in enumerate(vec):
                        if v > 0.5 and i < len(technique_label_names):
                            tech_counts[technique_label_names[i]] += 1
            top_tech = [t for t, _ in tech_counts.most_common(2)]
            if top_tech:
                parts.append("+".join(top_tech[:2]))

        # 2. 最多 harm_type
        if technique_label_names:
            harm_labels = [t for t in technique_label_names if t.startswith("harm:")]
            if harm_labels:
                harm_counts = Counter()
                harm_start_indices = {i for i, t in enumerate(technique_label_names) if t.startswith("harm:")}
                for m in members:
                    if m in features and "technique" in features[m]:
                        vec = features[m]["technique"]
                        for i, v in enumerate(vec):
                            if v > 0.5 and i in harm_start_indices:
                                harm_counts[technique_label_names[i].replace("harm:", "")] += 1
                top_harm = [h for h, _ in harm_counts.most_common(1)]
                if top_harm:
                    parts.append(f"→{top_harm[0]}")

        # 3. TF-IDF 关键词
        keywords = _extract_tfidf_keywords(method_prompts, members, top_n=3)
        if keywords:
            kw_str = "/".join(kw for kw, _ in keywords[:2])
            parts.append(f"[{kw_str}]")

        name = " ".join(parts) if parts else f"簇{cid}"
        cluster_names[cid] = name

    return cluster_names


# ============================================================
# 5. 外部验证 (NMI, ARI vs 人工分类)
# ============================================================
def compute_external_validation(
    labels: dict[str, int],
    meta: dict,
    features: dict,
) -> dict:
    """
    以攻击集的 category 为金标准，计算 NMI 和 ARI。
    """
    methods = meta.get("method_names", [])
    technique_label_names = meta.get("technique_label_names", [])

    # 从 technique 标签中提取每个方法的 category
    cat_labels = {}
    for m in methods:
        if m in features and "technique" in features[m]:
            vec = features[m]["technique"]
            # 找第一个 category 标签
            for i, label in enumerate(technique_label_names):
                if label.startswith("cat:") and i < len(vec) and vec[i] > 0.5:
                    cat_labels[m] = label.replace("cat:", "")
                    break
        if m not in cat_labels:
            cat_labels[m] = "unknown"

    # 过滤掉噪声点
    valid_methods = [m for m in methods if labels.get(m, -1) >= 0]
    if len(valid_methods) < 2:
        return {"nmi": 0, "ari": 0, "note": "样本不足"}

    y_pred = [labels[m] for m in valid_methods]
    y_true = [cat_labels.get(m, "unknown") for m in valid_methods]

    # 字符串标签转整数
    unique_true = sorted(set(y_true))
    true_to_int = {v: i for i, v in enumerate(unique_true)}
    y_true_int = [true_to_int[v] for v in y_true]

    nmi = normalized_mutual_info_score(y_true_int, y_pred)
    ari = adjusted_rand_score(y_true_int, y_pred)

    return {"nmi": round(float(nmi), 4), "ari": round(float(ari), 4)}


# ============================================================
# 6. 主流程
# ============================================================
def run_clustering_pipeline(
    features: dict,
    meta: dict,
    method: str = "hdbscan",
    k: int = None,
    min_cluster_size: int = 3,
    weights: tuple = (0.35, 0.25, 0.10, 0.30),
    verbose: bool = True,
) -> dict:
    """
    聚类主流程。

    参数:
        features: features.extract_all_features 的输出
        meta: features.extract_all_features 的元信息
        method: "hdbscan" | "kmeans" | "hierarchical"
        k: K-Means/层次聚类的簇数
        min_cluster_size: HDBSCAN 最小簇大小
        weights: (emb, tech, intent, defense) 复合距离权重
        verbose: 是否打印进度

    返回: 聚类报告 dict
    """
    methods = meta["method_names"]
    n = len(methods)
    if n < 2:
        return {"error": "方法数不足，至少需要 2 种方法进行聚类"}

    if verbose:
        print(f"\n{'='*60}")
        print(f"🎯 聚类分析 (方法数={n})")
        print(f"{'='*60}")

    # ---- Step 1: 构建复合距离矩阵 ----
    if verbose:
        print("📏 构建复合距离矩阵 ...")
    dist_matrix, block_info = build_composite_distance(features, methods, weights=weights)
    if verbose:
        for block_name, info in block_info.items():
            w = info["weight"]
            if w > 0:
                print(f"  {block_name}: weight={w} dim={info['dim']} method={info['method']}")

    # ---- Step 2: 聚类 ----
    if verbose:
        print(f"🔬 {method.upper()} 聚类 ...")

    hdbscan_params = {}
    if method == "hdbscan":
        labels = run_hdbscan(dist_matrix, methods, min_cluster_size=min_cluster_size)
        # 记录实际使用的 k-distance 参数
        try:
            _, eps = knee_eps(dist_matrix)
            hdbscan_params["k_distance_eps"] = round(float(eps), 4)
        except Exception:
            hdbscan_params["k_distance_eps"] = 0.0
    elif method == "kmeans":
        k_val = k or min(6, max(2, n // 3))
        labels = run_kmeans(dist_matrix, methods, k=k_val)
    elif method == "hierarchical":
        k_val = k or min(6, max(2, n // 3))
        labels = run_hierarchical(dist_matrix, methods, n_clusters=k_val)
    else:
        raise ValueError(f"未知聚类方法: {method}")

    # 统计
    cluster_ids = sorted(set(labels.values()))
    n_clusters = len([c for c in cluster_ids if c >= 0])
    n_noise = sum(1 for v in labels.values() if v == -1)
    if verbose:
        print(f"  簇数: {n_clusters} (+ {n_noise} 噪声点)")
        for cid in cluster_ids:
            members = [m for m, c in labels.items() if c == cid]
            tag = "🟡 噪声" if cid == -1 else f"簇{cid}"
            print(f"    {tag}: {len(members)} 种方法 - {', '.join(members[:5])}{'...' if len(members) > 5 else ''}")

    # ---- Step 3: 聚类验证 ----
    validation = {}
    if verbose:
        print("📊 聚类验证 ...")

    # 内部指标
    valid_idx = [i for i, m in enumerate(methods) if labels.get(m, -1) >= 0]
    if len(valid_idx) >= 3 and len(set(labels[m] for m in methods if labels.get(m, -1) >= 0)) >= 2:
        valid_methods = [methods[i] for i in valid_idx]
        y_valid = [labels[m] for m in valid_methods]
        d_sub = dist_matrix[np.ix_(valid_idx, valid_idx)]

        try:
            sil = silhouette_score(d_sub, y_valid, metric="precomputed")
            validation["silhouette"] = round(float(sil), 4)
        except Exception:
            validation["silhouette"] = 0.0

        try:
            db = davies_bouldin_score(d_sub, y_valid)
            validation["davies_bouldin"] = round(float(db), 4)
        except Exception:
            validation["davies_bouldin"] = 0.0
    else:
        validation["silhouette"] = 0.0
        validation["davies_bouldin"] = 0.0

    # 外部指标
    ext_val = compute_external_validation(labels, meta, features)
    validation.update(ext_val)

    if verbose:
        print(f"  轮廓系数: {validation['silhouette']:.4f}")
        print(f"  Davies-Bouldin: {validation['davies_bouldin']:.4f}")
        print(f"  NMI (vs 人工分类): {validation.get('nmi', 0):.4f}")
        print(f"  ARI (vs 人工分类): {validation.get('ari', 0):.4f}")

    # ---- Step 4: 自动命名 ----
    if verbose:
        print("🏷️ 自动命名 ...")
    # 构建 prompt 映射：meta["method_prompts"] 由 features.extract_all_features 提供
    # （每个方法的代表 prompt），供 TF-IDF 关键词命名使用
    method_prompts_dict = meta.get("method_prompts", {})
    cluster_names = auto_name_clusters(labels, features, meta, method_prompts_dict)
    if verbose:
        for cid, name in sorted(cluster_names.items()):
            tag = "🟡 噪声" if cid == -1 else f"簇{cid}"
            print(f"    {tag}: {name}")

    # ---- Step 5: 簇画像 ----
    cluster_profiles = build_cluster_profiles(labels, features, meta, cluster_names)

    # ---- 组装报告 ----
    report = {
        "generated_at": datetime.now().isoformat(),
        "method_count": n,
        "clustering_method": method,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "hdbscan_params": hdbscan_params,
        "weights": {
            "embedding": weights[0], "technique": weights[1],
            "intent": weights[2], "defense": weights[3],
        },
        "validation": validation,
        "block_info": {k: {kk: vv for kk, vv in v.items() if kk != "method"} for k, v in block_info.items()},
        "cluster_names": cluster_names,
        "cluster_profiles": cluster_profiles,
        "method_labels": {m: labels[m] for m in sorted(labels.keys())},
    }

    # ---- 导出 ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CLUSTER_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 导出特征矩阵 CSV
    _export_matrix(labels, features, meta)

    # ---- 保存聚类 artifacts（用于 ELO 冷启动预测） ----
    artifacts = {
        "features": features,
        "meta": meta,
        "labels": labels,
        "weights": weights,
        "block_info": block_info,
        "dist_matrix": dist_matrix,
        "cluster_names": cluster_names,
        "cluster_profiles": cluster_profiles,
        "hdbscan_params": hdbscan_params,
        "generated_at": report["generated_at"],
    }
    os.makedirs(os.path.dirname(CLUSTER_ARTIFACTS_FILE) or ".", exist_ok=True)
    joblib.dump(artifacts, CLUSTER_ARTIFACTS_FILE)

    if verbose:
        print(f"\n  📁 聚类报告: {CLUSTER_REPORT_FILE}")
        print(f"  📁 特征矩阵: {CLUSTER_MATRIX_FILE}")
        print(f"  📁 聚类 artifacts: {CLUSTER_ARTIFACTS_FILE}")

    return report


def build_cluster_profiles(
    labels: dict[str, int],
    features: dict,
    meta: dict,
    cluster_names: dict[int, str],
) -> dict[str, dict]:
    """为每个簇构建统计画像。"""
    methods = meta["method_names"]
    clusters = defaultdict(list)
    for m, cid in labels.items():
        clusters[cid].append(m)

    profiles = {}
    for cid, members in clusters.items():
        profile = {
            "size": len(members),
            "label": "noise" if cid == -1 else f"cluster_{cid}",
            "name": cluster_names.get(cid, f"簇{cid}"),
            "members": sorted(members),
        }

        # 文本统计均值
        textual_names = meta.get("textual_feature_names", [])
        for i, tn in enumerate(textual_names):
            vals = []
            for m in members:
                if m in features and "textual" in features[m]:
                    vec = features[m]["textual"]
                    if i < len(vec):
                        vals.append(vec[i])
            if vals:
                profile[f"textual_{tn}_mean"] = round(float(np.mean(vals)), 4)

        # 防御特征均值
        defense_names = meta.get("defense_feature_names", [])
        for i, dn in enumerate(defense_names):
            vals = []
            for m in members:
                if m in features and "defense" in features[m]:
                    vec = features[m]["defense"]
                    if i < len(vec):
                        vals.append(vec[i])
            if vals:
                profile[f"defense_{dn}_mean"] = round(float(np.mean(vals)), 4)

        profiles[str(cid)] = profile

    return profiles


def _export_matrix(labels: dict[str, int], features: dict, meta: dict):
    """导出特征矩阵 CSV。"""
    methods = meta["method_names"]
    textual_names = meta.get("textual_feature_names", [])
    intent_names = meta.get("intent_feature_names", [])
    defense_names = meta.get("defense_feature_names", [])
    technique_names = meta.get("technique_label_names", [])

    col_names = ["method", "cluster"] + textual_names + intent_names + defense_names + technique_names
    with open(CLUSTER_MATRIX_FILE, "w", encoding="utf-8") as f:
        f.write(",".join(f'"{c}"' for c in col_names) + "\n")
        for method in methods:
            row = [f'"{method}"', str(labels.get(method, -1))]
            feat = features.get(method, {})

            # textual
            tvec = feat.get("textual", np.zeros(len(textual_names)))
            for i in range(len(textual_names)):
                row.append(str(round(float(tvec[i]), 6) if i < len(tvec) else 0))

            # intent
            ivec = feat.get("intent", np.zeros(len(intent_names)))
            for i in range(len(intent_names)):
                row.append(str(round(float(ivec[i]), 6) if i < len(ivec) else 0))

            # defense
            dvec = feat.get("defense", np.zeros(len(defense_names)))
            for i in range(len(defense_names)):
                row.append(str(round(float(dvec[i]), 6) if i < len(dvec) else 0))

            # technique
            tecvec = feat.get("technique", np.zeros(len(technique_names)))
            for i in range(len(technique_names)):
                row.append(str(int(tecvec[i])) if i < len(tecvec) else "0")

            f.write(",".join(row) + "\n")


# ============================================================
# 7. 预聚类（攻击前）与最终聚类（攻击后）
# ============================================================
def run_pre_clustering(
    features: dict,
    meta: dict,
    target_min_clusters: int = 5,
    target_max_clusters: int = 10,
    weights: tuple = (0.35, 0.25, 0.10, 0.30),
) -> dict:
    """
    攻击前预聚类：只用攻击本身特征，把方法压到 5~10 个簇。
    用于固定簇采样与种子选择。

    参数:
        features: extract_all_features 输出
        meta: extract_all_features 元信息
        target_min_clusters: 最小簇数
        target_max_clusters: 最大簇数
        weights: 复合距离权重（只用攻击特征，defense 权重会被置 0）

    返回: 预聚类报告 dict
    """
    methods = meta["method_names"]
    n = len(methods)
    if n < 2:
        return {"error": "方法数不足", "labels": {m: 0 for m in methods}}

    # 只用攻击特征，防御特征权重置 0
    pre_weights = (weights[0], weights[1], weights[2], 0.0)
    dist_matrix, block_info = build_composite_distance(features, methods, weights=pre_weights)

    # 目标簇数：5~10 之间，且不超过 n
    if n <= target_min_clusters:
        target_k = n
    else:
        target_k = max(target_min_clusters, min(target_max_clusters, n // 10))

    labels = run_hierarchical(dist_matrix, methods, n_clusters=target_k)

    # 统计
    cluster_ids = sorted(set(labels.values()))
    n_clusters = len([c for c in cluster_ids if c >= 0])
    n_noise = sum(1 for v in labels.values() if v == -1)

    # 验证指标
    validation = {}
    valid_idx = [i for i, m in enumerate(methods) if labels.get(m, -1) >= 0]
    if len(valid_idx) >= 3 and len(set(labels[m] for m in methods if labels.get(m, -1) >= 0)) >= 2:
        valid_methods = [methods[i] for i in valid_idx]
        y_valid = [labels[m] for m in valid_methods]
        d_sub = dist_matrix[np.ix_(valid_idx, valid_idx)]
        try:
            validation["silhouette"] = round(float(silhouette_score(d_sub, y_valid, metric="precomputed")), 4)
        except Exception:
            validation["silhouette"] = 0.0
        try:
            validation["davies_bouldin"] = round(float(davies_bouldin_score(d_sub, y_valid)), 4)
        except Exception:
            validation["davies_bouldin"] = 0.0
    else:
        validation["silhouette"] = 0.0
        validation["davies_bouldin"] = 0.0

    cluster_names = auto_name_clusters(labels, features, meta, meta.get("method_prompts", {}))
    cluster_profiles = build_cluster_profiles(labels, features, meta, cluster_names)

    report = {
        "generated_at": datetime.now().isoformat(),
        "method_count": n,
        "clustering_method": "pre_agglomerative",
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "target_k": target_k,
        "validation": validation,
        "block_info": {k: {kk: vv for kk, vv in v.items() if kk != "method"} for k, v in block_info.items()},
        "cluster_names": cluster_names,
        "cluster_profiles": cluster_profiles,
        "method_labels": {m: labels[m] for m in sorted(labels.keys())},
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CLUSTER_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    artifacts = {
        "features": features,
        "meta": meta,
        "labels": labels,
        "weights": pre_weights,
        "block_info": block_info,
        "dist_matrix": dist_matrix,
        "cluster_names": cluster_names,
        "cluster_profiles": cluster_profiles,
        "pre_cluster_report": report,
        "generated_at": report["generated_at"],
    }
    joblib.dump(artifacts, CLUSTER_ARTIFACTS_FILE)

    return report


def run_final_clustering(
    features: dict,
    meta: dict,
    weights: tuple = (0.35, 0.25, 0.10, 0.30),
) -> dict:
    """
    攻击后最终聚类：DBSCAN + Agglomerative 两步。

    第一步 DBSCAN 用全部特征找密度核心簇；
    第二步 Agglomerative 把噪声点并入最近核心簇，或当核心簇不足时补足到 target_k。

    参数:
        features: extract_all_features 输出
        meta: extract_all_features 元信息
        weights: 复合距离权重

    返回: 最终聚类报告 dict
    """
    methods = meta["method_names"]
    n = len(methods)
    if n < 2:
        return {"error": "方法数不足", "labels": {m: 0 for m in methods}}

    dist_matrix, block_info = build_composite_distance(features, methods, weights=weights)

    # ---- Step 1: DBSCAN ----
    labels_dbscan = run_dbscan(dist_matrix, methods)
    core_ids = sorted(set(labels_dbscan.values()) - {-1})
    n_core = len(core_ids)
    n_noise_dbscan = sum(1 for v in labels_dbscan.values() if v == -1)

    # ---- Step 2: Agglomerative ----
    target_k = max(5, n // 10)
    if n_core >= target_k:
        # 核心簇已足够，只把噪声点并入最近核心簇
        labels = dict(labels_dbscan)
        method_to_idx = {m: i for i, m in enumerate(methods)}
        for m, cid in labels.items():
            if cid != -1:
                continue
            idx = method_to_idx[m]
            distances = dist_matrix[idx].copy()
            distances[idx] = np.inf
            sorted_idx = np.argsort(distances)
            for neighbor_idx in sorted_idx:
                neighbor = methods[neighbor_idx]
                if labels[neighbor] != -1:
                    labels[m] = labels[neighbor]
                    break
    else:
        # 核心簇不足，直接用 Agglomerative 补足到 target_k
        labels = run_hierarchical(dist_matrix, methods, n_clusters=target_k)

    # 统计
    cluster_ids = sorted(set(labels.values()))
    n_clusters = len([c for c in cluster_ids if c >= 0])
    n_noise = sum(1 for v in labels.values() if v == -1)

    # 验证指标
    validation = {}
    valid_idx = [i for i, m in enumerate(methods) if labels.get(m, -1) >= 0]
    if len(valid_idx) >= 3 and len(set(labels[m] for m in methods if labels.get(m, -1) >= 0)) >= 2:
        valid_methods = [methods[i] for i in valid_idx]
        y_valid = [labels[m] for m in valid_methods]
        d_sub = dist_matrix[np.ix_(valid_idx, valid_idx)]
        try:
            validation["silhouette"] = round(float(silhouette_score(d_sub, y_valid, metric="precomputed")), 4)
        except Exception:
            validation["silhouette"] = 0.0
        try:
            validation["davies_bouldin"] = round(float(davies_bouldin_score(d_sub, y_valid)), 4)
        except Exception:
            validation["davies_bouldin"] = 0.0
    else:
        validation["silhouette"] = 0.0
        validation["davies_bouldin"] = 0.0

    cluster_names = auto_name_clusters(labels, features, meta, meta.get("method_prompts", {}))
    cluster_profiles = build_cluster_profiles(labels, features, meta, cluster_names)

    report = {
        "generated_at": datetime.now().isoformat(),
        "method_count": n,
        "clustering_method": "final_dbscan_agglomerative",
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "dbscan_core_clusters": n_core,
        "dbscan_noise": n_noise_dbscan,
        "target_k": target_k,
        "validation": validation,
        "block_info": {k: {kk: vv for kk, vv in v.items() if kk != "method"} for k, v in block_info.items()},
        "cluster_names": cluster_names,
        "cluster_profiles": cluster_profiles,
        "method_labels": {m: labels[m] for m in sorted(labels.keys())},
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CLUSTER_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    _export_matrix(labels, features, meta)

    artifacts = {
        "features": features,
        "meta": meta,
        "labels": labels,
        "weights": weights,
        "block_info": block_info,
        "dist_matrix": dist_matrix,
        "cluster_names": cluster_names,
        "cluster_profiles": cluster_profiles,
        "final_cluster_report": report,
        "generated_at": report["generated_at"],
    }
    joblib.dump(artifacts, CLUSTER_ARTIFACTS_FILE)

    return report


def run_dbscan(
    dist_matrix: np.ndarray,
    method_names: list[str],
) -> dict[str, int]:
    """
    用 DBSCAN 聚类（基于预计算距离矩阵）。
    用 knee_eps 自动选 eps 和 min_samples。
    返回: {method_name: cluster_id}，噪声点为 -1。
    """
    from sklearn.cluster import DBSCAN

    n = dist_matrix.shape[0]
    if n < 2:
        return {name: 0 for name in method_names}

    auto_min_samples, eps = knee_eps(dist_matrix)
    min_samples = min(auto_min_samples, max(2, n // 5))
    min_samples = min(min_samples, n - 1)
    eps = min(eps, dist_matrix.max())

    sorted_distances = np.sort(dist_matrix[np.triu_indices_from(dist_matrix, k=1)])
    min_positive_dist = float(sorted_distances[sorted_distances > 0].min()) if np.any(sorted_distances > 0) else 1e-6

    eps_candidates = [eps]
    for factor in [0.75, 0.5, 0.33, 0.25]:
        candidate = max(eps * factor, min_positive_dist * 1.05)
        if candidate < eps_candidates[-1]:
            eps_candidates.append(candidate)

    last_labels = None
    for try_eps in eps_candidates:
        logger.info("DBSCAN 参数: min_samples=%d, eps=%.4f", min_samples, try_eps)
        clusterer = DBSCAN(eps=try_eps, min_samples=min_samples, metric="precomputed")
        labels = clusterer.fit_predict(dist_matrix)
        n_clusters = len(set(labels) - {-1})
        n_noise = sum(1 for v in labels if v == -1)
        last_labels = {name: int(label) for name, label in zip(method_names, labels)}
        if n_clusters >= 2 and n_noise < n / 2:
            return last_labels
        logger.info("DBSCAN eps=%.4f 结果不理想 (簇=%d, 噪声=%d)，尝试更小 eps", try_eps, n_clusters, n_noise)

    return last_labels if last_labels is not None else {name: -1 for name in method_names}
