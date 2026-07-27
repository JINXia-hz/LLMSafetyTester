#!/usr/bin/env python3
"""
聚类工具模块（重构后精简版）

保留的职责：
- 欧氏距离矩阵（白化空间上的 DBSCAN 用）
- knee_eps：k-distance 图自动选 DBSCAN 参数
- run_dbscan / run_dbscan_recursive：密度聚类（大簇自动递归细分）
- auto_name_clusters：簇自动命名（技术标签 + harm_type + TF-IDF 关键词，
  小簇/单点簇用代表方法名兜底，保证非默认名）
- build_cluster_profiles：簇统计画像（含后验 defense 均值——仅画像，不进入度量）
- _export_matrix：方法×特征矩阵 CSV 导出

聚类主流程在 llmsec.clustering.tree（Ward 树 + 多指标拐点 auto-k），
度量空间构建在 llmsec.clustering.space（阻尼白化轻量马氏空间）。
"""

import re
from collections import Counter, defaultdict

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_extraction.text import TfidfVectorizer

from llmsec.core import CLUSTER_MATRIX_FILE, get_logger

logger = get_logger(__name__)


# ============================================================
# 距离计算
# ============================================================
def euclidean_distance_matrix(vectors: np.ndarray, standardize: bool = True) -> np.ndarray:
    """计算欧氏距离矩阵。可选 Z-score 标准化（白化坐标上调用时传 False）。"""
    if standardize and vectors.shape[0] > 1:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        vectors = scaler.fit_transform(vectors)
    return squareform(pdist(vectors, metric="euclidean"))


def knee_eps(dist_matrix: np.ndarray, k_candidates: list[int] | None = None) -> tuple[int, float]:
    """
    用 k-distance 图找 DBSCAN 的推荐 min_samples 与 eps。

    对多个 k 候选，计算每个点到第 k 近邻的距离并排序，
    使用 Kneedle 算法找 k-distance 曲线的"肩部"（距离对角线最远的点）。

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
        distances = np.abs(y - x)
        knee_idx = int(np.argmax(distances))
        eps = float(k_dists[knee_idx])
        score = float(distances[knee_idx])

        if score > best_score and eps > 1e-6:
            best_score = score
            best_k = k
            best_eps = eps

    return int(best_k), float(best_eps)


# ============================================================
# DBSCAN（含递归细分）
# ============================================================
def _build_eps_candidates(dist_matrix: np.ndarray, eps: float) -> list[float]:
    """以自动 eps 为中心构造候选序列（逐步缩小，避免噪声过多）。"""
    candidates = [eps]
    for factor in (0.8, 0.6, 0.45, 0.35):
        candidates.append(eps * factor)
    upper = float(np.median(dist_matrix[dist_matrix > 0])) if np.any(dist_matrix > 0) else eps
    return [min(c, upper) for c in candidates]


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

    eps_candidates = _build_eps_candidates(dist_matrix, eps)

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


def run_dbscan_recursive(
    dist_matrix: np.ndarray,
    method_names: list[str],
    max_cluster_size: int = 20,
    depth: int = 0,
    max_depth: int = 2,
) -> dict[str, int]:
    """
    递归 DBSCAN：先做一次 DBSCAN，然后对每个过大的簇用更小的 eps 再次 DBSCAN。
    解决大型聚类吞噬导致部分簇数据量不足的问题。

    返回: {method_name: cluster_id}，噪声点为 -1。
    """
    if depth >= max_depth or len(method_names) <= max_cluster_size:
        return run_dbscan(dist_matrix, method_names)

    labels = run_dbscan(dist_matrix, method_names)
    cluster_sizes = Counter(labels.values())

    # 找出需要细分的大簇（排除噪声）
    large_clusters = {cid for cid, size in cluster_sizes.items() if cid != -1 and size > max_cluster_size}
    if not large_clusters:
        return labels

    logger.info(
        "递归 DBSCAN: 发现 %d 个大簇超过 size=%d，用更小 eps 细分",
        len(large_clusters), max_cluster_size,
    )

    method_to_idx = {m: i for i, m in enumerate(method_names)}
    next_cluster_id = max(labels.values()) + 1 if labels else 0

    for cid in large_clusters:
        members = [m for m, c in labels.items() if c == cid]
        if len(members) <= max_cluster_size:
            continue

        indices = [method_to_idx[m] for m in members]
        sub_dist = dist_matrix[np.ix_(indices, indices)]

        sub_labels = run_dbscan_recursive(
            sub_dist, members,
            max_cluster_size=max_cluster_size,
            depth=depth + 1,
            max_depth=max_depth,
        )

        # 合并：子簇重新编号，噪声保持 -1
        sub_max = max(sub_labels.values()) if sub_labels else -1
        for m, sub_cid in sub_labels.items():
            if sub_cid == -1:
                labels[m] = -1
            else:
                labels[m] = next_cluster_id + sub_cid
        if sub_max >= 0:
            next_cluster_id += sub_max + 1

    return labels


# ============================================================
# 簇自动命名
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
    4. 小簇/单点簇兜底：代表方法名（保证非默认名）
    """
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

        if not parts and members:
            # 小簇/单点簇兜底：用代表方法名命名，保证非默认名
            parts.append(members[0][:30])
        name = " ".join(parts) if parts else f"簇{cid}"
        cluster_names[cid] = name

    return cluster_names


# ============================================================
# 簇画像与导出
# ============================================================
def build_cluster_profiles(
    labels: dict[str, int],
    features: dict,
    meta: dict,
    cluster_names: dict[int, str],
) -> dict[str, dict]:
    """
    为每个簇构建统计画像。
    defense（后验）均值只作画像展示，从未进入聚类度量。
    """
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

        # 防御特征均值（后验画像）
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
