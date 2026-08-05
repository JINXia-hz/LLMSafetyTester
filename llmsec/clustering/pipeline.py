#!/usr/bin/env python3
"""
聚类工具模块（命名 / 画像 / 导出）

- auto_name_clusters：簇自动命名（技术标签 + harm_type + TF-IDF 关键词，
  小簇/单点簇用代表方法名兜底，保证非默认名）
- build_cluster_profiles：簇统计画像（含后验 defense 均值——仅画像，不进入度量）
- _export_matrix：方法×特征矩阵 CSV 导出

聚类主流程在 llmsec.clustering.hdb（HDBSCAN），
度量空间在 llmsec.clustering.space（阻尼白化轻量马氏），
树层选择在 llmsec.clustering.tree（算法无关）。
"""

import re
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from llmsec.core import CLUSTER_MATRIX_FILE, get_logger
from llmsec.core.logging import get_logger




logger = get_logger(__name__)
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
    """
    导出特征矩阵 CSV。

    列序：method, cluster + 度量空间全量块（textual → embedding → technique →
    intent → prior，顺序与 space.PRIOR_BLOCKS 一致，CSV 可按列区间复现聚类度量）
    + 附加后验块（defense → cross_model，不参与度量，仅画像）。
    embedding 块维度以实际特征向量为准（emb_0 .. emb_{d-1}）。
    """
    methods = meta["method_names"]
    textual_names = meta.get("textual_feature_names", [])
    technique_names = meta.get("technique_label_names", [])
    intent_names = meta.get("intent_feature_names", [])
    prior_names = meta.get("prior_feature_names", [])
    defense_names = meta.get("defense_feature_names", [])
    cross_model_names = meta.get("cross_model_feature_names", [])

    # embedding 块无名列表，按实际维度生成 emb_i 列名
    emb_dim = 0
    for m in methods:
        vec = features.get(m, {}).get("embedding")
        if vec is not None:
            emb_dim = max(emb_dim, len(np.atleast_1d(vec)))
    embedding_names = [f"emb_{i}" for i in range(emb_dim)]

    # 度量空间块（PRIOR_BLOCKS 顺序）+ 附加后验块
    metric_blocks = [
        ("textual", textual_names),
        ("embedding", embedding_names),
        ("technique", technique_names),
        ("intent", intent_names),
        ("prior", prior_names),
    ]
    extra_blocks = [
        ("defense", defense_names),
        ("cross_model", cross_model_names),
    ]

    col_names = ["method", "cluster"]
    for _, names in metric_blocks + extra_blocks:
        col_names += list(names)

    with open(CLUSTER_MATRIX_FILE, "w", encoding="utf-8") as f:
        f.write(",".join(f'"{c}"' for c in col_names) + "\n")
        for method in methods:
            row = [f'"{method}"', str(labels.get(method, -1))]
            feat = features.get(method, {})
            for block, names in metric_blocks + extra_blocks:
                vec = np.atleast_1d(feat.get(block, np.zeros(len(names))))
                for i in range(len(names)):
                    v = float(vec[i]) if i < len(vec) else 0.0
                    # technique 等多标签块按整数写，其余保留 6 位小数
                    row.append(str(int(v)) if block == "technique" else str(round(v, 6)))
            f.write(",".join(row) + "\n")
