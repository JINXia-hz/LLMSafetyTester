#!/usr/bin/env python3
"""
HDBSCAN 聚类主管线（post-test，整个测试流程结束后运行）。

流程：
1. 先验特征 → （可选弱监督特征权重）→ 阻尼白化空间（谱拐点截断）
2. HDBSCAN 密度聚类 → flat labels（含噪声 -1，仅作密度视图旁挂）
3. 主标签 = scipy Ward linkage 树（同一白化坐标；HDBSCAN 的
   single_linkage_tree_ 链式合并会退化，无法用于切层）
   → 树层 sweep + argmax → 关键层 k* + top3（前端缩放预设停点）
4. 全簇命名（含小簇；密度视图噪声组命名为"稀疏区"）
5. （可选）后验簇效验证 ANOVA / Kruskal-Wallis（posterior.py）

本模块不直接解析 eval 数据；反应相关输入由 posterior.py 准备。

用法:
    from llmsec.clustering.hdb import run_hdbscan_clustering
    report = run_hdbscan_clustering(features, meta, feature_weights=w, reactions=reactions)
"""

import hashlib
import json
from datetime import datetime

import joblib
import numpy as np

from llmsec.clustering.pipeline import (
    _export_matrix,
    auto_name_clusters,
    build_cluster_profiles,
)
from llmsec.clustering.space import build_whitened_space
from llmsec.clustering.tree import (
    candidate_ks,
    cut_tree,
    log_growth_k0,
    select_knee,
    sweep_candidates,
)
from llmsec.core.config import CLUSTER_REPORT_FILE, CLUSTER_RESULT_FILE, OUTPUT_DIR
from llmsec.core.logging import get_logger
from llmsec.params import HDBSCAN_MIN_CLUSTER_DIV

logger = get_logger(__name__)


def _method_set_hash(methods: list[str]) -> str:
    return hashlib.md5(",".join(sorted(set(methods))).encode("utf-8")).hexdigest()


def run_hdbscan_clustering(
    features: dict,
    meta: dict,
    feature_weights: np.ndarray | None = None,
    reactions: dict | None = None,
    write: bool = True,
) -> dict:
    """
    HDBSCAN 聚类主入口。

    参数:
        features: extract_all_features 输出 {method: 特征块}
        meta: extract_all_features 元信息
        feature_weights: posterior.learn_supervised_weights 的输出（可选）
        reactions: posterior.compute_method_reactions 的输出（可选，提供时做簇效验证）
        write: 是否写 cluster_report.json / cluster_result.pkl

    返回: 聚类报告 dict。
    """
    import hdbscan

    methods = sorted(features.keys())
    n = len(methods)
    if n < 2:
        return {"error": "方法数不足", "labels": {m: 0 for m in methods}}

    logger.info("🧬 HDBSCAN 聚类: %d 种方法", n)

    # 1. 阻尼白化空间（弱监督加权 + 谱拐点截断）
    space = build_whitened_space(features, methods, feature_weights=feature_weights)
    coords = space["coords"]
    logger.info(
        "  白化空间: %d 维 (累计解释方差 %.1f%%)%s",
        space["n_dims"], space["kept_variance"] * 100,
        " + 弱监督加权" if feature_weights is not None else "",
    )

    # 2. HDBSCAN
    # 2. HDBSCAN 密度视图（flat labels + 稀疏区）
    # min_samples=1 放宽互达距离，集中空间里显著降低噪声率；
    # min_cluster_size 随规模温和增长
    min_cluster_size = max(3, n // HDBSCAN_MIN_CLUSTER_DIV)
    clf = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric="euclidean",
    )
    flat_arr = clf.fit_predict(coords)
    flat_labels = {m: int(v) for m, v in zip(methods, flat_arr)}
    n_flat = len(set(flat_labels.values()) - {-1})
    n_noise = sum(1 for v in flat_labels.values() if v == -1)
    logger.info(
        "  HDBSCAN 密度视图: %d 簇, 噪声 %d (%.0f%%), min_cluster_size=%d",
        n_flat, n_noise, n_noise / n * 100, min_cluster_size,
    )

    # 3. Ward 缩放树 + 关键层（主 labels）
    # 注意：HDBSCAN 的 single_linkage_tree_ 在最近邻链式合并下会产出
    # 97% 单簇 + 单点的退化切割（实测 127/132 一簇），无法用于切层；
    # condensed tree 的"留在父簇的点"身份不可枚举，还原 k 层成员脆弱。
    # 因此缩放/关键层/簇效分析改用均衡的 Ward 树（同一白化坐标）。
    from scipy.cluster.hierarchy import linkage

    Z = linkage(coords, method="ward")
    ks = candidate_ks(n)
    sweep = sweep_candidates(coords, Z, methods, ks)
    k_best, top_ks = select_knee(sweep)
    labels = cut_tree(Z, methods, k_best)
    logger.info("  关键层: 候选 %s → k*=%d (top3 %s)", ks, k_best, top_ks)

    # 4. 命名（关键层各簇 + 密度视图各簇；噪声组固定命名为稀疏区）
    cluster_names = auto_name_clusters(labels, features, meta, meta.get("method_prompts", {}))
    flat_names = auto_name_clusters(flat_labels, features, meta, meta.get("method_prompts", {}))
    if -1 in flat_names:
        flat_names[-1] = "稀疏区（低密度噪声）"
    cluster_profiles = build_cluster_profiles(labels, features, meta, cluster_names)

    # 关键层 labels 的验证指标（取 sweep 中 k* 条目，另附轮廓/DB 一致性）
    best_entry = next((s for s in sweep if s["k"] == k_best), {})
    validation = {
        "silhouette": best_entry.get("silhouette", 0.0),
        "calinski_harabasz": best_entry.get("calinski_harabasz", 0.0),
        "davies_bouldin": best_entry.get("davies_bouldin", 0.0),
    }

    report = {
        "generated_at": datetime.now().isoformat(),
        "method_count": n,
        "clustering_method": "ward_autok+hdbscan",
        "n_clusters": k_best,
        # ward 主标签无噪声概念；取 HDBSCAN 密度视图的真实噪声数（同嵌套 hdbscan.n_noise）
        "n_noise": n_noise,
        "chosen_k": k_best,
        "k0_log_growth": log_growth_k0(n),
        "top_ks": top_ks,
        "candidate_sweep": sweep,
        "validation": validation,
        "cluster_names": {str(k): v for k, v in cluster_names.items()},
        "cluster_profiles": cluster_profiles,
        "method_labels": {m: labels[m] for m in sorted(labels.keys())},
        "hdbscan": {
            "n_clusters": n_flat,
            "n_noise": n_noise,
            "noise_ratio": round(n_noise / n, 4),
            "min_cluster_size": min_cluster_size,
            "cluster_names": {str(k): v for k, v in flat_names.items()},
            "method_labels": {m: flat_labels[m] for m in sorted(flat_labels.keys())},
        },
    }

    # 5. 后验簇效验证（ANOVA / Kruskal-Wallis，验证对象 = 关键层切割）
    if reactions:
        from llmsec.clustering.posterior import reaction_validation

        rv = reaction_validation(labels, reactions)
        rv["validated_on"] = f"ward_cut_k{k_best}"
        report["reaction_validation"] = rv

    if write:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(CLUSTER_REPORT_FILE, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        _export_matrix(labels, features, meta)

        artifacts = {
            "schema_version": 1,
            "kind": "cluster_result",
            "features": features,
            "meta": meta,
            "labels": labels,
            "hdbscan_labels": flat_labels,
            "cluster_names": cluster_names,
            "cluster_profiles": cluster_profiles,
            "linkage": Z,
            "whitened_coords": coords,
            "space_info": {
                "n_dims": space["n_dims"],
                "kept_variance": space["kept_variance"],
                "explained_variance_ratio": space["explained_variance_ratio"].tolist(),
                "lambda_w": space["lambda_w"],
                "damp": space["damp"],
            },
            "feature_weights": (
                feature_weights.tolist() if feature_weights is not None else None
            ),
            "chosen_k": k_best,
            "top_ks": top_ks,
            "candidate_sweep": sweep,
            "reaction_validation": report.get("reaction_validation"),
            "method_set_hash": _method_set_hash(methods),
            "hdbscan_report": report,
            "generated_at": report["generated_at"],
        }
        joblib.dump(artifacts, CLUSTER_RESULT_FILE)
        logger.info(
            "✅ 聚类完成: 关键层 k*=%d, 密度视图 %d 簇+%d 噪声, silhouette=%.4f",
            k_best, n_flat, n_noise, validation.get("silhouette", 0.0),
        )

    return report
