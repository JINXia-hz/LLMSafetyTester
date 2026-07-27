#!/usr/bin/env python3
"""
层次树聚类 + 多指标拐点 auto-k。

核心思路（一次建树，多次切割）：
1. 在马氏白化坐标上做一次 Ward 层次聚类，保留完整 linkage 树
2. 候选 k 以 log 增长的 k0 = ceil(log2(n)) 为中心取 log 间隔点
3. 每个 k 从同一棵树 fcluster 切出，计算 轮廓系数 / Calinski-Harabasz(方差比) / DB 指数
4. 三指标归一化合成 S(k)，取增益开始平缓的拐点（差分 < max(Δ)×0.2）之前的最佳 k
5. 保留 top-3 k 作为前端树图缩放的预设停点

用法:
    from llmsec.clustering.tree import run_tree_clustering
    report = run_tree_clustering(features, meta)
"""

import hashlib
import math
from datetime import datetime

import joblib
import numpy as np

from llmsec.clustering.pipeline import (
    _export_matrix,
    auto_name_clusters,
    build_cluster_profiles,
)
from llmsec.clustering.space import build_whitened_space
from llmsec.core.config import CLUSTER_ARTIFACTS_FILE, CLUSTER_REPORT_FILE, OUTPUT_DIR
from llmsec.core.logging import get_logger

logger = get_logger(__name__)

# 拐点判定：增益低于最大增益的该比例即视为"开始平缓"
KNEE_FLATTEN_RATIO = 0.2


# ============================================================
# auto-k：log 增长 + 候选集
# ============================================================
def log_growth_k0(n: int, k_min: int = 4, k_max: int = 20) -> int:
    """聚类量随数据规模 log 增长：n=100→7, n=1000→10, n=10000→14。"""
    return max(k_min, min(int(math.ceil(math.log2(max(n, 2)))), k_max))


def candidate_ks(n: int) -> list[int]:
    """以 k0 为中心生成 log 间隔的候选 k（约 8~10 个）。"""
    if n < 4:
        return [1] if n >= 1 else []
    k0 = log_growth_k0(n)
    lo = max(2, k0 - 2)
    hi = min(n - 1, 2 * k0)
    if hi <= lo:
        return sorted({min(max(k0, 2), n - 1)})
    num = min(10, hi - lo + 1)
    ks = sorted(set(int(round(v)) for v in np.geomspace(lo, hi, num=num)))
    return [k for k in ks if 2 <= k <= n - 1]


# ============================================================
# 树构建与切割
# ============================================================
def build_tree(coords: np.ndarray) -> np.ndarray | None:
    """Ward 层次聚类，返回 scipy linkage 矩阵。"""
    from scipy.cluster.hierarchy import linkage

    n = coords.shape[0]
    if n < 2:
        return None
    return linkage(coords, method="ward")


def cut_tree(Z: np.ndarray, methods: list[str], k: int) -> dict[str, int]:
    """在 linkage 树上切出 k 个簇（标签规范化为 0..k-1）。"""
    from scipy.cluster.hierarchy import fcluster

    raw = fcluster(Z, t=k, criterion="maxclust")
    return {m: int(c) - 1 for m, c in zip(methods, raw)}


# ============================================================
# 候选评估与拐点选择
# ============================================================
def _evaluate_cut(coords: np.ndarray, labels: dict[str, int], methods: list[str]) -> dict:
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    y = np.array([labels[m] for m in methods])
    n_clusters = len(set(y.tolist()))
    if n_clusters < 2 or n_clusters >= len(methods):
        return {"silhouette": 0.0, "calinski_harabasz": 0.0, "davies_bouldin": float("inf")}

    def _safe(fn, default=0.0):
        try:
            return float(fn(coords, y))
        except Exception:
            return default

    return {
        "silhouette": round(_safe(silhouette_score), 4),
        "calinski_harabasz": round(_safe(calinski_harabasz_score), 4),
        "davies_bouldin": round(_safe(davies_bouldin_score, default=float("inf")), 4),
    }


def sweep_candidates(
    coords: np.ndarray,
    Z: np.ndarray,
    methods: list[str],
    ks: list[int] | None = None,
) -> list[dict]:
    """
    对每个候选 k 切树并评估三指标，归一化后合成 score。
    返回按 k 升序的 [{"k", "silhouette", "calinski_harabasz", "davies_bouldin", "score"}]。
    """
    if ks is None:
        ks = candidate_ks(len(methods))
    entries = []
    for k in ks:
        labels = cut_tree(Z, methods, k)
        metrics = _evaluate_cut(coords, labels, methods)
        entries.append({"k": k, **metrics})

    # 归一化：轮廓/CH 越高越好，DB 越低越好（取反）
    def _norm(key, invert=False):
        vals = np.array([e[key] for e in entries], dtype=float)
        lo, hi = float(vals.min()), float(vals.max())
        if hi - lo < 1e-12:
            return [0.5] * len(entries)
        out = (vals - lo) / (hi - lo)
        if invert:
            out = 1.0 - out
        return out.tolist()

    ns = _norm("silhouette")
    nc = _norm("calinski_harabasz")
    nd = _norm("davies_bouldin", invert=True)
    for i, e in enumerate(entries):
        e["score"] = round((ns[i] + nc[i] + nd[i]) / 3.0, 4)
    return entries


def select_knee(sweep: list[dict], flatten_ratio: float = KNEE_FLATTEN_RATIO) -> tuple[int, list[int]]:
    """
    拐点选择：增益 Δ 首次低于 max(Δ)×flatten_ratio 视为开始平缓，
    在拐点（含）之前的候选中选 score 最优；全程无拐点则取全局 argmax。
    返回 (best_k, top3_ks)。
    """
    if not sweep:
        return 1, [1]
    scores = np.array([s["score"] for s in sweep], dtype=float)
    chosen_idx = int(np.argmax(scores))

    deltas = np.diff(scores)
    if deltas.size and float(deltas.max()) > 0:
        thr = flatten_ratio * float(deltas.max())
        for i, d in enumerate(deltas):
            if d < thr:
                chosen_idx = int(np.argmax(scores[: i + 2]))
                break

    top3 = [s["k"] for s in sorted(sweep, key=lambda s: -s["score"])[:3]]
    return sweep[chosen_idx]["k"], sorted(top3)


# ============================================================
# 主入口
# ============================================================
def _method_set_hash(methods: list[str]) -> str:
    return hashlib.md5(",".join(sorted(set(methods))).encode("utf-8")).hexdigest()


def run_final_tree_clustering(
    features: dict,
    meta: dict,
    write: bool = True,
) -> dict:
    """
    最终聚类（攻击完成后，含真实评估数据）：
    1. 树聚类（auto-k 主标签，供采样/分析/展示）
    2. 白化坐标上的递归 DBSCAN（密度视图：核心小簇 + 噪声，全部命名）
    3. 后验画像由 build_cluster_profiles 的 defense 均值承担（不进入度量）

    返回: 聚类报告 dict（含 dbscan 密度视图段）。
    """
    from llmsec.clustering.pipeline import (
        euclidean_distance_matrix,
        run_dbscan_recursive,
    )

    report = run_tree_clustering(features, meta, write=write)

    methods = sorted(features.keys())
    n = len(methods)
    if n < 2 or "error" in report:
        return report

    # 密度视图：白化坐标 → 距离矩阵 → 递归 DBSCAN（大簇自动细分）
    space = build_whitened_space(features, methods)
    dist = euclidean_distance_matrix(space["coords"], standardize=False)
    dbscan_labels = run_dbscan_recursive(
        dist, methods, max_cluster_size=max(10, n // 8)
    )
    dbscan_names = auto_name_clusters(
        dbscan_labels, features, meta, meta.get("method_prompts", {})
    )

    n_core = len(set(dbscan_labels.values()) - {-1})
    n_noise = sum(1 for v in dbscan_labels.values() if v == -1)
    report["dbscan"] = {
        "n_core_clusters": n_core,
        "n_noise": n_noise,
        "noise_ratio": round(n_noise / n, 4),
        "cluster_names": {str(k): v for k, v in dbscan_names.items()},
        "method_labels": {m: dbscan_labels[m] for m in sorted(methods)},
    }
    report["clustering_method"] = "tree_ward_autok+dbscan_recursive"

    if write:
        import json

        artifacts = joblib.load(CLUSTER_ARTIFACTS_FILE)
        artifacts["dbscan_labels"] = dbscan_labels
        artifacts["dbscan_cluster_names"] = dbscan_names
        artifacts["is_final_cluster"] = True
        joblib.dump(artifacts, CLUSTER_ARTIFACTS_FILE)
        with open(CLUSTER_REPORT_FILE, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(
        "✅ 最终聚类: 树 k*=%d, DBSCAN 核心簇=%d, 噪声=%d",
        report.get("n_clusters", 0), n_core, n_noise,
    )
    return report


def run_tree_clustering(
    features: dict,
    meta: dict,
    write: bool = True,
) -> dict:
    """
    树聚类主入口：白化空间 → Ward 树 → 拐点 auto-k → 切割 → 命名 → 画像。

    产出（与既有消费方 schema 兼容）：
    - cluster_report.json：method_labels / cluster_names / validation / sweep / top_ks
    - cluster_artifacts.pkl：labels / features / meta / linkage / whitened_coords / space_info

    返回: 聚类报告 dict。
    """
    methods = sorted(features.keys())
    n = len(methods)
    if n < 2:
        return {"error": "方法数不足", "labels": {m: 0 for m in methods}}

    logger.info("🌲 树聚类: %d 种方法", n)

    # 1. 马氏白化空间
    space = build_whitened_space(features, methods)
    coords = space["coords"]
    logger.info(
        "  白化空间: %d 维 (累计解释方差 %.1f%%)",
        space["n_dims"], space["kept_variance"] * 100,
    )

    # 2. Ward 层次树（只建一次）
    Z = build_tree(coords)

    # 3. 候选 k 扫描 + 拐点选择
    ks = candidate_ks(n)
    sweep = sweep_candidates(coords, Z, methods, ks)
    k_best, top_ks = select_knee(sweep)
    logger.info("  auto-k: 候选 %s → k*=%d (top3 %s)", ks, k_best, top_ks)

    # 4. 切割 + 命名 + 画像
    labels = cut_tree(Z, methods, k_best)
    cluster_names = auto_name_clusters(labels, features, meta, meta.get("method_prompts", {}))
    cluster_profiles = build_cluster_profiles(labels, features, meta, cluster_names)

    best_entry = next((s for s in sweep if s["k"] == k_best), {})
    validation = {
        "silhouette": best_entry.get("silhouette", 0.0),
        "calinski_harabasz": best_entry.get("calinski_harabasz", 0.0),
        "davies_bouldin": best_entry.get("davies_bouldin", 0.0),
    }

    report = {
        "generated_at": datetime.now().isoformat(),
        "method_count": n,
        "clustering_method": "tree_ward_autok",
        "n_clusters": k_best,
        "n_noise": 0,
        "target_k": k_best,
        "k0_log_growth": log_growth_k0(n),
        "top_ks": top_ks,
        "candidate_sweep": sweep,
        "validation": validation,
        "cluster_names": {str(k): v for k, v in cluster_names.items()},
        "cluster_profiles": cluster_profiles,
        "method_labels": {m: labels[m] for m in sorted(labels.keys())},
    }

    if write:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        import json

        with open(CLUSTER_REPORT_FILE, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        _export_matrix(labels, features, meta)

        artifacts = {
            "features": features,
            "meta": meta,
            "labels": labels,
            "cluster_names": cluster_names,
            "cluster_profiles": cluster_profiles,
            "linkage": Z,
            "whitened_coords": coords,
            "space_info": {
                "n_dims": space["n_dims"],
                "kept_variance": space["kept_variance"],
                "explained_variance_ratio": space["explained_variance_ratio"].tolist(),
                "lambda_w": space["lambda_w"],
            },
            "chosen_k": k_best,
            "top_ks": top_ks,
            "candidate_sweep": sweep,
            "method_set_hash": _method_set_hash(methods),
            "tree_cluster_report": report,
            "generated_at": report["generated_at"],
        }
        joblib.dump(artifacts, CLUSTER_ARTIFACTS_FILE)
        logger.info("✅ 树聚类完成: k*=%d, silhouette=%.4f", k_best, validation["silhouette"])

    return report
