#!/usr/bin/env python3
"""
层次树层选择工具（算法无关）。

对任意 scipy 兼容 linkage 树（Ward / HDBSCAN single_linkage_tree_ 均可）：
1. 候选 k 以 log 增长的 k0 = ceil(log2(n)) 为中心取 log 间隔点
2. 每个 k 从同一棵树 fcluster 切出，计算 轮廓系数 / Calinski-Harabasz(方差比) / DB 指数
3. 归一化合成 S(k)，全局 argmax 选关键层；边界仍上升时标注 k 可能低估
4. 保留 top-3 k 作为前端树图缩放的预设停点

聚类主流程见 llmsec.clustering.hdb（HDBSCAN）。
"""

import math

import numpy as np

from llmsec.core.logging import get_logger
from llmsec.params import (
    KNEE_FLATTEN_RATIO,  # 边界上升判定：末尾增益高于最大增益的该比例即视为"仍在上升"
    TREE_K_MAX,
    TREE_K_MIN,
)

logger = get_logger(__name__)


# ============================================================
# auto-k：log 增长 + 候选集
# ============================================================
def log_growth_k0(n: int, k_min: int = TREE_K_MIN, k_max: int = TREE_K_MAX) -> int:
    """聚类量随数据规模 log 增长：n=100→7, n=1000→10, n=10000→14。

    小样本时 k_min 下限按 n//4 收缩（n<16 时 TREE_K_MIN=4 会恒把 k0 抬到 4，
    失去 log 增长意义）；n>=16 时下限即 TREE_K_MIN，行为不变。
    """
    lo = min(k_min, max(2, n // 4))
    return max(lo, min(int(math.ceil(math.log2(max(n, 2)))), k_max))


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
# 树切割
# ============================================================
def cut_tree(Z: np.ndarray, methods: list[str], k: int) -> dict[str, int]:
    """在 linkage 树上切出 k 个簇（标签规范化为 0..k-1）。"""
    from scipy.cluster.hierarchy import fcluster

    raw = fcluster(Z, t=k, criterion="maxclust")
    return {m: int(c) - 1 for m, c in zip(methods, raw)}


# ============================================================
# 候选评估与关键层选择
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
    auto-k 选择：全局 argmax S 为主规则。

    历史教训：小 k 端的非单调抖动会让"首个平缓点即停"把主峰丢掉
    （真实数据误判 k*=6，实际峰值 k=12）。argmax 在合成团簇上同样在
    真实簇数 ±2 内，且不受起始段抖动影响。
    若 argmax 落在最大候选且末尾仍在快速上升，标注 k 可能被低估。

    返回 (best_k, top3_ks)。
    """
    if not sweep:
        return 1, [1]
    scores = np.array([s["score"] for s in sweep], dtype=float)
    chosen_idx = int(np.argmax(scores))

    deltas = np.diff(scores)
    if (
        chosen_idx == len(sweep) - 1
        and deltas.size
        and float(deltas.max()) > 0
        and float(deltas[-1]) > flatten_ratio * float(deltas.max())
    ):
        # 边界仍在上升：k 可能被低估，候选范围偏窄
        sweep[-1]["boundary_rising"] = True
        logger.info("auto-k: 最大候选 k=%d 处得分仍在上升，k 可能被低估", sweep[-1]["k"])

    top3 = [s["k"] for s in sorted(sweep, key=lambda s: -s["score"])[:3]]
    return sweep[chosen_idx]["k"], sorted(top3)
