#!/usr/bin/env python3
"""
层次树层选择工具（算法无关）。

对任意 scipy 兼容 linkage 树（Ward / HDBSCAN single_linkage_tree_ 均可）：
1. 候选 k 以 log 增长的 k0 = ceil(log2(n)) 为中心取 log 间隔点
2. 每个 k 从同一棵树 fcluster 切出，计算 轮廓系数 / Calinski-Harabasz(方差比) / DB 指数
3. 归一化合成 S(k)，全局 argmax 选关键层；边界仍上升时自动外扩候选重 sweep（P8），扩满仍上升才标注 k 可能低估
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

# ============================================================
# auto-k：log 增长 + 候选集
# ============================================================

logger = get_logger(__name__)

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
        # F-2 修复：用有限大值（1e6）替代 float("inf")，避免 inf 经 _norm → NaN
        # 导致 json.dump(allow_nan=False) 崩溃整个聚类报告
        return {"silhouette": 0.0, "calinski_harabasz": 0.0, "davies_bouldin": 1e6}

    def _safe(fn, default=0.0, name=""):
        try:
            return float(fn(coords, y))
        except Exception as e:
            # T-1：逐 k 静默失败会污染 auto-k 合成 score（失败项=0→score 被拉低→chosen_k 偏移）。
            # 补 per-k 日志让指标失败可见。
            import logging
            logging.getLogger(__name__).debug("k=%d 簇指标 %s 失败（用默认值 %s）: %s",
                                              n_clusters, name, default, e)
            return default

    return {
        "silhouette": round(_safe(silhouette_score, name="silhouette"), 4),
        "calinski_harabasz": round(_safe(calinski_harabasz_score, name="CH"), 4),
        "davies_bouldin": round(_safe(davies_bouldin_score, default=1e6, name="DB"), 4),
    }


# P8：argmax 顶到最大候选且末尾仍上升时，自动向外扩展候选的次数上限（防无限扩展）
_MAX_K_EXPANSIONS = 2


def _assign_scores(entries: list[dict]) -> None:
    """归一化三指标并合成 score 写回 entries：轮廓/CH 越高越好，DB 越低越好（取反）。"""

    def _norm(key, invert=False):
        vals = np.array([e[key] for e in entries], dtype=float)
        # F-2 修复：钳位非有限值（inf/nan），防止 (inf-lo)/(hi-lo) 产生 NaN 污染 score
        vals = np.nan_to_num(vals, nan=0.0, posinf=1e6, neginf=-1e6)
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


def _argmax_boundary_rising(entries: list[dict]) -> bool:
    """argmax 落在最大候选且末尾仍在快速上升（与 select_knee 的边界判定同口径）。"""
    if len(entries) < 2:
        return False
    scores = np.array([e["score"] for e in entries], dtype=float)
    if int(np.argmax(scores)) != len(entries) - 1:
        return False
    deltas = np.diff(scores)
    return bool(
        deltas.size
        and float(deltas.max()) > 0
        and float(deltas[-1]) > KNEE_FLATTEN_RATIO * float(deltas.max())
    )


def sweep_candidates(
    coords: np.ndarray,
    Z: np.ndarray,
    methods: list[str],
    ks: list[int] | None = None,
    max_expansions: int = _MAX_K_EXPANSIONS,
) -> list[dict]:
    """
    对每个候选 k 切树并评估三指标，归一化后合成 score。
    返回按 k 升序的 [{"k", "silhouette", "calinski_harabasz", "davies_bouldin", "score"}]。

    P8：argmax 落在最大候选且末尾仍在上升时，自动向外扩展候选重 sweep——
    每次把 hi ×1.5（上限 max(n//5, 2*k0)，且 ≤ n-1），最多 max_expansions 次，
    避免候选范围偏窄把 k* 截断在边界；扩满仍上升由 select_knee 标注 k 可能低估。
    """
    if ks is None:
        ks = candidate_ks(len(methods))
    n = len(methods)
    entries: list[dict] = []
    evaluated: set[int] = set()
    pending = sorted(set(ks))
    expansions = 0
    while pending:
        for k in pending:
            labels = cut_tree(Z, methods, k)
            metrics = _evaluate_cut(coords, labels, methods)
            entries.append({"k": k, **metrics})
            evaluated.add(k)
        entries.sort(key=lambda e: e["k"])
        _assign_scores(entries)
        if expansions >= max_expansions or not _argmax_boundary_rising(entries):
            break
        hi = entries[-1]["k"]
        cap = min(n - 1, max(n // 5, 2 * log_growth_k0(n)))
        new_hi = min(int(hi * 1.5), cap)
        if new_hi <= hi:
            break
        num = min(4, new_hi - hi)
        pending = sorted(
            k
            for k in {int(round(v)) for v in np.geomspace(hi + 1, new_hi, num=num)}
            if k not in evaluated and hi < k <= n - 1
        )
        if not pending:
            break
        expansions += 1
        logger.info(
            "auto-k: 最大候选 k=%d 处得分仍在上升，候选向外扩展至 k=%d（第 %d 次扩展）",
            hi, new_hi, expansions,
        )
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
    raw_idx = int(np.argmax(scores))
    # A5 修复：对合成 score 做窗口=3 移动均值平滑，消除单点尖峰干扰。
    # 但仅在原始峰值不显著（≤邻居 1.1×）时采纳平滑结果——真正的主峰（远高于邻居）
    # 保持原始 argmax，防平滑把真实主峰拖低。
    chosen_idx = raw_idx
    if len(scores) >= 3:
        kernel = np.ones(3) / 3.0
        smoothed = np.convolve(scores, kernel, mode="same")
        smoothed_idx = int(np.argmax(smoothed))
        if smoothed_idx != raw_idx:
            left = float(scores[raw_idx - 1]) if raw_idx > 0 else 0.0
            right = float(scores[raw_idx + 1]) if raw_idx < len(scores) - 1 else 0.0
            if float(scores[raw_idx]) <= max(left, right, 1e-9) * 1.1:
                chosen_idx = smoothed_idx

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
