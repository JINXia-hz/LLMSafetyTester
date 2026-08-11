#!/usr/bin/env python3
"""
D-Optimality 主动学习。

与 SVD-Ridge 预测模型同源：Ridge 的信息矩阵 M = X_gtᵀ X_gt + λI，
未测点 x 的 MAP 预测方差正比于 xᵀ M⁻¹ x（D-optimality 杠杆）。
选杠杆最大的点做真实评估，就是对预测矩阵信息量最大的学习。

贪心选择：每选中一个点，用 Sherman–Morrison 公式秩1更新 M⁻¹：
    M⁻¹ ← M⁻¹ - (M⁻¹ x xᵀ M⁻¹) / (1 + xᵀ M⁻¹ x)
精确且每次更新 O(d²)，无需重求逆。

用法:
    from llmsec.evaluation.predictors.active_learning import greedy_d_optimal
    idx = greedy_d_optimal(X_candidates, n=8, lam=1.0, X_gt=X_tested)
"""

import numpy as np


def d_optimal_scores(X: np.ndarray, M_inv: np.ndarray) -> np.ndarray:
    """逐行计算 D-optimality 杠杆 xᵀ M⁻¹ x。"""
    return np.sum((X @ M_inv) * X, axis=1)


def greedy_d_optimal(
    X: np.ndarray,
    n_select: int,
    lam: float = 1.0,
    X_gt: np.ndarray | None = None,
) -> list[int]:
    """
    贪心 D-optimal 选择。

    参数:
        X: (n, d) 候选点特征矩阵（应已标准化）
        n_select: 选择数量
        lam: Ridge 正则（信息矩阵地板；GT 为空时 M = λI，
            自动退化为最大范数点优先，覆盖特征空间）
        X_gt: 已有 ground truth 的特征矩阵（可选，纳入信息矩阵）

    返回: 选中点的行索引列表（按选择顺序）
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    if n == 0 or n_select <= 0:
        return []

    if X_gt is not None and len(X_gt) > 0:
        M = X_gt.T @ X_gt + lam * np.eye(d)
        M_inv = np.linalg.inv(M)
    else:
        M_inv = np.eye(d) / lam

    selected: list[int] = []
    available = np.arange(n)
    for _ in range(min(n_select, n)):
        scores = d_optimal_scores(X[available], M_inv)
        pick_pos = int(np.argmax(scores))
        idx = int(available[pick_pos])
        selected.append(idx)

        x = X[idx]
        Mx = M_inv @ x
        M_inv = M_inv - np.outer(Mx, Mx) / (1.0 + float(x @ Mx))
        available = np.delete(available, pick_pos)

    return selected
