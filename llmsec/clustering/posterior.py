#!/usr/bin/env python3
"""
后验统计模块 —— 聚类相关代码中唯一接触评估结果（eval_results / Elo）的模块。

职责：
1. compute_method_reactions：从评估结果汇总每个方法的机器反应
2. learn_supervised_weights：弱监督特征权重（相同机器反应在度量中拉近）
3. reaction_validation：簇效验证（ANOVA / Kruskal-Wallis + 效应量）

设计约束：
- 聚类主流程（hdb.py / space.py）不直接解析 eval 数据，只接收本模块准备好的输入
- 所有统计只用真实 ground truth；未测方法的预测值不参与（防特征-预测自相关）
"""

import math

import numpy as np

from llmsec.core.logging import get_logger
from llmsec.params import (

    RV_ALPHA,
    RV_EFFECT_THRESHOLD,
    RV_MIN_GROUP,
    RV_POWER_COEF,
    SUPERVISED_WEIGHT_CLIP,
    SUPERVISED_WEIGHT_MIN_SAMPLES,
)




logger = get_logger(__name__)

def _finite(v: float, fallback: float) -> float:
    """统计量兜底：nan/inf（如组内零方差时 f_oneway/kruskal 的输出）替换为有限值，
    避免非法 JSON token（NaN/Infinity）写入报告。"""
    v = float(v)
    return v if math.isfinite(v) else fallback


# ============================================================
# 1. 机器反应汇总
# ============================================================
def compute_method_reactions(eval_results: list[dict]) -> dict[str, dict]:
    """
    从评估结果汇总每个方法的机器反应。

    返回: {method: {"mean_score": float, "n": int, "win_rate": float}}
    """
    from collections import defaultdict

    scores: dict[str, list[float]] = defaultdict(list)
    for r in eval_results:
        m = r.get("method")
        s = r.get("eval_score")
        if m is None or s is None:
            continue
        scores[m].append(float(s))

    return {
        m: {
            "mean_score": round(float(np.mean(v)), 4),
            "n": len(v),
            "win_rate": round(sum(1 for s in v if s > 0) / len(v), 4),
        }
        for m, v in scores.items()
    }


# ============================================================
# 2. 弱监督特征权重
# ============================================================
def learn_supervised_weights(
    X: np.ndarray,
    methods: list[str],
    y_by_method: dict[str, float],
    clip: tuple[float, float] = SUPERVISED_WEIGHT_CLIP,
    min_samples: int = SUPERVISED_WEIGHT_MIN_SAMPLES,
) -> np.ndarray:
    """
    弱监督特征权重：与机器反应相关性高的特征方向放大，无关方向压低。

    只在有真实反应的方法（ground truth）上计算 |pearson(X_j, y)|，
    未测方法的预测值不参与——否则特征与自身预测自相关，权重虚高。

    参数:
        X: (n, d) 原始特征矩阵
        methods: 与 X 行对齐的方法名
        y_by_method: {method: 反应值（如 mean eval_score / 真实 Elo）}
        clip: 权重裁剪范围（默认 params.SUPERVISED_WEIGHT_CLIP）
        min_samples: 有反应样本少于此数时不做加权（返回全 1，
            默认 params.SUPERVISED_WEIGHT_MIN_SAMPLES）

    返回: (d,) 权重向量，均值 ≈ 1，逐元素落在 clip 范围内
    """
    d = X.shape[1]
    y = np.array([y_by_method.get(m, np.nan) for m in methods], dtype=np.float64)
    mask = ~np.isnan(y)

    if int(mask.sum()) < min_samples:
        logger.info("弱监督样本不足 (%d < %d)，跳过特征加权", int(mask.sum()), min_samples)
        return np.ones(d)

    Xg, yg = X[mask], y[mask]
    y_std = yg.std()
    if y_std < 1e-12:
        return np.ones(d)

    x_std = Xg.std(axis=0)
    valid = x_std > 1e-12
    corr = np.zeros(d)
    if valid.any():
        Xc = (Xg[:, valid] - Xg[:, valid].mean(axis=0)) / x_std[valid]
        yc = (yg - yg.mean()) / y_std
        corr[valid] = np.abs(Xc.T @ yc) / len(yg)

    if corr.max() < 1e-6:
        logger.info("弱监督信号为零（特征与反应无相关），跳过特征加权")
        return np.ones(d)

    # 归一到均值 1 并裁剪，避免单个特征主导或消失。
    # 顺序固定为"先归一后 clip"：clip 后再归一（w /= w.mean()）会把最大值重新抬出
    # clip 上限（如 5.0），失去裁剪的保护意义；先归一后 clip 均值仍 ≈ 1。
    w = corr / corr.mean() if corr.mean() > 0 else np.ones(d)
    w = np.clip(w, clip[0], clip[1])

    logger.info(
        "弱监督权重: GT=%d, 最大相关=%.3f, 权重范围 [%.2f, %.2f]",
        int(mask.sum()), float(corr.max()), float(w.min()), float(w.max()),
    )
    return w


# ============================================================
# 3. 簇效验证（ANOVA / Kruskal-Wallis）
# ============================================================
def reaction_validation(
    labels: dict[str, int],
    reactions: dict[str, dict],
    min_group: int = RV_MIN_GROUP,
) -> dict:
    """
    簇效验证：不同簇的机器反应是否有显著差异。

    4 分支判定（significant × large_effect 完整 2×2 矩阵）：
      effective   p<α ∧ eta²>θ → 特征确实有效（簇对应不同机器反应）
      promising   p≥α ∧ eta²>θ → 效应量大但样本不足，特征方向正确
      weak        p<α ∧ eta²≤θ → 显著但效应量小
      ineffective p≥α ∧ eta²≤θ → 确实不相关

    附带统计功效评估（Cohen 经验式 adequate_n = RV_POWER_COEF·k + k²），
    样本不足时标注 underpowered，避免把"检验功效低"误读为"特征无用"。

    参数:
        labels: {method: cluster_id}（噪声 -1 单独作为一组）
        reactions: compute_method_reactions 的输出（只用真实 GT）
        min_group: 每簇至少这么多个已测方法才参与检验

    返回: {
        "available": bool,
        "p_anova": float, "p_kruskal": float,
        "eta2": float, "epsilon2": float,
        "effective": bool, "status": str, "verdict": str,
        "n_total": int, "n_groups": int, "adequate_n": int, "underpowered": bool,
        "per_cluster": {cid: {"n_tested", "mean_score", "win_rate"}},
    }
    """
    from scipy import stats


    groups: dict[int, list[float]] = {}
    wins: dict[int, list[float]] = {}
    for m, cid in labels.items():
        r = reactions.get(m)
        if r is None:
            continue
        groups.setdefault(int(cid), []).append(r["mean_score"])
        wins.setdefault(int(cid), []).append(r["win_rate"])

    per_cluster = {
        str(cid): {
            "n_tested": len(v),
            "mean_score": round(float(np.mean(v)), 3),
            "win_rate": round(float(np.mean(wins[cid])), 3),
        }
        for cid, v in sorted(groups.items())
    }

    testable = {c: v for c, v in groups.items() if len(v) >= min_group}
    if len(testable) < 2:
        return {
            "available": False,
            "reason": f"有效簇不足（每簇至少 {min_group} 个已测方法，当前 {len(testable)} 簇达标）",
            "per_cluster": per_cluster,
        }

    arrays = list(testable.values())
    F, p_anova = stats.f_oneway(*arrays)
    H, p_kw = stats.kruskal(*arrays)
    # 组内全同值（零方差）时 scipy 返回 nan/inf：p 值兜底 1.0（不显著），统计量兜底 0.0
    F = _finite(F, 0.0)
    p_anova = _finite(p_anova, 1.0)
    H = _finite(H, 0.0)
    p_kw = _finite(p_kw, 1.0)

    # 效应量：eta²（ANOVA）与 epsilon²（KW）
    all_vals = np.concatenate(arrays)
    grand = float(all_vals.mean())
    ss_between = sum(len(g) * (float(np.mean(g)) - grand) ** 2 for g in arrays)
    ss_total = float(((all_vals - grand) ** 2).sum())
    eta2 = _finite(ss_between / ss_total if ss_total > 1e-12 else 0.0, 0.0)
    k_groups, n_total = len(arrays), len(all_vals)
    epsilon2 = _finite(max(0.0, (H - k_groups + 1) / (n_total - k_groups)) if n_total > k_groups else 0.0, 0.0)

    # 统计功效评估：Cohen 经验式 adequate_n = COEF*k + k²
    adequate_n = RV_POWER_COEF * k_groups + k_groups ** 2
    underpowered = n_total < adequate_n

    # 4 分支判定（完整 2×2 矩阵）
    p_min = min(p_anova, p_kw)
    significant = p_min < RV_ALPHA
    large_effect = eta2 > RV_EFFECT_THRESHOLD or epsilon2 > RV_EFFECT_THRESHOLD

    if significant and large_effect:
        status = "effective"
        effective = True
        verdict = f"特征有效：不同簇的机器反应差异显著（p={p_min:.3f}）且效应量大（eta²={eta2:.2f}）"
    elif not significant and large_effect:
        status = "promising"
        effective = False
        if underpowered:
            verdict = (f"效应量大（eta²={eta2:.2f}）但未达显著（p={p_min:.2f}）——"
                       f"特征方向正确，但样本严重不足（{n_total}/{adequate_n} 建议），需更多数据确认")
        else:
            verdict = (f"效应量大（eta²={eta2:.2f}）但未达显著（p={p_min:.2f}）——"
                       f"特征方向正确，边界情况，建议积累更多数据")
    elif significant and not large_effect:
        status = "weak"
        effective = False
        verdict = (f"差异统计显著（p={p_min:.3f}）但效应量小（eta²={eta2:.2f}）："
                   f"簇与机器反应仅弱相关，特征抽象有提升空间")
    else:
        status = "ineffective"
        effective = False
        if underpowered and n_total < k_groups * 3:
            verdict = (f"p={p_min:.2f} 且 eta²={eta2:.2f}：样本极少（{n_total}/{adequate_n} 建议），"
                       f"无法区分'确实不相关'与'功效不足'")
        else:
            verdict = (f"p={p_min:.2f} 且 eta²={eta2:.2f}：不同簇的机器反应确实无差异，"
                       f"特征抓到的文本结构与该机器关心的不相关，特征抽象需升级")

    logger.info(
        "簇效验证: p_anova=%.4f, p_kw=%.4f, eta²=%.3f, ε²=%.3f, n=%d/%d → [%s] %s",
        p_anova, p_kw, eta2, epsilon2, n_total, adequate_n, status, verdict,
    )
    return {
        "available": True,
        "p_anova": round(float(p_anova), 6),
        "p_kruskal": round(float(p_kw), 6),
        "eta2": round(float(eta2), 4),
        "epsilon2": round(float(epsilon2), 4),
        "effective": effective,
        "status": status,
        "verdict": verdict,
        "n_total": n_total,
        "n_groups": k_groups,
        "adequate_n": adequate_n,
        "underpowered": underpowered,
        "per_cluster": per_cluster,
    }
