#!/usr/bin/env python3
"""
聚类级安全分析

基于聚类结果和当前 Elo 状态，对每个簇计算安全指标，
识别高风险簇、盲区簇和稳定簇，为人工审查和自适应采样提供依据。

用法:
    from llmsec.evaluation.cluster_analysis import analyze_clusters
    from llmsec.evaluation import ELOTracker

    tracker = ELOTracker()
    tracker = derive_elo(ResultsMatrix.load(), "model-name")
    analysis = analyze_clusters(tracker)
"""

from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

from llmsec.clustering import parse_cluster_id
from llmsec.core import config as _config  # 重绑常量调期动态读（work-dir 兼容）
from llmsec.core.config import CLUSTER_SECURITY_ANALYSIS_FILE
from llmsec.core.io import read_json, write_json
from llmsec.core.logging import get_logger
from llmsec.evaluation.elo import ELOTracker
from llmsec.params import (
    CLUSTER_BLIND_SPOT_ELO_MARGIN,
    CLUSTER_COVERAGE_BOUNDARY,
    CLUSTER_HIGH_RISK_ELO_MARGIN,
    CLUSTER_HIGH_RISK_MIN_SUCCESS,
    CLUSTER_STABLE_MAX_ELO_STD,
)

# ============================================================
# 数据加载
# ============================================================

logger = get_logger(__name__)

def load_cluster_artifacts(path: Path | str | None = None) -> dict | None:
    """加载聚类产物 pickle（cluster_result.pkl，完整 schema）。"""
    if path is None:
        path = _config.CLUSTER_RESULT_FILE
    path = Path(path)
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    # M-38：类 refactor 后 joblib.load 会抛 AttributeError/ModuleNotFoundError（pickle
    # 反序列化找不到旧类路径），原只捕 EOF/Value/Import 会漏，让诊断 CLI 整体崩溃
    except (EOFError, ValueError, ImportError, AttributeError, ModuleNotFoundError) as e:
        logger.error("聚类 artifacts 加载失败（%s）: %s", path, e)
        return None


def load_cluster_report(path: Path | str | None = None) -> dict | None:
    """加载 cluster_report.json。"""
    if path is None:
        path = _config.CLUSTER_REPORT_FILE
    return read_json(path)


# ============================================================
# SVD-Ridge 预测模型诊断
# ============================================================
def build_svd_ridge_summary(tracker: ELOTracker) -> dict | None:
    """
    汇总 SVD-Ridge Elo 预测模型的诊断信息：
    - 正则化路径（λ vs K-Fold 验证误差）与最优 λ
    - 特征重要性（Ridge 系数绝对值排序）
    - 未测方法的预测 Elo 置信区间（均值 ± t·σ）

    H7 修复：derive_elo 路径从不调 predict_batch → model.w 恒 None → 诊断恒空。
    此处检测到未训练时，用 artifacts 特征按需训练一次再取诊断。
    """
    predictor = getattr(tracker, "predictor", None)
    model = getattr(predictor, "model", None) if predictor else None
    if model is None:
        return None

    # H7：derive 路径下 model.w 为 None——按需训练
    if model.w is None:
        feats = predictor.artifacts.get("features", {}) if predictor.artifacts else {}
        if not feats or predictor.ground_truth_count() < predictor.min_cluster_size:
            return None
        # 簇粒度：artifacts 特征是 method 级而 GT 键是 unit_id——按需训练前换成
        # unit 级质心特征（units 由 final_fit 随聚类产物持久化）
        units = predictor.artifacts.get("units") if predictor.artifacts else None
        if units:
            from llmsec.core.units import build_unit_features
            feats = build_unit_features(feats, units)
            orig = predictor.artifacts
            predictor.artifacts = {**orig, "features": feats}
            try:
                predictor.predict_batch({m: {} for m in feats})
            except Exception as e:
                logger.warning("SVD-Ridge 模型按需训练失败，跳过诊断: %s", e)
                return None
            finally:
                predictor.artifacts = orig
        else:
            # 构造最小 method_records（build_prior_features 接受空 record）
            method_records = {m: {} for m in feats}
            try:
                predictor.predict_batch(method_records)
            except Exception as e:
                logger.warning("SVD-Ridge 模型按需训练失败，跳过诊断: %s", e)
                return None
        if model.w is None:
            return None

    predictions = {}
    for method, pred in (predictor.last_predictions or {}).items():
        if pred.get("source") != "svd_ridge":
            continue
        predictions[method] = {
            "elo": pred.get("elo"),
            "std": pred.get("std"),
            "ci95": pred.get("ci95"),
        }

    return {
        "lambda_opt": model.lambda_opt,
        "sigma2": round(model.sigma2, 4) if model.sigma2 is not None else None,
        "regularization_path": model.get_regularization_path(),
        "pca_summary": model.get_pca_summary(),
        "feature_importance": model.get_feature_importance(top_n=20),
        "n_ground_truth": predictor.ground_truth_count(),
        "predictions": predictions,
    }


def build_blend_predictor_summary(tracker: ELOTracker) -> dict | None:
    """构建 BlendPredictor（多模型 sim-加权）并返回诊断：发现层 donor 相似度 / sim-加权状态。

    供看板"多模型层"展示——live 冷启动实际用的预测器（区别于 ColdStartPredictor 单模型层）。
    特征不可用 / R 为空时返回 None（看板显示"无多模型诊断"）。
    """
    pred = getattr(tracker, "predictor", None)
    feats = pred.artifacts.get("features", {}) if pred and pred.artifacts else {}
    if not feats:
        return None
    # 簇粒度：R 行键 = 记录 id、derive_elo 按 extra.unit 聚合出 unit 键——
    # 特征同步换成 unit 级质心，键空间才一致
    units = pred.artifacts.get("units") if pred.artifacts else None
    if units:
        from llmsec.core.units import build_unit_features
        feats = build_unit_features(feats, units)
    try:
        from llmsec.core.results import ResultsMatrix
        from llmsec.evaluation.predictors.blend import load_or_fit_blend_predictor

        R = ResultsMatrix.load()
        if not R.all_models():
            return None
        catalog = sorted(feats.keys())
        bp = load_or_fit_blend_predictor(R, feats, method_catalog=catalog)
        return bp.diagnostics()
    except Exception as e:
        logger.warning("BlendPredictor 诊断生成失败: %s", e, exc_info=True)
        return {"error": str(e)}


# ============================================================
# 分析核心
# ============================================================
def analyze_clusters(
    tracker: ELOTracker,
    cluster_report: dict | None = None,
    cluster_artifacts: dict | None = None,
    defender_name: str | None = None,
) -> dict:
    """
    对聚类结果做安全分析。

    参数:
        tracker: ELOTracker 实例
        cluster_report: 聚类报告 dict；None 则自动加载
        cluster_artifacts: 聚类 artifacts；None 则自动加载
        defender_name: 防御方名称；None 则取第一个防御方

    返回:
        {
            "defender_name": str,
            "defender_elo": float,
            "n_methods": int,
            "n_clusters": int,
            "clusters": {cid: {...}},
            "high_risk_clusters": [...],
            "blind_spot_clusters": [...],
            "stable_clusters": [...],
            "svd_ridge": {...},  # SVD-Ridge 模型已训练时存在
        }
    """
    if cluster_report is None:
        cluster_report = load_cluster_report()
    if cluster_artifacts is None:
        cluster_artifacts = load_cluster_artifacts()

    if not cluster_report and not cluster_artifacts:
        return {"error": "无聚类数据，请先运行聚类"}

    # 优先使用 artifacts 中的 labels，其次 report
    labels = {}
    if cluster_artifacts and "labels" in cluster_artifacts:
        labels = cluster_artifacts["labels"]
    elif cluster_report and "method_labels" in cluster_report:
        labels = cluster_report["method_labels"]

    cluster_names = {}
    if cluster_report and "cluster_names" in cluster_report:
        cluster_names = cluster_report["cluster_names"]

    # 簇粒度（unit）路径：artifacts 带 units 时，tracker 的评级键就是 unit_id——
    # 每个 unit 即一个簇，members 传入 [uid] 直接命中 tracker 的键空间
    units = (cluster_artifacts or {}).get("units") or {}

    if defender_name is None:
        defender_name = (
            list(tracker.defender_ratings.keys())[0]
            if tracker.defender_ratings
            else "target-model"
        )

    defender_elo = tracker.get_defender_elo(defender_name)

    # 按簇分组（unit 路径：label → [unit_id]；否则 method 粒度旧路径：label → [method]）
    clusters = defaultdict(list)
    if units:
        for uid, u in units.items():
            clusters[parse_cluster_id(u.get("label", 0))].append(uid)
        if not cluster_names:
            cluster_names = {str(u.get("label", 0)): u.get("name") for u in units.values()}
    else:
        for method, cid in labels.items():
            clusters[parse_cluster_id(cid)].append(method)

    # 从 history 计算每个键（unit/method）的 eval_score 历史（用于 ASR）
    method_scores: dict[str, list[float]] = defaultdict(list)
    for h in tracker.history:
        method_scores[h["attacker"]].append(h["eval_score"])

    cluster_details = {}
    for cid, members in clusters.items():
        detail = analyze_single_cluster(
            cid,
            members,
            tracker,
            defender_elo,
            cluster_names,
            method_scores,
        )
        if units:
            # unit 即簇：size/成员/覆盖率按 unit 真实语义修正（单键入参时默认 size=1）
            u = units.get(members[0], {})
            n_match = tracker.attacker_stats.get(members[0], {}).get("n_matches", 0)
            detail["unit_id"] = members[0]
            detail["size"] = u.get("size", 1)
            detail["members"] = sorted(u.get("members", []))[:50]
            detail["tested_members"] = []
            detail["tested_records"] = n_match
            detail["test_coverage"] = round(min(1.0, n_match / max(1, u.get("size", 1))), 4)
        cluster_details[str(cid)] = detail

    # 分类簇
    high_risk = []
    blind_spots = []
    stable = []

    for cid_str, detail in cluster_details.items():
        # 高风险：高成功率 + 接近或高于边界
        if detail["mean_success_rate"] >= CLUSTER_HIGH_RISK_MIN_SUCCESS and detail["mean_elo"] >= defender_elo - CLUSTER_HIGH_RISK_ELO_MARGIN:
            high_risk.append(cid_str)
        # 盲区：测试覆盖低 + 平均 Elo 接近边界
        elif detail["test_coverage"] < CLUSTER_COVERAGE_BOUNDARY and abs(detail["mean_elo"] - defender_elo) <= CLUSTER_BLIND_SPOT_ELO_MARGIN:
            blind_spots.append(cid_str)
        # 稳定：覆盖足够 + 方差低
        elif detail["test_coverage"] >= CLUSTER_COVERAGE_BOUNDARY and detail["elo_std"] <= CLUSTER_STABLE_MAX_ELO_STD:
            stable.append(cid_str)

    analysis = {
        "defender_name": defender_name,
        "defender_elo": round(defender_elo, 2),
        "n_methods": len(labels),
        "n_clusters": len([c for c in clusters.keys() if c != -1]),
        "n_noise": len(clusters.get(-1, [])),
        "clusters": cluster_details,
        "high_risk_clusters": high_risk,
        "blind_spot_clusters": blind_spots,
        "stable_clusters": stable,
    }

    # H-12 修复：透传聚类后验簇效验证（ANOVA/Kruskal-Wallis），让下游知道簇间差异
    # 是否有统计显著性。原模块只做描述性统计，结论无统计支撑。
    if cluster_report and "reaction_validation" in cluster_report:
        analysis["reaction_validation"] = cluster_report["reaction_validation"]

    # SVD-Ridge 预测模型诊断（正则化路径 / 最优 λ / 特征重要性 / 预测置信区间）
    try:
        svd_summary = build_svd_ridge_summary(tracker)
        if svd_summary:
            analysis["svd_ridge"] = svd_summary
        else:
            # 诊断：模型未训练时记录原因，便于排查（不再静默丢失）
            pred = getattr(tracker, "predictor", None)
            model = getattr(pred, "model", None) if pred else None
            gt_n = pred.ground_truth_count() if pred else 0
            if model is None or model.w is None:
                feats = pred.artifacts.get("features", {}) if pred and pred.artifacts else {}
                gt_all = sorted(pred.ground_truth.keys()) if pred else []
                # 簇粒度：GT 键是 unit_id 而 artifacts 特征是 method 级——
                # 用 unit 质心特征做匹配判断，避免误报"stale 污染"
                _units = (pred.artifacts or {}).get("units") if pred else None
                if _units and feats:
                    from llmsec.core.units import build_unit_features
                    feats = build_unit_features(feats, _units)
                gt_in_feats = [m for m in gt_all if m in feats]
                stale = len(gt_all) - len(gt_in_feats)
                if stale > 0:
                    reason = f"GT {gt_n} 中 {stale} 个单位不在当前特征集（跨攻击集 stale 污染），可用 {len(gt_in_feats)}"
                elif not feats:
                    reason = "特征缓存为空（extract_all_features 可能失败）"
                elif gt_n < (pred.min_cluster_size if pred else 3):
                    reason = f"GT 数 {gt_n} 不足（min_cluster_size={pred.min_cluster_size if pred else 3}）"
                else:
                    # 兜底：GT 够、特征也在，但模型仍未训练。细化原因而非"未知"：
                    # 1) predictor.model 本身为 None（冷启动未装配模型）
                    # 2) 训练后 model.w 仍 None（特征矩阵奇异/零方差/数值问题导致 Ridge 解退化）
                    if model is None:
                        reason = f"GT {gt_n} 充足但 predictor.model 未装配（冷启动模型装配失败）"
                    elif getattr(model, "w", None) is None:
                        reason = (f"GT {gt_n} 充足但 Ridge 解退化（model.w=None）："
                                  "特征矩阵可能奇异或零方差，无法求逆。建议检查特征提取是否产出常量列。")
                    else:
                        reason = f"GT {gt_n} 但模型未训练（未预期路径，请附日志反馈）"
                analysis["svd_ridge_skipped"] = reason
                logger.warning("SVD-Ridge 诊断跳过: %s", reason)
    except Exception as e:
        logger.warning("SVD-Ridge 诊断生成失败: %s", e, exc_info=True)
        analysis["svd_ridge_error"] = str(e)

    # BlendPredictor（多模型 sim-加权）诊断——看板"多模型层"（live 冷启动实际用的预测器）
    try:
        bp_summary = build_blend_predictor_summary(tracker)
        if bp_summary:
            analysis["blend_predictor"] = bp_summary
    except Exception as e:
        logger.warning("BlendPredictor 诊断写入失败: %s", e)

    return analysis


def analyze_single_cluster(
    cid: int,
    members: list[str],
    tracker: ELOTracker,
    defender_elo: float,
    cluster_names: dict,
    method_scores: dict[str, list[float]],
) -> dict:
    """分析单个簇的安全指标。"""
    elos = []
    test_counts = []
    success_rates = []
    above_boundary = 0
    distances = []

    tested_members = set()
    all_scores = []

    for method in members:
        elo = tracker.get_attacker_elo(method)
        elos.append(elo)
        distances.append(abs(elo - defender_elo))
        if elo > defender_elo:
            above_boundary += 1

        stats = tracker.attacker_stats.get(method, {})
        n = stats.get("n_matches", 0)
        test_counts.append(n)
        if n > 0:
            tested_members.add(method)
            wins = stats.get("wins", 0)
            success_rates.append(wins / n)
            all_scores.extend(method_scores.get(method, []))
        else:
            success_rates.append(0.0)

    elos_arr = np.array(elos) if elos else np.array([0.0])
    test_counts_arr = np.array(test_counts) if test_counts else np.array([0.0])
    success_rates_arr = np.array(success_rates) if success_rates else np.array([0.0])

    asr = (
        sum(1 for s in all_scores if s > 0) / len(all_scores)
        if all_scores
        else 0.0
    )

    return {
        "cluster_id": cid,
        "name": cluster_names.get(str(cid), f"簇{cid}"),
        "size": len(members),
        "members": sorted(members),
        "tested_members": sorted(tested_members),
        "test_coverage": round(len(tested_members) / len(members), 4) if members else 0.0,
        "mean_elo": round(float(np.mean(elos_arr)), 2),
        "elo_std": round(float(np.std(elos_arr)), 2),
        "min_elo": round(float(np.min(elos_arr)), 2),
        "max_elo": round(float(np.max(elos_arr)), 2),
        "mean_tests": round(float(np.mean(test_counts_arr)), 2),
        "mean_success_rate": round(float(np.mean(success_rates_arr)), 4),
        "asr": round(asr, 4),
        "methods_above_boundary": above_boundary,
        "distance_to_boundary": round(float(np.mean(distances)), 2) if distances else 0.0,
    }


# ============================================================
# 导出
# ============================================================
def save_cluster_analysis(
    analysis: dict,
    output_path: Path | str | None = None,
):
    """保存聚类安全分析结果到 JSON。"""
    if output_path is None:
        output_path = CLUSTER_SECURITY_ANALYSIS_FILE
    output_path = Path(output_path)
    write_json(output_path, analysis)
    return output_path


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="聚类级安全分析")
    parser.add_argument(
        "--defender",
        type=str,
        default=None,
        help="防御方名称；默认取 ELO 文件中第一个防御方",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径；默认 output/cluster_security_analysis.json",
    )
    args = parser.parse_args()

    # M-31：R 为原始观测——经 elo_access 统一入口派生，不再直读易漂移的 state.json。
    # 旧实现只读 state.json：只有 results.json 的部署会用全初始 Elo 生成"所有簇都是盲区"
    # 的误导性分析并落盘。
    from llmsec.core.results import ResultsMatrix
    from llmsec.evaluation.elo import derive_elo
    from llmsec.evaluation.elo_access import active_model


    R = ResultsMatrix.load()
    model = args.defender or active_model()
    if model is None or not R.model_column(model):
        logger.warning("⚠ 结果矩阵 R 无任何模型数据（请先跑评估）。聚类安全分析需真实 Elo，"
              "以下为空分析。")
        tracker = ELOTracker()
    else:
        tracker = derive_elo(R, model)

    analysis = analyze_clusters(tracker, defender_name=model or args.defender)
    out_path = save_cluster_analysis(analysis, args.output)
    logger.info(f"聚类安全分析已保存: {out_path}")
    logger.info(f"  簇数: {analysis.get('n_clusters', 0)}")
    logger.info(f"  高风险簇: {analysis.get('high_risk_clusters', [])}")
    logger.info(f"  盲区簇: {analysis.get('blind_spot_clusters', [])}")
    logger.info(f"  稳定簇: {analysis.get('stable_clusters', [])}")
