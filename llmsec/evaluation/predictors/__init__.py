"""
evaluation.predictors — 冷启动 ELO 预测器子包。

预测器职责：为**未测过**的攻击方法/模型预测初始 Elo（冷启动），填补评估矩阵的空白。
与 elo.py 的 ELOTracker（真实评级的增量更新）互补——预测器只在缺数据时提供先验。

模块：
  - svd_ridge.py    EloPredictorModel：纯 ML 模型（SVD + Ridge 回归 + K-Fold 选 λ）
  - cold_start.py   ColdStartPredictor：编排器（GT 库 / 特征缓存 / 工件 / 变体兜底 / 预测分发）
  - blend.py        BlendPredictor：统一层 + 模型层双层预测（贝叶斯收缩加权）
  - fingerprint.py  模型防御指纹：D-optimal 哨兵种子的 per-seed Elo 向量，量化模型相似度
  - active_learning.py  D-Optimality 贪心主动学习（哨兵种子选择）

依赖方向（无环）：
  active_learning（叶）← cold_start ← elo.py(ELOTracker 内嵌一个预测器)
  svd_ridge（叶）← cold_start, blend
  fingerprint（叶）← blend（lazy）, attack_phase（lazy）
"""

from llmsec.evaluation.predictors.active_learning import greedy_d_optimal
from llmsec.evaluation.predictors.blend import BlendPredictor, load_or_fit_blend_predictor
from llmsec.evaluation.predictors.cold_start import ColdStartPredictor
from llmsec.evaluation.predictors.fingerprint import (
    compute_fingerprint,
    donor_similarities,
    load_probes,
    model_similarity,
    save_probe,
)
from llmsec.evaluation.predictors.svd_ridge import EloPredictorModel

__all__ = [
    # ML 模型
    "EloPredictorModel",
    # 编排器
    "ColdStartPredictor",
    # 双层预测器
    "BlendPredictor", "load_or_fit_blend_predictor",
    # 指纹
    "compute_fingerprint", "model_similarity", "donor_similarities",
    "save_probe", "load_probes",
    # 主动学习
    "greedy_d_optimal",
]
