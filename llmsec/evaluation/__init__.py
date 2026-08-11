"""
llmsec.evaluation — 评估子包

  - judge.py / prescreen_ml.py  LLM-as-Judge 评分 + 拒绝预筛
  - elo.py / elo_access.py      双轨 ELO 追踪 + R 读写网关
  - evaluator.py / scoring.py   evaluate_single 评估核心 + 评分纯函数
  - safe_twin.py                安全孪生生成与过敏（FPR）检测
  - samplers.py                 攻击采样策略
  - cluster_analysis.py         聚类级安全分析
  - predictors/                 冷启动 ELO 预测器子包（svd_ridge/cold_start/blend/fingerprint/active_learning）

常用符号再导出，供 runner 等模块按 `from llmsec.evaluation import ...` 使用。
"""

from llmsec.evaluation.cluster_analysis import analyze_clusters, save_cluster_analysis
from llmsec.evaluation.elo import ELOTracker, derive_elo
from llmsec.evaluation.elo_access import (
    active_model,
    attacker_ratings_for,
    elo_state_for,
    publish_tracker,
)
from llmsec.evaluation.evaluator import (
    build_summary,
    evaluate_single,
    update_elo,
)
from llmsec.evaluation.judge import (
    FAST_HARMFUL_SIGNALS,
    FAST_REFUSAL_PATTERNS,
    Judge,
    create_judge_client,
    fast_prescreen,
    parse_compliance_level,
)
from llmsec.evaluation.predictors import (
    BlendPredictor,
    ColdStartPredictor,
    EloPredictorModel,
    compute_fingerprint,
    donor_similarities,
    greedy_d_optimal,
    load_or_fit_blend_predictor,
    load_probes,
    model_similarity,
    save_probe,
)
from llmsec.evaluation.prescreen_ml import predict as prescreen_predict
from llmsec.evaluation.prescreen_ml import train as train_prescreen
from llmsec.evaluation.safe_twin import SAFE_TWIN_SYSTEM, generate_safe_twin
from llmsec.evaluation.samplers import (
    AttackSampler,
    CoordinateDescentSampler,
    GapMinSampler,
    HybridSampler,
    InfoGainSampler,
    build_sampler,
)
from llmsec.evaluation.scoring import (
    compute_eval_score_v2,
    compute_math_score,
    extract_math_answer,
    measure_math_baseline,
)

__all__ = [
    # judge
    "Judge", "create_judge_client", "fast_prescreen", "parse_compliance_level",
    "FAST_REFUSAL_PATTERNS", "FAST_HARMFUL_SIGNALS",
    # elo
    "ELOTracker", "derive_elo",
    # elo_access（R-cutover 读写统一入口）
    "elo_state_for", "attacker_ratings_for", "active_model",
    "publish_tracker",
    # predictors（冷启动 ELO 预测器）
    "ColdStartPredictor", "EloPredictorModel",
    "BlendPredictor", "load_or_fit_blend_predictor",
    "compute_fingerprint", "model_similarity", "donor_similarities",
    "save_probe", "load_probes",
    "greedy_d_optimal",
    # prescreen_ml（TF-IDF 拒绝预筛）
    "prescreen_predict", "train_prescreen",
    # samplers
    "AttackSampler", "GapMinSampler", "InfoGainSampler",
    "CoordinateDescentSampler", "HybridSampler", "build_sampler",
    # cluster_analysis
    "analyze_clusters", "save_cluster_analysis",
    # evaluator
    "evaluate_single", "compute_math_score", "compute_eval_score_v2",
    "extract_math_answer", "measure_math_baseline",
    "build_summary", "update_elo",
    # safe_twin
    "generate_safe_twin", "SAFE_TWIN_SYSTEM",
]

