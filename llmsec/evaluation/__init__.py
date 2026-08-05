"""
llmsec.evaluation — 评估子包

由原根目录 judge.py / elo.py / evaluate.py / safe_twin.py 平移而来：
  - judge.py      LLM-as-Judge 多层次评分 + 快速预筛
  - elo.py        双轨 ELO 追踪（攻击方法 × 防御模型）
  - evaluator.py  evaluate_single 唯一评估核心 + 汇总统计 + CLI
  - safe_twin.py  安全孪生生成与过敏（FPR）检测

常用符号再导出，供 runner 等模块按 `from llmsec.evaluation import ...` 使用。
"""

from llmsec.evaluation.active_learning import greedy_d_optimal
from llmsec.evaluation.elo import ELOTracker, derive_elo
from llmsec.evaluation.elo_cluster import ClusterEloPredictor, EloPredictorModel
from llmsec.evaluation.elo_access import (
    active_model,
    attacker_ratings_for,
    elo_state_for,
    publish_tracker,
)
from llmsec.evaluation.samplers import (
    AttackSampler,
    CoordinateDescentSampler,
    GapMinSampler,
    HybridSampler,
    InfoGainSampler,
    build_sampler,
)
from llmsec.evaluation.cluster_analysis import analyze_clusters, save_cluster_analysis
from llmsec.evaluation.evaluator import (
    compute_eval_score_v2,
    compute_math_score,
    evaluate_single,
    extract_math_answer,
    measure_math_baseline,
)
from llmsec.evaluation.judge import (
    FAST_HARMFUL_SIGNALS,
    FAST_REFUSAL_PATTERNS,
    Judge,
    create_judge_client,
    fast_prescreen,
)
from llmsec.evaluation.safe_twin import SAFE_TWIN_SYSTEM, generate_safe_twin

__all__ = [
    # judge
    "Judge", "create_judge_client", "fast_prescreen",
    "FAST_REFUSAL_PATTERNS", "FAST_HARMFUL_SIGNALS",
    # elo
    "ELOTracker", "derive_elo",
    # elo_access（R-cutover 读写统一入口）
    "elo_state_for", "attacker_ratings_for", "active_model",
    "publish_tracker",
    # elo_cluster
    "ClusterEloPredictor", "EloPredictorModel",
    # active_learning
    "greedy_d_optimal",
    # samplers
    "AttackSampler", "GapMinSampler", "InfoGainSampler",
    "CoordinateDescentSampler", "HybridSampler", "build_sampler",
    # cluster_analysis
    "analyze_clusters", "save_cluster_analysis",
    # evaluator
    "evaluate_single", "compute_math_score", "compute_eval_score_v2",
    "extract_math_answer", "measure_math_baseline",
    # safe_twin
    "generate_safe_twin", "SAFE_TWIN_SYSTEM",
]
