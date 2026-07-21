"""
llmsec.evaluation — 评估子包

由原根目录 judge.py / elo.py / evaluate.py / safe_twin.py 平移而来：
  - judge.py      LLM-as-Judge 多层次评分 + 快速预筛
  - elo.py        双轨 ELO 追踪（攻击方法 × 防御模型）
  - evaluator.py  evaluate_single 唯一评估核心 + 汇总统计 + CLI
  - safe_twin.py  安全孪生生成与过敏（FPR）检测

常用符号再导出，供 runner 等模块按 `from llmsec.evaluation import ...` 使用。
"""

from llmsec.evaluation.elo import ELOTracker, update_elo_from_results
from llmsec.evaluation.evaluator import (
    compute_eval_score_legacy,
    compute_eval_score_v2,
    compute_math_score,
    detect_harmful_legacy,
    detect_refusal_legacy,
    evaluate_single,
    extract_math_answer,
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
    "ELOTracker", "update_elo_from_results",
    # evaluator
    "evaluate_single", "compute_math_score", "compute_eval_score_v2",
    "compute_eval_score_legacy", "extract_math_answer",
    "detect_refusal_legacy", "detect_harmful_legacy",
    # safe_twin
    "generate_safe_twin", "SAFE_TWIN_SYSTEM",
]
