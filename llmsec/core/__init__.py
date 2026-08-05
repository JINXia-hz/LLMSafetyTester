"""
llmsec.core — 基础设施层

常用符号再导出，方便 `from llmsec.core import load_env, OUTPUT_DIR, ...`。
"""

from llmsec.core.config import (
    ALLERGY_REPORT_FILE,
    ATTACK_SET_L1_FILE,
    ATTACKS_DIR,
    CLUSTER_FEATURES_FILE,
    CLUSTER_MATRIX_FILE,
    CLUSTER_REPORT_FILE,
    CLUSTER_RESULT_FILE,
    CLUSTER_SECURITY_ANALYSIS_FILE,
    DATA_DIR,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    ELO_CACHE_FILE,
    FEATURE_CACHE_FILE,
    METHOD_REGISTRY_FILE,
    OUTPUT_DIR,
    PREDICTORS_DIR,
    PROJECT_ROOT,
    REPORT_FILE,
    RESULTS_FILE,
    RUNS_DIR,
    SAFE_TWINS_FILE,
    STATE_DIR,
    TASK_LOG_DIR,
    TREE_FILE,
    TWIN_RESULT_FILE,
    GeneratorConfig,
    JudgeConfig,
    TargetConfig,
    load_env,
    load_targets,
)
from llmsec.core.io import (
    CorruptedFileError,
    append_jsonl,
    iter_jsonl,
    load_artifact,
    load_done_ids,
    read_csv,
    read_json,
    read_jsonl,
    save_artifact,
    write_csv,
    write_json,
    write_jsonl,
)
from llmsec.core.llm import chat_with_retry, create_openai_client, retry_call
from llmsec.core.logging import get_logger, setup_console
from llmsec.core.results import MatchResult, ResultsMatrix
from llmsec.core.seed import get_global_seed, set_global_seed
from llmsec.core.text import (
    estimate_tokens,
    gen_math,
    inject_math_tax,
    strip_math_tax,
)

__all__ = [
    # config
    "PROJECT_ROOT", "OUTPUT_DIR", "STATE_DIR", "ATTACKS_DIR", "RUNS_DIR", "DATA_DIR",
    "SAFE_TWINS_FILE", "ATTACK_SET_L1_FILE",
    "RESULTS_FILE", "PREDICTORS_DIR", "ELO_CACHE_FILE",
    "CLUSTER_REPORT_FILE", "CLUSTER_MATRIX_FILE", "CLUSTER_FEATURES_FILE",
    "FEATURE_CACHE_FILE", "CLUSTER_RESULT_FILE",
    "TREE_FILE", "REPORT_FILE", "METHOD_REGISTRY_FILE",
    "CLUSTER_SECURITY_ANALYSIS_FILE",
    "ALLERGY_REPORT_FILE", "TWIN_RESULT_FILE", "TASK_LOG_DIR",
    "DEFAULT_BASE_URL", "DEFAULT_MODEL",
    "load_env", "load_targets",
    "TargetConfig", "GeneratorConfig", "JudgeConfig",
    # io
    "read_jsonl", "iter_jsonl", "write_jsonl", "append_jsonl", "load_done_ids",
    "read_json", "write_json", "load_artifact", "save_artifact", "read_csv", "write_csv",
    "CorruptedFileError",
    # results
    "ResultsMatrix", "MatchResult",
    # llm
    "create_openai_client", "chat_with_retry", "retry_call",
    # logging
    "setup_console", "get_logger",
    # seed
    "get_global_seed", "set_global_seed",
    # text
    "strip_math_tax", "estimate_tokens", "gen_math", "inject_math_tax",
]
