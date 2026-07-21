"""
llmsec.core — 基础设施层

常用符号再导出，方便 `from llmsec.core import load_env, OUTPUT_DIR, ...`。
"""

from llmsec.core.config import (
    ATTACK_SET_L1_FILE,
    ATTACKS_DIR,
    CLUSTER_FEATURES_FILE,
    CLUSTER_MATRIX_FILE,
    CLUSTER_REPORT_FILE,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    ELO_FILE,
    LEGACY_ATTACK_SET_L1_FILE,
    LEGACY_ELO_FILE,
    LEGACY_SAFE_TWINS_FILE,
    OUTPUT_DIR,
    PROJECT_ROOT,
    RUNS_DIR,
    SAFE_TWINS_FILE,
    STATE_DIR,
    GeneratorConfig,
    JudgeConfig,
    TargetConfig,
    load_env,
    resolve_existing,
)
from llmsec.core.io import (
    append_jsonl,
    iter_jsonl,
    load_done_ids,
    read_jsonl,
    write_jsonl,
)
from llmsec.core.llm import chat_with_retry, create_openai_client
from llmsec.core.logging import get_logger, setup_console
from llmsec.core.text import estimate_tokens, strip_math_tax

__all__ = [
    # config
    "PROJECT_ROOT", "OUTPUT_DIR", "STATE_DIR", "ATTACKS_DIR", "RUNS_DIR",
    "ELO_FILE", "SAFE_TWINS_FILE", "ATTACK_SET_L1_FILE",
    "CLUSTER_REPORT_FILE", "CLUSTER_MATRIX_FILE", "CLUSTER_FEATURES_FILE",
    "LEGACY_ELO_FILE", "LEGACY_SAFE_TWINS_FILE", "LEGACY_ATTACK_SET_L1_FILE",
    "DEFAULT_BASE_URL", "DEFAULT_MODEL",
    "load_env", "resolve_existing",
    "TargetConfig", "GeneratorConfig", "JudgeConfig",
    # io
    "read_jsonl", "iter_jsonl", "write_jsonl", "append_jsonl", "load_done_ids",
    # llm
    "create_openai_client", "chat_with_retry",
    # logging
    "setup_console", "get_logger",
    # text
    "strip_math_tax", "estimate_tokens",
]
