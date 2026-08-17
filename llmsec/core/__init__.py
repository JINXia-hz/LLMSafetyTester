"""
llmsec.core — 基础设施层

常用符号再导出，方便 `from llmsec.core import load_env, ...`。

路径常量只 re-export **永不重绑**的静态锚点（PROJECT_ROOT/ATTACKS_DIR 等）——
会被 work-dir 隔离重绑的常量（RESULTS_*/CATALOG_DB/CLUSTER_*/STATE_DIR 等）
不提供包层 re-export：冻结导入会静默绕过隔离（tests/test_audit_r9_guard 的
AST 守卫口径），消费方一律 `import llmsec.core.config as _config` 调期读。
"""

from llmsec.core.config import (
    ATTACK_SET_L1_FILE,
    ATTACKS_DIR,
    DATA_DIR,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    OUTPUT_DIR,
    PROJECT_ROOT,
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
    read_json,
    read_jsonl,
    save_artifact,
    write_csv,
    write_json,
    write_jsonl,
)
from llmsec.core.llm import (
    chat_with_retry,
    create_openai_client,
    extract_message_text,
    retry_call,
)
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
    # config（仅静态锚点；重绑常量见模块 docstring）
    "PROJECT_ROOT", "OUTPUT_DIR", "ATTACKS_DIR", "DATA_DIR", "ATTACK_SET_L1_FILE",
    "DEFAULT_BASE_URL", "DEFAULT_MODEL",
    "load_env", "load_targets",
    "TargetConfig", "GeneratorConfig", "JudgeConfig",
    # io
    "read_jsonl", "iter_jsonl", "write_jsonl", "append_jsonl", "load_done_ids",
    "read_json", "write_json", "load_artifact", "save_artifact", "write_csv",
    "CorruptedFileError",
    # results
    "ResultsMatrix", "MatchResult",
    # llm
    "create_openai_client", "chat_with_retry", "retry_call", "extract_message_text",
    # logging
    "setup_console", "get_logger",
    # seed
    "get_global_seed", "set_global_seed",
    # text
    "strip_math_tax", "estimate_tokens", "gen_math", "inject_math_tax",
]
