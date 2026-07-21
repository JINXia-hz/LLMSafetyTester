"""
llmsec.reporting — 报告子包

由原根目录 report.py 平移而来：
  - report.py  五维树形安全画像 + LLM 叙事报告 + 方法注册表

常用符号再导出，供 runner 等模块按 `from llmsec.reporting import ...` 使用。
"""

from llmsec.reporting.report import (
    build_method_registry,
    build_method_stats,
    build_tree,
    generate_fallback_report,
    generate_narrative,
    load_all_results,
    load_allergy,
    load_elo,
    load_prompt_metadata,
    main,
    parse_args,
)

__all__ = [
    "load_all_results", "load_elo", "load_allergy", "load_prompt_metadata",
    "build_method_stats", "build_method_registry", "build_tree",
    "generate_narrative", "generate_fallback_report",
    "parse_args", "main",
]
