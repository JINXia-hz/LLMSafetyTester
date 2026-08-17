"""control.core.storage — 存储层薄契约（llmsec.storage.contract 的 re-export）。

权威实现见 ``llmsec/storage/``（2026-08 数据库重构：DAO 收口 + SQLModel，
SQL/ORM 只存在于 storage 包内）。本模块使 control 内部代码从
``control.core.storage`` 导入而保持"control 不 import llmsec 内部 API"的
边界——经本契约模块是唯一许可路径（仿 ``control/core/paths.py`` 先例）。

用法规约：
  - run 发现/解析一律走 ``query_runs`` / ``get_run``（先 ``reconcile_runs``
    对账，目录库是可重建的派生索引）；
  - runner_report 指标提取一律走 ``extract_report_metrics``（control 原私有
    ``extract_elo_fields`` 的超集，字段单一来源）；
  - 不要在 control 内 import sqlite3/sqlalchemy/sqlmodel（AST 守卫拦截）。
"""

from __future__ import annotations

from llmsec.storage.contract import (
    RUN_ARTIFACTS,
    RUN_NAME_RE,
    Run,
    Task,
    Trial,
    extract_report_metrics,
    get_run,
    query_runs,
    query_tasks,
    query_trials,
    reconcile_runs,
    register_run,
    remove_run,
)

__all__ = [
    "RUN_ARTIFACTS", "RUN_NAME_RE", "Run", "Task", "Trial",
    "extract_report_metrics", "get_run", "query_runs", "query_tasks",
    "query_trials", "reconcile_runs", "register_run", "remove_run",
]
