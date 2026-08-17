"""storage.contract — 存储层对外的唯一公开契约。

service 层（management / server / experiments / tui / mcp）与 control 层
（经 ``control/core/storage.py`` 薄 re-export，仿 control/core/paths.py 先例）
只允许 import 本模块——这是"SQL/ORM 只存在于 llmsec/storage/ 内"边界的
出口。AST 守卫（tests/test_audit_r9_guard.py）负责把边界钉在机器上：

  1. storage 包外禁止 import sqlite3 / sqlalchemy / sqlmodel；
  2. 路径常量（含 CATALOG_DB）禁止顶层冻结导入（work-dir 隔离兼容）。

新增对外能力时：先在 catalog / rstore 内实现，再在此登记导出——
不经过本模块的存储消费都视为违例。
"""

from __future__ import annotations

from llmsec.core.results import extract_report_metrics
from llmsec.storage.catalog import (
    RUN_ARTIFACTS,
    RUN_NAME_RE,
    RUN_TS_FORMAT,
    allocate_runs_dir,
    get_run,
    get_task,
    query_runs,
    query_tasks,
    query_trials,
    rebuild_runs,
    reconcile_runs,
    reconcile_tasks,
    register_run,
    register_task,
    register_trial,
    remove_run,
    update_task,
    update_trial,
    upsert_task,
)
from llmsec.storage.db import catalog_db, db_for
from llmsec.storage.models import Run, Task, Trial

__all__ = [
    "RUN_ARTIFACTS", "RUN_NAME_RE", "RUN_TS_FORMAT",
    "Run", "Task", "Trial",
    "allocate_runs_dir", "catalog_db", "db_for", "extract_report_metrics",
    "get_run", "get_task", "query_runs", "query_tasks", "query_trials",
    "reconcile_runs", "reconcile_tasks",
    "register_run", "register_task", "register_trial",
    "remove_run", "rebuild_runs", "update_task", "update_trial", "upsert_task",
]
