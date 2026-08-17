"""storage.contract — 存储层对外的唯一公开契约。

service 层（management / server / experiments / tui / mcp）与 control 层
（经 ``control/core/storage.py`` 薄 re-export，仿 control/core/paths.py 先例）
只允许 import 本模块——这是"SQL/ORM 只存在于 llmsec/storage/ 内"边界的
出口。AST 守卫（tests/test_audit_r9_guard.py）负责把边界钉在机器上：

  1. storage 包外禁止 import sqlite3 / sqlalchemy / sqlmodel；
  2. 路径常量（含 CATALOG_DB）禁止顶层冻结导入（work-dir 隔离兼容）。

新增对外能力时：先在 catalog / rstore / ctlstore 内实现，再在此登记导出——
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
    remove_run,
    update_task,
    upsert_task,
)
from llmsec.storage.ctlstore import (
    append_event,
    append_workspace_gc,
    clear_ticket,
    clear_tickets_for_plan,
    delete_env_snapshot,
    delete_workspace_row,
    enqueue_plan,
    finish_queue_item,
    gazette_events,
    gazette_meta,
    get_ctl_plan,
    get_env_snapshot,
    get_ticket,
    get_workspace,
    list_ctl_plans,
    list_env_snapshots,
    list_gazette_meta,
    list_workspaces,
    mark_queue_running,
    queued_plans,
    reset_ctl_plans,
    reset_gazette,
    reset_queue,
    reset_tickets,
    save_ctl_plan,
    save_env_snapshot,
    save_ticket,
    save_workspace,
    workspace_gc_log,
)
from llmsec.storage.ctlstore import search_gazette as search_gazettes
from llmsec.storage.db import catalog_db, db_for
from llmsec.storage.models import Run, Task, Trial
from llmsec.storage.rstore import backup as backup_results
from llmsec.storage.rstore import clone_from_run, results_stats
from llmsec.storage.rstore import close_db as close_results_db

__all__ = [
    "RUN_ARTIFACTS", "RUN_NAME_RE", "RUN_TS_FORMAT",
    "Run", "Task", "Trial",
    "allocate_runs_dir", "catalog_db", "db_for", "extract_report_metrics",
    "backup_results", "clone_from_run", "close_results_db", "results_stats",
    "search_gazettes",
    "append_event", "gazette_events", "gazette_meta", "list_gazette_meta", "reset_gazette",
    "save_ctl_plan", "get_ctl_plan", "list_ctl_plans", "reset_ctl_plans",
    "clear_ticket", "clear_tickets_for_plan", "get_ticket", "reset_tickets", "save_ticket",
    "enqueue_plan", "queued_plans", "mark_queue_running", "finish_queue_item", "reset_queue",
    "save_workspace", "get_workspace", "list_workspaces", "delete_workspace_row",
    "append_workspace_gc", "workspace_gc_log",
    "save_env_snapshot", "list_env_snapshots", "get_env_snapshot", "delete_env_snapshot",
    "get_run", "get_task", "query_runs", "query_tasks", "query_trials",
    "reconcile_runs", "reconcile_tasks",
    "register_run", "register_task",
    "remove_run", "rebuild_runs", "update_task", "upsert_task",
]
