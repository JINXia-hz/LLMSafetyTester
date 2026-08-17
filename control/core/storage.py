"""control.core.storage — 存储层薄契约（llmsec.storage.contract 的 re-export）。

权威实现见 ``llmsec/storage/``（2026-08 数据库重构：DAO 收口 + SQLModel，
SQL/ORM 只存在于 storage 包内）。本模块使 control 内部代码从
``control.core.storage`` 导入而保持"control 不 import llmsec 内部 API"的
边界——经本契约模块是唯一许可路径（仿 ``control/core/paths.py`` 先例）。

用法规约：
  - run 发现/解析一律走 ``query_runs`` / ``get_run``（先 ``reconcile_runs``
    对账，目录库是可重建的派生索引）；
  - runner_report 指标提取一律走 ``extract_report_metrics``（字段单一来源）；
  - 不要在 control 内 import sqlite3/sqlalchemy/sqlmodel（AST 守卫拦截）。
"""

from __future__ import annotations

from llmsec.storage.contract import (
    RUN_ARTIFACTS,
    RUN_NAME_RE,
    Run,
    Task,
    Trial,
    append_event,
    append_workspace_gc,
    backup_results,
    clear_ticket,
    clear_tickets_for_plan,
    clone_from_run,
    close_results_db,
    delete_env_snapshot,
    delete_workspace_row,
    enqueue_plan,
    extract_report_metrics,
    finish_queue_item,
    gazette_events,
    gazette_meta,
    get_ctl_plan,
    get_env_snapshot,
    get_run,
    get_ticket,
    get_workspace,
    list_ctl_plans,
    list_env_snapshots,
    list_gazette_meta,
    list_workspaces,
    mark_queue_running,
    query_runs,
    query_tasks,
    query_trials,
    reconcile_runs,
    register_run,
    reset_ctl_plans,
    reset_gazette,
    reset_queue,
    reset_tickets,
    results_stats,
    save_ctl_plan,
    save_env_snapshot,
    save_ticket,
    save_workspace,
    search_gazettes,
    workspace_gc_log,
)

__all__ = [
    "RUN_ARTIFACTS", "RUN_NAME_RE", "Run", "Task", "Trial",
    "backup_results", "clone_from_run", "close_results_db", "results_stats",
    "search_gazettes",
    "append_event", "gazette_events", "gazette_meta", "list_gazette_meta", "reset_gazette",
    "save_ctl_plan", "get_ctl_plan", "list_ctl_plans", "reset_ctl_plans",
    "clear_ticket", "clear_tickets_for_plan", "get_ticket", "reset_tickets", "save_ticket",
    "enqueue_plan", "mark_queue_running", "finish_queue_item", "reset_queue",
    "save_workspace", "get_workspace", "list_workspaces", "delete_workspace_row",
    "append_workspace_gc", "workspace_gc_log",
    "save_env_snapshot", "list_env_snapshots", "get_env_snapshot", "delete_env_snapshot",
    "extract_report_metrics", "get_run", "query_runs", "query_tasks",
    "query_trials", "reconcile_runs", "register_run",
]
