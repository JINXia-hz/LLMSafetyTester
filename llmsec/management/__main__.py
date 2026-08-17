"""management — CLI 入口

用法：
  python -m llmsec.management runs list [--json] [--filter ...] [--junk-only]
  python -m llmsec.management runs delete <run...> [--delete-r] [--yes] [--json]
  python -m llmsec.management cache list [--json]
  python -m llmsec.management cache clean <cat...> [--yes] [--json]
  python -m llmsec.management cache prune [--max N] [--yes] [--json]
  python -m llmsec.management storage reindex|verify|gc-tasks|trials|migrate-layouts
  python -m llmsec.management snapshot export [--source global|run:<name>] [--out PATH] [--json]

机器友好契约：
  - 所有 list/export 支持 --json 结构化输出（供控制层/agent 解析）
  - 所有写操作默认 dry-run，--yes 才执行
  - 删除走软删除（output/.trash/），可恢复
"""

from __future__ import annotations

import argparse
import sys

from llmsec.core.logging import get_logger

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m llmsec.management",
        description="llmsec 信息管理：过滤/清理历史、清缓存、导出快照",
    )
    sub = parser.add_subparsers(dest="group", required=True)

    # ---- runs ----
    runs = sub.add_parser("runs", help="run 历史管理")
    runs_sub = runs.add_subparsers(dest="cmd", required=True)

    p_list = runs_sub.add_parser("list", help="列出/过滤 run")
    p_list.add_argument("--json", action="store_true", help="结构化 JSON 输出")
    p_list.add_argument("--target", default=None, help="按目标名过滤")
    p_list.add_argument("--since", default=None, help="起始日期 YYYY-MM-DD（含）")
    p_list.add_argument("--until", default=None, help="截止日期 YYYY-MM-DD（含）")
    p_list.add_argument("--level", default=None, help="按 security_level 精确过滤")
    p_list.add_argument("--no-report", action="store_true", help="仅列无报告的（失败/垃圾）")
    p_list.add_argument("--min-size", type=int, default=None, help="最小字节数过滤")
    p_list.add_argument("--junk-only", action="store_true", help="仅列垃圾 run（无报告）")

    p_del = runs_sub.add_parser("delete", help="删除 run（软删除，可恢复）")
    p_del.add_argument("names", nargs="+", help="run 名（ts/target 或 ts）")
    p_del.add_argument("--delete-r", action="store_true", help="同时从 R 矩阵删该 model 列")
    p_del.add_argument("--yes", action="store_true", help="确认执行（默认 dry-run）")
    p_del.add_argument("--json", action="store_true", help="结构化 JSON 输出")

    # ---- cache ----
    cache = sub.add_parser("cache", help="缓存管理")
    cache_sub = cache.add_subparsers(dest="cmd", required=True)

    p_clist = cache_sub.add_parser("list", help="列出各类缓存占用")
    p_clist.add_argument("--json", action="store_true", help="结构化 JSON 输出")

    p_clean = cache_sub.add_parser("clean", help="清理缓存（软删除）")
    p_clean.add_argument("categories", nargs="+",
                         help="类别: elo_cache predictors predictors_legacy feature_cluster task_logs")
    p_clean.add_argument("--yes", action="store_true", help="确认执行（默认 dry-run）")
    p_clean.add_argument("--json", action="store_true", help="结构化 JSON 输出")

    p_prune = cache_sub.add_parser("prune", help="predictors LRU 修剪（按最近命中保留最新 N 个）")
    p_prune.add_argument("--max", type=int, default=200, help="保留的最新缓存数（默认 200）")
    p_prune.add_argument("--yes", action="store_true", help="确认执行（默认 dry-run）")
    p_prune.add_argument("--json", action="store_true", help="结构化 JSON 输出")

    # ---- storage ----（目录库管理：2026-08 数据库重构新增）
    stg = sub.add_parser("storage", help="目录库管理（runs/trials/tasks 索引）")
    stg_sub = stg.add_subparsers(dest="cmd", required=True)

    p_reindex = stg_sub.add_parser("reindex", help="全量重建目录库索引")
    p_reindex.add_argument("--no-empty", action="store_true", help="跳过零产物空壳目录")
    p_reindex.add_argument("--json", action="store_true", help="结构化 JSON 输出")

    p_verify = stg_sub.add_parser("verify", help="完整性校验（库行 ↔ 目录树双向对账）")
    p_verify.add_argument("--json", action="store_true", help="结构化 JSON 输出")

    p_gc = stg_sub.add_parser("gc-tasks", help="清理终态任务文件（软删 + 删库行）")
    p_gc.add_argument("--older-than", type=float, default=14.0, help="终态超过 N 天才清理（默认 14）")
    p_gc.add_argument("--yes", action="store_true", help="确认执行（默认 dry-run）")
    p_gc.add_argument("--json", action="store_true", help="结构化 JSON 输出")

    p_trials = stg_sub.add_parser("trials", help="列出 trials 登记行")
    p_trials.add_argument("study", nargs="?", default=None, help="按 study 名过滤")
    p_trials.add_argument("--json", action="store_true", help="结构化 JSON 输出")

    p_mig = stg_sub.add_parser("migrate-layouts", help="Gen1/Gen2 扁平 run 归一为 Gen3 <ts>/<target>/ 布局")
    p_mig.add_argument("--yes", action="store_true", help="确认执行（默认 dry-run）")
    p_mig.add_argument("--json", action="store_true", help="结构化 JSON 输出")

    p_mc = stg_sub.add_parser("migrate-control", help="control 层旧文件（gazette/plans/三索引）一次性导入目录库")
    p_mc.add_argument("--yes", action="store_true", help="确认执行（默认 dry-run）")
    p_mc.add_argument("--json", action="store_true", help="结构化 JSON 输出")

    p_bk = stg_sub.add_parser("backup-r", help="备份 R 库（sqlite3 backup API，WAL 安全）")
    p_bk.add_argument("out", nargs="?", default=None, help="备份目标路径（默认 output/state/results.backup.<ts>.db）")
    p_bk.add_argument("--json", action="store_true", help="结构化 JSON 输出")

    # ---- snapshot ----
    snap = sub.add_parser("snapshot", help="快照导出")
    snap_sub = snap.add_subparsers(dest="cmd", required=True)

    p_exp = snap_sub.add_parser("export", help="导出快照（控制层 fork 用）")
    p_exp.add_argument("--source", default="global",
                       help="来源: global 或 run:<name>（从 state.json 重建）")
    p_exp.add_argument("--out", default=None, help="输出目录或 .tar.gz 文件")
    p_exp.add_argument("--json", action="store_true", help="结构化 JSON 输出")

    # ---- merge ----
    p_merge = sub.add_parser("merge", help="合并 R 矩阵（显式统一动作）")
    p_merge.add_argument("--sources", nargs="+", required=True,
                         help="源列表：global / ws:<name> / 目录路径（一个或多个）")
    p_merge.add_argument("--target", required=True,
                         help="目标：global 或 ws:<name>")
    p_merge.add_argument("--models", nargs="*", default=None,
                         help="只合并指定 model 列（默认全部）")
    p_merge.add_argument("--yes", action="store_true", help="确认执行（默认 dry-run）")
    p_merge.add_argument("--json", action="store_true", help="结构化 JSON 输出")

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.group == "runs":
        from llmsec.management import runs as runs_mod
        if args.cmd == "list":
            return runs_mod.cmd_list(
                target=args.target, since=args.since, until=args.until,
                level=args.level,
                has_report=False if args.no_report else None,
                min_size=args.min_size,
                junk_only=args.junk_only,
                json_mode=args.json,
            )
        if args.cmd == "delete":
            return runs_mod.cmd_delete(
                args.names, delete_r=args.delete_r, yes=args.yes, json_mode=args.json,
            )

    if args.group == "cache":
        from llmsec.management import caches
        if args.cmd == "list":
            return caches.cmd_list(json_mode=args.json)
        if args.cmd == "clean":
            return caches.cmd_clean(args.categories, yes=args.yes, json_mode=args.json)
        if args.cmd == "prune":
            return caches.cmd_prune(args.max, yes=args.yes, json_mode=args.json)

    if args.group == "storage":
        from llmsec.management import storage as storage_mod
        if args.cmd == "reindex":
            return storage_mod.cmd_reindex(
                include_empty=not args.no_empty, json_mode=args.json)
        if args.cmd == "verify":
            return storage_mod.cmd_verify(json_mode=args.json)
        if args.cmd == "gc-tasks":
            return storage_mod.cmd_gc_tasks(
                args.older_than, yes=args.yes, json_mode=args.json)
        if args.cmd == "trials":
            return storage_mod.cmd_trials(args.study, json_mode=args.json)
        if args.cmd == "migrate-layouts":
            return storage_mod.cmd_migrate_layouts(yes=args.yes, json_mode=args.json)
        if args.cmd == "migrate-control":
            return storage_mod.cmd_migrate_control(yes=args.yes, json_mode=args.json)
        if args.cmd == "backup-r":
            return storage_mod.cmd_backup_r(args.out, json_mode=args.json)

    if args.group == "snapshot":
        from llmsec.management import snapshot
        if args.cmd == "export":
            return snapshot.cmd_export(
                source=args.source, out=args.out, json_mode=args.json,
            )

    if args.group == "merge":
        from llmsec.management import merge
        return merge.cmd_merge(
            args.sources, args.target, models=args.models,
            yes=args.yes, json_mode=args.json,
        )

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
