"""management.runs — 过滤 / 列出 / 清理 run 历史。

storage 重构后目录扫描收敛为 storage.catalog 单一实现，本模块保留
management 口径（含失败 run）与多维过滤/dry-run 删除能力：

  - list：目录占用（size）随目录库登记行缓存，支持多维过滤。
  - delete：默认 dry-run 预览；软删除到 output/.trash/；可选 --delete-r 删 R 列
    （阶段 2：rstore.remove_models 单事务 SQL 删除）。
"""

from __future__ import annotations

from pathlib import Path

from llmsec.core.config import RESULTS_DB, RUNS_DIR
from llmsec.core.io import read_json
from llmsec.core.logging import get_logger
from llmsec.core.paths import safe_subpath
from llmsec.core.results import ResultsMatrix
from llmsec.management.common import (
    Plan,
    dir_size,
    emit,
    fmt_size,
    print_table,
    soft_rmtree,
)

logger = get_logger(__name__)


# ============================================================
# 扫描（目录库单一实现，本层只做口径适配）
# ============================================================
def discover_runs(runs_dir: Path | None = None) -> list[dict]:
    """列出 run（时间倒序）。

    management 口径 = 有任意 RUN_ARTIFACTS 产物的目录（含无报告的失败 run，
    供 detect_junk 识别）。字段与 data_query 口径兼容并额外含 ``size``——
    两者都来自目录库登记行的 as_dict()（字段超集）。
    """
    from llmsec.storage import contract as _storage

    return [r.as_dict() for r in _storage.query_runs(runs_root=runs_dir, has_artifact=True)]


def run_dir_for(name: str, runs_dir: Path | None = None) -> Path | None:
    """run 名 → 目录路径。name 可以是 'ts/target' 或 'ts'。

    name 外部可控，走 safe_subpath 逐段校验防穿越；非法名称视为目录不存在。
    """
    runs_dir = runs_dir or RUNS_DIR
    try:
        d = safe_subpath(runs_dir, *name.split("/"))
    except ValueError:
        return None
    return d if d.is_dir() else None


# ============================================================
# 过滤
# ============================================================
def filter_runs(
    runs: list[dict],
    *,
    target: str | None = None,
    since: str | None = None,        # ISO 日期 YYYY-MM-DD（含）
    until: str | None = None,
    level: str | None = None,        # security_level 精确匹配
    has_report: bool | None = None,
    min_size: int | None = None,     # 字节
) -> list[dict]:
    """按多维度过滤 run 列表。各维度 AND 组合，None 表示不过滤。"""
    out = []
    for r in runs:
        if target and r["target"] != target and r["target_model"] != target:
            continue
        # batch 名前缀即日期
        batch_date = r["batch"][:10]
        if since and batch_date < since:
            continue
        if until and batch_date > until:
            continue
        if level and r.get("security_level") != level:
            continue
        if has_report is not None and r["has_report"] != has_report:
            continue
        if min_size is not None and r["size"] < min_size:
            continue
        out.append(r)
    return out


def detect_junk(runs: list[dict]) -> list[dict]:
    """识别「垃圾」run：无报告 / 失败（无 has_report 的 run 目录）。

    返回子集，供「快捷清理」用。
    """
    return [r for r in runs if not r["has_report"]]


# ============================================================
# list 子命令
# ============================================================
def cmd_list(
    *,
    target: str | None = None,
    since: str | None = None,
    until: str | None = None,
    level: str | None = None,
    has_report: bool | None = None,
    min_size: int | None = None,
    junk_only: bool = False,
    json_mode: bool = False,
) -> int:
    runs = discover_runs()
    if junk_only:
        runs = detect_junk(runs)
    else:
        runs = filter_runs(
            runs, target=target, since=since, until=until,
            level=level, has_report=has_report, min_size=min_size,
        )
    if json_mode:
        emit({"runs": runs, "count": len(runs)}, json_mode=True, title="runs")
    else:
        if not runs:
            print("（无 run 匹配）")
            return 0
        rows = []
        for r in runs:
            asr = r.get("asr")
            asr_s = f"{asr:.1%}" if isinstance(asr, (int, float)) else "-"
            elo = r.get("boundary_elo")
            elo_s = f"{elo:.0f}" if isinstance(elo, (int, float)) else "-"
            rows.append([
                r["name"],
                r["target_model"],
                (r.get("security_level") or "inconclusive")[:8],
                asr_s,
                elo_s,
                fmt_size(r["size"]),
                r["mtime"][:19],
            ])
        print_table(
            rows,
            headers=["run", "target", "level", "asr", "elo", "size", "mtime"],
        )
        print(f"\n共 {len(runs)} 个 run，占用 {fmt_size(sum(r['size'] for r in runs))}")
    return 0


# ============================================================
# delete 子命令
# ============================================================
def plan_delete(
    names: list[str],
    *,
    delete_r: bool = False,
) -> Plan:
    """构造删除预览（dry-run）。不执行任何写操作。"""
    plan = Plan(action="delete", dry_run=True)
    r_models_affected: set[str] = set()
    r_rows_total = 0
    for name in names:
        run_dir = run_dir_for(name)
        if run_dir is None:
            # 名称非法或目录不存在——用 try 拿到（校验后的）路径仅用于展示，
            # 该项 kind=missing 不会被 execute_delete 删除
            try:
                disp = safe_subpath(RUNS_DIR, *name.split("/"))
            except ValueError:
                disp = RUNS_DIR / "<invalid>"
            plan.add(disp, size=0, kind="missing", detail="目录不存在，跳过")
            continue
        size = dir_size(run_dir)
        plan.add(run_dir, size=size, kind="run_dir", detail="将软删除到 .trash/")
        if delete_r:
            # 从 runner_report.json 取 target_model，定位 R 列
            report = read_json(run_dir / "runner_report.json") or {}
            model = report.get("target_model")
            if model:
                try:
                    R = ResultsMatrix.load()
                    col = R.model_column(model)
                    n = len(col)
                    if n:
                        r_models_affected.add(model)
                        r_rows_total += n
                        plan.extra.setdefault("r_models", []).append(model)
                        plan.add(
                            RESULTS_DB, size=0, kind="r_column",
                            detail=f"将从 R 删除 model={model} 的 {n} 条观测",
                        )
                except Exception as e:
                    plan.add(RESULTS_DB, size=0, kind="r_error", detail=f"读 R 失败: {e}")
    plan.extra["r_models_affected"] = sorted(r_models_affected)
    plan.extra["r_rows_total"] = r_rows_total
    return plan


def execute_delete(plan: Plan, *, delete_r: bool = False) -> Plan:
    """执行删除（已确认）。返回执行后的 plan（dry_run=False）。"""
    from llmsec.core.config import OUTPUT_DIR

    done = Plan(action="delete", dry_run=False, extra=dict(plan.extra))
    for item in plan.items:
        if item.kind != "run_dir":
            continue
        # plan 存的是相对 OUTPUT_DIR 的路径，还原绝对路径
        src = OUTPUT_DIR / item.path
        if src.exists():
            dest = soft_rmtree(src)
            done.add(src, size=item.size, kind="run_dir",
                     detail=f"已移到 {dest}" if dest else "失败")
        else:
            done.add(src, size=0, kind="missing", detail="已不存在")
    # 删 R 列（阶段 2：单事务 SQL 删除，取代"文件锁 load→remove→save"——
    # _file_lock 与 LockTimeout 从 R 路径退役）
    if delete_r and plan.extra.get("r_models_affected"):
        try:
            from llmsec.storage import rstore
            total = rstore.remove_models(list(plan.extra["r_models_affected"]))
            done.extra["r_rows_removed"] = total
            logger.info("已从 R 删除 %d 条观测（models=%s）", total, plan.extra["r_models_affected"])
        except Exception as e:
            done.extra["r_error"] = str(e)
            logger.error("删 R 列失败: %s", e)
    return done


def cmd_delete(
    names: list[str],
    *,
    delete_r: bool = False,
    yes: bool = False,
    json_mode: bool = False,
) -> int:
    plan = plan_delete(names, delete_r=delete_r)
    if json_mode:
        if not yes:
            emit(plan.to_dict(), json_mode=True, title="delete (dry-run)")
            return 0
        done = execute_delete(plan, delete_r=delete_r)
        emit(done.to_dict(), json_mode=True, title="delete (executed)")
        return 0
    # 人可读
    _print_plan(plan)
    if plan.extra.get("r_rows_total"):
        print(f"\n⚠ R 矩阵影响：将删除 {plan.extra['r_rows_total']} 条观测"
              f"（models={plan.extra['r_models_affected']}）")
    if not yes:
        print(f"\n（dry-run 预览，{len(plan.items)} 项，将释放 {fmt_size(plan.total_size)}）"
              "\n确认执行请加 --yes")
        return 0
    print("\n执行中...")
    done = execute_delete(plan, delete_r=delete_r)
    print(f"✓ 完成：软删除 {len([i for i in done.items if i.kind == 'run_dir'])} 个 run 目录"
          f"，释放 {fmt_size(done.total_size)}")
    if done.extra.get("r_rows_removed") is not None:
        print(f"✓ R 矩阵：删除 {done.extra['r_rows_removed']} 条观测")
    return 0


def _print_plan(plan: Plan) -> None:
    """人可读打印 dry-run plan。"""
    print(f"操作: {plan.action}  模式: {'dry-run' if plan.dry_run else 'executed'}")
    rows = []
    for item in plan.items:
        rows.append([item.kind, item.path, fmt_size(item.size), item.detail])
    print_table(rows, headers=["kind", "path", "size", "detail"])
    print(f"合计: {len(plan.items)} 项, {fmt_size(plan.total_size)}")
