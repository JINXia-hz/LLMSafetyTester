"""management.runs — 过滤 / 列出 / 清理 run 历史。

复用 data_query._discover_runs 的扫描逻辑（双布局：runs/<ts>/<target>/ 与旧 runs/<ts>/），
但在本包内独立实现，不依赖 server 层，保持 CLI 纯净。

关键能力：
  - list：递归算目录大小（_discover_runs 只取 mtime，size 是空白），支持多维过滤。
  - delete：默认 dry-run 预览；软删除到 output/.trash/；可选 --delete-r 删 R 列
    （经 ResultsMatrix.remove_model + save(backup=True) + _file_lock，复用现有原子写）。
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from llmsec.core.config import RUNS_DIR
from llmsec.core.io import read_json
from llmsec.core.logging import get_logger
from llmsec.core.results import RESULTS_FILE, ResultsMatrix, _file_lock, extract_report_metrics
from llmsec.management.common import (
    Plan,
    dir_size,
    emit,
    fmt_size,
    print_table,
    soft_rmtree,
)

logger = get_logger(__name__)

RUN_NAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}$")

# run 目录内可能出现的产物文件（用于 size 统计与删除清单）
RUN_ARTIFACTS = (
    "runner_report.json",
    "security_tree.json",
    "security_report.md",
    "attack_results.jsonl",
    "state.json",
    "allergy.json",
    "sampler_log.jsonl",
    "cluster_security_analysis.json",
    "cluster_report.json",
)


# ============================================================
# 扫描（独立于 server 层，CLI 纯净）
# ============================================================
def discover_runs(runs_dir: Path | None = None) -> list[dict]:
    """扫描 run 目录，返回列表（时间倒序）。

    支持双布局：
      新布局 runs/<ts>/<target>/（含任意产物即视为 run，runner_report.json 可缺失=失败/垃圾）
      旧布局 runs/<ts>/runner_report.json

    与 data_query._discover_runs 字段兼容，并额外加 ``size``。
    关键差异：本实现把「无报告但有部分产物」的失败 run 也扫出来（has_report=False），
    供 detect_junk 识别——data_query 版只扫有 runner_report.json 的目录，会漏掉失败 run。
    """
    runs_dir = runs_dir or RUNS_DIR
    if not runs_dir.exists():
        return []
    runs: list[dict] = []
    for batch_dir in sorted(runs_dir.iterdir(), reverse=True):
        if not batch_dir.is_dir() or not RUN_NAME_RE.match(batch_dir.name):
            continue
        has_target_subdirs = False
        for target_dir in batch_dir.iterdir():
            if not target_dir.is_dir():
                continue
            # 任意已知产物存在即视为一个 run（失败 run 可能只有部分文件）
            if not any((target_dir / a).exists() for a in RUN_ARTIFACTS):
                continue
            has_target_subdirs = True
            report = read_json(target_dir / "runner_report.json") or {}
            runs.append(_run_entry(target_dir, batch_dir.name, target_dir.name, report))
        # 旧布局兼容
        if not has_target_subdirs and any((batch_dir / a).exists() for a in RUN_ARTIFACTS):
            report = read_json(batch_dir / "runner_report.json") or {}
            target = report.get("target_model", "") or batch_dir.name
            runs.append(_run_entry(batch_dir, batch_dir.name, target, report))
    runs.sort(key=lambda x: x["name"], reverse=True)
    return runs


def _run_entry(run_dir: Path, batch: str, target: str, report: dict) -> dict:
    """构造单条 run 元数据。"""
    size = dir_size(run_dir)
    mtime = datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat()
    m = extract_report_metrics(report)
    return {
        "name": f"{batch}/{target}" if batch != target else batch,
        "batch": batch,
        "target": target,
        "target_model": report.get("target_model", target),
        "security_level": report.get("security_level", "inconclusive"),
        "asr": m["asr"],
        "boundary_elo": m["boundary_elo"],
        "has_report": (run_dir / "runner_report.json").exists(),
        "has_md": (run_dir / "security_report.md").exists(),
        "mtime": mtime,
        "size": size,
    }


def run_dir_for(name: str, runs_dir: Path | None = None) -> Path | None:
    """run 名 → 目录路径。name 可以是 'ts/target' 或 'ts'。"""
    runs_dir = runs_dir or RUNS_DIR
    d = runs_dir / name
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
            plan.add(RUNS_DIR / name, size=0, kind="missing", detail="目录不存在，跳过")
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
                            RESULTS_FILE, size=0, kind="r_column",
                            detail=f"将从 R 删除 model={model} 的 {n} 条观测",
                        )
                except Exception as e:
                    plan.add(RESULTS_FILE, size=0, kind="r_error", detail=f"读 R 失败: {e}")
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
    # 删 R 列（一次性 load→remove→save，经 _file_lock）
    if delete_r and plan.extra.get("r_models_affected"):
        try:
            with _file_lock(RESULTS_FILE):
                R = ResultsMatrix.load()
                total = 0
                for model in plan.extra["r_models_affected"]:
                    total += R.remove_model(model)
                R.save(_locked=True)
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
