"""management.storage — 目录库管理 CLI（llmsec-manage storage 子命令组）。

能力：
  reindex          全量重建 runs/tasks 索引（目录库可删可重建的兑现入口）
  verify           完整性校验：库行 ↔ 目录树双向对账（"搬迁即检测"的检测端）
  gc-tasks         终态任务文件软删（task 只增不减的治理入口）+ 同步删库行
  trials           列出 trials 登记行（study 维度）
  migrate-layouts  Gen1/Gen2 扁平 run 物理归一为 Gen3 <ts>/<target>/ 布局
  backup-r         R 库备份（sqlite3 backup API）——migrate-r 已完成使命退役

所有删除路径一律走 management.common 的软删（.trash 可回滚），dry-run 默认。
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from llmsec.core import config as _config
from llmsec.core.io import read_json
from llmsec.core.logging import get_logger
from llmsec.management.common import (
    Plan,
    emit,
    fmt_size,
    print_table,
    soft_remove,
)
from llmsec.storage import catalog
from llmsec.storage import contract as _storage

logger = get_logger(__name__)

# Gen2 产物文件的 "__<model>" 后缀模式：allergy__minimax.json / attack_results__X.jsonl
_GEN2_SUFFIX_RE = re.compile(r"^(?P<stem>[a-z_]+)__(?P<model>.+)$")


# ============================================================
# reindex
# ============================================================
def cmd_reindex(*, include_empty: bool = True, json_mode: bool = False) -> int:
    """全量重建目录库（清空 runs 行后重扫 + tasks 对账）。"""
    runs_st = catalog.rebuild_runs(include_empty=include_empty)
    tasks_st = catalog.reconcile_tasks()
    summary = {
        "db": str(_storage.catalog_db()),
        "runs": runs_st,
        "tasks": tasks_st,
    }
    if json_mode:
        emit(summary, json_mode=True, title="reindex")
    else:
        print(f"目录库: {summary['db']}")
        print(f"runs : 重建完成——重扫 {runs_st['rescanned']} / 新入册 {runs_st['adopted']} / 清理 {runs_st['removed']}")
        print(f"tasks: 导入 {tasks_st['imported']} / 清理 {tasks_st['removed']}")
    return 0


# ============================================================
# verify
# ============================================================
def cmd_verify(*, json_mode: bool = False) -> int:
    """完整性校验（非破坏）：

    1. 幂等性：对账后再次对账应零动作（库与树一致）；
    2. 物理一致：每行 dir_path 存在、has_report 与实际文件一致；
    3. 覆盖一致：树上每个含产物的 run 目录都有库行（无不可见 run——
       RUN_NAME_RE 裂缝类问题的直接检测）；
    （P4 起 tasks 库行即真相，不再与文件对账。）
    """
    problems: list[str] = []
    root = Path(_config.RUNS_DIR)
    dbp = catalog.db.db_for(root)

    catalog.reconcile_runs(root, db_path=dbp)  # 第一次：吸收树上的新变更
    st2 = catalog.reconcile_runs(root, db_path=dbp)
    if st2["rescanned"] or st2["removed"] or st2["adopted"]:
        problems.append(f"对账不幂等: 第二次仍动作 {st2}")

    runs = catalog.query_runs(runs_root=root, db_path=dbp, reconcile=False)
    tree_names: set[str] = set()
    for run_dir, batch, target_hint, _has in catalog._scan_candidates(
            root, satellite=catalog._is_satellite(root), include_empty=False):
        tree_names.add(catalog.run_name(batch, target_hint))
        if not run_dir.is_dir():
            problems.append(f"库行目录不存在: {run_dir}")
    row_names = {r.name for r in runs}
    for nm in sorted(tree_names - row_names):
        problems.append(f"树上 run 无库行（不可见）: {nm}")
    for r in runs:
        d = Path(r.dir_path)
        if not d.is_dir():
            problems.append(f"库行目录缺失: {r.name} -> {r.dir_path}")
        elif r.has_report != (d / "runner_report.json").exists():
            problems.append(f"has_report 与实际不符: {r.name}")

    catalog.reconcile_tasks()  # 第一次：吸收 meta.json 的变更
    tasks_st2 = catalog.reconcile_tasks()
    if tasks_st2["imported"] or tasks_st2["removed"]:
        problems.append(f"tasks 对账不幂等: {tasks_st2}")

    ok = not problems
    out = {"ok": ok, "runs_indexed": len(row_names), "problems": problems}
    if json_mode:
        emit(out, json_mode=True, title="verify")
    else:
        print(f"{'✓' if ok else '✗'} 目录库校验{'通过' if ok else '失败'}"
              f"（runs {len(row_names)} 行，库 {dbp}）")
        for p in problems:
            print(f"  - {p}")
    return 0 if ok else 1


# ============================================================
# gc-tasks
# ============================================================
def plan_gc_tasks(older_than_days: float) -> Plan:
    """构造终态任务清理预览（P4：行驱动）。

    终态（success/failed/cancelled/ended/unknown）且 updated_at 早于阈值的行
    → 三件套文件（.log/.progress.jsonl/.meta.json）软删 + 删行。
    running/queued 永不清理。旧世代 meta.json（无行的）按文件 mtime 判龄清理。
    """
    tdir = Path(_config.TASK_LOG_DIR)
    cutoff = time.time() - older_than_days * 86400
    plan = Plan(action="gc-tasks", dry_run=True)
    if not tdir.is_dir():
        return plan
    terminal = ("success", "failed", "cancelled", "ended", "unknown", "external")
    rows = catalog.query_tasks(reconcile=False)
    by_id = {r.task_id: r for r in rows}
    doomed: set[str] = set()
    for r in rows:
        if r.status in terminal and (r.updated_at or 0) < cutoff:
            doomed.add(r.task_id)
    # 旧世代残件：有文件无行的 meta.json（reindex 前）
    legacy = 0
    for m in tdir.glob("*.meta.json"):
        tid = m.name[: -len(".meta.json")]
        if tid in by_id:
            continue
        data = read_json(m) or {}
        if str(data.get("status") or "unknown") in terminal and m.stat().st_mtime < cutoff:
            doomed.add(tid)
            legacy += 1
    for tid in sorted(doomed):
        size = 0
        for suffix in (".meta.json", ".log", ".progress.jsonl"):
            p = tdir / f"{tid}{suffix}"
            if p.exists():
                size += p.stat().st_size
        plan.add(tdir / f"{tid}.meta.json", size=size, kind="task_files",
                 detail=f"{tid}（终态，三件套软删 + 删库行）")
    plan.extra["tasks"] = len(doomed)
    plan.extra["scanned"] = len(rows)
    plan.extra["legacy_files"] = legacy
    return plan


def execute_gc_tasks(plan: Plan) -> Plan:
    from llmsec.storage import db as storage_db

    done = Plan(action="gc-tasks", dry_run=False, extra=dict(plan.extra))
    tdir = Path(_config.TASK_LOG_DIR)
    for item in plan.items:
        if item.kind != "task_files":
            continue
        # plan.path 存的是 .meta.json 完整路径（相对 OUTPUT_DIR）
        meta = _config.OUTPUT_DIR / item.path
        tid = meta.name[: -len(".meta.json")]
        for suffix in (".meta.json", ".log", ".progress.jsonl"):
            p = tdir / f"{tid}{suffix}"
            if p.exists():
                soft_remove(p)
        try:
            with storage_db.tx(storage_db.catalog_db()) as s:
                row = s.get(_storage.Task, tid)
                if row is not None:
                    s.delete(row)
        except Exception as e:  # 库行删除失败不影响文件清理（reconcile 会自愈反向）
            logger.warning("gc-tasks 删库行失败: %s", e)
        done.add(meta, size=item.size, kind="task_files", detail=f"{tid} 已清理")
    return done


def cmd_gc_tasks(older_than_days: float, *, yes: bool = False, json_mode: bool = False) -> int:
    plan = plan_gc_tasks(older_than_days)
    if json_mode:
        if not yes:
            emit(plan.to_dict(), json_mode=True, title="gc-tasks (dry-run)")
            return 0
        emit(execute_gc_tasks(plan).to_dict(), json_mode=True, title="gc-tasks (executed)")
        return 0
    print(f"扫描 {plan.extra['scanned']} 个任务，{plan.extra['tasks']} 个终态超期（>{older_than_days:g} 天）")
    if not plan.items:
        print("无可清理任务。")
        return 0
    for item in plan.items:
        print(f"  - {item.path}  {fmt_size(item.size)}")
    if not yes:
        print(f"\n（dry-run 预览，将释放 {fmt_size(plan.total_size)}）\n确认执行请加 --yes")
        return 0
    print("\n执行中...")
    execute_gc_tasks(plan)
    print(f"✓ 完成：清理 {plan.extra['tasks']} 个任务，释放 {fmt_size(plan.total_size)}")
    return 0


# ============================================================
# trials
# ============================================================
def cmd_trials(study: str | None, *, json_mode: bool = False) -> int:
    rows = catalog.query_trials(study)
    if json_mode:
        emit({"trials": [r.as_dict() for r in rows], "count": len(rows)},
             json_mode=True, title="trials")
    elif rows:
        print_table(
            [[r.study, r.idx, r.target or "-", r.seed or "-", r.status or "-",
              str((r.metrics or {}).get("conv_rounds", "-"))] for r in rows],
            headers=["study", "idx", "target", "seed", "status", "conv_rounds"],
        )
        print(f"\n共 {len(rows)} 个 trial（真相源：trials.jsonl）")
    else:
        print("（无 trial 登记——study 运行时由 executor 入册）")
    return 0


# ============================================================
# migrate-layouts（Gen1/Gen2 → Gen3）
# ============================================================
def _gen2_split_target(filename: str) -> tuple[str, str] | None:
    """'attack_results__minimax.jsonl' → ('attack_results', 'minimax')；无后缀返回 None。"""
    m = _GEN2_SUFFIX_RE.match(Path(filename).stem)
    if not m:
        return None
    stem, model = m.group("stem"), m.group("model")
    return (stem, model) if stem in ("allergy", "attack_results", "sampler_log", "state") else None


def plan_migrate_layouts() -> Plan:
    """构造 Gen1/Gen2 → Gen3 归一预览。

    Gen1（扁平无后缀，单目标）：全部产物移入 <ts>/<target>/（target 取
    runner_report.target_model，缺报告用批次目录名兜底→has_report=False）。
    Gen2（扁平 __<model> 后缀，多目标）：按后缀拆 <ts>/<model>/，文件剥后缀；
    批次级 runner_report.json 归其 target_model 主人；multi_target_report.json
    留在批次级（共享聚合产物，不参与 run 判定）。
    """
    plan = Plan(action="migrate-layouts", dry_run=True)
    root = Path(_config.RUNS_DIR)
    if not root.is_dir():
        return plan
    for batch_dir in sorted(root.iterdir()):
        if not batch_dir.is_dir() or not catalog.RUN_NAME_RE.match(batch_dir.name):
            continue
        files = [f for f in batch_dir.iterdir() if f.is_file()]
        if not files:
            continue
        suffixed = [f for f in files if _gen2_split_target(f.name)]
        if suffixed:
            # Gen2：按后缀模型拆分
            report = read_json(batch_dir / "runner_report.json") or {}
            report_owner = report.get("target_model")
            for f in files:
                split = _gen2_split_target(f.name)
                if split:
                    stem, model = split
                    dest = batch_dir / model / f"{stem}{f.suffix}"
                elif f.name == "runner_report.json" and report_owner:
                    dest = batch_dir / report_owner / f.name
                else:
                    continue  # multi_target_report.json 等共享文件留批次级
                plan.add(f, size=f.stat().st_size, kind="move",
                         detail=f"→ {dest.relative_to(batch_dir)}")
        else:
            # Gen1：整体移入单 target 子目录
            report = read_json(batch_dir / "runner_report.json") or {}
            target = report.get("target_model") or batch_dir.name
            for f in files:
                plan.add(f, size=f.stat().st_size, kind="move",
                         detail=f"→ {target}/{f.name}")
            plan.extra.setdefault("gen1_batches", []).append(batch_dir.name)
    return plan


def execute_migrate_layouts(plan: Plan) -> Plan:
    done = Plan(action="migrate-layouts", dry_run=False)
    moved = 0
    for item in plan.items:
        if item.kind != "move":
            continue
        src = _config.OUTPUT_DIR / item.path
        rel_dest = item.detail.removeprefix("→ ").strip().replace("\\", "/")
        batch = Path(item.path).parent
        dest = _config.OUTPUT_DIR / batch / rel_dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            src.replace(dest)
            moved += 1
            done.add(src, size=item.size, kind="moved", detail=f"→ {rel_dest}")
        except OSError as e:
            done.add(src, size=0, kind="error", detail=f"移动失败: {e}")
    # 迁移后重对账（mtime 全变，自动重扫入册新布局行）
    if moved:
        catalog.reconcile_runs(include_empty=True)
    done.extra["moved"] = moved
    return done


# ============================================================
# migrate-r / backup-r（阶段 2：R 矩阵 → SQLite）
# ============================================================
def cmd_backup_r(out: str | None, *, json_mode: bool = False) -> int:
    """R 库备份（sqlite3 backup API，WAL 安全；.bak 轮转的替代）。"""
    from llmsec.storage import rstore

    dest = Path(out) if out else _config.STATE_DIR / f"results.backup.{time.strftime('%Y%m%d_%H%M%S')}.db"
    rstore.backup(dest)
    emit({"ok": True, "dest": str(dest), "size": dest.stat().st_size},
         json_mode=json_mode, title="backup-r")
    print(f"✓ 已备份 R 库 → {dest}（{fmt_size(dest.stat().st_size)}）")
    return 0


def cmd_migrate_layouts(*, yes: bool = False, json_mode: bool = False) -> int:
    plan = plan_migrate_layouts()
    if json_mode:
        if not yes:
            emit(plan.to_dict(), json_mode=True, title="migrate-layouts (dry-run)")
            return 0
        emit(execute_migrate_layouts(plan).to_dict(), json_mode=True, title="migrate-layouts (executed)")
        return 0
    moves = [i for i in plan.items if i.kind == "move"]
    if not moves:
        print("（无 Gen1/Gen2 扁平布局需要迁移——已全部是 Gen3）")
        return 0
    print(f"将把 {len(moves)} 个产物文件归一为 Gen3 <ts>/<target>/ 布局：")
    for item in moves[:40]:
        print(f"  {item.path}  {item.detail}")
    if len(moves) > 40:
        print(f"  ...（共 {len(moves)} 项）")
    if not yes:
        print("\n（dry-run 预览；执行后自动重对账目录库）\n确认执行请加 --yes")
        return 0
    print("\n执行中...")
    done = execute_migrate_layouts(plan)
    print(f"✓ 完成：移动 {done.extra['moved']} 个文件")
    return 0
