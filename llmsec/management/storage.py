"""management.storage — 目录库管理 CLI（llmsec-manage storage 子命令组）。

能力：
  reindex          全量重建 runs/tasks 索引（目录库可删可重建的兑现入口）
  verify           完整性校验：库行 ↔ 目录树双向对账（"搬迁即检测"的检测端）
  gc-tasks         终态任务文件软删（task 只增不减的治理入口）+ 同步删库行
  trials           列出 trials 登记行（study 维度）
  migrate-layouts  Gen1/Gen2 扁平 run 物理归一为 Gen3 <ts>/<target>/ 布局

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
    4. tasks：meta.json 文件集与库行集一致。
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
    """构造终态任务清理预览：终态（success/failed/cancelled/ended/unknown）
    且 meta.json mtime 早于阈值的任务三件套（.log/.progress.jsonl/.meta.json）。

    running/queued 永不清理（meta.json 是跨进程可见性通道，动了会变孤儿）。
    """
    tdir = Path(_config.TASK_LOG_DIR)
    cutoff = time.time() - older_than_days * 86400
    plan = Plan(action="gc-tasks", dry_run=True)
    if not tdir.is_dir():
        return plan
    terminal = ("success", "failed", "cancelled", "ended", "unknown", "external")
    metas = sorted(tdir.glob("*.meta.json"))
    doomed: set[str] = set()
    for m in metas:
        data = read_json(m) or {}
        status = str(data.get("status") or "unknown")
        if status in terminal and m.stat().st_mtime < cutoff:
            doomed.add(m.name[: -len(".meta.json")])
    for tid in sorted(doomed):
        size = 0
        for suffix in (".meta.json", ".log", ".progress.jsonl"):
            p = tdir / f"{tid}{suffix}"
            if p.exists():
                size += p.stat().st_size
        plan.add(tdir / f"{tid}.meta.json", size=size, kind="task_files",
                 detail=f"{tid}（终态，三件套软删 + 删库行）")
    plan.extra["tasks"] = len(doomed)
    plan.extra["scanned"] = len(metas)
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
def _matrix_digests(mat) -> dict:
    """矩阵三关摘要：总行数 / 逐模型列指纹 / 逐模型时序序列。"""
    import hashlib

    fps: dict[str, str] = {}
    seqs: dict[str, list] = {}
    for model in mat.all_models():
        payload = mat.column_payload(model, extra_fields=("unit", "round")) or ""
        fps[model] = hashlib.md5(payload.encode("utf-8")).hexdigest()
        seqs[model] = [
            (r.record, r.eval_score, repr(r.ts), r.status, sorted((r.extra or {}).items()))
            for r in mat.ordered_results(model)
        ]
    return {"rows": sum(len(c) for c in mat._r.values()),
            "units": list(mat.all_units()),
            "fingerprints": fps, "sequences": seqs}


def cmd_migrate_r(*, yes: bool = False, force: bool = False, json_mode: bool = False) -> int:
    """results.json → results.db 全量搬迁（"搬迁即检测"：三关校验任一不过即回退）。

    三关：总行数相等；逐模型列指纹（record:score:ts:unit:round 的 MD5）相等；
    ordered_results 时序序列逐条相等。校验通过后额外导出 results.snapshot.json
    人读快照；原 results.json 原样保留（回滚兜底）。
    """
    from llmsec.storage import rstore

    legacy = Path(_config.RESULTS_FILE)
    dbp = Path(_config.RESULTS_DB)
    if not legacy.exists():
        msg = (f"R 库已就绪: {dbp}" if dbp.exists()
               else f"无可迁移的 {legacy}（首次评估时自动建库）")
        emit({"ok": True, "detail": msg}, json_mode=json_mode, title="migrate-r")
        print(msg)
        return 0
    if dbp.exists() and not force:
        try:
            existing = rstore.load_matrix(dbp)
            if sum(len(c) for c in existing._r.values()) > 0:
                msg = (f"R 库已迁移且有数据: {dbp}（如需从 {legacy.name} 重建用 --force；"
                       "注意会覆盖库内 json 之后新增的观测）")
                emit({"ok": True, "detail": msg}, json_mode=json_mode, title="migrate-r")
                print(msg)
                return 0
        except RuntimeError:
            pass  # 库损坏 → 允许从 json 重建

    from llmsec.core.results import ResultsMatrix
    old = ResultsMatrix.load(legacy)  # 遗留 json 直读
    old_digest = _matrix_digests(old)
    if not yes:
        summary = {"dry_run": True, "rows": old_digest["rows"],
                   "models": len(old_digest["fingerprints"]),
                   "units": len(old_digest["units"])}
        emit(summary, json_mode=json_mode, title="migrate-r (dry-run)")
        if not json_mode:
            print(f"将把 {legacy} 迁入 {dbp}：{summary['rows']} 条观测，"
                  f"{summary['models']} 个模型，{summary['units']} 个单位")
            print("（dry-run 预览；执行含三关校验 + 导出人读快照）\n确认执行请加 --yes")
        return 0

    print("执行中...")
    rstore.save_matrix(old, path=dbp)
    problems: list[str] = []
    try:
        new_digest = _matrix_digests(rstore.load_matrix(dbp))
        if new_digest["rows"] != old_digest["rows"]:
            problems.append(f"行数不符: {old_digest['rows']} → {new_digest['rows']}")
        if new_digest["units"] != old_digest["units"]:
            problems.append("单位目录不符")
        for model, fp in old_digest["fingerprints"].items():
            if new_digest["fingerprints"].get(model) != fp:
                problems.append(f"列指纹不符: {model}")
        for model, seq in old_digest["sequences"].items():
            if new_digest["sequences"].get(model) != seq:
                problems.append(f"时序序列不符: {model}")
    except Exception as e:
        problems.append(f"迁移后读取失败: {e}")
    if problems:
        # 回退：释放引擎句柄（Windows 上持句柄删不掉文件）后删 db（含 WAL 伴生件），
        # 原 json 未动
        from llmsec.storage import db as storage_db
        storage_db.close()
        for suffix in ("", "-wal", "-shm"):
            p = Path(str(dbp) + suffix)
            if p.exists():
                p.unlink()
        emit({"ok": False, "problems": problems, "rolled_back": True},
             json_mode=json_mode, title="migrate-r (failed)")
        for p in problems:
            print(f"  ✗ {p}")
        print("已回退（原 results.json 未动）。")
        return 1

    snapshot = rstore.export_legacy_json(_config.STATE_DIR / "results.snapshot.json")
    emit({"ok": True, "rows": old_digest["rows"],
          "models": len(old_digest["fingerprints"]), "db": str(dbp),
          "snapshot": str(snapshot)},
         json_mode=json_mode, title="migrate-r")
    print(f"✓ 迁移完成：{old_digest['rows']} 条观测 → {dbp}")
    print("✓ 三关校验通过（行数 / 逐模型指纹 / 时序序列）")
    print(f"✓ 人读快照: {snapshot}（原 {legacy.name} 保留作回滚兜底）")
    return 0


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
