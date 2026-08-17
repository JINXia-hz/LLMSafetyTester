"""storage.catalog — 目录库读写实现（runs / trials / tasks 三张表的 DAO）。

本模块是"发现 run"的**唯一实现**，取代此前的三份目录扫描
（server/data_query._discover_runs、management/runs.discover_runs、
control/compare.discover_workspace_runs）与两份字段提取。

核心机制——增量对账（reconcile）：
  目录库是派生索引，真相在文件。查询前先对账：扫描 runs 根的两级目录，
  凡 run 目录 mtime 与库行不一致（新建/产物更新/消失）即重扫入库。
  这保证：
    1. 写入口登记（register_run）失败或进程崩溃不丢可见性（下次查询自愈）；
    2. 重构前的存量目录在首次查询时自动入册（无需手工迁移）；
    3. 同秒撞名目录（`<ts>_2` 后缀，RUN_NAME_RE 裂缝）不再不可见。

布局支持（三种世代 + workdir/workspace 卫星布局）：
  gen3   runs/<ts>/<target>/          产物在 target 子目录
  gen1/2 runs/<ts>/                   产物扁平在时间戳目录（gen2 文件名带 __模型 后缀）
  卫星   <root>/<target>/             work-dir / workspace / trial：target 目录直接在根下
  （根下的 state/ predictors/ logs/ 等隔离子目录因不含 RUN_ARTIFACTS 自然被跳过）
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from sqlmodel import or_, select

from llmsec.core import config as _config
from llmsec.core.io import read_json
from llmsec.core.results import extract_report_metrics
from llmsec.storage import db
from llmsec.storage.models import Run, Task, Trial, dir_size, run_name

# run 目录命名契约的单一来源（原 management/runs.py 与 data_query 各一份且
# 带 $ 锚——`_allocate_runs_dir` 产生的 `<ts>_2` 撞名后缀不匹配，同秒第二个
# run 对两个发现实现都不可见；现放宽并收敛到此处）。
RUN_TS_FORMAT = "%Y-%m-%d_%H%M%S"
RUN_NAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}(?:_\d+)?$")

# run 目录内可能出现的产物文件（判定"这是一个 run"的最低证据；失败 run 可能
# 只有部分文件）。原 management/runs.py 的元组迁此。
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
# 扫描（rebuild / reconcile 共用）
# ============================================================

def _has_any_artifact(d: Path) -> bool:
    return any((d / a).exists() for a in RUN_ARTIFACTS)


def _scan_run(run_dir: Path, batch: str, target_hint: str | None) -> Run:
    """读单个 run 目录 → Run 登记行（读 report + 目录 stat，较重，仅在变更时调用）。"""
    report = read_json(run_dir / "runner_report.json") or {}
    target = target_hint or report.get("target_model", "") or run_dir.name
    try:
        mtime = run_dir.stat().st_mtime
    except OSError:
        mtime = time.time()
    return Run(
        name=run_name(batch, target_hint),
        batch=batch,
        target=target,
        layout=3 if target_hint else 2,
        dir_path=str(run_dir.resolve()),
        mtime=mtime,
        registered_at=time.time(),
        target_model=report.get("target_model") or (target_hint or None),
        security_level=report.get("security_level"),
        has_report=bool(report),
        has_md=(run_dir / "security_report.md").exists(),
        has_tree=(run_dir / "security_tree.json").exists(),
        has_cluster=(run_dir / "cluster_security_analysis.json").exists(),
        has_artifact=_has_any_artifact(run_dir),
        size=dir_size(run_dir),
        metrics=extract_report_metrics(report) if report else None,
    )


def _is_satellite(root: Path) -> bool:
    """该根是否卫星布局（work-dir/workspace/trial：target 目录直接在根下）。

    全局 runs 根（= config.RUNS_DIR，测试里被 patch 到 tmp 也算）只认时间戳
    批次目录，非时间戳目录一律忽略（旧发现口径：not-a-run 之类杂物不可见）；
    卫星根下非时间戳目录就是 target 本身。
    """
    try:
        return root.resolve() != Path(_config.RUNS_DIR).resolve()
    except OSError:
        return True


def _scan_candidates(root: Path, *, satellite: bool, include_empty: bool = False) -> list[tuple[Path, str, str | None, bool]]:
    """扫 runs 根的两级结构，产出 (run_dir, batch, target_hint, has_artifact) 候选。

    - 时间戳批次目录（匹配 RUN_NAME_RE）：其下含产物的 target 子目录各是一个
      run（gen3）；批次自身含产物且无产物子目录 → gen1/2 扁平 run。
    - 卫星布局（satellite=True）时：非时间戳的子目录含产物即一个 run，
      target_hint=目录名（全局模式下非时间戳目录是杂物，一律跳过）。
    - include_empty：把零产物的空壳目录也产出（供 reindex/detect_junk 识别
      中止 run 残壳；常规查询不带，保持与旧发现实现输出一致）。
    """
    out: list[tuple[Path, str, str | None, bool]] = []
    try:
        entries = list(root.iterdir())
    except OSError:
        return out
    for d in entries:
        if not d.is_dir():
            continue
        if RUN_NAME_RE.match(d.name):
            # 时间戳批次：扫 target 子目录
            try:
                subs = [t for t in d.iterdir() if t.is_dir()]
            except OSError:
                subs = []
            has_artifact_sub = False
            for t in subs:
                has = _has_any_artifact(t)
                if has or include_empty:
                    out.append((t, d.name, t.name, has))
                has_artifact_sub = has_artifact_sub or has
            # gen1/2：批次自身是 run（无产物子目录时）
            self_has = _has_any_artifact(d)
            if not has_artifact_sub and (self_has or include_empty):
                out.append((d, d.name, None, self_has))
        elif satellite:
            # 卫星布局：根下直接是 target 目录
            has = _has_any_artifact(d)
            if has or include_empty:
                out.append((d, root.name, d.name, has))
    return out


# ============================================================
# runs：登记 / 对账 / 查询
# ============================================================

def register_run(run_dir: Path, *, batch: str | None = None, target: str | None = None) -> None:
    """写入口轻登记：per-target run 目录刚创建（尚无产物）时占一行。

    作用：进行中的 run 从创建起即可见（junk/监控），同秒撞名目录不再依赖
    读方扫描发现。产物落盘后的富化（report/metrics/size）由 reconcile 按
    mtime 自动补齐——登记必须廉价，不能出现在评估热路径上。
    """
    run_dir = Path(run_dir)
    batch = batch or run_dir.parent.name
    try:
        mtime = run_dir.stat().st_mtime
    except OSError:
        return
    row = Run(
        name=run_name(batch, target),
        batch=batch,
        target=target or run_dir.name,
        layout=3 if target else 2,
        dir_path=str(run_dir.resolve()),
        mtime=mtime,
        registered_at=time.time(),
    )
    with db.tx() as s:
        s.merge(row)


def allocate_runs_dir(base_dir: Path, name: str) -> Path:
    """撞名分配：name 已存在时追加 _2/_3 后缀（原 runner._allocate_runs_dir）。

    返回不冲突的目录路径（已 mkdir）。目录名即撞名事实的载体（`<ts>_2`），
    登记发生在 per-target run 目录创建处（register_run）——批次级不登记，
    避免制造无产物的"幻影 run 行"。
    """
    base_dir = Path(base_dir)
    candidate = base_dir / name
    suffix = 2
    while candidate.exists():
        candidate = base_dir / f"{name}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def reconcile_runs(runs_root: Path | str | None = None, *, db_path=None, include_empty: bool = False) -> dict:
    """增量对账：新/变更/消失的 run 目录 ↔ 库行同步。查询前自动调用。

    Returns: {"rescanned": n, "removed": n, "adopted": n}
    """
    root = Path(runs_root) if runs_root is not None else Path(_config.RUNS_DIR)
    dbp = Path(db_path) if db_path is not None else db.db_for(root)
    if not root.exists():
        return {"rescanned": 0, "removed": 0, "adopted": 0}
    with db.session(dbp) as s:
        known: dict[str, float] = {r.name: r.mtime for r in s.exec(select(Run)).all()}
    if not root.is_dir():
        return {"rescanned": 0, "removed": 0, "adopted": 0}

    cands = _scan_candidates(root, satellite=_is_satellite(root), include_empty=include_empty)
    rescanned = adopted = 0
    with db.tx(dbp) as s:
        seen_dirs: set[str] = set()
        for run_dir, batch, target_hint, _has in cands:
            try:
                mtime = run_dir.stat().st_mtime
            except OSError:
                continue
            seen_dirs.add(str(run_dir.resolve()))
            nm = run_name(batch, target_hint)
            if known.get(nm) == mtime:
                continue  # 无变化，跳过重扫（report 读 + dir_size 是重活）
            if nm in known:
                rescanned += 1
            else:
                adopted += 1
            s.merge(_scan_run(run_dir, batch, target_hint))
        # 扁平行被 gen3 取代：批次已含产物子目录时，批级别的 gen1/2 旧行过期
        # （migrate-layouts 把扁平产物搬进子目录后，批次目录本身还在——消失
        # 检测打不到它，必须按"批次有了子 run"判定取代）
        superseded = {b for _, b, t, _ in cands if t} & known.keys()
        removed = 0
        for b in superseded:
            stale_row = s.get(Run, b)
            if stale_row is not None and stale_row.layout != 3:
                s.delete(stale_row)
                removed += 1
        # 消失的行：库里有、目录没了（被软删/手动删）
        for nm in list(known):
            row = s.get(Run, nm)
            if row is not None and row.dir_path not in seen_dirs and not Path(row.dir_path).is_dir():
                s.delete(row)
                removed += 1
    return {"rescanned": rescanned, "removed": removed, "adopted": adopted}


def query_runs(
    *,
    runs_root: Path | str | None = None,
    db_path=None,
    target: str | None = None,
    has_report: bool | None = None,
    has_artifact: bool | None = None,
    since: str | None = None,       # ISO 日期 YYYY-MM-DD（含），按 batch 前缀
    until: str | None = None,
    limit: int | None = None,
    reconcile: bool = True,
) -> list[Run]:
    """查询 run 登记（name 倒序，与旧 discover 实现排序一致）。默认先对账。

    消费方口径映射（保持与旧实现输出等价）：
      data_query（只认有报告的）  → has_report=True
      management（含失败 run）    → has_artifact=True
      两者默认 None=不过滤（全部行，含登记未开工的进行中 run）。
    """
    root = Path(runs_root) if runs_root is not None else Path(_config.RUNS_DIR)
    dbp = Path(db_path) if db_path is not None else db.db_for(root)
    if reconcile:
        reconcile_runs(root, db_path=dbp)
    q = select(Run)
    if target:
        q = q.where(or_(Run.target == target, Run.target_model == target))
    if has_report is not None:
        q = q.where(Run.has_report == has_report)
    if has_artifact is not None:
        q = q.where(Run.has_artifact == has_artifact)
    if since:
        q = q.where(Run.batch >= since)
    if until:
        q = q.where(Run.batch <= until + "_zz")
    q = q.order_by(Run.name.desc())
    if limit:
        q = q.limit(limit)
    with db.session(dbp) as s:
        return list(s.exec(q).all())


def get_run(name: str, *, runs_root: Path | str | None = None, db_path=None) -> Run | None:
    """run 名 → 登记行（'batch/target' / 'batch' / 卫星库 target 名）。"""
    root = Path(runs_root) if runs_root is not None else Path(_config.RUNS_DIR)
    dbp = Path(db_path) if db_path is not None else db.db_for(root)
    with db.session(dbp) as s:
        return s.get(Run, name)


def get_task(task_id: str, *, tasks_dir: Path | str | None = None, db_path=None) -> Task | None:
    """task_id → 登记行（默认先对账 meta.json，外部进程写的任务入册）。"""
    dbp = db.catalog_db() if db_path is None else Path(db_path)
    if tasks_dir is not None or db_path is None:
        reconcile_tasks(tasks_dir=tasks_dir, db_path=dbp)
    with db.session(dbp) as s:
        return s.get(Task, task_id)


def remove_run(name: str, *, db_path=None) -> bool:
    """删登记行（run 目录被软删/清理后同步库）。返回是否删到了。"""
    dbp = Path(db_path) if db_path is not None else db.catalog_db()
    with db.tx(dbp) as s:
        row = s.get(Run, name)
        if row is None:
            return False
        s.delete(row)
        return True


def rebuild_runs(runs_root: Path | str | None = None, *, db_path=None, include_empty: bool = True) -> dict:
    """全量重建 runs 索引（清空该库 runs 行后重扫；storage reindex 用）。"""
    root = Path(runs_root) if runs_root is not None else Path(_config.RUNS_DIR)
    dbp = Path(db_path) if db_path is not None else db.db_for(root)
    with db.tx(dbp) as s:
        for row in s.exec(select(Run)).all():
            s.delete(row)
    return reconcile_runs(root, db_path=dbp, include_empty=include_empty)


# ============================================================
# trials：登记 / 查询（真相源 trials.jsonl 不动）
# ============================================================

def upsert_trial_record(study: str, rec: dict) -> None:
    """trial record 整行 upsert（study.py 的唯一写点，P4 起 db 为真相）。

    rec 形状 = run_trial 返回 + study 补登记（trial/idx/target/seed/search_fp/
    search_params）；同 (study, idx) 覆盖。
    """
    idx = rec.get("trial", rec.get("idx"))
    if idx is None:
        return
    now = time.time()
    row = Trial(
        study=study, idx=int(idx), work_dir=str(rec.get("work_dir", "")),
        registered_at=now, target=rec.get("target"), seed=rec.get("seed"),
        status=rec.get("status"), metrics=rec.get("metrics") or None,
        updated_at=now, params=rec.get("params") or None,
        search_fp=rec.get("search_fp"), search_params=rec.get("search_params") or None,
        returncode=rec.get("returncode"), error=rec.get("error"),
        elapsed_s=rec.get("elapsed_s"),
    )
    with db.tx() as s:
        existing = s.get(Trial, (study, int(idx)))
        if existing is not None:
            row.registered_at = existing.registered_at  # 首登时间不漂移
        s.merge(row)


def register_trial(study: str, idx: int, *, work_dir: Path | str, target: str | None = None,
                   seed: str | None = None, status: str | None = "running",
                   metrics: dict | None = None, db_path=None) -> None:
    row = Trial(
        study=study, idx=idx, work_dir=str(Path(work_dir).resolve()),
        registered_at=time.time(), target=target, seed=seed,
        status=status, metrics=metrics, updated_at=time.time(),
    )
    with db.tx(db.catalog_db() if db_path is None else db_path) as s:
        s.merge(row)


def update_trial(study: str, idx: int, *, status: str | None = None,
                 metrics: dict | None = None, db_path=None) -> None:
    with db.tx(db.catalog_db() if db_path is None else db_path) as s:
        row = s.get(Trial, (study, idx))
        if row is None:
            return
        if status is not None:
            row.status = status
        if metrics is not None:
            row.metrics = metrics
        row.updated_at = time.time()
        s.add(row)


def query_trials(study: str | None = None, *, db_path=None) -> list[Trial]:
    dbp = db.catalog_db() if db_path is None else Path(db_path)
    with db.session(dbp) as s:
        q = select(Trial)
        if study:
            q = q.where(Trial.study == study)
        return list(s.exec(q.order_by(Trial.study, Trial.idx)).all())


# ============================================================
# tasks：登记 / 对账 / 查询（跨进程真相仍是 meta.json）
# ============================================================

def register_task(task_id: str, kind: str, *, cmd: str | None = None, pid: int | None = None,
                  status: str = "queued", log_path: Path | str | None = None,
                  started_at: str | None = None, meta: dict | None = None, db_path=None) -> None:
    row = Task(
        task_id=task_id, kind=kind, cmd=cmd, pid=pid, status=status,
        log_path=str(log_path) if log_path else None,
        started_at=started_at, meta=meta,
        registered_at=time.time(), updated_at=time.time(),
    )
    with db.tx(db.catalog_db() if db_path is None else db_path) as s:
        s.merge(row)


def update_task(task_id: str, *, status: str | None = None, pid: int | None = None,
                meta: dict | None = None, db_path=None) -> None:
    with db.tx(db.catalog_db() if db_path is None else db_path) as s:
        row = s.get(Task, task_id)
        if row is None:
            return
        if status is not None:
            row.status = status
        if pid is not None:
            row.pid = pid
        if meta is not None:
            row.meta = meta
        row.updated_at = time.time()
        s.add(row)


def upsert_task(task_id: str, kind: str, *, cmd: str | None = None, pid: int | None = None,
                status: str = "unknown", log_path: Path | str | None = None,
                started_at: str | None = None, meta: dict | None = None, db_path=None) -> None:
    """任务元数据 upsert（保留首登时间）。task_manager._persist_meta 的落库镜像：
    meta.json（跨进程真相）写完就镜像一行，注册/状态迁移统一入口。"""
    with db.tx(db.catalog_db() if db_path is None else db_path) as s:
        row = s.get(Task, task_id)
        if row is None:
            row = Task(task_id=task_id, kind=kind, status=status,
                       registered_at=time.time(), cmd=cmd, pid=pid,
                       log_path=str(log_path) if log_path else None,
                       started_at=started_at,
                       meta=meta if isinstance(meta, dict) else None,
                       updated_at=time.time())
        else:
            row.kind = kind
            row.cmd = cmd
            row.pid = pid
            row.status = status
            row.log_path = str(log_path) if log_path else None
            row.started_at = started_at
            row.meta = meta if isinstance(meta, dict) else None
            row.updated_at = time.time()
        s.add(row)


def reconcile_tasks(tasks_dir: Path | str | None = None, *, db_path=None) -> dict:
    """meta.json 文件 ↔ tasks 行对账：导入新/变更，清理消失（gc 后）。

    task_manager 每次状态迁移都会覆盖写 meta.json（mtime 变化），据此做增量。
    持有进程崩溃无人回写终态的行以 meta.json 为准（PID 通道语义不变）。
    """
    tdir = Path(tasks_dir) if tasks_dir is not None else Path(_config.TASK_LOG_DIR)
    dbp = db.catalog_db() if db_path is None else Path(db_path)
    if not tdir.is_dir():
        return {"imported": 0, "removed": 0}
    with db.session(dbp) as s:
        known: dict[str, float] = {t.task_id: (t.updated_at or 0.0) for t in s.exec(select(Task)).all()}

    disk: dict[str, float] = {}
    for f in tdir.iterdir():
        if not f.name.endswith(".meta.json"):
            continue
        try:
            disk[f.name[: -len(".meta.json")]] = f.stat().st_mtime
        except OSError:
            continue

    imported = 0
    with db.tx(dbp) as s:
        for tid, mtime in disk.items():
            if known.get(tid) == mtime:
                continue
            try:
                data = json.loads((tdir / f"{tid}.meta.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            imported += 1
            s.merge(Task(
                task_id=tid,
                kind=str(data.get("kind") or tid.split("-", 1)[0]),
                cmd=data.get("cmd"),
                pid=data.get("pid") if isinstance(data.get("pid"), int) else None,
                status=str(data.get("status") or "unknown"),
                log_path=str(tdir / f"{tid}.log"),
                started_at=data.get("started_at"),
                meta=data.get("meta") if isinstance(data.get("meta"), dict) else None,
                registered_at=mtime,  # 首见时间不可考，用文件 mtime 近似
                updated_at=mtime,
            ))
        removed = 0
        for tid in list(known):
            if tid not in disk:
                row = s.get(Task, tid)
                if row is not None:
                    s.delete(row)
                    removed += 1
    return {"imported": imported, "removed": removed}


def query_tasks(*, kinds: tuple[str, ...] | None = None, active_only: bool = False,
                limit: int | None = None, db_path=None, tasks_dir: Path | str | None = None,
                reconcile: bool = True) -> list[Task]:
    """查询任务登记（registered_at 倒序）。默认先对账（meta.json 入册/清理）。

    tasks_dir：对账来源目录（默认 config.TASK_LOG_DIR）。TaskStore 等支持
    注入目录的消费者传入自己的目录，避免对账打到真实 output/tasks。
    """
    dbp = db.catalog_db() if db_path is None else Path(db_path)
    if reconcile:
        reconcile_tasks(tasks_dir=tasks_dir, db_path=dbp)
    with db.session(dbp) as s:
        q = select(Task)
        if kinds:
            q = q.where(Task.kind.in_(list(kinds)))
        if active_only:
            q = q.where(Task.status.in_(["queued", "running"]))
        q = q.order_by(Task.registered_at.desc())
        if limit:
            q = q.limit(limit)
        return list(s.exec(q).all())
