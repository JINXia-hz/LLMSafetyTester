"""storage.rstore — R 矩阵（原始观测）的 SQLite 后端（数据库重构阶段 2）。

设计要点：
  - ``Observation`` 表 = R[record][model] 的扁平化；``runits``/``rmodels``
    保序目录表保持 all_units/all_models 的插入序语义。
  - **ts 精确保存**：ts_json 列存 ``json.dumps(ts)``——int 5 与 float 5.0 在
    column_payload 指纹（elo_cache/predictor 缓存键）里是不同字符串，必须
    逐位保真；类型经 json.loads 还原。
  - **ins_order**：全局插入序，ordered_results 平局（同 ts）稳定排序的依据。
  - **并发写零丢失**：upsert_observations / remove_models 单事务
    （BEGIN IMMEDIATE），取代旧的"文件锁 load→改→save RMW"——两个并发
    publish 各自的事务串行提交，不存在后写覆盖先写。
  - 损坏处置：load 时 PRAGMA quick_check 轻量校验（替代 .bak/.corrupt.bak
    机器——SQLite 事务本身保证不落半截写）；显式备份走 ``backup()``
    （sqlite3 backup API，WAL 安全）。遗留 results.json 读写通道已删除
    （本项目不做版本兼容）。
"""

from __future__ import annotations

import json
import sqlite3 as _sqlite3
import time
from pathlib import Path

from sqlalchemy import JSON, Column, delete, func
from sqlmodel import Field, SQLModel
from sqlmodel import select as _select

from llmsec.core import config as _config
from llmsec.core.io import read_json
from llmsec.core.logging import get_logger
from llmsec.core.results import MatchResult, ResultsMatrix
from llmsec.storage import db as _db

logger = get_logger(__name__)


class Observation(SQLModel, table=True):
    """一条 (record × model) 攻击观测——R 矩阵的原子行。"""

    __tablename__ = "observations"

    record: str = Field(primary_key=True)
    model: str = Field(primary_key=True)
    eval_score: float
    status: str = ""
    ts_json: str = "null"  # json.dumps(ts)：int/float/str/None 类型逐位保真
    ins_order: int = 0     # 全局插入序（ordered_results 平局稳定排序）
    extra: dict | None = Field(default=None, sa_column=Column(JSON))


class RUnit(SQLModel, table=True):
    """评级单位（簇）目录，保序。"""

    __tablename__ = "runits"

    unit: str = Field(primary_key=True)
    position: int = 0


class RModel(SQLModel, table=True):
    """已观测模型目录，保序（all_models 的 _models 序）。"""

    __tablename__ = "rmodels"

    model: str = Field(primary_key=True)
    position: int = 0


class EloCache(SQLModel, table=True):
    """Elo 派生缓存行（P2：原 elo_cache.json 表化——指纹命中 + 事务 upsert，
    取代文件锁 RMW）。payload 含 _version（schema 漂移即条目作废）。"""

    __tablename__ = "elo_cache"

    model: str = Field(primary_key=True)
    fingerprint: str
    payload: dict = Field(default_factory=dict, sa_column=Column(JSON))
    updated_at: float = 0.0


# ============================================================
# 路径归一
# ============================================================

def results_db() -> Path:
    """R 观测表所在的库路径（P7 统一库：与目录库同一文件 catalog.db）。

    保留函数名作为语义入口（R 域的默认库），实现委托 catalog_db——
    work-dir 卫星库里 observations 与 runs/tasks 行同库。
    """
    return _db.catalog_db()


def _as_db_path(path: Path | str | None) -> Path:
    """路径归一：None → 当前进程的 R 真相库（work-dir 隔离经 config 重绑）。"""
    return Path(path) if path is not None else results_db()


# A-13：quick_check 是整库扫描——load_matrix 高频调用（publish 每模型一次、
# safe_twin._asr_from_results 裸调等），每次全检纯开销。进程内按路径记忆
# "已检通过"，同一库只检一次（重启/换库/work-dir 卫星库各自独立首次校验）。
_quick_check_ok: set[str] = set()


def _quick_check(dbp: Path) -> None:
    """SQLite 完整性快检（替代 .bak/.corrupt.bak 机器）。损坏抛 RuntimeError。"""
    # timeout=5：写锁竞争时等待而非立刻 OperationalError——否则锁竞争会被
    # 误报成"库损坏"
    key = str(dbp)
    if key in _quick_check_ok:
        return
    conn = _sqlite3.connect(str(dbp), timeout=5.0)
    try:
        try:
            row = conn.execute("PRAGMA quick_check").fetchone()
        except _sqlite3.DatabaseError as e:
            # 非 db 文件（伪装后缀/半截拷贝）——统一按损坏上报
            raise RuntimeError(f"R 库完整性校验失败: {dbp}: {e}") from e
        if row and row[0] != "ok":
            raise RuntimeError(f"R 库完整性校验失败: {dbp}: {row[0]}")
        _quick_check_ok.add(key)
    finally:
        conn.close()


# ============================================================
# 读写实现
# ============================================================

def load_matrix(path: Path | str | None = None) -> ResultsMatrix:
    """全量构建内存矩阵（13 处调用点零改动——ResultsMatrix.load 委托此处）。"""
    dbp = _as_db_path(path)
    if not dbp.exists():
        return ResultsMatrix()  # 空库语义（与旧"文件不存在=空矩阵"一致）
    _quick_check(dbp)
    mat = ResultsMatrix()
    with _db.session(dbp) as s:
        obs = s.exec(_select(Observation).order_by(Observation.ins_order)).all()
        for o in obs:
            ts = json.loads(o.ts_json)
            mat._r.setdefault(o.record, {})[o.model] = MatchResult(
                o.record, o.model, float(o.eval_score), o.status or "", ts,
                dict(o.extra or {}),
            )
        for u in s.exec(_select(RUnit).order_by(RUnit.position)).all():
            mat._units.append(u.unit)
        for m in s.exec(_select(RModel).order_by(RModel.position)).all():
            mat._models.append(m.model)
        mat._ins_order = obs[-1].ins_order if obs else 0
    return mat


def save_matrix(matrix: ResultsMatrix, path: Path | str | None = None) -> Path:
    """全量覆写（单事务）。供 ResultsMatrix.save 与快照/工作区写入。"""
    dbp = _as_db_path(path)
    with _db.tx(dbp) as s:
        # bulk DELETE：不再全量载入 ORM 行逐条 delete（大 R 库下省一次全表物化）
        s.execute(delete(Observation))
        s.execute(delete(RUnit))
        s.execute(delete(RModel))
        order = 0
        for record, col in matrix._r.items():
            for model, res in col.items():
                order += 1
                s.add(Observation(
                    record=record, model=model, eval_score=float(res.eval_score),
                    status=res.status or "", ts_json=json.dumps(res.ts),
                    ins_order=order, extra=dict(res.extra or {}) or None,
                ))
        for i, u in enumerate(matrix._units):
            s.add(RUnit(unit=u, position=i))
        for i, m in enumerate(matrix.all_models()):
            s.add(RModel(model=m, position=i))
    return dbp


def upsert_observations(items: list[MatchResult], path: Path | str | None = None) -> int:
    """增量 upsert（单事务）——publish_tracker / merge 的并发安全写入路径。

    已存在的 (record, model) 覆盖内容、保留原 ins_order（时序平局稳定）；
    新观测接续全局计数器。返回写入条数。
    """
    if not items:
        return 0
    dbp = _as_db_path(path)
    with _db.tx(dbp) as s:
        order = s.exec(
            _select(Observation.ins_order).order_by(Observation.ins_order.desc())
        ).first() or 0
        for it in items:
            existing = s.get(Observation, (it.record, it.model))
            if existing is not None:
                existing.eval_score = float(it.eval_score)
                existing.status = it.status or ""
                # ts=None（publish_tracker 恒缺省）不覆盖已有数值 ts——重发布
                # 不应把该观测在 Elo 回放时间轴上挪到末尾
                if it.ts is not None:
                    existing.ts_json = json.dumps(it.ts)
                existing.extra = dict(it.extra or {}) or None
                s.add(existing)
            else:
                order = (order or 0) + 1
                s.add(Observation(
                    record=it.record, model=it.model,
                    eval_score=float(it.eval_score), status=it.status or "",
                    ts_json=json.dumps(it.ts), ins_order=order,
                    extra=dict(it.extra or {}) or None,
                ))
        for m in {it.model for it in items}:
            if s.get(RModel, m) is None:
                last = s.exec(
                    _select(RModel.position).order_by(RModel.position.desc())
                ).first() or -1
                s.add(RModel(model=m, position=last + 1))
    return len(items)


def remove_models(models: list[str], path: Path | str | None = None) -> int:
    """删除模型列（单事务）。返回删除的观测条数（=旧 remove_model 之和）。"""
    dbp = _as_db_path(path)
    with _db.tx(dbp) as s:
        n = 0
        for m in models:
            n += s.execute(
                delete(Observation).where(Observation.model == m)).rowcount or 0
            rm = s.get(RModel, m)
            if rm is not None:
                s.delete(rm)
        return n


def set_units(units: list[str], path: Path | str | None = None) -> None:
    """评级单位目录覆写（set_unit_catalog 的落库路径，单事务）。"""
    dbp = _as_db_path(path)
    with _db.tx(dbp) as s:
        s.execute(delete(RUnit))
        for i, u in enumerate(units):
            s.add(RUnit(unit=u, position=i))


def backup(dest: Path | str, path: Path | str | None = None) -> Path:
    """sqlite3 backup API 备份（WAL 安全，替代 .bak 轮转）。

    用独立连接而非引擎池里的连接——池连接归池管理，借出备份会与借还语义
    冲突，且引擎持有句柄时 Windows 上无法删除/改名库文件（回滚路径依赖此点）。
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    src_conn = _sqlite3.connect(str(_as_db_path(path)))
    try:
        out = _sqlite3.connect(str(dest))
        try:
            src_conn.backup(out)
        finally:
            out.close()
    finally:
        # mkdir/第二个 connect 抛错也必须关 src——Windows 上滞留句柄会挡住
        # 后续对该库的删除/改名
        src_conn.close()
    return dest


def get_elo_cache(model: str, path: Path | str | None = None) -> tuple[str, dict] | None:
    """取某模型的派生缓存行：返回 (fingerprint, payload)；无行返回 None。"""
    dbp = _as_db_path(path)
    if not dbp.exists():
        return None
    with _db.session(dbp) as s:
        row = s.get(EloCache, model)
        return (row.fingerprint, dict(row.payload)) if row is not None else None


def upsert_elo_cache(model: str, fingerprint: str, payload: dict,
                     path: Path | str | None = None) -> None:
    """派生缓存 upsert（单事务——文件锁 RMW 的替代）。"""
    dbp = _as_db_path(path)
    with _db.tx(dbp) as s:
        row = s.get(EloCache, model)
        if row is None:
            row = EloCache(model=model, fingerprint=fingerprint,
                           payload=dict(payload), updated_at=time.time())
        else:
            row.fingerprint = fingerprint
            row.payload = dict(payload)
            row.updated_at = time.time()
        s.add(row)


def close_db(path: Path | str | None = None) -> None:
    """释放该 R 库的引擎句柄（Windows 上持句柄删不掉文件——delete/gc 前调用）。"""
    _db.close(_as_db_path(path))


def results_stats(path: Path | str | None = None) -> dict:
    """R 库统计（fork 记索引 / 快照 manifest 用）：{models, records, observations, units}。"""
    dbp = _as_db_path(path)
    if not dbp.exists():
        return {"models": [], "records": 0, "observations": 0, "units": 0}
    with _db.session(dbp) as s:
        models = [m.model for m in s.exec(_select(RModel).order_by(RModel.position)).all()]
        n_obs = s.exec(_select(func.count()).select_from(Observation)).one()
        n_rec = s.exec(
            _select(func.count(func.distinct(Observation.record))).select_from(Observation)
        ).one()
        n_units = s.exec(_select(func.count()).select_from(RUnit)).one()
        return {"models": models, "records": n_rec,
                "observations": n_obs, "units": n_units}


def clone_from_run(run_name: str, dest: Path | str) -> dict:
    """从 run 的 state.json 重建 R 并落库到 dest（fork 的 run:<name> 源）。

    返回 results_stats(dest)。state.json 是 ELOTracker 快照（history 含每场
    对局），重建为 record→model→MatchResult；模型名恒从 runner_report 取
    （ELOTracker.save 不写 defender_name 键）。
    """
    from llmsec.core.paths import safe_subpath

    parts = run_name.split("/")
    run_dir = safe_subpath(Path(_config.RUNS_DIR), *parts)
    state_path = run_dir / "state.json"
    if not state_path.exists() and len(parts) > 1:
        # 旧布局（gen1/2 扁平）：run 目录为首段；单段名时 fallback 与首选同路径，
        # 无需再试。runner_report 必须同步换到旧布局目录，否则 target_model 恒 None。
        run_dir = safe_subpath(Path(_config.RUNS_DIR), parts[0])
        state_path = run_dir / "state.json"
    if not state_path.exists():
        raise FileNotFoundError(
            f"run {run_name!r} 无 state.json，无法重建 R。"
            "（仅 global 源或含 state.json 的 run 可导出）"
        )
    state = read_json(state_path) or {}
    report = read_json(run_dir / "runner_report.json") or {}
    model = report.get("target_model")
    history = state.get("history", [])
    if not model and any(h.get("defender") is None for h in history):
        # A-3：runner_report 缺失/损坏且 history 行无 defender 键——此前 def_ 恒
        # None、所有行被 continue 静默跳过，save_matrix 写出空 R、fork/快照带着
        # 空数据继续走（静默数据丢失）。与 state.json 缺失同等对待：显式失败。
        raise ValueError(
            f"run {run_name!r} 无 runner_report.json（target_model 不可得）且 "
            "history 缺 defender 键，无法归属模型列——拒绝导出空 R。"
            "（run 产物可能损坏；用 llmsec-manage runs list 检查该 run）")
    mat = ResultsMatrix()
    for h in history:
        rec = h.get("record")
        def_ = h.get("defender") or model
        if not rec or not def_:
            continue
        mat.upsert(
            record=rec, model=def_,
            eval_score=float(h.get("eval_score") or 0.0),
            status=h.get("status", ""),
            ts=h.get("round"),
            extra={"unit": h.get("unit"), "round": h.get("round")},
        )
    save_matrix(mat, path=dest)
    return results_stats(dest)
