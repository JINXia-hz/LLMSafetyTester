"""storage.rstore — R 矩阵（唯一真相）的 SQLite 后端（数据库重构阶段 2）。

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
  - **遗留 json 自迁移**：db 缺失而旁有 results.json（旧 workspace/全局
    存量）时，load 首次自动导入；显式 ``.json`` 路径的 load/save 走
    遗留 JSON 读写（快照导出格式，人读友好）。
  - 损坏处置：load 时 PRAGMA quick_check 轻量校验（替代 .bak/.corrupt.bak
    机器——SQLite 事务本身保证不落半截写）；显式备份走 ``backup()``
    （sqlite3 backup API，WAL 安全）。
"""

from __future__ import annotations

import json
import sqlite3 as _sqlite3
from pathlib import Path

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel
from sqlmodel import select as _select

from llmsec.core import config as _config
from llmsec.core.io import CorruptedFileError, read_json, write_json
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
# 路径与遗留迁移
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


def matrix_from_legacy_json(filepath: Path) -> ResultsMatrix:
    """读遗留 results.json → 内存矩阵（保留 F1/F-3 的损坏处置语义）。

    - v2：正常解析。
    - v1（method 键纪元）：归档为同目录 results.method-era.bak 后返回空矩阵。
    - 损坏：残文件备份为 <name>.corrupt.bak 供取证 → 尝试 <name>.bak 恢复 →
      仍失败 raise CorruptedFileError（不返空矩阵防"空矩阵被写回=永久丢失"）。
    """
    filepath = Path(filepath)
    data = None
    try:
        data = read_json(filepath, strict=True)
    except CorruptedFileError as e:
        logger.error("遗留 results.json 损坏: %s。原因: %s", filepath, e.cause)
        try:
            import shutil
            shutil.copy2(filepath, str(filepath) + ".corrupt.bak")
        except OSError:
            pass
        bak = filepath.with_suffix(filepath.suffix + ".bak")
        if bak.exists():
            try:
                data = read_json(bak, strict=True)
                logger.warning("已从备份 %s 恢复", bak)
            except CorruptedFileError:
                data = None
        if data is None:
            raise
    if not data:
        return ResultsMatrix()
    if data.get("version") != 2:
        # v1（method 键）废弃：归档后重建（行键是方法名，无法还原记录级观测）
        try:
            import shutil
            bak = filepath.with_name("results.method-era.bak")
            shutil.copy2(filepath, bak)
            logger.warning("遗留 results.json 为 v1 schema，已归档为 %s 并按 v2 重建", bak)
        except OSError:
            pass
        return ResultsMatrix()
    mat = ResultsMatrix(units=data.get("units", []), models=data.get("models", []))
    for record, col in data.get("results", {}).items():
        for model, d in col.items():
            try:
                res = MatchResult.from_store(record, model, d)
            except ValueError as e:
                logger.warning("跳过损坏记录: %s", e)
                continue
            mat._r.setdefault(record, {})[model] = res
            try:
                t = float(res.ts) if res.ts is not None else 0
            except (TypeError, ValueError):
                t = 0
            if t > mat._ins_order:
                mat._ins_order = int(t)
    return mat


def _quick_check(dbp: Path) -> None:
    """SQLite 完整性快检（替代 .bak/.corrupt.bak 机器）。损坏抛 RuntimeError。"""
    conn = _sqlite3.connect(str(dbp))
    try:
        try:
            row = conn.execute("PRAGMA quick_check").fetchone()
        except _sqlite3.DatabaseError as e:
            # 非 db 文件（伪装后缀/半截拷贝）——统一按损坏上报
            raise RuntimeError(f"R 库完整性校验失败: {dbp}: {e}") from e
        if row and row[0] != "ok":
            raise RuntimeError(f"R 库完整性校验失败: {dbp}: {row[0]}")
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
        for row in s.exec(_select(Observation)).all():
            s.delete(row)
        for row in s.exec(_select(RUnit)).all():
            s.delete(row)
        for row in s.exec(_select(RModel)).all():
            s.delete(row)
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
            rows = s.exec(_select(Observation).where(Observation.model == m)).all()
            n += len(rows)
            for row in rows:
                s.delete(row)
            rm = s.get(RModel, m)
            if rm is not None:
                s.delete(rm)
        return n


def set_units(units: list[str], path: Path | str | None = None) -> None:
    """评级单位目录覆写（set_unit_catalog 的落库路径，单事务）。"""
    dbp = _as_db_path(path)
    with _db.tx(dbp) as s:
        for row in s.exec(_select(RUnit)).all():
            s.delete(row)
        for i, u in enumerate(units):
            s.add(RUnit(unit=u, position=i))


def backup(dest: Path | str, path: Path | str | None = None) -> Path:
    """sqlite3 backup API 备份（WAL 安全，替代 .bak 轮转）。

    用独立连接而非引擎池里的连接——池连接归池管理，借出备份会与借还语义
    冲突，且引擎持有句柄时 Windows 上无法删除/改名库文件（回滚路径依赖此点）。
    """
    src_conn = _sqlite3.connect(str(_as_db_path(path)))
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out = _sqlite3.connect(str(dest))
    try:
        src_conn.backup(out)
    finally:
        out.close()
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
    import time as _time
    dbp = _as_db_path(path)
    with _db.tx(dbp) as s:
        row = s.get(EloCache, model)
        if row is None:
            row = EloCache(model=model, fingerprint=fingerprint,
                           payload=dict(payload), updated_at=_time.time())
        else:
            row.fingerprint = fingerprint
            row.payload = dict(payload)
            row.updated_at = _time.time()
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
        obs = s.exec(_select(Observation)).all()
        models = [m.model for m in s.exec(_select(RModel).order_by(RModel.position)).all()]
        units = s.exec(_select(RUnit)).all()
        return {"models": models, "records": len({o.record for o in obs}),
                "observations": len(obs), "units": len(units)}


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
    if not state_path.exists():
        # 旧布局
        state_path = safe_subpath(Path(_config.RUNS_DIR), parts[0]) / "state.json"
    if not state_path.exists():
        raise FileNotFoundError(
            f"run {run_name!r} 无 state.json，无法重建 R。"
            "（仅 global 源或含 state.json 的 run 可导出）"
        )
    state = read_json(state_path) or {}
    model = read_json(run_dir / "runner_report.json") or {}
    model = model.get("target_model")
    mat = ResultsMatrix()
    for h in state.get("history", []):
        rec = h.get("record")
        def_ = h.get("defender") or model
        if not rec or not def_:
            continue
        mat.upsert(
            record=rec, model=def_,
            eval_score=float(h.get("eval_score", 0.0)),
            status=h.get("status", ""),
            ts=h.get("round"),
            extra={"unit": h.get("unit"), "round": h.get("round")},
        )
    save_matrix(mat, path=dest)
    return results_stats(dest)


def export_legacy_json(out_path: Path | str, path: Path | str | None = None) -> Path:
    """导出人读快照 results.json（派生产物；快照/分发用）。"""
    mat = load_matrix(path)
    out = Path(out_path)
    write_json(out, mat.to_store_dict(), allow_nan=False)
    return out
