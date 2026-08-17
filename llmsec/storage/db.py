"""storage.db — 引擎/会话基建（DAO 层唯一握 SQLAlchemy/SQLModel 的地方）。

选型（2026-08 重构定案）：SQLModel（pydantic + SQLAlchemy 2.x）——业务代码
零 SQL 字符串。本模块内仅剩两处驱动级字面量：连接 PRAGMA 与 BEGIN IMMEDIATE
（事务隔离配置，不是查询）。

并发模型：dashboard / MCP / TUI / CLI / control 多进程共开同一个库文件。
WAL 模式下读不阻塞写、写不阻塞读；写写竞争由 busy_timeout（5s）排队。

读写引擎分离（同一路径两个引擎、两个池）：
  - 读引擎：不做事务改写——SELECT 走 pysqlite 自动提交，零锁；
  - 写引擎：连接关掉隐式 BEGIN + "begin" 事件改写为 BEGIN IMMEDIATE——
    立刻拿写锁排队（防 DEFERRED 事务"读后升级写锁"的 SQLITE_BUSY 升级失败）。
    若把改写挂在唯一引擎上，只读会话也会抢写锁（曾致并发回归失败）。

Windows 连接竞态防护：多线程并发首次打开同一 WAL 库会在 -shm 创建上偶发
SQLITE_CANTOPEN——creator 在全局锁内建连接（含 PRAGMA），并带短暂重试。
连接建立是低频事件（池化复用），锁开销可忽略。

路径解析（调期动态读 config，兼容 work-dir 隔离重绑，见 core.isolation）：
  - 全局目录库：output/state/catalog.db（config.CATALOG_DB）
  - 卫星库（work-dir / workspace / trial workdir）：任意 runs 根目录下的
    <root>/catalog.db——work-dir 是自包含单元，索引随数据走。
"""

from __future__ import annotations

import sqlite3 as _sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

from llmsec.core import config as _config

_BUSY_TIMEOUT_MS = 5000
_SCHEMA_VERSION = 1  # PRAGMA user_version（降级守卫用；目录库可删库重建，无历史迁移）

_ENGINES: dict[str, dict[str, Engine]] = {}
_ENGINES_LOCK = threading.Lock()
_CONNECT_LOCK = threading.Lock()  # Windows：串行化 DBAPI 连接建立（-shm 创建竞态）


def catalog_db() -> Path:
    """当前进程的全局目录库路径（调期动态读——work-dir 隔离经 config 重绑生效）。"""
    return Path(_config.CATALOG_DB)


def db_for(runs_root: Path | str | None) -> Path:
    """runs 根目录 → 该根的目录库路径。

    全局 runs 根（output/runs/）的索引在 output/state/catalog.db（统一库）；其它根（work-dir / workspace / trial workdir）用 <root>/catalog.db
    卫星库——隔离单元自带索引，删除/归档单元时索引随之消失，不污染全局。
    """
    if runs_root is None:
        return catalog_db()
    root = Path(runs_root)
    try:
        if root.resolve() == Path(_config.RUNS_DIR).resolve():
            return catalog_db()
    except OSError:
        pass
    return root / "catalog.db"


def _make_creator(path_str: str, *, write: bool):
    """DBAPI 连接工厂：锁内建连接 + PRAGMA（连接级配置只在建立时做一次）。"""

    def creator() -> _sqlite3.Connection:
        with _CONNECT_LOCK:
            last: Exception | None = None
            for attempt in range(5):
                try:
                    conn = _sqlite3.connect(
                        path_str,
                        timeout=_BUSY_TIMEOUT_MS / 1000.0,
                        check_same_thread=False,  # 池串行借还下跨线程复用（FastAPI to_thread）
                    )
                    break
                except _sqlite3.OperationalError as e:  # Windows -shm 创建竞态的短暂 CANTOPEN
                    last = e
                    time.sleep(0.05 * (attempt + 1))
            else:
                raise last  # type: ignore[misc]
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
            cur.close()
            if write:
                # 关闭 pysqlite 隐式 BEGIN/COMMIT（SQLAlchemy 文档配方）：
                # 事务边界由写引擎的 "begin" 事件统一改写为 BEGIN IMMEDIATE
                conn.isolation_level = None
            return conn

    return creator


def _engine_for(db_path: Path | str | None, *, write: bool) -> Engine:
    path = Path(db_path) if db_path is not None else catalog_db()
    key = str(path.resolve())
    with _ENGINES_LOCK:
        # 表模型注册到 SQLModel.metadata（create_all 依赖；models 不反向依赖 db，无环）
        from llmsec.storage import models as _models  # noqa: F401

        pair = _ENGINES.get(key)
        if pair is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            pair = {}
            for w in (False, True):
                eng = create_engine(
                    "sqlite://",  # 实际路径由 creator 全权控制
                    creator=_make_creator(key, write=w),
                    pool_size=8,
                )
                if w:
                    event.listen(eng, "begin", _emit_begin_immediate)
                pair["write" if w else "read"] = eng
            # 后续任一步失败（降级守卫拒绝 / create_all 出错）都必须 dispose——
            # 半初始化的引擎不 dispose 会泄漏连接池（Windows 上还会占住库文件句柄）
            try:
                # 降级守卫：库 schema 比当前代码新 → 拒绝打开（防旧代码写坏新库）
                with pair["read"].connect() as conn:
                    v = int(conn.exec_driver_sql("PRAGMA user_version").scalar() or 0)
                    if v > _SCHEMA_VERSION:
                        raise RuntimeError(
                            f"目录库 schema v{v} 比当前代码支持的 v{_SCHEMA_VERSION} 新——"
                            "请升级代码后再访问"
                        )
                SQLModel.metadata.create_all(pair["write"])
                _ensure_columns(pair["write"])  # 模型扩列后旧库自动 ALTER ADD
                with pair["write"].begin() as conn:
                    conn.exec_driver_sql(f"PRAGMA user_version={_SCHEMA_VERSION}")
            except BaseException:
                for eng in pair.values():
                    eng.dispose()
                raise
            _ENGINES[key] = pair
        return pair["write" if write else "read"]


def _ensure_columns(eng: Engine) -> None:
    """模型新增列 → 旧库 ALTER TABLE ADD COLUMN（幂等）。

    SQLite 的 ALTER 只支持 ADD COLUMN（无约束变更），恰好覆盖"扩列不改约束"
    的演进；列删除/改型走删库重建（目录库本来就是派生索引）。
    """
    with eng.begin() as conn:
        for table in SQLModel.metadata.sorted_tables:
            existing = {row[1] for row in conn.exec_driver_sql(
                f"PRAGMA table_info({table.name})")}
            if not existing:
                continue  # 表刚建（create_all 已含全部列）
            for col in table.columns:
                if col.name in existing:
                    continue
                coltype = col.type.compile(eng.dialect)
                conn.exec_driver_sql(f"ALTER TABLE {table.name} ADD COLUMN {col.name} {coltype}")


def _emit_begin_immediate(conn) -> None:
    conn.exec_driver_sql("BEGIN IMMEDIATE")


def engine_for(db_path: Path | str | None = None) -> Engine:
    """读引擎（SELECT 自动提交，零事务改写）。引擎创建即幂等建表——任何入口
    先拿引擎，不存在"忘了初始化"的失败路径。"""
    return _engine_for(db_path, write=False)


@contextmanager
def session(db_path: Path | str | None = None) -> Iterator[Session]:
    """只读会话（autocommit SELECT）。写操作必须用 tx()。"""
    with Session(engine_for(db_path), expire_on_commit=False) as s:
        yield s


@contextmanager
def tx(db_path: Path | str | None = None) -> Iterator[Session]:
    """写事务：写引擎 + Session.begin()（首条语句触发 BEGIN IMMEDIATE）→ COMMIT/ROLLBACK。

    用法：``with db.tx() as s: s.add(run); ...``——离开 with 自动提交/回滚。
    """
    eng = _engine_for(db_path, write=True)
    with Session(eng, expire_on_commit=False) as s:
        with s.begin():
            yield s


def close(db_path: Path | str | None = None) -> None:
    """释放引擎（测试 / 路径重绑后防旧句柄滞留）。"""
    with _ENGINES_LOCK:
        if db_path is None:
            for pair in _ENGINES.values():
                for eng in pair.values():
                    eng.dispose()
            _ENGINES.clear()
            return
        key = str(Path(db_path).resolve())
        pair = _ENGINES.pop(key, None)
        if pair is not None:
            for eng in pair.values():
                eng.dispose()
