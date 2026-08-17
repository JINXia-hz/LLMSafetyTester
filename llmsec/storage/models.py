"""storage.models — 目录库表模型（SQLModel：模型即 schema）。

此前"一次运行"没有统一模型：management/data_query/control 各返回 ad-hoc
dict，字段集漂移（management 多 size/security_level，data_query 多
has_tree/has_cluster）。本模块的表模型是唯一 schema 定义处，``as_dict()``
输出旧两套实现的**超集**，消费方（dashboard API / management CLI /
control / MCP / TUI）零字段损失。

表语义（P9 所有权翻转：db 独占状态，文件只做制品/blob/进程通道）：
  - runs：评估运行登记。register（创建）→ finalize（报告后富化）→ remove
    （软删）全生命周期由写入口维护；报告/树/md 等制品仍在 run 目录。
  - trials：HPO trial 登记。db 唯一真相；旧 trials.jsonl 仅一次性导入源。
  - tasks：后台任务登记。db 唯一跨进程真相；.log/.progress.jsonl 是日志通道。
  - probes：模型防御指纹（原 state/probes.json 表化）。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


def _iso(ts: float | None) -> str:
    """epoch → ISO 字符串（旧 discover 系实现的 mtime 口径）。None → 空串。"""
    if ts is None:
        return ""
    return datetime.fromtimestamp(ts).isoformat()


class Run(SQLModel, table=True):
    """runs 表：一次评估运行的登记项。

    name：'batch/target'（gen3 全局布局）/'batch'（gen1/2 扁平布局）/target
    目录名（work-dir/workspace 卫星库布局）。
    layout：0=登记时尚无产物（进行中）1/2=扁平世代 3=<ts>/<target> 世代。
    dir_path：绝对路径——派生索引可随时重建，不追求仓库搬迁后幸存。
    """

    __tablename__ = "runs"

    name: str = Field(primary_key=True)
    batch: str = Field(index=True)
    target: str
    layout: int = 0
    dir_path: str
    mtime: float
    registered_at: float
    target_model: str | None = None
    security_level: str | None = None
    has_report: bool = False
    has_md: bool = False
    has_tree: bool = False
    has_cluster: bool = False
    has_artifact: bool = False   # RUN_ARTIFACTS 九种产物任一存在（management 发现口径）
    size: int | None = None
    metrics: dict | None = Field(default=None, sa_column=Column(JSON))

    def as_dict(self) -> dict:
        """旧 management.discover_runs / data_query._discover_runs 的字段超集。"""
        d = {
            "name": self.name,
            "batch": self.batch,
            "target": self.target,
            "target_model": self.target_model if self.target_model is not None else self.target,
            "security_level": self.security_level or "inconclusive",
            "has_report": self.has_report,
            "has_md": self.has_md,
            "has_tree": self.has_tree,
            "has_cluster": self.has_cluster,
            "mtime": _iso(self.mtime),
            "size": self.size if self.size is not None else 0,
            "layout": self.layout,
            "dir_path": self.dir_path,
        }
        if self.metrics:
            d.update(self.metrics)
        return d


class Trial(SQLModel, table=True):
    """trials 表：HPO trial 记录（P4 起唯一真相——原 trials.jsonl append-only 退役，
    断点续跑改读表；旧 jsonl 经 study.load_trial_records 一次性导入）。"""

    __tablename__ = "trials"

    study: str = Field(primary_key=True)
    idx: int = Field(primary_key=True)
    work_dir: str
    registered_at: float
    target: str | None = None
    seed: str | None = None
    status: str | None = None
    metrics: dict | None = Field(default=None, sa_column=Column(JSON))
    updated_at: float | None = None
    # P4 扩列（db.ensure_columns 对旧库自动 ALTER ADD）
    params: dict | None = Field(default=None, sa_column=Column(JSON))
    search_fp: str | None = None
    search_params: dict | None = Field(default=None, sa_column=Column(JSON))
    returncode: int | None = None
    error: str | None = None
    elapsed_s: float | None = None

    def as_dict(self) -> dict:
        """与 study 侧 trial record 形状兼容（键统一为 idx）。"""
        return {
            "study": self.study,
            "idx": self.idx,
            "target": self.target,
            "seed": self.seed,
            "work_dir": self.work_dir,
            "status": self.status,
            "metrics": self.metrics or {},
            "params": self.params or {},
            "search_fp": self.search_fp,
            "search_params": self.search_params or {},
            "returncode": self.returncode,
            "error": self.error,
            "elapsed_s": self.elapsed_s,
        }


class Task(SQLModel, table=True):
    """tasks 表：后台任务登记（P4 起库行即跨进程唯一真相；legacy meta.json 经 reconcile 吸收）。"""

    __tablename__ = "tasks"

    task_id: str = Field(primary_key=True)
    kind: str
    status: str = Field(index=True)
    registered_at: float
    cmd: str | None = None
    pid: int | None = None
    log_path: str | None = None
    started_at: str | None = None
    meta: dict | None = Field(default=None, sa_column=Column(JSON))
    updated_at: float | None = None

    def as_dict(self) -> dict:
        return {
            "id": self.task_id,
            "kind": self.kind,
            "cmd": self.cmd or "",
            "pid": self.pid,
            "status": self.status,
            "log_path": self.log_path,
            "started_at": self.started_at or _iso(self.registered_at),
            "meta": self.meta,
        }


class PredictorCache(SQLModel, table=True):
    """预测器登记行（P8：真 LRU——last_hit 行内记录，取代 mtime-touch 近似。

    blob 仍是 predictors/<key>.pkl 文件（登记只存元数据）；行是可重建的
    派生索引：删库后 prune 前对账重建（created 用文件 mtime 兜底）。"""

    __tablename__ = "predictor_cache"

    key: str = Field(primary_key=True)   # cache_key（文件名去 .pkl）
    size: int = 0
    created: float = 0.0
    last_hit: float = 0.0
    hits: int = 0


class Probe(SQLModel, table=True):
    """模型防御指纹（P9 表化——原 state/probes.json 的 RMW 文件退役，
    单事务 upsert 跨进程竞态消失。经 config 重绑自动落 work-dir 卫星库）。"""

    __tablename__ = "probes"

    model: str = Field(primary_key=True)
    fingerprint: dict = Field(default_factory=dict, sa_column=Column(JSON))
    seed_methods: list = Field(default_factory=list, sa_column=Column(JSON))
    n: int = 0
    computed_at: str = ""


# ============================================================
# control 侧表（P5：control 层数据库化——原 _index.json×3 / gazette jsonl /
# plans json / 内存 tickets / queue 落库；经 storage.contract 供 control 消费）
# ============================================================

class CtlEvent(SQLModel, table=True):
    """文牍事件流（append-only INSERT；自增 id 即事件序）。"""

    __tablename__ = "ctl_events"

    id: int | None = Field(default=None, primary_key=True)
    ts: float = Field(index=True)
    kind: str = Field(index=True)
    dept: str
    plan_id: str = Field(index=True)
    step_id: str | None = None
    session_id: str | None = Field(default=None, index=True)
    detail: dict | None = Field(default=None, sa_column=Column(JSON))

    def event_dict(self) -> dict:
        """与原 GazetteEvent.to_dict() 形状一致（消费方零改动）。"""
        return {
            "ts": self.ts, "kind": self.kind, "dept": self.dept,
            "plan_id": self.plan_id, "step_id": self.step_id,
            "session_id": self.session_id, "detail": self.detail or {},
        }


class CtlPlanMeta(SQLModel, table=True):
    """文牍 Plan 元数据（原 gazette/_index.json 的 plans 项）。"""

    __tablename__ = "ctl_plan_meta"

    plan_id: str = Field(primary_key=True)
    intent: str = ""
    session_id: str | None = None
    created: float = 0.0
    status: str = "active"
    last_event: str = ""
    last_ts: float = 0.0
    finished: float | None = None

    def as_dict(self) -> dict:
        d = {
            "plan_id": self.plan_id, "intent": self.intent,
            "session_id": self.session_id, "created": self.created,
            "status": self.status, "last_event": self.last_event,
            "last_ts": self.last_ts,
        }
        if self.finished is not None:
            d["finished"] = self.finished
        return d


class CtlPlan(SQLModel, table=True):
    """三省 Plan 快照（整 plan 一个 JSON blob 行——executor 线程池内改共享
    对象后整体覆写，与原 save_plan 语义一致且原子）。"""

    __tablename__ = "ctl_plans"

    id: str = Field(primary_key=True)
    intent: str = ""
    status: str = ""
    session_id: str | None = None
    created: float = 0.0
    updated_at: float = 0.0
    payload: dict = Field(default_factory=dict, sa_column=Column(JSON))  # 完整 to_dict

    def as_dict(self) -> dict:
        return dict(self.payload)


class CtlTicket(SQLModel, table=True):
    """封驳令（P5 落库：原内存 _TICKETS 重启即丢——封驳静默放行的安全修复）。

    列形状与 BlockTicket.to_dict() 一致（control.agent.menxia.block）。"""

    __tablename__ = "ctl_tickets"

    plan_id: str = Field(primary_key=True)
    step_id: str = Field(primary_key=True)
    token: str = ""
    capability: str = ""
    risk_level: str = ""
    summary: str = ""
    detail: str = ""
    created: float = 0.0


class CtlQueueItem(SQLModel, table=True):
    """Plan 队列项（内容落库获重启恢复；worker 线程协议留内存）。"""

    __tablename__ = "ctl_queue"

    id: int | None = Field(default=None, primary_key=True)
    plan_id: str = Field(index=True)
    queued_at: float = 0.0
    status: str = "queued"   # queued / running / done


class CtlWorkspace(SQLModel, table=True):
    """workspace 索引（原 workspaces/_index.json；P9：gc_log 哨兵行审计链已删）。"""

    __tablename__ = "ctl_workspaces"

    name: str = Field(primary_key=True)
    path: str = ""
    source: str = ""
    note: str = ""
    created: str = ""
    models: list = Field(default_factory=list, sa_column=Column(JSON))
    records: int = 0
    merged: bool = False
    merged_at: str | None = None
    merged_to: str | None = None

    def as_dict(self) -> dict:
        d = {
            "name": self.name, "path": self.path, "source": self.source,
            "note": self.note, "created": self.created, "models": self.models or [],
            "records": self.records, "merged": self.merged,
            "merged_at": self.merged_at, "merged_to": self.merged_to,
        }
        return d


class CtlEnvSnapshot(SQLModel, table=True):
    """env_snapshot 索引（原 env_snapshots/_index.json；.env 文件本体保留）。"""

    __tablename__ = "ctl_env_snapshots"

    name: str = Field(primary_key=True)
    path: str = ""
    source: str = ""
    note: str = ""
    created: str = ""
    keys: list = Field(default_factory=list, sa_column=Column(JSON))
    merged_to_global: str | None = None

    def as_dict(self) -> dict:
        return {
            "name": self.name, "path": self.path, "source": self.source,
            "note": self.note, "created": self.created, "keys": self.keys or [],
            "merged_to_global": self.merged_to_global,
        }


# ============================================================
# 纯助手（不落库）
# ============================================================

def run_name(batch: str, target: str | None) -> str:
    """构造 run 名（management.runs._run_entry 的 'batch/target' 口径）。"""
    return f"{batch}/{target}" if target and batch != target else batch


def dir_size(path: Path) -> int:
    """递归目录大小（management.common.dir_size 的等价实现）。

    storage 包自带一份：catalog 扫描是唯一调用方，避免反向 import
    management（DAO 层不依赖 service 层）。入口语义与 common 版对齐：
    不存在返回 0、文件入口返回其大小。
    """
    path = Path(path)
    try:
        if not path.exists():
            return 0
        if path.is_file():
            return path.stat().st_size
    except OSError:
        return 0
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total
