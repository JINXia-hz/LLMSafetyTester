"""storage.models — 目录库表模型（SQLModel：模型即 schema）。

此前"一次运行"没有统一模型：management/data_query/control 各返回 ad-hoc
dict，字段集漂移（management 多 size/security_level，data_query 多
has_tree/has_cluster）。本模块的表模型是唯一 schema 定义处，``as_dict()``
输出旧两套实现的**超集**，消费方（dashboard API / management CLI /
control / MCP / TUI）零字段损失。

表语义（目录库是**可重建的派生索引**，真相在文件）：
  - runs：一次评估运行的登记行。真相 = runs 树目录 + 产物文件。
  - trials：HPO trial 登记。真相 = output/experiments/<study>/trials.jsonl。
  - tasks：后台任务登记。跨进程真相 = output/tasks/<id>.meta.json（PID 通道）。
"""

from __future__ import annotations

import json
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
        """与 study 侧 trial record 形状兼容（"trial" 键 = idx）。"""
        return {
            "study": self.study,
            "trial": self.idx,
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
    """tasks 表：后台任务登记（跨进程真相仍是 meta.json，本表让查询免扫目录）。"""

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


# ============================================================
# 纯助手（不落库）
# ============================================================

def run_name(batch: str, target: str | None) -> str:
    """构造 run 名（management.runs._run_entry 的 'batch/target' 口径）。"""
    return f"{batch}/{target}" if target and batch != target else batch


def metrics_to_json(metrics: dict | None) -> str | None:
    """metrics dict → JSON 字符串（verify/dump 对账用；ORM 路径自动走 JSON 列）。"""
    return json.dumps(metrics, ensure_ascii=False) if metrics else None


def dir_size(path: Path) -> int:
    """递归目录大小（management.common.dir_size 的等价实现）。

    storage 包自带一份：catalog 扫描是唯一调用方，避免反向 import
    management（DAO 层不依赖 service 层）。
    """
    total = 0
    for p in Path(path).rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total
