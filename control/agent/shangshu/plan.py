"""control.agent.shangshu.plan — 结构化 Plan 数据结构。

Plan = 有序步骤列表，每步指向一个 capability + 参数 + 依赖。
执行时按依赖拓扑分层，同层步骤可并行，被封驳的步骤跳过但其依赖者也标 blocked。

持久化：Plan 存内存注册表（_PLANS），同时序列化到 output/plans/<id>.json 供前端轮询。
重启后内存清空（Plan 是短生命周期的，一次任务用完即弃；需要持久历史另做）。
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

from control.config import OUTPUT_DIR

# 步骤状态
S_PENDING = "pending"      # 未开始
S_RUNNING = "running"      # 执行中
S_DONE = "done"            # 成功
S_BLOCKED = "blocked"      # 被门下省封驳
S_FAILED = "failed"        # 执行异常
S_SKIPPED = "skipped"      # 因前置步 blocked/failed 而跳过

# Plan 状态
P_DRAFTED = "drafted"      # 已拟案，待用户准奏
P_APPROVED = "approved"    # 用户已准奏，待执行
P_EXECUTING = "executing"  # 执行中
P_DONE = "done"            # 全部完成（含部分 blocked/failed）
P_REJECTED = "rejected"    # 用户驳回


@dataclass
class Step:
    """Plan 中的一个步骤。"""
    id: str                          # "s1", "s2"...
    capability: str                  # capabilities.py 清单里的 name
    args: dict                       # 该 capability 的参数
    depends_on: list[str] = field(default_factory=list)   # 前置步骤 id（空=可立即执行）
    description: str = ""            # 人类可读的一句描述（尚书省拟案时写）
    # 运行时状态（执行器填写）
    status: str = S_PENDING
    result: dict | None = None       # 执行结果（成功时）
    error: str | None = None         # 失败原因
    ticket: dict | None = None       # 封驳令（blocked 时）
    # 执行时间戳（executor 填写）
    started: float | None = None     # 执行开始时间
    finished: float | None = None    # 执行结束时间
    block_history: list[dict] = field(default_factory=list)  # 封驳历史（append，不覆盖）


@dataclass
class Plan:
    """一个完整的执行计划。"""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    intent: str = ""                 # 用户的原始意图（中书省转交时附）
    steps: list[Step] = field(default_factory=list)
    status: str = P_DRAFTED
    created: float = field(default_factory=time.time)
    approved: float | None = None
    finished: float | None = None
    summary: str = ""                # 执行完毕的总结（门下省简报附此）
    session_id: str | None = None   # 关联 session（用户身份追溯）
    started: float | None = None    # 执行开始时间

    def topological_layers(self) -> list[list[Step]]:
        """按依赖拓扑排序，返回分层列表（同层无依赖关系，可并行）。

        实现：Kahn 算法分层——每轮取所有入度=0 的步骤为一层，移除后重复。
        被标记 blocked/failed 的步骤的依赖者不会被移除入度（它们将在执行时被判 skipped）。
        若有环（不应发生），剩余步骤作为最后一层返回。
        """
        remaining = {s.id: s for s in self.steps}
        deps = {s.id: set(s.depends_on) for s in self.steps}
        layers: list[list[Step]] = []
        while remaining:
            # 入度=0：依赖都在已分层的层里
            ready = [s for sid, s in remaining.items()
                     if all(d not in remaining for d in deps[sid])]
            if not ready:
                # 环或全部有剩余依赖（异常）——把剩余全放一层兜底
                ready = list(remaining.values())
            ready.sort(key=lambda s: s.id)
            layers.append(ready)
            for s in ready:
                del remaining[s.id]
        return layers

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "intent": self.intent,
            "status": self.status,
            "created": self.created,
            "approved": self.approved,
            "finished": self.finished,
            "started": self.started,
            "summary": self.summary,
            "session_id": self.session_id,
            "steps": [
                {
                    "id": s.id, "capability": s.capability, "args": s.args,
                    "depends_on": s.depends_on, "description": s.description,
                    "status": s.status, "result": s.result, "error": s.error,
                    "ticket": s.ticket, "started": s.started, "finished": s.finished,
                    "block_history": s.block_history,
                }
                for s in self.steps
            ],
        }


# ============================================================
# Plan 注册表（内存 + 磁盘持久化）
# ============================================================
_PLANS: dict[str, Plan] = {}
_LOCK = Lock()

_PLANS_DIR = OUTPUT_DIR / "plans"


def _plans_dir() -> Path:
    _PLANS_DIR.mkdir(parents=True, exist_ok=True)
    return _PLANS_DIR


def save_plan(plan: Plan) -> None:
    """持久化 Plan 到内存注册表 + 磁盘（output/plans/<id>.json）。"""
    with _LOCK:
        _PLANS[plan.id] = plan
    p = _plans_dir() / f"{plan.id}.json"
    p.write_text(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_plan(plan_id: str) -> Plan | None:
    """取 Plan：先查内存，没有则从磁盘加载（重启后补全）。"""
    with _LOCK:
        p = _PLANS.get(plan_id)
    if p is not None:
        return p
    # 尝试磁盘
    fp = _plans_dir() / f"{plan_id}.json"
    if not fp.exists():
        return None
    return _from_json(json.loads(fp.read_text(encoding="utf-8")))


def _from_json(d: dict) -> Plan:
    """从 dict 重建 Plan 对象。"""
    plan = Plan(
        id=d["id"], intent=d.get("intent", ""), status=d.get("status", P_DRAFTED),
        created=d.get("created", time.time()), approved=d.get("approved"),
        finished=d.get("finished"), summary=d.get("summary", ""),
        session_id=d.get("session_id"), started=d.get("started"),
    )
    for sd in d.get("steps", []):
        plan.steps.append(Step(
            id=sd["id"], capability=sd["capability"], args=sd.get("args", {}),
            depends_on=sd.get("depends_on", []), description=sd.get("description", ""),
            status=sd.get("status", S_PENDING), result=sd.get("result"),
            error=sd.get("error"), ticket=sd.get("ticket"),
            started=sd.get("started"), finished=sd.get("finished"),
            block_history=sd.get("block_history", []),
        ))
    with _LOCK:
        _PLANS[plan.id] = plan
    return plan


def list_plans(*, recent: int = 20) -> list[dict]:
    """列出最近的 Plan（按创建时间倒序，默认 20 条）。供前端展示历史。"""
    with _LOCK:
        plans = list(_PLANS.values())
    plans.sort(key=lambda p: p.created, reverse=True)
    return [p.to_dict() for p in plans[:recent]]


def reset_plans() -> None:
    """清空注册表（测试用）。"""
    with _LOCK:
        _PLANS.clear()


def make_plan_from_llm(intent: str, steps_raw: list[dict]) -> Plan:
    """从 LLM submit_plan 工具返回的原始步骤列表构造 Plan。

    steps_raw: [{id, capability, args, depends_on, description}, ...]
    自动补全缺失的 id / depends_on。
    """
    steps: list[Step] = []
    for i, sr in enumerate(steps_raw):
        sid = sr.get("id") or f"s{i + 1}"
        steps.append(Step(
            id=sid,
            capability=sr["capability"],
            args=sr.get("args", {}),
            depends_on=sr.get("depends_on", []),
            description=sr.get("description", ""),
        ))
    return Plan(intent=intent, steps=steps)
