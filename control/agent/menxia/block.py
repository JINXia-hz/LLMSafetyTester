"""control.agent.menxia.block — 封驳令管理。

BlockTicket 附在某个被执行步骤上，等用户确认（准奏放行 / 作罢跳过）。
封驳令按 (plan_id, step_id) 索引，内存存储（进程生命周期内有效）。
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict, dataclass


@dataclass
class BlockTicket:
    """门下省封驳令（附在某步骤上，待用户确认）。"""
    token: str
    plan_id: str
    step_id: str
    capability: str
    risk_level: str
    summary: str         # 一句话劝谏标题
    detail: str          # 详细影响说明
    created: float

    def to_dict(self) -> dict:
        return asdict(self)


# _TICKETS: key=(plan_id, step_id) → BlockTicket
# 持锁访问：issue_block 在 executor 线程池的总线回调里执行，list_pending_blocks
# 在 API 线程里迭代 values()——无锁并发可触发 "dictionary changed size during iteration"
_TICKETS: dict[tuple[str, str], BlockTicket] = {}
_LOCK = threading.Lock()


def issue_block(plan_id: str, step_id: str, capability: str, risk_level: str,
                assessment: dict) -> BlockTicket:
    """对一个危险步骤发封驳令。"""
    ticket = BlockTicket(
        token=uuid.uuid4().hex[:12],
        plan_id=plan_id, step_id=step_id,
        capability=capability, risk_level=risk_level,
        summary=assessment["summary"], detail=assessment["detail"],
        created=time.time(),
    )
    with _LOCK:
        _TICKETS[(plan_id, step_id)] = ticket
    return ticket


def get_block(plan_id: str, step_id: str) -> BlockTicket | None:
    with _LOCK:
        return _TICKETS.get((plan_id, step_id))


def clear_block(plan_id: str, step_id: str) -> bool:
    """用户准奏后清除封驳（让 executor 重试该步）。返回是否清除成功。"""
    with _LOCK:
        return _TICKETS.pop((plan_id, step_id), None) is not None


def clear_all_for_plan(plan_id: str) -> int:
    """清除某 Plan 的所有封驳（plan 驳回/重置时）。"""
    with _LOCK:
        keys = [k for k in _TICKETS if k[0] == plan_id]
        for k in keys:
            del _TICKETS[k]
    return len(keys)


def approve_block(plan_id: str, step_id: str) -> bool:
    """用户准奏某步的封驳（= clear_block，executor 重入时重试）。"""
    return clear_block(plan_id, step_id)


def list_pending_blocks() -> list[dict]:
    """列出所有待确认封驳（供前端展示）。"""
    with _LOCK:
        return [t.to_dict() for t in _TICKETS.values()]


def reset_blocks() -> None:
    """清空所有封驳（测试用）。"""
    with _LOCK:
        _TICKETS.clear()
