"""control.agent.menxia.block — 封驳令管理。

BlockTicket 附在某个被执行步骤上，等用户确认（准奏放行 / 作罢跳过）。
按 (plan_id, step_id) 索引。

P5（control 库化）：落目录库 ctl_tickets 表——原内存 _TICKETS 重启即丢，
executor 的"blocked 且 ticket=None → 回 pending 重试"语义意味着进程重启
即静默放行全部封驳；落库即修复。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass

from control.core.storage import (
    clear_ticket as _clear_row,
)
from control.core.storage import (
    clear_tickets_for_plan as _clear_rows,
)
from control.core.storage import (
    get_ticket as _get_row,
)
from control.core.storage import (
    reset_tickets as _reset_rows,
)
from control.core.storage import (
    save_ticket as _save_row,
)


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
    _save_row(ticket.to_dict())
    return ticket


def get_block(plan_id: str, step_id: str) -> BlockTicket | None:
    d = _get_row(plan_id, step_id)
    return BlockTicket(**d) if d is not None else None


def clear_block(plan_id: str, step_id: str) -> bool:
    """用户准奏后清除封驳（让 executor 重试该步）。返回是否清除成功。"""
    return _clear_row(plan_id, step_id)


def clear_all_for_plan(plan_id: str) -> int:
    """清除某 Plan 的所有封驳（plan 驳回/重置时）。"""
    return _clear_rows(plan_id)


def reset_blocks() -> None:
    """清空所有封驳（测试用）。"""
    _reset_rows()
