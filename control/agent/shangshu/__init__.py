"""control.agent.shangshu — 尚书省（执行调度层）。

职责（三省制中的尚书省）：
  - 拟案：收中书省转交的用户意图，基于完整能力清单，产出结构化 Plan（步骤 + 依赖）。
  - 执行：用户准奏后，拓扑分层执行 Plan，每步经门下省审查，分批汇报进度。
  - 汇报：执行完毕经总线通知门下省审查呈递简报。

尚书省有**完整的能力文档**（capabilities.py + docs.py），是三省中阅读理解能力最强的执行单元。
中书省只有简略概览，判断复杂度后转交尚书省。

对外接口：
  draft_plan(intent, context) → Plan          拟案（planner.py）
  execute_plan(plan_id, on_progress) → Plan   执行（executor.py）
  approve_plan(plan_id) → Plan                用户准奏（设状态）
  reject_plan(plan_id) → Plan                 用户驳回
  load_plan(plan_id) / list_plans()           查询
"""

from __future__ import annotations

from control.agent.shangshu.executor import execute_plan
from control.agent.shangshu.plan import (
    P_APPROVED,
    P_DONE,
    P_DRAFTED,
    P_EXECUTING,
    P_REJECTED,
    Plan,
    Step,
    list_plans,
    load_plan,
    make_plan_from_llm,
    reset_plans,
    save_plan,
)
from control.agent.shangshu.planner import draft_plan


def approve_plan(plan_id: str) -> Plan:
    """用户准奏 Plan：状态 drafted → approved + 文牍记录 + 总线通知。"""
    plan = load_plan(plan_id)
    if plan is None:
        raise KeyError(f"Plan 不存在: {plan_id}")
    if plan.status != P_DRAFTED:
        raise RuntimeError(f"Plan 状态不允许准奏: {plan.status}（需 {P_DRAFTED}）")
    import time
    plan.status = P_APPROVED
    plan.approved = time.time()
    save_plan(plan)
    from control.agent import gazette
    from control.agent.bus import KIND_PLAN_APPROVED, SHANGSHU, USER, notify
    gazette.append_event(plan.id, gazette.EV_PLAN_APPROVED, USER,
                         session_id=plan.session_id, intent=plan.intent,
                         detail={"approved_at": plan.approved})
    notify(KIND_PLAN_APPROVED, from_dept=USER, to_dept=SHANGSHU,
           plan_id=plan.id, intent=plan.intent, session_id=plan.session_id)
    return plan


def reject_plan(plan_id: str) -> Plan:
    """用户驳回 Plan + 文牍记录 + 总线通知。"""
    plan = load_plan(plan_id)
    if plan is None:
        raise KeyError(f"Plan 不存在: {plan_id}")
    plan.status = P_REJECTED
    save_plan(plan)
    from control.agent import gazette
    from control.agent.bus import KIND_PLAN_REJECTED, USER, notify
    gazette.append_event(plan.id, gazette.EV_PLAN_REJECTED, USER,
                         session_id=plan.session_id, intent=plan.intent,
                         detail={})
    notify(KIND_PLAN_REJECTED, from_dept=USER, plan_id=plan.id,
           intent=plan.intent, session_id=plan.session_id)
    return plan


__all__ = [
    "draft_plan", "execute_plan", "approve_plan", "reject_plan",
    "load_plan", "list_plans", "save_plan", "reset_plans",
    "make_plan_from_llm",
    "Plan", "Step",
    "P_DRAFTED", "P_APPROVED", "P_EXECUTING", "P_DONE", "P_REJECTED",
]
