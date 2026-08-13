"""control.agent.menxia.listener — 总线订阅（门下省挂起监听）。

门下省是三省中唯一常态挂起、全局监听的部门。它订阅总线消息：
  - step_start → 封驳审查（capability.block_message 决定是否封）
  - plan_done → 自动审查呈递简报（capability.extract_review_target 提取 run 名）
  - step_failed → 异常呈递

封驳粒度：只挡该步，不依赖它的步骤继续执行。
用户准奏 → 清该步 ticket → executor 重入时重试。
"""

from __future__ import annotations

from control.agent.bus import (
    ALL,
    KIND_BLOCK,
    KIND_PLAN_DONE,
    KIND_REVIEW,
    KIND_STEP_FAILED,
    KIND_STEP_START,
    MENXIA,
    ZHONGSHU,
    BusMessage,
    get_bus,
)
from control.agent.menxia.block import (
    get_block,
    issue_block,
)


# ============================================================
# 封驳审查（数据化判据）
# ============================================================
def assess_step(cap, args: dict) -> dict | None:
    """审查一个步骤是否需要封驳。

    判据完全由 capability 自身提供（block_message 字段）：
    - cap.block_message 为 None → 永不封驳
    - cap.block_message(args) 返回 None → 本次放行
    - cap.block_message(args) 返回 dict → 封驳

    Args:
        cap: Capability 对象（来自 shangshu.capabilities）
        args: 该步骤的参数

    Returns:
        dict（含 summary/detail）= 需封驳；None = 放行
    """
    if cap.block_message is None:
        return None
    return cap.block_message(args)


# ============================================================
# 总线订阅
# ============================================================
_INITIALIZED = False


def init_menxia() -> None:
    """初始化门下省：订阅总线消息。进程启动时调一次。幂等。"""
    global _INITIALIZED
    if _INITIALIZED:
        return
    bus = get_bus()
    bus.subscribe(MENXIA, [KIND_STEP_START], _on_step_start)
    bus.subscribe(MENXIA, [KIND_PLAN_DONE], _on_plan_done)
    bus.subscribe(MENXIA, [KIND_STEP_FAILED], _on_step_failed)
    _INITIALIZED = True


def reinit_menxia() -> None:
    """强制重新订阅（测试 reset_bus 后用）。"""
    global _INITIALIZED
    _INITIALIZED = False
    init_menxia()


def _on_step_start(msg: BusMessage) -> None:
    """步骤开始前审查。dangerous → 发封驳令 + 发 KIND_BLOCK 消息。"""
    payload = msg.payload
    capability_name = payload.get("capability", "")
    risk_level = payload.get("risk_level", "low")
    plan_id = payload.get("plan_id", "")
    step_id = payload.get("step_id", "")
    args = payload.get("args", {})

    # 从 capability 清单取 block_message 判据
    from control.agent.shangshu.capabilities import capability_by_name
    cap = capability_by_name(capability_name)
    if cap is None:
        return  # 未知 capability，不审查（executor 会报错）

    assessment = assess_step(cap, args)
    if assessment is None:
        return  # 放行

    # 已有封驳令且未被清除 → 保持封驳
    existing = get_block(plan_id, step_id)
    if existing is not None:
        bus = get_bus()
        bus.publish(BusMessage(
            from_dept=MENXIA, to_dept=ALL, kind=KIND_BLOCK,
            payload={"plan_id": plan_id, "step_id": step_id,
                     "ticket": existing.to_dict()},
        ))
        return

    ticket = issue_block(plan_id, step_id, capability_name, risk_level, assessment)
    bus = get_bus()
    bus.publish(BusMessage(
        from_dept=MENXIA, to_dept=ALL, kind=KIND_BLOCK,
        payload={"plan_id": plan_id, "step_id": step_id, "ticket": ticket.to_dict()},
    ))


def _on_plan_done(msg: BusMessage) -> None:
    """Plan 执行完，自动审查所有产生的 run，呈递简报到中书省面板。"""
    payload = msg.payload
    plan_id = payload.get("plan_id", "")

    from control.agent.shangshu.capabilities import capability_by_name
    from control.agent.shangshu.plan import load_plan
    plan = load_plan(plan_id)
    if plan is None:
        return

    reviews = []
    for s in plan.steps:
        if s.status != "done":
            continue
        cap = capability_by_name(s.capability)
        if cap is None or cap.extract_review_target is None:
            continue
        run_name = cap.extract_review_target(s.result or {}, s.args)
        if run_name is None:
            continue
        review = _try_review(run_name)
        if review:
            reviews.append({"step_id": s.id, "review": review})

    if not reviews:
        return

    bus = get_bus()
    bus.publish(BusMessage(
        from_dept=MENXIA, to_dept=ZHONGSHU, kind=KIND_REVIEW,
        payload={"plan_id": plan_id, "reviews": reviews},
    ))


def _try_review(run_name: str) -> dict | None:
    """尝试审查一个 run。失败静默返回 None。"""
    try:
        from control.agent.menxia.review import review_run
        return review_run(run_name)
    except Exception:
        return None


def _on_step_failed(msg: BusMessage) -> None:
    """步骤失败，呈递异常简报到中书省面板。"""
    payload = msg.payload
    bus = get_bus()
    bus.publish(BusMessage(
        from_dept=MENXIA, to_dept=ZHONGSHU, kind=KIND_REVIEW,
        payload={
            "plan_id": payload.get("plan_id", ""),
            "step_id": payload.get("step_id", ""),
            "capability": payload.get("capability", ""),
            "error": payload.get("error", ""),
            "type": "failure_report",
        },
    ))
