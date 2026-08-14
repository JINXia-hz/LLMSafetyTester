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
    KIND_BLOCK,
    KIND_PLAN_APPROVED,
    KIND_PLAN_DONE,
    KIND_PLAN_DRAFTED,
    KIND_REVIEW,
    KIND_STEP_FAILED,
    KIND_STEP_START,
    MENXIA,
    ZHONGSHU,
    BusMessage,
    get_bus,
    notify,
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
    """初始化门下省：订阅总线消息。进程启动时调一次。幂等。

    三阶段全监控：
      - 拟案阶段（plan_drafted）→ 整体合理性审查（报告，非封驳）
      - 准奏阶段（plan_approved）→ 整体风险评估（报告）
      - 执行阶段（step_start / plan_done / step_failed）→ 单步封驳 + 审查
    """
    global _INITIALIZED
    if _INITIALIZED:
        return
    bus = get_bus()
    # 拟案 + 准奏阶段
    bus.subscribe(MENXIA, [KIND_PLAN_DRAFTED], _on_plan_drafted)
    bus.subscribe(MENXIA, [KIND_PLAN_APPROVED], _on_plan_approved)
    # 执行阶段
    bus.subscribe(MENXIA, [KIND_STEP_START], _on_step_start)
    bus.subscribe(MENXIA, [KIND_PLAN_DONE], _on_plan_done)
    bus.subscribe(MENXIA, [KIND_STEP_FAILED], _on_step_failed)
    _INITIALIZED = True


def reinit_menxia() -> None:
    """强制重新订阅（测试 reset_bus 后用）。"""
    global _INITIALIZED
    _INITIALIZED = False
    init_menxia()


def _load_plan_from_msg(msg: BusMessage):
    """从总线消息取 plan_id 并加载 Plan。

    返回 (plan_id, plan)；plan_id 为空或 Plan 不存在时 plan 为 None。
    """
    plan_id = msg.plan_id or ""
    if not plan_id:
        return "", None
    from control.agent.shangshu.plan import load_plan
    return plan_id, load_plan(plan_id)


def _on_plan_drafted(msg: BusMessage) -> None:
    """拟案阶段：整体合理性审查（报告，非封驳）。

    门下省在尚书省拟完案、中书省展示给用户之前，审查 Plan 整体：
    - 步骤过多 → 建议拆分
    - 含 critical 步骤 → 提示用户细看
    - 引用不存在的资源 → 报告
    发现问题经总线发 KIND_REVIEW 报告给中书省面板。
    """
    plan_id, plan = _load_plan_from_msg(msg)
    if plan is None:
        return

    findings = []

    # 步骤过多
    if len(plan.steps) > 10:
        findings.append(f"计划含 {len(plan.steps)} 步，过于复杂，建议拆分为多个小计划")

    # 统计风险等级
    from control.agent.shangshu.capabilities import capability_by_name
    risk_counts = {"critical": 0, "high": 0, "medium": 0}
    for s in plan.steps:
        cap = capability_by_name(s.capability)
        if cap:
            risk_counts[cap.risk_level] = risk_counts.get(cap.risk_level, 0) + 1

    if risk_counts["critical"] > 0:
        findings.append(f"含 {risk_counts['critical']} 个不可逆操作（critical），请陛下细看每步说明")
    if risk_counts["high"] > 3:
        findings.append(f"含 {risk_counts['high']} 个高危步骤（high），执行将消耗大量 API 额度")

    # 矛盾检测：同时删 R 列 + merge 到全局
    has_delete_r = any(
        s.capability == "delete_runs" and s.args.get("delete_r")
        for s in plan.steps
    )
    has_merge_global = any(
        s.capability == "merge_results" and s.args.get("target") == "global"
        for s in plan.steps
    )
    if has_delete_r and has_merge_global:
        findings.append("⚠ 计划同时删 R 列和 merge 到全局——逻辑矛盾，请确认")

    if not findings:
        return  # 无异常，不报告

    report = "门下省拟案审查：" + "；".join(findings)
    notify(KIND_REVIEW, from_dept=MENXIA, to_dept=ZHONGSHU,
           plan_id=plan_id, intent=msg.intent, session_id=msg.session_id,
           type="draft_review", findings=findings, report=report)


def _on_plan_approved(msg: BusMessage) -> None:
    """准奏阶段：整体风险评估（报告，不阻塞执行）。

    用户准奏后、执行开始前，门下省做最后一道整体检查。
    """
    plan_id, plan = _load_plan_from_msg(msg)
    if plan is None:
        return

    from control.agent.shangshu.capabilities import capability_by_name
    high_risk = 0
    for s in plan.steps:
        cap = capability_by_name(s.capability)
        if cap and cap.risk_level in ("critical", "high"):
            high_risk += 1

    if high_risk == 0:
        return  # 无高危步骤，不报告

    report = f"此计划含 {high_risk} 个高危步骤。执行时门下省将逐一封驳确认。"
    notify(KIND_REVIEW, from_dept=MENXIA, to_dept=ZHONGSHU,
           plan_id=plan_id, intent=msg.intent, session_id=msg.session_id,
           type="approval_review", high_risk_count=high_risk, report=report)


def _on_step_start(msg: BusMessage) -> None:
    """步骤开始前审查。dangerous → 发封驳令 + notify(KIND_BLOCK)。

    放行豁免：如果此步骤在当前 Plan 中曾被封驳且已被用户放行（文牍有
    EV_STEP_UNBLOCKED），跳过封驳——同一 Plan 内同一步骤只封驳一次。
    """
    payload = msg.payload
    capability_name = payload.get("capability", "")
    risk_level = payload.get("risk_level", "low")
    plan_id = msg.plan_id or payload.get("plan_id", "")
    step_id = payload.get("step_id", "")
    args = payload.get("args", {})

    from control.agent.shangshu.capabilities import capability_by_name
    cap = capability_by_name(capability_name)
    if cap is None:
        return

    # ★ 放行豁免：查文牍，此步是否曾被封驳且已被用户放行
    from control.agent import gazette
    ctx = gazette.read_plan_context(plan_id)
    if ctx and step_id in ctx.get("steps", {}):
        step_info = ctx["steps"][step_id]
        if step_info.get("block_count", 0) > 0 and step_info["status"] == "pending":
            return  # 用户已放行过，不再封驳

    assessment = assess_step(cap, args)
    if assessment is None:
        return  # 放行

    # 已有封驳令且未被清除 → 保持封驳
    existing = get_block(plan_id, step_id)
    if existing is not None:
        notify(KIND_BLOCK, from_dept=MENXIA, plan_id=plan_id,
               intent=msg.intent, session_id=msg.session_id,
               step_id=step_id, ticket=existing.to_dict())
        return

    ticket = issue_block(plan_id, step_id, capability_name, risk_level, assessment)
    notify(KIND_BLOCK, from_dept=MENXIA, plan_id=plan_id,
           intent=msg.intent, session_id=msg.session_id,
           step_id=step_id, ticket=ticket.to_dict())


def _on_plan_done(msg: BusMessage) -> None:
    """Plan 执行完，自动审查所有产生的 run，呈递简报 + 写文牍。"""
    from control.agent import gazette
    from control.agent.shangshu.capabilities import capability_by_name
    from control.agent.shangshu.plan import load_plan

    plan_id = msg.plan_id or msg.payload.get("plan_id", "")
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
            # 写文牍：审查呈递
            gazette.append_event(plan_id, gazette.EV_REVIEW_FILED, MENXIA,
                                 step_id=s.id, session_id=msg.session_id, intent=msg.intent,
                                 detail={"run_name": run_name,
                                         "digest": (review.get("digest", "") or "")[:500]})

    if not reviews:
        return

    notify(KIND_REVIEW, from_dept=MENXIA, to_dept=ZHONGSHU,
           plan_id=plan_id, intent=msg.intent, session_id=msg.session_id,
           reviews=reviews)


def _try_review(run_name: str) -> dict | None:
    """尝试审查一个 run。失败静默返回 None。"""
    try:
        from control.agent.menxia.review import review_run
        return review_run(run_name)
    except Exception:
        return None


def _on_step_failed(msg: BusMessage) -> None:
    """步骤失败，呈递异常简报到中书省面板。"""
    notify(KIND_REVIEW, from_dept=MENXIA, to_dept=ZHONGSHU,
           plan_id=msg.plan_id, intent=msg.intent, session_id=msg.session_id,
           step_id=msg.payload.get("step_id", ""),
           capability=msg.payload.get("capability", ""),
           error=msg.payload.get("error", ""),
           type="failure_report")
