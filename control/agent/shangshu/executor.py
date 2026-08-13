"""control.agent.shangshu.executor — 尚书省执行（execute_plan）。

拓扑分层执行 Plan：
  1. 按 depends_on 分层，同层步骤并行（ThreadPoolExecutor）。
  2. 每步执行前发 BusMessage(step_start) → 门下省审查。
     - 门下省封驳 → 该步标 blocked，发 step_blocked。
     - 依赖它的后续步标 skipped（数据依赖无法满足）。
     - 不依赖它的步骤继续。
  3. 每步执行后发 step_done（成功）/ step_failed（异常），附结果。
  4. 全部完成发 plan_done → 门下省自动审查呈递简报。

同步阻塞：execute_plan 在一个调用内跑完，通过 on_progress 回调推前端进度。
长任务（跑评估）会阻塞——Phase 1 取舍，预留 async 接口 Phase 2 接 job 系统。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from control.agent.bus import (
    ALL,
    KIND_PLAN_DONE,
    KIND_PLAN_PROGRESS,
    KIND_STEP_BLOCKED,
    KIND_STEP_DONE,
    KIND_STEP_FAILED,
    KIND_STEP_START,
    SHANGSHU,
    BusMessage,
    get_bus,
)
from control.agent.shangshu import capabilities as caps_mod
from control.agent.shangshu import plan as plan_mod
from control.agent.shangshu.plan import (
    P_APPROVED,
    P_DONE,
    P_EXECUTING,
    S_BLOCKED,
    S_DONE,
    S_FAILED,
    S_PENDING,
    S_RUNNING,
    S_SKIPPED,
    Plan,
    Step,
)

# 进度回调：on_progress(plan: Plan) -> None（前端据此渲染）
ProgressCallback = Callable[[Plan], None]


def execute_plan(
    plan_id: str,
    *,
    on_progress: ProgressCallback | None = None,
    max_workers: int = 4,
) -> Plan:
    """执行 Plan（拓扑分层 + 门下省封驳 + 进度回调）。

    Args:
        plan_id: Plan id（须已 load_plan 进内存或磁盘存在）
        on_progress: 每步状态变化时回调（传当前 Plan 快照）
        max_workers: 同层并行步的最大线程数

    Returns:
        执行完的 Plan（status=done，各步有最终状态）

    封驳处理：门下省在 step_start 时审查，返回非 None → 该步 blocked。
    被挡步骤的依赖者标 skipped。不依赖被挡步骤的继续执行。
    """
    plan = plan_mod.load_plan(plan_id)
    if plan is None:
        raise KeyError(f"Plan 不存在: {plan_id}")
    if plan.status not in (P_APPROVED, P_EXECUTING):
        # 允许 approved 或重入 executing（封驳后用户确认重试）
        if plan.status == P_DONE:
            return plan  # 已完成，幂等返回
        raise RuntimeError(f"Plan 状态不允许执行: {plan.status}（需 {P_APPROVED}）")

    plan.status = P_EXECUTING
    plan_mod.save_plan(plan)
    _notify_progress(plan, on_progress)

    bus = get_bus()

    # 拓扑分层执行
    layers = plan.topological_layers()
    for layer in layers:
        # 跳过已经终态的步骤（重入时已完成的不重跑）
        runnable = [s for s in layer if s.status in (S_PENDING, S_BLOCKED)]
        # blocked 步骤若用户已确认放行（ticket 被清除），重试——这里简化为：
        # blocked 且 ticket 仍存在的保持 blocked，ticket=None 的转 pending 重试
        for s in runnable:
            if s.status == S_BLOCKED and s.ticket is not None:
                s.status = S_BLOCKED  # 仍被挡，不动
            elif s.status == S_BLOCKED and s.ticket is None:
                s.status = S_PENDING  # 已放行，重试
        to_run = [s for s in layer if s.status == S_PENDING]

        if not to_run:
            continue

        # 同层并行
        if len(to_run) == 1:
            _execute_step(to_run[0], plan, on_progress, bus)
        else:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(to_run))) as pool:
                futures = {pool.submit(_execute_step, s, plan, on_progress, bus): s for s in to_run}
                for fut in as_completed(futures):
                    fut.result()  # 异常已在 _execute_step 内捕获，不会抛

        # 层结束后：传播 blocked/failed 到依赖者
        _propagate_blockage(plan)
        plan_mod.save_plan(plan)
        _notify_progress(plan, on_progress)

    # 全部完成
    plan.status = P_DONE
    plan.finished = time.time()
    plan_mod.save_plan(plan)

    # 发 plan_done → 门下省自动审查呈递简报
    bus.publish(BusMessage(
        from_dept=SHANGSHU, to_dept=ALL, kind=KIND_PLAN_DONE,
        payload={"plan_id": plan.id, "intent": plan.intent,
                 "steps": _steps_summary(plan)},
    ))
    _notify_progress(plan, on_progress)
    return plan


def _execute_step(step: Step, plan: Plan, on_progress: ProgressCallback | None, bus) -> None:
    """执行单个步骤（含门下省审查）。线程安全：每步只改自己的字段。"""
    # 1. 发 step_start → 门下省审查
    cap = caps_mod.capability_by_name(step.capability)
    if cap is None:
        step.status = S_FAILED
        step.error = f"未知能力: {step.capability}"
        bus.publish(BusMessage(
            from_dept=SHANGSHU, to_dept=ALL, kind=KIND_STEP_FAILED,
            payload={"plan_id": plan.id, "step_id": step.id,
                     "capability": step.capability, "error": step.error},
        ))
        return

    bus.publish(BusMessage(
        from_dept=SHANGSHU, to_dept=ALL, kind=KIND_STEP_START,
        payload={"plan_id": plan.id, "step_id": step.id,
                 "capability": step.capability, "args": step.args,
                 "risk_level": cap.risk_level, "description": step.description},
    ))

    # 2. 门下省审查（经总线同步派发，门下省回调可能发 block 消息）
    #    检查是否有针对此步的封驳（门下省在 _on_step_start 里发 KIND_BLOCK）
    #    给门下省一点时间处理（总线是同步的，publish 返回时回调已执行完）
    block_msg = _check_block_for_step(bus, plan.id, step.id)
    if block_msg is not None:
        step.status = S_BLOCKED
        step.ticket = block_msg
        bus.publish(BusMessage(
            from_dept=SHANGSHU, to_dept=ALL, kind=KIND_STEP_BLOCKED,
            payload={"plan_id": plan.id, "step_id": step.id,
                     "capability": step.capability, "ticket": block_msg},
        ))
        return

    # 3. 执行
    step.status = S_RUNNING
    _notify_progress(plan, on_progress)
    try:
        result = cap.handler(step.args)
        step.status = S_DONE
        step.result = _sanitize_result(result)
        bus.publish(BusMessage(
            from_dept=SHANGSHU, to_dept=ALL, kind=KIND_STEP_DONE,
            payload={"plan_id": plan.id, "step_id": step.id,
                     "capability": step.capability, "result": step.result},
        ))
    except Exception as e:
        step.status = S_FAILED
        step.error = f"{type(e).__name__}: {e}"
        bus.publish(BusMessage(
            from_dept=SHANGSHU, to_dept=ALL, kind=KIND_STEP_FAILED,
            payload={"plan_id": plan.id, "step_id": step.id,
                     "capability": step.capability, "error": step.error},
        ))


def _check_block_for_step(bus, plan_id: str, step_id: str) -> dict | None:
    """检查门下省是否对此步发了封驳。

    门下省在收到 step_start 时（同步回调内）会 publish 一个 KIND_BLOCK 消息。
    我们查 bus.recent 里最近发来的、针对此步的 block。

    简化实现：step_start publish 后，总线同步派发完门下省回调，
    门下省回调内若封驳会再 publish(KIND_BLOCK)。我们查 recent 末尾。
    """
    recent = bus.recent(since_ts=time.time() - 5.0)  # 最近 5 秒
    for m in reversed(recent):
        if (m.kind == "block"
                and m.payload.get("plan_id") == plan_id
                and m.payload.get("step_id") == step_id):
            return m.payload.get("ticket")
    return None


def _propagate_blockage(plan: Plan) -> None:
    """层结束后传播 blocked/failed：依赖被挡步骤的后续步标 skipped。"""
    blocked_or_failed = {s.id for s in plan.steps if s.status in (S_BLOCKED, S_FAILED)}
    for s in plan.steps:
        if s.status == S_PENDING:
            if any(d in blocked_or_failed for d in s.depends_on):
                s.status = S_SKIPPED
                s.error = "前置步骤被封驳或失败，跳过"


def _notify_progress(plan: Plan, on_progress: ProgressCallback | None) -> None:
    """推进度（回调 + 总线消息）。"""
    if on_progress:
        try:
            on_progress(plan)
        except Exception:
            pass  # 回调异常不影响执行
    bus = get_bus()
    bus.publish(BusMessage(
        from_dept=SHANGSHU, to_dept=ALL, kind=KIND_PLAN_PROGRESS,
        payload={"plan_id": plan.id, "status": plan.status,
                 "steps": [{"id": s.id, "status": s.status} for s in plan.steps]},
    ))


def _sanitize_result(result) -> dict | None:
    """把 handler 结果清洗成可序列化的 dict（存 Plan + 推前端）。"""
    import json
    if result is None:
        return None
    if isinstance(result, (dict, list, str, int, float, bool)):
        # 尝试序列化（dict/list 里可能有 Path 等不可序列化对象）
        try:
            json.dumps(result, ensure_ascii=False, default=str)
            return result if isinstance(result, dict) else {"value": result}
        except (TypeError, ValueError):
            return {"value": str(result)[:2000]}
    return {"value": str(result)[:2000]}


def _steps_summary(plan: Plan) -> list[dict]:
    return [{"id": s.id, "capability": s.capability, "status": s.status,
             "description": s.description} for s in plan.steps]
