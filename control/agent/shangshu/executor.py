"""control.agent.shangshu.executor — 尚书省执行（execute_plan）。

拓扑分层执行 Plan：
  1. 按 depends_on 分层，同层步骤并行（ThreadPoolExecutor）。
  2. 每步执行前 notify(step_start) → 门下省审查。
     - 门下省封驳 → 该步标 blocked，文牍记 step_blocked。
     - 依赖它的后续步标 skipped（数据依赖无法满足）。
     - 不依赖它的步骤继续。
  3. 每步执行后文牍记 step_succeeded/failed，notify 推进度。
  4. 全部完成文牍记 plan_finished + notify(plan_done) → 门下省自动审查。

文牍：每个事件 append 到 output/gazette/<plan_id>.jsonl，持久化完整执行历史。
总线：用 notify() 统一发布，带 plan_id/intent/session_id 公共信封。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from control.agent import gazette
from control.agent.bus import (
    KIND_PLAN_DONE,
    KIND_PLAN_PROGRESS,
    KIND_STEP_BLOCKED,
    KIND_STEP_DONE,
    KIND_STEP_FAILED,
    KIND_STEP_START,
    SHANGSHU,
    notify,
        notify_routed,
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

ProgressCallback = Callable[[Plan], None]


def execute_plan(
    plan_id: str,
    *,
    on_progress: ProgressCallback | None = None,
    max_workers: int = 4,
) -> Plan:
    """执行 Plan（拓扑分层 + 门下省封驳 + 文牍记录 + 进度回调）。"""
    plan = plan_mod.load_plan(plan_id)
    if plan is None:
        raise KeyError(f"Plan 不存在: {plan_id}")
    if plan.status not in (P_APPROVED, P_EXECUTING):
        if plan.status == P_DONE:
            return plan
        raise RuntimeError(f"Plan 状态不允许执行: {plan.status}（需 {P_APPROVED}）")

    plan.status = P_EXECUTING
    plan.started = time.time()
    plan_mod.save_plan(plan)

    gazette.append_event(plan.id, gazette.EV_PLAN_STARTED, SHANGSHU,
                         session_id=plan.session_id, intent=plan.intent,
                         detail={})
    _notify_progress(plan, on_progress)

    # 拓扑分层执行
    layers = plan.topological_layers()
    for layer in layers:
        for s in layer:
            if s.status == S_BLOCKED and s.ticket is None:
                s.status = S_PENDING  # 门下省已放行，重试
        to_run = [s for s in layer if s.status == S_PENDING]

        if not to_run:
            continue

        if len(to_run) == 1:
            _execute_step(to_run[0], plan, on_progress)
        else:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(to_run))) as pool:
                futures = {pool.submit(_execute_step, s, plan, on_progress): s for s in to_run}
                for fut in as_completed(futures):
                    try:
                        fut.result()
                    except Exception as e:
                        # worker 线程内 gazette 落盘等基础设施异常（非 handler 异常——
                        # 那些已在 _execute_step 内消化）。吞掉并记录，保证后续层照常
                        # 执行、Plan 能推进到终态；上抛会让 execute_plan 中断且无人恢复，
                        # Plan 永久卡在 executing。
                        import sys
                        print(f"[executor] 步骤 {futures[fut].id} 线程异常: "
                              f"{type(e).__name__}: {e}", file=sys.stderr)

        _propagate_blockage(plan)
        plan_mod.save_plan(plan)
        _notify_progress(plan, on_progress)

    # 全部完成
    plan.status = P_DONE
    plan.finished = time.time()
    plan_mod.save_plan(plan)

    # 统计
    stats = {}
    for s in plan.steps:
        stats[s.status] = stats.get(s.status, 0) + 1

    gazette.append_event(plan.id, gazette.EV_PLAN_FINISHED, SHANGSHU,
                         session_id=plan.session_id, intent=plan.intent,
                         detail={"step_stats": stats, "duration_s": round(
                             (plan.finished - (plan.started or plan.created)), 1)})

    notify_routed(KIND_PLAN_DONE, from_dept=SHANGSHU,
                  plan_id=plan.id, intent=plan.intent, session_id=plan.session_id,
                  steps=_steps_summary(plan))
    _notify_progress(plan, on_progress)
    return plan


def _execute_step(step: Step, plan: Plan, on_progress: ProgressCallback | None) -> None:
    """执行单个步骤（含门下省审查 + 文牍记录 + 时间戳）。"""
    cap = caps_mod.capability_by_name(step.capability)
    if cap is None:
        step.status = S_FAILED
        step.error = f"未知能力: {step.capability}"
        gazette.append_event(plan.id, gazette.EV_STEP_FAILED, SHANGSHU,
                             step_id=step.id, session_id=plan.session_id, intent=plan.intent,
                             detail={"capability": step.capability, "error": step.error})
        notify_routed(KIND_STEP_FAILED, from_dept=SHANGSHU, plan_id=plan.id,
                      intent=plan.intent, session_id=plan.session_id,
                      step_id=step.id, capability=step.capability, error=step.error)
        return

    # 1. 文牍记 + 总线通知 step_start（collect_replies：门下省在同步派发中
    #    直接返回封驳裁决——放行返回 None，封驳返回 ticket dict）
    step.started = time.time()
    gazette.append_event(plan.id, gazette.EV_STEP_STARTED, SHANGSHU,
                         step_id=step.id, session_id=plan.session_id, intent=plan.intent,
                         detail={"capability": step.capability, "description": step.description,
                                 "args": step.args})
    replies = notify_routed(KIND_STEP_START, from_dept=SHANGSHU, plan_id=plan.id,
                            intent=plan.intent, session_id=plan.session_id,
                            step_id=step.id, capability=step.capability, args=step.args,
                            risk_level=cap.risk_level, description=step.description,
                            collect_replies=True)

    # 2. 门下省裁决直收（按 plan/step 双重匹配，防其他订阅者的无关返回值混入）
    block_msg = next(
        (r for r in (replies or [])
         if isinstance(r, dict) and r.get("plan_id") == plan.id and r.get("step_id") == step.id),
        None)
    if block_msg is not None:
        step.status = S_BLOCKED
        step.ticket = block_msg
        step.finished = time.time()
        step.block_history.append({"ticket": block_msg, "ts": time.time()})
        gazette.append_event(plan.id, gazette.EV_STEP_BLOCKED, SHANGSHU,
                             step_id=step.id, session_id=plan.session_id, intent=plan.intent,
                             detail={"capability": step.capability, "ticket": block_msg})
        notify_routed(KIND_STEP_BLOCKED, from_dept=SHANGSHU, plan_id=plan.id,
                      intent=plan.intent, session_id=plan.session_id,
                      step_id=step.id, capability=step.capability, ticket=block_msg)
        return

    # 3. 执行
    step.status = S_RUNNING
    _notify_progress(plan, on_progress)
    try:
        result = cap.handler(step.args)
        step.status = S_DONE
        step.result = _sanitize_result(result)
        step.finished = time.time()
        duration = round(step.finished - step.started, 1)
        gazette.append_event(plan.id, gazette.EV_STEP_SUCCEEDED, SHANGSHU,
                             step_id=step.id, session_id=plan.session_id, intent=plan.intent,
                             detail={"capability": step.capability,
                                     "result_digest": _result_digest(step.result),
                                     "duration_s": duration})
        notify_routed(KIND_STEP_DONE, from_dept=SHANGSHU, plan_id=plan.id,
                      intent=plan.intent, session_id=plan.session_id,
                      step_id=step.id, capability=step.capability, result=step.result)
    except Exception as e:
        step.status = S_FAILED
        step.error = f"{type(e).__name__}: {e}"
        step.finished = time.time()
        gazette.append_event(plan.id, gazette.EV_STEP_FAILED, SHANGSHU,
                             step_id=step.id, session_id=plan.session_id, intent=plan.intent,
                             detail={"capability": step.capability, "error": step.error})
        notify(KIND_STEP_FAILED, from_dept=SHANGSHU, plan_id=plan.id,
               intent=plan.intent, session_id=plan.session_id,
               step_id=step.id, capability=step.capability, error=step.error)


def _propagate_blockage(plan: Plan) -> None:
    """层结束后传播 blocked/failed：依赖被挡步骤的后续步标 skipped + 文牍记录。"""
    blocked_or_failed = {s.id for s in plan.steps if s.status in (S_BLOCKED, S_FAILED)}
    for s in plan.steps:
        if s.status == S_PENDING:
            if any(d in blocked_or_failed for d in s.depends_on):
                s.status = S_SKIPPED
                s.error = "前置步骤被封驳或失败，跳过"
                gazette.append_event(plan.id, gazette.EV_STEP_SKIPPED, SHANGSHU,
                                     step_id=s.id, session_id=plan.session_id, intent=plan.intent,
                                     detail={"capability": s.capability,
                                             "reason": s.error})


def _notify_progress(plan: Plan, on_progress: ProgressCallback | None) -> None:
    """推进度（回调 + 总线消息，带信封）。"""
    if on_progress:
        try:
            on_progress(plan)
        except Exception:
            pass
    notify(KIND_PLAN_PROGRESS, from_dept=SHANGSHU, plan_id=plan.id,
           intent=plan.intent, session_id=plan.session_id,
           status=plan.status,
           steps=[{"id": s.id, "status": s.status} for s in plan.steps])


def _sanitize_result(result) -> dict | None:
    """把 handler 结果清洗成可序列化的 dict。"""
    import json
    if result is None:
        return None
    if isinstance(result, (dict, list, str, int, float, bool)):
        try:
            json.dumps(result, ensure_ascii=False, default=str)
            return result if isinstance(result, dict) else {"value": result}
        except (TypeError, ValueError):
            return {"value": str(result)[:2000]}
    return {"value": str(result)[:2000]}


def _result_digest(result: dict | None) -> str:
    """结果摘要（给文牍记，不存完整 result）。"""
    if not result:
        return ""
    if isinstance(result, dict):
        ok = result.get("ok")
        if ok is not None:
            return f"ok={ok}"
        return str(result)[:200]
    return str(result)[:200]


def _steps_summary(plan: Plan) -> list[dict]:
    return [{"id": s.id, "capability": s.capability, "status": s.status,
             "description": s.description} for s in plan.steps]
