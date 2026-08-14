"""control.agent.shangshu.planner — 尚书省拟案（draft_plan）。

收中书省转交的用户意图，经 LLM 产出结构化 Plan。

拟案时 LLM 可先查询项目现状（list_runs / list_workspaces / list_env_snapshots /
get_env_config），再调 submit_plan 提交计划。这是 ReAct 循环（最多 4 轮）：
  轮 1: LLM 可能先查"现在有哪些 run / 配了什么模型"
  轮 2: LLM 基于查询结果调 submit_plan 提交计划
  （或直接调 submit_plan 不查，也行）
"""

from __future__ import annotations

import json

from control.agent.llm import chat_with_tools, is_llm_configured, parse_tool_args, rebuild_tool_calls
from control.agent.shangshu import plan as plan_mod
from control.agent.shangshu.docs import build_system_prompt

# 拟案时可用的查询工具（只读，让 LLM 了解项目现状）
_QUERY_CAPS = {"list_runs", "list_workspaces", "list_env_snapshots", "get_env_config"}


def _submit_plan_schema() -> dict:
    """submit_plan 工具的 schema（约束 LLM 产出结构化 Plan）。"""
    return {
        "type": "function",
        "function": {
            "name": "submit_plan",
            "description": (
                "提交一个结构化执行计划。你应该在充分了解项目现状后调用此工具。"
                "如果需要查 run 历史/工作区/配置，先调查询工具，再调此工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "description": "你理解的用户意图（一句话概括）",
                    },
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "description": "步骤 id，如 s1/s2"},
                                "capability": {"type": "string", "description": "能力清单里的 name"},
                                "args": {"type": "object", "description": "该能力的参数"},
                                "depends_on": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "前置步骤 id（空数组=无依赖）",
                                },
                                "description": {"type": "string", "description": "一句话说明这步干什么"},
                            },
                            "required": ["id", "capability", "args", "description"],
                        },
                    },
                },
                "required": ["intent", "steps"],
            },
        },
    }


def _query_tool_schemas() -> list[dict]:
    """从 capabilities 清单派生查询工具 schema（只读类）。"""
    from control.agent.shangshu.capabilities import all_capabilities
    return [c.to_schema() for c in all_capabilities() if c.name in _QUERY_CAPS]


def _run_query_tool(name: str, args: dict) -> str:
    """执行查询工具，返回截断的结果字符串（给 LLM 回灌）。"""
    from control.agent.shangshu.capabilities import call
    try:
        result = call(name, args)
        return json.dumps(result, ensure_ascii=False, default=str)[:1200]
    except Exception as e:
        return f"查询失败: {type(e).__name__}: {e}"


def draft_plan(intent: str, *, session_id: str | None = None) -> plan_mod.Plan:
    """中书省转交的意图 → 结构化 Plan。

    ReAct 循环（最多 4 轮）：LLM 可先查项目现状再拟案。

    Args:
        intent: 用户意图（中书省转交时已清晰化）
        session_id: 关联 session（写入 Plan 供文牍追溯用户身份）

    Returns:
        Plan（已持久化，status=drafted，待用户准奏）
    """
    if not is_llm_configured():
        raise RuntimeError("尚书省拟案需要 LLM 配置（GENERATOR_* 环境变量）")

    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": f"请为以下指令拟定执行计划：\n\n{intent}"},
    ]

    tools_schema = [_submit_plan_schema()] + _query_tool_schemas()

    for _round in range(4):
        resp = chat_with_tools(messages, tools=tools_schema)
        msg = resp.choices[0].message

        if not msg.tool_calls:
            # LLM 没调工具（不应该发生，因为 tool_choice 默认 auto + 有 submit_plan）
            raise RuntimeError(f"尚书省未产出 Plan：{msg.content[:200]}")

        # 检查是否调了 submit_plan
        for tc in msg.tool_calls:
            if tc.function.name == "submit_plan":
                return _extract_plan(tc, intent, session_id)

        # 否则执行查询工具，结果回灌，继续循环
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": rebuild_tool_calls(msg),
        })
        for tc in msg.tool_calls:
            name = tc.function.name
            args = parse_tool_args(tc)
            result_str = _run_query_tool(name, args) if name in _QUERY_CAPS else f"未知工具: {name}"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_str})

    # 达到轮次仍未提交 Plan → 最后一轮强制只要 submit_plan
    messages.append({"role": "user", "content": "请立即调 submit_plan 提交计划。"})
    resp = chat_with_tools(messages, tools=[_submit_plan_schema()], tool_choice="required")
    msg = resp.choices[0].message
    if msg.tool_calls:
        for tc in msg.tool_calls:
            if tc.function.name == "submit_plan":
                return _extract_plan(tc, intent, session_id)
    raise RuntimeError("尚书省拟案超时（4 轮查询后仍未提交 Plan）")


def _extract_plan(tc, intent: str, session_id: str | None) -> plan_mod.Plan:
    """从 submit_plan 工具调用提取 Plan。"""
    try:
        args = json.loads(tc.function.arguments) if tc.function.arguments else {}
    except json.JSONDecodeError as e:
        raise RuntimeError(f"submit_plan 参数解析失败: {e}") from e

    steps_raw = args.get("steps", [])
    if not steps_raw:
        raise RuntimeError("submit_plan 返回空步骤列表")

    from control.agent.shangshu.capabilities import capability_by_name
    for s in steps_raw:
        cap_name = s.get("capability", "")
        if capability_by_name(cap_name) is None:
            raise RuntimeError(f"步骤 {s.get('id')} 引用了未知能力: {cap_name}")

    plan = plan_mod.make_plan_from_llm(args.get("intent", intent), steps_raw)
    plan.session_id = session_id
    # 便宜行事标注：自动判断每个步骤是否可以跳过用户准奏
    _annotate_auto_execute(plan, session_id)
    plan_mod.save_plan(plan)
    return plan


def _annotate_auto_execute(plan: plan_mod.Plan, session_id: str | None) -> None:
    """便宜行事标注——自动判断每个步骤是否可以跳过用户准奏。

    规则：
    - risk_level=low → auto_execute=True（查询/只读/创建隔离副本）
    - risk_level=medium → 查文牍：同一 session 内该 capability 曾被准奏过 → True
    - risk_level=high/critical → False（一律走准奏）
    """
    from control.agent.shangshu.capabilities import capability_by_name

    # 查文牍历史：同 session 内哪些 capability 曾被准奏过
    prior_caps = set()
    if session_id:
        try:
            from control.agent import gazette
            for g in gazette.list_gazettes(session_id=session_id, recent=50):
                events = gazette.read_events(g["plan_id"])
                # 找该 Plan 的事件中：step_succeeded 的 capability（说明曾执行过=曾准奏过）
                for ev in events:
                    # 只认 EV_STEP_SUCCEEDED：EV_PLAN_APPROVED 的 detail 只有 approved_at，
                    # 从无 capability 字段（曾列入枚举属无效逻辑）
                    if ev.kind == gazette.EV_STEP_SUCCEEDED:
                        cap = ev.detail.get("capability", "")
                        if cap:
                            prior_caps.add(cap)
        except Exception:
            pass  # 文牍查询失败不影响拟案

    for step in plan.steps:
        cap = capability_by_name(step.capability)
        if cap is None:
            step.auto_execute = False
        elif cap.risk_level == "low":
            step.auto_execute = True
        elif cap.risk_level == "medium":
            step.auto_execute = step.capability in prior_caps  # 熟能生巧
        else:
            step.auto_execute = False  # high/critical 一律走准奏
