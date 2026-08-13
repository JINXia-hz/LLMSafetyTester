"""control.agent.shangshu.planner — 尚书省拟案（draft_plan）。

收中书省转交的用户意图，经 LLM + submit_plan 工具产出结构化 Plan。
LLM 用尚书省专属 system prompt（含完整能力文档），在能力清单内组合步骤。
"""

from __future__ import annotations

import json

from control.agent.llm import chat_with_tools, is_llm_configured
from control.agent.shangshu import plan as plan_mod
from control.agent.shangshu.docs import build_system_prompt


def _submit_plan_schema() -> dict:
    """submit_plan 工具的 schema（约束 LLM 产出结构化 Plan）。"""
    return {
        "type": "function",
        "function": {
            "name": "submit_plan",
            "description": (
                "提交一个结构化执行计划。你必须在收到指令后**只调用此工具一次**，"
                "把拆解好的步骤序列提交。不要先调别的工具。"
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
                                    "description": "前置步骤 id（空数组=无依赖，可立即执行）",
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


def draft_plan(intent: str, context: dict | None = None) -> plan_mod.Plan:
    """中书省转交的意图 → 结构化 Plan。

    用尚书省专属 LLM 会话（完整能力文档），经 submit_plan 工具产出 Plan。
    LLM 只调一次 submit_plan 就返回（不做 ReAct 多轮）。

    Args:
        intent: 用户意图（中书省转交时已清晰化）
        context: 可选上下文（如当前有哪些 workspace/env_snapshot，帮 LLM 决策）

    Returns:
        Plan（已持久化，status=drafted，待用户准奏）

    Raises:
        RuntimeError: LLM 未配置或拟案失败
    """
    if not is_llm_configured():
        raise RuntimeError("尚书省拟案需要 LLM 配置（GENERATOR_* 环境变量）")

    context_str = ""
    if context:
        context_str = "\n\n[上下文信息]\n"
        for k, v in context.items():
            context_str += f"- {k}: {json.dumps(v, ensure_ascii=False)[:500] if not isinstance(v, str) else v[:500]}\n"

    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": f"请为以下指令拟定执行计划：\n\n{intent}{context_str}"},
    ]

    resp = chat_with_tools(messages, tools=[_submit_plan_schema()], tool_choice="required")
    msg = resp.choices[0].message

    if not msg.tool_calls:
        raise RuntimeError(f"尚书省未产出 Plan（LLM 未调 submit_plan）：{msg.content[:200]}")

    tc = msg.tool_calls[0]
    try:
        args = json.loads(tc.function.arguments) if tc.function.arguments else {}
    except json.JSONDecodeError as e:
        raise RuntimeError(f"submit_plan 参数解析失败: {e}") from e

    # 校验 + 构造 Plan
    steps_raw = args.get("steps", [])
    if not steps_raw:
        raise RuntimeError("submit_plan 返回空步骤列表")

    # 校验每个步骤的 capability 存在
    from control.agent.shangshu.capabilities import capability_by_name
    for s in steps_raw:
        cap_name = s.get("capability", "")
        if capability_by_name(cap_name) is None:
            raise RuntimeError(f"步骤 {s.get('id')} 引用了未知能力: {cap_name}")

    plan = plan_mod.make_plan_from_llm(args.get("intent", intent), steps_raw)
    plan_mod.save_plan(plan)
    return plan
