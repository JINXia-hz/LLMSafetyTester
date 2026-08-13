"""control.agent.zhongshu — 中书省（前台 + 意图理解 + 润色）。

重构后的中书省是对话主入口，但**不再自己执行复杂任务**：

  简单查询（list_runs/compare/list_workspaces 等）→ 自己用工具处理
  复杂指令（多步/跨模块/改配置+跑实验）→ 调尚书省.draft_plan
    → 收到 Plan → LLM 润色成给用户的中文方案 → 返回 plan_pending
    → 用户准奏 → 尚书省.execute_plan

中书省有**简略能力概览**（不像尚书省有完整文档），
判断复杂度的依据写进 system prompt：
  指令含「然后/再/接着/同时/对比 A 和 B 之后」等多步信号 → 复杂 → 转交尚书省。

人设（保持原有）：
  - 称用户「陛下」，自称「臣」
  - 简洁得体，有古风不迂腐
  - 数据说话，不空谈
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from control.agent import session as sess
from control.agent.llm import chat_with_tools, is_llm_configured
from control.agent.loop import chat_one as _rule_chat_one
from control.agent.prompts import ZHONGSHU_PROMPT as _SYSTEM_PROMPT

# 中书省 system prompt 在 prompts.py（避免循环导入）


# ============================================================
# 中书省的简单查询工具（复杂操作转交尚书省）
# ============================================================
def _simple_tools() -> list[dict]:
    """中书省保留的简单查询工具 schema（执行类交给尚书省）。"""
    from control.agent.tools import all_tools
    keep = {"list_runs", "compare_runs", "review_run", "list_workspaces"}
    return [t.to_schema() for t in all_tools() if t.name in keep]


def _shangshu_plan_schema() -> dict:
    """request_shangshu_plan 工具：转交尚书省拟案。"""
    return {
        "type": "function",
        "function": {
            "name": "request_shangshu_plan",
            "description": (
                "把一个复杂指令转交尚书省拟执行计划。"
                "用于多步/跨模块/改配置后操作/批量等复杂任务。"
                "调用后你会收到尚书省拟好的结构化 Plan，你需要把它润色成给用户看的中文方案。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "description": "清晰、准确转述的用户指令（尚书省靠这个拟案）",
                    },
                },
                "required": ["intent"],
            },
        },
    }


# ============================================================
# 一轮对话的结果
# ============================================================
@dataclass
class ZhongshuTurn:
    """中书省一轮对话的结果。"""
    user_text: str
    reply: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    plan_pending: dict | None = None   # {plan_id, rendered_plan}（复杂指令拟案后）
    mode: str = "llm"                   # llm / rule / fallback / error
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "user": self.user_text,
            "reply": self.reply,
            "tool_calls": self.tool_calls,
            "plan_pending": self.plan_pending,
            "mode": self.mode,
            "error": self.error,
        }


# ============================================================
# 主入口
# ============================================================
def handle_message(
    user_text: str,
    *,
    session_id: str | None = None,
) -> dict:
    """中书省主入口。处理用户消息，返回结构化结果。

    流程：
      1. LLM 理解意图 + 判断复杂度
      2. 简单 → 自己用工具处理
      3. 复杂 → 调 request_shangshu_plan → 收到 Plan → 润色 → 返回 plan_pending
      4. LLM 未配置/失败 → 规则兜底

    返回 dict（含 mode/reply/plan_pending/tool_calls/session_id）。
    """
    session_id, messages = sess.get_or_create(session_id)

    if not is_llm_configured():
        reply = _rule_chat_one(user_text)
        sess.append(session_id, "user", user_text)
        sess.append(session_id, "assistant", reply)
        return {"mode": "rule", "reply": reply, "session_id": session_id}

    messages.append({"role": "user", "content": user_text})

    try:
        turn = _react_loop(user_text, messages, session_id)
        return turn.to_dict() | {"session_id": session_id}
    except Exception as e:
        # LLM 失败 → 规则兜底
        reply = _rule_chat_one(user_text)
        sess.append(session_id, "assistant", reply)
        return {"mode": "fallback", "reply": reply, "session_id": session_id,
                "llm_error": f"{type(e).__name__}: {e}"}


def _react_loop(user_text: str, messages: list[dict], session_id: str,
                max_rounds: int = 4) -> ZhongshuTurn:
    """中书省 ReAct 循环：LLM + 简单工具 + request_shangshu_plan。

    最多 4 轮（复杂任务转交尚书省，不需要长循环）。
    """
    from control.agent.tools import call_tool
    turn = ZhongshuTurn(user_text=user_text)

    tools_schema = _simple_tools() + [_shangshu_plan_schema()]

    for _round in range(max_rounds):
        resp = chat_with_tools(messages, tools=tools_schema)
        msg = resp.choices[0].message

        # LLM 没调工具 → 最终回复
        if not msg.tool_calls:
            turn.reply = msg.content or "(臣无以作答)"
            messages.append({"role": "assistant", "content": turn.reply})
            sess.append(session_id, "assistant", turn.reply)
            return turn

        # 执行工具调用
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {}

            # request_shangshu_plan → 转交尚书省
            if name == "request_shangshu_plan":
                plan_dict = _hand_to_shangshu(args.get("intent", user_text), messages, session_id)
                if plan_dict:
                    turn.plan_pending = plan_dict
                    turn.reply = plan_dict.get("rendered_plan", "尚书省已拟案，请陛下过目。")
                    # 回灌 tool 结果
                    messages.append({
                        "role": "tool", "tool_call_id": tc.id,
                        "content": json.dumps({"plan_id": plan_dict["plan_id"],
                                              "steps_count": len(plan_dict.get("steps", []))},
                                             ensure_ascii=False),
                    })
                    sess.append(session_id, "assistant", turn.reply)
                    return turn
                else:
                    messages.append({
                        "role": "tool", "tool_call_id": tc.id,
                        "content": "尚书省拟案失败，请简化指令或稍后再试。",
                    })
                continue

            # 简单工具 → 直接调
            if name in {"list_runs", "compare_runs", "review_run", "list_workspaces"}:
                try:
                    result = call_tool(name, args)
                    result_str = _summarize_result(result)
                    turn.tool_calls.append({"name": name, "args": args, "result": result_str[:500]})
                    messages.append({
                        "role": "tool", "tool_call_id": tc.id, "content": result_str,
                    })
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                    messages.append({
                        "role": "tool", "tool_call_id": tc.id, "content": f"工具失败: {err}",
                    })
            else:
                messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": f"中书省不直接执行 {name}——请转交尚书省拟案。",
                })

    # 达到轮次上限 → 让 LLM 收尾
    messages.append({"role": "user", "content": "请基于已有信息总结回复。"})
    resp = chat_with_tools(messages, tools=None)
    turn.reply = resp.choices[0].message.content or "(臣无以作答)"
    messages.append({"role": "assistant", "content": turn.reply})
    sess.append(session_id, "assistant", turn.reply)
    return turn


def _hand_to_shangshu(intent: str, messages: list[dict], session_id: str) -> dict | None:
    """转交尚书省拟案，再润色成给用户看的方案。

    Returns:
        {plan_id, steps, rendered_plan} 或 None（拟案失败）
    """
    try:
        from control.agent.shangshu import draft_plan

        # 收集上下文（当前 workspace/env_snapshot 列表，帮尚书省决策）
        context = _collect_context()
        plan = draft_plan(intent, context=context)

        # 润色：LLM 把技术步骤翻译成给用户的话
        rendered = _render_plan_for_user(plan, messages)

        return {
            "plan_id": plan.id,
            "steps": [s.to_dict() if hasattr(s, 'to_dict') else _step_dict(s) for s in plan.steps],
            "rendered_plan": rendered,
        }
    except Exception as e:
        import sys
        print(f"[中书省] 转交尚书省失败: {e}", file=sys.stderr)
        return None


def _step_dict(s) -> dict:
    """Step 对象转 dict（plan.py 的 Step 不是 dataclass to_dict，手动转）。"""
    return {
        "id": s.id, "capability": s.capability, "args": s.args,
        "depends_on": s.depends_on, "description": s.description,
        "status": s.status,
    }


def _collect_context() -> dict:
    """收集当前状态上下文（帮尚书省拟案）。失败静默跳过。"""
    ctx = {}
    try:
        from control.core.workspace import list_workspaces
        ctx["workspaces"] = [w["name"] for w in list_workspaces()]
    except Exception:
        pass
    try:
        from control.core.env_snapshot import list_snapshots
        snaps = list_snapshots()
        ctx["env_snapshots"] = [s["name"] for s in snaps]
    except Exception:
        pass
    return ctx


def _render_plan_for_user(plan, messages: list[dict]) -> str:
    """LLM 润色：把尚书省的技术 Plan 翻译成给用户看的中文方案。"""
    steps_text = "\n".join(
        f"  {i+1}. [{s.id}] {s.description}（能力：{s.capability}）"
        for i, s in enumerate(plan.steps)
    )
    prompt = (
        f"尚书省拟好了执行计划，请把下面的技术步骤润色成给天子（用户）看的中文方案。\n"
        f"用简洁得体的语言，不要照搬技术参数细节，重点说清每步干什么、为什么。\n"
        f"格式用编号列表。末尾问天子是否准奏。\n\n"
        f"用户原始意图：{plan.intent}\n\n"
        f"尚书省的技术步骤：\n{steps_text}"
    )
    try:
        resp = chat_with_tools(
            [{"role": "system", "content": _SYSTEM_PROMPT},
             {"role": "user", "content": prompt}],
            tools=None,
        )
        return resp.choices[0].message.content or steps_text
    except Exception:
        return f"尚书省已拟案（{len(plan.steps)} 步）：\n{steps_text}\n\n请陛下准奏。"


def _summarize_result(result) -> str:
    """把工具结果压缩成给 LLM 看的字符串。"""
    if isinstance(result, list):
        n = len(result)
        if n == 0:
            return "（无结果）"
        sample = result[:10]
        lines = [f"共 {n} 项，前 10 项："]
        for it in sample:
            if isinstance(it, dict):
                name = it.get("name", it.get("workspace", "?"))
                target = it.get("target_model", it.get("target", ""))
                asr = it.get("asr")
                asr_s = f"{asr:.1%}" if isinstance(asr, (int, float)) else "-"
                lines.append(f"  - {name} (target={target}, asr={asr_s})")
            else:
                lines.append(f"  - {it}")
        return "\n".join(lines)
    if isinstance(result, dict):
        return json.dumps(result, ensure_ascii=False, default=str)[:1500]
    return str(result)[:1500]
