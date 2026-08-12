"""control.agent.chat — LLM 驱动的对话中间者（tool-calling ReAct 循环）。

替代 loop.py 的规则版 _parse_intent：用 LLM（经 control.agent.llm）理解自然语言意图，
通过 OpenAI tool calling 调用 control 的 7 个 tool（list_runs/compare/fork/...）。

流程（ReAct）：
  用户输入 → LLM（带 tool schema）→ 若 LLM 调 tool → 执行 tool → 结果回灌 → 再问 LLM
  → 重复直到 LLM 不再调 tool（给出自然语言总结）→ 返回对话轨迹

兜底：LLM 未配置或调用失败时，回退到 loop.py 的规则版 chat_one。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from control.agent.llm import chat_with_tools, is_llm_configured
from control.agent.loop import chat_one as _rule_chat_one
from control.agent.tools import Tool, all_tools

# 控制台对话的系统提示（让 LLM 知道它的角色 + 可用工具）
_SYSTEM_PROMPT = """你是 llmsec 安全评估框架的中书省控制助手。你通过调用工具帮用户管理评测工作单元（fork 分支）、对比历史结果、审查报告、合并数据。

你的能力（工具）：
- list_runs: 列出评测 run（含 fork 分支内的 run）
- compare_runs: 对比多个 run 的安全指标
- review_run: 审查一个 run 的安全报告（识别异常 + 呈递摘要）
- fork_workspace: 创建隔离的 fork 测试环境
- list_workspaces: 列出已创建的 fork 工作区
- delete_workspace: 删除一个 fork 工作区
- orchestrate: 批量并行 fork + run
- merge: 把工作区结果合并到全局或另一工作区

工作原则（中书省职责——起草与调度）：
1. **先规划后执行**：收到用户命令后，先在回复正文里简述执行计划（「我将：1.… 2.… 3.…」），再调用工具。简单查询（列一下/查一个数）可省略计划直接调。
2. **多步任务必拟计划**：涉及多个工具调用时，先说明整体步骤，让用户知道你要做什么。
3. 调用工具后，基于返回结果用简洁中文回答。涉及数据时用表格或要点。
4. 危险操作（delete/merge 到全局）会被门下省封驳要求二次确认——你在调用前应已判断用户意图明确。
5. run 名格式：历史 run 为 'YYYY-MM-DD_HHMMSS/target'，fork 分支 run 为 'ws:<分支名>/<target>'。
6. 不确定时多问一句，不要擅自假设 run 名或工作区名。
7. 审查报告时重点看「真实盲区」（surprise_score 高的威胁，即低 Elo 却成功），不是 Elo 高低；inconclusive 的结论要标注「数字待验证」。
"""


@dataclass
class ChatTurn:
    """一轮对话的轨迹（供前端渲染思考过程）。"""
    user_text: str
    tool_calls: list[dict] = field(default_factory=list)   # [{name, args, result_summary}]
    reply: str = ""
    error: str | None = None
    plan: str = ""   # 中书省拟定的执行计划（先规划后执行）

    def to_dict(self) -> dict:
        return {
            "user": self.user_text,
            "tool_calls": self.tool_calls,
            "reply": self.reply,
            "error": self.error,
            "plan": self.plan,
        }


def chat_with_llm(
    user_text: str,
    messages: list[dict],
    *,
    max_tool_rounds: int = 5,
    confirmed_token: str | None = None,
    on_blocked=None,
) -> ChatTurn:
    """LLM 驱动的单轮对话（多轮 tool-calling ReAct），带上下文记忆 + 门下省封驳。

    Args:
        user_text: 用户自然语言输入
        messages: 会话历史（会被本函数追加 user/assistant/tool 消息，调用方负责持久化）
        max_tool_rounds: 最多工具调用轮次（防死循环）
        confirmed_token: 用户对门下省劝谏的确认令牌（None=无待确认/首次请求）
        on_blocked: 回调（ticket_dict）→ 调用方据此存 pending_confirm

    Returns:
        ChatTurn（含 tool_calls 轨迹 + 最终自然语言回复 + 可能的 blocked 标记）
    """
    from control.agent import gatekeeper

    turn = ChatTurn(user_text=user_text)
    tools_schema = [t.to_schema() for t in all_tools()]
    tool_map: dict[str, Tool] = {t.name: t for t in all_tools()}

    # 追加用户消息到会话历史
    messages.append({"role": "user", "content": user_text})

    for _round in range(max_tool_rounds):
        resp = chat_with_tools(messages, tools=tools_schema)
        msg = resp.choices[0].message

        # LLM 没调工具 → 最终回复（也可能是纯对话/纯计划）
        if not msg.tool_calls:
            turn.reply = msg.content or "(模型未给出回复)"
            messages.append({"role": "assistant", "content": turn.reply})
            return turn

        # 捕获中书省计划（第一轮 tool_call 附带的 content = 执行计划）
        if _round == 0 and msg.content and not turn.plan:
            turn.plan = msg.content.strip()

        # 把 assistant 的 tool_calls 消息加入历史
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        })

        # 执行每个工具调用，结果回灌
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {}

            # ★ 门下省审查：危险操作封驳（除非用户已确认）
            assessment = gatekeeper.assess(name, args)
            if assessment is not None and not confirmed_token:
                # 发劝谏令牌，封驳执行
                ticket = gatekeeper.issue_ticket(name, args, assessment)
                turn.error = "blocked"
                turn.reply = (
                    f"⚠ **门下省封驳**：{assessment['summary']}\n\n"
                    f"{assessment['detail']}\n\n"
                    f"如确认执行，请回复「确认」或点击确认按钮。"
                )
                if on_blocked:
                    on_blocked(ticket.to_dict())
                # 回灌一个 tool 结果告诉 LLM 被封驳（让 LLM 后续知道）
                messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": f"门下省封驳：{assessment['summary']}。等待用户确认。",
                })
                return turn

            tool = tool_map.get(name)
            if tool is None:
                result_str = f"错误：未知工具 {name}"
            else:
                try:
                    result = tool.call(args)
                    result_str = _summarize_result(result)
                except Exception as e:
                    result_str = f"工具执行失败: {type(e).__name__}: {e}"
            turn.tool_calls.append({"name": name, "args": args, "result": result_str[:500]})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            })

    # 达到 max_tool_rounds 仍未结束 → 让 LLM 收尾
    messages.append({"role": "user", "content": "已达到工具调用上限，请基于已有信息给出总结。"})
    resp = chat_with_tools(messages, tools=None)
    turn.reply = resp.choices[0].message.content or "(模型未给出回复)"
    messages.append({"role": "assistant", "content": turn.reply})
    return turn


def _summarize_result(result) -> str:
    """把工具结果压缩成给 LLM 看的字符串（避免超长）。"""
    if isinstance(result, list):
        # list_runs / discover_workspace_runs 的结果
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
        # compare / fork / merge 等的结构化结果
        return json.dumps(result, ensure_ascii=False, default=str)[:1500]
    return str(result)[:1500]


def chat_once_robust(
    user_text: str,
    *,
    session_id: str | None = None,
    confirm_token: str | None = None,
) -> dict:
    """对外统一入口（带 session 上下文记忆 + 门下省封驳）。

    流程：
      1. 有 pending_confirm 且 confirm_token 匹配 → 执行被封驳的操作
      2. 否则走 LLM 对话（带 session history）
      3. LLM 未配置/失败 → 规则兜底

    返回 {mode, reply, tool_calls, session_id, blocked?, confirm?}
    """
    from control.agent import session as sess

    # 确保 session 存在，拿到 history
    session_id, messages = sess.get_or_create(session_id)

    # ★ 门下省确认流程：用户回传了 confirm_token
    if confirm_token and confirm_token != "REJECT":
        pending = sess.pop_pending_confirm_if_match(session_id, confirm_token)
        if pending is not None:
            # 确认通过（原子取出+清除）→ 执行被封驳的操作
            tool_name = pending["tool_name"]
            tool_args = pending["tool_args"]
            try:
                from control.agent.tools import call_tool
                result = call_tool(tool_name, tool_args)
                reply = f"✓ 已执行（经门下省确认）：{pending['summary']}\n\n{_summarize_result(result)}"
                mode = "confirmed"
            except Exception as e:
                reply = f"✗ 确认后执行仍失败：{type(e).__name__}: {e}"
                mode = "error"
            sess.append(session_id, "user", user_text or "（确认执行）")
            sess.append(session_id, "assistant", reply)
            return {"mode": mode, "reply": reply, "session_id": session_id,
                    "tool_calls": [{"name": tool_name, "args": tool_args}]}

    # 用户拒绝了（发来 REJECT 信号）→ 清 pending
    if confirm_token == "REJECT":
        sess.set_pending_confirm(session_id, None)
        reply = "已取消该操作。"
        sess.append(session_id, "user", user_text or "（取消）")
        sess.append(session_id, "assistant", reply)
        return {"mode": "cancelled", "reply": reply, "session_id": session_id}

    # 正常对话分支：清除 stale pending_confirm（用户没确认也没拒绝就发了新问题）
    sess.set_pending_confirm(session_id, None)

    if not is_llm_configured():
        reply = _rule_chat_one(user_text)
        sess.append(session_id, "user", user_text)
        sess.append(session_id, "assistant", reply)
        return {"mode": "rule", "reply": reply, "session_id": session_id}

    blocked_ticket = None

    def _on_blocked(ticket_dict):
        nonlocal blocked_ticket
        blocked_ticket = ticket_dict
        sess.set_pending_confirm(session_id, ticket_dict)

    try:
        turn = chat_with_llm(
            user_text, messages,
            confirmed_token=confirm_token,
            on_blocked=_on_blocked,
        )
        result = {
            "mode": "llm", "reply": turn.reply, "session_id": session_id,
            "tool_calls": turn.tool_calls,
            "plan": turn.plan,
        }
        if turn.error == "blocked" and blocked_ticket:
            result["blocked"] = blocked_ticket
        return result
    except Exception as e:
        # LLM 失败 → 规则兜底（但仍记录进 session）
        reply = _rule_chat_one(user_text)
        sess.append(session_id, "assistant", reply)
        return {"mode": "fallback", "reply": reply, "session_id": session_id,
                "llm_error": f"{type(e).__name__}: {e}"}
