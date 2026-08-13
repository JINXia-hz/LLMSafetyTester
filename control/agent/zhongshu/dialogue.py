"""control.agent.zhongshu.dialogue — 中书省对话主循环（前台 + 意图理解 + 润色）。

中书省是对话主入口：
  简单查询 → 自己用工具处理
  复杂指令 → 调尚书省 draft_plan → 润色成给用户的中文方案 → 返回 plan_pending

人设：称用户「陛下」，自称「臣」。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from control.agent.llm import chat_with_tools, is_llm_configured
from control.agent.prompts import ZHONGSHU_PROMPT as _SYSTEM_PROMPT
from control.agent.zhongshu import session as sess
from control.agent.zhongshu.fallback import chat_one as _rule_chat_one
from control.agent.zhongshu.tools import all_tools, call_tool


def _simple_tools() -> list[dict]:
    """中书省保留的简单查询工具 schema（执行类交给尚书省）。"""
    keep = {"list_runs", "compare_runs", "review_run", "list_workspaces"}
    return [t.to_schema() for t in all_tools() if t.name in keep]


def _search_history_schema() -> dict:
    """search_history 工具：按关键词搜索历史文牍，帮中书省了解过去做过什么。"""
    return {
        "type": "function",
        "function": {
            "name": "search_history",
            "description": (
                "按关键词搜索历史执行记录（文牍），了解过去做过什么、怎么做的。"
                "用于：用户说「上次」「之前」「按照以前的」等引用历史的指令时查找参考。"
                "返回匹配的 Plan 摘要（意图、步骤、执行结果）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "搜索关键词，如 ['minimax', '评估'] 或 ['judge', '改']",
                    },
                    "recent": {
                        "type": "integer",
                        "description": "最多返回多少条（默认 5）",
                        "default": 5,
                    },
                },
                "required": ["keywords"],
            },
        },
    }


def _do_search_history(args: dict) -> str:
    """执行历史搜索：在文牍 intent/user_text/steps 里匹配关键词。"""
    keywords = [k.lower() for k in args.get("keywords", [])]
    recent = args.get("recent", 5)
    if not keywords:
        return "（未提供关键词）"

    from control.agent import gazette
    results = []
    for g in gazette.list_gazettes(recent=50):
        events = gazette.read_events(g.get("plan_id", ""))
        if not events:
            continue
        # 拼接可搜索文本：intent + user_text + steps descriptions
        searchable = (g.get("intent", "") or "").lower()
        steps_desc = []
        for ev in events:
            if ev.kind == gazette.EV_PLAN_DRAFTED:
                searchable += " " + (ev.detail.get("user_text", "") or "").lower()
                for s in ev.detail.get("steps_summary", []):
                    desc = s.get("description", "")
                    steps_desc.append(desc)
                    searchable += " " + desc.lower()
        # 任一关键词命中即收录
        if any(kw in searchable for kw in keywords):
            # 统计执行结果
            stats = {}
            for ev in events:
                if ev.kind == gazette.EV_PLAN_FINISHED:
                    stats = ev.detail.get("step_stats", {})
            results.append({
                "plan_id": g.get("plan_id", "")[:12],
                "intent": g.get("intent", "")[:100],
                "status": g.get("status", "?"),
                "steps": steps_desc[:5],
                "step_stats": stats,
            })
        if len(results) >= recent:
            break

    if not results:
        return f"未找到匹配「{', '.join(keywords)}」的历史记录。"
    import json
    return json.dumps(results, ensure_ascii=False, default=str)


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
                "如果你搜到了历史参考，在 reference 里附上让尚书省据此拟案。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "description": "清晰、准确转述的用户指令（尚书省靠这个拟案）",
                    },
                    "reference": {
                        "type": "string",
                        "description": "（可选）历史参考信息——如「参考文牍 X，上次对 minimax 跑评估用了 5 轮自适应」",
                    },
                },
                "required": ["intent"],
            },
        },
    }


@dataclass
class ZhongshuTurn:
    """中书省一轮对话的结果。"""
    user_text: str
    reply: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    plan_pending: dict | None = None
    mode: str = "llm"
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


def handle_message(
    user_text: str,
    *,
    session_id: str | None = None,
) -> dict:
    """中书省主入口。处理用户消息，返回结构化结果。"""
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
        reply = _rule_chat_one(user_text)
        sess.append(session_id, "assistant", reply)
        return {"mode": "fallback", "reply": reply, "session_id": session_id,
                "llm_error": f"{type(e).__name__}: {e}"}


def _react_loop(user_text: str, messages: list[dict], session_id: str,
                max_rounds: int = 4) -> ZhongshuTurn:
    """中书省 ReAct 循环：LLM + 简单工具 + 历史搜索 + request_shangshu_plan。"""
    turn = ZhongshuTurn(user_text=user_text)
    tools_schema = _simple_tools() + [_search_history_schema()] + [_shangshu_plan_schema()]

    for _round in range(max_rounds):
        resp = chat_with_tools(messages, tools=tools_schema)
        msg = resp.choices[0].message

        if not msg.tool_calls:
            turn.reply = msg.content or "(臣无以作答)"
            messages.append({"role": "assistant", "content": turn.reply})
            sess.append(session_id, "assistant", turn.reply)
            return turn

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

            if name == "request_shangshu_plan":
                intent = args.get("intent", user_text)
                reference = args.get("reference", "")
                plan_dict = _hand_to_shangshu(intent, messages, session_id, reference=reference)
                if plan_dict:
                    turn.plan_pending = plan_dict
                    turn.reply = plan_dict.get("rendered_plan", "尚书省已拟案，请陛下过目。")
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

            if name == "search_history":
                result_str = _do_search_history(args)
                turn.tool_calls.append({"name": name, "args": args, "result": result_str[:500]})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_str})
                continue

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

    messages.append({"role": "user", "content": "请基于已有信息总结回复。"})
    resp = chat_with_tools(messages, tools=None)
    turn.reply = resp.choices[0].message.content or "(臣无以作答)"
    messages.append({"role": "assistant", "content": turn.reply})
    sess.append(session_id, "assistant", turn.reply)
    return turn


def _hand_to_shangshu(intent: str, messages: list[dict], session_id: str,
                      reference: str = "") -> dict | None:
    """转交尚书省拟案，再润色成给用户看的方案。

    文牍流程：拟案 → 记录拟案完成（含用户原始指令）。
    reference 是中书省搜到的历史参考，附加到 intent 帮尚书省拟更精准的方案。
    """
    try:
        from control.agent import gazette
        from control.agent.shangshu import draft_plan

        # 如果有历史参考，附加到 intent 让尚书省看到
        full_intent = intent
        if reference:
            full_intent = f"{intent}\n\n[历史参考]\n{reference}"

        plan = draft_plan(full_intent, session_id=session_id)
        rendered = _render_plan_for_user(plan, messages)

        # 记录拟案完成（user_text 存原始指令，不再用 __pending__ 占位）
        gazette.append_event(plan.id, gazette.EV_PLAN_DRAFTED, "尚书省",
                             session_id=session_id, intent=plan.intent,
                             detail={"user_text": intent,
                                     "steps_count": len(plan.steps),
                                     "steps_summary": [
                                         {"id": s.id, "capability": s.capability,
                                          "description": s.description}
                                         for s in plan.steps]})

        # 通知门下省审查拟案（门下省在拟案阶段做整体合理性审查）
        from control.agent.bus import KIND_PLAN_DRAFTED, ZHONGSHU, notify
        notify(KIND_PLAN_DRAFTED, from_dept=ZHONGSHU,
               plan_id=plan.id, intent=plan.intent, session_id=session_id,
               steps_count=len(plan.steps))

        # 便宜行事：如果所有步骤都标了 auto_execute=True，直接提交执行不走准奏
        all_auto = all(s.auto_execute for s in plan.steps)
        if all_auto:
            from control.agent.shangshu import approve_plan, get_queue
            approve_plan(plan.id)  # 自动准奏
            get_queue().submit(plan.id)  # 提交执行队列
            return {
                "plan_id": plan.id,
                "steps": [_step_dict(s) for s in plan.steps],
                "rendered_plan": rendered,
                "auto_executed": True,  # ★ 前端据此不展示准奏卡片
            }

        return {
            "plan_id": plan.id,
            "steps": [_step_dict(s) for s in plan.steps],
            "rendered_plan": rendered,
        }
    except Exception as e:
        import sys
        print(f"[中书省] 转交尚书省失败: {e}", file=sys.stderr)
        return None


def _step_dict(s) -> dict:
    """Step 对象转 dict。"""
    return {
        "id": s.id, "capability": s.capability, "args": s.args,
        "depends_on": s.depends_on, "description": s.description,
        "status": s.status, "auto_execute": s.auto_execute,
    }


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
