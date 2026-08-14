"""control.agent.zhongshu.dialogue — 中书省对话主循环（前台 + 意图理解 + 润色）。

中书省是对话主入口：
  简单查询 → 自己用工具处理
  复杂指令 → 调尚书省 draft_plan → 润色成给用户的中文方案 → 返回 plan_pending

人设：称用户「陛下」，自称「臣」。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from control.agent.llm import chat_with_tools, is_llm_configured, parse_tool_args, rebuild_tool_calls
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
                "多个关键词为 AND 语义（须同时命中），返回匹配的 Plan 摘要（意图、步骤、执行结果），"
                "按相关性排序。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "搜索关键词（AND 语义——须全部命中），如 ['minimax', '评估'] 或 ['judge', '改']",
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


# 文牍可搜索文本的进程内缓存。
# 缓存每个 plan 的分层文本（intent/user_text/steps_text/steps_desc/stats），
# 按 _index.json 的 mtime 失效——append_event 写索引时 mtime 更新，缓存自动重建。
# 阶段 2 优化：消除每次 search_history 重复读 jsonl 全文 + 逐行 json.loads 的开销。
# 数据天然小（每 plan ~60 字符），缓存让"多次搜索"零 I/O 成本；未来文牍增长到数百时
# 叠加 2-gram 倒排索引进一步加速（当前规模缓存已足够，索引暂不引入）。
_GAZETTE_TEXT_CACHE: tuple[float, dict[str, dict]] | None = None


def _load_gazette_texts() -> dict[str, dict]:
    """加载所有文牍的可搜索文本（进程内缓存，按 _index.json mtime 失效）。

    Returns:
        {plan_id: {intent, user_text, steps_text, steps_desc, stats, status, idx_entry}}
    """
    global _GAZETTE_TEXT_CACHE
    from control.agent import gazette

    idx_path = gazette._store.path
    try:
        sig = idx_path.stat().st_mtime
    except OSError:
        sig = 0.0
    if _GAZETTE_TEXT_CACHE and _GAZETTE_TEXT_CACHE[0] == sig:
        return _GAZETTE_TEXT_CACHE[1]

    texts: dict[str, dict] = {}
    for g in gazette.list_gazettes(recent=500):
        pid = g.get("plan_id", "")
        events = gazette.read_events(pid)
        if not events:
            continue
        intent_text = (g.get("intent", "") or "").lower()
        user_text = ""
        steps_text = ""
        steps_desc = []
        stats = {}
        for ev in events:
            if ev.kind == gazette.EV_COMMISSION:
                user_text = (ev.detail.get("user_text", "") or "").lower()
            elif ev.kind == gazette.EV_PLAN_DRAFTED:
                ut = (ev.detail.get("user_text", "") or "").lower()
                if ut:
                    user_text = ut
                for s in ev.detail.get("steps_summary", []):
                    desc = s.get("description", "")
                    steps_desc.append(desc)
                    steps_text += " " + desc.lower()
            elif ev.kind == gazette.EV_STEP_STARTED:
                desc = ev.detail.get("description", "")
                if desc:
                    steps_text += " " + desc.lower()
            elif ev.kind == gazette.EV_PLAN_FINISHED:
                stats = ev.detail.get("step_stats", {})
        texts[pid] = {
            "intent": g.get("intent", "")[:100],
            "status": g.get("status", "?"),
            "_intent_lc": intent_text,
            "_user_text_lc": user_text,
            "_steps_text_lc": steps_text,
            "steps_desc": steps_desc[:5],
            "step_stats": stats,
        }
    _GAZETTE_TEXT_CACHE = (sig, texts)
    return texts


def _do_search_history(args: dict) -> str:
    """执行历史搜索：在文牍的多层文本里 AND 匹配关键词，按相关性排序。

    召回质量优化（调研发现：原 OR 语义 + 高同质数据 = 无区分度）：
      - AND 语义：所有关键词都须命中才收录，精确查找"同时涉及多个方面的 Plan"
      - 文本源补全：intent（索引层）+ commission.user_text（用户原话）
        + plan_drafted.steps_summary + step_started.description（更丰富的步骤描述）
      - 相关性排序：intent 命中权重最高(×3)，user_text 次之(×2)，步骤描述最低(×1)
      - 全量搜索：数据天然小（每 plan 平均 ~60 字符可搜索文本），无 recent 上限
    """
    keywords = [k.lower() for k in args.get("keywords", [])]
    recent = args.get("recent", 5)
    if not keywords:
        return "（未提供关键词）"

    texts = _load_gazette_texts()
    matches = []
    for pid, t in texts.items():
        intent_lc = t["_intent_lc"]
        user_lc = t["_user_text_lc"]
        steps_lc = t["_steps_text_lc"]

        # AND 语义：所有关键词都须命中（在任一文本层）
        all_texts = [intent_lc, user_lc, steps_lc]
        if not all(any(kw in txt for txt in all_texts) for kw in keywords):
            continue

        # 相关性评分：intent 命中权重最高，user_text 次之，步骤描述最低
        score = 0
        for kw in keywords:
            if kw in intent_lc:
                score += 3
            if kw in user_lc:
                score += 2
            if kw in steps_lc:
                score += 1

        matches.append({
            "plan_id": pid[:12],
            "intent": t["intent"],
            "status": t["status"],
            "steps": t["steps_desc"],
            "step_stats": t["step_stats"],
            "_score": score,  # 排序用，不回传给 LLM
        })

    # 按相关性降序（同分按 plan_id 保持稳定）
    matches.sort(key=lambda m: (-m["_score"], m["plan_id"]))
    results = [{k: v for k, v in m.items() if k != "_score"} for m in matches[:recent]]

    if not results:
        return f"未找到匹配「{', '.join(keywords)}」的历史记录（AND 语义，所有关键词须同时命中）。"
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
            "tool_calls": rebuild_tool_calls(msg),
        })

        for tc in msg.tool_calls:
            name = tc.function.name
            args = parse_tool_args(tc)

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
        rendered = _render_plan_for_user(plan)

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


def _render_plan_for_user(plan) -> str:
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
