"""control.agent.session — 对话 session 管理（后端存 history）。

每个 session 维护一份 LLM messages 历史，使控制台对话有跨消息的上下文记忆。
用户可追问、可指代「刚才那个」。

设计：
  - 进程内 dict 存储（_SESSIONS），按 session_id 索引。
  - TTL 自动清理：超过 2h 未活动的 session 回收，防内存泄漏。
  - history 上限 40 条（~20 轮对话），超出时滑窗丢弃最旧的非 system 消息。
  - 首次对话时注入 system prompt（控制助手角色）。

session_id 由前端生成（首次进入控制台时 uuid），随每次 chat 请求带上。
"""

from __future__ import annotations

import time
import uuid
from threading import Lock

from control.agent.chat import _SYSTEM_PROMPT

# session 存储：session_id → {"messages": [...], "last_active": ts, "pending_confirm": ...}
_SESSIONS: dict[str, dict] = {}
_LOCK = Lock()

_SESSION_TTL = 7200          # 2h 未活动 → 清理
_HISTORY_MAX = 40            # 每会话最多保留 40 条消息


def _gc() -> None:
    """清理过期 session（调用方持锁）。"""
    now = time.time()
    expired = [sid for sid, s in _SESSIONS.items() if now - s["last_active"] > _SESSION_TTL]
    for sid in expired:
        del _SESSIONS[sid]


def get_or_create(session_id: str | None) -> tuple[str, list[dict]]:
    """获取或创建 session，返回 (session_id, messages)。

    session_id 为 None 时新建（分配 uuid）。
    messages 是该 session 的 LLM 消息历史（含 system）。
    """
    with _LOCK:
        _gc()
        if session_id is None or session_id not in _SESSIONS:
            session_id = session_id or uuid.uuid4().hex[:12]
            _SESSIONS[session_id] = {
                "messages": [{"role": "system", "content": _SYSTEM_PROMPT}],
                "last_active": time.time(),
                "pending_confirm": None,   # 门下省待确认的操作
            }
        s = _SESSIONS[session_id]
        s["last_active"] = time.time()
        # 返回原始 list 引用（非副本）：chat_with_llm 会直接 append 到此 list，
        # 使上下文跨请求累积。对话按 session 串行（前端 _chatBusy 保证），无并发写。
        return session_id, s["messages"]


def append(session_id: str, role: str, content: str, **extra) -> None:
    """向 session 追加一条消息（role/content + 可选 tool_calls/tool_call_id）。"""
    with _LOCK:
        s = _SESSIONS.get(session_id)
        if s is None:
            return
        msg = {"role": role, "content": content}
        msg.update(extra)
        s["messages"].append(msg)
        s["last_active"] = time.time()
        # 滑窗：超出上限时丢弃最旧的非 system 消息
        while len(s["messages"]) > _HISTORY_MAX and s["messages"][0]["role"] != "system":
            s["messages"].pop(0)


def append_raw(session_id: str, message: dict) -> None:
    """追加一条已构造好的完整消息（含 tool_calls 等复杂结构）。"""
    with _LOCK:
        s = _SESSIONS.get(session_id)
        if s is None:
            return
        s["messages"].append(message)
        s["last_active"] = time.time()
        while len(s["messages"]) > _HISTORY_MAX and s["messages"][0]["role"] != "system":
            s["messages"].pop(0)


def get_pending_confirm(session_id: str) -> dict | None:
    """获取该 session 待确认的危险操作（门下省封驳用）。"""
    with _LOCK:
        s = _SESSIONS.get(session_id)
        return s["pending_confirm"] if s else None


def set_pending_confirm(session_id: str, confirm: dict | None) -> None:
    """设置/清除待确认操作。confirm = {token, action, summary, detail}。"""
    with _LOCK:
        s = _SESSIONS.get(session_id)
        if s is not None:
            s["pending_confirm"] = confirm


def reset(session_id: str) -> None:
    """清空 session 历史（用户点「重新开始」）。"""
    with _LOCK:
        if session_id in _SESSIONS:
            _SESSIONS[session_id] = {
                "messages": [{"role": "system", "content": _SYSTEM_PROMPT}],
                "last_active": time.time(),
                "pending_confirm": None,
            }


def session_count() -> int:
    """当前活跃 session 数（诊断用）。"""
    with _LOCK:
        return len(_SESSIONS)
