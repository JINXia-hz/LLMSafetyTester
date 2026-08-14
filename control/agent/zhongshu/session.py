"""control.agent.session — 对话 session 管理（后端存 history）。

每个 session 维护一份 LLM messages 历史，使控制台对话有跨消息的上下文记忆。
用户可追问、可指代「刚才那个」。

设计：
  - 进程内 dict 存储（_SESSIONS），按 session_id 索引。
  - TTL 自动清理：超过 2h 未活动的 session 回收，防内存泄漏。
  - history 上限 40 条（~20 轮对话），超出时滑窗丢弃最旧的非 system 消息。
  - 首次对话时注入 system prompt（中书省角色）。

session_id 由前端生成（首次进入控制台时 uuid），随每次 chat 请求带上。

注意：封驳（门下省）不再经 session 管理——走总线 + menxia.py 的 block 机制。
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock

from control.agent.prompts import ZHONGSHU_PROMPT as _SYSTEM_PROMPT

# session 存储：session_id → {"messages": [...], "last_active": ts}
_SESSIONS: dict[str, dict] = {}
_LOCK = Lock()

# per-session 对话处理锁：同一 session 的 handle_message 服务端串行化
_SESSION_LOCKS: dict[str, Lock] = {}
_LOCKS_GUARD = Lock()

_SESSION_TTL = 7200          # 2h 未活动 → 清理
_HISTORY_MAX = 40            # 每会话最多保留 40 条消息


def _gc() -> None:
    """清理过期 session（调用方持锁）。对应的对话锁一并回收。"""
    now = time.time()
    expired = [sid for sid, s in _SESSIONS.items() if now - s["last_active"] > _SESSION_TTL]
    for sid in expired:
        del _SESSIONS[sid]
        with _LOCKS_GUARD:
            _SESSION_LOCKS.pop(sid, None)


def _session_lock(session_id: str) -> Lock:
    with _LOCKS_GUARD:
        lock = _SESSION_LOCKS.get(session_id)
        if lock is None:
            lock = Lock()
            _SESSION_LOCKS[session_id] = lock
        return lock


@contextmanager
def conversation_lock(session_id: str) -> Iterator[None]:
    """同一 session 的对话处理串行化（服务端）。

    此前"同一 session 同时只有一条 chat 在处理"由前端 _chatBusy 保证——
    服务端并发调用同一 session 时，两条对话的 tool 消息会交错、上下文互相
    污染。锁在 handle_message 全程持有（含 LLM 往返），后来的请求排队等待。
    """
    lock = _session_lock(session_id)
    lock.acquire()
    try:
        yield
    finally:
        lock.release()


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
            }
        s = _SESSIONS[session_id]
        s["last_active"] = time.time()
        # 返回原始 list 引用（非副本）：调用方会经 append_message 持锁写入，
        # 使上下文跨请求累积。对话按 session 串行（conversation_lock 保证，
        # 前端 _chatBusy 只是第一道防线）。
        return session_id, s["messages"]


def append(session_id: str, role: str, content: str) -> None:
    """向 session 追加一条纯文本消息（带 tool_calls/tool_call_id 的消息走 append_message）。"""
    append_message(session_id, {"role": role, "content": content})


def append_message(session_id: str, msg: dict) -> None:
    """向 session 追加一条已构造好的消息 dict（持锁）。

    dialogue 的 ReAct 循环需要追加带 tool_calls/tool_call_id 的原始消息——
    此前直接对 messages 引用无锁 append，与 append() 的持锁写不一致。
    """
    with _LOCK:
        s = _SESSIONS.get(session_id)
        if s is None:
            return
        s["messages"].append(msg)
        s["last_active"] = time.time()
        _trim(s["messages"])


def _trim(messages: list[dict]) -> None:
    """滑窗：超出上限时丢弃最旧的非 system 消息（system 固定在 index 0）。

    按完整对话单元裁剪：弹掉一条消息后，若因此产生孤儿（无对应 assistant 的
    tool 应答、或 tool_calls 后紧跟的不是 tool 应答），把孤儿一并弹掉——
    OpenAI API 要求 tool 消息必须紧跟携带对应 tool_call_id 的 assistant 消息，
    配对被拆散的 history 会让该 session 的后续请求整体报废。
    """
    while len(messages) > _HISTORY_MAX and len(messages) > 1:
        messages.pop(1)
        while len(messages) > 1:
            head = messages[1]
            nxt = messages[2] if len(messages) > 2 else None
            if head.get("role") == "tool":
                # 它对应的 assistant(tool_calls) 已被裁掉 → 孤儿应答
                messages.pop(1)
                continue
            if (head.get("role") == "assistant" and head.get("tool_calls")
                    and (nxt is None or nxt.get("role") != "tool")):
                # 它的 tool 应答已被裁掉 → 孤儿 tool_calls
                messages.pop(1)
                continue
            break


def reset(session_id: str) -> None:
    """清空 session 历史（用户点「重新开始」）。"""
    with _LOCK:
        if session_id in _SESSIONS:
            _SESSIONS[session_id] = {
                "messages": [{"role": "system", "content": _SYSTEM_PROMPT}],
                "last_active": time.time(),
            }
