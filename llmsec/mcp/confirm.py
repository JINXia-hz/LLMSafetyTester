"""llmsec.mcp.confirm — 危险写操作的两步确认 token 机制。

模式：preview → 返回 {summary, confirm_token} → agent 二次调 *_confirm(token) 才真执行。

设计要点：
  - token 存内存 dict（MCP server 单进程生命周期内有效，重启即失效）
  - TTL 5 分钟，过期自动失效
  - 一次性：确认后即删，防止重放
  - 线程安全（MCP server 可能并发处理请求）
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any

_TTL_SECONDS = 300  # 5 分钟
_LOCK = threading.Lock()


@dataclass
class _Pending:
    """一个待确认的操作。"""

    action: str                       # 操作类型标识，如 "delete_runs"
    summary: dict[str, Any]           # 人类可读的影响摘要（原样返回给 agent）
    execute_fn: Any                   # 确认时调用的零参可调用对象，返回执行结果
    args_repr: str = ""               # 参数的人类可读描述（调试用）
    created: float = field(default_factory=time.time)


_PENDING: dict[str, _Pending] = {}


def issue(action: str, summary: dict[str, Any], execute_fn: Any, *, args_repr: str = "") -> str:
    """登记一个待确认操作，返回 confirm_token。

    Args:
        action: 操作类型标识（用于日志/调试）。
        summary: 影响摘要 dict，确认时原样回传给 agent 供其参考。
        execute_fn: 零参可调用对象，确认时调用，返回值作为执行结果。
        args_repr: 参数的可读描述（可选）。

    Returns:
        confirm_token 字符串。
    """
    token = secrets.token_urlsafe(8)
    with _LOCK:
        _gc()  # M-7：_gc 迭代/修改 _PENDING，必须持锁调用（见 _gc docstring）
        _PENDING[token] = _Pending(
            action=action, summary=summary, execute_fn=execute_fn, args_repr=args_repr
        )
    return token


def confirm(token: str) -> dict[str, Any]:
    """用 token 执行待确认操作。

    Returns:
        {"status": "executed", "result": <execute_fn 返回值>}
        {"status": "expired", ...}   token 不存在或已过期
        {"status": "already_confirmed", ...}  token 已被用过（一次性）
    """
    with _LOCK:
        _gc()
        p = _PENDING.pop(token, None)
    if p is None:
        # pop 后无法区分"不存在"和"已确认过"——两者对 agent 行为一致（都需重新 preview）
        return {
            "status": "expired_or_already_confirmed",
            "detail": "token 无效、已过期或已使用。请重新调用 preview 获取新 token。",
        }
    result = p.execute_fn()
    return {"status": "executed", "result": result}


def _gc() -> None:
    """清除过期条目。M-7：**调用方必须已持有 _LOCK**。

    _gc 迭代并修改 _PENDING；锁外调用与 confirm 的持锁 pop 并发会抛
    "dictionary changed size during iteration"（threading.Lock 不可重入，
    不能由 _gc 自己加锁——confirm 在持锁状态下调用它）。
    """
    now = time.time()
    expired = [t for t, p in _PENDING.items() if now - p.created > _TTL_SECONDS]
    for t in expired:
        _PENDING.pop(t, None)


def clear() -> None:
    """清空所有待确认操作（测试用）。"""
    with _LOCK:
        _PENDING.clear()
