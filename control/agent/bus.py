"""control.agent.bus — 三省消息总线。

中书省 / 尚书省 / 门下省 不直接互调（避免循环依赖），而是经本总线通信：

  publish(BusMessage) → 总线同步派发给所有匹配的订阅者 → 订阅者回调

设计：
  - 进程内单例（_BUS），线程安全（订阅/发布可跨线程，如尚书省线程池内 publish）。
  - 订阅者注册 (dept, kinds, callback)；发布的消息 to_dept 匹配或为「全员」时派发。
  - 最近 N 条消息留存内存，供前端 /api/control/bus/feed?since=<ts> 拉取补全。
  - 派发同步串行（一个消息的所有订阅者依次执行），回调内异常不中断后续派发（记录但不抛）。

为什么需要总线：
  门下省「常态挂起，监听三省所有内容」——它是订阅者而非被动函数。
  总线让门下省能监听尚书省每一步、随时介入，而不需要尚书省知道门下省存在（解耦）。
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from threading import Lock

# 部门标识
ZHONGSHU = "中书省"
SHANGSHU = "尚书省"
MENXIA = "门下省"
USER = "用户"
ALL = "全员"  # 广播

# 消息 kind 常量（三省约定的词汇表）
# 中书省 → 尚书省
KIND_PLAN_REQUEST = "plan_request"        # 中书省请尚书省拟案
# 尚书省 → 中书省（经总线回呈）
KIND_PLAN_DRAFTED = "plan_drafted"        # 尚书省拟好了 Plan
KIND_PLAN_PROGRESS = "plan_progress"      # Plan 执行进度更新（某步状态变化）
KIND_PLAN_DONE = "plan_done"              # Plan 全部执行完
# 尚书省 → 门下省（每步执行前后广播）
KIND_STEP_START = "step_start"            # 某步即将执行（门下省审查时机）
KIND_STEP_DONE = "step_done"              # 某步成功
KIND_STEP_BLOCKED = "step_blocked"        # 某步被封驳
KIND_STEP_FAILED = "step_failed"          # 某步执行异常
# 门下省 → 中书省 / 尚书省
KIND_REVIEW = "review"                    # 门下省审查简报（plan_done 后自动，或主动）
KIND_BLOCK = "block"                      # 门下省封驳令（附 ticket）
KIND_UNBLOCK = "unblock"                  # 用户准奏后放行
# 用户经中书省
KIND_PLAN_APPROVED = "plan_approved"      # 用户准奏
KIND_PLAN_REJECTED = "plan_rejected"      # 用户驳回


@dataclass
class BusMessage:
    """总线消息。"""
    from_dept: str               # 发送部门
    to_dept: str                 # 接收部门（ALL=广播）
    kind: str                    # 消息类型（KIND_* 常量）
    payload: dict                # 消息体（结构由 kind 约定）
    ts: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> dict:
        return asdict(self)


# 订阅者：callback(BusMessage) -> None
Subscriber = Callable[[BusMessage], None]

_BUS_MAX_RECENT = 500   # 留存最近 500 条（前端补全用）


class MessageBus:
    """进程内消息总线单例。"""

    def __init__(self) -> None:
        self._lock = Lock()
        self._subs: list[tuple[str, list[str], Subscriber]] = []  # (dept, kinds, cb)
        self._recent: list[BusMessage] = []

    def subscribe(self, dept: str, kinds: list[str], callback: Subscriber) -> None:
        """订阅。dept 匹配 to_dept（含 ALL 广播）；kinds 为空列表=接收所有 kind。"""
        with self._lock:
            self._subs.append((dept, kinds, callback))

    def publish(self, msg: BusMessage) -> None:
        """发布消息。同步派发给所有匹配订阅者（回调内异常不中断后续派发）。"""
        with self._lock:
            self._recent.append(msg)
            if len(self._recent) > _BUS_MAX_RECENT:
                self._recent = self._recent[-_BUS_MAX_RECENT:]
            # 拍照（持锁内拷贝订阅者列表，回调在锁外执行避免死锁）
            targets = [
                cb for (dept, kinds, cb) in self._subs
                if (msg.to_dept in (dept, ALL)) and (not kinds or msg.kind in kinds)
            ]
        for cb in targets:
            try:
                cb(msg)
            except Exception:
                # 订阅者异常不阻断总线（记录到 stderr 但不抛）
                import sys
                print(f"[bus] 订阅者回调异常 (kind={msg.kind}): {sys.exc_info()[1]}", file=sys.stderr)

    def recent(self, since_ts: float = 0.0, dept: str | None = None,
              kinds: list[str] | None = None) -> list[BusMessage]:
        """取 since_ts 之后的消息（可选按部门/kind 过滤）。供前端 feed 拉取。"""
        with self._lock:
            msgs = list(self._recent)
        out = []
        for m in msgs:
            if m.ts <= since_ts:
                continue
            if dept and m.to_dept not in (dept, ALL) and m.from_dept != dept:
                continue
            if kinds and m.kind not in kinds:
                continue
            out.append(m)
        return out

    def clear(self) -> None:
        """清空留存（测试用）。"""
        with self._lock:
            self._recent.clear()


# 模块级单例
_BUS: MessageBus | None = None


def get_bus() -> MessageBus:
    """获取全局总线单例（惰性创建）。"""
    global _BUS
    if _BUS is None:
        _BUS = MessageBus()
    return _BUS


def reset_bus() -> None:
    """重置单例（测试用）。"""
    global _BUS
    _BUS = None
