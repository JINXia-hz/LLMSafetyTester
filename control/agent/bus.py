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
# 尚书省广播：Plan 拟案/执行进度 + 每步状态
KIND_PLAN_DRAFTED = "plan_drafted"       # 尚书省拟案完成（中书省发通知，门下省审查时机）
KIND_PLAN_PROGRESS = "plan_progress"      # Plan 执行进度更新（某步状态变化）
KIND_PLAN_DONE = "plan_done"              # Plan 全部执行完
KIND_STEP_START = "step_start"            # 某步即将执行（门下省审查时机）
KIND_STEP_DONE = "step_done"              # 某步成功
KIND_STEP_BLOCKED = "step_blocked"        # 某步被封驳
KIND_STEP_UNBLOCKED = "step_unblocked"    # 封驳令被清除（用户放行 / 随 Plan 驳回撤销）
KIND_STEP_FAILED = "step_failed"          # 某步执行异常
# 门下省 → 中书省
KIND_REVIEW = "review"                    # 门下省审查简报（plan_done 后自动，或主动）
KIND_BLOCK = "block"                      # 门下省封驳令（附 ticket）
# 用户经中书省
KIND_PLAN_APPROVED = "plan_approved"      # 用户准奏
KIND_PLAN_REJECTED = "plan_rejected"      # 用户驳回

# r8/病根2：kind → 接收部门的**声明式路由表**。发布方一律走 notify_routed()，
# 不再在调用点手写 to_dept——H-2 的教训：emit 点写错部门名（to_dept=SHANGSHU 而
# 唯一订阅方是 MENXIA）时 bus 的 to_dept in (dept, ALL) 过滤永不匹配，订阅方
# 静默失联、无任何报错。路由集中在此处后，发布点不可能再拼错部门。
KIND_ROUTES: dict[str, str] = {
    KIND_PLAN_DRAFTED: ALL,       # 门下省（订阅）+ 面板存档
    KIND_PLAN_PROGRESS: ALL,      # 面板/进度回调存档
    KIND_PLAN_DONE: ALL,          # 门下省（订阅，事后审查）
    KIND_STEP_START: MENXIA,      # 请求-应答：门下省同步返回封驳裁决
    KIND_STEP_DONE: ALL,
    KIND_STEP_BLOCKED: ALL,
    KIND_STEP_UNBLOCKED: ALL,     # 门下省面板据此递减待裁计数（跨标签页/重放配平）
    KIND_STEP_FAILED: ALL,        # 门下省（订阅，异常呈递）
    KIND_REVIEW: ZHONGSHU,        # 门下省 → 中书省面板展示
    KIND_BLOCK: ALL,              # 封驳令归档（票据本体走 block store）
    KIND_PLAN_APPROVED: ALL,      # 门下省（订阅，准奏阶段风险评估）
    KIND_PLAN_REJECTED: ALL,
}


@dataclass
class BusMessage:
    """总线消息。

    公共信封字段（plan_id / intent / session_id）让每条消息自解释——
    前端轮询 bus feed 时不需要额外反查 Plan 就知道"这是哪个计划的、为了什么"。
    """
    from_dept: str               # 发送部门
    to_dept: str                 # 接收部门（ALL=广播）
    kind: str                    # 消息类型（KIND_* 常量）
    payload: dict                # 消息体（结构由 kind 约定）
    ts: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    # 公共信封（可选，plan 相关消息必带）
    plan_id: str | None = None
    intent: str | None = None
    session_id: str | None = None

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

    def publish(self, msg: BusMessage, *, collect_replies: bool = False) -> list | None:
        """发布消息。同步派发给所有匹配订阅者（回调内异常不中断后续派发）。

        collect_replies=True 时收集各订阅者的非 None 返回值并返回（请求-应答式
        派发：尚书省 step_start → 门下省同步返回封驳裁决，避免"事后扫 recent
        反推裁决"的旁路——时间窗/消息挤出/消费去重等一整类问题随之消失）。
        """
        with self._lock:
            self._recent.append(msg)
            if len(self._recent) > _BUS_MAX_RECENT:
                self._recent = self._recent[-_BUS_MAX_RECENT:]
            # 拍照（持锁内拷贝订阅者列表，回调在锁外执行避免死锁）
            targets = [
                cb for (dept, kinds, cb) in self._subs
                if (msg.to_dept in (dept, ALL)) and (not kinds or msg.kind in kinds)
            ]
        replies: list = []
        for cb in targets:
            try:
                r = cb(msg)
            except Exception:
                # 订阅者异常不阻断总线（记录到 stderr 但不抛）
                import sys
                print(f"[bus] 订阅者回调异常 (kind={msg.kind}): {sys.exc_info()[1]}", file=sys.stderr)
                continue
            if collect_replies and r is not None:
                replies.append(r)
        return replies if collect_replies else None

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


# 模块级单例（锁内惰性创建：两线程首次并发 get_bus 不各建一个——
# 各建的总线订阅者只挂在其中一个上，消息丢失）
_BUS: MessageBus | None = None
_BUS_LOCK = Lock()


def get_bus() -> MessageBus:
    """获取全局总线单例（惰性创建）。"""
    global _BUS
    if _BUS is None:
        with _BUS_LOCK:
            if _BUS is None:
                _BUS = MessageBus()
    return _BUS


def reset_bus() -> None:
    """重置单例（测试用）。"""
    global _BUS
    _BUS = None


def notify(
    kind: str,
    *,
    from_dept: str,
    to_dept: str = ALL,
    plan_id: str | None = None,
    intent: str | None = None,
    session_id: str | None = None,
    collect_replies: bool = False,
    **payload,
) -> list | None:
    """构造带公共信封的 BusMessage 并发布。三省统一用这个发消息。

    payload 的 key-value 直接成为消息体，无需手动包 dict。
    例：notify(KIND_STEP_START, from_dept=SHANGSHU, plan_id=p.id, intent=p.intent,
              step_id=s.id, capability=s.capability)

    collect_replies=True：返回匹配订阅者的非 None 返回值列表（请求-应答式派发，
    供 executor 直收门下省的封驳裁决）；默认 None。
    """
    msg = BusMessage(
        from_dept=from_dept, to_dept=to_dept, kind=kind,
        payload=dict(payload),
        plan_id=plan_id, intent=intent, session_id=session_id,
    )
    return get_bus().publish(msg, collect_replies=collect_replies)


def notify_routed(
    kind: str,
    *,
    from_dept: str,
    plan_id: str | None = None,
    intent: str | None = None,
    session_id: str | None = None,
    collect_replies: bool = False,
    **payload,
) -> list | None:
    """按 KIND_ROUTES 路由表发布消息（r8/病根2：发布点不再手写 to_dept）。

    生产代码一律用本函数；notify() 保留给测试/特殊场景（显式指定接收方）。
    kind 不在路由表中时抛 KeyError——新 kind 必须先在表里登记路由，
    避免拼错 kind 静默广播。
    """
    if kind not in KIND_ROUTES:
        raise KeyError(f"未登记路由的消息 kind: {kind!r}（请先在 bus.KIND_ROUTES 登记）")
    return notify(
        kind, from_dept=from_dept, to_dept=KIND_ROUTES[kind],
        plan_id=plan_id, intent=intent, session_id=session_id,
        collect_replies=collect_replies, **payload,
    )
