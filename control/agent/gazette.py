"""control.agent.gazette — 文牍存储（三省任务的持久化事件流）。

P5（control 库化）：事件流迁入目录库 ctl_events 表（append-only INSERT）+
ctl_plan_meta 元数据表——原 <plan_id>.jsonl + _index.json 双结构、复合锁、
跨进程 RMW 全部退役（单事务保证"事件 + 元数据"原子）。

与 Plan 快照（ctl_plans 表，当前状态视图）互补：文牍是事件流——记录
"怎么走到这个状态"，不可变，重启不丢。三省协作的**共享记忆**：
  - 门下省封驳时从文牍取上下文（intent / 前置步骤结果 / 封驳历史）
  - 尚书省审查时从文牍知道哪个步骤产出了哪个 run
  - 前端轮询时每条总线消息可从文牍补充来龙去脉

消费经 control.core.storage 薄契约（llmsec.storage.ctlstore）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from control.core.storage import (
    append_event as _append_row,
)
from control.core.storage import (
    gazette_events as _rows,
)
from control.core.storage import (
    gazette_meta as _meta_row,
)
from control.core.storage import (
    list_gazette_meta as _list_rows,
)
from control.core.storage import (
    reset_gazette as _reset_rows,
)

# ============================================================
# 事件类型
# ============================================================
# Plan 级
EV_PLAN_DRAFTED = "plan_drafted"        # 尚书省拟案完成
EV_PLAN_APPROVED = "plan_approved"      # 用户准奏
EV_PLAN_REJECTED = "plan_rejected"      # 用户驳回
EV_PLAN_STARTED = "plan_started"        # 尚书省开始执行
EV_PLAN_FINISHED = "plan_finished"      # 执行完毕
EV_COMMISSION = "commission"            # 中书省下旨（记录用户原话 + session）

# Step 级
EV_STEP_STARTED = "step_started"        # 某步开始
EV_STEP_SUCCEEDED = "step_succeeded"    # 某步成功
EV_STEP_BLOCKED = "step_blocked"        # 某步被封驳
EV_STEP_UNBLOCKED = "step_unblocked"    # 用户准奏放行
EV_STEP_FAILED = "step_failed"          # 某步失败
EV_STEP_SKIPPED = "step_skipped"        # 因前置失败跳过

# 门下省
EV_REVIEW_FILED = "review_filed"        # 审查呈递


@dataclass
class GazetteEvent:
    """文牍事件（形状与库行一致——读路径直构，写路径经 to_dict）。"""
    ts: float               # 事件时间
    kind: str               # 事件类型（EV_* 常量）
    dept: str               # 发起部门（中书省/尚书省/门下省/用户）
    plan_id: str            # 关联 Plan
    step_id: str | None     # 关联步骤（plan 级事件为 None）
    session_id: str | None  # 关联 session（用户身份追溯）
    detail: dict            # 事件详情（各 kind 不同）

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# 公共 API
# ============================================================
def append_event(
    plan_id: str,
    kind: str,
    dept: str,
    *,
    detail: dict | None = None,
    step_id: str | None = None,
    session_id: str | None = None,
    intent: str | None = None,
) -> None:
    """向文牍追加一条事件（单事务：INSERT 事件 + upsert Plan 元数据）。"""
    _append_row(plan_id, kind, dept, detail=detail, step_id=step_id,
                session_id=session_id, intent=intent)


def read_events(plan_id: str) -> list[GazetteEvent]:
    """读某 Plan 的全部事件流（事件序升序）。"""
    return [GazetteEvent(**d) for d in _rows(plan_id)]


def read_plan_context(plan_id: str) -> dict | None:
    """从事件流重建 Plan 上下文快照。

    返回:
        {plan_id, intent, session_id, steps: {step_id: {status, capability, ...}},
         blocks: [...], reviews: [...], events_count} 或 None（无文牍）

    供门下省封驳/审查时取上下文——不再盲判。
    intent/session_id 优先从元数据表取（首个 append_event 时记入）。
    """
    events = read_events(plan_id)
    if not events:
        return None

    idx_entry = _meta_row(plan_id) or {}
    ctx = {
        "plan_id": plan_id,
        "intent": idx_entry.get("intent", ""),
        "session_id": idx_entry.get("session_id"),
        "steps": {},
        "blocks": [],
        "reviews": [],
        "events_count": len(events),
    }
    for ev in events:
        if ev.session_id and not ctx["session_id"]:
            ctx["session_id"] = ev.session_id
        if not ctx["intent"] and ev.kind == EV_COMMISSION and ev.detail.get("user_text"):
            ctx["intent"] = ev.detail["user_text"]

        sid = ev.step_id
        if sid:
            step_info = ctx["steps"].setdefault(sid, {
                "step_id": sid, "status": "pending", "capability": "",
                "description": "", "block_count": 0,
            })
            if ev.kind == EV_STEP_STARTED:
                step_info["status"] = "running"
                step_info["capability"] = ev.detail.get("capability", "")
                step_info["description"] = ev.detail.get("description", "")
            elif ev.kind == EV_STEP_SUCCEEDED:
                step_info["status"] = "done"
            elif ev.kind == EV_STEP_BLOCKED:
                step_info["status"] = "blocked"
                step_info["block_count"] += 1
                step_info["unblocked"] = False  # 新封驳令覆盖旧放行标记
                ctx["blocks"].append({
                    "step_id": sid, "ticket": ev.detail.get("ticket"),
                    "ts": ev.ts,
                })
            elif ev.kind == EV_STEP_UNBLOCKED:
                step_info["status"] = "pending"  # 放行后待重试
                # 显式放行标记：豁免判定不能用 status——重试时 executor 会先写
                # EV_STEP_STARTED 把 status 翻成 running，"pending" 判据必失效
                step_info["unblocked"] = True
            elif ev.kind == EV_STEP_FAILED:
                step_info["status"] = "failed"
                step_info["error"] = ev.detail.get("error", "")
            elif ev.kind == EV_STEP_SKIPPED:
                step_info["status"] = "skipped"

        if ev.kind == EV_REVIEW_FILED:
            ctx["reviews"].append({
                "step_id": sid, "run_name": ev.detail.get("run_name"),
                "digest": ev.detail.get("digest", "")[:500],
                "ts": ev.ts,
            })

    return ctx


def list_gazettes(*, session_id: str | None = None, recent: int = 20) -> list[dict]:
    """列出最近的文牍（元数据表），可按 session 过滤。"""
    return _list_rows(session_id=session_id, recent=recent)


def reset_gazettes() -> None:
    """清空文牍（测试用）。"""
    _reset_rows()
