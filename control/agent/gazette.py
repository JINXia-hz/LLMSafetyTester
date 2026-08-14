"""control.agent.gazette — 文牍存储（三省任务的持久化事件流）。

每个 Plan 的完整生命周期记录在一个 append-only JSONL 文件里。
与 Plan JSON 快照（output/plans/<id>.json，当前状态视图）互补：
文牍是事件流——记录"怎么走到这个状态"，不可变，重启不丢。

存储：
  output/gazette/
  ├── <plan_id>.jsonl     # 每个 Plan 的事件流（一行一个事件）
  └── _index.json         # 所有 Plan 的元信息索引

事件流是三省协作的**共享记忆**：
  - 门下省封驳时从文牍取上下文（intent / 前置步骤结果 / 封驳历史）
  - 尚书省审查时从文牍知道哪个步骤产出了哪个 run
  - 前端轮询时每条总线消息可从文牍补充来龙去脉
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from control.config import OUTPUT_DIR
from control.core.paths import safe_component
from control.core.store import AtomicIndexStore

_GAZETTE_DIR = OUTPUT_DIR / "gazette"
# append JSONL 事件 + 更新 _index.json 是复合操作，需在同一把锁内
_LOCK = threading.Lock()

# 文牍索引存储（原子读写 + Windows PermissionError 重试）
# base_dir 传 lambda：测试期 monkeypatch _GAZETTE_DIR 后能动态生效
_store = AtomicIndexStore(lambda: _GAZETTE_DIR, "plans")


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
    """文牍事件。"""
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
# 存储
# ============================================================
def _gazette_path(plan_id: str) -> Path:
    _store.ensure_dir()
    # plan_id 外部可控，走校验防穿越（`../x` 逃出 gazette 目录）
    return safe_component(_GAZETTE_DIR, f"{plan_id}.jsonl")


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
    """向文牍 append 一条事件（原子追加，带锁）。同时更新 _index.json。

    Args:
        plan_id: 关联 Plan id
        kind: 事件类型（EV_* 常量）
        dept: 发起部门
        detail: 事件详情
        step_id: 关联步骤（plan 级事件为 None）
        session_id: 关联 session
        intent: Plan 意图（首次记录时写入 _index）
    """
    event = GazetteEvent(
        ts=time.time(), kind=kind, dept=dept, plan_id=plan_id,
        step_id=step_id, session_id=session_id, detail=detail or {},
    )
    line = json.dumps(event.to_dict(), ensure_ascii=False) + "\n"
    with _LOCK:
        # append 事件
        with open(_gazette_path(plan_id), "a", encoding="utf-8") as f:
            f.write(line)
        # 更新索引
        idx = _store.load()
        entry = idx["plans"].setdefault(plan_id, {
            "plan_id": plan_id,
            "intent": intent or "",
            "session_id": session_id,
            "created": event.ts,
            "status": "active",
            "last_event": kind,
            "last_ts": event.ts,
        })
        # 首次记录时补 intent/session
        if intent and not entry.get("intent"):
            entry["intent"] = intent
        if session_id and not entry.get("session_id"):
            entry["session_id"] = session_id
        entry["last_event"] = kind
        entry["last_ts"] = event.ts
        # 根据 kind 更新 status
        if kind == EV_PLAN_FINISHED:
            entry["status"] = "finished"
            entry["finished"] = event.ts
        elif kind == EV_PLAN_REJECTED:
            entry["status"] = "rejected"
            entry["finished"] = event.ts
        _store.save(idx)


def read_events(plan_id: str) -> list[GazetteEvent]:
    """读某 Plan 的全部事件流（按 ts 排序）。"""
    p = _gazette_path(plan_id)
    if not p.exists():
        return []
    events = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            events.append(GazetteEvent(
                ts=d["ts"], kind=d["kind"], dept=d["dept"],
                plan_id=d["plan_id"], step_id=d.get("step_id"),
                session_id=d.get("session_id"), detail=d.get("detail", {}),
            ))
        except (json.JSONDecodeError, KeyError):
            continue
    events.sort(key=lambda e: e.ts)
    return events


def read_plan_context(plan_id: str) -> dict | None:
    """从事件流重建 Plan 上下文快照。

    返回:
        {plan_id, intent, session_id, steps: {step_id: {status, capability, ...}},
         blocks: [...], reviews: [...], events_count} 或 None（无文牍）

    供门下省封驳/审查时取上下文——不再盲判。
    intent/session_id 优先从 _index.json 取（首个 append_event 时记入）。
    """
    events = read_events(plan_id)
    if not events:
        return None

    # 先从索引取 intent/session（比从事件流里拼更可靠）
    with _LOCK:
        idx = _store.load()
    idx_entry = idx.get("plans", {}).get(plan_id, {})

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
                ctx["blocks"].append({
                    "step_id": sid, "ticket": ev.detail.get("ticket"),
                    "ts": ev.ts,
                })
            elif ev.kind == EV_STEP_UNBLOCKED:
                step_info["status"] = "pending"  # 放行后待重试
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
    """列出最近的文牍（_index.json），可按 session 过滤。过滤掉 __pending__。"""
    idx = _store.load()
    plans = [p for p in idx.get("plans", {}).values() if p.get("plan_id") != "__pending__"]
    if session_id:
        plans = [p for p in plans if p.get("session_id") == session_id]
    plans.sort(key=lambda p: p.get("last_ts", 0), reverse=True)
    return plans[:recent]


def reset_gazettes() -> None:
    """清空文牍（测试用）。"""
    with _LOCK:
        import shutil
        try:
            if _GAZETTE_DIR.exists():
                shutil.rmtree(_GAZETTE_DIR)
        except OSError:
            pass  # Windows 并发文件锁，忽略
