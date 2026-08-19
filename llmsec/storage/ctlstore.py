"""storage.ctlstore — control 侧表 DAO（P5：control 层数据库化）。

表：ctl_events / ctl_plan_meta / ctl_plans / ctl_tickets / ctl_queue /
ctl_workspaces / ctl_env_snapshots（定义见 models.py）。

设计要点：
  - 全部落在目录库（catalog.db）——"登记"域单一库，引擎/事务基建复用；
  - append_event 单事务（INSERT 事件 + upsert 元数据），取代"jsonl 追加 +
    _index.json 跨进程锁 RMW"的复合操作；
  - control 只经 contract 消费（AST 守卫禁 control 直连 SQL/ORM）；
  - 旧文件（gazette/*.jsonl、_index.json×3、plans/*.json）经
    ``management.storage migrate-control`` 一次性导入。
"""

from __future__ import annotations

import time

from sqlmodel import select as _select

from llmsec.storage import db as _db
from llmsec.storage.models import (
    CtlEnvSnapshot,
    CtlEvent,
    CtlPlan,
    CtlPlanMeta,
    CtlQueueItem,
    CtlTicket,
    CtlWorkspace,
)

# 元数据 status 状态机：事件 kind → plan 终态
_PLAN_FINISHED_KINDS = {"plan_finished": "finished", "plan_rejected": "rejected"}


# ============================================================
# 文牍（gazette）
# ============================================================

def append_event(plan_id: str, kind: str, dept: str, *, detail: dict | None = None,
                 step_id: str | None = None, session_id: str | None = None,
                 intent: str | None = None) -> None:
    """追加一条文牍事件（单事务：INSERT 事件 + upsert Plan 元数据）。"""
    ts = time.time()
    with _db.tx() as s:
        s.add(CtlEvent(ts=ts, kind=kind, dept=dept, plan_id=plan_id,
                       step_id=step_id, session_id=session_id,
                       detail=dict(detail or {}) or None))
        meta = s.get(CtlPlanMeta, plan_id)
        if meta is None:
            meta = CtlPlanMeta(plan_id=plan_id, intent=intent or "",
                               session_id=session_id, created=ts, status="active")
        if intent and not meta.intent:
            meta.intent = intent
        if session_id and not meta.session_id:
            meta.session_id = session_id
        meta.last_event = kind
        meta.last_ts = ts
        if kind in _PLAN_FINISHED_KINDS:
            meta.status = _PLAN_FINISHED_KINDS[kind]
            meta.finished = ts
        s.add(meta)


def gazette_events(plan_id: str, *, limit: int | None = None) -> list[dict]:
    """某 Plan 的事件流（事件序升序；dict 形状与原 GazetteEvent.to_dict 一致）。"""
    with _db.session() as s:
        q = _select(CtlEvent).where(CtlEvent.plan_id == plan_id).order_by(CtlEvent.id)
        if limit:
            q = q.limit(limit)
        return [e.event_dict() for e in s.exec(q).all()]


def gazette_meta(plan_id: str) -> dict | None:
    with _db.session() as s:
        row = s.get(CtlPlanMeta, plan_id)
        return row.as_dict() if row is not None else None


def list_gazette_meta(*, session_id: str | None = None, recent: int = 20) -> list[dict]:
    with _db.session() as s:
        q = _select(CtlPlanMeta)
        if session_id:
            q = q.where(CtlPlanMeta.session_id == session_id)
        rows = s.exec(q.order_by(CtlPlanMeta.last_ts.desc())).all()
        return [r.as_dict() for r in rows[:recent]]


def search_gazette(keywords: list[str], *, limit: int = 500) -> list[dict]:
    """按关键词查事件（候选集 SQL LIKE，OR 命中；打分留给调用方——沿用原
    dialogue 的相关性排序语义）。

    返回 [{event_dict…, "intent": plan intent}] 候选（ts 升序）。
    """
    if not keywords:
        return []
    import json as _json

    pats = [k.lower() for k in keywords]
    with _db.session() as s:
        events = s.exec(_select(CtlEvent).order_by(CtlEvent.id.desc()).limit(limit)).all()
        metas = {m.plan_id: m for m in s.exec(_select(CtlPlanMeta)).all()}
        out = []
        for e in reversed(events):
            d = e.event_dict()
            d["intent"] = metas[e.plan_id].intent if e.plan_id in metas else ""
            searchable = _json.dumps(d, ensure_ascii=False, default=str).lower()
            if any(p in searchable for p in pats):
                out.append(d)
        return out


def reset_gazette() -> None:
    """清空文牍（测试用）。"""
    with _db.tx() as s:
        for row in s.exec(_select(CtlEvent)).all():
            s.delete(row)
        for row in s.exec(_select(CtlPlanMeta)).all():
            s.delete(row)


# ============================================================
# Plan 快照
# ============================================================

def save_ctl_plan(plan_dict: dict) -> None:
    """整 plan upsert（payload = 完整 to_dict；原子，取代裸 write_text）。"""
    now = time.time()
    with _db.tx() as s:
        pid = plan_dict["id"]
        row = s.get(CtlPlan, pid)
        if row is None:
            row = CtlPlan(id=pid, created=plan_dict.get("created", now))
        row.intent = plan_dict.get("intent", "")
        row.status = plan_dict.get("status", "")
        row.session_id = plan_dict.get("session_id")
        row.payload = dict(plan_dict)
        row.updated_at = now
        s.add(row)


def get_ctl_plan(plan_id: str) -> dict | None:
    with _db.session() as s:
        row = s.get(CtlPlan, plan_id)
        return row.as_dict() if row is not None else None


def list_ctl_plans(*, recent: int = 20) -> list[dict]:
    with _db.session() as s:
        rows = s.exec(_select(CtlPlan).order_by(CtlPlan.created.desc())).all()
        return [r.as_dict() for r in rows[:recent]]


def reset_ctl_plans() -> None:
    with _db.tx() as s:
        for row in s.exec(_select(CtlPlan)).all():
            s.delete(row)


# ============================================================
# 封驳令
# ============================================================

def save_ticket(ticket: dict) -> None:
    """ticket 形状 = BlockTicket.to_dict()。"""
    with _db.tx() as s:
        key = (ticket["plan_id"], ticket["step_id"])
        row = s.get(CtlTicket, key)
        if row is None:
            row = CtlTicket(plan_id=key[0], step_id=key[1])
        for k in ("token", "capability", "risk_level", "summary", "detail", "created"):
            if k in ticket:
                setattr(row, k, ticket[k])
        s.add(row)


def get_ticket(plan_id: str, step_id: str) -> dict | None:
    with _db.session() as s:
        row = s.get(CtlTicket, (plan_id, step_id))
        if row is None:
            return None
        return {
            "token": row.token, "plan_id": row.plan_id, "step_id": row.step_id,
            "capability": row.capability, "risk_level": row.risk_level,
            "summary": row.summary, "detail": row.detail, "created": row.created,
        }


def clear_ticket(plan_id: str, step_id: str) -> bool:
    with _db.tx() as s:
        row = s.get(CtlTicket, (plan_id, step_id))
        if row is None:
            return False
        s.delete(row)
        return True


def clear_tickets_for_plan(plan_id: str) -> int:
    with _db.tx() as s:
        rows = s.exec(_select(CtlTicket).where(CtlTicket.plan_id == plan_id)).all()
        for r in rows:
            s.delete(r)
        return len(rows)


def list_tickets_for_plan(plan_id: str) -> list[dict]:
    """某 Plan 现存全部封驳令（撤销前先取清单，供逐令广播 step_unblocked）。"""
    with _db.session() as s:
        rows = s.exec(
            _select(CtlTicket).where(CtlTicket.plan_id == plan_id)
        ).all()
        return [
            {
                "token": r.token, "plan_id": r.plan_id, "step_id": r.step_id,
                "capability": r.capability, "risk_level": r.risk_level,
                "summary": r.summary, "detail": r.detail, "created": r.created,
            }
            for r in rows
        ]


def reset_tickets() -> None:
    with _db.tx() as s:
        for row in s.exec(_select(CtlTicket)).all():
            s.delete(row)


# ============================================================
# Plan 队列（内容落库；worker 协议留内存）
# ============================================================

_DONE_KEEP = 20  # done 行只作近期审计，超出即清（防无限增长）


def enqueue_plan(plan_id: str) -> None:
    with _db.tx() as s:
        if any(r.plan_id == plan_id and r.status == "queued"
               for r in s.exec(_select(CtlQueueItem)).all()):
            return
        s.add(CtlQueueItem(plan_id=plan_id, queued_at=time.time(), status="queued"))


def mark_queue_running(plan_id: str) -> None:
    with _db.tx() as s:
        for r in s.exec(_select(CtlQueueItem).where(CtlQueueItem.plan_id == plan_id)).all():
            r.status = "running"
            s.add(r)


def finish_queue_item(plan_id: str) -> None:
    with _db.tx() as s:
        for r in s.exec(_select(CtlQueueItem).where(CtlQueueItem.plan_id == plan_id)).all():
            r.status = "done"
            s.add(r)
        done = list(s.exec(_select(CtlQueueItem).where(CtlQueueItem.status == "done")
                           .order_by(CtlQueueItem.queued_at.desc())).all())
        for r in done[_DONE_KEEP:]:
            s.delete(r)


def pending_queue_plans() -> list[str]:
    """重启恢复读侧：queued + 崩溃遗留的 running（重置回 queued，按入队序）。

    E-6：此前只回填 queued——worker 崩溃/进程被杀时 running 行永久搁浅，
    Plan 卡在 executing、其封驳令成孤儿。恢复为排队重跑：executor 按
    step.status 跳过已 done 步骤、复位僵死 running 步骤（execute_plan 入口），
    重跑代价可控。单进程部署假设下启动时回收 running 行是安全的（此刻本
    进程尚无 worker 在跑）。
    """
    with _db.tx() as s:
        rows = s.exec(_select(CtlQueueItem)
                      .where(CtlQueueItem.status.in_(["queued", "running"]))
                      .order_by(CtlQueueItem.queued_at)).all()
        ids = []
        for r in rows:
            if r.status == "running":
                r.status = "queued"
                s.add(r)
            ids.append(r.plan_id)
        return ids


def reset_queue() -> None:
    with _db.tx() as s:
        for row in s.exec(_select(CtlQueueItem)).all():
            s.delete(row)


# ============================================================
# workspace / env_snapshot 索引
# ============================================================

def save_workspace(info: dict) -> None:
    with _db.tx() as s:
        row = s.get(CtlWorkspace, info["name"])
        if row is None:
            row = CtlWorkspace(name=info["name"])
        for k in ("path", "source", "note", "created", "models", "records",
                  "merged", "merged_at", "merged_to"):
            if k in info:
                setattr(row, k, info[k])
        s.add(row)


def get_workspace(name: str) -> dict | None:
    with _db.session() as s:
        row = s.get(CtlWorkspace, name)
        return row.as_dict() if row is not None else None


def list_workspaces() -> list[dict]:
    with _db.session() as s:
        return [r.as_dict() for r in s.exec(
            _select(CtlWorkspace).order_by(CtlWorkspace.created.desc())).all()]


def delete_workspace_row(name: str) -> dict | None:
    with _db.tx() as s:
        row = s.get(CtlWorkspace, name)
        if row is None:
            return None
        d = row.as_dict()
        s.delete(row)
        return d


def save_env_snapshot(info: dict) -> None:
    with _db.tx() as s:
        row = s.get(CtlEnvSnapshot, info["name"])
        if row is None:
            row = CtlEnvSnapshot(name=info["name"])
        for k in ("path", "source", "note", "created", "keys", "merged_to_global"):
            if k in info:
                setattr(row, k, info[k])
        s.add(row)


def list_env_snapshots() -> list[dict]:
    with _db.session() as s:
        return [r.as_dict() for r in s.exec(
            _select(CtlEnvSnapshot).order_by(CtlEnvSnapshot.created.desc())).all()]


def get_env_snapshot(name: str) -> dict | None:
    with _db.session() as s:
        row = s.get(CtlEnvSnapshot, name)
        return row.as_dict() if row is not None else None


def delete_env_snapshot(name: str) -> bool:
    with _db.tx() as s:
        row = s.get(CtlEnvSnapshot, name)
        if row is None:
            return False
        s.delete(row)
        return True
