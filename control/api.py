"""control.api — 控制层（control/）的 FastAPI router，把三省能力暴露给看板 UI。

薄封装 control.agent.* / control.core.* 为端点。本模块属于 control 包，
守隔离边界（不 import llmsec 内部，日志用标准库 logging）；由 llmsec 的
dashboard_api 作为**组合根**唯一一处 import 挂载——llmsec→control 的依赖
收口到这一行，不再是散落在 routers/ 里的隐性倒置。

端点：
  GET    /api/control/workspaces          列出 fork 工作区
  POST   /api/control/fork                fork 新工作区
  DELETE /api/control/workspaces/{name}   删除工作区
  POST   /api/control/compare             对比 run
  POST   /api/control/merge               合并 R 矩阵
  POST   /api/control/chat                中书省对话（复杂指令→尚书省拟案）
  POST   /api/control/chat/reset          清空 session
  GET    /api/control/llm-status          LLM 是否已配置
  GET    /api/control/tools               列出中书省工具 schema
  GET    /api/control/capabilities        列出尚书省能力清单
  --- Plan 管理 ---
  POST   /api/control/plan/approve        用户准奏 Plan → 触发尚书省执行
  POST   /api/control/plan/reject         用户驳回 Plan
  POST   /api/control/plan/block/approve  用户准奏某步封驳（放行该步）
  GET    /api/control/plan/{id}/status    查 Plan 执行状态
  --- 总线 feed（三省面板轮询）---
  GET    /api/control/bus/feed            总线消息流

（审查清理：fork-and-run / review / plan/queue / plans / blocks /
env-snapshots×3 共 8 个端点经全仓 grep 确认无任何调用方，已删除。）
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from control.agent import menxia
from control.agent.bus import get_bus
from control.agent.llm import is_llm_configured
from control.agent.shangshu import capabilities as caps_mod
from control.agent.zhongshu import handle_message as zhongshu_handle
from control.agent.zhongshu import session as sess
from control.agent.zhongshu import tools as zs_tools
from control.core import compare as compare_mod
from control.core import workspace as ws_mod

router = APIRouter()

logger = logging.getLogger("control.api")

# 门下省在 router 加载时初始化（订阅总线）
menxia.init_menxia()


# ============================================================
# 请求模型
# ============================================================
class ForkRequest(BaseModel):
    name: str
    source: str = "global"
    note: str = ""


class CompareRequest(BaseModel):
    runs: list[str]


class MergeRequest(BaseModel):
    sources: list[str]
    target: str
    models: list[str] | None = None
    confirm: bool = False


class ChatRequest(BaseModel):
    text: str
    session_id: str | None = None


class ResetRequest(BaseModel):
    session_id: str | None = None


class PlanApproveRequest(BaseModel):
    plan_id: str
    session_id: str | None = None


class PlanRejectRequest(BaseModel):
    plan_id: str


class BlockApproveRequest(BaseModel):
    plan_id: str
    step_id: str


# ============================================================
# 工作区管理
# ============================================================
@router.get("/api/control/workspaces")
def api_list_workspaces():
    try:
        return {"workspaces": ws_mod.list_workspaces()}
    except Exception:
        # 不把内部异常文本回传客户端（可能含路径等敏感信息），与 api_chat 同策略
        logger.exception("处理失败")
        raise HTTPException(status_code=500, detail="处理失败，请查看服务端日志")


@router.post("/api/control/fork")
def api_fork(req: ForkRequest):
    try:
        return ws_mod.fork(req.name, source=req.source, note=req.note)
    except (FileExistsError, FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        # 不把内部异常文本回传客户端（可能含路径等敏感信息），与 api_chat 同策略
        logger.exception("处理失败")
        raise HTTPException(status_code=500, detail="处理失败，请查看服务端日志")


@router.delete("/api/control/workspaces/{name}")
def api_delete_workspace(name: str):
    try:
        return ws_mod.delete_workspace(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"工作区不存在: {name}")
    except ValueError as e:
        # 名称非法（路径穿越）
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        # 不把内部异常文本回传客户端（可能含路径等敏感信息），与 api_chat 同策略
        logger.exception("处理失败")
        raise HTTPException(status_code=500, detail="处理失败，请查看服务端日志")


# ============================================================
# 对比 / 合并
# ============================================================
@router.post("/api/control/compare")
def api_compare(req: CompareRequest):
    if len(req.runs) < 2:
        raise HTTPException(status_code=400, detail="至少需要 2 个 run")
    try:
        return compare_mod.compare(req.runs)
    except Exception:
        # 不把内部异常文本回传客户端（可能含路径等敏感信息），与 api_chat 同策略
        logger.exception("处理失败")
        raise HTTPException(status_code=500, detail="处理失败，请查看服务端日志")


@router.post("/api/control/merge")
def api_merge(req: MergeRequest):
    """合并 R 矩阵。经 control merge capability（confirm=True 时执行 + 回写 merged 状态）。"""
    try:
        result = caps_mod.call("merge_results", {
            "sources": req.sources, "target": req.target,
            "models": req.models, "confirm": req.confirm,
        })
        return result
    except Exception:
        # 不把内部异常文本回传客户端（可能含路径等敏感信息），与 api_chat 同策略
        logger.exception("处理失败")
        raise HTTPException(status_code=500, detail="处理失败，请查看服务端日志")


# ============================================================
# 中书省对话（复杂指令→尚书省拟案）
# ============================================================
@router.get("/api/control/llm-status")
def api_llm_status():
    return {"configured": is_llm_configured()}


@router.get("/api/control/tools")
def api_tools():
    """中书省保留的简单工具 schema。"""
    return {"tools": [t.to_schema() for t in zs_tools.all_tools()]}


@router.get("/api/control/capabilities")
def api_capabilities():
    """尚书省完整能力清单。"""
    return {"capabilities": [
        {"name": c.name, "description": c.description,
         "risk_level": c.risk_level, "parameters": c.parameters}
        for c in caps_mod.all_capabilities()
    ]}


@router.post("/api/control/chat")
def api_chat(req: ChatRequest):
    """中书省对话（简单自处理，复杂转交尚书省拟案）。"""
    try:
        result = zhongshu_handle(req.text, session_id=req.session_id)
        return result
    except Exception:
        # 不把内部异常文本回传客户端（可能含文件路径等敏感信息），仅记日志
        logger.exception("api_chat 处理失败")
        raise HTTPException(status_code=500, detail="处理失败，请重试或查看服务端日志")


@router.post("/api/control/chat/reset")
def api_chat_reset(req: ResetRequest):
    """清空 session 历史（重新开始对话）。"""
    if req.session_id:
        sess.reset(req.session_id)
    return {"session_id": req.session_id, "reset": True}


# ============================================================
# Plan 管理（尚书省执行）
# ============================================================
@router.post("/api/control/plan/approve")
def api_plan_approve(req: PlanApproveRequest):
    """用户准奏 Plan → 提交执行队列（异步，不阻塞）。"""
    from control.agent.shangshu import approve_plan, get_queue, load_plan
    try:
        plan = load_plan(req.plan_id)
        if plan is None:
            raise HTTPException(status_code=404, detail=f"Plan 不存在: {req.plan_id}")
        if plan.status == "drafted":
            approve_plan(req.plan_id)
        # 提交到队列，不阻塞——worker 线程串行执行
        queue_status = get_queue().submit(req.plan_id)
        return {"plan_id": req.plan_id, "queue_status": queue_status}
    except HTTPException:
        raise
    except Exception:
        # 不把内部异常文本回传客户端（可能含路径等敏感信息），与 api_chat 同策略
        logger.exception("处理失败")
        raise HTTPException(status_code=500, detail="处理失败，请查看服务端日志")


@router.post("/api/control/plan/reject")
def api_plan_reject(req: PlanRejectRequest):
    """用户驳回 Plan。"""
    from control.agent.shangshu import reject_plan
    try:
        plan = reject_plan(req.plan_id)
        # 清除该 plan 的所有封驳
        menxia.clear_all_for_plan(req.plan_id)
        return plan.to_dict()
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Plan 不存在: {req.plan_id}")
    except Exception:
        # 不把内部异常文本回传客户端（可能含路径等敏感信息），与 api_chat 同策略
        logger.exception("处理失败")
        raise HTTPException(status_code=500, detail="处理失败，请查看服务端日志")


@router.post("/api/control/plan/block/approve")
def api_block_approve(req: BlockApproveRequest):
    """用户准奏某步封驳（放行该步）。写文牍 step_unblocked。

    如果 Plan 已执行完（status=done）且有 blocked 步骤被放行，
    自动重新提交到执行队列——让 executor 重入时重试放行的步骤。
    """
    from control.agent import gazette
    from control.agent.shangshu import get_queue, load_plan
    ok = menxia.clear_block(req.plan_id, req.step_id)
    if not ok:
        raise HTTPException(status_code=404, detail="封驳令不存在（可能已放行或已过期）")
    gazette.append_event(req.plan_id, gazette.EV_STEP_UNBLOCKED, "用户",
                         step_id=req.step_id,
                         detail={"capability": "放行重试"})
    # 检查是否需要重新入队
    plan = load_plan(req.plan_id)
    requeued = False
    if plan and plan.status == "done":
        # 放行的步骤 + 因它被跳过的后续步骤都要重置
        unblocked_sids = set()
        for s in plan.steps:
            if s.id == req.step_id:
                s.status = "pending"
                s.ticket = None
                unblocked_sids.add(s.id)
        # 因依赖被放行步骤而 skipped 的也要重置
        changed = True
        while changed:
            changed = False
            for s in plan.steps:
                if s.status == "skipped" and any(d in unblocked_sids for d in s.depends_on):
                    s.status = "pending"
                    unblocked_sids.add(s.id)
                    changed = True
        # 其他 blocked 步骤（未被放行的）也重置为 pending（它们的 ticket 可能已过时）
        for s in plan.steps:
            if s.status == "blocked":
                s.status = "pending"
                s.ticket = None
                unblocked_sids.add(s.id)
        has_pending = any(s.status == "pending" for s in plan.steps)
        if has_pending:
            plan.status = "approved"
            from control.agent.shangshu import save_plan
            save_plan(plan)
            get_queue().submit(req.plan_id)
            requeued = True
    return {"plan_id": req.plan_id, "step_id": req.step_id, "approved": True,
            "requeued": requeued}


@router.get("/api/control/plan/{plan_id}/status")
def api_plan_status(plan_id: str):
    """查 Plan 执行状态。"""
    from control.agent.shangshu import load_plan
    plan = load_plan(plan_id)
    if plan is None:
        raise HTTPException(status_code=404, detail=f"Plan 不存在: {plan_id}")
    return plan.to_dict()


# ============================================================
# 总线 feed（三省面板轮询）
# ============================================================
@router.get("/api/control/bus/feed")
def api_bus_feed(since: float = 0.0, dept: str | None = None):
    """总线消息流（供前端面板轮询补全）。"""
    bus = get_bus()
    msgs = bus.recent(since_ts=since, dept=dept)  # kinds 缺省=所有 kind（前端按需过滤）
    return {"messages": [m.to_dict() for m in msgs], "latest_ts": msgs[-1].ts if msgs else since}
