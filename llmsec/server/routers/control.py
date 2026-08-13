"""control router — 把控制层（control/）能力暴露给看板 UI。

薄封装 control.agent.* / control.core.* 为 FastAPI 端点。
control 内部仍守隔离边界（不 import llmsec 内部），本 router 只是 control 的一个 caller。

端点：
  GET    /api/control/workspaces          列出 fork 工作区
  POST   /api/control/fork                fork 新工作区
  POST   /api/control/fork-and-run        fork 并异步起 runner（复用 tasks 机制）
  DELETE /api/control/workspaces/{name}   删除工作区
  POST   /api/control/compare             对比 run
  POST   /api/control/merge               合并 R 矩阵
  POST   /api/control/chat                中书省对话（复杂指令→尚书省拟案）
  POST   /api/control/chat/reset          清空 session
  POST   /api/control/review              门下省审查某 run
  GET    /api/control/llm-status          LLM 是否已配置
  GET    /api/control/tools               列出中书省工具 schema
  GET    /api/control/capabilities        列出尚书省能力清单
  --- Plan 管理 ---
  POST   /api/control/plan/approve        用户准奏 Plan → 触发尚书省执行
  POST   /api/control/plan/reject         用户驳回 Plan
  POST   /api/control/plan/block/approve  用户准奏某步封驳（放行该步）
  GET    /api/control/plan/{id}/status    查 Plan 执行状态
  GET    /api/control/plans               列出最近 Plan
  --- 总线 feed（三省面板轮询）---
  GET    /api/control/bus/feed            总线消息流
  GET    /api/control/blocks              当前待确认封驳列表
  --- .env 快照 ---
  GET    /api/control/env-snapshots       列出 .env 快照
  POST   /api/control/env-snapshots       创建快照
  DELETE /api/control/env-snapshots/{name} 删除快照
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from control.agent import menxia
from control.agent.bus import get_bus
from control.agent.llm import is_llm_configured
from control.agent.shangshu import capabilities as caps_mod
from control.agent.zhongshu import handle_message as zhongshu_handle
from control.agent.zhongshu import session as sess
from control.agent.zhongshu import tools as zs_tools
from control.config import WORKSPACES_DIR
from control.core import compare as compare_mod
from control.core import workspace as ws_mod

router = APIRouter()

# 门下省在 router 加载时初始化（订阅总线）
menxia.init_menxia()


# ============================================================
# 请求模型
# ============================================================
class ForkRequest(BaseModel):
    name: str
    source: str = "global"
    note: str = ""


class ForkRunRequest(BaseModel):
    name: str
    source: str = "global"
    note: str = ""
    target: str | None = None
    input_file: str = "attacks/l1.jsonl"
    max_rounds: int = 5
    seed: int | None = None


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


class ReviewRequest(BaseModel):
    run: str
    use_llm: bool = True


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


class EnvSnapshotCreateRequest(BaseModel):
    name: str
    source: str = "global"
    note: str = ""


# ============================================================
# 工作区管理
# ============================================================
@router.get("/api/control/workspaces")
def api_list_workspaces():
    try:
        return {"workspaces": ws_mod.list_workspaces()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/control/fork")
def api_fork(req: ForkRequest):
    try:
        return ws_mod.fork(req.name, source=req.source, note=req.note)
    except (FileExistsError, FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/control/fork-and-run")
def api_fork_and_run(req: ForkRunRequest):
    """fork 后异步起 runner（复用 tasks 子系统的任务跟踪 + SSE）。"""
    try:
        info = ws_mod.fork(req.name, source=req.source, note=req.note)
    except (FileExistsError, FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    from llmsec.server.routers.tasks import _start_task
    ws_dir = WORKSPACES_DIR / req.name
    argv = ["-m", "llmsec.pipeline.runner", "--work-dir", str(ws_dir),
            "--input", req.input_file, "--max-rounds", str(req.max_rounds),
            "--phase", "all", "--no-early-stop"]
    if req.target:
        argv += ["--target", req.target]
    if req.seed is not None:
        argv += ["--seed", str(req.seed)]
    task = _start_task("control-run", argv)
    return {"workspace": info, "task": task}


@router.delete("/api/control/workspaces/{name}")
def api_delete_workspace(name: str):
    try:
        return ws_mod.delete_workspace(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"工作区不存在: {name}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 对比 / 合并
# ============================================================
@router.post("/api/control/compare")
def api_compare(req: CompareRequest):
    if len(req.runs) < 2:
        raise HTTPException(status_code=400, detail="至少需要 2 个 run")
    try:
        return compare_mod.compare(req.runs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/control/merge")
def api_merge(req: MergeRequest):
    """合并 R 矩阵。经 control merge capability（confirm=True 时执行 + 回写 merged 状态）。"""
    try:
        result = caps_mod.call("merge_results", {
            "sources": req.sources, "target": req.target,
            "models": req.models, "confirm": req.confirm,
        })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/control/chat/reset")
def api_chat_reset(req: ResetRequest):
    """清空 session 历史（重新开始对话）。"""
    if req.session_id:
        sess.reset(req.session_id)
    return {"session_id": req.session_id, "reset": True}


@router.post("/api/control/review")
def api_review(req: ReviewRequest):
    """门下省审查：读 run 报告，识别异常，呈递摘要。"""
    from control.agent.menxia import review_run
    try:
        result = review_run(req.run, use_llm=req.use_llm)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Plan 管理（尚书省执行）
# ============================================================
@router.post("/api/control/plan/approve")
def api_plan_approve(req: PlanApproveRequest):
    """用户准奏 Plan → 尚书省执行（同步阻塞，长任务前端轮询 bus feed 看进度）。"""
    from control.agent.shangshu import approve_plan, execute_plan, load_plan
    try:
        plan = load_plan(req.plan_id)
        if plan is None:
            raise HTTPException(status_code=404, detail=f"Plan 不存在: {req.plan_id}")
        if plan.status == "drafted":
            approve_plan(req.plan_id)
        # 执行（同步）—— 短任务直接返回结果；长任务前端会先看到 bus feed 进度
        plan = execute_plan(req.plan_id)
        return plan.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/control/plan/block/approve")
def api_block_approve(req: BlockApproveRequest):
    """用户准奏某步封驳（放行该步，下次执行时重试）。写文牍 step_unblocked。"""
    from control.agent import gazette
    ok = menxia.approve_block(req.plan_id, req.step_id)
    if not ok:
        raise HTTPException(status_code=404, detail="封驳令不存在（可能已放行或已过期）")
    gazette.append_event(req.plan_id, gazette.EV_STEP_UNBLOCKED, "用户",
                         step_id=req.step_id,
                         detail={"capability": "放行重试"})
    return {"plan_id": req.plan_id, "step_id": req.step_id, "approved": True}


@router.get("/api/control/plan/{plan_id}/status")
def api_plan_status(plan_id: str):
    """查 Plan 执行状态。"""
    from control.agent.shangshu import load_plan
    plan = load_plan(plan_id)
    if plan is None:
        raise HTTPException(status_code=404, detail=f"Plan 不存在: {plan_id}")
    return plan.to_dict()


@router.get("/api/control/plans")
def api_plans():
    """列出最近 Plan。"""
    from control.agent.shangshu import list_plans
    return {"plans": list_plans()}


# ============================================================
# 总线 feed（三省面板轮询）
# ============================================================
@router.get("/api/control/bus/feed")
def api_bus_feed(since: float = 0.0, dept: str | None = None):
    """总线消息流（供前端面板轮询补全）。"""
    bus = get_bus()
    kinds = None  # 返回所有 kind（前端按需过滤）
    msgs = bus.recent(since_ts=since, dept=dept, kinds=kinds)
    return {"messages": [m.to_dict() for m in msgs], "latest_ts": msgs[-1].ts if msgs else since}


@router.get("/api/control/blocks")
def api_blocks():
    """当前待确认封驳列表。"""
    return {"blocks": menxia.list_pending_blocks()}


# ============================================================
# .env 快照
# ============================================================
@router.get("/api/control/env-snapshots")
def api_env_snapshots_list():
    from control.core import env_snapshot
    return {"snapshots": env_snapshot.list_snapshots()}


@router.post("/api/control/env-snapshots")
def api_env_snapshots_create(req: EnvSnapshotCreateRequest):
    from control.core import env_snapshot
    try:
        return env_snapshot.create(req.name, source=req.source, note=req.note)
    except (FileExistsError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/control/env-snapshots/{name}")
def api_env_snapshots_delete(name: str):
    from control.core import env_snapshot
    try:
        return env_snapshot.delete(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"快照不存在: {name}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
