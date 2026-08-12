"""control router — 把控制层（control/）能力暴露给看板 UI。

薄封装 control.agent.tools / control.core.* 为 FastAPI 端点。
control 内部仍守隔离边界（不 import llmsec 内部），本 router 只是 control 的一个 caller。

端点：
  GET    /api/control/workspaces          列出 fork 工作区
  POST   /api/control/fork                fork 新工作区
  POST   /api/control/fork-and-run        fork 并异步起 runner（复用 tasks 机制）
  DELETE /api/control/workspaces/{name}   删除工作区
  POST   /api/control/compare             对比 run
  POST   /api/control/merge               合并 R 矩阵
  POST   /api/control/chat                LLM 对话（tool-calling）
  GET    /api/control/llm-status          LLM 是否已配置
  GET    /api/control/tools               列出可用工具 schema
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from control.agent.chat import chat_once_robust
from control.agent.llm import is_llm_configured
from control.agent.tools import all_tools
from control.config import WORKSPACES_DIR
from control.core import compare as compare_mod
from control.core import workspace as ws_mod

router = APIRouter()


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
    max_tool_rounds: int = 5


# ============================================================
# 工作区管理
# ============================================================
@router.get("/api/control/workspaces")
async def api_list_workspaces():
    try:
        return {"workspaces": ws_mod.list_workspaces()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/control/fork")
async def api_fork(req: ForkRequest):
    try:
        return ws_mod.fork(req.name, source=req.source, note=req.note)
    except (FileExistsError, FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/control/fork-and-run")
async def api_fork_and_run(req: ForkRunRequest):
    """fork 后异步起 runner（复用 tasks 子系统的任务跟踪 + SSE）。"""
    try:
        info = ws_mod.fork(req.name, source=req.source, note=req.note)
    except (FileExistsError, FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    # 异步起 runner（复用 tasks._start_task，看板自动获任务卡片 + 进度 + SSE）
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
async def api_delete_workspace(name: str):
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
async def api_compare(req: CompareRequest):
    if len(req.runs) < 2:
        raise HTTPException(status_code=400, detail="至少需要 2 个 run")
    try:
        return compare_mod.compare(req.runs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/control/merge")
async def api_merge(req: MergeRequest):
    """合并 R 矩阵。经 control merge tool（confirm=True 时执行 + 回写 merged 状态）。"""
    from control.agent.tools import call_tool, reset_registry
    reset_registry()
    try:
        result = call_tool("merge", {
            "sources": req.sources, "target": req.target,
            "models": req.models, "confirm": req.confirm,
        })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# LLM 对话
# ============================================================
@router.get("/api/control/llm-status")
async def api_llm_status():
    return {"configured": is_llm_configured()}


@router.get("/api/control/tools")
async def api_tools():
    return {"tools": [t.to_schema() for t in all_tools()]}


@router.post("/api/control/chat")
async def api_chat(req: ChatRequest):
    """LLM 对话（tool-calling ReAct 循环）。LLM 未配置时兜底规则版。"""
    try:
        result = chat_once_robust(req.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
