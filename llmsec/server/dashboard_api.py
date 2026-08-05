#!/usr/bin/env python3
"""
LLMSEC 安全评估 Web 面板（FastAPI + 原生 HTML/JS）

功能：
- 只读数据 API：总览（雷达图）、威胁看板、ELO 排名与收敛曲线、
  Markdown 报告、聚类分析、SVD-Ridge 预测模型诊断
- 操作 API：图形化触发生成攻击集 / 自适应评估 / 聚类分析（子进程任务 + 状态轮询）

启动（在仓库根目录下执行）:
    .venv/Scripts/uvicorn llmsec.server.dashboard_api:app --host 127.0.0.1 --port 8080

访问:
    http://localhost:8080
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from llmsec.core.config import RUNS_DIR
from llmsec.core.logging import get_logger
from llmsec.server.routers import cluster_viz, data_query, tasks
from llmsec.server.routers.cluster_viz import _CACHE_MAX_SIZE, _cache_put
from llmsec.server.routers.data_query import _convergence_score
from llmsec.server.routers.tasks import (
    TASKS,
    EvaluateRequest,
    _refresh_task_status,
    _start_task,
)

logger = get_logger(__name__)

# ============================================================
# 路径
# ============================================================
SERVER_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = SERVER_DIR / "templates"
STATIC_DIR = SERVER_DIR / "static"
# RUNS_DIR 由 core.config 统一定义（见 import）；作为模块属性保留，
# 供测试 monkeypatch 重定向（路由经 data_query._runs_dir() 现取此值）。

app = FastAPI(title="LLMSEC Dashboard")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ============================================================
# 页面
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


# ============================================================
# 路由注册
# ============================================================
app.include_router(data_query.router)
app.include_router(cluster_viz.router)
app.include_router(tasks.router)


# ============================================================
# 向后兼容 / 测试访问的再导出
# ============================================================
# 路由拆分后，部分历史符号被测试与外部以 `dashboard_api.X` 形式引用；
# 这里集中再导出，保持 `uvicorn llmsec.server.dashboard_api:app` 入口与 API 不变。
__all__ = [
    "RUNS_DIR",
    "TASKS",
    "EvaluateRequest",
    "_CACHE_MAX_SIZE",
    "_cache_put",
    "_convergence_score",
    "_refresh_task_status",
    "_start_task",
    "app",
    "logger",
]
