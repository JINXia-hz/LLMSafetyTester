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

from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from control.api import router as control_router
from llmsec.core import config as _config
from llmsec.core.config import RUNS_DIR
from llmsec.core.logging import get_logger
from llmsec.server import task_manager
from llmsec.server.routers import cluster_viz, data_query, hpo, tasks

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


class _NoCacheStaticFiles(StaticFiles):
    """静态资源一律 Cache-Control: no-cache（每次协商重校验，ETag/Last-Modified
    仍让 304 足够便宜）——浏览器启发式缓存曾导致 JS 更新后必须 ctrl+F5 才生效。"""

    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


if STATIC_DIR.exists():
    app.mount("/static", _NoCacheStaticFiles(directory=str(STATIC_DIR)), name="static")


# ============================================================
# 页面
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


# ============================================================
# 健康检查（容器编排探针用，轻量无副作用）
# ============================================================
@app.get("/health")
@app.get("/healthz")
async def health():
    """liveness/readiness 探针：返回进程存活 + 任务队列概要。

    供 docker-compose healthcheck / K8s 探针 / 外部监控调用。
    /health 与 /healthz 双路径（后者是 K8s 惯例）。
    """
    try:
        all_tasks = task_manager.list_tasks()
        running = sum(1 for t in all_tasks if t.get("status") == "running")
        queued = sum(1 for t in all_tasks if t.get("status") == "queued")
    except Exception:
        running, queued = 0, 0
    return JSONResponse({
        "status": "ok",
        "ts": datetime.now().isoformat(),
        "tasks_running": running,
        "tasks_queued": queued,
    })


@app.get("/ready")
async def ready():
    """readiness 探针：检查 R 矩阵（唯一真相）可读。

    R 矩阵不存在时返回 503（首次部署/数据未初始化时看板尚未就绪）。
    """
    if _config.RESULTS_DB.exists():
        try:
            from llmsec.storage import rstore
            rstore.load_matrix()  # quick_check 一并探过
            return JSONResponse({"status": "ready", "results_db": str(_config.RESULTS_DB)})
        except (OSError, RuntimeError):
            pass
    return JSONResponse({"status": "not_ready", "results_db": str(_config.RESULTS_DB)}, status_code=503)


# ============================================================
# 路由注册
# ============================================================
app.include_router(data_query.router)
app.include_router(cluster_viz.router)
app.include_router(tasks.router)
app.include_router(hpo.router)
app.include_router(control_router)


# ============================================================
# 模块级 RUNS_DIR 作为 data_query._runs_dir() 的 monkeypatch 入口保留
# ============================================================
__all__ = [
    "RUNS_DIR",
    "app",
]
