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
from llmsec.server.routers import cluster_viz, control, data_query, hpo, tasks

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
# 路由注册
# ============================================================
app.include_router(data_query.router)
app.include_router(cluster_viz.router)
app.include_router(tasks.router)
app.include_router(hpo.router)
app.include_router(control.router)


# ============================================================
# 模块级 RUNS_DIR 作为 data_query._runs_dir() 的 monkeypatch 入口保留
# ============================================================
__all__ = [
    "RUNS_DIR",
    "app",
]
