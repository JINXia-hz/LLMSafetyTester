"""control.config — 定位 llmsec 工作单元（解释器 / 仓库根 / output）。

控制层需要知道三件事来调用 llmsec：
  1. 用哪个 python（.venv/Scripts/python.exe，runner 有 .venv 自重启逻辑）
  2. llmsec 仓库根在哪（subprocess 的 cwd）
  3. output/ 在哪（读 run 产物做对比、定位 work-dir）

默认从本文件位置推导（control/ 在仓库根下，与 llmsec/ 并列），可被环境变量覆盖。
"""

from __future__ import annotations

import os
from pathlib import Path

# control/ 的父目录 = 仓库根（与 llmsec/ 并列）
REPO_ROOT = Path(__file__).resolve().parent.parent

# llmsec 仓库根（默认=本仓库；LLMSEC_REPO_ROOT 可指向别处安装的 llmsec）
LLMSEC_REPO = Path(os.environ.get("LLMSEC_REPO_ROOT", REPO_ROOT))

# output 目录（llmsec 的产物根；work-dir 也落在其下或别处）
OUTPUT_DIR = LLMSEC_REPO / "output"
RUNS_DIR = OUTPUT_DIR / "runs"
WORKSPACES_DIR = OUTPUT_DIR / "workspaces"   # 控制层管理的 fork 工作区根

# python 解释器：优先 .venv（runner 的 __main__ 也会重定向到 .venv，直接用省一次重启）
_VENV_PYTHON = LLMSEC_REPO / ".venv" / "Scripts" / "python.exe"
if not _VENV_PYTHON.exists():
    # 非 Windows venv 布局
    _VENV_PYTHON = LLMSEC_REPO / ".venv" / "bin" / "python"
PYTHON = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else os.environ.get("PYTHON", "python")


def ensure_workspaces_dir() -> Path:
    """确保 workspaces 目录存在。"""
    WORKSPACES_DIR.mkdir(parents=True, exist_ok=True)
    return WORKSPACES_DIR
