"""llmsec.mcp — MCP 工具库接口（llmsec 的第四种入口）。

把 llmsec 退回为"安全测试工具库"，让外部 agent（ZCode/Cursor/Claude）成为
真正的决策者。核心设计：

  - 主路线：工具库 + 轻护栏（不走路由进三省制）
  - 分层暴露：Tier 1 纯函数 / Tier 2 只读查询 / Tier 3 写操作（带两步确认）/ Tier 4 长任务
  - 配置：最小配置（只配 TARGET_*/GENERATOR_*/JUDGE_*，不强制配 control LLM）
  - 危险操作：dry-run 预览 + confirm token 两步确认（决策权在 agent）

快速开始：
  from llmsec.mcp.server import create_server
  mcp = create_server()
  mcp.run()   # stdio 传输

或命令行：
  llmsec-mcp                          # stdio（IDE 集成）
  llmsec-mcp --transport http         # HTTP（远程）
"""

from __future__ import annotations

# MCP server 可能从任意 CWD 启动（IDE/agent 的子进程），需确保项目根在 sys.path 中，
# 否则同级包 control（不在 setuptools packages 里，靠 CWD/PYTHONPATH 发现）无法 import。
# llmsec.mcp 的 __file__ = <root>/llmsec/mcp/__init__.py，parents[2] 即项目根。
import sys as _sys
from pathlib import Path as _Path

_ROOT = str(_Path(__file__).resolve().parents[2])
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from llmsec.mcp.server import create_server, main

__all__ = ["create_server", "main"]
