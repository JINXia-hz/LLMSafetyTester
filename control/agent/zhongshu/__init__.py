"""control.agent.zhongshu — 中书省（前台 + 意图理解 + 润色）。

三省中的对话主入口：
  - 简单查询 → 自己用工具处理（list_runs/compare_runs/review_run/list_workspaces）
  - 复杂指令 → 调尚书省 draft_plan → 润色成给用户的中文方案 → 返回 plan_pending

子模块：
  dialogue.py — 对话主循环（handle_message）
  session.py  — 会话管理（get_or_create / append / reset）
  tools.py    — 查询工具（6 个：list_runs/compare_runs/review_run/fork/list/delete_workspace）
  fallback.py — 规则版兜底（LLM 未配置/失败时）

对外接口（本 __init__ re-export）：
  handle_message(text, session_id)    对话主入口
  session.get_or_create / reset       会话管理
  tools.all_tools / call_tool         查询工具
"""

from control.agent.zhongshu import session
from control.agent.zhongshu.dialogue import ZhongshuTurn, handle_message
from control.agent.zhongshu.fallback import chat_loop, chat_one
from control.agent.zhongshu.tools import (
    Tool,
    all_tools,
    call_tool,
    reset_registry,
    tool_by_name,
)

__all__ = [
    "handle_message", "ZhongshuTurn",
    "session",
    "all_tools", "call_tool", "tool_by_name", "reset_registry", "Tool",
    "chat_one", "chat_loop",
]
