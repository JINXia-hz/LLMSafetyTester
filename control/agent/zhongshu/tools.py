"""control.agent.tools — 中书省 + 规则兜底共用的查询/管理工具。

三省重构后，工具职责划分：
  - 中书省保留**查询类**工具（list_runs/compare_runs/review_run/list_workspaces）
    + **简单管理类**（fork_workspace/delete_workspace，规则兜底 loop.py 也会解析这两个意图）
  - 执行类操作（merge/clean_cache/delete_runs/orchestrate）统一由尚书省 capabilities 承担

每个 tool = {name, description, parameters (JSON schema), call(args) -> result}。
供 zhongshu._react_loop / loop._parse_intent / cli tool 子命令调用。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from control.core import compare as compare_mod
from control.core import workspace as ws_mod
from control.core.invoker import list_runs


# ============================================================
# Tool 类型
# ============================================================
class Tool:
    """一个可被 agent 调用的工具。"""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict,        # JSON schema
        call: Callable[[dict], Any],
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.call = call

    def to_schema(self) -> dict:
        """导出为 OpenAI function-calling 兼容的 schema。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ============================================================
# Tool 实现
# ============================================================
def _tool_list_runs():
    return Tool(
        name="list_runs",
        description=(
            "列出 run 历史。包含两类：output/runs/ 下的历史 run，以及 output/workspaces/ 下"
            "各 fork 分支内的 run（name 形如 'ws:<分支>/<target>'）。可按目标/日期/垃圾过滤。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "按目标模型名过滤（可选）"},
                "since": {"type": "string", "description": "起始日期 YYYY-MM-DD（可选）"},
                "junk_only": {"type": "boolean", "description": "仅列失败/无报告的垃圾 run", "default": False},
                "include_workspaces": {"type": "boolean", "description": "是否包含 workspace 分支内的 run", "default": True},
            },
        },
        call=lambda args: _do_list_runs(args),
    )


def _do_list_runs(args: dict) -> list[dict]:
    """列出历史 run +（可选）workspace 分支内的 run。"""
    runs = list_runs(
        target=args.get("target"),
        since=args.get("since"),
        junk_only=args.get("junk_only", False),
    )
    if args.get("include_workspaces", True):
        from control.core.compare import discover_workspace_runs
        ws_runs = discover_workspace_runs()
        if args.get("target"):
            ws_runs = [r for r in ws_runs if r.get("target") == args["target"]
                       or r.get("target_model") == args["target"]]
        runs = runs + ws_runs
    return runs


def _tool_compare_runs():
    return Tool(
        name="compare_runs",
        description=(
            "对比多个 run 的安全评测指标（asr/fpr/elo/coverage 等）+ 威胁树差异。至少 2 个 run。"
            "支持历史 run（'ts/target'）与 workspace 分支 run（'ws:<分支>/<target>' 或 'ws:<分支>'）。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "runs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "run 名列表，如 ['2026-08-11_120000/minimax', 'ws:ab_test/minimax']",
                },
            },
            "required": ["runs"],
        },
        call=lambda args: compare_mod.compare(args["runs"]),
    )


def _tool_review_run():
    """门下省事后审查：读 run 报告，识别异常，呈递关键摘要。"""
    return Tool(
        name="review_run",
        description=(
            "审查一个 run 的安全评测报告：读取 runner_report.json + security_tree.json，"
            "用阈值规则识别异常（ASR/FPR/收敛/覆盖率/真实盲区），生成中文审查摘要。"
            "run 名支持 'ts/target'（历史）或 'ws:<分支>/<target>'（fork 分支）。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "run": {"type": "string", "description": "run 名，如 '2026-08-11_151938/minimax' 或 'ws:ab1/minimax'"},
            },
            "required": ["run"],
        },
        call=lambda args: _do_review(args),
    )


def _do_review(args: dict) -> dict:
    """调 menxia.review_run（不经 subprocess，直接读文件 + LLM）。"""
    from control.agent.menxia import review_run
    return review_run(args["run"])


def _tool_fork_workspace():
    return Tool(
        name="fork_workspace",
        description="fork 一个新的隔离测试环境（工作区），以当前全局或指定 run 的状态为起点。不污染全局。",
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "新工作区名（唯一）"},
                "source": {"type": "string", "description": "'global' 或 'run:<run_name>'", "default": "global"},
                "note": {"type": "string", "description": "备注", "default": ""},
            },
            "required": ["name"],
        },
        call=lambda args: ws_mod.fork(
            args["name"], source=args.get("source", "global"), note=args.get("note", ""),
        ),
    )


def _tool_list_workspaces():
    return Tool(
        name="list_workspaces",
        description="列出所有已创建的 fork 工作区。",
        parameters={"type": "object", "properties": {}},
        call=lambda _: ws_mod.list_workspaces(),
    )


def _tool_delete_workspace():
    return Tool(
        name="delete_workspace",
        description="删除一个 fork 工作区（仅删隔离副本，不影响全局）。",
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        call=lambda args: ws_mod.delete_workspace(args["name"]),
    )


# ============================================================
# 注册表（模块级单例：tool 实例稳定，便于 monkeypatch 单测与外部 agent 复用）
# ============================================================
_REGISTRY: list[Tool] | None = None


def all_tools() -> list[Tool]:
    """返回全部已注册 tool（首次调用构建，之后复用同一批实例）。

    工具集 = 查询类（list_runs/compare_runs/review_run/list_workspaces）
           + 简单管理类（fork_workspace/delete_workspace）。
    执行类操作（merge/clean_cache/delete_runs/orchestrate）由尚书省 capabilities 承担。
    """
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = [
            _tool_list_runs(),
            _tool_compare_runs(),
            _tool_review_run(),
            _tool_fork_workspace(),
            _tool_list_workspaces(),
            _tool_delete_workspace(),
        ]
    return _REGISTRY


def reset_registry() -> None:
    """重置注册表（测试用：重建所有 tool，丢弃 monkeypatch）。"""
    global _REGISTRY
    _REGISTRY = None


def tool_by_name(name: str) -> Tool | None:
    for t in all_tools():
        if t.name == name:
            return t
    return None


def call_tool(name: str, args: dict) -> Any:
    """按名调 tool，返回结果。tool 不存在抛 KeyError。"""
    t = tool_by_name(name)
    if t is None:
        raise KeyError(f"未知 tool: {name}")
    return t.call(args)
