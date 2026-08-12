"""control.agent.tools — 把控制层能力暴露为结构化 tool 定义。

每个 tool = {name, description, parameters (JSON schema), call(args) -> result}。
供 agent/loop.py 的对话循环调用，也可被外部 agent 框架（如 LLM tool-calling）直接消费。

设计：tool 是纯函数 + schema 声明，不绑定任何特定 agent 框架。
切换 LLM 后端（OpenAI function calling / 自研循环）只需改 loop.py，tools 不变。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from control.core import compare as compare_mod
from control.core import orchestrator as orch_mod
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


def _tool_orchestrate():
    return Tool(
        name="orchestrate",
        description="批量并行 fork + run 多个工作单元（A/B 对比实验）。每个 spec 起一个隔离 run。",
        parameters={
            "type": "object",
            "properties": {
                "specs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "source": {"type": "string", "default": "global"},
                            "target": {"type": "string"},
                            "max_rounds": {"type": "integer", "default": 5},
                            "seed": {"type": "integer"},
                        },
                        "required": ["name"],
                    },
                },
                "max_workers": {"type": "integer", "default": 2},
                "compare_after": {"type": "boolean", "default": True},
            },
            "required": ["specs"],
        },
        call=lambda args: orch_mod.orchestrate(
            [orch_mod.RunSpec(**s) for s in args["specs"]],
            max_workers=args.get("max_workers", 2),
            compare_after=args.get("compare_after", True),
        ),
    )


def _tool_merge():
    """merge：把一个或多个工作区/run 的结果合并到全局或另一工作区（显式统一动作）。"""
    return Tool(
        name="merge",
        description=(
            "把一个或多个源（work-dir/workspace/global）的 R 矩阵观测合并到目标（global 或 ws:<name>）。"
            "runner 默认不再自动 publish 全局 R，更新全局 R 必须经此 tool 显式触发。"
            "默认 dry-run 预览，confirm=True 执行。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "源列表：'global' / 'ws:<name>' / work-dir 目录路径",
                },
                "target": {"type": "string", "description": "目标：'global' 或 'ws:<name>'"},
                "models": {"type": "array", "items": {"type": "string"}, "description": "只合并指定 model（默认全部）"},
                "confirm": {"type": "boolean", "description": "True 执行合并（默认 False=dry-run 预览）", "default": False},
            },
            "required": ["sources", "target"],
        },
        call=lambda args: _do_merge(args),
    )


def _do_merge(args: dict) -> dict:
    """经 invoker 调 llmsec-manage merge（控制层不碰 R 内部）。

    执行合并（confirm=True）后，对每个 ws:<name> 源回写 _index.json 的 merged 状态，
    使分支生命循环的「合并」阶段可追踪。
    """
    from control.core.invoker import _manage_argv, _run
    sub = ["merge", "--sources", *args["sources"], "--target", args["target"], "--json"]
    if args.get("models"):
        sub += ["--models", *args["models"]]
    if args.get("confirm"):
        sub.append("--yes")
    res = _run(_manage_argv(sub))
    res.require_ok()
    result = res.json or {}
    # 状态闭合：执行合并后标记各 workspace 源为已合并（容错——merge 已成功不可逆）
    if args.get("confirm") and result.get("dry_run") is False:
        from control.core.workspace import mark_merged
        target = args["target"]
        for src in args["sources"]:
            if src.startswith("ws:"):
                ok = mark_merged(src[3:], target)
                if not ok:
                    result.setdefault("warnings", []).append(
                        f"工作区 {src[3:]} 状态更新失败（merge 已生效）")
    return result


def _tool_review_run():
    """门下省事后审查：读 run 报告，识别异常，呈递关键摘要。"""
    return Tool(
        name="review_run",
        description=(
            "审查一个 run 的安全评测报告：读取 runner_report.json + security_tree.json，"
            "用阈值规则识别异常（ASR/FPR/收敛/覆盖率/真实盲区），生成中文审查摘要。"
            "任务完成后自动审查，用户也可主动要求「审查/总结一下 X」。"
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
    """调 review.review_run（不经 subprocess，直接读文件 + LLM）。"""
    from control.agent.review import review_run
    return review_run(args["run"])


# ============================================================
# 注册表（模块级单例：tool 实例稳定，便于 monkeypatch 单测与外部 agent 复用）
# ============================================================
_REGISTRY: list[Tool] | None = None


def all_tools() -> list[Tool]:
    """返回全部已注册 tool（首次调用构建，之后复用同一批实例）。"""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = [
            _tool_list_runs(),
            _tool_compare_runs(),
            _tool_fork_workspace(),
            _tool_list_workspaces(),
            _tool_delete_workspace(),
            _tool_orchestrate(),
            _tool_merge(),
            _tool_review_run(),
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
