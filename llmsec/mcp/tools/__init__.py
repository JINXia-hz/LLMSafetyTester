"""llmsec.mcp.tools — MCP 工具按风险分层的注册模块。

register_all(mcp) 汇总注册全部工具：
  - compute:  Tier 1 纯函数（obfuscate/scoring/features/metrics）
  - query:    Tier 2 只读查询（list_runs/compare/elo 派生）
  - actions:  Tier 3 写操作（delete/clean/fork/snapshot，带两步确认）
  - tasks:    Tier 4 长任务（run_evaluation/get_task_status/cancel）
"""

from __future__ import annotations

from typing import Any

from llmsec.mcp.tools import actions, compute, query, tasks


def _warmup_imports() -> None:
    """在主线程预 import 重模块，避免首次工具调用时在工作线程触发 lazy import。

    FastMCP 默认 run_in_thread=True，工具函数在子线程执行。Python 的 import lock
    机制下，若重模块（elo/judge/results 等）的首次 import 发生在子线程，
    可能与主线程的并发活动竞争 import lock，导致极长的阻塞（实测 elo 模块
    import 链近 2s，在线程内可能被 lock 放大到数十秒甚至超时）。

    在 register_all 时（主线程、server 启动前）预 import，让首次加载的开销
    发生在启动阶段而非首次调用。
    """
    import control.agent.gazette  # noqa: F401
    import control.agent.menxia.review  # noqa: F401
    import control.agent.shangshu.capabilities  # noqa: F401
    import control.agent.shangshu.plan  # noqa: F401

    # control 包（同级，靠 sys.path 发现）
    import control.core.compare  # noqa: F401
    import control.core.env_snapshot  # noqa: F401
    import control.core.orchestrator  # noqa: F401
    import control.core.workspace  # noqa: F401
    import llmsec.attacks.obfuscators  # noqa: F401
    import llmsec.core.results  # noqa: F401
    import llmsec.evaluation.cluster_analysis  # noqa: F401
    import llmsec.evaluation.elo  # noqa: F401
    import llmsec.evaluation.scoring  # noqa: F401
    import llmsec.management.merge  # noqa: F401
    import llmsec.management.runs  # noqa: F401
    import llmsec.reporting.report  # noqa: F401


def _warmup_caches() -> None:
    """预热运行时缓存，避免首次工具调用时在子线程触发慢操作。

    get_thresholds 首次调用会 subprocess 跑 llmsec-manage（~2s），
    在并发请求场景下会导致超时。在启动时（主线程）预填充缓存。
    """
    try:
        from control.agent.menxia import review

        review.get_thresholds()
    except Exception:
        pass  # 预热失败不阻塞启动，后续调用会重试


def register_all(mcp: Any) -> None:
    """把全部工具注册到 FastMCP server。"""
    _warmup_imports()
    _warmup_caches()
    compute.register(mcp)
    query.register(mcp)
    actions.register(mcp)
    tasks.register(mcp)
