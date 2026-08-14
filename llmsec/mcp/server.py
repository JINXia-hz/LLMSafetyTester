"""llmsec.mcp.server — FastMCP server 实例创建与启动。

把 llmsec 作为 MCP 工具库暴露，让外部 agent（ZCode/Cursor/Claude）成为
真正的决策者：自己读工具描述、自己决定调什么、自己组织编排。

启动方式：
  stdio（默认，适配 IDE 集成）:
      llmsec-mcp                         # 或 python -m llmsec.mcp
  HTTP（适配远程/多客户端）:
      llmsec-mcp --transport http --port 8765

配置：复用项目根的 .env（TARGET_*/GENERATOR_*/JUDGE_* 等）。
不需要配置 control 层的三省制 LLM——本接口不走路由进三省制。
"""

from __future__ import annotations

from typing import Any


def create_server() -> Any:
    """创建并配置 FastMCP server，注册全部工具。

    Returns:
        配置好的 FastMCP 实例（未启动）。
    """
    from fastmcp import FastMCP

    from llmsec.mcp.tools import register_all

    mcp = FastMCP("llmsec")
    register_all(mcp)
    return mcp


def main() -> None:
    """命令行入口：解析参数并启动 server。"""
    import argparse
    import os

    parser = argparse.ArgumentParser(
        prog="llmsec-mcp",
        description="llmsec MCP server —— 把 llmsec 作为工具库暴露给外部 agent",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="传输方式：stdio（默认，适配 IDE/ZCode 集成）或 http（远程访问）",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="HTTP 模式监听地址（默认 127.0.0.1）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="HTTP 模式监听端口（默认 8765）",
    )
    args = parser.parse_args()

    mcp = create_server()

    if args.transport == "stdio":
        # 防僵尸进程：注册 atexit + SIGBREAK/SIGINT 处理器确保进程退出时干净释放。
        # Windows 上 Ctrl+Break (SIGBREAK) 比 Ctrl+C 更可靠地被捕获。
        # FastMCP 的 stdio transport 在正常情况下会自行检测 stdin EOF 并退出，
        # 这里是兜底——防止异常路径（如子线程阻塞）导致进程残留锁住 exe 文件。
        import atexit
        import signal

        def _clean_exit(signum=None, frame=None):
            os._exit(0 if signum is None else 128 + signum)

        atexit.register(_clean_exit)
        signal.signal(signal.SIGINT, _clean_exit)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, _clean_exit)

        mcp.run()
    else:
        mcp.run(transport="http", host=args.host, port=args.port)
