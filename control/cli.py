"""control.cli — 控制层命令入口。

用法：
  python -m control workspace fork <name> [--source global|run:<run>] [--note ...]
  python -m control workspace list
  python -m control workspace delete <name>
  python -m control compare <run...> [--json]
  python -m control orchestrate <specs.json> [--workers N] [--json]
  python -m control chat
  python -m control tool <name> [args.json]

所有输出支持 --json（供 agent / 脚本消费）。
"""

from __future__ import annotations

import argparse
import json
import sys


def _print(obj, *, json_mode: bool, title: str = "") -> None:
    if json_mode:
        out = {"_title": title, **obj} if isinstance(obj, dict) and title else obj
        print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    else:
        from control.agent.loop import _render
        print(_render(obj) if not title else f"=== {title} ===\n{_render(obj)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m control",
        description="llmsec 控制层：fork / 对比 / 编排 / 对话",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- workspace ----
    ws = sub.add_parser("workspace", help="fork 工作区管理")
    ws_sub = ws.add_subparsers(dest="ws_cmd", required=True)

    p_fork = ws_sub.add_parser("fork", help="fork 一个隔离工作区")
    p_fork.add_argument("name")
    p_fork.add_argument("--source", default="global", help="global 或 run:<run_name>")
    p_fork.add_argument("--note", default="")
    p_fork.add_argument("--json", action="store_true")
    p_fork.add_argument("--run", action="store_true", help="fork 后立即起一个 llmsec run")
    p_fork.add_argument("--target", default=None)
    p_fork.add_argument("--max-rounds", type=int, default=5)
    p_fork.add_argument("--seed", type=int, default=None)

    p_wslist = ws_sub.add_parser("list", help="列出工作区")
    p_wslist.add_argument("--json", action="store_true")

    p_wsd = ws_sub.add_parser("delete", help="删除工作区")
    p_wsd.add_argument("name")
    p_wsd.add_argument("--json", action="store_true")

    # ---- compare ----
    p_cmp = sub.add_parser("compare", help="对比多个 run")
    p_cmp.add_argument("runs", nargs="+")
    p_cmp.add_argument("--json", action="store_true")

    # ---- orchestrate ----
    p_orc = sub.add_parser("orchestrate", help="批量并行编排")
    p_orc.add_argument("specs", help="JSON 文件路径，内容为 RunSpec 列表")
    p_orc.add_argument("--workers", type=int, default=2)
    p_orc.add_argument("--json", action="store_true")

    # ---- chat ----
    sub.add_parser("chat", help="交互式对话中间者")

    # ---- tool ----
    p_tool = sub.add_parser("tool", help="直接调一个 tool（供脚本/agent）")
    p_tool.add_argument("name")
    p_tool.add_argument("args", nargs="?", default="{}", help="JSON 参数")
    p_tool.add_argument("--json", action="store_true")

    args = parser.parse_args()

    if args.cmd == "workspace":
        from control.core import workspace as ws_mod
        if args.ws_cmd == "fork":
            if args.run:
                from control.core.workspace import fork_and_run
                r = fork_and_run(args.name, source=args.source, target=args.target,
                                 max_rounds=args.max_rounds, seed=args.seed, note=args.note)
                _print(r, json_mode=args.json, title="fork+run")
            else:
                r = ws_mod.fork(args.name, source=args.source, note=args.note)
                _print(r, json_mode=args.json, title="fork")
            return 0
        if args.ws_cmd == "list":
            r = ws_mod.list_workspaces()
            _print(r, json_mode=args.json, title="workspaces")
            return 0
        if args.ws_cmd == "delete":
            r = ws_mod.delete_workspace(args.name)
            _print(r, json_mode=args.json, title="delete workspace")
            return 0

    if args.cmd == "compare":
        from control.core.compare import compare
        r = compare(args.runs)
        _print(r, json_mode=args.json, title="compare")
        return 0

    if args.cmd == "orchestrate":
        from control.core.orchestrator import RunSpec, orchestrate
        specs_data = json.loads(__import__("pathlib").Path(args.specs).read_text(encoding="utf-8"))
        specs = [RunSpec(**s) for s in specs_data]
        r = orchestrate(specs, max_workers=args.works)
        _print(r, json_mode=args.json, title="orchestrate")
        return 0

    if args.cmd == "chat":
        from control.agent.loop import chat_loop
        chat_loop()
        return 0

    if args.cmd == "tool":
        from control.agent.tools import call_tool
        tool_args = json.loads(args.args) if args.args else {}
        r = call_tool(args.name, tool_args)
        _print(r, json_mode=True)  # tool 调用默认 JSON（机器消费）
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
