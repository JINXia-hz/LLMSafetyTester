"""control.agent.loop — 最小对话循环（人机互动中间者）。

设计为「可扩展的中间者」：
  - 当前：基于规则的意图解析（中文/英文关键词 → tool 调用）。
    足够覆盖「列一下 run / 对比 X 和 Y / fork 一个叫 Z 的环境」这类指令。
  - 未来：把 _parse_intent 换成 LLM tool-calling（接同一套 tools，不改主循环）。

循环：读输入 → 解析意图 → 调 tool → 渲染结果 → 等下一步。
退出：quit / exit / Ctrl-D。
"""

from __future__ import annotations

import json
import re
from typing import Any

from control.agent.zhongshu.tools import all_tools, call_tool


# ============================================================
# 结果渲染（人可读）
# ============================================================
def _render(result: Any, max_rows: int = 20) -> str:
    """把 tool 结果渲染为人可读文本。"""
    if isinstance(result, list):
        return _render_list(result, max_rows)
    if isinstance(result, dict):
        # 按「已知形状」选渲染器
        if "runs" in result and isinstance(result["runs"], list) and result["runs"] and "asr" in result["runs"][0]:
            return _render_compare(result)
        if "results" in result and "summary" in result:
            return _render_orchestrate(result)
        if "workspace" in result and "run" in result:
            return _render_fork_run(result)
        return _render_dict(result)
    return str(result)


def _render_list(items: list, max_rows: int) -> str:
    if not items:
        return "（无）"
    out = [f"共 {len(items)} 项："]
    for i, it in enumerate(items[:max_rows], 1):
        if isinstance(it, dict):
            # run 列表
            if "name" in it and "asr" in it:
                asr = it.get("asr")
                asr_s = f"{asr:.1%}" if isinstance(asr, (int, float)) else "-"
                elo = it.get("boundary_elo")
                elo_s = f"{elo:.0f}" if isinstance(elo, (int, float)) else "-"
                size = it.get("size", 0)
                size_s = f"{size / 1024:.0f}KB" if size else "-"
                out.append(f"  {i}. {it['name']}  target={it.get('target_model','-')}  "
                           f"level={(it.get('security_level') or '-')[:8]}  asr={asr_s}  elo={elo_s}  size={size_s}")
            elif "name" in it and "source" in it:
                # workspace 列表
                out.append(f"  {i}. {it['name']}  source={it.get('source')}  "
                           f"records={it.get('records',0)}  created={it.get('created','-')}")
            else:
                out.append(f"  {i}. {json.dumps(it, ensure_ascii=False)[:120]}")
        else:
            out.append(f"  {i}. {it}")
    if len(items) > max_rows:
        out.append(f"  ...（省略 {len(items) - max_rows} 项）")
    return "\n".join(out)


def _render_compare(report: dict) -> str:
    out = ["对比结果：", ""]
    rows = report.get("runs", [])
    if not rows:
        return "（无可用 run 对比）"
    # 列名必须与 compare.run_metrics 的行字段一致（run/target_model/asr/fpr/
    # boundary_elo/conv_rounds/security_level）。旧版用 target/elo/level，
    # 与字段名不符的三列恒渲染成 "-"。
    cols = ["run", "target_model", "asr", "fpr", "boundary_elo", "conv_rounds", "security_level"]
    # 动态列宽：按该列实际内容最长 + 表头
    def _cell(r, c):
        v = r.get(c)
        if isinstance(v, float):
            return f"{v:.3f}" if v < 1 else f"{v:.1f}"
        return str(v) if v is not None else "-"
    widths = []
    for c in cols:
        w = max([len(c)] + [len(_cell(r, c)) for r in rows])
        widths.append(min(w, 30))
    out.append("  ".join(c.ljust(widths[i]) for i, c in enumerate(cols)))
    out.append("  ".join("-" * widths[i] for i in range(len(cols))))
    for r in rows:
        cells = [_cell(r, c).ljust(widths[i]) for i, c in enumerate(cols)]
        out.append("  ".join(cells))
    if report.get("missing"):
        out.append(f"\n缺少报告（跳过）: {report['missing']}")
    return "\n".join(out)


def _render_orchestrate(rep: dict) -> str:
    s = rep.get("summary", {})
    out = [f"批量编排完成：{s.get('success',0)}/{s.get('total',0)} 成功"]
    for r in rep.get("results", []):
        status = r.get("status")
        name = r.get("spec", {}).get("name", "?")
        if r.get("run"):
            out.append(f"  {name}: {status} (rc={r['run'].get('returncode')}, "
                       f"{r['run'].get('elapsed_s')}s)")
        else:
            out.append(f"  {name}: {status} {r.get('error','')}")
    return "\n".join(out)


def _render_fork_run(r: dict) -> str:
    ws = r.get("workspace", {})
    run = r.get("run", {})
    return (f"✓ fork+run 完成: {ws.get('name')} (source={ws.get('source')})\n"
            f"  run: {'成功' if run.get('ok') else '失败'} "
            f"(rc={run.get('returncode')}, {run.get('elapsed_s')}s)\n"
            f"  日志: {run.get('log')}")


def _render_dict(d: dict) -> str:
    return json.dumps(d, ensure_ascii=False, indent=2)[:1000]


# ============================================================
# 意图解析（规则版，可替换为 LLM）
# ============================================================
def _parse_intent(text: str) -> tuple[str, dict] | None:
    """把自然语言/命令解析为 (tool_name, args)。

    支持的指令模式：
      list runs / 列 run / 列一下 run [--target X] [--junk]
      compare A B / 对比 A B / 对比 A、B、C
      fork NAME [--from global|run:X] / fork 一个叫 NAME 的
      workspaces / 列工作区
      orchestrate ...（复杂，建议直接用 CLI）
    无法识别返回 None。
    """
    text = text.strip()
    low = text.lower()

    # list runs（中文「列」后无 word boundary，不用 \b）
    # 用子串判断替代 (list|列|show).*(run) 正则——避免交替+.* 的 ReDoS 启发式告警。
    # 注意语义有放宽：原正则要求 list 类词在前、run 在后；子串判断不限词序
    # （如「run 列表」也会命中）。对启发式意图解析而言可接受。
    if (any(k in low for k in ("list", "列", "show")) and "run" in low) or low in ("runs", "run", "历史"):
        args: dict = {}
        m = re.search(r"(?:target|目标)[=\s]+(\S+)", text)
        if m:
            args["target"] = m.group(1).strip("\"'")
        if "junk" in low or "垃圾" in text or "失败" in text:
            args["junk_only"] = True
        return ("list_runs", args)

    # workspaces
    if "workspace" in low or "工作区" in text:
        if any(k in low for k in ("list", "列", "show")) or "列出" in text or "看一下" in text:
            return ("list_workspaces", {})
        # delete workspace（删/删除/delete/remove 工作区 NAME）
        if any(k in low for k in ("delete", "删", "删除", "remove")):
            # NAME = 工作区关键词后的 token（支持中英、连字符）。
            # 用单段正则提取，不用 .*? 桥接（消除 ReDoS 启发式告警）。
            m = re.search(r"(?:workspace|工作区)\s+([\w.-]+)", text, re.IGNORECASE)
            if m:
                return ("delete_workspace", {"name": m.group(1)})

    # compare（中文词两侧均属 \w，\b 不产生边界，正则只保留英文；中文走子串判断）
    if "compare" in low or "对比" in text or "比较" in text:
        # 提取 run 名（支持空格/顿号/逗号分隔，带斜杠的 ts/target）
        runs = re.findall(r"[\w.-]+/[\w.-]+|\d{4}-\d{2}-\d{2}_\d{6}", text)
        if len(runs) >= 2:
            return ("compare_runs", {"runs": runs})
        return None

    # fork（\b(fork) 能匹配时 "fork" in low 必为 True，正则冗余已删）
    if "fork" in low or "建环境" in text or "新环境" in text:
        # 名字：fork NAME 或 叫 NAME 的
        m = re.search(r"fork\s+(\S+)", text) or re.search(r"叫\s*(\S+?)\s*(?:的|环境)", text)
        if not m:
            return None
        name = m.group(1).strip("\"'")
        args = {"name": name}
        src = re.search(r"(?:from|来源|源)[=\s:]+(global|run:\S+)", text)
        if src:
            args["source"] = src.group(1)
        return ("fork_workspace", args)

    return None


# ============================================================
# 对话循环
# ============================================================
def _help() -> str:
    tools = all_tools()
    lines = ["可用指令（自然语言或命令）：", ""]
    for t in tools:
        lines.append(f"  {t.name}: {t.description}")
    lines += ["", "也可直接输入 JSON: {\"tool\": \"list_runs\", \"args\": {}}",
              "退出: quit / exit"]
    return "\n".join(lines)


def chat_one(text: str) -> str:
    """处理单轮输入，返回渲染后的回复。供 CLI / 外部调用复用。"""
    text = text.strip()
    if not text:
        return ""
    # JSON 直调
    if text.startswith("{"):
        tool_name = "?"  # try 前绑定：异常分支不隐式依赖"解析已成功"的分支顺序
        try:
            req = json.loads(text)
            tool_name = req.get("tool") or "?"
            result = call_tool(req["tool"], req.get("args", {}))
            return _render(result)
        except (json.JSONDecodeError, KeyError) as e:
            return f"❌ JSON 调用失败: {e}"
        except Exception as e:
            # call_tool 执行期异常（fork 重名 / require_ok 失败等）也不能上抛——
            # 此函数同时服务 CLI 与 dialogue 规则分支，上抛会让 API 直接 500
            return f"❌ 执行 {tool_name} 失败: {type(e).__name__}: {e}"
    # 意图解析
    parsed = _parse_intent(text)
    if parsed is None:
        return f"未识别指令。输入 help 查看可用指令。\n原始: {text[:80]}"
    tool_name, args = parsed
    try:
        result = call_tool(tool_name, args)
        return _render(result)
    except Exception as e:
        return f"❌ 执行 {tool_name} 失败: {type(e).__name__}: {e}"


def chat_loop() -> None:
    """交互式对话循环（REPL）。"""
    print("=== llmsec 控制层 · 对话中间者 ===")
    print("输入指令（help 查看帮助，quit 退出）\n")
    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            break
        if text.lower() in ("quit", "exit", "q"):
            print("再见")
            break
        if text.lower() in ("help", "?", "帮助"):
            print(_help())
            continue
        reply = chat_one(text)
        if reply:
            print(reply)
        print()
