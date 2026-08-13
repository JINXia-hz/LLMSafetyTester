"""control.agent.shangshu.docs — 尚书省能力文档（注入 LLM system prompt）。

尚书省拥有三省中最完整的文档——它需要理解每个能力的参数/约束/组合方式，
才能把中书省转交的用户意图拆解成正确的步骤序列。

prompt 主体（角色定位 + 拟案原则 + 示例）在 prompts.py 统一管理；
本模块负责运行时动态拼接能力清单（{caps} 占位）+ 详细文档。

文档是手写维护的（不是从代码自动生成），因为它教 LLM「怎么组合能力」，
这是代码注释无法替代的知识。
"""

from __future__ import annotations

from control.agent.prompts import SHANGSHU_PROMPT
from control.agent.shangshu.capabilities import all_capabilities


def _caps_overview() -> str:
    """能力一览（name + 一句话描述 + risk_level），注入 system prompt。"""
    lines = []
    for c in all_capabilities():
        lines.append(f"  - {c.name} [{c.risk_level}]: {c.description.split('。')[0]}。")
    return "\n".join(lines)


def _caps_detail() -> str:
    """能力详细文档（含 doc 字段），注入 system prompt。"""
    lines = ["\n\n=== 能力详细文档 ==="]
    for c in all_capabilities():
        lines.append(f"\n【{c.name}】risk={c.risk_level}")
        lines.append(f"  {c.description}")
        if c.doc:
            lines.append(f"  说明：{c.doc}")
        # 参数摘要
        props = c.parameters.get("properties", {})
        required = set(c.parameters.get("required", []))
        if props:
            param_strs = []
            for pk, pv in props.items():
                req = "必填" if pk in required else "可选"
                desc = pv.get("description", "")
                param_strs.append(f"    - {pk} ({req}): {desc}")
            lines.append("  参数：")
            lines.append("\n".join(param_strs))
    return "\n".join(lines)


def build_system_prompt() -> str:
    """组装完整的尚书省 system prompt（概览 + 详细文档）。"""
    base = SHANGSHU_PROMPT.replace("{caps}", _caps_overview())
    return base + _caps_detail()
