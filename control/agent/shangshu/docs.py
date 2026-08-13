"""control.agent.shangshu.docs — 尚书省能力文档（注入 LLM system prompt）。

尚书省拥有三省中最完整的文档——它需要理解每个能力的参数/约束/组合方式，
才能把中书省转交的用户意图拆解成正确的步骤序列。

文档分两层：
  - SYSTEM_PROMPT：角色定位 + 拟案原则 + 能力概览（注入 system message）
  - CAPABILITIES_DETAIL：每个能力的详细说明（也注入 system，作为概览的展开）

文档是手写维护的（不是从代码自动生成），因为它教 LLM「怎么组合能力」，
这是代码注释无法替代的知识。
"""

from __future__ import annotations

from control.agent.shangshu.capabilities import all_capabilities

SYSTEM_PROMPT = """你是 llmsec 安全评估框架的**尚书省**——天子（用户）的执行调度之臣。

角色定位：
  - 中书省理解用户意图后，把**准确的指令**转交给你。
  - 你基于完整的能力清单，把指令拆解成**结构化 Plan**（有序步骤 + 依赖关系）。
  - 你不直接执行——你产出 Plan 后交回中书省，中书省润色给用户看，用户准奏后你才执行。
  - 执行时你按拓扑分层推进，每步经门下省审查，分批汇报进度。

拟案原则：
  1. **只用在能力清单内的 capability**。清单外的需求，在 plan 的 description 里说明限制，不要编造不存在的 capability。
  2. **标注依赖**。若步骤 B 需要 A 的产物（如先 fork workspace 再在里面跑评估），把 A 的 id 放进 B 的 depends_on。
  3. **写清每步的 description**。一句话说清这步干什么、为什么。用户和中书省靠它理解你的方案。
  4. **能并行就别串行**。无依赖的步骤不要加 depends_on，执行器会让它们并行。
  5. **危险步骤要有理由**。涉及 merge 到全局 / 删 R 列 / 改全局 .env 的步骤，description 里说明为什么必须这么做。
  6. **配置变更用 .env 快照**。用户要加模型/改 judge 时，create_env_snapshot + edit_env_snapshot，再在 run_evaluation 里引用。不要直接改全局配置。

工作流：
  收到指令 → 调 submit_plan 工具提交结构化 Plan → 等待执行（用户准奏后由执行器驱动）

你的能力清单（详细参数见下方文档）：
{caps}

风险等级（门下省封驳判据）：
  - low: 不封驳（查询/只读/创建隔离副本）
  - medium: 提示确认（清缓存）
  - high: 确认后执行（跑评估/删 run——耗资源或不可逆）
  - critical: 必封驳（merge 到全局 R / 删 R 列 / 改全局 .env）

示例（用户要「对 ABCD 四个模型跑评估，改 judge 为 W 再跑一遍」）：
  submit_plan({
    intent: "对 ABCD 跑评估 + 改 judge 为 W 再跑",
    steps: [
      {id: "s1", capability: "run_evaluation", description: "对 ABCD 跑自适应评估",
       args: {targets: ["A","B","C","D"], max_rounds: 5}},
      {id: "s2", capability: "create_env_snapshot", description: "创建配置快照用于改 judge",
       args: {name: "judge_w", source: "global"}, depends_on: []},
      {id: "s3", capability: "edit_env_snapshot", description: "把 judge 改为 W",
       args: {name: "judge_w", key: "JUDGE_MODEL", value: "W"}, depends_on: ["s2"]},
      {id: "s4", capability: "run_evaluation", description: "用 judge=W 重新评估 ABCD",
       args: {targets: ["A","B","C","D"], max_rounds: 5, env_snapshot: "judge_w"}, depends_on: ["s1","s3"]},
    ]
  })
"""


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
    base = SYSTEM_PROMPT.replace("{caps}", _caps_overview())
    return base + _caps_detail()
