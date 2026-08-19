#!/usr/bin/env python3
"""生成器出口自检——攻击生成侧的统一出口（Step 2「收编」）。

ensure_contract 是生成器出口的唯一硬约束：仓内两支生成器（generate.py 的
L1 生成、harmbench.py 的模板组装）、外部导入通道（management 的
attacks import）以及 Step 3 的进化算子都只承诺一件事——**产物能过
AttackRecord 契约校验**。不做更重的抽象：等 Step 3 的真实需求落地后
再长出参数面，避免接口先行于用例（原 AttackGenerator Protocol 仅测试
消费，已删）。
"""
from __future__ import annotations


def ensure_contract(entries: list[dict], *, where: str) -> None:
    """生成器输出自检：任一条目违反契约立即抛错，不静默落盘。

    where：报错时的来源标注（如 "generate.py 1.1.1"），方便定位是哪支
    生成器、哪一段产物坏了。
    """
    from llmsec.attacks.schema import validate_record

    for e in entries:
        _, issues = validate_record(e)
        if issues:
            raise ValueError(f"生成器产物违反契约（{where}）{e.get('id', '?')}: {issues}")
