#!/usr/bin/env python3
"""生成器薄接口 + 输出自检——攻击生成侧的统一出口（Step 2「收编」）。

AttackGenerator 是刻意保持极薄的协议：仓内两支生成器（generate.py 的
L1 生成、harmbench.py 的模板组装）、外部导入通道（management 的
attacks import）以及 Step 3 的进化算子（obfuscate/LLM 合成）都只承诺
一件事——**产物能过 AttackRecord 契约校验**。不做更重的抽象：等 Step 3
的真实需求落地后再长出参数面，避免接口先行于用例。
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable


@runtime_checkable
class AttackGenerator(Protocol):
    """攻击生成器契约。

    实现方承诺：
      - source：产地的稳定标识（schema.SOURCES 之一）
      - generate(**params)：惰性产出记录 dict 流，逐条须过
        llmsec.attacks.schema.validate_record（用 ensure_contract 自检）
    """

    source: str

    def generate(self, **params) -> Iterator[dict]: ...


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
