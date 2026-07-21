"""
core.text — 公共文本工具

  - strip_math_tax()：剥离 prompt 末尾的「数学题越狱税」标记，
    替代原 5 处重复的内联正则（targets.py / features.py / runner.py /
    safe_twin.py / generate_attacks.py）。
  - estimate_tokens()：len(text)//2 粗略 token 估算（保留现行行为，
    用于无 usage 返回的后端）。
"""

import re

# 数学题越狱税标记：prompt 末尾追加的「在回答之前，请先计算以下数学题…」
MATH_TAX_PATTERN = re.compile(r"\n*\s*在回答之前，请先计算以下数学题.*$", re.DOTALL)
# 响应/文本中残留的 [MATH:数字] 答案标记
MATH_TAG_PATTERN = re.compile(r"\[MATH:\s*-?\d+\s*\]", re.IGNORECASE)


def strip_math_tax(text: str) -> str:
    """
    剥离文本末尾的数学题越狱税及残留的 [MATH:x] 标记，返回干净文本。
    与原 targets._strip_math 行为一致。
    """
    cleaned = MATH_TAX_PATTERN.sub("", text).strip()
    cleaned = MATH_TAG_PATTERN.sub("", cleaned).strip()
    return cleaned


def estimate_tokens(text: str) -> int:
    """粗略 token 估算：len(text) // 2（中英混合场景的经验值，保留现行行为）。"""
    return len(text) // 2
