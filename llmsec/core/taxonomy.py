"""
core.taxonomy — harm_type / category 归一化层

设计背景：攻击集来源多样（L1 生成器 6 类、外部源 11 类、HarmBench 原生分类），
三套词表并存且无统一映射，导致：
  - 聚类特征维度因词表断裂而错误分裂（features.py:385 动态派生 label_names）
  - 冷启动预测器静默丢弃未知词表的特征位（cold_start.py:716 in label_names 检查）
  - ASR 报告维度碎片化（evaluator.py:301 按原始字符串分桶）

本模块提供唯一归一化入口 normalize_harm_type()，在所有读取边界统一调用，
保证 fit/predict 两侧词表对称。
"""
from __future__ import annotations

# 标准危害类型词表（12 类，按频率排序，与 all_merged 一致 + copyright）
CANONICAL_HARM_TYPES = [
    "other",
    "hate",
    "violence",
    "illegal",
    "cyber",
    "weapon_drugs",
    "fraud",
    "privacy",
    "self_harm",
    "sexual",
    "misinfo",
    "copyright",
]

# 别名 → 标准类型（小写匹配）
# 覆盖 HarmBench 原生 SemanticCategory、大小写变体、中文别名
_HARM_ALIASES: dict[str, str] = {
    # HarmBench 原生 SemanticCategory → 标准
    "chemical_biological": "weapon_drugs",
    "chemical_weapons": "weapon_drugs",
    "cybercrime_intrusion": "cyber",
    "cybercrime": "cyber",
    "misinformation_disinformation": "misinfo",
    "misinformation": "misinfo",
    "harassment_bullying": "hate",
    "harassment": "hate",
    "harmful": "other",
    # 中文别名（生成器上下文 / 手工标注可能使用）
    "暴力": "violence",
    "仇恨": "hate",
    "欺诈": "fraud",
    "隐私": "privacy",
    "自残": "self_harm",
    "自伤": "self_harm",
    "违法": "illegal",
    "非法": "illegal",
    "色情": "sexual",
    "网络攻击": "cyber",
    "武器毒品": "weapon_drugs",
    "虚假信息": "misinfo",
    "版权": "copyright",
    "其他": "other",
}


def normalize_harm_type(raw: str) -> str:
    """将任意 harm_type 字符串归一化到标准词表。

    匹配顺序：精确匹配标准词表 → 别名映射 → 兜底 'other'。
    大小写不敏感；空值/None → 'other'。
    """
    if not raw:
        return "other"
    key = raw.strip().lower()
    if key in CANONICAL_HARM_TYPES:
        return key
    return _HARM_ALIASES.get(key, "other")


def normalize_harm_types(values) -> list[str]:
    """批量归一化（去重保序）。"""
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        nv = normalize_harm_type(v)
        if nv not in seen:
            seen.add(nv)
            out.append(nv)
    return out
