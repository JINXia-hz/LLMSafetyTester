#!/usr/bin/env python3
"""攻击记录契约——攻击集数据的唯一 schema 权威（Step 1「契约+体检」）。

背景：攻击集 JSONL 产自多条生成链路（仓内 generate.py/harmbench.py +
外部同事脚本导入），schema 已各自为政。本模块定义唯一的 AttackRecord
契约，供体检校验（validate.py）与后续收编/进化使用。

设计约束：
  - **只做校验与标准化视图**，不替换现有 dict 流动路径——runner/
    attack_phase 的读取零改动，契约通过 model_validate 对存量数据逐条
    体检（违规被收集为报告项，而非整批拒绝）。
  - extra="allow"：harmbench 溯源字段（behavior_id/jailbreak_template_*
    等）原样透传，契约只声明跨数据集公共字段。
  - 血统字段（evolved/operator/parent_id/generation）为 Step 3 自适应
    攻击进化预留：存量数据默认 evolved=False，进化体必须填全。
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llmsec.params import ATTACK_MOJIBAKE_CHARS, ATTACK_PROMPT_MAX_CHARS

# 六类危害 + 兜底 other（wildjailbreak 系数据集的 harm_type 几乎全为 other，
# 这是体检要量化的重点，不是契约要拒绝的理由）
HARM_TYPES = ("violence", "hate", "fraud", "privacy", "self_harm", "illegal", "other")

# 已知攻击集来源（l1/harmbench 为仓内生成器；其余为外部导入；evolved 为
# Step 3 进化体预留；unknown 为无法推断）
SOURCES = (
    "l1", "harmbench", "wildjailbreak", "in_the_wild", "rubend18",
    "jailbreakv28k", "jailbreakdb", "evolved", "unknown",
)

# 文件名 → source 推断表（记录自带 source 字段时以记录为准）
_FILENAME_SOURCE_MAP = {
    "l1": "l1",
    "harmbench_ensemble": "harmbench",
    "wildjailbreak": "wildjailbreak",
    "in_the_wild": "in_the_wild",
    "rubend18": "rubend18",
    "jailbreakv28k": "jailbreakv28k",
    "jailbreakdb": "jailbreakdb",
}


def infer_source(filename: str, record_source: str | None = None) -> str:
    """推断一条记录的 source：记录自带字段 > 文件名映射 > unknown。"""
    if record_source:
        return record_source
    stem = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    stem = stem.rsplit(".", 1)[0]
    return _FILENAME_SOURCE_MAP.get(stem, "unknown")


def detect_mojibake(text: str) -> list[str]:
    """检测 UTF-8 被按 GBK 二次解码的特征字符（启发式，只报告不修删）。

    返回命中的特征字符列表（去重保序）；空列表 = 干净。特征字符集见
    params.ATTACK_MOJIBAKE_CHARS——这些字在正常简体中文语料中近乎不出现
    （如 jailbreakv28k 的 "doesn鈥檛"），但存在理论误报可能，结论以
    Step 2 清洗时的人工/自动复核为准。
    """
    hits: list[str] = []
    for ch in ATTACK_MOJIBAKE_CHARS:
        if ch in text and ch not in hits:
            hits.append(ch)
    return hits


class AttackRecord(BaseModel):
    """一条攻击 prompt 的契约视图。

    必填三件套（id/method/prompt）缺失或 prompt 超界 → ValidationError，
    体检计为 error 级；harm_type/source 为宽松 str（未知值计 warn 级分布，
    不拒收）——存量数据先量化再清洗，不在契约层一刀切。
    """

    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1)
    method: str = Field(min_length=1)
    prompt: str = Field(min_length=1, max_length=ATTACK_PROMPT_MAX_CHARS)

    # 宽松枚举：未知值不拒收，由体检报告分布
    harm_type: str = "other"
    source: str = "unknown"

    # 跨数据集公共可选字段（各家命名差异在此归位）
    category: str | None = None
    category_name: str | None = None
    math_problem: str | None = None
    expected_answer: object | None = None  # int/str 皆有（l1=int，其余多为 str）
    build_difficulty: str | None = None    # "L1"/"L2"/"L3"；外部数据集普遍缺失
    functional_category: str | None = None

    # ---- 血统预留（Step 3 自适应攻击进化）----
    evolved: bool = False
    operator: str | None = None   # 生成算子：obfuscate:<name> / llm_synth
    parent_id: str | None = None  # 父代记录 id；evolved=True 时必填（体检校验）
    generation: int | None = None  # 进化代数；原生数据为 None


def validate_record(raw: dict, *, source: str | None = None) -> tuple[AttackRecord | None, list[str]]:
    """校验单条记录，返回 (记录或 None, 违规描述列表)。

    违规描述为短字符串（"prompt: String should have at most ..." 风格的
    pydantic 摘要），供体检逐条归档；记录合法时违规列表为空。source
    传入时覆盖记录自带的 source 字段（供文件级推断）。
    """
    data = dict(raw)
    if source:
        data["source"] = source
    try:
        rec = AttackRecord.model_validate(data)
    except ValidationError as e:
        issues = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"]) or "<root>"
            issues.append(f"{loc}: {err['msg']}")
        return None, issues
    # 血统一致性：进化体必须可溯源（原生数据 evolved=False 不受影响）
    extra: list[str] = []
    if rec.evolved and not rec.parent_id:
        extra.append("lineage: evolved=True 但缺 parent_id")
    return rec, extra


__all__ = [
    "ATTACK_PROMPT_MAX_CHARS",
    "AttackRecord",
    "HARM_TYPES",
    "SOURCES",
    "detect_mojibake",
    "infer_source",
    "validate_record",
]
