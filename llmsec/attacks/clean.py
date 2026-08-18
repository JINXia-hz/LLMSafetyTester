#!/usr/bin/env python3
"""攻击集清洗器——修复外部数据集的编码事故与元数据损伤（Step 2「收编」）。

用法：
  python -m llmsec.attacks.clean                 # 清洗五份外部数据集 → attacks/cleaned/
  python -m llmsec.attacks.clean a.jsonl b.jsonl # 指定文件
  python -m llmsec.attacks.clean --no-merged     # 不重建 all_merged

清洗操作（原件零改动，产物落 attacks/cleaned/）：
  1. mojibake 分段修复：非 ASCII 连续段尝试 GBK 逆解码——成功 = 该段是
     "UTF-8 被 GBK 误读"的产物，替换为修复文本；失败 = 真 UTF-8 中文
     （如数学税）保留。实测 jailbreakv28k 2090 条损坏中 1255 条可这样
     确定性复原。
  2. 孤立标记补全（启发式）：剩余 835 条的标记（如 "./cmd 鈥 Can"）是
     多字节序列的第三字节在当初误读时被吃掉——字节已丢失，无法确定性
     复原。按上下文补最可能的原文：前邻 ASCII 字母/数字 → 右引号 ”，
     其余 → em-dash —。每处补全计入该条记录的 repaired 字段（区分
     确定性修复与启发式补全，人工复核时优先看后者）。
  3. method 去序号：外部数据集的 method 逐条唯一（"tpl-0000"），
     去 `-\\d{4,}$` 尾缀恢复模板族聚合（原值存 method_raw）。
  4. harm_original 保全：harm_type 不在六类枚举里的（harmbench 的
     copyright 等）原值另存 harm_original，harm_type 本身不动。

清洗后自动跑体检并打印 before/after 对比；all_merged.jsonl 从清洗后
成员重建，保留成员原 id（不再重排、不重新注题——恢复 id 可连接性）。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from llmsec.attacks.schema import HARM_TYPES, validate_record
from llmsec.core.logging import get_logger, setup_console
from llmsec.params import ATTACK_MOJIBAKE_CHARS

logger = get_logger(__name__)
setup_console()

# 五份外部数据集（体检结论：mojibake / method 唯一化 / other 泛滥都集中在这）
EXTERNAL_DATASETS = ("wildjailbreak", "in_the_wild", "rubend18", "jailbreakv28k", "jailbreakdb")

_ASCII_RUN = re.compile(r"([\x00-\x7f]+)")
_METHOD_SUFFIX = re.compile(r"-\d{4,}$")
_MARKERS = frozenset(ATTACK_MOJIBAKE_CHARS)


# ============================================================
# 修复原语
# ============================================================
def repair_mojibake(text: str) -> tuple[str, list[str]]:
    """修复文本中的 mojibake，返回 (修复后文本, 操作列表)。

    两遍策略：第 1 遍分段逆解码（确定性）；第 2 遍孤立标记启发式补全
    （前邻 ASCII 字母数字 → 右引号，其余 → em-dash）。操作列表元素为
    "mojibake_segment" / "marker_quote" / "marker_dash"，供 repaired 字段
    区分确定性修复与启发式补全。幂等：干净文本返回原文与空列表。
    """
    ops: list[str] = []
    # 第 1 遍：分段逆解码
    parts: list[str] = []
    for seg in _ASCII_RUN.split(text):
        if seg and not seg.isascii():
            try:
                parts.append(seg.encode("gbk").decode("utf-8"))
                ops.append("mojibake_segment")
            except (UnicodeEncodeError, UnicodeDecodeError):
                parts.append(seg)  # 真 UTF-8 中文段（数学税等）
        else:
            parts.append(seg)
    text = "".join(parts)

    # 第 2 遍：孤立标记补全（连片标记一次跳过，各记一次操作）
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in _MARKERS:
            prev = text[i - 1] if i > 0 else ""
            if prev.isascii() and prev.isalnum():
                out.append("\u201d")   # ” 右引号（如 "unhinged鈥 state"）
                ops.append("marker_quote")
            else:
                out.append("\u2014")   # — em-dash（如 "./cmd 鈥 Can"）
                ops.append("marker_dash")
            while i + 1 < n and text[i + 1] in _MARKERS:
                i += 1
        else:
            out.append(ch)
        i += 1
    return "".join(out), ops


def normalize_method(method: str) -> tuple[str, str | None]:
    """去掉 method 尾部序号恢复模板族，返回 (归一值, 原值或 None)。

    外部导入时 method 被写成 "模板名-0000"（逐条唯一，无法聚合）；
    去 `-\\d{4,}$` 后缀即恢复模板族（实测 wildjailbreak 2000→1409、
    jailbreakdb 2407→1111）。仅记录有变化时返回原值。
    """
    stripped = _METHOD_SUFFIX.sub("", method)
    if stripped and stripped != method:
        return stripped, method
    return method, None


def clean_record(raw: dict) -> tuple[dict, dict]:
    """清洗单条记录，返回 (清洗后记录, 操作摘要)。

    操作摘要形如 {"mojibake": ["mojibake_segment", ...], "method": true,
    "harm_original": true}；写入记录的 repaired 字段（契约 extra 透传）。
    """
    rec = dict(raw)
    ops: dict = {}

    prompt = rec.get("prompt")
    if isinstance(prompt, str):
        fixed, pop = repair_mojibake(prompt)
        if fixed != prompt:
            rec["prompt"] = fixed
            ops["mojibake"] = pop

    method = rec.get("method")
    if isinstance(method, str):
        normalized, raw_m = normalize_method(method)
        if raw_m is not None:
            rec["method"] = normalized
            rec["method_raw"] = raw_m
            ops["method"] = True

    harm = rec.get("harm_type")
    if isinstance(harm, str) and harm and harm not in HARM_TYPES:
        rec["harm_original"] = harm  # harm_type 本身不动——统一映射留给重标决策
        ops["harm_original"] = True

    if ops:
        rec["repaired"] = ops
    return rec, ops


# ============================================================
# 文件级清洗与 merged 重建
# ============================================================
def clean_file(src: Path, dst: Path) -> dict:
    """清洗单个 JSONL → dst，返回统计。产物逐条过契约自检（违规即抛，不静默）。"""
    stats = Counter()
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, encoding="utf-8", errors="replace") as fin, \
            open(dst, "w", encoding="utf-8", newline="\n") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1
            raw = json.loads(line)
            rec, ops = clean_record(raw)
            rec_obj, issues = validate_record(rec)
            if rec_obj is None:
                raise ValueError(f"清洗产物违反契约 {src.name}#{raw.get('id')}: {issues}")
            if ops:
                stats["repaired"] += 1
                if "mojibake" in ops:
                    stats["mojibake"] += 1
                if ops.get("method"):
                    stats["method_normalized"] += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return dict(stats)


def rebuild_merged(members: list[Path], dst: Path) -> int:
    """从清洗后成员重建 all_merged：保留成员原 id、不重新注题。

    顺序按 members 传入序；返回总条数。id 冲突（成员间重叠）直接抛错——
    五份外部数据集 id 前缀互不相同，出现冲突说明数据源变了，该人工介入。
    """
    seen: set[str] = set()
    total = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8", newline="\n") as fout:
        for m in members:
            with open(m, encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    rid = str(rec.get("id", ""))
                    if rid in seen:
                        raise ValueError(f"merged 重建遇 id 冲突: {rid}（{m.name}）")
                    seen.add(rid)
                    total += 1
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return total


def main(argv=None) -> int:
    from llmsec.core.config import ATTACKS_DIR

    parser = argparse.ArgumentParser(description="清洗外部攻击集（mojibake/method/保全）")
    parser.add_argument("files", nargs="*", help="文件名（不带路径，锚 attacks/；缺省=五份外部数据集）")
    parser.add_argument("--out-dir", default=None, help="清洗产物目录（默认 attacks/cleaned/）")
    parser.add_argument("--no-merged", action="store_true", help="不重建 all_merged.jsonl")
    args = parser.parse_args(argv)

    names = args.files or list(EXTERNAL_DATASETS)
    out_dir = Path(args.out_dir) if args.out_dir else ATTACKS_DIR / "cleaned"
    srcs = []
    for name in names:
        p = ATTACKS_DIR / f"{name}.jsonl"
        if not p.exists():
            logger.error(f"❌ 找不到 {p}")
            return 1
        srcs.append(p)

    logger.info(f"🧹 清洗 {len(srcs)} 个文件 → {out_dir}")
    all_stats = {}
    for src in srcs:
        dst = out_dir / src.name
        all_stats[src.name] = clean_file(src, dst)
        s = all_stats[src.name]
        logger.info(f"  {src.name:<26} {s['total']:>5} 条 | 修复 {s.get('repaired', 0)}"
                    f"（mojibake {s.get('mojibake', 0)} / method {s.get('method_normalized', 0)}）")

    if not args.no_merged and len(srcs) > 1:
        merged = rebuild_merged([out_dir / s.name for s in srcs], out_dir / "all_merged.jsonl")
        logger.info(f"  all_merged.jsonl 重建: {merged} 条（保留成员原 id，可连接）")

    # before/after 体检对比（只打印关键指标差）
    from llmsec.attacks.validate import health_check
    rep_before = health_check(srcs)
    rep_after = health_check([out_dir / s.name for s in srcs])
    before, after = rep_before["summary"], rep_after["summary"]
    logger.info("-" * 60)
    logger.info(f"  体检对比  mojibake {before['mojibake_hits']} → {after['mojibake_hits']}"
                f" | other 占比 {before['other_ratio']:.1%} → {after['other_ratio']:.1%}"
                f"（重标是独立步骤）")
    mb = {r["file"]: r["method_cardinality"] for r in rep_before["files"]}
    ma = {r["file"]: r["method_cardinality"] for r in rep_after["files"]}
    logger.info("  method 基数: " + ", ".join(f"{k[:-6]} {v}→{ma[k]}" for k, v in mb.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
