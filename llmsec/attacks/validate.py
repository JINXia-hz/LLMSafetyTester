#!/usr/bin/env python3
"""攻击集体检校验器——把"质量参差不齐"变成量化事实（Step 1「契约+体检」）。

用法：
  python -m llmsec.attacks.validate              # 体检 attacks/ 下全部 JSONL
  python -m llmsec.attacks.validate a.jsonl b.jsonl --out report.json

逐文件报告：schema 违规（AttackRecord 契约口径）/ harm_type 分布与 other
占比 / mojibake 特征命中 / 文件内重复 prompt 与重复 id / method-category
基数 / source 分布；汇总另含跨文件重复组（all_merged.jsonl 是合并视图，
与其成员碰撞属预期，报告中有标注）。

定位是体检不是门禁：exit 恒 0（除非文件不可读），违规只进报告——
清洗与拦截依体检数据在 Step 2 立项。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from llmsec.attacks.schema import HARM_TYPES, SOURCES, detect_mojibake, infer_source, validate_record
from llmsec.core.logging import get_logger, setup_console

logger = get_logger(__name__)
setup_console()

_SAMPLE_CAP = 5          # 各类问题的样例 id 上限（报告可读性）
_TOP_N = 5               # method/category top-N
_CROSS_FILE_GROUP_CAP = 10


def _digest(prompt: str) -> str:
    return hashlib.sha1(prompt.strip().encode("utf-8")).hexdigest()


def check_file(path: Path) -> dict:
    """体检单个 JSONL 文件，返回单文件报告 dict。"""
    rel = path.name
    harm_dist: Counter[str] = Counter()
    source_dist: Counter[str] = Counter()
    error_kinds: Counter[str] = Counter()
    error_samples: list[dict] = []
    mojibake_chars: Counter[str] = Counter()
    mojibake_ids: list[str] = []
    prompt_hashes: dict[str, str] = {}   # digest -> 首个 id
    dup_prompt_ids: list[str] = []
    seen_ids: dict[str, int] = Counter()
    dup_id_list: list[str] = []
    method_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()

    total = valid = bad_json = hashed = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                error_kinds["<line>: JSON 解析失败"] += 1
                continue
            if not isinstance(raw, dict):
                error_kinds["<line>: 非对象行"] += 1
                continue

            rec_id = str(raw.get("id", "?"))
            rec, issues = validate_record(raw, source=infer_source(rel, raw.get("source")))
            if rec is not None:
                valid += 1
                for issue in issues:  # warn 级（如血统缺失）同样计数
                    error_kinds[issue] += 1
                harm_dist[rec.harm_type] += 1
                source_dist[rec.source] += 1
                method_counts[rec.method] += 1
                if rec.category:
                    category_counts[rec.category] += 1
                hits = detect_mojibake(rec.prompt)
                if hits:
                    for ch in hits:
                        mojibake_chars[ch] += 1
                    if len(mojibake_ids) < _SAMPLE_CAP:
                        mojibake_ids.append(rec_id)
                d = _digest(rec.prompt)
                hashed += 1
                if d in prompt_hashes:
                    if len(dup_prompt_ids) < _SAMPLE_CAP:
                        dup_prompt_ids.append(rec_id)
                else:
                    prompt_hashes[d] = rec_id
            else:
                for issue in issues:
                    error_kinds[issue] += 1
                if len(error_samples) < _SAMPLE_CAP:
                    error_samples.append({"id": rec_id, "issues": issues})
            seen_ids[rec_id] += 1
            if seen_ids[rec_id] == 2 and len(dup_id_list) < _SAMPLE_CAP:
                dup_id_list.append(rec_id)

    dup_id_count = sum(c - 1 for c in seen_ids.values() if c > 1)
    other_n = harm_dist.get("other", 0)
    unknown_harm = {k: v for k, v in harm_dist.items() if k not in HARM_TYPES}
    return {
        "file": rel,
        "total": total,
        "valid": valid,
        "bad_json_lines": bad_json,
        "errors": dict(error_kinds.most_common()),
        "error_samples": error_samples,
        "harm_dist": dict(harm_dist.most_common()),
        "other_ratio": round(other_n / valid, 4) if valid else 0.0,
        "unknown_harm_types": unknown_harm,
        "source_dist": {k: v for k, v in source_dist.most_common() if k in SOURCES} | {
            k: v for k, v in source_dist.items() if k not in SOURCES
        },
        "mojibake": {"count": sum(mojibake_chars.values()), "chars": dict(mojibake_chars),
                     "sample_ids": mojibake_ids},
        "dup_prompt": {"count": hashed - len(prompt_hashes),
                       "sample_ids": dup_prompt_ids},
        "dup_id": {"count": dup_id_count, "sample_ids": dup_id_list},
        "method_cardinality": len(method_counts),
        "category_cardinality": len(category_counts),
        "top_methods": method_counts.most_common(_TOP_N),
        "top_categories": category_counts.most_common(_TOP_N),
    }


def health_check(files: list[Path]) -> dict:
    """体检一组文件：逐文件报告 + 跨文件重复组汇总。"""
    file_reports = [check_file(p) for p in files]

    # 跨文件重复：digest -> {files, sample_id}（all_merged 是合并视图，碰撞属预期）
    cross: dict[str, dict] = {}
    for p, _rep in zip(files, file_reports):
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(raw, dict) and raw.get("prompt"):
                    d = _digest(str(raw["prompt"]))
                    slot = cross.setdefault(d, {"files": set(), "sample_id": str(raw.get("id", "?"))})
                    slot["files"].add(p.name)
    cross_groups = [
        {"files": sorted(v["files"]), "sample_id": v["sample_id"]}
        for v in cross.values() if len(v["files"]) >= 2
    ]
    cross_groups.sort(key=lambda g: (-len(g["files"]), g["sample_id"]))

    total_records = sum(r["total"] for r in file_reports)
    total_valid = sum(r["valid"] for r in file_reports)
    return {
        "files": file_reports,
        "summary": {
            "file_count": len(file_reports),
            "total_records": total_records,
            "valid_records": total_valid,
            "invalid_records": total_records - total_valid,
            "other_ratio": round(
                sum(r["harm_dist"].get("other", 0) for r in file_reports) / total_valid, 4
            ) if total_valid else 0.0,
            "mojibake_hits": sum(r["mojibake"]["count"] for r in file_reports),
            "note_cross_file_dup": (
                "精确哈希口径。实测 all_merged.jsonl 与成员文件零碰撞：它在合并时"
                "重排了 id 并重新注入数学税（prompt 全文已变），并非成员文件的"
                "逐字拷贝——Step 2 清洗时以 prompt 语义（聚类层）对齐，勿依赖 id"
            ),
            "cross_file_dup_groups": cross_groups[:_CROSS_FILE_GROUP_CAP],
            "cross_file_dup_group_count": len(cross_groups),
        },
    }


def _print_report(report: dict) -> None:
    s = report["summary"]
    logger.info("=" * 78)
    logger.info(f"🩺 攻击集体检：{s['file_count']} 个文件 / {s['total_records']} 条记录")
    logger.info(f"   契约有效 {s['valid_records']} | 违规 {s['invalid_records']} | "
                f"other 危害占比 {s['other_ratio']:.1%} | mojibake 命中 {s['mojibake_hits']} 条")
    logger.info("-" * 78)
    for r in report["files"]:
        flags = []
        if r["total"] - r["valid"]:
            flags.append(f"违规 {r['total'] - r['valid']}")
        if r["mojibake"]["count"]:
            flags.append(f"乱码 {r['mojibake']['count']}")
        if r["dup_prompt"]["count"]:
            flags.append(f"重复prompt {r['dup_prompt']['count']}")
        if r["dup_id"]["count"]:
            flags.append(f"重复id {r['dup_id']['count']}")
        top_harm = ", ".join(f"{k}:{v}" for k, v in list(r["harm_dist"].items())[:3])
        logger.info(f"  {r['file']:<28} {r['total']:>5} 条 | method {r['method_cardinality']:>3} 种 | "
                    f"other {r['other_ratio']:.0%} | {top_harm}"
                    + (f" | ⚠ {'; '.join(flags)}" if flags else ""))
    logger.info("-" * 78)
    logger.info(f"   跨文件重复组 {s['cross_file_dup_group_count']} 个"
                f"（{s['note_cross_file_dup']}）")
    logger.info("=" * 78)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="攻击集体检（契约校验/分布/乱码/重复）")
    parser.add_argument("files", nargs="*", help="JSONL 文件（缺省= attacks/ 下全部）")
    parser.add_argument("--out", default=None, help="报告 JSON 输出路径（默认 output/attack_set_health.json）")
    args = parser.parse_args(argv)

    if args.files:
        files = [Path(p) for p in args.files]
    else:
        from llmsec.core.config import ATTACKS_DIR
        files = sorted(ATTACKS_DIR.glob("*.jsonl"))
    files = [p for p in files if p.exists()]
    if not files:
        logger.error("❌ 未找到任何待体检文件")
        return 1

    report = health_check(files)
    _print_report(report)

    if args.out:
        out = Path(args.out)
    else:
        from llmsec.core.config import OUTPUT_DIR
        out = OUTPUT_DIR / "attack_set_health.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"📄 明细报告: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
