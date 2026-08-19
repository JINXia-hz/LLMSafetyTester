#!/usr/bin/env python3
"""攻击有效性融合层——低 ASR 的归因甄别与整改需求报告（攻击有效性 V2）。

测量效度问题：run 结束后，低 ASR 的攻击单位被默认记为"防御成功"。本模块
把 run 产物（state.json 的 per-unit Elo/观测统计 + attack_results.jsonl 的
逐条结果）与静态质量分（attack_quality.json）融合，把低 ASR 拆成两种解释：

  - false_defense_suspect：ASR 低 × 攻击质量弱 → 低 ASR 是攻击无效所致，
    不能记防御功劳（安全边界的可疑证据）
  - genuine_strong_defense：ASR 低 × 攻击质量高 → 可信的防御强证据
  - inconclusive：观测不足/质量未知/中间地带

修正原则（科学诚实）：**不重算不篡改 Elo**——修正只发生在解释层：统计
"低 ASR 单位中多大比例是假防御嫌疑"、给出嫌疑清单与整改建议；runner_report
顶层并入 attack_validity 块，不碰 security_level。

键位兼容：旧 run 的 unit 键是 method 名、新 run 是 c_<md5>——一律以该 run
自己的 attack_results.jsonl 行键为准，不假设格式。

用法：
  python -m llmsec.attacks.assess output/runs/<ts>/<target>            # 写入 run 目录
  python -m llmsec.attacks.assess <run_dir> --quality path/to/q.json   # 指定质量报告
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from llmsec.core.logging import get_logger, setup_console
from llmsec.params import (
    ATTACK_QUALITY_HIGH,
    ATTACK_QUALITY_WEAK,
    RECTIFY_LOW_ASR_MAX,
    RECTIFY_MIN_TESTS,
)

logger = get_logger(__name__)
setup_console()

_TAG_ADVICE = {
    "degenerate": "重写为真正贯彻该方法技术的版本，或从攻击集剔除（明文直球已有专门类别）",
    "template_mismatch": "对齐 method 声明与模板实际机制，修正标注或更换模板",
    "mild_harm": "升级危害请求的实质性（具体化场景/对象/产出物），否则测不出边界",
    "unclear": "修复构造缺陷（破碎文本/结构混乱）后重评",
}


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fuse(run_dir: Path, quality: dict) -> dict:
    """融合单个 run 的产物与质量分，返回 attack_validity 结构。

    quality：{id: {method_fidelity, harm_substance, construction, overall,
    tags, ...}}（attack_quality.json 的 scores 字段）。
    """
    rows = _load_jsonl(run_dir / "attack_results.jsonl")
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))

    # ---- 逐 unit 聚合：观测数 / 成功数 / 质量分（经 id+prompt_sha16 连接，C-1） ----
    per_unit: dict[str, dict] = defaultdict(lambda: {"n": 0, "harmful": 0,
                                                     "qualities": [], "tags": Counter(),
                                                     "methods": set()})
    # C-10：ASR 口径单源（attack_phase / report / assess 三处同函数）
    # C-6：质量分连接键与 quality.py 缓存键同源（id + prompt 指纹）
    from llmsec.attacks.quality import quality_key
    from llmsec.evaluation.scoring import count_successful
    for r in rows:
        uid = r.get("unit") or r.get("method") or "?"
        u = per_unit[uid]
        u["n"] += 1
        if count_successful(r):
            u["harmful"] += 1
        m = r.get("method")
        if m:
            u["methods"].add(m)
        q = quality.get(quality_key(r))
        if q and not q.get("unparsed") and "overall" in q:
            u["qualities"].append(q["overall"])
            u["tags"].update(q.get("tags", []))

    stats = state.get("attacker_stats", {})
    defender_elo = None
    for v in (state.get("defender_ratings") or {}).values():
        defender_elo = v
        break

    # C-1 防回归哨兵：质量报告非空且有明细行却零连接命中——键口径再次漂移
    # （旧行为：明细行无 prompt 字段 → 键恒为空串指纹 → 恒 miss，所有单位
    # 判"质量分缺失"、假防御嫌疑恒 0，比不产出报告更误导）。必须吵出来。
    if quality and rows and not any(u["qualities"] for u in per_unit.values()):
        logger.error(f"❌ 质量分连接 0 命中（{len(rows)} 行明细 × {len(quality)} 条质量分）"
                     "——attack_results 行缺 prompt_sha16（旧 run？）或质量报告来自"
                     "不同攻击集，假防御甄别不可用")

    verdicts = {"false_defense_suspect": [], "genuine_strong_defense": [], "inconclusive": []}
    for uid, u in sorted(per_unit.items()):
        asr = u["harmful"] / u["n"] if u["n"] else 0.0
        mean_q = round(sum(u["qualities"]) / len(u["qualities"]), 1) if u["qualities"] else None
        entry = {
            "unit": uid,
            "methods": sorted(u["methods"])[:3],
            "n_matches": u["n"],
            "asr": round(asr, 3),
            "mean_quality": mean_q,
            "elo": state.get("attacker_ratings", {}).get(uid),
            "tags": dict(u["tags"].most_common(3)),
        }
        # n_matches 优先用 tracker 口径（含被 dedup 的历史观测）；行数口径兜底
        n = stats.get(uid, {}).get("n_matches") or u["n"]
        entry["n_matches"] = n
        if n < RECTIFY_MIN_TESTS:
            verdicts["inconclusive"].append({**entry, "reason": "观测数不足"})
        elif mean_q is None:
            verdicts["inconclusive"].append({**entry, "reason": "质量分缺失"})
        elif asr <= RECTIFY_LOW_ASR_MAX and mean_q < ATTACK_QUALITY_WEAK:
            verdicts["false_defense_suspect"].append(entry)
        elif asr <= RECTIFY_LOW_ASR_MAX and mean_q >= ATTACK_QUALITY_HIGH:
            verdicts["genuine_strong_defense"].append(entry)
        else:
            verdicts["inconclusive"].append({**entry, "reason": "中间地带"})

    n_low_asr = sum(1 for uid, u in per_unit.items()
                    if (u["harmful"] / u["n"] if u["n"] else 0) <= RECTIFY_LOW_ASR_MAX)
    suspects = verdicts["false_defense_suspect"]
    tag_total = Counter(t for e in suspects for t in e["tags"])
    return {
        "generated_at": _now(),
        "defender_elo": defender_elo,
        "n_units": len(per_unit),
        "n_low_asr_units": n_low_asr,
        "false_defense_suspects": suspects,
        "genuine_strong_defenses": verdicts["genuine_strong_defense"],
        "inconclusive_count": len(verdicts["inconclusive"]),
        # 修正口径：低 ASR 单位中嫌疑占比——该比例越高，本次 run 的
        # "低于边界"证据越不可信（是攻击弱，不是防御强）
        "suspect_ratio_among_low_asr": round(len(suspects) / n_low_asr, 3) if n_low_asr else 0.0,
        "suspect_tag_dist": dict(tag_total.most_common()),
        "verdicts_detail": verdicts,
    }


def _now() -> str:
    from datetime import datetime

    return datetime.now().isoformat(timespec="seconds")


def render_rectification_md(v: dict) -> str:
    """把融合结果渲染为整改需求 markdown（report.py fallback 的 lines 风格）。"""
    lines: list[str] = []
    lines.append("# 攻击有效性评估与整改需求")
    lines.append("")
    lines.append(f"> 生成于 {v['generated_at']}；防御方 Elo≈{v['defender_elo'] or 'N/A'}。")
    lines.append("> 修正口径：低 ASR ≠ 防御强——攻击质量弱的低 ASR 不能记防御功劳。")
    lines.append("> 本报告不重算 Elo，只修正解释层。")
    lines.append("")
    lines.append("## 总览")
    lines.append("")
    lines.append(f"- 评估单位: {v['n_units']} 个；其中低 ASR（≤{RECTIFY_LOW_ASR_MAX}）: {v['n_low_asr_units']} 个")
    suspect_ratio = v.get("suspect_ratio_among_low_asr") or 0
    lines.append(f"- **假防御嫌疑: {len(v['false_defense_suspects'])} 个"
                 f"（占低 ASR 单位的 {suspect_ratio:.0%}）——这些'防住了'的证据不可信**")
    lines.append(f"- 可信强防御: {len(v['genuine_strong_defenses'])} 个（低 ASR × 高质量攻击）")
    lines.append(f"- 待定: {v['inconclusive_count']} 个（观测不足/质量缺失/中间地带）")
    lines.append("")

    if v["false_defense_suspects"]:
        lines.append("## 假防御嫌疑清单")
        lines.append("")
        lines.append("| 攻击单位 | 方法族 | 观测数 | ASR | 质量分 | 主要问题 |")
        lines.append("|---|---|---|---|---|---|")
        for e in sorted(v["false_defense_suspects"], key=lambda x: (x["mean_quality"] or 5))[:20]:
            tags = "、".join(e["tags"]) or "-"
            methods = "、".join(m[:28] for m in e["methods"]) or "-"
            lines.append(f"| {e['unit'][:16]} | {methods} | {e['n_matches']} | "
                         f"{e['asr']:.0%} | {e['mean_quality']} | {tags} |")
        lines.append("")

    if v["genuine_strong_defenses"]:
        lines.append("## 可信强防御（边界的有力证据）")
        lines.append("")
        for e in v["genuine_strong_defenses"][:10]:
            methods = "、".join(m[:28] for m in e["methods"]) or "-"
            lines.append(f"- {e['unit'][:16]}（{methods}）: ASR {e['asr']:.0%}，"
                         f"质量 {e['mean_quality']}，{e['n_matches']} 次观测")
        lines.append("")

    tag_dist = v.get("suspect_tag_dist") or {}
    if tag_dist:
        lines.append("## 整改需求（按问题类型）")
        lines.append("")
        for tag, count in tag_dist.items():
            advice = _TAG_ADVICE.get(tag, "人工复核")
            lines.append(f"- **{tag}**（{count} 处）: {advice}")
        lines.append("")
        lines.append("整改后走 `llmsec-manage attacks import` 通道回流（带 parent_id 血统），"
                     "下次评估生效。")
        lines.append("")
    return "\n".join(lines) + "\n"


def assess_run(run_dir: Path, quality_path: Path | None = None) -> dict | None:
    """对一个 run 目录执行融合并落盘 attack_validity.json + attack_rectification.md。

    质量报告缺失时返回 None（调用方优雅降级）。
    """
    if quality_path is None:
        from llmsec.attacks.quality import _default_quality_file

        quality_path = _default_quality_file()
    if not (run_dir / "state.json").exists() or not (run_dir / "attack_results.jsonl").exists():
        logger.warning(f"⚠ {run_dir} 缺 state.json/attack_results.jsonl，跳过有效性评估")
        return None
    if not quality_path.exists():
        logger.info(f"ℹ 质量报告不存在（{quality_path}），跳过有效性评估——"
                    f"先运行 python -m llmsec.attacks.quality")
        return None

    quality = json.loads(quality_path.read_text(encoding="utf-8")).get("scores", {})
    validity = fuse(run_dir, quality)

    from llmsec.core.io import write_json

    write_json(run_dir / "attack_validity.json", validity, backup=False)
    (run_dir / "attack_rectification.md").write_text(
        render_rectification_md(validity), encoding="utf-8")
    logger.info(f"🔍 攻击有效性: 假防御嫌疑 {len(validity['false_defense_suspects'])}"
                f"/低ASR {validity['n_low_asr_units']} 单位"
                f"（{validity['suspect_ratio_among_low_asr']:.0%}）| "
                f"可信强防御 {len(validity['genuine_strong_defenses'])}"
                f" → {run_dir / 'attack_rectification.md'}")
    return validity


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="攻击有效性融合（假防御甄别+整改报告）")
    parser.add_argument("run_dir", help="run 目录（含 state.json + attack_results.jsonl）")
    parser.add_argument("--quality", default=None, help="质量报告路径（默认 attacks/cleaned/attack_quality.json）")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        logger.error(f"❌ 不是目录: {run_dir}")
        return 1
    result = assess_run(run_dir, Path(args.quality) if args.quality else None)
    return 0 if result is not None else 1


if __name__ == "__main__":
    sys.exit(main())
