"""attacks 子命令组——外部攻击集的体检与导入通道（Step 2「收编」）。

对接约定（docs/攻击集导入.md）：外部产物不要求交出生成代码，走
`llmsec-manage attacks import` 进来即合规——契约校验 → source 登记 →
id 冲突检测（对 attacks/、attacks/cleaned/、attacks/imported/ 三个
id 空间）→ 落盘 attacks/imported/<source>.jsonl。写操作默认 dry-run。
"""
from __future__ import annotations

import json
from pathlib import Path

from llmsec.attacks.schema import SOURCES, validate_record
from llmsec.core.logging import get_logger

logger = get_logger(__name__)


def cmd_health(files: list[str] | None, json_mode: bool = False) -> int:
    """attacks health——包装体检校验器（契约/分布/乱码/重复）。"""
    from llmsec.attacks.validate import main as validate_main

    return validate_main(files)


def _existing_id_spaces(attacks_dir: Path) -> dict[str, str]:
    """收集三个 id 空间的现有 id → 所属文件名（冲突检测用）。"""
    seen: dict[str, str] = {}
    for sub in (attacks_dir, attacks_dir / "cleaned", attacks_dir / "imported"):
        if not sub.is_dir():
            continue
        for f in sorted(sub.glob("*.jsonl")):
            with open(f, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rid = rec.get("id") if isinstance(rec, dict) else None
                    if rid:
                        seen.setdefault(str(rid), f.name)
    return seen


def cmd_import(file: str, source: str, yes: bool = False, json_mode: bool = False) -> int:
    """attacks import——校验 + 登记 + 冲突检测 → attacks/imported/<source>.jsonl。

    默认 dry-run（只报告）；--yes 落盘。source 必须是 schema.SOURCES 之一
    （新产地先在契约里登记，避免 id 空间无主扩张）。
    """
    from llmsec.core.config import ATTACKS_DIR

    if source not in SOURCES:
        logger.error(f"❌ 未知 source {source!r}，可用: {list(SOURCES)}（新产地先登记到 schema.SOURCES）")
        return 1
    src = Path(file)
    if not src.exists():
        logger.error(f"❌ 文件不存在: {src}")
        return 1

    rows = []
    violations: list[str] = []
    with open(src, encoding="utf-8", errors="replace") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                violations.append(f"line {ln}: JSON 解析失败 ({e})")
                continue
            rec, issues = validate_record(raw, source=source)
            if issues:
                violations.append(f"line {ln} ({raw.get('id', '?')}): {'; '.join(issues)}")
            else:
                rows.append(rec.model_dump())

    if violations:
        logger.error(f"❌ 契约校验未过（{len(violations)} 处，前 5 条）：")
        for v in violations[:5]:
            logger.error(f"   {v}")
        logger.error("   修复后重试；契约见 docs/攻击集导入.md")
        return 1

    known = _existing_id_spaces(ATTACKS_DIR)
    conflicts = [(r["id"], known[r["id"]]) for r in rows if r["id"] in known]

    dst = ATTACKS_DIR / "imported" / f"{source}.jsonl"
    if json_mode:
        import json as _json
        print(_json.dumps({
            "file": str(src), "source": source, "records": len(rows),
            "conflicts": [{"id": i, "existing_in": w} for i, w in conflicts],
            "dry_run": not yes, "dest": str(dst),
        }, ensure_ascii=False))
    logger.info(f"📥 导入 {src.name} → {dst}")
    logger.info(f"   {len(rows)} 条 | source={source} | id 冲突 {len(conflicts)} 处")
    for i, w in conflicts[:5]:
        logger.info(f"     冲突: {i}（已存在于 {w}）")

    if conflicts:
        logger.error("❌ id 冲突：重命名 id 或确认数据源后重试（不覆盖既有空间）")
        return 1
    if not yes:
        logger.info("   dry-run（--yes 落盘）")
        return 0

    dst.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if dst.exists() else "w"  # 同 source 追加式累积，不整体覆盖
    if mode == "a":
        # 追加前重查目标文件内部的 id（known 只含首条同 id 归属）
        with open(dst, encoding="utf-8", errors="replace") as fh:
            existing = {str(json.loads(line)["id"]) for line in fh if line.strip()}
        dup = [r["id"] for r in rows if r["id"] in existing]
        if dup:
            logger.error(f"❌ 与 {dst.name} 已有记录重复: {dup[:5]}")
            return 1
    with open(dst, mode, encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"   ✅ 已写入 {dst}（{len(rows)} 条）")
    return 0
