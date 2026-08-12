#!/usr/bin/env python3
"""一次性数据修复脚本：修正攻击集中的历史遗留问题。

C2: example.jsonl 中 12 条重复 prompt（同 method 不同混淆变体产生相同文本）→ 去重保留首个
C3: example.jsonl / harmbench_ensemble.jsonl 中负值 expected_answer → 取绝对值
    （gen_math 保证 answer > 0，负值来自旧版生成器；LLM 回答 [MATH:-107] 不合直觉）

用法:
    python scripts/fix_attack_data.py          # 预览（dry-run）
    python scripts/fix_attack_data.py --apply  # 实际写入
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ATTACKS = ROOT / "attacks"


def fix_duplicates(rows: list[dict]) -> list[dict]:
    """C2: 按 (method, prompt 完整文本) 去重，保留首个变体。

    只有 method + prompt 完全相同时才视为重复（同一 method 的不同混淆变体
    本应有不同 prompt 文本；重复说明生成器产出了完全相同的行）。
    """
    seen = set()
    out = []
    removed = 0
    for r in rows:
        key = (r.get("method", ""), r.get("prompt", ""))
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        out.append(r)
    return out, removed


def fix_negative_answers(rows: list[dict]) -> list[dict]:
    """C3: expected_answer 负值 → 绝对值。"""
    fixed = 0
    for r in rows:
        ea = r.get("expected_answer", 0)
        if isinstance(ea, (int, float)) and ea < 0:
            r["expected_answer"] = abs(int(ea))
            fixed += 1
    return rows, fixed


def process_file(path: Path, apply: bool = False) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    original_count = len(rows)

    dups_fixed = 0
    negs_fixed = 0

    if path.name == "example.jsonl":
        rows, dups_fixed = fix_duplicates(rows)

    if path.name in ("example.jsonl", "harmbench_ensemble.jsonl"):
        rows, negs_fixed = fix_negative_answers(rows)

    result = {
        "file": path.name,
        "original": original_count,
        "after": len(rows),
        "dups_removed": dups_fixed,
        "negs_fixed": negs_fixed,
    }

    if apply and (dups_fixed or negs_fixed):
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )
        result["written"] = True
    else:
        result["written"] = False

    return result


def main():
    apply = "--apply" in sys.argv

    print(f"{'=== APPLY MODE ===' if apply else '=== DRY RUN (use --apply to write) ==='}")
    print()

    targets = [ATTACKS / "example.jsonl", ATTACKS / "harmbench_ensemble.jsonl"]
    for p in targets:
        if not p.exists():
            print(f"⚠️  {p.name}: 文件不存在，跳过")
            continue
        r = process_file(p, apply=apply)
        status = "✅ 已写入" if r["written"] else "📋 预览"
        print(f"{status} {r['file']}:")
        print(f"  原始行数: {r['original']}")
        print(f"  去重移除: {r['dups_removed']}")
        print(f"  负值修正: {r['negs_fixed']}")
        print(f"  修正后行数: {r['after']}")
        print()


if __name__ == "__main__":
    main()
