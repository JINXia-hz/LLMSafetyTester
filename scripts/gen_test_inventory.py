#!/usr/bin/env python3
"""生成 tests/INVENTORY.md —— 测试文件 × 收集用例数清单。

用法：
  python scripts/gen_test_inventory.py           # 生成/刷新 tests/INVENTORY.md
  python scripts/gen_test_inventory.py --check   # CI 用：与已提交版本不一致则 exit 1

数据来自 `pytest --collect-only -q`（清空 addopts，含 real_api/e2e 标记的用例），
比 grep "def test_" 准确（覆盖 parametrize 展开、类方法、fixture 参数化）。
tests/README.md 的分组表不再硬编码用例数，以本文件为准。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "INVENTORY.md"


def collect_counts() -> Counter[str]:
    """跑 pytest --collect-only，返回 {测试文件名: 用例数}。"""
    # -o addopts=''：清空 pyproject 的 -n 4 与 -m 过滤——收集要覆盖全量用例
    # （含默认排除的 real_api/e2e），且收集无需并行
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-o", "addopts=", "--collect-only", "-q",
         "tests/"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if proc.returncode not in (0, 5):  # 5 = 收集到 0 个用例（空目录场景）
        print(proc.stdout[-2000:], file=sys.stderr)
        print(proc.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"pytest --collect-only 失败（exit {proc.returncode}）")
    counts: Counter[str] = Counter()
    total = 0
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("tests/") or "::" not in line:
            continue
        fname = line.split("::", 1)[0]
        counts[fname.replace("\\", "/")] += 1
        total += 1
    if total == 0:
        raise SystemExit("未收集到任何用例（collect 输出异常）")
    return counts


def render(counts: Counter[str]) -> str:
    lines = [
        "# 测试清单（自动生成，勿手改）",
        "",
        f"> 由 `scripts/gen_test_inventory.py` 生成于 {datetime.now().isoformat(timespec='seconds')}；",
        "> CI 会校验本文件与实际收集结果一致（`--check`），过期即失败。",
        "> 本地刷新：`python scripts/gen_test_inventory.py`。",
        "",
        f"合计 **{len(counts)}** 个文件 / **{sum(counts.values())}** 个用例（含 parametrize 展开；",
        "含默认排除的 real_api/e2e 用例——它们需手动 `pytest -m real_api` / `-m e2e` 触发）。",
        "",
        "| 测试文件 | 用例数 |",
        "|---|---|",
    ]
    for fname in sorted(counts):
        lines.append(f"| {fname} | {counts[fname]} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="不写盘：与现有 INVENTORY.md 不一致时 exit 1（CI 用）")
    args = ap.parse_args()

    content = render(collect_counts())
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        # 比较时忽略首部时间戳行（每次生成都变，不参与过期判定）
        def _strip_ts(s: str) -> list[str]:
            return [ln for ln in s.splitlines() if not ln.startswith("> 由 `scripts/")]
        if _strip_ts(current) == _strip_ts(content):
            print("INVENTORY.md 与实际收集一致")
            return 0
        print("INVENTORY.md 过期：与 pytest 实际收集结果不一致，请运行 "
              "`python scripts/gen_test_inventory.py` 并提交", file=sys.stderr)
        return 1
    OUT.write_text(content, encoding="utf-8")
    print(f"已生成 {OUT.relative_to(ROOT)}（{sum(1 for _ in open(OUT, encoding='utf-8'))} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
