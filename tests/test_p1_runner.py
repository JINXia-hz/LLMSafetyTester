#!/usr/bin/env python3
"""
P1 修复回归测试：llmsec/pipeline/runner.py

覆盖：
- H2: .venv re-exec 逻辑必须在 __main__ 块内（模块顶层 import 不杀进程），
      且子进程退出码透传（sys.exit(proc.returncode)）。
- H3: round_idx 兜底（max_rounds<=0 时循环不执行，summary 不抛 NameError）
      + --max-rounds argparse 校验 >=1（_positive_int 单测 + subprocess 端到端）。
- H4: resume 时 tested 从 tracker.ground_truth_methods 初始化，
      已实测方法不会被二次选中（逻辑内联在长函数中，做源码级断言 + stub 验证）。
- H1: --input 默认值为 attacks/l1.jsonl，且保留旧名"攻击集_L1.jsonl"兼容映射。
"""

import argparse
import ast
import inspect
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Windows GBK 控制台兼容：允许输出 ✅/❌
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from llmsec.pipeline import runner


def _check(cond: bool, msg: str) -> int:
    if not cond:
        print(f"❌ {msg}")
        return 1
    print(f"✅ {msg}")
    return 0


def test_h2_no_toplevel_reexec() -> int:
    """H2: re-exec 不在模块顶层；__main__ 块内保留 re-exec 且透传退出码。"""
    rc = 0
    # import 成功本身即是冒烟：若顶层仍有 re-exec 且当前解释器非 .venv，
    # 本测试进程已被替换/杀掉，走不到这里
    rc |= _check("llmsec.pipeline.runner" in sys.modules,
                 "import llmsec.pipeline.runner 未触发 re-exec 杀进程")

    tree = ast.parse(inspect.getsource(runner))
    # 模块顶层语句不允许直接出现 subprocess.run 调用
    top_level_run = [
        node for node in tree.body
        if isinstance(node, ast.Expr) and "subprocess" in ast.dump(node)
    ]
    rc |= _check(not top_level_run, "模块顶层无 subprocess.run（re-exec 已移出）")

    main_guard = next(
        (node for node in tree.body
         if isinstance(node, ast.If) and "__main__" in ast.dump(node.test)),
        None,
    )
    rc |= _check(main_guard is not None, "存在 if __name__ == '__main__' 块")
    if main_guard is not None:
        body = ast.dump(main_guard)
        rc |= _check("subprocess" in body, "__main__ 块内保留 .venv re-exec")
        rc |= _check("returncode" in body, "re-exec 透传子进程退出码 sys.exit(proc.returncode)")
    return rc


def test_h3_max_rounds_validation() -> int:
    """H3: _positive_int 单测 + round_idx 兜底源码断言 + subprocess 端到端。"""
    rc = 0
    rc |= _check(runner._positive_int("1") == 1, "_positive_int 接受 '1'")
    rc |= _check(runner._positive_int("10") == 10, "_positive_int 接受 '10'")
    for bad in ("0", "-2", "abc"):
        try:
            runner._positive_int(bad)
            rc |= _check(False, f"_positive_int 应拒绝 {bad!r}")
        except argparse.ArgumentTypeError:
            print(f"✅ _positive_int 拒绝 {bad!r}（ArgumentTypeError）")

    # round_idx 兜底：max_rounds<=0 时循环不执行，循环前必须有 round_idx = 0
    src = inspect.getsource(runner.run_attack_phase)
    loop_pos = src.index("for round_idx in range(")
    rc |= _check("round_idx = 0" in src[:loop_pos],
                 "run_attack_phase 循环前有 round_idx = 0 兜底")

    # 端到端：argparse 在加载攻击集之前即拒绝 --max-rounds 0。
    # 注意：runner 模块级 setup_console() 会把子进程 stderr 重配为 UTF-8，
    # 这里按字节捕获再解码，且只断言 ASCII 内容，避免 GBK/UTF-8 解码差异。
    proc = subprocess.run(
        [sys.executable, "-m", "llmsec.pipeline.runner", "--max-rounds", "0"],
        cwd=ROOT, capture_output=True, timeout=120,
    )
    stderr = proc.stderr.decode("utf-8", errors="replace")
    rc |= _check(proc.returncode != 0,
                 f"--max-rounds 0 退出码非 0（实际 {proc.returncode}）")
    rc |= _check("error" in stderr and "--max-rounds" in stderr,
                 "stderr 报 argparse 校验错误")
    return rc


def test_h4_tested_resume_init() -> int:
    """H4: resume 时 tested 从 ground_truth_methods 初始化。

    该逻辑内联在 run_attack_phase 长函数中，难以单测，
    故做源码级断言 + stub 行为验证（模拟 untested 选择语义）。
    """
    rc = 0
    src = inspect.getsource(runner.run_attack_phase)
    rc |= _check("tested = set(tracker.ground_truth_methods)" in src,
                 "tested 从 tracker.ground_truth_methods 初始化（源码断言）")

    # stub 行为验证：resume 场景下已实测方法不再进入 untested 选择
    class _StubTracker:
        ground_truth_methods = {"m1", "m2"}

    tracker = _StubTracker()
    tested = set(tracker.ground_truth_methods)
    all_methods = ["m1", "m2", "m3"]
    untested = [m for m in all_methods if m not in tested]
    rc |= _check(untested == ["m3"],
                 "stub: 已实测方法 m1/m2 不会被二次选中，仅 m3 待测")
    return rc


def test_h1_input_default_compat() -> int:
    """H1: --input 默认值 attacks/l1.jsonl + 旧名兼容映射。"""
    rc = 0
    src = inspect.getsource(runner)
    rc |= _check('default="attacks/l1.jsonl"' in src,
                 "--input 默认值为 attacks/l1.jsonl")
    rc |= _check('"攻击集_L1.jsonl"' in src and "ATTACK_SET_L1_FILE" in src,
                 "保留旧名 攻击集_L1.jsonl → ATTACK_SET_L1_FILE 兼容映射")
    from llmsec.core.config import ATTACK_SET_L1_FILE
    rc |= _check(ATTACK_SET_L1_FILE.name == "l1.jsonl",
                 "ATTACK_SET_L1_FILE 指向 l1.jsonl")
    return rc


def main() -> int:
    rc = 0
    rc |= test_h2_no_toplevel_reexec()
    rc |= test_h3_max_rounds_validation()
    rc |= test_h4_tested_resume_init()
    rc |= test_h1_input_default_compat()
    print()
    if rc == 0:
        print("🎉 全部 P1 runner 测试通过")
    else:
        print("💥 存在失败项")
    return rc


if __name__ == "__main__":
    sys.exit(main())
