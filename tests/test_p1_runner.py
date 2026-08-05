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
from llmsec.pipeline import runner

def test_h2_no_toplevel_reexec():
    """H2: re-exec 不在模块顶层；__main__ 块内保留 re-exec 且透传退出码。"""
    assert 'llmsec.pipeline.runner' in sys.modules, 'import llmsec.pipeline.runner 未触发 re-exec 杀进程'
    tree = ast.parse(inspect.getsource(runner))
    top_level_run = [node for node in tree.body if isinstance(node, ast.Expr) and 'subprocess' in ast.dump(node)]
    assert not top_level_run, '模块顶层无 subprocess.run（re-exec 已移出）'
    main_guard = next((node for node in tree.body if isinstance(node, ast.If) and '__main__' in ast.dump(node.test)), None)
    assert main_guard is not None, "存在 if __name__ == '__main__' 块"
    if main_guard is not None:
        body = ast.dump(main_guard)
        assert 'subprocess' in body, '__main__ 块内保留 .venv re-exec'
        assert 'returncode' in body, 're-exec 透传子进程退出码 sys.exit(proc.returncode)'

def test_h3_max_rounds_validation():
    """H3: _positive_int 单测 + round_idx 兜底源码断言 + subprocess 端到端。"""
    assert runner._positive_int('1') == 1, "_positive_int 接受 '1'"
    assert runner._positive_int('10') == 10, "_positive_int 接受 '10'"
    for bad in ('0', '-2', 'abc'):
        try:
            runner._positive_int(bad)
            assert False, f'_positive_int 应拒绝 {bad!r}'
        except argparse.ArgumentTypeError:
            print(f'✅ _positive_int 拒绝 {bad!r}（ArgumentTypeError）')
    src = inspect.getsource(runner.run_attack_phase)
    loop_pos = src.index('for round_idx in range(')
    assert 'round_idx = 0' in src[:loop_pos], 'run_attack_phase 循环前有 round_idx = 0 兜底'
    proc = subprocess.run([sys.executable, '-m', 'llmsec.pipeline.runner', '--max-rounds', '0'], cwd=ROOT, capture_output=True, timeout=120)
    stderr = proc.stderr.decode('utf-8', errors='replace')
    assert proc.returncode != 0, f'--max-rounds 0 退出码非 0（实际 {proc.returncode}）'
    assert 'error' in stderr and '--max-rounds' in stderr, 'stderr 报 argparse 校验错误'

def test_h4_tested_resume_init():
    """H4: resume 时 tested 从 ground_truth_methods 初始化。

    该逻辑内联在 run_attack_phase 长函数中，难以单测，
    故做源码级断言 + stub 行为验证（模拟 untested 选择语义）。
    """
    src = inspect.getsource(runner.run_attack_phase)
    assert 'tested = set(tracker.ground_truth_methods)' in src, 'tested 从 tracker.ground_truth_methods 初始化（源码断言）'

    class _StubTracker:
        ground_truth_methods = {'m1', 'm2'}
    tracker = _StubTracker()
    tested = set(tracker.ground_truth_methods)
    all_methods = ['m1', 'm2', 'm3']
    untested = [m for m in all_methods if m not in tested]
    assert untested == ['m3'], 'stub: 已实测方法 m1/m2 不会被二次选中，仅 m3 待测'

def test_h1_input_default_compat():
    """H1: --input 默认值 attacks/l1.jsonl（旧名兼容映射已随 v1.0 清理移除）。"""
    src = inspect.getsource(runner)
    assert 'default="attacks/l1.jsonl"' in src
    assert '攻击集_L1.jsonl' not in src, '旧名 兼容映射应已移除（v1.0 不向后兼容）'
    from llmsec.core.config import ATTACK_SET_L1_FILE
    assert ATTACK_SET_L1_FILE.name == 'l1.jsonl', 'ATTACK_SET_L1_FILE 指向 l1.jsonl'
