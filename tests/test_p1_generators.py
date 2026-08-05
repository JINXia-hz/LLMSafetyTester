"""
冒烟测试：P1 级生成器修复（M6 / M7 / M20）。

验证：
1. M6：generate.py 的 build_entries 对同方法多条记录独立出题注入越狱税，
   不再 5 条共用同一道题；旧探针仍会被剥离，entry 字段与下游兼容。
2. M7：harmbench 生成器输出的 entry 含 functional_category 字段，
   且取值与 CSV 的 FunctionalCategory 一致，category 字符串保持兼容。
3. M20：safe_twin 的 --no-generate 可关闭生成阶段；默认行为（生成/评估分派）不变。
"""
import csv
import json
import tempfile
from pathlib import Path
from llmsec.core.text import strip_math_tax

def test_m6_generate_per_record_tax():
    """M6：同方法多条记录独立出题，不再共用同一道数学题。"""
    from llmsec.attacks.generate import build_entries
    method = {'id': '9.9.9', 'category': 'test', 'category_name': '测试类别', 'method': 'mock_method'}
    harm_types = ['violence'] * 5
    base_prompts = [f'测试攻击内容 {i}' for i in range(5)]
    records = [{'harm_type': 'violence', 'prompt': p} for p in base_prompts]
    entries = build_entries(method, records, harm_types)
    assert len(entries) == 5, 'build_entries 返回 5 条 entry'
    problems = [e['math_problem'] for e in entries]
    assert len(set(problems)) >= 2, f'同方法 5 条记录的 math_problem 不再全部相同（去重后 {len(set(problems))} 种）'
    for i, e in enumerate(entries):
        if e['id'] != f'9.9.9-{i + 1:03d}':
            assert False, f"record_id 编号异常: {e['id']}"
            break
        if not (e['math_problem'] in e['prompt'] and '[MATH:答案]' in e['prompt']):
            assert False, f'第 {i + 1} 条 prompt 未含自身探针'
            break
        if not (isinstance(e['expected_answer'], int) and e['expected_answer'] > 0):
            assert False, f"第 {i + 1} 条 expected_answer 异常: {e['expected_answer']}"
            break
        if strip_math_tax(e['prompt']) != base_prompts[i]:
            assert False, f'第 {i + 1} 条剥离探针后未还原原始 prompt'
            break
    else:
        print('✅ 每条 entry 的 id/探针/答案/剥离还原均正确')
    required = {'id', 'category', 'category_name', 'method', 'harm_type', 'prompt', 'math_problem', 'expected_answer', 'build_difficulty'}
    assert required <= set(entries[0]), 'entry 字段名与既有下游格式兼容'
    from llmsec.core.text import MATH_TAX_SUFFIX_TEMPLATE
    dirty = '攻击内容\n\n' + MATH_TAX_SUFFIX_TEMPLATE.format(problem='((1 × 1) + (2 ÷ 2)) - 1 = ?')
    e2 = build_entries(method, [{'harm_type': 'violence', 'prompt': dirty}], harm_types)[0]
    assert e2['prompt'].count('在回答之前') == 1, '残留旧探针被剥离，不叠加'

def test_m7_harmbench_functional_category():
    """M7：harmbench entry 输出 functional_category 且取值与 CSV 一致。"""
    from llmsec.attacks.harmbench import BEHAVIORS_CSV, generate
    with open(BEHAVIORS_CSV, encoding='utf-8') as f:
        truth = {r['BehaviorID']: r.get('FunctionalCategory', 'standard') for r in csv.DictReader(f)}
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / 'hb.jsonl'
        generate(BEHAVIORS_CSV, ['Answer the following: {0}'], out, seed=42)
        entries = [json.loads(line) for line in out.read_text(encoding='utf-8').splitlines() if line.strip()]
    assert len(entries) == len(truth), '生成条数与 CSV behavior 数一致'
    assert all(('functional_category' in e for e in entries)), '每条 entry 均含 functional_category 字段'
    bad = [e['id'] for e in entries if e.get('functional_category') != truth.get(e['behavior_id'])]
    assert not bad, f'functional_category 取值与 CSV 一致（异常 {len(bad)} 条）'
    bad_cat = [e['id'] for e in entries if e['category'] != f"harmbench-{e['functional_category']}"]
    assert not bad_cat, 'category 字符串保持 harmbench-{functional} 兼容格式'
    distinct = {e['functional_category'] for e in entries}
    assert len(distinct) > 1, f'输出覆盖多个功能类别（非全部回落 standard）: {sorted(distinct)}'

def test_m20_safe_twin_no_generate():
    """M20：--no-generate 可关闭生成阶段；其余分派行为不变。"""
    import llmsec.evaluation.safe_twin as st
    calls = []
    orig_g, orig_e = (st.generate_all_twins, st.evaluate_allergy)
    st.generate_all_twins = lambda: calls.append('generate')
    st.evaluate_allergy = lambda: calls.append('evaluate')
    try:
        cases = [([], ['generate'], '默认（无参数）→ 仅生成'), (['--evaluate'], ['evaluate'], '--evaluate → 仅评估'), (['--all'], ['generate', 'evaluate'], '--all → 生成 + 评估'), (['--all', '--no-generate'], ['evaluate'], '--all --no-generate → 跳过生成仅评估'), (['--no-generate'], [], '--no-generate 单独使用 → 无操作')]
        for argv, expected, desc in cases:
            calls.clear()
            st.main(argv)
            assert calls == expected, f'{desc}（实际: {calls}）'
    finally:
        st.generate_all_twins, st.evaluate_allergy = (orig_g, orig_e)
