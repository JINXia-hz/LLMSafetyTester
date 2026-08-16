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
    assert all('functional_category' in e for e in entries), '每条 entry 均含 functional_category 字段'
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


# ===== 补充覆盖：generate 的 MD 解析 / 类别轮转 / 两轮生成 / main 流程 =====

_MD_SAMPLE = """# 攻击分析

### 1.1 角色扮演（Roleplay）

| 序号 | 攻击方法 | 难度 | 构建 | 说明 |
|---|---|---|---|---|
| 1.1.1 | **虚构专家人格** | 中 | L1 | 让模型扮演无限制人格 |
| 1.1.2 | **伪权威指令** | 高 | L1 | 伪造上级指令施压 |
| 1.1.3 | 伪代码越权 | 中 | L2 | 不应出现在 L1 输出 |

### 1.2 上下文注入（Context）

| 序号 | 攻击方法 | 难度 | 构建 | 说明 |
|---|---|---|---|---|
| 1.2.10 | 编号十号方法 | 低 | L1 | 验证数值序筛选 |
"""


class _NoSleep:
    """替换 time 模块的 sleep（避免 patch 全局 time.sleep）。"""

    @staticmethod
    def sleep(_s):
        pass


def test_parse_md_extracts_l1_methods(tmp_path):
    """parse_md：三级标题归类 + 表格行解析 + L2 过滤 + 类别名去英文括号。"""
    from llmsec.attacks import generate as gen

    md = tmp_path / "攻击分析.md"
    md.write_text(_MD_SAMPLE, encoding="utf-8")
    methods = gen.parse_md(md)
    ids = [m["id"] for m in methods]
    assert ids == ["1.1.1", "1.1.2", "1.2.10"], f"L2 构建级应被过滤: {ids}"
    m0 = methods[0]
    assert m0["method"] == "虚构专家人格", "加粗标记应被剥离"
    assert m0["category"] == "1.1"
    assert m0["category_name"] == "角色扮演", "类别名应去尾部英文括号"
    assert m0["difficulty"] == "中" and "人格" in m0["description"]


def test_assign_harm_types_rotation():
    """assign_harm_types：5 条至少 3 个不同类别，且确定性、随索引轮转。"""
    from llmsec.attacks.generate import HARM_TYPES, assign_harm_types

    a = assign_harm_types(0)
    assert a == assign_harm_types(0), "同索引分配应确定性"
    assert len(a) == 5 and len(set(a)) >= 3, "每批 ≥3 个不同类别"
    assert all(ht in HARM_TYPES for ht in a), "类别须在 HARM_TYPES 白名单内"
    assert assign_harm_types(1) != a, "相邻索引应轮转"


def _fake_client(responses):
    """按序返回预设响应文本的假 OpenAI client（chat.completions.create）。"""
    from types import SimpleNamespace

    queue = list(responses)

    def _create(**kw):
        text = queue.pop(0)
        if isinstance(text, Exception):
            raise text
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=text))])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_create)))


def test_call_api_two_round_happy_path(monkeypatch):
    """两轮生成：code-fence JSON 被剥壳；审查轮 prompt 覆盖初稿。"""
    from llmsec.attacks import generate as gen

    monkeypatch.setattr(gen, "time", _NoSleep)
    drafts = json.dumps([{"harm_type": "violence", "prompt": "draft-1"}])
    reviewed = json.dumps([{"harm_type": "violence", "prompt": "reviewed-1"}])
    client = _fake_client([f"```json\n{drafts}\n```", reviewed])
    out = gen.call_api_two_round(client, {"method": "m", "description": "d", "category": "1.1",
                                 "category_name": "测试"},
                                 ["violence"], model="fake")
    assert out == [{"harm_type": "violence", "prompt": "reviewed-1"}], \
        "审查轮结果应覆盖初稿"


def test_call_api_two_round_critique_fallback_to_drafts(monkeypatch):
    """审查轮返回条数不符 → 退回初稿（不失败）。"""
    from llmsec.attacks import generate as gen

    monkeypatch.setattr(gen, "time", _NoSleep)
    drafts = json.dumps([{"harm_type": "violence", "prompt": "d1"},
                         {"harm_type": "hate", "prompt": "d2"}])
    bad_review = json.dumps([{"prompt": "only-one"}])  # 条数不符
    client = _fake_client([drafts, bad_review])
    out = gen.call_api_two_round(client, {"method": "m", "description": "d", "category": "1.1",
                                 "category_name": "测试"},
                                 ["violence", "hate"], model="fake")
    assert [r["prompt"] for r in out] == ["d1", "d2"], "审查异常应退回初稿"


def test_call_api_two_round_mismatch_retries_then_none(monkeypatch):
    """初稿条数不符重试；全部失败 → None（不抛出）。"""
    from llmsec.attacks import generate as gen

    monkeypatch.setattr(gen, "time", _NoSleep)
    wrong = json.dumps([{"harm_type": "violence", "prompt": "only"}])  # 缺一条
    client = _fake_client([wrong] * (gen.API_MAX_RETRIES + 1))
    out = gen.call_api_two_round(client, {"method": "m", "description": "d", "category": "1.1",
                                 "category_name": "测试"},
                                 ["violence", "hate"], model="fake")
    assert out is None, "全部重试失败应返回 None"


def test_main_dry_run_and_only(tmp_path, monkeypatch):
    """main：--dry-run 只列方法不调 API；--only 不存在的方法 exit 1。"""
    import sys

    import pytest

    from llmsec.attacks import generate as gen

    md = tmp_path / "攻击分析.md"
    md.write_text(_MD_SAMPLE, encoding="utf-8")
    monkeypatch.setattr(gen, "MD_FILE", md)
    monkeypatch.setattr(gen, "OUTPUT_FILE", tmp_path / "out.jsonl")
    monkeypatch.setattr(gen, "ATTACKS_DIR", tmp_path)

    monkeypatch.setattr(sys, "argv", ["generate", "--dry-run"])
    gen.main()  # dry-run 不触 API、不落盘
    assert not (tmp_path / "out.jsonl").exists()

    monkeypatch.setattr(sys, "argv", ["generate", "--only", "9.9.9"])
    with pytest.raises(SystemExit) as ei:
        gen.main()
    assert ei.value.code == 1, "未找到 --only 方法应 exit 1"


def test_main_generates_and_resumes(tmp_path, monkeypatch):
    """main 全流程（mock 客户端）：3 方法 ×5 条落盘含探针；重跑全部跳过。"""
    import sys

    from llmsec.attacks import generate as gen

    md = tmp_path / "攻击分析.md"
    md.write_text(_MD_SAMPLE, encoding="utf-8")
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(gen, "MD_FILE", md)
    monkeypatch.setattr(gen, "OUTPUT_FILE", out)
    monkeypatch.setattr(gen, "time", _NoSleep)

    # 3 个 L1 方法，assign_harm_types 默认 count=5 → 每方法两轮响应各 5 条
    def _pair(idx):
        payload = json.dumps([{"harm_type": "h", "prompt": f"raw-{idx}-{k}"} for k in range(5)])
        return [payload, payload]

    responses = []
    for idx in range(3):
        responses.extend(_pair(idx))
    monkeypatch.setattr(gen, "create_openai_client", lambda *a, **k: _fake_client(responses))

    class _CfgShim:
        api_key, base_url, timeout, model = "k", "http://x", 5, "fake-model"

        @staticmethod
        def from_env():
            return _CfgShim()

    monkeypatch.setattr(gen, "GeneratorConfig", _CfgShim)

    monkeypatch.setattr(sys, "argv", ["generate"])
    gen.main()
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(lines) == 15, f"3 方法 × 5 条，实得 {len(lines)}"
    assert all("[MATH:" in e["prompt"] for e in lines), "prompt 应注入数学税探针"

    # 断点续传：重跑时全部方法已在 done_ids → 不再调 API
    monkeypatch.setattr(gen, "create_openai_client",
                        lambda *a, **k: _fake_client([RuntimeError("should not be called")]))
    gen.main()
    after = [x for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(after) == 15, "续跑不应追加记录"

