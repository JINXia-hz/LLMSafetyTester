"""
回归测试：report.py 的 P1 修复（H9 / H10 / M9）。

验证：
1. H9 load_all_results 两来源互斥：
   (a) 只有 *_结果.jsonl        → 读 evaluator 数据
   (b) 只有 runs/<ts>/attack_results.jsonl → 读最新 run 数据
   (c) 两者皆有                 → 只用最新 run 数据，记录数不翻倍
2. F4 _load_elo_tracker：始终从 R 派生（不再读 state.json），R 空时返回 None。
3. M9 generate_narrative 创建 OpenAI 客户端时透传 timeout=cfg.timeout。
"""
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
from llmsec.evaluation.elo import ELOTracker
from llmsec.reporting import report


def _write_jsonl(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

def _eval_records(n: int, tag: str):
    return [{'method': f'{tag}_method_{i}', 'is_harmful': i % 2 == 0} for i in range(n)]

def test_h9_mutual_exclusion():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        _write_jsonl(out / 'run1_结果.jsonl', _eval_records(3, 'eval'))
        got = report.load_all_results(out)
        assert len(got) == 3, 'H9(a)：仅 *_结果.jsonl 时读 evaluator 数据（3 条）'
        assert all(r['method'].startswith('eval_') for r in got), 'H9(a)：记录来自 evaluator 文件'
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        _write_jsonl(out / 'runs' / '20260801_120000' / 'attack_results.jsonl', _eval_records(5, 'run'))
        got = report.load_all_results(out)
        assert len(got) == 5, 'H9(b)：仅 run 目录时读 attack_results.jsonl（5 条）'
        assert all(r['method'].startswith('run_') for r in got), 'H9(b)：记录来自最新 run'
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        _write_jsonl(out / 'run1_结果.jsonl', _eval_records(3, 'eval'))
        _write_jsonl(out / 'runs' / '20260801_120000' / 'attack_results.jsonl', _eval_records(5, 'run'))
        got = report.load_all_results(out)
        assert len(got) == 5, 'H9(c)：两来源并存时记录数不翻倍（5 条而非 8 条）'
        assert all(r['method'].startswith('run_') for r in got), 'H9(c)：优先选择 runner 来源'
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        _write_jsonl(out / 'runs' / '20260801_120000' / 'attack_results.jsonl', _eval_records(4, 'run'))
        newer = out / 'runs' / '20260802_120000'
        newer.mkdir(parents=True)
        import os
        import time
        future = time.time() + 10
        os.utime(newer, (future, future))
        got = report.load_all_results(out)
        assert len(got) == 4, 'H9：最新 run 缺 attack_results.jsonl 时取次新 run'

def test_f4_elo_tracker_from_R_not_state():
    """F4：_load_elo_tracker 始终从 R 派生，不读 state.json 快照。"""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        state_dir = out / 'state'
        state_dir.mkdir(parents=True)
        # 写一个 state.json 但 _load_elo_tracker 不应读它
        with open(state_dir / 'state.json', 'w', encoding='utf-8') as f:
            json.dump({'attacker_ratings': {'custom_probe_method': 1888.0}, 'defender_ratings': {'custom_target': 1666.0}, 'history': []}, f, ensure_ascii=False)
        tracker = report._load_elo_tracker(out)
        # 不管全局 R 是否有数据，tracker 都不应包含 state.json 的 custom_probe_method
        if tracker is not None:
            assert tracker.attacker_ratings.get('custom_probe_method') != 1888.0, \
                'F4：不应从 state.json 读 Elo（custom_probe_method=1888 仅存在于 state.json）'

    # 无参数调用也不崩
    tracker = report._load_elo_tracker()
    assert tracker is None or isinstance(tracker, ELOTracker), 'F4：缺省调用不崩'

def test_m9_timeout_passthrough():
    captured = {}

    class _FakeMessage:
        content = '# 报告\n内容'

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeResponse:
        choices = [_FakeChoice()]
    orig_client, orig_chat = (report.create_openai_client, report.chat_with_retry)
    orig_cfg = report._report_config
    try:

        def fake_client(api_key, base_url, **kwargs):
            captured.update(kwargs)
            return object()
        report.create_openai_client = fake_client
        report.chat_with_retry = lambda client, **kw: _FakeResponse()
        report._report_config = lambda: report.GeneratorConfig(api_key='k', base_url='http://x', model='m', timeout=123.0)
        tree = {'overall': {'asr': 0.1, 'fpr': 0.0, 'elo_boundary': 1500, 'elo_confidence': 0, 'security_level': 'safe', 'total_methods': 1, 'total_tests': 2, 'jailbreak_tax_mean': None, 'jailbreak_tax': None}, 'dimensions': {}, 'top_threats': [], 'strong_defenses': [], 'upsets': {'weakness': [], 'strength': []}}
        md = report.generate_narrative(tree, ROOT / 'output')
        assert captured.get('timeout') == 123.0, 'M9：create_openai_client 收到 timeout=cfg.timeout（123.0）'
        assert isinstance(md, str) and md.startswith('# 报告'), 'M9：mock 链路下叙事报告正常返回'
    finally:
        report.create_openai_client = orig_client
        report.chat_with_retry = orig_chat
        report._report_config = orig_cfg
