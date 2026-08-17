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
        _write_jsonl(out / 'runs' / '20260801_120000' / 't1' / 'attack_results.jsonl', _eval_records(5, 'run'))
        got = report.load_all_results(out)
        assert len(got) == 5, 'H9(b)：仅 run 目录时读 attack_results.jsonl（5 条）'
        assert all(r['method'].startswith('run_') for r in got), 'H9(b)：记录来自最新 run'
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        _write_jsonl(out / 'run1_结果.jsonl', _eval_records(3, 'eval'))
        _write_jsonl(out / 'runs' / '20260801_120000' / 't1' / 'attack_results.jsonl', _eval_records(5, 'run'))
        got = report.load_all_results(out)
        assert len(got) == 5, 'H9(c)：两来源并存时记录数不翻倍（5 条而非 8 条）'
        assert all(r['method'].startswith('run_') for r in got), 'H9(c)：优先选择 runner 来源'
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        _write_jsonl(out / 'runs' / '20260801_120000' / 't1' / 'attack_results.jsonl', _eval_records(4, 'run'))
        newer = out / 'runs' / '20260802_120000' / 'empty_target'
        newer.mkdir(parents=True)
        import os
        import time
        future = time.time() + 10
        os.utime(newer.parent, (future, future))
        got = report.load_all_results(out)
        assert len(got) == 4, 'H9：最新 run 缺 attack_results.jsonl 时取次新 run'

def test_f4_elo_tracker_from_R_not_state(tmp_path, monkeypatch):
    """F4：_load_elo_tracker 始终从 R 派生，不读 state.json 快照（tmp 隔离，不碰全局 R）。"""
    import llmsec.core.config as _results_cfg
    import llmsec.evaluation.elo_access as ea
    monkeypatch.setattr(_results_cfg, 'CATALOG_DB', tmp_path / 'catalog.db')
    monkeypatch.setattr(ea, 'active_model', lambda: 'model_r')
    # 写一个 state.json（哨兵值仅存于此），_load_elo_tracker 不应读它
    state_dir = tmp_path / 'state'
    state_dir.mkdir()
    (state_dir / 'state.json').write_text(json.dumps({
        'attacker_ratings': {'custom_probe_method': 1888.0},
        'defender_ratings': {'custom_target': 1666.0}, 'history': []},
    ), encoding='utf-8')

    # R 为空 → None（不再回退 state.json）
    assert report._load_elo_tracker() is None, 'F4：R 空应返回 None 而非读快照'

    # R 有数据 → 从 R 派生，值来自 R 而非 state.json
    from llmsec.core.results import ResultsMatrix

    R = ResultsMatrix()
    R.upsert('r1', 'model_r', 3.0, status='fully_compliant', ts=1)
    R.save()
    tracker = report._load_elo_tracker()
    assert isinstance(tracker, ELOTracker), 'F4：R 有数据应返回 tracker'
    assert tracker.attacker_ratings.get('custom_probe_method') != 1888.0, \
        'F4：不应从 state.json 读 Elo（1888 仅存在于 state.json）'

    # active_model 无值 → None（缺省调用不崩）
    monkeypatch.setattr(ea, 'active_model', lambda: None)
    assert report._load_elo_tracker() is None

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
        md = report.generate_narrative(tree)
        assert captured.get('timeout') == 123.0, 'M9：create_openai_client 收到 timeout=cfg.timeout（123.0）'
        assert isinstance(md, str) and md.startswith('# 报告'), 'M9：mock 链路下叙事报告正常返回'
    finally:
        report.create_openai_client = orig_client
        report.chat_with_retry = orig_chat
        report._report_config = orig_cfg


# ===== 补充覆盖：load_elo / load_allergy 分支 / fallback 报告 =====

def test_load_elo_prefers_active_model(monkeypatch):
    """load_elo：指定模型用指定列；缺省取 active_model；R 空 → {}。"""
    import llmsec.evaluation.elo_access as ea

    monkeypatch.setattr(ea, "active_model", lambda: "active-m")
    monkeypatch.setattr(ea, "attacker_ratings_for",
                        lambda m: {"DAN": 1511.0} if m == "active-m" else {"x": 1.0})
    assert report.load_elo() == {"DAN": 1511.0}, '缺省取 active_model 列'
    assert report.load_elo("other") == {"x": 1.0}, '显式模型覆盖缺省'

    monkeypatch.setattr(ea, "active_model", lambda: None)
    assert report.load_elo() == {}, 'R 空 → {}（F3：不回退 state.json）'


def _touch(path: Path, delay=0.0):
    import os
    import time

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    if delay:
        future = time.time() + delay
        os.utime(path, (future, future))


def test_load_allergy_model_files_and_run_fallback(tmp_path, monkeypatch):
    """load_allergy：优先按模型的 allergy__<model>.json；否则最新 mtime；再回退 run 产物。"""
    import llmsec.evaluation.elo_access as ea

    monkeypatch.setattr(ea, "active_model", lambda: "model/a")

    # 按模型分文件：活跃模型有专属文件时优先于更新的其它模型文件
    import os
    import time

    own = tmp_path / "allergy__model_a.json"
    other = tmp_path / "allergy__other.json"
    own.write_text(json.dumps({"summary": {"false_positive_rate": 0.1}}), encoding="utf-8")
    other.write_text(json.dumps({"summary": {"false_positive_rate": 0.9}}), encoding="utf-8")
    now = time.time()
    os.utime(own, (now, now))          # 较旧
    os.utime(other, (now + 5, now + 5))  # 较新（显式错开，避免同刻 mtime 排序不稳）
    got = report.load_allergy(tmp_path)
    assert got["summary"]["false_positive_rate"] == 0.1, '应优先取活跃模型自己的文件'

    # 无活跃模型匹配：取 mtime 最新的分文件
    monkeypatch.setattr(ea, "active_model", lambda: "ghost")
    got = report.load_allergy(tmp_path)
    assert got["summary"]["false_positive_rate"] == 0.9, '无匹配模型时取最新 mtime 文件'

    # 无分文件：回退 runs/<ts>/<target>/allergy.json
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        _write_jsonl(out / "runs" / "b1" / "t1" / "keep.jsonl", [])
        run_allergy = out / "runs" / "b1" / "t1" / "allergy.json"
        run_allergy.write_text(json.dumps({"summary": {"false_positive_rate": 0.5}}),
                               encoding="utf-8")
        got = report.load_allergy(out)
        assert got["summary"]["false_positive_rate"] == 0.5, '回退读 run 的 allergy.json'
        assert report.load_allergy(out / "nonexistent") == {}, '无任何产物 → {}'


def _fallback_tree(**overall_over):
    overall = {
        "security_level": "safe", "asr": 0.25, "fpr": 0.05,
        "elo_boundary": 1500.5, "elo_confidence": 0.9,
        "total_methods": 10, "total_tests": 42,
        "jailbreak_tax_mean": None, "jailbreak_tax": None,
    }
    overall.update(overall_over)
    return {
        "overall": overall,
        "dimensions": {"by_harm_type": {"fraud": {"label": "欺诈", "asr": 0.4, "count": 3}}},
        "top_threats": [{"method": "DAN", "elo": 1480.0, "asr": 0.8,
                          "surprise_score": 60, "weakness_count": 2, "mean_jailbreak_tax": 2.5}],
        "strong_defenses": [{"method": "hard", "elo": 1900.0, "asr": 0.1, "max_strength_gap": 400}],
        "upsets": {"weakness": [], "strength": []},
    }


def test_fallback_tax_line_three_modes():
    o = _fallback_tree()["overall"]
    assert "未测试" in report._fallback_tax_line(o), '无税数据 → 未测试'
    o2 = _fallback_tree(jailbreak_tax_mean=2.0)["overall"]
    assert "越狱税均值: 2.00" in report._fallback_tax_line(o2), '只有均值 → 无基线对照形式'
    o3 = _fallback_tree(jailbreak_tax={"baseline_accuracy": 0.9, "attack_accuracy": 0.6,
                                       "accuracy_drop": 0.3})["overall"]
    line = report._fallback_tax_line(o3)
    assert "基线正确率 90%" in line and "退化 30%" in line, '有基线 → 对比式呈现'


def test_generate_fallback_report_renders_sections():
    md = report.generate_fallback_report(_fallback_tree())
    assert "目标模型安全评估报告" in md and "SAFE" in md, '标题与安全等级'
    assert "ASR (攻击成功率): 25.0%" in md, 'ASR 百分比'
    assert "DAN" in md and "越狱税=2.50" in md, '高威胁方法含税值'
    assert "hard" in md and "max_strength_gap=400" in md, '防御强项段'
    assert "按有害类别" in md and "欺诈" in md, '维度分解（中文维度名映射）'

    md2 = report.generate_fallback_report(_fallback_tree(fpr=None))
    assert "未测（无有效过敏样本）" in md2, 'FPR=None → 未测文案'


def test_generate_narrative_empty_content_falls_back(monkeypatch):
    """LLM 返回空内容（content/reasoning 均空）→ 走 fallback 报告而非空字符串。"""
    from types import SimpleNamespace

    monkeypatch.setattr(report, "_report_config",
                        lambda: report.GeneratorConfig(api_key="k", base_url="http://x",
                                                        model="m", timeout=1.0))
    monkeypatch.setattr(report, "create_openai_client", lambda *a, **k: object())
    monkeypatch.setattr(report, "chat_with_retry",
                        lambda client, **kw: SimpleNamespace(choices=[
                            SimpleNamespace(message=SimpleNamespace(content=""))]))
    md = report.generate_narrative(_fallback_tree())
    assert "目标模型安全评估报告" in md, '空响应应退回 fallback 报告'

    # LLM 抛异常 → 同样退回 fallback
    def _boom(client, **kw):
        raise RuntimeError("api down")

    monkeypatch.setattr(report, "chat_with_retry", _boom)
    md2 = report.generate_narrative(_fallback_tree())
    assert "LLM生成失败" in md2, '异常应退回 fallback 报告'


def test_load_allergy_mtime_tie_deterministic(tmp_path, monkeypatch):
    """同刻 mtime 平局：按文件名次级裁决，选取确定（不依赖枚举顺序）。"""
    import os
    import time

    import llmsec.evaluation.elo_access as ea

    monkeypatch.setattr(ea, "active_model", lambda: None)
    a = tmp_path / "allergy__alpha.json"
    b = tmp_path / "allergy__beta.json"
    a.write_text(json.dumps({"summary": {"false_positive_rate": 0.1}}), encoding="utf-8")
    b.write_text(json.dumps({"summary": {"false_positive_rate": 0.9}}), encoding="utf-8")
    now = time.time()
    os.utime(a, (now, now))
    os.utime(b, (now, now))  # 完全同刻：排序只能靠 name 裁决

    for _ in range(3):  # 重复读取稳定
        got = report.load_allergy(tmp_path)
        assert got["summary"]["false_positive_rate"] == 0.9, \
            "平局应按文件名降序取 allergy__beta（确定性）"
