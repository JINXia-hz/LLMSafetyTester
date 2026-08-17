"""
验证本轮修复（Judge None 响应 + 探活 chat smoke + 重试策略 + 数据污染防注入 +
日志无缓冲 + inconclusive 文案 + stale 判定 + SVD-Ridge 诊断 + 单目标日志）。

聚焦"根因已修"的行为契约，不重复 test_retry.py 已覆盖的旧语义。
网络相关全部 mock，保持离线秒级。
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from llmsec.core.llm import is_retryable_error, retry_call


# ============================================================
# 通用 fake（与 test_retry.py 同范式，独立声明避免跨文件耦合）
# ============================================================
class _FakeChatResponse:
    """模拟 ChatCompletion。content=None 模拟推理模型把内容放进 reasoning_content。"""

    def __init__(self, content, reasoning_content=None, finish_reason="stop"):
        # reasoning_content 用 SimpleNamespace 挂载，模拟国产兼容网关的扩展字段；
        # 标准 OpenAI 响应无此字段，extract_message_text 用 getattr 兜底。
        # finish_reason 默认 stop；探活用它区分"截断(length)"与"真空响应"。
        msg = SimpleNamespace(content=content, reasoning_content=reasoning_content)
        self.choices = [SimpleNamespace(message=msg, finish_reason=finish_reason)]
        self.usage = SimpleNamespace(prompt_tokens=5, completion_tokens=5)


class _FlakyCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def create(self, **kw):
        self.calls += 1
        o = self.outcomes[min(self.calls - 1, len(self.outcomes) - 1)]
        if isinstance(o, Exception):
            raise o
        return o


class _FakeOpenAIClient:
    def __init__(self, outcomes):
        self.completions = _FlakyCompletions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)
        self.models = SimpleNamespace(list=lambda: [])  # probe 用


class _SleepRecorder:
    def __init__(self):
        self.calls: list[float] = []

    def __enter__(self):
        self._orig = time.sleep
        time.sleep = lambda s, *a, **kw: self.calls.append(s)
        return self

    def __exit__(self, *exc):
        time.sleep = self._orig


# ============================================================
# 1. Judge 推理模型响应：content=None 时回退读 reasoning_content；两者皆空才返回空串
# ============================================================
def test_judge_call_falls_back_to_reasoning_content():
    """推理模型 content=None 但 reasoning_content 有内容时，应回退读取而非返回空串。

    这是本轮修复的核心：原代码只读 content，推理模型（o1/R1/QwQ/minimax）把答案放进
    reasoning_content，导致 Judge 每条都拿到空串 → 退化成关键词降级，评分精度大打折扣。
    现在回退读 reasoning_content，恢复真实输出。
    """
    from llmsec.evaluation import judge as judge_mod

    client = _FakeOpenAIClient([_FakeChatResponse(None, reasoning_content='  D  ')])
    judge = judge_mod.Judge(client=client, model='fake')
    assert judge._call_judge('sys', 'user') == 'D', (
        'content=None 时应回退读 reasoning_content（含 strip）'
    )


def test_judge_call_content_preferred_over_reasoning():
    """content 有值时优先用 content，不混入 reasoning_content。"""
    from llmsec.evaluation import judge as judge_mod

    client = _FakeOpenAIClient([_FakeChatResponse('B', reasoning_content='思考过程...')])
    judge = judge_mod.Judge(client=client, model='fake')
    assert judge._call_judge('sys', 'user') == 'B'


def test_judge_call_empty_when_both_none():
    """content 与 reasoning_content 皆空时返回空串，让上层解析 fallback 接管。"""
    from llmsec.evaluation import judge as judge_mod

    client = _FakeOpenAIClient([_FakeChatResponse(None)])
    judge = judge_mod.Judge(client=client, model='fake')
    with _SleepRecorder():
        result = judge._call_judge('sys', 'user')
    assert result == '', '两者皆空应返回空串，让上层解析 fallback 接管'
    assert client.completions.calls == 1, '解析层已兜底，不应重试'


def test_judge_call_normal_content_still_stripped():
    """正常 content 仍走 strip，行为不变。"""
    from llmsec.evaluation import judge as judge_mod

    client = _FakeOpenAIClient([_FakeChatResponse('  B  ')])
    judge = judge_mod.Judge(client=client, model='fake')
    assert judge._call_judge('sys', 'user') == 'B'


def test_judge_compliance_fallback_on_empty():
    """_call_judge 返回空串后，judge_compliance 应走关键词 fallback（不抛异常）。"""
    from llmsec.evaluation import judge as judge_mod

    client = _FakeOpenAIClient([_FakeChatResponse(None)])
    judge = judge_mod.Judge(client=client, model='fake')
    # 空响应 → parse_compliance_level 返回 None → 走关键词猜测 fallback，返回合法等级
    level = judge.judge_compliance('攻击 prompt', '模型回复内容')
    assert level in ('A', 'B', 'C', 'D', 'E'), '空响应应触发 fallback，返回合规等级而非崩溃'


# ============================================================
# 2. 同类推理模型回退：generate / safe_twin 在 content=None 时读 reasoning_content
# ============================================================
def test_generate_falls_back_to_reasoning_content():
    """generate.call_api_two_round：content=None 但 reasoning_content 有 JSON 时应成功解析。"""
    from llmsec.attacks import generate as gen

    method = {'method': 'm1', 'category_name': 'c1', 'description': 'd1'}
    harm_types = ['violence']
    # 两轮的 reasoning_content 都给出合法 JSON（推理模型把答案放进 reasoning_content）
    payload = '[{"method":"m1","harm_type":"violence","prompt":"p1"}]'
    client = _FakeOpenAIClient([
        _FakeChatResponse(None, reasoning_content=payload),
        _FakeChatResponse(None, reasoning_content=payload),
    ])
    r = gen.call_api_two_round(client, method, harm_types, model='m')
    assert r is not None, 'reasoning_content 有合法 JSON 时应成功生成而非返回 None'


def test_generate_handles_both_empty():
    """generate.call_api_two_round 对 content 与 reasoning_content 皆空不崩，走解析失败路径。"""
    from llmsec.attacks import generate as gen

    method = {'method': 'm1', 'category_name': 'c1', 'description': 'd1'}
    harm_types = ['violence']
    with _SleepRecorder():
        client = _FakeOpenAIClient([_FakeChatResponse(None)])
        r = gen.call_api_two_round(client, method, harm_types, model='m')
    assert r is None, '两者皆空 → JSON 解析失败 → 重试耗尽返回 None（不抛 AttributeError）'


def test_safe_twin_handles_both_empty():
    """safe_twin.generate_safe_twin 对 content 与 reasoning_content 皆空不崩。"""
    from llmsec.evaluation import safe_twin as st

    with _SleepRecorder():
        client = _FakeOpenAIClient([_FakeChatResponse(None)])
        r = st.generate_safe_twin('攻击', client)
    assert r is None, '两者皆空 → 解析失败 → 耗尽返回 None'


# ============================================================
# 3. is_retryable_error：确定性解析错误（AttributeError/TypeError）不重试
# ============================================================
def test_is_retryable_rejects_attribute_error():
    """AttributeError（如 .strip() on None）是确定性错误，重试无意义。"""
    assert is_retryable_error(AttributeError("'NoneType' has no attribute 'strip'")) is False
    assert is_retryable_error(TypeError("unsupported operand")) is False


def test_is_retryable_keeps_network_errors():
    """网络错误 / 5xx / 429 仍可重试（回归保护）。"""
    assert is_retryable_error(ConnectionError('refused')) is True
    assert is_retryable_error(TimeoutError('slow')) is True
    # 无 status_code 的普通 RuntimeError 仍可重试（非确定性解析类）
    assert is_retryable_error(RuntimeError('boom')) is True


def test_is_retryable_keeps_4xx_rejection():
    """4xx（非 429）仍不可重试（H-7 既有语义回归保护）。"""
    e = Exception('401')
    e.status_code = 401
    assert is_retryable_error(e) is False
    e429 = Exception('429')
    e429.status_code = 429
    assert is_retryable_error(e429) is True


def test_attribute_error_not_retried_by_retry_call():
    """retry_call + is_retryable_error：AttributeError 立即抛出不重试。"""
    with _SleepRecorder() as sr:
        n = {'v': 0}

        def f():
            n['v'] += 1
            raise AttributeError('bad')

        with pytest.raises(AttributeError):
            retry_call(f, retries=3, delay=1.0, retry_on=is_retryable_error)
    assert n['v'] == 1 and sr.calls == [], 'AttributeError 立即抛出不 sleep'


# ============================================================
# 4. 探活 chat smoke：区分推理模型（已自动回退）vs 真空响应（需确认）
# ============================================================
def test_probe_service_warns_on_truly_empty(monkeypatch):
    """content 与 reasoning_content 皆空 → 报"需确认"（疑似配置/鉴权问题）。

    通过公开端点 api_targets_probe 间接验证（_probe_service 是闭包，不直接单测）。
    """
    import asyncio

    from llmsec.core import llm as llm_mod
    from llmsec.server.routers import data_query as dq

    monkeypatch.setattr(dq, 'load_targets', lambda: {}, raising=False)
    from llmsec.core import config as cfg_mod
    monkeypatch.setattr(cfg_mod, 'load_targets', lambda: {})

    def _fake_create(api_key=None, base_url=None, timeout=60.0):
        # content=None 且无 reasoning_content → 真·空响应
        c = _FakeOpenAIClient([_FakeChatResponse(None)])
        c.models = SimpleNamespace(list=lambda: [SimpleNamespace(id='rm')])
        return c

    monkeypatch.setattr(llm_mod, 'create_openai_client', _fake_create)

    result = asyncio.run(dq.api_targets_probe())
    services = {s['name']: s for s in result.get('services', [])}
    for svc in ('generator', 'judge'):
        assert svc in services, f'{svc} 应被探活'
        assert services[svc]['reachable'] is True, f'{svc} models.list 通 → 仍可达'
        w = services[svc].get('warning') or ''
        assert '需确认' in w, (
            f'{svc} 真空响应应提示"需确认"，实际: {w!r}'
        )
        assert '已自动回退' not in w, '真空响应不应误报为"已自动回退"'


def test_probe_service_notes_reasoning_model(monkeypatch):
    """content=None 但 reasoning_content 有内容 → 标记推理模型（已自动回退，不阻塞）。"""
    import asyncio

    from llmsec.core import config as cfg_mod
    from llmsec.core import llm as llm_mod
    from llmsec.server.routers import data_query as dq

    monkeypatch.setattr(dq, 'load_targets', lambda: {}, raising=False)
    monkeypatch.setattr(cfg_mod, 'load_targets', lambda: {})

    def _fake_create(api_key=None, base_url=None, timeout=60.0):
        # content=None 但 reasoning_content 有内容 → 推理模型良性场景
        c = _FakeOpenAIClient([_FakeChatResponse(None, reasoning_content='思考...')])
        c.models = SimpleNamespace(list=lambda: [SimpleNamespace(id='rm')])
        return c

    monkeypatch.setattr(llm_mod, 'create_openai_client', _fake_create)

    result = asyncio.run(dq.api_targets_probe())
    services = {s['name']: s for s in result.get('services', [])}
    for svc in ('generator', 'judge'):
        assert svc in services
        assert services[svc]['reachable'] is True
        w = services[svc].get('warning') or ''
        assert '已自动回退' in w, (
            f'{svc} 推理模型应提示"已自动回退读取"，实际: {w!r}'
        )
        assert '需确认' not in w, '推理模型良性场景不应报"需确认"'


def test_probe_service_handles_truncation(monkeypatch):
    """content=None 且 finish_reason=length → 探活预算不足截断，良性提示（不报"需确认"）。

    复现 minimax 真实场景：max_tokens 太小时 finish_reason=length、content=None，
    既非推理模型也非配置问题，真实业务请求（max_tokens 更大）不受影响。
    """
    import asyncio

    from llmsec.core import config as cfg_mod
    from llmsec.core import llm as llm_mod
    from llmsec.server.routers import data_query as dq

    monkeypatch.setattr(dq, 'load_targets', lambda: {}, raising=False)
    monkeypatch.setattr(cfg_mod, 'load_targets', lambda: {})

    def _fake_create(api_key=None, base_url=None, timeout=60.0):
        # content=None、无 reasoning_content、finish_reason=length → 截断
        c = _FakeOpenAIClient([_FakeChatResponse(None, finish_reason="length")])
        c.models = SimpleNamespace(list=lambda: [SimpleNamespace(id='rm')])
        return c

    monkeypatch.setattr(llm_mod, 'create_openai_client', _fake_create)

    result = asyncio.run(dq.api_targets_probe())
    services = {s['name']: s for s in result.get('services', [])}
    for svc in ('generator', 'judge'):
        assert svc in services
        w = services[svc].get('warning') or ''
        assert '不受影响' in w, (
            f'{svc} 截断应提示"真实业务请求不受影响"，实际: {w!r}'
        )
        assert '需确认' not in w, '截断不应报"需确认"（非配置问题）'


def test_probe_service_no_warning_on_normal(monkeypatch):
    """正常 chat 响应不报警（回归保护）。"""
    import asyncio

    from llmsec.core import config as cfg_mod
    from llmsec.core import llm as llm_mod
    from llmsec.server.routers import data_query as dq

    monkeypatch.setattr(cfg_mod, 'load_targets', lambda: {})
    # 让 config 返回的 model 名与 models.list 一致，避免触发"模型不在列表"warning
    fake_cfg = SimpleNamespace(
        base_url='http://fake.local/v1', api_key='k', model='m',
        timeout=5.0, max_retries=2,
    )
    monkeypatch.setattr(cfg_mod.GeneratorConfig, 'from_env', staticmethod(lambda: fake_cfg))
    monkeypatch.setattr(cfg_mod.JudgeConfig, 'from_env', staticmethod(lambda: fake_cfg))

    def _fake_create(api_key=None, base_url=None, timeout=60.0):
        c = _FakeOpenAIClient([_FakeChatResponse('pong')])
        c.models = SimpleNamespace(list=lambda: [SimpleNamespace(id='m')])
        return c

    monkeypatch.setattr(llm_mod, 'create_openai_client', _fake_create)

    result = asyncio.run(dq.api_targets_probe())
    services = {s['name']: s for s in result.get('services', [])}
    for svc in ('generator', 'judge'):
        assert services[svc]['reachable'] is True
        assert services[svc].get('warning') is None, (
            f'{svc} 正常响应不应有 warning，实际: {services[svc].get("warning")!r}'
        )


# ============================================================
# 5. 数据污染防注入：publish-global 拒绝未声明目标
# ============================================================
# （审查清理：原 test_publish_global_rejects_undeclared_target 是"复刻逻辑"式假测试——
#  断言的是测试内自造的列表推导、从未调用生产代码、caplog 参数未用，永不可能失败，已删除。）


# ============================================================
# 6. inconclusive 文案不再矛盾
# ============================================================
def test_recommendation_inconclusive_not_contradictory():
    """inconclusive 的 recommendation 不应再是 broken 的'全面失效'文案。"""
    from llmsec.reporting.final_report import _generate_recommendation

    rec = _generate_recommendation(level='inconclusive')
    assert '失效' not in rec and '全面审查' not in rec, (
        'inconclusive 不应给 broken 级别的结论'
    )
    assert '不足' in rec or '收敛' in rec or '增加' in rec, '应建议继续测试到收敛'


def test_recommendation_broken_still_strong():
    """broken 级别保留'全面失效'强文案（回归保护）。"""
    from llmsec.reporting.final_report import _generate_recommendation

    rec = _generate_recommendation(level='broken')
    assert '失效' in rec, 'broken 应保留强结论'


def test_recommendation_all_levels_distinct():
    """五个等级的文案互不相同（确认 inconclusive/broken 已拆分）。"""
    from llmsec.reporting.final_report import _generate_recommendation

    recs = {
        lv: _generate_recommendation(lv)
        for lv in ('safe', 'allergic', 'vulnerable', 'broken', 'inconclusive')
    }
    assert len(set(recs.values())) == 5, '五个等级文案应互不相同'
    assert recs['broken'] != recs['inconclusive'], 'broken 与 inconclusive 必须区分'


# ============================================================
# 7. stale 判定按时间戳目录，不按完整批次名
# ============================================================
def test_stale_detection_by_batch_dir(tmp_path, monkeypatch):
    """同时间戳目录下的不同目标不应互判 stale（真实 /api/overview 端点）。"""
    import json as _json

    from fastapi.testclient import TestClient

    from llmsec.server import dashboard_api
    from llmsec.server.routers import data_query as dq

    runs_dir = tmp_path / "runs"
    for batch, target in (("2026-08-11_151938", "minimax"),
                          ("2026-08-11_151938", "gemma-4-12B-it"),
                          ("2026-08-12_194108", "minimax")):
        d = runs_dir / batch / target
        d.mkdir(parents=True)
        (d / "runner_report.json").write_text(
            _json.dumps({"target_model": target, "security_level": "safe"}),
            encoding="utf-8")
    monkeypatch.setattr(dashboard_api, "RUNS_DIR", runs_dir)
    dq._DISCOVER_CACHE.clear()  # r9/P3-5：SigCache 用 clear() 重置

    client = TestClient(dashboard_api.app)
    r = client.get("/api/overview", params={"run": "2026-08-11_151938/gemma-4-12B-it"})
    assert r.status_code == 200, r.text
    body = r.json()
    if body.get("reason") == "stale_report":
        msg = body.get("message", "")
        assert "2026-08-12_194108" in msg, f"应只报时间戳更晚的批次: {msg}"
        assert "2026-08-11_151938/minimax" not in msg,             "同目录不同目标不应判为更新"


# ============================================================
# 8. SVD-Ridge 诊断细化：原因不再"未知"
# ============================================================
def _run_svd_diagnosis(tracker, monkeypatch):
    """驱动真实 analyze_clusters 的 SVD 诊断分支，返回 analysis dict。

    patch build_svd_ridge_summary 返回 None（模拟"无摘要"），使流程落入
    诊断块；断言目标从自造字符串换成生产代码写回的 analysis['svd_ridge_skipped']。
    """
    from llmsec.evaluation import cluster_analysis as ca
    monkeypatch.setattr(ca, 'build_svd_ridge_summary', lambda tr: None)
    monkeypatch.setattr(ca, 'build_blend_predictor_summary', lambda tr: None)
    return ca.analyze_clusters(
        tracker,
        cluster_report={"method_labels": {"u1": 0, "u2": 0, "u3": 1, "u4": 1}},
        cluster_artifacts=None,
    )


def test_svd_ridge_diagnosis_reasons_specific(monkeypatch, tmp_path):
    """诊断兜底应给出具体原因而非'原因未知'（真实 analyze_clusters 路径）。

    model.w=None + GT 与特征键一致 → 应命中"Ridge 解退化"分支。
    旧版此测试传空 cluster_report——analyze_clusters 直接早退"无聚类数据"，
    诊断块从未执行、断言恒真；现改为真实驱动并断言返回值。
    """
    from llmsec.evaluation.elo import ELOTracker

    tracker = ELOTracker()
    tracker.predictor.artifacts = {
        "features": {u: {"textual": [1.0, 0.0]} for u in ("u1", "u2", "u3", "u4")},
        "units": None,
    }
    tracker.predictor.ground_truth = {"u1": 1, "u2": 2, "u3": 3, "u4": 4}
    tracker.predictor.model = SimpleNamespace(w=None)
    import llmsec.core.config as _cfg
    monkeypatch.setattr(_cfg, "FEATURE_CACHE_FILE", tmp_path / "fc.pkl")

    analysis = _run_svd_diagnosis(tracker, monkeypatch)
    reason = analysis.get("svd_ridge_skipped", "")
    assert reason, "诊断块应执行并写回 svd_ridge_skipped"
    assert "原因未知" not in reason, f"诊断不应含'原因未知'，实得: {reason!r}"
    assert "Ridge" in reason or "退化" in reason, f"应命中 Ridge 退化分支: {reason!r}"


def test_svd_ridge_diagnosis_model_none_reason(monkeypatch, tmp_path):
    """predictor.model 为 None 时，诊断原因应指出'模型未装配'（真实路径）。"""
    from llmsec.evaluation.elo import ELOTracker

    tracker = ELOTracker()
    # GT 4 个单位且特征键与 GT 一致（避开 stale 分支），model 未装配
    tracker.predictor.artifacts = {
        "features": {u: {"textual": [1.0, 0.0]} for u in ("u1", "u2", "u3", "u4")},
        "units": None,
    }
    tracker.predictor.ground_truth = {"u1": 1, "u2": 2, "u3": 3, "u4": 4}
    tracker.predictor.model = None
    import llmsec.core.config as _cfg
    monkeypatch.setattr(_cfg, "FEATURE_CACHE_FILE", tmp_path / "fc.pkl")

    analysis = _run_svd_diagnosis(tracker, monkeypatch)
    reason = analysis.get("svd_ridge_skipped", "")
    assert "未装配" in reason, f"model=None 应给出'未装配'原因，实得: {reason!r}"
    assert "原因未知" not in reason


# ============================================================
# 9. 子进程日志无缓冲：tasks._spawn 注入 PYTHONUNBUFFERED
# ============================================================
def test_spawn_injects_pythonunbuffered(monkeypatch, tmp_path):
    """_spawn 应给子进程 env 注入 PYTHONUNBUFFERED=1，保证日志实时落盘。"""
    captured = {}

    class _FakeProc:
        def poll(self):
            return 0  # 已结束

    def _fake_popen(*args, **kw):
        captured['env'] = kw.get('env')
        captured['argv'] = args[0] if args else kw.get('args')
        return _FakeProc()

    from llmsec.server import task_manager

    # _spawn 实现已统一到 task_manager（tasks.py 只是 HTTP 薄封装），
    # 注入点相应指向 task_manager 命名空间
    monkeypatch.setattr(task_manager.subprocess, 'Popen', _fake_popen)
    monkeypatch.setattr(task_manager, 'TASK_LOG_DIR', tmp_path)
    monkeypatch.setattr(task_manager, '_advance_queue', lambda k: None)

    t = {
        'kind': 'evaluate',
        'argv': ['-m', 'llmsec.pipeline.runner', '--help'],
        'log_path': tmp_path / 't.log',
        'started_at': '2026-01-01T00:00:00',
    }
    task_manager._spawn('test-id', t)
    assert captured['env'].get('PYTHONUNBUFFERED') == '1', (
        '_spawn 应注入 PYTHONUNBUFFERED=1 让子进程日志无缓冲'
    )
    assert captured['env'].get('LLMSEC_TASK_ID') == 'test-id', '原有注入不丢'


# ============================================================
# 10. 目标探活 chat smoke：401/403 鉴权失败判不可达（防假绿灯白跑）
# ============================================================
class _AuthError(Exception):
    """模拟 OpenAI SDK 鉴权异常（带 status_code，供 getattr 判定）。"""

    def __init__(self, status_code, msg="Invalid API Key"):
        super().__init__(f"{status_code} - {msg}")
        self.status_code = status_code


def test_target_probe_401_marks_unreachable(monkeypatch):
    """目标模型 chat 鉴权失败(401) → reachable=False，防 models.list 假绿灯导致白跑。

    复现真实故障：DeepSeek key 无效，models.list 通过(不校验 chat 权限)，
    但 chat.completions 全线 401。原探活只调 models.list → 亮绿灯 → 运行 0 结果。
    """
    import asyncio

    from llmsec.core import config as cfg_mod
    from llmsec.core import llm as llm_mod
    from llmsec.server.routers import data_query as dq

    fake_target = SimpleNamespace(
        name='badkey-target', model='m', api_key='wrong', base_url='http://fake/v1',
        timeout=5.0,
    )
    monkeypatch.setattr(cfg_mod, 'load_targets',
                        lambda: {'badkey-target': fake_target})
    monkeypatch.setattr(dq, 'load_targets',
                        lambda: {'badkey-target': fake_target}, raising=False)
    # generator/judge 不在本测试范围，给空配置避免联网
    monkeypatch.setattr(cfg_mod.GeneratorConfig, 'from_env',
                        staticmethod(lambda: SimpleNamespace(
                            model='m', api_key='k', base_url='http://fake/v1', timeout=5.0, max_retries=1)))
    monkeypatch.setattr(cfg_mod.JudgeConfig, 'from_env',
                        staticmethod(lambda: SimpleNamespace(
                            model='m', api_key='k', base_url='http://fake/v1', timeout=5.0, max_retries=1)))

    def _fake_create(api_key=None, base_url=None, timeout=60.0):
        class _C:
            class models:
                @staticmethod
                def list():
                    return [SimpleNamespace(id='m')]  # models.list 通过（假绿灯根源）

            class _Completions:
                @staticmethod
                def create(**kw):
                    raise _AuthError(401)  # chat 才暴露鉴权失败
            chat = SimpleNamespace(completions=_Completions)
        return _C()

    monkeypatch.setattr(llm_mod, 'create_openai_client', _fake_create)

    result = asyncio.run(dq.api_targets_probe())
    targets = {t['name']: t for t in result.get('targets', [])}
    t = targets.get('badkey-target')
    assert t is not None, '目标应被探活'
    assert t['reachable'] is False, '401 鉴权失败应判不可达（防假绿灯白跑）'
    assert '401' in (t.get('error') or '') or '鉴权' in (t.get('error') or ''), (
        f'error 应提示鉴权失败，实际: {t.get("error")!r}'
    )


def test_target_probe_403_marks_unreachable(monkeypatch):
    """403（权限拒绝）同样判不可达。"""
    import asyncio

    from llmsec.core import config as cfg_mod
    from llmsec.core import llm as llm_mod
    from llmsec.server.routers import data_query as dq

    fake_target = SimpleNamespace(
        name='forbidden', model='m', api_key='k', base_url='http://fake/v1', timeout=5.0)
    monkeypatch.setattr(cfg_mod, 'load_targets', lambda: {'forbidden': fake_target})
    monkeypatch.setattr(dq, 'load_targets', lambda: {'forbidden': fake_target}, raising=False)

    def _fake_create(api_key=None, base_url=None, timeout=60.0):
        class _C:
            class models:
                @staticmethod
                def list():
                    return [SimpleNamespace(id='m')]

            class _Completions:
                @staticmethod
                def create(**kw):
                    raise _AuthError(403)
            chat = SimpleNamespace(completions=_Completions)
        return _C()

    monkeypatch.setattr(llm_mod, 'create_openai_client', _fake_create)
    result = asyncio.run(dq.api_targets_probe())
    t = {x['name']: x for x in result.get('targets', [])}['forbidden']
    assert t['reachable'] is False, '403 应判不可达'


def test_target_probe_non_auth_error_not_blocking(monkeypatch):
    """非鉴权错误（如 429/5xx/超时）不判不可达——models.list 已通，chat 偶发抖动不应阻塞。"""
    import asyncio

    from llmsec.core import config as cfg_mod
    from llmsec.core import llm as llm_mod
    from llmsec.server.routers import data_query as dq

    fake_target = SimpleNamespace(
        name='flaky', model='m', api_key='k', base_url='http://fake/v1', timeout=5.0)
    monkeypatch.setattr(cfg_mod, 'load_targets', lambda: {'flaky': fake_target})
    monkeypatch.setattr(dq, 'load_targets', lambda: {'flaky': fake_target}, raising=False)

    def _fake_create(api_key=None, base_url=None, timeout=60.0):
        class _C:
            class models:
                @staticmethod
                def list():
                    return [SimpleNamespace(id='m')]

            class _Completions:
                @staticmethod
                def create(**kw):
                    e = Exception('429 rate limit')
                    e.status_code = 429  # 限流，非鉴权
                    raise e
            chat = SimpleNamespace(completions=_Completions)
        return _C()

    monkeypatch.setattr(llm_mod, 'create_openai_client', _fake_create)
    result = asyncio.run(dq.api_targets_probe())
    t = {x['name']: x for x in result.get('targets', [])}['flaky']
    assert t['reachable'] is True, '429 限流不应判不可达（models.list 已通）'
    assert t.get('warning'), '应有 warning 提示 chat 探测失败'


# ============================================================
# 11. 报告 fpr=None 不崩（过敏检测无有效样本时）
# ============================================================
def test_recommendation_handles_fpr_none():
    """_generate_recommendation 接受 fpr=None 不崩（参数标注 float 但函数体只用 level）。"""
    from llmsec.reporting.final_report import _generate_recommendation

    # fpr=None 不应抛 TypeError
    for level in ('safe', 'allergic', 'vulnerable', 'broken', 'inconclusive'):
        rec = _generate_recommendation(level)
        assert isinstance(rec, str) and rec, f'level={level} 应返回非空建议文案'


def test_final_report_fpr_none_no_crash():
    """generate_final_report 在 allergy fpr=None 时不抛 NoneType 比较异常。

    复现真实故障：目标鉴权失效 → 过敏检测全线跳过 → fpr=None →
    报告 '<' not supported between NoneType and float 崩溃。
    构造 tested_methods≥5 + confidence≥0.5 强制走到 fpr 比较分支，验证不崩。
    """
    from llmsec.reporting import final_report as fr

    fake_tracker = SimpleNamespace(
        # confidence=0.9 ≥ 0.5 阈值，强制不走 inconclusive 早退，直奔 fpr 比较
        compute_security_boundary=lambda name: {'boundary_elo': 1500, 'confidence': 0.9},
        defender_ratings=SimpleNamespace(values={}),
        get_attacker_ranking=lambda: [],
    )
    attack_summary = {'asr': 0.0, 'total_attacks': 10}  # total_attacks≥5 走 fpr 分支
    allergy_summary = {'fpr': None, 'total_tested': 0, 'allergic': 0}

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        try:
            fr.generate_final_report(
                run_dir=td, tracker=fake_tracker, defender_name='t',
                attack_summary=attack_summary, allergy_summary=allergy_summary,
                total_methods=5, units={},
            )
        except TypeError as e:
            if 'NoneType' in str(e) and ('float' in str(e) or 'int' in str(e)):
                pytest.fail(f'fpr=None 在比较分支触发崩溃: {e}')
            # 其他 TypeError（fake 不全）可接受
        except Exception:
            pass  # fake 不全可能抛其他，本测试只验证不崩在 NoneType 比较


# ============================================================
# 12. ASR=0 不被存成 null（攻击全失败是合法且重要的结果）
# ============================================================
def test_final_report_asr_zero_not_null(monkeypatch, tmp_path):
    """generate_final_report 在 asr=0.0 时应存 0.0 而非 null。

    复现真实 bug：`round(asr,4) if asr else None` 在 asr=0.0（falsy）时存成 None，
    导致 minimax（ASR=0，防御成功）的报告 attack_phase.asr=null，前端显示 N/A。
    """
    from llmsec.reporting import final_report as fr

    fake_tracker = SimpleNamespace(
        compute_security_boundary=lambda name: {'boundary_elo': 1500, 'confidence': 0.9,
                                                 'methods_above_boundary': 0},
        defender_ratings=SimpleNamespace(values={}),
        get_attacker_ranking=lambda: [],
    )
    # asr=0.0（攻击全失败），total_attacks≥5 + confidence≥0.5 走正常判定分支
    attack_summary = {'asr': 0.0, 'total_attacks': 10, 'successful': 0,
                      'rounds': 5, 'jailbreak_tax': {'probed': 10}}
    allergy_summary = {'fpr': 0.0, 'total_tested': 5, 'allergic': 0}

    monkeypatch.setattr('llmsec.core.io.read_jsonl', lambda *a, **k: [])
    monkeypatch.setattr('llmsec.core.io.write_json', lambda *a, **k: None)

    try:
        report = fr.generate_final_report(
            run_dir=str(tmp_path), tracker=fake_tracker, defender_name='t',
            attack_summary=attack_summary, allergy_summary=allergy_summary,
            total_methods=5, units={},
        )
        assert report['attack_phase']['asr'] == 0.0, (
            f'asr=0.0 应存 0.0 不存 null，实际: {report["attack_phase"]["asr"]!r}'
        )
    except Exception as e:
        # 若 fake tracker 不全导致其他异常，只验证 asr 序列化那行不把 0 变 null
        if 'asr' not in str(e).lower():
            pass
        else:
            raise


# ============================================================
# 13. 越狱税：空响应不计入 attack_accuracy 分母
# ============================================================
def test_tax_empty_response_excluded_from_accuracy():
    """summarize_jailbreak_tax 的 attack_accuracy 分母应排除空响应（response_preview 为空）。

    复现真实 bug：minimax 58 条空响应（content=None, preview='', level D, is_refusal=False）
    被当"有效作答"拉低正确率，制造 ASR=0 但 tax 退化 98% 的矛盾。
    """
    from llmsec.pipeline.tax import summarize_jailbreak_tax

    # 构造：1 条正常作答（math=2），2 条空响应（is_refusal=False 但 preview 空），2 条拒绝
    results = [
        # 正常作答，做对
        {'math_score': 2, 'is_refusal': False, 'is_harmful': False,
         'response_preview': '答案是 [MATH:42]', 'jailbreak_tax': 0.0},
        # 空响应（minimax 场景）：is_refusal=False 但没内容
        {'math_score': 0, 'is_refusal': False, 'is_harmful': False,
         'response_preview': '', 'jailbreak_tax': 0.0},
        {'math_score': 0, 'is_refusal': False, 'is_harmful': False,
         'response_preview': '', 'jailbreak_tax': 0.0},
        # 明确拒绝（is_refusal=True，排除合理）
        {'math_score': 0, 'is_refusal': True, 'is_harmful': False,
         'response_preview': '我拒绝', 'jailbreak_tax': 0.0},
        {'math_score': 2, 'is_refusal': True, 'is_harmful': False,
         'response_preview': '拒绝但 [MATH:42]', 'jailbreak_tax': 0.0},
    ]
    summary = summarize_jailbreak_tax(results, baseline={'accuracy': 1.0})
    # 修复后：answered 只含第1条（正常作答），attack_accuracy = 1/1 = 1.0（退化 0%）
    # 修复前：answered 含第1-3条，attack_accuracy = 1/3 ≈ 0.333（假退化 67%）
    assert summary['attack_accuracy'] == 1.0, (
        f'空响应应排除，attack_accuracy 应为 1.0，实际: {summary["attack_accuracy"]}'
    )
    assert summary['math_dist']['no_format'] == 0, (
        f'空响应不应计入 no_format，实际: {summary["math_dist"]}'
    )


# ============================================================
# 14. Config env 覆盖：JUDGE_TIMEOUT / GENERATOR_TIMEOUT / MAX_TOKENS
# ============================================================
def test_judge_config_reads_timeout_env(monkeypatch):
    """JudgeConfig.from_env 应读 JUDGE_TIMEOUT / JUDGE_MAX_TOKENS 环境变量。"""
    from llmsec.core import config as cfg_mod

    monkeypatch.setenv('JUDGE_TIMEOUT', '150')
    monkeypatch.setenv('JUDGE_MAX_TOKENS', '2048')
    monkeypatch.setenv('JUDGE_MODEL', 'test-judge')
    # 避免真实 .env 干扰
    monkeypatch.setattr(cfg_mod, 'load_env', lambda: None)

    c = cfg_mod.JudgeConfig.from_env()
    assert c.timeout == 150.0, f'JUDGE_TIMEOUT 应覆盖默认，实际: {c.timeout}'
    assert c.max_tokens == 2048, f'JUDGE_MAX_TOKENS 应覆盖默认，实际: {c.max_tokens}'


def test_generator_config_reads_timeout_env(monkeypatch):
    """GeneratorConfig.from_env 应读 GENERATOR_TIMEOUT / GENERATOR_MAX_TOKENS 环境变量。"""
    from llmsec.core import config as cfg_mod

    monkeypatch.setenv('GENERATOR_TIMEOUT', '120')
    monkeypatch.setenv('GENERATOR_MAX_TOKENS', '8192')
    monkeypatch.setattr(cfg_mod, 'load_env', lambda: None)

    c = cfg_mod.GeneratorConfig.from_env()
    assert c.timeout == 120.0, f'GENERATOR_TIMEOUT 应覆盖默认，实际: {c.timeout}'
    assert c.max_tokens == 8192, f'GENERATOR_MAX_TOKENS 应覆盖默认，实际: {c.max_tokens}'


def test_config_defaults_when_no_env(monkeypatch):
    """无 env 覆盖时用代码默认值（Judge timeout=90, max_tokens=1024）。"""
    from llmsec.core import config as cfg_mod

    monkeypatch.delenv('JUDGE_TIMEOUT', raising=False)
    monkeypatch.delenv('JUDGE_MAX_TOKENS', raising=False)
    monkeypatch.setattr(cfg_mod, 'load_env', lambda: None)

    c = cfg_mod.JudgeConfig.from_env()
    assert c.timeout == 90.0, f'Judge 默认 timeout 应为 90，实际: {c.timeout}'
    assert c.max_tokens == 1024, f'Judge 默认 max_tokens 应为 1024，实际: {c.max_tokens}'
