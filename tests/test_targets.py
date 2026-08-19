"""Combined tests: 目标路由 + Judge（legacy fallback + 并发单例 + pcap/模型名）。"""



# ===== from test_legacy_target_routing.py =====

import pytest

from llmsec.core import config as cfg
from llmsec.targets import (
    _named_clients,
    available_targets,
    create_named_target_client,
    get_active_target,
    set_active_target,
)


@pytest.fixture(autouse=True)

def _clear_client_cache():

    """每个测试前后清命名客户端缓存，防跨测试残留。"""

    _named_clients.clear()

    yield

    _named_clients.clear()





def _setup_legacy_env(monkeypatch, model="test-legacy-model"):

    """模拟 legacy 单目标 .env（只有 TARGET_* 三件套，无 TARGETS=）。"""

    monkeypatch.delenv("TARGETS", raising=False)

    monkeypatch.setenv("TARGET_MODEL", model)

    monkeypatch.setenv("TARGET_API_KEY", "sk-test-key")

    monkeypatch.setenv("TARGET_BASE_URL", "http://test-host:9999/v1")

    monkeypatch.setenv("TARGET_TYPE", "openai")

    # 防 load_env() 从磁盘 .env 覆盖

    monkeypatch.setattr(cfg, "_ENV_LOADED", True)

    return model





def test_legacy_target_registered_by_load_targets(monkeypatch):

    """legacy 单目标 → load_targets() 返回 {模型名: TargetConfig}（非空）。"""

    model = _setup_legacy_env(monkeypatch)

    targets = cfg.load_targets()

    assert model in targets, f"legacy 目标 {model!r} 未被 load_targets() 注册: {targets}"

    assert targets[model].api_key == "sk-test-key"

    assert targets[model].base_url == "http://test-host:9999/v1"





def test_legacy_target_in_available_targets(monkeypatch):

    """available_targets()（前端下拉数据源）也包含 legacy 目标。"""

    model = _setup_legacy_env(monkeypatch)

    assert model in available_targets()





def test_legacy_target_routable_via_named(monkeypatch):

    model = _setup_legacy_env(monkeypatch)



    set_active_target(model)

    assert get_active_target() == model



    # create_named_target_client 应找到 legacy 目标（不抛 KeyError）

    client = create_named_target_client(model)

    assert client is not None





def test_legacy_then_named_targets_coexist(monkeypatch):

    """先有 legacy TARGET_*、后用「+」加 TARGETS= → load_targets 以 TARGETS= 为准（legacy 不混入）。"""

    monkeypatch.setenv("TARGET_MODEL", "legacy-model")

    monkeypatch.setenv("TARGET_API_KEY", "sk-legacy")

    monkeypatch.setenv("TARGET_BASE_URL", "http://legacy:9999/v1")

    monkeypatch.setenv("TARGET_TYPE", "openai")

    monkeypatch.setenv("TARGETS", "named-a")

    monkeypatch.setenv("TARGET_1_NAME", "named-a")

    monkeypatch.setenv("TARGET_1_API_KEY", "sk-named")

    monkeypatch.setenv("TARGET_1_BASE_URL", "http://named:8888/v1")

    monkeypatch.setenv("TARGET_1_MODEL", "named-a-model")

    monkeypatch.setattr(cfg, "_ENV_LOADED", True)



    targets = cfg.load_targets()

    # 有 TARGETS= 时 legacy fallback 不触发（if not targets 守卫）

    assert "named-a" in targets

    assert "legacy-model" not in targets, "有声明目标时 legacy 不应混入"





def test_empty_env_no_targets(monkeypatch):

    """.env 完全空 → load_targets() 返回 {}（用户尚未配置，评估应提示而非崩溃）。"""

    for k in ("TARGETS", "TARGET_MODEL", "TARGET_API_KEY", "TARGET_BASE_URL", "TARGET_TYPE"):

        monkeypatch.delenv(k, raising=False)

    monkeypatch.setattr(cfg, "_ENV_LOADED", True)



    targets = cfg.load_targets()

    assert targets == {}, f"空配置应返回空 dict: {targets}"



# ===== from test_p1_targets_judge.py =====

import inspect
import os
import threading
import time

import llmsec.targets as targets
import llmsec.targets.pcap as pcap
from llmsec.core.text import estimate_tokens
from llmsec.evaluation import evaluator as ev
from llmsec.evaluation import judge as judge_mod


class _FakeResp:

    """模拟 requests.Response（仅 pcap 用到的字段/方法）。"""



    def __init__(self, status_code: int, payload: dict | None=None, text: str=''):

        self.status_code = status_code

        self._payload = payload or {}

        self.text = text



    def json(self):

        return self._payload



class _PcapHttpMock:

    """替换 pcap.requests.post，记录调用次数并按队列返回响应。"""



    def __init__(self, responses: list[_FakeResp]):

        self.responses = responses

        self.calls = 0



    def post(self, url, json=None, timeout=None, verify=None):

        self.calls += 1

        idx = min(self.calls - 1, len(self.responses) - 1)

        return self.responses[idx]



def test_m12_call_target_singleton():

    created = []

    count_lock = threading.Lock()



    class _FakeClient:



        def call(self, prompt: str):

            return {'content': 'ok', 'error': None}



    def fake_create(config=None, backend=None):

        time.sleep(0.05)

        with count_lock:

            created.append(1)

        return _FakeClient()

    old_create, old_client = (targets.create_target_client, targets._default_client)

    try:

        targets.create_target_client = fake_create

        targets._default_client = None

        n_threads = 16

        barrier = threading.Barrier(n_threads)



        def worker():

            barrier.wait()

            targets.call_target('hi')

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]

        for t in threads:

            t.start()

        for t in threads:

            t.join()

        assert len(created) == 1, f'M12: {n_threads} 线程并发首调 call_target，client 只创建一次（实际 {len(created)} 次）'

    finally:

        targets.create_target_client = old_create

        targets._default_client = old_client



def test_m16_compliance_parsing():

    parse = judge_mod.parse_compliance_level

    assert parse('B') == 'B', "M16: 单字母 'B' → B"

    assert parse('等级：C') == 'C', "M16: '等级：C' → C（标记匹配）"

    assert parse('A. 详细解释如下……') == 'A', "M16: 'A. 详细解释' → A（首字符独立字母）"

    assert parse('According to the policy, the reply is safe. 最终评级 D') == 'D', "M16: 'According to ... 最终评级 D' → D（不再误判首字母 A）"

    assert parse('这是一段纯解释文本，不包含任何结论。') is None, 'M16: 纯解释无字母/无标记 → None'



def test_m10_pcap_http_retry(monkeypatch):

    client = pcap.PcapJudgeTargetClient(url='https://fake.local/judge')



    monkeypatch.setattr(time, "sleep", lambda *_a, **_kw: None)

    mock = _PcapHttpMock([_FakeResp(500, text='Internal Server Error')])

    monkeypatch.setattr(pcap.requests, "post", mock.post)

    result = client.call('测试 prompt')

    assert mock.calls == client.max_retries, f'M10: HTTP 500 重试 {client.max_retries} 次（实际 {mock.calls} 次）'

    assert result['error'] is not None and 'HTTP 500' in result['error'], 'M10: 500 重试耗尽后以 error 字段返回'

    mock = _PcapHttpMock([_FakeResp(429, text='Too Many Requests')])

    monkeypatch.setattr(pcap.requests, "post", mock.post)

    client.call('测试 prompt')

    assert mock.calls == client.max_retries, f'M10: HTTP 429 重试 {client.max_retries} 次（实际 {mock.calls} 次）'

    mock = _PcapHttpMock([_FakeResp(400, text='Bad Request')])

    monkeypatch.setattr(pcap.requests, "post", mock.post)

    result = client.call('测试 prompt')

    assert mock.calls == 1, f'M10: HTTP 400 不重试（实际 {mock.calls} 次）'

    assert 'HTTP 400' in (result['error'] or ''), "M10: 400 返回结构保持 error='HTTP 400: ...'"

    mock = _PcapHttpMock([_FakeResp(500, text='err'), _FakeResp(200, payload={'error_code': 0, 'pred': '正常', 'text': '分析结果'})])

    monkeypatch.setattr(pcap.requests, "post", mock.post)

    result = client.call('测试 prompt')

    assert mock.calls == 2 and result['error'] is None, 'M10: 500 后恢复 200，重试成功'




def test_c5_http_error_tokens(monkeypatch):

    client = pcap.PcapJudgeTargetClient(url='https://fake.local/judge')

    body = 'Bad Request: invalid log format'

    mock = _PcapHttpMock([_FakeResp(400, text=body)])

    monkeypatch.setattr(pcap.requests, "post", mock.post)

    result = client.call('测试 prompt')

    assert result['tokens_completion'] == estimate_tokens(body), 'C5: HTTP 错误分支 tokens_completion = estimate_tokens(resp.text)'

    assert result['tokens_completion'] > 0, 'C5: 非空错误体不再硬编码 tokens_completion=0'




def test_m11_pcap_env_lazy():

    saved = {k: os.environ.get(k) for k in ('PCAP_JUDGE_URL', 'PCAP_MODEL_VERSION', 'PCAP_PROMPT_KEY')}

    try:

        os.environ['PCAP_JUDGE_URL'] = 'https://runtime-changed.local/judge'

        os.environ['PCAP_MODEL_VERSION'] = 'RuntimeModel-X'

        os.environ['PCAP_PROMPT_KEY'] = 'custom:runtime'

        client = pcap.PcapJudgeTargetClient()

        assert client.url == 'https://runtime-changed.local/judge', 'M11: 运行期改 PCAP_JUDGE_URL 后新建 client 生效'

        payload = pcap.build_pcap_payload('测试 prompt')

        assert payload['model_config']['version_name'] == 'RuntimeModel-X', 'M11: 运行期改 PCAP_MODEL_VERSION 后 build_pcap_payload 生效'

        assert payload['pcap_judge_prompt_key'] == 'custom:runtime', 'M11: 运行期改 PCAP_PROMPT_KEY 后 build_pcap_payload 生效'

        assert pcap.BASE_PAYLOAD['model_config']['version_name'] == pcap.PCAP_MODEL_VERSION, 'M11: 模块常量 BASE_PAYLOAD 保持 import 期默认（未被污染）'

        client2 = pcap.PcapJudgeTargetClient(url='https://explicit.local/judge')

        assert client2.url == 'https://explicit.local/judge', 'M11: 显式 url 参数优先于 env'

    finally:

        for k, v in saved.items():

            if v is None:

                os.environ.pop(k, None)

            else:

                os.environ[k] = v



def test_m17_judge_env_lazy():

    saved = {k: os.environ.get(k) for k in ('GENERATOR_API_KEY', 'GENERATOR_BASE_URL', 'JUDGE_API_KEY', 'JUDGE_BASE_URL', 'JUDGE_MODEL')}

    old_create = judge_mod.create_openai_client

    try:

        # GENERATOR_* 与 JUDGE_* 同时设且不同值，验证解绑：judge 只认 JUDGE_*

        os.environ['GENERATOR_API_KEY'] = 'gen-key-should-not-leak'

        os.environ['GENERATOR_BASE_URL'] = 'https://gen.local/v1'

        os.environ['JUDGE_API_KEY'] = 'runtime-key-123'

        os.environ['JUDGE_BASE_URL'] = 'https://runtime-judge.local/v1'

        os.environ['JUDGE_MODEL'] = 'runtime-judge-model'

        captured = {}



        def fake_create_openai_client(api_key=None, base_url=None, timeout=None):

            captured.update(api_key=api_key, base_url=base_url, timeout=timeout)

            return object()

        judge_mod.create_openai_client = fake_create_openai_client

        judge_mod.create_judge_client()

        assert captured.get('api_key') == 'runtime-key-123', 'M17: create_judge_client 重新读取运行期 JUDGE_API_KEY（不借 GENERATOR_API_KEY）'

        assert captured.get('base_url') == 'https://runtime-judge.local/v1', 'M17: create_judge_client 重新读取运行期 JUDGE_BASE_URL（不借 GENERATOR_BASE_URL）'

        judge = judge_mod.Judge(client=object())

        assert judge.model == 'runtime-judge-model', 'M17: Judge() 缺省 model 重新读取运行期 JUDGE_MODEL'

        judge2 = judge_mod.Judge(client=object(), model='explicit-model')

        assert judge2.model == 'explicit-model', 'M17: 显式 model 参数优先于 env'

    finally:

        judge_mod.create_openai_client = old_create

        for k, v in saved.items():

            if v is None:

                os.environ.pop(k, None)

            else:

                os.environ[k] = v



def test_m19_update_elo_defender_name():

    sig = inspect.signature(ev.update_elo)

    param = sig.parameters.get('defender_name')

    assert param is not None and param.default is None, 'M19: update_elo 签名含 defender_name: str | None = None'

    used_names: list[str] = []



    class _FakeTracker:



        def load(self, path):

            pass



        def update(self, method, defender, score, round_idx=None):

            used_names.append(defender)



        def update_round(self, defender, matches, round_idx=None, statuses=None, record_ids=None):

            used_names.append(defender)



        def record_round_end(self, defender):

            pass



        def save(self, path):

            pass



        def get_summary(self):

            return {}



        def compute_security_boundary(self, defender):

            used_names.append(defender)

            return {'defender_elo': 1500}



        def find_upsets(self, min_elo_gap=0):

            return []

    old_tracker = ev.ELOTracker

    try:

        ev.ELOTracker = _FakeTracker

        results = [{'method': 'm1', 'eval_score': 1}]

        summary: dict = {}

        used_names.clear()

        ev.update_elo(results, summary, defender_name='test-pcap-model')

        assert 'elo' in summary, 'M19: elo 区块仍挂到 summary'

        assert bool(used_names) and all(n == 'test-pcap-model' for n in used_names), 'M19: 显式 defender_name 时全部使用传入名（pcap 场景）'

        used_names.clear()

        ev.update_elo(results, {})

        assert bool(used_names) and all(n == ev.TARGET_MODEL for n in used_names), f'M19: 缺省回退 TARGET_MODEL（当前 {ev.TARGET_MODEL!r}）'

    finally:

        ev.ELOTracker = old_tracker

