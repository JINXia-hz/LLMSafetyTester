"""
P1 修复回归测试：targets(pcap/单例) 与 judge/evaluator。

验证：
1. M12：多线程并发首调 call_target，client 只创建一次（双重检查锁定）。
2. M16：judge.parse_compliance_level 分级匹配，解释性文本不再误判首个 A-E 字母。
3. M10：pcap HTTP 5xx/429 纳入重试（3 次），4xx（除 429）立即返回不重试。
4. C5 ：pcap HTTP 错误分支 tokens_completion 用 estimate_tokens(resp.text)。
5. M11：pcap env 惰性读取——运行期改 os.environ 后建客户端/构造 payload 生效。
6. M17：judge env 惰性读取——create_judge_client()/Judge() 重新 from_env()。
7. M19：evaluator.update_elo 支持 defender_name 参数，缺省回退 TARGET_MODEL。
"""
import inspect
import os
import threading
import time
from pathlib import Path
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

def test_m10_pcap_http_retry():
    client = pcap.PcapJudgeTargetClient(url='https://fake.local/judge')
    old_post = pcap.requests.post
    old_sleep = time.sleep
    try:
        time.sleep = lambda *_a, **_kw: None
        mock = _PcapHttpMock([_FakeResp(500, text='Internal Server Error')])
        pcap.requests.post = mock.post
        result = client.call('测试 prompt')
        assert mock.calls == client.max_retries, f'M10: HTTP 500 重试 {client.max_retries} 次（实际 {mock.calls} 次）'
        assert result['error'] is not None and 'HTTP 500' in result['error'], 'M10: 500 重试耗尽后以 error 字段返回'
        mock = _PcapHttpMock([_FakeResp(429, text='Too Many Requests')])
        pcap.requests.post = mock.post
        client.call('测试 prompt')
        assert mock.calls == client.max_retries, f'M10: HTTP 429 重试 {client.max_retries} 次（实际 {mock.calls} 次）'
        mock = _PcapHttpMock([_FakeResp(400, text='Bad Request')])
        pcap.requests.post = mock.post
        result = client.call('测试 prompt')
        assert mock.calls == 1, f'M10: HTTP 400 不重试（实际 {mock.calls} 次）'
        assert 'HTTP 400' in (result['error'] or ''), "M10: 400 返回结构保持 error='HTTP 400: ...'"
        mock = _PcapHttpMock([_FakeResp(500, text='err'), _FakeResp(200, payload={'error_code': 0, 'pred': '正常', 'text': '分析结果'})])
        pcap.requests.post = mock.post
        result = client.call('测试 prompt')
        assert mock.calls == 2 and result['error'] is None, 'M10: 500 后恢复 200，重试成功'
    finally:
        pcap.requests.post = old_post
        time.sleep = old_sleep

def test_c5_http_error_tokens():
    client = pcap.PcapJudgeTargetClient(url='https://fake.local/judge')
    old_post = pcap.requests.post
    try:
        body = 'Bad Request: invalid log format'
        mock = _PcapHttpMock([_FakeResp(400, text=body)])
        pcap.requests.post = mock.post
        result = client.call('测试 prompt')
        assert result['tokens_completion'] == estimate_tokens(body), 'C5: HTTP 错误分支 tokens_completion = estimate_tokens(resp.text)'
        assert result['tokens_completion'] > 0, 'C5: 非空错误体不再硬编码 tokens_completion=0'
    finally:
        pcap.requests.post = old_post

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
    saved = {k: os.environ.get(k) for k in ('GENERATOR_API_KEY', 'GENERATOR_BASE_URL', 'JUDGE_MODEL')}
    old_create = judge_mod.create_openai_client
    try:
        os.environ['GENERATOR_API_KEY'] = 'runtime-key-123'
        os.environ['GENERATOR_BASE_URL'] = 'https://runtime-judge.local/v1'
        os.environ['JUDGE_MODEL'] = 'runtime-judge-model'
        captured = {}

        def fake_create_openai_client(api_key=None, base_url=None, timeout=None):
            captured.update(api_key=api_key, base_url=base_url, timeout=timeout)
            return object()
        judge_mod.create_openai_client = fake_create_openai_client
        judge_mod.create_judge_client()
        assert captured.get('api_key') == 'runtime-key-123', 'M17: create_judge_client 重新读取运行期 GENERATOR_API_KEY'
        assert captured.get('base_url') == 'https://runtime-judge.local/v1', 'M17: create_judge_client 重新读取运行期 GENERATOR_BASE_URL'
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

        def update(self, method, defender, score):
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
        assert bool(used_names) and all((n == 'test-pcap-model' for n in used_names)), 'M19: 显式 defender_name 时全部使用传入名（pcap 场景）'
        used_names.clear()
        ev.update_elo(results, {})
        assert bool(used_names) and all((n == ev.TARGET_MODEL for n in used_names)), f'M19: 缺省回退 TARGET_MODEL（当前 {ev.TARGET_MODEL!r}）'
    finally:
        ev.ELOTracker = old_tracker
