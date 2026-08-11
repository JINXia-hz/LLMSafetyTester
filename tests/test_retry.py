"""
P1 冒烟测试：统一重试辅助 retry_call（M5）。

验证：
1. core.llm.retry_call：成功不重试、前 N-1 次失败后成功、全部失败抛最后异常、
   retries/delay 参数生效（patch time.sleep 记录，不真等）、retry_on 谓词、
   on_retry 覆盖间隔（429/限流特殊间隔的机制）。
2. chat_with_retry：签名/行为不变（report.py 在用）。
3. 迁移后各模块重试行为冒烟（mock 底层 HTTP/SDK 调用）：
   - judge          2 次 / 2s
   - openai_backend 3 次 / 3s
   - pcap           3 次 / 3s，5xx/429 重试、4xx 立即返回
   - generate       3 次，内容类失败 2s / API 类失败 5s
   - safe_twin      3 次 / 2s，耗尽返回 None
"""
import time

from llmsec.core.llm import chat_with_retry, retry_call


class _SleepRecorder:
    """替换全局 time.sleep，记录每次间隔（不真正等待）。"""

    def __init__(self):
        self.calls: list[float] = []
        self._orig = None

    def __enter__(self):
        self._orig = time.sleep
        time.sleep = lambda s, *a, **kw: self.calls.append(s)
        return self

    def __exit__(self, *exc):
        time.sleep = self._orig

class _FakeUsage:
    prompt_tokens = 10
    completion_tokens = 20

class _FakeChatResponse:
    """模拟 openai 的 ChatCompletion（仅各模块用到的字段）。"""

    def __init__(self, content: str='ok'):
        self.choices = [type('C', (), {'message': type('M', (), {'content': content})()})()]
        self.usage = _FakeUsage()

class _FlakyCompletions:
    """create() 按队列出结果：Exception 抛出，其余返回；队列用尽重复最后一项。"""

    def __init__(self, outcomes: list):
        self.outcomes = list(outcomes)
        self.calls = 0

    def create(self, **kw):
        self.calls += 1
        outcome = self.outcomes[min(self.calls - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

class _FakeOpenAIClient:

    def __init__(self, outcomes: list):
        self.completions = _FlakyCompletions(outcomes)
        self.chat = type('Chat', (), {'completions': self.completions})()

def test_retry_call_success_no_retry():
    with _SleepRecorder() as sr:
        n = {'v': 0}

        def f():
            n['v'] += 1
            return 'ok'
        r = retry_call(f, retries=3, delay=1.0)
    assert r == 'ok' and n['v'] == 1, '成功不重试：1 次调用直接返回'
    assert sr.calls == [], '成功不 sleep'

def test_retry_call_fail_then_success():
    with _SleepRecorder() as sr:
        outcomes = iter([ValueError('e1'), ValueError('e2'), 'ok'])
        n = {'v': 0}

        def f():
            n['v'] += 1
            o = next(outcomes)
            if isinstance(o, Exception):
                raise o
            return o
        r = retry_call(f, retries=3, delay=2.0)
    assert r == 'ok' and n['v'] == 3, '前 N-1 次失败后成功：共 3 次尝试'
    assert sr.calls == [2.0, 2.0], '失败间隔 = delay，共 retries-1 次 sleep'

def test_retry_call_all_fail_raises():
    with _SleepRecorder() as sr:
        n = {'v': 0}

        def f():
            n['v'] += 1
            raise RuntimeError(f"boom-{n['v']}")
        try:
            retry_call(f, retries=3, delay=1.5)
            err = None
        except RuntimeError as e:
            err = e
    assert err is not None and str(err) == 'boom-3', '全部失败：抛出最后一次异常'
    assert n['v'] == 3 and sr.calls == [1.5, 1.5], '全部失败：retries 次尝试、retries-1 次 sleep'

def test_retry_call_retry_on_predicate():
    with _SleepRecorder() as sr:
        n = {'v': 0}

        def f():
            n['v'] += 1
            raise TypeError('不可重试')
        try:
            retry_call(f, retries=3, delay=1.0, retry_on=lambda e: isinstance(e, ValueError))
            err = None
        except TypeError as e:
            err = e
    assert err is not None and n['v'] == 1 and (sr.calls == []), 'retry_on 判定不可重试：立即抛出，不 sleep 不重试'
    with _SleepRecorder() as sr2:
        n2 = {'v': 0}

        def f2():
            n2['v'] += 1
            raise ValueError('可重试')
        try:
            retry_call(f2, retries=2, delay=1.0, retry_on=lambda e: isinstance(e, ValueError))
        except ValueError:
            pass
    assert n2['v'] == 2 and sr2.calls == [1.0], 'retry_on 判定可重试：按 retries 重试'

def test_retry_call_on_retry_override():
    seen = []
    with _SleepRecorder() as sr:
        n = {'v': 0}

        def f():
            n['v'] += 1
            raise RuntimeError('429 限流')

        def on_retry(attempt, e):
            seen.append((attempt, str(e)))
            return 5.0
        try:
            retry_call(f, retries=3, delay=2.0, on_retry=on_retry)
        except RuntimeError:
            pass
    assert sr.calls == [5.0, 5.0], 'on_retry 返回数值覆盖本次间隔（429 特殊间隔）'
    assert seen == [(1, '429 限流'), (2, '429 限流')], 'on_retry 在每次重试前收到 (attempt, e)，最后一次失败不触发'
    with _SleepRecorder() as sr2:
        try:
            retry_call(lambda: (_ for _ in ()).throw(ValueError('x')), retries=2, delay=3.0, on_retry=lambda a, e: None)
        except ValueError:
            pass
    assert sr2.calls == [3.0], 'on_retry 返回 None 时使用 delay'

def test_chat_with_retry_compat():
    with _SleepRecorder() as sr:
        client = _FakeOpenAIClient([RuntimeError('boom'), _FakeChatResponse('hi')])
        r = chat_with_retry(client, model='m', messages=[], max_retries=3, delay=1.0)
    assert r.choices[0].message.content == 'hi' and client.completions.calls == 2, 'chat_with_retry：失败一次后成功'
    assert sr.calls == [1.0], 'chat_with_retry：默认间隔语义不变'
    with _SleepRecorder() as sr2:
        client2 = _FakeOpenAIClient([RuntimeError('boom')])
        try:
            chat_with_retry(client2, model='m', messages=[], max_retries=3, delay=3.0)
            err = None
        except RuntimeError as e:
            err = e
    assert err is not None and client2.completions.calls == 3 and (sr2.calls == [3.0, 3.0]), 'chat_with_retry：耗尽抛异常（report.py 的 3 次/3s 用法）'

def test_judge_retry():
    from llmsec.evaluation import judge as judge_mod
    assert judge_mod.JUDGE_MAX_RETRIES == 2, 'judge：JUDGE_MAX_RETRIES=2 语义不变'
    with _SleepRecorder() as sr:
        client = _FakeOpenAIClient([RuntimeError('boom')])
        judge = judge_mod.Judge(client=client, model='fake-judge')
        try:
            judge._call_judge('sys', 'user')
            err = None
        except RuntimeError as e:
            err = e
    assert err is not None and client.completions.calls == 2, 'judge：2 次尝试后抛出（旧语义 2 次重试）'
    assert sr.calls == [judge_mod.JUDGE_RETRY_DELAY] == [2.0], 'judge：重试间隔 2s'
    with _SleepRecorder():
        client2 = _FakeOpenAIClient([RuntimeError('boom'), _FakeChatResponse(' B ')])
        judge2 = judge_mod.Judge(client=client2, model='fake-judge')
        r = judge2._call_judge('sys', 'user')
    assert r == 'B' and client2.completions.calls == 2, 'judge：失败一次后重试成功，返回值 strip 不变'

def test_openai_backend_retry():
    from llmsec.core.config import TargetConfig
    from llmsec.targets.openai_backend import OpenAITargetClient

    def _client(outcomes):
        c = OpenAITargetClient.__new__(OpenAITargetClient)
        c.config = TargetConfig()
        c.client = _FakeOpenAIClient(outcomes)
        return c
    with _SleepRecorder() as sr:
        c = _client([RuntimeError('boom')])
        result = c.call('hi')
    assert c.client.completions.calls == c.config.max_retries == 3, 'openai_backend：3 次尝试（TargetConfig.max_retries）'
    assert sr.calls == [3.0, 3.0], 'openai_backend：重试间隔 3s'
    assert result['error'] == 'boom' and result['meta']['attempts'] == 3, 'openai_backend：耗尽后 error 字段兜底返回，结构不变'
    with _SleepRecorder():
        c2 = _client([RuntimeError('boom'), _FakeChatResponse('正常内容')])
        result2 = c2.call('hi')
    assert result2['error'] is None and result2['content'] == '正常内容' and (result2['tokens_prompt'] == 10) and (result2['tokens_completion'] == 20), 'openai_backend：重试成功，返回结构/token 字段不变'

def test_pcap_retry():
    import llmsec.targets.pcap as pcap

    class _FakeResp:

        def __init__(self, status_code, payload=None, text=''):
            self.status_code = status_code
            self._payload = payload or {}
            self.text = text

        def json(self):
            return self._payload

    class _HttpMock:

        def __init__(self, responses):
            self.responses = responses
            self.calls = 0

        def post(self, url, json=None, timeout=None, verify=None):
            self.calls += 1
            return self.responses[min(self.calls - 1, len(self.responses) - 1)]
    client = pcap.PcapJudgeTargetClient(url='https://fake.local/judge')
    old_post = pcap.requests.post
    try:
        with _SleepRecorder() as sr:
            mock = _HttpMock([_FakeResp(500, text='Internal Server Error')])
            pcap.requests.post = mock.post
            result = client.call('测试 prompt')
        assert mock.calls == client.max_retries == 3, 'pcap：HTTP 500 重试 3 次'
        assert sr.calls == [3.0, 3.0], 'pcap：重试间隔 3s'
        assert 'HTTP 500' in (result['error'] or '') and result['meta']['status'] == 500, "pcap：500 耗尽后 error='HTTP 500: ...'，结构不变"
        with _SleepRecorder():
            mock = _HttpMock([_FakeResp(429, text='Too Many Requests')])
            pcap.requests.post = mock.post
            client.call('测试 prompt')
        assert mock.calls == 3, 'pcap：HTTP 429 重试 3 次'
        with _SleepRecorder() as sr3:
            mock = _HttpMock([_FakeResp(400, text='Bad Request')])
            pcap.requests.post = mock.post
            result = client.call('测试 prompt')
        assert mock.calls == 1 and sr3.calls == [], 'pcap：HTTP 400 立即返回不重试'
        assert 'HTTP 400' in (result['error'] or ''), "pcap：400 返回结构保持 error='HTTP 400: ...'"
        with _SleepRecorder() as sr4:
            mock_calls = {'n': 0}

            def _post(*a, **kw):
                mock_calls['n'] += 1
                raise ConnectionError('conn refused')
            pcap.requests.post = _post
            result = client.call('测试 prompt')
        assert mock_calls['n'] == 3 and sr4.calls == [3.0, 3.0], 'pcap：网络异常同样 3 次 / 3s 重试'
        assert 'ConnectionError' in (result['error'] or ''), 'pcap：异常耗尽后 error 含异常类型名（旧格式）'
    finally:
        pcap.requests.post = old_post

def test_generate_retry():
    from llmsec.attacks import generate as gen
    from llmsec.params import API_MAX_RETRIES, API_RATE_LIMIT_DELAY, API_RETRY_DELAY
    method = {'method': 'm1', 'category_name': 'c1', 'description': 'd1'}
    harm_types = ['violence']
    with _SleepRecorder() as sr:
        client = _FakeOpenAIClient([RuntimeError('429 rate limit')])
        r = gen.call_api_two_round(client, method, harm_types, model='m')
    assert r is None and client.completions.calls == API_MAX_RETRIES == 3, 'generate：API 失败 3 次尝试后返回 None'
    assert sr.calls == [API_RATE_LIMIT_DELAY] * 2 == [5.0, 5.0], 'generate：API/429 失败间隔 5s'
    with _SleepRecorder() as sr2:
        client2 = _FakeOpenAIClient([_FakeChatResponse('这不是JSON')])
        r2 = gen.call_api_two_round(client2, method, harm_types, model='m')
    assert r2 is None and client2.completions.calls == 3, 'generate：JSON 解析失败 3 次尝试后返回 None'
    assert sr2.calls == [API_RETRY_DELAY] * 2 == [2.0, 2.0], 'generate：JSON 解析失败间隔 2s'
    with _SleepRecorder() as sr3:
        bad = _FakeChatResponse('[{"harm_type":"violence","prompt":"a"},{"harm_type":"violence","prompt":"b"}]')
        client3 = _FakeOpenAIClient([bad])
        r3 = gen.call_api_two_round(client3, method, harm_types, model='m')
    assert r3 is None and client3.completions.calls == 3, 'generate：初稿条数不符按一次失败重试'
    assert sr3.calls == [2.0, 2.0], 'generate：条数不符间隔 2s'
    with _SleepRecorder():
        good = _FakeChatResponse('[{"harm_type":"violence","prompt":"改写后"}]')
        client4 = _FakeOpenAIClient([good, good])
        r4 = gen.call_api_two_round(client4, method, harm_types, model='m')
    assert isinstance(r4, list) and r4[0]['prompt'] == '改写后' and (client4.completions.calls == 2), 'generate：两轮成功路径返回结构不变'

def test_safe_twin_retry():
    from llmsec.evaluation import safe_twin as st
    from llmsec.params import API_MAX_RETRIES, API_RETRY_DELAY
    with _SleepRecorder() as sr:
        client = _FakeOpenAIClient([RuntimeError('boom')])
        r = st.generate_safe_twin('攻击 prompt', client)
    assert r is None and client.completions.calls == API_MAX_RETRIES == 3, 'safe_twin：3 次尝试后返回 None'
    assert sr.calls == [API_RETRY_DELAY] * 2 == [2.0, 2.0], 'safe_twin：重试间隔 2s'
    with _SleepRecorder():
        ok = _FakeChatResponse('{"safe_prompt": "安全版", "replacement": "炸弹→蛋糕"}')
        client2 = _FakeOpenAIClient([RuntimeError('boom'), ok])
        r2 = st.generate_safe_twin('攻击 prompt', client2)
    assert isinstance(r2, dict) and r2['safe_prompt'] == '安全版' and (client2.completions.calls == 2), 'safe_twin：失败一次后重试成功，返回结构不变'
