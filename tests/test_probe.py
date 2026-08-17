"""pipeline.probe 探测脚本测试：路由分发 / 字段扫描 / 各错误分支（全 mock，零网络）。"""

from types import SimpleNamespace


def test_probe_routes_by_target_type(monkeypatch):
    """probe()：TARGET_TYPE=pcap_judge → probe_pcap；其余 → probe_openai。"""
    import llmsec.pipeline.probe as pb

    calls = []
    monkeypatch.setattr(pb, "probe_pcap", lambda text: calls.append(("pcap", text)))
    monkeypatch.setattr(pb, "probe_openai", lambda text, tt: calls.append(("openai", tt)))

    monkeypatch.setenv("TARGET_TYPE", "pcap_judge")
    pb.probe("t1")
    monkeypatch.setenv("TARGET_TYPE", "openai")
    pb.probe("t2")
    monkeypatch.setenv("TARGET_TYPE", "local_sim")
    pb.probe("t3")
    assert calls == [("pcap", "t1"), ("openai", "openai"), ("openai", "local_sim")]


def test_probe_openai_success_and_error(monkeypatch, caplog):
    """openai 探测：成功打印响应；调用失败打印错误不抛；target_refused 有告警。"""
    import logging

    import llmsec.pipeline.probe as pb

    monkeypatch.setattr(pb, "call_target", lambda text: {
        "content": "好的，这是回复。", "error": None, "latency_ms": 12.3,
        "tokens_prompt": 5, "tokens_completion": 9, "meta": {"backend": "fake"},
    })
    with caplog.at_level(logging.INFO, logger="llmsec.pipeline.probe"):
        pb.probe_openai("hi", "openai")
    assert any("连接成功" in r.message for r in caplog.records)

    caplog.clear()
    monkeypatch.setattr(pb, "call_target", lambda text: {
        "content": "", "error": "conn refused", "latency_ms": 1.0,
        "tokens_prompt": 0, "tokens_completion": 0, "meta": {},
    })
    with caplog.at_level(logging.INFO, logger="llmsec.pipeline.probe"):
        pb.probe_openai("hi", "openai")
    assert any("调用失败" in r.message for r in caplog.records)

    caplog.clear()
    monkeypatch.setattr(pb, "call_target", lambda text: {
        "content": "拒答", "error": None, "latency_ms": 1.0,
        "tokens_prompt": 1, "tokens_completion": 1, "meta": {}, "target_refused": True,
    })
    with caplog.at_level(logging.INFO, logger="llmsec.pipeline.probe"):
        pb.probe_openai("hi", "openai")
    assert any("target_refused" in r.message for r in caplog.records)


def _fake_resp(status=200, json_data=None, text="raw-text"):
    def _json():
        if json_data is None:
            raise ValueError("no json")
        return json_data

    return SimpleNamespace(status_code=status, text=text, headers={"Content-Type": "application/json"},
                           json=_json)


def test_probe_pcap_json_and_error_branches(monkeypatch, caplog):
    """pcap 探测：正常 JSON 响应走字段扫描；SSL/连接/超时错误分支各有提示。"""
    import logging

    import requests

    import llmsec.pipeline.probe as pb

    monkeypatch.setattr(pb, "build_pcap_payload", lambda text, strip_math: {"q": text})
    monkeypatch.setattr(pb, "pcap_judge_url", lambda: "http://fake-judge")

    # 正常 JSON 响应
    monkeypatch.setattr(pb.requests, "post",
                        lambda *a, **kw: _fake_resp(json_data={"choices": [{"message": {"content": "x" * 30}}]}))
    with caplog.at_level(logging.INFO, logger="llmsec.pipeline.probe"):
        pb.probe_pcap("t")
    assert any("字段扫描" in r.message for r in caplog.records)

    # 非 JSON 响应
    caplog.clear()
    monkeypatch.setattr(pb.requests, "post", lambda *a, **kw: _fake_resp(json_data=None))
    with caplog.at_level(logging.INFO, logger="llmsec.pipeline.probe"):
        pb.probe_pcap("t")
    assert any("不是有效 JSON" in r.message for r in caplog.records)

    # 三类网络错误分支
    def _mk_raise(exc):
        def _raise(*a, **kw):
            raise exc
        return _raise

    for exc, hint in (
        (requests.exceptions.SSLError("bad cert"), "SSL 错误"),
        (requests.exceptions.ConnectionError("refused"), "连接失败"),
        (requests.exceptions.Timeout("t"), "请求超时"),
    ):
        caplog.clear()
        monkeypatch.setattr(pb.requests, "post", _mk_raise(exc))
        with caplog.at_level(logging.INFO, logger="llmsec.pipeline.probe"):
            pb.probe_pcap("t")  # 不应抛出
        assert any(hint in r.message for r in caplog.records), f"{hint} 分支未走到"


def test_scan_for_text_fields_recursion(caplog):
    """字段扫描：dict 递归、长/短字符串分档、列表只下钻首元素。"""
    import logging

    import llmsec.pipeline.probe as pb

    data = {
        "short": "abc",
        "long": "x" * 25,
        "nested": {"inner": "y" * 30},
        "choices": [{"a": "first"}, {"a": "second"}],
        "empty_list": [],
    }
    with caplog.at_level(logging.INFO, logger="llmsec.pipeline.probe"):
        pb.scan_for_text_fields(data)
    msgs = [r.message for r in caplog.records]
    assert any("short" in m and "abc" in m for m in msgs), "短字段应打印"
    assert any("long" in m for m in msgs), "长字段应打印（截断形式）"
    assert any("nested.inner" in m for m in msgs), "嵌套 dict 应带路径下钻"
    assert any("choices[0].a" in m for m in msgs), "列表只下钻首元素"
    assert not any("second" in m for m in msgs), "列表其余元素不扫"
