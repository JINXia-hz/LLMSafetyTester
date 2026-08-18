"""pipeline.probe / core.probe 探测测试：路由分发 / 字段扫描 / 各错误分支 /
PCAP TLS 校验开关（全 mock，零网络）。"""

from types import SimpleNamespace


# ============================================================
# PCAP_VERIFY_TLS：三处请求点（targets/pcap、core/probe、pipeline/probe）
# 共用同一开关，默认 False（内网自签名），置 1/true 开启校验
# ============================================================
def test_pcap_verify_tls_env_parsing(monkeypatch):
    """pcap_verify_tls()：缺省/假值 → False；1/true/yes/on（大小写不敏感）→ True。"""
    import llmsec.targets.pcap as pc

    monkeypatch.delenv("PCAP_VERIFY_TLS", raising=False)
    assert pc.pcap_verify_tls() is False
    for val in ("0", "", "false", "no", "off"):
        monkeypatch.setenv("PCAP_VERIFY_TLS", val)
        assert pc.pcap_verify_tls() is False, f"{val!r} 应为 False"
    for val in ("1", "true", "YES", "On"):
        monkeypatch.setenv("PCAP_VERIFY_TLS", val)
        assert pc.pcap_verify_tls() is True, f"{val!r} 应为 True"


def test_probe_target_pcap_respects_verify_tls(monkeypatch):
    """core.probe.probe_target pcap 分支：verify 传参跟随 PCAP_VERIFY_TLS。"""
    from types import SimpleNamespace as NS

    import llmsec.core.probe as cp
    import llmsec.targets.pcap as pc

    monkeypatch.setattr("llmsec.targets.target_backend", lambda name: "pcap_judge")
    monkeypatch.setattr(cp, "models_list", lambda *a, **kw: (1.0, []))
    seen = {}

    def _get(url, timeout=None, verify=None):
        seen["verify"] = verify
        return NS(status_code=200, raise_for_status=lambda: None)

    monkeypatch.setattr("requests.get", _get)
    cfg = NS(model="pcap-m")

    monkeypatch.setenv("PCAP_JUDGE_URL", "http://fake-judge")
    monkeypatch.delenv("PCAP_VERIFY_TLS", raising=False)
    monkeypatch.setattr(pc, "_warning_suppressed", False)
    out = cp.probe_target("t", cfg)
    assert out["reachable"] is True and seen["verify"] is False

    monkeypatch.setenv("PCAP_VERIFY_TLS", "1")
    out = cp.probe_target("t", cfg)
    assert out["reachable"] is True and seen["verify"] is True


def test_probe_pcap_script_respects_verify_tls(monkeypatch):
    """pipeline.probe.probe_pcap：requests.post 的 verify 传参跟随开关。"""
    import llmsec.pipeline.probe as pb
    import llmsec.targets.pcap as pc

    monkeypatch.setattr(pb, "build_pcap_payload", lambda text, strip_math: {"q": text})
    monkeypatch.setattr(pb, "pcap_judge_url", lambda: "http://fake-judge")
    seen = {}
    monkeypatch.setattr(
        pb.requests, "post",
        lambda url, json=None, timeout=None, verify=None: (
            seen.__setitem__("verify", verify),
            _fake_resp(json_data={"text": "x"}),
        )[1],
    )

    monkeypatch.delenv("PCAP_VERIFY_TLS", raising=False)
    monkeypatch.setattr(pc, "_warning_suppressed", False)
    pb.probe_pcap("t")
    assert seen["verify"] is False

    monkeypatch.setenv("PCAP_VERIFY_TLS", "true")
    pb.probe_pcap("t")
    assert seen["verify"] is True


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
