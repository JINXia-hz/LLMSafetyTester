"""targets/base + local_sim + pipeline/probe + clustering/cli 基础覆盖。"""


def test_base_error_result_and_abstract():
    """TargetClient._error_result 构造标准失败 dict；类为抽象不可实例化。"""
    from llmsec.targets.base import TargetClient

    r = TargetClient._error_result("boom", {"backend": "x"})
    assert r["error"] == "boom" and r["content"] == "" and r["latency_ms"] == 0
    assert r["target_refused"] is False and r["meta"]["backend"] == "x"
    assert TargetClient.__abstractmethods__, "TargetClient 应有抽象方法 call"
    print("✅ targets.base 通过")


def test_local_sim_client():
    """LocalSimTargetClient 继承 OpenAI 后端，仅 backend_name 标 local_sim。"""
    from llmsec.targets.local_sim import LocalSimTargetClient
    from llmsec.targets.openai_backend import OpenAITargetClient

    assert issubclass(LocalSimTargetClient, OpenAITargetClient)
    assert LocalSimTargetClient.backend_name == "local_sim"
    print("✅ targets.local_sim 通过")


def test_probe_routing(monkeypatch):
    """probe() 按 TARGET_TYPE 路由到 probe_openai / probe_pcap（不发网络请求）。"""
    import llmsec.pipeline.probe as probe_mod

    calls = {}
    monkeypatch.setattr(probe_mod, "probe_openai", lambda text, tt="": calls.setdefault("openai", text))
    monkeypatch.setattr(probe_mod, "probe_pcap", lambda text: calls.setdefault("pcap", text))

    monkeypatch.setenv("TARGET_TYPE", "openai")
    probe_mod.probe("hello")
    assert "openai" in calls and "pcap" not in calls

    monkeypatch.setenv("TARGET_TYPE", "pcap_judge")
    probe_mod.probe("hello")
    assert "pcap" in calls, "TARGET_TYPE=pcap_judge 应路由到 probe_pcap"
    print("✅ probe 路由通过")


def test_probe_scan_for_text_fields():
    """scan_for_text_fields 纯函数递归扫描，不抛。"""
    from llmsec.pipeline.probe import scan_for_text_fields

    scan_for_text_fields({"a": {"b": "x" * 30}})
    scan_for_text_fields([{"k": "short"}, {"k": "y" * 40}])
    scan_for_text_fields({"n": 123, "empty": ""})
    print("✅ probe scan_for_text_fields 通过")


def test_clustering_cli_smoke():
    """clustering.cli 可导入且有 main 入口（CLI 整合聚类，重流程不做实跑）。"""
    import llmsec.clustering.cli as cli

    assert callable(getattr(cli, "main", None)), "clustering.cli 应有 main 入口"
    print("✅ clustering.cli 冒烟通过")
