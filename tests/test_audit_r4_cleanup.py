"""代码审查第 4 轮修复的回归测试：冗余统一 + 死代码/无用参数清理。

覆盖：
  C1  8 个无调用方端点已删（404）
  C2  探活统一走 llmsec.core.probe（dashboard 端点与 MCP 同一实现）
  C3  write_csv 正确处理含逗号字段（_export_matrix 不再手写拼接）
  C4  KMeans 兜底统一为 _kmeans_fallback
  C5  死符号确已删除（函数/常量/再导出）
  C6  compare.list_all_runs 统一口径（target 过滤 + include_workspaces 开关）
"""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# C1：死端点 404
# ============================================================
def test_r4_dead_endpoints_removed():
    from llmsec.server.dashboard_api import app

    client = TestClient(app)
    gone = [
        ("POST", "/api/control/fork-and-run"),
        ("POST", "/api/control/review"),
        ("GET", "/api/control/plan/queue"),
        ("GET", "/api/control/plans"),
        ("GET", "/api/control/blocks"),
        ("GET", "/api/control/env-snapshots"),
        ("POST", "/api/control/env-snapshots"),
        ("DELETE", "/api/control/env-snapshots/x"),
    ]
    for method, url in gone:
        r = client.request(method, url, json={})
        assert r.status_code == 404, f"C1: {method} {url} 应已删除（实得 {r.status_code}）"

    # 存活端点抽查（防止删多）
    for url in ("/api/control/workspaces", "/api/control/bus/feed",
                "/api/control/llm-status", "/api/control/plan/queue-x/status"):
        pass
    assert client.get("/api/control/workspaces").status_code != 404
    assert client.get("/api/control/bus/feed").status_code == 200


# ============================================================
# C2：探活统一实现
# ============================================================
def test_r4_probe_unified(monkeypatch, tmp_path):
    import llmsec.core.probe as probe_mod
    from llmsec.server.dashboard_api import app
    called = []

    def _fake_probe_target(name, cfg):
        called.append(name)
        return {"name": name, "model": "m", "reachable": True,
                "latency_ms": 1, "error": None, "warning": None}

    monkeypatch.setattr(probe_mod, "probe_target", _fake_probe_target)
    # targets 配置隔离
    from llmsec.core import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "load_targets", lambda: {
        "t1": cfg_mod.TargetConfig(model="m", api_key="k", base_url="http://x"),
    })
    client = TestClient(app)
    r = client.get("/api/targets/probe?name=t1")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["targets"] and body["targets"][0]["name"] == "t1"
    assert called == ["t1"], "C2: dashboard 探活应走 llmsec.core.probe 共享实现"


# ============================================================
# C3：write_csv 逗号字段
# ============================================================
def test_r4_write_csv_comma_field(tmp_path):
    import csv

    from llmsec.core.io import write_csv

    path = tmp_path / "matrix.csv"
    write_csv(path, [
        {"method": "harm:other,cat:test", "cluster": 1, "v": 0.5},
        {"method": "plain", "cluster": 2, "v": 1.5},
    ])
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 and rows[0]["method"] == "harm:other,cat:test", \
        "C3: 含逗号方法名须被正确加引号（手写拼接时代会破格式）"
    assert rows[0]["cluster"] == "1" and float(rows[1]["v"]) == 1.5


# ============================================================
# C4：KMeans 兜底统一
# ============================================================
def test_r4_kmeans_fallback_shared():
    import llmsec.pipeline.attack_phase as ap

    assert hasattr(ap, "_kmeans_fallback"), "C4: 兜底应统一为 _kmeans_fallback"
    features = {f"m{i}": {"textual": [float(i), 1.0], "embedding": [0.1 * i, 0.2]}
                for i in range(6)}
    labels = ap._emergency_cluster(
        {m: {"method": m} for m in features}, {"features": features})
    assert labels is not None and set(labels.values()) <= set(range(6))
    assert len(labels) == 6


# ============================================================
# C5：死符号确已删除
# ============================================================
def test_r4_dead_symbols_removed():
    from control.core import env_snapshot
    from llmsec.core import io as core_io
    from llmsec.core import monitoring as core_monitoring
    from llmsec.core import results as core_results
    from llmsec.pipeline import allergy_phase

    assert not hasattr(core_results.ResultsMatrix, "summary")
    assert not hasattr(core_io, "read_csv")
    assert not hasattr(core_monitoring, "shutdown")
    assert not hasattr(allergy_phase, "compute_min_twin_sample_size")
    assert not hasattr(env_snapshot, "get_snapshot")

    import llmsec.core as core
    import llmsec.core.config as cfg
    for name in ("TREE_FILE", "REPORT_FILE", "METHOD_REGISTRY_FILE", "CLUSTER_FEATURES_FILE"):
        assert not hasattr(cfg, name), f"C5: 死常量 {name} 应已删除"
        assert not hasattr(core, name), f"C5: 再导出 {name} 应已删除"

    # invoker 死参数已移除
    import inspect

    from control.core import invoker
    assert "capture" not in inspect.signature(invoker._run).parameters
    assert "extra_argv" not in inspect.signature(invoker.run_runner).parameters
    from control.core.orchestrator import RunSpec
    assert "extra_argv" not in RunSpec.__dataclass_fields__


def test_r4_dead_script_removed():
    assert not (_ROOT / "scripts" / "fix_attack_data.py").exists(), \
        "C5: 一次性修复脚本（数据已修完、无 CI/lint 覆盖）应已删除"


# ============================================================
# C6：list_all_runs 统一口径
# ============================================================
def test_r4_list_all_runs_unified(monkeypatch, tmp_path):
    from control.core import compare as cmp
    from control.core import invoker

    ws_root = tmp_path / "workspaces"
    ws_run = ws_root / "ab1" / "minimax"
    ws_run.mkdir(parents=True)
    (ws_run / "runner_report.json").write_text(
        '{"target_model": "minimax", "security_level": "safe"}', encoding="utf-8")
    monkeypatch.setattr(cmp, "WORKSPACES_DIR", ws_root)

    hist = [{"name": "2026-01-01_000000/minimax", "target_model": "minimax"},
            {"name": "2026-01-02_000000/gemma", "target_model": "gemma"}]

    def _fake_list_runs(*, target=None, since=None, junk_only=False):
        if target:  # 模拟 invoker 的服务端 target 过滤
            return [r for r in hist if r.get("target_model") == target]
        return hist

    monkeypatch.setattr(invoker, "list_runs", _fake_list_runs)

    all_runs = cmp.list_all_runs()
    assert len(all_runs) == 3, "C6: 历史 2 + workspace 1"

    filt = cmp.list_all_runs(target="minimax")
    names = [r["name"] for r in filt]
    assert names == ["2026-01-01_000000/minimax", "ws:ab1/minimax"], \
        f"C6: target 过滤须同时作用于历史与 workspace（实得 {names}）"

    no_ws = cmp.list_all_runs(include_workspaces=False)
    assert all(not r["name"].startswith("ws:") for r in no_ws)
