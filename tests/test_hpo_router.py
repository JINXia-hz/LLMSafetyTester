"""server.routers.hpo 路由测试（HPO 配置台 API：params / preview / run）。"""

from fastapi.testclient import TestClient

from llmsec.server.dashboard_api import app

client = TestClient(app)


def test_hpo_params():
    r = client.get("/api/hpo/params")
    assert r.status_code == 200
    params = r.json()["params"]
    assert len(params) > 0
    assert all("name" in p and "group" in p and "type" in p for p in params)
    names = {p["name"] for p in params}
    assert "K_FACTOR" in names and "sampler" in names
    print("✅ /api/hpo/params 通过")


def test_hpo_preview_grid():
    """grid 策略：n_configs = 笛卡尔积；repeats=1 → n_trials = n_configs。"""
    body = {
        "name": "prev_ut",
        "strategy": "grid",
        "max_trials": 100,
        "repeats": 1,
        "space": {
            "K_FACTOR": {"type": "float", "low": 16, "high": 32, "step": 16},        # 2 值
            "sampler": {"type": "categorical", "choices": ["hybrid", "gap"]},        # ×2 → 4
        },
    }
    r = client.post("/api/hpo/preview", json=body)
    assert r.status_code == 200
    d = r.json()
    assert d["n_configs"] == 4, "grid 笛卡尔积 = 2×2"
    assert d["n_trials"] == 4
    assert "est_method_calls" in d and isinstance(d["warnings"], list)
    print("✅ /api/hpo/preview 通过")


def test_hpo_preview_no_factors_warns():
    """空 space → warnings 提示、n_configs=1（仅 fixed 跑 repeats 次）。"""
    r = client.post("/api/hpo/preview", json={"name": "x", "strategy": "bayesian",
                                              "max_trials": 3, "repeats": 2})
    assert r.status_code == 200
    d = r.json()
    assert d["n_configs"] == 3 and d["n_trials"] == 6  # bayesian: n_configs=max_trials, ×repeats
    assert any("因子" in w for w in d["warnings"]), "空 space 应告警"
    print("✅ /api/hpo/preview 空因子告警通过")


def test_run_hpo_starts_task(monkeypatch, tmp_path):
    """/api/run/hpo：写 study.yaml + 以 hpo kind 启动任务（monkeypatch _start_task 防真起子进程）。"""
    import llmsec.server.routers.hpo as hpo_mod

    monkeypatch.setattr(hpo_mod, "OUTPUT_DIR", tmp_path)
    captured = {}

    def fake_start(kind, argv):
        captured["kind"] = kind
        captured["argv"] = list(argv)
        return {"id": "fake-hpo", "kind": kind, "cmd": " ".join(argv), "argv": list(argv),
                "status": "queued", "returncode": None, "log_path": tmp_path / "h.log",
                "log_file": None, "started_at": "2026-01-01T00:00:00", "error": None, "proc": None}

    monkeypatch.setattr(hpo_mod, "_start_task", fake_start)

    r = client.post("/api/run/hpo", json={"name": "utstudy", "strategy": "bayesian", "max_trials": 5,
                                          "targets": ["modelA"]})
    assert r.status_code == 200, r.text
    assert captured["kind"] == "hpo"
    yamls = list((tmp_path / "experiments").glob("_dashboard_utstudy.yaml"))
    assert len(yamls) == 1, "应落盘 study.yaml 供 experiments run 读取"
    print("✅ /api/run/hpo 通过")


def test_run_hpo_no_targets_rejected():
    """无 targets 且 fixed 无 target → 400（防空转 study：前端漏传目标时的防线）。"""
    r = client.post("/api/run/hpo", json={"name": "notarget", "strategy": "bayesian", "max_trials": 5})
    assert r.status_code == 400, r.text
    assert "目标" in r.json()["detail"]
    print("✅ /api/run/hpo 空目标 400 通过")


def test_run_hpo_fixed_target_ok(monkeypatch, tmp_path):
    """fixed.target 兜底：无 targets 列表但 fixed 带 target 时允许启动。"""
    import llmsec.server.routers.hpo as hpo_mod

    monkeypatch.setattr(hpo_mod, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(hpo_mod, "_start_task",
                        lambda kind, argv: {"id": "fake", "kind": kind, "cmd": "", "argv": argv,
                                            "status": "queued", "returncode": None,
                                            "log_path": tmp_path / "h.log", "log_file": None,
                                            "started_at": "2026-01-01T00:00:00", "error": None, "proc": None})

    r = client.post("/api/run/hpo", json={"name": "fixedtgt", "strategy": "random", "max_trials": 2,
                                          "fixed": {"target": "modelB", "input": "l1.jsonl"}})
    assert r.status_code == 200, r.text
    print("✅ /api/run/hpo fixed.target 兜底通过")
