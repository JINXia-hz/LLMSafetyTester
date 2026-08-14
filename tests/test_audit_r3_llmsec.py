"""代码审查第 3 轮修复的回归测试：llmsec/ 逻辑隐患 + 前端配套。

覆盖：
  L1  data_query._validate_run 逐段校验（路径穿越拒绝）
  L2  hpo study 名 safe_component 校验（yaml 写穿越拒绝）
  L3  search.tell 失败时归还在飞 trial（队列状态不丢）
  L4  io.write_jsonl 并发覆写不损坏（tmp 名带 pid/tid）
  L5  task_view log_tail 只读尾部（大日志不全量读也能取到末尾内容）
  L6  control/api.py 500 响应不回传内部异常文本
  L7  local_model_server / report / features 模块 docstring 恢复
  L8  前端轮询卸载钩子存在（unload*Section 定义 + core.js 接线）
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient


# ============================================================
# L1：_validate_run 防穿越
# ============================================================
def test_r3_validate_run_rejects_traversal(monkeypatch, tmp_path):
    import llmsec.server.routers.data_query as dq
    from llmsec.server import dashboard_api

    monkeypatch.setattr(dashboard_api, "RUNS_DIR", tmp_path)
    (tmp_path / "2026-01-01_000000").mkdir()

    # 合法值放行
    assert dq._validate_run("2026-01-01_000000") == "2026-01-01_000000"
    assert dq._validate_run("2026-01-01_000000/gemma") == "2026-01-01_000000/gemma"

    for bad in ("2026-01-01_000000/../../x", "2026-01-01_000000/..",
                "not-a-ts"):
        with pytest.raises(HTTPException) as ei:
            dq._validate_run(bad)
        assert ei.value.status_code == 400, f"L1: {bad!r} 应被拒绝"


# ============================================================
# L2：hpo study 名校验
# ============================================================
_HPO_BODY = {
    "name": "safe-name",
    "targets": ["t1"],
    "space": {},
    "fixed": {},
    "strategy": "bayesian",
}


def test_r3_hpo_name_traversal_rejected(monkeypatch, tmp_path):
    from llmsec.server.dashboard_api import app
    from llmsec.server.routers import hpo as hpo_mod

    monkeypatch.setattr(hpo_mod, "OUTPUT_DIR", tmp_path)
    started: list[list] = []
    # hpo 已直调 task_manager.start_task（tasks 别名层已删），注入点随之迁移
    monkeypatch.setattr(hpo_mod.task_manager, "start_task",
                        lambda kind, argv: started.append(argv))

    client = TestClient(app)
    r = client.post("/api/run/hpo", json=_HPO_BODY | {"name": "../../evil"})
    assert r.status_code == 400, f"L2: 穿越 study 名应 400（实得 {r.status_code}）"
    assert not started

    r2 = client.post("/api/run/hpo", json=_HPO_BODY)
    assert r2.status_code == 200
    assert started and str(tmp_path) in " ".join(started[0]), \
        "L2: 合法 study 应写到受控 experiments 目录内"


# ============================================================
# L3：search.tell 失败归还 trial
# ============================================================
def test_r3_search_tell_failure_returns_trial():
    from llmsec.experiments.search import BayesianSearch

    class _BoomStudy:
        def tell(self, trial, value):
            raise ValueError("optuna rejected")

    eng = object.__new__(BayesianSearch)
    eng._study = _BoomStudy()
    eng._param_order = ["x"]
    eng._pending = {}

    trial = object()
    from collections import deque
    eng._pending[eng._key({"x": 1})] = deque([trial])

    with pytest.raises(ValueError):
        eng.tell({"x": 1}, 0.5)

    q = eng._pending.get(eng._key({"x": 1}))
    assert q and q[0] is trial, "L3: tell 失败后应归还在飞 trial（队列状态不丢）"


# ============================================================
# L4：write_jsonl 并发覆写
# ============================================================
def test_r3_write_jsonl_concurrent_no_corruption(tmp_path):
    from llmsec.core.io import read_jsonl, write_jsonl

    path = tmp_path / "shared.jsonl"

    def _work(w):
        for i in range(20):
            write_jsonl(path, [{"w": w, "i": i, "pad": "x" * 200}])

    threads = [threading.Thread(target=_work, args=(w,)) for w in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    rows = list(read_jsonl(path))
    assert len(rows) == 1, "L4: 每次覆写都是完整单行，最终文件应是某次完整写入"
    assert not list(tmp_path.glob("shared.jsonl.tmp.*")), "L4: 不应残留 tmp"


# ============================================================
# L5：log_tail 尾部读取
# ============================================================
def test_r3_task_view_log_tail_reads_end(tmp_path):
    from llmsec.server import task_manager

    log = tmp_path / "big.log"
    # 生成 ~2MB 日志，末行是哨兵
    with open(log, "w", encoding="utf-8") as f:
        for i in range(20000):
            f.write(f"line {i} padding-padding-padding\n")
        f.write("SENTINEL_LAST_LINE\n")

    tid = "smoke-r3-tail"
    task_manager.TASKS[tid] = {
        "kind": "smoke", "cmd": "x", "argv": [], "proc": None,
        "log_path": log, "log_file": None,
        "status": "success", "returncode": 0,
        "started_at": "2026-01-01T00:00:00",
    }
    try:
        view = task_manager.task_view(tid)
        assert len(view["log_tail"]) <= 4000
        assert "SENTINEL_LAST_LINE" in view["log_tail"], \
            "L5: 尾部读取必须覆盖日志末行"
    finally:
        task_manager.TASKS.pop(tid, None)


# ============================================================
# L6：control router 500 脱敏
# ============================================================
def test_r3_control_500_sanitized(monkeypatch):
    from control.core import workspace as ws_mod
    from llmsec.server.dashboard_api import app

    def _boom():
        raise RuntimeError("secret internal path C:/Users/x/keys")

    monkeypatch.setattr(ws_mod, "list_workspaces", _boom)
    client = TestClient(app)
    r = client.get("/api/control/workspaces")
    assert r.status_code == 500
    assert "secret internal path" not in r.text, \
        "L6: 内部异常文本不得回传客户端"
    assert "处理失败" in r.text


# ============================================================
# L7：模块 docstring 恢复（import 曾在 docstring 之前使 __doc__ 为 None）
# ============================================================
def test_r3_module_docstrings_present():
    import llmsec.clustering.features as features
    import llmsec.reporting.report as report
    import llmsec.server.local_model_server as lms

    assert features.__doc__ and "特征" in features.__doc__
    assert report.__doc__ and "报告" in report.__doc__
    assert lms.__doc__ and "模拟" in lms.__doc__


# ============================================================
# L8：前端轮询卸载钩子
# ============================================================
def test_r3_frontend_poll_unload_wiring():
    root = Path(__file__).resolve().parents[1]

    menxia = (root / "llmsec/server/static/js/menxia.js").read_text(encoding="utf-8")
    shangshu = (root / "llmsec/server/static/js/shangshu.js").read_text(encoding="utf-8")
    control = (root / "llmsec/server/static/js/control.js").read_text(encoding="utf-8")
    core = (root / "llmsec/server/static/js/core.js").read_text(encoding="utf-8")

    assert "function unloadMenxiaSection" in menxia and "clearInterval(_mxPollTimer)" in menxia
    assert "function unloadShangshuSection" in shangshu and "clearInterval(_pollTimer)" in shangshu
    assert "function unloadControlSection" in control
    assert "unloadControlSection" in core, "L8: core.js 离开宣政殿时须调用卸载钩子"
