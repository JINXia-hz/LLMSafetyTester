"""第 7 轮审计回归——服务端/看板/MCP（M-3 / M-8 / M-9 / M-10 / M-11 / M-12）。

  - M-3: pcap_judge 探活必须探 PCAP_JUDGE_URL（与评估端点一致），而非 cfg.base_url。
  - M-8: run_evaluation 的 workspace/work_dir_name 走 safe_component（防穿越）。
  - M-9: 取消排队任务 / 子进程启动失败后推进同 kind 队列。
  - M-10: .env 写入——写前备份旧内容、跨进程锁、值按 dotenv 规则加引号。
  - M-11: 聚类可视化/units 加载的 CPU 重计算放 to_thread，不阻塞事件循环。
  - M-12: _tail_text 按 limit 定位读，full_log 不再恒截 4KB。
"""

from __future__ import annotations

import asyncio
import threading
import types

import pytest

# ============================================================
# M-3: pcap 探活端点
# ============================================================

class _Resp:
    def raise_for_status(self):
        pass


def test_probe_pcap_uses_pcap_judge_url(monkeypatch):
    """pcap 后端探活必须打 PCAP_JUDGE_URL，不得打 OpenAI 型 cfg.base_url。"""
    import requests as requests_mod

    import llmsec.core.probe as probe_mod
    import llmsec.targets as tgt
    import llmsec.targets.pcap as pcap_mod

    monkeypatch.setattr(tgt, "target_backend", lambda name: "pcap_judge")
    monkeypatch.setattr(pcap_mod, "pcap_judge_url", lambda: "http://pcap-judge:9000")

    called = {}

    def fake_get(url, **kw):
        called["url"] = url
        return _Resp()
    monkeypatch.setattr(requests_mod, "get", fake_get)

    fake_cfg = types.SimpleNamespace(api_key="k", base_url="http://openai-style/v1", model="m")
    result = probe_mod.probe_target("p1", fake_cfg)

    assert called.get("url") == "http://pcap-judge:9000", (
        "M-3：探活必须探实际评估端点 PCAP_JUDGE_URL，而非 cfg.base_url")
    assert result["reachable"] is True


def test_probe_pcap_unconfigured(monkeypatch):
    """PCAP_JUDGE_URL 未配置时明确报不可达。"""
    import llmsec.core.probe as probe_mod
    import llmsec.targets as tgt
    import llmsec.targets.pcap as pcap_mod

    monkeypatch.setattr(tgt, "target_backend", lambda name: "pcap_judge")
    monkeypatch.setattr(pcap_mod, "pcap_judge_url", lambda: "")

    fake_cfg = types.SimpleNamespace(api_key="k", base_url="http://openai-style/v1", model="m")
    result = probe_mod.probe_target("p1", fake_cfg)
    assert result["reachable"] is False
    assert "PCAP_JUDGE_URL" in result["error"]


# ============================================================
# M-8: run_evaluation 路径校验
# ============================================================

class TestRunEvaluationPathValidation:
    @pytest.fixture(autouse=True)
    def _dirs(self, tmp_path, monkeypatch):
        import control.config as cconfig
        import control.core.invoker as inv

        ws_root = tmp_path / "workspaces"
        (ws_root / "legit").mkdir(parents=True)
        out_root = tmp_path / "output"
        out_root.mkdir()
        monkeypatch.setattr(cconfig, "WORKSPACES_DIR", ws_root)
        monkeypatch.setattr(cconfig, "OUTPUT_DIR", out_root)

        self.started = []
        monkeypatch.setattr(inv, "run_runner", lambda wd, **kw: self.started.append(wd) or types.SimpleNamespace(
            returncode=0, ok=True, elapsed_s=0.0, stdout="", stderr=""))
        self.ws_root = ws_root
        self.out_root = out_root

    def test_workspace_traversal_rejected(self):
        from control.agent.shangshu.capabilities import _h_run_evaluation

        with pytest.raises(ValueError):
            _h_run_evaluation({"workspace": "../escape"})
        with pytest.raises(ValueError):
            _h_run_evaluation({"workspace": "a/b"})
        assert not self.started

    def test_work_dir_name_traversal_rejected(self):
        from control.agent.shangshu.capabilities import _h_run_evaluation

        with pytest.raises(ValueError):
            _h_run_evaluation({"work_dir_name": "../evil"})
        assert not self.started

    def test_legit_workspace_passes(self):
        from control.agent.shangshu.capabilities import _h_run_evaluation

        _h_run_evaluation({"workspace": "legit"})
        assert self.started == [self.ws_root / "legit"], (
            "合法 workspace 必须解析为 WORKSPACES_DIR 内的路径")

    def test_generated_dir_stays_inside(self):
        from control.agent.shangshu.capabilities import _h_run_evaluation

        _h_run_evaluation({"work_dir_name": "eval_x"})
        assert self.started[0] == self.out_root / "eval_runs" / "eval_x"


# ============================================================
# M-9: 任务队列推进
# ============================================================

@pytest.fixture
def _task_env(tmp_path, monkeypatch):
    import llmsec.server.task_manager as tm

    tm.TASKS.clear()
    monkeypatch.setattr(tm, "TASK_LOG_DIR", tmp_path / "logs")
    yield tm
    tm.TASKS.clear()


def _mk_task(tm, tid, kind="r7"):
    return {"kind": kind, "cmd": "x", "argv": ["x"], "env_override": None,
            "meta": None, "proc": None, "log_path": tm.TASK_LOG_DIR / f"{tid}.log",
            "log_file": None, "status": "queued", "started_at": "now",
            "_task_id": tid}


def test_cancel_queued_advances_queue(_task_env, monkeypatch):
    """取消队头排队任务（无同 kind running）后，下一个 queued 必须被拉起。"""
    tm = _task_env
    tm.TASKS["q1"] = _mk_task(tm, "q1")
    tm.TASKS["q2"] = _mk_task(tm, "q2")

    spawned = []

    def fake_spawn(tid, t):
        spawned.append(tid)
        t["status"] = "running"
    monkeypatch.setattr(tm, "_spawn", fake_spawn)

    view = tm.cancel_task("q1")
    assert view["status"] == "cancelled"
    assert spawned == ["q2"], (
        "M-9：取消队头 queued 任务后必须推进同 kind 队列——修复前 q2 永久滞留排队")


def test_spawn_oserror_advances_queue(_task_env, monkeypatch):
    """子进程启动失败（Popen OSError）不得让后续 queued 任务永久搁浅。"""
    tm = _task_env
    tm.TASKS["q1"] = _mk_task(tm, "q1")
    tm.TASKS["q2"] = _mk_task(tm, "q2")

    def boom_popen(*a, **kw):
        raise OSError("spawn denied (simulated)")
    monkeypatch.setattr(tm.subprocess, "Popen", boom_popen)

    tm._advance_queue("r7")

    assert tm.TASKS["q1"]["status"] == "failed"
    assert tm.TASKS["q2"]["status"] == "failed", (
        "M-9：启动失败的队列必须继续推进——修复前 q2 会永久滞留 queued")


# ============================================================
# M-10: .env 写入（备份时机 / 引号）
# ============================================================

@pytest.fixture
def _env_env(tmp_path, monkeypatch):
    import llmsec.core.config as cfg

    monkeypatch.setattr(cfg, "PROJECT_ROOT", tmp_path)
    out_dir = tmp_path / "output"
    monkeypatch.setattr(cfg, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(cfg, "load_env", lambda: None)

    # 端点会写 os.environ（TARGET_*/TARGETS），测试后恢复
    import os
    before = {k: v for k, v in os.environ.items() if k == "TARGETS" or k.startswith("TARGET_")}
    yield tmp_path
    for k in list(os.environ):
        if k == "TARGETS" or (k.startswith("TARGET_") and k not in before):
            os.environ.pop(k, None)
    os.environ.update(before)


def test_targets_add_pre_write_backup_and_quoting(_env_env):
    from llmsec.server.routers.data_query import AddTargetRequest, api_targets_add

    tmp_path = _env_env
    env = tmp_path / ".env"
    old_content = "TARGETS=t1\nTARGET_1_NAME=t1\nTARGET_1_MODEL=m1\n"
    env.write_text(old_content, encoding="utf-8")

    req = AddTargetRequest(name="new1", model="nm", base_url="http://x/#frag",
                           api_key="sk 123")
    res = asyncio.run(api_targets_add(req))
    assert res["ok"] is True

    new_content = env.read_text(encoding="utf-8")
    # 写前备份（.env 同目录）必须是旧内容——回滚手段
    assert (tmp_path / ".env.bak").read_text(encoding="utf-8") == old_content, (
        "M-10：.env.bak 应是写前旧内容（原先备的是新内容，旧配置不可恢复）")
    # 值按 dotenv 规则加引号 + round-trip 解析不截断
    assert 'TARGET_2_BASE_URL="http://x/#frag"' in new_content
    from dotenv import dotenv_values
    vals = dotenv_values(env)
    assert vals["TARGET_2_BASE_URL"] == "http://x/#frag"
    assert vals["TARGET_2_API_KEY"] == "sk 123"
    # output 卷的 .env.bak 是新内容（docker entrypoint 恢复语义，保持不变）
    out_bak = tmp_path / "output" / ".env.bak"
    assert out_bak.read_text(encoding="utf-8") == new_content


def test_update_env_vars_pre_write_backup_and_quoting(_env_env):
    import os

    from llmsec.server.routers.data_query import _update_env_vars

    tmp_path = _env_env
    env = tmp_path / ".env"
    old_content = "JUDGE_MODEL=old-model\n# 注释保留\n"
    env.write_text(old_content, encoding="utf-8")

    saved = os.environ.get("JUDGE_MODEL")
    try:
        _update_env_vars({"JUDGE_MODEL": "new model #tag"})
    finally:
        if saved is None:
            os.environ.pop("JUDGE_MODEL", None)
        else:
            os.environ["JUDGE_MODEL"] = saved

    assert (tmp_path / ".env.bak").read_text(encoding="utf-8") == old_content
    from dotenv import dotenv_values
    vals = dotenv_values(env)
    assert vals["JUDGE_MODEL"] == "new model #tag", "含 # 的值必须加引号防截断"


# ============================================================
# M-11: 重计算放 to_thread
# ============================================================

def test_cluster_endpoints_offload_to_thread(monkeypatch):
    """投影/树/切割的计算函数必须跑在事件循环之外的线程。"""
    import llmsec.server.routers.cluster_viz as cv

    main_thread = threading.get_ident()
    worker_threads = {}

    def fake_proj(method):
        worker_threads["proj"] = threading.get_ident()
        return {"available": False}

    def fake_tree():
        worker_threads["tree"] = threading.get_ident()
        return {"available": False}

    def fake_cut(k):
        worker_threads["cut"] = threading.get_ident()
        return {"available": False}

    monkeypatch.setattr(cv, "_compute_projection", fake_proj)
    monkeypatch.setattr(cv, "_compute_tree", fake_tree)
    monkeypatch.setattr(cv, "_compute_cut", fake_cut)

    asyncio.run(cv.api_cluster_projection("pca"))
    asyncio.run(cv.api_cluster_tree())
    asyncio.run(cv.api_cluster_cut(k=3))

    assert set(worker_threads) == {"proj", "tree", "cut"}
    for name, tid in worker_threads.items():
        assert tid != main_thread, f"{name} 计算必须离开事件循环线程（M-11）"


# ============================================================
# M-12: _tail_text 按 limit 读取
# ============================================================

def test_tail_text_respects_limit(tmp_path):
    from llmsec.tui.task_store import _tail_text

    p = tmp_path / "t.log"
    p.write_text("A" * 10240, encoding="utf-8")

    full = _tail_text(p, limit=2_000_000)
    assert len(full) == 10240, "M-12：full_log（大 limit）必须拿到完整日志而非恒 4KB"
    assert len(_tail_text(p)) == 4000, "默认 4KB tail 口径保持不变"
