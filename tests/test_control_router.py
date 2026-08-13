"""control router 测试：端点冒烟 + chat 兜底 + fork/merge 闭环（全沙盒化）。

所有 fork/merge 操作经 monkeypatch 重定向到 tmp_path，绝不碰真实 output/。
export_snapshot（走 subprocess）被 mock，避免 subprocess 不受 monkeypatch 影响的问题。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    """把 control 层的路径常量全重定向到 tmp，并 mock export_snapshot（不经 subprocess）。

    这是 control router 测试沙盒化的关键：
      - control.config.{OUTPUT_DIR, WORKSPACES_DIR, LLMSEC_REPO, RUNS_DIR} → tmp
      - workspace.py / compare.py 的模块级路径常量 → tmp
      - invoker.export_snapshot → mock（返回 tmp 内的假快照，不经 subprocess）
    这样 fork/list/compare/merge 全在 tmp 下，不碰真实 output/。
    """
    from control import config as ctrl_cfg
    out = tmp_path / "output"
    out.mkdir()
    (out / "workspaces").mkdir()
    (out / "runs").mkdir()

    monkeypatch.setattr(ctrl_cfg, "OUTPUT_DIR", out)
    monkeypatch.setattr(ctrl_cfg, "WORKSPACES_DIR", out / "workspaces")
    monkeypatch.setattr(ctrl_cfg, "RUNS_DIR", out / "runs")
    monkeypatch.setattr(ctrl_cfg, "LLMSEC_REPO", tmp_path)

    # workspace.py 持有这些常量的模块级引用（import 时绑定）
    from control.core import workspace as ws
    monkeypatch.setattr(ws, "WORKSPACES_DIR", out / "workspaces")
    monkeypatch.setattr(ws, "LLMSEC_REPO", tmp_path)
    # compare.py
    from control.core import compare as cmp
    monkeypatch.setattr(cmp, "RUNS_DIR", out / "runs")
    monkeypatch.setattr(cmp, "WORKSPACES_DIR", out / "workspaces")
    # merge.py（经 management）
    from llmsec.management import merge as merge_mod
    monkeypatch.setattr(merge_mod, "WORKSPACES_DIR", out / "workspaces")
    monkeypatch.setattr(merge_mod, "OUTPUT_DIR", out)
    monkeypatch.setattr(merge_mod, "RESULTS_FILE", out / "state" / "results.json")
    (out / "state").mkdir()

    # mock export_snapshot：在 tmp 下造假快照（不经 subprocess）
    def fake_export(source="global", out=None):
        snap_dir = out if out else (ctrl_cfg.OUTPUT_DIR / "snapshots" / "fake")
        Path(snap_dir).mkdir(parents=True, exist_ok=True)
        # 写一个最小 R
        R = {"version": 2, "units": [], "models": ["test_model"],
             "results": {"r1": {"test_model": {"eval_score": 1.0, "status": "ok", "ts": 1, "extra": {}}}}}
        Path(snap_dir, "results.json").write_text(json.dumps(R), encoding="utf-8")
        return {"snapshot": str(Path(snap_dir).relative_to(ctrl_cfg.OUTPUT_DIR)).replace("\\", "/"),
                "models": ["test_model"], "records": 1}

    from control.core import invoker
    monkeypatch.setattr(invoker, "export_snapshot", fake_export)
    monkeypatch.setattr(ws, "export_snapshot", fake_export)
    return out


def _client():
    from llmsec.server.dashboard_api import app
    return TestClient(app)


# ============================================================
# 端点冒烟（不需沙盒，只读 + 结构校验）
# ============================================================
class TestControlRouterSmoke:
    def test_llm_status(self):
        r = _client().get("/api/control/llm-status")
        assert r.status_code == 200
        assert "configured" in r.json()

    def test_tools_list(self):
        r = _client().get("/api/control/tools")
        assert r.status_code == 200
        names = {t["function"]["name"] for t in r.json()["tools"]}
        # 中书省保留 6 个查询/管理工具（执行类在 /api/control/capabilities）
        assert {"list_runs", "compare_runs", "fork_workspace", "list_workspaces",
                "delete_workspace", "review_run"} <= names
        assert "orchestrate" not in names  # 执行类已移至尚书省 capabilities
        assert "merge" not in names
        for t in r.json()["tools"]:
            assert t["type"] == "function" and "parameters" in t["function"]

    def test_index_has_control_section(self):
        r = _client().get("/")
        assert r.status_code == 200
        assert 'data-section="control"' in r.text
        assert 'id="sec-control"' in r.text
        assert "control.js" in r.text


# ============================================================
# chat 兜底（mock LLM）—— 新中书省流程（zhongshu.handle_message）
# ============================================================
class TestChatFallback:
    def test_chat_rule_mode_when_unconfigured(self, monkeypatch):
        from control.agent import zhongshu as zs_mod
        monkeypatch.setattr(zs_mod, "is_llm_configured", lambda: False)
        r = _client().post("/api/control/chat", json={"text": "列一下 run"})
        assert r.status_code == 200
        d = r.json()
        assert d["mode"] == "rule" and d["reply"]

    def test_chat_fallback_on_llm_error(self, monkeypatch):
        from control.agent import zhongshu as zs_mod
        monkeypatch.setattr(zs_mod, "is_llm_configured", lambda: True)

        def boom(messages, **k):
            raise RuntimeError("LLM 连不上")
        monkeypatch.setattr(zs_mod, "chat_with_tools", boom)
        r = _client().post("/api/control/chat", json={"text": "列工作区"})
        assert r.status_code == 200
        d = r.json()
        assert d["mode"] == "fallback"
        assert "LLM 连不上" in d["llm_error"] and d["reply"]

    def test_chat_llm_mode_with_mocked_reply(self, monkeypatch):
        """中书省 LLM 模式：mock chat_with_tools 返回纯文本回复（无 tool call）。"""
        from control.agent import zhongshu as zs_mod
        monkeypatch.setattr(zs_mod, "is_llm_configured", lambda: True)

        class FakeMsg:
            content = "当前没有工作区。"
            tool_calls = None
        class FakeResp:
            choices = [type("C", (), {"message": FakeMsg()})()]

        def fake_chat(messages, **k):
            return FakeResp()
        monkeypatch.setattr(zs_mod, "chat_with_tools", fake_chat)
        r = _client().post("/api/control/chat", json={"text": "有哪些工作区"})
        assert r.status_code == 200
        d = r.json()
        assert d["mode"] == "llm" and "工作区" in d["reply"]


# ============================================================
# fork/merge 闭环（全沙盒化，不碰真实 output）
# ============================================================
class TestForkMergeFlow:
    def test_fork_then_list_then_delete(self, sandbox):
        c = _client()
        r = c.post("/api/control/fork", json={"name": "ws1", "source": "global"})
        assert r.status_code == 200
        assert r.json()["name"] == "ws1"
        assert r.json()["records"] == 1  # fake snapshot 的 1 条
        # list 含它
        r = c.get("/api/control/workspaces")
        assert "ws1" in [w["name"] for w in r.json()["workspaces"]]
        # delete
        r = c.delete("/api/control/workspaces/ws1")
        assert r.status_code == 200 and r.json()["deleted"] == "ws1"
        # list 不再含
        r = c.get("/api/control/workspaces")
        assert "ws1" not in [w["name"] for w in r.json()["workspaces"]]

    def test_fork_duplicate_400(self, sandbox):
        c = _client()
        assert c.post("/api/control/fork", json={"name": "dup", "source": "global"}).status_code == 200
        r = c.post("/api/control/fork", json={"name": "dup", "source": "global"})
        assert r.status_code == 400  # FileExistsError → 400
        c.delete("/api/control/workspaces/dup")

    def test_merge_dry_run_isolated(self, sandbox):
        """merge dry-run 在沙盒内，不碰真实全局 R。"""
        c = _client()
        c.post("/api/control/fork", json={"name": "mrg", "source": "global"})
        # 给全局 R 造个文件（merge target = global）
        from control import config as ctrl_cfg
        global_r = ctrl_cfg.OUTPUT_DIR / "state" / "results.json"
        global_r.write_text(json.dumps({"version": 2, "units": [], "models": [], "results": {}}), encoding="utf-8")
        r = c.post("/api/control/merge", json={"sources": ["ws:mrg"], "target": "global", "confirm": False})
        assert r.status_code == 200
        d = r.json()
        assert d["dry_run"] is True
        assert "per_model" in d["extra"]
        c.delete("/api/control/workspaces/mrg")

    def test_compare_needs_two_runs(self):
        r = _client().post("/api/control/compare", json={"runs": ["only-one"]})
        assert r.status_code == 400
