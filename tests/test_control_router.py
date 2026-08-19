"""control router 测试：端点冒烟 + chat 兜底 + fork/merge 闭环（全沙盒化）。

所有 fork/merge 操作经 monkeypatch 重定向到 tmp_path，绝不碰真实 output/。
库级 clone（control.core.storage）被 mock，不经 subprocess。
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    """把 control 层的路径常量全重定向到 tmp，并 mock export_snapshot（不经 subprocess）。

    这是 control router 测试沙盒化的关键：
      - control.config.{OUTPUT_DIR, WORKSPACES_DIR, LLMSEC_REPO, RUNS_DIR} → tmp
      - workspace.py / compare.py 的模块级路径常量 → tmp
      - storage.backup → mock（在 tmp 内落真实迷你 R 库，不经 subprocess）
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
    import llmsec.core.config as cfg
    monkeypatch.setattr(cfg, "CATALOG_DB", out / "state" / "catalog.db")
    (out / "state").mkdir()

    # mock 库级 clone（P3：fork 直调 backup，无 subprocess 快照握手）
    from control.core import storage as cstorage

    def fake_backup(dest):
        from llmsec.core.results import ResultsMatrix
        from llmsec.storage import rstore
        mat = ResultsMatrix()
        mat.upsert("r1", "test_model", 1.0, ts=1, status="ok")
        rstore.save_matrix(mat, path=dest)

    def fake_stats(path):
        return {"models": ["test_model"], "records": 1, "observations": 1, "units": 0}

    monkeypatch.setattr(cstorage, "backup", fake_backup)
    monkeypatch.setattr(cstorage, "results_stats", fake_stats)
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
        from control.agent.zhongshu import dialogue as zs_mod
        monkeypatch.setattr(zs_mod, "is_llm_configured", lambda: False)
        r = _client().post("/api/control/chat", json={"text": "列一下 run"})
        assert r.status_code == 200
        d = r.json()
        assert d["mode"] == "rule" and d["reply"]

    def test_chat_fallback_on_llm_error(self, monkeypatch):
        from control.agent.zhongshu import dialogue as zs_mod
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
        from control.agent.zhongshu import dialogue as zs_mod
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


# ============================================================
# 封驳解除广播（step_unblocked）：跨标签页/刷新重放的待裁计数配平
# ============================================================
class TestUnblockBusBroadcast:
    """放行 / 驳回清除封驳令时必须广播 step_unblocked——门下省面板据此
    递减待裁计数并翻按钮。此前只在"点按钮的那一页"本地递减，他页放行、
    刷新重放都会让徽标恒卡在「封驳 N 起 · 待圣裁」。
    """

    @pytest.fixture(autouse=True)
    def _sandbox_bus_plan(self, monkeypatch, tmp_path):
        """目录库重定向 tmp + 总线/封驳令/Plan 复位（结束同样复位防泄漏）。"""
        import llmsec.core.config as cfg
        state = tmp_path / "state"
        state.mkdir()
        monkeypatch.setattr(cfg, "CATALOG_DB", state / "catalog.db")

        from control.agent import bus as bus_mod
        from control.agent.menxia import block as block_mod
        from control.agent.menxia import listener
        from control.agent.shangshu import plan as plan_mod
        plan_mod.reset_plans()
        bus_mod.reset_bus()
        block_mod.reset_blocks()
        listener.reinit_menxia()
        yield
        plan_mod.reset_plans()
        bus_mod.reset_bus()
        block_mod.reset_blocks()
        listener.reinit_menxia()

    def _feed_kinds(self, kind):
        from control.agent.bus import get_bus
        return [m for m in get_bus().recent() if m.kind == kind]

    def test_block_approve_emits_step_unblocked(self):
        """放行封驳 → 总线广播 step_unblocked（信封带 plan_id，payload 带 step_id/reason）。"""
        from control.agent.menxia import block as block_mod
        from control.agent.shangshu.plan import Plan, Step, save_plan

        plan = Plan(intent="t", steps=[Step(id="s1", capability="clean_cache", args={})],
                    status="drafted")
        save_plan(plan)
        block_mod.issue_block(plan.id, "s1", "clean_cache", "medium",
                              {"summary": "即将清理缓存", "detail": "可恢复"})

        r = _client().post("/api/control/plan/block/approve",
                           json={"plan_id": plan.id, "step_id": "s1"})
        assert r.status_code == 200
        d = r.json()
        assert d["approved"] is True and d["requeued"] is False

        unb = self._feed_kinds("step_unblocked")
        assert len(unb) == 1
        assert unb[0].plan_id == plan.id, "信封必须带 plan_id（前端据此配对卡片）"
        assert unb[0].payload["step_id"] == "s1"
        assert unb[0].payload["reason"] == "approve"
        assert block_mod.get_block(plan.id, "s1") is None, "放行后封驳令应已清除"

    def test_plan_reject_emits_step_unblocked_per_ticket(self):
        """驳回 Plan 清除全部封驳令 → 每张令各广播一条（reason=reject）。"""
        from control.agent.menxia import block as block_mod
        from control.agent.shangshu.plan import Plan, Step, save_plan

        plan = Plan(intent="t", steps=[
            Step(id="s1", capability="clean_cache", args={}),
            Step(id="s2", capability="delete_runs", args={"names": ["x"]}),
        ], status="drafted")
        save_plan(plan)
        block_mod.issue_block(plan.id, "s1", "clean_cache", "medium",
                              {"summary": "a", "detail": "d"})
        block_mod.issue_block(plan.id, "s2", "delete_runs", "high",
                              {"summary": "b", "detail": "d"})

        r = _client().post("/api/control/plan/reject", json={"plan_id": plan.id})
        assert r.status_code == 200

        unb = self._feed_kinds("step_unblocked")
        assert {m.payload["step_id"] for m in unb} == {"s1", "s2"}
        assert all(m.plan_id == plan.id for m in unb)
        assert all(m.payload["reason"] == "reject" for m in unb)
        assert block_mod.list_blocks_for_plan(plan.id) == [], "驳回后封驳令应全部清除"

    def test_block_approve_404_when_ticket_missing(self):
        """不存在的封驳令 → 404（不广播）。"""
        from control.agent.shangshu.plan import Plan, Step, save_plan
        plan = Plan(intent="t", steps=[Step(id="s1", capability="clean_cache", args={})],
                    status="drafted")
        save_plan(plan)
        r = _client().post("/api/control/plan/block/approve",
                           json={"plan_id": plan.id, "step_id": "s1"})
        assert r.status_code == 404
        assert self._feed_kinds("step_unblocked") == []


# ============================================================
# P1-6 回归：执行期放行封驳不再静默丢失（api 清内存票 + executor 收尾自动重入队）
# ============================================================
class TestExecutingStateApprove:
    @pytest.fixture(autouse=True)
    def _sandbox_bus_plan(self, monkeypatch, tmp_path):
        import llmsec.core.config as cfg
        state = tmp_path / "state"
        state.mkdir()
        monkeypatch.setattr(cfg, "CATALOG_DB", state / "catalog.db")
        from control.agent import bus as bus_mod
        from control.agent.menxia import block as block_mod
        from control.agent.menxia import listener
        from control.agent.shangshu import plan as plan_mod
        plan_mod.reset_plans()
        bus_mod.reset_bus()
        block_mod.reset_blocks()
        listener.reinit_menxia()
        yield
        plan_mod.reset_plans()
        bus_mod.reset_bus()
        block_mod.reset_blocks()
        listener.reinit_menxia()

    def test_executing_approve_clears_memory_ticket(self):
        """executing 态放行：内存 Step.ticket 必须同步置空（executor 重试判据）。

        修复前只清库票、不重入队、内存票残留——该步所在层已过永不重试，
        终态后 404 门再挡住 done 态重入队，放行被静默吞掉。
        """
        from control.agent.menxia import block as block_mod
        from control.agent.shangshu import load_plan
        from control.agent.shangshu.plan import Plan, Step, save_plan

        ticket = {"token": "tk1", "plan_id": "p1", "step_id": "s1"}
        plan = Plan(intent="t", steps=[
            Step(id="s1", capability="clean_cache", args={}, status="blocked", ticket=ticket),
        ], status="executing")
        save_plan(plan)
        block_mod.issue_block(plan.id, "s1", "clean_cache", "medium",
                              {"summary": "a", "detail": "d"})

        r = _client().post("/api/control/plan/block/approve",
                           json={"plan_id": plan.id, "step_id": "s1"})
        assert r.status_code == 200
        assert r.json()["requeued"] is False, "executing 态由 executor 收尾重入队，api 不直接 submit"

        p2 = load_plan(plan.id)
        assert p2.steps[0].ticket is None, "内存票必须同步置空（否则 executor 永不重试）"
        assert p2.steps[0].status == "blocked"

    def test_done_approve_keeps_other_blocked_steps_blocked(self):
        """done 态放行 s1：s2 的封驳令与其 blocked 状态必须原样保留（E-2）。

        旧实现把 s2 也重置 pending 却不清其库票据——重跑时门下省再发新令，
        前端双计数、封驳卡重复渲染。
        """
        from control.agent.menxia import block as block_mod
        from control.agent.shangshu import load_plan
        from control.agent.shangshu.plan import Plan, Step, save_plan

        t2 = {"token": "tk2", "plan_id": "p2", "step_id": "s2"}
        plan = Plan(intent="t", steps=[
            Step(id="s1", capability="clean_cache", args={}, status="blocked",
                 ticket={"token": "tk1", "plan_id": "p2", "step_id": "s1"}),
            Step(id="s2", capability="delete_runs", args={}, status="blocked", ticket=t2),
        ], status="done")
        save_plan(plan)
        # 只为 s1 发库票（放行 s1 需要库行存在）
        block_mod.issue_block(plan.id, "s1", "clean_cache", "medium",
                              {"summary": "a", "detail": "d"})
        submitted = []
        import control.agent.shangshu.queue as queue_mod
        monkey = pytest.MonkeyPatch()
        monkey.setattr(queue_mod.PlanQueue, "submit",
                       lambda self, pid: submitted.append(pid))
        try:
            r = _client().post("/api/control/plan/block/approve",
                               json={"plan_id": plan.id, "step_id": "s1"})
        finally:
            monkey.undo()
        assert r.status_code == 200 and r.json()["requeued"] is True
        assert submitted == [plan.id], "done 态放行应重新入队"

        p2 = load_plan(plan.id)
        assert p2.steps[0].status == "pending" and p2.steps[0].ticket is None
        assert p2.steps[1].status == "blocked" and p2.steps[1].ticket == t2, \
            "未放行的封驳步骤必须保持原状（含内存票）"

    def test_executor_finalize_auto_requeues_in_flight_approval(self, monkeypatch):
        """执行期放行竞态：executor 收尾检测 blocked+ticket-None 自动重入队（不终结为 done）。"""
        from control.agent import bus as bus_mod
        from control.agent.bus import KIND_STEP_BLOCKED
        from control.agent.shangshu import plan as plan_mod
        from control.agent.shangshu.executor import execute_plan
        from control.agent.shangshu.plan import Plan, Step, save_plan

        plan = Plan(intent="t", steps=[
            Step(id="s1", capability="list_runs", args={}),   # low risk，正常放行执行
            Step(id="s2", capability="delete_runs", args={}, depends_on=["s1"]),
        ], status="approved")
        save_plan(plan)

        # 门下省对 s2 封驳（sync 回复）；同时模拟用户在执行期立刻放行 s2：
        # 订阅 step_blocked 的回调同步清内存票（复现 api 放行的竞态注入点）
        def fake_menxia_reply(msg):
            if msg.payload.get("step_id") == "s2" and msg.kind == "step_start":
                return {"plan_id": msg.plan_id, "step_id": "s2",
                        "ticket": {"token": "tk9", "plan_id": msg.plan_id, "step_id": "s2"}}
            return None
        bus_mod.get_bus().subscribe(bus_mod.MENXIA, ["step_start"], fake_menxia_reply)

        def user_approves_immediately(msg):
            if msg.kind == KIND_STEP_BLOCKED:
                p = plan_mod.load_plan(msg.plan_id)
                for s in p.steps:
                    if s.id == msg.payload.get("step_id") and s.status == "blocked":
                        s.ticket = None
        bus_mod.get_bus().subscribe(bus_mod.SHANGSHU, [KIND_STEP_BLOCKED], user_approves_immediately)

        submitted = []
        import control.agent.shangshu.queue as queue_mod
        monkeypatch.setattr(queue_mod.PlanQueue, "submit",
                            lambda self, pid: submitted.append(pid))

        result = execute_plan(plan.id)

        assert result.status == "approved", "收尾检测到执行期放行的步骤必须重入队而非终结"
        assert submitted == [plan.id], "应自动重新提交执行队列"
        s2 = [s for s in result.steps if s.id == "s2"][0]
        assert s2.status == "pending" and s2.ticket is None, "放行步骤应已重置待重跑"
