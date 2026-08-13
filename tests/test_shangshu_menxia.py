"""三省新架构测试：bus + plan + capabilities + env_snapshot + menxia + executor。

覆盖：
  - bus：发布/订阅/留存/过滤
  - plan：拓扑分层 + 依赖传播 + 持久化
  - capabilities：清单完整 + handler 可调
  - env_snapshot：CRUD + 编辑 + merge
  - menxia：封驳判据 + block 管理
  - executor：简单 plan 执行 + 封驳拦截 + 依赖传播
"""

from __future__ import annotations

import time

import pytest


# ============================================================
# 消息总线
# ============================================================
class TestBus:
    def setup_method(self):
        from control.agent.bus import get_bus, reset_bus
        reset_bus()
        self.bus = get_bus()

    def test_publish_subscribe(self):
        from control.agent.bus import ALL, BusMessage
        received = []
        self.bus.subscribe("门下省", ["step_start"], lambda m: received.append(m))
        msg = BusMessage(from_dept="尚书省", to_dept=ALL, kind="step_start", payload={"x": 1})
        self.bus.publish(msg)
        assert len(received) == 1
        assert received[0].payload["x"] == 1

    def test_dept_filter(self):
        from control.agent.bus import MENXIA, ZHONGSHU, BusMessage
        menxia_got = []
        zhongshu_got = []
        self.bus.subscribe(MENXIA, [], lambda m: menxia_got.append(m))
        self.bus.subscribe(ZHONGSHU, [], lambda m: zhongshu_got.append(m))
        self.bus.publish(BusMessage(from_dept="尚书省", to_dept=MENXIA, kind="x", payload={}))
        self.bus.publish(BusMessage(from_dept="尚书省", to_dept=ZHONGSHU, kind="x", payload={}))
        assert len(menxia_got) == 1 and len(zhongshu_got) == 1

    def test_broadcast(self):
        from control.agent.bus import ALL, MENXIA, ZHONGSHU, BusMessage
        got = []
        self.bus.subscribe(MENXIA, [], lambda m: got.append("menxia"))
        self.bus.subscribe(ZHONGSHU, [], lambda m: got.append("zhongshu"))
        self.bus.publish(BusMessage(from_dept="尚书省", to_dept=ALL, kind="x", payload={}))
        assert "menxia" in got and "zhongshu" in got

    def test_kind_filter(self):
        from control.agent.bus import ALL, BusMessage
        got = []
        self.bus.subscribe("门下省", ["step_start"], lambda m: got.append(m))
        self.bus.publish(BusMessage(from_dept="尚书省", to_dept=ALL, kind="step_done", payload={}))
        self.bus.publish(BusMessage(from_dept="尚书省", to_dept=ALL, kind="step_start", payload={}))
        assert len(got) == 1  # 只收到 step_start

    def test_recent_since(self):
        from control.agent.bus import ALL, BusMessage
        self.bus.publish(BusMessage(from_dept="x", to_dept=ALL, kind="a", payload={}))
        t1 = self.bus.recent()[-1].ts
        # 确保 ts 严格递增（Windows 时钟精度问题）
        time.sleep(0.01)
        self.bus.publish(BusMessage(from_dept="x", to_dept=ALL, kind="b", payload={}))
        msgs = self.bus.recent(since_ts=t1)
        assert len(msgs) == 1 and msgs[0].kind == "b"

    def test_subscriber_exception_isolated(self):
        """订阅者回调异常不阻断总线。"""
        from control.agent.bus import ALL, BusMessage
        ok = []
        self.bus.subscribe("A", [], lambda m: (_ for _ in ()).throw(ValueError("boom")))
        self.bus.subscribe("B", [], lambda m: ok.append(m))
        self.bus.publish(BusMessage(from_dept="x", to_dept=ALL, kind="k", payload={}))
        assert len(ok) == 1  # B 仍收到


# ============================================================
# Plan 数据结构
# ============================================================
class TestPlan:
    def setup_method(self):
        from control.agent.shangshu import plan
        plan.reset_plans()

    def test_topological_layers_simple(self):
        from control.agent.shangshu.plan import Plan, Step
        plan = Plan(steps=[
            Step(id="s1", capability="x", args={}),
            Step(id="s2", capability="y", args={}, depends_on=["s1"]),
            Step(id="s3", capability="z", args={}, depends_on=["s2"]),
        ])
        layers = plan.topological_layers()
        assert len(layers) == 3
        assert layers[0][0].id == "s1"
        assert layers[1][0].id == "s2"
        assert layers[2][0].id == "s3"

    def test_topological_layers_parallel(self):
        from control.agent.shangshu.plan import Plan, Step
        plan = Plan(steps=[
            Step(id="s1", capability="x", args={}),
            Step(id="s2", capability="y", args={}),  # 无依赖，与 s1 同层
            Step(id="s3", capability="z", args={}, depends_on=["s1", "s2"]),
        ])
        layers = plan.topological_layers()
        assert len(layers) == 2
        assert {s.id for s in layers[0]} == {"s1", "s2"}
        assert layers[1][0].id == "s3"

    def test_save_load_plan(self, tmp_path, monkeypatch):
        from control.agent.shangshu import plan as plan_mod
        monkeypatch.setattr(plan_mod, "_PLANS_DIR", tmp_path / "plans")
        plan_mod._PLANS_DIR.mkdir(parents=True, exist_ok=True)
        plan_mod.reset_plans()

        from control.agent.shangshu.plan import Plan, Step, load_plan, save_plan
        p = Plan(intent="测试", steps=[
            Step(id="s1", capability="list_runs", args={}, description="查")
        ])
        save_plan(p)
        # 内存命中
        assert load_plan(p.id) is not None
        # 清内存后磁盘命中
        plan_mod.reset_plans()
        loaded = load_plan(p.id)
        assert loaded is not None
        assert loaded.intent == "测试"
        assert len(loaded.steps) == 1

    def test_make_plan_from_llm_auto_id(self):
        from control.agent.shangshu.plan import make_plan_from_llm
        p = make_plan_from_llm("意图", [
            {"capability": "list_runs", "args": {}},
            {"capability": "list_workspaces", "args": {}},
        ])
        assert len(p.steps) == 2
        assert p.steps[0].id == "s1"
        assert p.steps[1].id == "s2"


# ============================================================
# Capabilities 清单
# ============================================================
class TestCapabilities:
    def test_all_capabilities_have_required_fields(self):
        from control.agent.shangshu.capabilities import all_capabilities
        for c in all_capabilities():
            assert c.name, "capability 缺 name"
            assert c.description, f"{c.name} 缺 description"
            assert c.handler is not None, f"{c.name} 缺 handler"
            assert c.risk_level in ("low", "medium", "high", "critical"), f"{c.name} risk_level 非法"
            assert c.parameters.get("type") == "object", f"{c.name} parameters 不是 object"

    def test_expected_capabilities_exist(self):
        from control.agent.shangshu.capabilities import all_capabilities
        names = {c.name for c in all_capabilities()}
        expected = {
            "run_evaluation", "run_batch_experiment", "fork_workspace",
            "merge_results", "delete_runs", "clean_cache",
            "create_env_snapshot", "edit_env_snapshot", "list_env_snapshots",
            "delete_env_snapshot", "merge_env_to_global",
            "list_runs", "compare_runs", "list_workspaces", "delete_workspace",
            "request_review",
        }
        assert expected <= names, f"缺失: {expected - names}"

    def test_capability_by_name(self):
        from control.agent.shangshu.capabilities import capability_by_name
        c = capability_by_name("list_runs")
        assert c is not None and c.name == "list_runs"
        assert capability_by_name("不存在") is None


# ============================================================
# env_snapshot
# ============================================================
class TestEnvSnapshot:
    def _setup(self, monkeypatch, tmp_path):
        """统一设置：ENV_SNAPSHOTS_DIR + _GLOBAL_ENV + LLMSEC_REPO 都指向 tmp。"""
        from control.core import env_snapshot
        snaps = tmp_path / "env_snaps"
        snaps.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(env_snapshot, "ENV_SNAPSHOTS_DIR", snaps)
        monkeypatch.setattr(env_snapshot, "_GLOBAL_ENV", tmp_path / ".env")
        monkeypatch.setattr(env_snapshot, "LLMSEC_REPO", tmp_path)
        return env_snapshot

    def test_create_from_blank(self, tmp_path, monkeypatch):
        env_snapshot = self._setup(monkeypatch, tmp_path)
        info = env_snapshot.create("snap1", source="blank")
        assert info["name"] == "snap1"
        assert info["keys"] == []
        snaps = env_snapshot.list_snapshots()
        assert any(s["name"] == "snap1" for s in snaps)

    def test_create_from_global(self, tmp_path, monkeypatch):
        env_snapshot = self._setup(monkeypatch, tmp_path)
        global_env = tmp_path / ".env"
        global_env.write_text("TARGETS=A,B\nJUDGE_MODEL=deepseek\nGENERATOR_API_KEY=sk-x\n", encoding="utf-8")
        info = env_snapshot.create("snap_g", source="global")
        assert "TARGETS" in info["keys"]
        assert "JUDGE_MODEL" in info["keys"]

    def test_edit_key(self, tmp_path, monkeypatch):
        env_snapshot = self._setup(monkeypatch, tmp_path)
        env_snapshot.create("s", source="blank")
        result = env_snapshot.edit_key("s", "TARGETS", "X,Y,Z")
        assert "TARGETS" in result["keys"]
        d = env_snapshot.load_env_dict("s")
        assert d["TARGETS"] == "X,Y,Z"

    def test_edit_disallowed_key(self, tmp_path, monkeypatch):
        env_snapshot = self._setup(monkeypatch, tmp_path)
        env_snapshot.create("s", source="blank")
        with pytest.raises(ValueError, match="不允许"):
            env_snapshot.edit_key("s", "RANDOM_KEY", "bad")

    def test_delete(self, tmp_path, monkeypatch):
        env_snapshot = self._setup(monkeypatch, tmp_path)
        env_snapshot.create("s", source="blank")
        result = env_snapshot.delete("s")
        assert result["deleted"] == "s"
        assert env_snapshot.get_snapshot("s") is None

    def test_merge_to_global(self, tmp_path, monkeypatch):
        env_snapshot = self._setup(monkeypatch, tmp_path)
        global_env = tmp_path / ".env"
        global_env.write_text("TARGETS=A\nKEEP=this\n", encoding="utf-8")
        env_snapshot.create("s", source="blank")
        env_snapshot.edit_key("s", "TARGETS", "X,Y,Z")
        result = env_snapshot.merge_to_global("s")
        assert "TARGETS" in result["changed_keys"]
        d = env_snapshot._parse_env(global_env.read_text(encoding="utf-8"))
        assert d["TARGETS"] == "X,Y,Z"
        assert d["KEEP"] == "this"


# ============================================================
# 门下省封驳判据
# ============================================================
class TestMenxiaAssess:
    """封驳判据测试——基于 capability.block_message（数据化判据）。"""
    def _cap(self, name):
        from control.agent.shangshu.capabilities import capability_by_name
        return capability_by_name(name)

    def test_critical_merge_to_global(self):
        from control.agent.menxia import assess_step
        cap = self._cap("merge_results")
        a = assess_step(cap, {"target": "global", "sources": ["ws:x"]})
        assert a is not None
        assert "全局 R" in a["summary"]

    def test_merge_to_ws_also_blocks(self):
        """merge target=ws 也封驳（分支融合）。"""
        from control.agent.menxia import assess_step
        cap = self._cap("merge_results")
        a = assess_step(cap, {"target": "ws:other", "sources": ["ws:x"]})
        assert a is not None and "ws:other" in a["summary"]

    def test_critical_delete_r(self):
        from control.agent.menxia import assess_step
        cap = self._cap("delete_runs")
        a = assess_step(cap, {"delete_r": True, "names": ["m1"]})
        assert a is not None and "R 矩阵" in a["summary"]

    def test_delete_runs_without_r(self):
        from control.agent.menxia import assess_step
        cap = self._cap("delete_runs")
        a = assess_step(cap, {"delete_r": False, "names": ["m1"]})
        assert a is not None and "删除" in a["summary"]

    def test_high_run_evaluation(self):
        from control.agent.menxia import assess_step
        cap = self._cap("run_evaluation")
        a = assess_step(cap, {"targets": ["A"], "max_rounds": 5})
        assert a is not None and "评估" in a["summary"]

    def test_medium_clean_cache(self):
        from control.agent.menxia import assess_step
        cap = self._cap("clean_cache")
        a = assess_step(cap, {"categories": ["elo_cache"]})
        assert a is not None and "elo_cache" in a["summary"]

    def test_low_passthrough(self):
        """low capability（block_message=None）永不封驳。"""
        from control.agent.menxia import assess_step
        assert assess_step(self._cap("list_runs"), {}) is None
        assert assess_step(self._cap("list_workspaces"), {}) is None
        assert assess_step(self._cap("fork_workspace"), {}) is None

    def test_block_lifecycle(self):
        from control.agent.menxia import approve_block, get_block, issue_block, reset_blocks
        reset_blocks()
        issue_block("p1", "s1", "run_evaluation", "high", {"summary": "s", "detail": "d"})
        assert get_block("p1", "s1") is not None
        assert approve_block("p1", "s1") is True
        assert get_block("p1", "s1") is None
        reset_blocks()


# ============================================================
# Executor（封驳 + 依赖传播）
# ============================================================
class TestExecutor:
    def setup_method(self):
        from control.agent.bus import get_bus, reset_bus
        from control.agent.menxia import reinit_menxia, reset_blocks
        from control.agent.shangshu import plan as plan_mod
        plan_mod.reset_plans()
        reset_bus()
        reset_blocks()
        reinit_menxia()  # 重新订阅新总线（reset_bus 创建了新实例）
        self.bus = get_bus()

    def test_simple_plan_executes(self, tmp_path, monkeypatch):
        """简单 plan（无封驳）执行成功。"""
        from control.agent import gazette
        from control.agent.shangshu import plan as plan_mod
        monkeypatch.setattr(plan_mod, "_PLANS_DIR", tmp_path / "plans")
        monkeypatch.setattr(gazette, "_GAZETTE_DIR", tmp_path / "gazette")
        plan_mod._PLANS_DIR.mkdir(parents=True, exist_ok=True)
        plan_mod.reset_plans()

        from control.agent.shangshu import executor
        from control.agent.shangshu.plan import P_APPROVED, Plan, Step, save_plan

        # mock capability handler（low risk，不封驳）
        call_log = []
        original_by_name = executor.caps_mod.capability_by_name
        def mock_by_name(name):
            c = original_by_name(name)
            if c is None:
                return None
            def fake_handler(args):
                call_log.append(name)
                return {"ok": True}
            # 返回一个 handler 被 mock 的副本
            from control.agent.shangshu.capabilities import Capability
            return Capability(name=c.name, description=c.description,
                            parameters=c.parameters, handler=fake_handler,
                            risk_level=c.risk_level, doc=c.doc)
        monkeypatch.setattr(executor.caps_mod, "capability_by_name", mock_by_name)

        p = Plan(intent="测", steps=[
            Step(id="s1", capability="list_workspaces", args={}, description="列工作区"),
        ], status=P_APPROVED)
        save_plan(p)

        result = executor.execute_plan(p.id)
        assert result.status == "done"
        assert result.steps[0].status == "done"
        assert "list_workspaces" in call_log

    def test_blocked_step_skips_dependents(self, tmp_path, monkeypatch):
        """被封驳的步骤，其依赖者标 skipped。"""
        from control.agent import gazette
        from control.agent.shangshu import plan as plan_mod
        monkeypatch.setattr(plan_mod, "_PLANS_DIR", tmp_path / "plans")
        monkeypatch.setattr(gazette, "_GAZETTE_DIR", tmp_path / "gazette")
        plan_mod._PLANS_DIR.mkdir(parents=True, exist_ok=True)
        plan_mod.reset_plans()

        from control.agent import menxia
        from control.agent.shangshu import executor
        from control.agent.shangshu.plan import P_APPROVED, Plan, Step, save_plan

        # 发一个封驳令，让 run_evaluation 步被封
        menxia.reset_blocks()

        p = Plan(intent="测", steps=[
            Step(id="s1", capability="run_evaluation", args={"target": "A"}, description="跑评估"),
            Step(id="s2", capability="list_runs", args={}, description="查", depends_on=["s1"]),
        ], status=P_APPROVED)
        save_plan(p)

        result = executor.execute_plan(p.id)
        assert result.status == "done"
        assert result.steps[0].status == "blocked"  # s1 被门下省封驳
        assert result.steps[1].status == "skipped"   # s2 因 s1 blocked 而 skipped

    def test_non_dependent_continues_after_block(self, tmp_path, monkeypatch):
        """被封驳步骤的不依赖者继续执行。"""
        from control.agent import gazette
        from control.agent.shangshu import plan as plan_mod
        monkeypatch.setattr(plan_mod, "_PLANS_DIR", tmp_path / "plans")
        monkeypatch.setattr(gazette, "_GAZETTE_DIR", tmp_path / "gazette")
        plan_mod._PLANS_DIR.mkdir(parents=True, exist_ok=True)
        plan_mod.reset_plans()

        from control.agent import menxia
        from control.agent.shangshu import executor
        from control.agent.shangshu.plan import P_APPROVED, Plan, Step, save_plan

        menxia.reset_blocks()

        p = Plan(intent="测", steps=[
            Step(id="s1", capability="run_evaluation", args={"target": "A"}, description="跑评估"),
            Step(id="s2", capability="list_workspaces", args={}, description="查工作区"),  # 不依赖 s1
        ], status=P_APPROVED)
        save_plan(p)

        result = executor.execute_plan(p.id)
        assert result.steps[0].status == "blocked"   # s1 封驳
        assert result.steps[1].status == "done"      # s2 照常完成
