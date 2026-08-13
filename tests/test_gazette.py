"""文牍存储测试：append/read/context/list。"""

from __future__ import annotations


class TestGazette:
    def test_append_and_read(self, tmp_path, monkeypatch):
        from control.agent import gazette
        monkeypatch.setattr(gazette, "_GAZETTE_DIR", tmp_path / "gazette")
        gazette.reset_gazettes()

        gazette.append_event("p1", gazette.EV_PLAN_DRAFTED, "尚书省",
                             intent="测试意图", session_id="s1",
                             detail={"steps_count": 3})
        gazette.append_event("p1", gazette.EV_PLAN_APPROVED, "用户",
                             session_id="s1", detail={})
        gazette.append_event("p1", gazette.EV_STEP_STARTED, "尚书省",
                             step_id="s1", session_id="s1",
                             detail={"capability": "run_evaluation"})

        events = gazette.read_events("p1")
        assert len(events) == 3
        assert events[0].kind == gazette.EV_PLAN_DRAFTED
        assert events[1].kind == gazette.EV_PLAN_APPROVED
        assert events[2].kind == gazette.EV_STEP_STARTED
        assert events[0].detail["steps_count"] == 3

    def test_read_plan_context(self, tmp_path, monkeypatch):
        """从事件流重建 Plan 上下文：intent + steps 状态 + 封驳历史 + 审查记录。"""
        from control.agent import gazette
        monkeypatch.setattr(gazette, "_GAZETTE_DIR", tmp_path / "gazette")
        gazette.reset_gazettes()

        # 拟案
        gazette.append_event("p1", gazette.EV_PLAN_DRAFTED, "尚书省",
                             intent="跑评估", session_id="s1",
                             detail={"steps_count": 2})
        # 步骤开始 + 被封驳
        gazette.append_event("p1", gazette.EV_STEP_STARTED, "尚书省",
                             step_id="s1", detail={"capability": "run_evaluation",
                                                   "description": "跑A"})
        gazette.append_event("p1", gazette.EV_STEP_BLOCKED, "尚书省",
                             step_id="s1", detail={"capability": "run_evaluation",
                                                   "ticket": {"summary": "危险"}})
        # 另一步成功
        gazette.append_event("p1", gazette.EV_STEP_STARTED, "尚书省",
                             step_id="s2", detail={"capability": "list_workspaces"})
        gazette.append_event("p1", gazette.EV_STEP_SUCCEEDED, "尚书省",
                             step_id="s2", detail={"capability": "list_workspaces"})
        # 审查
        gazette.append_event("p1", gazette.EV_REVIEW_FILED, "门下省",
                             step_id="s2", detail={"run_name": "ws:x/A",
                                                   "digest": "安全等级=safe"})

        ctx = gazette.read_plan_context("p1")
        assert ctx is not None
        assert ctx["intent"] == "跑评估"
        assert ctx["session_id"] == "s1"
        assert ctx["steps"]["s1"]["status"] == "blocked"
        assert ctx["steps"]["s1"]["block_count"] == 1
        assert ctx["steps"]["s2"]["status"] == "done"
        assert len(ctx["blocks"]) == 1
        assert len(ctx["reviews"]) == 1
        assert ctx["reviews"][0]["run_name"] == "ws:x/A"

    def test_list_gazettes(self, tmp_path, monkeypatch):
        from control.agent import gazette
        monkeypatch.setattr(gazette, "_GAZETTE_DIR", tmp_path / "gazette")
        gazette.reset_gazettes()

        gazette.append_event("p1", gazette.EV_PLAN_DRAFTED, "尚书省",
                             intent="任务1", session_id="s1", detail={})
        gazette.append_event("p2", gazette.EV_PLAN_DRAFTED, "尚书省",
                             intent="任务2", session_id="s2", detail={})
        gazette.append_event("p2", gazette.EV_PLAN_FINISHED, "尚书省",
                             session_id="s2", detail={})

        all_plans = gazette.list_gazettes()
        assert len(all_plans) == 2
        # p2 后更新，排前面
        assert all_plans[0]["plan_id"] == "p2"
        assert all_plans[0]["status"] == "finished"

        # 按 session 过滤
        s1_plans = gazette.list_gazettes(session_id="s1")
        assert len(s1_plans) == 1
        assert s1_plans[0]["plan_id"] == "p1"

    def test_read_empty(self, tmp_path, monkeypatch):
        from control.agent import gazette
        monkeypatch.setattr(gazette, "_GAZETTE_DIR", tmp_path / "gazette")
        gazette.reset_gazettes()
        assert gazette.read_events("nonexistent") == []
        assert gazette.read_plan_context("nonexistent") is None
