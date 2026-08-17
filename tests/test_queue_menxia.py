"""Plan 执行队列 + 门下省三阶段监控测试。"""

from __future__ import annotations

import time


# ============================================================
# PlanQueue
# ============================================================
class TestPlanQueue:
    def setup_method(self):
        from control.agent.shangshu.queue import reset_queue
        reset_queue()

    def test_submit_queued(self, tmp_path, monkeypatch):
        from control.agent.shangshu import plan as plan_mod
        from control.agent.shangshu.plan import P_APPROVED, Plan, Step, save_plan
        from control.agent.shangshu.queue import get_queue
        plan_mod.reset_plans()

        # mock execute_plan 避免真正执行
        from control.agent.shangshu import executor
        executed = []
        def fake_execute(plan_id):
            executed.append(plan_id)
            time.sleep(0.1)  # 模拟耗时
        monkeypatch.setattr(executor, "execute_plan", fake_execute)

        plan = Plan(intent="test", steps=[Step(id="s1", capability="list_runs", args={})],
                    status=P_APPROVED)
        save_plan(plan)

        result = get_queue().submit(plan.id)
        assert result == "queued"

        # 等待 worker 执行
        time.sleep(0.5)
        assert plan.id in executed

    def test_duplicate_rejected(self, tmp_path, monkeypatch):
        from control.agent.shangshu import plan as plan_mod
        from control.agent.shangshu.plan import P_APPROVED, Plan, Step, save_plan
        from control.agent.shangshu.queue import get_queue
        plan_mod.reset_plans()

        from control.agent.shangshu import executor
        def slow_execute(plan_id):
            time.sleep(1.0)  # 慢执行，确保第二个 submit 时第一个还在跑
        monkeypatch.setattr(executor, "execute_plan", slow_execute)

        plan = Plan(intent="test", steps=[Step(id="s1", capability="list_runs", args={})],
                    status=P_APPROVED)
        save_plan(plan)

        get_queue().submit(plan.id)
        # 再次提交同一个 → duplicate
        result = get_queue().submit(plan.id)
        assert result == "duplicate"

    def test_status(self, tmp_path, monkeypatch):
        from control.agent.shangshu import plan as plan_mod
        from control.agent.shangshu.plan import P_APPROVED, Plan, Step, save_plan
        from control.agent.shangshu.queue import get_queue
        plan_mod.reset_plans()

        from control.agent.shangshu import executor
        def slow_execute(plan_id):
            time.sleep(1.0)
        monkeypatch.setattr(executor, "execute_plan", slow_execute)

        plan = Plan(intent="test", steps=[Step(id="s1", capability="list_runs", args={})],
                    status=P_APPROVED)
        save_plan(plan)

        st = get_queue().status()
        assert "running" in st and "queued" in st

        get_queue().submit(plan.id)
        time.sleep(0.2)
        st = get_queue().status()
        # worker 正在执行（running 不为 None 或 queued 不为空）
        assert st["running"] is not None or len(st["queued"]) > 0

    def test_cancel_queued(self, tmp_path, monkeypatch):
        from control.agent.shangshu import plan as plan_mod
        from control.agent.shangshu.plan import P_APPROVED, Plan, Step, save_plan
        from control.agent.shangshu.queue import get_queue
        plan_mod.reset_plans()

        from control.agent.shangshu import executor
        blocker = {"wait": True}
        def blocking_execute(plan_id):
            while blocker["wait"]:
                time.sleep(0.05)
        monkeypatch.setattr(executor, "execute_plan", blocking_execute)

        plan1 = Plan(intent="p1", steps=[Step(id="s1", capability="list_runs", args={})],
                     status=P_APPROVED)
        plan2 = Plan(intent="p2", steps=[Step(id="s1", capability="list_runs", args={})],
                     status=P_APPROVED)
        save_plan(plan1)
        save_plan(plan2)

        get_queue().submit(plan1.id)
        time.sleep(0.2)  # 等 plan1 开始执行
        try:
            get_queue().submit(plan2.id)  # plan2 排队

            # 取消排队的 plan2
            assert get_queue().cancel(plan2.id) is True
            # 再次取消 → False（已不在队列）
            assert get_queue().cancel(plan2.id) is False
        finally:
            # 断言失败也必须放行，否则 worker 线程在 blocking_execute 里永久自旋
            blocker["wait"] = False
        time.sleep(0.3)


# ============================================================
# 门下省三阶段监控
# ============================================================
class TestMenxiaThreeStage:
    def setup_method(self):
        from control.agent.bus import reset_bus
        from control.agent.menxia import reset_blocks
        from control.agent.menxia.listener import reinit_menxia
        reset_bus()
        reset_blocks()
        reinit_menxia()

    def test_menxia_subscribes_drafted_and_approved(self):
        """门下省订阅了 plan_drafted 和 plan_approved。"""
        from control.agent.bus import get_bus
        bus = get_bus()
        # 获取订阅列表
        subs = bus._subs
        all_kinds = set()
        for _dept, kinds, _cb in subs:
            all_kinds.update(kinds)
        assert "plan_drafted" in all_kinds
        assert "plan_approved" in all_kinds

    def test_plan_drafted_triggers_review(self, tmp_path, monkeypatch):
        """拟案通知 → 门下省审查 → 发 review 报告（如果有问题）。"""
        from control.agent.bus import (
            KIND_PLAN_DRAFTED,
            KIND_REVIEW,
            ZHONGSHU,
            get_bus,
            notify,
        )
        from control.agent.shangshu import plan as plan_mod
        plan_mod.reset_plans()

        # 构造一个含 critical 步骤的 Plan
        from control.agent.shangshu.plan import P_DRAFTED, Plan, Step, save_plan
        plan = Plan(
            intent="合并到全局",
            steps=[Step(id="s1", capability="merge_results",
                       args={"sources": ["ws:x"], "target": "global", "confirm": True},
                       description="合并到全局")],
            status=P_DRAFTED,
        )
        save_plan(plan)

        # 先订阅 review（门下省发 to_dept=ZHONGSHU），再发通知
        reviews = []
        get_bus().subscribe(ZHONGSHU, [KIND_REVIEW],
                           lambda m: reviews.append(m))

        notify(KIND_PLAN_DRAFTED, from_dept=ZHONGSHU,
               plan_id=plan.id, intent=plan.intent,
               steps_count=1)

        assert len(reviews) >= 1
        assert reviews[0].kind == KIND_REVIEW
        assert "critical" in (reviews[0].payload.get("report", "") or
                              str(reviews[0].payload.get("findings", "")))

    def test_plan_approved_triggers_risk_report(self, tmp_path, monkeypatch):
        """准奏通知 → 门下省发高危步骤报告。"""
        from control.agent.bus import (
            KIND_PLAN_APPROVED,
            KIND_REVIEW,
            ZHONGSHU,
            get_bus,
            notify,
        )
        from control.agent.shangshu import plan as plan_mod
        plan_mod.reset_plans()

        from control.agent.shangshu.plan import P_APPROVED, Plan, Step, save_plan
        plan = Plan(
            intent="跑评估",
            steps=[Step(id="s1", capability="run_evaluation",
                       args={"target": "A", "max_rounds": 5})],
            status=P_APPROVED,
        )
        save_plan(plan)

        reviews = []
        get_bus().subscribe(ZHONGSHU, [KIND_REVIEW],
                           lambda m: reviews.append(m))

        notify(KIND_PLAN_APPROVED, from_dept="用户", plan_id=plan.id,
               intent=plan.intent)

        assert len(reviews) >= 1
        assert "高危" in (reviews[0].payload.get("report", "") or "")
