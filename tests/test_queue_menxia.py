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

    def test_submit_duplicate_and_running_rerun(self, tmp_path, monkeypatch):
        """排队中重复提交 → duplicate；运行中提交 → queued（E-2 重跑语义）。

        E-2 之前 running 中的 plan 再提交恒被 "duplicate" 拒绝——executor 收尾
        自动重入队（执行期放行）与 api 的 done 态补入队都发生在这个窗口，
        放行被静默吞掉。新语义：当前轮结束后重跑一轮（排队行与 running 行并存）。
        """
        from control.agent.shangshu import plan as plan_mod
        from control.agent.shangshu.plan import P_APPROVED, Plan, Step, save_plan
        from control.agent.shangshu.queue import PlanQueue, get_queue
        plan_mod.reset_plans()

        plan = Plan(intent="test", steps=[Step(id="s1", capability="list_runs", args={})],
                    status=P_APPROVED)
        save_plan(plan)

        # 场景1（排队中重复提交）：不启 worker，plan 稳定留在 deque——
        # 旧写法靠 sleep 与 worker 抢跑，天然竞态
        monkeypatch.setattr(PlanQueue, "_ensure_worker", lambda self: None)
        q = get_queue()
        assert q.submit(plan.id) == "queued"
        assert q.submit(plan.id) == "duplicate"

        # 场景2（运行中提交）：worker 仍持 _running——E-2 修复后是"重跑"而非拒绝
        q._queue.clear()
        with q._lock:
            q._running = plan.id
        try:
            assert q.submit(plan.id) == "queued", "运行中重提交必须是重跑语义（E-2）"
            assert plan.id in q._queue
        finally:
            with q._lock:
                q._running = None
                q._queue.clear()

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

    def test_restart_recovers_queued_plans(self, tmp_path, monkeypatch):
        """P9/B2 重启恢复：ctl_queue 的 queued 行在新 PlanQueue 构造时回填
        （此前只写不读，重启即静默丢队列）。E-6：崩溃遗留的 running 行也重置回
        queued 一并恢复（executor 按 step.status 跳过已 done 步骤，重跑代价可控）。"""
        import threading

        from control.agent.shangshu.queue import PlanQueue
        from control.core.storage import (
            enqueue_plan,
            finish_queue_item,
            mark_queue_running,
            pending_queue_plans,
            reset_queue,
        )

        reset_queue()
        enqueue_plan("plan-a")
        enqueue_plan("plan-b")
        mark_queue_running("plan-a")   # 崩溃时在跑
        enqueue_plan("plan-c")
        # E-6：running 行回收为 queued，按入队序（plan-a 先入队故排最前）
        assert pending_queue_plans() == ["plan-a", "plan-b", "plan-c"]
        # 回收后行状态已重置（幂等再读仍是三行 queued）
        assert pending_queue_plans() == ["plan-a", "plan-b", "plan-c"]

        # fake_execute 阻塞到断言完成——worker 构造即开始消费恢复的队列，
        # 直接读 q._queue 会与 popleft 竞态；按 status() 终态判定
        released = threading.Event()

        def _fake_execute(plan_id):
            released.wait(timeout=5)

        monkeypatch.setattr("control.agent.shangshu.executor.execute_plan", _fake_execute)

        q = PlanQueue()  # 模拟重启后新队列
        deadline = time.time() + 5
        while time.time() < deadline:
            st = q.status()
            # E-6：plan-a（崩溃遗留 running）回收后按入队序最先执行
            if st["running"] == "plan-a" and st["queued"] == ["plan-b", "plan-c"]:
                break
            time.sleep(0.05)
        else:
            raise AssertionError(f"恢复失败：{q.status()}（queued/崩溃 running 行应全部回填）")

        released.set()
        time.sleep(0.3)  # 放行 worker 排空退出

        # done 行滚动清理：finish 超过保留数后旧行删除
        reset_queue()
        for i in range(25):
            enqueue_plan(f"p{i}")
            mark_queue_running(f"p{i}")
            finish_queue_item(f"p{i}")
        from llmsec.storage import ctlstore
        with ctlstore._db.session() as s:
            n_done = len(s.exec(ctlstore._select(ctlstore.CtlQueueItem)).all())
        assert n_done <= 20, f"done 行应滚动清理到 ≤20，实际 {n_done}"


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
