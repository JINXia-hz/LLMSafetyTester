"""代码审查第 1 轮修复的回归测试：高严重度正确性 bug（H1–H14）。

对照审查报告：
  H1  attack_phase 种子批发错目标（ambient 未设置 → 回退全局默认客户端）
  H2  build_tree 只认落盘 allergy.json 形态，内存扁平 summary 恒取 fpr=None
  H3  /health 读 task_manager、任务却登记在 tasks.py 的另一份 TASKS
  H4  _discover_runs 缓存签名感知不到 batch 内新增 target 子目录
      （回归已并入 test_audit_r6_root.py G4，此处不重复）
  H5  generate --start-from 按字典序过滤（'1.10.1' < '1.3.1'）
  H6  filelock 被 import 但未声明为依赖
  H7  control 不入 wheel（llmsec 模块级 import control）
  H8  PlanQueue 并发 submit 可启动双 worker
  H9  PlanQueue worker 退出竞态 → 任务永久搁浅
  H10 封驳检测用 5 秒时间窗 → 放行重试被旧 block 反复拦截
  H11 SSE 对 queued 任务立即发 done 并关流
  H12 .env 原子写临时文件名 .env.env.tmp（with_suffix 对点文件的坑）
  H13 gazette 索引 RMW 绕过跨进程文件锁 → 丢更新
  H14 fallback compare 表格列名与 run_metrics 字段不符（3 列恒 "-"）

全部离线：mock/合成数据，不触真实 API 与真实 output/。
"""
from __future__ import annotations

import asyncio
import contextlib
import re
import threading
import time
from pathlib import Path

import pytest


# ============================================================
# H1：种子批发给正确的目标
# ============================================================
def _records(methods):
    return [{
        "id": m, "method": m, "prompt": f"attack prompt {m}",
        "expected_answer": 0, "category": "test", "harm_type": "test",
    } for m in methods]


def _eval_ok(prompt, ea, judge, use_judge=True):
    return {
        "content": "ok", "is_refusal": False, "is_harmful": True,
        "eval_score": 3.0, "jailbreak_tax": 0.5, "status": "fully_compliant",
        "compliance_level": "A", "latency_ms": 100,
    }


def test_h1_seed_round_routes_to_defender(tmp_path, monkeypatch):
    """种子轮 evaluate_single 调用时 ambient 目标必须是 defender_name。

    修复前：种子循环不设置 threading.local ambient，call_target 回退全局默认
    客户端——--target X（X≠全局 TARGET_MODEL）时整批种子发给错误模型。
    """
    import llmsec.pipeline.attack_phase as ap
    from llmsec.core.results import ResultsMatrix
    from llmsec.evaluation.elo import ELOTracker
    from llmsec.targets import get_active_target, set_active_target

    set_active_target(None)  # 线程卫生：不继承其他测试残留的 ambient

    import llmsec.core.config as cfg
    monkeypatch.setattr(cfg, "FEATURE_CACHE_FILE", tmp_path / "feature_cache.pkl")
    monkeypatch.setattr(ap, "analyze_clusters", lambda tracker: {})

    seen: list[str | None] = []

    def _spy(prompt, ea, judge, use_judge=True):
        seen.append(get_active_target())  # 记录调用瞬间的路由目标
        return _eval_ok(prompt, ea, judge, use_judge=use_judge)

    monkeypatch.setattr(ap, "evaluate_single", _spy)

    methods = ["m0", "m1", "m2", "m3", "m4", "m5"]
    monkeypatch.setattr(ap, "_quick_precluster", lambda *a, **k: None)

    from llmsec.evaluation.predictors.cold_start import (
        _compute_method_set_hash,
        current_feature_config_hash,
    )
    tracker = ELOTracker()
    tracker.predictor.artifacts = {
        "features": {
            m: {"textual": [1.0, 0.0], "embedding": [0.1 * i, 0.2]}
            for i, m in enumerate(methods)
        },
        "method_set_hash": _compute_method_set_hash(sorted(methods)),
        "meta": {"feature_config_hash": current_feature_config_hash()},
    }

    try:
        ap.run_attack_phase(
            _records(methods), judge=None, tracker=tracker,
            batch_size=6, max_rounds=1, attack_file=tmp_path / "attack.jsonl",
            sampler="gap", coordinate_rounds=1,
            state_file=str(tmp_path / "state.json"),
            defender_name="def-h1", r_snapshot=ResultsMatrix(),
        )
    finally:
        set_active_target(None)

    # 种子轮先于主循环执行（GT 为空 → 必有种子），所有调用都必须路由到 def-h1
    assert seen, "H1: 冷启动应触发种子评估"
    bad = [a for a in seen if a != "def-h1"]
    assert not bad, f"H1: 有 {len(bad)}/{len(seen)} 次评估未路由到 def-h1（实得 {set(seen)}）"


# ============================================================
# H2：build_tree 兼容内存扁平 allergy summary
# ============================================================
def _tree_with_allergy(monkeypatch, tmp_path, allergy_data):
    import llmsec.reporting.report as rep

    class _StubTracker:
        attacker_ratings = {"u1": 1600.0}
        defender_ratings = {"def": 1500.0}

        def compute_security_boundary(self, *a, **k):
            return {"confidence": 0.9, "boundary_elo": 1500.0}

        def find_surprises(self, min_elo_gap=0):
            return {"weakness": [], "strength": []}

    monkeypatch.setattr(rep, "_load_elo_tracker", lambda output_dir=None: _StubTracker())
    results = [{"method": f"m{i}", "is_harmful": False, "harm_type": "t", "category": "c"}
               for i in range(10)]
    method_stats = rep.build_method_stats(results, {}, {})
    return rep.build_tree(method_stats, allergy_data)


def test_h2_flat_summary_fpr_takes_effect(monkeypatch, tmp_path):
    """内存扁平 {"fpr": 0.5} → fpr 超标 → level=allergic（修复前恒 safe）。"""
    tree = _tree_with_allergy(monkeypatch, tmp_path, {"fpr": 0.5})
    level = tree["overall"]["security_level"]
    assert level == "allergic", f"H2: 扁平 fpr=0.5 应判 allergic（实得 {level}）"
    assert tree["overall"].get("fpr") == pytest.approx(0.5)


def test_h2_nested_disk_shape_still_works(monkeypatch, tmp_path):
    """落盘 allergy.json 形态（summary.false_positive_rate）不受影响。"""
    tree = _tree_with_allergy(
        monkeypatch, tmp_path, {"summary": {"false_positive_rate": 0.5}})
    assert tree["overall"]["security_level"] == "allergic"
    assert tree["overall"].get("fpr") == pytest.approx(0.5)


def test_h2_missing_fpr_means_no_evidence(monkeypatch, tmp_path):
    """fpr 缺失（无过敏证据）+ ASR 低 → safe，不误判 allergic。"""
    tree = _tree_with_allergy(monkeypatch, tmp_path, {})
    assert tree["overall"]["security_level"] == "safe"


# ============================================================
# H3：tasks router 与 task_manager 共用同一份 TASKS
# ============================================================
def test_h3_single_task_registry():
    from llmsec.server import task_manager
    from llmsec.server.routers import tasks

    assert tasks.TASKS is task_manager.TASKS, \
        "H3: router 与 task_manager 必须共用同一份 TASKS（否则 /health 看不到看板任务）"


def test_h3_health_counts_running_tasks(tmp_path):
    """/health 应统计到看板启动的 running 任务（修复前恒 0）。"""
    from fastapi.testclient import TestClient

    from llmsec.server import task_manager
    from llmsec.server.dashboard_api import app

    class _StubProc:
        def poll(self):
            return None

    tid = "evaluate-h3-probe"
    task_manager.TASKS[tid] = {
        "kind": "evaluate", "cmd": "probe", "argv": [], "proc": _StubProc(),
        "log_path": tmp_path / "probe.log", "log_file": None,
        "status": "running", "started_at": "2026-01-01T00:00:00",
    }
    try:
        client = TestClient(app)
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["tasks_running"] >= 1, \
            f"H3: /health 应看到 running 任务（实得 {body}）"
    finally:
        task_manager.TASKS.pop(tid, None)


# ============================================================
# H5：--start-from 数值序
# ============================================================
def test_h5_filter_methods_numeric_order():
    from llmsec.attacks.generate import _filter_methods

    ms = [{"id": i} for i in ("1.1.1", "1.3.1", "1.10.1", "2.1")]
    out = _filter_methods(ms, only=None, start_from="1.3.1")
    assert [m["id"] for m in out] == ["1.3.1", "1.10.1", "2.1"], \
        "H5: '1.10.1' ≥ '1.3.1' 应按数值序保留（字典序会错误跳过）"
    assert [m["id"] for m in _filter_methods(ms, only="1.10.1", start_from=None)] == ["1.10.1"]
    assert _filter_methods(ms, only=None, start_from=None) == ms


# ============================================================
# H6/H7：依赖声明与打包范围（回归守卫）
# ============================================================
_ROOT = Path(__file__).resolve().parents[1]


def test_h6_filelock_declared_everywhere():
    pyproject = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    reqs_in = (_ROOT / "llmsec" / "requirements.in").read_text(encoding="utf-8")
    reqs_txt = (_ROOT / "llmsec" / "requirements.txt").read_text(encoding="utf-8")
    assert re.search(r'"filelock>=3', pyproject), "H6: pyproject dependencies 缺 filelock"
    assert "filelock>=3,<4" in reqs_in, "H6: requirements.in 缺 filelock"
    assert re.search(r"^filelock==", reqs_txt, re.M), "H6: requirements.txt 缺 filelock 锁定"


def test_h7_control_packaged_into_wheel():
    pyproject = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(
        r"\[tool\.setuptools\.packages\.find\]\s*(?:#[^\n]*\n\s*)*include\s*=\s*\[(.*?)\]",
        pyproject, re.S)
    assert m and '"control*"' in m.group(1), \
        "H7: 打包范围必须含 control*（dashboard_api 模块级 import control）"


# ============================================================
# H8/H9：PlanQueue worker 生命周期
# ============================================================
def _install_exec_stub(monkeypatch, recorder):
    from control.agent.shangshu import executor

    def _stub(plan_id):
        recorder["idents"].add(threading.get_ident())
        recorder["done"].append(plan_id)

    monkeypatch.setattr(executor, "execute_plan", _stub)


def _wait_until(cond, timeout=8.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(interval)
    return False


def test_h9_worker_restarts_after_exit(monkeypatch):
    """worker 空闲退出后再次 submit 必须能执行（修复前存在永久搁浅窗口）。"""
    from control.agent.shangshu.queue import PlanQueue

    recorder = {"idents": set(), "done": []}
    _install_exec_stub(monkeypatch, recorder)
    q = PlanQueue()

    assert q.submit("p1") == "queued"
    assert _wait_until(lambda: "p1" in recorder["done"]), "H9: 首个任务未执行"
    # 等待 worker 空闲退出（wait 超时 1s 后摘牌退出）
    assert _wait_until(lambda: q._worker is None, timeout=5), "H9: worker 未按期退出"

    assert q.submit("p2") == "queued"
    assert _wait_until(lambda: "p2" in recorder["done"]), \
        "H9: worker 退出后重新 submit 的任务被永久搁浅"


def test_h8_concurrent_submit_single_worker(monkeypatch):
    """worker 已死时并发 submit 只能启动一个新 worker（单 Plan 串行不变量）。"""
    from control.agent.shangshu.queue import PlanQueue

    recorder = {"idents": set(), "done": []}
    _install_exec_stub(monkeypatch, recorder)
    q = PlanQueue()

    # 先跑一个任务并等 worker 退出，制造"worker 已死"的起点
    q.submit("init")
    assert _wait_until(lambda: "init" in recorder["done"])
    assert _wait_until(lambda: q._worker is None, timeout=5)
    # 计数只针对并发阶段（W1 已合法退出，不计入）
    recorder["idents"].clear()

    n = 8
    barrier = threading.Barrier(n)

    def _submit(i):
        barrier.wait()
        q.submit(f"p{i}")

    threads = [threading.Thread(target=_submit, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    expected = {"init"} | {f"p{i}" for i in range(n)}
    assert _wait_until(lambda: set(recorder["done"]) == expected, timeout=10), \
        f"H8: 有任务未执行（done={recorder['done']}）"
    assert len(recorder["idents"]) == 1, \
        f"H8: 并发 submit 启动了 {len(recorder['idents'])} 个 worker（必须始终单 worker）"


# ============================================================
# H10：封驳裁决经总线同步回执直收（R6 病根修复后的语义）
# ============================================================
def test_h10_block_from_sync_reply():
    """step_start 的 collect_replies 直收门下省裁决，不重复、不依赖时间窗。

    旧机制（扫 bus.recent + 消费去重）已整体移除：一次派发一次裁决，
    重试时门下省重新裁决，天然无旧消息重复命中问题。
    """
    from control.agent import bus as bus_mod
    from control.agent.bus import notify

    bus_mod.reset_bus()
    try:
        bus = bus_mod.get_bus()
        verdicts = [{"plan_id": "p1", "step_id": "s1", "token": "t1"}, None]

        def _fake_menxia(msg):
            if msg.kind == bus_mod.KIND_STEP_START and verdicts:
                return verdicts.pop(0)
            return None

        bus.subscribe("门下省", [bus_mod.KIND_STEP_START], _fake_menxia)

        r1 = notify(bus_mod.KIND_STEP_START, from_dept="尚书省",
                    plan_id="p1", step_id="s1", collect_replies=True)
        assert r1 and r1[0].get("token") == "t1", "H10: 封驳裁决应经回执直收"

        r2 = notify(bus_mod.KIND_STEP_START, from_dept="尚书省",
                    plan_id="p1", step_id="s1", collect_replies=True)
        assert r2 == [], "H10: 放行重试不应收到旧封驳（回执机制无重复命中）"
    finally:
        bus_mod.reset_bus()


def test_h10_executor_receives_block_and_exempts_on_retry(tmp_path, monkeypatch):
    """端到端：危险步骤被封驳 → 用户放行（清令 + 文牍）→ 重试经豁免放行执行。

    真实 menxia listener + 真实 gazette 文牍豁免，不再依赖任何扫描机制。
    """
    from control.agent import bus as bus_mod
    from control.agent import gazette
    from control.agent.menxia import block as block_mod
    from control.agent.menxia import listener
    from control.agent.shangshu import executor
    from control.agent.shangshu import plan as plan_mod
    from control.agent.shangshu.plan import P_APPROVED, Plan, Step, save_plan

    monkeypatch.setattr(gazette, "_GAZETTE_DIR", tmp_path / "gazette")
    monkeypatch.setattr(plan_mod, "_PLANS_DIR", tmp_path / "plans")
    plan_mod._PLANS_DIR.mkdir(parents=True, exist_ok=True)
    plan_mod.reset_plans()
    bus_mod.reset_bus()
    listener.reinit_menxia()
    block_mod.reset_blocks()

    executed = []
    from control.agent.shangshu import capabilities as caps_mod
    from control.agent.shangshu.capabilities import Capability

    def _fake_cap_by_name(name):
        # 假 capability：handler 只记录不真跑（此前误 patch caps_mod.call——
        # executor 走的是 cap.handler，导致真实 runner 子进程被拉起）
        if name == "run_evaluation":
            return Capability(
                name=name, description="t", parameters={},
                handler=lambda args: executed.append(name) or {"ok": True},
                risk_level="high",
                block_message=lambda args: {"summary": "高危", "detail": "确认？"},
            )
        return None

    monkeypatch.setattr(caps_mod, "capability_by_name", _fake_cap_by_name)

    plan = Plan(intent="t", steps=[Step(id="s1", capability="run_evaluation",
                                        args={"max_rounds": 1})],
                status=P_APPROVED)
    save_plan(plan)

    # 第一次执行：run_evaluation 属高危 → 被封驳
    executor.execute_plan(plan.id, max_workers=1)
    p1 = plan_mod.load_plan(plan.id)
    assert p1.steps[0].status == "blocked" and p1.steps[0].ticket, \
        "H10: 高危步骤应经同步回执被封驳"
    assert executed == [], "被封驳的步骤不应执行 handler"

    # 用户放行：清封驳令 + 写文牍 step_unblocked（router api_block_approve 同款流程）
    block_mod.approve_block(plan.id, "s1")
    gazette.append_event(plan.id, gazette.EV_STEP_UNBLOCKED, "用户", step_id="s1",
                         detail={"capability": "放行重试"})
    for s in p1.steps:
        s.status = "pending"
        s.ticket = None
    p1.status = P_APPROVED
    save_plan(p1)

    # 第二次执行：文牍豁免生效 → 放行执行
    executor.execute_plan(plan.id, max_workers=1)
    p2 = plan_mod.load_plan(plan.id)
    assert p2.steps[0].status == "done", \
        f"H10: 放行重试应执行成功（实得 {p2.steps[0].status}）"
    assert executed == ["run_evaluation"]

    plan_mod.reset_plans()
    bus_mod.reset_bus()
    listener.reinit_menxia()
    block_mod.reset_blocks()


# ============================================================
# H11：SSE 对 queued 任务不得立即发 done
# ============================================================
def test_h11_sse_queued_task_not_immediately_done(monkeypatch, tmp_path):
    from llmsec.server import task_manager
    from llmsec.server.routers import tasks as tasks_mod

    tid = "smoke-h11-queued"
    task_manager.TASKS[tid] = {
        "kind": "smoke", "cmd": "probe", "argv": [], "proc": None,
        "log_path": tmp_path / "probe.log", "log_file": None,
        "status": "queued", "started_at": "2026-01-01T00:00:00",
    }

    # 假 sleep：第 2 次起把任务置为终态，让生成器自然收尾（免真等 0.5s×N）
    calls = {"n": 0}

    async def _fake_sleep(sec):
        calls["n"] += 1
        if calls["n"] >= 2:
            task_manager.TASKS[tid]["status"] = "success"
            task_manager.TASKS[tid]["returncode"] = 0

    monkeypatch.setattr(tasks_mod.asyncio, "sleep", _fake_sleep)
    try:
        resp = asyncio.run(tasks_mod.api_task_stream(tid))

        async def _collect():
            out = []
            ait = resp.body_iterator
            while True:
                try:
                    chunk = await asyncio.wait_for(ait.__anext__(), timeout=2.0)
                except StopAsyncIteration:
                    break
                out.append(chunk)
            return out

        events = asyncio.run(_collect())
    finally:
        task_manager.TASKS.pop(tid, None)

    done_events = [e for e in events if "event: done" in e]
    assert done_events, "H11: 任务终态后应发出 done 事件"
    assert '"queued"' not in done_events[0], \
        f"H11: 排队中的任务被立即当作结束（events={events}）"
    assert '"success"' in done_events[0]


# ============================================================
# H12：.env 原子写临时文件名
# ============================================================
def test_h12_env_tmp_naming():
    # Path 语义佐证：点文件 suffix 为空，with_suffix 会产出 .env.env.tmp
    assert Path(".env").with_suffix(".env.tmp").name == ".env.env.tmp"

    src = Path(__import__("llmsec.server.routers.data_query", fromlist=["x"]).__file__) \
        .read_text(encoding="utf-8")
    assert '.with_suffix(".env.tmp")' not in src, \
        "H12: .env 临时文件不得用 with_suffix（点文件坑）"
    assert src.count('with_name(env_path.name + ".tmp")') >= 2, \
        "H12: 两处 .env 原子写应统一用 with_name 拼接"


# ============================================================
# H13：gazette 索引并发不丢更新
# ============================================================
def test_h13_gazette_index_no_lost_updates(tmp_path, monkeypatch):
    """禁用 gazette 进程内锁（模拟另一进程的锁保护不到本进程），并发 append
    不得丢索引条目——并发安全只能靠 _store.update 的跨进程文件锁。"""
    from control.agent import gazette

    monkeypatch.setattr(gazette, "_GAZETTE_DIR", tmp_path / "gazette")
    gazette.reset_gazettes()
    monkeypatch.setattr(gazette, "_LOCK", contextlib.nullcontext())

    n_threads, n_plans = 8, 5
    barrier = threading.Barrier(n_threads)

    def _work(w):
        barrier.wait()
        for i in range(n_plans):
            gazette.append_event(f"p{w}_{i}", gazette.EV_PLAN_DRAFTED, "尚书省",
                                 detail={})

    threads = [threading.Thread(target=_work, args=(w,)) for w in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    idx = gazette._store.load()
    names = set(idx.get("plans", {}))
    expected = {f"p{w}_{i}" for w in range(n_threads) for i in range(n_plans)}
    missing = expected - names
    assert not missing, f"H13: 索引丢更新（缺 {len(missing)}/{len(expected)} 个条目）"


# ============================================================
# H14：compare 表格列名与 run_metrics 字段一致
# ============================================================
def test_h14_render_compare_columns():
    from control.agent.zhongshu.fallback import _render_compare

    report = {"runs": [{
        "run": "2026-01-01_000000/gemma", "target_model": "gemma",
        "asr": 0.25, "fpr": 0.1, "boundary_elo": 1480.0,
        "conv_rounds": 3, "security_level": "vulnerable",
    }], "missing": []}
    text = _render_compare(report)
    assert "target_model" in text and "boundary_elo" in text and "security_level" in text, \
        "H14: 表头应使用 run_metrics 的真实字段名"
    # 三个曾恒为 "-" 的列现在必须渲染出真实值
    assert "gemma" in text, "H14: target_model 列未渲染目标名"
    assert "vulnerable" in text, "H14: security_level 列未渲染等级"
    assert "1480" in text, "H14: boundary_elo 列未渲染 ELO"
