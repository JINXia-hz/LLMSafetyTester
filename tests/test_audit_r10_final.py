"""终审第五轮（r10）回归测试——上线前终审发现的 P0/P1 修复机器锚点。

覆盖：C-1 质量分连接 / C-12 跨集撞名 / B-1 Elo 过滤单源 / C-2 过敏降级剔除 /
A-1 state.json 损坏守卫 / D-1 外部任务 SSE / E-1 fail-closed 序列化 /
E-2 执行期放行重入队（真实队列）/ G-3 驳回终局 / 队列行 per-entry 生命周期。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

import llmsec.core.config as cfg

# ============================================================
# C-1：assess 质量分连接（prompt_sha16 双侧同键）
# ============================================================

def test_quality_key_prefers_prompt_sha16():
    """攻击集记录（带 prompt）与明细行（带 prompt_sha16）必须算出同一个键。"""
    import hashlib

    from llmsec.attacks.quality import quality_key

    sha = hashlib.sha256(b"P").hexdigest()[:16]
    assert quality_key({"id": "atk-1", "prompt": "P"}) == f"atk-1:{sha}"
    assert quality_key({"id": "atk-1", "prompt_sha16": sha}) == f"atk-1:{sha}"
    # 两侧同给时 sha16 优先（防 prompt 全文意外混入明细行时口径漂移）
    assert quality_key({"id": "atk-1", "prompt": "其他", "prompt_sha16": sha}) == f"atk-1:{sha}"


def test_build_attack_row_carries_prompt_sha16():
    """明细行必须落 prompt_sha16——缺此字段时键恒为空串指纹、连接恒 miss（C-1 病根）。"""
    import hashlib

    from llmsec.pipeline.attack_phase import _build_attack_row

    rec = {"id": "atk-1", "method": "m", "prompt": "P"}
    result = {"eval_score": 0.0, "jailbreak_tax": None, "status": "refused",
              "latency_ms": 5, "content": "x"}
    row = _build_attack_row(rec, result, 0, "seed", unit="u1")
    assert row["prompt_sha16"] == hashlib.sha256(b"P").hexdigest()[:16]


def _make_run_dir(tmp_path: Path, rows: list[dict]) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "attack_results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    (run_dir / "state.json").write_text(json.dumps({
        "attacker_stats": {}, "attacker_ratings": {"u1": 1400.0},
        "defender_ratings": {"def": 1500.0}, "history": [],
    }), encoding="utf-8")
    return run_dir


def test_fuse_joins_quality_and_flags_zero_join(tmp_path, caplog):
    """带 prompt_sha16 的行连接命中；无该字段的旧行触发零命中 ERROR 哨兵。"""
    import logging

    from llmsec.attacks.assess import fuse
    from llmsec.attacks.quality import quality_key
    from llmsec.params import RECTIFY_MIN_TESTS

    n = RECTIFY_MIN_TESTS + 1
    # 攻击集侧记录 → 质量缓存键
    records = [{"id": f"atk-{i}", "prompt": f"prompt-{i}", "method": "m"} for i in range(n)]
    quality = {quality_key(r): {"overall": 1.5, "tags": ["degenerate"]} for r in records}

    # 新式明细行：低 ASR × 弱质量 → 假防御嫌疑
    rows = [{"unit": "u1", "method": "m", "id": r["id"],
             "prompt_sha16": r_key.split(":")[1],
             "is_harmful": False, "eval_score": 0.0, "round": 0}
            for r, r_key in zip(records, [quality_key(r) for r in records])]
    v = fuse(_make_run_dir(tmp_path, rows), quality)
    assert v["false_defense_suspects"], "连接修好后低 ASR × 弱质量必须是假防御嫌疑"
    assert v["false_defense_suspects"][0]["mean_quality"] == 1.5

    # 旧式明细行（无 prompt_sha16，C-1 修复前的恒 miss 形态）→ 零命中哨兵必须吵
    caplog.clear()
    old_rows = [{k: v2 for k, v2 in r.items() if k != "prompt_sha16"} for r in rows]
    with caplog.at_level(logging.ERROR, logger="llmsec.attacks.assess"):
        v2 = fuse(_make_run_dir(tmp_path / "old", old_rows), quality)
    assert v2["inconclusive_count"] == 1 and not v2["false_defense_suspects"]
    assert any("质量分连接 0 命中" in rec.message for rec in caplog.records), \
        "零命中必须打 ERROR（防键口径再次静默漂移）"


# ============================================================
# C-12：跨攻击集撞名 id 防护
# ============================================================

def test_resume_tested_recs_cross_set_id_collision():
    """R 行 unit 与本集 id 所属 unit 不一致（跨集撞名）不得标记已测。"""
    from llmsec.core.results import ResultsMatrix
    from llmsec.pipeline.attack_phase import _resume_tested_recs

    tested: dict[str, set] = {"u-cur": set()}
    rec_to_unit = {"atk-1": "u-cur"}

    r_other = ResultsMatrix()
    r_other.upsert(record="atk-1", model="def", eval_score=1.0, status="harmful",
                   extra={"unit": "u-other-set"})
    _resume_tested_recs(r_other, "def", rec_to_unit, tested)
    assert tested["u-cur"] == set(), "跨集同 id（unit 不一致）不得标记已测（C-12）"

    r_same = ResultsMatrix()
    r_same.upsert(record="atk-1", model="def", eval_score=1.0, status="harmful",
                  extra={"unit": "u-cur"})
    _resume_tested_recs(r_same, "def", rec_to_unit, tested)
    assert tested["u-cur"] == {"atk-1"}, "同 unit 的真实已测记录照常标记"


# ============================================================
# B-1：Elo 过滤单源（judge_parse_fallback 不得进 live/R Elo）
# ============================================================

def test_elo_eligible_excludes_all_judge_failure_modes():
    from llmsec.evaluation.scoring import elo_eligible

    assert elo_eligible({"judge_mode": "judge", "status": "harmful"})
    assert elo_eligible({"judge_mode": "no_judge", "status": "harmful"})
    assert not elo_eligible({"judge_mode": "fallback_keyword", "status": "judge_error"})
    assert not elo_eligible({"judge_mode": "judge_parse_fallback", "status": "judge_error"}), \
        "B-1：第二种降级模式必须同样剔除（此前 attack_phase 漏滤）"
    assert not elo_eligible({"judge_mode": "judge", "status": "judge_error"})


def test_update_elo_excludes_parse_fallback_from_r(tmp_path, monkeypatch):
    """evaluator 回放：judge_parse_fallback 行不进 R（与 live 喂入口径一致）。"""
    from llmsec.evaluation.evaluator import update_elo
    from llmsec.storage.contract import load_matrix

    rows = [
        {"method": "u-ok", "id": "rec-ok", "round": 0, "eval_score": 3.0,
         "status": "harmful", "judge_mode": "judge"},
        {"method": "u-bad", "id": "rec-bad", "round": 0, "eval_score": 0.0,
         "status": "judge_error", "judge_mode": "judge_parse_fallback"},
    ]
    update_elo(rows, summary={}, defender_name="r10-def")
    col = load_matrix().model_column("r10-def")
    assert "rec-ok" in col
    assert "rec-bad" not in col, "judge_parse_fallback 行不得进入 R（B-1）"


# ============================================================
# C-2：allergy_phase 降级条目剔除 FPR 分子分母 + 阈值对齐
# ============================================================

class _FakeTracker2:
    defender_ratings = {"fake_def": 1500.0}

    def get_attacker_ranking(self):
        # 覆盖足够大的窗口（select_twin_candidates 按排名截 n_window）
        return [{"unit": f"m{i}", "elo": 1490.0} for i in range(1, 10)]

    def compute_security_boundary(self, _name):
        return {"boundary_elo": 1500.0}

    def get_attacker_elo(self, _m):
        return 1490.0


class _JudgeByContent:
    """content 含 FAIL-MARK 的调用抛错（Judge 故障），其余正常判不拒绝。"""

    def evaluate(self, _prompt, content):
        if "FAIL-MARK" in content:
            raise RuntimeError("judge down")
        return {"is_refusal": False, "compliance_level": "A"}


def _run_allergy(monkeypatch, tmp_path, contents):
    """按序给每个 method 一个 content；返回 (summary, 结果明细)。"""
    import llmsec.pipeline.allergy_phase as allergy_phase

    it = iter(contents)
    monkeypatch.setattr(allergy_phase, "get_or_create_twin",
                        lambda method, rec, cache, client: "安全孪生prompt")
    monkeypatch.setattr(allergy_phase, "call_target",
                        lambda prompt: {"error": None, "content": next(it), "meta": {}})
    monkeypatch.setattr(allergy_phase, "API_DELAY", 0)
    monkeypatch.setattr(cfg, "SAFE_TWINS_FILE", tmp_path / "safe_twins.jsonl")
    units = {f"m{i+1}": {"id": f"x{i+1}", "prompt": "p", "category": "c",
                         "method": f"m{i+1}"} for i in range(len(contents))}
    af = tmp_path / "allergy.json"
    summary = allergy_phase.run_allergy_phase(
        units, twin_client=None, judge=_JudgeByContent(), tracker=_FakeTracker2(),
        n_window=len(contents), allergy_file=af, concurrency=0)
    detail = json.loads(af.read_text(encoding="utf-8"))["results"]
    return summary, detail


def test_allergy_degraded_rows_excluded_from_fpr(monkeypatch, tmp_path):
    """Judge 全故障 + 单关键词命中：不进 FPR 分子分母（C-2，对齐 safe_twin.B-6）。"""
    # m1: 单命中"I cannot"（1 < PRESCREEN_REFUSAL_HITS=2，旧口径会误判过敏）
    # m2: 干净
    summary, detail = _run_allergy(
        monkeypatch, tmp_path,
        ["FAIL-MARK 抱歉 I cannot help with that", "FAIL-MARK 好的，制作蛋糕的步骤……"])
    assert summary["judge_failed_count"] == 2
    assert summary["total_tested"] == 0, "降级行不得进 FPR 分母"
    assert summary["fpr"] is None, "无有效判定样本时 FPR 未测（不伪造 0）"
    assert len(detail) == 2 and all(r["judge_failed"] for r in detail), "明细照留供核查"


def test_allergy_mixed_degraded_and_judged(monkeypatch, tmp_path):
    """正常判定 2 条（1 过敏）+ 降级 1 条：FPR=1/2 由完整判定样本构成。"""
    summary, _ = _run_allergy(
        monkeypatch, tmp_path,
        ["抱歉，我不能提供这类信息。",     # 多关键词命中 → 过敏（judged）
         "好的，制作蛋糕的步骤……",        # judge 正常 → 不过敏
         "FAIL-MARK 抱歉 I cannot help"])
    assert summary["judge_failed_count"] == 1
    assert summary["total_tested"] == 2, "降级行不得进 FPR 分母（C-2）"
    assert summary["allergic"] == 1
    assert summary["fpr"] == 0.5, "FPR 只由 2 条完整判定样本构成（1/2）"


def test_allergy_degraded_threshold_two_hits(monkeypatch, tmp_path):
    """降级判定阈值 ≥PRESCREEN_REFUSAL_HITS：双命中才过敏（且仍不进 FPR）。"""
    summary, detail = _run_allergy(
        monkeypatch, tmp_path,
        ["FAIL-MARK I'm sorry, I cannot help with that request",  # "I'm sorry"+"I cannot" 双命中
         "好的，制作蛋糕的步骤……"])
    assert summary["judge_failed_count"] == 1
    assert detail[0]["is_allergic"] is True, "降级路径双关键词命中仍应判过敏（明细层）"
    assert summary["total_tested"] == 1 and summary["fpr"] == 0.0, \
        "但降级行不得混进 FPR 分子分母"


# ============================================================
# A-1：clone_from_run 对损坏 state.json 显式失败
# ============================================================

def test_clone_from_run_rejects_corrupt_state_json(tmp_path, monkeypatch):
    from llmsec.storage.contract import clone_from_run

    monkeypatch.setattr(cfg, "RUNS_DIR", tmp_path)
    run_dir = tmp_path / "2026-01-01_000000" / "def"
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text("{half-written", encoding="utf-8")
    with pytest.raises(ValueError, match="state.json 损坏"):
        clone_from_run("2026-01-01_000000/def", tmp_path / "out.db")
    assert not (tmp_path / "out.db").exists(), "拒绝导出空 R（不得落产物）"


# ============================================================
# D-1：外部任务 SSE 不因 Task.returncode 缺列崩溃
# ============================================================

def test_external_task_stream_survives_missing_returncode(monkeypatch):
    """外部（跨进程）任务行无 returncode 列——SSE 必须正常发 done 而非 AttributeError。"""
    from fastapi.testclient import TestClient

    from llmsec.server import task_manager
    from llmsec.server.dashboard_api import app
    from llmsec.storage.contract import upsert_task

    upsert_task("r10-ext-stream", kind="evaluate", status="success",
                pid=12345, cmd="external")
    client = TestClient(app)
    try:
        monkeypatch.setattr(task_manager, "TASKS", {})  # 不在本进程表 → 走外部行
        r = client.get("/api/tasks/r10-ext-stream/stream")
        assert r.status_code == 200
        assert "event: done" in r.text, "外部任务流必须以 done 收尾（D-1 修复前零事件即断）"
    finally:
        client.close()


# ============================================================
# E-1：fail-closed 故障票必须是可序列化 dict
# ============================================================

def test_fail_closed_ticket_is_serializable_dict():
    """门下省订阅回调异常 → fail-closed 封驳票为 dict，Plan 不崩、文牍落库。"""
    from control.agent import bus as bus_mod
    from control.agent.shangshu.executor import execute_plan
    from control.agent.shangshu.plan import P_APPROVED, Plan, Step, load_plan, save_plan
    from control.core.storage import gazette_events

    bus_mod.reset_bus()

    def bad_reply(_msg):
        raise RuntimeError("menxia boom")

    bus_mod.get_bus().subscribe(bus_mod.MENXIA, ["step_start"], bad_reply)
    plan = Plan(intent="r10", steps=[Step(id="s1", capability="run_evaluation", args={})],
                status=P_APPROVED)
    save_plan(plan)
    try:
        execute_plan(plan.id)  # E-1 修复前：TypeError: BlockTicket is not JSON serializable
    finally:
        bus_mod.reset_bus()
    p2 = load_plan(plan.id)
    s1 = p2.steps[0]
    assert s1.status == "blocked"
    assert isinstance(s1.ticket, dict), "故障票必须是 dict（E-1：BlockTicket 对象序列化即崩）"
    json.dumps(s1.ticket)  # 不得抛
    # 文牍链完整：step_started → step_blocked（修复前事务回滚、事件丢失）
    from control.agent import gazette
    kinds = [e["kind"] for e in gazette_events(plan.id)]
    assert gazette.EV_STEP_BLOCKED in kinds, f"文牍必须落 step_blocked，实际: {kinds}"


# ============================================================
# E-2 / G-3：真实队列下的执行期放行重入队 / 驳回终局
# ============================================================

class _RequeueHarness:
    """两步场景：s1(low) 层1；s2(high, 先封驳) + s3(low, 可阻塞) 层2。

    门下省对 s2 第一次封驳、之后放行；s3 的 handler 挂在 Event 上，
    用于在层 2 执行期间注入用户动作（放行 / 驳回）。
    """

    def __init__(self, monkeypatch):
        from control.agent import bus as bus_mod
        from control.agent.shangshu import capabilities as caps
        from control.agent.shangshu.plan import P_APPROVED, Plan, Step, save_plan

        bus_mod.reset_bus()
        self.bus = bus_mod
        self.calls = {"s2": 0, "s3": 0}
        self.gate = threading.Event()
        self.blocked_once = False

        def menxia_reply(msg):
            if msg.payload.get("step_id") == "s2" and not self.blocked_once:
                self.blocked_once = True
                return {"plan_id": msg.plan_id, "step_id": "s2",
                        "ticket": {"token": "tk-r10", "plan_id": msg.plan_id,
                                   "step_id": "s2", "risk_level": "high",
                                   "summary": "r10 封驳", "detail": "测试票"}}
            return None  # 重跑（第二次 step_start）放行

        bus_mod.get_bus().subscribe(bus_mod.MENXIA, ["step_start"], menxia_reply)

        # s2 的 handler 打桩（真实 run_evaluation 会起子进程）；s3 挂门闸
        cap2 = caps.capability_by_name("run_evaluation")
        cap3 = caps.capability_by_name("list_runs")
        monkeypatch.setattr(cap2, "handler",
                            lambda args: self.calls.__setitem__("s2", self.calls["s2"] + 1)
                            or {"ok": True})
        monkeypatch.setattr(cap3, "handler",
                            lambda args: self.gate.wait(timeout=30)
                            and self.calls.__setitem__("s3", self.calls["s3"] + 1)
                            or {"ok": True})

        self.plan = Plan(intent="r10-queue", steps=[
            Step(id="s1", capability="list_env_snapshots", args={}),
            Step(id="s2", capability="run_evaluation", args={}, depends_on=["s1"]),
            Step(id="s3", capability="list_runs", args={}, depends_on=["s1"]),
        ], status=P_APPROVED)
        save_plan(self.plan)

    def wait_blocked(self, timeout=10.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            s2 = [s for s in self.plan.steps if s.id == "s2"][0]
            if s2.status == "blocked" and s2.ticket is not None:
                return True
            time.sleep(0.02)
        return False

    def approve_s2(self):
        """复刻 api_block_approve 的同步面：清库票 + 清内存票。"""
        from control.core.storage import clear_ticket
        clear_ticket(self.plan.id, "s2")
        for s in self.plan.steps:
            if s.id == "s2":
                s.ticket = None

    def wait_status(self, statuses, timeout=15.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.plan.status in statuses:
                return self.plan.status
            time.sleep(0.05)
        return self.plan.status

    def teardown(self):
        self.gate.set()
        self.bus.reset_bus()


@pytest.fixture()
def real_queue(monkeypatch):
    from control.agent.shangshu.queue import get_queue, reset_queue
    reset_queue()
    get_queue()  # 预热单例（恢复行为在隔离 catalog 下为空）
    yield
    reset_queue()


def test_inflight_approval_requeues_through_real_queue(monkeypatch, real_queue):
    """E-2：执行期放行后 executor 收尾重入队必须真的重跑（非 duplicate 吞单）。"""
    from control.agent.shangshu.queue import get_queue

    h = _RequeueHarness(monkeypatch)
    try:
        assert get_queue().submit(h.plan.id) == "queued"
        assert h.wait_blocked(), "s2 应被门下省封驳"
        h.approve_s2()          # 层 2 执行期间放行（s3 仍挂起）
        h.gate.set()            # 放行 s3 → 层 2 结束 → 收尾重入队（E-2 修复点）
        # 只等 done：approved 是重入队的合法中间态（worker 摘牌重跑的窗口），
        # 提前捕获它会制造竞态；若重入队被吞则 15s 后停在 approved 断言失败
        final = h.wait_status({"done"}, timeout=20.0)
        assert final == "done", \
            f"重入队后应执行到终态 done（E-2 修复前恒卡 approved），实际 {final}"
        assert h.calls["s2"] == 1, "放行的步骤必须真正重跑一次"
        assert h.calls["s3"] == 1
    finally:
        h.teardown()


def test_last_layer_reject_preserved(monkeypatch, real_queue):
    """G-3 变体2：最后层执行期间驳回 → 收尾不得覆盖为 done。"""
    from control.agent.shangshu import reject_plan
    from control.agent.shangshu.queue import get_queue

    h = _RequeueHarness(monkeypatch)
    try:
        assert get_queue().submit(h.plan.id) == "queued"
        assert h.wait_blocked()
        reject_plan(h.plan.id)   # 层 2 执行期间驳回（s3 仍挂起）
        h.gate.set()
        final = h.wait_status({"done", "rejected", "approved"})
        assert final == "rejected", \
            f"驳回终局不得被收尾覆盖（G-3 修复前变 done/approved），实际 {final}"
    finally:
        h.teardown()


def test_approve_then_reject_keeps_rejected(monkeypatch, real_queue):
    """G-3 变体1：放行 + 驳回 → 驳回仍优先（不得改判 approved 三不管）。"""
    from control.agent.shangshu import reject_plan
    from control.agent.shangshu.queue import get_queue

    h = _RequeueHarness(monkeypatch)
    try:
        assert get_queue().submit(h.plan.id) == "queued"
        assert h.wait_blocked()
        h.approve_s2()
        reject_plan(h.plan.id)
        h.gate.set()
        final = h.wait_status({"done", "rejected", "approved"})
        assert final == "rejected", \
            f"放行+驳回的终局必须是 rejected（G-3 修复前被 requeue 改判 approved），实际 {final}"
    finally:
        h.teardown()


# ============================================================
# 队列行 per-entry 生命周期（E-2 的持久层配套）
# ============================================================

def test_queue_rows_per_entry_lifecycle():
    """running 行旁可并存 queued 行；finish 只关 running；恢复去重。"""
    from control.core.storage import (
        enqueue_plan,
        finish_queue_item,
        mark_queue_running,
        pending_queue_plans,
        reset_queue,
    )

    reset_queue()
    enqueue_plan("r10-p")
    mark_queue_running("r10-p")
    enqueue_plan("r10-p")           # 执行中重入队 → queued 行与 running 行并存
    finish_queue_item("r10-p")      # worker 收尾（默认只关 running）
    assert pending_queue_plans() == ["r10-p"], \
        "重入队的 queued 行必须存活（E-2 修复前被 finish 全行清掉）"
    finish_queue_item("r10-p", include_queued=True)  # cancel 口径
    assert pending_queue_plans() == []
    reset_queue()


def test_pending_queue_plans_dedups_parallel_rows():
    """同 plan 的 running+queued 两行（崩溃窗口）恢复时只排一次。"""
    from control.core.storage import (
        enqueue_plan,
        mark_queue_running,
        pending_queue_plans,
        reset_queue,
    )

    reset_queue()
    enqueue_plan("r10-q")
    mark_queue_running("r10-q")
    enqueue_plan("r10-q")   # 崩溃前的重入队：两行并存（running + queued）
    assert pending_queue_plans().count("r10-q") == 1, "同一 plan 不得恢复出两个队列条目"
    reset_queue()
