"""回顾轮（R6）修复的回归测试：病根收口 + 架构整理。

覆盖：
  G1  session.conversation_lock 服务端串行化（同 session 排队、异 session 并行）
  G2  gazette unblocked 显式标记（豁免判据不被 EV_STEP_STARTED 的 status 覆盖）
  G3  runner.partition_publish_names（publish-global 守卫纯函数，替代被删假测试）
  G4  fsig 签名 helper（子目录 mtime 变化可感知）
  G5  probe.ModelsProbeResult（NamedTuple 契约）
  G6  tasks 兼容别名层已删除（墓碑用例已删，单一命名空间 task_manager）
  G7  mutator 不再自行 save（update 统一写回）
"""
from __future__ import annotations

import threading
import time


# ============================================================
# G1：session 服务端串行化
# ============================================================
def test_g1_conversation_lock_serializes_same_session():
    from control.agent.zhongshu import session as sess

    order = []
    lock_held = {"n": 0}

    def _work(i):
        with sess.conversation_lock("s-g1"):
            lock_held["n"] += 1
            assert lock_held["n"] == 1, "同 session 的锁必须互斥"
            order.append(f"enter{i}")
            time.sleep(0.05)
            order.append(f"exit{i}")
            lock_held["n"] -= 1

    threads = [threading.Thread(target=_work, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    # 串行：每个 enter 后必跟同号 exit
    assert order == [f"{ph}{i}" for i in range(4) for ph in ("enter", "exit")], \
        f"G1: 同 session 未串行化（实得 {order}）"


def test_g1_different_sessions_not_blocked():
    from control.agent.zhongshu import session as sess

    both_inside = threading.Barrier(2, timeout=3)

    def _work(sid):
        with sess.conversation_lock(sid):
            both_inside.wait()  # 两把锁同时持有才算通过（互阻则超时）

    t1 = threading.Thread(target=_work, args=("s-a",))
    t2 = threading.Thread(target=_work, args=("s-b",))
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert not t1.is_alive() and not t2.is_alive(), "G1: 不同 session 不应互相阻塞"


# ============================================================
# G2：gazette unblocked 标记
# ============================================================
def test_g2_gazette_unblocked_flag_survives_step_started(tmp_path, monkeypatch):
    from control.agent import gazette

    gazette.reset_gazettes()

    gazette.append_event("p", gazette.EV_STEP_STARTED, "尚书省", step_id="s1", detail={})
    gazette.append_event("p", gazette.EV_STEP_BLOCKED, "尚书省", step_id="s1",
                         detail={"ticket": {"t": 1}})
    gazette.append_event("p", gazette.EV_STEP_UNBLOCKED, "用户", step_id="s1", detail={})
    # 重试：executor 会先写新的 EV_STEP_STARTED（status 翻成 running）
    gazette.append_event("p", gazette.EV_STEP_STARTED, "尚书省", step_id="s1", detail={})

    ctx = gazette.read_plan_context("p")
    info = ctx["steps"]["s1"]
    assert info["status"] == "running"
    assert info["block_count"] == 1 and info.get("unblocked") is True, \
        "G2: 豁免判据须用显式 unblocked 标记（status 已被重试的 STARTED 覆盖）"

    # 再封驳：unblocked 标记被新封驳令翻转
    gazette.append_event("p", gazette.EV_STEP_BLOCKED, "尚书省", step_id="s1",
                         detail={"ticket": {"t": 2}})
    ctx2 = gazette.read_plan_context("p")
    assert ctx2["steps"]["s1"].get("unblocked") is False, \
        "G2: 新封驳令应覆盖旧放行标记（再次封驳生效）"


# ============================================================
# G3：publish-global 守卫纯函数
# ============================================================
def test_g3_partition_publish_names():
    from llmsec.pipeline.runner import partition_publish_names

    allowed, skipped = partition_publish_names(
        ["minimax", "test_model", "gemma"], {"minimax", "gemma"})
    assert allowed == ["minimax", "gemma"] and skipped == ["test_model"], \
        "G3: 未声明目标应被拒、声明目标放行"

    # declared 为空（load_targets 失败/未配置）：不校验，全部放行
    allowed2, skipped2 = partition_publish_names(["anything"], set())
    assert allowed2 == ["anything"] and skipped2 == []


# ============================================================
# G4：fsig 签名 helper（r7：dir_sig 无生产调用方已删，此处只测 file_sig）
# ============================================================
# （g4 fsig：模块已随 P5 库化删除——mtime 签名缓存的消费者已不存在）



def test_g4_data_query_uses_shared_sig(tmp_path, monkeypatch):
    import llmsec.server.routers.data_query as dq
    from llmsec.server import dashboard_api

    monkeypatch.setattr(dashboard_api, "RUNS_DIR", tmp_path)
    batch = tmp_path / "2026-01-01_000000"
    batch.mkdir()
    from llmsec.storage import contract as _storage
    _storage.reconcile_runs(runs_root=tmp_path)  # P9：查询纯读——造盘后显式入册
    t = batch / "gemma"
    t.mkdir()
    (t / "runner_report.json").write_text('{"target_model": "gemma"}', encoding="utf-8")
    _storage.reconcile_runs(runs_root=tmp_path)
    names = [r["name"] for r in dq._discover_runs()]
    assert "2026-01-01_000000/gemma" in names


# ============================================================
# G5：ModelsProbeResult 契约
# ============================================================
def test_g5_models_probe_result_namedtuple():
    from llmsec.core.probe import ModelsProbeResult, probe_service

    class _Cfg:
        api_key = "k"
        base_url = "http://x"
        model = "m"

    # 复用第一段结果（错误形态）：probe_service 直接落不可达，不再重发 models.list
    r = probe_service("judge", _Cfg(),
                      models_result=ModelsProbeResult(None, None, "boom"))
    assert r["reachable"] is False and r["error"] == "boom"
    fields = ModelsProbeResult._fields
    assert fields == ("latency_ms", "model_ids", "error"), \
        f"G5: NamedTuple 字段契约（实得 {fields}）"


# ============================================================
# G7：mutator 不再自行 save（update 统一写回）
# ============================================================
def test_g7_index_ops_still_persist(tmp_path, monkeypatch):
    """别名清理后索引写入仍生效（update 写回路径的行为验证）。"""
    from control.core import env_snapshot as es

    monkeypatch.setattr(es, "ENV_SNAPSHOTS_DIR", tmp_path / "snaps")
    monkeypatch.setattr(es, "LLMSEC_REPO", tmp_path)
    es.create("s1", source="blank")
    names = [s["name"] for s in es.list_snapshots()]
    assert "s1" in names, "G7: create 后索引须持久化（update 写回）"
