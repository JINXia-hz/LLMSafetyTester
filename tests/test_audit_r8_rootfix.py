"""第 8 轮回归——复盘修复（P1 / P2 + 病根 4、病根 5 见各节）。

  - P1: weekend_hpo 的 best_overall 不得接收 None（跨 study 混合 None/数值
    时 `mv < None` TypeError）。
  - P2: cluster_viz 缓存淘汰的并发安全（to_thread 迁移后的新竞态）。
"""

from __future__ import annotations

import threading

import llmsec.core.config as cfg  # P9: TASK_LOG_DIR 动态读后统一 patch cfg

# ============================================================
# P1: best_overall 不接收 None
# ============================================================

def test_hpo_report_mixed_none_and_value_no_crash(tmp_path, monkeypatch):
    """stage1 缺 ci_half、stage2 有值：跳过 None 条目，best_overall 取数值条目。"""
    import pytest

    import llmsec.experiments.study as study_mod

    # scripts/ 已移出仓库（本地维护脚本）——本地有则测、CI 无则跳过
    wh = pytest.importorskip("scripts.weekend_hpo")

    fake_summaries = {
        "weekend_stage1": {"best": {"ci_half_mean": None, "ci_half_std": 0,
                                    "params": {"bad": 1}}},
        "weekend_stage2": {"best": {"ci_half_mean": 7.5, "ci_half_std": 0.2,
                                    "params": {"good": 2}}},
        "weekend_stage3": {"best": {"ci_half_mean": 9.0, "ci_half_std": 0.1,
                                    "params": {"worse": 3}}},
    }

    def fake_dir(name):
        d = tmp_path / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "study.yaml").write_text("name: x\n", encoding="utf-8")
        return d

    monkeypatch.setattr(study_mod, "study_dir", fake_dir)
    monkeypatch.setattr(
        study_mod.StudyConfig, "from_yaml",
        classmethod(lambda cls, path: type("C", (), {"name": path.parent.name})()))
    monkeypatch.setattr(study_mod, "summarize",
                        lambda cfg: fake_summaries.get(cfg.name, {}))

    rc = wh.cmd_report()
    assert rc == 0, "混合 None/数值时 cmd_report 不得崩溃（r8/P1）"


# ============================================================
# P2: cluster_viz 缓存并发淘汰
# ============================================================

def test_cache_put_concurrent_eviction_no_crash():
    """并发读写 + 触发上限淘汰，不得抛异常（r8/P2 语义在 r9 SigCache 上延续）。"""
    import random

    from llmsec.core.caches import SigCache
    from llmsec.server.routers import cluster_viz as cv

    cache = SigCache(maxsize=64)
    errors: list[Exception] = []

    def worker(seed):
        try:
            rnd = random.Random(seed)
            for i in range(400):
                cache.get(f"m{rnd.randrange(1000)}", float(i),
                          lambda i=i: {"n": i})
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(s,)) for s in range(6)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=10)
        assert not th.is_alive(), "压测线程应正常退出（未死锁）"

    assert not errors, f"并发缓存淘汰抛异常: {errors[:3]}"
    assert len(cache._data) <= 64, "缓存上限必须成立"
    # cluster_viz 的两个缓存必须是 SigCache 实例（迁移完整性）
    from llmsec.core.caches import SigCache as _SC
    assert isinstance(cv._PROJECTION_CACHE, _SC) and isinstance(cv._CUT_CACHE, _SC)


# ============================================================
# 病根1: units 代理持久化（phase 2 键空间与 state.json 严格同源）
# ============================================================

def test_phase2_units_file_written_and_preferred(tmp_path, monkeypatch):
    """phase 1 落 units.json；phase 2 在"聚类漂移"（重推导必产出不同键）时
    仍以文件为准——FPR 不再静默失效。"""
    import json

    from tests.test_audit_r7_high import _offline_runner_env

    rn, fixed_run, base_argv, deps = _offline_runner_env(tmp_path, monkeypatch)
    rn.main(base_argv[1:] + ["--phase", "1"], deps=deps)

    units_file = fixed_run / "units.json"
    assert units_file.exists(), "phase 1 应随 run 落盘 units.json"
    proxies = json.loads(units_file.read_text(encoding="utf-8"))
    assert proxies and all(k.startswith("c_") for k in proxies), (
        "units.json 的键应为 unit_id（c_<md5>），与 state.json 的 attacker_ratings 同源")

    # 模拟病根场景：特征配置漂移 → 重推导会得到完全不同的 unit_id 集合
    # （全部 method 并成一簇 → 1 个全新 unit_id）。修复前 phase 2 用重推导结果
    # 查 state.json 恒 miss；修复后以 units.json 为准不受影响。
    monkeypatch.setattr(rn, "_quick_precluster",
                        lambda tracker, methods: {m: 0 for m in methods})

    res = rn.main(base_argv[1:] + ["--phase", "2"], deps=deps)
    info = res["per_target"]["t1"]
    assert info.get("fpr") is not None, (
        "病根1：聚类漂移时 phase 2 必须以 units.json 为准（键空间与 state.json 同源），"
        "不得依赖重推导")


def test_phase2_falls_back_to_derivation_without_file(tmp_path, monkeypatch):
    """无 units.json（手工构造/极旧 run）时退回确定性重推导，流程不崩。"""

    from tests.test_audit_r7_high import _offline_runner_env

    rn, fixed_run, base_argv, deps = _offline_runner_env(tmp_path, monkeypatch)
    rn.main(base_argv[1:] + ["--phase", "1"], deps=deps)
    (fixed_run / "units.json").unlink()  # 模拟无文件

    res = rn.main(base_argv[1:] + ["--phase", "2"], deps=deps)
    assert res["per_target"]["t1"].get("fpr") is not None, (
        "无 units.json 时应退回 H-1 的确定性重推导口径")


# ============================================================
# 病根2: 总线声明式路由
# ============================================================

def test_menxia_subscribed_kinds_route_correctly():
    """表驱动交叉验证（H-2 病根）：门下省订阅的每个 kind，其路由必须是
    MENXIA 或 ALL——bus 过滤是 to_dept in (dept, ALL)，其它取值即静默失联。"""
    from control.agent import bus as bus_mod
    from control.agent.menxia.listener import reinit_menxia

    bus_mod.reset_bus()
    reinit_menxia()  # 本测试进程尚未初始化门下省订阅

    menxia_kinds = [k for (_dept, kinds, _cb) in bus_mod.get_bus()._subs
                    if _dept == bus_mod.MENXIA for k in kinds]
    assert menxia_kinds, "前置：门下省应有总线订阅"
    for kind in menxia_kinds:
        route = bus_mod.KIND_ROUTES.get(kind)
        assert route in (bus_mod.MENXIA, bus_mod.ALL), (
            f"kind={kind} 路由到 {route!r}——门下省订阅该消息但永远收不到（H-2 病根）")


def test_notify_routed_resolves_and_rejects_unknown():
    from control.agent import bus as bus_mod

    bus_mod.reset_bus()
    got = []
    bus_mod.get_bus().subscribe(bus_mod.MENXIA, [bus_mod.KIND_PLAN_APPROVED],
                                lambda m: got.append(m))
    bus_mod.notify_routed(bus_mod.KIND_PLAN_APPROVED, from_dept=bus_mod.USER,
                           plan_id="p1")
    assert got and got[0].to_dept == bus_mod.KIND_ROUTES[bus_mod.KIND_PLAN_APPROVED]

    import pytest
    with pytest.raises(KeyError):
        bus_mod.notify_routed("no_such_kind", from_dept=bus_mod.USER)


def test_no_hand_addressed_notify_outside_bus(monkeypatch):
    """源码守卫：control 层不得再手写 to_dept=（发布一律走路由表）。

    H-2 的教训是调用点拼错部门名无任何报错；把"不许手写地址"固化为
    源码扫描断言，新增消息必须登记 KIND_ROUTES。
    """
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "control"
    offenders = []
    for py in root.rglob("*.py"):
        if py.name == "bus.py":
            continue
        for lineno, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if "to_dept=" in line and not stripped.startswith("#"):
                offenders.append(f"{py.relative_to(root)}:{lineno}")
    assert not offenders, (
        f"发现手写 to_dept 的发布点（应改用 notify_routed/登记 KIND_ROUTES）: {offenders}")


# ============================================================
# 病根4: load_artifact 异常分类
# ============================================================

def test_load_artifact_oserror_not_corruption(tmp_path, monkeypatch):
    """IO 错误（如 PermissionError）不伪装成 CorruptedFileError（r8/病根4）。"""
    import joblib
    import pytest

    from llmsec.core.io import load_artifact

    f = tmp_path / "cache.pkl"
    f.write_bytes(b"\x80\x05dummy")

    # joblib.load 内部自行 open——在 joblib 命名空间模拟瞬时占用
    def denied_load(path, *a, **kw):
        raise PermissionError(5, "Access is denied (simulated)")
    monkeypatch.setattr(joblib, "load", denied_load)

    with pytest.raises(PermissionError):
        load_artifact(f, strict=True)
    # 非严格：IO 错误静默返回 default
    assert load_artifact(f, default={"d": 1}) == {"d": 1}


def test_load_artifact_garbage_is_corruption(tmp_path):
    import pytest

    from llmsec.core.io import CorruptedFileError, load_artifact

    f = tmp_path / "cache.pkl"
    f.write_bytes(b"not a pickle at all")
    assert load_artifact(f, default="D") == "D"
    with pytest.raises(CorruptedFileError):
        load_artifact(f, strict=True)


# ============================================================
# 病根5: task 终态迁移中央化
# ============================================================

def test_spawn_oserror_persists_row_and_advances(tmp_path, monkeypatch):
    """Popen 失败：终态入库（P4：目录库行取代 meta.json）+ 推进队列。"""
    import llmsec.core.config as cfg
    import llmsec.server.task_manager as tm

    tm.TASKS.clear()
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(cfg, "TASK_LOG_DIR", log_dir)
    monkeypatch.setattr(cfg, "CATALOG_DB", tmp_path / "catalog.db")
    from llmsec.storage import db as storage_db
    storage_db.close()
    try:
        t1 = {"kind": "r8", "cmd": "x", "argv": ["x"], "env_override": None,
              "meta": None, "proc": None, "log_path": log_dir / "q1.log",
              "log_file": None, "status": "queued", "started_at": "now",
              "_task_id": "q1"}
        tm.TASKS["q1"] = t1

        def boom_popen(*a, **kw):
            raise OSError("spawn denied (simulated)")
        monkeypatch.setattr(tm.subprocess, "Popen", boom_popen)

        tm._advance_queue("r8")
        assert t1["status"] == "failed"
        from llmsec.storage import catalog
        row = catalog.get_task("q1", tasks_dir=None, db_path=cfg.CATALOG_DB)
        assert row is not None and row.status == "failed", (
            "r8/病根5：spawn 失败的终态必须入库（外部可见）")
    finally:
        tm.TASKS.clear()
        storage_db.close()


def test_refresh_terminal_advances_queue(tmp_path, monkeypatch):
    """running 任务自然结束（poll 返回）后，同 kind 的 queued 必须被拉起。"""
    import llmsec.server.task_manager as tm

    tm.TASKS.clear()
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(cfg, "TASK_LOG_DIR", log_dir)
    try:
        spawned = []

        def fake_spawn(tid, t):
            spawned.append(tid)
            t["status"] = "running"
        monkeypatch.setattr(tm, "_spawn", fake_spawn)

        class _FakeProc:
            def poll(self):
                return 0            # 已正常退出
            @property
            def returncode(self):
                return 0

        t1 = {"kind": "r8b", "cmd": "x", "argv": ["x"], "env_override": None,
              "meta": None, "proc": _FakeProc(), "log_path": log_dir / "q1.log",
              "log_file": None, "status": "running", "started_at": "now",
              "_task_id": "q1"}
        t2 = {"kind": "r8b", "cmd": "x", "argv": ["x"], "env_override": None,
              "meta": None, "proc": None, "log_path": log_dir / "q2.log",
              "log_file": None, "status": "queued", "started_at": "now",
              "_task_id": "q2"}
        tm.TASKS["q1"] = t1
        tm.TASKS["q2"] = t2

        tm._refresh_task_status(t1)
        assert t1["status"] == "success"
        assert spawned == ["q2"], "r8/病根5：终态刷新必须经 _finish 推进队列"
    finally:
        tm.TASKS.clear()
