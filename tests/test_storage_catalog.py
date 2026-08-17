"""storage 目录库单元测试（阶段 1 地基）。

覆盖：三世代布局 + 卫星布局发现、同秒撞名可见性（RUN_NAME_RE 裂缝回归）、
增量对账（无变化零重扫 / 新变更入库 / 消失清理）、线程并发登记、
work-dir 隔离重绑、tasks/trials 登记、旧 dict 口径超集。
"""

from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path

import pytest

import llmsec.core.config as cfg
from llmsec.storage import contract
from llmsec.storage import db as storage_db

REPORT = {
    "target_model": "minimax",
    "security_level": "high",
    "attack_phase": {"asr": 0.25, "rounds": 3, "total_tested": 40},
    "elo": {"boundary_elo": 1650.0, "converged": True, "conv_rounds": 3},
    "allergy": {"fpr": 0.02},
}


@pytest.fixture()
def iso(tmp_path, monkeypatch):
    """路径隔离：runs 根与目录库都落 tmp（catalog 调期动态读 config）。"""
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(cfg, "RUNS_DIR", runs)
    monkeypatch.setattr(cfg, "CATALOG_DB", tmp_path / "state" / "catalog.db")
    storage_db.close()  # 清掉可能指向真实 output 的引擎缓存
    yield runs
    storage_db.close()


def _mk_run(root: Path, batch: str, target: str | None, *, report: bool = True,
            extra: str = "state.json") -> Path:
    d = root / batch / target if target else root / batch
    d.mkdir(parents=True, exist_ok=True)
    if report:
        (d / "runner_report.json").write_text(json.dumps(REPORT), encoding="utf-8")
    (d / extra).write_text("{}", encoding="utf-8")
    return d


# ============================================================
# 发现与布局
# ============================================================

def test_gen3_layout_discovered(iso):
    _mk_run(iso, "2026-08-17_120000", "minimax")
    rows = contract.query_runs()
    assert [r.name for r in rows] == ["2026-08-17_120000/minimax"]
    r = rows[0]
    assert r.layout == 3 and r.has_report and r.has_artifact
    assert r.target_model == "minimax"
    assert r.metrics["asr"] == 0.25 and r.metrics["boundary_elo"] == 1650.0


def test_same_second_suffix_visible(iso):
    """RUN_NAME_RE 裂缝回归：`<ts>_2` 撞名目录对发现可见（原正则带 $ 锚漏掉）。"""
    _mk_run(iso, "2026-08-17_120000", "minimax")
    _mk_run(iso, "2026-08-17_120000_2", "qwen")
    names = {r.name for r in contract.query_runs()}
    assert "2026-08-17_120000_2/qwen" in names
    assert contract.RUN_NAME_RE.match("2026-08-17_120000_2")


def test_gen1_flat_layout_discovered(iso):
    _mk_run(iso, "2026-08-06_145558", None, report=False, extra="attack_results.jsonl")
    rows = contract.query_runs()
    assert [r.name for r in rows] == ["2026-08-06_145558"]
    assert rows[0].layout == 2 and not rows[0].has_report and rows[0].has_artifact


def test_empty_batch_shell_invisible_by_default(iso):
    """零产物空壳默认不可见（与旧发现实现口径一致）；include_empty 时入册供清残。"""
    (iso / "2026-08-15_204228").mkdir(parents=True)
    assert contract.query_runs() == []
    st = contract.reconcile_runs(include_empty=True)
    assert st["adopted"] == 1
    assert len(contract.query_runs()) == 1


def test_workspace_satellite_layout(iso, tmp_path):
    """卫星布局：target 目录直接在根下；state/predictors/logs 隔离目录不含
    RUN_ARTIFACTS，自然被跳过。卫星库落 <root>/catalog.db，不碰全局库。"""
    ws = tmp_path / "ws_alpha"
    for sub in ("minimax", "state", "predictors", "logs"):
        (ws / sub).mkdir(parents=True)
    (ws / "minimax" / "runner_report.json").write_text(json.dumps(REPORT), encoding="utf-8")
    (ws / "state" / "results.json").write_text("{}", encoding="utf-8")

    rows = contract.query_runs(runs_root=ws)
    assert [r.name for r in rows] == ["ws_alpha/minimax"] or [r.name for r in rows] == ["minimax"]
    # batch = 根名（ws_alpha），target = 子目录名
    assert rows[0].batch == "ws_alpha" and rows[0].target == "minimax"
    assert (ws / "catalog.db").exists()
    # 全局库不受污染
    assert not cfg.CATALOG_DB.exists() or len(contract.query_runs()) == 0


# ============================================================
# 增量对账
# ============================================================

def test_reconcile_no_change_zero_rescan(iso):
    _mk_run(iso, "2026-08-17_120000", "minimax")
    contract.query_runs()
    st = contract.reconcile_runs()
    assert st == {"rescanned": 0, "removed": 0, "adopted": 0}


def test_reconcile_report_arrival_updates(iso):
    """报告落盘（目录 mtime 变化）后对账自动富化 has_report/metrics。"""
    d = _mk_run(iso, "2026-08-17_120000", "minimax", report=False)
    rows = contract.query_runs()
    assert not rows[0].has_report and rows[0].metrics is None
    (d / "runner_report.json").write_text(json.dumps(REPORT), encoding="utf-8")
    rows = contract.query_runs()
    assert rows[0].has_report and rows[0].metrics["asr"] == 0.25


def test_reconcile_removes_deleted(iso):
    _mk_run(iso, "2026-08-17_120000", "minimax")
    assert contract.query_runs()
    shutil.rmtree(iso / "2026-08-17_120000")
    st = contract.reconcile_runs()
    assert st["removed"] == 1
    assert contract.query_runs() == []


def test_rebuild_full(iso):
    _mk_run(iso, "2026-08-17_120000", "minimax")
    _mk_run(iso, "2026-08-16_090000", "qwen")
    contract.query_runs()
    st = contract.rebuild_runs(iso)
    assert st["adopted"] == 2
    assert len(contract.query_runs()) == 2


def test_register_run_then_reconcile_enriches(iso):
    """写入口轻登记（无产物）→ 对账不丢行；产物出现后富化。"""
    d = iso / "2026-08-17_130000" / "minimax"
    d.mkdir(parents=True)
    contract.register_run(d, batch="2026-08-17_130000", target="minimax")
    rows = contract.query_runs()  # reconcile 不产候选但已知行保留
    assert [r.name for r in rows] == ["2026-08-17_130000/minimax"]
    assert not rows[0].has_artifact
    (d / "runner_report.json").write_text(json.dumps(REPORT), encoding="utf-8")
    rows = contract.query_runs()
    assert rows[0].has_report


def test_allocate_runs_dir_suffix(iso):
    p1 = contract.allocate_runs_dir(iso, "2026-08-17_140000")
    p2 = contract.allocate_runs_dir(iso, "2026-08-17_140000")
    assert p1.name == "2026-08-17_140000" and p2.name == "2026-08-17_140000_2"


# ============================================================
# 并发与隔离
# ============================================================

def test_concurrent_register(iso):
    """多线程并发登记不丢行（连接池串行 + BEGIN IMMEDIATE）。"""
    errors: list[Exception] = []

    def worker(i: int) -> None:
        try:
            for j in range(5):
                d = iso / f"2026-08-17_15000{i}" / f"t{j}"
                d.mkdir(parents=True, exist_ok=True)
                (d / "state.json").write_text("{}", encoding="utf-8")
                contract.query_runs()
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(contract.query_runs()) == 20


def test_workdir_satellite_catalog(tmp_path, monkeypatch):
    """work-dir 隔离语义：CATALOG_DB 重绑后索引落 <wd>/catalog.db（monkeypatch
    自动还原；rebind_to_workdir 的全量重绑覆盖见 test_audit_r7_isolation）。"""
    wd = tmp_path / "trial_x"
    monkeypatch.setattr(cfg, "CATALOG_DB", wd / "catalog.db")
    storage_db.close()
    d = wd / "minimax"
    d.mkdir(parents=True)
    (d / "runner_report.json").write_text(json.dumps(REPORT), encoding="utf-8")
    rows = contract.query_runs(runs_root=wd)
    assert [r.target for r in rows] == ["minimax"]
    assert rows[0].batch == "trial_x"
    assert (wd / "catalog.db").exists()


# ============================================================
# tasks / trials
# ============================================================

def test_tasks_reconcile_lifecycle(iso, tmp_path, monkeypatch):
    tdir = tmp_path / "tasks"
    tdir.mkdir()
    monkeypatch.setattr(cfg, "TASK_LOG_DIR", tdir)

    def write_meta(tid: str, status: str) -> None:
        (tdir / f"{tid}.meta.json").write_text(json.dumps({
            "id": tid, "kind": tid.split("-", 1)[0], "cmd": "x", "pid": 123,
            "status": status, "started_at": "2026-08-17T14:30:25",
        }), encoding="utf-8")

    write_meta("evaluate-143025-ab12cd", "running")
    rows = contract.query_tasks()
    assert [r.task_id for r in rows] == ["evaluate-143025-ab12cd"]
    assert rows[0].pid == 123 and rows[0].status == "running"

    # 状态迁移（meta.json 覆盖写 → mtime 变化 → 对账更新）
    write_meta("evaluate-143025-ab12cd", "success")
    rows = contract.query_tasks()
    assert rows[0].status == "success"

    # 文件消失（gc）→ 行保留（P4 语义：reconcile 不反向删行——原生行无
    # meta.json，按文件删行会误杀；行清理是 gc-tasks 的职责）
    (tdir / "evaluate-143025-ab12cd.meta.json").unlink()
    rows = contract.query_tasks(reconcile=False)
    assert [r.task_id for r in rows] == ["evaluate-143025-ab12cd"]
    assert rows[0].status == "success"


def test_register_and_query_trials(iso):
    contract.register_trial("s1", 0, work_dir="/tmp/t0", target="m", seed="42")
    contract.register_trial("s1", 1, work_dir="/tmp/t1", target="m", seed="43")
    contract.update_trial("s1", 0, status="success", metrics={"conv_rounds": 5})
    rows = contract.query_trials("s1")
    assert [r.idx for r in rows] == [0, 1]
    assert rows[0].status == "success" and rows[0].metrics == {"conv_rounds": 5}
    assert contract.query_trials("nope") == []


# ============================================================
# 旧口径兼容
# ============================================================

def test_as_dict_superset_of_legacy_implementations(iso):
    """as_dict 必须是旧 management.discover_runs / data_query._discover_runs
    两套 dict 的字段超集（消费方零损失收口的前提）。"""
    _mk_run(iso, "2026-08-17_120000", "minimax")
    d = contract.query_runs()[0].as_dict()
    management_keys = {
        "name", "batch", "target", "target_model", "security_level",
        "asr", "boundary_elo", "has_report", "has_md", "mtime", "size",
    }
    data_query_keys = {
        "name", "batch", "target", "target_model", "has_report", "has_md",
        "has_tree", "has_cluster", "mtime",
    }
    assert management_keys <= set(d)
    assert data_query_keys <= set(d)
    assert d["target_model"] == "minimax"
    assert d["security_level"] == "high"
    assert isinstance(d["mtime"], str) and d["mtime"]
