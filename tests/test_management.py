"""management 包测试：dry-run 正确性 / 软删除可恢复 / 删R后 elo 失效 / snapshot 自包含。

所有测试用 monkeypatch 把 OUTPUT_DIR/RUNS_DIR 等重定向到 tmp_path，绝不碰真实数据。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmsec.core import config as cfg
from llmsec.core import results as res_mod
from llmsec.core.io import read_json, write_json
from llmsec.core.results import ResultsMatrix


# ============================================================
# fixture：在 tmp_path 下重建 output 结构
# ============================================================
@pytest.fixture
def iso_output(monkeypatch, tmp_path):
    """把所有相关路径常量重定向到 tmp_path/output，隔离真实数据。"""
    out = tmp_path / "output"
    out.mkdir()
    state = out / "state"
    state.mkdir()
    runs = out / "runs"
    runs.mkdir()
    predictors = out / "predictors"
    predictors.mkdir()
    tasks = out / "tasks"
    tasks.mkdir()
    snapshots = out / "snapshots"
    snapshots.mkdir()

    monkeypatch.setattr(cfg, "OUTPUT_DIR", out)
    monkeypatch.setattr(cfg, "STATE_DIR", state)
    monkeypatch.setattr(cfg, "RUNS_DIR", runs)
    monkeypatch.setattr(cfg, "PREDICTORS_DIR", predictors)
    monkeypatch.setattr(cfg, "TASK_LOG_DIR", tasks)
    monkeypatch.setattr(cfg, "RESULTS_FILE", state / "results.json")
    monkeypatch.setattr(cfg, "ELO_CACHE_FILE", state / "elo_cache.json")
    monkeypatch.setattr(cfg, "FEATURE_CACHE_FILE", out / "feature_cache.pkl")
    monkeypatch.setattr(cfg, "CLUSTER_RESULT_FILE", out / "cluster_result.pkl")

    # management.common 也持有 OUTPUT_DIR/TRASH_DIR 的 import-time 引用，patch 它们
    from llmsec.management import common
    monkeypatch.setattr(common, "OUTPUT_DIR", out)
    monkeypatch.setattr(common, "TRASH_DIR", out / ".trash")
    # snapshot 模块的 SNAPSHOT_DIR
    from llmsec.management import snapshot
    monkeypatch.setattr(snapshot, "SNAPSHOT_DIR", snapshots)
    monkeypatch.setattr(snapshot, "OUTPUT_DIR", out)
    # runs 模块的 RUNS_DIR
    from llmsec.management import runs as runs_mod
    monkeypatch.setattr(runs_mod, "RUNS_DIR", runs)
    # caches 模块的路径常量（import 时从 cfg 绑定）
    from llmsec.management import caches
    monkeypatch.setattr(caches, "ELO_CACHE_FILE", state / "elo_cache.json")
    monkeypatch.setattr(caches, "PREDICTORS_DIR", predictors)
    monkeypatch.setattr(caches, "FEATURE_CACHE_FILE", out / "feature_cache.pkl")
    monkeypatch.setattr(caches, "CLUSTER_RESULT_FILE", out / "cluster_result.pkl")
    monkeypatch.setattr(caches, "TASK_LOG_DIR", tasks)
    monkeypatch.setattr(caches, "OUTPUT_DIR", out)
    # results 模块的 RESULTS_FILE（ResultsMatrix.load/save 默认读它）
    monkeypatch.setattr(res_mod, "RESULTS_FILE", state / "results.json")
    return out


def _make_run(runs_dir: Path, ts: str, target: str, *, has_report=True, asr=None, level="inconclusive") -> Path:
    """在 tmp runs 目录下造一个 run 目录（新布局 ts/target/）。"""
    d = runs_dir / ts / target
    d.mkdir(parents=True)
    (d / "attack_results.jsonl").write_text('{"id":"x"}\n', encoding="utf-8")
    if has_report:
        report = {
            "target_model": target,
            "security_level": level,
            "attack_phase": {"asr": asr} if asr is not None else {},
            "elo": {"boundary_elo": 1700},
        }
        write_json(d / "runner_report.json", report)
    return d


# ============================================================
# results.remove_model / remove_record
# ============================================================
class TestRemoveMethods:
    def test_remove_model_deletes_column_and_cleans_empty_rows(self):
        R = ResultsMatrix()
        R.upsert("r1", "modelA", 1.0)
        R.upsert("r2", "modelA", 0.5)
        R.upsert("r1", "modelB", -1.0)
        assert R.n_for_model("modelA") == 2

        n = R.remove_model("modelA")
        assert n == 2
        assert R.n_for_model("modelA") == 0
        assert "modelA" not in R.all_models()
        # r2 只剩 modelA → 删后整行应被清空（record_row 为空）
        assert R.record_row("r2") == {}
        # r1 还有 modelB
        assert R.get("r1", "modelB") is not None

    def test_remove_record_deletes_row(self):
        R = ResultsMatrix()
        R.upsert("r1", "mA", 1.0)
        R.upsert("r1", "mB", 0.5)
        R.upsert("r2", "mA", 0.0)
        n = R.remove_record("r1")
        assert n == 2
        assert R.record_row("r1") == {}
        assert R.get("r2", "mA") is not None

    def test_column_payload_none_after_remove_triggers_elo_invalidation(self):
        """删 model 列后 column_payload 返回 None → elo_cache 指纹失效（契约验证）。"""
        R = ResultsMatrix()
        R.upsert("r1", "mA", 1.0, ts=1)
        assert R.column_payload("mA") is not None
        R.remove_model("mA")
        assert R.column_payload("mA") is None  # 指纹变 None → elo_state_for 返回 {}


# ============================================================
# runs: discover / filter / size
# ============================================================
class TestRunsList:
    def test_discover_runs_with_size(self, iso_output, monkeypatch):
        from llmsec.management import runs as runs_mod
        runs_dir = cfg.RUNS_DIR
        _make_run(runs_dir, "2026-08-11_120000", "modelA", asr=0.5)
        _make_run(runs_dir, "2026-08-10_100000", "modelB", asr=0.1)

        found = runs_mod.discover_runs()
        assert len(found) == 2
        # 时间倒序
        assert found[0]["batch"] == "2026-08-11_120000"
        # size > 0（attack_results.jsonl + runner_report.json 都有内容）
        assert all(r["size"] > 0 for r in found)
        # asr 富化
        assert found[0]["asr"] == 0.5

    def test_filter_by_target_and_since(self, iso_output):
        from llmsec.management import runs as runs_mod
        runs_dir = cfg.RUNS_DIR
        _make_run(runs_dir, "2026-08-11_120000", "modelA")
        _make_run(runs_dir, "2026-08-10_100000", "modelB")
        _make_run(runs_dir, "2026-08-11_130000", "modelB")

        all_runs = runs_mod.discover_runs()
        # 按 target
        assert len(runs_mod.filter_runs(all_runs, target="modelA")) == 1
        # 按 since
        assert len(runs_mod.filter_runs(all_runs, since="2026-08-11")) == 2
        # 组合
        assert len(runs_mod.filter_runs(all_runs, target="modelB", since="2026-08-11")) == 1

    def test_junk_detection(self, iso_output):
        from llmsec.management import runs as runs_mod
        runs_dir = cfg.RUNS_DIR
        _make_run(runs_dir, "2026-08-11_120000", "good", has_report=True)
        _make_run(runs_dir, "2026-08-11_130000", "bad", has_report=False)
        all_runs = runs_mod.discover_runs()
        junk = runs_mod.detect_junk(all_runs)
        assert len(junk) == 1
        assert junk[0]["target"] == "bad"


# ============================================================
# runs delete: dry-run + 软删除可恢复
# ============================================================
class TestRunsDelete:
    def test_delete_dry_run_does_not_touch_disk(self, iso_output, capsys):
        from llmsec.management import runs as runs_mod
        runs_dir = cfg.RUNS_DIR
        run_dir = _make_run(runs_dir, "2026-08-11_120000", "modelA")
        report_size_before = (run_dir / "runner_report.json").stat().st_size

        plan = runs_mod.plan_delete(["2026-08-11_120000/modelA"])
        assert plan.dry_run is True
        assert plan.total_size > 0
        assert len(plan.items) == 1
        # 目录仍在（dry-run 不动盘）
        assert run_dir.exists()
        assert (run_dir / "runner_report.json").stat().st_size == report_size_before

    def test_execute_delete_soft_deletes_and_recoverable(self, iso_output):
        """软删除：目录移到 .trash/，可 mv 回来恢复。"""
        from llmsec.management import runs as runs_mod
        from llmsec.management.common import TRASH_DIR
        runs_dir = cfg.RUNS_DIR
        run_dir = _make_run(runs_dir, "2026-08-11_120000", "modelA")
        report_content = (run_dir / "runner_report.json").read_bytes()

        plan = runs_mod.plan_delete(["2026-08-11_120000/modelA"])
        done = runs_mod.execute_delete(plan)
        assert len(done.items) == 1 and done.items[0].kind == "run_dir"
        # 原目录已移走
        assert not run_dir.exists()
        # .trash 下能找到
        trash_items = list(TRASH_DIR.rglob("runner_report.json"))
        assert len(trash_items) == 1
        # 内容一致（可恢复）
        assert trash_items[0].read_bytes() == report_content

    def test_delete_with_r_removes_model_column(self, iso_output):
        """--delete-r 真删 R 列，且 R save 走原子写。"""
        from llmsec.management import runs as runs_mod
        # 先在 R 里放一个 model 列
        R = ResultsMatrix()
        R.upsert("r1", "modelA", 1.0, ts=1)
        R.upsert("r2", "modelA", 0.5, ts=2)
        R.upsert("r1", "modelB", -1.0, ts=1)
        R.save()
        assert ResultsMatrix.load().n_for_model("modelA") == 2

        runs_dir = cfg.RUNS_DIR
        _make_run(runs_dir, "2026-08-11_120000", "modelA")

        plan = runs_mod.plan_delete(["2026-08-11_120000/modelA"], delete_r=True)
        assert plan.extra["r_models_affected"] == ["modelA"]
        assert plan.extra["r_rows_total"] == 2

        done = runs_mod.execute_delete(plan, delete_r=True)
        assert done.extra.get("r_rows_removed") == 2
        # R 已持久化，重读验证 modelA 列没了
        R2 = ResultsMatrix.load()
        assert R2.n_for_model("modelA") == 0
        assert R2.n_for_model("modelB") == 1  # 其他模型不受影响


# ============================================================
# caches: list + clean 软删除
# ============================================================
class TestCaches:
    def test_list_categories_with_sizes(self, iso_output):
        from llmsec.management import caches
        # 造 elo_cache + 一个 predictor + 一个 task log
        write_json(cfg.ELO_CACHE_FILE, {"_version": 3, "mA": {}})
        (cfg.PREDICTORS_DIR / "blend_abc.pkl").write_bytes(b"\x80\x04" * 100)
        (cfg.TASK_LOG_DIR / "eval-1.log").write_text("log line\n", encoding="utf-8")

        summaries = caches.all_category_summaries()
        by_name = {s["name"]: s for s in summaries}
        assert by_name["elo_cache"]["file_count"] == 1
        assert by_name["elo_cache"]["size"] > 0
        assert by_name["predictors"]["file_count"] == 1
        assert by_name["task_logs"]["file_count"] == 1

    def test_clean_dry_run_no_touch(self, iso_output):
        from llmsec.management import caches
        write_json(cfg.ELO_CACHE_FILE, {"_version": 3})
        plan = caches.plan_clean(["elo_cache"])
        assert plan.dry_run is True
        assert cfg.ELO_CACHE_FILE.exists()  # 未动

    def test_clean_execute_soft_deletes(self, iso_output):
        from llmsec.management import caches
        from llmsec.management.common import TRASH_DIR
        write_json(cfg.ELO_CACHE_FILE, {"_version": 3})
        (cfg.PREDICTORS_DIR / "blend_x.pkl").write_bytes(b"\x80\x04")

        plan = caches.plan_clean(["elo_cache", "predictors"])
        done = caches.execute_clean(plan)
        assert len([i for i in done.items if i.kind in ("cache_file", "cache_dir")]) == 2
        assert not cfg.ELO_CACHE_FILE.exists()
        assert not (cfg.PREDICTORS_DIR / "blend_x.pkl").exists()
        # trash 里有
        assert (TRASH_DIR / "state" / "elo_cache.json").exists() or \
               any(TRASH_DIR.rglob("elo_cache.json"))
        assert any(TRASH_DIR.rglob("blend_x.pkl"))


# ============================================================
# snapshot: 自包含 + 可作为 fork 起点
# ============================================================
class TestSnapshot:
    def test_export_global_self_contained(self, iso_output):
        """global 快照含 results.json + manifest，且 R 内容与源一致。"""
        from llmsec.management import snapshot
        R = ResultsMatrix()
        R.upsert("r1", "modelA", 1.0, ts=1)
        R.upsert("r2", "modelB", 0.5, ts=2)
        R.save()

        info = snapshot.export_snapshot("global")
        snap_dir = cfg.OUTPUT_DIR / info["snapshot"]
        assert (snap_dir / "results.json").exists()
        assert (snap_dir / "manifest.json").exists()
        # manifest 结构
        m = read_json(snap_dir / "manifest.json")
        assert m["source"] == "global"
        assert set(m["results"]["models"]) == {"modelA", "modelB"}
        assert m["results"]["results_total"] == 2
        # 快照 R 内容与源一致（自包含）
        R2 = ResultsMatrix.load(snap_dir / "results.json")
        assert R2.n_for_model("modelA") == 1
        assert R2.get("r2", "modelB") is not None

    def test_export_can_seed_fork_workdir(self, iso_output, tmp_path):
        """快照 results.json 复制到新 work-dir 后能被 ResultsMatrix.load 读回
        （模拟控制层 fork：复制快照 → 新 work-dir）。"""
        from llmsec.management import snapshot
        R = ResultsMatrix()
        R.upsert("r1", "modelA", 1.0, ts=1)
        R.save()

        info = snapshot.export_snapshot("global")
        snap_dir = cfg.OUTPUT_DIR / info["snapshot"]

        # 模拟控制层 fork：复制快照到全新 work-dir
        work_dir = tmp_path / "fork_env"
        work_dir.mkdir()
        import shutil
        shutil.copy2(snap_dir / "results.json", work_dir / "results.json")

        # 新工作区的 R 可独立加载，与全局隔离
        R_fork = ResultsMatrix.load(work_dir / "results.json")
        assert R_fork.n_for_model("modelA") == 1
        # 在 fork 里增删不影响全局
        R_fork.upsert("r3", "modelA", 2.0, ts=3)
        R_fork.save(work_dir / "results.json")
        R_global = ResultsMatrix.load()
        assert R_global.get("r3", "modelA") is None  # 全局未受污染

    def test_export_unknown_source_raises(self, iso_output):
        from llmsec.management import snapshot
        with pytest.raises(ValueError):
            snapshot.export_snapshot("bogus")

    def test_export_run_source_without_state_raises(self, iso_output):
        from llmsec.management import snapshot
        runs_dir = cfg.RUNS_DIR
        _make_run(runs_dir, "2026-08-11_120000", "modelA")
        # 该 run 无 state.json
        with pytest.raises(FileNotFoundError):
            snapshot.export_snapshot("run:2026-08-11_120000/modelA")


# ============================================================
# CLI: __main__ 端到端
# ============================================================
class TestCLI:
    def test_cli_runs_list_json(self, iso_output, capsys, monkeypatch):
        from llmsec.management import __main__ as cli
        _make_run(cfg.RUNS_DIR, "2026-08-11_120000", "modelA", asr=0.3)
        monkeypatch.setattr("sys.argv", ["llmsec.management", "runs", "list", "--json"])
        rc = cli.main()
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["count"] == 1
        assert data["runs"][0]["target"] == "modelA"

    def test_cli_delete_dry_run_default(self, iso_output, capsys, monkeypatch):
        from llmsec.management import __main__ as cli
        run_dir = _make_run(cfg.RUNS_DIR, "2026-08-11_120000", "modelA")
        monkeypatch.setattr("sys.argv",
                            ["llmsec.management", "runs", "delete", "2026-08-11_120000/modelA"])
        rc = cli.main()
        assert rc == 0
        assert run_dir.exists()  # dry-run 未删
        out = capsys.readouterr().out
        assert "dry-run" in out
