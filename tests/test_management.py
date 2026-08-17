"""management 包测试：dry-run 正确性 / 软删除可恢复 / 删R后 elo 失效 / snapshot 自包含。

所有测试用 monkeypatch 把 OUTPUT_DIR/RUNS_DIR 等重定向到 tmp_path，绝不碰真实数据。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmsec.core import config as cfg
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
    monkeypatch.setattr(cfg, "CATALOG_DB", state / "catalog.db")
    monkeypatch.setattr(cfg, "FEATURE_CACHE_FILE", out / "feature_cache.pkl")
    monkeypatch.setattr(cfg, "CLUSTER_RESULT_FILE", out / "cluster_result.pkl")

    # caches 的 OUTPUT_DIR/TASK_LOG_DIR 是模块级冻结导入（静态锚点，守卫不拦），patch 模块属性
    from llmsec.management import caches as _caches
    monkeypatch.setattr(_caches, "OUTPUT_DIR", out)
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
    # results 模块的 RESULTS_FILE（ResultsMatrix.load/save 默认读它）
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
        # 造一个 predictor + 一个 model_state 文件（P8：task_logs 类别已并入 gc-tasks）
        (cfg.PREDICTORS_DIR / "blend_abc.pkl").write_bytes(b"\x80\x04" * 100)
        (cfg.STATE_DIR / "probes.json").write_text("{}", encoding="utf-8")

        summaries = caches.all_category_summaries()
        by_name = {s["name"]: s for s in summaries}
        assert by_name["predictors"]["file_count"] == 1
        assert by_name["model_state"]["file_count"] == 1

    def test_clean_dry_run_no_touch(self, iso_output):
        from llmsec.management import caches
        plan = caches.plan_clean(["predictors"])
        assert plan.dry_run is True

    def test_clean_execute_soft_deletes(self, iso_output):
        from llmsec.management import caches
        from llmsec.management.common import TRASH_DIR
        (cfg.PREDICTORS_DIR / "blend_x.pkl").write_bytes(b"\x80\x04")

        plan = caches.plan_clean(["predictors"])
        done = caches.execute_clean(plan)
        assert len([i for i in done.items if i.kind in ("cache_file", "cache_dir")]) == 1
        assert not (cfg.PREDICTORS_DIR / "blend_x.pkl").exists()
        # trash 里有
        assert any(TRASH_DIR.rglob("blend_x.pkl"))


# ============================================================
# snapshot: 自包含 + 可作为 fork 起点
# ============================================================
class TestSnapshot:
    def test_export_global_self_contained(self, iso_output):
        """global 快照含 results.db + manifest，且 R 内容与源一致。"""
        from llmsec.management import snapshot
        R = ResultsMatrix()
        R.upsert("r1", "modelA", 1.0, ts=1)
        R.upsert("r2", "modelB", 0.5, ts=2)
        R.save()

        info = snapshot.export_snapshot("global")
        snap_dir = cfg.OUTPUT_DIR / info["snapshot"]
        assert (snap_dir / "catalog.db").exists()
        assert (snap_dir / "manifest.json").exists()
        # manifest 结构
        m = read_json(snap_dir / "manifest.json")
        assert m["source"] == "global"
        assert set(m["results"]["models"]) == {"modelA", "modelB"}
        assert m["results"]["observations"] == 2
        # 快照 R 内容与源一致（自包含）
        R2 = ResultsMatrix.load(snap_dir / "catalog.db")
        assert R2.n_for_model("modelA") == 1
        assert R2.get("r2", "modelB") is not None

    def test_export_can_seed_fork_workdir(self, iso_output, tmp_path):
        """快照 results.db 复制到新 work-dir 后能被 ResultsMatrix.load 读回
        （fork 语义自包含；生产路径 workspace.fork 已直调库级 clone）。"""
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
        shutil.copy2(snap_dir / "catalog.db", work_dir / "catalog.db")

        # 新工作区的 R 可独立加载，与全局隔离
        R_fork = ResultsMatrix.load(work_dir / "catalog.db")
        assert R_fork.n_for_model("modelA") == 1
        # 在 fork 里增删不影响全局
        R_fork.upsert("r3", "modelA", 2.0, ts=3)
        R_fork.save(work_dir / "catalog.db")
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


# ============================================================
# snapshot：run:<name> 源重建 / tar.gz 打包 / out 越界 / cmd 返回码
# ============================================================
class TestSnapshotSources:
    def test_export_run_source_rebuilds_r(self, iso_output):
        """run:<name> 源：从 run 目录 state.json 的 history 重建 R，缺 record 的条目跳过。"""
        from llmsec.management import snapshot
        run_dir = cfg.RUNS_DIR / "2026-08-11_120000" / "modelA"
        run_dir.mkdir(parents=True)
        write_json(run_dir / "state.json", {
            "defender_ratings": {"modelA": 1600.0},
            "round_defender_elos": {"modelA": [1600.0, 1601.0]},
            "history": [
                {"record": "r1", "defender": "modelA", "eval_score": 1.5,
                 "status": "fully_compliant", "round": 1, "unit": "c1"},
                {"record": "r2", "defender": "modelA", "eval_score": -1.0,
                 "status": "refused", "round": 2, "unit": "c2"},
                {"defender": "modelA", "eval_score": 2.0, "round": 2},  # 无 record → 跳过
            ],
        })
        write_json(run_dir / "runner_report.json", {"target_model": "modelA"})

        out_dir = cfg.OUTPUT_DIR / "snap_run"
        info = snapshot.export_snapshot("run:2026-08-11_120000/modelA", out=out_dir)
        assert info["models"] == ["modelA"], "❌1 重建 R 应只有 modelA 列"
        assert info["records"] == 2, "❌2 r1/r2 入 R，缺 record 条目应跳过"
        R2 = ResultsMatrix.load(out_dir / "catalog.db")
        assert R2.get("r1", "modelA").eval_score == 1.5, "❌4 快照 R 内容应与 history 一致"
        m = read_json(out_dir / "manifest.json")
        assert "state.json 重建" in m["source_desc"], "❌5 manifest 应标注来源描述"

    def test_export_run_source_defender_from_report(self, iso_output):
        """history 无 defender 键时，模型名回退 runner_report.target_model。"""
        from llmsec.management import snapshot
        run_dir = cfg.RUNS_DIR / "2026-08-12_090000" / "modelB"
        run_dir.mkdir(parents=True)
        write_json(run_dir / "state.json", {
            "history": [{"record": "r1", "eval_score": 0.5, "round": 1, "unit": "c1"}],
        })
        write_json(run_dir / "runner_report.json", {"target_model": "modelB"})
        info = snapshot.export_snapshot("run:2026-08-12_090000/modelB",
                                        out=cfg.OUTPUT_DIR / "snap_b")
        assert info["models"] == ["modelB"], "❌1 defender 应回退到报告里的 target_model"
        assert info["records"] == 1, "❌2 该条目应入 R"

    def test_export_run_source_old_layout_fallback(self, iso_output):
        """旧布局回退：ts/target 无 state.json 时用 ts/state.json。"""
        from llmsec.management import snapshot
        batch = cfg.RUNS_DIR / "2026-08-10_000000"
        batch.mkdir(parents=True)
        write_json(batch / "state.json", {
            "history": [{"record": "r9", "defender": "modelC", "eval_score": 1.0, "round": 1}],
        })
        info = snapshot.export_snapshot("run:2026-08-10_000000/modelC",
                                        out=cfg.OUTPUT_DIR / "snap_old")
        assert info["records"] == 1 and info["models"] == ["modelC"], \
            "❌1 旧布局 ts/state.json 应被回退命中"

    def test_export_out_escape_rejected(self, iso_output, tmp_path):
        """out 越界（output/ 之外，绝对路径或 ../ 相对）→ ValueError，绝不写出。"""
        from llmsec.management import snapshot
        with pytest.raises(ValueError, match="越界"):
            snapshot.export_snapshot("global", out=tmp_path / "escape")
        with pytest.raises(ValueError, match="越界"):
            snapshot.export_snapshot("global", out=Path("../evil"))
        assert not (tmp_path / "evil").exists(), "❌1 越界路径不得被创建"

    def test_export_relative_out_anchors_to_output(self, iso_output, tmp_path, monkeypatch):
        """相对 out 锚到 OUTPUT_DIR：校验与写盘同锚点，不落 CWD、不在 manifest 阶段崩。

        回归：原实现校验按 OUTPUT_DIR 解析、写盘按 CWD 解析——相对 out 先把
        快照写到 output/ 之外，再在 relative_to(OUTPUT_DIR) 处 ValueError。
        """
        from llmsec.management import snapshot
        R = ResultsMatrix()
        R.upsert("r1", "modelA", 1.0, ts=1)
        R.save()
        monkeypatch.chdir(tmp_path)  # CWD 与 OUTPUT_DIR 分离，验证不漂移

        info = snapshot.export_snapshot("global", out=Path("relsnap"))
        assert info["snapshot"] == "relsnap", f"❌1 snapshot 应为 output 内相对路径: {info['snapshot']}"
        assert (cfg.OUTPUT_DIR / "relsnap" / "catalog.db").exists(), "❌2 应落盘到 OUTPUT_DIR/relsnap"
        assert not (tmp_path / "relsnap").exists(), "❌3 CWD 下不得出现快照目录（写盘锚点漂移回归）"

        # 相对路径子目录同样锚定（tar.gz 打包分支已随 P3 删除——备份用 backup-r）
        info2 = snapshot.export_snapshot("global", out=Path("sub/inner"))
        assert (cfg.OUTPUT_DIR / "sub" / "inner" / "catalog.db").exists(), "❌4 子目录同样锚定"
        assert not (tmp_path / "sub").exists(), "❌5 不得写进 CWD"
        assert info2["snapshot"].replace("\\", "/") == "sub/inner"

    def test_snapshot_is_fresh_db_copy(self, iso_output):
        """快照 results.db 是源的独立副本：源后续写入不渗入快照（backup 语义，
        取代已删除的 tar.gz 打包分支）。"""
        from llmsec.management import snapshot
        R = ResultsMatrix()
        R.upsert("r1", "modelA", 1.0, ts=1)
        R.save()
        snapshot.export_snapshot("global", out=cfg.OUTPUT_DIR / "snapdb")
        snap = cfg.OUTPUT_DIR / "snapdb" / "catalog.db"
        assert snap.exists() and (cfg.OUTPUT_DIR / "snapdb" / "manifest.json").exists()

        R.upsert("r2", "modelA", 9.0, ts=2)
        R.save()
        R2 = ResultsMatrix.load(snap)
        assert R2.get("r2", "modelA") is None, "快照是导出时点的独立副本"
        assert R2.n_for_model("modelA") == 1

    def test_cmd_export_return_codes(self, iso_output, capsys):
        """cmd_export：未知源 / 无 state.json 的 run 返回 1；成功（json/人读）返回 0。"""
        from llmsec.management import snapshot
        assert snapshot.cmd_export("bogus") == 1, "❌1 未知 source 应返回 1"
        assert snapshot.cmd_export("run:no-such-run") == 1, "❌2 无 state.json 的 run 应返回 1"
        rc = snapshot.cmd_export("global", out=str(cfg.OUTPUT_DIR / "ok_json"), json_mode=True)
        assert rc == 0, "❌3 json 模式成功应返回 0"
        data = json.loads(capsys.readouterr().out)
        assert data["source"] == "global", "❌4 json 输出应含 source"
        rc2 = snapshot.cmd_export("global", out=str(cfg.OUTPUT_DIR / "ok_human"))
        assert rc2 == 0, "❌5 人读模式成功应返回 0"
        assert "snapshot" in capsys.readouterr().out, "❌6 人读模式应打印表格"


# ============================================================
# caches：list/clean 子命令、legacy 判定、未知类别告警
# ============================================================
class TestCachesCommands:
    def test_cmd_list_json_and_human(self, iso_output, capsys):
        """cmd_list：json 输出全部类别汇总；人读输出含表格与"绝不清"提示。"""
        from llmsec.management import caches
        rc = caches.cmd_list(json_mode=True)
        assert rc == 0, "❌1 cmd_list 应返回 0"
        data = json.loads(capsys.readouterr().out)
        assert data["count"] == len(caches.CACHE_CATEGORIES), "❌2 类别数应齐全"
        rc2 = caches.cmd_list()
        out = capsys.readouterr().out
        assert rc2 == 0 and "predictors" in out and "绝不清" in out, "❌4 人读模式应有表格与提示"

    def test_legacy_predictor_split(self, iso_output):
        """predictors 按现行前缀 blend_v2_ 判活；无版本盐的旧键归 predictors_legacy。"""
        from llmsec.management import caches
        (cfg.PREDICTORS_DIR / "blend_v2_abc.pkl").write_bytes(b"x" * 16)
        (cfg.PREDICTORS_DIR / "blend_abc.pkl").write_bytes(b"y" * 16)
        live = caches.category_summary("predictors")
        legacy = caches.category_summary("predictors_legacy")
        assert live["file_count"] == 2, "❌1 predictors 应含新旧全部 pkl"
        assert legacy["file_count"] == 1, "❌2 legacy 只应含 blend_ 旧前缀"
        assert legacy["rebuildable"] == "disposable", "❌3 legacy 应标记 disposable"

    def test_model_state_paths_exact(self, iso_output):
        """model_state 精确点名 probes.json + prescreen_model.joblib（P8 新类别）。"""
        from llmsec.management import caches
        (cfg.STATE_DIR / "probes.json").write_text("{}", encoding="utf-8")
        (cfg.STATE_DIR / "unrelated.txt").write_text("x", encoding="utf-8")
        s = caches.category_summary("model_state")
        assert s["file_count"] == 1, f"❌1 只应计 probes.json，实际 {s['file_count']}"

    def test_plan_clean_unknown_category_marked(self, iso_output):
        """未知类别不展开任何路径，标记 unknown_category 且提示。"""
        from llmsec.management import caches
        plan = caches.plan_clean(["no_such_cat"])
        kinds = [(i.kind, i.detail) for i in plan.items]
        assert any(k == "unknown_category" and "未知类别" in d for k, d in kinds), \
            f"❌1 未知类别未标记: {kinds}"
        assert not any(k == "cache_file" for k, _ in kinds), "❌2 未知类别不得展开路径"

    def test_execute_clean_skips_unknown_and_missing(self, iso_output):
        """执行期：未知类别条目跳过；已不存在的文件标记 missing 而非失败。"""
        from llmsec.management import caches
        (cfg.PREDICTORS_DIR / "gone.pkl").write_bytes(b"x")
        plan = caches.plan_clean(["predictors"])
        (cfg.PREDICTORS_DIR / "gone.pkl").unlink()  # 计划后文件消失
        done = caches.execute_clean(plan)
        by_kind = {i.kind: i for i in done.items}
        assert by_kind["missing"].detail == "已不存在", "❌1 缺失文件应标记 missing"

    def test_cmd_clean_dry_run_then_yes(self, iso_output, capsys):
        """clean：默认 dry-run 不动盘；--yes 软删到 .trash 且原文件可寻回。"""
        from llmsec.management import caches
        (cfg.PREDICTORS_DIR / "blend_v2_k.pkl").write_bytes(b"z" * 16)
        rc = caches.cmd_clean(["predictors"])
        assert rc == 0, "❌1 dry-run 应返回 0"
        out = capsys.readouterr().out
        assert "dry-run" in out and "--yes" in out, "❌2 应提示 dry-run 与 --yes"
        assert (cfg.PREDICTORS_DIR / "blend_v2_k.pkl").exists(), "❌4 dry-run 不得动盘"
        rc2 = caches.cmd_clean(["predictors"], yes=True)
        assert rc2 == 0, "❌5 --yes 应返回 0"
        assert not (cfg.PREDICTORS_DIR / "blend_v2_k.pkl").exists(), "❌6 --yes 后原文件应消失"
        assert any((cfg.OUTPUT_DIR / ".trash").rglob("blend_v2_k.pkl")), "❌7 应软删进 .trash"

    def test_cmd_clean_json_modes(self, iso_output, capsys):
        """clean --json：dry-run 输出 Plan 序列化且不动盘；--yes 输出执行结果且真删。"""
        from llmsec.management import caches
        rc = caches.cmd_clean(["predictors"], yes=False, json_mode=True)
        data = json.loads(capsys.readouterr().out)
        assert rc == 0 and data["_title"] == "clean (dry-run)" and data["dry_run"] is True, \
            "❌1 json dry-run 结构错误"
        rc2 = caches.cmd_clean(["predictors"], yes=True, json_mode=True)
        data2 = json.loads(capsys.readouterr().out)
        assert rc2 == 0 and data2["_title"] == "clean (executed)" and data2["dry_run"] is False, \
            "❌3 json 执行结构错误"

    def test_cmd_clean_invalid_category_warns(self, iso_output, capsys):
        """clean 带未知类别：提示未知与可选项，不影响返回 0。"""
        from llmsec.management import caches
        rc = caches.cmd_clean(["ghost_cat"])
        out = capsys.readouterr().out
        assert rc == 0, "❌1 未知类别不应非零退出"
        assert "未知类别" in out and "predictors" in out, "❌2 应提示未知类别与可选项"

    def test_missing_dirs_yield_empty_categories(self, iso_output, monkeypatch):
        """目录不存在时 predictor 类别安静为空（早退分支），不报错。"""
        from llmsec.management import caches
        assert caches._predictor_paths() == [], "❌1 目录缺失应返回空列表"
        assert caches.category_summary("predictors")["file_count"] == 0, "❌3 汇总应为 0"
