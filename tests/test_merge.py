"""merge 测试：通用 R 矩阵合并 + fork→merge 回路。

验证单元化的「显式统一」动作：plan 预览正确、execute 真合并、dry-run 不写、
多源合并、--models 过滤、ws↔global 双向。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmsec.core import config as cfg
from llmsec.core.results import ResultsMatrix


@pytest.fixture
def iso_output(monkeypatch, tmp_path):
    """重定向全局 output 到 tmp_path（隔离真实数据）。"""
    out = tmp_path / "output"
    state = out / "state"
    state.mkdir(parents=True)
    (out / "runs").mkdir()
    (out / "predictors").mkdir()
    (out / "workspaces").mkdir()

    monkeypatch.setattr(cfg, "OUTPUT_DIR", out)
    monkeypatch.setattr(cfg, "STATE_DIR", state)
    monkeypatch.setattr(cfg, "RESULTS_DB", state / "results.db")
    monkeypatch.setattr(cfg, "RESULTS_FILE", state / "results.json")
    monkeypatch.setattr(cfg, "ELO_CACHE_FILE", state / "elo_cache.json")

    from llmsec.management import merge as merge_mod
    monkeypatch.setattr(merge_mod, "RESULTS_DB", state / "results.db")
    monkeypatch.setattr(merge_mod, "WORKSPACES_DIR", out / "workspaces")
    monkeypatch.setattr(merge_mod, "OUTPUT_DIR", out)
    return out


def _save_R(path: Path, records: dict) -> None:
    """造一个 R 矩阵文件。records = {model: [(record, score), ...]}。"""
    R = ResultsMatrix()
    for model, recs in records.items():
        for rec, score in recs:
            R.upsert(rec, model, score, ts=1, extra={"unit": "u", "round": 1})
    R.save(path)


class TestMerge:
    def test_plan_detects_new_records(self, iso_output):
        from llmsec.management import merge
        global_R = cfg.RESULTS_DB
        ws_R = cfg.OUTPUT_DIR / "workspaces" / "ws1" / "results.json"
        ws_R.parent.mkdir(parents=True)
        _save_R(global_R, {"mA": [("r1", 1.0), ("r2", 0.5)]})
        _save_R(ws_R, {"mA": [("r1", 1.0), ("r2", 0.5), ("r3", 0.0)]})  # r3 新

        plan = merge.plan_merge(["ws:ws1"], "global")
        assert plan.dry_run is True
        pm = plan.extra["per_model"]["mA"]
        assert pm["target_existing"] == 2
        assert pm["source_records"] == 3
        assert pm["new_to_target"] == 1
        assert plan.extra["total_new"] == 1

    def test_execute_merges_new_records(self, iso_output):
        from llmsec.management import merge
        global_R = cfg.RESULTS_DB
        ws_R = cfg.OUTPUT_DIR / "workspaces" / "ws1" / "results.json"
        ws_R.parent.mkdir(parents=True)
        _save_R(global_R, {"mA": [("r1", 1.0)]})
        _save_R(ws_R, {"mA": [("r2", 0.5)], "mB": [("r1", 0.0)]})

        done = merge.execute_merge(["ws:ws1"], "global")
        R = ResultsMatrix.load(global_R)
        assert R.n_for_model("mA") == 2          # r1 + r2
        assert R.n_for_model("mB") == 1          # 新增 mB 列
        assert done.extra["total_merged"] >= 2

    def test_dry_run_does_not_write(self, iso_output):
        from llmsec.management import merge
        global_R = cfg.RESULTS_DB
        ws_R = cfg.OUTPUT_DIR / "workspaces" / "ws1" / "results.json"
        ws_R.parent.mkdir(parents=True)
        _save_R(global_R, {"mA": [("r1", 1.0)]})
        _save_R(ws_R, {"mA": [("r2", 0.5)]})

        merge.plan_merge(["ws:ws1"], "global")  # dry-run
        R = ResultsMatrix.load(global_R)
        assert R.n_for_model("mA") == 1  # 未变

    def test_models_filter(self, iso_output):
        from llmsec.management import merge
        global_R = cfg.RESULTS_DB
        ws_R = cfg.OUTPUT_DIR / "workspaces" / "ws1" / "results.json"
        ws_R.parent.mkdir(parents=True)
        _save_R(global_R, {})
        _save_R(ws_R, {"mA": [("r1", 1.0)], "mB": [("r2", 0.5)]})

        merge.execute_merge(["ws:ws1"], "global", models=["mA"])
        R = ResultsMatrix.load(global_R)
        assert R.n_for_model("mA") == 1
        assert R.n_for_model("mB") == 0  # 被过滤掉

    def test_multiple_sources_merge(self, iso_output):
        """两个 ws 合并到 global，各自的新记录都进。"""
        from llmsec.management import merge
        global_R = cfg.RESULTS_DB
        ws1 = cfg.OUTPUT_DIR / "workspaces" / "ws1" / "results.json"
        ws2 = cfg.OUTPUT_DIR / "workspaces" / "ws2" / "results.json"
        ws1.parent.mkdir(parents=True)
        ws2.parent.mkdir(parents=True)
        _save_R(global_R, {})
        _save_R(ws1, {"mA": [("r1", 1.0)]})
        _save_R(ws2, {"mA": [("r2", 0.5)], "mB": [("r3", 0.0)]})

        merge.execute_merge(["ws:ws1", "ws:ws2"], "global")
        R = ResultsMatrix.load(global_R)
        assert R.n_for_model("mA") == 2
        assert R.n_for_model("mB") == 1

    def test_merge_to_workspace_target(self, iso_output):
        """target=ws:<name>：合并进另一个工作区（分支融合）。"""
        from llmsec.management import merge
        ws1 = cfg.OUTPUT_DIR / "workspaces" / "ws1" / "results.json"
        ws2 = cfg.OUTPUT_DIR / "workspaces" / "ws2" / "results.json"
        ws1.parent.mkdir(parents=True)
        ws2.parent.mkdir(parents=True)
        _save_R(ws1, {"mA": [("r1", 1.0)]})
        _save_R(ws2, {"mB": [("r2", 0.5)]})

        merge.execute_merge(["ws:ws1"], "ws:ws2")
        R = ResultsMatrix.load(ws2.with_suffix(".db"))  # 阶段 2：目标真相在 db
        assert R.n_for_model("mA") == 1   # 从 ws1 融合进来
        assert R.n_for_model("mB") == 1   # 原有保留

    def test_path_source(self, iso_output, tmp_path):
        """source 直接给目录路径（work-dir）。"""
        from llmsec.management import merge
        global_R = cfg.RESULTS_DB
        workdir = tmp_path / "some-workdir"
        workdir.mkdir()
        _save_R(global_R, {})
        _save_R(workdir / "results.json", {"mA": [("r1", 1.0)]})

        merge.execute_merge([str(workdir)], "global")
        R = ResultsMatrix.load(global_R)
        assert R.n_for_model("mA") == 1

    def test_overwrite_same_record(self, iso_output):
        """同 record+model：source 覆盖 target（upsert 语义）。"""
        from llmsec.management import merge
        global_R = cfg.RESULTS_DB
        ws_R = cfg.OUTPUT_DIR / "workspaces" / "ws1" / "results.json"
        ws_R.parent.mkdir(parents=True)
        _save_R(global_R, {"mA": [("r1", 1.0)]})       # 旧分 1.0
        _save_R(ws_R, {"mA": [("r1", 0.2)]})            # 新分 0.2

        merge.execute_merge(["ws:ws1"], "global")
        R = ResultsMatrix.load(global_R)
        assert R.n_for_model("mA") == 1                 # 仍 1 条
        assert R.get("r1", "mA").eval_score == 0.2      # 被覆盖


class TestMergeCLI:
    def test_cli_merge_dry_run_json(self, iso_output, capsys, monkeypatch):
        from llmsec.management import __main__ as cli
        global_R = cfg.RESULTS_DB
        ws_R = cfg.OUTPUT_DIR / "workspaces" / "ws1" / "results.json"
        ws_R.parent.mkdir(parents=True)
        _save_R(global_R, {"mA": [("r1", 1.0)]})
        _save_R(ws_R, {"mA": [("r2", 0.5)]})

        monkeypatch.setattr("sys.argv",
                            ["m", "merge", "--sources", "ws:ws1", "--target", "global", "--json"])
        rc = cli.main()
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["action"] == "merge"
        assert data["dry_run"] is True
        assert data["extra"]["total_new"] == 1
        # dry-run 未写
        assert ResultsMatrix.load(global_R).n_for_model("mA") == 1
