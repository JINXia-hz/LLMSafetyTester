"""MCP 工具层补充覆盖：query / actions / tasks / server。

与 test_mcp.py 同一策略（直接调 Python 工具函数，不走 MCP 协议，全程离线）：
  - tmp_path 重定向 llmsec + control 的全部路径常量，绝不写真实 output/
  - 外部副作用（subprocess 阈值 CLI / fork 的 invoker / 探活 / task 启动）一律 mock
  - 两步确认流在每个用例前后 clear()，防 token 泄漏到其他用例
"""

from __future__ import annotations

import json
import sys

import pytest

from llmsec.core.results import ResultsMatrix
from llmsec.mcp import confirm as confirm_mod
from llmsec.mcp.tools import actions, query
from llmsec.mcp.tools import tasks as tasks_mod


# ============================================================
# 公共 fixture / 造数
# ============================================================
@pytest.fixture
def iso_out(monkeypatch, tmp_path):
    """把 llmsec / control 的路径常量全部重定向到 tmp，返回 tmp output 根。

    覆盖三层：
      - llmsec.core.config（函数内 from-import 动态取值）
      - llmsec.core.results / management.*（模块级 from-import，须 patch 各自命名空间）
      - control.core.compare / control.config（同上）
    """
    from control.core import compare as compare_mod
    from llmsec.core import config as cfg
    from llmsec.management import common as common_mod
    from llmsec.management import runs as runs_mod

    out = tmp_path / "output"
    (out / "runs").mkdir(parents=True)
    (out / "state").mkdir(parents=True)
    res_file = out / "state" / "results.json"

    monkeypatch.setattr(cfg, "OUTPUT_DIR", out)
    monkeypatch.setattr(cfg, "RESULTS_FILE", res_file)
    monkeypatch.setattr(cfg, "RUNS_DIR", out / "runs")
    monkeypatch.setattr(cfg, "ELO_CACHE_FILE", out / "state" / "elo_cache.json")
    monkeypatch.setattr(cfg, "TASK_LOG_DIR", out / "tasks")
    monkeypatch.setattr(common_mod, "OUTPUT_DIR", out)
    monkeypatch.setattr(common_mod, "TRASH_DIR", out / ".trash")
    monkeypatch.setattr(runs_mod, "RUNS_DIR", out / "runs")
    monkeypatch.setattr(runs_mod, "RESULTS_FILE", res_file)
    monkeypatch.setattr(compare_mod, "RUNS_DIR", out / "runs")
    monkeypatch.setattr(compare_mod, "WORKSPACES_DIR", out / "workspaces")

    import control.config as control_cfg
    import control.core.workspace as ws_core

    monkeypatch.setattr(control_cfg, "OUTPUT_DIR", out)
    monkeypatch.setattr(control_cfg, "RUNS_DIR", out / "runs")
    monkeypatch.setattr(control_cfg, "WORKSPACES_DIR", out / "workspaces")
    monkeypatch.setattr(ws_core, "WORKSPACES_DIR", out / "workspaces")
    return out


def make_report(**over):
    """构造一份标准的 runner_report.json 内容。"""
    rep = {
        "target_model": "model-a",
        "security_level": "safe",
        "attack_phase": {"asr": 0.1, "rounds": 3, "total_tested": 50},
        "elo": {
            "boundary_elo": 1520.0, "converged": True, "coverage": 0.9,
            "ci_half": 10.0, "drift": 1.0, "boundary_confidence": 0.8,
        },
        "allergy": {"fpr": 0.02},
    }
    rep.update(over)
    return rep


def make_run(out, batch, target, report=None, tree=None):
    """造一个 run 目录（runs/<batch>/<target>/runner_report.json）。"""
    d = out / "runs" / batch / target
    d.mkdir(parents=True, exist_ok=True)
    rep = report if report is not None else make_report(target_model=target)
    (d / "runner_report.json").write_text(json.dumps(rep), encoding="utf-8")
    if tree is not None:
        (d / "security_tree.json").write_text(json.dumps(tree), encoding="utf-8")
    return d


def make_junk_run(out, batch, target):
    """造一个失败/垃圾 run：只有 attack_results.jsonl，无 runner_report。"""
    d = out / "runs" / batch / target
    d.mkdir(parents=True, exist_ok=True)
    (d / "attack_results.jsonl").write_text('{"id": "a1"}\n', encoding="utf-8")
    return d


def seed_results(path, model, unit_scores, prefix="rec"):
    """往 path 写一份小 R 矩阵：[(unit, eval_score), ...]。prefix 区分来源防行键撞车。"""
    R = ResultsMatrix()
    for i, (unit, score) in enumerate(unit_scores):
        R.upsert(f"{prefix}-{i}", model, float(score), ts=i,
                 extra={"unit": unit, "round": 0})
    R.save(path)
    return path


@pytest.fixture
def offline_thresholds(monkeypatch):
    """把 menxia.review 的阈值获取替换为 fallback 常量（不打 subprocess CLI）。"""
    from control.agent.menxia import review as review_mod
    from control.config import _FALLBACK_THRESHOLDS

    monkeypatch.setattr(review_mod, "get_thresholds",
                        lambda: dict(_FALLBACK_THRESHOLDS))


# ============================================================
# query.py 补充覆盖：run 查询 / 对比 / 报告
# ============================================================
class TestQueryRuns:
    def test_list_runs_empty(self, iso_out):
        assert query.list_runs() == []

    def test_list_runs_discovers_both_layouts_and_ignores_non_run_dirs(self, iso_out):
        make_run(iso_out, "2024-01-01_120000", "model-a")
        make_junk_run(iso_out, "2024-01-02_120000", "model-b")
        # 不匹配 RUN_NAME_RE 的目录应被忽略
        (iso_out / "runs" / "not-a-run").mkdir(parents=True)
        (iso_out / "runs" / "not-a-run" / "runner_report.json").write_text("{}", encoding="utf-8")

        runs = query.list_runs()
        names = [r["name"] for r in runs]
        assert names == ["2024-01-02_120000/model-b", "2024-01-01_120000/model-a"]  # 时间倒序
        good = runs[1]
        assert good["target"] == "model-a"
        assert good["asr"] == 0.1
        assert good["security_level"] == "safe"
        assert good["boundary_elo"] == 1520.0
        assert good["has_report"] is True
        assert good["size"] > 0
        assert runs[0]["has_report"] is False  # junk run 也被扫出

    def test_list_runs_filters(self, iso_out):
        make_run(iso_out, "2024-01-01_120000", "model-a")
        make_junk_run(iso_out, "2024-01-02_120000", "model-b")

        by_target = query.list_runs(target="model-a")
        assert [r["target"] for r in by_target] == ["model-a"]

        with_report = query.list_runs(has_report=True)
        assert [r["target"] for r in with_report] == ["model-a"]

        by_since = query.list_runs(since="2024-01-02")
        assert [r["target"] for r in by_since] == ["model-b"]

        assert query.list_runs(min_size=10**9) == []

    def test_list_runs_junk_only_reuses_scan(self, iso_out):
        make_run(iso_out, "2024-01-01_120000", "model-a")
        make_junk_run(iso_out, "2024-01-02_120000", "model-b")

        junk = query.list_runs(junk_only=True)
        assert [r["name"] for r in junk] == ["2024-01-02_120000/model-b"]

    def test_compare_runs_missing_annotated(self, iso_out):
        r = query.compare_runs(["ghost-1", "ghost-2"])
        assert r["runs"] == []
        assert r["missing"] == ["ghost-1", "ghost-2"]

    def test_compare_runs_two_runs_with_trees(self, iso_out):
        tree = {"dimensions": {"by_harm_type": {"weapon": {"asr": 0.2}}}}
        make_run(iso_out, "2024-01-01_120000", "model-a",
                 report=make_report(target_model="model-a", attack_phase={"asr": 0.2, "total_tested": 10}),
                 tree=tree)
        make_run(iso_out, "2024-01-02_120000", "model-b",
                 report=make_report(target_model="model-b", attack_phase={"asr": 0.4, "total_tested": 20}),
                 tree=tree)

        r = query.compare_runs(["2024-01-01_120000/model-a", "2024-01-02_120000/model-b"])
        assert r["missing"] == []
        assert {row["run"] for row in r["runs"]} == {
            "2024-01-01_120000/model-a", "2024-01-02_120000/model-b"}
        assert r["metrics"]["asr"]["2024-01-01_120000/model-a"] == 0.2
        # 威胁树类目对齐
        assert r["threat_diff"]["by_harm_type"]["weapon"][
            "2024-01-01_120000/model-a"] == 0.2

    def test_read_run_report_roundtrip_and_traversal(self, iso_out):
        make_run(iso_out, "2024-01-01_120000", "model-a", tree={"top_threats": []})

        r = query.read_run_report("2024-01-01_120000/model-a")
        assert r is not None
        assert r["report"]["target_model"] == "model-a"
        assert r["tree"] == {"top_threats": []}
        assert r["run_name"] == "2024-01-01_120000/model-a"

        # 不存在 / 路径穿越名 → None（safe_subpath 拒绝，错误被 _try 吞掉）
        assert query.read_run_report("ghost") is None
        assert query.read_run_report("../../etc") is None

    def test_assess_run_findings_critical_asr(self, iso_out, offline_thresholds):
        make_run(iso_out, "2024-01-01_120000", "model-a",
                 report=make_report(security_level="vulnerable",
                                    attack_phase={"asr": 0.8, "total_tested": 50}))
        r = query.assess_run_findings("2024-01-01_120000/model-a")
        assert r["run_name"] == "2024-01-01_120000/model-a"
        assert r["thresholds"]["PORTRAIT_ASR_SAFE"] == 0.3
        asr_findings = [f for f in r["findings"] if f["metric"] == "ASR"]
        assert asr_findings and asr_findings[0]["severity"] == "critical"

    def test_assess_run_findings_nonexistent(self, iso_out, offline_thresholds):
        r = query.assess_run_findings("ghost")
        assert r == {"run_name": "ghost", "findings": [], "error": "run 不存在或无报告"}

    def test_review_run_offline_template(self, iso_out, offline_thresholds):
        make_run(iso_out, "2024-01-01_120000", "model-a")
        r = query.review_run("2024-01-01_120000/model-a", use_llm=False)
        assert "安全等级=safe" in r["summary"]
        assert isinstance(r["findings"], list)
        assert r["metrics"]["target"] == "model-a"
        assert "model-a" in r["digest"]

        missing = query.review_run("ghost", use_llm=False)
        assert "error" in missing

    def test_get_thresholds_tool(self, monkeypatch):
        from control.agent.menxia import review as review_mod

        monkeypatch.setattr(review_mod, "get_thresholds",
                            lambda: {"PORTRAIT_MIN_TESTED": 7})
        assert query.get_thresholds() == {"PORTRAIT_MIN_TESTED": 7}


# ============================================================
# query.py 补充覆盖：R 矩阵 / Elo 派生
# ============================================================
class TestQueryResultsAndElo:
    def test_get_results_summary_missing_file(self, iso_out):
        r = query.get_results_summary()
        assert r["models"] == []
        assert "note" in r  # results.json 不存在提示

    def test_get_results_summary_with_data(self, iso_out):
        seed_results(iso_out / "state" / "results.json", "m-x",
                     [("u1", 1.0), ("u2", 0.0)])
        r = query.get_results_summary()
        assert r["models"] == ["m-x"]
        assert r["records"] == 2
        assert r["total_observations"] == 2
        assert r["results_file"].endswith("results.json")

    def test_elo_tools_missing_results_file(self, iso_out):
        err = query.elo_ranking("m-elo")
        assert "error" in err and err["model"] == "m-elo"
        assert "error" in query.elo_security_boundary("m-elo")
        assert "error" in query.elo_find_surprises("m-elo")

    def test_elo_ranking_and_boundary(self, iso_out):
        seed_results(iso_out / "state" / "results.json", "m-elo",
                     [("u-win", 5.0), ("u-loss", 0.0)])

        ranking = query.elo_ranking("m-elo")
        assert [row["unit"] for row in ranking] == ["u-win", "u-loss"]  # 降序
        assert ranking[0]["elo"] > ranking[1]["elo"]
        assert all("predicted" in row for row in ranking)

        boundary = query.elo_security_boundary("m-elo")
        assert boundary["defender"] == "m-elo"
        assert isinstance(boundary["boundary_elo"], float)
        assert "converged" in boundary and "methods_above_boundary" in boundary

    def test_elo_find_surprises_both_directions(self, iso_out):
        seed_results(iso_out / "state" / "results.json", "m-elo",
                     [("u-win", 5.0), ("u-loss", 0.0)])
        r = query.elo_find_surprises("m-elo")
        assert {w["attacker"] for w in r["weakness"]} == {"u-win"}  # 低分攻击成功
        assert {s["attacker"] for s in r["strength"]} == {"u-loss"}  # 攻击失败

    def test_elo_suggest_next_pairing(self, iso_out):
        # 用独立模型名 + 不同分数，避免与相邻用例命中同一列指纹的进程内 tracker 缓存
        seed_results(iso_out / "state" / "results.json", "m-pair",
                     [("p1", 3.0), ("p2", 7.0)])
        pairs = query.elo_suggest_next_pairing("m-pair", n=5)
        assert 1 <= len(pairs) <= 5
        assert all(p["defender"] == "m-pair" for p in pairs)
        assert {p["attacker"] for p in pairs} == {"p1", "p2"}

    def test_elo_unknown_model_returns_empty(self, iso_out):
        seed_results(iso_out / "state" / "results.json", "m-real", [("u1", 1.0)])
        assert query.elo_ranking("m-ghost") == []


# ============================================================
# query.py 补充覆盖：env / 探活 / 过敏 / 目标
# ============================================================
class TestQueryEnv:
    def test_get_allergy_report(self, iso_out):
        (iso_out / "allergy__model-a.json").write_text(
            json.dumps({"summary": {"false_positive_rate": 0.03}}), encoding="utf-8")
        r = query.get_allergy_report()
        assert r["summary"]["false_positive_rate"] == 0.03

    def test_get_allergy_report_empty(self, iso_out):
        assert query.get_allergy_report() == {}

    def test_list_targets_masks_api_keys(self, monkeypatch):
        from llmsec.core import config as cfg
        from llmsec.core.config import TargetConfig

        monkeypatch.setattr(cfg, "load_targets", lambda: {
            "alpha": TargetConfig(api_key="sk-abcdefgh12345678", model="m1"),
            "beta": TargetConfig(api_key="short", model="m2"),
        })
        targets = {t["name"]: t for t in query.list_targets()}
        assert targets["alpha"]["api_key"] == "sk-abcde***"  # 只显示前 8 位
        assert targets["beta"]["api_key"] == "***"
        assert targets["alpha"]["model"] == "m1"
        assert "sk-abcdefgh12345678" not in json.dumps(targets)  # 全 key 不泄漏

    def test_probe_targets_single_skips_services(self, monkeypatch):
        import llmsec.core.probe as probe_mod
        from llmsec.core import config as cfg
        from llmsec.core.config import TargetConfig

        monkeypatch.setattr(cfg, "load_targets", lambda: {
            "alpha": TargetConfig(model="m1"), "beta": TargetConfig(model="m2")})
        monkeypatch.setattr(probe_mod, "probe_target",
                            lambda n, c: {"name": n, "reachable": True, "latency_ms": 5})
        service_calls = []
        monkeypatch.setattr(probe_mod, "probe_service",
                            lambda n, c: service_calls.append(n))

        r = query.probe_targets(name="beta")
        assert [t["name"] for t in r["targets"]] == ["beta"]
        assert r["targets"][0]["reachable"] is True
        assert r["services"] == []
        assert service_calls == []  # name 模式不探 generator/judge

    def test_probe_targets_all_keeps_declaration_order(self, monkeypatch):
        import llmsec.core.probe as probe_mod
        from llmsec.core import config as cfg
        from llmsec.core.config import TargetConfig

        monkeypatch.setattr(cfg, "load_targets", lambda: {
            "alpha": TargetConfig(model="m1"), "beta": TargetConfig(model="m2")})
        monkeypatch.setattr(probe_mod, "probe_target",
                            lambda n, c: {"name": n, "reachable": True})
        monkeypatch.setattr(probe_mod, "probe_service",
                            lambda n, c: {"name": n, "reachable": False})

        r = query.probe_targets()
        assert [t["name"] for t in r["targets"]] == ["alpha", "beta"]
        assert [s["name"] for s in r["services"]] == ["generator", "judge"]

    def test_probe_targets_load_failure(self, monkeypatch):
        from llmsec.core import config as cfg

        def _boom():
            raise RuntimeError("bad .env")

        monkeypatch.setattr(cfg, "load_targets", _boom)
        r = query.probe_targets()
        assert r["targets"] == [] and r["services"] == []
        assert "load_targets 失败" in r["error"]


# ============================================================
# query.py 补充覆盖：Plan / 文牍 / workspace / params
# ============================================================
class TestQueryPlansAndGazettes:
    @pytest.fixture
    def iso_plans(self, monkeypatch, iso_out):
        from control.agent.shangshu import plan as plan_mod

        monkeypatch.setattr(plan_mod, "_PLANS_DIR", iso_out / "plans")
        monkeypatch.setattr(plan_mod, "_PLANS", {})
        return plan_mod

    def test_list_and_get_plan(self, iso_plans):
        p = iso_plans.Plan(
            intent="跑个测试",
            steps=[iso_plans.Step(id="s1", capability="run_evaluation",
                                  args={"target": "m"})])
        iso_plans.save_plan(p)

        listed = query.list_plans()
        assert [pl["id"] for pl in listed] == [p.id]
        assert listed[0]["intent"] == "跑个测试"

        got = query.get_plan(p.id)
        assert got["steps"][0]["capability"] == "run_evaluation"

    def test_get_plan_missing_and_traversal(self, iso_plans):
        assert query.get_plan("nope") is None
        assert query.get_plan("../evil") is None  # safe_component 拒穿越
        assert query.list_plans() == []

    def test_gazette_tools_roundtrip(self, monkeypatch, iso_out):
        from control.agent import gazette

        monkeypatch.setattr(gazette, "_GAZETTE_DIR", iso_out / "gazette")
        gazette.append_event("plan-1", gazette.EV_PLAN_DRAFTED, "尚书省",
                             intent="意图A", session_id="s1", detail={})
        gazette.append_event("plan-1", gazette.EV_STEP_STARTED, "尚书省",
                             step_id="s1", detail={"capability": "run_evaluation"})

        listed = query.list_gazettes()
        assert [g["plan_id"] for g in listed] == ["plan-1"]

        ctx = query.get_plan_context("plan-1")
        assert ctx["intent"] == "意图A"
        assert ctx["steps"]["s1"]["status"] == "running"
        assert ctx["events_count"] == 2

        events = query.read_plan_events("plan-1")
        assert [e["kind"] for e in events] == ["plan_drafted", "step_started"]

    def test_gazette_tools_missing(self, monkeypatch, iso_out):
        from control.agent import gazette

        monkeypatch.setattr(gazette, "_GAZETTE_DIR", iso_out / "gazette")
        assert query.list_gazettes() == []
        assert query.get_plan_context("nope") is None
        assert query.read_plan_events("nope") == []

    def test_list_workspaces_and_workspace_runs(self, iso_out):
        # workspace 内的 run（runner_report.json 在 <ws>/<target>/ 下）
        d = iso_out / "workspaces" / "exp1" / "model-a"
        d.mkdir(parents=True)
        (d / "runner_report.json").write_text(
            json.dumps(make_report(target_model="model-a")), encoding="utf-8")

        runs = query.list_workspace_runs()
        assert [r["name"] for r in runs] == ["ws:exp1/model-a"]
        assert runs[0]["target"] == "model-a"
        assert runs[0]["asr"] == 0.1

        # 索引为空 → list_workspaces 返回 []
        assert query.list_workspaces() == []

    def test_get_cluster_report(self, monkeypatch, iso_out):
        import llmsec.evaluation.cluster_analysis as ca

        f = iso_out / "cluster_report.json"
        monkeypatch.setattr(ca, "CLUSTER_REPORT_FILE", f)

        assert query.get_cluster_report() is None  # 未跑过聚类
        f.write_text(json.dumps({"n_clusters": 3}), encoding="utf-8")
        assert query.get_cluster_report() == {"n_clusters": 3}

    def test_get_params_all_and_category(self):
        r = query.get_params()
        assert isinstance(r, dict) and len(r) >= 5
        # 内层条目带 value/type/description 三元组
        some_cat = next(iter(r.values()))
        entry = next(iter(some_cat.values()))
        assert {"value", "type", "description"} <= set(entry)

        elo = query.get_params("elo")
        assert elo
        elo_keys = set()
        for group in elo.values():
            elo_keys.update(group)
        assert "K_FACTOR" in elo_keys

        bad = query.get_params("不存在的分组")
        assert "error" in bad and isinstance(bad["available"], list)


# ============================================================
# actions.py 补充覆盖：merge 入口校验（防路径穿越）
# ============================================================
class TestMergeSpecValidation:
    def test_global_and_ws_pass(self, iso_out):
        assert actions._validate_merge_spec("global") == "global"
        assert actions._validate_merge_spec("ws:exp1") == "ws:exp1"

    def test_target_rejects_bare_path(self, iso_out, tmp_path):
        with pytest.raises(ValueError, match="target 仅支持"):
            actions._validate_merge_spec(str(tmp_path), is_target=True)

    def test_source_rejects_traversal(self, iso_out):
        with pytest.raises(ValueError, match="穿越"):
            actions._validate_merge_spec("../outside")

    def test_source_rejects_path_outside_output(self, iso_out, tmp_path):
        with pytest.raises(ValueError, match="越界"):
            actions._validate_merge_spec(str(tmp_path / "elsewhere"))

    def test_source_inside_output_passes(self, iso_out):
        rel = "snapshots/foo"
        assert actions._validate_merge_spec(rel) == rel
        inside = str(iso_out / "snapshots" / "x")
        assert actions._validate_merge_spec(inside) == inside

    def test_preview_wraps_validation_errors(self, iso_out):
        r = actions.merge_workspaces_preview(sources=["../evil"], target="global")
        assert "error" in r and "hint" in r

        r2 = actions.merge_workspaces_preview(sources=["ws:x"], target="/abs/path")
        assert "error" in r2


# ============================================================
# actions.py 补充覆盖：preview → confirm 两步流
# ============================================================
class TestActionsConfirmFlows:
    def setup_method(self):
        confirm_mod.clear()

    def teardown_method(self):
        confirm_mod.clear()

    def test_delete_runs_full_flow_with_r_column(self, iso_out):
        d = make_run(iso_out, "2024-01-01_120000", "model-a")
        res_file = iso_out / "state" / "results.json"
        seed_results(res_file, "model-a", [("u1", 1.0), ("u2", 2.0)])

        prev = actions.delete_runs_preview(["2024-01-01_120000/model-a"], delete_r=True)
        assert prev["action"] == "delete_runs"
        kinds = {i["kind"] for i in prev["summary"]["items"]}
        assert "run_dir" in kinds and "r_column" in kinds
        assert "model-a" in prev["impact_note"]  # R 列影响写进提示
        assert prev["ttl_seconds"] == 300

        res = actions.delete_runs_confirm(prev["confirm_token"])
        assert res["status"] == "executed"
        assert not d.exists()                       # 目录已软删
        assert (iso_out / ".trash").exists()        # 进了回收站
        assert ResultsMatrix.load(res_file).n_for_model("model-a") == 0

        # token 一次性
        again = actions.delete_runs_confirm(prev["confirm_token"])
        assert again["status"] == "expired_or_already_confirmed"

    def test_delete_runs_preview_missing_run_marked(self, iso_out):
        prev = actions.delete_runs_preview(["ghost-run"], delete_r=False)
        assert prev["action"] == "delete_runs"
        kinds = {i["kind"] for i in prev["summary"]["items"]}
        assert kinds == {"missing"}

    def test_clean_caches_full_flow(self, iso_out, monkeypatch):
        from llmsec.management import caches as caches_mod

        elo_cache = iso_out / "state" / "elo_cache.json"
        elo_cache.write_text("{}", encoding="utf-8")
        tasks_dir = iso_out / "tasks"
        tasks_dir.mkdir()
        (tasks_dir / "a.log").write_text("log", encoding="utf-8")
        (tasks_dir / "a.progress.jsonl").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(caches_mod, "ELO_CACHE_FILE", elo_cache)
        monkeypatch.setattr(caches_mod, "TASK_LOG_DIR", tasks_dir)
        monkeypatch.setattr(caches_mod, "OUTPUT_DIR", iso_out)

        prev = actions.clean_caches_preview(["elo_cache", "task_logs"])
        assert prev["action"] == "clean_caches"
        items = prev["summary"]["items"]
        assert len([i for i in items if i["kind"] == "cache_file"]) == 3

        # 未知类别不炸，标 unknown_category
        bad = actions.clean_caches_preview(["bogus_cat"])
        assert any(i["kind"] == "unknown_category" for i in bad["summary"]["items"])

        res = actions.clean_caches_confirm(prev["confirm_token"])
        assert res["status"] == "executed"
        assert not elo_cache.exists()
        assert not (tasks_dir / "a.log").exists()
        assert not (tasks_dir / "a.progress.jsonl").exists()

    def test_merge_workspaces_full_flow(self, iso_out, monkeypatch):
        from llmsec.management import merge as merge_mod

        res_file = iso_out / "state" / "results.json"
        monkeypatch.setattr(merge_mod, "RESULTS_FILE", res_file)
        monkeypatch.setattr(merge_mod, "WORKSPACES_DIR", iso_out / "workspaces")
        seed_results(res_file, "m1", [("u0", 1.0)], prefix="global")
        ws_res = iso_out / "workspaces" / "exp1" / "results.json"
        seed_results(ws_res, "m1", [("u1", 1.0), ("u2", 0.0)], prefix="ws")

        prev = actions.merge_workspaces_preview(["ws:exp1"], target="global")
        assert prev["action"] == "merge_workspaces"
        assert prev["summary"]["extra"]["total_new"] == 2
        assert "confirm_token" in prev

        res = actions.merge_workspaces_confirm(prev["confirm_token"])
        assert res["status"] == "executed"
        assert ResultsMatrix.load(res_file).n_for_model("m1") == 3

        # models 过滤：只合并指定列
        seed_results(ws_res, "m2", [("w1", 1.0)])
        prev2 = actions.merge_workspaces_preview(["ws:exp1"], target="global",
                                                 models=["m2"])
        assert prev2["summary"]["extra"]["per_model"].keys() == {"m2"}

    def test_merge_confirm_fake_token(self):
        r = actions.merge_workspaces_confirm("no-such-token")
        assert r["status"] == "expired_or_already_confirmed"


# ============================================================
# actions.py 补充覆盖：env snapshot CRUD
# ============================================================
class TestEnvSnapshotTools:
    @pytest.fixture(autouse=True)
    def iso_env(self, monkeypatch, iso_out, tmp_path):
        import control.core.env_snapshot as es

        monkeypatch.setattr(es, "ENV_SNAPSHOTS_DIR", iso_out / "env_snapshots")
        monkeypatch.setattr(es, "LLMSEC_REPO", tmp_path)  # info.path 的 relative_to 基准
        global_env = tmp_path / ".env"
        global_env.write_text(
            "GENERATOR_API_KEY=sk-1234567890abcdef\nGENERATOR_BASE_URL=http://g\n",
            encoding="utf-8")
        monkeypatch.setattr(es, "_GLOBAL_ENV", global_env)
        self.global_env = global_env
        yield

    def setup_method(self):
        confirm_mod.clear()

    def teardown_method(self):
        confirm_mod.clear()

    def test_snapshot_crud_and_merge_to_global(self):
        info = actions.create_env_snapshot("s1", source="blank", note="实验")
        assert info["name"] == "s1"
        assert info["keys"] == []

        # 重名 → FileExistsError 被包装成 error dict
        dup = actions.create_env_snapshot("s1")
        assert "error" in dup

        edit = actions.edit_env_snapshot("s1", "TARGET_1_API_KEY", "k1")
        assert edit["keys"] == ["TARGET_1_API_KEY"]

        # 不受管理的 key 前缀被拒
        bad = actions.edit_env_snapshot("s1", "HACKED_KEY", "x")
        assert "error" in bad

        listed = actions.list_env_snapshots()
        assert [s["name"] for s in listed] == ["s1"]

        cfg = actions.get_env_config()
        assert cfg["configured"]["GENERATOR_API_KEY"] == "sk-12345***"  # 脱敏
        assert cfg["missing_essential"] == []

        # preview → confirm 写回全局
        prev = actions.merge_env_snapshot_to_global_preview("s1")
        assert prev["action"] == "merge_env_to_global"
        assert prev["will_change_keys"] == ["TARGET_1_API_KEY"]
        res = actions.merge_env_snapshot_to_global_confirm(prev["confirm_token"])
        assert res["status"] == "executed"
        assert "TARGET_1_API_KEY=k1" in self.global_env.read_text(encoding="utf-8")

        deleted = actions.delete_env_snapshot("s1")
        assert deleted["deleted"] == "s1"
        assert "error" in actions.delete_env_snapshot("s1")  # 再删报错
        assert actions.list_env_snapshots() == []

    def test_merge_preview_missing_snapshot(self):
        r = actions.merge_env_snapshot_to_global_preview("nope")
        assert "error" in r


# ============================================================
# actions.py 补充覆盖：workspace / snapshot
# ============================================================
class TestWorkspaceAndSnapshot:
    @pytest.fixture
    def iso_ws(self, monkeypatch, iso_out, tmp_path):
        import control.core.workspace as ws_mod

        monkeypatch.setattr(ws_mod, "WORKSPACES_DIR", iso_out / "workspaces")
        monkeypatch.setattr(ws_mod, "LLMSEC_REPO", tmp_path)

        def fake_export(source="global"):
            snap = iso_out / "snapshots" / "fake"
            snap.mkdir(parents=True, exist_ok=True)
            seed_results(snap / "results.json", "m1", [("u1", 1.0)])
            return {"snapshot": "snapshots/fake", "models": ["m1"], "records": 1}

        monkeypatch.setattr(ws_mod, "export_snapshot", fake_export)
        return ws_mod

    def test_fork_and_delete_workspace(self, iso_ws):
        info = actions.fork_workspace("exp1", note="隔离实验")
        assert info["name"] == "exp1"
        assert info["records"] == 1
        assert (iso_ws.WORKSPACES_DIR / "exp1" / "results.json").exists()
        assert [w["name"] for w in query.list_workspaces()] == ["exp1"]

        dup = actions.fork_workspace("exp1")
        assert "error" in dup  # 已存在

        r = actions.delete_workspace("exp1")
        assert r["deleted"] == "exp1"
        assert not (iso_ws.WORKSPACES_DIR / "exp1").exists()
        assert "error" in actions.delete_workspace("nope")

    def test_fork_workspace_export_failure(self, iso_ws, monkeypatch):
        monkeypatch.setattr(iso_ws, "export_snapshot", lambda source="global": {})
        r = actions.fork_workspace("exp-bad")
        assert "error" in r and "hint" in r

    def test_gc_merged_workspaces(self, iso_ws):
        actions.fork_workspace("exp-gc")
        # 标记已 merge 且超期（merged_at 拨回 2020 年）
        iso_ws._store.update(lambda idx: idx["workspaces"]["exp-gc"].update(
            {"merged": True, "merged_at": "2020-01-01T00:00:00", "merged_to": "global"}))

        r = actions.gc_merged_workspaces(older_than_days=7)
        assert [c["name"] for c in r["cleaned"]] == ["exp-gc"]
        assert r["gc_log_size"] >= 1  # 审计日志保留合并去向
        assert not (iso_ws.WORKSPACES_DIR / "exp-gc").exists()

    def test_export_snapshot_tool(self, iso_out, tmp_path, monkeypatch):
        import llmsec.management.snapshot as snap_mod

        res_file = iso_out / "state" / "results.json"
        seed_results(res_file, "m1", [("u1", 1.0)])
        monkeypatch.setattr(snap_mod, "OUTPUT_DIR", iso_out)
        monkeypatch.setattr(snap_mod, "SNAPSHOT_DIR", iso_out / "snapshots")
        monkeypatch.setattr(snap_mod, "ELO_CACHE_FILE", iso_out / "state" / "elo_cache.json")

        out_dir = iso_out / "snapshots" / "manual"
        r = actions.export_snapshot(source="global", out=str(out_dir))
        assert r["models"] == ["m1"] and r["records"] == 1
        assert (out_dir / "results.json").exists()
        assert (out_dir / "manifest.json").exists()

        assert "error" in actions.export_snapshot(source="bogus")  # 未知 source
        # out 越界（tmp 根不在 output/ 内）被拒
        assert "error" in actions.export_snapshot(source="global", out=str(tmp_path / "evil"))


# ============================================================
# tasks.py 补充覆盖：状态查询 / 取消 / 轮询
# ============================================================
class TestTaskQueries:
    @pytest.fixture
    def task_env(self, monkeypatch, tmp_path):
        import llmsec.server.task_manager as tm

        saved = dict(tm.TASKS)
        tm.TASKS.clear()
        log_dir = tmp_path / "tasklogs"
        log_dir.mkdir()
        monkeypatch.setattr(tm, "TASK_LOG_DIR", log_dir)
        yield tm, log_dir
        tm.TASKS.clear()
        tm.TASKS.update(saved)

    @staticmethod
    def _add_task(tm, log_dir, task_id="t1", status="queued"):
        tm.TASKS[task_id] = {
            "kind": "evaluate", "cmd": "-m runner", "argv": ["-m", "runner"],
            "env_override": None, "meta": {"targets": ["m1"]}, "proc": None,
            "log_path": log_dir / f"{task_id}.log", "log_file": None,
            "status": status, "started_at": "2024-01-01T00:00:00",
            "_task_id": task_id,
        }

    def test_status_log_and_progress(self, task_env):
        tm, log_dir = task_env
        self._add_task(tm, log_dir)
        (log_dir / "t1.log").write_text("hello log line\n", encoding="utf-8",
                                        newline="\n")
        (log_dir / "t1.progress.jsonl").write_text(
            json.dumps({"target": "m1", "round": 1, "asr": 0.5}) + "\n"
            + json.dumps({"target": "m1", "round": 2, "asr": 0.6}) + "\n",
            encoding="utf-8")

        v = tasks_mod.get_task_status("t1")
        assert v["status"] == "queued"
        assert v["log_tail"] == "hello log line\n"
        assert v["meta"]["targets"] == ["m1"]

        # 进度取每目标最后一条
        p = tasks_mod.get_task_progress("t1")
        assert p["kind"] == "evaluate"
        assert p["progress"]["m1"]["round"] == 2

        l = tasks_mod.get_task_log("t1")
        assert l == {"id": "t1", "log": "hello log line\n"}

        assert len(tasks_mod.list_tasks()) == 1

    def test_progress_and_log_missing(self, task_env):
        tm, log_dir = task_env
        self._add_task(tm, log_dir)
        assert tasks_mod.get_task_progress("t1")["progress"] == {}
        assert "note" in tasks_mod.get_task_log("t1")  # 日志为空提示

        assert tasks_mod.get_task_progress("ghost") == {"error": "任务不存在: ghost"}
        assert tasks_mod.get_task_log("ghost")["log"] == ""

    def test_cancel_task_lifecycle(self, task_env):
        tm, log_dir = task_env
        self._add_task(tm, log_dir)

        r = tasks_mod.cancel_task("t1")           # queued → 直接取消
        assert r["status"] == "cancelled"

        r2 = tasks_mod.cancel_task("t1")          # 已结束
        assert r2["error"] == "任务已结束，无法取消"
        assert r2["current_status"] == "cancelled"

        r3 = tasks_mod.cancel_task("ghost")       # 不存在
        assert r3["error"] == "任务不存在: ghost"


# ============================================================
# tasks.py 补充覆盖：run_evaluation / orchestrate_runs 启动链
# ============================================================
class TestTaskLaunch:
    @pytest.fixture
    def launch_env(self, monkeypatch, tmp_path):
        import control.core.env_snapshot as es
        import llmsec.server.task_manager as tm
        from llmsec.core import config as cfg

        attacks = tmp_path / "attacks"
        attacks.mkdir()
        (attacks / "l1.jsonl").write_text('{"id": "a1"}\n', encoding="utf-8")
        monkeypatch.setattr(cfg, "ATTACKS_DIR", attacks)
        monkeypatch.setattr(cfg, "load_targets",
                            lambda: {"m1": object(), "m2": object()})
        monkeypatch.setattr(es, "ENV_SNAPSHOTS_DIR", tmp_path / "env_snapshots")

        captured = {}

        def fake_start(kind, argv, *, env_override=None, meta=None):
            captured.update(kind=kind, argv=argv,
                            env_override=env_override, meta=meta)
            return {"id": f"{kind}-1", "kind": kind, "status": "queued",
                    "cmd": " ".join(argv)}

        monkeypatch.setattr(tm, "start_task", fake_start)
        return captured

    def test_run_evaluation_single_target(self, launch_env):
        r = tasks_mod.run_evaluation(target="m1", max_rounds=3, seed=42)
        assert r["id"] == "evaluate-1"
        assert "轮询" in r["next_step"]
        assert launch_env["kind"] == "evaluate"
        argv = launch_env["argv"]
        assert "--target" in argv and argv[argv.index("--target") + 1] == "m1"
        assert "--publish-global" in argv          # 看板/MCP 默认 publish
        assert launch_env["meta"] == {"targets": ["m1"], "max_rounds": 3,
                                      "input": argv[argv.index("--input") + 1]}

    def test_run_evaluation_multi_targets_full_concurrency(self, launch_env):
        r = tasks_mod.run_evaluation(targets=["m1", "m2"])
        assert r["status"] == "queued"
        argv = launch_env["argv"]
        assert argv[argv.index("--targets") + 1] == "m1,m2"
        assert argv[argv.index("--target-concurrency") + 1] == "2"  # 默认全并发

    def test_run_evaluation_param_overrides_become_env(self, launch_env):
        tasks_mod.run_evaluation(target="m1", param_overrides={"K_FACTOR": 32})
        assert launch_env["env_override"] == {"LLMSEC_PARAM_K_FACTOR": "32"}

    def test_run_evaluation_target_and_targets_mutually_exclusive(self, launch_env):
        r = tasks_mod.run_evaluation(target="a", targets=["b"])
        assert "互斥" in r["error"]

    def test_run_evaluation_missing_env_snapshot(self, launch_env):
        r = tasks_mod.run_evaluation(target="m1", env_snapshot="nope")
        assert "env 快照不存在" in r["error"]
        assert "hint" in r  # LaunchError 的 hint 透传

    def test_orchestrate_runs_validation(self, monkeypatch, iso_out):
        import control.core.env_snapshot as es

        monkeypatch.setattr(es, "ENV_SNAPSHOTS_DIR", iso_out / "env_snapshots")
        assert "specs 不能为空" in tasks_mod.orchestrate_runs([])["error"]
        assert "缺少 name" in tasks_mod.orchestrate_runs([{"target": "x"}])["error"]
        err = tasks_mod.orchestrate_runs([{"name": "e1"}], env_snapshot="nope")
        assert "env 快照不存在" in err["error"]

    def test_orchestrate_runs_starts_task(self, launch_env):
        r = tasks_mod.orchestrate_runs(
            [{"name": "e1", "target": "m1", "param_overrides": {"K_FACTOR": 32}},
             {"name": "e2"}],
            max_workers=2, compare_after=False)
        assert r["id"] == "orchestrate-1"
        assert "轮询" in r["next_step"]
        assert launch_env["kind"] == "orchestrate"
        script = launch_env["argv"][1]  # ["-c", script]
        assert "orchestrate" in script
        # param_overrides → LLMSEC_PARAM_* env 通道（脚本里内联构造）
        assert "LLMSEC_PARAM_" in script and "K_FACTOR" in script


# ============================================================
# server.py 补充覆盖：main 入口 + 全量工具名注册
# ============================================================
class TestServerMain:
    def test_main_http_transport(self, monkeypatch):
        pytest.importorskip("fastmcp")
        from llmsec.mcp import server as server_mod

        calls = {}

        class FakeMCP:
            def run(self, **kwargs):
                calls.update(kwargs)

        monkeypatch.setattr(server_mod, "create_server", lambda: FakeMCP())
        monkeypatch.setattr(
            sys, "argv",
            ["llmsec-mcp", "--transport", "http", "--host", "0.0.0.0", "--port", "9999"])
        server_mod.main()
        assert calls == {"transport": "http", "host": "0.0.0.0", "port": 9999}

    def test_main_stdio_transport(self, monkeypatch):
        pytest.importorskip("fastmcp")
        from llmsec.mcp import server as server_mod

        ran = []

        class FakeMCP:
            def run(self, **kwargs):
                ran.append(kwargs)

        monkeypatch.setattr(server_mod, "create_server", lambda: FakeMCP())
        monkeypatch.setattr(sys, "argv", ["llmsec-mcp"])
        server_mod.main()
        assert ran == [{}]  # stdio：无参 run

    def test_create_server_registers_every_tool_by_name(self):
        pytest.importorskip("fastmcp")
        import asyncio

        from llmsec.mcp.server import create_server

        names = {t.name for t in asyncio.run(create_server().list_tools())}
        expected = {
            # compute（7）
            "obfuscate_prompt", "compute_eval_score", "compute_math_score",
            "extract_math_answer", "extract_textual_features",
            "extract_report_metrics", "aggregate_metrics",
            # query（23）
            "list_runs", "compare_runs", "read_run_report", "assess_run_findings",
            "review_run", "get_thresholds", "get_results_summary",
            "elo_ranking", "elo_security_boundary", "elo_find_surprises",
            "elo_suggest_next_pairing", "get_allergy_report", "list_targets",
            "probe_targets", "list_workspaces", "list_workspace_runs",
            "get_cluster_report", "get_params", "list_plans", "get_plan",
            "list_gazettes", "get_plan_context", "read_plan_events",
            # actions（17）
            "delete_runs_preview", "delete_runs_confirm",
            "clean_caches_preview", "clean_caches_confirm",
            "fork_workspace", "export_snapshot", "create_env_snapshot",
            "edit_env_snapshot", "list_env_snapshots", "get_env_config",
            "delete_env_snapshot", "merge_workspaces_preview",
            "merge_workspaces_confirm", "merge_env_snapshot_to_global_preview",
            "merge_env_snapshot_to_global_confirm", "delete_workspace",
            "gc_merged_workspaces",
            # tasks（7）
            "run_evaluation", "get_task_status", "get_task_progress",
            "get_task_log", "cancel_task", "list_tasks", "orchestrate_runs",
        }
        missing = expected - names
        assert not missing, f"缺少工具注册: {missing}"
        assert names == expected, f"出现未预期的工具: {names - expected}"
