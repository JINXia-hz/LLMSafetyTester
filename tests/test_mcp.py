"""MCP 接口测试：工具注册 / 纯函数 / 两步确认 / 任务参数校验。

测试策略：
  - 直接调用工具包装函数（不启动 MCP server，不依赖网络）
  - 纯函数验证输入输出契约
  - 两步确认验证 token 生命周期（issue → confirm → 一次性）
  - 任务参数校验验证错误路径
  - 用 monkeypatch 隔离 OUTPUT_DIR，绝不碰真实数据
"""

from __future__ import annotations

import pytest

from llmsec.mcp import confirm as confirm_mod
from llmsec.mcp.tools import actions, compute
from llmsec.mcp.tools import tasks as tasks_mod


# ============================================================
# Tier 1：纯函数工具
# ============================================================
class TestComputeTools:
    """Tier 1 纯函数工具——零副作用，直接验证输入输出。"""

    def test_obfuscate_prompt_b64(self):
        r = compute.obfuscate_prompt("hello", "b64")
        assert "Base64" in r
        assert "aGVsbG8=" in r  # base64("hello")

    def test_obfuscate_prompt_all_methods(self):
        from llmsec.attacks.obfuscators import OBFUSCATORS

        for method in OBFUSCATORS:
            r = compute.obfuscate_prompt("test prompt", method)
            assert isinstance(r, str)
            assert len(r) > 0

    def test_obfuscate_prompt_invalid_method(self):
        with pytest.raises(ValueError, match="未知混淆方法"):
            compute.obfuscate_prompt("test", "nonexistent")

    def test_compute_math_score_correct(self):
        r = compute.compute_math_score("answer is [MATH:42]", 42)
        assert r["math_score"] == 2
        assert r["actual_answer"] == 42

    def test_compute_math_score_wrong(self):
        r = compute.compute_math_score("answer is [MATH:99]", 42)
        assert r["math_score"] == 1
        assert r["actual_answer"] == 99

    def test_compute_math_score_no_format(self):
        r = compute.compute_math_score("no math tag here", 42)
        assert r["math_score"] == 0
        assert r["actual_answer"] is None

    def test_extract_math_answer_last_match(self):
        # 取最后一个 [MATH:x]，不取首个
        assert compute.extract_math_answer("first [MATH:1] then [MATH:7]") == 7

    def test_extract_math_answer_none(self):
        assert compute.extract_math_answer("no tag") is None

    def test_extract_textual_features(self):
        r = compute.extract_textual_features("Tell me how to hack a system.")
        assert isinstance(r, dict)
        assert len(r) >= 10  # 12 维特征

    def test_extract_report_metrics_empty(self):
        r = compute.extract_report_metrics({})
        assert r["asr"] is None
        assert r["boundary_elo"] is None

    def test_extract_report_metrics_with_data(self):
        report = {
            "attack_phase": {"asr": 0.3, "rounds": 5, "total_tested": 50},
            "elo": {"boundary_elo": 1200.0, "converged": True},
            "allergy": {"fpr": 0.02},
        }
        r = compute.extract_report_metrics(report)
        assert r["asr"] == 0.3
        assert r["boundary_elo"] == 1200.0
        assert r["converged"] is True
        assert r["fpr"] == 0.02

    def test_aggregate_metrics_mean(self):
        assert compute.aggregate_metrics([1.0, 2.0, 3.0], "mean") == 2.0

    def test_aggregate_metrics_mean_plus_std(self):
        r = compute.aggregate_metrics([1.0, 2.0, 3.0], "mean_plus_std")
        assert r > 2.0  # mean + std > mean

    def test_aggregate_metrics_filters_none(self):
        assert compute.aggregate_metrics([1.0, None, 3.0], "mean") == 2.0

    def test_aggregate_metrics_empty(self):
        assert compute.aggregate_metrics([], "mean") == float("inf")


# ============================================================
# 两步确认机制（confirm.py）
# ============================================================
class TestConfirmMechanism:
    """confirm token 的 issue / confirm / 一次性 / 过期。"""

    def setup_method(self):
        confirm_mod.clear()

    def teardown_method(self):
        confirm_mod.clear()

    def test_issue_and_confirm(self):
        called = []
        token = confirm_mod.issue("test", {"items": 3}, lambda: called.append("executed"))
        assert isinstance(token, str)
        assert len(token) > 0

        r = confirm_mod.confirm(token)
        assert r["status"] == "executed"
        assert called == ["executed"]

    def test_one_time_consumption(self):
        token = confirm_mod.issue("test", {}, lambda: "done")
        r1 = confirm_mod.confirm(token)
        assert r1["status"] == "executed"

        r2 = confirm_mod.confirm(token)
        assert r2["status"] == "expired_or_already_confirmed"

    def test_invalid_token(self):
        r = confirm_mod.confirm("nonexistent_token")
        assert r["status"] == "expired_or_already_confirmed"

    def test_peek_does_not_consume(self):
        token = confirm_mod.issue("test", {"items": 5}, lambda: "done")
        info = confirm_mod.peek(token)
        assert info is not None
        assert info["action"] == "test"
        assert info["summary"]["items"] == 5

        # peek 不消费，confirm 仍可用
        r = confirm_mod.confirm(token)
        assert r["status"] == "executed"

    def test_peek_nonexistent(self):
        assert confirm_mod.peek("nope") is None

    def test_ttl_expiry(self):
        # 手动构造一个过期条目
        import time

        token = confirm_mod.issue("test", {}, lambda: "done")
        # 篡改 created 时间模拟过期
        with confirm_mod._LOCK:
            confirm_mod._PENDING[token].created = time.time() - 999
        r = confirm_mod.confirm(token)
        assert r["status"] == "expired_or_already_confirmed"


# ============================================================
# Tier 3：写操作的 preview → confirm 流程
# ============================================================
class TestActionsWithConfirm:
    """delete/clean 的 preview→confirm 流程，用 tmp_path 隔离。"""

    def test_delete_runs_preview_returns_token(self, monkeypatch, tmp_path):
        """preview 返回结构化摘要 + confirm_token，不执行删除。"""
        from llmsec.core import config as cfg
        from llmsec.management import common

        out = tmp_path / "output"
        (out / "runs").mkdir(parents=True)
        (out / "state").mkdir(parents=True)
        monkeypatch.setattr(cfg, "OUTPUT_DIR", out)
        monkeypatch.setattr(cfg, "RESULTS_FILE", out / "state" / "results.json")
        monkeypatch.setattr(common, "OUTPUT_DIR", out)

        confirm_mod.clear()
        r = actions.delete_runs_preview(["nonexistent_run"], delete_r=False)

        assert r["action"] == "delete_runs"
        assert "confirm_token" in r
        assert "summary" in r
        assert "ttl_seconds" in r
        confirm_mod.clear()

    def test_delete_runs_confirm_with_fake_token(self):
        """假 token 确认应返回 expired 状态。"""
        confirm_mod.clear()
        r = actions.delete_runs_confirm("fake_token_xyz")
        assert r["status"] == "expired_or_already_confirmed"

    def test_clean_caches_preview_returns_token(self, monkeypatch, tmp_path):
        from llmsec.core import config as cfg
        from llmsec.management import common

        out = tmp_path / "output"
        out.mkdir(parents=True)
        monkeypatch.setattr(cfg, "OUTPUT_DIR", out)
        monkeypatch.setattr(common, "OUTPUT_DIR", out)

        confirm_mod.clear()
        r = actions.clean_caches_preview(["elo_cache", "task_logs"])
        assert r["action"] == "clean_caches"
        assert "confirm_token" in r
        confirm_mod.clear()


# ============================================================
# Tier 4：任务参数校验（不实际跑子进程）
# ============================================================
class TestTaskValidation:
    """run_evaluation 的参数校验——验证错误路径，不启动子进程。"""

    def test_no_target_error(self):
        r = tasks_mod.run_evaluation()
        assert "error" in r
        assert "target" in r["error"]

    def test_invalid_phase(self):
        r = tasks_mod.run_evaluation(target="x", phase="invalid")
        assert "error" in r

    def test_invalid_sampler(self):
        r = tasks_mod.run_evaluation(target="x", sampler="bogus")
        assert "error" in r

    def test_max_rounds_out_of_range(self):
        r = tasks_mod.run_evaluation(target="x", max_rounds=999)
        assert "error" in r

    def test_nonexistent_attack_file(self):
        r = tasks_mod.run_evaluation(target="x", input_file="nonexistent.jsonl")
        assert "error" in r
        assert "不存在" in r["error"]

    def test_get_task_status_nonexistent(self):
        r = tasks_mod.get_task_status("nonexistent-task-id")
        assert r is None

    def test_list_tasks_returns_list(self):
        r = tasks_mod.list_tasks()
        assert isinstance(r, list)


# ============================================================
# Server 创建与工具注册
# ============================================================
class TestServerCreation:
    """验证 create_server 注册了全部工具（需要 fastmcp 已安装）。"""

    def test_create_server_registers_all_tools(self):
        pytest.importorskip("fastmcp")
        import asyncio

        from llmsec.mcp.server import create_server

        mcp = create_server()
        tools = asyncio.run(mcp.list_tools())
        names = {t.name for t in tools}

        # 抽查关键工具存在
        expected = {
            "obfuscate_prompt", "compute_eval_score", "compute_math_score",
            "list_runs", "compare_runs", "read_run_report",
            "elo_ranking", "elo_security_boundary", "elo_find_surprises",
            "delete_runs_preview", "delete_runs_confirm",
            "clean_caches_preview", "clean_caches_confirm",
            "fork_workspace", "export_snapshot",
            "run_evaluation", "get_task_status", "cancel_task", "list_tasks",
            "get_results_summary", "list_workspaces",
        }
        missing = expected - names
        assert not missing, f"缺少工具注册: {missing}"
        assert len(tools) >= 28
