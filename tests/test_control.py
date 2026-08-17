"""control 包测试。

策略：
  - invoker：monkeypatch subprocess，验证 argv 构造正确（不真起 llmsec）
  - workspace/compare：在 tmp_path 下造产物文件，测纯逻辑（不依赖真实 output/）
  - tools：验证 schema 导出 + call 分发
  - loop：验证意图解析 + JSON 直调

控制层的核心契约：**绝不 import llmsec 内部**，只经文件 + subprocess。
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

# （路径注入由 tests/conftest.py 统一完成——control 与 llmsec 并列于仓库根）


# ============================================================
# invoker：argv 构造（mock subprocess，不真起 llmsec）
# ============================================================
class TestInvoker:
    def test_list_runs_argv(self):
        from control.core import invoker
        captured = {}

        def fake_run(argv, **kw):
            captured["argv"] = argv
            captured["cwd"] = kw.get("cwd")

            class P:
                returncode = 0
                stdout = json.dumps({"runs": [{"name": "ts/t", "asr": 0.1}], "count": 1})
                stderr = ""
            return P()

        with patch.object(invoker.subprocess, "run", fake_run):
            runs = invoker.list_runs(target="modelA")
        assert runs == [{"name": "ts/t", "asr": 0.1}]
        argv = captured["argv"]
        assert "-m" in argv and "llmsec.management" in argv
        assert "runs" in argv and "list" in argv and "--json" in argv
        assert "--target" in argv and "modelA" in argv

    def test_run_runner_uses_work_dir(self):
        """runner 调用必须带 --work-dir（隔离的核心）。"""
        from control.core import invoker
        captured = {}

        def fake_run(argv, **kw):
            captured["argv"] = argv
            captured["env"] = kw.get("env", {})

            class P:
                returncode = 0
                stdout = '{"ok":true}'
                stderr = ""
            return P()

        with patch.object(invoker.subprocess, "run", fake_run):
            invoker.run_runner(Path("/tmp/wd"), target="modelA", max_rounds=3, seed=42)
        argv = captured["argv"]
        assert "--work-dir" in argv
        wd_idx = argv.index("--work-dir")
        assert argv[wd_idx + 1] == str(Path("/tmp/wd"))
        assert "--target" in argv and "modelA" in argv
        assert "--max-rounds" in argv and "3" in argv
        assert "--seed" in argv and "42" in argv
        assert "--no-early-stop" in argv
        # PYTHONUNBUFFERED 注入
        assert captured["env"].get("PYTHONUNBUFFERED") == "1"

    def test_require_ok_raises_on_failure(self):
        from control.core.invoker import InvokeResult
        res = InvokeResult(argv=["x"], returncode=1, stderr="boom")
        with pytest.raises(RuntimeError, match="boom"):
            res.require_ok()


# ============================================================
# compare：读 runner_report.json 聚合
# ============================================================
class TestCompare:
    def _make_run(self, runs_dir: Path, ts: str, target: str, *, asr, elo=1700, fpr=0.1):
        d = runs_dir / ts / target
        d.mkdir(parents=True)
        report = {
            "target_model": target,
            "security_level": "secure",
            "attack_phase": {"asr": asr, "total_tested": 100, "rounds": 5, "jailbreak_tax": {"probed": 10}},
            "elo": {"boundary_elo": elo, "coverage": 0.8, "conv_rounds": 4, "ci_half": 50,
                    "converged": True, "total_methods": 20, "methods_above_boundary": 3},
            "allergy": {"fpr": fpr},
        }
        (d / "runner_report.json").write_text(json.dumps(report), encoding="utf-8")
        # security_tree
        tree = {
            "dimensions": {
                "by_harm_type": {"harm_a": {"asr": 0.5}, "harm_b": {"asr": 0.1}},
            },
        }
        (d / "security_tree.json").write_text(json.dumps(tree), encoding="utf-8")
        return d

    def test_run_metrics(self, tmp_path, monkeypatch):
        from control import config
        from control.core import compare as cmp
        monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
        monkeypatch.setattr(cmp, "RUNS_DIR", tmp_path)
        self._make_run(tmp_path, "2026-08-11_120000", "modelA", asr=0.3)
        m = cmp.run_metrics("2026-08-11_120000/modelA")
        assert m is not None
        assert m["asr"] == 0.3
        assert m["boundary_elo"] == 1700
        assert m["target_model"] == "modelA"

    def test_compare_multiple_runs(self, tmp_path, monkeypatch):
        from control import config
        from control.core import compare as cmp
        monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
        monkeypatch.setattr(cmp, "RUNS_DIR", tmp_path)
        self._make_run(tmp_path, "2026-08-11_120000", "modelA", asr=0.3, elo=1800)
        self._make_run(tmp_path, "2026-08-11_130000", "modelB", asr=0.7, elo=1500)
        report = cmp.compare(["2026-08-11_120000/modelA", "2026-08-11_130000/modelB"])
        assert len(report["runs"]) == 2
        # 指标透视
        assert report["metrics"]["asr"]["2026-08-11_120000/modelA"] == 0.3
        assert report["metrics"]["asr"]["2026-08-11_130000/modelB"] == 0.7
        assert report["metrics"]["boundary_elo"]["2026-08-11_120000/modelA"] == 1800
        # 威胁树 diff
        assert "by_harm_type" in report["threat_diff"]
        assert report["threat_diff"]["by_harm_type"]["harm_a"]["2026-08-11_120000/modelA"] == 0.5

    def test_compare_handles_missing_run(self, tmp_path, monkeypatch):
        from control import config
        from control.core import compare as cmp
        monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
        monkeypatch.setattr(cmp, "RUNS_DIR", tmp_path)
        self._make_run(tmp_path, "2026-08-11_120000", "modelA", asr=0.3)
        report = cmp.compare(["2026-08-11_120000/modelA", "nonexistent/run"])
        assert report["missing"] == ["nonexistent/run"]
        assert len(report["runs"]) == 1


# ============================================================
# 观测闭合：workspace run 可被 compare / list 看到
# ============================================================
class TestWorkspaceObservability:
    """验证 fork 分支跑完后结果可被 compare / discover_workspace_runs 观测。"""

    def _make_ws_run(self, ws_dir: Path, target: str, *, asr, elo=1700):
        """在 workspace 目录下造 <target>/runner_report.json（模拟 fork_and_run 产物）。"""
        d = ws_dir / target
        d.mkdir(parents=True)
        report = {
            "target_model": target,
            "security_level": "secure",
            "attack_phase": {"asr": asr, "total_tested": 100, "rounds": 5, "jailbreak_tax": {"probed": 10}},
            "elo": {"boundary_elo": elo, "coverage": 0.8, "conv_rounds": 4, "ci_half": 50,
                    "converged": True, "total_methods": 20, "methods_above_boundary": 3},
            "allergy": {"fpr": 0.1},
        }
        (d / "runner_report.json").write_text(json.dumps(report), encoding="utf-8")

    def test_resolve_run_dir_ws_prefix(self, tmp_path, monkeypatch):
        """_resolve_run_dir 识别 ws:<name> 前缀，定位 <ws>/<target>/。"""
        from control import config
        from control.core import compare as cmp
        runs = tmp_path / "runs"
        runs.mkdir()
        ws = tmp_path / "workspaces" / "ab1"
        monkeypatch.setattr(config, "RUNS_DIR", runs)
        monkeypatch.setattr(config, "WORKSPACES_DIR", tmp_path / "workspaces")
        monkeypatch.setattr(cmp, "RUNS_DIR", runs)
        monkeypatch.setattr(cmp, "WORKSPACES_DIR", tmp_path / "workspaces")
        self._make_ws_run(ws, "minimax", asr=0.3)

        # ws:<name> 未指定 target → 自动找第一个含报告的子目录
        d = cmp._resolve_run_dir("ws:ab1")
        assert d is not None and d.name == "minimax"
        assert (d / "runner_report.json").exists()
        # ws:<name>/<target> 显式指定
        d2 = cmp._resolve_run_dir("ws:ab1/minimax")
        assert d2 is not None and d2.name == "minimax"
        # 不存在的 workspace
        assert cmp._resolve_run_dir("ws:nope") is None

    def test_run_metrics_reads_workspace(self, tmp_path, monkeypatch):
        """run_metrics 经 ws: 前缀读 workspace 报告。"""
        from control import config
        from control.core import compare as cmp
        runs = tmp_path / "runs"
        runs.mkdir()
        ws = tmp_path / "workspaces" / "ab1"
        monkeypatch.setattr(config, "RUNS_DIR", runs)
        monkeypatch.setattr(config, "WORKSPACES_DIR", tmp_path / "workspaces")
        monkeypatch.setattr(cmp, "RUNS_DIR", runs)
        monkeypatch.setattr(cmp, "WORKSPACES_DIR", tmp_path / "workspaces")
        self._make_ws_run(ws, "minimax", asr=0.42, elo=1850)

        m = cmp.run_metrics("ws:ab1/minimax")
        assert m is not None
        assert m["asr"] == 0.42
        assert m["boundary_elo"] == 1850
        assert m["run"] == "ws:ab1/minimax"

    def test_compare_mixes_history_and_workspace(self, tmp_path, monkeypatch):
        """compare 可混合对比历史 run 与 workspace run。"""
        from control import config
        from control.core import compare as cmp
        runs = tmp_path / "runs"
        runs.mkdir()
        ws = tmp_path / "workspaces" / "ab1"
        monkeypatch.setattr(config, "RUNS_DIR", runs)
        monkeypatch.setattr(config, "WORKSPACES_DIR", tmp_path / "workspaces")
        monkeypatch.setattr(cmp, "RUNS_DIR", runs)
        monkeypatch.setattr(cmp, "WORKSPACES_DIR", tmp_path / "workspaces")
        # 历史 run（复用 TestCompare._make_run 的写法）
        hist = runs / "2026-08-11_120000" / "modelA"
        hist.mkdir(parents=True)
        (hist / "runner_report.json").write_text(json.dumps({
            "target_model": "modelA", "security_level": "secure",
            "attack_phase": {"asr": 0.1}, "elo": {"boundary_elo": 1800},
            "allergy": {"fpr": 0.05},
        }), encoding="utf-8")
        # workspace run
        self._make_ws_run(ws, "minimax", asr=0.5, elo=1600)

        report = cmp.compare(["2026-08-11_120000/modelA", "ws:ab1/minimax"])
        assert len(report["runs"]) == 2
        assert report["missing"] == []
        assert report["metrics"]["asr"]["ws:ab1/minimax"] == 0.5

    def test_discover_workspace_runs(self, tmp_path, monkeypatch):
        """discover_workspace_runs 列出所有 workspace 内的 run。"""
        from control import config
        from control.core import compare as cmp
        ws_root = tmp_path / "workspaces"
        monkeypatch.setattr(config, "WORKSPACES_DIR", ws_root)
        monkeypatch.setattr(cmp, "WORKSPACES_DIR", ws_root)
        self._make_ws_run(ws_root / "ab1", "minimax", asr=0.3)
        self._make_ws_run(ws_root / "ab1", "gemma", asr=0.7)
        self._make_ws_run(ws_root / "ab2", "minimax", asr=0.1)

        runs = cmp.discover_workspace_runs()
        assert len(runs) == 3
        names = [r["name"] for r in runs]
        assert "ws:ab1/minimax" in names
        assert "ws:ab1/gemma" in names
        assert "ws:ab2/minimax" in names

    def test_list_runs_tool_includes_workspaces(self, tmp_path, monkeypatch):
        """list_runs tool 默认包含 workspace run。"""
        from control import config
        from control.agent.zhongshu import tools
        from control.core import compare as cmp
        ws_root = tmp_path / "workspaces"
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        monkeypatch.setattr(config, "WORKSPACES_DIR", ws_root)
        monkeypatch.setattr(config, "RUNS_DIR", runs_dir)
        monkeypatch.setattr(cmp, "WORKSPACES_DIR", ws_root)
        self._make_ws_run(ws_root / "ab1", "minimax", asr=0.3)

        # mock list_runs（历史 run）返回空，只验证 workspace 部分
        #（实现已下沉到 compare.list_all_runs → invoker.list_runs，注入点随之迁移）
        from control.core import invoker
        monkeypatch.setattr(invoker, "list_runs", lambda **kw: [])
        tools.reset_registry()
        result = tools.call_tool("list_runs", {})
        assert any(r["name"] == "ws:ab1/minimax" for r in result)
        # include_workspaces=False 不含
        result2 = tools.call_tool("list_runs", {"include_workspaces": False})
        assert all(not r["name"].startswith("ws:") for r in result2)


# ============================================================
# 状态闭合：merge 后 mark_merged 回写库行
# ============================================================
class TestMergeStatusClosure:
    def test_mark_merged_updates_row(self, tmp_path, monkeypatch):
        """mark_merged 把 workspace 的 merged/merged_at/merged_to 写进库行。"""
        from control import config
        from control.core import workspace as ws
        ws_root = tmp_path / "workspaces"
        ws_root.mkdir()
        monkeypatch.setattr(config, "WORKSPACES_DIR", ws_root)
        monkeypatch.setattr(ws, "WORKSPACES_DIR", ws_root)
        from control.core.storage import get_workspace, save_workspace
        save_workspace({"name": "ab1", "merged": False, "merged_at": None, "merged_to": None})

        assert ws.mark_merged("ab1", "global") is True
        row = get_workspace("ab1")
        assert row["merged"] is True
        assert row["merged_to"] == "global"
        assert row["merged_at"] is not None


# ============================================================
# workspace：fork 编排（mock invoker，验证流程 + 隔离契约）
# ============================================================
class TestWorkspace:
    def test_fork_clones_results_db(self, tmp_path, monkeypatch):
        """fork（P3 库级 clone）：backup → <ws>/catalog.db + 记索引。"""
        from control import config
        from control.core import storage as cstorage
        from control.core import workspace as ws

        workspaces = tmp_path / "workspaces"
        monkeypatch.setattr(config, "WORKSPACES_DIR", workspaces)
        monkeypatch.setattr(ws, "WORKSPACES_DIR", workspaces)
        monkeypatch.setattr(config, "LLMSEC_REPO", tmp_path)
        monkeypatch.setattr(ws, "LLMSEC_REPO", tmp_path)

        # mock 库级 clone：落一个真实可读的迷你 R 库 + 统计
        def fake_backup(dest):
            from llmsec.core.results import ResultsMatrix
            from llmsec.storage import rstore
            mat = ResultsMatrix()
            mat.upsert("r1", "mA", 1.0, ts=1)
            rstore.save_matrix(mat, path=dest)

        def fake_stats(path):
            return {"models": ["mA"], "records": 1, "observations": 1, "units": 0}

        monkeypatch.setattr(cstorage, "backup", fake_backup)
        monkeypatch.setattr(cstorage, "results_stats", fake_stats)

        info = ws.fork("ws1", source="global", note="test")
        assert info["name"] == "ws1"
        assert info["models"] == ["mA"]
        # results.db 落到 workspace（库级 clone 产物）
        assert (workspaces / "ws1" / "catalog.db").exists()
        # 索引行（P5：ctl_workspaces 表）
        from control.core.storage import get_workspace
        assert get_workspace("ws1") is not None

    def test_fork_duplicate_name_raises(self, tmp_path, monkeypatch):
        from control import config
        from control.core import workspace as ws
        workspaces = tmp_path / "workspaces"
        monkeypatch.setattr(config, "WORKSPACES_DIR", workspaces)
        monkeypatch.setattr(ws, "WORKSPACES_DIR", workspaces)
        monkeypatch.setattr(config, "LLMSEC_REPO", tmp_path)
        monkeypatch.setattr(ws, "LLMSEC_REPO", tmp_path)
        (workspaces / "ws1").mkdir(parents=True)
        with pytest.raises(FileExistsError):
            ws.fork("ws1")

    def test_list_and_delete(self, tmp_path, monkeypatch):
        from control import config
        from control.core import workspace as ws
        workspaces = tmp_path / "workspaces"
        workspaces.mkdir()
        monkeypatch.setattr(config, "WORKSPACES_DIR", workspaces)
        monkeypatch.setattr(ws, "WORKSPACES_DIR", workspaces)
        monkeypatch.setattr(config, "LLMSEC_REPO", tmp_path)
        monkeypatch.setattr(ws, "LLMSEC_REPO", tmp_path)
        # 造索引行 + 目录
        (workspaces / "ws1").mkdir()
        from control.core.storage import save_workspace
        save_workspace({"name": "ws1", "path": "output/workspaces/ws1",
                        "source": "global", "created": "2026-01-01", "records": 5})

        ws_list = ws.list_workspaces()
        assert len(ws_list) == 1 and ws_list[0]["name"] == "ws1"

        res = ws.delete_workspace("ws1")
        assert res["deleted"] == "ws1"
        assert not (workspaces / "ws1").exists()


# ============================================================
# tools：schema + 分发
# ============================================================
class TestTools:
    def test_all_tools_have_schema(self):
        from control.agent.zhongshu.tools import all_tools
        for t in all_tools():
            schema = t.to_schema()
            assert schema["type"] == "function"
            assert "name" in schema["function"]
            assert "parameters" in schema["function"]
            assert schema["function"]["name"] == t.name

    def test_call_tool_dispatch(self, monkeypatch):
        """call_tool 按名分发到对应 tool.call，透传 args。"""
        from control.agent.zhongshu import tools
        t = tools.tool_by_name("list_runs")
        monkeypatch.setattr(t, "call", lambda args: {"echoed": args})
        result = tools.call_tool("list_runs", {"target": "mA"})
        assert result == {"echoed": {"target": "mA"}}

    def test_call_unknown_tool_raises(self):
        from control.agent.zhongshu.tools import call_tool
        with pytest.raises(KeyError):
            call_tool("bogus", {})


# ============================================================
# loop：意图解析
# ============================================================
class TestLoop:
    def test_parse_list_runs(self):
        from control.agent.zhongshu.fallback import _parse_intent
        assert _parse_intent("列一下 run") == ("list_runs", {})
        assert _parse_intent("list runs") == ("list_runs", {})
        r = _parse_intent("列 run target=modelA")
        assert r == ("list_runs", {"target": "modelA"})

    def test_parse_list_junk(self):
        from control.agent.zhongshu.fallback import _parse_intent
        r = _parse_intent("列一下垃圾 run")
        assert r is not None and r[0] == "list_runs" and r[1].get("junk_only") is True

    def test_parse_compare(self):
        from control.agent.zhongshu.fallback import _parse_intent
        r = _parse_intent("对比 2026-08-11_120000/A 和 2026-08-11_130000/B")
        assert r is not None and r[0] == "compare_runs"
        assert len(r[1]["runs"]) == 2

    def test_parse_fork(self):
        from control.agent.zhongshu.fallback import _parse_intent
        r = _parse_intent("fork ws1")
        assert r == ("fork_workspace", {"name": "ws1"})
        r = _parse_intent("fork ws1 from run:2026-08-11_120000/A")
        assert r == ("fork_workspace", {"name": "ws1", "source": "run:2026-08-11_120000/A"})

    def test_parse_workspace_list_and_delete(self):
        from control.agent.zhongshu.fallback import _parse_intent
        assert _parse_intent("列工作区") == ("list_workspaces", {})
        assert _parse_intent("list workspaces") == ("list_workspaces", {})
        r = _parse_intent("删工作区 demo-ws")
        assert r == ("delete_workspace", {"name": "demo-ws"})
        r = _parse_intent("delete workspace demo-ws")
        assert r == ("delete_workspace", {"name": "demo-ws"})

    def test_chat_one_json_direct_call(self, monkeypatch):
        from control.agent.zhongshu import fallback as loop
        monkeypatch.setattr(loop, "call_tool", lambda name, args: [{"name": "t"}])
        out = loop.chat_one('{"tool": "list_runs", "args": {}}')
        assert "共 1 项" in out

    def test_chat_one_unknown(self):
        from control.agent.zhongshu.fallback import chat_one
        out = chat_one("xyzzy nonsense")
        assert "未识别" in out


# ============================================================
# session：上下文记忆
# ============================================================
class TestSession:
    def test_get_or_create_assigns_id(self):
        from control.agent.zhongshu import session as sess
        sess._SESSIONS.clear()
        sid, msgs = sess.get_or_create(None)
        assert sid is not None and len(sid) > 0
        assert len(msgs) >= 1 and msgs[0]["role"] == "system"

    def test_reuse_existing_session(self):
        from control.agent.zhongshu import session as sess
        sess._SESSIONS.clear()
        sid1, _ = sess.get_or_create(None)
        sid2, msgs2 = sess.get_or_create(sid1)
        assert sid1 == sid2  # 同一 session

    def test_messages_accumulate(self):
        """跨调用累积消息（上下文记忆的核心）。"""
        from control.agent.zhongshu import session as sess
        sess._SESSIONS.clear()
        sid, msgs = sess.get_or_create(None)
        sess.append(sid, "user", "帮我 fork")
        sess.append(sid, "assistant", "已 fork")
        _, msgs2 = sess.get_or_create(sid)
        assert len(msgs2) == 3  # system + user + assistant
        assert msgs2[1]["content"] == "帮我 fork"
        assert msgs2[2]["content"] == "已 fork"

    def test_reset_clears_history(self):
        from control.agent.zhongshu import session as sess
        sess._SESSIONS.clear()
        sid, _ = sess.get_or_create(None)
        sess.append(sid, "user", "hello")
        sess.reset(sid)
        _, msgs = sess.get_or_create(sid)
        assert len(msgs) == 1 and msgs[0]["role"] == "system"

    def test_history_window_trims_to_max(self):
        """超过 _HISTORY_MAX 时滑窗裁剪，system 永远保留在 index 0。"""
        from control.agent.zhongshu import session as sess
        sess._SESSIONS.clear()
        sid, _ = sess.get_or_create(None)
        for i in range(sess._HISTORY_MAX + 10):
            sess.append(sid, "user", f"msg-{i}")
        _, msgs = sess.get_or_create(sid)
        assert len(msgs) <= sess._HISTORY_MAX
        assert msgs[0]["role"] == "system"


# ============================================================
# review：门下省事后审查（读报告 → 规则判定 findings）
# ============================================================
class TestReview:
    def _make_report(self, *, asr=0.1, fpr=0.01, level="safe", tested=50,
                     converged=True, coverage=0.5, ci_half=15, drift=2,
                     confidence=0.8):
        return {
            "target_model": "testModel", "security_level": level,
            "overall_verdict": "ok",
            "attack_phase": {"asr": asr, "total_tested": tested},
            "elo": {"boundary_elo": 1700, "boundary_confidence": confidence,
                    "converged": converged, "ci_half": ci_half, "drift": drift,
                    "coverage": coverage, "conv_rounds": 5},
            "allergy": {"fpr": fpr},
            "recommendation": "ok",
        }

    def test_high_asr_is_critical(self):
        from control.agent.menxia.review import assess_findings
        report = self._make_report(asr=0.8, level="broken")
        findings = assess_findings(report, None)
        asr_f = [f for f in findings if f["metric"] == "ASR"]
        assert len(asr_f) == 1 and asr_f[0]["severity"] == "critical"

    def test_safe_asr_is_good(self):
        from control.agent.menxia.review import assess_findings
        findings = assess_findings(self._make_report(asr=0.05), None)
        asr_f = [f for f in findings if f["metric"] == "ASR"]
        assert asr_f[0]["severity"] == "good"

    def test_high_fpr_flags(self):
        from control.agent.menxia.review import assess_findings
        findings = assess_findings(self._make_report(fpr=0.2), None)
        fpr_f = [f for f in findings if f["metric"] == "FPR"]
        assert fpr_f[0]["severity"] == "critical"

    def test_inconclusive_flagged(self):
        from control.agent.menxia.review import assess_findings
        findings = assess_findings(self._make_report(level="inconclusive"), None)
        assert any(f["metric"] == "security_level" for f in findings)

    def test_low_coverage_flagged(self):
        from control.agent.menxia.review import assess_findings
        findings = assess_findings(self._make_report(coverage=0.1), None)
        assert any(f["metric"] == "coverage" for f in findings)

    def test_not_converged_flagged(self):
        from control.agent.menxia.review import assess_findings
        findings = assess_findings(self._make_report(converged=False), None)
        assert any(f["metric"] == "converged" for f in findings)

    def test_zero_tested_flagged(self):
        """tested==0（最不充分）必须触发样本不足告警，不能被真值门短路跳过。"""
        from control.agent.menxia.review import assess_findings
        findings = assess_findings(self._make_report(tested=0), None)
        assert any(f["metric"] == "total_tested" for f in findings)

    def test_findings_sorted_by_severity(self):
        from control.agent.menxia.review import assess_findings
        findings = assess_findings(
            self._make_report(asr=0.8, fpr=0.2, coverage=0.1, level="broken"), None)
        severities = [f["severity"] for f in findings]
        # critical 应排在 warning/info 前面
        if "critical" in severities and "warning" in severities:
            assert severities.index("critical") < severities.index("warning")

    def test_get_thresholds_direct(self):
        """P5：阈值直读 llmsec.params 单一来源（CLI/缓存/fallback 已删）。"""
        from control.agent.menxia import review
        from llmsec.params import PORTRAIT_ASR_SAFE
        th = review.get_thresholds()
        assert th["PORTRAIT_ASR_SAFE"] == PORTRAIT_ASR_SAFE
        assert set(th) == {
            "PORTRAIT_MIN_TESTED", "PORTRAIT_MIN_CONFIDENCE", "PORTRAIT_ASR_SAFE",
            "ALLERGY_FPR_SAFE", "TWIN_SEVERITY_FPR_MED", "CONV_CI_TARGET",
            "CONV_DRIFT_TARGET", "MIN_COVERAGE_RATIO", "MIN_COVERAGE_ABSOLUTE",
        }

    def test_review_run_returns_structure(self, tmp_path, monkeypatch):
        """review_run 完整流程（mock 读报告）。"""
        from control import config
        from control.agent.menxia import review
        from control.core import compare as cmp
        runs = tmp_path / "runs"
        runs.mkdir()
        d = runs / "2026-08-11_120000" / "modelA"
        d.mkdir(parents=True)
        report = self._make_report(asr=0.6, level="vulnerable")
        import json
        (d / "runner_report.json").write_text(json.dumps(report), encoding="utf-8")
        monkeypatch.setattr(config, "RUNS_DIR", runs)
        monkeypatch.setattr(cmp, "RUNS_DIR", runs)
        result = review.review_run("2026-08-11_120000/modelA", use_llm=False)
        assert "error" not in result
        assert result["summary"]
        assert len(result["findings"]) > 0
        assert "digest" in result
