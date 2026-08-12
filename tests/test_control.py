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
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# 确保 control 包可 import（它在仓库根，与 llmsec 并列）
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


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
        runs = tmp_path / "runs"; runs.mkdir()
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
        runs = tmp_path / "runs"; runs.mkdir()
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
        runs = tmp_path / "runs"; runs.mkdir()
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
        from control.agent import tools
        from control.core import compare as cmp
        ws_root = tmp_path / "workspaces"
        runs_dir = tmp_path / "runs"; runs_dir.mkdir()
        monkeypatch.setattr(config, "WORKSPACES_DIR", ws_root)
        monkeypatch.setattr(config, "RUNS_DIR", runs_dir)
        monkeypatch.setattr(cmp, "WORKSPACES_DIR", ws_root)
        self._make_ws_run(ws_root / "ab1", "minimax", asr=0.3)

        # mock list_runs（历史 run）返回空，只验证 workspace 部分
        monkeypatch.setattr(tools, "list_runs", lambda **kw: [])
        tools.reset_registry()
        result = tools.call_tool("list_runs", {})
        assert any(r["name"] == "ws:ab1/minimax" for r in result)
        # include_workspaces=False 不含
        result2 = tools.call_tool("list_runs", {"include_workspaces": False})
        assert all(not r["name"].startswith("ws:") for r in result2)


# ============================================================
# 状态闭合：merge 后 mark_merged 回写 _index.json
# ============================================================
class TestMergeStatusClosure:
    def test_mark_merged_updates_index(self, tmp_path, monkeypatch):
        """mark_merged 把 workspace 的 merged/merged_at/merged_to 写进 _index.json。"""
        from control import config
        from control.core import workspace as ws
        ws_root = tmp_path / "workspaces"
        ws_root.mkdir()
        monkeypatch.setattr(config, "WORKSPACES_DIR", ws_root)
        monkeypatch.setattr(ws, "WORKSPACES_DIR", ws_root)
        # 造一个已登记的 workspace
        idx = {"workspaces": {"ab1": {"name": "ab1", "merged": False, "merged_at": None, "merged_to": None}}}
        (ws_root / "_index.json").write_text(json.dumps(idx), encoding="utf-8")

        ws.mark_merged("ab1", "global")
        idx2 = json.loads((ws_root / "_index.json").read_text(encoding="utf-8"))
        assert idx2["workspaces"]["ab1"]["merged"] is True
        assert idx2["workspaces"]["ab1"]["merged_to"] == "global"
        assert idx2["workspaces"]["ab1"]["merged_at"] is not None

    def test_merge_tool_confirm_marks_workspace_merged(self, tmp_path, monkeypatch):
        """merge tool confirm=True 执行后回写 ws 源的 merged 状态。"""
        from control import config
        from control.agent import tools
        from control.core import invoker
        from control.core import workspace as ws
        ws_root = tmp_path / "workspaces"
        ws_root.mkdir()
        monkeypatch.setattr(config, "WORKSPACES_DIR", ws_root)
        monkeypatch.setattr(ws, "WORKSPACES_DIR", ws_root)
        idx = {"workspaces": {"ab1": {"name": "ab1", "merged": False}}}
        (ws_root / "_index.json").write_text(json.dumps(idx), encoding="utf-8")

        # mock _do_merge 经 invoker._run 的 subprocess 调用，返回 executed 结果
        _FakeRes = type("R", (), {
            "require_ok": lambda self: self, "json": {"action": "merge", "dry_run": False},
            "returncode": 0, "stdout": "", "stderr": "", "elapsed_s": 0,
        })
        monkeypatch.setattr(invoker, "_run", lambda argv: _FakeRes())
        tools.reset_registry()
        result = tools.call_tool("merge", {
            "sources": ["ws:ab1"], "target": "global", "confirm": True,
        })
        assert result["dry_run"] is False
        # _index.json 已回写
        idx2 = json.loads((ws_root / "_index.json").read_text(encoding="utf-8"))
        assert idx2["workspaces"]["ab1"]["merged"] is True
        assert idx2["workspaces"]["ab1"]["merged_to"] == "global"

    def test_merge_tool_dry_run_does_not_mark(self, tmp_path, monkeypatch):
        """merge tool confirm=False（dry-run）不回写 merged 状态。"""
        from control import config
        from control.agent import tools
        from control.core import invoker
        from control.core import workspace as ws
        ws_root = tmp_path / "workspaces"
        ws_root.mkdir()
        monkeypatch.setattr(config, "WORKSPACES_DIR", ws_root)
        monkeypatch.setattr(ws, "WORKSPACES_DIR", ws_root)
        idx = {"workspaces": {"ab1": {"name": "ab1", "merged": False}}}
        (ws_root / "_index.json").write_text(json.dumps(idx), encoding="utf-8")

        _FakeRes = type("R", (), {
            "require_ok": lambda self: self, "json": {"action": "merge", "dry_run": True},
            "returncode": 0, "stdout": "", "stderr": "", "elapsed_s": 0,
        })
        monkeypatch.setattr(invoker, "_run", lambda argv: _FakeRes())
        tools.reset_registry()
        tools.call_tool("merge", {"sources": ["ws:ab1"], "target": "global", "confirm": False})
        idx2 = json.loads((ws_root / "_index.json").read_text(encoding="utf-8"))
        assert idx2["workspaces"]["ab1"]["merged"] is False


# ============================================================
# workspace：fork 编排（mock invoker，验证流程 + 隔离契约）
# ============================================================
class TestWorkspace:
    def test_fork_copies_snapshot_to_workspace(self, tmp_path, monkeypatch):
        """fork：调 export_snapshot → 复制 results.json → 记索引。"""
        from control import config
        from control.core import workspace as ws

        workspaces = tmp_path / "workspaces"
        monkeypatch.setattr(config, "WORKSPACES_DIR", workspaces)
        monkeypatch.setattr(ws, "WORKSPACES_DIR", workspaces)
        monkeypatch.setattr(config, "LLMSEC_REPO", tmp_path)
        monkeypatch.setattr(ws, "LLMSEC_REPO", tmp_path)

        # mock export_snapshot：造一个临时快照目录
        def fake_export(source="global", out=None):
            snap_dir = tmp_path / "output" / "snapshots" / "fake"
            snap_dir.mkdir(parents=True)
            (snap_dir / "results.json").write_text(
                json.dumps({"version": 2, "models": ["mA"], "results": {"r1": {"mA": {"eval_score": 1.0}}}}),
                encoding="utf-8",
            )
            return {"snapshot": "snapshots/fake", "models": ["mA"], "records": 1}

        monkeypatch.setattr(ws, "export_snapshot", fake_export)
        # OUTPUT_DIR for snap_path 解析
        monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "output")

        info = ws.fork("ws1", source="global", note="test")
        assert info["name"] == "ws1"
        assert info["models"] == ["mA"]
        # results.json 复制到 workspace
        assert (workspaces / "ws1" / "results.json").exists()
        # 索引记录
        idx = json.loads((workspaces / "_index.json").read_text(encoding="utf-8"))
        assert "ws1" in idx["workspaces"]

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
        # 造索引 + 目录
        (workspaces / "ws1").mkdir()
        idx = {"workspaces": {"ws1": {"name": "ws1", "path": "output/workspaces/ws1",
                                      "source": "global", "created": "2026-01-01", "records": 5}}}
        (workspaces / "_index.json").write_text(json.dumps(idx), encoding="utf-8")

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
        from control.agent.tools import all_tools
        for t in all_tools():
            schema = t.to_schema()
            assert schema["type"] == "function"
            assert "name" in schema["function"]
            assert "parameters" in schema["function"]
            assert schema["function"]["name"] == t.name

    def test_call_tool_dispatch(self, monkeypatch):
        """call_tool 按名分发到对应 tool.call，透传 args。"""
        from control.agent import tools
        t = tools.tool_by_name("list_runs")
        monkeypatch.setattr(t, "call", lambda args: {"echoed": args})
        result = tools.call_tool("list_runs", {"target": "mA"})
        assert result == {"echoed": {"target": "mA"}}

    def test_call_unknown_tool_raises(self):
        from control.agent.tools import call_tool
        with pytest.raises(KeyError):
            call_tool("bogus", {})


# ============================================================
# loop：意图解析
# ============================================================
class TestLoop:
    def test_parse_list_runs(self):
        from control.agent.loop import _parse_intent
        assert _parse_intent("列一下 run") == ("list_runs", {})
        assert _parse_intent("list runs") == ("list_runs", {})
        r = _parse_intent("列 run target=modelA")
        assert r == ("list_runs", {"target": "modelA"})

    def test_parse_list_junk(self):
        from control.agent.loop import _parse_intent
        r = _parse_intent("列一下垃圾 run")
        assert r is not None and r[0] == "list_runs" and r[1].get("junk_only") is True

    def test_parse_compare(self):
        from control.agent.loop import _parse_intent
        r = _parse_intent("对比 2026-08-11_120000/A 和 2026-08-11_130000/B")
        assert r is not None and r[0] == "compare_runs"
        assert len(r[1]["runs"]) == 2

    def test_parse_fork(self):
        from control.agent.loop import _parse_intent
        r = _parse_intent("fork ws1")
        assert r == ("fork_workspace", {"name": "ws1"})
        r = _parse_intent("fork ws1 from run:2026-08-11_120000/A")
        assert r == ("fork_workspace", {"name": "ws1", "source": "run:2026-08-11_120000/A"})

    def test_parse_workspace_list_and_delete(self):
        from control.agent.loop import _parse_intent
        assert _parse_intent("列工作区") == ("list_workspaces", {})
        assert _parse_intent("list workspaces") == ("list_workspaces", {})
        r = _parse_intent("删工作区 demo-ws")
        assert r == ("delete_workspace", {"name": "demo-ws"})
        r = _parse_intent("delete workspace demo-ws")
        assert r == ("delete_workspace", {"name": "demo-ws"})

    def test_chat_one_json_direct_call(self, monkeypatch):
        from control.agent import loop
        monkeypatch.setattr(loop, "call_tool", lambda name, args: [{"name": "t"}])
        out = loop.chat_one('{"tool": "list_runs", "args": {}}')
        assert "共 1 项" in out

    def test_chat_one_unknown(self):
        from control.agent.loop import chat_one
        out = chat_one("xyzzy nonsense")
        assert "未识别" in out


# ============================================================
# session：上下文记忆
# ============================================================
class TestSession:
    def test_get_or_create_assigns_id(self):
        from control.agent import session as sess
        sess._SESSIONS.clear()
        sid, msgs = sess.get_or_create(None)
        assert sid is not None and len(sid) > 0
        assert len(msgs) >= 1 and msgs[0]["role"] == "system"

    def test_reuse_existing_session(self):
        from control.agent import session as sess
        sess._SESSIONS.clear()
        sid1, _ = sess.get_or_create(None)
        sid2, msgs2 = sess.get_or_create(sid1)
        assert sid1 == sid2  # 同一 session

    def test_messages_accumulate(self):
        """跨调用累积消息（上下文记忆的核心）。"""
        from control.agent import session as sess
        sess._SESSIONS.clear()
        sid, msgs = sess.get_or_create(None)
        sess.append(sid, "user", "帮我 fork")
        sess.append(sid, "assistant", "已 fork")
        _, msgs2 = sess.get_or_create(sid)
        assert len(msgs2) == 3  # system + user + assistant
        assert msgs2[1]["content"] == "帮我 fork"
        assert msgs2[2]["content"] == "已 fork"

    def test_append_raw_with_tool_calls(self):
        from control.agent import session as sess
        sess._SESSIONS.clear()
        sid, _ = sess.get_or_create(None)
        sess.append_raw(sid, {"role": "assistant", "content": None, "tool_calls": [{"id": "x"}]})
        _, msgs = sess.get_or_create(sid)
        assert len(msgs) == 2 and msgs[1].get("tool_calls")

    def test_pending_confirm_set_get_clear(self):
        from control.agent import session as sess
        sess._SESSIONS.clear()
        sid, _ = sess.get_or_create(None)
        assert sess.get_pending_confirm(sid) is None
        sess.set_pending_confirm(sid, {"token": "abc", "action": "merge"})
        assert sess.get_pending_confirm(sid)["token"] == "abc"
        sess.set_pending_confirm(sid, None)
        assert sess.get_pending_confirm(sid) is None

    def test_reset_clears_history(self):
        from control.agent import session as sess
        sess._SESSIONS.clear()
        sid, _ = sess.get_or_create(None)
        sess.append(sid, "user", "hello")
        sess.reset(sid)
        _, msgs = sess.get_or_create(sid)
        assert len(msgs) == 1 and msgs[0]["role"] == "system"


# ============================================================
# gatekeeper：门下省封驳
# ============================================================
class TestGatekeeper:
    def test_merge_to_global_is_blocked(self):
        from control.agent import gatekeeper
        a = gatekeeper.assess("merge", {"sources": ["ws:ab"], "target": "global"})
        assert a is not None
        assert a["action"] == "merge_to_global"
        assert "不可逆" in a["detail"]

    def test_merge_to_workspace_not_blocked(self):
        from control.agent import gatekeeper
        a = gatekeeper.assess("merge", {"sources": ["ws:ab"], "target": "ws:other"})
        assert a is None  # 融合到另一个 ws，不触发封驳

    def test_delete_runs_with_delete_r_blocked(self):
        from control.agent import gatekeeper
        a = gatekeeper.assess("delete_runs", {"names": ["m1"], "delete_r": True})
        assert a is not None and a["action"] == "delete_r_column"

    def test_delete_runs_without_delete_r_not_blocked(self):
        from control.agent import gatekeeper
        a = gatekeeper.assess("delete_runs", {"names": ["m1"], "delete_r": False})
        assert a is None

    def test_safe_operations_not_blocked(self):
        from control.agent import gatekeeper
        for name, args in [("list_runs", {}), ("compare_runs", {"runs": ["a", "b"]}),
                           ("fork_workspace", {"name": "x"}), ("list_workspaces", {})]:
            assert gatekeeper.assess(name, args) is None

    def test_issue_ticket_has_token(self):
        from control.agent import gatekeeper
        a = gatekeeper.assess("merge", {"sources": ["ws:x"], "target": "global"})
        t = gatekeeper.issue_ticket("merge", {"sources": ["ws:x"], "target": "global"}, a)
        assert t.token and len(t.token) > 0
        assert t.action == "merge_to_global"
        assert "merge" in t.tool_name

    def test_is_confirmed_matches_token(self):
        from control.agent import gatekeeper
        a = gatekeeper.assess("merge", {"sources": ["ws:x"], "target": "global"})
        t = gatekeeper.issue_ticket("merge", {"sources": ["ws:x"], "target": "global"}, a)
        assert gatekeeper.is_confirmed(t, t.token) is True
        assert gatekeeper.is_confirmed(t, "wrong") is False
        assert gatekeeper.is_confirmed(None, "x") is False
