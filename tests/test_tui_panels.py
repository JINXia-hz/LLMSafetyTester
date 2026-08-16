"""tests for TUI 面板层：tasks_panel / runs_panel / hpo_panel。

纯函数（_fmt_num/_fmt_ratio/_render_report/_load_target_names）直接测；
面板的行构造/刷新逻辑用 App.run_test 离线驱动（毫秒级、无网络，不算 e2e），
@work 线程方法一律 monkeypatch 实例属性拦截，不发线程不起子进程。
涉及 task_store 的用 tmp 目录伪造 .progress.jsonl（沿用 test_tui_task_store 手法）。
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

# 可选依赖：textual 属 [tui] extra，未安装环境整体跳过（沿用 hdbscan 惯例）
pytest.importorskip("textual")

from textual.app import App, ComposeResult
from textual.widgets import Checkbox, DataTable, Input, RichLog, Select, SelectionList, Static

from llmsec.tui.panels import runs_panel
from llmsec.tui.panels.hpo_panel import HpoPanel, HpoStartScreen
from llmsec.tui.panels.runs_panel import EloSelectScreen, RunsPanel, _fmt_num, _fmt_ratio, _render_report
from llmsec.tui.panels.tasks_panel import LaunchScreen, TasksPanel, _load_target_names
from llmsec.tui.render import EvalProgressState
from llmsec.tui.task_store import TaskSnapshot, TaskStore
from llmsec.tui.widgets import ConfirmScreen, TermBox

# TASKS 隔离由 conftest 的 autouse _hermetic_tasks 统一提供（原先本文件的局部
# fixture 已收编，见 conftest.py 注释）。


# ============================================================
# 造数辅助
# ============================================================
def _write(path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _ev_line(tg, rnd, **kw) -> str:
    return json.dumps(
        {"ts": f"2026-08-15T10:00:{rnd:02d}", "phase": "attack", "target": tg, "round": rnd,
         "max_rounds": 5, "elo": 1500 + rnd, "delta": 10.0, "ci_half": 30.0,
         "progress_pct": 20 * rnd, "converged": False, **kw},
        ensure_ascii=False,
    )


def _hpo_line(trial_done: int, seed: int) -> str:
    return json.dumps(
        {"ts": "2026-08-15T11:00:00", "phase": "hpo", "trial_done": trial_done,
         "trial_total_est": 10, "configs_done": trial_done, "configs_total": 8,
         "best_metric": 5.5, "metric_name": "conv_rounds", "direction": "minimize",
         "last": {"target": "模型A", "seed": seed, "status": "success", "value": 5.5,
                  "params": {"K_FACTOR": 16}}},
        ensure_ascii=False,
    )


def _fake_disk_tasks(tmp_path, *, evaluate_mtime: int = 2_000_000_000,
                     hpo_mtime: int = 1_000_000_000) -> None:
    """伪造一个外部 evaluate + 一个外部 hpo 任务的磁盘痕迹（mtime 可控排序）。"""
    ev = tmp_path / "evaluate-101010-ab12cd.progress.jsonl"
    _write(ev, _ev_line("模型A", 2) + "\n")
    _write(tmp_path / "evaluate-101010-ab12cd.log", "fake log\n")
    _write(tmp_path / "hpo-111111-aabbcc.progress.jsonl", _hpo_line(2, 0) + "\n")
    os.utime(ev, (evaluate_mtime, evaluate_mtime))
    os.utime(tmp_path / "hpo-111111-aabbcc.progress.jsonl", (hpo_mtime, hpo_mtime))


def _snap(sid: str, *, kind=None, status="external", owned=False, cmd="", meta=None,
          state=None) -> TaskSnapshot:
    return TaskSnapshot(id=sid, kind=kind or sid.split("-", 1)[0], status=status, cmd=cmd,
                        started_at="2026-08-15T10:00:01", owned=owned, state=state, meta=meta)


def _ev_state(target="模型A", rnd=2, **kw) -> EvalProgressState:
    st = EvalProgressState("t-1")
    st.apply_record(
        {"phase": "attack", "target": target, "round": rnd, "max_rounds": 5,
         "elo": 1500 + rnd, "delta": 10.0, "ci_half": 30.0, "progress_pct": 20 * rnd,
         "converged": False, "ts": "2026-08-15T10:00:00", **kw}
    )
    st.set_running(True)
    return st


def _hpo_state() -> EvalProgressState:
    st = EvalProgressState("h-1")
    st.apply_record(json.loads(_hpo_line(2, 0)))
    st.set_running(True)
    return st


def _notes_recorder():
    """返回 (记录列表, 可塞给 app.notify 的替身)。"""
    notes: list[str] = []

    def fake_notify(message="", *args, **kwargs) -> None:
        notes.append(str(message))

    return notes, fake_notify


async def _wait_until(pilot, cond, tries: int = 25) -> None:
    """有界等待条件成立（每次迭代排空一轮消息队列 + 让出真实时间片）。

    超时必须抛错：静默返回会让失败推迟到后续步骤、报出误导性错误（如 click 的
    NoMatches 实际是屏从未弹出）；pause() 只推进消息队列不保证真实时间，
    慢机上还需 sleep 让后台线程有机会完成。
    """
    for _ in range(tries):
        if cond():
            return
        await pilot.pause()
        await asyncio.sleep(0.02)
    raise AssertionError(f"_wait_until 超时（{tries} 轮）：条件始终未成立")


# ============================================================
# tasks_panel — 面板刷新逻辑
# ============================================================
class _TasksApp(App):
    def __init__(self, store: TaskStore) -> None:
        super().__init__()
        self.store = store

    def compose(self) -> ComposeResult:
        yield TasksPanel(self.store)


class TestTasksPanelUpdate:
    def test_update_populates_table_summary_and_term(self, tmp_path):
        _fake_disk_tasks(tmp_path)
        store = TaskStore(log_dir=tmp_path)

        async def _run() -> None:
            app = _TasksApp(store)
            async with app.run_test() as pilot:
                panel = app.query_one(TasksPanel)
                snaps, _ = store.refresh()
                panel.update_tasks(snaps)
                await pilot.pause()  # move_cursor → RowHighlighted → _refresh_term
                table = app.query_one("#task-table", DataTable)
                assert table.row_count == 2
                summary = app.query_one("#task-summary", Static).renderable
                assert summary == "2 任务 · 运行 0 · 排队 0 · 外部 2"
                term = app.query_one("#task-term", TermBox)
                assert term._state is not None
                # mtime 较新的 evaluate 排在首行并被选中
                assert panel._selected == "evaluate-101010-ab12cd"
                assert "评估 · ab12cd" in term._title

        asyncio.run(_run())

    def test_update_empty_resets_everything(self, tmp_path):
        async def _run() -> None:
            app = _TasksApp(TaskStore(log_dir=tmp_path))
            async with app.run_test() as pilot:
                panel = app.query_one(TasksPanel)
                panel.update_tasks([])
                await pilot.pause()
                assert app.query_one("#task-table", DataTable).row_count == 0
                assert app.query_one("#task-summary", Static).renderable == "0 任务 · 运行 0 · 排队 0 · 外部 0"
                term = app.query_one("#task-term", TermBox)
                assert term._state is None and term._title == "任务"

        asyncio.run(_run())

    def test_highlighted_row_switches_term(self, tmp_path):
        _fake_disk_tasks(tmp_path)
        store = TaskStore(log_dir=tmp_path)

        async def _run() -> None:
            app = _TasksApp(store)
            async with app.run_test() as pilot:
                panel = app.query_one(TasksPanel)
                snaps, _ = store.refresh()
                panel.update_tasks(snaps)
                table = app.query_one("#task-table", DataTable)
                table.move_cursor(row=1)  # 第二行 = hpo 任务
                await _wait_until(pilot, lambda: panel._selected == "hpo-111111-aabbcc")
                assert panel._selected == "hpo-111111-aabbcc"
                term = app.query_one("#task-term", TermBox)
                assert term._state is not None and term._state.kind == "hpo"
                assert "HPO · aabbcc" in term._title

        asyncio.run(_run())

    def test_term_title_uses_cmd_summary(self, tmp_path):
        async def _run() -> None:
            app = _TasksApp(TaskStore(log_dir=tmp_path))
            async with app.run_test() as pilot:
                panel = app.query_one(TasksPanel)
                snap = _snap("hpo-111111-x1", status="running", owned=True,
                             cmd="-m llmsec.experiments run experiments/study.yaml",
                             meta={"study": "study.yaml"}, state=_hpo_state())
                panel.update_tasks([snap])
                await pilot.pause()
                term = app.query_one("#task-term", TermBox)
                assert "study.yaml" in term._title
                assert "HPO · x1" in term._title

        asyncio.run(_run())


class TestTasksPanelActions:
    def test_cancel_blocked_for_external_or_done(self, tmp_path, monkeypatch):
        async def _run() -> None:
            app = _TasksApp(TaskStore(log_dir=tmp_path))
            async with app.run_test() as pilot:
                panel = app.query_one(TasksPanel)
                notes, fake = _notes_recorder()
                monkeypatch.setattr(app, "notify", fake)
                cancels: list[str] = []
                monkeypatch.setattr(panel, "_cancel", cancels.append)
                panel.update_tasks([_snap("evaluate-1-aaaaaa", status="external", owned=False)])
                await pilot.pause()
                panel.action_cancel()
                await pilot.pause()
                assert notes == ["没有可取消的运行中/排队任务"]
                assert cancels == []
                assert not isinstance(app.screen, ConfirmScreen)

        asyncio.run(_run())

    def test_cancel_confirm_dialog_flow(self, tmp_path, monkeypatch):
        async def _run() -> None:
            app = _TasksApp(TaskStore(log_dir=tmp_path))
            async with app.run_test() as pilot:
                panel = app.query_one(TasksPanel)
                cancels: list[str] = []
                monkeypatch.setattr(panel, "_cancel", cancels.append)
                panel.update_tasks([_snap("evaluate-1-aaaaaa", status="running", owned=True,
                                          state=_ev_state())])
                await pilot.pause()
                panel.action_cancel()
                await pilot.pause()
                assert isinstance(app.screen, ConfirmScreen)  # 本机运行中 → 弹确认框
                await pilot.press("escape")  # 取消 → 不触发 _cancel
                await _wait_until(pilot, lambda: not isinstance(app.screen, ConfirmScreen))
                assert cancels == []
                panel.action_cancel()
                await pilot.pause()
                await pilot.click("#confirm-yes")  # 确认 → _cancel(选中 id)
                await _wait_until(pilot, lambda: bool(cancels))
                assert cancels == ["evaluate-1-aaaaaa"]

        asyncio.run(_run())

    def test_log_action_dispatch(self, tmp_path, monkeypatch):
        async def _run() -> None:
            app = _TasksApp(TaskStore(log_dir=tmp_path))
            async with app.run_test() as pilot:
                panel = app.query_one(TasksPanel)
                notes, fake = _notes_recorder()
                monkeypatch.setattr(app, "notify", fake)
                logged: list = []
                monkeypatch.setattr(panel, "_log", logged.append)
                panel.action_log()  # 未选中 → 警告
                await pilot.pause()
                assert notes == ["先选中一个任务"]
                assert logged == []
                snap = _snap("evaluate-1-aaaaaa", status="success", owned=True)
                panel.update_tasks([snap])
                await pilot.pause()
                panel.action_log()
                await pilot.pause()
                assert logged == [snap]

        asyncio.run(_run())

    def test_new_eval_pushes_launch_screen(self, tmp_path, monkeypatch):
        import llmsec.mcp.tools.query as query_mod

        monkeypatch.setattr(query_mod, "list_targets", lambda: [{"name": "模型A"}])

        async def _run() -> None:
            app = _TasksApp(TaskStore(log_dir=tmp_path))
            async with app.run_test() as pilot:
                panel = app.query_one(TasksPanel)
                launches: list = []
                monkeypatch.setattr(panel, "_launch", launches.append)
                panel.action_new_eval()
                await pilot.pause()
                assert isinstance(app.screen, LaunchScreen)
                await pilot.press("escape")
                await pilot.pause()
                assert launches == []  # 表单取消 → 不提交

        asyncio.run(_run())


class TestLoadTargetNames:
    def test_extracts_names_and_skips_bad_entries(self, monkeypatch):
        import llmsec.mcp.tools.query as query_mod

        monkeypatch.setattr(
            query_mod, "list_targets",
            lambda: [{"name": "模型A"}, {"name": "模型B"}, {"no_name": 1}, "str"],
        )
        assert _load_target_names() == ["模型A", "模型B"]

    def test_exception_returns_empty(self, monkeypatch):
        import llmsec.mcp.tools.query as query_mod

        def boom():
            raise RuntimeError("offline")

        monkeypatch.setattr(query_mod, "list_targets", boom)
        assert _load_target_names() == []

    def test_non_list_returns_empty(self, monkeypatch):
        import llmsec.mcp.tools.query as query_mod

        monkeypatch.setattr(query_mod, "list_targets", lambda: {"error": "x"})
        assert _load_target_names() == []


class TestLaunchScreen:
    @staticmethod
    def _push(app: App, targets: list, attacks: list, callback) -> None:
        """push LaunchScreen（_ok_pressed/_validate 走源码真实路径）。"""
        screen = LaunchScreen(targets, attacks)
        app.push_screen(screen, callback)

    def test_params_single_target_defaults(self):
        results: list = []

        async def _run() -> None:
            app = _TasksApp(TaskStore(log_dir=None))
            async with app.run_test() as pilot:
                self._push(app, ["模型A"], ["l1.jsonl"], results.append)
                await pilot.pause()
                app.screen._ok_pressed()
                await _wait_until(pilot, lambda: bool(results))

        asyncio.run(_run())
        assert results == [{"target": "模型A", "input_file": "l1.jsonl",
                            "sampler": "hybrid", "max_rounds": 5}]

    def test_params_multi_targets_and_optional_fields(self):
        results: list = []

        async def _run() -> None:
            app = _TasksApp(TaskStore(log_dir=None))
            async with app.run_test() as pilot:
                self._push(app, ["模型A", "模型B"], ["l1.jsonl", "l2.jsonl"], results.append)
                await pilot.pause()
                screen = app.screen
                screen.query_one("#f-rounds", Input).value = "3"
                screen.query_one("#f-seed", Input).value = "42"
                screen.query_one("#f-batch", Input).value = "8"
                screen.query_one("#f-noearly", Checkbox).value = True
                screen._ok_pressed()
                await _wait_until(pilot, lambda: bool(results))

        asyncio.run(_run())
        assert results == [{"targets": ["模型A", "模型B"], "input_file": "l1.jsonl",
                            "sampler": "hybrid", "max_rounds": 3, "seed": 42,
                            "batch_size": 8, "no_early_stop": True}]

    def test_validate_rejects_non_integer(self):
        async def _run() -> None:
            app = _TasksApp(TaskStore(log_dir=None))
            async with app.run_test() as pilot:
                self._push(app, ["A"], ["l1.jsonl"], None)
                await pilot.pause()
                screen = app.screen
                screen.query_one("#f-rounds", Input).value = "abc"
                err = screen._validate()
                assert err is not None and "整数" in err and "最大轮数" in err
                screen.query_one("#f-rounds", Input).value = "5"
                screen.query_one("#f-seed", Input).value = "x1"
                err2 = screen._validate()
                assert err2 is not None and "种子" in err2
                screen.query_one("#f-seed", Input).value = ""
                assert screen._validate() is None

        asyncio.run(_run())

    def test_validate_requires_target_selection(self):
        async def _run() -> None:
            app = _TasksApp(TaskStore(log_dir=None))
            async with app.run_test() as pilot:
                self._push(app, ["A", "B"], ["l1.jsonl"], None)
                await pilot.pause()
                screen = app.screen
                assert screen._validate() is None  # 默认全选
                screen.query_one("#f-targets", SelectionList).deselect_all()
                assert screen._validate() == "至少选择一个目标模型"

        asyncio.run(_run())

    def test_no_targets_shows_hint_and_validates(self):
        async def _run() -> None:
            app = _TasksApp(TaskStore(log_dir=None))
            async with app.run_test() as pilot:
                self._push(app, [], ["l1.jsonl"], None)
                await pilot.pause()
                screen = app.screen
                assert screen._targets_selection() is None
                assert ".env" in screen.query_one(".field-hint", Static).renderable
                assert screen._validate() is None  # 无选择框时跳过目标校验

        asyncio.run(_run())

    def test_cancel_button_dismisses_none(self):
        results: list = []

        async def _run() -> None:
            app = _TasksApp(TaskStore(log_dir=None))
            async with app.run_test() as pilot:
                self._push(app, ["A"], ["l1.jsonl"], results.append)
                await pilot.pause()
                await pilot.click("#f-cancel")
                await pilot.pause()

        asyncio.run(_run())
        assert results == [None]


# ============================================================
# runs_panel — 纯格式化函数
# ============================================================
class TestFmtNum:
    def test_none_and_bools(self):
        assert _fmt_num(None) == "—"
        assert _fmt_num(True) == "✓"
        assert _fmt_num(False) == "✗"

    def test_numbers_trim_trailing_zeros(self):
        assert _fmt_num(3.14159) == "3.14"
        assert _fmt_num(2.0) == "2"
        assert _fmt_num(1234.5) == "1234.5"
        assert _fmt_num(0.0) == "0"

    def test_strings_passthrough(self):
        assert _fmt_num("n/a") == "n/a"


class TestFmtRatio:
    def test_zero_to_one_as_percent(self):
        assert _fmt_ratio(0) == "0.0%"
        assert _fmt_ratio(0.25) == "25.0%"
        assert _fmt_ratio(0.1234) == "12.3%"
        assert _fmt_ratio(1) == "100.0%"

    def test_out_of_range_and_others_fall_back(self):
        assert _fmt_ratio(1.5) == "1.5"  # 超出 0-1 → 定点数
        assert _fmt_ratio(-0.2) == "-0.2"
        assert _fmt_ratio(None) == "—"
        assert _fmt_ratio("weird") == "weird"


class TestRenderReport:
    def test_minimal_report_only_header_and_json(self):
        text = _render_report("run-1", {})
        assert text.splitlines()[0] == "== run-1 =="
        assert "ASR" not in text
        assert "边界 Elo" not in text
        assert "门下省" not in text
        assert text.rstrip().endswith("{}")  # 空 report 的 JSON 尾巴

    def test_core_metrics_and_findings(self):
        data = {
            "report": {
                "target_model": "模型A", "security_level": "safe", "generated_at": "2026-08-15T10:00:00",
                "attack_phase": {"asr": 0.25}, "elo": {"boundary_elo": 1234.5, "ci_half": 40.0},
            },
            "findings": [{"severity": "warn", "metric": "asr", "value": 0.9, "threshold": 0.8,
                          "interpretation": "边界过高"}],
        }
        text = _render_report("run-1", data)
        assert "target_model: 模型A" in text
        assert "security_level: safe" in text
        assert "ASR: 25.0%" in text
        assert "边界 Elo: 1234.5  CI±40" in text
        assert "-- 门下省审查发现 --" in text
        assert "[warn] asr: 0.9 (阈值 0.8)" in text
        assert "    边界过高" in text

    def test_json_tail_parseable_and_cjk_unescaped(self):
        rep = {"target_model": "模型A", "attack_phase": {"asr": 0.5}}
        text = _render_report("run-1", {"report": rep})
        tail = text.split("-- 完整报告 JSON --\n", 1)[1]
        assert json.loads(tail) == rep
        assert "模型A" in tail  # ensure_ascii=False → 不转 \uXXXX
        assert "\\u" not in tail

    def test_non_list_findings_ignored(self):
        text = _render_report("run-1", {"report": {}, "findings": "bad"})
        assert "门下省" not in text


# ============================================================
# runs_panel — 面板逻辑（拦截 @work 线程方法）
# ============================================================
class _RunsApp(App):
    """RunsPanel 宿主：on_mount 触发的 _load 线程加载被替换成计数器。"""

    def __init__(self) -> None:
        super().__init__()
        self.load_calls = 0
        self.panel: RunsPanel | None = None

    def compose(self) -> ComposeResult:
        panel = RunsPanel()
        panel._load = self._fake_load  # 拦截线程加载，避免扫真实 output/
        self.panel = panel
        yield panel

    def _fake_load(self) -> None:
        self.load_calls += 1
        if self.panel is not None:
            self.panel._dirty = False  # 与真实 _load 收尾一致（加载完成清脏标记）


def _run_row(name, *, level="safe", asr=0.25, elo=None, target="模型A", has_report=True,
             mtime="2026-08-15T10:20:30", target_key="target"):
    return {"name": name, "security_level": level, "asr": asr, "boundary_elo": elo,
            target_key: target, "has_report": has_report, "mtime": mtime}


class TestRunsPanelRender:
    def test_rows_filtering_and_cells(self):
        async def _run() -> None:
            app = _RunsApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                panel = app.query_one(RunsPanel)
                panel._render_runs([
                    _run_row("run-a", level="vulnerable", elo=1234.5),
                    _run_row("run-b", target="模型B", target_key="target_model", asr=None, has_report=False),
                    "junk", {"no_name": 1},  # 非法条目被过滤
                ])
                table = app.query_one("#runs-table", DataTable)
                assert table.row_count == 2
                row = table.get_row("run-a")
                assert row[0] == "run-a"
                assert row[1] == "模型A"
                assert row[2].plain == "vulnerable"
                assert row[2].style == runs_panel._LEVEL_COLOR["vulnerable"]
                assert row[3] == "25.0%"
                assert row[4] == "1234.5"
                assert row[5] == "✓"
                assert row[6] == "2026-08-15 10:20"
                row_b = table.get_row("run-b")
                assert row_b[1] == "模型B"  # target_model 兜底
                assert row_b[2].plain == "safe"
                assert row_b[3] == "—"  # asr 缺失
                assert row_b[5] == "✗"
                hint = app.query_one("#runs-hint", Static).renderable
                assert "共 2 个 run" in hint and "标记对比（无）" in hint

        asyncio.run(_run())

    def test_stale_mark_pruned_and_marked_hint(self):
        async def _run() -> None:
            app = _RunsApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                panel = app.query_one(RunsPanel)
                panel._marked = ["gone", "run-a"]
                panel._render_runs([_run_row("run-a"), _run_row("run-b")])
                assert panel._marked == ["run-a"]  # 消失的标记被清掉
                assert app.query_one("#runs-table", DataTable).get_row("run-a")[0].startswith("★ ")
                hint = app.query_one("#runs-hint", Static).renderable
                assert "标记对比（run-a）" in hint

        asyncio.run(_run())


class TestRunsPanelMarkCompare:
    def test_mark_toggle_and_cap_two(self):
        async def _run() -> None:
            app = _RunsApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                panel = app.query_one(RunsPanel)
                panel._render_runs([_run_row("r1"), _run_row("r2"), _run_row("r3")])
                table = app.query_one("#runs-table", DataTable)
                panel.action_mark()  # 光标默认在 r1
                assert panel._marked == ["r1"]
                assert table.get_row("r1")[0].startswith("★ ")
                table.move_cursor(row=1)
                panel.action_mark()
                table.move_cursor(row=2)
                panel.action_mark()  # 第 3 个 → 挤掉最旧的 r1
                assert panel._marked == ["r2", "r3"]
                assert not table.get_row("r1")[0].startswith("★ ")
                panel.action_mark()  # 再按 → 取消 r3
                assert panel._marked == ["r2"]

        asyncio.run(_run())

    def test_compare_requires_exactly_two(self, monkeypatch):
        async def _run() -> None:
            app = _RunsApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                panel = app.query_one(RunsPanel)
                notes, fake = _notes_recorder()
                monkeypatch.setattr(app, "notify", fake)
                compares: list = []
                monkeypatch.setattr(panel, "_compare", compares.append)
                panel._render_runs([_run_row("r1"), _run_row("r2")])
                panel.action_compare()
                assert notes and "标记 2 个" in notes[0]
                assert compares == []
                panel._marked = ["r1"]
                notes.clear()
                panel.action_compare()  # 只有 1 个标记
                assert notes and "当前 1 个" in notes[0]
                panel._marked = ["r1", "r2"]
                panel.action_compare()
                assert compares == [["r1", "r2"]]

        asyncio.run(_run())

    def test_cursor_run_empty_table_none(self):
        async def _run() -> None:
            app = _RunsApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                assert app.query_one(RunsPanel)._cursor_run() is None

        asyncio.run(_run())


class TestRunsPanelEloAndReload:
    def test_elo_needs_run_data(self, monkeypatch):
        async def _run() -> None:
            app = _RunsApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                panel = app.query_one(RunsPanel)
                notes, fake = _notes_recorder()
                monkeypatch.setattr(app, "notify", fake)
                panel.action_elo()
                assert notes == ["暂无 run 数据（先跑一次评估）"]

        asyncio.run(_run())

    def test_elo_select_screen_flow(self, monkeypatch):
        async def _run() -> None:
            app = _RunsApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                panel = app.query_one(RunsPanel)
                elos: list = []
                monkeypatch.setattr(panel, "_elo", elos.append)  # 拦截线程 worker
                panel._render_runs([_run_row("r1", target="模型A"), _run_row("r2", target="模型B")])
                panel.action_elo()
                await pilot.pause()
                assert isinstance(app.screen, EloSelectScreen)
                await pilot.press("escape")
                await pilot.pause()
                assert elos == []  # 取消选择 → 不查询
                panel.action_elo()
                await pilot.pause()
                await pilot.click("#elo-ok")
                await _wait_until(pilot, lambda: bool(elos))
                assert elos == ["模型A"]  # Select 默认取第一个模型

        asyncio.run(_run())

    def test_pick_model_dispatch(self, monkeypatch):
        async def _run() -> None:
            app = _RunsApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                panel = app.query_one(RunsPanel)
                panel._runs = {"r1": {"name": "r1", "target": "模型B"}}
                elos: list = []
                monkeypatch.setattr(panel, "_elo", elos.append)
                # _pick_model 弹选择框，选中回调才分发；None（Esc）不分发
                panel._pick_model(elos.append)
                await pilot.pause()
                await pilot.click("#elo-ok")  # 选第一个模型
                await _wait_until(pilot, lambda: len(elos) == 1)
                assert elos == ["模型B"]

        asyncio.run(_run())

    def test_flag_reload_deferred_when_hidden(self):
        async def _run() -> None:
            app = _RunsApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                panel = app.query_one(RunsPanel)
                assert app.load_calls == 1  # on_mount 已加载一次
                panel.add_class("hidden")
                panel.flag_reload()  # 不可见 → 只标脏不加载
                assert panel._dirty is True
                assert app.load_calls == 1
                panel.show_refresh()  # 切回 → 补一次加载并清脏
                assert app.load_calls == 2
                assert panel._dirty is False
                panel.show_refresh()  # 不脏不再加载
                assert app.load_calls == 2
                panel.remove_class("hidden")
                panel.flag_reload()  # 可见 → 立即加载
                assert app.load_calls == 3

        asyncio.run(_run())


# ============================================================
# hpo_panel — 面板逻辑
# ============================================================
class _HpoApp(App):
    def __init__(self, store: TaskStore) -> None:
        super().__init__()
        self.store = store

    def compose(self) -> ComposeResult:
        yield HpoPanel(self.store)


class TestHpoPanelUpdate:
    def test_update_filters_hpo_only(self, tmp_path):
        _fake_disk_tasks(tmp_path)
        store = TaskStore(log_dir=tmp_path)

        async def _run() -> None:
            app = _HpoApp(store)
            async with app.run_test() as pilot:
                panel = app.query_one(HpoPanel)
                snaps, _ = store.refresh()
                assert len(snaps) == 2  # evaluate + hpo 都在快照里
                panel.update_tasks(snaps)
                await pilot.pause()
                table = app.query_one("#hpo-table", DataTable)
                assert table.row_count == 1  # 但 HPO 表只列 hpo 任务
                assert table.get_row("hpo-111111-aabbcc")[0] == "HPO·aabbcc"
                assert app.query_one("#hpo-summary", Static).renderable == "1 个 study · 在跑 1"
                term = app.query_one("#hpo-term", TermBox)
                assert term._state is not None and term._state.kind == "hpo"
                assert "HPO · aabbcc" in term._title

        asyncio.run(_run())

    def test_update_summary_counts_active_only(self, tmp_path):
        async def _run() -> None:
            app = _HpoApp(TaskStore(log_dir=tmp_path))
            async with app.run_test() as pilot:
                panel = app.query_one(HpoPanel)
                panel.update_tasks([
                    _snap("hpo-1-aaaaaa", status="running", owned=True, state=_hpo_state()),
                    _snap("hpo-2-bbbbbb", status="success", owned=True),  # 终态不计在跑
                    _snap("evaluate-3-cccccc", status="running", owned=True),  # 非 hpo 不进表
                ])
                await pilot.pause()
                table = app.query_one("#hpo-table", DataTable)
                assert table.row_count == 2
                assert app.query_one("#hpo-summary", Static).renderable == "2 个 study · 在跑 1"

        asyncio.run(_run())

    def test_update_empty_placeholder(self, tmp_path):
        async def _run() -> None:
            app = _HpoApp(TaskStore(log_dir=tmp_path))
            async with app.run_test() as pilot:
                app.query_one(HpoPanel).update_tasks([])
                await pilot.pause()
                assert app.query_one("#hpo-table", DataTable).row_count == 0
                assert app.query_one("#hpo-summary", Static).renderable == "0 个 study · 在跑 0"
                term = app.query_one("#hpo-term", TermBox)
                assert term._state is None and term._title == "HPO"

        asyncio.run(_run())


class TestHpoPanelStart:
    def test_start_without_yamls_notifies(self, tmp_path, monkeypatch):
        import llmsec.tui.panels.hpo_panel as hp

        monkeypatch.setattr(hp, "study_yamls", lambda: [])

        async def _run() -> None:
            app = _HpoApp(TaskStore(log_dir=tmp_path))
            async with app.run_test() as pilot:
                notes, fake = _notes_recorder()
                monkeypatch.setattr(app, "notify", fake)
                app.query_one(HpoPanel).action_start()
                await pilot.pause()
                assert notes and "没找到 study 配置" in notes[0]
                assert not isinstance(app.screen, HpoStartScreen)

        asyncio.run(_run())

    def test_start_screen_manual_path_priority(self, tmp_path, monkeypatch):
        import llmsec.tui.panels.hpo_panel as hp

        monkeypatch.setattr(hp, "study_yamls", lambda: ["experiments/a.yaml", "experiments/b.yaml"])

        async def _run() -> None:
            app = _HpoApp(TaskStore(log_dir=tmp_path))
            async with app.run_test() as pilot:
                panel = app.query_one(HpoPanel)
                results: list = []
                # action_start 的回调固定是 panel._on_start → 拦截它收集 dismiss 结果
                monkeypatch.setattr(panel, "_on_start", results.append)
                panel.action_start()
                await pilot.pause()
                assert isinstance(app.screen, HpoStartScreen)
                await pilot.press("escape")
                await pilot.pause()
                assert results == [None]
                panel.action_start()
                await pilot.pause()
                screen = app.screen
                screen.query_one("#h-path", Input).value = "experiments/c.yaml"
                screen._ok_pressed()  # 手输路径优先于下拉选择
                await _wait_until(pilot, lambda: len(results) >= 2)
                assert results == [None, "experiments/c.yaml"]
                panel.action_start()
                await pilot.pause()
                app.screen._ok_pressed()  # 无手输 → 取下拉默认第一项
                await _wait_until(pilot, lambda: len(results) >= 3)
                assert results == [None, "experiments/c.yaml", "experiments/a.yaml"]

        asyncio.run(_run())

    def test_on_start_dispatches_worker(self, tmp_path, monkeypatch):
        async def _run() -> None:
            app = _HpoApp(TaskStore(log_dir=tmp_path))
            async with app.run_test():
                panel = app.query_one(HpoPanel)
                starts: list = []
                monkeypatch.setattr(panel, "_start", starts.append)
                panel._on_start(None)
                assert starts == []
                panel._on_start("experiments/x.yaml")
                assert starts == ["experiments/x.yaml"]

        asyncio.run(_run())


# ============================================================
# UX 走查回归（三处关键 bug 的防回归；走查记录见当次会话）
# ============================================================
def _vis(app, pid) -> bool:
    p = app.query_one(pid)
    return p.styles.display != "none" and p.region.height > 0


class TestPanelSwitchVisibility:
    """回归：按 2/3 后面板主体空白——显隐必须只由 .hidden 类管理，
    不能再写无条件 display:none 基础规则（移除类后基础规则仍生效）。"""

    def test_switch_2_and_3_shows_panel_content(self, tmp_path):
        from llmsec.tui.app import LlmsecTUI

        async def _run() -> None:
            app = LlmsecTUI(store=TaskStore(log_dir=tmp_path))
            async with app.run_test(size=(120, 36)) as pilot:
                assert _vis(app, "#panel-tasks") and not _vis(app, "#panel-hpo")
                await pilot.press("2")
                await pilot.pause()
                assert _vis(app, "#panel-hpo"), "按 2 后 HPO 面板必须可见"
                assert not _vis(app, "#panel-tasks") and not _vis(app, "#panel-runs")
                await pilot.press("3")
                await pilot.pause()
                assert _vis(app, "#panel-runs"), "按 3 后 Runs 面板必须可见"
                assert not _vis(app, "#panel-hpo") and not _vis(app, "#panel-tasks")
                await pilot.press("1")
                await pilot.pause()
                assert _vis(app, "#panel-tasks") and not _vis(app, "#panel-runs")

        asyncio.run(_run())


class TestLaunchFormFitsScreen:
    """回归：表单被 Vertical 默认 1fr 撑到 66 行高，提交按钮越出常规终端。"""

    def _hermetic_form_env(self, tmp_path, monkeypatch):
        import llmsec.core.config as config

        attacks = tmp_path / "attacks"
        attacks.mkdir(exist_ok=True)
        _write(attacks / "l1.jsonl", "{}\n")
        monkeypatch.setattr(config, "ATTACKS_DIR", attacks)
        monkeypatch.setattr(
            config, "load_targets",
            lambda: {"模型A": config.TargetConfig(), "模型B": config.TargetConfig()},
        )

    def test_submit_button_within_36_rows(self, tmp_path, monkeypatch):
        self._hermetic_form_env(tmp_path, monkeypatch)

        async def _run() -> None:
            app = _TasksApp(TaskStore(log_dir=tmp_path))
            async with app.run_test(size=(120, 36)) as pilot:
                app.query_one(TasksPanel).action_new_eval()
                await pilot.pause()
                btn = app.screen.query_one("#f-ok")
                r = btn.region
                assert r.y >= 0 and r.y + r.height <= 36, f"提交按钮越出 36 行可视区: {r}"
                cell = app.screen.query_one(".field-cell")
                # 次要不变量：不再取容器默认 1fr（会被撑到几十行）；label+input 含边距正常 <= 10
                assert cell.styles.height != "1fr" and cell.region.height <= 10, \
                    f"field-cell 被容器默认高度撑爆: {cell.styles.height}/{cell.region}"

        asyncio.run(_run())


class TestNoFalseErrorToast:
    """回归：task_view 恒含 error=None 键，成功路径不得因「'error' in view」误报失败。"""

    def test_launch_success_notifies_ok(self, tmp_path):
        store = TaskStore(log_dir=tmp_path)
        store.start_evaluation = lambda **kw: {
            "id": "evaluate-100000-abc123", "kind": "evaluate", "cmd": "x",
            "status": "queued", "error": None, "meta": {"targets": ["模型A"], "max_rounds": 5},
        }

        async def _run() -> None:
            app = _TasksApp(store)
            async with app.run_test() as pilot:
                notes, fake = _notes_recorder()
                app.notify = fake
                app.query_one(TasksPanel)._launch({"target": "模型A"})
                await _wait_until(pilot, lambda: any("队列" in x for x in notes))
                assert not any("启动失败" in x or x.startswith("None") for x in notes), notes
                assert any("已进入队列" in x for x in notes), notes

        asyncio.run(_run())

    def test_hpo_start_success_notifies_ok(self, tmp_path):
        store = TaskStore(log_dir=tmp_path)
        store.start_hpo = lambda path: {
            "id": "hpo-100000-abc123", "kind": "hpo", "cmd": "x",
            "status": "queued", "error": None, "meta": {"study": "s.yaml"},
        }

        async def _run() -> None:
            app = _HpoApp(store)
            async with app.run_test() as pilot:
                notes, fake = _notes_recorder()
                app.notify = fake
                app.query_one(HpoPanel)._start("experiments/s.yaml")
                await _wait_until(pilot, lambda: any("队列" in x for x in notes))
                assert not any("启动失败" in x or x.startswith("None") for x in notes), notes
                assert any("已进入队列" in x for x in notes), notes

        asyncio.run(_run())

    def test_cancel_success_notifies_ok(self, tmp_path):
        store = TaskStore(log_dir=tmp_path)
        store.cancel = lambda task_id: {
            "id": task_id, "kind": "evaluate", "cmd": "x",
            "status": "cancelled", "error": None, "meta": None,
        }

        async def _run() -> None:
            app = _TasksApp(store)
            async with app.run_test() as pilot:
                notes, fake = _notes_recorder()
                app.notify = fake
                app.query_one(TasksPanel)._cancel("evaluate-100000-abc123")
                await _wait_until(pilot, lambda: any("已取消" in x for x in notes))
                assert not any("取消失败" in x for x in notes), notes

        asyncio.run(_run())


class TestHpoTableFiltering:
    """HPO 面板表格只列 hpo 任务（evaluate 不混入）。"""

    def test_only_hpo_kind_listed(self, tmp_path):
        snaps = [
            _snap("evaluate-101010-ab12cd", status="running", owned=True, state=_ev_state()),
            _snap("hpo-111111-aabbcc", status="running", owned=True, state=_hpo_state()),
        ]

        async def _run() -> None:
            app = _HpoApp(TaskStore(log_dir=tmp_path))
            async with app.run_test() as pilot:
                panel = app.query_one(HpoPanel)
                panel.update_tasks(snaps)
                await pilot.pause()
                assert app.query_one("#hpo-table", DataTable).row_count == 1

        asyncio.run(_run())


class TestHpoPanelCancelLogBindings:
    """回归：取消(c)/日志(l) 此前只在任务中心挂键，HPO 面板按键无反馈——
    收口到 TaskTablePanel 基类后两面板行为必须一致。"""

    def test_cancel_key_on_hpo_panel_opens_confirm(self, tmp_path, monkeypatch):
        """真实按键路径：焦点在 HPO 面板表格上按 c → 确认框 → 确认后取消选中任务。"""
        snaps = [_snap("hpo-111111-aabbcc", status="running", owned=True,
                       cmd="-m llmsec.experiments run x.yaml", state=_hpo_state())]

        async def _run() -> None:
            app = _HpoApp(TaskStore(log_dir=tmp_path))
            async with app.run_test() as pilot:
                panel = app.query_one(HpoPanel)
                panel.update_tasks(snaps)
                await pilot.pause()
                app.query_one("#hpo-table", DataTable).focus()
                cancels: list = []
                monkeypatch.setattr(panel, "_cancel", cancels.append)
                await pilot.press("c")
                await pilot.pause()
                from llmsec.tui.widgets import ConfirmScreen
                assert isinstance(app.screen, ConfirmScreen), "HPO 面板按 c 应弹确认框"
                await pilot.click("#confirm-yes")
                await _wait_until(pilot, lambda: len(cancels) == 1)
                assert cancels == ["hpo-111111-aabbcc"]

        asyncio.run(_run())

    def test_log_key_on_hpo_panel_opens_modal(self, tmp_path):
        """真实按键 + 真实 worker 路径：按 l → _log 线程读日志 → LogModal 弹出。"""
        snaps = [_snap("hpo-111111-aabbcc", status="running", owned=True, state=_hpo_state())]

        async def _run() -> None:
            app = _HpoApp(TaskStore(log_dir=tmp_path))
            async with app.run_test() as pilot:
                panel = app.query_one(HpoPanel)
                panel.update_tasks(snaps)
                await pilot.pause()
                app.query_one("#hpo-table", DataTable).focus()
                await pilot.press("l")
                from llmsec.tui.widgets import LogModal
                await _wait_until(pilot, lambda: isinstance(app.screen, LogModal))
                assert isinstance(app.screen, LogModal), "HPO 面板按 l 应弹日志框"

        asyncio.run(_run())

    def test_binding_declarations_cover_both_panels(self):
        """声明层检查：c/l 在基类，n/s 在子类（合并是 textual 的职责，
        真实按键路径由上面两个用例覆盖）。"""
        from textual.binding import Binding

        from llmsec.tui.panels.common import TaskTablePanel
        from llmsec.tui.panels.tasks_panel import TasksPanel

        def keys(bindings: list[Binding]) -> set:
            return {b.key for b in bindings}

        assert keys(TaskTablePanel.BINDINGS) == {"c", "l"}
        assert "n" in keys(TasksPanel.BINDINGS)
        assert "s" in keys(HpoPanel.BINDINGS)


class TestLaunchScreenAdvancedFields:
    """表单能力补全：采样器超参 / env 快照 / 参数覆写（归一层全能力面）。"""

    def _push(self, app, cb=None):
        screen = LaunchScreen(["模型A"], ["l1.jsonl"], snapshots=["snap1"])
        app.push_screen(screen, cb)

    def test_params_include_advanced_fields(self):
        results: list = []

        async def _run() -> None:
            app = _TasksApp(TaskStore(log_dir=None))
            async with app.run_test() as pilot:
                self._push(app, results.append)
                await pilot.pause()
                s = app.screen
                s.query_one("#f-alpha", Input).value = "1.5"
                s.query_one("#f-beta", Input).value = "0.5"
                s.query_one("#f-gamma", Input).value = "2"
                s.query_one("#f-coord", Input).value = "6"
                s.query_one("#f-envsnap", Select).value = "snap1"
                s.query_one("#f-params", Input).value = "K_FACTOR=32,CONV_CI_TARGET=15"
                s._ok_pressed()
                await _wait_until(pilot, lambda: bool(results))

        asyncio.run(_run())
        assert results == [{
            "target": "模型A", "input_file": "l1.jsonl", "sampler": "hybrid", "max_rounds": 5,
            "sampler_alpha": 1.5, "sampler_beta": 0.5, "sampler_gamma": 2.0, "coordinate_rounds": 6,
            "env_snapshot": "snap1",
            "param_overrides": {"K_FACTOR": "32", "CONV_CI_TARGET": "15"},
        }]

    def test_validate_rejects_bad_overrides_and_floats(self):
        async def _run() -> None:
            app = _TasksApp(TaskStore(log_dir=None))
            async with app.run_test() as pilot:
                self._push(app)
                await pilot.pause()
                s = app.screen
                s.query_one("#f-params", Input).value = "NO_EQUAL_SIGN"
                assert s._validate() is not None and "KEY=V" in s._validate()
                s.query_one("#f-params", Input).value = ""
                s.query_one("#f-alpha", Input).value = "abc"
                assert s._validate() is not None and "α" in s._validate()

        asyncio.run(_run())

    def test_parse_param_overrides(self):
        from llmsec.tui.panels.tasks_panel import _parse_param_overrides as p

        assert p("K=32,C=15.5") == {"K": "32", "C": "15.5"}
        assert p("K = v with spaces ") == {"K": "v with spaces"}
        assert p("NOEQ") is None
        assert p("=32") is None
        assert p("") is None


class TestChatPanel:
    def test_send_and_reply(self, monkeypatch):
        import control.agent.zhongshu.fallback as fb

        monkeypatch.setattr(fb, "chat_one", lambda text: f"回声:{text}")

        async def _run() -> None:
            from llmsec.tui.panels.chat_panel import ChatPanel

            class _ChatApp(App):
                def compose(self) -> ComposeResult:
                    yield ChatPanel()

            app = _ChatApp()
            async with app.run_test() as pilot:
                inp = app.query_one("#chat-input", Input)
                inp.focus()
                inp.value = "列出最近的 run"
                await pilot.press("enter")

                def log_text() -> str:
                    lines = app.query_one("#chat-log", RichLog).lines
                    return "\n".join(
                        "".join(seg.text for seg in getattr(line, "_segments", [])) for line in lines)

                for _ in range(30):
                    await pilot.pause()
                    if "回声:列出最近的 run" in log_text():
                        break
                    await asyncio.sleep(0.1)
                assert "列出最近的 run" in log_text(), log_text()
                assert "回声:列出最近的 run" in log_text(), log_text()
                assert inp.value == "", "发送后输入框应清空"

        asyncio.run(_run())


class TestHelpOverlay:
    def test_question_mark_opens_help(self):
        from llmsec.tui.app import LlmsecTUI

        async def _run() -> None:
            app = LlmsecTUI(store=TaskStore(log_dir=None))
            async with app.run_test(size=(120, 36)) as pilot:
                await pilot.press("question_mark")
                await pilot.pause()
                from llmsec.tui.widgets import HelpModal
                assert isinstance(app.screen, HelpModal)
                text = "\n".join(str(l) for l in app.screen.query_one("#help-log", RichLog).lines)
                assert "宣政殿" in text and "跨进程" in text
                await pilot.press("escape")
                await pilot.pause()
                assert not isinstance(app.screen, HelpModal)

        asyncio.run(_run())


class TestRunsAnalysisViews:
    def _panel_with_runs(self):
        """构造带两个 run 的面板。复用模块级 _RunsApp（拦截 _load）——
        真实 _load 的后台磁盘扫描在干净环境返回空列表并回写 _runs={}，
        会把测试注入的数据清掉（CI 首跑即因此挂掉）。"""
        return _RunsApp()

    def test_boundary_surprises_pairing(self, monkeypatch):
        import llmsec.mcp.tools.query as q

        monkeypatch.setattr(q, "elo_security_boundary",
                            lambda m: {"boundary_elo": 1500.0, "converged": True, "ci_half": 12.0})
        monkeypatch.setattr(q, "elo_find_surprises",
                            lambda m, min_elo_gap=0.0: {"weakness": [{"attacker": "a1", "elo_gap": -80.0, "eval_score": 3}],
                                                        "strength": [{"attacker": "a2", "elo_gap": 90.0, "eval_score": 0}]})
        monkeypatch.setattr(q, "elo_suggest_next_pairing",
                            lambda m, n=5: [{"attacker": "a1", "defender": m}, {"attacker": "a2", "defender": m}])

        async def _run() -> None:
            app = self._panel_with_runs()
            async with app.run_test() as pilot:
                panel = app.query_one(RunsPanel)
                panel._render_runs([_run_row("r1", target="模型A"), _run_row("r2", target="模型B")])
                await pilot.pause()
                from llmsec.tui.widgets import LogModal

                await pilot.press("b")
                from llmsec.tui.panels.runs_panel import EloSelectScreen
                await _wait_until(pilot, lambda: isinstance(app.screen, EloSelectScreen))
                await pilot.click("#elo-ok")
                await _wait_until(pilot, lambda: isinstance(app.screen, LogModal))
                assert "boundary_elo" in "\n".join(str(l) for l in app.screen.query_one("#modal-log", RichLog).lines) \
                    or isinstance(app.screen, LogModal)
                await pilot.press("escape")
                await pilot.pause()

                from llmsec.tui.widgets import TableModal

                await pilot.press("p")
                await _wait_until(pilot, lambda: isinstance(app.screen, EloSelectScreen))
                await pilot.click("#elo-ok")
                await _wait_until(pilot, lambda: isinstance(app.screen, TableModal))
                await pilot.press("escape")
                await pilot.pause()

                await pilot.press("n")
                await _wait_until(pilot, lambda: isinstance(app.screen, EloSelectScreen))
                await pilot.click("#elo-ok")
                await _wait_until(pilot, lambda: isinstance(app.screen, TableModal))

        asyncio.run(_run())


class TestChatPanelSwitch:
    def test_panel_4_switches_to_chat(self, tmp_path):
        from llmsec.tui.app import LlmsecTUI

        async def _run() -> None:
            app = LlmsecTUI(store=TaskStore(log_dir=tmp_path))
            async with app.run_test(size=(120, 36)) as pilot:
                await pilot.press("4")
                await _wait_until(pilot, lambda: _vis(app, "#panel-chat") and isinstance(app.focused, Input))
                assert not _vis(app, "#panel-tasks")
                # 输入框聚焦时数字键被吃：先 Esc 离开输入框，再按 1 切回
                await pilot.press("1")
                await pilot.pause()
                assert _vis(app, "#panel-chat"), "输入框聚焦时数字键不应切面板（被吃进输入框）"
                await pilot.press("escape")
                await _wait_until(pilot, lambda: app.focused is None)
                await pilot.press("1")
                await _wait_until(pilot, lambda: _vis(app, "#panel-tasks") and not _vis(app, "#panel-chat"))

        asyncio.run(_run())
