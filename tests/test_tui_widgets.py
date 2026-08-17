"""tests for llmsec.tui.widgets + llmsec.tui.views — TUI 组件层（v4 控制台范式）。

TermBox 的纯渲染逻辑（未挂载直接调 render/_blink，不依赖事件循环）；
LogModal 与 refresh_task_table 用 App.run_test 离线驱动（毫秒级、无网络，
不算 e2e——沿用 asyncio.run 手法）。
"""

from __future__ import annotations

import asyncio

import pytest

# 可选依赖：textual 属 [tui] extra（rich 随其安装），未安装环境整体跳过（沿用 hdbscan 惯例）
pytest.importorskip("textual")

from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import DataTable, RichLog, Static

from llmsec.tui.render import C_DIM, EvalProgressState
from llmsec.tui.task_store import TaskSnapshot, kind_label, short_cmd, task_summary
from llmsec.tui.views import refresh_task_table
from llmsec.tui.widgets import LogModal, TermBox


# ============================================================
# 造数辅助
# ============================================================
def _ev_state(target: str = "模型A", rnd: int = 2, max_rounds: int = 5, **kw) -> EvalProgressState:
    """一个运行中的 evaluate 状态（单目标一条记录）。"""
    st = EvalProgressState("t-1")
    st.apply_record(
        {
            "phase": "attack",
            "target": target,
            "round": rnd,
            "max_rounds": max_rounds,
            "elo": 1500 + rnd,
            "delta": 10.0,
            "ci_half": 30.0,
            "progress_pct": 20 * rnd,
            "converged": False,
            "ts": "2026-08-15T10:00:00",
            **kw,
        }
    )
    st.set_running(True)
    return st


def _hpo_state(n_trials: int = 4, configs_total: int = 4) -> EvalProgressState:
    """n 条 trial 的 hpo 状态（偶数 seed 成功、奇数失败，值递减）。"""
    st = EvalProgressState("h-1")
    for i in range(n_trials):
        st.apply_record(
            {
                "phase": "hpo",
                "trial_done": i + 1,
                "trial_total_est": 10,
                "configs_done": i + 1,
                "configs_total": configs_total,
                "best_metric": 5.0 - i * 0.5,
                "metric_name": "conv_rounds",
                "direction": "minimize",
                "last": {
                    "target": "模型A",
                    "seed": i,
                    "status": "success" if i % 2 == 0 else "failed",
                    "value": 5.0 - i * 0.5,
                    "params": {"K_FACTOR": 16},
                },
            }
        )
    st.set_running(True)
    return st


def _box_plains(box: TermBox) -> list[str]:
    """未挂载直接 render：Group → 各 Text 的 plain（header, 空行, *body）。"""
    return [t.plain for t in box.render().renderables]


class _ModalApp(App):
    """只承载模态 screen 的最小宿主。"""

    def compose(self) -> ComposeResult:
        yield Static("base")


async def _wait_until(pilot, cond, tries: int = 25) -> None:
    """有界等待条件成立（pause 只推进消息队列不保证真实时间，慢机需让出时间片）。"""
    for _ in range(tries):
        if cond():
            return
        await pilot.pause()
        await asyncio.sleep(0.02)
    raise AssertionError(f"_wait_until 超时（{tries} 轮）：条件始终未成立")


# ============================================================
# TermBox — 纯渲染逻辑
# ============================================================
class TestTermBoxRender:
    def test_no_state_placeholder(self):
        lines = _box_plains(TermBox())
        assert lines[0].count("●") == 3  # 标题栏三点
        assert "任务" in lines[0]  # 默认标题
        assert lines[2] == "（无任务进度）"

    def test_show_updates_title_and_evaluate_body(self):
        box = TermBox()
        box.show("评估 · ab12cd", _ev_state())
        plains = _box_plains(box)
        assert "评估 · ab12cd" in plains[0]
        body = "\n".join(plains[2:])
        assert "❯ 模型A" in body
        assert "R2/5" in body
        assert "运行中" in body
        assert "▋" in body  # cursor_on 初值 True → active 行尾块光标

    def test_hpo_state_renders_hpo_lines(self):
        box = TermBox()
        box.show("HPO · aabbcc", _hpo_state(n_trials=3, configs_total=4))
        body = "\n".join(_box_plains(box)[2:])
        assert "config 3/4" in body
        assert "trial 3/10" in body
        assert "✓" in body and "✗" in body  # 成功/失败 trial 各有
        assert "趋势" in body  # ≥2 个成功值 → sparkline

    def test_recent_limits_trial_lines(self):
        box = TermBox(recent=2)
        box.show("HPO", _hpo_state(n_trials=4))
        body = "\n".join(_box_plains(box)[2:])
        assert "s2" in body and "s3" in body  # 只留最近 2 条 trial
        assert "s0" not in body and "s1" not in body

    def test_done_state_no_cursor(self):
        st = _ev_state(rnd=5, converged=True)  # round == max_rounds → done
        assert st.active_target is None
        box = TermBox()
        box.show("评估", st)
        body = "\n".join(_box_plains(box)[2:])
        assert "已收敛" in body
        assert "▋" not in body
        assert "运行中" not in body


class TestTermBoxBlink:
    def test_blink_toggles_cursor_when_active(self):
        box = TermBox()
        box.show("评估", _ev_state())
        assert box._cursor_on is True
        box._blink()
        assert box._cursor_on is False
        assert "▋" not in "\n".join(_box_plains(box)[2:])  # 相位翻转后光标消失

    def test_blink_noop_without_state(self):
        box = TermBox()
        box._blink()
        assert box._cursor_on is True  # 无状态不翻转、不重绘

    def test_blink_noop_when_done(self):
        box = TermBox()
        box.show("评估", _ev_state(rnd=5, converged=True))
        box._blink()
        assert box._cursor_on is True  # 无 active 目标（终态）不闪

    def test_blink_hpo_state_toggles(self):
        box = TermBox()
        box.show("HPO", _hpo_state(n_trials=1))
        box._blink()
        assert box._cursor_on is False  # hpo 直播也闪


# ============================================================
# LogModal — cat 命令的查看器（run_test 离线驱动）
# ============================================================
class TestLogModal:
    def test_writes_text_and_escapes(self):
        results: list = []

        async def _run() -> None:
            app = _ModalApp()
            async with app.run_test() as pilot:
                app.push_screen(LogModal("日志 · t-1", "hello\n世界"), results.append)
                await pilot.pause()
                text = "".join(
                    getattr(strip, "text", "") for strip in app.screen.query_one("#modal-log", RichLog).lines
                )
                assert "hello" in text
                assert "世界" in text
                await pilot.press("escape")
                await _wait_until(pilot, lambda: results)

        asyncio.run(_run())
        assert results == [None]

    def test_q_also_closes(self):
        results: list = []

        async def _run() -> None:
            app = _ModalApp()
            async with app.run_test() as pilot:
                app.push_screen(LogModal("日志 · t-1", "x"), results.append)
                await pilot.pause()
                await pilot.press("q")
                await _wait_until(pilot, lambda: results)

        asyncio.run(_run())
        assert results == [None]

    def test_empty_text_placeholder(self):
        async def _run() -> None:
            app = _ModalApp()
            async with app.run_test() as pilot:
                app.push_screen(LogModal("日志 · t-1", ""), None)
                await pilot.pause()
                text = "".join(
                    getattr(strip, "text", "") for strip in app.screen.query_one("#modal-log", RichLog).lines
                )
                assert "（空）" in text

        asyncio.run(_run())


# ============================================================
# task_store 展示助手 — 纯函数
# ============================================================
class TestKindLabel:
    def test_known_and_unknown(self):
        assert kind_label("evaluate") == "评估"
        assert kind_label("hpo") == "HPO"
        assert kind_label("migrate") == "migrate"  # 未知 kind 原样透传


class TestShortCmd:
    def test_empty(self):
        assert short_cmd("") == ""

    def test_target_flag(self):
        cmd = "python -m llmsec.pipeline.runner evaluate --target 模型A --rounds 5"
        assert short_cmd(cmd) == "模型A"

    def test_targets_flag_commas_to_plus(self):
        assert short_cmd("run --targets A,B,C extra") == "A+B+C"

    def test_experiments_yaml_basename(self):
        cmd = "python -m llmsec.experiments run output/experiments/study.yaml"
        assert short_cmd(cmd) == "study.yaml"

    def test_experiments_windows_path(self):
        cmd = "python -m llmsec.experiments run C:\\repo\\experiments\\s.yaml"
        assert short_cmd(cmd) == "s.yaml"

    def test_fallback_truncated(self):
        cmd = "x" * 60
        assert short_cmd(cmd) == "x" * 48


class TestTaskSummary:
    def test_meta_targets_joined(self):
        snap = TaskSnapshot(id="t", kind="evaluate", status="running", meta={"targets": ["A", "B"]})
        assert task_summary(snap) == "A+B"

    def test_meta_study(self):
        snap = TaskSnapshot(id="t", kind="hpo", status="running", meta={"study": "s.yaml"})
        assert task_summary(snap) == "s.yaml"

    def test_fallback_to_cmd(self):
        snap = TaskSnapshot(id="t", kind="evaluate", status="external", cmd="run --target 模型A", meta=None)
        assert task_summary(snap) == "模型A"

    def test_nothing_available(self):
        assert task_summary(TaskSnapshot(id="t", kind="evaluate", status="external")) == ""


# ============================================================
# refresh_task_table（views）— DataTable 重建 + 选中保持
# ============================================================
class _TableApp(App):
    def compose(self) -> ComposeResult:
        table = DataTable()
        # refresh_task_table 只加行不加列；列由宿主预置（与 TaskLiveScreen 一致）
        table.add_columns("任务", "状态", "进度", "开始", "来源", "命令")
        yield table


def _with_table(fn) -> None:
    """挂载一个 DataTable 并把实例交给 fn（断言直接在 fn 里抛）。"""

    async def _run() -> None:
        app = _TableApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            fn(app.query_one(DataTable))

    asyncio.run(_run())


def _snap(
    sid: str = "evaluate-101010-ab12cd", *, state=None, owned=False, cmd="", meta=None, status="external"
) -> TaskSnapshot:
    return TaskSnapshot(
        id=sid,
        kind=sid.split("-", 1)[0],
        status=status,
        cmd=cmd,
        started_at="2026-08-15T10:00:01",
        owned=owned,
        state=state,
        meta=meta,
    )


class TestRefreshTaskTable:
    def test_rows_content_and_first_selected(self):
        with_pct = _snap(state=_ev_state(progress_pct=40))
        no_state = _snap("evaluate-111111-ff00ff", owned=True, status="running")

        def fn(table: DataTable) -> None:
            sel = refresh_task_table(table, [with_pct, no_state], None)
            assert table.row_count == 2
            assert sel == with_pct.id  # 无选中 → 回落到首行
            row = table.get_row(with_pct.id)
            assert row[0] == "评估·ab12cd"
            assert row[1].plain == "外部"
            assert row[2].plain == "40%"
            assert "D9B45C" in row[2].style  # 进度描金
            assert row[3] == "10:00:01"  # started_at 取 [11:19] 时间段
            assert row[4] == "外部"
            assert row[5] == ""  # 无 meta/cmd → 空摘要
            row2 = table.get_row(no_state.id)
            assert row2[1].plain == "运行中"
            assert row2[2].plain == "—"  # 无进度状态 → 占位符
            assert row2[2].style == C_DIM  # 压暗样式
            assert row2[4] == "本机"

        _with_table(fn)

    def test_selection_preserved_when_still_present(self):
        a = _snap("evaluate-101010-aaaaaa")
        b = _snap("evaluate-111111-bbbbbb")

        def fn(table: DataTable) -> None:
            sel = refresh_task_table(table, [a, b], b.id)
            assert sel == b.id
            assert table.cursor_row == 1  # 光标回跳到 b 所在行

        _with_table(fn)

    def test_stale_selection_falls_back_to_first(self):
        a = _snap("evaluate-101010-aaaaaa")

        def fn(table: DataTable) -> None:
            assert refresh_task_table(table, [a, _snap("evaluate-111111-bbbbbb")], "gone-id") == a.id

        _with_table(fn)

    def test_empty_snaps(self):
        def fn(table: DataTable) -> None:
            assert refresh_task_table(table, [], "whatever") is None
            assert table.row_count == 0

        _with_table(fn)


# ============================================================
# 单元格类型（防把 Text 当 str 比较）
# ============================================================
def test_status_cell_is_rich_text():
    """refresh_task_table 状态/进度列必须是 rich Text（携带样式），名字列是 str。"""

    def fn(table: DataTable) -> None:
        snap = _snap(state=_ev_state())
        refresh_task_table(table, [snap], None)
        row = table.get_row(snap.id)
        assert isinstance(row[0], str)
        assert isinstance(row[1], Text)
        assert isinstance(row[2], Text)
        assert len(row) == 6

    _with_table(fn)
