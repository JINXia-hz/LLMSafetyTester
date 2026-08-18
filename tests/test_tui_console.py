"""tests for llmsec.tui.console — 控制台范式 e2e（run_test 离线驱动）。

覆盖：命令执行链（help/eval/kill/rm→confirm//agent）、拼错自动纠错、
Tab 补全、↑ 历史、浮层显隐、top 视图推入/返回。全程无网络：
launch 层与 MCP 动作全部 monkeypatch，绝不触发真实评估。
"""

from __future__ import annotations

import asyncio

import pytest

# 可选依赖：textual 属 [tui] extra，未安装环境整体跳过（沿用 hdbscan 惯例）
pytest.importorskip("textual")

from textual.widgets import Input, RichLog, Static

from llmsec.tui.app import LlmsecTUI
from llmsec.tui.console import CommandInput, ConsoleScreen
from llmsec.tui.task_store import TaskSnapshot, TaskStore
from llmsec.tui.views import TaskLiveScreen

# ============================================================
# 驱动辅助
# ============================================================
from tests.utils import wait_until as _wait_until


def _console_text(app) -> str:
    log = app.query_one("#console", RichLog)
    return "\n".join(getattr(strip, "text", "") for strip in log.lines)


def _input(app) -> CommandInput:
    return app.query_one("#cmd-bar", CommandInput)


async def _submit(app, pilot, line: str) -> None:
    """填入命令行并回车（CJK 直接设值——pilot.press 只可靠支持单字节键）。

    程序化设值后 caret 可能停在 0 且 Changed 滞后一拍，显式置行尾光标 +
    同步刷新补全态，等价于真人敲完整行后的状态（决策路径见 _on_submitted）。"""
    tick = getattr(app, "_inject_tick", None)
    if tick:
        # 有注入：先等补丁后首个轮询 tick 完成（见 _inject），排干在途真实
        # refresh 的空快照消息，命令才不会读到被清空的 _snaps
        await _wait_until(pilot, lambda: bool(tick))
        await pilot.pause()
    inp = _input(app)
    inp.value = line
    inp.cursor_position = len(line)
    app._console._refresh_assist(inp.value, inp.cursor_position)
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()


def _run(fn, *, tmp_path=None, monkeypatch=None):
    """最小宿主：挂 LlmsecTUI（隔离 log_dir + 历史文件）跑 fn(app, pilot)。"""
    store = TaskStore(log_dir=tmp_path) if tmp_path else TaskStore()

    async def _main() -> None:
        app = LlmsecTUI(store=store, warm=False)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            # 历史文件隔离到 tmp（默认写仓库 STATE_DIR，测试不得污染真实历史）
            hist = (tmp_path or store.log_dir) / "hist.txt"
            # 测试注入：历史文件指向 tmp
            app._console._hist_path = lambda: hist
            await fn(app, pilot)

    asyncio.run(_main())


def _snap(sid="evaluate-101010-ab12cd", *, status="running", owned=True, pid=None) -> TaskSnapshot:
    return TaskSnapshot(
        id=sid,
        kind=sid.split("-", 1)[0],
        status=status,
        owned=owned,
        started_at="2026-08-17T10:00:01",
        pid=pid,
        meta={"targets": ["模型A"], "max_rounds": 5},
    )


def _inject(app, monkeypatch, snaps: list[TaskSnapshot]) -> None:
    """注入任务快照并钉住 store.refresh——否则轮询线程 2s 周期会用磁盘扫描的
    空结果覆盖 _snaps，命令读取的时机变成竞态（真实 app 中轮询=磁盘真相，测试
    中注入=真相，二者必须同源）。

    竞态根治：补丁落点可能晚于轮询 worker 在途的一次真实 refresh——它的空
    快照消息会在注入之后才被处理（全套跑时机器慢、窗口更宽，ls -l tasks 曾
    因此间歇超时）。这里记录补丁后首个 tick 标记并 _wake 立即触发下一轮；
    _submit 提交前等该标记——worker 单线程顺序发消息，补丁后的 tick 入队时
    在途空消息必已处理完（先冲掉、再由 tick 恢复注入快照），命令读取
    时机不再靠运气。"""
    app._console.update_snapshots(snaps)
    tick: list[bool] = []

    def _fake_refresh():
        tick.append(True)
        return snaps, False

    monkeypatch.setattr(app.store, "refresh", _fake_refresh)
    app._inject_tick = tick
    app._wake.set()


# ============================================================
# 基础：横幅 / help / 未知命令 / 纠错
# ============================================================
class TestBasics:
    def test_banner(self, tmp_path):
        async def fn(app, pilot):
            await _wait_until(pilot, lambda: "终端指挥台" in _console_text(app))

        _run(fn, tmp_path=tmp_path)

    def test_help_lists_commands(self, tmp_path):
        async def fn(app, pilot):
            await _submit(app, pilot, "help")
            await _wait_until(pilot, lambda: "发起红队评估" in _console_text(app))

        _run(fn, tmp_path=tmp_path)

    def test_help_single_command_usage(self, tmp_path):
        async def fn(app, pilot):
            await _submit(app, pilot, "help eval")
            await _wait_until(pilot, lambda: "--target" in _console_text(app))

        _run(fn, tmp_path=tmp_path)

    def test_unknown_command(self, tmp_path):
        async def fn(app, pilot):
            await _submit(app, pilot, "zzzz")
            await _wait_until(pilot, lambda: "未知命令" in _console_text(app))
            assert "✗" in _console_text(app)

        _run(fn, tmp_path=tmp_path)

    def test_typo_autocorrected(self, tmp_path):
        async def fn(app, pilot):
            await _submit(app, pilot, "lss tasks")
            await _wait_until(pilot, lambda: "已纠错：lss → ls" in _console_text(app))
            await _wait_until(pilot, lambda: "无任务" in _console_text(app))

        _run(fn, tmp_path=tmp_path)

    def test_clear(self, tmp_path):
        async def fn(app, pilot):
            await _submit(app, pilot, "help")
            await _wait_until(pilot, lambda: "发起红队评估" in _console_text(app))
            await _submit(app, pilot, "clear")
            await pilot.pause()
            assert "发起红队评估" not in _console_text(app)

        _run(fn, tmp_path=tmp_path)


# ============================================================
# 补全浮层 / 提示行 / 历史
# ============================================================
class TestAssist:
    def test_popup_appears_while_typing(self, tmp_path):
        async def fn(app, pilot):
            _input(app).value = "l"
            await pilot.pause()
            popup = app.query_one("#cmd-complete", Static)
            assert not popup.has_class("hidden")
            assert "ls" in popup.renderable

        _run(fn, tmp_path=tmp_path)

    def test_popup_hidden_after_submit(self, tmp_path):
        async def fn(app, pilot):
            await _submit(app, pilot, "help")
            await _wait_until(pilot, lambda: "发起红队评估" in _console_text(app))
            assert app.query_one("#cmd-complete", Static).has_class("hidden")

        _run(fn, tmp_path=tmp_path)

    def test_tab_completes_command(self, tmp_path):
        async def fn(app, pilot):
            _input(app).value = "ev"
            await pilot.pause()
            await pilot.press("tab")
            await pilot.pause()
            assert _input(app).value.startswith("eval ")

        _run(fn, tmp_path=tmp_path)

    def test_tab_completes_flag_value(self, tmp_path, monkeypatch):
        async def fn(app, pilot):
            app._console._tgt = lambda: ["glm4", "glm4-air"]  # 注入目标补全源
            _input(app).value = "eval -t glm"
            _input(app).cursor_position = len("eval -t glm")
            await pilot.pause()
            await pilot.press("tab")
            await pilot.pause()
            assert _input(app).value == "eval -t glm4 "

        _run(fn, tmp_path=tmp_path)

    def test_enter_applies_highlighted_completion(self, tmp_path):
        """浮层可见时回车=应用高亮项（fish 语义）：↑↓ 选中 runs → 回车接上，
        再回车才执行。"""

        async def fn(app, pilot):
            _input(app).value = "ls "  # 浮层列出资源候选（tasks/runs/…）
            await pilot.pause()
            await pilot.press("down")  # 高亮第二项 runs
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert _input(app).value == "ls runs ", "回车应接上高亮补全而非执行"
            assert "❯ ls" not in _console_text(app), "第一次回车不应执行"
            # 补全后无新候选（资源位已填）→ 浮层收起，第二次回车执行
            assert app.query_one("#cmd-complete", Static).has_class("hidden")
            await pilot.press("enter")
            await _wait_until(pilot, lambda: "❯ ls runs" in _console_text(app))

        _run(fn, tmp_path=tmp_path)

    def test_enter_executes_when_completion_is_noop(self, tmp_path, monkeypatch):
        """高亮项与已输入相同（应用=无操作）时回车直接执行——不追加重复词。"""

        async def fn(app, pilot):
            app._console._tgt = lambda: ["glm4"]
            monkeypatch.setattr(
                app.store,
                "start_evaluation",
                lambda **kw: {"id": "t-fake", "status": "queued", "meta": {"targets": ["glm4"], "max_rounds": 5}},
            )
            _input(app).value = "eval -t glm4"  # 浮层唯一候选 glm4 == 已输入
            _input(app).cursor_position = len("eval -t glm4")
            await pilot.pause()
            assert not app.query_one("#cmd-complete", Static).has_class("hidden")
            await pilot.press("enter")
            await _wait_until(pilot, lambda: "❯ eval -t glm4" in _console_text(app))
            assert _input(app).value == "", "执行后输入清空"

        _run(fn, tmp_path=tmp_path)

    def test_esc_hides_popup_then_clears(self, tmp_path):
        async def fn(app, pilot):
            _input(app).value = "l"
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert app.query_one("#cmd-complete", Static).has_class("hidden")
            assert _input(app).value == "l"  # 第一次 Esc 只关浮层
            await pilot.press("escape")
            await pilot.pause()
            assert _input(app).value == ""  # 第二次清空

        _run(fn, tmp_path=tmp_path)

    def test_history_up(self, tmp_path):
        async def fn(app, pilot):
            await _submit(app, pilot, "help")
            await _wait_until(pilot, lambda: "发起红队评估" in _console_text(app))
            await pilot.press("up")
            await pilot.pause()
            assert _input(app).value == "help"
            await pilot.press("down")
            await pilot.pause()
            assert _input(app).value == ""  # 回到底稿（空）

        _run(fn, tmp_path=tmp_path)

    def test_hint_line_updates(self, tmp_path):
        async def fn(app, pilot):
            _input(app).value = "zzzz"
            await pilot.pause()
            hint = app.query_one("#cmd-hint", Static).renderable
            assert "未知" in str(hint)
            _input(app).value = "eval"
            await pilot.pause()
            hint = app.query_one("#cmd-hint", Static).renderable
            assert "红队评估" in str(hint)

        _run(fn, tmp_path=tmp_path)


# ============================================================
# eval / hpo —— launch 层 monkeypatch（绝不真跑评估）
# ============================================================
def _patch_launch(monkeypatch, tmp_path, calls):
    from llmsec.server import launch as launch_mod
    from llmsec.server import task_manager

    monkeypatch.setattr(launch_mod, "resolve_attack_file", lambda f: tmp_path / "l1.jsonl")
    monkeypatch.setattr(launch_mod, "check_targets_declared", lambda spec: None)

    def fake_start(kind, argv, **kw):
        calls.append((kind, argv, kw.get("meta")))
        return {
            "id": f"{kind}-999999-zzz",
            "kind": kind,
            "status": "queued",
            "cmd": " ".join(argv),
            "meta": kw.get("meta"),
        }

    monkeypatch.setattr(task_manager, "start_task", fake_start)


class TestEval:
    def test_eval_launches(self, tmp_path, monkeypatch):
        calls: list = []
        _patch_launch(monkeypatch, tmp_path, calls)

        async def fn(app, pilot):
            await _submit(app, pilot, "eval -t glm4 -r 3")
            await _wait_until(pilot, lambda: "已入队" in _console_text(app))
            text = _console_text(app)
            assert "glm4" in text and "3 轮" in text
            assert len(calls) == 1
            kind, argv, meta = calls[0]
            assert kind == "evaluate"
            assert "--target" in argv and "glm4" in argv
            assert meta["targets"] == ["glm4"]

        _run(fn, tmp_path=tmp_path)

    def test_eval_launch_error_reported(self, tmp_path, monkeypatch):
        from llmsec.server import launch as launch_mod
        from llmsec.server.launch import LaunchError

        monkeypatch.setattr(launch_mod, "resolve_attack_file", lambda f: tmp_path / "l1.jsonl")

        def boom(spec):
            raise LaunchError("目标未在 .env TARGETS 中声明: x", reason="undeclared", hint="用 ls targets 查看")

        monkeypatch.setattr(launch_mod, "check_targets_declared", boom)

        async def fn(app, pilot):
            await _submit(app, pilot, "eval -t x")
            await _wait_until(pilot, lambda: "未在 .env TARGETS" in _console_text(app))
            assert "ls targets" in _console_text(app)  # hint 一并展示

        _run(fn, tmp_path=tmp_path)


# ============================================================
# kill —— 本机直杀 / 外部 y/N 确认
# ============================================================
class TestKill:
    def test_kill_unknown(self, tmp_path):
        async def fn(app, pilot):
            await _submit(app, pilot, "kill zzz")
            await _wait_until(pilot, lambda: "找不到任务" in _console_text(app))

        _run(fn, tmp_path=tmp_path)

    def test_kill_local_direct(self, tmp_path, monkeypatch):
        cancelled: list = []

        async def fn(app, pilot):
            _inject(app, monkeypatch, [_snap()])
            monkeypatch.setattr(
                app.store, "cancel", lambda tid: cancelled.append(tid) or {"id": tid, "status": "cancelled"}
            )
            await _submit(app, pilot, "kill ab12")
            await _wait_until(pilot, lambda: "已取消" in _console_text(app))
            assert cancelled == ["evaluate-101010-ab12cd"]

        _run(fn, tmp_path=tmp_path)

    def test_kill_external_requires_y(self, tmp_path, monkeypatch):
        cancelled: list = []

        async def fn(app, pilot):
            _inject(app, monkeypatch, [_snap(status="running", owned=False, pid=4242)])
            monkeypatch.setattr(
                app.store, "cancel", lambda tid: cancelled.append(tid) or {"id": tid, "status": "cancelled"}
            )
            await _submit(app, pilot, "kill ab12")
            await _wait_until(pilot, lambda: "跨进程强杀" in _console_text(app), tries=120)
            assert not cancelled  # 未确认不执行
            await _submit(app, pilot, "y")
            await _wait_until(pilot, lambda: cancelled == ["evaluate-101010-ab12cd"], tries=120)
            await _wait_until(pilot, lambda: "已取消" in _console_text(app))

        _run(fn, tmp_path=tmp_path)

    def test_kill_external_n_cancels(self, tmp_path, monkeypatch):
        cancelled: list = []

        async def fn(app, pilot):
            _inject(app, monkeypatch, [_snap(status="running", owned=False, pid=4242)])
            monkeypatch.setattr(
                app.store, "cancel", lambda tid: cancelled.append(tid) or {"id": tid, "status": "cancelled"}
            )
            await _submit(app, pilot, "kill ab12")
            await _wait_until(pilot, lambda: "跨进程强杀" in _console_text(app), tries=120)
            await _submit(app, pilot, "n")
            await _wait_until(pilot, lambda: "已取消（n）" in _console_text(app) or "已取消" in _console_text(app))
            assert not cancelled

        _run(fn, tmp_path=tmp_path)


# ============================================================
# rm → confirm 两步
# ============================================================
class TestTwoStepWrite:
    def test_rm_preview_then_confirm(self, tmp_path, monkeypatch):
        confirmed: list = []
        monkeypatch.setattr(
            "llmsec.mcp.tools.actions.delete_runs_preview",
            lambda names, delete_r=False: {"confirm_token": "tok1", "runs": names},
        )
        monkeypatch.setattr(
            "llmsec.mcp.tools.actions.delete_runs_confirm",
            lambda token: confirmed.append(token) or {"deleted": ["run_x"]},
        )

        async def fn(app, pilot):
            await _submit(app, pilot, "rm run_x")
            await _wait_until(pilot, lambda: "已预览" in _console_text(app))
            await _wait_until(pilot, lambda: "tok1" in _console_text(app))
            assert "tok1" in app._console._tokens  # noqa: SLF001
            assert not confirmed
            await _submit(app, pilot, "confirm tok1")
            await _wait_until(pilot, lambda: "已执行" in _console_text(app))
            assert confirmed == ["tok1"]

        _run(fn, tmp_path=tmp_path)

    def test_confirm_unknown_token(self, tmp_path):
        async def fn(app, pilot):
            await _submit(app, pilot, "confirm nope")
            await _wait_until(pilot, lambda: "未知 token" in _console_text(app))

        _run(fn, tmp_path=tmp_path)


# ============================================================
# /agent —— 宣政殿（规则引擎 monkeypatch）
# ============================================================
class TestAgent:
    def test_agent_reply(self, tmp_path, monkeypatch):
        monkeypatch.setattr("control.agent.zhongshu.fallback.chat_one", lambda t: f"收到:{t}")

        async def fn(app, pilot):
            await _submit(app, pilot, "/agent 你好")
            await _wait_until(pilot, lambda: "收到:你好" in _console_text(app))
            assert "中书 ❯" in _console_text(app)

        _run(fn, tmp_path=tmp_path)


# ============================================================
# top —— 直播视图推入/返回
# ============================================================
class TestTop:
    def test_top_push_and_back(self, tmp_path):
        async def fn(app, pilot):
            await _submit(app, pilot, "top")
            await _wait_until(pilot, lambda: isinstance(app.screen, TaskLiveScreen))
            await pilot.press("q")
            await _wait_until(pilot, lambda: isinstance(app.screen, ConsoleScreen))

        _run(fn, tmp_path=tmp_path)

    def test_top_with_snapshots_shows_rows(self, tmp_path, monkeypatch):
        async def fn(app, pilot):
            _inject(
                app,
                monkeypatch,
                [
                    _snap("evaluate-101010-ab12cd"),
                    _snap("hpo-111111-aabbcc"),
                ],
            )
            await _submit(app, pilot, "top")
            await _wait_until(pilot, lambda: isinstance(app.screen, TaskLiveScreen))
            app.screen.update_tasks(app.store.refresh()[0])
            # 收敛式断言：补丁生效前在途的真实轮询回调（空快照）可能晚到清表
            # （见 _inject 的竞态说明）——补丁后所有 tick 都带注入快照，终态必为
            # 2 行；慢机（CI Windows）上单次 pause 后直接断言会撞到中间态。
            # 查询异常按 -1 计（防御慢机上瞬时的未挂载窗口），不炸轮询 lambda。
            live = app.screen

            def _rows() -> int:
                try:
                    return live.query_one("#live-table").row_count
                except Exception:
                    return -1

            await _wait_until(pilot, lambda: _rows() == 2, tries=120)

        _run(fn, tmp_path=tmp_path)


# ============================================================
# ls tasks —— 表格渲染（磁盘快照注入）
# ============================================================
class TestLsTasks:
    def test_ls_tasks_renders_table(self, tmp_path, monkeypatch):
        async def fn(app, pilot):
            _inject(app, monkeypatch, [_snap()])
            await _submit(app, pilot, "ls tasks")
            await _wait_until(pilot, lambda: "模型A" in _console_text(app))
            assert "1 任务" in _console_text(app)

        _run(fn, tmp_path=tmp_path)

    def test_ls_tasks_long_includes_meta(self, tmp_path, monkeypatch):
        async def fn(app, pilot):
            _inject(app, monkeypatch, [_snap()])
            await _submit(app, pilot, "ls -l tasks")
            await _wait_until(pilot, lambda: "meta" in _console_text(app))
            assert "targets" in _console_text(app)

        _run(fn, tmp_path=tmp_path)


# ============================================================
# 输入框类型与焦点
# ============================================================
def test_command_input_is_input_subclass():
    assert issubclass(CommandInput, Input)
