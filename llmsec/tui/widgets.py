"""llmsec.tui.widgets — TUI 复用组件（v4 控制台范式）。

TermBox 对应 web 端 .progress-box（「真·终端窗口」：标题栏三点 + 主体行），
供 top 直播视图使用；LogModal 是 cat 命令的全屏文本查看器（日志/报告）。
v1-v3 面板范式的 ConfirmScreen/TableModal/HelpModal 随交互重构移除——
写操作确认改为 rm/clean 的 preview+confirm 两步，帮助改为 help 命令。
"""

from __future__ import annotations

from rich.console import Group, RenderableType
from rich.text import Text
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import Label, RichLog, Static

from llmsec.tui.render import (
    C_DIM,
    C_GOLD,
    C_SAFE,
    C_WARN,
    EvalProgressState,
    progress_lines,
)


class TermBox(Static):
    """仿终端进度窗口：标题栏（朱/金/石绿三点）+ 主体（progress_lines）。

    active 行尾块光标 ▋ 用 set_interval 翻转相位实现闪烁
    （web 端对应 CSS pgBlink 1.06s step-end）。
    """

    BLINK_INTERVAL = 1.06  # 秒（与 web 端动画周期一致）

    def __init__(self, *, recent: int = 4, **kwargs) -> None:
        super().__init__(**kwargs)
        self._title = "任务"
        self._state: EvalProgressState | None = None
        self._cursor_on = True
        self._recent = recent

    def on_mount(self) -> None:
        self.set_interval(self.BLINK_INTERVAL, self._blink)

    def show(self, title: str, state: EvalProgressState | None) -> None:
        self._title = title
        self._state = state
        self.refresh()

    def _blink(self) -> None:
        # 只在有 active 目标时闪（终态/等待不闪，避免无谓重绘）
        if self._state is not None and (self._state.active_target or self._state.kind == "hpo"):
            self._cursor_on = not self._cursor_on
            self.refresh()

    def render(self) -> RenderableType:
        header = Text()
        header.append("● ", style=C_WARN)
        header.append("● ", style=C_GOLD)
        header.append("● ", style=C_SAFE)
        header.append(f"  {self._title}", style=f"dim {C_DIM}")
        if self._state is None:
            body = [Text("（无任务进度）", style=f"dim {C_DIM}")]
        else:
            body = progress_lines(self._state, cursor_on=self._cursor_on, recent=self._recent)
        return Group(header, Text(), *body)


class LogModal(ModalScreen[None]):
    """只读长文本查看（cat tasks/<id> 的完整日志 / cat runs/<名> 的报告）。"""

    BINDINGS = [
        Binding("escape", "dismiss_none", "关闭", show=False),
        Binding("q", "dismiss_none", "关闭", show=False),
    ]

    def __init__(self, title: str, text: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._title = title
        self._text = text

    def compose(self):
        yield Label(self._title, classes="modal-title")
        yield RichLog(id="modal-log", wrap=False, highlight=False, markup=False)

    def on_mount(self) -> None:
        log = self.query_one("#modal-log", RichLog)
        log.write(self._text or "（空）")

    def action_dismiss_none(self) -> None:
        self.dismiss(None)
