"""宣政殿 · 中书省对话面板——自然语言/JSON 指令操作整个控制层。

复用 control.agent.zhongshu.fallback.chat_one（规则版意图引擎：意图解析 →
call_tool → 渲染）。LLM 版对话在看板 POST /api/control/chat（需开服务），
TUI 独立进程走规则版。chat_one 可能触发子进程 CLI（编排/快照类指令），
一律放线程 worker，绝不卡 UI。
"""

from __future__ import annotations

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Input, RichLog, Static

from llmsec.tui.render import C_DIM, C_GOLD

_INTRO = (
    "中书省 · 对话中间者（规则引擎）——输入自然语言或 JSON 指令。\n"
    '  · 自然语言：如「列出最近的 run」「对比 run1 和 run2」\n'
    '  · JSON 直调：{"tool": "list_runs", "args": {}}\n'
    "  · help 查看全部可用指令；q 退出为全局键，对话框内回车发送"
)


class ChatPanel(Vertical):
    # 输入框聚焦时数字/字母键会被吃掉（全局 1/2/3/q 不可用）——Esc 先离开
    # 输入框（blur），全局键恢复；点击/Tab 可回到输入框。
    BINDINGS = [Binding("escape", "blur_input", "离开输入框（恢复全局键）", show=False)]

    def __init__(self, **kwargs) -> None:
        super().__init__(id="panel-chat", **kwargs)
        self._busy = False

    def action_blur_input(self) -> None:
        self.screen.set_focus(None)

    def compose(self) -> ComposeResult:
        yield Static("宣政殿 · 中书省", classes="panel-title")
        yield Static(_INTRO, id="chat-hint")
        yield RichLog(id="chat-log", markup=False, wrap=True, highlight=False)
        yield Input(placeholder="输入指令，回车发送（help 查看可用指令）", id="chat-input")

    def on_mount(self) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.write("中书省已就位。")

    @property
    def input_widget(self) -> Input:
        return self.query_one("#chat-input", Input)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        self.input_widget.value = ""
        self._append("你", text)
        self._ask(text)

    def _append(self, who: str, text: str) -> None:
        from rich.text import Text

        line = Text()
        line.append(f"{who} ❯ ", style=C_GOLD if who == "你" else C_DIM)
        line.append(text)
        self.query_one("#chat-log", RichLog).write(line)

    @work(thread=True, exclusive=False, group="chat")
    def _ask(self, text: str) -> None:
        from control.agent.zhongshu.fallback import _help, chat_one

        if text.lower() in ("help", "?", "帮助"):
            reply = _help()
        else:
            try:
                reply = chat_one(text)
            except Exception as e:
                reply = f"❌ chat_one 异常: {type(e).__name__}: {e}"
        if reply:
            self.app.call_from_thread(self._append, "中书", reply)
