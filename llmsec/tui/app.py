"""llmsec.tui.app — Textual 应用骨架：三面板 + 轮询 + 全局键。

全局键：1/2/3 切面板 · r 立即刷新 · q 退出（延续 web 看板 1-7/r 的快捷键习惯）。
面板只在获得焦点时接收自己的按键（modal 表单内按键被输入框消费，不会误触）。
"""

from __future__ import annotations

import threading

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.message import Message
from textual.widgets import Footer, Header

from llmsec.tui.panels.chat_panel import ChatPanel
from llmsec.tui.panels.hpo_panel import HpoPanel
from llmsec.tui.panels.runs_panel import RunsPanel
from llmsec.tui.panels.tasks_panel import TasksPanel
from llmsec.tui.task_store import TaskSnapshot, TaskStore
from llmsec.tui.widgets import HelpModal

HELP_TEXT = """\
全局键
  1 / 2 / 3 / 4   切面板：任务中心 / HPO 直播 / Runs 浏览 / 宣政殿对话
  r               立即刷新（唤醒 2s 轮询 + 重载 runs）
  ?               本帮助
  q               退出

任务中心（1）/ HPO 直播（2）——共用键
  n               发起评估（仅任务中心）
  s               启动 HPO study（仅 HPO 面板）
  c               取消选中任务（本机直接终止；外部任务经 PID 跨进程强杀）
  l               查看选中任务完整日志
  ↑/↓/Enter      表格导航

Runs 浏览（3）
  enter           读报告（核心指标 + 门下省 findings + 完整 JSON）
  m / v           标记两个 run / 对比已标记
  e / b / p / n   ELO 榜 / 安全边界 / 意外发现 / 下一批测试建议

宣政殿（4）
  回车            发送指令（自然语言或 {"tool": ...,"args": ...} JSON 直调）
  help            查看对话引擎全部可用指令
  Esc             离开输入框——之后 1/2/3/q 等全局键可用（输入框聚焦时会吃键）

外部任务说明：由看板/MCP 启动或 TUI 重启前的任务标「外部」，状态与进度照常
显示（meta.json + progress 直播），带存活 PID 的可跨进程取消。
"""

# 漆夜玄朱：延续 web 端暗色主题（index.html :83-95）
_CSS = """
Screen {
    background: #15120E;
}
Header {
    background: #241F19;
}
Footer {
    background: #241F19;
}
#panel-tasks, #panel-hpo, #panel-runs {
    height: 1fr;
    padding: 0 1;
}
/* 面板显隐只由 .hidden 类管理（初始态在 compose 时给 hpo/runs 挂 hidden）。
   不能再写 "#panel-hpo { display: none }" 之类无条件基础规则——.hidden 移除后
   基础规则仍生效，面板会永远不可见（UX 走查踩过）。 */
.hidden {
    display: none;
}
.panel-head {
    height: 1;
    margin-top: 0;
}
.panel-title {
    color: #D9B45C;
    text-style: bold;
    width: auto;
}
#task-summary, #hpo-summary {
    color: #9A8F76;
    margin-left: 2;
}
#runs-hint {
    color: #9A8F76;
    height: 1;
}
#task-table, #hpo-table, #runs-table {
    height: 1fr;
    margin-top: 0;
}
TermBox {
    border: round #4B4136;
    background: #20242B;
    color: #E7DFC8;
    padding: 0 1;
    height: auto;
    max-height: 20;
    margin-top: 1;
}
DataTable {
    background: #1C1814;
}
/* ---- 模态 ---- */
LaunchScreen, HpoStartScreen, ConfirmScreen, LogModal, TableModal, EloSelectScreen, ReportModal {
    align: center middle;
    background: #15120E 60%;
}
.modal-box {
    width: 72;
    max-width: 92%;
    height: auto;
    max-height: 88%;
    overflow-y: auto;  /* 极矮终端下兜底可滚，而不是把提交按钮裁出屏幕 */
    border: round #D9B45C;
    background: #241F19;
    padding: 1 2;
}
.modal-title {
    color: #D9B45C;
    text-style: bold;
    margin-bottom: 1;
}
.field-label {
    color: #9A8F76;
    margin-top: 1;
}
.field-hint {
    color: #C0492B;
    margin-top: 1;
}
.field-row {
    height: auto;
}
.field-cell {
    width: 1fr;
    height: auto;  /* Vertical 容器默认 1fr，不显式收敛会把一行三格撑到几十行 */
    margin-right: 1;
}
.modal-buttons {
    height: auto;
    align-horizontal: right;
    margin-top: 1;
}
#f-targets {
    max-height: 8;
    border: round #4B4136;
}
#confirm-box {
    width: 60;
}
#confirm-text {
    margin-bottom: 1;
}
#modal-log, #report-log {
    height: 1fr;
    max-height: 30;
    border: round #4B4136;
    background: #20242B;
    color: #E7DFC8;
}
#modal-table {
    height: 1fr;
    max-height: 28;
}
/* ---- 宣政殿 chat ---- */
#panel-chat {
    height: 1fr;
    padding: 0 1;
}
#chat-hint {
    color: #9A8F76;
    height: auto;
    margin-bottom: 1;
}
#chat-log {
    height: 1fr;
    border: round #4B4136;
    background: #20242B;
    color: #E7DFC8;
    padding: 0 1;
}
#chat-input {
    dock: bottom;
    border: round #D9B45C;
}
#help-log {
    height: 1fr;
    max-height: 32;
}
"""


class TasksUpdated(Message):
    """轮询线程 → UI：最新任务快照 + 是否有任务新进入终态。

    定义在模块级（无 namespace）→ handler 名 on_tasks_updated；
    嵌在 App 类里会被 textual 自动加 namespace 导致 handler 失配。
    """

    def __init__(self, snapshots: list[TaskSnapshot], runs_dirty: bool) -> None:
        super().__init__()
        self.snapshots = snapshots
        self.runs_dirty = runs_dirty


class LlmsecTUI(App):
    TITLE = "llmsec · 终端指挥台"
    CSS = _CSS

    BINDINGS = [
        Binding("1", "panel('tasks')", "任务", show=True),
        Binding("2", "panel('hpo')", "HPO", show=True),
        Binding("3", "panel('runs')", "Runs", show=True),
        Binding("4", "panel('chat')", "宣政殿", show=True),
        Binding("question_mark", "help", "帮助", show=True),
        Binding("r", "refresh_all", "刷新", show=True),
        Binding("q", "quit", "退出", show=True),
    ]

    def __init__(self, store: TaskStore | None = None) -> None:
        super().__init__()
        self.store = store or TaskStore()
        self._wake = threading.Event()  # r 键立即唤醒轮询线程
        self._active = "tasks"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield TasksPanel(self.store)
        yield HpoPanel(self.store, classes="hidden")
        yield RunsPanel(classes="hidden")
        yield ChatPanel(classes="hidden")
        yield Footer()

    def on_mount(self) -> None:
        self.run_worker(self._poll_tasks, thread=True, exclusive=True, group="poll")
        self.action_panel("tasks")

    # ============================================================
    # 任务轮询（线程 worker → 消息驱动 UI 更新）
    # ============================================================
    @work(thread=True, exclusive=True, group="poll")
    def _poll_tasks(self) -> None:
        while self.is_running:  # 应用退出时协作结束（worker 线程无法被强杀）
            try:
                snaps, dirty = self.store.refresh()
                self.post_message(TasksUpdated(snaps, dirty))
            except Exception:
                pass  # 轮询永不死——单轮失败下一轮再来
            # 2s 周期，r 键可立即唤醒
            self._wake.wait(2.0)
            self._wake.clear()

    def on_tasks_updated(self, msg: TasksUpdated) -> None:
        self.query_one(TasksPanel).update_tasks(msg.snapshots)
        self.query_one(HpoPanel).update_tasks(msg.snapshots)
        if msg.runs_dirty:
            self.query_one(RunsPanel).flag_reload()

    # ============================================================
    # 面板切换 / 手动刷新
    # ============================================================
    _PANEL_IDS = {"tasks": "#panel-tasks", "hpo": "#panel-hpo", "runs": "#panel-runs", "chat": "#panel-chat"}
    _PANEL_FOCUS = {"tasks": "#task-table", "hpo": "#hpo-table", "runs": "#runs-table", "chat": "#chat-input"}

    def action_panel(self, name: str) -> None:
        for panel_name, selector in self._PANEL_IDS.items():
            node = self.query_one(selector)
            if panel_name == name:
                node.remove_class("hidden")
            else:
                node.add_class("hidden")
        self._active = name
        try:
            self.query_one(self._PANEL_FOCUS[name]).focus()
        except Exception:
            pass
        if name == "runs":
            self.query_one(RunsPanel).show_refresh()

    def action_help(self) -> None:
        self.push_screen(HelpModal())

    def action_refresh_all(self) -> None:
        self._wake.set()
        self.query_one(RunsPanel).show_refresh()
        self.notify("已刷新", timeout=2)


def main() -> None:
    """llmsec-tui 入口：UTF-8 控制台兜底后启动 Textual 应用。"""
    from llmsec.core.logging import setup_console

    setup_console()
    LlmsecTUI().run()


if __name__ == "__main__":
    main()
