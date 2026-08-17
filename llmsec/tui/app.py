"""llmsec.tui.app — 应用骨架：控制台屏为家 + 轮询 + top 视图路由。

v4 范式：常态是一台 shell 式控制台（console.ConsoleScreen：日志流 + 命令行，
无常驻可视化）；可视化由 top 命令全屏唤起（views.TaskLiveScreen，q/Esc 返回）。
任务轮询（2s，线程 worker → 消息）驱动控制台补全源与直播视图。
"""

from __future__ import annotations

import threading

from textual import work
from textual.app import App
from textual.message import Message

from llmsec.tui.console import ConsoleScreen
from llmsec.tui.task_store import TERMINAL_STATUSES, TaskSnapshot, TaskStore
from llmsec.tui.views import TaskLiveScreen

# 漆夜玄朱：延续 web 端暗色主题（index.html :83-95）。
# 只保留全局组件样式（DataTable/TermBox/LogModal 模态）；控制台样式在
# ConsoleScreen.CSS，直播视图样式在 TaskLiveScreen.CSS。
_CSS = """
Screen {
    background: #15120E;
}
DataTable {
    background: #1C1814;
}
TermBox {
    border: round #4B4136;
    background: #20242B;
    color: #E7DFC8;
    padding: 0 1;
    height: auto;
    max-height: 24;
    margin-top: 1;
}
/* cat 命令的日志/报告查看模态 */
LogModal {
    align: center middle;
    background: #15120E 60%;
}
.modal-box {
    width: 72;
    max-width: 92%;
    height: 1fr;
    max-height: 88%;
    border: round #D9B45C;
    background: #241F19;
    padding: 1 2;
}
.modal-title {
    color: #D9B45C;
    text-style: bold;
    margin-bottom: 1;
}
#modal-log {
    height: 1fr;
    border: round #4B4136;
    background: #20242B;
    color: #E7DFC8;
}
"""


class TasksUpdated(Message):
    """轮询线程 → UI：最新任务快照。

    定义在模块级（无 namespace）→ handler 名 on_tasks_updated；
    嵌在 App 类里会被 textual 自动加 namespace 导致 handler 失配。
    """

    def __init__(self, snapshots: list[TaskSnapshot]) -> None:
        super().__init__()
        self.snapshots = snapshots


class LlmsecTUI(App):
    TITLE = "llmsec · 终端指挥台"
    CSS = _CSS

    def __init__(self, store: TaskStore | None = None, *, warm: bool = True) -> None:
        super().__init__()
        self.store = store or TaskStore()
        self._warm = warm
        self._wake = threading.Event()  # refresh 命令立即唤醒轮询线程
        self._console: ConsoleScreen | None = None
        self._status_seen: dict[str, str] = {}  # 终态 toast 去重（首见不算新终态）

    def get_default_screen(self) -> ConsoleScreen:
        self._console = ConsoleScreen(self.store, warm=self._warm)
        return self._console

    def on_mount(self) -> None:
        self.run_worker(self._poll_tasks, thread=True, exclusive=True, group="poll")

    # ============================================================
    # 任务轮询（线程 worker → 消息驱动 UI 更新）
    # ============================================================
    @work(thread=True, exclusive=True, group="poll")
    def _poll_tasks(self) -> None:
        while self.is_running:  # 应用退出时协作结束（worker 线程无法被强杀）
            try:
                snaps, _dirty = self.store.refresh()
                self.post_message(TasksUpdated(snaps))
            except Exception:
                pass  # 轮询永不死——单轮失败下一轮再来
            self._wake.wait(2.0)
            self._wake.clear()

    def on_tasks_updated(self, msg: TasksUpdated) -> None:
        if self._console is not None:
            self._console.update_snapshots(msg.snapshots)
        if isinstance(self.screen, TaskLiveScreen):
            self.screen.update_tasks(msg.snapshots)
        # 终态 toast：新进入终态的任务提醒一次（首见已终态的旧任务不提示）
        for s in msg.snapshots:
            first = s.id not in self._status_seen
            prev = self._status_seen.get(s.id)
            self._status_seen[s.id] = s.status
            if not first and prev != s.status and s.status in TERMINAL_STATUSES:
                self.notify(f"{s.id} 已结束（{s.status}）· top 查看", timeout=6)

    # ============================================================
    # top 视图路由 / 刷新
    # ============================================================
    def open_top(self, kind: str | None = None, focus: str | None = None) -> None:
        self.push_screen(TaskLiveScreen(self.store, kind=kind, focus_id=focus))

    def action_refresh_all(self) -> None:
        self._wake.set()
        if self._console is not None:
            self._console.refresh_runs()


def main() -> None:
    """llmsec-tui 入口：UTF-8 控制台兜底后启动 Textual 应用。"""
    from llmsec.core.logging import setup_console

    setup_console()
    LlmsecTUI().run()


if __name__ == "__main__":
    main()
