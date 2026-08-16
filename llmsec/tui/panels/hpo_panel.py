"""HPO 搜索直播面板：hpo 任务表 + 选中任务的放大直播视图。

面板键：s 启动 study · c 取消选中 · l 完整日志（c/l 与任务中心同行为，
实现在 TaskTablePanel 基类——UX 走查发现只在任务中心挂键导致 HPO 面板
按键无反馈）。study 启动选现有 yaml，与看板 POST /api/run/hpo 同一链路；
因子选择表单是 web 端强项，TUI v1 从简——直接跑已有 study 配置。
"""

from __future__ import annotations

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static

from llmsec.tui.panels.common import TaskTablePanel
from llmsec.tui.task_store import TaskSnapshot, TaskStore, study_yamls


class HpoPanel(TaskTablePanel):
    BINDINGS = [Binding("s", "start", "启动 study")]  # c/l 继承自基类

    PANEL_TITLE = "HPO 直播"
    EMPTY_TITLE = "HPO"
    TABLE_ID = "hpo-table"
    SUMMARY_ID = "hpo-summary"
    TERM_ID = "hpo-term"
    CMD_COLUMN = "study"
    TERM_RECENT = 12
    KIND_FILTER = "hpo"

    def __init__(self, store: TaskStore, **kwargs) -> None:
        super().__init__(store, id="panel-hpo", **kwargs)

    def summary_text(self, shown: list[TaskSnapshot]) -> str:
        running = sum(1 for s in shown if s.status in ("running", "queued", "external"))
        return f"{len(shown)} 个 study · 在跑 {running}"

    # ---- 启动 study ----
    def action_start(self) -> None:
        yamls = study_yamls()
        if not yamls:
            self.app.notify(
                "没找到 study 配置（experiments/ 与 output/experiments/ 下无 yaml）",
                title="启动 HPO",
                severity="warning",
            )
            return
        self.app.push_screen(HpoStartScreen(yamls), self._on_start)

    def _on_start(self, path: str | None) -> None:
        if path:
            self._start(path)

    @work(thread=True, exclusive=True, group="hpo-start")
    def _start(self, path: str) -> None:
        view = self.store.start_hpo(path)
        # task_view 恒含 error=None 键：按值判空（同 tasks_panel._launch 的教训）
        if view.get("error") is not None:
            self.app.call_from_thread(
                self.app.notify, f"{view['error']}\n{view.get('hint', '')}", title="启动失败", severity="error", timeout=8
            )
            return
        self.app.call_from_thread(
            self.app.notify, f"任务 {view.get('id')} 已进入队列", title="HPO 已提交", timeout=5
        )


class HpoStartScreen(ModalScreen[str | None]):
    """选择 study.yaml 启动（下拉选现有配置，或手输仓库内路径）。"""

    BINDINGS = [Binding("escape", "cancel", "取消", show=False)]

    def __init__(self, yamls: list[str], **kwargs) -> None:
        super().__init__(**kwargs)
        self._yamls = yamls

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-box", id="hpo-start-box"):
            yield Static("启动 HPO study", classes="modal-title")
            yield Label("现有配置（experiments/ 与 output/experiments/）", classes="field-label")
            yield Select([(y, y) for y in self._yamls], value=self._yamls[0], allow_blank=False, id="h-yaml")
            yield Label("或手输仓库内路径（优先生效）", classes="field-label")
            yield Input(placeholder="如 experiments/my_study.yaml", id="h-path")
            with Horizontal(classes="modal-buttons"):
                yield Button("启动", variant="primary", id="h-ok")
                yield Button("取消", variant="default", id="h-cancel")

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#h-cancel")
    def _cancel_pressed(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#h-ok")
    def _ok_pressed(self) -> None:
        manual = self.query_one("#h-path", Input).value.strip()
        path = manual or str(self.query_one("#h-yaml", Select).value)
        if not path:
            self.app.notify("先选择或输入 study 配置", severity="warning")
            return
        self.dismiss(path)
