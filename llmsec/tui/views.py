"""llmsec.tui.views — top 命令唤起的任务直播全屏视图。

top / top hpo / top <id前缀> 推入本屏（任务表 + TermBox 进度窗，App 轮询驱动
持续刷新），q/Esc 弹回控制台。enter 打开选中任务完整日志（LogModal）。
这是 v4 控制台范式下唯一的常驻可视化入口——看完即走，不留常驻区域。
"""

from __future__ import annotations

from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import DataTable, Static

from llmsec.tui.render import C_DIM, C_GOLD, status_text, themed_css
from llmsec.tui.task_store import TaskSnapshot, TaskStore, kind_label, task_summary
from llmsec.tui.widgets import LogModal, TermBox


def refresh_task_table(
    table: DataTable,
    snaps: list[TaskSnapshot],
    selected: str | None,
) -> str | None:
    """重建任务表并尽量保持选中行。返回恢复后的 selected id。

    2s 轮询全量重建（≤ 84 行 + 状态着色，成本可忽略），cursor 通过
    move_cursor 回跳——会触发 RowHighlighted，调用方据此刷新详情区。
    """
    table.clear()
    for s in snaps:
        pct = s.state.overall_pct() if s.state else None
        table.add_row(
            f"{kind_label(s.kind)}·{s.id.split('-')[-1]}",
            status_text(s.status),
            Text(f"{pct}%", style=C_GOLD) if pct is not None else Text("—", style=C_DIM),
            (s.started_at or "")[11:19],
            "本机" if s.owned else "外部",
            task_summary(s),
            key=s.id,
        )
    ids = [s.id for s in snaps]
    sel = selected if selected in ids else (ids[0] if ids else None)
    if sel is not None:
        table.move_cursor(row=ids.index(sel), animate=False)
    return sel


class TaskLiveScreen(Screen):
    """任务直播视图（top 唤起）。kind 过滤（"hpo" 只看 HPO），focus_id 直达选中。"""

    CSS = themed_css("""
    #live-head { height: 1; }
    .panel-title { color: $AZURE; text-style: bold; width: auto; }
    #live-summary { color: $DIM; margin-left: 2; }
    #live-table { height: 1fr; margin-top: 1; }
    """)

    BINDINGS = [
        Binding("q", "close", "返回控制台"),
        Binding("escape", "close", "返回", show=False),
    ]

    def __init__(self, store: TaskStore, *, kind: str | None = None, focus_id: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.store = store
        self._kind = kind
        self._focus_id = focus_id
        self._snaps: dict[str, TaskSnapshot] = {}
        self._selected: str | None = focus_id

    def compose(self) -> ComposeResult:
        title = "任务直播" + (f" · {kind_label(self._kind)}" if self._kind else "")
        with Horizontal(id="live-head"):
            yield Static(title, classes="panel-title")
            yield Static("", id="live-summary")
        table = DataTable(id="live-table", cursor_type="row", zebra_stripes=True)
        table.add_columns("任务", "状态", "进度", "开始", "来源", "命令")
        yield table
        yield TermBox(id="live-term", recent=12 if self._kind == "hpo" else 6)

    def on_mount(self) -> None:
        self.query_one("#live-table", DataTable).focus()

    # ---- 数据更新（App 轮询消息驱动）----
    def update_tasks(self, snaps: list[TaskSnapshot]) -> None:
        shown = [s for s in snaps if s.kind == self._kind] if self._kind else list(snaps)
        self._snaps = {s.id: s for s in shown}
        table = self.query_one("#live-table", DataTable)
        self._selected = refresh_task_table(table, shown, self._selected)
        running = sum(1 for s in shown if s.status == "running")
        queued = sum(1 for s in shown if s.status == "queued")
        external = sum(1 for s in shown if not s.owned)
        self.query_one("#live-summary", Static).update(
            f"{len(shown)} 任务 · 运行 {running} · 排队 {queued} · 外部 {external}"
        )
        self._refresh_term()

    @on(DataTable.RowHighlighted, "#live-table")
    def _row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is not None and event.row_key.value:
            self._selected = event.row_key.value
            self._refresh_term()

    @on(DataTable.RowSelected, "#live-table")
    def _row_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key is not None and event.row_key.value:
            self._open_log(str(event.row_key.value))

    def _refresh_term(self) -> None:
        snap = self._snaps.get(self._selected or "")
        term = self.query_one("#live-term", TermBox)
        if snap is None:
            term.show("任务", None)
            return
        title = f"{kind_label(snap.kind)} · {snap.id.split('-')[-1]}"
        summary = task_summary(snap)
        if summary:
            title += f" · {summary}"
        term.show(title, snap.state)

    @work(thread=True, exclusive=True, group="livelog")
    def _open_log(self, task_id: str) -> None:
        text = self.store.full_log(task_id)
        self.app.call_from_thread(self.app.push_screen, LogModal(f"日志 · {task_id}", text or "（空日志）"))

    def action_close(self) -> None:
        self.app.pop_screen()
