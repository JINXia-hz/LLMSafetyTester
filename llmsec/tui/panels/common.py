"""任务表共享逻辑与面板基类（任务中心与 HPO 面板复用）。"""

from __future__ import annotations

from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Static

from llmsec.tui.render import C_DIM, C_GOLD, status_text
from llmsec.tui.task_store import TaskSnapshot, TaskStore
from llmsec.tui.widgets import ConfirmScreen, LogModal, TermBox

_KIND_LABEL = {"evaluate": "评估", "hpo": "HPO"}
_CANCELABLE = ("running", "queued")


def kind_label(kind: str) -> str:
    return _KIND_LABEL.get(kind, kind)


def short_cmd(cmd: str) -> str:
    """从任务命令行提取有辨识度的短摘要（目标名 / yaml 名）。

    仅作 meta 缺席时的兜底（launch 层统一携带 meta 后，常规任务不走这里）。
    """
    if not cmd:
        return ""
    toks = cmd.split()
    for flag in ("--target", "--targets"):
        if flag in toks:
            i = toks.index(flag)
            if i + 1 < len(toks):
                v = toks[i + 1]
                return v.replace(",", "+") if flag == "--targets" else v
    if "llmsec.experiments" in cmd:
        # study.yaml 路径取末段文件名
        last = toks[-1].replace("\\", "/").rsplit("/", 1)[-1]
        return last
    return cmd[:48]


def task_summary(snap: TaskSnapshot) -> str:
    """任务短摘要：优先 launch 层 meta（结构化，无反向解析），兜底 short_cmd。"""
    meta = snap.meta or {}
    if meta.get("targets"):
        return "+".join(meta["targets"])
    if meta.get("study"):
        return str(meta["study"])
    return short_cmd(snap.cmd)


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


class TaskTablePanel(Vertical):
    """任务表面板基类（任务中心 / HPO 直播共用）。

    提供：头部标题+摘要、任务表、TermBox 详情、行选中联动，以及面板级
    动作 c 取消 / l 完整日志——两个面板行为一致（UX 走查发现 HPO 面板
    按这两个键无反馈，收口到基类杜绝再次分叉）。

    子类只声明差异：类属性（标题/表格列/过滤 kind/详情行数）+ 摘要文案 +
    各自的启动入口（n / s）。textual 的 BINDINGS 沿 MRO 合并，子类追加
    自己的键即可。
    """

    BINDINGS = [
        Binding("c", "cancel", "取消任务"),
        Binding("l", "log", "完整日志"),
    ]

    PANEL_TITLE = "任务"
    EMPTY_TITLE = "任务"        # 无选中时 TermBox 标题
    TABLE_ID = "task-table"
    SUMMARY_ID = "task-summary"
    TERM_ID = "task-term"
    CMD_COLUMN = "命令"
    TERM_RECENT = 6
    KIND_FILTER: str | None = None  # None = 不过滤（全部任务）

    def __init__(self, store: TaskStore, **kwargs) -> None:
        super().__init__(**kwargs)
        self.store = store
        self._snaps: dict[str, TaskSnapshot] = {}
        self._selected: str | None = None

    # ---- 布局 ----
    def compose(self) -> ComposeResult:
        with Horizontal(classes="panel-head"):
            yield Static(self.PANEL_TITLE, classes="panel-title")
            yield Static("", id=self.SUMMARY_ID)
        table = DataTable(id=self.TABLE_ID, cursor_type="row", zebra_stripes=True)
        table.add_columns("任务", "状态", "进度", "开始", "来源", self.CMD_COLUMN)
        yield table
        yield TermBox(id=self.TERM_ID, recent=self.TERM_RECENT)

    # ---- 数据更新（App 轮询消息驱动）----
    def visible_snaps(self, snaps: list[TaskSnapshot]) -> list[TaskSnapshot]:
        if self.KIND_FILTER is None:
            return list(snaps)
        return [s for s in snaps if s.kind == self.KIND_FILTER]

    def summary_text(self, shown: list[TaskSnapshot]) -> str:
        return f"{len(shown)} 任务"

    def update_tasks(self, snaps: list[TaskSnapshot]) -> None:
        shown = self.visible_snaps(snaps)
        self._snaps = {s.id: s for s in shown}
        table = self.query_one(f"#{self.TABLE_ID}", DataTable)
        self._selected = refresh_task_table(table, shown, self._selected)
        self.query_one(f"#{self.SUMMARY_ID}", Static).update(self.summary_text(shown))
        self._refresh_term()

    @on(DataTable.RowHighlighted)
    def _row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        # 事件只从本面板自己的表冒泡上来（DOM 祖先唯一），无需校验来源
        if event.row_key is not None and event.row_key.value:
            self._selected = event.row_key.value
            self._refresh_term()

    def _refresh_term(self) -> None:
        snap = self._snaps.get(self._selected or "")
        term = self.query_one(f"#{self.TERM_ID}", TermBox)
        if snap is None:
            term.show(self.EMPTY_TITLE, None)
            return
        title = f"{kind_label(snap.kind)} · {snap.id.split('-')[-1]}"
        summary = task_summary(snap)
        if summary:
            title += f" · {summary}"
        term.show(title, snap.state)

    # ---- 取消 / 日志（两面板共用的面板级动作）----
    def action_cancel(self) -> None:
        snap = self._snaps.get(self._selected or "")
        if snap is None or snap.status not in _CANCELABLE:
            self.app.notify("没有可取消的运行中/排队任务", severity="warning")
            return
        # 外部任务须带存活 PID 才能跨进程强杀（meta.json 提供）
        if not snap.owned and snap.pid is None:
            self.app.notify("外部任务无 PID 信息，无法跨进程取消", severity="warning")
            return
        way = "跨进程强杀（连子进程树）" if not snap.owned else "子进程将被终止"
        self.app.push_screen(
            ConfirmScreen(f"取消任务 {kind_label(snap.kind)} · {snap.id.split('-')[-1]}？\n{way}，已观测结果保留。"),
            self._on_confirm_cancel,
        )

    def _on_confirm_cancel(self, ok: bool | None) -> None:
        if ok and self._selected:
            self._cancel(self._selected)

    @work(thread=True, exclusive=True, group="cancel")
    def _cancel(self, task_id: str) -> None:
        view = self.store.cancel(task_id)
        if view.get("error") is not None:
            self.app.call_from_thread(self.app.notify, view["error"], title="取消失败", severity="error")
            return
        self.app.call_from_thread(self.app.notify, f"已取消 {task_id}", title="任务取消", timeout=5)

    def action_log(self) -> None:
        snap = self._snaps.get(self._selected or "")
        if snap is None:
            self.app.notify("先选中一个任务", severity="warning")
            return
        self._log(snap)

    @work(thread=True, exclusive=True, group="tasklog")
    def _log(self, snap: TaskSnapshot) -> None:
        text = self.store.full_log(snap.id)
        title = f"日志 · {kind_label(snap.kind)} · {snap.id}"
        self.app.call_from_thread(self.app.push_screen, LogModal(title, text))
