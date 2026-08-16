"""任务运行中心面板：全部任务表 + 选中任务的终端进度窗 + 发起评估。

面板键（焦点在本面板内时）：n 发起评估 · c 取消选中 · l 完整日志
（c/l 与 HPO 面板同行为，实现在 TaskTablePanel 基类）。
"""

from __future__ import annotations

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    Input,
    Label,
    Select,
    SelectionList,
    Static,
)
from textual.widgets.selection_list import Selection

from llmsec.tui.panels.common import TaskTablePanel
from llmsec.tui.task_store import TaskSnapshot, TaskStore, attack_files

_SAMPLERS = ("hybrid", "gap", "infogain", "coordinate")


class TasksPanel(TaskTablePanel):
    BINDINGS = [Binding("n", "new_eval", "发起评估")]  # c/l 继承自基类

    PANEL_TITLE = "任务中心"
    EMPTY_TITLE = "任务"
    TABLE_ID = "task-table"
    SUMMARY_ID = "task-summary"
    TERM_ID = "task-term"
    CMD_COLUMN = "命令"
    TERM_RECENT = 6

    def __init__(self, store: TaskStore, **kwargs) -> None:
        super().__init__(store, id="panel-tasks", **kwargs)

    def summary_text(self, shown: list[TaskSnapshot]) -> str:
        running = sum(1 for s in shown if s.status == "running")
        queued = sum(1 for s in shown if s.status == "queued")
        external = sum(1 for s in shown if not s.owned)
        return f"{len(shown)} 任务 · 运行 {running} · 排队 {queued} · 外部 {external}"

    # ---- 发起评估 ----
    def action_new_eval(self) -> None:
        targets = _load_target_names()
        attacks = attack_files() or ["l1.jsonl"]
        self.app.push_screen(LaunchScreen(targets, attacks), self._on_launch)

    def _on_launch(self, params: dict | None) -> None:
        if params:
            self._launch(params)

    @work(thread=True, exclusive=True, group="launch")
    def _launch(self, params: dict) -> None:
        view = self.store.start_evaluation(**params)
        # task_view 恒含 error=None 键：必须按值判空，不能用 "error" in view（键恒在）
        if view.get("error") is not None:
            self.app.call_from_thread(
                self.app.notify, f"{view['error']}\n{view.get('hint', '')}", title="启动失败", severity="error", timeout=8
            )
            return
        self.app.call_from_thread(
            self.app.notify, f"任务 {view.get('id')} 已进入队列", title="评估已提交", timeout=5
        )


def _load_target_names() -> list[str]:
    """读 .env 声明的目标模型名（毫秒级同步调用；失败返回空表让用户手输）。"""
    try:
        from llmsec.mcp.tools.query import list_targets

        data = list_targets()
        if isinstance(data, list):
            return [t["name"] for t in data if isinstance(t, dict) and t.get("name")]
    except Exception:
        pass
    return []


def _env_snapshot_names() -> list[str]:
    """env 快照名列表（隔离评估用）；失败返回空表。"""
    try:
        from llmsec.mcp.tools.actions import list_env_snapshots

        data = list_env_snapshots()
        if isinstance(data, list):
            return [s["name"] for s in data if isinstance(s, dict) and s.get("name")]
    except Exception:
        pass
    return []


def _parse_param_overrides(raw: str) -> dict | None:
    """「KEY=V,KEY2=V2」→ dict；任一段无 = 返回 None（表单校验用）。"""
    out: dict = {}
    for seg in raw.split(","):
        seg = seg.strip()
        if not seg:
            continue
        if "=" not in seg:
            return None
        k, v = seg.split("=", 1)
        k = k.strip()
        if not k:
            return None
        out[k] = v.strip()
    return out or None


class LaunchScreen(ModalScreen[dict | None]):
    """发起评估的模态表单。字段对齐 LaunchSpec 常用参数（归一层全能力面：
    含 env 快照隔离 / 参数覆写 / 采样器超参）。"""

    BINDINGS = [Binding("escape", "cancel", "取消", show=False)]

    def __init__(self, targets: list[str], attacks: list[str], snapshots: list[str] | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._targets = targets
        self._attacks = attacks
        self._snapshots = snapshots if snapshots is not None else _env_snapshot_names()

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-box", id="launch-box"):
            yield Static("发起评估", classes="modal-title")
            yield Label("目标模型（可多选）", classes="field-label")
            if self._targets:
                yield SelectionList(
                    *[Selection(n, n, True) for n in self._targets], id="f-targets"
                )
            else:
                yield Static("（.env 未声明 TARGETS——请先配置或用 llmsec-manage 检查）", classes="field-hint")
            yield Label("攻击集", classes="field-label")
            yield Select(
                [(a, a) for a in self._attacks],
                value=self._attacks[0] if self._attacks else Select.BLANK,
                allow_blank=False,
                id="f-input",
            )
            yield Label("采样策略", classes="field-label")
            yield Select([(s, s) for s in _SAMPLERS], value="hybrid", allow_blank=False, id="f-sampler")
            with Horizontal(classes="field-row"):
                yield Vertical(Label("最大轮数", classes="field-label"), Input(value="5", id="f-rounds"), classes="field-cell")
                yield Vertical(Label("种子（可空）", classes="field-label"), Input(value="", id="f-seed"), classes="field-cell")
                yield Vertical(Label("批量（可空）", classes="field-label"), Input(value="", id="f-batch"), classes="field-cell")
            yield Label("采样器超参（可空，用 params 默认值）", classes="field-label")
            with Horizontal(classes="field-row"):
                yield Vertical(Label("α", classes="field-label"), Input(value="", id="f-alpha"), classes="field-cell")
                yield Vertical(Label("β", classes="field-label"), Input(value="", id="f-beta"), classes="field-cell")
                yield Vertical(Label("γ", classes="field-label"), Input(value="", id="f-gamma"), classes="field-cell")
                yield Vertical(Label("探索轮", classes="field-label"), Input(value="", id="f-coord"), classes="field-cell")
            yield Label("隔离与覆写（可空）", classes="field-label")
            with Horizontal(classes="field-row"):
                yield Vertical(
                    Label("env 快照", classes="field-label"),
                    Select([("（不使用）", "")] + [(s, s) for s in self._snapshots],
                           value="", allow_blank=False, id="f-envsnap"),
                    classes="field-cell",
                )
                yield Vertical(
                    Label("参数覆写 KEY=V,逗号分隔", classes="field-label"),
                    Input(placeholder="如 K_FACTOR=32,CONV_CI_TARGET=15", id="f-params"),
                    classes="field-cell",
                )
            yield Checkbox("跑满轮数不早停（no_early_stop）", id="f-noearly")
            with Horizontal(classes="modal-buttons"):
                yield Button("提交", variant="primary", id="f-ok")
                yield Button("取消", variant="default", id="f-cancel")

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#f-cancel")
    def _cancel_pressed(self) -> None:
        self.dismiss(None)

    def _targets_selection(self) -> list[str] | None:
        """目标多选结果；无选择框（.env 未声明目标）时返回 None。

        注意 Screen 在当前 textual 版本没有 query_optional，用 query_one + 捕获。
        """
        try:
            widget = self.query_one("#f-targets", SelectionList)
        except Exception:
            return None
        return [str(v) for v in widget.selected]

    @on(Button.Pressed, "#f-ok")
    def _ok_pressed(self) -> None:
        err = self._validate()
        if err:
            self.app.notify(err, title="参数有误", severity="error")
            return
        params: dict = {}
        sel = self._targets_selection() or []
        if len(sel) == 1:
            params["target"] = sel[0]
        elif len(sel) > 1:
            params["targets"] = sel
        params["input_file"] = self.query_one("#f-input", Select).value
        params["sampler"] = self.query_one("#f-sampler", Select).value
        params["max_rounds"] = int(self.query_one("#f-rounds", Input).value or "5")
        seed = self.query_one("#f-seed", Input).value.strip()
        batch = self.query_one("#f-batch", Input).value.strip()
        if seed:
            params["seed"] = int(seed)
        if batch:
            params["batch_size"] = int(batch)
        # 采样器超参（α/β/γ 浮点，探索轮整数）
        for field_id, key, cast in (
            ("#f-alpha", "sampler_alpha", float),
            ("#f-beta", "sampler_beta", float),
            ("#f-gamma", "sampler_gamma", float),
            ("#f-coord", "coordinate_rounds", int),
        ):
            v = self.query_one(field_id, Input).value.strip()
            if v:
                params[key] = cast(v)
        # 隔离与覆写
        env_snap = str(self.query_one("#f-envsnap", Select).value or "")
        if env_snap:
            params["env_snapshot"] = env_snap
        po_raw = self.query_one("#f-params", Input).value.strip()
        if po_raw:
            params["param_overrides"] = _parse_param_overrides(po_raw)
        if self.query_one("#f-noearly", Checkbox).value:
            params["no_early_stop"] = True
        self.dismiss(params)

    def _validate(self) -> str | None:
        sel = self._targets_selection()
        if sel is not None and not sel:
            return "至少选择一个目标模型"
        for field_id, name in (("#f-rounds", "最大轮数"), ("#f-seed", "种子"), ("#f-batch", "批量")):
            err = self._validate_int(field_id, name)
            if err:
                return err
        for field_id, name in (("#f-alpha", "α"), ("#f-beta", "β"), ("#f-gamma", "γ")):
            v = self.query_one(field_id, Input).value.strip()
            if v:
                try:
                    float(v)
                except ValueError:
                    return f"采样器超参 {name} 须为数字：{v!r}"
        err = self._validate_int("#f-coord", "探索轮")
        if err:
            return err
        po_raw = self.query_one("#f-params", Input).value.strip()
        if po_raw and _parse_param_overrides(po_raw) is None:
            return f"参数覆写格式须为 KEY=V 逗号分隔：{po_raw!r}"
        return None

    def _validate_int(self, field_id: str, name: str) -> str | None:
        v = self.query_one(field_id, Input).value.strip()
        if v:
            try:
                int(v)
            except ValueError:
                return f"{name}须为整数：{v!r}"
        return None
