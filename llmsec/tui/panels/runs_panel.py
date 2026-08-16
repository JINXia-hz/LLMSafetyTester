"""Runs 浏览与 ELO 面板。

面板键：enter 看报告 · m 标记对比（≤2 个）· v 对比已标记 · e ELO 榜。
数据全部走 MCP 查询工具（纯读磁盘，线程 worker 调用）。
"""

from __future__ import annotations

import json

from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Label, RichLog, Select, Static

from llmsec.tui.render import C_DIM, C_GOLD, C_SAFE, C_WARN
from llmsec.tui.widgets import LogModal, TableModal

_LEVEL_COLOR = {
    "safe": C_SAFE,
    "allergic": C_GOLD,
    "vulnerable": C_WARN,
    "broken": C_WARN,
    "inconclusive": C_DIM,
}
# 已知 0-1 比例字段 → 百分数展示；其余数值定点展示
_RATIO_METRICS = ("asr", "fpr", "coverage")
_METRIC_LABEL = {
    "asr": "ASR", "fpr": "FPR", "boundary_elo": "边界 Elo",
    "boundary_confidence": "边界置信度", "coverage": "覆盖率", "conv_rounds": "收敛轮数",
    "ci_half": "CI 半宽", "total_methods": "方法总数", "methods_above_boundary": "超边界方法数",
}


def _fmt_num(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "✓" if v else "✗"
    if isinstance(v, (int, float)):
        s = f"{float(v):.2f}".rstrip("0").rstrip(".")
        return s or "0"
    return str(v)


def _fmt_ratio(v) -> str:
    if isinstance(v, (int, float)) and 0 <= float(v) <= 1:
        return f"{float(v) * 100:.1f}%"
    return _fmt_num(v)


class RunsPanel(Vertical):
    BINDINGS = [
        Binding("m", "mark", "标记对比"),
        Binding("v", "compare", "对比已标记"),
        Binding("e", "elo", "ELO 榜"),
        Binding("b", "boundary", "安全边界"),
        Binding("p", "surprises", "意外发现"),
        Binding("n", "pairing", "下一对建议"),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(id="panel-runs", **kwargs)
        self._runs: dict[str, dict] = {}
        self._marked: list[str] = []
        self._dirty = True

    def compose(self) -> ComposeResult:
        yield Static("Runs 浏览与 ELO", classes="panel-title")
        yield Static("", id="runs-hint")
        table = DataTable(id="runs-table", cursor_type="row", zebra_stripes=True)
        table.add_columns("run", "目标", "等级", "ASR", "边界Elo", "报告", "修改时间")
        yield table

    def on_mount(self) -> None:
        self._update_hint()
        self._load()

    # ---- 数据加载 ----
    @work(thread=True, exclusive=True, group="runs-load")
    def _load(self) -> None:
        from llmsec.mcp.tools.query import list_runs

        try:
            runs = list_runs()
        except Exception:
            runs = []
        if not isinstance(runs, list):
            runs = []
        self._dirty = False
        self.app.call_from_thread(self._render_runs, runs)

    def _render_runs(self, runs: list[dict]) -> None:
        self._runs = {r["name"]: r for r in runs if isinstance(r, dict) and r.get("name")}
        self._marked = [m for m in self._marked if m in self._runs]
        table = self.query_one("#runs-table", DataTable)
        table.clear()
        for r in runs:
            if not isinstance(r, dict) or not r.get("name"):
                continue
            name = r["name"]
            level = r.get("security_level") or "—"
            table.add_row(
                ("★ " if name in self._marked else "") + name,
                r.get("target") or r.get("target_model") or "—",
                Text(level, style=_LEVEL_COLOR.get(level, C_DIM)),
                _fmt_ratio(r.get("asr")),
                _fmt_num(r.get("boundary_elo")),
                "✓" if r.get("has_report") else "✗",
                (r.get("mtime") or "")[:16].replace("T", " "),
                key=name,
            )
        self._update_hint()

    def _update_hint(self) -> None:
        marked = " vs ".join(self._marked) if self._marked else "无"
        self.query_one("#runs-hint", Static).update(
            f"enter 报告 · m 标记对比（{marked}）· v 对比 · e ELO 榜 · b 边界 · p 意外 · n 下对    共 {len(self._runs)} 个 run"
        )

    def flag_reload(self) -> None:
        """任务终态后数据已变；面板可见则立即重载，否则标记待切回时重载。"""
        if self.has_class("hidden"):
            self._dirty = True
        else:
            self._load()

    def show_refresh(self) -> None:
        """切回本面板时由 App 调用：有待重载标记则刷新。"""
        if self._dirty:
            self._load()

    # ---- 报告 ----
    @on(DataTable.RowSelected, "#runs-table")
    def _row_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key is not None and event.row_key.value:
            self._report(str(event.row_key.value))

    @work(thread=True, exclusive=True, group="runs-report")
    def _report(self, name: str) -> None:
        from llmsec.mcp.tools.query import assess_run_findings, read_run_report

        data = read_run_report(name)
        if not isinstance(data, dict) or "error" in data or not data.get("report"):
            err = data.get("error", "run 不存在或无报告") if isinstance(data, dict) else "读取失败"
            self.app.call_from_thread(self.app.notify, str(err), title="读取报告失败", severity="error")
            return
        findings = assess_run_findings(name)
        if isinstance(findings, dict):
            data = {**data, "findings": findings.get("findings")}
        text = _render_report(name, data)
        self.app.call_from_thread(self.app.push_screen, ReportModal(name, text))

    # ---- 标记与对比 ----
    def _cursor_run(self) -> str | None:
        table = self.query_one("#runs-table", DataTable)
        try:
            row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        except Exception:
            return None
        return row_key.value if row_key is not None else None

    def action_mark(self) -> None:
        name = self._cursor_run()
        if not name:
            return
        if name in self._marked:
            self._marked.remove(name)
        else:
            if len(self._marked) >= 2:
                self._marked.pop(0)
            self._marked.append(name)
        self._render_runs(list(self._runs.values()))
        # 重建后光标回跳到刚才操作的行
        table = self.query_one("#runs-table", DataTable)
        ids = list(self._runs)
        if name in ids:
            table.move_cursor(row=ids.index(name), animate=False)

    def action_compare(self) -> None:
        if len(self._marked) != 2:
            self.app.notify(f"需先标记 2 个 run（当前 {len(self._marked)} 个，用 m 标记）", severity="warning")
            return
        self._compare(list(self._marked))

    @work(thread=True, exclusive=True, group="runs-compare")
    def _compare(self, names: list[str]) -> None:
        from llmsec.mcp.tools.query import compare_runs

        data = compare_runs(names)
        if not isinstance(data, dict) or "error" in data:
            err = data.get("error", "对比失败") if isinstance(data, dict) else "对比失败"
            self.app.call_from_thread(self.app.notify, str(err), title="对比失败", severity="error")
            return
        rows = []
        for metric, values in (data.get("metrics") or {}).items():
            label = _METRIC_LABEL.get(metric, metric)
            fmt = _fmt_ratio if metric in _RATIO_METRICS else _fmt_num
            rows.append([label] + [fmt(v) for v in values.values()])
        if data.get("missing"):
            rows.append(["缺失 run"] + [str(m) for m in data["missing"]])
        cols = ["指标"] + [n.split("/")[-1] for n in names]
        self.app.call_from_thread(
            self.app.push_screen, TableModal(f"对比 · {' vs '.join(names)}", cols, rows)
        )

    # ---- ELO 与分析视图（e/b/p/n 都先选目标模型）----
    def _pick_model(self, cb) -> None:
        models = sorted({r.get("target") or r.get("target_model") for r in self._runs.values()} - {None})
        if not models:
            self.app.notify("暂无 run 数据（先跑一次评估）", severity="warning")
            return
        self.app.push_screen(EloSelectScreen(models), lambda m: cb(m) if m else None)

    def action_elo(self) -> None:
        self._pick_model(self._elo)

    def action_boundary(self) -> None:
        self._pick_model(self._boundary)

    def action_surprises(self) -> None:
        self._pick_model(self._surprises)

    def action_pairing(self) -> None:
        self._pick_model(self._pairing)

    @work(thread=True, exclusive=True, group="runs-elo")
    def _elo(self, model: str) -> None:
        from llmsec.mcp.tools.query import elo_ranking

        data = elo_ranking(model)
        if not isinstance(data, list):
            self.app.call_from_thread(self.app.notify, "ELO 派生失败（R 矩阵无该模型列？）", severity="error")
            return
        rows = [
            [str(r.get("attacker", "?")), _fmt_num(r.get("elo")), _fmt_num(r.get("played")), _fmt_num(r.get("predicted"))]
            for r in data
            if isinstance(r, dict)
        ]
        self.app.call_from_thread(
            self.app.push_screen,
            TableModal(f"攻击方 Elo 榜 · {model}（高 Elo = 强攻击）", ["攻击方法", "Elo", "场次", "预测Elo"], rows),
        )

    @work(thread=True, exclusive=True, group="runs-analysis")
    def _boundary(self, model: str) -> None:
        """安全边界：boundary_elo / 收敛 / 置信度 / CI / 边界上下方法数。"""
        from llmsec.mcp.tools.query import elo_security_boundary

        data = elo_security_boundary(model)
        if not isinstance(data, dict) or data.get("error") or not data:
            self.app.call_from_thread(self.app.notify, "派生失败（R 矩阵无该模型列？）", severity="error")
            return
        lines = [f"== {model} · 安全边界 =="]
        order = ["boundary_elo", "converged", "confidence", "ci_half",
                 "methods_above_boundary", "methods_below_boundary", "total_methods"]
        for k in order + sorted(set(data) - set(order) - {"error"}):
            if k in data:
                v = data[k]
                if isinstance(v, bool):
                    v = "✓" if v else "✗"
                lines.append(f"{k}: {v}")
        self.app.call_from_thread(
            self.app.push_screen, LogModal(f"安全边界 · {model}", "\n".join(lines)))

    @work(thread=True, exclusive=True, group="runs-analysis")
    def _surprises(self, model: str) -> None:
        """双向意外：weakness=低 Elo 攻击得手（短板）/ strength=高 Elo 攻击失手（强项）。"""
        from llmsec.mcp.tools.query import elo_find_surprises

        data = elo_find_surprises(model)
        rows = []
        if isinstance(data, dict) and not data.get("error"):
            for kind, label in (("weakness", "短板"), ("strength", "强项")):
                for r in data.get(kind) or []:
                    if isinstance(r, dict):
                        rows.append([label, str(r.get("attacker", "?")), _fmt_num(r.get("elo_gap")),
                                     _fmt_num(r.get("eval_score"))])
        if not rows:
            self.app.call_from_thread(self.app.notify, "无意外事件（或派生失败）", severity="warning")
            return
        self.app.call_from_thread(
            self.app.push_screen,
            TableModal(f"意外发现 · {model}（短板=低Elo得手 / 强项=高Elo失手）",
                       ["类型", "攻击方法", "Elo差", "评分"], rows),
        )

    @work(thread=True, exclusive=True, group="runs-analysis")
    def _pairing(self, model: str) -> None:
        """下一批测试建议：|攻Elo - 防Elo| 最小的配对（不确定性最大，测试获益最高）。"""
        from llmsec.mcp.tools.query import elo_suggest_next_pairing

        data = elo_suggest_next_pairing(model, n=8)
        rows = [
            [str(r.get("attacker", "?")), str(r.get("defender", model))]
            for r in data
            if isinstance(r, dict)
        ] if isinstance(data, list) else []
        if not rows:
            self.app.call_from_thread(self.app.notify, "无配对建议（或派生失败）", severity="warning")
            return
        self.app.call_from_thread(
            self.app.push_screen,
            TableModal(f"下一批测试建议 · {model}（Elo 差距最小 = 最值得测）", ["攻击方法", "防御方"], rows),
        )


class EloSelectScreen(ModalScreen[str | None]):
    """选目标模型查看 ELO 榜。"""

    BINDINGS = [Binding("escape", "cancel", "取消", show=False)]

    def __init__(self, models: list[str], **kwargs) -> None:
        super().__init__(**kwargs)
        self._models = models

    def compose(self):
        yield Static("选择目标模型", classes="modal-title")
        yield Select([(m, m) for m in self._models], value=self._models[0], allow_blank=False, id="elo-model")
        with Horizontal(classes="modal-buttons"):
            yield Button("查看", variant="primary", id="elo-ok")
            yield Button("取消", variant="default", id="elo-cancel")

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#elo-cancel")
    def _cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#elo-ok")
    def _ok(self) -> None:
        self.dismiss(str(self.query_one("#elo-model", Select).value))


class ReportModal(ModalScreen[None]):
    """run 报告查看（核心指标 + 门下省 findings + 完整 JSON）。"""

    BINDINGS = [Binding("escape", "close", "关闭", show=False)]

    def __init__(self, name: str, text: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._name = name
        self._text = text

    def compose(self):
        yield Label(f"报告 · {self._name}", classes="modal-title")
        yield RichLog(id="report-log", wrap=True, highlight=False, markup=False)

    def on_mount(self) -> None:
        self.query_one("#report-log").write(self._text or "（空）")

    def action_close(self) -> None:
        self.dismiss(None)


def _render_report(name: str, data: dict) -> str:
    """报告 dict → 可读文本：核心指标 + findings（如可得）+ 完整 JSON。"""
    rep = data.get("report") or {}
    lines = [f"== {name} =="]
    for k in ("target_model", "security_level", "generated_at"):
        if rep.get(k) is not None:
            lines.append(f"{k}: {rep[k]}")
    attack = rep.get("attack_phase") or {}
    elo = rep.get("elo") or {}
    if attack.get("asr") is not None:
        lines.append(f"ASR: {_fmt_ratio(attack.get('asr'))}")
    if elo.get("boundary_elo") is not None:
        lines.append(f"边界 Elo: {_fmt_num(elo.get('boundary_elo'))}  CI±{_fmt_num(elo.get('ci_half'))}")
    findings = data.get("findings")
    if isinstance(findings, list) and findings:
        lines.append("")
        lines.append("-- 门下省审查发现 --")
        for f in findings:
            if isinstance(f, dict):
                lines.append(f"[{f.get('severity', '?')}] {f.get('metric')}: {f.get('value')} (阈值 {f.get('threshold')})")
                if f.get("interpretation"):
                    lines.append(f"    {f['interpretation']}")
    lines.append("")
    lines.append("-- 完整报告 JSON --")
    lines.append(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    return "\n".join(lines)
