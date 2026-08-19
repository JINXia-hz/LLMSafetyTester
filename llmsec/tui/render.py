"""llmsec.tui.render — 终端字符渲染层（从 web 端 run-control.js 移植）。

移植对应关系（llmsec/server/static/js/run-control.js）：
  braille_bar          ← _textBar (:760)    盲文进度条
  TargetProgressTracker← _recomputeDisp (:818)  OLS 回归进度平滑
  EvalProgressState    ← progressState/applyProgress/_mergeSnapshot/recomputeEvalState
  target_row           ← renderTargetRow (:868)
  hpo_lines/sparkline  ← renderHpoBox (:901)

全部为纯函数/纯状态类（只依赖 rich，不依赖 textual），可独立单测。
配色为「敦煌暮色」：取敦煌壁画矿物颜料（月白/金箔/石绿/石青/朱砂/赭石），
只做文字配色、不画背景（界面底色交给终端默认），暖金担主线、寒青做对比——
TUI 独立创作，不与 web 端「漆夜玄朱」（index.html）对齐。
"""

from __future__ import annotations

import json
from string import Template

from rich.cells import cell_len
from rich.text import Text

# ---- 配色（「敦煌暮色」文字/语义色：月白为主，石青结构，金仅小面积点缀）----
C_TEXT = "#E9E3D3"  # 月白：主体文字
C_DIM = "#8A8B7E"  # 灰绿：次要文字/idle 行
C_GOLD = "#D9A441"  # 三彩黄：仅提示符/进度填充等点缀（不大面积用）
C_SAFE = "#87B08C"  # 石绿：done 标记/完成态
C_WARN = "#C4553A"  # 朱砂：失败 trial
C_UP = "#9FBE8C"  # 头绿：delta 上涨
C_DOWN = "#CE8F6E"  # 赭红：delta 下跌
C_MUTED = "#9C9484"  # 灰沙：空槽/状态字
C_AZURE = "#5E9AB8"  # 石青（结构色）：标题/表头/命令名/运行态/加载动画

# ---- 结构色（无背景界面：边框是唯一结构手段；RAISED 仅供模态遮底）----
C_RAISED = "#161616"  # 中性黑：cat 模态盒底（无背景会被下层文字透穿，不可用）
C_BORDER = "#43433C"  # 灰绿线：方角边框

STYLE_EMPTY = f"dim {C_MUTED}"  # 盲文空槽（web 端 opacity .4 的等价物）

# Textual CSS 模板变量：app/views 两处 CSS 共用，色值单一事实源。
THEME: dict[str, str] = {
    "RAISED": C_RAISED,
    "BORDER": C_BORDER,
    "TEXT": C_TEXT,
    "DIM": C_DIM,
    "GOLD": C_GOLD,
    "SAFE": C_SAFE,
    "WARN": C_WARN,
    "AZURE": C_AZURE,
}


def themed_css(css: str) -> str:
    """把 CSS 模板里的 $BORDER/$GOLD/... 替换为 THEME 色值。"""
    return Template(css).substitute(THEME)

# 任务状态 → (中文标签, 颜色)。external/ended 为 TUI 特有：外部任务无元数据 /
# 持有进程已退出但无人回写终态（meta.json + PID 探活推断）。
STATUS_LABELS: dict[str, tuple[str, str]] = {
    "running": ("运行中", C_AZURE),
    "queued": ("排队中", C_MUTED),
    "success": ("完成", C_SAFE),
    "failed": ("失败", C_WARN),
    "cancelled": ("已取消", C_MUTED),
    "external": ("外部", C_DIM),
    "ended": ("已结束", C_DIM),
}

_HPO_TRIALS_CAP = 30  # trial 明细封顶（与 web 端一致）


def fmt_num(v, digits: int = 0) -> str | None:
    """数字格式化（对齐 web 端 fmtNum：0 位取整，其余定点小数）。None → None。"""
    if v is None:
        return None
    if digits <= 0:
        return f"{int(round(float(v)))}"
    return f"{float(v):.{digits}f}"


def pad_cells(s: str, width: int) -> str:
    """按显示宽度右补空格到 width 列，尾部恒留一个空格（CJK 名字对齐用——
    JS 的 padEnd 按码元计，终端里必须按显示宽度计）。"""
    cur = cell_len(s)
    if cur < width:
        return s + " " * (width - cur) + " "
    return s + " "


def braille_bar(pct: float | None, width: int = 14) -> Text:
    """盲文进度条 [⣿⣿⣿⣦⣀⣀]：盲文字符等高（不像 █/░ 高低不齐）。
    已填 ⣿ + 过渡 ⣦（frac>=0.5）走金色，空槽 ⣀ 走暗色。"""
    p = 0.0 if pct is None else max(0.0, min(100.0, float(pct)))
    total = p / 100.0 * width
    full = min(width, int(total))
    frac = total - full
    filled = "⣿" * full
    n_empty = width - full
    if full < width and frac >= 0.5:
        filled += "⣦"
        n_empty -= 1
    t = Text("[", style=C_DIM)
    t.append(filled, style=C_GOLD)
    t.append("⣀" * max(0, n_empty), style=STYLE_EMPTY)
    t.append("]", style=C_DIM)
    return t


def sparkline(values: list[float], direction: str = "minimize") -> Text:
    """目标值 sparkline：归一化到 ▁▂▃▄▅▆▇█ 8 级，最优值字符描金。少于 2 点返回空。"""
    if len(values) < 2:
        return Text()
    chars = "▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    best = hi if direction == "maximize" else lo
    t = Text()
    for v in values:
        lvl = 3 if hi == lo else round((v - lo) / (hi - lo) * 7)
        ch = chars[max(0, min(7, lvl))]
        t.append(ch, style=C_GOLD if v == best else C_TEXT)
    return t


class TargetProgressTracker:
    """单目标展示进度平滑器（移植 _recomputeDisp）。

    对 (round, progress_pct) 历史做 OLS 线性回归，把受 ci_half 噪声（前中期常为
    0、非单调）的收敛进度"拉成"近线性上升；叠加 round/max_rounds 线性地板与
    单调高水位 → 条整体线性上升、永不归零/倒退。纯展示用，不影响后端数据。
    """

    def __init__(self) -> None:
        self.hist: list[tuple[float, float]] = []  # (round, progress_pct)，按 round upsert
        self.disp_pct: int = 0  # 单调高水位

    def update(self, rec: dict, max_rounds: float | None) -> int:
        # 1) 入历史（按 round 去重 upsert；progress_pct 为 None 的不进回归）
        rec_round = rec.get("round")
        rec_pct = rec.get("progress_pct")
        if rec_round is not None and rec_pct is not None:
            for i, (x, _) in enumerate(self.hist):
                if x == rec_round:
                    self.hist[i] = (x, float(rec_pct))
                    break
            else:
                self.hist.append((float(rec_round), float(rec_pct)))

        # 2) OLS 拟合 progress_pct ~ round
        ols: float | None = None
        n = len(self.hist)
        if n >= 2:
            sx = sum(x for x, _ in self.hist)
            sy = sum(y for _, y in self.hist)
            sxx = sum(x * x for x, _ in self.hist)
            sxy = sum(x * y for x, y in self.hist)
            den = n * sxx - sx * sx
            if abs(den) > 1e-9:
                b = (n * sxy - sx * sy) / den
                a = (sy - b * sx) / n
                x_cur = float(rec_round) if rec_round is not None else sx / n
                ols = a + b * x_cur
            else:
                ols = sy / n
        elif n == 1:
            ols = self.hist[0][1]

        # 3) 线性地板 round/max_rounds（保证从第 1 轮就上升，不被早期全 0 困住）
        floor = 0.0
        if max_rounds and rec_round is not None:
            floor = (rec_round / max_rounds) * 100.0

        # 4) 取较大者 → 终态封顶 100 → clamp
        est = max(ols if ols is not None else 0.0, floor)
        if rec.get("phase") == "attack_done" or rec.get("converged"):
            est = 100.0
        est = max(0.0, min(100.0, est))

        # 5) 单调高水位（永不倒退/归零）
        self.disp_pct = max(self.disp_pct, int(round(est)))
        return self.disp_pct


class EvalProgressState:
    """单任务的进度回放状态（evaluate/hpo 通用，移植 web 端 progressState 条目）。

    数据源是 progress.jsonl 逐条记录；apply_record 增量应用（等价 web 端 SSE 单条），
    可从头全量回放也可增量 tail。
    """

    def __init__(self, task_id: str, kind: str = "evaluate") -> None:
        self.task_id = task_id
        self.kind = kind
        self.running = False
        self.max_rounds: float | None = None
        self.order: list[str] = []
        self.targets: dict[str, dict] = {}
        self.done: set[str] = set()
        self.active_target: str | None = None
        self.hpo: dict | None = None
        self.hpo_trials: list[dict] = []
        self._trackers: dict[str, TargetProgressTracker] = {}
        self._trial_seen: set[tuple] = set()

    # ---- 写入 ----
    def set_running(self, running: bool) -> None:
        self.running = running
        self._recompute()

    def declare_targets(self, names: list[str], max_rounds: int | None = None) -> None:
        """预声明目标名单（launch 层 meta）：progress 记录到达前先渲染「等待中」占位行。

        与 web 端 progress 快照的 targets 占位语义对齐；幂等（已声明的跳过）。
        """
        for n in names:
            if n not in self.order:
                self.order.append(n)
                self.targets[n] = {}
        if self.max_rounds is None and max_rounds:
            self.max_rounds = max_rounds
        self._recompute()

    def apply_record(self, rec: dict) -> None:
        """应用一条 progress.jsonl 记录（移植 applyProgress，含 HPO trial 去重）。"""
        if rec.get("phase") == "hpo":
            self.kind = "hpo"
            self.hpo = rec
            last = rec.get("last") or {}
            if last.get("target") is not None:
                key = (
                    last.get("target"),
                    last.get("seed"),
                    json.dumps(last.get("params"), sort_keys=True, ensure_ascii=False),
                    rec.get("trial_done"),
                )
                if key not in self._trial_seen:
                    self._trial_seen.add(key)
                    self.hpo_trials.append(dict(last))
                    if len(self.hpo_trials) > _HPO_TRIALS_CAP:
                        self.hpo_trials = self.hpo_trials[-_HPO_TRIALS_CAP:]
            return
        tg = rec.get("target")
        if tg is not None:
            if tg not in self.order:
                self.order.append(tg)
            self.targets[tg] = rec
            self._tracker(tg).update(rec, self.max_rounds)
        if self.max_rounds is None and rec.get("max_rounds"):
            self.max_rounds = rec.get("max_rounds")
            # max_rounds 到位后重算已有目标（此前地板按 0 算）
            for t in self.targets:
                self._tracker(t).update(self.targets[t], self.max_rounds)
        self.running = True
        self._recompute()

    def _tracker(self, tg: str) -> TargetProgressTracker:
        if tg not in self._trackers:
            self._trackers[tg] = TargetProgressTracker()
        return self._trackers[tg]

    def _recompute(self) -> None:
        """重算 done/active（移植 recomputeEvalState）。"""
        if self.kind == "hpo":
            self.active_target = None
            return
        self.done = set()
        for tg, rec in self.targets.items():
            if (
                rec.get("phase") == "attack_done"
                or rec.get("converged")
                or (self.max_rounds and rec.get("round") is not None and rec["round"] >= self.max_rounds)
            ):
                self.done.add(tg)
        self.active_target = None
        if self.running:
            best: str | None = None
            best_ts = ""
            for tg, rec in self.targets.items():
                if tg in self.done:
                    continue
                ts = rec.get("ts") or ""
                if ts >= best_ts:  # >=：并列时取后遍历者（与 web 一致）
                    best_ts = ts
                    best = tg
            self.active_target = best

    # ---- 读取 ----
    def disp_pct(self, tg: str) -> int:
        tr = self._trackers.get(tg)
        return tr.disp_pct if tr else 0

    def overall_pct(self) -> int | None:
        """汇总进度 %（移植 _overallPct）：evaluate 取回归平滑均值，HPO 取 config 进度。"""
        if self.kind == "hpo":
            rec = self.hpo or {}
            tot, done = rec.get("configs_total"), rec.get("configs_done")
            if isinstance(tot, (int, float)) and tot and isinstance(done, (int, float)):
                return int(round(done / tot * 100))
            return None
        # 只统计有进度历史的目标（占位目标无 tracker，不计入——对齐 web 端 dispPct 过滤）
        pcts = [tr.disp_pct for tr in self._trackers.values()]
        if pcts:
            return int(round(sum(pcts) / len(pcts)))
        if self.max_rounds:
            with_round = [r.get("round") for r in self.targets.values() if r.get("round") is not None]
            if with_round:
                return int(round(sum(with_round) / (len(with_round) * self.max_rounds) * 100))
        return None


# ============================================================
# 行渲染
# ============================================================
def target_row(tg: str, rec: dict, state: EvalProgressState) -> Text:
    """单目标行（移植 renderTargetRow 的终端拟真三态）：
    active ❯ 描金提示符 + 行尾闪烁块光标；done ✓ 石绿整行压暗；idle ❯ 暗色。"""
    is_done = tg in state.done
    is_active = state.active_target == tg and not is_done
    line_style = C_DIM if is_done else (C_TEXT if is_active else C_DIM)
    t = Text(style=line_style)
    if is_done:
        t.append("✓ ", style=C_SAFE)
    elif is_active:
        t.append("❯ ", style=C_GOLD)
    else:
        t.append("❯ ", style=f"dim {C_DIM}")
    name = pad_cells(tg, 14)
    if rec.get("round") is None and not is_done:
        t.append(name)
        t.append("等待中")
        return t
    t.append(name)
    t.append(f"R{fmt_num(rec.get('round'), 0) or '?'}/{fmt_num(state.max_rounds, 0) or '?'}")
    t.append("  ")
    t.append(f"ELO {fmt_num(rec.get('elo'), 0) or '—'}")
    delta = rec.get("delta")
    if delta is not None and delta != 0:
        arrow, color = ("↑", C_UP) if delta > 0 else ("↓", C_DOWN)
        t.append(" ")
        t.append(f"{arrow}{fmt_num(abs(delta), 0)}", style=color)
    t.append("  ")
    t.append(f"CI±{fmt_num(rec.get('ci_half'), 0) or '—'}")
    t.append("  ")
    if is_done:
        t.append("已收敛" if rec.get("converged") else "完成", style=C_SAFE)
    else:
        pct = state.disp_pct(tg)
        t.append(braille_bar(pct))
        t.append(f" {pct}%")
    if is_active:
        t.append(" ")
        t.append("运行中", style=C_MUTED)
    return t


def _fmt_params(p: dict) -> str:
    """trial 参数紧凑展示：前 3 个 k=v + 溢出计数，⟨⟩ 包裹（移植 fmtParams）。"""
    if not isinstance(p, dict) or not p:
        return ""
    ents = list(p.items())
    shown = " ".join(f"{k}={fmt_num(v, 2) if isinstance(v, (int, float)) else v}" for k, v in ents[:3])
    extra = f" +{len(ents) - 3}" if len(ents) > 3 else ""
    return f"⟨{shown}{extra}⟩"


def hpo_lines(state: EvalProgressState, recent: int = 4) -> list[Text]:
    """HPO 直播：汇总行 + 盲文进度条 + 目标值 sparkline + 最近 trial 流水
    （移植 renderHpoBox；recent 可放大到面板级 12 条）。"""
    rec = state.hpo
    if not rec:
        return [Text("❯ HPO 搜索准备中", style=f"dim {C_DIM}")]
    t_done = rec.get("trial_done") or 0
    t_tot = rec.get("trial_total_est")
    c_done = rec.get("configs_done") or 0
    c_tot = rec.get("configs_total")
    c_tot_n = c_tot if isinstance(c_tot, (int, float)) else None
    pct = int(round(c_done / c_tot_n * 100)) if c_tot_n else 0
    arrow = "↑" if rec.get("direction") == "maximize" else "↓"

    head = Text("❯ ", style=C_GOLD)
    head.append(
        f"config {c_done}/{c_tot if c_tot is not None else '?'} · trial {t_done}/{t_tot if t_tot is not None else '?'}"
    )
    if rec.get("best_metric") is not None:
        head.append(f" · 最佳 {rec.get('metric_name') or ''}={fmt_num(rec['best_metric'], 3)} ")
        head.append(f"({arrow}更佳)", style=f"dim {C_MUTED}")

    bar = Text("    ", style=C_TEXT)
    bar.append(braille_bar(pct))
    bar.append(f" {pct}% ")

    lines = [head, bar]
    # 目标值 sparkline：成功 trial 的 value 归一化，最优值字符描金
    ok = [tr for tr in state.hpo_trials if tr.get("status") == "success" and isinstance(tr.get("value"), (int, float))]
    if len(ok) >= 2:
        trend = Text("    ", style=C_TEXT)
        trend.append("趋势 ", style=f"dim {C_MUTED}")
        trend.append(sparkline([float(tr["value"]) for tr in ok], rec.get("direction") or "minimize"))
        lines.append(trend)
    # 最近 trial 流水（新在下）：✓ 成功带目标值，✗ 失败带状态
    metric = rec.get("metric_name") or ""
    for tr in state.hpo_trials[-recent:]:
        who = pad_cells(str(tr.get("target")), 14)
        tag = f"s{tr.get('seed') or 0}"
        line = Text()
        if tr.get("status") == "success":
            line.append("✓ ", style=C_SAFE)
            line.append(f"{who}{tag}  {metric}={fmt_num(tr.get('value'), 2)}  ", style=C_TEXT)
        else:
            line.append("✗ ", style=C_WARN)
            line.append(f"{who}{tag}  {tr.get('status') or 'failed'}  ", style=C_TEXT)
        line.append(_fmt_params(tr.get("params")), style=f"dim {C_MUTED}")
        lines.append(line)
    return lines


def progress_lines(state: EvalProgressState, cursor_on: bool = True, recent: int = 4) -> list[Text]:
    """进度窗口主体行（evaluate → 每目标一行；hpo → HPO 直播）。
    cursor_on：active 行尾闪烁块光标的当前相位（TermBox 定时翻转）。"""
    if state.kind == "hpo":
        return hpo_lines(state, recent=recent)
    lines = [target_row(tg, state.targets[tg], state) for tg in state.order]
    if cursor_on and state.active_target:
        row = state.order.index(state.active_target) if state.active_target in state.order else None
        if row is not None:
            lines[row].append(" ")
            lines[row].append("▋", style=C_TEXT)
    return lines


def status_text(status: str) -> Text:
    """任务状态徽标（着色中文标签）。"""
    label, color = STATUS_LABELS.get(status, (status, C_MUTED))
    return Text(label, style=color)
