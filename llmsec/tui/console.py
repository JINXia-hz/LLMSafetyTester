"""llmsec.tui.console — 控制台屏：全屏日志流 + 常驻命令输入行。

范式（v4 定稿）：TUI = 原生 CLI 外面加一层 shell 转译壳。常态无任何可视化
区域；可视化由 ``top`` 命令全屏唤起（views.TaskLiveScreen，q/Esc 返回）。

本模块 = 转译层的 UI 宿主 + 动作分发（executor）：
  CommandInput   Input 子类，接管 ↑↓（浮层导航/历史）/ Tab（补全）/ Esc
  ConsoleScreen  #console(RichLog) + #cmd-complete(浮层) + #cmd-hint + #cmd-bar

所有命令动作映射到既有后端（launch 层 / task_manager / MCP query·actions），
后端零改动。IO 一律 thread worker（group="cmd" 串行输出不交错）。
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import cast

from rich.table import Table
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Input, RichLog, Static

from llmsec.tui.commands import (
    CACHE_CATEGORIES,
    COMMANDS,
    LS_RESOURCES,
    REGISTRY,
    complete,
    parse,
    tokens_with_partial,
    usage,
)
from llmsec.tui.render import (
    C_DIM,
    C_GOLD,
    C_SAFE,
    C_WARN,
    status_text,
)
from llmsec.tui.task_store import (
    TERMINAL_STATUSES,
    TaskSnapshot,
    TaskStore,
    attack_files,
    kind_label,
    study_yamls,
    task_summary,
)
from llmsec.tui.widgets import LogModal

_HIST_MAX = 200
_POPUP_MAX_ROWS = 6
_PROMPT = Text("❯ ", style=C_GOLD)

# ============================================================
# 小工具
# ============================================================


def _sampler_names() -> list[str]:
    """补全源：SAMPLERS 单源派生（r7），hybrid 默认置顶。"""
    from llmsec.params import SAMPLERS

    return ["hybrid", *(s for s in SAMPLERS if s != "hybrid")]


class _TTL:
    """带 TTL 的惰性缓存（补全源用：击键期不能反复打磁盘）。"""

    def __init__(self, fn: Callable[[], list[str]], ttl: float = 8.0) -> None:
        self._fn = fn
        self._ttl = ttl
        self._at = 0.0
        self._val: list[str] = []

    def __call__(self) -> list[str]:
        if time.time() - self._at > self._ttl:
            try:
                self._val = [str(x) for x in self._fn()]
            except Exception:
                self._val = []
            self._at = time.time()
        return self._val


class _RunsCache:
    """runs 名/目标模型缓存（list_runs 读盘，60s TTL，可强制刷新）。"""

    def __init__(self) -> None:
        self._at = 0.0
        self.names: list[str] = []
        self.models: list[str] = []

    def refresh(self, force: bool = False) -> None:
        if not force and time.time() - self._at < 60:
            return
        try:
            from llmsec.mcp.tools.query import list_runs

            runs = [r for r in (list_runs() or []) if isinstance(r, dict) and r.get("name")]
            self.names = [str(r["name"]) for r in runs]
            self.models = sorted({str(r.get("target") or r.get("target_model")) for r in runs} - {"None"})
            self._at = time.time()
        except Exception:
            pass


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


def _table(title: str | None, cols: list[str]) -> Table:
    t = Table(box=None, pad_edge=False, title=title, title_style=f"bold {C_GOLD}")
    for c in cols:
        t.add_column(Text(c, style=f"bold {C_GOLD}"))
    return t


def _dict_block(d: dict, limit: int = 1200) -> Text:
    s = json.dumps(d, ensure_ascii=False, indent=2, default=str)
    if len(s) > limit:
        s = s[:limit] + "\n…（截断）"
    return Text(s, style=C_DIM)


# ============================================================
# 输入框（接管 ↑↓ / Tab / Esc——Input 原生不绑这些键）
# ============================================================


class CommandInput(Input):
    BINDINGS = [
        Binding("tab", "apply", "补全", show=False),
        Binding("up", "up", "浮层/历史", show=False),
        Binding("down", "down", "浮层/历史", show=False),
        Binding("escape", "escape", "关闭/清空", show=False),
    ]

    @property
    def _console(self) -> ConsoleScreen:
        return cast("ConsoleScreen", self.screen)

    def action_apply(self) -> None:
        self._console.assist_apply()

    def action_up(self) -> None:
        self._console.assist_up()

    def action_down(self) -> None:
        self._console.assist_down()

    def action_escape(self) -> None:
        self._console.assist_escape()


# ============================================================
# 控制台屏
# ============================================================


class ConsoleScreen(Screen):
    CSS = """
    #console {
        height: 1fr;
        border: round #4B4136;
        background: #1C1814;
        color: #E7DFC8;
        padding: 0 1;
    }
    #cmd-complete {
        height: auto;
        max-height: 8;
        border: round #4B4136;
        background: #241F19;
        padding: 0 1;
        margin: 0 1;
    }
    #cmd-complete.hidden { display: none; }
    #cmd-hint {
        height: 1;
        color: #9A8F76;
        padding: 0 2;
    }
    #cmd-bar { border: round #D9B45C; }
    """

    BINDINGS: list[Binding] = []

    def __init__(self, store: TaskStore, *, warm: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.store = store
        self._warm = warm  # 后台预热补全源（真实体验用；测试关掉避免拖慢/争抢）
        self._snaps: list[TaskSnapshot] = []
        self._runs = _RunsCache()
        self._hist: list[str] = []
        self._hist_idx: int | None = None
        self._hist_draft: str = ""
        self._assist = None  # 最近一次 complete() 结果
        self._partial: str = ""
        self._sel: int = 0
        self._hist_suppress: bool = False  # 历史召回抑制一轮浮层（见 _on_input_changed）
        self._pending_confirm: Callable[[], None] | None = None  # kill 外部强杀 y/N
        self._tokens: dict[str, tuple[str, str]] = {}  # preview token -> (动作, 描述)
        # 补全源（击键惰性求值；重 IO 的带 TTL 缓存）
        self._snap_names = _TTL(lambda: self._snapshot_names())
        self._tgt = _TTL(self._load_target_names, ttl=10.0)
        self._snaps_env = _TTL(self._load_env_snapshot_names, ttl=10.0)
        self._yamls = _TTL(study_yamls, ttl=10.0)
        self._attacks = _TTL(lambda: [a for a in attack_files()], ttl=10.0)
        self._wss = _TTL(self._load_workspace_names, ttl=20.0)

    # ---- 布局 ----
    def compose(self) -> ComposeResult:
        yield RichLog(id="console", markup=False, wrap=True, highlight=False, max_lines=4000)
        yield Static("", id="cmd-complete", classes="hidden")
        yield Static("", id="cmd-hint")
        yield CommandInput(
            placeholder="输入命令（Tab 补全 · help 速查 · /agent 对话 · top 任务直播）",
            id="cmd-bar",
        )

    def on_mount(self) -> None:
        self._load_history()
        self.query_one("#cmd-bar", CommandInput).focus()
        self.out(Text("llmsec 终端指挥台 · shell 式命令", style=f"bold {C_GOLD}"))
        self.out(
            Text(
                "  help 命令速查 · Tab 补全 · ↑↓ 历史 · top 任务直播 · /agent 宣政殿对话 · quit 退出",
                style=f"dim {C_DIM}",
            )
        )
        if self._warm:
            self.run_worker(self._warm_sources, thread=True, group="warm")

    def _warm_sources(self) -> None:
        """后台预热补全源：首次 import mcp 工具链（~1s）+ list_runs 读库
        （~0.3s）绝不能落在击键路径的 UI 线程上——首键冻结期间浮层状态会
        滞后于输入。预热后 TTL 过期重取只剩毫秒级磁盘读。"""
        self._tgt()
        self._runs.refresh()
        self._snaps_env()
        self._wss()
        self._yamls()
        self._attacks()

    # ---- 输出 ----
    @property
    def _log(self) -> RichLog:
        return self.query_one("#console", RichLog)

    def echo(self, line: str) -> None:
        self._log.write(Text("❯ ", style=C_GOLD) + Text(line))

    def out(self, item: object = "") -> None:
        self._log.write(item)

    def ok(self, msg: str) -> None:
        self._log.write(Text.assemble(("✓ ", C_SAFE), (msg,)))

    def err(self, msg: str) -> None:
        self._log.write(Text.assemble(("✗ ", C_WARN), (msg,)))

    def dim(self, msg: str) -> None:
        self._log.write(Text(msg, style=f"dim {C_DIM}"))

    def _write(self, items: list[object]) -> None:
        for i in items:
            self._log.write(i)

    # ============================================================
    # 补全浮层 / 提示行 / 历史
    # ============================================================
    @on(Input.Changed, "#cmd-bar")
    def _on_input_changed(self, event: Input.Changed) -> None:
        inp = cast(CommandInput, event.input)
        if self._hist_suppress:
            # 历史召回（程序化设值）：本轮不弹浮层——否则召回值恰为命令前缀
            # 会立刻弹层，把随后的 ↑↓ 劫持成浮层导航（readline 语义应继续翻历史）
            self._hist_suppress = False
            self.query_one("#cmd-complete", Static).add_class("hidden")
            return
        self._refresh_assist(inp.value, inp.cursor_position)

    def _refresh_assist(self, value: str, caret: int) -> None:
        r = complete(value, caret, self._sources())
        self._assist = r
        self._partial = tokens_with_partial(value[:caret])[1]
        self._sel = 0
        self.query_one("#cmd-hint", Static).update(Text(r.hint, style=C_WARN if r.hint_error else f"dim {C_DIM}"))
        popup = self.query_one("#cmd-complete", Static)
        # 空输入（刚执行完/初始态）不弹全量命令浮层——打字后才浮现
        if r.items and value.strip():
            popup.remove_class("hidden")
            self._render_popup()
        else:
            popup.add_class("hidden")

    def _render_popup(self) -> None:
        """浮层整体用单个 Static 渲染（避免逐行 mount/remove 的异步竞态）。"""
        popup = self.query_one("#cmd-complete", Static)
        assert self._assist is not None
        text = Text()
        for i, item in enumerate(self._assist.items[:_POPUP_MAX_ROWS]):
            if i:
                text.append("\n")
            if i == self._sel:
                text.append(f"❯ {item.label}", style=f"bold {C_GOLD}")
            else:
                text.append(f"  {item.label}", style="#E7DFC8")
            if item.help:
                text.append(f"  {item.help}", style=f"dim {C_DIM}")
        popup.update(text)

    def assist_apply(self) -> None:
        r = self._assist
        if r is None or not r.items:
            return
        inp = self.query_one("#cmd-bar", CommandInput)
        # 陈旧防御：Changed 未及处理时（程序化设值/极速击键后立刻按键），
        # _partial 落后于输入框真实内容——用旧 partial 计算替换区间会拼坏输入。
        # 先同步刷新（选择重置），本次不应用。
        _, cur_partial = tokens_with_partial(inp.value[: inp.cursor_position])
        if cur_partial != self._partial:
            self._refresh_assist(inp.value, inp.cursor_position)
            return
        item = r.items[min(self._sel, len(r.items) - 1)]
        value, caret = inp.value, inp.cursor_position
        start = max(0, caret - len(self._partial))
        new = value[:start] + item.insert + value[caret:]
        inp.value = new
        inp.cursor_position = start + len(item.insert)

    def assist_up(self) -> None:
        if (
            self._assist is not None
            and self._assist.items
            and not self.query_one("#cmd-complete", Static).has_class("hidden")
        ):
            self._sel = max(0, self._sel - 1)
            self._render_popup()
            return
        self._hist_nav(-1)

    def assist_down(self) -> None:
        if (
            self._assist is not None
            and self._assist.items
            and not self.query_one("#cmd-complete", Static).has_class("hidden")
        ):
            self._sel = min(len(self._assist.items) - 1, self._sel + 1)
            self._render_popup()
            return
        self._hist_nav(1)

    def assist_escape(self) -> None:
        popup = self.query_one("#cmd-complete", Static)
        if not popup.has_class("hidden"):
            popup.add_class("hidden")
            return
        inp = self.query_one("#cmd-bar", CommandInput)
        if inp.value:
            inp.value = ""

    # ---- 历史（STATE_DIR/tui_history.txt 持久化） ----
    def _hist_path(self):
        from llmsec.core.config import STATE_DIR

        return STATE_DIR / "tui_history.txt"

    def _load_history(self) -> None:
        try:
            p = self._hist_path()
            if p.exists():
                self._hist = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()][-_HIST_MAX:]
        except OSError:
            self._hist = []

    def _hist_push(self, line: str) -> None:
        if self._hist and self._hist[-1] == line:
            return
        self._hist.append(line)
        self._hist = self._hist[-_HIST_MAX:]
        try:
            p = self._hist_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("\n".join(self._hist) + "\n", encoding="utf-8")
        except OSError:
            pass

    def _hist_nav(self, delta: int) -> None:
        if not self._hist:
            return
        inp = self.query_one("#cmd-bar", CommandInput)
        self._hist_suppress = True
        if self._hist_idx is None:
            if delta < 0:
                self._hist_draft = inp.value
                self._hist_idx = len(self._hist) - 1
            else:
                self._hist_suppress = False
                return
        else:
            self._hist_idx += delta
            if self._hist_idx >= len(self._hist):
                self._hist_idx = None
                inp.value = self._hist_draft
                return
        self._hist_idx = max(0, self._hist_idx)
        inp.value = self._hist[self._hist_idx]
        inp.cursor_position = len(inp.value)

    # ============================================================
    # 补全源
    # ============================================================
    def _snapshot_names(self) -> list[str]:
        return [s.id for s in self._snaps]

    def _load_target_names(self) -> list[str]:
        from llmsec.core.config import load_targets

        return sorted(load_targets())

    def _load_env_snapshot_names(self) -> list[str]:
        from llmsec.mcp.tools.actions import list_env_snapshots

        data = list_env_snapshots()
        if isinstance(data, list):
            return [s["name"] for s in data if isinstance(s, dict) and s.get("name")]
        return []

    def _load_workspace_names(self) -> list[str]:
        from llmsec.mcp.tools.query import list_workspaces

        data = list_workspaces()
        if isinstance(data, list):
            return [w["name"] for w in data if isinstance(w, dict) and w.get("name")]
        return []

    def _sources(self) -> dict[str, Callable[[], list[str]]]:
        tids = self._snap_names
        tgt = self._tgt
        runs = self._runs
        return {
            "targets": tgt,
            "models": lambda: sorted({*self._tgt(), *runs.models}),
            "attacks": self._attacks,
            "samplers": _sampler_names,
            "snapshots": self._snaps_env,
            "yamls": self._yamls,
            "taskids": tids,
            "topsel": lambda: ["hpo", *tids()],
            "cat_objects": lambda: [f"tasks/{t}" for t in tids()[:20]] + [f"runs/{r}" for r in runs.names[:20]],
            "ls_resources": lambda: [*LS_RESOURCES, *(f"runs/{m}" for m in runs.models)],
            "runs": lambda: runs.names,
            "workspaces": self._wss,
            "tokens": lambda: list(self._tokens),
            "cache_categories": lambda: list(CACHE_CATEGORIES),
        }

    def update_snapshots(self, snaps: list[TaskSnapshot]) -> None:
        """App 轮询消息驱动：刷新任务快照（补全源 + kill 解析用）。"""
        self._snaps = snaps
        self._snap_names = _TTL(self._snapshot_names)  # 立即生效

    def refresh_runs(self) -> None:
        """refresh 命令入口：线程里强制重载 runs 缓存（不打断 UI）。"""
        self.run_worker(lambda: self._runs.refresh(force=True), thread=True, group="refresh")

    # ============================================================
    # 提交 → 分发
    # ============================================================
    @on(Input.Submitted, "#cmd-bar")
    def _on_submitted(self, event: Input.Submitted) -> None:
        inp = cast(CommandInput, event.input)

        def _clear_and_sync() -> None:
            # 程序化清空后立即同步刷新浮层/提示——不等队列里滞后的 Changed
            # （否则下一次按键前浮层仍展示旧内容，即"删了输入提示还挂着"）
            inp.value = ""
            self._refresh_assist(inp.value, inp.cursor_position)

        if self._pending_confirm is not None:
            line = event.value.strip()
            _clear_and_sync()
            self.echo(line or "n")
            fn = self._pending_confirm
            self._pending_confirm = None
            if line.lower() in ("y", "yes", "是"):
                fn()
            else:
                self.dim("已取消")
            return
        # 回车优先应用高亮补全（fish/IDE 语义）：↑↓ 选中→回车接上。三重条件
        # 防陈旧误应用（程序化设值/极速击键后 Changed 滞后，_assist 落后于输入框，
        # 曾把 "clear" 回车误拼成 "ls clear"）：
        #   ① 光标在行尾（补全槽位=当前输入尾部；行首/行中不做回车应用）；
        #   ② 现场重算候选与浮层一致（滞后时重算必不同 → 放弃应用，直接执行）；
        #   ③ 高亮项与已输入不同（应用是无操作 → 直接执行）。
        r = self._assist
        apply_done = False
        if (
            r is not None
            and r.items
            and inp.cursor_position >= len(inp.value)
            and not self.query_one("#cmd-complete", Static).has_class("hidden")
        ):
            fresh = complete(inp.value, inp.cursor_position, self._sources())
            if [c.label for c in fresh.items] == [c.label for c in r.items]:
                item = r.items[min(self._sel, len(r.items) - 1)]
                if item.insert.rstrip() != self._partial:
                    self.assist_apply()
                    apply_done = True
        if not apply_done:
            line = event.value.strip()
            _clear_and_sync()
            if not line:
                return
            self.echo(line)
            self._hist_push(line)
            p = parse(line)
            for c in p.corrections:
                self.dim(f"✎ 已纠错：{c}")
            if not p.ok:
                for e in p.errors:
                    self.err(e)
                return
            self._dispatch(p)

    def _dispatch(self, p) -> None:
        name = p.name
        if name == "ls":
            self._cmd_ls(p)
        elif name == "cat":
            self._cmd_cat(p)
        elif name == "mkdir":
            self._cmd_mkdir(p)
        elif name == "rmdir":
            self._cmd_rmdir(p)
        elif name == "rm":
            self._cmd_rm(p)
        elif name == "clean":
            self._cmd_clean(p)
        elif name == "kill":
            self._cmd_kill(p)
        elif name == "top":
            self._cmd_top(p)
        elif name == "eval":
            self._cmd_eval(p)
        elif name == "hpo":
            self._cmd_hpo(p)
        elif name == "probe":
            self._cmd_probe(p)
        elif name == "elo":
            self._cmd_elo(p)
        elif name == "boundary":
            self._cmd_boundary(p)
        elif name == "surprise":
            self._cmd_surprise(p)
        elif name == "pairing":
            self._cmd_pairing(p)
        elif name == "compare":
            self._cmd_compare(p)
        elif name == "snapshot list":
            self._cmd_snapshot_list(p)
        elif name == "snapshot new":
            self._cmd_snapshot_new(p)
        elif name == "snapshot set":
            self._cmd_snapshot_set(p)
        elif name == "snapshot rm":
            self._cmd_snapshot_rm(p)
        elif name == "confirm":
            self._cmd_confirm(p)
        elif name == "help":
            self._cmd_help(p)
        elif name == "clear":
            self._cmd_clear(p)
        elif name == "refresh":
            self._cmd_refresh(p)
        elif name == "quit":
            self.app.exit()
        elif name == "/agent":
            self._cmd_agent(p)

    # ============================================================
    # ls —— 各资源渲染
    # ============================================================
    @work(thread=True, exclusive=True, group="cmd")
    def _cmd_ls(self, p) -> None:
        resource = p.positionals[0] if p.positionals else "tasks"
        base, _, sub = resource.partition("/")
        want_all = bool(p.values.get("all"))
        want_long = bool(p.values.get("long"))
        app = self.app
        try:
            if base == "tasks":
                # 直接消费轮询快照（2s 新鲜度足够；省一次磁盘 refresh，
                # 且与 kill/top 的任务解析同口径——不出现"看得到杀不掉"）
                snaps = list(self._snaps)
                if not want_all:
                    snaps = [s for s in snaps if s.status not in TERMINAL_STATUSES and s.status != "ended"]
                out: list[object]
                if not snaps:
                    out = [Text("无任务（-a 看全部；eval 发起评估）", style=f"dim {C_DIM}")]
                elif want_long:
                    out = self._tasks_long(snaps)
                else:
                    out = [self._tasks_table(snaps)]
                app.call_from_thread(self._write, out)
                return
            if base == "runs":
                from llmsec.mcp.tools.query import list_runs

                runs = [r for r in (list_runs(target=sub or None) or []) if isinstance(r, dict)]
                if want_all:
                    runs += [
                        r
                        for r in (list_runs(target=sub or None, junk_only=True) or [])
                        if isinstance(r, dict) and r.get("name") not in {x.get("name") for x in runs}
                    ]
                app.call_from_thread(self._write, self._runs_render(runs, want_long))
                return
            if base == "targets":
                from llmsec.mcp.tools.query import list_targets

                rows = [r for r in (list_targets() or []) if isinstance(r, dict)]
                if not rows:
                    app.call_from_thread(self.out, Text("（.env 未声明目标）", style=f"dim {C_DIM}"))
                    return
                t = _table(f"{len(rows)} 目标", ["目标", "模型", "base_url"])
                for r in rows:
                    t.add_row(str(r.get("name", "?")), str(r.get("model", "—")), str(r.get("base_url", "")))
                app.call_from_thread(self.out, t)
                return
            if base == "attacks":
                files = attack_files()
                app.call_from_thread(
                    self.out,
                    Text("  ".join(files) if files else "（attacks/ 为空）", style=None if files else f"dim {C_DIM}"),
                )
                return
            if base == "studies":
                ys = study_yamls()
                app.call_from_thread(
                    self.out,
                    Text(
                        "\n".join(f"  {y}" for y in ys) if ys else "（无 study yaml）",
                        style=None if ys else f"dim {C_DIM}",
                    ),
                )
                return
            if base == "snapshots":
                from llmsec.mcp.tools.actions import list_env_snapshots

                data = list_env_snapshots()
                t = _table(None, ["快照", "来源", "键数", "备注", "创建"])
                for s in data if isinstance(data, list) else []:
                    if isinstance(s, dict):
                        t.add_row(
                            str(s.get("name", "?")),
                            str(s.get("source", "—")),
                            str(len(s.get("keys") or [])),
                            str(s.get("note", "")),
                            str(s.get("created", ""))[:16],
                        )
                app.call_from_thread(
                    self.out, t if isinstance(data, list) and data else Text("（无 env 快照）", style=f"dim {C_DIM}")
                )
                return
            if base == "workspaces":
                from llmsec.mcp.tools.query import list_workspaces

                data = list_workspaces()
                t = _table(None, ["工作区", "备注"])
                for w in data if isinstance(data, list) else []:
                    if isinstance(w, dict):
                        t.add_row(str(w.get("name", "?")), str(w.get("note", w.get("status", ""))))
                app.call_from_thread(
                    self.out,
                    t if isinstance(data, list) and data else Text("（无工作区——mkdir 开辟）", style=f"dim {C_DIM}"),
                )
                return
            if base == "cache":
                from llmsec.mcp.tools.actions import clean_caches_preview

                data = clean_caches_preview(list(CACHE_CATEGORIES))
                app.call_from_thread(self.out, _dict_block(data) if isinstance(data, dict) else Text(str(data)))
                return
            if base == "params":
                from llmsec.mcp.tools.query import get_params

                data = get_params()
                if isinstance(data, dict) and "categories" in data:
                    data = data["categories"]
                if isinstance(data, dict):
                    lines = [
                        Text(f"  {k}: {len(v) if isinstance(v, (list, dict)) else v} 项", style=f"dim {C_DIM}")
                        for k, v in data.items()
                    ]
                    app.call_from_thread(self._write, lines or [Text("（无）")])
                else:
                    app.call_from_thread(self.out, _dict_block(data if isinstance(data, dict) else {"data": data}))
                return
            app.call_from_thread(self.err, f"未知资源 {base}")
        except Exception as e:
            app.call_from_thread(self.err, f"ls {base} 失败: {type(e).__name__}: {e}")

    def _tasks_table(self, snaps: list[TaskSnapshot]) -> Table:
        t = _table(f"{len(snaps)} 任务", ["任务", "状态", "进度", "开始", "来源", "摘要"])
        for s in snaps:
            pct = s.state.overall_pct() if s.state else None
            t.add_row(
                f"{kind_label(s.kind)}·{s.id.split('-')[-1]}",
                status_text(s.status),
                Text(f"{pct}%", style=C_GOLD) if pct is not None else Text("—", style=C_DIM),
                (s.started_at or "")[11:19],
                "本机" if s.owned else "外部",
                task_summary(s),
            )
        return t

    def _tasks_long(self, snaps: list[TaskSnapshot]) -> list[object]:
        out: list[object] = []
        for s in snaps:
            head = Text()
            head.append(f"{s.id}  ", style=C_GOLD)
            head.append(status_text(s.status))
            head.append(f"  {'本机' if s.owned else '外部'}  {s.started_at or ''}", style=f"dim {C_DIM}")
            out.append(head)
            if s.pid:
                out.append(Text(f"  pid {s.pid}", style=f"dim {C_DIM}"))
            if s.cmd:
                out.append(Text(f"  cmd {s.cmd[:120]}", style=f"dim {C_DIM}"))
            if s.meta:
                out.append(Text(f"  meta {json.dumps(s.meta, ensure_ascii=False)}", style=f"dim {C_DIM}"))
        return out

    def _runs_render(self, runs: list[dict], long_flag: bool) -> list[object]:
        if not runs:
            return [Text("无 run（eval 发起评估）", style=f"dim {C_DIM}")]
        level_color = {"safe": C_SAFE, "allergic": C_GOLD, "vulnerable": C_WARN, "broken": C_WARN}
        t = _table(f"{len(runs)} run", ["run", "目标", "等级", "ASR", "边界Elo", "报告", "修改时间"])
        for r in runs:
            level = r.get("security_level") or "—"
            t.add_row(
                str(r.get("name", "?")),
                str(r.get("target") or r.get("target_model") or "—"),
                Text(level, style=level_color.get(level, C_DIM)),
                _fmt_ratio(r.get("asr")),
                _fmt_num(r.get("boundary_elo")),
                "✓" if r.get("has_report") else "✗",
                str(r.get("mtime") or "")[:16].replace("T", " "),
            )
        out: list[object] = [t]
        if long_flag:
            for r in runs[:10]:
                out.append(Text(f"  {r.get('name')}: {_dict_compact(r)}", style=f"dim {C_DIM}"))
        return out

    # ============================================================
    # cat —— 日志 / 报告查看器
    # ============================================================
    @work(thread=True, exclusive=True, group="cmd")
    def _cmd_cat(self, p) -> None:
        obj = p.positionals[0]
        base, _, key = obj.partition("/")
        app = self.app
        try:
            if base == "tasks":
                hits = [s for s in self._snaps if key in s.id]
                if not hits:
                    recent = ", ".join(s.id for s in self._snaps[:3]) or "（无任务）"
                    app.call_from_thread(self.err, f"找不到任务 {key}（最近：{recent}）")
                    return
                if len(hits) > 1:
                    app.call_from_thread(self.out, Text("多个匹配：" + " ".join(s.id for s in hits[:6]), style=C_WARN))
                    return
                snap = hits[0]
                text = self.store.full_log(snap.id)
                app.call_from_thread(app.push_screen, LogModal(f"日志 · {snap.id}", text or "（空日志）"))
                return
            if base == "runs":
                from llmsec.mcp.tools.query import assess_run_findings, read_run_report

                data = read_run_report(key)
                if not isinstance(data, dict) or data.get("error") or not data.get("report"):
                    err = data.get("error", "run 不存在或无报告") if isinstance(data, dict) else "读取失败"
                    app.call_from_thread(self.err, str(err))
                    return
                findings = assess_run_findings(key)
                if isinstance(findings, dict):
                    data = {**data, "findings": findings.get("findings")}
                app.call_from_thread(app.push_screen, LogModal(f"报告 · {key}", _render_report(key, data)))
                return
        except Exception as e:
            app.call_from_thread(self.err, f"cat 失败: {type(e).__name__}: {e}")

    # ============================================================
    # 工作区 / 删除 / 清理 / kill / top
    # ============================================================
    @work(thread=True, exclusive=True, group="cmd")
    def _cmd_mkdir(self, p) -> None:
        from llmsec.mcp.tools.actions import fork_workspace

        name = p.positionals[0]
        try:
            res = fork_workspace(name, source=p.values.get("source") or "global", note=p.values.get("note") or "")
            self._wss = _TTL(self._load_workspace_names)
            if isinstance(res, dict) and res.get("error"):
                self.app.call_from_thread(self.err, str(res["error"]))
            else:
                self.app.call_from_thread(self.ok, f"工作区 {name} 已开辟（源 {p.values.get('source') or 'global'}）")
        except Exception as e:
            self.app.call_from_thread(self.err, f"mkdir 失败: {e}")

    @work(thread=True, exclusive=True, group="cmd")
    def _cmd_rmdir(self, p) -> None:
        from llmsec.mcp.tools.actions import delete_workspace

        name = p.positionals[0]
        try:
            res = delete_workspace(name)
            self._wss = _TTL(self._load_workspace_names)
            if isinstance(res, dict) and res.get("error"):
                self.app.call_from_thread(self.err, str(res["error"]))
            else:
                self.app.call_from_thread(self.ok, f"工作区 {name} 已删除")
        except Exception as e:
            self.app.call_from_thread(self.err, f"rmdir 失败: {e}")

    @work(thread=True, exclusive=True, group="cmd")
    def _cmd_rm(self, p) -> None:
        from llmsec.mcp.tools.actions import delete_runs_preview

        names = list(p.positionals)
        try:
            res = delete_runs_preview(names, delete_r=bool(p.values.get("delete-r")))
            if isinstance(res, dict) and res.get("error"):
                self.app.call_from_thread(self.err, str(res["error"]))
                return
            token = res.get("confirm_token") if isinstance(res, dict) else None
            if not token:
                self.app.call_from_thread(self.out, _dict_block(res if isinstance(res, dict) else {"res": res}))
                return
            self.app.call_from_thread(self._register_preview, token, "runs", f"删除 {len(names)} 个 run")
        except Exception as e:
            self.app.call_from_thread(self.err, f"rm 失败: {e}")

    @work(thread=True, exclusive=True, group="cmd")
    def _cmd_clean(self, p) -> None:
        from llmsec.mcp.tools.actions import clean_caches_preview

        cats = list(p.positionals)
        try:
            res = clean_caches_preview(cats)
            if isinstance(res, dict) and res.get("error"):
                self.app.call_from_thread(self.err, str(res["error"]))
                return
            token = res.get("confirm_token") if isinstance(res, dict) else None
            if not token:
                self.app.call_from_thread(self.out, _dict_block(res if isinstance(res, dict) else {"res": res}))
                return
            self.app.call_from_thread(self._register_preview, token, "cache", f"清理 {'/'.join(cats)}")
        except Exception as e:
            self.app.call_from_thread(self.err, f"clean 失败: {e}")

    def _register_preview(self, token: str, kind: str, desc: str) -> None:
        """preview 结果落控制台并登记 token（confirm 命令 + 补全源用）。"""
        self._tokens[token] = (kind, desc)
        self.ok(f"已预览：{desc}")
        self.out(_dict_block({"token": token, "说明": "5 分钟内输入 confirm <token> 执行"}))

    @work(thread=True, exclusive=True, group="cmd")
    def _cmd_confirm(self, p) -> None:
        token = p.positionals[0]
        entry = self._tokens.get(token)
        if entry is None:
            self.app.call_from_thread(self.err, f"未知 token {token}（先 rm / clean 预览）")
            return
        kind, desc = entry
        try:
            if kind == "runs":
                from llmsec.mcp.tools.actions import delete_runs_confirm as confirm_fn
            else:
                from llmsec.mcp.tools.actions import clean_caches_confirm as confirm_fn
            res = confirm_fn(token)
            self._tokens.pop(token, None)
            if isinstance(res, dict) and res.get("error"):
                self.app.call_from_thread(self.err, str(res["error"]))
            else:
                self.app.call_from_thread(self.ok, f"已执行：{desc}")
        except Exception as e:
            self.app.call_from_thread(self.err, f"confirm 失败: {e}")

    def _resolve_task(self, key: str) -> TaskSnapshot | list[TaskSnapshot] | None:
        if key == "latest":
            active = [s for s in self._snaps if s.status in ("running", "queued")]
            return active[0] if active else None
        hits = [s for s in self._snaps if s.id.startswith(key) or key in s.id]
        if len(hits) == 1:
            return hits[0]
        return hits or None

    def _cmd_kill(self, p) -> None:
        key = p.positionals[0]
        resolved = self._resolve_task(key)
        if resolved is None:
            recent = ", ".join(s.id.split("-")[-1] for s in self._snaps[:3]) or "（无任务）"
            self.err(f"找不到任务 {key}（最近：{recent}）")
            return
        if isinstance(resolved, list):
            self.out(Text("多个匹配：" + " ".join(s.id for s in resolved[:6]), style=C_WARN))
            return
        snap = resolved
        if snap.status in TERMINAL_STATUSES or snap.status == "ended":
            self.err(f"{snap.id} 已是终态（{snap.status}），无需取消")
            return
        if not snap.owned and snap.pid is not None:
            self.out(Text(f"外部任务 {snap.id} 持有 PID {snap.pid}，跨进程强杀将连子进程树一起终止。", style=C_WARN))
            self._pending_confirm = lambda: self._do_kill(snap.id)
            self.dim("输入 y 确认 / 其它取消：")
            return
        self._do_kill(snap.id)

    @work(thread=True, exclusive=True, group="cmd")
    def _do_kill(self, task_id: str) -> None:
        view = self.store.cancel(task_id)
        if view.get("error") is not None:
            self.app.call_from_thread(self.err, f"取消失败：{view['error']}")
        else:
            self.app.call_from_thread(self.ok, f"已取消 {task_id}")

    def _cmd_top(self, p) -> None:
        arg = p.positionals[0] if p.positionals else ""
        kind = None
        focus = None
        if arg:
            if arg in ("hpo", "evaluate"):
                kind = arg
            else:
                resolved = self._resolve_task(arg)
                if resolved is None or isinstance(resolved, list):
                    self.err(f"找不到任务 {arg}（top / top hpo / top <id前缀>）")
                    return
                focus = resolved.id
                kind = resolved.kind
        self.app.open_top(kind=kind, focus=focus)

    # ============================================================
    # 评估 / HPO / 探测
    # ============================================================
    @work(thread=True, exclusive=True, group="cmd")
    def _cmd_eval(self, p) -> None:
        v = p.values
        targets = v.get("target")
        if v.get("all"):
            targets = None
        kwargs = dict(
            target=targets[0] if isinstance(targets, list) and len(targets) == 1 else None,
            targets=targets if isinstance(targets, list) and len(targets) > 1 else None,
            input_file=v.get("input") or "attacks/l1.jsonl",
            phase=v.get("phase") or "all",
            max_rounds=v.get("max-rounds", 5),
            sampler=v.get("sampler") or "hybrid",
            batch_size=v.get("batch-size"),
            seed=v.get("seed"),
            sampler_alpha=v.get("sampler-alpha"),
            sampler_beta=v.get("sampler-beta"),
            sampler_gamma=v.get("sampler-gamma"),
            coordinate_rounds=v.get("coordinate-rounds"),
            twin_window=v.get("twin-window"),
            no_early_stop=bool(v.get("no-early-stop")),
            env_snapshot=v.get("env-snap"),
            param_overrides=v.get("param"),
        )
        view = self.store.start_evaluation(**kwargs)
        # task_view 恒含 error=None 键：必须按值判空（键恒在）
        if view.get("error") is not None:
            msg = f"{view['error']}"
            if view.get("hint"):
                msg += f"\n  {view['hint']}"
            self.app.call_from_thread(self.err, msg)
            return
        meta = view.get("meta") or {}
        tg = "+".join(meta.get("targets") or []) or "全部目标"
        self.app.call_from_thread(self.ok, f"任务 {view.get('id')} 已入队 · {tg} · {meta.get('max_rounds', '?')} 轮")
        self.app.call_from_thread(self.dim, f"  top 看直播 · log {view.get('id', '')} 看日志")

    @work(thread=True, exclusive=True, group="cmd")
    def _cmd_hpo(self, p) -> None:
        yaml_path = p.positionals[0]
        view = self.store.start_hpo(yaml_path)
        if view.get("error") is not None:
            msg = f"{view['error']}"
            if view.get("hint"):
                msg += f"\n  {view['hint']}"
            self.app.call_from_thread(self.err, msg)
            return
        self.app.call_from_thread(self.ok, f"HPO 任务 {view.get('id')} 已入队 · top hpo 看直播")

    @work(thread=True, exclusive=True, group="cmd")
    def _cmd_probe(self, p) -> None:
        from llmsec.mcp.tools.query import probe_targets

        name = p.positionals[0] if p.positionals else None
        res = probe_targets(name)
        if isinstance(res, dict) and res.get("error"):
            self.app.call_from_thread(self.err, str(res["error"]))
            return
        self.app.call_from_thread(self.out, _dict_block(res if isinstance(res, dict) else {"res": res}))

    # ============================================================
    # 查询 / 分析
    # ============================================================
    def _pick_model(self, given: str | None) -> str | None:
        """在 worker 线程内调用：模型缺省时唯一模型自动选，否则报候选（回 UI 线程打印）。"""
        if given:
            return given
        models = sorted({*self._tgt(), *self._runs.models})
        if len(models) == 1:
            return models[0]
        msg = "请指定目标模型：" + (" / ".join(models[:6]) if models else "（先跑 eval 或声明 TARGETS）")
        self.app.call_from_thread(self.err, msg)
        return None

    @work(thread=True, exclusive=True, group="cmd")
    def _cmd_elo(self, p) -> None:
        from llmsec.mcp.tools.query import elo_ranking

        model = self._pick_model(p.positionals[0] if p.positionals else None)
        if model is None:
            return
        data = elo_ranking(model)
        if not isinstance(data, list):
            self.app.call_from_thread(self.err, "ELO 派生失败（R 矩阵无该模型列？）")
            return
        t = _table(f"攻击方 Elo 榜 · {model}（高 = 强攻击）", ["攻击方法", "Elo", "场次", "预测Elo"])
        for r in data:
            if isinstance(r, dict):
                t.add_row(
                    str(r.get("attacker", "?")),
                    _fmt_num(r.get("elo")),
                    _fmt_num(r.get("played")),
                    _fmt_num(r.get("predicted")),
                )
        self.app.call_from_thread(self.out, t)

    @work(thread=True, exclusive=True, group="cmd")
    def _cmd_boundary(self, p) -> None:
        from llmsec.mcp.tools.query import elo_security_boundary

        model = p.positionals[0]
        data = elo_security_boundary(model)
        if not isinstance(data, dict) or data.get("error") or not data:
            self.app.call_from_thread(self.err, "派生失败（R 矩阵无该模型列？）")
            return
        lines = [Text(f"== {model} · 安全边界 ==", style=f"bold {C_GOLD}")]
        for k, v in data.items():
            if k == "error":
                continue
            if isinstance(v, bool):
                v = "✓" if v else "✗"
            lines.append(Text(f"  {k}: {v}"))
        self.app.call_from_thread(self._write, lines)

    @work(thread=True, exclusive=True, group="cmd")
    def _cmd_surprise(self, p) -> None:
        from llmsec.mcp.tools.query import elo_find_surprises

        model = self._pick_model(p.positionals[0] if p.positionals else None)
        if model is None:
            return
        data = elo_find_surprises(model)
        rows = []
        if isinstance(data, dict) and not data.get("error"):
            for kind, label in (("weakness", "短板"), ("strength", "强项")):
                for r in data.get(kind) or []:
                    if isinstance(r, dict):
                        rows.append(
                            [
                                label,
                                str(r.get("attacker", "?")),
                                _fmt_num(r.get("elo_gap")),
                                _fmt_num(r.get("eval_score")),
                            ]
                        )
        if not rows:
            self.app.call_from_thread(self.err, "无意外事件（或派生失败）")
            return
        t = _table(f"意外发现 · {model}（短板=低Elo得手 / 强项=高Elo失手）", ["类型", "攻击方法", "Elo差", "评分"])
        for r in rows:
            t.add_row(*r)
        self.app.call_from_thread(self.out, t)

    @work(thread=True, exclusive=True, group="cmd")
    def _cmd_pairing(self, p) -> None:
        from llmsec.mcp.tools.query import elo_suggest_next_pairing

        model = self._pick_model(p.positionals[0] if p.positionals else None)
        if model is None:
            return
        n = p.values.get("n") or 8
        data = elo_suggest_next_pairing(model, n=n)
        t = _table(f"下一批测试建议 · {model}（Elo 差距最小 = 最值得测）", ["攻击方法", "防御方"])
        got = False
        for r in data if isinstance(data, list) else []:
            if isinstance(r, dict):
                t.add_row(str(r.get("attacker", "?")), str(r.get("defender", model)))
                got = True
        if not got:
            self.app.call_from_thread(self.err, "无配对建议（或派生失败）")
            return
        self.app.call_from_thread(self.out, t)

    @work(thread=True, exclusive=True, group="cmd")
    def _cmd_compare(self, p) -> None:
        from llmsec.mcp.tools.query import compare_runs

        a, b = p.positionals[0], p.positionals[1]
        data = compare_runs([a, b])
        if not isinstance(data, dict) or data.get("error"):
            err = data.get("error", "对比失败") if isinstance(data, dict) else "对比失败"
            self.app.call_from_thread(self.err, str(err))
            return
        metric_label = {
            "asr": "ASR",
            "fpr": "FPR",
            "boundary_elo": "边界 Elo",
            "coverage": "覆盖率",
            "conv_rounds": "收敛轮数",
        }
        t = _table(f"对比 · {a} vs {b}", ["指标", a.split("/")[-1], b.split("/")[-1]])
        for metric, values in (data.get("metrics") or {}).items():
            label = metric_label.get(metric, metric)
            fmt = _fmt_ratio if metric in ("asr", "fpr", "coverage") else _fmt_num
            t.add_row(label, *[fmt(v) for v in values.values()])
        if data.get("missing"):
            t.add_row("缺失 run", *[str(m) for m in data["missing"]][:2])
        self.app.call_from_thread(self.out, t)

    # ============================================================
    # env 快照
    # ============================================================
    @work(thread=True, exclusive=True, group="cmd")
    def _cmd_snapshot_list(self, p) -> None:
        data = self._snaps_env()
        if not data:
            self.app.call_from_thread(self.out, Text("（无 env 快照——snapshot new 创建）", style=f"dim {C_DIM}"))
            return
        t = _table(None, ["快照", "来源", "键数", "备注", "创建"])
        from llmsec.mcp.tools.actions import list_env_snapshots

        for s in list_env_snapshots() or []:
            if isinstance(s, dict):
                t.add_row(
                    str(s.get("name", "?")),
                    str(s.get("source", "—")),
                    str(len(s.get("keys") or [])),
                    str(s.get("note", "")),
                    str(s.get("created", ""))[:16],
                )
        self.app.call_from_thread(self.out, t)

    @work(thread=True, exclusive=True, group="cmd")
    def _cmd_snapshot_new(self, p) -> None:
        from llmsec.mcp.tools.actions import create_env_snapshot

        name = p.positionals[0]
        res = create_env_snapshot(name, source=p.values.get("source") or "global", note=p.values.get("note") or "")
        self._snaps_env = _TTL(self._load_env_snapshot_names)
        if isinstance(res, dict) and res.get("error"):
            self.app.call_from_thread(self.err, str(res["error"]))
        else:
            self.app.call_from_thread(self.ok, f"env 快照 {name} 已创建")

    @work(thread=True, exclusive=True, group="cmd")
    def _cmd_snapshot_set(self, p) -> None:
        from llmsec.mcp.tools.actions import edit_env_snapshot

        name, kv = p.positionals[0], p.positionals[1]
        if "=" not in kv:
            self.app.call_from_thread(self.err, f"kv 须为 KEY=V：{kv!r}")
            return
        k, v = kv.split("=", 1)
        res = edit_env_snapshot(name, k, v)
        if isinstance(res, dict) and res.get("error"):
            self.app.call_from_thread(self.err, str(res["error"]))
        else:
            self.app.call_from_thread(self.ok, f"{name}: {k}={v} 已写入")

    @work(thread=True, exclusive=True, group="cmd")
    def _cmd_snapshot_rm(self, p) -> None:
        from llmsec.mcp.tools.actions import delete_env_snapshot

        name = p.positionals[0]
        res = delete_env_snapshot(name)
        self._snaps_env = _TTL(self._load_env_snapshot_names)
        if isinstance(res, dict) and res.get("error"):
            self.app.call_from_thread(self.err, str(res["error"]))
        else:
            self.app.call_from_thread(self.ok, f"env 快照 {name} 已删除")

    # ============================================================
    # 杂项
    # ============================================================
    def _cmd_help(self, p) -> None:
        arg = p.positionals[0] if p.positionals else None
        if arg:
            cmd = REGISTRY.get(arg.lower())
            if cmd is None:
                self.err(f"未知命令 {arg}（help 查看全部）")
                return
            lines = [Text(f"{usage(cmd)}", style=f"bold {C_GOLD}"), Text(f"  {cmd.help}", style=None)]
            for a in cmd.args:
                tag = "…" if a.variadic else ("" if a.required else "（可选）")
                lines.append(Text(f"  <{a.name}>{tag}  {a.help}", style=f"dim {C_DIM}"))
            for o in cmd.opts:
                short = f"/-{o.short}" if o.short else ""
                lines.append(Text(f"  --{o.long}{short}  {o.help}", style=f"dim {C_DIM}"))
            self._write(lines)
            return
        t = _table("命令（Tab 补全 · 拼错自动纠错）", ["命令", "说明"])
        for c in COMMANDS:
            t.add_row(c.name, c.help)
        self.out(t)
        self.dim("别名：q/exit→quit · evaluate→eval ｜ 键位：↑↓ 历史 · Tab 补全 · Esc 关闭 · q/Esc 退出视图")

    def _cmd_clear(self, p) -> None:
        self._log.clear()

    def _cmd_refresh(self, p) -> None:
        self.app.action_refresh_all()
        self.ok("已刷新（任务 + runs 缓存）")

    @work(thread=True, exclusive=False, group="chat")
    def _cmd_agent(self, p) -> None:
        from control.agent.zhongshu.fallback import _help, chat_one

        text = " ".join(p.positionals).strip()
        try:
            reply = _help() if not text else chat_one(text)
        except Exception as e:
            reply = f"❌ chat_one 异常: {type(e).__name__}: {e}"
        if reply:
            self.app.call_from_thread(self.out, Text.assemble(("中书 ❯ ", C_DIM), (reply,)))


def _dict_compact(d: dict, limit: int = 160) -> str:
    s = json.dumps(d, ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[:limit] + "…"


def _render_report(name: str, data: dict) -> str:
    """报告 dict → 可读文本：核心指标 + findings + 完整 JSON（自 runs_panel 迁移）。"""
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
                lines.append(
                    f"[{f.get('severity', '?')}] {f.get('metric')}: {f.get('value')} (阈值 {f.get('threshold')})"
                )
                if f.get("interpretation"):
                    lines.append(f"    {f['interpretation']}")
    lines.append("")
    lines.append("-- 完整报告 JSON --")
    lines.append(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    return "\n".join(lines)
