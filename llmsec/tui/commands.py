"""llmsec.tui.commands — shell 式命令转译层（纯逻辑，无 textual 依赖）。

三层管道的第一层：用户输入（"ls -al tasks"、"eval -t glm4 -r 5"、"kill ab12"）
在这里完成解析（parse）、位置感知补全（complete）、实时提示（hint）与拼写纠错
（fuzzy）。注册表只声明「语法面」（动词 / 参数 / 旗标 / 补全源名），不 import
任何后端——动作分发在 console.py 的 executor，后端仍是 launch 层 /
task_manager / MCP 工具（零改动）。

补全源（sources）以名字引用，调用方注入 ``name -> () -> list[str]`` 的惰性
提供者（TUI 用 store / 磁盘 / .env，测试注入假源）；静态源（资源名 / 子命令 /
命令名）由本模块默认提供。
"""

from __future__ import annotations

import difflib
import shlex
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

# ============================================================
# 语法声明
# ============================================================


@dataclass(frozen=True)
class Opt:
    """命名旗标。long 不带 --（如 "target"），short 不带 -（如 "t"）。"""

    long: str
    short: str | None = None
    kind: str = "str"  # str / int / float / bool / kv
    help: str = ""
    completer: str | None = None  # 值补全源名
    multi: bool = False  # 可重复 / 逗号分隔聚合为 list


@dataclass(frozen=True)
class Arg:
    """位置参数。variadic 吞并剩余位置参数（/agent 文本、rm 多 run 名）。"""

    name: str
    help: str = ""
    completer: str | None = None
    required: bool = True
    variadic: bool = False


@dataclass(frozen=True)
class Command:
    name: str  # 单词 "eval"、多词 "snapshot new"、斜杠 "/agent"
    help: str = ""
    args: tuple[Arg, ...] = ()
    opts: tuple[Opt, ...] = ()
    require_any: tuple[str, ...] = ()  # opts 长名：至少出现其一
    slash: bool = False  # "/agent" 等 TUI 特制命名空间

    @property
    def verb(self) -> str:
        return self.name.split()[0]


LS_RESOURCES = (
    "tasks",
    "runs",
    "targets",
    "attacks",
    "studies",
    "snapshots",
    "workspaces",
    "cache",
    "params",
)
CACHE_CATEGORIES = ("predictors", "feature_cluster", "model_state")
CAT_PREFIXES = ("tasks", "runs")

_COMMANDS: tuple[Command, ...] = (
    # ---- shell 动词（转译特制）----
    Command(
        "ls",
        "列出资源（默认 tasks）",
        args=(
            Arg("resource", required=False, completer="ls_resources", help="tasks/runs/targets/…，或 runs/<目标> 过滤"),
        ),
        opts=(
            Opt("all", "a", "bool", "含已结束/外部记录"),
            Opt("long", "l", "bool", "长格式（完整 id/命令行/元数据）"),
        ),
    ),
    Command(
        "cat",
        "查看详情：tasks/<id前缀> 完整日志 | runs/<名> 报告",
        args=(Arg("object", completer="cat_objects", help="tasks/<id前缀> 或 runs/<run名>"),),
    ),
    Command(
        "mkdir",
        "开辟隔离工作区",
        args=(Arg("name", help="新工作区名"),),
        opts=(Opt("source", None, "str", "来源：global / run:<run名>"),),
    ),
    Command("rmdir", "删除工作区", args=(Arg("name", completer="workspaces"),)),
    Command(
        "rm",
        "删除 run（预览 → confirm 两步）",
        args=(Arg("runs", variadic=True, completer="runs"),),
        opts=(Opt("delete-r", None, "bool", "连 run 的 R 矩阵数据一并删"),),
    ),
    Command(
        "clean",
        "清理缓存（预览 → confirm 两步）",
        args=(
            Arg(
                "categories",
                variadic=True,
                completer="cache_categories",
                help="predictors/feature_cluster/model_state",
            ),
        ),
    ),
    Command(
        "kill",
        "取消任务（外部任务跨进程强杀前确认）",
        args=(Arg("task", completer="taskids", help="任务 id 前缀 / latest"),),
    ),
    Command(
        "top",
        "唤起任务直播视图（q/Esc 返回）",
        args=(Arg("task", required=False, completer="topsel", help="任务 id 前缀 / hpo"),),
    ),
    # ---- domain 动词 ----
    Command(
        "eval",
        "发起红队评估",
        opts=(
            Opt("target", "t", "str", "目标模型（可重复 / 逗号分隔）", completer="targets", multi=True),
            Opt("all", None, "bool", "跑全部 .env 声明目标"),
            Opt("input", "i", "str", "攻击集文件名", completer="attacks"),
            Opt("max-rounds", "r", "int", "最大轮数（默认 5）"),
            Opt("sampler", None, "str", "采样策略", completer="samplers"),
            Opt("batch-size", None, "int", "批量大小"),
            Opt("seed", None, "int", "随机种子"),
            Opt("sampler-alpha", None, "float", "采样器 α"),
            Opt("sampler-beta", None, "float", "采样器 β"),
            Opt("sampler-gamma", None, "float", "采样器 γ"),
            Opt("coordinate-rounds", None, "int", "坐标下降轮数"),
            Opt("twin-window", None, "int", "过敏检测窗口"),
            Opt("phase", None, "str", "阶段 all/1/2"),
            Opt("no-early-stop", None, "bool", "跑满轮数不早停"),
            Opt("env-snap", None, "str", "env 快照隔离", completer="snapshots"),
            Opt("param", None, "kv", "参数覆写 KEY=V,KEY2=V2"),
        ),
        require_any=("target", "all"),
    ),
    Command("hpo", "启动 HPO study", args=(Arg("yaml", completer="yamls", help="study yaml 路径"),)),
    Command("probe", "探测目标 API 连通性", args=(Arg("target", required=False, completer="targets"),)),
    Command("elo", "攻击方 Elo 榜", args=(Arg("model", required=False, completer="models"),)),
    Command("boundary", "安全边界", args=(Arg("model", completer="models"),)),
    Command("surprise", "双向意外发现（短板/强项）", args=(Arg("model", required=False, completer="models"),)),
    Command(
        "pairing",
        "下一批测试建议（Elo 差距最小配对）",
        args=(Arg("model", required=False, completer="models"),),
        opts=(Opt("n", None, "int", "条数（默认 8）"),),
    ),
    Command("compare", "对比两个 run", args=(Arg("a", completer="runs"), Arg("b", completer="runs"))),
    Command("snapshot list", "列出 env 快照"),
    Command(
        "snapshot new",
        "创建 env 快照",
        args=(Arg("name", help="快照名"),),
        opts=(Opt("source", None, "str", "来源：global/blank/其他快照"), Opt("note", None, "str", "备注")),
    ),
    Command("snapshot set", "向快照写入一个 KEY=V", args=(Arg("name", completer="snapshots"), Arg("kv", help="KEY=V"))),
    Command("snapshot rm", "删除 env 快照", args=(Arg("name", completer="snapshots"),)),
    Command("confirm", "执行预览过的写操作", args=(Arg("token", completer="tokens", help="preview 返回的 token"),)),
    Command("help", "命令速查（help <命令> 看用法）", args=(Arg("command", required=False, completer="verbs"),)),
    Command("clear", "清空控制台"),
    Command("refresh", "立即刷新任务与数据缓存"),
    Command("quit", "退出"),
    # ---- / 特制 ----
    Command(
        "/agent",
        "宣政殿对话（自然语言 / JSON 指令；无参看引擎 help）",
        args=(Arg("text", variadic=True, required=False),),
        slash=True,
    ),
)

COMMANDS = _COMMANDS
REGISTRY: dict[str, Command] = {c.name.lower(): c for c in _COMMANDS}
ALIASES = {"q": "quit", "exit": "quit", "evaluate": "eval"}
# 多词命令的首词（"snapshot"）：单独出现时提示补全子命令
_MULTI_PREFIXES = {c.verb for c in _COMMANDS if " " in c.name}


def _default_sources() -> dict[str, Callable[[], list[str]]]:
    return {
        "ls_resources": lambda: list(LS_RESOURCES),
        "cache_categories": lambda: list(CACHE_CATEGORIES),
        "verbs": lambda: [c.name for c in _COMMANDS if not c.slash] + [c.name for c in _COMMANDS if c.slash],
    }


# ============================================================
# 拼写纠错
# ============================================================


def _edit_distance(a: str, b: str) -> int:
    """Damerau-Levenshtein（OSA）：相邻换位计 1 次编辑（evla→eval 距离 1）。"""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev2 = None
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if prev2 is not None and i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
    return prev[lb]


def strong_match(word: str, candidates: Iterable[str]) -> str | None:
    """唯一且足够近的匹配（编辑距离≤1 或相似度≥0.8）；多命中/无命中返回 None。"""
    w = word.lower().strip()
    if not w:
        return None
    hits: list[str] = []
    for c in candidates:
        cl = c.lower()
        if cl == w:
            return c
        if _edit_distance(w, cl) <= 1 or difflib.SequenceMatcher(None, w, cl).ratio() >= 0.8:
            hits.append(c)
    return hits[0] if len(hits) == 1 else None


def weak_matches(word: str, candidates: Iterable[str], n: int = 4) -> list[str]:
    """宽松候选（did-you-mean 列表用）。"""
    cands = [c for c in candidates if c.lower() != word.lower()]
    return difflib.get_close_matches(word.lower(), [c.lower() for c in cands], n=n, cutoff=0.6)


# ============================================================
# 解析
# ============================================================


@dataclass
class Parsed:
    """一条用户输入的解析结果。errors 非空时不执行。"""

    cmd: Command | None
    name: str
    values: dict = field(default_factory=dict)
    positionals: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    corrections: list[str] = field(default_factory=list)  # ["lss → ls"]（自动纠错回显）

    @property
    def ok(self) -> bool:
        return self.cmd is not None and not self.errors


def _verbs(slash: bool) -> list[str]:
    return [c.verb for c in _COMMANDS if c.slash == slash]


def _match_command(tokens: list[str]) -> tuple[Command | None, list[str], list[str] | None, list[str] | None]:
    """匹配命令名（两词优先 → 一词 → 别名 → 纠错一次）。

    Returns:
        (command, 剩余 tokens, 纠错说明或 None, 致命错误或 None)
    """
    head = tokens[0]
    low = head.lower()
    # 两词命令（snapshot new …）；第二词不能是旗标
    if len(tokens) >= 2 and not tokens[1].startswith("-"):
        key = f"{low} {tokens[1].lower()}"
        if key in REGISTRY:
            return REGISTRY[key], tokens[2:], None, None
    # 单词命令 / 别名
    resolved = REGISTRY.get(low) or REGISTRY.get(ALIASES.get(low, ""))
    if resolved is not None and " " not in resolved.name:
        return resolved, tokens[1:], None, None
    # 多词首词单独出现：提示子命令
    if low in _MULTI_PREFIXES:
        subs = [c.name for c in _COMMANDS if c.verb == low]
        return None, [], None, [f"{low} 需要子命令：{' / '.join(subs)}"]
    # 纠错一次（slash 命名空间独立）
    slash = low.startswith("/")
    fixed = strong_match(low, _verbs(slash))
    if fixed is not None:
        cmd2, rest2, _corr, fatal2 = _match_command([fixed, *tokens[1:]])
        return cmd2, rest2, f"{head} → {fixed}", fatal2
    cands = weak_matches(low, _verbs(slash)) or []
    err = [f"未知命令 {head}"]
    if cands:
        err.append(f"你是想输入：{' / '.join(cands)}？（Tab 补全 · help 查看全部）")
    return None, [], None, err


def _parse_kv(raw: str, flag: str) -> tuple[dict | None, str | None]:
    out: dict[str, str] = {}
    for seg in raw.split(","):
        seg = seg.strip()
        if not seg:
            continue
        if "=" not in seg:
            return None, f"{flag} 格式须 KEY=V 逗号分隔：{seg!r}"
        k, v = seg.split("=", 1)
        if not k.strip():
            return None, f"{flag} 的 KEY 不能为空：{seg!r}"
        out[k.strip()] = v.strip()
    return (out, None) if out else (None, f"{flag} 未解析到任何 KEY=V")


def parse(line: str) -> Parsed:
    """解析一行用户输入。纯语法层：类型转换 / 未知旗标 / 参数数量 / 资源名校验。"""
    values: dict = {}
    positionals: list[str] = []
    errors: list[str] = []
    corrections: list[str] = []

    raw = line.strip()
    if not raw:
        return Parsed(None, "", errors=["空输入"])
    try:
        tokens = shlex.split(raw)
    except ValueError:
        return Parsed(None, "", errors=['引号不匹配——空格参数请用 "..." 包裹'])
    if not tokens:
        return Parsed(None, "", errors=["空输入"])

    cmd, rest, corr, fatal = _match_command(tokens)
    if corr:
        corrections.append(corr)
    if cmd is None:
        return Parsed(None, "", errors=fatal or ["未知命令"])
    name = cmd.name

    long_map = {o.long: o for o in cmd.opts}
    short_map = {o.short: o for o in cmd.opts if o.short}
    short_bools = {o.short for o in cmd.opts if o.short and o.kind == "bool"}

    def set_value(opt: Opt, raw_val: str, flag_disp: str) -> None:
        if opt.kind == "int":
            try:
                v: object = int(raw_val)
            except ValueError:
                errors.append(f"{flag_disp} 须为整数：{raw_val!r}")
                return
        elif opt.kind == "float":
            try:
                v = float(raw_val)
            except ValueError:
                errors.append(f"{flag_disp} 须为数字：{raw_val!r}")
                return
        elif opt.kind == "kv":
            kv, err = _parse_kv(raw_val, flag_disp)
            if err:
                errors.append(err)
                return
            v = kv
        else:
            v = raw_val
        if opt.multi:
            pieces = [p.strip() for p in str(raw_val).split(",") if p.strip()] if opt.kind == "str" else [v]
            values.setdefault(opt.long, [])
            values[opt.long].extend(pieces if opt.kind == "str" else [v])
        else:
            values[opt.long] = v

    def unknown_flag(disp: str) -> None:
        near = weak_matches(disp.lstrip("-").replace("-", ""), [o.long.replace("-", "") for o in cmd.opts])
        msg = f"未知旗标 {disp}"
        if near:
            real = next(o.long for o in cmd.opts if o.long.replace("-", "") == near[0])
            msg += f"（是 --{real}？）"
        errors.append(msg)

    i = 0
    while i < len(rest):
        tok = rest[i]
        if tok == "--":
            positionals.extend(rest[i + 1 :])
            break
        if tok.startswith("--") and len(tok) > 2:
            fname, eq, inline = tok[2:].partition("=")
            opt = long_map.get(fname)
            if opt is None:
                unknown_flag(f"--{fname}")
                i += 1
                continue
            if opt.kind == "bool":
                if eq and inline.lower() not in ("true", "1"):
                    errors.append(f"--{fname} 是布尔开关（不接受 = 值）")
                values[opt.long] = True
                i += 1
                continue
            if eq:
                set_value(opt, inline, f"--{fname}")
                i += 1
            elif i + 1 < len(rest):
                set_value(opt, rest[i + 1], f"--{fname}")
                i += 2
            else:
                errors.append(f"--{fname} 缺值（{opt.help or opt.long}）")
                i += 1
        elif tok.startswith("-") and len(tok) > 1:
            body = tok[1:]
            # 组合布尔短旗标 "-al"
            if len(body) > 1 and all(c in short_bools for c in body):
                for c in body:
                    values[short_map[c].long] = True
                i += 1
                continue
            ch, stuck = body[0], body[1:]
            opt = short_map.get(ch)
            if opt is None:
                unknown_flag(f"-{ch}")
                i += 1
                continue
            if opt.kind == "bool":
                values[opt.long] = True
                if stuck and not all(c in short_bools for c in stuck):
                    errors.append(f"-{body} 组合中含非布尔短旗标")
                elif stuck:
                    for c in stuck:
                        values[short_map[c].long] = True
                i += 1
            elif stuck:
                set_value(opt, stuck, f"-{ch}")
                i += 1
            elif i + 1 < len(rest):
                set_value(opt, rest[i + 1], f"-{ch}")
                i += 2
            else:
                errors.append(f"-{ch} 缺值（{opt.help or opt.long}）")
                i += 1
        else:
            positionals.append(tok)
            i += 1

    fixed_args = [a for a in cmd.args if not a.variadic]
    variadic = next((a for a in cmd.args if a.variadic), None)
    for idx, a in enumerate(fixed_args):
        if a.required and idx >= len(positionals):
            errors.append(f"缺参数 <{a.name}>（{a.help}）" if a.help else f"缺参数 <{a.name}>")
    if not variadic and len(positionals) > len(fixed_args):
        errors.append(f"多余参数：{' '.join(positionals[len(fixed_args) :])}")
    if cmd.require_any and not any(values.get(n) for n in cmd.require_any):
        wanted = " / ".join(f"--{n}" for n in cmd.require_any)
        errors.append(f"缺 {wanted}")

    # 资源/对象名校验（ls 的资源名、cat 的路径前缀），可纠错
    if name == "ls" and positionals:
        res = positionals[0]
        base = res.split("/", 1)[0].lower()
        if base not in LS_RESOURCES:
            fixed = strong_match(base, LS_RESOURCES)
            if fixed:
                corrections.append(f"{base} → {fixed}")
                positionals[0] = fixed + (res[len(base) :] if "/" in res else "")
            else:
                errors.append(f"未知资源 {base}（可用：{'/'.join(LS_RESOURCES)}）")
    if name == "cat" and positionals:
        base = positionals[0].split("/", 1)[0].lower()
        if base not in CAT_PREFIXES:
            fixed = strong_match(base, CAT_PREFIXES)
            if fixed:
                corrections.append(f"{base} → {fixed}")
                positionals[0] = fixed + positionals[0][len(base) :]
            else:
                errors.append("cat 支持 tasks/<id前缀> 与 runs/<run名> 两种对象")

    return Parsed(cmd=cmd, name=name, values=values, positionals=positionals, errors=errors, corrections=corrections)


# ============================================================
# 补全与提示
# ============================================================


@dataclass(frozen=True)
class Completion:
    insert: str  # 应用到当前词的文本（含尾随空格）
    label: str  # 浮层显示名
    help: str = ""


@dataclass(frozen=True)
class CompleteResult:
    items: list[Completion]
    hint: str
    hint_error: bool = False


def usage(cmd: Command) -> str:
    parts = [cmd.name]
    for a in cmd.args:
        if a.variadic:
            parts.append(f"<{a.name}…>")
        elif a.required:
            parts.append(f"<{a.name}>")
        else:
            parts.append(f"[{a.name}]")
    if cmd.opts:
        parts.append("[旗标…]")
    return " ".join(parts)


def tokens_with_partial(text: str) -> tuple[list[str], str]:
    """(已完成 tokens, 当前词)。text 以空白结尾 → 当前词为空（补下一个词）。"""
    if text and text[-1].isspace():
        try:
            return shlex.split(text), ""
        except ValueError:
            return text.split(), ""
    try:
        toks = shlex.split(text)
    except ValueError:
        toks = text.split()
    if toks:
        return toks[:-1], toks[-1].lstrip("\"'")
    return [], ""


def _flag_opt(tok: str, long_map: dict, short_map: dict) -> Opt | None:
    if tok.startswith("--") and len(tok) > 2:
        return long_map.get(tok[2:].split("=", 1)[0])
    if tok.startswith("-") and len(tok) > 1:
        return short_map.get(tok[1])
    return None


def _source_values(name: str, sources: Mapping[str, Callable[[], list[str]]]) -> list[str]:
    provider = sources.get(name) or _default_sources().get(name)
    if provider is None:
        return []
    try:
        return [str(x) for x in provider()]
    except Exception:
        return []


def complete(
    line: str, caret: int | None = None, sources: Mapping[str, Callable[[], list[str]]] | None = None
) -> CompleteResult:
    """位置感知补全：命令名 → 子命令 → 旗标名 → 旗标值 → 位置参数。"""
    src = dict(_default_sources())
    if sources:
        src.update(sources)
    text = line if caret is None else line[:caret]
    toks, partial = tokens_with_partial(text)
    pkey = partial.lower()
    max_items = 8

    def done(items: list[Completion], hint: str, err: bool = False) -> CompleteResult:
        return CompleteResult(items[:max_items], hint, err)

    # ---- 第一词：命令名 ----
    if not toks:
        if pkey.startswith("/"):
            cmds = [c for c in _COMMANDS if c.slash and c.name.lower().startswith(pkey)]
            if not cmds:
                return done([], f"未知命令 {partial}（/agent 对话）", err=True)
            h = f"{cmds[0].name} · {cmds[0].help}" if len(cmds) == 1 else "/ 前缀为 TUI 特制命令（对话 / 视图）"
            return done([Completion(c.name + " ", c.name, c.help) for c in cmds], h)
        cmds = [c for c in _COMMANDS if not c.slash and (c.name.lower().startswith(pkey) or c.verb.startswith(pkey))]
        if not pkey:  # 空输入也展示 / 特制，保证可发现性
            cmds += [c for c in _COMMANDS if c.slash]
        items = [Completion(c.name + " ", c.name, c.help) for c in cmds]
        for alias, target in ALIASES.items():
            if alias.startswith(pkey):
                items.append(Completion(alias + " ", alias, f"→ {target}"))
        if not items and pkey:
            return done([], f"未知命令 {partial}（help 查看全部）", err=True)
        # 提示行给出首个候选的说明（打字母即见该命令用途，而非通用文案）
        h = (
            f"{cmds[0].name} · {cmds[0].help}"
            if pkey and cmds
            else "命令 Tab 补全 · / 前缀为 TUI 特制（/agent）· help 查看全部"
        )
        return done(items, h)

    # ---- 子命令（snapshot 单独出现）----
    low0 = toks[0].lower()
    if len(toks) == 1 and low0 in _MULTI_PREFIXES:
        subs = [c for c in _COMMANDS if c.verb == low0 and c.name.split()[1].startswith(pkey)]
        if not subs and partial:
            return done(
                [], f"{low0} 的子命令：{' / '.join(c.name.split()[1] for c in _COMMANDS if c.verb == low0)}", err=True
            )
        return done(
            [Completion(c.name.split()[1] + " ", c.name, c.help) for c in subs],
            f"{low0} 子命令：{' / '.join(c.name.split()[1] for c in _COMMANDS if c.verb == low0)}",
        )

    # ---- 命令解析 ----
    cmd, rest, _corr, fatal = _match_command(toks)
    if cmd is None:
        return done([], fatal[0] if fatal else f"未知命令 {toks[0]}", err=True)

    long_map = {o.long: o for o in cmd.opts}
    short_map = {o.short: o for o in cmd.opts if o.short}
    used = set()

    # 遍历命令词之后的 tokens：跳过旗标及其值，统计位置参数个数
    n_fixed = len(cmd.name.split())
    pos_count = 0
    j = n_fixed
    while j < len(toks):
        t = toks[j]
        if t == "--":
            pos_count += len(toks) - j - 1
            break
        o = _flag_opt(t, long_map, short_map)
        if o is not None:
            used.add(o.long)
            if o.kind != "bool" and "=" not in t.lstrip("-") and (j + 1 < len(toks) or partial):
                j += 1  # 跳过值
        else:
            pos_count += 1
        j += 1

    hint = f"{usage(cmd)} · {cmd.help}"

    # ---- 旗标值补全（前一词是取值旗标 / --k=部分值）----
    if partial.startswith("--") and "=" in partial:
        fname, _, vpart = partial[2:].partition("=")
        o = long_map.get(fname)
        if o is not None and o.completer:
            cands = _source_values(o.completer, src)
            items = [
                Completion(f"--{fname}={c} ", f"--{fname}={c}", o.help)
                for c in cands
                if c.lower().startswith(vpart.lower())
            ]
            return done(items, hint)
    if toks and not partial.startswith("-"):
        o = _flag_opt(toks[-1], long_map, short_map)
        if o is not None and o.kind != "bool":
            cands = _source_values(o.completer or "", src) if o.completer else []
            items = [Completion(c + " ", c, o.help) for c in cands if not partial or c.lower().startswith(pkey)]
            return done(items, hint)

    # ---- 旗标名补全 ----
    if partial.startswith("-"):
        items: list[Completion] = []
        for o in cmd.opts:
            if o.long in used:
                continue
            if f"--{o.long}".startswith(partial):
                items.append(Completion(f"--{o.long} ", f"--{o.long}", o.help))
            if o.short and f"-{o.short}" == partial:
                items.insert(0, Completion(f"-{o.short} ", f"-{o.short}", o.help))
        if not items:
            return done([], f"没有匹配 {partial} 的旗标", err=True)
        return done(items, hint)

    # ---- 位置参数补全 ----
    fixed_args = [a for a in cmd.args if not a.variadic]
    arg = fixed_args[pos_count] if pos_count < len(fixed_args) else next((a for a in cmd.args if a.variadic), None)
    if arg is None or not arg.completer:
        return done([], hint)
    cands = _source_values(arg.completer, src)
    items = [Completion(c + " ", c, arg.help) for c in cands if not partial or c.lower().startswith(pkey)]
    return done(items, hint)


def hint(line: str, sources: Mapping[str, Callable[[], list[str]]] | None = None) -> tuple[str, bool]:
    """实时提示行：(文案, 是否错误)。"""
    if not line.strip():
        return "命令 Tab 补全 · / 前缀为 TUI 特制（/agent）· help 查看全部", False
    r = complete(line, len(line), sources)
    return r.hint, r.hint_error
