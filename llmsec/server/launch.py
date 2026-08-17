"""llmsec.server.launch — 评估 / HPO 任务的统一启动层。

归一此前各自为政且已漂移的三方启动路径（argv 构造 + 校验 + env 注入）：
  - Web 看板  routers/tasks.py::api_run_evaluate —— 校验最全（pydantic 模式/范围/
    目标声明校验），但无 env_snapshot / param_overrides
  - MCP 工具  mcp/tools/tasks.py::run_evaluation —— 能力面最全（env 快照隔离 /
    params 覆写），但不校验目标声明、无采样器超参
  - TUI       tui/task_store.py —— 评估委派 MCP；HPO argv 与 hpo router 重复

本层取三方并集：LaunchSpec（参数超集 dataclass）→ launch_evaluation(spec) 统一
「校验 → 解析攻击集 → 构造 argv → 注入 env → task_manager.start_task（携带 meta）」。
Web / MCP / TUI 都只做自身协议的薄映射：

  HTTP:   LaunchError → HTTPException(4xx)
  MCP:    LaunchError → {"error": ..., "hint": ...}（fastmcp 工具不抛异常的约定）
  TUI:    LaunchError → notify

meta（targets/max_rounds/study 等）随任务注册进 TASKS 并出现在 task_view——
消费者不再反向解析 argv（原 web 端 _parse_eval_argv / TUI short_cmd 均据此废除）。

不在本层管辖（语义不同，刻意保持独立）：
  - experiments/executor._runner_argv：HPO trial 的 work-dir 隔离 + phase 1 +
    no-early-stop 固定语义（schema.CLI_FACTORS 驱动）
  - control/core/invoker.run_runner：控制层架构约定「不 import llmsec 任何模块，
    只经 CLI + 文件交互」
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from llmsec.params import MAX_ROUNDS_LIMIT
from llmsec.params import SAMPLERS as _SAMPLERS  # r7：单源（params.SAMPLERS）

# task_manager 经模块属性引用（测试 monkeypatch 拦截 start_task 的既有范式）
from llmsec.server import task_manager

_PHASES = ("all", "1", "2")


class LaunchError(ValueError):
    """启动参数/输入不合法。reason 供 HTTP 侧映射状态码：not_found→404，其余→400。"""

    def __init__(self, msg: str, *, reason: str = "invalid", hint: str = "") -> None:
        super().__init__(msg)
        self.reason = reason
        self.hint = hint


@dataclass
class LaunchSpec:
    """一次评估任务的完整参数（三方超集）。

    与 runner CLI 旗标的对应见 build_eval_argv；batch_size/twin_window/concurrency
    为 None 时不传旗标（runner 用自身默认/自适应），与 MCP 既有语义一致。
    """

    target: str | None = None
    targets: list[str] | None = None
    input_file: str = "attacks/l1.jsonl"
    phase: str = "all"
    batch_size: int | None = None
    max_rounds: int = 5
    sampler: str = "hybrid"
    # 采样器超参（web 端原有；MCP/TUI 此前不可达 → 归一补齐）
    sampler_alpha: float | None = None
    sampler_beta: float | None = None
    sampler_gamma: float | None = None
    coordinate_rounds: int | None = None
    seed: int | None = None
    twin_window: int | None = None
    no_early_stop: bool = False
    concurrency: int | None = None
    target_concurrency: int | None = None
    # 看板/MCP/TUI 评估默认 publish 到全局 R（runner CLI 本身默认关，此处显式开启）
    publish_global: bool = True
    # env 隔离（MCP 端原有；web/TUI 此前不可达 → 归一补齐）
    env_snapshot: str | None = None
    param_overrides: dict | None = None

    def target_names(self) -> list[str]:
        """归一后的目标名列表（单/多目标统一视角，供校验与 meta）。"""
        if self.target:
            return [self.target]
        return list(self.targets or [])


# ============================================================
# 校验与解析
# ============================================================
def validate_spec(spec: LaunchSpec) -> None:
    """参数校验（三方统一规则；错误消息沿用 MCP 既有文案，中文）。

    target/targets 都不传是合法的（runner 语义：跑全部 .env 声明目标）——
    更严的「必须指定」契约由个别调用面（如 MCP 工具）自行叠加。
    """
    if spec.target and spec.targets:
        raise LaunchError("target 与 targets 互斥，二选一（runner CLI 语义）")
    if spec.phase not in _PHASES:
        raise LaunchError(f"phase 须为 all/1/2，收到 {spec.phase!r}")
    if not (1 <= spec.max_rounds <= MAX_ROUNDS_LIMIT):
        raise LaunchError(f"max_rounds 须在 1-{MAX_ROUNDS_LIMIT}，收到 {spec.max_rounds}")
    if spec.sampler not in _SAMPLERS:
        raise LaunchError(f"sampler 须为 {'/'.join(_SAMPLERS)}，收到 {spec.sampler!r}")


def resolve_attack_file(input_file: str) -> Path:
    """解析攻击集为 ATTACKS_DIR 下的绝对路径。

    合并两套既有防御：取末段文件名防路径穿越（web + MCP 共有）+ .jsonl 后缀约束
    （web 原有，MCP 缺失 → 归一补齐）+ 存在性检查。
    """
    from llmsec.core.config import ATTACKS_DIR

    name = Path(input_file).name
    if not name.endswith(".jsonl"):
        raise LaunchError(f"input 必须是 .jsonl 文件名，收到 {input_file!r}")
    path = ATTACKS_DIR / name
    if not path.exists():
        raise LaunchError(
            f"攻击集不存在: {input_file}",
            reason="not_found",
            hint="可用攻击集在 attacks/ 目录下",
        )
    return path


def check_targets_declared(spec: LaunchSpec) -> None:
    """目标须在 .env TARGETS 中声明（web 端原有规则，原只查单目标 → 归一为单/多都查）。

    静默丢弃会张冠李戴（runner 按声明路由配置）。load_targets 失败/为空时无法
    校验，放行交由 runner 自身报错（与 web 既有兜底一致）。
    """
    from llmsec.core.config import load_targets

    try:
        declared = load_targets()
    except Exception:
        declared = {}
    if not declared:
        return
    bad = [n for n in spec.target_names() if n not in declared]
    if bad:
        raise LaunchError(
            f"目标未在 .env TARGETS 中声明: {', '.join(bad)}",
            reason="undeclared",
            hint="用 list_targets 查已声明目标",
        )


def attack_has_tax_probe(attack_path: Path) -> bool:
    """越狱税探针预检：读首条记录的 expected_answer（非 0/None 即含数学探针）。"""
    try:
        with open(attack_path, encoding="utf-8") as f:
            first_line = f.readline()
        if not first_line.strip():
            return False
        ea = json.loads(first_line).get("expected_answer")
        return ea not in (0, None)
    except Exception:
        return False


# ============================================================
# argv 构造（纯函数）
# ============================================================
def build_eval_argv(spec: LaunchSpec, *, attack_rel: str) -> list[str]:
    """构造 runner 子进程 argv（不含 python 可执行文件）。

    多目标未显式指定 target_concurrency 时默认全并发（每目标独立端点，无共享
    限速）——web/MCP 既有行为，归一固化于此。
    """
    argv = [
        "-m", "llmsec.pipeline.runner",
        "--phase", spec.phase,
        "--input", attack_rel,
        "--max-rounds", str(spec.max_rounds),
        "--sampler", spec.sampler,
    ]
    if spec.batch_size is not None:
        argv += ["--batch-size", str(spec.batch_size)]
    if spec.seed is not None:
        argv += ["--seed", str(spec.seed)]
    if spec.twin_window is not None:
        argv += ["--twin-window", str(spec.twin_window)]
    if spec.no_early_stop:
        argv += ["--no-early-stop"]
    if spec.concurrency is not None:
        argv += ["--concurrency", str(spec.concurrency)]
    for flag, v in (
        ("--sampler-alpha", spec.sampler_alpha),
        ("--sampler-beta", spec.sampler_beta),
        ("--sampler-gamma", spec.sampler_gamma),
        ("--coordinate-rounds", spec.coordinate_rounds),
    ):
        if v is not None:
            argv += [flag, str(v)]
    if spec.target:
        argv += ["--target", spec.target]
    elif spec.targets:
        argv += ["--targets", ",".join(spec.targets)]
        tc = spec.target_concurrency if spec.target_concurrency is not None else len(spec.targets)
        argv += ["--target-concurrency", str(tc)]
    elif spec.target_concurrency is not None:
        argv += ["--target-concurrency", str(spec.target_concurrency)]
    if spec.publish_global:
        argv += ["--publish-global"]
    return argv


def build_hpo_argv(yaml_path: Path) -> list[str]:
    """构造 study 运行 argv（hpo router / TUI 共用，替代两份重复实现）。"""
    return ["-m", "llmsec.experiments", "run", str(yaml_path)]


# ============================================================
# 启动入口（三方共用）
# ============================================================
def launch_evaluation(spec: LaunchSpec) -> dict:
    """提交一次评估任务：校验 → 解析攻击集 → argv → env 注入 → start_task。

    Returns:
        task_view dict（额外含 meta 结构化信息，随 task_view 一并暴露）。
    Raises:
        LaunchError: 任一校验不通过（调用方按各自协议翻译）。
    """
    validate_spec(spec)
    attack_path = resolve_attack_file(spec.input_file)
    check_targets_declared(spec)
    env_override = _load_env_override(spec)
    attack_rel = str(attack_path).replace("\\", "/")
    argv = build_eval_argv(spec, attack_rel=attack_rel)
    meta = {
        "targets": spec.target_names(),
        "max_rounds": spec.max_rounds,
        "input": attack_rel,
    }
    view = task_manager.start_task("evaluate", argv, env_override=env_override, meta=meta)
    if spec.env_snapshot:
        view["env_snapshot"] = spec.env_snapshot
    return view


def launch_hpo_study(yaml_path: str | Path) -> dict:
    """以 hpo 任务启动一个 study.yaml（与看板 POST /api/run/hpo 同一链路）。

    路径须存在、.yaml/.yml 后缀、且在仓库目录内（yaml 内容由 experiments
    schema 解析，路径外置文件无意义且有穿越面）。
    """
    from llmsec.core.config import PROJECT_ROOT

    p = Path(yaml_path).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p = p.resolve()
    if not p.exists() or p.suffix.lower() not in (".yaml", ".yml"):
        raise LaunchError(
            f"study 文件不存在: {yaml_path}",
            reason="not_found",
            hint="用 llmsec.tui.task_store.study_yamls() 查可用配置",
        )
    try:
        p.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        raise LaunchError("study 文件必须在仓库目录内") from None
    meta = {"study": p.name}
    return task_manager.start_task("hpo", build_hpo_argv(p), meta=meta)


def _load_env_override(spec: LaunchSpec) -> dict | None:
    """env_snapshot（隔离连接配置）+ param_overrides（LLMSEC_PARAM_*）→ 子进程 env 覆盖。"""
    env: dict | None = None
    if spec.env_snapshot:
        from control.core.env_snapshot import load_env_dict

        try:
            env = dict(load_env_dict(spec.env_snapshot))
        except FileNotFoundError:
            raise LaunchError(
                f"env 快照不存在: {spec.env_snapshot}",
                reason="not_found",
                hint="用 list_env_snapshots 查可用快照",
            ) from None
        except Exception as e:
            raise LaunchError(f"读取 env 快照失败: {e}") from None
    if spec.param_overrides:
        env = env or {}
        for k, v in spec.param_overrides.items():
            env[f"LLMSEC_PARAM_{k}"] = str(v)
    return env
