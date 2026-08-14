"""llmsec.mcp.tools.tasks — Tier 4 长任务工具（异步提交 + 轮询）。

run_evaluation 这类评估跑几分钟到几十分钟（调 LLM API），不能在单个 MCP tool call 里
同步阻塞。采用"提交 → 返回 task_id → 轮询状态"的异步模式。

MCP server 有自己独立的 task 队列（llmsec.server.task_manager），与 HTTP dashboard 的
task 队列互不干扰——各自管各自的子进程。

暴露的工具：
  - run_evaluation       提交一次评估，返回 task_id
  - get_task_status      查任务状态 + log_tail
  - get_task_progress    查进度快照（逐目标/逐轮）
  - get_task_log         读完整日志
  - cancel_task          取消排队中/运行中的任务
  - list_tasks           列出所有任务
"""

from __future__ import annotations

from typing import Any


# ============================================================
# 辅助
# ============================================================
def _try(fn, *, error_hint: str = "") -> Any:
    try:
        return fn()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "hint": error_hint}


def _build_eval_argv(
    target: str | None = None,
    targets: list[str] | None = None,
    input_file: str = "attacks/l1.jsonl",
    max_rounds: int = 5,
    phase: str = "all",
    batch_size: int | None = None,
    sampler: str = "hybrid",
    seed: int | None = None,
    twin_window: int | None = None,
    no_early_stop: bool = False,
    concurrency: int | None = None,
) -> list[str]:
    """构造 runner 的 argv（不含 python 可执行文件）。"""
    argv = [
        "-m", "llmsec.pipeline.runner",
        "--phase", phase,
        "--input", input_file,
        "--max-rounds", str(max_rounds),
        "--sampler", sampler,
    ]
    if batch_size is not None:
        argv += ["--batch-size", str(batch_size)]
    if seed is not None:
        argv += ["--seed", str(seed)]
    if twin_window is not None:
        argv += ["--twin-window", str(twin_window)]
    if no_early_stop:
        argv += ["--no-early-stop"]
    if concurrency is not None:
        argv += ["--concurrency", str(concurrency)]
    if target:
        argv += ["--target", target]
    elif targets:
        argv += ["--targets", ",".join(targets)]
        # 多目标默认全并发（每目标独立端点）
        argv += ["--target-concurrency", str(len(targets))]
    # 看板评估默认 publish 到全局 R
    argv += ["--publish-global"]
    return argv


# ============================================================
# 工具函数
# ============================================================
def run_evaluation(
    target: str | None = None,
    targets: list[str] | None = None,
    input_file: str = "attacks/l1.jsonl",
    max_rounds: int = 5,
    phase: str = "all",
    batch_size: int | None = None,
    sampler: str = "hybrid",
    seed: int | None = None,
    env_snapshot: str | None = None,
    twin_window: int | None = None,
    no_early_stop: bool = False,
    concurrency: int | None = None,
) -> dict[str, Any]:
    """提交一次 llmsec 安全评估任务（异步，立即返回 task_id）。

    评估会跑自适应攻击流水线（几分钟到几十分钟，取决于 API 延迟和轮数）。
    本工具不阻塞等待——返回 task_id 后用 get_task_status 轮询进度。

    连接配置（GENERATOR_*/TARGET_*/JUDGE_*）默认从全局 .env 读取。如果指定了
    env_snapshot，则用快照里的配置覆盖全局（隔离评估，不碰全局 .env）。
    用 create_env_snapshot + edit_env_snapshot 创建和编辑快照。

    Args:
        target:       单个目标模型名（与 targets 二选一）。须在 .env TARGETS 中声明。
        targets:      多目标列表（与 target 二选一），默认全并发。
        input_file:   攻击集文件路径（相对仓库根，如 "attacks/l1.jsonl"）。
        max_rounds:   自适应最大轮数（1-50，默认 5）。
        phase:        评估阶段："all"（默认）/"1"（仅攻击）/"2"（仅过敏）。
        batch_size:   每轮批量大小（不传用默认自适应策略）。
        sampler:      采样策略："hybrid"（默认）/"gap"/"infogain"/"coordinate"。
        seed:         随机种子（可复现）。
        env_snapshot: .env 快照名。指定时用快照里的连接配置覆盖全局 .env（隔离评估）。
                      用 create_env_snapshot 创建快照。
        twin_window:  过敏检测方法数上限（控制过敏阶段成本）。None 用默认自适应。
        no_early_stop: 跑满 max_rounds 不提前停止（实验可比性，固定预算下 ci_half 可比）。
        concurrency:  批内并行度（每轮同时发起多少个攻击请求）。None 用默认全并发。

    Returns:
        task_view dict（含 id/status/started_at）。用 id 轮询 get_task_status。
    """
    from pathlib import Path

    from llmsec.core.config import ATTACKS_DIR
    from llmsec.params import MAX_ROUNDS_LIMIT

    def _do() -> dict[str, Any]:
        # 参数校验
        if not target and not targets:
            return {"error": "必须指定 target 或 targets 之一"}
        if phase not in ("all", "1", "2"):
            return {"error": f"phase 须为 all/1/2，收到 {phase!r}"}
        if not (1 <= max_rounds <= MAX_ROUNDS_LIMIT):
            return {"error": f"max_rounds 须在 1-{MAX_ROUNDS_LIMIT}，收到 {max_rounds}"}
        if sampler not in ("gap", "infogain", "coordinate", "hybrid"):
            return {"error": f"sampler 须为 gap/infogain/coordinate/hybrid，收到 {sampler!r}"}

        # 攻击集存在性检查
        resolved_input = input_file
        attack_path = Path(input_file)
        if not attack_path.is_absolute():
            attack_path = Path(".") / input_file
        if not attack_path.exists():
            # 也检查 ATTACKS_DIR
            alt = ATTACKS_DIR / Path(input_file).name
            if alt.exists():
                resolved_input = str(alt).replace("\\", "/")
            else:
                return {"error": f"攻击集不存在: {input_file}", "hint": "可用攻击集在 attacks/ 目录下"}

        # 加载 env_snapshot（如果指定）
        env_override = None
        if env_snapshot:
            try:
                from control.core.env_snapshot import load_env_dict

                env_override = load_env_dict(env_snapshot)
            except FileNotFoundError:
                return {"error": f"env 快照不存在: {env_snapshot}", "hint": "用 list_env_snapshots 查可用快照"}
            except Exception as e:
                return {"error": f"读取 env 快照失败: {e}"}

        argv = _build_eval_argv(
            target=target, targets=targets, input_file=resolved_input,
            max_rounds=max_rounds, phase=phase, batch_size=batch_size,
            sampler=sampler, seed=seed,
            twin_window=twin_window, no_early_stop=no_early_stop, concurrency=concurrency,
        )
        from llmsec.server.task_manager import start_task

        view = start_task("evaluate", argv, env_override=env_override)
        view["next_step"] = "用 get_task_status(task_id) 轮询进度，status 为 success/failed/cancelled 时结束"
        if env_snapshot:
            view["env_snapshot"] = env_snapshot
        return view

    return _try(_do, error_hint="检查 .env 是否配置了 GENERATOR_* 和 TARGET_*，或用 env_snapshot 参数指定隔离配置")


def get_task_status(task_id: str) -> dict[str, Any] | None:
    """查询任务当前状态（含日志尾部 4KB）。

    Args:
        task_id: run_evaluation 返回的任务 id。

    Returns:
        task_view dict（id/status/returncode/log_tail/...）；task 不存在返回 None。
        status: queued（排队）/ running（执行中）/ success / failed / cancelled。
    """
    from llmsec.server.task_manager import task_view

    return task_view(task_id)


def get_task_progress(task_id: str) -> dict[str, Any]:
    """查询任务的进度快照（逐目标/逐轮的实时数据）。

    比 get_task_status 更详细——包含每个目标的当前轮次、ASR、Elo 变化等。
    数据来自子进程写入的 progress.jsonl。

    Args:
        task_id: 任务 id。

    Returns:
        {kind, status, targets, max_rounds, progress: {target: {...}}}。
    """
    from llmsec.server.task_manager import read_progress, task_view

    def _do() -> dict[str, Any]:
        view = task_view(task_id)
        if view is None:
            return {"error": f"任务不存在: {task_id}"}
        records = read_progress(task_id)
        # 每目标取最后一条进度
        by_target: dict[str, dict] = {}
        for r in records:
            tg = r.get("target")
            if tg:
                by_target[tg] = r
        return {
            "kind": view["kind"],
            "status": view["status"],
            "progress": by_target,
            "log_tail": view.get("log_tail", ""),
        }

    return _try(_do)


def get_task_log(task_id: str) -> dict[str, Any]:
    """读取任务的完整日志（task_view 只有尾部 4KB）。

    任务失败后看完整上下文用此工具。

    Args:
        task_id: 任务 id。

    Returns:
        {id, log}，log 为完整日志文本。
    """
    from llmsec.server.task_manager import read_full_log

    log = read_full_log(task_id)
    if not log:
        return {"id": task_id, "log": "", "note": "日志为空（任务不存在或尚未产生输出）"}
    return {"id": task_id, "log": log}


def cancel_task(task_id: str) -> dict[str, Any]:
    """取消排队中或运行中的任务。

    queued 状态直接取消；running 状态发 SIGTERM（5s 宽限后 SIGKILL）。
    已结束的任务（success/failed/cancelled）无法取消，返回错误提示。

    取消后已观测的评估结果保留在 R 矩阵中（runner 每场攻击实时 upsert）。

    Args:
        task_id: 要取消的任务 id。

    Returns:
        取消后的 task_view；task 不存在或已结束返回相应错误。
    """
    from llmsec.server.task_manager import cancel_task as _cancel

    def _do() -> dict[str, Any]:
        view = _cancel(task_id)
        if view is None:
            # 区分"不存在"和"已结束"
            from llmsec.server.task_manager import task_view

            if task_view(task_id) is None:
                return {"error": f"任务不存在: {task_id}"}
            return {"error": "任务已结束，无法取消", "current_status": task_view(task_id)["status"]}
        return view

    return _try(_do)


def list_tasks() -> list[dict[str, Any]]:
    """列出当前所有任务的状态（时间倒序）。

    Returns:
        task_view dict 列表。
    """
    from llmsec.server.task_manager import list_tasks as _lt

    return _lt()


# ============================================================
# orchestrate — 批量并行实验（异步 task）
# ============================================================
def orchestrate_runs(
    specs: list[dict[str, Any]],
    max_workers: int = 2,
    compare_after: bool = True,
) -> dict[str, Any]:
    """提交批量并行评估任务（A/B 对比 / 参数扫描）。

    每个 spec 会 fork 一个隔离工作区并在其中跑 runner，全部并行执行。
    结束后可选自动生成对比报告。立即返回 task_id，不阻塞等待。

    Args:
        specs:          工作单元规格列表，每条含：
            - name (必须): workspace 名（唯一）
            - target:      目标模型名
            - source:      fork 来源（默认 "global"）
            - input_file:  攻击集（默认 "attacks/l1.jsonl"）
            - max_rounds:  最大轮数（默认 5）
            - seed:        随机种子
            - note:        备注
        max_workers:    并行度（同时跑多少个 runner 子进程，默认 2）。
        compare_after:  全部完成后是否自动跑 compare。

    Returns:
        task_view dict（含 id/status）。用 get_task_status 轮询进度。
    """
    import json as _json

    from llmsec.server.task_manager import start_task

    def _do() -> dict[str, Any]:
        if not specs:
            return {"error": "specs 不能为空"}
        for i, s in enumerate(specs):
            if not s.get("name"):
                return {"error": f"specs[{i}] 缺少 name 字段"}

        # 构造一个 python -c 脚本，让 task_manager 子进程执行 orchestrate
        script = (
            "from control.core.orchestrator import orchestrate, RunSpec\n"
            "import json, sys\n"
            f"specs_data = {repr(_json.dumps(specs))}\n"
            f"max_workers = {max_workers}\n"
            f"compare_after = {compare_after}\n"
            "specs = [RunSpec(**s) for s in json.loads(specs_data)]\n"
            "result = orchestrate(specs, max_workers=max_workers, compare_after=compare_after)\n"
            "print(json.dumps(result, ensure_ascii=False, default=str))\n"
        )
        argv = ["-c", script]
        view = start_task("orchestrate", argv)
        view["next_step"] = "用 get_task_status(task_id) 轮询进度"
        return view

    return _try(_do, error_hint="检查 specs 格式（每条须含 name 字段）")


# ============================================================
# 注册
# ============================================================
def register(mcp: Any) -> None:
    """把本模块所有工具注册到 FastMCP server。"""
    mcp.tool(run_evaluation)
    mcp.tool(get_task_status)
    mcp.tool(get_task_progress)
    mcp.tool(get_task_log)
    mcp.tool(cancel_task)
    mcp.tool(list_tasks)
    mcp.tool(orchestrate_runs)
