"""control.core.invoker — subprocess 调 llmsec CLI 的封装。

控制层与 llmsec 的唯一交互通道。每条命令封装为：
  - argv 构造（用 config.PYTHON + ``-m llmsec...``）
  - subprocess 执行（同步 / 后台 / 带超时）
  - 结果结构化（returncode / stdout / stderr / 解析后的 JSON）

参照 llmsec/experiments/executor.py 的调用模式，但控制层不 import llmsec 任何模块，
只经 CLI + 文件交互。所有命令默认 ``--json`` 拿结构化输出。

支持三类调用：
  - list_runs():        ``llmsec-manage runs list --json``
  - run_runner():       ``python -m llmsec.pipeline.runner --work-dir ... ``
  - manage_delete():    ``llmsec-manage runs delete ... --yes --json``
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from control.config import LLMSEC_REPO, PYTHON


@dataclass
class InvokeResult:
    """一次 llmsec CLI 调用的结果。"""
    argv: list[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    json: Any = None           # stdout 解析出的 JSON（若 --json）
    elapsed_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def require_ok(self) -> InvokeResult:
        """断言成功，否则抛 RuntimeError（带 stderr 摘要）。"""
        if not self.ok:
            raise RuntimeError(
                f"llmsec 命令失败 (rc={self.returncode}): {' '.join(self.argv)}\n"
                f"stderr: {self.stderr[-500:]}"
            )
        return self


def _run(
    argv: list[str],
    *,
    cwd: Path = LLMSEC_REPO,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
    log_file: Path | None = None,
) -> InvokeResult:
    """执行 argv，返回 InvokeResult。

    log_file 指定时 stdout/stderr 重定向到该文件（后台/批量任务用），不抓回内存；
    此前 run_runner 的 log_file 分支单独实现了一份 env 合并/超时/计时逻辑，
    现统一收口到本函数（原 capture 参数无人传 False，已移除）。
    """
    started = time.time()
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    res = InvokeResult(argv=argv, returncode=-1)
    try:
        if log_file is not None:
            with open(log_file, "w", encoding="utf-8") as f:
                proc = subprocess.run(
                    argv, cwd=str(cwd), env=full_env, timeout=timeout,
                    stdout=f, stderr=subprocess.STDOUT,
                )
            res.returncode = proc.returncode
        else:
            proc = subprocess.run(
                argv, cwd=str(cwd), env=full_env, timeout=timeout,
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            res.stdout = proc.stdout or ""
            res.stderr = proc.stderr or ""
            res.returncode = proc.returncode
            # 尝试解析 JSON（stdout 是纯 JSON 时）
            stripped = res.stdout.strip()
            if stripped.startswith("{") or stripped.startswith("["):
                try:
                    res.json = json.loads(stripped)
                except json.JSONDecodeError:
                    pass  # 混了日志行，不解析
    except subprocess.TimeoutExpired:
        res.returncode = -1
        res.stderr = f"超时（>{timeout}s）"
    res.elapsed_s = round(time.time() - started, 2)
    return res


# ============================================================
# 命令封装
# ============================================================
def _manage_argv(subcmd: list[str]) -> list[str]:
    """构造 llmsec-manage 调用 argv。subcmd 如 ['runs', 'list', '--json']。"""
    return [PYTHON, "-m", "llmsec.management", *subcmd]


def run_manage(subcmd: list[str], *, timeout: float | None = 600) -> InvokeResult:
    """执行 llmsec-manage 子命令（capabilities 等外部消费者的公共入口——
    不跨模块引用 _run/_manage_argv 私有实现）。"""
    return _run(_manage_argv(subcmd), timeout=timeout)


def list_runs(*, target: str | None = None, since: str | None = None,
              junk_only: bool = False) -> list[dict]:
    """列出 llmsec 的 run 历史（结构化）。经 ``llmsec-manage runs list --json``。"""
    sub = ["runs", "list", "--json"]
    if target:
        sub += ["--target", target]
    if since:
        sub += ["--since", since]
    if junk_only:
        sub += ["--junk-only"]
    # 超时保护：本函数会在门下省总线同步派发的回调里被调用，
    # 无超时的挂起子进程会卡死正在执行的整个 Plan 步骤与队列 worker
    res = _run(_manage_argv(sub), timeout=120).require_ok()
    data = res.json or {}
    return data.get("runs", []) if isinstance(data, dict) else data


def delete_runs(names: list[str], *, delete_r: bool = False) -> dict:
    """删除 run（软删除，已确认执行）。"""
    sub = ["runs", "delete", *names, "--yes", "--json"]
    if delete_r:
        sub.append("--delete-r")
    res = _run(_manage_argv(sub), timeout=600).require_ok()
    return res.json or {}


def clean_caches(categories: list[str]) -> dict:
    """清理缓存（经 llmsec-manage cache clean，已过门下省确认，强制 --yes）。"""
    sub = ["cache", "clean", *categories, "--yes", "--json"]
    res = _run(_manage_argv(sub), timeout=600).require_ok()
    return res.json or {}


def _anchor_input_file(input_file: str) -> str:
    """裸文件名锚定到 LLMSEC_REPO/attacks/（与 server/launch.py 同一口径）。

    runner 把相对 --input 锚 PROJECT_ROOT（仓库根）：用户自然语言/LLM 拟案常给
    裸文件名（"audit_small.jsonl"），原样透传必报"攻击集不存在"（P1-7）。
    launch 层早已做 attacks/ 归一化，控制层此前漏做。已带路径/绝对路径且存在则
    原样；都不存在也原样透传，交给 runner 报缺文件错误（保持既有错误语义）。
    """
    p = Path(input_file)
    if p.is_absolute():
        return input_file
    for cand in (LLMSEC_REPO / input_file, LLMSEC_REPO / "attacks" / input_file):
        if cand.exists():
            return cand.as_posix()
    return input_file


def run_runner(
    work_dir: Path,
    *,
    target: str | None = None,
    targets: list[str] | None = None,
    input_file: str = "attacks/l1.jsonl",
    max_rounds: int = 5,
    phase: str = "all",
    seed: int | None = None,
    env_override: dict[str, str] | None = None,
    timeout: float | None = None,
    log_file: Path | None = None,
) -> InvokeResult:
    """起一个 llmsec runner 工作单元（隔离 work-dir）。

    参照 experiments/executor.py：``python -m llmsec.pipeline.runner --work-dir ...``。
    work-dir 模式全局零污染（runner.py:183-194 重绑 state 路径）。
    log_file 指定时 stdout/stderr 重定向到文件（后台/批量用），不抓回内存。
    """
    argv = [
        PYTHON, "-m", "llmsec.pipeline.runner",
        "--work-dir", str(work_dir),
        "--input", _anchor_input_file(input_file),
        "--max-rounds", str(max_rounds),
        "--phase", phase,
        "--no-early-stop",  # work-dir 模式 runner 会强制，显式传便于阅读
    ]
    if target:
        argv += ["--target", target]
    if targets:
        argv += ["--targets", ",".join(targets)]
    if seed is not None:
        argv += ["--seed", str(seed)]

    env = {"PYTHONUNBUFFERED": "1"}
    if env_override:
        env.update(env_override)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
    return _run(argv, timeout=timeout, env=env, log_file=log_file)
