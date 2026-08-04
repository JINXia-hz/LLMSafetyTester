"""
experiments.executor — 单 trial 执行（隔离 subprocess + manifest + 指标提取）。

run_trial(config, seed, work_dir, study_config) → trial 记录 dict。
每个 trial：独立 work-dir，runner 子进程，不碰全局 state/results。
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from llmsec.experiments.manifest import capture_manifest
from llmsec.experiments.metrics import extract_metrics
from llmsec.experiments.schema import resolve_trial


def _runner_argv(work_dir: Path, seed: int, config: dict) -> tuple[list[str], dict[str, str]]:
    """构造 runner 子进程 argv 与 env 覆盖。config 的 input 应已归一化。"""
    argv_extra, env_override = resolve_trial(config)
    argv = [
        sys.executable, "-m", "llmsec.pipeline.runner",
        "--phase", "1",            # 实验只跑攻击阶段（收敛度量来自这里）
        "--work-dir", str(work_dir),
        "--seed", str(seed),
        *argv_extra,
    ]
    env_override = {**env_override, "PYTHONUNBUFFERED": "1"}  # 子进程实时刷出，便于观测
    return argv, env_override


def run_trial(
    config: dict,
    seed: int,
    work_dir: Path,
    study_name: str,
    trial_idx: int,
) -> dict:
    """
    执行一个 trial（单 seed）。返回记录 dict（含 metrics / status / 耗时）。

    config: {factor_name: value}（搜索值 ∪ fixed），含 input/target/max_rounds 等。
    """
    # input 路径归一化（无分隔符 → 补 attacks/），统一用于 argv 与 manifest
    config = dict(config)
    inp = config.get("input")
    if inp and "/" not in str(inp) and "\\" not in str(inp):
        config["input"] = f"attacks/{inp}"

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    argv, env_override = _runner_argv(work_dir, seed, config)
    capture_manifest(work_dir, argv, env_override, seed,
                     attack_set=_resolve_attack_set_path(config), config=config)

    log_path = work_dir / "runner.log"
    started = datetime.now()
    status = "running"
    returncode = None
    err = None
    try:
        full_env = {**__import__("os").environ, **env_override}
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT,
                                  env=full_env, cwd=str(Path(__file__).resolve().parents[2]))
        returncode = proc.returncode
        status = "success" if returncode == 0 else "failed"
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        status = "error"

    elapsed = (datetime.now() - started).total_seconds()
    metrics = extract_metrics(work_dir, max_rounds=_max_rounds_of(config)) if status != "error" else {}

    return {
        "study": study_name,
        "trial": trial_idx,
        "seed": seed,
        "params": config,
        "status": status,
        "returncode": returncode,
        "error": err,
        "elapsed_s": round(elapsed, 1),
        "metrics": metrics,
        "work_dir": str(work_dir),
    }


def _resolve_attack_set_path(config: dict) -> str | None:
    """从 config 的 input 因子解析攻击集绝对路径（manifest 记 hash 用）。"""
    rel = config.get("input")
    if not rel:
        return None
    from llmsec.core.config import OUTPUT_DIR
    p = Path(rel)
    return str(p if p.is_absolute() else OUTPUT_DIR / p)


def _max_rounds_of(config: dict) -> int | None:
    mr = config.get("max_rounds")
    try:
        return int(mr) if mr is not None else None
    except (TypeError, ValueError):
        return None
