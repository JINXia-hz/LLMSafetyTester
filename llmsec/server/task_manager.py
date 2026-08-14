"""llmsec.server.task_manager — 框架无关的子进程任务管理 core。

从 server/routers/tasks.py 抽出，供 FastAPI router 和 MCP server 共用。
不含任何 HTTP/FastAPI 依赖，纯同步逻辑 + subprocess。

职责：
  - 维护全局 TASKS dict（task_id → 状态 + Popen 句柄）
  - 按 kind 串行执行（同 kind 有 running 时排队）
  - 状态刷新（poll 子进程终态）
  - 日志/进度读取
  - 取消（SIGTERM → SIGKILL）

线程安全说明：TASKS 在 FastAPI 和 MCP 场景下都是单进程内共享。
FastAPI 的 endpoint 是 async 但这些函数是同步的（cancel 在 router 里用 to_thread 包装）。
MCP 的 call_tool 同理。由于 GIL + 操作本身简短，未加显式锁。
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

from llmsec.core.config import TASK_LOG_DIR

# 子进程 cwd = 仓库根（task_manager.py 位于 llmsec/server/，parents[2] 即仓库根）
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

# 全局任务注册表（task_id → dict）
TASKS: dict[str, dict] = {}

# TASKS 上限：新任务入列时淘汰最旧的终态任务（running 不淘汰）
_TASKS_MAX = 64
_TERMINAL_STATUSES = ("success", "failed", "cancelled")

# 僵尸任务检测：running 超 N 分钟无 progress 产出则告警（不自动杀，避免误杀慢任务）
_ZOMBIE_MINUTES = float(os.getenv("LLMSEC_ZOMBIE_MINUTES", "60") or "60")


# ============================================================
# 内部
# ============================================================
def _evict_tasks() -> None:
    """TASKS 超 _TASKS_MAX 时按插入序淘汰最旧的终态任务，并关闭其日志句柄。"""
    while len(TASKS) > _TASKS_MAX:
        victim = next(
            (tid for tid, t in TASKS.items() if t["status"] in _TERMINAL_STATUSES),
            None,
        )
        if victim is None:
            break  # 全是 running，不淘汰
        t = TASKS.pop(victim)
        log_file = t.get("log_file")
        if log_file is not None:
            log_file.close()


def _refresh_task_status(t: dict) -> None:
    """刷新任务状态：子进程已结束但 status 仍为 running 时更新为 success/failed。

    子进程可能崩溃且无人轮询，若不在每次 _task_view 里更新状态，TASKS 会残留
    永久 running 的任务（阻塞同 kind 队列推进、log_file 句柄泄漏）。

    告警：
      - 终态=failed → emit_alert（监控设施 try/except 兜底，绝不影响状态刷新）
      - running 超 _ZOMBIE_MINUTES 无产出 → 僵尸告警（告警但不自动杀）
    """
    if t["status"] != "running":
        return
    proc: subprocess.Popen = t["proc"]
    rc = proc.poll()
    if rc is None:
        # 进程仍活着：检查僵尸态（超时无产出）
        _check_zombie(t)
        return
    t["status"] = "success" if rc == 0 else "failed"
    t["returncode"] = rc
    log_file = t.get("log_file")
    if log_file is not None:
        log_file.close()
        t["log_file"] = None
    # 失败告警（监控设施故障不影响状态机）
    if t["status"] == "failed":
        try:
            from llmsec.core.monitoring import alert_task_failed

            alert_task_failed(
                task_id=t.get("_task_id", "?"),
                kind=t["kind"],
                cmd=t.get("cmd", ""),
                log_path=str(t["log_path"]),
                returncode=rc,
            )
        except Exception:
            pass
    _advance_queue(t["kind"])


def _check_zombie(t: dict) -> None:
    """僵尸任务检测：running 超 _ZOMBIE_MINUTES 且 progress.jsonl 无新写入则告警。

    告警但不自动杀（避免误杀慢任务）。每个任务只告警一次（靠 monitoring 去抖）。
    """
    try:
        spawned = t.get("spawned_at")
        if spawned is None:
            return
        running_minutes = (datetime.now() - spawned).total_seconds() / 60.0
        if running_minutes < _ZOMBIE_MINUTES:
            return
        # progress.jsonl 最近修改时间（无文件或无更新视为僵尸）
        prog_path = _progress_path(t.get("_task_id", ""))
        if prog_path.exists():
            mtime = datetime.fromtimestamp(prog_path.stat().st_mtime)
            idle_minutes = (datetime.now() - mtime).total_seconds() / 60.0
            if idle_minutes < _ZOMBIE_MINUTES:
                return  # progress 近期有更新，不是僵尸
        from llmsec.core.monitoring import alert_zombie_task

        alert_zombie_task(
            task_id=t.get("_task_id", "?"),
            kind=t["kind"],
            cmd=t.get("cmd", ""),
            running_minutes=running_minutes,
        )
    except Exception:
        pass


def _spawn(task_id: str, t: dict) -> None:
    """启动已入队任务的子进程（打开日志、Popen、置 running）。Popen 失败置 failed。"""
    TASK_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = open(t["log_path"], "w", encoding="utf-8")
    try:
        env = os.environ.copy()
        env["LLMSEC_TASK_ID"] = task_id
        env["PYTHONUNBUFFERED"] = "1"
        # env_override（来自 env_snapshot）：注入隔离的连接配置
        env_override = t.get("env_override")
        if env_override:
            env.update(env_override)
        proc = subprocess.Popen(
            [sys.executable, *t["argv"]],
            cwd=WORKSPACE_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )
    except OSError as e:
        log_file.close()
        t["status"] = "failed"
        t["returncode"] = -1
        t["error"] = f"任务启动失败: {e}"
        return
    t["proc"] = proc
    t["log_file"] = log_file
    t["status"] = "running"
    t["spawned_at"] = datetime.now()


def _advance_queue(kind: str) -> None:
    """该 kind 无 running 任务时，启动最早的 queued 任务（FIFO）。"""
    if any(t["kind"] == kind and t["status"] == "running" for t in TASKS.values()):
        return
    for tid, t in TASKS.items():
        if t["kind"] == kind and t["status"] == "queued":
            _spawn(tid, t)
            return


def _progress_path(task_id: str) -> Path:
    """<task_id>.progress.jsonl 路径（与 .log 同目录）。"""
    return TASK_LOG_DIR / f"{task_id}.progress.jsonl"


# ============================================================
# 公开 API
# ============================================================
def start_task(kind: str, argv: list[str], *, env_override: dict[str, str] | None = None) -> dict:
    """入队一个新任务并返回其 task_view。

    同 kind 有 running 任务时排队，否则立即启动。

    Args:
        kind: 任务类型（"evaluate" / "hpo"），用于串行队列分组。
        argv: 子进程参数（不含 python 可执行文件，会自动加 sys.executable）。
        env_override: 注入子进程的环境变量（来自 env_snapshot，覆盖全局 .env 的同名 key）。

    Returns:
        task_view dict（id/kind/cmd/status/started_at/log_tail/...）。
    """
    # 先刷新所有 running 任务的真实状态
    for t in TASKS.values():
        _refresh_task_status(t)

    task_id = f"{kind}-{datetime.now().strftime('%H%M%S')}-{uuid.uuid4().hex[:6]}"
    TASKS[task_id] = {
        "kind": kind,
        "cmd": " ".join(argv),
        "argv": argv,
        "env_override": env_override,
        "proc": None,
        "log_path": TASK_LOG_DIR / f"{task_id}.log",
        "log_file": None,
        "status": "queued",
        "started_at": datetime.now().isoformat(),
        "_task_id": task_id,  # 供告警/僵尸检测引用
    }
    _evict_tasks()
    _advance_queue(kind)
    return task_view(task_id)


def task_view(task_id: str) -> dict | None:
    """返回任务的视图 dict（刷新状态 + log_tail）。task 不存在返回 None。"""
    t = TASKS.get(task_id)
    if t is None:
        return None
    _refresh_task_status(t)
    status = t["status"]
    rc = t.get("returncode")
    log_tail = ""
    log_path: Path = t["log_path"]
    if log_path.exists():
        try:
            log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        except OSError:
            pass
    return {
        "id": task_id,
        "kind": t["kind"],
        "cmd": t["cmd"],
        "status": status,
        "returncode": rc,
        "started_at": t["started_at"],
        "log_tail": log_tail,
        "error": t.get("error"),
    }


def list_tasks() -> list[dict]:
    """列出全部任务的视图（时间倒序）。"""
    return [
        task_view(tid)
        for tid, _ in sorted(TASKS.items(), reverse=True)
    ]


def read_progress(task_id: str) -> list[dict]:
    """读取任务的 progress.jsonl 全部记录。文件不存在返回 []。"""
    import json

    p = _progress_path(task_id)
    if not p.exists():
        return []
    out: list[dict] = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def read_full_log(task_id: str) -> str:
    """读取任务完整日志（task_view 只有尾部 4KB）。task 不存在返回空串。"""
    t = TASKS.get(task_id)
    if t is None:
        return ""
    log_path: Path = t["log_path"]
    if not log_path.exists():
        return ""
    try:
        return log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def cancel_task(task_id: str) -> dict | None:
    """取消排队中或运行中的任务，置 cancelled。

    queued：直接标记取消。running：SIGTERM → 5s 宽限 → SIGKILL。
    Windows 无 SIGTERM 语义（Popen.terminate 即强杀）。
    已结束的任务返回 None（调用方应判断 status）。

    Returns:
        取消后的 task_view；task 不存在或已结束返回 None。
    """

    t = TASKS.get(task_id)
    if t is None:
        return None
    _refresh_task_status(t)
    if t["status"] not in ("running", "queued"):
        return None  # 已结束
    if t["status"] == "queued":
        t["status"] = "cancelled"
        return task_view(task_id)
    proc: subprocess.Popen = t["proc"]
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    t["status"] = "cancelled"
    t["returncode"] = proc.returncode
    log_file = t.get("log_file")
    if log_file is not None:
        log_file.close()
        t["log_file"] = None
    _advance_queue(t["kind"])
    return task_view(task_id)
