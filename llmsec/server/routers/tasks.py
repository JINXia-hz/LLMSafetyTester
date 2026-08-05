"""任务管理路由：图形化触发生成 / 评估 / 聚类分析（子进程任务 + 状态轮询 / SSE 流）。"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from llmsec.core.config import ATTACKS_DIR, TASK_LOG_DIR
from llmsec.params import ADAPTIVE_BATCH_MAX

router = APIRouter()

# ============================================================
# 操作 API（子进程任务）
# ============================================================
# 子进程 cwd = 仓库根（dashboard_api 同级约定的 WORKSPACE_ROOT）。
# tasks.py 位于 llmsec/server/routers/，parents[3] 即仓库根。
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]

TASKS: dict[str, dict] = {}

# TASKS 上限：新任务入列时淘汰最旧的终态任务（running 不淘汰），防长期运行内存/句柄堆积
_TASKS_MAX = 64
_TERMINAL_STATUSES = ("success", "failed", "cancelled")


def _evict_tasks() -> None:
    """TASKS 超 _TASKS_MAX 时按插入序淘汰最旧的终态任务，并确保其日志句柄关闭。"""
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


class EvaluateRequest(BaseModel):
    phase: str = Field(default="all", pattern="^(all|1|2)$")
    input: str = "l1.jsonl"
    # runner._adaptive_batch_size 会把 batch 压到 [ADAPTIVE_BATCH_MIN, ADAPTIVE_BATCH_MAX] 内，
    # 上限与 runner 对齐，避免用户传 >ADAPTIVE_BATCH_MAX 时被静默压回；默认值随上限自适应
    batch_size: int = Field(default=min(10, ADAPTIVE_BATCH_MAX), ge=1, le=ADAPTIVE_BATCH_MAX)
    max_rounds: int = Field(default=5, ge=1, le=50)
    sampler: str = Field(default="hybrid", pattern="^(gap|infogain|coordinate|hybrid)$")
    # 目标模型（.env TARGETS 中声明的名字）；None = .env 默认目标。
    # pattern 防异常字符（argv 以列表传递不走 shell，仍做白名单校验）
    target: str | None = Field(default=None, pattern=r"^[\w.\-:]+$")


def _refresh_task_status(t: dict) -> None:
    """刷新任务状态：子进程已结束但 status 仍为 running 时更新为 success/failed，
    并关闭 log_file 句柄（置 None）。

    子进程可能崩溃且无人轮询接口，若只在 _task_view 里更新状态，
    TASKS 中会残留永久 running 的任务（导致 _start_task 的 409 检查误拒同类新任务），
    log_file 句柄也随 TASKS 常驻泄漏。
    """
    if t["status"] != "running":
        return
    proc: subprocess.Popen = t["proc"]
    rc = proc.poll()
    if rc is None:
        return
    t["status"] = "success" if rc == 0 else "failed"
    t["returncode"] = rc
    log_file = t.get("log_file")
    if log_file is not None:
        log_file.close()
        t["log_file"] = None


def _task_view(task_id: str, t: dict) -> dict:
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
    }


def _start_task(kind: str, argv: list[str]) -> dict:
    # 先刷新所有 running 任务的真实状态，避免子进程崩溃后无人轮询
    # 导致 status 永久 running（409 误拒同类新任务）与 log_file 句柄泄漏
    for t in TASKS.values():
        _refresh_task_status(t)
    for tid, t in TASKS.items():
        if t["kind"] == kind and t["status"] == "running":
            raise HTTPException(status_code=409, detail=f"{kind} 任务正在运行中 (id={tid})")

    TASK_LOG_DIR.mkdir(parents=True, exist_ok=True)
    task_id = f"{kind}-{datetime.now().strftime('%H%M%S')}-{uuid.uuid4().hex[:6]}"
    log_path = TASK_LOG_DIR / f"{task_id}.log"
    log_file = open(log_path, "w", encoding="utf-8")

    try:
        proc = subprocess.Popen(
            [sys.executable, *argv],
            cwd=WORKSPACE_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
    except OSError as e:
        log_file.close()
        raise HTTPException(status_code=500, detail=f"任务启动失败: {e}")

    TASKS[task_id] = {
        "kind": kind,
        "cmd": " ".join(argv),
        "proc": proc,
        "log_path": log_path,
        "log_file": log_file,
        "status": "running",
        "started_at": datetime.now().isoformat(),
    }
    _evict_tasks()
    return _task_view(task_id, TASKS[task_id])


@router.post("/api/run/generate")
async def api_run_generate():
    return _start_task("generate", ["-m", "llmsec.attacks.generate"])


@router.post("/api/run/evaluate")
async def api_run_evaluate(req: EvaluateRequest):
    # input 只允许 output/attacks/ 下的 jsonl 文件名，防路径穿越
    input_name = Path(req.input).name
    if not input_name.endswith(".jsonl"):
        raise HTTPException(status_code=400, detail="input 必须是 .jsonl 文件名")
    if not (ATTACKS_DIR / input_name).exists():
        raise HTTPException(status_code=404, detail=f"攻击集不存在: attacks/{input_name}")

    argv = [
        "-m", "llmsec.pipeline.runner",
        "--phase", req.phase,
        "--input", f"attacks/{input_name}",
        "--batch-size", str(req.batch_size),
        "--max-rounds", str(req.max_rounds),
        "--sampler", req.sampler,
    ]
    if req.target:
        # 目标须在 .env TARGETS 已声明，否则 400（静默丢弃会张冠李戴）；
        # load_targets 失败/为空时无法校验，放行交由 runner 自身报错
        from llmsec.core.config import load_targets
        try:
            declared = load_targets()
        except Exception:
            declared = {}
        if declared and req.target not in declared:
            raise HTTPException(status_code=400, detail=f"目标未在 TARGETS 中声明: {req.target!r}")
        argv += ["--target", req.target]
    return _start_task("evaluate", argv)


@router.post("/api/run/cluster-analysis")
async def api_run_cluster_analysis():
    return _start_task("cluster-analysis", ["-m", "llmsec.evaluation.cluster_analysis"])


@router.get("/api/tasks")
async def api_tasks():
    return {"tasks": [_task_view(tid, t) for tid, t in sorted(TASKS.items(), reverse=True)]}


@router.get("/api/tasks/{task_id}")
async def api_task(task_id: str):
    t = TASKS.get(task_id)
    if t is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return _task_view(task_id, t)


@router.get("/api/tasks/{task_id}/log")
async def api_task_log(task_id: str, download: bool = False):
    """完整任务日志（log_tail 只有尾部 4KB；任务失败后看完整上下文用此接口）。

    ?download=1 时以 text/plain + Content-Disposition 返回，便于直接下载 .log。
    """
    t = TASKS.get(task_id)
    if t is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    log_path: Path = t["log_path"]
    text = ""
    if log_path.exists():
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
    if download:
        return PlainTextResponse(
            text,
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{task_id}.log"'},
        )
    return {"id": task_id, "log": text}


@router.post("/api/tasks/{task_id}/cancel")
async def api_task_cancel(task_id: str):
    """取消运行中的任务：SIGTERM → 5s 宽限 → SIGKILL，置 cancelled 状态。

    Windows 无 SIGTERM 语义：Popen.terminate 即 TerminateProcess 强杀，宽限期仅对 POSIX 有效。
    proc.wait 经 asyncio.to_thread 包裹，避免同步等待阻塞事件循环。
    runner 每场攻击实时 upsert 进 R，故取消后已观测的结果保留在结果矩阵中。
    已结束的任务返回 409。
    """
    t = TASKS.get(task_id)
    if t is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    _refresh_task_status(t)
    if t["status"] != "running":
        raise HTTPException(status_code=409, detail=f"任务已结束（{t['status']}），无法取消")
    proc: subprocess.Popen = t["proc"]
    proc.terminate()
    try:
        await asyncio.to_thread(proc.wait, timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        await asyncio.to_thread(proc.wait)
    t["status"] = "cancelled"
    t["returncode"] = proc.returncode
    log_file = t.get("log_file")
    if log_file is not None:
        log_file.close()
        t["log_file"] = None
    return _task_view(task_id, t)


@router.get("/api/tasks/{task_id}/stream")
async def api_task_stream(task_id: str):
    """SSE 实时日志流：连接时先吐尾部 2KB 上下文，之后跟随新增字节（直播）。

    子进程结束时发一个 event:done（携带 status/returncode）再关闭，前端据此
    停止跟随并刷新数据。取代运行控制页 2~3s 轮询 log_tail 的"看监控录像"体验。
    """
    t = TASKS.get(task_id)
    if t is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    log_path: Path = t["log_path"]

    async def event_gen():
        buf = ""

        def take_lines(text: str) -> list[str]:
            """把文本切成完整行返回，末尾不完整段留作 buf 等下次拼齐。"""
            nonlocal buf
            buf += text
            parts = buf.split("\n")
            buf = parts.pop()  # 最后一段可能不完整，保留
            return parts

        offset = 0
        # 连接初始上下文：尾部 2KB（起始半行丢弃，避免半行噪音）
        if log_path.exists():
            try:
                size = log_path.stat().st_size
                head = max(0, size - 2048)
                with open(log_path, encoding="utf-8", errors="replace") as f:
                    f.seek(head)
                    if head > 0:
                        f.readline()  # 丢弃起始半行
                    init = f.read()
                offset = size
            except OSError:
                init = ""
            for line in take_lines(init):
                yield f"data: {line}\n\n"

        while True:
            if log_path.exists():
                try:
                    size = log_path.stat().st_size
                except OSError:
                    size = offset
                if size > offset:
                    try:
                        with open(log_path, encoding="utf-8", errors="replace") as f:
                            f.seek(offset)
                            chunk = f.read(size - offset)
                        offset = size
                    except OSError:
                        chunk = ""
                    for line in take_lines(chunk):
                        yield f"data: {line}\n\n"
                elif size < offset:
                    # 文件被截断/轮转，重置偏移跟随新内容
                    offset = size
            _refresh_task_status(t)
            if t["status"] != "running":
                # 刷出残留 buffer 后发结束事件
                if buf:
                    yield f"data: {buf}\n\n"
                    buf = ""
                yield (
                    "event: done\ndata: "
                    + json.dumps(
                        {"status": t["status"], "returncode": t.get("returncode")},
                        ensure_ascii=False,
                    )
                    + "\n\n"
                )
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
