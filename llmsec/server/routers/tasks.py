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

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from llmsec.core.config import ATTACKS_DIR, TASK_LOG_DIR
from llmsec.params import ADAPTIVE_BATCH_MAX, DEFAULT_BATCH_SIZE, DEFAULT_MAX_ROUNDS, MAX_ROUNDS_LIMIT

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
    batch_size: int = Field(default=min(DEFAULT_BATCH_SIZE, ADAPTIVE_BATCH_MAX), ge=1, le=ADAPTIVE_BATCH_MAX)
    max_rounds: int = Field(default=DEFAULT_MAX_ROUNDS, ge=1, le=MAX_ROUNDS_LIMIT)
    sampler: str = Field(default="hybrid", pattern="^(gap|infogain|coordinate|hybrid)$")
    # 采样器权重（None = 用 params 默认值，不传 --xxx 旗标给 runner）
    sampler_alpha: float | None = None
    sampler_beta: float | None = None
    sampler_gamma: float | None = None
    coordinate_rounds: int | None = None
    target: str | None = Field(default=None, pattern=r"^[\w.\-:]+$")
    targets: str | None = None      # 多目标子集（逗号分隔，前端探活后只传可达的）
    # 多目标并发数：None + 多目标 → 默认全并发（每个目标是独立端点，无共享限速）
    target_concurrency: int | None = Field(default=None, ge=1, le=32)


def _refresh_task_status(t: dict) -> None:
    """刷新任务状态：子进程已结束但 status 仍为 running 时更新为 success/failed，
    并关闭 log_file 句柄（置 None）。

    子进程可能崩溃且无人轮询，若不在每次 _task_view 里更新状态，TASKS 会残留永久
    running 的任务（阻塞同 kind 队列推进、log_file 句柄常驻泄漏）。
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
    _advance_queue(t["kind"])   # running→终态，推进该 kind 的队列


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
        "error": t.get("error"),
    }


def _spawn(task_id: str, t: dict) -> None:
    """启动已入队任务的子进程（打开日志、Popen、置 running）。Popen 失败置 failed。"""
    TASK_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = open(t["log_path"], "w", encoding="utf-8")
    try:
        # 注入 LLMSEC_TASK_ID：子进程（runner/attack_phase、experiments/study）据此
        # 把逐轮/逐 trial 进度落到 output/tasks/<task_id>.progress.jsonl，供看板消费。
        env = os.environ.copy()
        env["LLMSEC_TASK_ID"] = task_id
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


def _advance_queue(kind: str) -> None:
    """该 kind 无 running 任务时，启动最早的 queued 任务（FIFO）。"""
    if any(t["kind"] == kind and t["status"] == "running" for t in TASKS.values()):
        return
    for tid, t in TASKS.items():  # dict 保序，最早入队在前
        if t["kind"] == kind and t["status"] == "queued":
            _spawn(tid, t)
            return


def _start_task(kind: str, argv: list[str]) -> dict:
    # 先刷新所有 running 任务的真实状态，避免子进程崩溃后无人轮询
    # 导致 status 永久 running（阻塞同 kind 队列推进）与 log_file 句柄泄漏
    for t in TASKS.values():
        _refresh_task_status(t)

    task_id = f"{kind}-{datetime.now().strftime('%H%M%S')}-{uuid.uuid4().hex[:6]}"
    TASKS[task_id] = {
        "kind": kind,
        "cmd": " ".join(argv),
        "argv": argv,
        "proc": None,
        "log_path": TASK_LOG_DIR / f"{task_id}.log",
        "log_file": None,
        "status": "queued",          # 同 kind 有 running 时排队；由 _advance_queue 在前一个结束后启动
        "started_at": datetime.now().isoformat(),
    }
    _evict_tasks()
    _advance_queue(kind)             # 无 running 时立即启动本任务；否则保持 queued
    return _task_view(task_id, TASKS[task_id])


@router.post("/api/run/evaluate")
async def api_run_evaluate(req: EvaluateRequest):
    # input 只允许 output/attacks/ 下的 jsonl 文件名，防路径穿越
    input_name = Path(req.input).name
    if not input_name.endswith(".jsonl"):
        raise HTTPException(status_code=400, detail="input 必须是 .jsonl 文件名")
    attack_file = ATTACKS_DIR / input_name
    if not attack_file.exists():
        raise HTTPException(status_code=404, detail=f"攻击集不存在: attacks/{input_name}")

    # 越狱税探针预检：读首条记录的 expected_answer（非 0/None 即含数学探针）
    has_tax_probe = False
    try:
        with open(attack_file, encoding="utf-8") as f:
            first_line = f.readline()
        if first_line.strip():
            ea = json.loads(first_line).get("expected_answer")
            has_tax_probe = ea not in (0, None)
    except Exception:
        has_tax_probe = False

    argv = [
        "-m", "llmsec.pipeline.runner",
        "--phase", req.phase,
        "--input", f"attacks/{input_name}",
        "--batch-size", str(req.batch_size),
        "--max-rounds", str(req.max_rounds),
        "--sampler", req.sampler,
    ]
    if req.sampler_alpha is not None:
        argv += ["--sampler-alpha", str(req.sampler_alpha)]
    if req.sampler_beta is not None:
        argv += ["--sampler-beta", str(req.sampler_beta)]
    if req.sampler_gamma is not None:
        argv += ["--sampler-gamma", str(req.sampler_gamma)]
    if req.coordinate_rounds is not None:
        argv += ["--coordinate-rounds", str(req.coordinate_rounds)]
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
    elif req.targets:
        # 前端探活后只传可达目标的子集（逗号分隔）
        argv += ["--targets", req.targets]
        # 多目标并发：未显式指定时默认全并发（每目标独立端点）。runner 内 min(tc, n) 兜底
        n_targets = len([t for t in req.targets.split(",") if t.strip()])
        tc = req.target_concurrency or max(1, n_targets)
        argv += ["--target-concurrency", str(tc)]
    elif req.target_concurrency:
        argv += ["--target-concurrency", str(req.target_concurrency)]
    view = _start_task("evaluate", argv)
    view["has_tax_probe"] = has_tax_probe
    return view


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


# ============================================================
# 任务进度（看板实时简略信息）
# ============================================================
def _progress_path(task_id: str) -> Path:
    """<task_id>.progress.jsonl 路径（与 .log 同目录）。可能不存在（任务刚启动/无 env）。"""
    return TASK_LOG_DIR / f"{task_id}.progress.jsonl"


def _read_progress(task_id: str) -> list[dict]:
    """读取 progress.jsonl 全部记录（坏行跳过）。文件不存在返回 []。"""
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


def _parse_eval_argv(argv: list[str]) -> tuple[list[str], int | None]:
    """从 runner argv 解析目标列表（--targets/--target）与 max_rounds。

    看板需要"全部声明目标"以渲染排队中占位行，而 progress.jsonl 只有已启动目标。
    """
    targets: list[str] = []
    max_rounds: int | None = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--targets" and i + 1 < len(argv):
            targets = [x.strip() for x in argv[i + 1].split(",") if x.strip()]
            i += 2
            continue
        if a == "--target" and i + 1 < len(argv):
            targets = [argv[i + 1].strip()]
            i += 2
            continue
        if a == "--max-rounds" and i + 1 < len(argv):
            try:
                max_rounds = int(argv[i + 1])
            except ValueError:
                pass
            i += 2
            continue
        i += 1
    return targets, max_rounds


@router.get("/api/tasks/{task_id}/progress")
async def api_task_progress(task_id: str):
    """任务进度快照：evaluate 返回每目标最后一条 + 全部声明目标（占位）；
    hpo 返回最后一条汇总。供看板初次渲染与 SSE 不可用时的轮询兜底。"""
    t = TASKS.get(task_id)
    if t is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    _refresh_task_status(t)
    kind = t["kind"]
    status = t["status"]
    records = _read_progress(task_id)

    if kind == "hpo":
        return {
            "kind": "hpo", "status": status,
            "progress": records[-1] if records else {},
        }

    # evaluate：每目标取最后一条；用 argv 补齐未启动目标的占位
    targets, max_rounds = _parse_eval_argv(t.get("argv", []))
    by_target: dict[str, dict] = {}
    for r in records:
        tg = r.get("target")
        if tg:
            by_target[tg] = r
    progress: dict[str, dict] = {tg: by_target.get(tg, {}) for tg in targets}
    # 兜底：progress 里出现但 argv 未声明的目标（如单 --target 之外的回退场景）
    for tg, rec in by_target.items():
        progress.setdefault(tg, rec)
    return {
        "kind": "evaluate", "status": status,
        "targets": targets, "max_rounds": max_rounds,
        "progress": progress,
    }


@router.post("/api/tasks/{task_id}/cancel")
async def api_task_cancel(task_id: str):
    """取消排队中或运行中的任务，置 cancelled。

    queued：直接标记取消（无子进程可杀）。running：SIGTERM → 5s 宽限 → SIGKILL，
    取消后推进同 kind 队列。Windows 无 SIGTERM 语义（Popen.terminate 即强杀）。
    runner 每场攻击实时 upsert 进 R，故取消后已观测的结果保留在结果矩阵中。
    已结束的任务返回 409。
    """
    t = TASKS.get(task_id)
    if t is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    _refresh_task_status(t)
    if t["status"] not in ("running", "queued"):
        raise HTTPException(status_code=409, detail=f"任务已结束（{t['status']}），无法取消")
    if t["status"] == "queued":
        t["status"] = "cancelled"
        return _task_view(task_id, t)
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
    _advance_queue(t["kind"])   # 取消 running 后，启动该 kind 队列里的下一个
    return _task_view(task_id, t)


@router.get("/api/tasks/{task_id}/stream")
async def api_task_stream(task_id: str):
    """SSE 实时进度流：跟随 progress.jsonl 新增行，每行发一个 event:progress（JSON）。

    连接时先回放射已有进度行（初次渲染上下文），之后跟随新增行直播。
    子进程结束时发一个 event:done（携带 status/returncode）再关闭，前端据此刷新数据。
    原始 .log 不再直播（仅 /api/tasks/{id}/log 下载）——运行框改为结构化简略信息。
    """
    t = TASKS.get(task_id)
    if t is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    progress_path = _progress_path(task_id)

    async def event_gen():
        offset = 0
        # 连接初始上下文：回放射全部已有进度行（每轮/每 trial 一行，量小）
        if progress_path.exists():
            try:
                init = progress_path.read_text(encoding="utf-8", errors="replace")
                offset = len(init.encode("utf-8"))
            except OSError:
                init = ""
            for line in init.splitlines():
                line = line.strip()
                if line:
                    yield f"event: progress\ndata: {line}\n\n"

        while True:
            if progress_path.exists():
                try:
                    size = progress_path.stat().st_size
                except OSError:
                    size = offset
                if size > offset:
                    try:
                        with open(progress_path, encoding="utf-8", errors="replace") as f:
                            f.seek(offset)
                            chunk = f.read(size - offset)
                        offset = size
                    except OSError:
                        chunk = ""
                    for line in chunk.splitlines():
                        line = line.strip()
                        if line:
                            yield f"event: progress\ndata: {line}\n\n"
                elif size < offset:
                    # 文件被截断/轮转，重置偏移跟随新内容
                    offset = size
            _refresh_task_status(t)
            if t["status"] != "running":
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


# ============================================================
# 攻击集上传
# ============================================================
@router.post("/api/attack-sets/upload")
async def upload_attack_set(file: UploadFile = File(...)):
    """拖拽/选择上传攻击集 .jsonl 文件。

    校验后缀 + 首行可 JSON parse，存到 ATTACKS_DIR。
    防路径穿越：只取 Path(file.filename).name。
    """
    if not file.filename or not file.filename.endswith(".jsonl"):
        raise HTTPException(status_code=400, detail="文件必须是 .jsonl 格式")

    # 防路径穿越：只取纯文件名
    safe_name = Path(file.filename).name
    if not safe_name:
        raise HTTPException(status_code=400, detail="无效文件名")

    content = await file.read()
    if not content.strip():
        raise HTTPException(status_code=400, detail="文件为空")

    # 校验首行可 parse
    first_line = content.decode("utf-8", errors="replace").split("\n", 1)[0].strip()
    if first_line:
        try:
            json.loads(first_line)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="首行不是有效 JSON（非标准 JSONL 格式）")

    ATTACKS_DIR.mkdir(parents=True, exist_ok=True)
    dest = ATTACKS_DIR / safe_name
    dest.write_bytes(content)

    n_records = sum(1 for line in content.decode("utf-8", errors="replace").splitlines() if line.strip())
    return {
        "name": safe_name,
        "size_kb": round(len(content) / 1024, 1),
        "n_records": n_records,
    }
