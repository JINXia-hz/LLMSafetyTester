"""任务管理路由：图形化触发生成 / 评估 / 聚类分析（子进程任务 + 状态轮询 / SSE 流）。

任务状态机/子进程管理统一在 llmsec.server.task_manager（与 MCP server 共用同一份
TASKS 注册表）——本文件只是 HTTP 薄封装。测试/调用方一律直接引用 task_manager
命名空间（不再保留本模块的兼容别名层）。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from llmsec.core.config import ATTACKS_DIR
from llmsec.params import ADAPTIVE_BATCH_MAX, DEFAULT_BATCH_SIZE, DEFAULT_MAX_ROUNDS, MAX_ROUNDS_LIMIT, SAMPLERS
from llmsec.server import task_manager
from llmsec.server.task_manager import (
    TASKS,
    _progress_path,
    _refresh_task_status,
    task_view,
)

router = APIRouter()


def _require_task(task_id: str) -> dict:
    """按 id 取任务，不存在则抛 404。"""
    t = TASKS.get(task_id)
    if t is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return t


def _external_task_row(task_id: str):
    """跨进程任务的目录库行回退（D-4）。

    list_tasks 是"本进程 TASKS ∪ 库行"的跨进程视图，但各任务端点此前只查
    本进程 TASKS——MCP/TUI 启动的任务（或已被 >64 淘汰出 TASKS 的历史任务）
    在看板"列表可见、详情/日志/进度/SSE 全 404"，前端 watcher 静默死亡。
    返回 Task 库行；无行返回 None。
    """
    from llmsec.storage import catalog

    try:
        return catalog.get_task(task_id)
    except Exception:
        return None


def _external_task_view(task_id: str, row) -> dict:
    """把目录库行构造成 task_view 同形 dict（只读：log_tail 尾部 4KB）。

    Task 表无 returncode 列（退出码在 _persist_task 只写进程内 dict）——
    getattr 兜 None，避免库行字段差异把端点打成 500。
    """
    log_path = Path(row.log_path) if row.log_path else None
    log_tail = ""
    if log_path is not None and log_path.exists():
        try:
            log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-4096:]
        except OSError:
            pass
    return {
        "id": task_id, "kind": row.kind, "status": row.status,
        "returncode": getattr(row, "returncode", None), "cmd": row.cmd,
        "log_tail": log_tail, "started_at": getattr(row, "started_at", None),
        "pid": getattr(row, "pid", None), "meta": None, "error": None,
        "external": True,
        "log_path": str(log_path) if log_path is not None else None,
    }


class EvaluateRequest(BaseModel):
    phase: str = Field(default="all", pattern="^(all|1|2)$")
    input: str = "l1.jsonl"
    batch_size: int = Field(default=min(DEFAULT_BATCH_SIZE, ADAPTIVE_BATCH_MAX), ge=1, le=ADAPTIVE_BATCH_MAX)
    max_rounds: int = Field(default=DEFAULT_MAX_ROUNDS, ge=1, le=MAX_ROUNDS_LIMIT)
    sampler: str = Field(default="hybrid", pattern="^(" + "|".join(SAMPLERS) + ")$")
    # 采样器权重（None = 用 params 默认值，不传 --xxx 旗标给 runner）
    sampler_alpha: float | None = None
    sampler_beta: float | None = None
    sampler_gamma: float | None = None
    coordinate_rounds: int | None = None
    target: str | None = Field(default=None, pattern=r"^[\w.\-:]+$")
    targets: str | None = None      # 多目标子集（逗号分隔，前端探活后只传可达的）
    # 多目标并发数：None + 多目标 → 默认全并发（每个目标是独立端点，无共享限速）
    target_concurrency: int | None = Field(default=None, ge=1, le=32)
    no_early_stop: bool = False     # 跑满 max_rounds 不早停（固定预算可比性）
    # env 隔离（归一新增，能力与 MCP run_evaluation 对齐）：快照覆盖全局 .env / 覆写 params
    env_snapshot: str | None = None
    param_overrides: dict | None = None


@router.post("/api/run/evaluate")
async def api_run_evaluate(req: EvaluateRequest):
    # argv 构造/校验/env 注入统一在 llmsec.server.launch（与 MCP/TUI 共用），
    # 本端点只做 HTTP 协议映射。攻击集先解析一次：越狱税探针预检需要路径。
    from llmsec.server.launch import (
        LaunchError,
        LaunchSpec,
        attack_has_tax_probe,
        launch_evaluation,
        resolve_attack_file,
    )

    try:
        attack_path = resolve_attack_file(req.input)
    except LaunchError as e:
        raise HTTPException(status_code=404 if e.reason == "not_found" else 400, detail=str(e)) from None

    spec = LaunchSpec(
        target=req.target,
        targets=_split_targets(req.targets),
        input_file=req.input,
        phase=req.phase,
        batch_size=req.batch_size,
        max_rounds=req.max_rounds,
        sampler=req.sampler,
        sampler_alpha=req.sampler_alpha,
        sampler_beta=req.sampler_beta,
        sampler_gamma=req.sampler_gamma,
        coordinate_rounds=req.coordinate_rounds,
        target_concurrency=req.target_concurrency,
        no_early_stop=req.no_early_stop,
        env_snapshot=req.env_snapshot,
        param_overrides=req.param_overrides,
    )
    try:
        view = launch_evaluation(spec)
    except LaunchError as e:
        raise HTTPException(status_code=404 if e.reason == "not_found" else 400, detail=str(e)) from None
    # 越狱税探针预检：该攻击集无数学探针时越狱税将不计算（前端提示用）
    view["has_tax_probe"] = attack_has_tax_probe(attack_path)
    return view


def _split_targets(raw: str | None) -> list[str] | None:
    """逗号分隔的多目标串 → 列表（空/空白项剔除；None 透传）。"""
    if raw is None:
        return None
    parts = [t.strip() for t in raw.split(",") if t.strip()]
    return parts or None


@router.get("/api/tasks")
async def api_tasks():
    return {"tasks": task_manager.list_tasks()}


@router.get("/api/tasks/{task_id}")
async def api_task(task_id: str):
    if task_id not in TASKS:
        # D-4：跨进程任务只读回退（详情不再 404）
        row = _external_task_row(task_id)
        if row is not None:
            return _external_task_view(task_id, row)
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    view = task_view(task_id)
    if view is None:  # 竞态：刚被淘汰出 TASKS
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return view


@router.get("/api/tasks/{task_id}/log")
async def api_task_log(task_id: str, download: bool = False):
    """完整任务日志（log_tail 只有尾部 4KB；任务失败后看完整上下文用此接口）。

    ?download=1 时以 text/plain + Content-Disposition 返回，便于直接下载 .log。
    """
    if task_id in TASKS:
        text = task_manager.read_full_log(task_id)
    else:
        # D-4：跨进程任务——log_path 在库行里
        row = _external_task_row(task_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
        text = ""
        if row.log_path and Path(row.log_path).exists():
            try:
                text = Path(row.log_path).read_text(encoding="utf-8", errors="replace")
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
@router.get("/api/tasks/{task_id}/progress")
async def api_task_progress(task_id: str):
    """任务进度快照：evaluate 返回每目标最后一条 + 全部声明目标（占位）；
    hpo 返回最后一条汇总。供看板初次渲染与 SSE 不可用时的轮询兜底。"""
    t = TASKS.get(task_id)
    if t is None:
        # D-4：跨进程任务——progress 文件路径可由 id 推导，状态取库行
        row = _external_task_row(task_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
        records = task_manager.read_progress(task_id)
        if row.kind == "hpo":
            return {"kind": "hpo", "status": row.status,
                    "progress": records[-1] if records else {},
                    "trials": [r["last"] for r in records if r.get("last")][-30:]}
        by_target: dict[str, dict] = {}
        for r in records:
            tg = r.get("target")
            if tg:
                by_target[tg] = r
        return {"kind": "evaluate", "status": row.status,
                "targets": list(by_target), "max_rounds": None,
                "progress": by_target}
    _refresh_task_status(t)
    kind = t["kind"]
    status = t["status"]
    records = task_manager.read_progress(task_id)

    if kind == "hpo":
        return {
            "kind": "hpo", "status": status,
            "progress": records[-1] if records else {},
            # 逐 trial 明细（供轮询兜底/中途打开页面重建历史；无 last 的旧记录跳过）
            "trials": [r["last"] for r in records if r.get("last")][-30:],
        }

    # evaluate：每目标取最后一条；用 launch 层写入的 meta 补齐未启动目标的占位
    # （原实现对 argv 反向解析——归一后 meta 由 start_task 时一次性结构化写入）
    meta = t.get("meta") or {}
    targets: list[str] = meta.get("targets") or []
    max_rounds = meta.get("max_rounds")
    by_target: dict[str, dict] = {}
    for r in records:
        tg = r.get("target")
        if tg:
            by_target[tg] = r
    progress: dict[str, dict] = {tg: by_target.get(tg, {}) for tg in targets}
    # 兜底：progress 里出现但 meta 未声明的目标（如不传 target 跑全部的回退场景）
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

    queued：直接标记取消（无子进程可杀）。running：Windows taskkill /T 树杀、
    POSIX SIGTERM → 5s 宽限 → SIGKILL，取消后推进同 kind 队列。
    runner 每场攻击实时 upsert 进 R，故取消后已观测的结果保留在结果矩阵中。
    已结束的任务返回 409；跨进程任务（MCP/TUI 启动、proc 句柄不在本进程）
    无法从看板取消，返回 409 明示。
    """
    if task_id not in TASKS:
        row = _external_task_row(task_id)
        if row is not None:
            raise HTTPException(
                status_code=409,
                detail="跨进程任务（其他入口启动）无法从看板取消——请在启动方"
                       "（TUI/MCP）执行取消，或直接结束其进程")
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    # cancel 内含 proc.wait(5)，阻塞等待放线程池，不卡事件循环
    view = await asyncio.to_thread(task_manager.cancel_task, task_id)
    if view is None:
        raise HTTPException(status_code=409, detail="任务已结束，无法取消")
    return view


@router.get("/api/tasks/{task_id}/stream")
async def api_task_stream(task_id: str):
    """SSE 实时进度流：跟随 progress.jsonl 新增行，每行发一个 event:progress（JSON）。

    连接时先回放射已有进度行（初次渲染上下文），之后跟随新增行直播。
    子进程结束时发一个 event:done（携带 status/returncode）再关闭，前端据此刷新数据。
    原始 .log 不再直播（仅 /api/tasks/{id}/log 下载）——运行框改为结构化简略信息。

    D-4：跨进程任务（TASKS 无、库行有）同样可流——状态改查目录库行（进程句柄
    不在本进程，无法 poll，库行由启动方 upsert）。
    D-6：行缓冲读——子进程写与 SSE 读并发时最后一行可能被"撕开"（半行 JSON），
    此前按行直发即丢该条进度记录；残行留在缓冲区与下轮 chunk 拼接。
    """
    t = TASKS.get(task_id)
    external_row = None
    if t is None:
        external_row = _external_task_row(task_id)
        if external_row is None:
            raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    progress_path = _progress_path(task_id)

    def _current_status() -> tuple[str, object]:
        """(status, returncode)：本进程任务实时 poll；外部任务查库行。"""
        if t is not None:
            _refresh_task_status(t)
            return t["status"], t.get("returncode")
        fresh = _external_task_row(task_id)
        if fresh is not None:
            # D-1：Task 模型无 returncode 列（MCP/TUI 写入的外部行）——裸属性
            # 访问让外部任务的 SSE 首轮即 AttributeError 断流，与 _external_task_view
            # 同口径用 getattr 兜底
            return fresh.status, getattr(fresh, "returncode", None)
        return "failed", None  # 库行消失（极端）：按终态关流

    async def event_gen():
        offset = 0
        buffer = ""  # D-6：跨读次的残行缓冲（字节 offset 前进、文本留在缓冲拼接）
        deadline = asyncio.get_event_loop().time() + 7200  # D-6 残项：流总生命周期 2h
        # 连接初始上下文：回放射全部已有进度行（每轮/每 trial 一行，量小）
        if progress_path.exists():
            try:
                init = progress_path.read_text(encoding="utf-8", errors="replace")
                offset = len(init.encode("utf-8"))
            except OSError:
                init = ""
            *complete, buffer = init.split("\n")
            for line in complete:
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
                    buffer += chunk
                    *complete, buffer = buffer.split("\n")
                    for line in complete:
                        line = line.strip()
                        if line:
                            yield f"event: progress\ndata: {line}\n\n"
                elif size < offset:
                    # 文件被截断/轮转，重置偏移跟随新内容（残行一并丢弃）
                    offset = size
                    buffer = ""
            status, returncode = _current_status()
            now = asyncio.get_event_loop().time()
            # D-6 残项：无总上限时任务对象若永不过渡，流永不关闭（连接泄漏）
            if now > deadline:
                yield (
                    "event: done\ndata: "
                    + json.dumps(
                        {"status": "stream_timeout", "returncode": None,
                         "note": "SSE 流达 2h 上限关闭（任务仍可能在跑，重连继续）"},
                        ensure_ascii=False,
                    )
                    + "\n\n"
                )
                return
            # queued 也是活跃态（排队中尚未 spawn）：只有终态才发 done 关流。
            # 旧版 `!= "running"` 会让排队任务一连流就收到 done。
            if status not in ("running", "queued"):
                yield (
                    "event: done\ndata: "
                    + json.dumps(
                        {"status": status, "returncode": returncode},
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
    # D-23：大小上限——整文件读内存前先拒（默认 64MB，覆盖万级攻击集 jsonl）
    if len(content) > 64 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="文件超过 64MB 上限")
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
