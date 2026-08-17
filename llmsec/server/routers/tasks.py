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
    _require_task(task_id)
    view = task_view(task_id)
    if view is None:  # 竞态：刚被淘汰出 TASKS
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return view


@router.get("/api/tasks/{task_id}/log")
async def api_task_log(task_id: str, download: bool = False):
    """完整任务日志（log_tail 只有尾部 4KB；任务失败后看完整上下文用此接口）。

    ?download=1 时以 text/plain + Content-Disposition 返回，便于直接下载 .log。
    """
    _require_task(task_id)
    text = task_manager.read_full_log(task_id)
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
    t = _require_task(task_id)
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

    queued：直接标记取消（无子进程可杀）。running：SIGTERM → 5s 宽限 → SIGKILL，
    取消后推进同 kind 队列。Windows 无 SIGTERM 语义（Popen.terminate 即强杀）。
    runner 每场攻击实时 upsert 进 R，故取消后已观测的结果保留在结果矩阵中。
    已结束的任务返回 409。
    """
    _require_task(task_id)
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
    """
    t = _require_task(task_id)
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
            # queued 也是活跃态（排队中尚未 spawn）：只有终态才发 done 关流。
            # 旧版 `!= "running"` 会让排队任务一连流就收到 done。
            if t["status"] not in ("running", "queued"):
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
