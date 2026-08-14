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
from llmsec.params import ADAPTIVE_BATCH_MAX, DEFAULT_BATCH_SIZE, DEFAULT_MAX_ROUNDS, MAX_ROUNDS_LIMIT
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
    # 看板评估默认走全局模式且 publish 到全局 R（保留旧行为；runner 已改为默认不 publish）。
    # 用户若要隔离评估，用 control 层的 fork。
    argv += ["--publish-global"]
    view = task_manager.start_task("evaluate", argv)
    view["has_tax_probe"] = has_tax_probe
    return view


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
