"""core.progress — 后台任务实时进度落盘（供看板 SSE / 快照端点消费）。

evaluate 任务：attack_phase 每轮（含 seed 轮）落一条 per-target 记录，结束时落
              一条 phase=attack_done 标记该目标完成。
hpo 任务：study 每个 trial 落盘后落一条 phase=hpo 汇总记录（trial 计数 + 最佳指标）。

落盘位置：output/tasks/<task_id>.progress.jsonl（与任务日志同目录）。
仅当环境变量 LLMSEC_TASK_ID 存在时写——CLI 直跑无该 env → no-op，终端输出不变。

线程安全：模块级 Lock，保证多目标并发（target_concurrency>1 / HPO max_concurrent>1）
时整行原子追加、行间不交错。
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime

from llmsec.core import config as _config  # TASK_LOG_DIR 调用期动态读（测试可重绑）

_lock = threading.Lock()


def emit_progress(record: dict) -> None:
    """追加一条进度记录到 <task_id>.progress.jsonl；无 LLMSEC_TASK_ID 时 no-op。

    record 由调用方填充业务字段（phase/target/round/...），本函数统一补 ts 时间戳。
    任何 IO 异常静默吞掉——进度是辅助可见性，绝不能拖垮评估主流程。
    """
    task_id = os.getenv("LLMSEC_TASK_ID")
    if not task_id:
        return
    try:
        # A-10：dumps 同入 try——record 含非 str 键等 TypeError 属"进度数据问题"，
        # 与 IO 异常同按"绝不能拖垮评估主流程"处理（模块 docstring 的承诺口径）
        line = json.dumps(
            {"ts": datetime.now().isoformat(timespec="seconds"), **record},
            ensure_ascii=False,
            default=str,
        )
        with _lock:
            _config.TASK_LOG_DIR.mkdir(parents=True, exist_ok=True)
            with open(_config.TASK_LOG_DIR / f"{task_id}.progress.jsonl", "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except (OSError, TypeError, ValueError):
        pass
