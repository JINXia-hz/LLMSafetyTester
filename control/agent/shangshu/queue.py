"""control.agent.shangshu.queue — Plan 执行队列（单 worker 线程）。

用户可以连续准奏多个 Plan，它们排队执行（同一时间只跑一个 Plan）。
这避免了并发执行多个 Plan 同时写 R 矩阵、同时消耗 API 额度的问题。

设计：
  - 单 worker 线程串行执行（Plan 之间有数据依赖——R 矩阵是共享状态）
  - submit() 不阻塞——立即返回 "queued" / "running" / "duplicate"
  - 前端轮询 bus feed 看执行进度（已有逻辑）
  - cancel() 只能取消排队中的，不能中断正在执行的
"""

from __future__ import annotations

import threading
from collections import deque
from threading import Lock, Thread


class PlanQueue:
    """Plan 执行队列。单 worker 线程串行执行。"""

    def __init__(self) -> None:
        self._queue: deque[str] = deque()
        self._running: str | None = None
        self._lock = Lock()
        self._worker: Thread | None = None
        self._cond = threading.Condition(self._lock)

    def submit(self, plan_id: str) -> str:
        """提交 Plan 到队列。

        Returns:
            "queued" — 已加入队列，等待执行
            "duplicate" — 该 Plan 已在队列或正在执行
        """
        with self._lock:
            if plan_id == self._running or plan_id in self._queue:
                return "duplicate"
            self._queue.append(plan_id)
            self._cond.notify()
        self._ensure_worker()
        return "queued"

    def status(self) -> dict:
        """返回队列状态。"""
        with self._lock:
            return {
                "running": self._running,
                "queued": list(self._queue),
            }

    def cancel(self, plan_id: str) -> bool:
        """从队列移除（只能取消排队中的，不能取消正在执行的）。"""
        with self._lock:
            if plan_id in self._queue:
                self._queue.remove(plan_id)
                return True
            return False

    def _ensure_worker(self) -> None:
        """确保 worker 线程在跑。"""
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = Thread(target=self._run_loop, daemon=True, name="plan-queue-worker")
        self._worker.start()

    def _run_loop(self) -> None:
        """Worker 线程主循环：从队列取 Plan 执行。"""
        from control.agent.shangshu.executor import execute_plan

        while True:
            with self._lock:
                while not self._queue and self._running is None:
                    self._cond.wait(timeout=1.0)
                    if not self._queue:
                        # 队列空且无任务，退出线程（下次 submit 会重新启动）
                        return
                if not self._queue:
                    return
                plan_id = self._queue.popleft()
                self._running = plan_id

            try:
                execute_plan(plan_id)
            except Exception as e:
                import sys
                print(f"[plan-queue] Plan {plan_id} 执行失败: {e}", file=sys.stderr)
            finally:
                with self._lock:
                    self._running = None


# 模块级单例
_QUEUE: PlanQueue | None = None


def get_queue() -> PlanQueue:
    """获取全局 Plan 队列单例。"""
    global _QUEUE
    if _QUEUE is None:
        _QUEUE = PlanQueue()
    return _QUEUE


def reset_queue() -> None:
    """重置队列（测试用）。"""
    global _QUEUE
    _QUEUE = None
