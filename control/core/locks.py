"""control.core.locks — 跨进程文件锁（control 层共享原语）。

隔离边界约束（control/__init__.py）：control 层绝不 import llmsec 内部 API，
故不能复用 llmsec/core/results.py 的 _file_lock。本模块基于环境已安装的
filelock 包（比手写 msvcrt/fcntl 更可靠——处理了锁文件清理、线程内重入、
Windows LockFileEx 边界），提供语义一致的跨进程文件锁。

与 llmsec/core/results.py:_file_lock 的对应关系：
  - 本模块的 strict 语义与 results.py 的 _file_lock(strict=...) 一致
  - LockTimeout 是 OSError 子类（与 results.LockTimeout 同构），便于上层统一捕获

用途：AtomicIndexStore.update 的跨进程 RMW 保护（workspace/env_snapshot/gazette
的 _index.json）。锁顺序约束：线程锁（AtomicIndexStore._lock）在外，文件锁在内。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout


class LockTimeout(OSError):
    """跨进程文件锁获取超时（strict 模式下抛出）。

    是 OSError 子类，便于上层用已有的 except Exception / except OSError 兜底。
    """


@contextmanager
def cross_process_lock(
    lock_path: Path | str,
    *,
    timeout: float = 5.0,
    strict: bool = True,
) -> Iterator[None]:
    """跨进程文件锁（基于 filelock 包）。

    Args:
        lock_path:  被锁资源的路径。锁文件 = str(lock_path) + ".lock"。
                    （与 results.py:_file_lock 一致：锁文件旁路在被锁文件同目录）
        timeout:    获取锁的超时秒数。默认 5s（索引 RMW 极快，5s 远超正常耗时）。
        strict:     True=超时抛 LockTimeout；False=超时放行（best-effort，仅记日志）。
                    默认 True——索引 RMW 静默放行必丢更新，失败显式报错更安全。

    线程内重入：filelock 的 FileLock 支持同线程多次 acquire（引用计数），
    不会自死锁。跨进程互斥由 OS 级锁保证。
    """
    lock = FileLock(str(lock_path) + ".lock", timeout=timeout)
    try:
        lock.acquire()
    except Timeout:
        if strict:
            raise LockTimeout(
                f"跨进程文件锁获取超时({timeout:.0f}s)，拒绝操作（strict 模式，防并发丢更新）: {lock_path}"
            )
        # strict=False：放行（best-effort）。调用方须自行承担罕见并发丢更新风险
        yield
        return
    try:
        yield
    finally:
        try:
            lock.release()
        except Exception:
            pass  # 锁文件可能已被清理；释放失败不影响正确性
