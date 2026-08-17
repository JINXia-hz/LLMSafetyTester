"""core.caches — 进程内缓存的统一助手（r9/P3-5）。

此前缓存实现散落 6 处、4 种失效口径（mtime 签名 / sig 元组 / TTL / 指纹），
且除个别外均无线程安全设计——cluster_viz 迁入 to_thread 后暴露的淘汰竞态
（r8/P2）正是这一类的第一例。本模块提供两个 ~40 行助手：

  - SigCache：签名失效缓存（key + sig 命中，否则 loader() 重建；上限淘汰）
  - TTLCache：单值 TTL 缓存（过期调 loader() 重建）

共同保证：
  - 线程安全（内部锁；loader 在锁外执行——慢加载不阻塞其它 key 的命中）
  - loader 并发重复加载允许（与既有各实现一致：值确定或近似，无正确性影响）
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any


class SigCache:
    """签名失效缓存：``get(key, sig, loader)``。

    - 同 key 且 sig 相同 → 直接返回缓存值
    - 否则调 ``loader()`` 重建并缓存（锁外执行），超 maxsize 按插入序淘汰最旧
    - sig 是任意可比较对象（mtime、(mtime, size) 元组、指纹……）
    """

    def __init__(self, maxsize: int = 128):
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self._data: dict[Any, tuple[Any, Any]] = {}  # key -> (sig, value)

    def get(self, key, sig, loader: Callable[[], Any]) -> Any:
        with self._lock:
            hit = self._data.get(key)
            if hit is not None and hit[0] == sig:
                return hit[1]
        value = loader()
        with self._lock:
            self._data[key] = (sig, value)
            while len(self._data) > self._maxsize:
                self._data.pop(next(iter(self._data)))
        return value

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


class TTLCache:
    """单值 TTL 缓存：``get(loader)``——未过期直接命中，过期调 loader() 重建。

    loader 抛异常不缓存（由调用方决定降级值并返回——见 review.get_thresholds
    的用法：loader 内部 try/except 返回 fallback，fallback 同样享受 TTL）。
    """

    def __init__(self, ttl: float):
        self._ttl = ttl
        self._lock = threading.Lock()
        self._value: Any = None
        self._at: float = 0.0

    def get(self, loader: Callable[[], Any]) -> Any:
        with self._lock:
            if self._value is not None and (time.time() - self._at) < self._ttl:
                return self._value
        value = loader()
        with self._lock:
            self._value = value
            self._at = time.time()
        return value

    def clear(self) -> None:
        with self._lock:
            self._value = None
            self._at = 0.0
