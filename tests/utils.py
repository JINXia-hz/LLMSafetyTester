"""共享测试辅助（跨测试文件复用的等待工具）。

tests 目录无 __init__.py：导入依赖 conftest.py 注入的项目根 + PEP 420
命名空间包——统一写 `from tests.utils import ...`。本文件名不匹配
python_files（test_*.py），不会被 pytest 收集为用例。
"""

import asyncio
import time


async def wait_until(pilot, cond, tries: int = 60) -> None:
    """有界等待条件成立（TUI pilot 版）。

    pilot.pause() 只推进消息队列、不保证真实时间流逝——慢机上渲染/worker
    线程可能尚未就绪，每轮补 20ms 真实 sleep 让出时间片。超时抛
    AssertionError（4 个历史副本曾在此各自漂移：默认轮数 25/40/60 不一，
    收敛后统一 60，只增耐心不减）。
    """
    for _ in range(tries):
        if cond():
            return
        await pilot.pause()
        await asyncio.sleep(0.02)
    raise AssertionError(f"wait_until 超时（{tries} 轮）：条件始终未成立")


def wait_until_sync(cond, timeout: float = 8.0, interval: float = 0.01) -> bool:
    """同步轮询等待条件成立；超时返回 False（调用方自行断言失败信息）。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(interval)
    return False
