#!/usr/bin/env python3
"""
冒烟测试：dashboard_api 的 P1 级修复（H6 / M8 / M18）。

验证：
1. H6：_refresh_task_status 对已结束子进程更新 status 并关闭 log_file（dict 中置 None）；
   _start_task 在 409 检查前先刷新，崩溃后无人轮询的任务不再永久占用 "running"。
2. M8：_cache_put 维护 _CACHE_MAX_SIZE 上限，超限时按插入顺序淘汰最旧条目。
3. M18：EvaluateRequest.batch_size 的 le 上限 == params.ADAPTIVE_BATCH_MAX，
   超限被 pydantic 拦截、上限值本身可接受。
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Windows GBK 控制台兼容：允许输出 ✅/❌
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from llmsec.params import ADAPTIVE_BATCH_MAX
from llmsec.server.dashboard_api import (
    _CACHE_MAX_SIZE,
    TASKS,
    EvaluateRequest,
    _cache_put,
    _refresh_task_status,
    _start_task,
)


def _check(cond: bool, msg: str) -> int:
    if not cond:
        print(f"❌ {msg}")
        return 1
    print(f"✅ {msg}")
    return 0


class _StubProc:
    """假子进程：poll() 直接返回预设退出码。"""

    def __init__(self, rc: int | None):
        self._rc = rc

    def poll(self) -> int | None:
        return self._rc


def _make_task(rc: int | None, kind: str = "evaluate") -> dict:
    """构造假任务：stub proc + 真实临时 log 文件句柄。"""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".log", delete=False)
    return {
        "kind": kind,
        "cmd": "stub",
        "proc": _StubProc(rc),
        "log_path": Path(tmp.name),
        "log_file": tmp,
        "status": "running",
        "started_at": "2026-08-03T00:00:00",
    }


def test_refresh_task_status() -> int:
    rc = 0
    # 崩溃任务（rc=1）：status → failed，log_file 关闭且置 None
    t = _make_task(1)
    handle = t["log_file"]
    _refresh_task_status(t)
    rc |= _check(t["status"] == "failed" and t["returncode"] == 1,
                 "H6: 崩溃子进程 status 更新为 failed")
    rc |= _check(handle.closed and t["log_file"] is None,
                 "H6: log_file 句柄已关闭且 dict 中置 None")
    t["log_path"].unlink(missing_ok=True)

    # 正常结束（rc=0）：status → success
    t = _make_task(0)
    _refresh_task_status(t)
    rc |= _check(t["status"] == "success" and t["log_file"] is None,
                 "H6: 正常结束 status 更新为 success")
    t["log_path"].unlink(missing_ok=True)

    # 仍在运行（poll 为 None）：状态与句柄保持不动
    t = _make_task(None)
    _refresh_task_status(t)
    rc |= _check(t["status"] == "running" and not t["log_file"].closed,
                 "H6: 运行中任务不被误刷新")
    t["log_file"].close()
    t["log_path"].unlink(missing_ok=True)
    return rc


def test_start_task_refreshes_before_409() -> int:
    rc = 0
    # 塞入一个同类（evaluate）的“假死”任务：status 仍 running 但子进程已崩溃
    stale = _make_task(1, kind="evaluate")
    TASKS["stale-evaluate"] = stale
    view = None
    try:
        # 若 _start_task 未先刷新，409 检查会把这个假死任务当成运行中而拒绝新任务
        view = _start_task("evaluate", ["-c", "print('p1-ok')"])
        rc |= _check(view["kind"] == "evaluate",
                     "H6: _start_task 刷新后不再被假死任务 409 误拒")
        rc |= _check(stale["status"] == "failed" and stale["log_file"] is None,
                     "H6: _start_task 顺带刷新假死任务并关闭其 log_file")
    finally:
        TASKS.pop("stale-evaluate", None)
        stale["log_path"].unlink(missing_ok=True)
        real = TASKS.pop(view["id"], None) if view is not None else None
        if real is not None:
            real["proc"].wait()
            if real.get("log_file") is not None:
                real["log_file"].close()
            real["log_path"].unlink(missing_ok=True)
    return rc


def test_cache_eviction() -> int:
    rc = 0
    cache: dict = {}
    for i in range(_CACHE_MAX_SIZE + 10):
        _cache_put(cache, ("k", i), i)
    rc |= _check(len(cache) == _CACHE_MAX_SIZE,
                 f"M8: 缓存大小被压在上限 {_CACHE_MAX_SIZE}")
    rc |= _check(("k", 0) not in cache and ("k", 9) not in cache,
                 "M8: 最旧的 10 条已按插入顺序淘汰")
    rc |= _check(cache.get(("k", _CACHE_MAX_SIZE + 9)) == _CACHE_MAX_SIZE + 9,
                 "M8: 最新条目保留")
    return rc


def test_batch_limit_matches_params() -> int:
    rc = 0
    from annotated_types import Le
    from pydantic import ValidationError

    field = EvaluateRequest.model_fields["batch_size"]
    le_values = [m.le for m in field.metadata if isinstance(m, Le)]
    rc |= _check(le_values == [ADAPTIVE_BATCH_MAX],
                 f"M18: batch_size le 上限 == ADAPTIVE_BATCH_MAX({ADAPTIVE_BATCH_MAX})")

    try:
        EvaluateRequest(batch_size=ADAPTIVE_BATCH_MAX)
        ok_at_max = True
    except ValidationError:
        ok_at_max = False
    rc |= _check(ok_at_max, "M18: batch_size=ADAPTIVE_BATCH_MAX 可接受")

    try:
        EvaluateRequest(batch_size=ADAPTIVE_BATCH_MAX + 1)
        over_rejected = False
    except ValidationError:
        over_rejected = True
    rc |= _check(over_rejected, "M18: batch_size 超上限被 422 拦截")
    return rc


def main() -> int:
    rc = 0
    rc |= test_refresh_task_status()
    rc |= test_start_task_refreshes_before_409()
    rc |= test_cache_eviction()
    rc |= test_batch_limit_matches_params()
    print()
    if rc == 0:
        print("🎉 全部 P1 仪表盘修复测试通过")
    else:
        print("💥 存在失败项")
    return rc


if __name__ == "__main__":
    sys.exit(main())
