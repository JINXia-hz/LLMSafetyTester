"""
冒烟测试：dashboard_api 的 P1 级修复（H6 / M8 / M18）。

验证：
1. H6：_refresh_task_status 对已结束子进程更新 status 并关闭 log_file（dict 中置 None）；
   _start_task 在 409 检查前先刷新，崩溃后无人轮询的任务不再永久占用 "running"。
2. M8：_cache_put 维护 _CACHE_MAX_SIZE 上限，超限时按插入顺序淘汰最旧条目。
3. M18：EvaluateRequest.batch_size 的 le 上限 == params.ADAPTIVE_BATCH_MAX，
   超限被 pydantic 拦截、上限值本身可接受。
"""
import tempfile
from pathlib import Path

from llmsec.params import ADAPTIVE_BATCH_MAX
from llmsec.server.routers.cluster_viz import _CACHE_MAX_SIZE, _cache_put
from llmsec.server.routers.tasks import (
    TASKS,
    EvaluateRequest,
    _refresh_task_status,
    _start_task,
)


class _StubProc:
    """假子进程：poll() 直接返回预设退出码。"""

    def __init__(self, rc: int | None):
        self._rc = rc

    def poll(self):
        return self._rc

def _make_task(rc: int | None, kind: str='evaluate'):
    """构造假任务：stub proc + 真实临时 log 文件句柄。"""
    tmp = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.log', delete=False)
    return {'kind': kind, 'cmd': 'stub', 'proc': _StubProc(rc), 'log_path': Path(tmp.name), 'log_file': tmp, 'status': 'running', 'started_at': '2026-08-03T00:00:00'}

def test_refresh_task_status():
    t = _make_task(1)
    handle = t['log_file']
    _refresh_task_status(t)
    assert t['status'] == 'failed' and t['returncode'] == 1, 'H6: 崩溃子进程 status 更新为 failed'
    assert handle.closed and t['log_file'] is None, 'H6: log_file 句柄已关闭且 dict 中置 None'
    t['log_path'].unlink(missing_ok=True)
    t = _make_task(0)
    _refresh_task_status(t)
    assert t['status'] == 'success' and t['log_file'] is None, 'H6: 正常结束 status 更新为 success'
    t['log_path'].unlink(missing_ok=True)
    t = _make_task(None)
    _refresh_task_status(t)
    assert t['status'] == 'running' and (not t['log_file'].closed), 'H6: 运行中任务不被误刷新'
    t['log_file'].close()
    t['log_path'].unlink(missing_ok=True)

def test_start_task_refreshes_before_409():
    stale = _make_task(1, kind='evaluate')
    TASKS['stale-evaluate'] = stale
    view = None
    try:
        view = _start_task('evaluate', ['-c', "print('p1-ok')"])
        assert view['kind'] == 'evaluate', 'H6: _start_task 刷新后不再被假死任务 409 误拒'
        assert stale['status'] == 'failed' and stale['log_file'] is None, 'H6: _start_task 顺带刷新假死任务并关闭其 log_file'
    finally:
        TASKS.pop('stale-evaluate', None)
        stale['log_path'].unlink(missing_ok=True)
        real = TASKS.pop(view['id'], None) if view is not None else None
        if real is not None:
            real['proc'].wait()
            if real.get('log_file') is not None:
                real['log_file'].close()
            real['log_path'].unlink(missing_ok=True)

def test_cache_eviction():
    cache: dict = {}
    for i in range(_CACHE_MAX_SIZE + 10):
        _cache_put(cache, ('k', i), i)
    assert len(cache) == _CACHE_MAX_SIZE, f'M8: 缓存大小被压在上限 {_CACHE_MAX_SIZE}'
    assert ('k', 0) not in cache and ('k', 9) not in cache, 'M8: 最旧的 10 条已按插入顺序淘汰'
    assert cache.get(('k', _CACHE_MAX_SIZE + 9)) == _CACHE_MAX_SIZE + 9, 'M8: 最新条目保留'

def test_batch_limit_matches_params():
    from annotated_types import Le
    from pydantic import ValidationError
    field = EvaluateRequest.model_fields['batch_size']
    le_values = [m.le for m in field.metadata if isinstance(m, Le)]
    assert le_values == [ADAPTIVE_BATCH_MAX], f'M18: batch_size le 上限 == ADAPTIVE_BATCH_MAX({ADAPTIVE_BATCH_MAX})'
    try:
        EvaluateRequest(batch_size=ADAPTIVE_BATCH_MAX)
        ok_at_max = True
    except ValidationError:
        ok_at_max = False
    assert ok_at_max, 'M18: batch_size=ADAPTIVE_BATCH_MAX 可接受'
    try:
        EvaluateRequest(batch_size=ADAPTIVE_BATCH_MAX + 1)
        over_rejected = False
    except ValidationError:
        over_rejected = True
    assert over_rejected, 'M18: batch_size 超上限被 422 拦截'
