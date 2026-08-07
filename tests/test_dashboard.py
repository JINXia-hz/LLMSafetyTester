"""Combined tests: Dashboard API (smoke + P1 fixes)."""

# ===== from test_dashboard_api.py =====
import timefrom fastapi.testclient import TestClientfrom llmsec.server.dashboard_api import appfrom llmsec.server.routers.tasks import TASKS, _start_taskclient = TestClient(app)

def test_index_and_data_apis():
    r = client.get('/')
    assert r.status_code == 200 and 'LLMSEC' in r.text, '首页 200 且包含标题'
    r = client.get('/api/runs')
    assert r.status_code == 200 and 'runs' in r.json(), '/api/runs 结构'
    r = client.get('/api/overview')
    assert r.status_code == 200, '/api/overview 200'
    d = r.json()
    if d.get('available'):
        assert len(d.get('radar', {}).get('labels', [])) == 5, '雷达图五维'
        assert len(d.get('radar', {}).get('values', [])) == 5, '雷达图五值'
        assert all(0 <= v <= 1 for v in d['radar']['values']), '雷达值域 [0,1]'
    for path in ['/api/threats', '/api/elo', '/api/report-md', '/api/clusters', '/api/model', '/api/attack-sets', '/api/tasks']:
        r = client.get(path)
        assert r.status_code == 200, f'{path} 200'
    print('✅ 首页与数据 API 通过')

def test_run_param_validation():
    r = client.get('/api/overview?run=../../etc')
    assert r.status_code == 400, '路径穿越被 400 拦截'
    r = client.get('/api/overview?run=2026-01-01_000000')
    assert r.status_code == 200, '合法但不存在的 run 不报错（available=False 或空）'
    print('✅ run 参数校验通过')

def test_model_fallback():
    r = client.get('/api/model')
    d = r.json()
    if not d['available']:
        assert 'run' in d, '无 svd_ridge 时优雅降级'
    print('✅ /api/model 容错通过')

def test_evaluate_validation():
    r = client.post('/api/run/evaluate', json={'input': '../../etc/passwd'})
    assert r.status_code in (400, 404), '非法 input 被拦截'
    r = client.post('/api/run/evaluate', json={'input': 'not_exists.jsonl'})
    assert r.status_code == 404, '不存在的攻击集 404'
    r = client.post('/api/run/evaluate', json={'input': 'l1.jsonl', 'phase': 'bogus'})
    assert r.status_code == 422, '非法 phase 被 pydantic 拦截'
    print('✅ 评估参数校验通过')

def test_task_lifecycle():
    view = _start_task('smoke', ['-c', "print('smoke-ok')"])
    task_id = view['id']
    assert not task_id not in TASKS, '❌ 任务未注册'
    deadline = time.time() + 30
    status = view['status']
    while time.time() < deadline:
        r = client.get(f'/api/tasks/{task_id}')
        status = r.json()['status']
        if status != 'running':
            break
        time.sleep(0.3)
    assert not status != 'success', f'❌ 任务未成功结束: {status}'
    r = client.get(f'/api/tasks/{task_id}')
    assert not 'smoke-ok' not in r.json().get('log_tail', ''), '❌ 日志尾缺少子进程输出'
    r = client.get('/api/tasks/nonexistent')
    assert not r.status_code != 404, '❌ 不存在任务应 404'
    print('✅ 任务生命周期通过')

def test_cluster_projection():
    r = client.get('/api/cluster-projection?method=pca')
    assert r.status_code == 200, 'pca 投影 200'
    d = r.json()
    assert 'available' in d, 'pca 投影含 available'
    if d.get('available'):
        assert d['n'] == len(d['points']), 'pca 点数与方法数一致'
        assert 'explained_variance' in d and len(d['explained_variance']) == 2, 'pca 含两维解释方差'
        p0 = d['points'][0]
        assert all(k in p0 for k in ('method', 'x', 'y', 'cluster', 'tested')), 'pca 点字段完整'
        assert isinstance(p0['x'], float) and isinstance(p0['y'], float), 'pca 坐标为数值'
    r = client.get('/api/cluster-projection?method=tsne')
    assert r.status_code == 200, 'tsne 投影 200'
    d = r.json()
    if d.get('available'):
        assert d['n'] == len(d['points']), 'tsne 点数与方法数一致'
        assert 'perplexity' in d and 1 <= d['perplexity'] < max(d['n'], 2), 'tsne perplexity 合法'
    r = client.get('/api/cluster-projection?method=umap')
    assert r.status_code == 400, '非法投影方法 400'
def test_cluster_tree_and_cut():
    r = client.get('/api/cluster-tree')
    assert r.status_code == 200, '/api/cluster-tree 200'
    d = r.json()
    assert 'available' in d, '/api/cluster-tree 含 available'
    if d.get('available'):
        assert d['n'] > 0 and len(d['icoord']) == d['n'] - 1, '树图坐标数量正确'
        assert len(d['merge_heights']) == d['n'] - 1, '合并高度数量正确'
        assert d['chosen_k'] >= 2, 'chosen_k 合法'
        n = d['n']
        r = client.get(f'/api/cluster-cut?k={min(5, n - 1)}')
        assert r.status_code == 200, '/api/cluster-cut 200'
        c = r.json()
        if c.get('available'):
            assert len(c['clusters']) == min(5, n - 1), '切割簇数 == k'
            assert all('name' in cl and 'members' in cl for cl in c['clusters']), '切割簇字段完整'
            total = sum(cl['size'] for cl in c['clusters'])
            assert total == n, '切割覆盖全部方法'
        r = client.get('/api/cluster-cut?k=99999')
        assert r.status_code == 400, '非法 k 被 400 拦截'
def test_run_endpoints_post_only():
    """任务端点只接受 POST（前端曾用 GET 调用导致 405）。"""
    for ep in ['/api/run/evaluate', '/api/run/hpo']:
        r = client.get(ep)
        assert r.status_code == 405, f'GET {ep} 应 405，实际 {r.status_code}'
def test_state_snapshot_priority(monkeypatch, tmp_path):
    import json    import llmsec.server.dashboard_api as api
    fake_runs = tmp_path / "runs"
    fake_runs.mkdir()
    monkeypatch.setattr(api, "RUNS_DIR", fake_runs)

    run_name = '2099-01-01_000000'
    run_dir = fake_runs / run_name
    tree = {'top_threats': [{'method': 'snapshot_only_method', 'elo': 1600.0}], 'strong_defenses': [], 'upsets': {}}
    snapshot_state = {'attacker_ratings': {'snapshot_only_method': 1600.0}, 'attacker_pred_std': {}, 'ground_truth': {'snapshot_only_method': {'elo': 1600.0}}}
    run_dir.mkdir(parents=True)
    (run_dir / 'security_tree.json').write_text(json.dumps(tree, ensure_ascii=False), encoding='utf-8')
    r = client.get(f'/api/threats?run={run_name}')
    assert r.status_code == 200, '/api/threats 无快照 200'
    threats = r.json().get('top_threats', [])
    assert bool(threats) and threats[0]['tested'] is False and (threats[0]['source'] in ('svd_ridge', 'predicted')), '无快照时回退全局 state（标 svd_ridge）'
    (run_dir / 'state.json').write_text(json.dumps(snapshot_state, ensure_ascii=False), encoding='utf-8')
    r = client.get(f'/api/threats?run={run_name}')
    assert r.status_code == 200, '/api/threats 有快照 200'
    threats = r.json().get('top_threats', [])
    assert bool(threats) and threats[0]['tested'] is True and (threats[0]['source'] == 'ground_truth'), '有快照时优先快照（标 ground_truth）'
    assert threats[0]['elo'] == 1600.0, '快照 Elo 生效'
    r = client.get(f'/api/clusters?run={run_name}')
    assert r.status_code == 200, '/api/clusters 200'
    assert r.json().get('validation', {}).get('sentinel') is not True, '无 cluster_report 快照时回退全局报告'
    (run_dir / 'cluster_report.json').write_text(json.dumps({'validation': {'silhouette': 0.9999, 'sentinel': True}}, ensure_ascii=False), encoding='utf-8')
    r = client.get(f'/api/clusters?run={run_name}')
    assert r.json().get('validation', {}).get('sentinel') is True, '有 cluster_report 快照时优先快照'
    assert r.json().get('validation', {}).get('silhouette') == 0.9999, '快照 validation 内容生效'

# ===== from test_p1_dashboard.py =====
import tempfilefrom pathlib import Pathfrom llmsec.params import ADAPTIVE_BATCH_MAXfrom llmsec.server.routers.cluster_viz import _CACHE_MAX_SIZE, _cache_putfrom llmsec.server.routers.tasks import (    EvaluateRequest,    _refresh_task_status,)class _StubProc:
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
    from annotated_types import Le    from pydantic import ValidationError
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
