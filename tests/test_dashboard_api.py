"""
冒烟测试：LLMSEC Web 面板 API。

验证：
1. 首页与全部数据 API 返回 200 且结构正确。
2. 非法 run 参数被 400 拦截（防路径穿越）。
3. /api/model 在缺少 svd_ridge 数据时优雅降级（available=False）。
4. 评估任务对不存在的攻击集返回 404。
5. 任务运行器能启动/跟踪/完成一个轻量子进程。
"""
import time

from fastapi.testclient import TestClient

from llmsec.server.dashboard_api import TASKS, _start_task, app

client = TestClient(app)

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
    for ep in ['/api/run/generate', '/api/run/cluster-analysis', '/api/run/evaluate']:
        r = client.get(ep)
        assert r.status_code == 405, f'GET {ep} 应 405，实际 {r.status_code}'
def test_state_snapshot_priority(monkeypatch, tmp_path):
    """run 目录内有 state.json 快照时，/api/threats 应优先用快照判定实测/预测；
    无快照时回退全局 state（历史批次行为不变）。

    RUNS_DIR 经 monkeypatch 重定向到 tmp_path，避免在真实 RUNS_DIR 建目录
    短暂成为"最新批次"污染同机看板。
    """
    import json

    import llmsec.server.dashboard_api as api
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
