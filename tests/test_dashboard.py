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



def test_emit_progress_env_gated():
    """emit_progress：无 LLMSEC_TASK_ID 时 no-op；有则写一行 JSON。"""
    import os

    from llmsec.core import progress as P

    task_id = 'ut-progress-' + str(int(time.time() * 1000))
    path = P.TASK_LOG_DIR / f"{task_id}.progress.jsonl"
    if path.exists():
        path.unlink()

    # 无 env → 不写文件
    os.environ.pop('LLMSEC_TASK_ID', None)
    P.emit_progress({'phase': 'attack', 'target': 'x', 'round': 1})
    assert not path.exists(), '无 LLMSEC_TASK_ID 时不应写 progress 文件'

    # 有 env → 写一行可 parse 的 JSON
    os.environ['LLMSEC_TASK_ID'] = task_id
    try:
        P.emit_progress({'phase': 'attack', 'target': 'deepseek', 'round': 2, 'elo': 1500.0})
        assert path.exists(), '有 LLMSEC_TASK_ID 时应写 progress 文件'
        import json
        rec = json.loads(path.read_text(encoding='utf-8').strip().splitlines()[-1])
        assert rec['target'] == 'deepseek' and rec['round'] == 2 and 'ts' in rec, '进度记录字段完整'
    finally:
        os.environ.pop('LLMSEC_TASK_ID', None)
        if path.exists():
            path.unlink()

    print('✅ emit_progress env 门控通过')



def test_task_progress_endpoint():
    """/api/tasks/{id}/progress：argv 解析目标 + 每 target 取末条；hpo 取末条。"""
    import json

    from llmsec.core.config import TASK_LOG_DIR
    from llmsec.server.routers.tasks import TASKS, _progress_path

    # evaluate：双目标，各有进度
    tid = 'evaluate-ut-' + str(int(time.time() * 1000))
    TASKS[tid] = {
        'kind': 'evaluate',
        'argv': ['-m', 'llmsec.pipeline.runner', '--targets', 'a,b', '--max-rounds', '10'],
        'status': 'success', 'returncode': 0, 'log_path': TASK_LOG_DIR / f"{tid}.log",
        'log_file': None, 'started_at': '2026-01-01T00:00:00', 'cmd': '', 'proc': None,
    }
    pp = _progress_path(tid)
    TASK_LOG_DIR.mkdir(parents=True, exist_ok=True)
    pp.write_text(
        json.dumps({'ts': 't1', 'phase': 'attack', 'target': 'a', 'round': 1, 'elo': 1490.0}) + '\n'
        + json.dumps({'ts': 't2', 'phase': 'attack', 'target': 'a', 'round': 2, 'elo': 1500.0}) + '\n'
        + json.dumps({'ts': 't3', 'phase': 'attack', 'target': 'b', 'round': 1, 'elo': 1510.0}) + '\n',
        encoding='utf-8')
    try:
        r = client.get(f'/api/tasks/{tid}/progress')
        assert r.status_code == 200
        d = r.json()
        assert d['kind'] == 'evaluate' and d['targets'] == ['a', 'b'] and d['max_rounds'] == 10
        # a 取末条（round 2），b 取末条（round 1）
        assert d['progress']['a']['round'] == 2 and d['progress']['b']['round'] == 1
    finally:
        TASKS.pop(tid, None)
        if pp.exists():
            pp.unlink()

    # hpo：取末条汇总 + 逐 trial 明细（无 last 的旧记录跳过）
    hid = 'hpo-ut-' + str(int(time.time() * 1000))
    TASKS[hid] = {
        'kind': 'hpo', 'argv': ['-m', 'llmsec.experiments', 'run', 'x.yaml'],
        'status': 'success', 'returncode': 0, 'log_path': TASK_LOG_DIR / f"{hid}.log",
        'log_file': None, 'started_at': '2026-01-01T00:00:00', 'cmd': '', 'proc': None,
    }
    hp = _progress_path(hid)
    hp.write_text(
        json.dumps({'phase': 'hpo', 'trial_done': 1, 'best_metric': None}) + '\n'
        + json.dumps({'phase': 'hpo', 'trial_done': 2, 'best_metric': 0.6,
                      'last': {'target': 'a', 'seed': 0, 'status': 'success',
                               'value': 0.6, 'params': {'K_FACTOR': 16}}}) + '\n'
        + json.dumps({'phase': 'hpo', 'trial_done': 3, 'best_metric': 0.5,
                      'last': {'target': 'a', 'seed': 0, 'status': 'timeout',
                               'value': None, 'params': {'K_FACTOR': 48}}}) + '\n',
        encoding='utf-8')
    try:
        r = client.get(f'/api/tasks/{hid}/progress')
        assert r.status_code == 200
        d = r.json()
        assert d['kind'] == 'hpo' and d['progress']['trial_done'] == 3 and d['progress']['best_metric'] == 0.5
        assert len(d['trials']) == 2, 'trials 应收 2 条带 last 的明细（首条无 last 跳过）'
        assert d['trials'][0]['value'] == 0.6 and d['trials'][1]['status'] == 'timeout'
        assert d['trials'][1]['params'] == {'K_FACTOR': 48}
    finally:
        TASKS.pop(hid, None)
        if hp.exists():
            hp.unlink()

    print('✅ /api/tasks/{id}/progress 通过')



def test_evaluate_concurrency_argv(monkeypatch):
    """多目标评估：argv 必须含 --targets 与 --target-concurrency（默认=目标数，全并发）。"""
    import llmsec.server.routers.tasks as tasks_mod

    captured = {}

    def fake_start(kind, argv):
        captured['argv'] = list(argv)
        return {"id": "fake-eval", "kind": kind, "cmd": " ".join(argv), "argv": list(argv),
                "status": "queued", "returncode": None, "log_path": tasks_mod.TASK_LOG_DIR / "fake.log",
                "log_file": None, "started_at": "2026-01-01T00:00:00", "error": None, "proc": None}

    monkeypatch.setattr(tasks_mod, "_start_task", fake_start)

    # 默认：多目标 → 全并发（target_concurrency = 目标数）
    r = client.post('/api/run/evaluate', json={
        "input": "example.jsonl", "targets": "a,b,c", "max_rounds": 3, "batch_size": 3})
    assert r.status_code == 200, r.text
    argv = captured['argv']
    assert "--targets" in argv and argv[argv.index("--targets") + 1] == "a,b,c"
    assert "--target-concurrency" in argv, "多目标必须拼 --target-concurrency"
    assert argv[argv.index("--target-concurrency") + 1] == "3", "默认全并发 = 目标数"

    # 显式覆盖
    captured.clear()
    r = client.post('/api/run/evaluate', json={
        "input": "example.jsonl", "targets": "a,b", "target_concurrency": 1, "max_rounds": 3, "batch_size": 3})
    argv = captured['argv']
    assert argv[argv.index("--target-concurrency") + 1] == "1", "显式 target_concurrency 生效"

    print('✅ 多目标并发 argv 拼接通过')



def test_spawn_injects_task_id_env(monkeypatch, tmp_path):
    """_spawn 必须把 LLMSEC_TASK_ID 注入子进程 env（进度落盘的钥匙）。"""
    import llmsec.server.routers.tasks as tasks_mod

    captured = {}

    class FakeProc:
        returncode = 0

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            pass

        def kill(self):
            pass

    def fake_popen(*args, **kwargs):
        captured['env'] = kwargs.get('env')
        return FakeProc()

    monkeypatch.setattr(tasks_mod.subprocess, "Popen", fake_popen)

    t = {"kind": "smoke", "argv": ["-c", "pass"], "cmd": "pass",
         "log_path": tmp_path / "t.log", "log_file": None,
         "status": "queued", "started_at": "2026-01-01T00:00:00", "proc": None}
    try:
        tasks_mod._spawn("tid-inject", t)
    finally:
        if t.get("log_file"):
            t["log_file"].close()

    assert captured.get('env', {}).get("LLMSEC_TASK_ID") == "tid-inject", \
        "子进程 env 必须注入 LLMSEC_TASK_ID"
    print('✅ _spawn 注入 LLMSEC_TASK_ID 通过')



def test_task_stream_progress_events(monkeypatch):
    """SSE /stream：回放射已有 progress 行（event:progress）+ 结束发 event:done。"""
    import json

    import llmsec.server.routers.tasks as tasks_mod

    tid = "stream-ut"
    pp = tasks_mod._progress_path(tid)
    tasks_mod.TASK_LOG_DIR.mkdir(parents=True, exist_ok=True)
    pp.write_text(
        json.dumps({"ts": "t1", "phase": "attack", "target": "a", "round": 1, "elo": 1500.0}) + "\n"
        + json.dumps({"ts": "t2", "phase": "attack", "target": "a", "round": 2, "elo": 1505.0}) + "\n",
        encoding="utf-8")
    # 终态：生成器首次 while 即发 done 返回（不挂起）
    tasks_mod.TASKS[tid] = {
        "kind": "evaluate", "argv": ["--targets", "a"], "cmd": "", "status": "success",
        "returncode": 0, "log_path": tasks_mod.TASK_LOG_DIR / f"{tid}.log", "log_file": None,
        "started_at": "2026-01-01T00:00:00", "proc": None}
    try:
        r = client.get(f"/api/tasks/{tid}/stream")
        assert r.status_code == 200
        body = r.text
        assert "event: progress" in body, "应回放射 progress 事件"
        assert "event: done" in body, "任务终态应发 done 事件"
        assert '"round": 2' in body, "应包含最新进度行（round 2）"
    finally:
        tasks_mod.TASKS.pop(tid, None)
        if pp.exists():
            pp.unlink()

    print('✅ SSE progress 流通过')


def test_runs_active_marking(monkeypatch, tmp_path):
    """/api/runs：有 running evaluate 任务时，批次 ts ≥ 任务 started_at 的 run 标 active。"""
    import json

    import llmsec.server.dashboard_api as api
    from llmsec.server.routers import data_query as dq
    from llmsec.server.routers.tasks import TASKS

    # 构造两个批次：旧批次（任务开始前）与新批次（任务进行中）
    for ts, tgt in [("2026-08-10_100000", "modelA"), ("2026-08-11_150000", "modelB")]:
        d = tmp_path / ts / tgt
        d.mkdir(parents=True)
        (d / "runner_report.json").write_text(json.dumps({
            "target_model": tgt, "security_level": "safe",
            "attack_phase": {"asr": 0.1}, "allergy": {}, "elo": {}}), encoding="utf-8")
    monkeypatch.setattr(api, "RUNS_DIR", tmp_path)
    dq._RUN_META_CACHE.clear()

    tid = "evaluate-active-ut"
    TASKS[tid] = {"kind": "evaluate", "status": "running",
                  "started_at": "2026-08-11T14:59:00", "argv": [], "cmd": "",
                  "returncode": None, "log_path": None, "log_file": None, "proc": None}
    try:
        r = client.get("/api/runs")
        assert r.status_code == 200
        by_name = {x["name"]: x for x in r.json()["runs"]}
        assert by_name["2026-08-11_150000/modelB"].get("active") is True, "进行中新批次应标 active"
        assert "active" not in by_name["2026-08-10_100000/modelA"], "旧批次不应标 active"
    finally:
        TASKS.pop(tid, None)

    # 无运行任务 → 全部无 active
    r = client.get("/api/runs")
    assert all("active" not in x for x in r.json()["runs"]), "无运行任务时不应有 active 标注"
    print("✅ /api/runs active 标注通过")


def test_run_meta_cache_invalidation(monkeypatch, tmp_path):
    """_run_meta：同目录覆写 runner_report.json（size 变）必须重读，不能吃目录 mtime 旧缓存。"""
    import json

    import llmsec.server.dashboard_api as api
    from llmsec.server.routers import data_query as dq

    run_dir = tmp_path / "2026-08-11_160000" / "modelC"
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(api, "RUNS_DIR", tmp_path)
    dq._RUN_META_CACHE.clear()

    (run_dir / "runner_report.json").write_text(json.dumps({
        "target_model": "modelC", "security_level": "safe", "attack_phase": {"asr": 0.10},
        "allergy": {}, "elo": {}}), encoding="utf-8")
    m1 = dq._run_meta(run_dir)
    assert m1["asr"] == 0.10

    # 覆写（内容更长 → size 变）：resume 续跑场景
    (run_dir / "runner_report.json").write_text(json.dumps({
        "target_model": "modelC", "security_level": "risky", "attack_phase": {"asr": 0.55},
        "allergy": {}, "elo": {}}), encoding="utf-8")
    m2 = dq._run_meta(run_dir)
    assert m2["asr"] == 0.55 and m2["security_level"] == "risky", "覆写后缓存必须失效重读"
    print("✅ _run_meta 缓存失效通过")


def test_probe_includes_services(monkeypatch):
    """/api/targets/probe：全量探活含 generator+judge；同端点复用 list；模型名不在列表 → warning。"""
    from types import SimpleNamespace

    import llmsec.core.config as cfg_mod
    import llmsec.core.llm as llm_mod

    monkeypatch.setattr(cfg_mod, "load_targets", lambda: {})
    gen_cfg = SimpleNamespace(api_key="k", base_url="http://g/v1", model="gen-x")
    judge_cfg = SimpleNamespace(api_key="k", base_url="http://g/v1", model="judge-y")
    monkeypatch.setattr(cfg_mod.GeneratorConfig, "from_env", classmethod(lambda cls: gen_cfg))
    monkeypatch.setattr(cfg_mod.JudgeConfig, "from_env", classmethod(lambda cls: judge_cfg))

    calls = {"n": 0}

    class FakeClient:
        def __init__(self, *a, **k):
            calls["n"] += 1

        class models:
            @staticmethod
            def list():
                return [SimpleNamespace(id="gen-x")]  # judge-y 不在列表

        # chat smoke（探活第二段）：正常返回非 None content，不产生 chat warning
        class _Completions:
            @staticmethod
            def create(**kw):
                return SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(content="pong"))])

        chat = SimpleNamespace(completions=_Completions())

    monkeypatch.setattr(llm_mod, "create_openai_client", lambda *a, **k: FakeClient())

    r = client.get("/api/targets/probe")
    assert r.status_code == 200
    d = r.json()
    svcs = {s["name"]: s for s in d["services"]}
    assert set(svcs) == {"generator", "judge"}, "services 应含 generator 与 judge"
    assert svcs["generator"]["reachable"] is True
    assert svcs["generator"]["warning"] is None, "gen-x 在列表且 chat 正常 → 无 warning"
    assert svcs["judge"]["warning"], "judge-y 不在列表 → 应有 warning"
    # models.list 同端点复用（1 次）；chat smoke 各 service 各建 1 次 client（+2）。
    # 原"只调 1 次"的断言因新增 chat smoke 而调整为 3。
    assert calls["n"] == 3, f"models.list 复用 1 + chat smoke generator/judge 各 1 = 3，实际 {calls['n']}"
    print("✅ /api/targets/probe services 通过")
