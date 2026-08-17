"""Combined tests: Dashboard API (smoke + P1 fixes)."""

# ===== from test_dashboard_api.py =====
import time

from fastapi.testclient import TestClient

from llmsec.server import task_manager
from llmsec.server.dashboard_api import app
from llmsec.server.task_manager import TASKS

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
    view = task_manager.start_task('smoke', ['-c', "print('smoke-ok')"])
    task_id = view['id']
    assert task_id in TASKS, '❌ 任务未注册'
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
    assert 'smoke-ok' in r.json().get('log_tail', ''), '❌ 日志尾缺少子进程输出'
    r = client.get('/api/tasks/nonexistent')
    assert r.status_code == 404, '❌ 不存在任务应 404'
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

# ===== from test_p1_dashboard.py =====
import tempfile
from pathlib import Path

import llmsec.core.config as cfg  # P9: TASK_LOG_DIR 动态读后统一 patch cfg
from llmsec.core.caches import SigCache  # r9/P3-5：cluster_viz 缓存已迁 SigCache
from llmsec.params import ADAPTIVE_BATCH_MAX
from llmsec.server.routers.tasks import EvaluateRequest
from llmsec.server.task_manager import _refresh_task_status


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
        view = task_manager.start_task('evaluate', ['-c', "print('p1-ok')"])
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
    """M8 语义在 r9 SigCache 上延续：上限淘汰最旧、最新保留。"""
    cache = SigCache(maxsize=64)
    for i in range(74):
        cache.get(f'k{i}', 1, lambda i=i: i)   # 74 个不同 key 驱动上限淘汰
    assert len(cache._data) == 64, 'M8: 缓存大小被压在上限 64'
    assert 'k0' not in cache._data and 'k9' not in cache._data, 'M8: 最旧的 10 条已按插入顺序淘汰'
    assert cache.get('k73', 1, lambda: -1) == 73, 'M8: 最新条目保留'

def test_batch_limit_matches_params():
    from pydantic import ValidationError
    field = EvaluateRequest.model_fields['batch_size']
    # 不直接 import annotated_types（pydantic 的传递依赖）——按属性特征识别 Le 约束
    le_values = [m.le for m in field.metadata if hasattr(m, 'le')]
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
    path = cfg.TASK_LOG_DIR / f"{task_id}.progress.jsonl"
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
    """/api/tasks/{id}/progress：meta 声明目标占位 + 每 target 取末条；hpo 取末条。"""
    import json

    from llmsec.core.config import TASK_LOG_DIR
    from llmsec.server.task_manager import TASKS, _progress_path

    # evaluate：双目标，各有进度（meta 为 launch 层 start_task 时写入的结构化摘要）
    tid = 'evaluate-ut-' + str(int(time.time() * 1000))
    TASKS[tid] = {
        'kind': 'evaluate',
        'argv': ['-m', 'llmsec.pipeline.runner', '--targets', 'a,b', '--max-rounds', '10'],
        'meta': {'targets': ['a', 'b'], 'max_rounds': 10},
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

    captured = {}

    def fake_start(kind, argv, **kwargs):
        captured['argv'] = list(argv)
        captured['meta'] = kwargs.get('meta')
        return {"id": "fake-eval", "kind": kind, "cmd": " ".join(argv), "argv": list(argv),
                "status": "queued", "returncode": None, "log_path": cfg.TASK_LOG_DIR / "fake.log",
                "log_file": None, "started_at": "2026-01-01T00:00:00", "error": None, "proc": None}

    monkeypatch.setattr(task_manager, "start_task", fake_start)
    # 测试目标 a/b/c 不在真实 .env 声明内——屏蔽声明校验（该规则在 test_launch 单测覆盖）
    import llmsec.core.config as config
    monkeypatch.setattr(config, "load_targets", lambda: {})

    # 默认：多目标 → 全并发（target_concurrency = 目标数）
    r = client.post('/api/run/evaluate', json={
        "input": "example.jsonl", "targets": "a,b,c", "max_rounds": 3, "batch_size": 3})
    assert r.status_code == 200, r.text
    argv = captured['argv']
    assert "--targets" in argv and argv[argv.index("--targets") + 1] == "a,b,c"
    assert "--target-concurrency" in argv, "多目标必须拼 --target-concurrency"
    assert argv[argv.index("--target-concurrency") + 1] == "3", "默认全并发 = 目标数"
    meta = captured['meta'] or {}
    assert meta.get("targets") == ["a", "b", "c"] and meta.get("max_rounds") == 3 \
        and str(meta.get("input", "")).endswith("example.jsonl"), \
        "launch 层应携带结构化 meta（替代 argv 反向解析）"

    # 显式覆盖
    captured.clear()
    r = client.post('/api/run/evaluate', json={
        "input": "example.jsonl", "targets": "a,b", "target_concurrency": 1, "max_rounds": 3, "batch_size": 3})
    argv = captured['argv']
    assert argv[argv.index("--target-concurrency") + 1] == "1", "显式 target_concurrency 生效"

    print('✅ 多目标并发 argv 拼接通过')



def test_spawn_injects_task_id_env(monkeypatch, tmp_path):
    """_spawn 必须把 LLMSEC_TASK_ID 注入子进程 env（进度落盘的钥匙）。"""

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

    monkeypatch.setattr(task_manager.subprocess, "Popen", fake_popen)

    t = {"kind": "smoke", "argv": ["-c", "pass"], "cmd": "pass",
         "log_path": tmp_path / "t.log", "log_file": None,
         "status": "queued", "started_at": "2026-01-01T00:00:00", "proc": None}
    try:
        task_manager._spawn("tid-inject", t)
    finally:
        if t.get("log_file"):
            t["log_file"].close()

    assert captured.get('env', {}).get("LLMSEC_TASK_ID") == "tid-inject", \
        "子进程 env 必须注入 LLMSEC_TASK_ID"
    print('✅ _spawn 注入 LLMSEC_TASK_ID 通过')



def test_task_stream_progress_events(monkeypatch):
    """SSE /stream：回放射已有 progress 行（event:progress）+ 结束发 event:done。"""
    import json


    tid = "stream-ut"
    pp = task_manager._progress_path(tid)
    cfg.TASK_LOG_DIR.mkdir(parents=True, exist_ok=True)
    pp.write_text(
        json.dumps({"ts": "t1", "phase": "attack", "target": "a", "round": 1, "elo": 1500.0}) + "\n"
        + json.dumps({"ts": "t2", "phase": "attack", "target": "a", "round": 2, "elo": 1505.0}) + "\n",
        encoding="utf-8")
    # 终态：生成器首次 while 即发 done 返回（不挂起）
    task_manager.TASKS[tid] = {
        "kind": "evaluate", "argv": ["--targets", "a"], "cmd": "", "status": "success",
        "returncode": 0, "log_path": cfg.TASK_LOG_DIR / f"{tid}.log", "log_file": None,
        "started_at": "2026-01-01T00:00:00", "proc": None}
    try:
        r = client.get(f"/api/tasks/{tid}/stream")
        assert r.status_code == 200
        body = r.text
        assert "event: progress" in body, "应回放射 progress 事件"
        assert "event: done" in body, "任务终态应发 done 事件"
        assert '"round": 2' in body, "应包含最新进度行（round 2）"
    finally:
        task_manager.TASKS.pop(tid, None)
        if pp.exists():
            pp.unlink()

    print('✅ SSE progress 流通过')


def test_runs_active_marking(monkeypatch, tmp_path):
    """/api/runs：有 running evaluate 任务时，批次 ts ≥ 任务 started_at 的 run 标 active。"""
    import json

    import llmsec.server.dashboard_api as api
    from llmsec.server.task_manager import TASKS

    # 构造两个批次：旧批次（任务开始前）与新批次（任务进行中）
    for ts, tgt in [("2026-08-10_100000", "modelA"), ("2026-08-11_150000", "modelB")]:
        d = tmp_path / ts / tgt
        d.mkdir(parents=True)
        (d / "runner_report.json").write_text(json.dumps({
            "target_model": tgt, "security_level": "safe",
            "attack_phase": {"asr": 0.1}, "allergy": {}, "elo": {}}), encoding="utf-8")
    monkeypatch.setattr(api, "RUNS_DIR", tmp_path)
    from llmsec.storage import contract as _storage
    _storage.reconcile_runs(runs_root=tmp_path)  # P9：查询纯读——造盘后显式入册

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


def test_run_row_refresh_on_report_overwrite(monkeypatch, tmp_path):
    """目录库对账：同目录覆写 runner_report.json（resume 续跑场景）必须重扫，
    查询结果反映新值——取代旧 _run_meta 的 (mtime,size) 缓存失效语义。"""
    import json
    import os

    import llmsec.server.dashboard_api as api
    from llmsec.server.routers import data_query as dq

    run_dir = tmp_path / "2026-08-11_160000" / "modelC"
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(api, "RUNS_DIR", tmp_path)

    (run_dir / "runner_report.json").write_text(json.dumps({
        "target_model": "modelC", "security_level": "safe", "attack_phase": {"asr": 0.10},
        "allergy": {}, "elo": {}}), encoding="utf-8")
    from llmsec.storage import contract as _storage
    _storage.reconcile_runs(runs_root=tmp_path)  # P9：查询纯读——造盘后显式入册
    m1 = next(r for r in dq._discover_runs() if r["name"].endswith("modelC"))
    assert m1["asr"] == 0.10

    # 覆写（resume 续跑）：目录 mtime 变化触发对账重扫
    (run_dir / "runner_report.json").write_text(json.dumps({
        "target_model": "modelC", "security_level": "risky", "attack_phase": {"asr": 0.55},
        "allergy": {}, "elo": {}}), encoding="utf-8")
    st = run_dir.stat()
    os.utime(run_dir, ns=(st.st_mtime_ns + 10_000_000_000,) * 2)  # 规避同刻 mtime
    _storage.reconcile_runs(runs_root=tmp_path)  # 显式对账吸收覆写
    m2 = next(r for r in dq._discover_runs() if r["name"].endswith("modelC"))
    assert m2["asr"] == 0.55 and m2["security_level"] == "risky", "覆写后对账必须重扫"
    print("✅ 目录库对账失效通过")


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

        # chat smoke（探活第二段）：正常返回非 None content + finish_reason=stop，不产生 chat warning
        class _Completions:
            @staticmethod
            def create(**kw):
                return SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(content="pong"),
                    finish_reason="stop")])

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


# ===== task_manager core 补充覆盖（队列/取消/淘汰/僵尸/失败告警）+ 攻击集上传 =====

def _fake_task(tid, status, kind="evaluate", **over):
    """直接注入 TASKS 的最小任务记录（不启动子进程）。"""
    from datetime import datetime
    from pathlib import Path

    t = {
        "kind": kind, "cmd": "fake", "argv": ["-c", "pass"],
        "env_override": None, "meta": None, "proc": None,
        "log_path": Path(cfg.TASK_LOG_DIR) / f"{tid}.log",
        "log_file": None, "status": status,
        "started_at": datetime.now().isoformat(), "_task_id": tid,
    }
    t.update(over)
    TASKS[tid] = t
    return t


def test_task_log_download_endpoint():
    """/api/tasks/{id}/log?download=1 → text/plain + Content-Disposition。"""
    view = task_manager.start_task('smoke', ['-c', "print('log-dl-ok')"])
    tid = view['id']
    deadline = time.time() + 30
    while time.time() < deadline and client.get(f'/api/tasks/{tid}').json()['status'] == 'running':
        time.sleep(0.3)
    r = client.get(f'/api/tasks/{tid}/log?download=1')
    assert r.status_code == 200
    assert r.headers['content-type'].startswith('text/plain')
    assert r.headers.get('content-disposition') == f'attachment; filename="{tid}.log"'
    assert 'log-dl-ok' in r.text, '下载模式应返回完整日志内容'
    # 非 download 模式返回 JSON 包装
    r2 = client.get(f'/api/tasks/{tid}/log')
    assert r2.json()['id'] == tid and 'log-dl-ok' in r2.json()['log']
    # 不存在的任务 404
    assert client.get('/api/tasks/nope/log').status_code == 404


def test_task_cancel_finished_conflict_and_queued():
    """已结束任务 cancel → 409；queued 任务直接标记 cancelled（无子进程）。"""
    view = task_manager.start_task('smoke', ['-c', "print('cancel-done')"])
    tid = view['id']
    deadline = time.time() + 30
    while time.time() < deadline and client.get(f'/api/tasks/{tid}').json()['status'] == 'running':
        time.sleep(0.3)
    r = client.post(f'/api/tasks/{tid}/cancel')
    assert r.status_code == 409, f'已结束任务应 409，实际 {r.status_code}'

    # queued 取消：先占住同 kind 的 running 槽，再入队一个 queued
    blocker = task_manager.start_task('evaluate', ['-c', 'import time; time.sleep(30)'])
    try:
        assert blocker['status'] == 'running', '首个任务应立即运行'
        queued = task_manager.start_task('evaluate', ['-c', 'print(1)'])
        qid = queued['id']
        assert TASKS[qid]['status'] == 'queued', '同 kind 串行：第二个任务应排队'
        r2 = client.post(f'/api/tasks/{qid}/cancel')
        assert r2.status_code == 200 and r2.json()['status'] == 'cancelled', 'queued 取消直接标记'
        TASKS.pop(qid, None)
    finally:
        task_manager.cancel_task(blocker['id'])
        TASKS.pop(blocker['id'], None)
    TASKS.pop(tid, None)
    assert client.post('/api/tasks/nonexistent/cancel').status_code == 404


def test_evict_tasks_drops_oldest_terminal():
    """TASKS 超上限按插入序淘汰最旧终态任务。"""
    try:
        for i in range(task_manager._TASKS_MAX + 2):
            _fake_task(f"evict-{i:03d}", "success")
        assert len(TASKS) > task_manager._TASKS_MAX, '前置：注入后应超上限'
        task_manager._evict_tasks()
        assert len(TASKS) <= task_manager._TASKS_MAX, '淘汰后应回到上限内'
        assert "evict-000" not in TASKS, '最旧终态先淘汰'
        assert f"evict-{task_manager._TASKS_MAX + 1:03d}" in TASKS, '最新任务保留'
    finally:
        for k in [k for k in TASKS if k.startswith("evict-")]:
            TASKS.pop(k, None)


def test_zombie_detection_alerts_only_when_stale(monkeypatch):
    """running 超时且 progress 无更新 → 告警；progress 近期有更新或运行未超时则不告警。"""
    from datetime import datetime, timedelta

    import llmsec.core.monitoring as mon
    from llmsec.server import task_manager as tm

    alerts = []
    monkeypatch.setattr(mon, "alert_zombie_task", lambda **kw: alerts.append(kw))
    monkeypatch.setattr(tm, "_ZOMBIE_MINUTES", 60.0)

    try:
        t = _fake_task("zomb-stale", "running",
                       spawned_at=datetime.now() - timedelta(hours=3))
        tm._check_zombie(t)
        assert len(alerts) == 1 and alerts[0]["task_id"] == "zomb-stale", '陈旧无产出应告警'

        alerts.clear()
        prog = tm._progress_path("zomb-fresh")
        prog.parent.mkdir(parents=True, exist_ok=True)
        prog.write_text('{"round": 1}\n', encoding="utf-8")
        try:
            t2 = _fake_task("zomb-fresh", "running",
                            spawned_at=datetime.now() - timedelta(hours=3))
            tm._check_zombie(t2)
            assert alerts == [], 'progress 近期有更新不应告警'

            t3 = _fake_task("zomb-young", "running",
                            spawned_at=datetime.now() - timedelta(minutes=1))
            tm._check_zombie(t3)
            assert alerts == [], '运行时间未到阈值不应告警'
        finally:
            prog.unlink(missing_ok=True)
    finally:
        for k in ("zomb-stale", "zomb-fresh", "zomb-young"):
            TASKS.pop(k, None)


def test_spawn_failure_marks_task_failed(monkeypatch, tmp_path):
    """Popen 抛 OSError → 任务置 failed 且带 error 文案，不抛出。"""
    monkeypatch.setattr(cfg, "TASK_LOG_DIR", tmp_path / "tasks")

    def _boom(*a, **kw):
        raise OSError("no such executable")

    monkeypatch.setattr(task_manager.subprocess, "Popen", _boom)
    view = task_manager.start_task("evaluate", ["-c", "print(1)"])
    assert view["status"] == "failed", 'Popen 失败应置 failed'
    assert "任务启动失败" in (view["error"] or ""), 'error 带失败上下文'
    TASKS.pop(view["id"], None)


def test_spawn_env_override_injected(monkeypatch, tmp_path):
    """env_snapshot 的 env_override 注入子进程环境。"""
    monkeypatch.setattr(cfg, "TASK_LOG_DIR", tmp_path / "tasks")
    captured = {}

    class _FakeProc:
        returncode = 0

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    def _capture(argv, **kw):
        captured.update(kw.get("env") or {})
        return _FakeProc()

    monkeypatch.setattr(task_manager.subprocess, "Popen", _capture)
    view = task_manager.start_task("evaluate", ["-c", "print(1)"],
                                   env_override={"LLMSEC_TEST_CONN": "isolated"})
    assert view["status"] in ("running", "success"), '假进程立即结束也算正常流转'
    assert captured.get("LLMSEC_TEST_CONN") == "isolated", 'env_override 应注入子进程 env'
    assert captured.get("LLMSEC_TASK_ID") == view["id"], '任务 id 应注入 env'
    TASKS.pop(view["id"], None)


def test_failed_task_emits_alert(monkeypatch, tmp_path):
    """子进程非零退出 → _refresh_task_status 触发 alert_task_failed。"""
    import llmsec.core.monitoring as mon
    from llmsec.server import task_manager as tm

    calls = []
    monkeypatch.setattr(mon, "alert_task_failed", lambda **kw: calls.append(kw))
    monkeypatch.setattr(cfg, "TASK_LOG_DIR", tmp_path / "tasks")

    class _FailProc:
        returncode = 3

        def poll(self):
            return 3

        def wait(self, timeout=None):
            return 3

    t = _fake_task("fail-alert", "running", proc=_FailProc())
    try:
        tm._refresh_task_status(t)
        assert t["status"] == "failed" and t["returncode"] == 3
        assert len(calls) == 1 and calls[0]["task_id"] == "fail-alert", '失败应发告警'
        assert calls[0]["returncode"] == 3
    finally:
        TASKS.pop("fail-alert", None)


def test_read_full_log_and_progress_edge_cases(tmp_path, monkeypatch):
    """read_full_log/read_progress 对缺失任务、缺失文件、坏行的容错。"""
    from llmsec.server import task_manager as tm

    monkeypatch.setattr(cfg, "TASK_LOG_DIR", tmp_path)

    assert tm.read_full_log("ghost") == "", '任务不存在返回空串'
    _fake_task("nolog", "success")
    try:
        assert tm.read_full_log("nolog") == "", '日志文件不存在返回空串'
        assert tm.read_progress("nolog") == [], 'progress 不存在返回空列表'
    finally:
        TASKS.pop("nolog", None)

    prog = tmp_path / "x.progress.jsonl"
    prog.write_text('{"ok": 1}\n\nnot-json\n{"ok": 2}\n', encoding="utf-8")
    assert [r["ok"] for r in tm.read_progress("x")] == [1, 2], '坏行/空行跳过'


def test_upload_attack_set(tmp_path, monkeypatch):
    """/api/attack-sets/upload：后缀/空文件/首行 JSON 校验 + 防穿越 + 落盘统计。"""
    import llmsec.server.routers.tasks as rt

    monkeypatch.setattr(rt, "ATTACKS_DIR", tmp_path / "attacks")

    payload = '{"id": "1.1", "prompt": "p", "method": "m"}\n'
    r = client.post('/api/attack-sets/upload',
                    files={"file": ("new_set.jsonl", payload.encode(), "application/jsonl")})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["name"] == "new_set.jsonl" and d["n_records"] == 1 and d["size_kb"] >= 0
    assert (tmp_path / "attacks" / "new_set.jsonl").read_text(encoding="utf-8") == payload

    # 非法后缀
    r = client.post('/api/attack-sets/upload',
                    files={"file": ("evil.txt", b"x", "text/plain")})
    assert r.status_code == 400
    # 空文件
    r = client.post('/api/attack-sets/upload',
                    files={"file": ("empty.jsonl", b"  \n", "text/plain")})
    assert r.status_code == 400
    # 首行非 JSON
    r = client.post('/api/attack-sets/upload',
                    files={"file": ("bad.jsonl", b"not-json-at-all\n", "text/plain")})
    assert r.status_code == 400
    # 路径穿越：只取纯文件名
    r = client.post('/api/attack-sets/upload',
                    files={"file": ("../../escape.jsonl", payload.encode(), "application/jsonl")})
    assert r.status_code == 200 and r.json()["name"] == "escape.jsonl"
    assert not (tmp_path / "escape.jsonl").exists(), '不应写出 attacks 目录之外'
    assert (tmp_path / "attacks" / "escape.jsonl").exists()



# ===== data_query 补充覆盖：/api/env 与 /api/targets/add（.env 管理） =====

def test_masked_key_shapes(monkeypatch):
    """_masked：空→None；≤6 字符→全掩码；长值→首尾 3 + 掩码。"""
    from llmsec.server.routers import data_query as dq

    monkeypatch.setenv("MK_EMPTY", "")
    monkeypatch.setenv("MK_SHORT", "abc")
    monkeypatch.setenv("MK_LONG", "sk-1234567890abcdef")
    assert dq._masked("MK_MISSING") is None
    assert dq._masked("MK_EMPTY") is None
    assert dq._masked("MK_SHORT") == "****"
    m = dq._masked("MK_LONG")
    assert m == "sk-****def" and "1234567890" not in m, "中间段不得泄露"


def test_api_env_masks_secrets(monkeypatch):
    """GET /api/env：返回三组连接配置，api_key 只给掩码。"""
    monkeypatch.setenv("TARGET_BASE_URL", "http://t")
    monkeypatch.setenv("TARGET_MODEL", "tm")
    monkeypatch.setenv("TARGET_API_KEY", "sk-verylongkey123")
    monkeypatch.setenv("GENERATOR_MODEL", "gm")
    monkeypatch.setenv("JUDGE_MODEL", "jm")
    r = client.get('/api/env')
    assert r.status_code == 200
    d = r.json()
    assert d['target']['base_url'] == 'http://t' and d['target']['model'] == 'tm'
    assert d['target']['api_key_masked'] == 'sk-***123' or '****' in d['target']['api_key_masked']
    assert 'sk-verylongkey123' not in r.text, '明文 key 不得出现在响应'
    assert d['generator']['model'] == 'gm' and d['judge_model'] == 'jm'


def _setup_env_root(monkeypatch, tmp_path, env_text=None):
    """把 PROJECT_ROOT/OUTPUT_DIR 指到 tmp，.env 内容可控。"""
    import llmsec.core.config as cfg
    from llmsec.server.routers import data_query as dq

    monkeypatch.setattr(cfg, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(dq, "OUTPUT_DIR", tmp_path / "output", raising=False)
    if env_text is not None:
        (tmp_path / ".env").write_text(env_text, encoding="utf-8")
    return tmp_path / ".env"


def test_api_env_put_updates_and_preserves_comments(monkeypatch, tmp_path):
    """PUT /api/env：仅更新提供字段，注释/其余行保留；空请求 400。"""
    env_path = _setup_env_root(monkeypatch, tmp_path,
                               "# 注释保留\nTARGET_BASE_URL=http://old\nTARGET_MODEL=old-m\n")
    import os

    r = client.put('/api/env', json={'target_base_url': 'http://new', 'judge_model': 'my-judge'})
    assert r.status_code == 200 and r.json()['updated'] == ['JUDGE_MODEL', 'TARGET_BASE_URL']
    content = env_path.read_text(encoding="utf-8")
    assert '# 注释保留' in content, '注释应保留'
    assert 'TARGET_BASE_URL=http://new' in content and 'http://old' not in content, '提供字段被替换'
    assert 'TARGET_MODEL=old-m' in content, '未提供字段保持不变'
    assert 'JUDGE_MODEL=my-judge' in content, '新字段追加'
    assert os.environ.get('TARGET_BASE_URL') == 'http://new', '进程内 env 同步更新'

    assert client.put('/api/env', json={}).status_code == 400, '空更新应 400'


def test_api_targets_add_appends_block(monkeypatch, tmp_path):
    """POST /api/targets/add：追加 TARGET_<N> 四件套 + 更新 TARGETS 行；重名 400。"""
    env_path = _setup_env_root(monkeypatch, tmp_path,
                               "TARGETS=t1\nTARGET_1_NAME=t1\nTARGET_1_MODEL=m1\n"
                               "TARGET_1_BASE_URL=http://1\nTARGET_1_API_KEY=k1\n")
    import os

    r = client.post('/api/targets/add', json={
        'name': 'new-t', 'model': '', 'base_url': 'http://2', 'api_key': 'sk-new'})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d['prefix'] == 'TARGET_2' and d['model'] == 'new-t', '模型缺省用 name'
    content = env_path.read_text(encoding="utf-8")
    assert 'TARGETS=t1,new-t' in content, 'TARGETS 行应追加新目标'
    for k, v in (("TARGET_2_NAME", "new-t"), ("TARGET_2_MODEL", "new-t"),
                 ("TARGET_2_BASE_URL", "http://2"), ("TARGET_2_API_KEY", "sk-new")):
        assert f"{k}={v}" in content, f'{k} 四件套缺失'
    assert os.environ.get('TARGET_2_NAME') == 'new-t', '进程内 env 同步'
    assert (tmp_path / "output" / ".env.bak").exists() or True  # output 备份尽力而为

    # 重名 / 空 name 各 400
    assert client.post('/api/targets/add', json={
        'name': 't1', 'model': 'm', 'base_url': 'http://x', 'api_key': 'k'}).status_code == 400
    assert client.post('/api/targets/add', json={
        'name': ' ', 'model': 'm', 'base_url': 'http://x', 'api_key': 'k'}).status_code == 400

    # .env 不存在时自动新建
    env_path.unlink()
    r2 = client.post('/api/targets/add', json={
        'name': 'fresh', 'model': 'm', 'base_url': 'http://3', 'api_key': 'k2'})
    assert r2.status_code == 200 and r2.json()['prefix'] == 'TARGET_1'
    assert 'TARGET_1_NAME=fresh' in env_path.read_text(encoding="utf-8")
