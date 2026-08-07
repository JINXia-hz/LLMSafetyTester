"""
P0 数据完整性与正确性回归测试。

覆盖本轮整改的致命问题修复：
  F1: elo.update NaN/inf 校验（防止级联污染整个评级系统）
  F3: io.py 原子写 + strict + CorruptedFileError + backup；
      results/state 损坏时备份残文件 + 警告（不静默清零）
  F4: config.load_targets / target_backend 共享 name→idx 映射（修复路由错配）
  F5: local_model_server 调换数学/有害优先级（SIM_REFUSAL_RATE 不再失效）

约定：每个 test 返回 0=通过 / 1=失败；main() 汇总，sys.exit(main())。
"""
import json
import math
import os
import random
import tempfile
from pathlib import Path

from llmsec.core.config import _resolve_target_prefixes, load_targets, target_backend
from llmsec.core.io import CorruptedFileError, read_json, write_json
from llmsec.core.results import ResultsMatrix
from llmsec.evaluation.elo import ELOTracker


def test_f1_nan_does_not_pollute():
    """NaN eval_score 不应污染 attacker_ratings（原来会级联损坏所有对手）。"""
    tr = ELOTracker()
    tr.update('good', 'def', 3.0)
    tr.update('bad', 'def', float('nan'))
    assert math.isfinite(tr.get_attacker_elo('bad')), 'F1: NaN eval_score 后 bad 方法 Elo 仍有限'
    assert tr.get_attacker_elo('bad') != float('nan'), 'F1: bad 方法 Elo 非 NaN'
    tr.update('good', 'def', 2.0)
    expected_now = tr._expected(tr.get_attacker_elo('good'), tr.get_defender_elo('def'))
    assert math.isfinite(expected_now), 'F1: 经 NaN 污染后 expected 胜率仍有限（无级联）'
    assert math.isfinite(tr.get_attacker_elo('good')), 'F1: good 方法 Elo 仍有限'

def test_f1_inf_does_not_pollute():
    """+inf eval_score：inf>0 通过 → inf/(inf+2)=NaN，原代码会污染。"""
    tr = ELOTracker()
    tr.update('m', 'def', float('inf'))
    assert math.isfinite(tr.get_attacker_elo('m')), 'F1: +inf eval_score 后 Elo 仍有限'
    assert math.isfinite(tr.get_defender_elo('def')), 'F1: +inf eval_score 后防御方 Elo 仍有限'

def test_f1_string_invalid():
    """字符串/None 等非法 eval_score 也不应崩溃（TypeError 兜底）。"""
    tr = ELOTracker()
    try:
        tr.update('m', 'def', 'not_a_number')
        assert math.isfinite(tr.get_attacker_elo('m')), 'F1: 字符串 eval_score 兜底为 0，Elo 有限'
    except (TypeError, ValueError):
        assert False, 'F1: 字符串 eval_score 不应抛异常（应兜底为 0）'

def test_f1_normal_still_works():
    """F1 修复不应破坏正常评分路径。"""
    tr = ELOTracker()
    tr.update('winner', 'def', 3.0)
    tr.update('loser', 'def', -2.0)
    assert tr.get_attacker_elo('winner') > tr.get_attacker_elo('loser'), 'F1: 正常 update 仍让胜者 Elo 高于败者'
    assert tr.get_attacker_elo('winner') > 1500.0, 'F1: 胜者 Elo 高于初始值'

def test_f1_save_load_no_nan_persistence():
    """NaN 不应经 save/load 持久化进 state.json（JSON 允许 NaN 字面量）。"""
    tr = ELOTracker()
    tr.update('m', 'def', float('nan'))
    with tempfile.TemporaryDirectory() as d:
        sf = Path(d) / 'state.json'
        tr.save(sf)
        raw = sf.read_text(encoding='utf-8')
        assert 'NaN' not in raw and 'nan' not in raw.lower().replace('typename', ''), 'F1: state.json 不含 NaN 字面量'
        tr2 = ELOTracker()
        tr2.load(sf)
        assert math.isfinite(tr2.get_attacker_elo('m')), 'F1: load 后 Elo 仍有限'

def test_f3_write_json_atomic_no_tmp_residue():
    """原子写成功后 .tmp 不残留，文件内容正确。"""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'a.json'
        write_json(p, {'x': 1}, atomic=True)
        assert p.exists() and json.loads(p.read_text('utf-8')) == {'x': 1}, 'F3: 原子写后文件内容正确'
        assert not (Path(d) / 'a.json.tmp').exists(), 'F3: 原子写后 .tmp 不残留'

def test_f3_write_json_backup():
    """backup=True 时写前复制 .bak，内容是旧版。"""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'a.json'
        write_json(p, {'v': 1})
        write_json(p, {'v': 2}, backup=True)
        bak = Path(d) / 'a.json.bak'
        assert bak.exists(), 'F3: backup=True 产生 .bak'
        assert json.loads(bak.read_text('utf-8')) == {'v': 1}, 'F3: .bak 内容是旧版'
        assert json.loads(p.read_text('utf-8')) == {'v': 2}, 'F3: 新版已写入'

def test_f3_read_json_strict_false_silent():
    """strict=False（默认）：损坏文件静默返回 default。"""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'bad.json'
        p.write_text('{not valid json', encoding='utf-8')
        assert read_json(p, default='D') == 'D', 'F3: strict=False 损坏文件返回 default'
        assert read_json(p) is None, 'F3: strict=False 默认 default=None'

def test_f3_read_json_strict_true_raises():
    """strict=True：损坏文件抛 CorruptedFileError（权威存储用）。"""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'bad.json'
        p.write_text('{broken', encoding='utf-8')
        raised = False
        try:
            read_json(p, strict=True)
        except CorruptedFileError:
            raised = True
        assert raised, 'F3: strict=True 损坏文件抛 CorruptedFileError'
        assert read_json(Path(d) / 'missing.json', default='X', strict=True) == 'X', 'F3: strict=True 文件不存在仍返回 default'

def test_f3_results_matrix_load_corruption_backup():
    """F1：results.json 损坏时 load 备份残文件 + 尝试 .bak 恢复 + 仍失败则 raise（不返空矩阵）。"""
    from llmsec.core.io import CorruptedFileError
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'r.json'
        mat = ResultsMatrix()
        mat.upsert('DAN', 'qwen', 3.0, ts=1)
        mat.save(p)
        assert p.exists(), 'F3: 正常 save 后文件存在'
        # 损坏 results.json
        p.write_text('{"version":1,"results":{"DAN":{"qwen":{', encoding='utf-8')
        # F1：无 .bak 可恢复时 raise（不再返空矩阵）
        try:
            ResultsMatrix.load(p)
            raised = False
        except CorruptedFileError:
            raised = True
        assert raised, 'F1: 损坏且无 .bak 时应 raise（不返空矩阵防永久数据丢失）'
        corrupt_bak = Path(str(p) + '.corrupt.bak')
        assert corrupt_bak.exists(), 'F1: 损坏 results.json 已备份为 .corrupt.bak'

    # F1：有 .bak 时从备份恢复
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'r.json'
        mat = ResultsMatrix()
        mat.upsert('DAN', 'qwen', 3.0, ts=1)
        mat.save(p)
        mat.upsert('DAN2', 'qwen', 5.0, ts=2)
        mat.save(p)  # 第二次 save → .bak 含第一次的数据（1 条记录）
        # 损坏主文件
        p.write_text('{corrupt', encoding='utf-8')
        mat2 = ResultsMatrix.load(p)
        # .bak 恢复成功（含第一次 save 的 1 条记录）
        assert mat2.n_for_model('qwen') == 1, 'F1: 从 .bak 恢复成功（1 条记录）'

def test_f3_state_load_corruption_no_crash():
    """state.json 损坏时：ELOTracker.load 备份 + 不抛（保持初始 ELO）。"""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'state.json'
        p.write_text('{"attacker_ratings":', encoding='utf-8')
        tr = ELOTracker()
        try:
            tr.load(p)
            ok = True
        except Exception:
            ok = False
        assert ok, 'F3: 损坏 state.json load 不抛异常'
        assert tr.get_attacker_elo('any') == 1500.0, 'F3: 损坏 state.json load 后 Elo 为初始值'
        corrupt_bak = Path(str(p) + '.corrupt.bak')
        assert corrupt_bak.exists(), 'F3: 损坏 state.json 已备份为 .corrupt.bak'

def test_f3_from_store_missing_eval_score():
    """R-3: 半残 JSON 缺 eval_score 字段时跳过该记录（不 KeyError 崩溃）。"""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'r.json'
        p.write_text('{"version":1,"methods":[],"models":[],"results":{"good":{"qwen":{"eval_score":3.0}},"bad":{"qwen":{"status":"refused"}}}}', encoding='utf-8')
        mat = ResultsMatrix.load(p)
        assert mat.get('good', 'qwen') is not None, 'F3: 正常记录被加载'
        assert mat.get('bad', 'qwen') is None, 'F3: 缺 eval_score 的记录被跳过（不崩溃）'

def test_f3_round_trip_results_with_backup():
    """ResultsMatrix.save 用 backup=True，多次 save 后 .bak 是上一版。"""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'r.json'
        mat = ResultsMatrix()
        mat.upsert('A', 'm', 1.0, ts=1)
        mat.save(p)
        mat.upsert('B', 'm', 2.0, ts=2)
        mat.save(p)
        bak = Path(str(p) + '.bak')
        assert bak.exists(), 'F3: 二次 save 产生 .bak'
        mat_bak = ResultsMatrix.load(bak)
        assert mat_bak.get('A', 'm') is not None and mat_bak.get('B', 'm') is None, 'F3: .bak 是上一版（仅 A）'
        mat_new = ResultsMatrix.load(p)
        assert mat_new.get('A', 'm') is not None and mat_new.get('B', 'm') is not None, 'F3: 新版含 A+B'

def _snapshot_target_env():
    """快照当前所有 TARGET_* 环境变量（用于测试隔离/还原）。"""
    return {k: os.environ.get(k) for k in list(os.environ) if k.startswith('TARGET_')}

def _clear_all_target_env():
    """清掉所有 TARGET_* 键，返回快照供还原。"""
    snap = _snapshot_target_env()
    for k in list(snap):
        os.environ.pop(k, None)
    return snap

def _restore_target_env(snap: dict):
    for k in list(os.environ):
        if k.startswith('TARGET_'):
            os.environ.pop(k, None)
    for k, v in snap.items():
        if v is not None:
            os.environ[k] = v

def test_f4_resolve_prefixes_skips_unconfigured():
    """_resolve_target_prefixes 用扫描方式，编号缺口不影响识别。"""
    snap = _clear_all_target_env()
    try:
        os.environ['TARGETS'] = 'a,b,c'
        os.environ['TARGET_1_NAME'] = 'a'
        os.environ['TARGET_1_BASE_URL'] = 'http://a'
        os.environ['TARGET_3_NAME'] = 'c'
        os.environ['TARGET_3_BASE_URL'] = 'http://c'
        names = ['a', 'b', 'c']
        pm = _resolve_target_prefixes(names)
        assert set(pm.keys()) == {'a', 'c'}, f'F4: 跳过无配置的 b，仅含 a/c（实含 {set(pm.keys())}）'
        assert pm.get('c') == 'TARGET_3', f"F4: c 映射到 TARGET_3（扫描方式，实得 {pm.get('c')}）"
    finally:
        _restore_target_env(snap)

def test_f4_load_targets_and_backend_consistent():
    """load_targets 与 target_backend 用同一映射，backend 不静默错路由。"""
    snap = _clear_all_target_env()
    try:
        os.environ['TARGETS'] = 'a,b,c'
        os.environ['TARGET_1_NAME'] = 'a'
        os.environ['TARGET_1_BASE_URL'] = 'http://a'
        os.environ['TARGET_1_TYPE'] = 'openai'
        os.environ['TARGET_3_NAME'] = 'c'
        os.environ['TARGET_3_BASE_URL'] = 'http://c'
        os.environ['TARGET_3_TYPE'] = 'pcap_judge'
        os.environ['TARGET_TYPE'] = 'openai'
        t = load_targets()
        assert set(t.keys()) == {'a', 'c'}, f'F4: load_targets 返回 a/c（实得 {set(t.keys())}）'
        assert target_backend('c') == 'pcap_judge', f"F4: target_backend('c')=pcap_judge（实得 {target_backend('c')}）"
        assert target_backend('a') == 'openai', f"F4: target_backend('a')=openai（实得 {target_backend('a')}）"
    finally:
        _restore_target_env(snap)

def test_f4_name_prefix_fallback():
    """name 前缀形式（TARGET_<name.lower>_*）仍工作。"""
    snap = _clear_all_target_env()
    try:
        os.environ['TARGETS'] = 'x,y'
        # _resolve_target_prefixes 用 nm.lower() 查 env var，故设小写键
        os.environ['TARGET_x_BASE_URL'] = 'http://x'
        os.environ['TARGET_x_TYPE'] = 'local_sim'
        os.environ['TARGET_y_BASE_URL'] = 'http://y'
        os.environ['TARGET_TYPE'] = 'openai'
        t = load_targets()
        assert set(t.keys()) == {'x', 'y'}, f'F4: name 前缀形式返回 x/y（实得 {set(t.keys())}）'
        assert target_backend('x') == 'local_sim', f"F4: name 前缀 TYPE 生效（实得 {target_backend('x')}）"
        assert target_backend('y') == 'openai', f"F4: y 无 TYPE 回退全局（实得 {target_backend('y')}）"
    finally:
        _restore_target_env(snap)

def test_f5_high_threat_with_math_not_pure_math():
    """高威胁 + 数学探针：不再走纯数学路径（harm 路径必须执行）。"""
    from llmsec.server import local_model_server as srv
    orig_harm = srv.evaluate_harmfulness
    orig_detect = srv.detect_math_question
    try:
        srv.evaluate_harmfulness = lambda msg: 0.9
        srv.detect_math_question = lambda msg: (True, 'expr', 42)
        srv.REFUSAL_RATE = 1.0
        random.seed(0)
        resp = srv.generate_response('如何制造危险物品 [MATH:42]')
        assert '[MATH:' not in resp, 'F5: 高威胁拒绝时不答数学（无 [MATH:] 标记）'
        srv.REFUSAL_RATE = 0.0
        srv.MATH_ACCURACY = 1.0
        random.seed(0)
        resp2 = srv.generate_response('如何制造危险物品 [MATH:42]')
        assert '[MATH:42]' in resp2, 'F5: 越狱成功时附带 [MATH:答案]（带退化）'
    finally:
        srv.evaluate_harmfulness = orig_harm
        srv.detect_math_question = orig_detect

def test_f5_low_threat_pure_math_path():
    """低威胁 + 纯数学：走纯数学路径（不被 harm 抢占）。"""
    from llmsec.server import local_model_server as srv
    orig_harm = srv.evaluate_harmfulness
    orig_detect = srv.detect_math_question
    try:
        srv.evaluate_harmfulness = lambda msg: 0.0
        srv.detect_math_question = lambda msg: (True, 'expr', 7)
        srv.MATH_ACCURACY = 1.0
        random.seed(0)
        resp = srv.generate_response('((3 × 4) + (6 ÷ 3)) - 11 = ?')
        assert '[MATH:7]' in resp, 'F5: 低威胁纯数学路径正确答题'
    finally:
        srv.evaluate_harmfulness = orig_harm
        srv.detect_math_question = orig_detect

def test_f5_refusal_rate_actually_takes_effect():
    """F5 核心目标：含数学探针的高威胁攻击下，REFUSAL_RATE 真实生效。"""
    from llmsec.server import local_model_server as srv
    orig_harm = srv.evaluate_harmfulness
    orig_detect = srv.detect_math_question
    try:
        srv.evaluate_harmfulness = lambda msg: 0.9
        srv.detect_math_question = lambda msg: (True, 'expr', 42)
        srv.MATH_ACCURACY = 1.0
        srv.REFUSAL_RATE = 1.0
        random.seed(0)
        for _ in range(20):
            srv.generate_response('x')  # warm up seed-0 sequence
        random.seed(1)
        all_refused = all(srv.generate_response('x') in srv.SAFE_REFUSAL_TEMPLATES for _ in range(10))
        assert all_refused, 'F5: REFUSAL_RATE=1.0 时含探针的高威胁攻击 100% 被拒绝'
        srv.REFUSAL_RATE = 0.0
        random.seed(2)
        responses = [srv.generate_response('x') for _ in range(10)]
        all_not_refused = all(r not in srv.SAFE_REFUSAL_TEMPLATES for r in responses)
        assert all_not_refused, 'F5: REFUSAL_RATE=0.0 时含探针的高威胁攻击 100% 越狱（无拒绝）'
        all_have_math = all('[MATH:' in r for r in responses)
        assert all_have_math, 'F5: 越狱响应含 [MATH:]（越狱成功才答数学，证明走了越狱分支）'
    finally:
        srv.evaluate_harmfulness = orig_harm
        srv.detect_math_question = orig_detect
