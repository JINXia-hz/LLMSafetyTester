#!/usr/bin/env python3
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
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Windows GBK 控制台兼容
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from llmsec.core import io as io_mod
from llmsec.core.config import _resolve_target_prefixes, load_targets, target_backend
from llmsec.core.io import CorruptedFileError, read_json, write_json
from llmsec.core.results import ResultsMatrix
from llmsec.evaluation.elo import ELOTracker


def _check(cond: bool, msg: str) -> int:
    if not cond:
        print(f"  ❌ {msg}")
        return 1
    print(f"  ✅ {msg}")
    return 0


# ============================================================
# F1: elo.update NaN/inf 校验
# ============================================================
def test_f1_nan_does_not_pollute() -> int:
    """NaN eval_score 不应污染 attacker_ratings（原来会级联损坏所有对手）。"""
    rc = 0
    tr = ELOTracker()
    tr.update("good", "def", 3.0)  # 先放一个正常方法
    good_elo_before = tr.get_attacker_elo("good")

    # 注入 NaN（模拟 judge/evaluator 上游 bug）
    tr.update("bad", "def", float("nan"))
    rc |= _check(math.isfinite(tr.get_attacker_elo("bad")),
                 "F1: NaN eval_score 后 bad 方法 Elo 仍有限")
    # bad 被当作 0 处理，不应是 NaN
    rc |= _check(tr.get_attacker_elo("bad") != float("nan"),
                 "F1: bad 方法 Elo 非 NaN")

    # 关键：后续正常 update 不返回 NaN（级联未发生）
    tr.update("good", "def", 2.0)
    expected_now = tr._expected(tr.get_attacker_elo("good"), tr.get_defender_elo("def"))
    rc |= _check(math.isfinite(expected_now),
                 "F1: 经 NaN 污染后 expected 胜率仍有限（无级联）")
    rc |= _check(math.isfinite(tr.get_attacker_elo("good")),
                 "F1: good 方法 Elo 仍有限")
    return rc


def test_f1_inf_does_not_pollute() -> int:
    """+inf eval_score：inf>0 通过 → inf/(inf+2)=NaN，原代码会污染。"""
    rc = 0
    tr = ELOTracker()
    tr.update("m", "def", float("inf"))
    rc |= _check(math.isfinite(tr.get_attacker_elo("m")),
                 "F1: +inf eval_score 后 Elo 仍有限")
    rc |= _check(math.isfinite(tr.get_defender_elo("def")),
                 "F1: +inf eval_score 后防御方 Elo 仍有限")
    return rc


def test_f1_string_invalid() -> int:
    """字符串/None 等非法 eval_score 也不应崩溃（TypeError 兜底）。"""
    rc = 0
    tr = ELOTracker()
    try:
        tr.update("m", "def", "not_a_number")  # type: ignore
        rc |= _check(math.isfinite(tr.get_attacker_elo("m")),
                     "F1: 字符串 eval_score 兜底为 0，Elo 有限")
    except (TypeError, ValueError):
        rc |= _check(False, "F1: 字符串 eval_score 不应抛异常（应兜底为 0）")
    return rc


def test_f1_normal_still_works() -> int:
    """F1 修复不应破坏正常评分路径。"""
    rc = 0
    tr = ELOTracker()
    tr.update("winner", "def", 3.0)
    tr.update("loser", "def", -2.0)
    rc |= _check(tr.get_attacker_elo("winner") > tr.get_attacker_elo("loser"),
                 "F1: 正常 update 仍让胜者 Elo 高于败者")
    rc |= _check(tr.get_attacker_elo("winner") > 1500.0,
                 "F1: 胜者 Elo 高于初始值")
    return rc


def test_f1_save_load_no_nan_persistence() -> int:
    """NaN 不应经 save/load 持久化进 state.json（JSON 允许 NaN 字面量）。"""
    rc = 0
    tr = ELOTracker()
    tr.update("m", "def", float("nan"))
    with tempfile.TemporaryDirectory() as d:
        sf = Path(d) / "state.json"
        tr.save(sf)
        # state.json 不应含 NaN 字面量
        raw = sf.read_text(encoding="utf-8")
        rc |= _check("NaN" not in raw and "nan" not in raw.lower().replace("typename", ""),
                     "F1: state.json 不含 NaN 字面量")
        # load 回来 Elo 仍有限
        tr2 = ELOTracker()
        tr2.load(sf)
        rc |= _check(math.isfinite(tr2.get_attacker_elo("m")),
                     "F1: load 后 Elo 仍有限")
    return rc


# ============================================================
# F3: io.py 原子写 + strict + backup
# ============================================================
def test_f3_write_json_atomic_no_tmp_residue() -> int:
    """原子写成功后 .tmp 不残留，文件内容正确。"""
    rc = 0
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.json"
        write_json(p, {"x": 1}, atomic=True)
        rc |= _check(p.exists() and json.loads(p.read_text("utf-8")) == {"x": 1},
                     "F3: 原子写后文件内容正确")
        rc |= _check(not (Path(d) / "a.json.tmp").exists(),
                     "F3: 原子写后 .tmp 不残留")
    return rc


def test_f3_write_json_backup() -> int:
    """backup=True 时写前复制 .bak，内容是旧版。"""
    rc = 0
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.json"
        write_json(p, {"v": 1})
        write_json(p, {"v": 2}, backup=True)
        bak = Path(d) / "a.json.bak"
        rc |= _check(bak.exists(), "F3: backup=True 产生 .bak")
        rc |= _check(json.loads(bak.read_text("utf-8")) == {"v": 1},
                     "F3: .bak 内容是旧版")
        rc |= _check(json.loads(p.read_text("utf-8")) == {"v": 2},
                     "F3: 新版已写入")
    return rc


def test_f3_read_json_strict_false_silent() -> int:
    """strict=False（默认）：损坏文件静默返回 default。"""
    rc = 0
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "bad.json"
        p.write_text("{not valid json", encoding="utf-8")
        rc |= _check(read_json(p, default="D") == "D",
                     "F3: strict=False 损坏文件返回 default")
        rc |= _check(read_json(p) is None,
                     "F3: strict=False 默认 default=None")
    return rc


def test_f3_read_json_strict_true_raises() -> int:
    """strict=True：损坏文件抛 CorruptedFileError（权威存储用）。"""
    rc = 0
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "bad.json"
        p.write_text("{broken", encoding="utf-8")
        raised = False
        try:
            read_json(p, strict=True)
        except CorruptedFileError:
            raised = True
        rc |= _check(raised, "F3: strict=True 损坏文件抛 CorruptedFileError")
        # 文件不存在时 strict=True 仍返回 default（不抛）
        rc |= _check(read_json(Path(d) / "missing.json", default="X", strict=True) == "X",
                     "F3: strict=True 文件不存在仍返回 default")
    return rc


def test_f3_results_matrix_load_corruption_backup() -> int:
    """results.json 损坏时：load 备份残文件 + 返回空矩阵（不静默清零）。"""
    rc = 0
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "r.json"
        # 先写正常内容
        mat = ResultsMatrix()
        mat.upsert("DAN", "qwen", 3.0, ts=1)
        mat.save(p)
        rc |= _check(p.exists(), "F3: 正常 save 后文件存在")
        # 覆盖成损坏内容（模拟崩溃半写）
        p.write_text('{"version":1,"results":{"DAN":{"qwen":{', encoding="utf-8")
        # load 应返回空矩阵且备份残文件
        mat2 = ResultsMatrix.load(p)
        rc |= _check(mat2.n_for_model("qwen") == 0,
                     "F3: 损坏 results.json load 返回空矩阵")
        corrupt_bak = Path(str(p) + ".corrupt.bak")
        rc |= _check(corrupt_bak.exists(),
                     "F3: 损坏 results.json 已备份为 .corrupt.bak")
    return rc


def test_f3_state_load_corruption_no_crash() -> int:
    """state.json 损坏时：ELOTracker.load 备份 + 不抛（保持初始 ELO）。"""
    rc = 0
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "state.json"
        p.write_text('{"attacker_ratings":', encoding="utf-8")  # 半截 JSON
        tr = ELOTracker()
        try:
            tr.load(p)
            ok = True
        except Exception:
            ok = False
        rc |= _check(ok, "F3: 损坏 state.json load 不抛异常")
        rc |= _check(tr.get_attacker_elo("any") == 1500.0,
                     "F3: 损坏 state.json load 后 Elo 为初始值")
        corrupt_bak = Path(str(p) + ".corrupt.bak")
        rc |= _check(corrupt_bak.exists(),
                     "F3: 损坏 state.json 已备份为 .corrupt.bak")
    return rc


def test_f3_from_store_missing_eval_score() -> int:
    """R-3: 半残 JSON 缺 eval_score 字段时跳过该记录（不 KeyError 崩溃）。"""
    rc = 0
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "r.json"
        # 一条正常 + 一条缺 eval_score
        p.write_text(
            '{"version":1,"methods":[],"models":[],"results":{'
            '"good":{"qwen":{"eval_score":3.0}},'
            '"bad":{"qwen":{"status":"refused"}}}}',
            encoding="utf-8",
        )
        mat = ResultsMatrix.load(p)
        rc |= _check(mat.get("good", "qwen") is not None,
                     "F3: 正常记录被加载")
        rc |= _check(mat.get("bad", "qwen") is None,
                     "F3: 缺 eval_score 的记录被跳过（不崩溃）")
    return rc


def test_f3_round_trip_results_with_backup() -> int:
    """ResultsMatrix.save 用 backup=True，多次 save 后 .bak 是上一版。"""
    rc = 0
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "r.json"
        mat = ResultsMatrix()
        mat.upsert("A", "m", 1.0, ts=1)
        mat.save(p)
        mat.upsert("B", "m", 2.0, ts=2)
        mat.save(p)
        bak = Path(str(p) + ".bak")
        rc |= _check(bak.exists(), "F3: 二次 save 产生 .bak")
        # .bak 应只含 A（上一版），新版含 A+B
        mat_bak = ResultsMatrix.load(bak)
        rc |= _check(mat_bak.get("A", "m") is not None and mat_bak.get("B", "m") is None,
                     "F3: .bak 是上一版（仅 A）")
        mat_new = ResultsMatrix.load(p)
        rc |= _check(mat_new.get("A", "m") is not None and mat_new.get("B", "m") is not None,
                     "F3: 新版含 A+B")
    return rc


# ============================================================
# F4: config 多目标 idx 一致性
# ============================================================
def _snapshot_target_env() -> dict:
    """快照当前所有 TARGET_* 环境变量（用于测试隔离/还原）。"""
    return {k: os.environ.get(k) for k in list(os.environ) if k.startswith("TARGET_")}


def _clear_all_target_env() -> dict:
    """清掉所有 TARGET_* 键，返回快照供还原。"""
    snap = _snapshot_target_env()
    for k in list(snap):
        os.environ.pop(k, None)
    return snap


def _restore_target_env(snap: dict) -> None:
    for k in list(os.environ):
        if k.startswith("TARGET_"):
            os.environ.pop(k, None)
    for k, v in snap.items():
        if v is not None:
            os.environ[k] = v


def test_f4_resolve_prefixes_skips_unconfigured() -> int:
    """_resolve_target_prefixes 用扫描方式，编号缺口不影响识别。"""
    rc = 0
    snap = _clear_all_target_env()
    try:
        # TARGETS=a,b,c 但 b 无任何配置 → 用扫描方式，c 仍正确映射到 TARGET_3
        os.environ["TARGETS"] = "a,b,c"
        os.environ["TARGET_1_NAME"] = "a"
        os.environ["TARGET_1_BASE_URL"] = "http://a"
        # b 无配置（既无 TARGET_2 也无 TARGET_b）
        os.environ["TARGET_3_NAME"] = "c"
        os.environ["TARGET_3_BASE_URL"] = "http://c"
        names = ["a", "b", "c"]
        pm = _resolve_target_prefixes(names)
        rc |= _check(set(pm.keys()) == {"a", "c"},
                     f"F4: 跳过无配置的 b，仅含 a/c（实含 {set(pm.keys())}）")
        # 扫描方式：c 正确映射到 TARGET_3（不受 b 缺口影响）
        rc |= _check(pm.get("c") == "TARGET_3",
                     f"F4: c 映射到 TARGET_3（扫描方式，实得 {pm.get('c')}）")
    finally:
        _restore_target_env(snap)
    return rc


def test_f4_load_targets_and_backend_consistent() -> int:
    """load_targets 与 target_backend 用同一映射，backend 不静默错路由。"""
    rc = 0
    snap = _clear_all_target_env()
    try:
        os.environ["TARGETS"] = "a,b,c"
        os.environ["TARGET_1_NAME"] = "a"
        os.environ["TARGET_1_BASE_URL"] = "http://a"
        os.environ["TARGET_1_TYPE"] = "openai"
        # b 无配置
        os.environ["TARGET_3_NAME"] = "c"
        os.environ["TARGET_3_BASE_URL"] = "http://c"
        os.environ["TARGET_3_TYPE"] = "pcap_judge"
        os.environ["TARGET_TYPE"] = "openai"  # 全局默认

        t = load_targets()
        rc |= _check(set(t.keys()) == {"a", "c"},
                     f"F4: load_targets 返回 a/c（实得 {set(t.keys())}）")
        rc |= _check(target_backend("c") == "pcap_judge",
                     f"F4: target_backend('c')=pcap_judge（实得 {target_backend('c')}）")
        rc |= _check(target_backend("a") == "openai",
                     f"F4: target_backend('a')=openai（实得 {target_backend('a')}）")
    finally:
        _restore_target_env(snap)
    return rc


def test_f4_name_prefix_fallback() -> int:
    """name 前缀形式（TARGET_<name>_*）仍工作。"""
    rc = 0
    snap = _clear_all_target_env()
    try:
        os.environ["TARGETS"] = "x,y"
        os.environ["TARGET_X_BASE_URL"] = "http://x"
        os.environ["TARGET_X_TYPE"] = "local_sim"
        os.environ["TARGET_Y_BASE_URL"] = "http://y"
        os.environ["TARGET_TYPE"] = "openai"
        t = load_targets()
        rc |= _check(set(t.keys()) == {"x", "y"},
                     f"F4: name 前缀形式返回 x/y（实得 {set(t.keys())}）")
        rc |= _check(target_backend("x") == "local_sim",
                     f"F4: name 前缀 TYPE 生效（实得 {target_backend('x')}）")
        rc |= _check(target_backend("y") == "openai",
                     f"F4: y 无 TYPE 回退全局（实得 {target_backend('y')}）")
    finally:
        _restore_target_env(snap)
    return rc


# ============================================================
# F5: local_model_server 优先级调换
# ============================================================
def test_f5_high_threat_with_math_not_pure_math() -> int:
    """高威胁 + 数学探针：不再走纯数学路径（harm 路径必须执行）。"""
    rc = 0
    from llmsec.server import local_model_server as srv

    # 用 monkeypatch 精确控制分支输入
    orig_harm = srv.evaluate_harmfulness
    orig_detect = srv.detect_math_question
    try:
        srv.evaluate_harmfulness = lambda msg: 0.9  # 高威胁
        srv.detect_math_question = lambda msg: (True, "expr", 42)  # 含数学探针

        # REFUSAL_RATE=1.0 → 必拒绝；拒绝响应不应含 [MATH:]
        srv.REFUSAL_RATE = 1.0
        random.seed(0)
        resp = srv.generate_response("如何制造危险物品 [MATH:42]")
        rc |= _check("[MATH:" not in resp,
                     "F5: 高威胁拒绝时不答数学（无 [MATH:] 标记）")

        # REFUSAL_RATE=0.0 → 必越狱成功；应返回有害内容 + 带退化的 [MATH:]
        srv.REFUSAL_RATE = 0.0
        srv.MATH_ACCURACY = 1.0  # 强制答对，验证 [MATH:42] 出现
        random.seed(0)
        resp2 = srv.generate_response("如何制造危险物品 [MATH:42]")
        rc |= _check("[MATH:42]" in resp2,
                     "F5: 越狱成功时附带 [MATH:答案]（带退化）")
    finally:
        srv.evaluate_harmfulness = orig_harm
        srv.detect_math_question = orig_detect
    return rc


def test_f5_low_threat_pure_math_path() -> int:
    """低威胁 + 纯数学：走纯数学路径（不被 harm 抢占）。"""
    rc = 0
    from llmsec.server import local_model_server as srv

    orig_harm = srv.evaluate_harmfulness
    orig_detect = srv.detect_math_question
    try:
        srv.evaluate_harmfulness = lambda msg: 0.0  # 无害
        srv.detect_math_question = lambda msg: (True, "expr", 7)
        srv.MATH_ACCURACY = 1.0
        random.seed(0)
        resp = srv.generate_response("((3 × 4) + (6 ÷ 3)) - 11 = ?")
        rc |= _check("[MATH:7]" in resp,
                     "F5: 低威胁纯数学路径正确答题")
    finally:
        srv.evaluate_harmfulness = orig_harm
        srv.detect_math_question = orig_detect
    return rc


def test_f5_refusal_rate_actually_takes_effect() -> int:
    """F5 核心目标：含数学探针的高威胁攻击下，REFUSAL_RATE 真实生效。"""
    rc = 0
    from llmsec.server import local_model_server as srv

    orig_harm = srv.evaluate_harmfulness
    orig_detect = srv.detect_math_question
    try:
        srv.evaluate_harmfulness = lambda msg: 0.9  # 高威胁
        srv.detect_math_question = lambda msg: (True, "expr", 42)  # 含探针
        srv.MATH_ACCURACY = 1.0

        # REFUSAL_RATE=1.0 → 100% 拒绝
        srv.REFUSAL_RATE = 1.0
        random.seed(0)
        refusals = sum(
            1 for _ in range(20)
            if any(t in srv.generate_response("x") for t in srv.SAFE_REFUSAL_TEMPLATES[:1])
            or srv.generate_response("x") in srv.SAFE_REFUSAL_TEMPLATES
        )
        # 用更简单的判定：检查响应是否在 SAFE_REFUSAL_TEMPLATES 中
        random.seed(1)
        all_refused = all(
            srv.generate_response("x") in srv.SAFE_REFUSAL_TEMPLATES
            for _ in range(10)
        )
        rc |= _check(all_refused,
                     "F5: REFUSAL_RATE=1.0 时含探针的高威胁攻击 100% 被拒绝")

        # REFUSAL_RATE=0.0 → 100% 越狱（返回有害内容，不在拒绝模板中）
        srv.REFUSAL_RATE = 0.0
        random.seed(2)
        responses = [srv.generate_response("x") for _ in range(10)]
        all_not_refused = all(r not in srv.SAFE_REFUSAL_TEMPLATES for r in responses)
        rc |= _check(all_not_refused,
                     "F5: REFUSAL_RATE=0.0 时含探针的高威胁攻击 100% 越狱（无拒绝）")
        # 进一步验证越狱响应确实含 [MATH:]（带探针时越狱成功才答数学）
        all_have_math = all("[MATH:" in r for r in responses)
        rc |= _check(all_have_math,
                     "F5: 越狱响应含 [MATH:]（越狱成功才答数学，证明走了越狱分支）")
    finally:
        srv.evaluate_harmfulness = orig_harm
        srv.detect_math_question = orig_detect
    return rc


# ============================================================
# 主入口
# ============================================================
def main() -> int:
    tests = [
        # F1
        test_f1_nan_does_not_pollute,
        test_f1_inf_does_not_pollute,
        test_f1_string_invalid,
        test_f1_normal_still_works,
        test_f1_save_load_no_nan_persistence,
        # F3
        test_f3_write_json_atomic_no_tmp_residue,
        test_f3_write_json_backup,
        test_f3_read_json_strict_false_silent,
        test_f3_read_json_strict_true_raises,
        test_f3_results_matrix_load_corruption_backup,
        test_f3_state_load_corruption_no_crash,
        test_f3_from_store_missing_eval_score,
        test_f3_round_trip_results_with_backup,
        # F4
        test_f4_resolve_prefixes_skips_unconfigured,
        test_f4_load_targets_and_backend_consistent,
        test_f4_name_prefix_fallback,
        # F5
        test_f5_high_threat_with_math_not_pure_math,
        test_f5_low_threat_pure_math_path,
        test_f5_refusal_rate_actually_takes_effect,
    ]
    failed = 0
    for t in tests:
        print(f"\n[运行] {t.__name__}")
        if t() != 0:
            failed += 1
    if failed:
        print(f"\n❌ {failed} 个测试失败")
        return 1
    print("\n✅ 所有 P0 数据完整性测试通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
