"""Combined tests: ELO 评分系统（收敛判据 + 边界健壮性 + R-cutover 派生访问层）。"""


# ===== from test_elo_convergence.py =====

"""

回归测试：Elo 收敛判定与固定簇预测。


验证：

1. predict 对同后缀变体优先使用同后缀 ground truth。

2. predict 对同基底变体使用变体 ground truth（同后缀不存在时回退）。

3. check_convergence 在 Elo 噪声大（95%CI 半宽超目标）时不判收敛（抗假阳性）。

4. check_convergence 在噪声小、漂移小、覆盖率足够时判收敛。

"""


from llmsec.clustering.features import _extract_variant_suffix, _strip_variant_suffix
from llmsec.evaluation.elo import ELOTracker
from llmsec.evaluation.predictors.cold_start import ColdStartPredictor


def test_strip_variant_suffix():

    """测试变体后缀剥离。"""

    cases = [

        ("method_rot13", "method"),

        ("method_b64", "method"),

        ("method_base64", "method"),

        ("method_code", "method"),

        ("method_story", "method"),

        ("method_0", "method"),

        ("method", "method"),

    ]

    for raw, expected in cases:

        got = _strip_variant_suffix(raw)

        assert not (got != expected), f"❌ _strip_variant_suffix({raw!r}) = {got!r}, expected {expected!r}"


def test_extract_variant_suffix():

    """测试变体后缀提取。"""

    cases = [

        ("method_rot13", "rot13"),

        ("method_b64", "b64"),

        ("method_base64", "b64"),

        ("method_code", "code"),

        ("method_story", "story"),

        ("method_0", "0"),

        ("method", ""),

    ]

    for raw, expected in cases:

        got = _extract_variant_suffix(raw)

        assert not (got != expected), f"❌ _extract_variant_suffix({raw!r}) = {got!r}, expected {expected!r}"


def test_predict_suffix_variant_fallback():

    """predict 应优先用同后缀变体的 ground truth。"""

    predictor = ColdStartPredictor()

    predictor.ground_truth = {

        "attack_a_rot13": {"elo": 1800.0},

        "attack_b_rot13": {"elo": 1700.0},

        "attack_c_b64": {"elo": 1200.0},

        "other_method": {"elo": 1500.0},

    }

    predictor.artifacts = {

        "labels": {

            "attack_a_rot13": 0,

            "attack_b_rot13": 0,

            "attack_c_b64": 0,

            "attack_d_rot13": 0,

            "attack_e_code": 0,

            "other_method": 1,

        },

        "features": {},

        "weights": (0.15, 0.45, 0.25, 0.0),

    }


    # 预测同后缀的新变体 attack_d_rot13：应接近 rot13 平均 (1800+1700)/2 = 1750

    pred = predictor.predict("attack_d_rot13")

    assert not (pred["source"] != "predicted_suffix_variant"), f"❌ predict 未优先使用同后缀变体兜底: source={pred['source']}"

    expected = (1800.0 + 1700.0) / 2

    assert not (abs(pred["elo"] - expected) > 1e-6), f"❌ predict 同后缀预测错误: elo={pred['elo']}, expected={expected}"


    # 预测同基底但不同后缀的 attack_e_code：应回退到同基底变体（但同基底只有 attack_c_b64，后缀不同）

    # attack_e_code 与 attack_c_b64 不是同基底（基底是 attack_c vs attack_e），也不是同后缀

    # 所以应该使用簇内/全局平均

    pred2 = predictor.predict("attack_e_code")

    assert not (pred2["source"] == "predicted_suffix_variant"), f"❌ predict 错误地把无关方法识别为同后缀变体: source={pred2['source']}"


def test_predict_base_variant_fallback():

    """predict 在同后缀不存在时应回退到同基底变体。"""

    predictor = ColdStartPredictor()

    predictor.ground_truth = {

        "attack_rot13": {"elo": 1800.0},

        "attack_b64": {"elo": 1200.0},

        "other_method": {"elo": 1500.0},

    }

    predictor.artifacts = {

        "labels": {

            "attack_rot13": 0,

            "attack_b64": 0,

            "attack_code": 0,

            "other_method": 1,

        },

        "features": {},

        "weights": (0.15, 0.45, 0.25, 0.0),

    }


    # attack_code 没有同后缀 ground truth，但有同基底变体 attack_rot13 和 attack_b64

    pred = predictor.predict("attack_code")

    assert not (pred["source"] != "predicted_variant"), f"❌ predict 未回退到同基底变体: source={pred['source']}"

    expected = (1800.0 + 1200.0) / 2

    assert not (abs(pred["elo"] - expected) > 1e-6), f"❌ predict 同基底预测错误: elo={pred['elo']}, expected={expected}"


def test_convergence_resists_false_positive():

    """Elo 噪声大（真值 Elo 95%CI 半宽超目标）时不应判收敛。"""

    tracker = ELOTracker()

    defender = "test-model"


    # 模拟多轮防御方 Elo 大幅波动（去趋势后噪声大 → CI 半宽远超 ±20 目标）

    # B1：CONV_WINDOW_MIN=6，需 >= 6 轮才判收敛

    tracker._round_defender_elos[defender] = [1500.0, 1560.0, 1490.0, 1555.0, 1505.0, 1545.0, 1495.0, 1510.0]

    tracker.defender_ratings[defender] = 1510.0


    # 构造足够的方法数以满足覆盖率

    for i in range(50):

        tracker.attacker_ratings[f"method_{i}"] = 700.0

    for i in range(15):

        tracker.ground_truth_methods.add(f"method_{i}")


    conv = tracker.check_convergence(defender, total_methods=50)

    assert not (conv["converged"]), f"❌ 假收敛未被拦截: ci_half={conv['ci_half']}, drift={conv['drift']}, coverage={conv['coverage']}"


def test_convergence_true_positive():

    """噪声小、漂移小、覆盖率足够时应判收敛。"""

    tracker = ELOTracker()

    defender = "test-model"


    # 防御方 Elo 稳定在 ~1500（低噪声 + 低漂移 → CI 半宽 < ±20）

    # B1：CONV_WINDOW_MIN=6，需 >= 6 轮才判收敛

    tracker.defender_ratings[defender] = 1498.0

    tracker._round_defender_elos[defender] = [1495.0, 1502.0, 1498.0, 1501.0, 1499.0, 1500.0, 1502.0, 1498.0]


    # 总方法 50，已测 15 => 覆盖率 30%

    for i in range(50):

        tracker.attacker_ratings[f"method_{i}"] = 1500.0

    for i in range(15):

        tracker.ground_truth_methods.add(f"method_{i}")


    conv = tracker.check_convergence(defender, total_methods=50)

    assert conv["converged"], f"❌ 真收敛未通过: {conv}"


def test_boundary_split_tested_predicted():

    """compute_security_boundary 应按实测/预测拆分边界以上统计。"""

    tracker = ELOTracker()

    defender = "test-model"

    tracker.defender_ratings[defender] = 1500.0


    # 2 个实测方法（1 个在边界上）、2 个预测方法（1 个在边界上）

    tracker.attacker_ratings = {

        "tested_high": 1600.0,

        "tested_low": 1400.0,

        "pred_high": 1600.0,

        "pred_low": 1400.0,

    }

    tracker.predictor.ground_truth = {

        "tested_high": {"elo": 1600.0},

        "tested_low": {"elo": 1400.0},

    }

    tracker.ground_truth_methods = {"tested_high", "tested_low"}


    b = tracker.compute_security_boundary(defender)

    assert not (b.get("tested_above_boundary") != 1), f"❌ tested_above_boundary={b.get('tested_above_boundary')}, expected 1"

    assert not (b.get("predicted_above_boundary") != 1), f"❌ predicted_above_boundary={b.get('predicted_above_boundary')}, expected 1"

    assert not (b.get("methods_above_boundary") != 2), f"❌ methods_above_boundary={b.get('methods_above_boundary')}, expected 2"

    assert not (b["tested_above_boundary"] + b["predicted_above_boundary"] != b["methods_above_boundary"]), "❌ 拆分之和 != 总数"


# ===== from test_elo_edge_cases.py =====

import tempfile
from pathlib import Path

import joblib
import numpy as np

from llmsec.evaluation.predictors.svd_ridge import EloPredictorModel


def test_boundary_no_defender_keys():

    """S-2：无防御方场次时 compute_security_boundary 不崩且键完整。"""

    tracker = ELOTracker()

    b = tracker.compute_security_boundary("ghost-defender")

    required = {"boundary_elo", "defender_elo", "converged", "confidence",

                "methods_above_boundary", "tested_above_boundary", "predicted_above_boundary"}

    assert required.issubset(b.keys()), f"早退 dict 缺键: {required - set(b.keys())}"

    assert b["converged"] is False

    assert b["confidence"] == 0.0

    # 旧代码曾 KeyError 的两个键，现在可直接下标

    _ = b["converged"], b["defender_elo"]


def test_update_accepts_string_score():

    """M-3：update_round 对数字字符串 eval_score 回写 float，不抛 TypeError。"""

    tracker = ELOTracker()

    tracker.update_round("def", [("DAN", "3.5")])  # 字符串分数

    assert tracker.get_attacker_elo("DAN") > 1500

    # 非数字字符串 → 视为 0（不崩；score=0 仍登记）

    tracker.update_round("def", [("X", "not-a-number")])

    assert "X" in tracker.attacker_ratings


def test_sigma2_floor_on_constant_gt():

    """M-7：GT Elo 全相同时 σ² 有下限（不产 std=0 的绝对确定预测）。"""

    blocks = ("textual", "embedding", "technique", "intent", "prior")

    features = {

        f"m{i}": {b: np.array([float(i) + k * 0.1], dtype=float) for k, b in enumerate(blocks)}

        for i in range(6)

    }

    gt = {f"m{i}": {"elo": 1500.0} for i in range(6)}  # 全相同 Elo → 残差 0

    m = EloPredictorModel()

    m.fit(features, gt)

    assert m.sigma2 >= 1e-6, f"σ² 应有下限 ≥1e-6（得 {m.sigma2}）"


def test_load_artifacts_prefers_cluster_result():
    """M-4：两文件并存时优先 cluster_result.pkl（含 labels），重启后标签不丢。"""
    import llmsec.core.config as cfg

    orig_cr = cfg.CLUSTER_RESULT_FILE

    orig_fc = cfg.FEATURE_CACHE_FILE

    with tempfile.TemporaryDirectory() as d:

        tmp = Path(d)

        cfg.CLUSTER_RESULT_FILE = tmp / "cluster_result.pkl"

        cfg.FEATURE_CACHE_FILE = tmp / "feature_cache.pkl"

        # feature_cache：仅 features（无 labels）

        joblib.dump({"features": {"m1": {"t": [0.0]}, "m2": {"t": [1.0]}}}, cfg.FEATURE_CACHE_FILE)

        # cluster_result：features + labels —— 应被优先加载

        joblib.dump(

            {"features": {"m1": {"t": [0.0]}, "m2": {"t": [1.0]}},

             "labels": {"m1": 0, "m2": 1}, "kind": "cluster_result"},

            cfg.CLUSTER_RESULT_FILE,

        )

        try:

            pred = ColdStartPredictor()

            pred._load_artifacts()

            assert pred.artifacts is not None

            assert pred.artifacts.get("labels"), "应优先加载含 labels 的 cluster_result"

        finally:

            cfg.CLUSTER_RESULT_FILE = orig_cr

            cfg.FEATURE_CACHE_FILE = orig_fc


# ===== from test_elo_access.py =====

import llmsec.core.config as _results_cfg
from llmsec.core.results import ResultsMatrix
from llmsec.evaluation import elo_access as ea
from llmsec.evaluation.predictors.blend import load_or_fit_blend_predictor


def _setup(tmp_path, monkeypatch):

    monkeypatch.setattr(_results_cfg, "RESULTS_DB", tmp_path / "results.db")
    monkeypatch.setattr(_results_cfg, "RESULTS_FILE", tmp_path / "results.json")


def test_publish_and_derive(tmp_path, monkeypatch):

    _setup(tmp_path, monkeypatch)

    tracker = ELOTracker()

    tracker.update_round("qwen9b", [("DAN", 3.5)])

    tracker.update_round("qwen9b", [("rot13", -1.0)])

    tracker.record_round_end("qwen9b")

    ea.publish_tracker(tracker, "qwen9b")


    R = ResultsMatrix.load()

    assert R.n_for_model("qwen9b") == 2

    st = ea.elo_state_for("qwen9b")

    assert "DAN" in st["attacker_ratings"]

    assert st["fingerprint"] is not None

    assert st.get("round_defender_elos", {}).get("qwen9b")


def test_cache_invalidates_on_R_change(tmp_path, monkeypatch):

    _setup(tmp_path, monkeypatch)

    t = ELOTracker()

    t.update_round("qwen9b", [("DAN", 3.5)])

    ea.publish_tracker(t, "qwen9b")

    fp1 = ea.elo_state_for("qwen9b")["fingerprint"]


    t2 = ELOTracker()

    t2.update_round("qwen9b", [("rot13", 2.0)])

    ea.publish_tracker(t2, "qwen9b")

    st2 = ea.elo_state_for("qwen9b")

    assert st2["fingerprint"] != fp1

    assert "rot13" in st2["attacker_ratings"]


def test_active_model_and_empty(tmp_path, monkeypatch):

    _setup(tmp_path, monkeypatch)

    assert ea.active_model() is None

    t = ELOTracker()

    t.update_round("modelA", [("DAN", 1.0)])

    ea.publish_tracker(t, "modelA")

    assert ea.active_model() == "modelA"

    assert ea.attacker_ratings_for("modelA").get("DAN") is not None

    assert ea.elo_state_for("nope") == {}


def test_blend_predictor_cache_reuse(tmp_path, monkeypatch):

    _setup(tmp_path, monkeypatch)


    import llmsec.core.config as _bp_cfg
    monkeypatch.setattr(_bp_cfg, "PREDICTORS_DIR", tmp_path / "predictors")

    R = ResultsMatrix()

    R.upsert("DAN", "qwen9b", 3.0, ts=1)

    R.upsert("rot13", "qwen9b", -1.0, ts=2)

    R.upsert("b64", "qwen9b", 2.0, ts=3)

    features = {"DAN": {"t": [1.0]}, "rot13": {"t": [0.0]}, "b64": {"t": [0.5]}}

    catalog = ["DAN", "rot13", "b64"]


    load_or_fit_blend_predictor(R, features, method_catalog=catalog)

    load_or_fit_blend_predictor(R, features, method_catalog=catalog)  # 二次命中缓存

    assert any((tmp_path / "predictors").glob("blend_*.pkl"))


    R.upsert("extra", "qwen9b", 1.0, ts=4)  # R 变动 → 新缓存

    load_or_fit_blend_predictor(R, features, method_catalog=catalog)

    files = list((tmp_path / "predictors").glob("blend_*.pkl"))

    assert len(files) >= 2


# ===== from test_eval_review_elo.py（评审修复回归：B 组）=====

import pytest


def test_recent_success_rate_dedupes_retested_methods():
    """同一方法重复测试时，窗口按 distinct 方法计数，各取最近一次结果。"""
    tr = ELOTracker()
    # 时间序：m1 胜 → m2 胜 → m1 重测败（m1 最近一次为败）
    tr.update_round("def", [("m1", 3.0)])
    tr.update_round("def", [("m2", 3.0)])
    tr.update_round("def", [("m1", -1.0)])

    rate = tr._recent_success_rate(window_methods=15)
    # 去重口径：{m1: 最近=败, m2: 胜} → 1/2 = 0.5（旧口径数场次 = 2/3）
    assert rate == pytest.approx(0.5), f"应按方法去重取最近一次结果，得 {rate}"


def test_recent_success_rate_window_counts_distinct_methods():
    """window_methods 限的是最近 N 个 distinct 方法而非最近 N 场。"""
    tr = ELOTracker()
    for _ in range(5):  # m1 连测 5 场全败，之后 m2 一场胜
        tr.update_round("def", [("m1", -1.0)])
    tr.update_round("def", [("m2", 3.0)])

    assert tr._recent_success_rate(window_methods=2) == pytest.approx(0.5)
    assert tr._recent_success_rate(window_methods=1) == pytest.approx(1.0)


def test_security_boundary_raises_on_multiple_defenders_without_name():
    """多于一个防御方且未显式指定时 raise ValueError（不再任意取插入序第一个）。"""
    tr = ELOTracker()
    tr.update_round("def_a", [("m1", 3.0)])
    tr.update_round("def_b", [("m1", -1.0)])

    with pytest.raises(ValueError, match="defender_name"):
        tr.compute_security_boundary()

    b = tr.compute_security_boundary("def_a")  # 显式指定则正常
    assert b["defender"] == "def_a"


def test_security_boundary_single_defender_default_still_works():
    """恰有一个防御方时缺省取它（唯一选择，无歧义）。"""
    tr = ELOTracker()
    tr.update_round("def_a", [("m1", 3.0)])

    assert tr.compute_security_boundary()["defender"] == "def_a"
    empty = ELOTracker().compute_security_boundary()  # 无防御方：早退 dict，不 raise
    assert empty["converged"] is False


# ===== elo_access 补充覆盖：tracker memoize / 缓存容错 / active_model =====

def test_elo_tracker_for_memoize_and_invalidate(tmp_path, monkeypatch):
    """elo_tracker_for：无数据 None；命中进程内缓存返回同一对象；R 变动重派生。"""
    _setup(tmp_path, monkeypatch)
    ea._TRACKER_CACHE.clear()
    try:
        assert ea.elo_tracker_for("ghost") is None, "R 无该模型列 → None"

        t = ELOTracker()
        t.update_round("m1", [("DAN", 3.5)])
        t.record_round_end("m1")
        ea.publish_tracker(t, "m1")

        tr1 = ea.elo_tracker_for("m1")
        assert tr1 is not None and "DAN" in tr1.attacker_ratings
        tr2 = ea.elo_tracker_for("m1")
        assert tr1 is tr2, "同指纹应命中进程内缓存（同一对象）"

        # R 变动 → 指纹变 → 重派生新对象
        R = ResultsMatrix.load()
        R.upsert("new-rec", "m1", 2.0, status="fully_compliant", extra={"unit": "DAN"})
        R.save()
        tr3 = ea.elo_tracker_for("m1")
        assert tr3 is not tr1, "R 列变动应使 tracker 缓存失效"
        assert ea.attacker_ratings_for("m1"), "便捷读取返回非空映射"
    finally:
        ea._TRACKER_CACHE.clear()


def test_elo_cache_row_version_drift_is_miss(tmp_path, monkeypatch):
    """P2 表化：elo_cache 行 payload 版本漂移 → 视为 miss 重算（指纹相同也不命中）。"""
    from llmsec.storage import rstore

    _setup(tmp_path, monkeypatch)
    rstore.upsert_elo_cache("qwen9b", "deadbeef", {"_version": 999, "k": 1})
    assert ea._cache_hit("qwen9b", "deadbeef") is None


def test_active_model_by_latest_ts(tmp_path, monkeypatch):
    """active_model 取结果 ts 最新的模型；ts 混合 int/str 也能比较；R 空 → None。"""
    _setup(tmp_path, monkeypatch)
    assert ea.active_model() is None, "空 R → None"

    R = ResultsMatrix()
    R.upsert("r1", "older", 3.0, status="ok", ts=10)
    R.upsert("r2", "newer", 3.0, status="ok", ts=20)
    R.save()
    assert ea.active_model() == "newer", "纯数字 ts：应取最大的模型"

    # 字符串 ts 走防 TypeError 的兜底分支（(1,str) 恒大于 (0,float)）：混型时
    # active_model 的"最新"判定无真值，此处仅锁定与 ordered_results 一致的确定性次序
    R.upsert("r3", "iso", 3.0, status="ok", ts="2026-01-01")
    R.save()
    assert ea.active_model() == "iso", "兜底次序：字符串 ts 排在数字 ts 之后"
    assert ea._ts_key(5) == (0, 5.0) and ea._ts_key("x") == (1, "x"), "数值/字符串双分支"
