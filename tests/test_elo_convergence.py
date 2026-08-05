#!/usr/bin/env python3
"""
回归测试：Elo 收敛判定与固定簇预测。

验证：
1. predict 对同后缀变体优先使用同后缀 ground truth。
2. predict 对同基底变体使用变体 ground truth（同后缀不存在时回退）。
3. check_convergence 在 Elo 噪声大（95%CI 半宽超目标）时不判收敛（抗假阳性）。
4. check_convergence 在噪声小、漂移小、覆盖率足够时判收敛。
"""



from llmsec.evaluation.elo import ELOTracker
from llmsec.evaluation.elo_cluster import (
    ClusterEloPredictor,
    _extract_variant_suffix,
    _strip_variant_suffix,
)


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
    predictor = ClusterEloPredictor()
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
    predictor = ClusterEloPredictor()
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
    tracker._round_defender_elos[defender] = [1500.0, 1560.0, 1490.0, 1555.0, 1505.0]
    tracker.defender_ratings[defender] = 1505.0

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
    tracker.defender_ratings[defender] = 1501.0
    tracker._round_defender_elos[defender] = [1495.0, 1502.0, 1498.0, 1501.0]

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


