#!/usr/bin/env python3
"""
回归测试：Elo 收敛判定与固定簇预测。

验证：
1. predict 对同后缀变体优先使用同后缀 ground truth。
2. predict 对同基底变体使用变体 ground truth（同后缀不存在时回退）。
3. check_convergence 在 Elo 标准差不满足时不判收敛（抗假阳性）。
4. check_convergence 在标准差小、覆盖率足够时判收敛。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llmsec.evaluation.elo import ELOTracker
from llmsec.evaluation.elo_cluster import (
    ClusterEloPredictor,
    _extract_variant_suffix,
    _strip_variant_suffix,
)


def test_strip_variant_suffix() -> int:
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
        if got != expected:
            print(f"❌ _strip_variant_suffix({raw!r}) = {got!r}, expected {expected!r}")
            return 1
    print("✅ 变体后缀剥离通过")
    return 0


def test_extract_variant_suffix() -> int:
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
        if got != expected:
            print(f"❌ _extract_variant_suffix({raw!r}) = {got!r}, expected {expected!r}")
            return 1
    print("✅ 变体后缀提取通过")
    return 0


def test_predict_suffix_variant_fallback() -> int:
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
    pred = predictor.predict("attack_d_rot13", record=None)
    if pred["source"] != "predicted_suffix_variant":
        print(f"❌ predict 未优先使用同后缀变体兜底: source={pred['source']}")
        return 1
    expected = (1800.0 + 1700.0) / 2
    if abs(pred["elo"] - expected) > 1e-6:
        print(f"❌ predict 同后缀预测错误: elo={pred['elo']}, expected={expected}")
        return 1

    # 预测同基底但不同后缀的 attack_e_code：应回退到同基底变体（但同基底只有 attack_c_b64，后缀不同）
    # attack_e_code 与 attack_c_b64 不是同基底（基底是 attack_c vs attack_e），也不是同后缀
    # 所以应该使用簇内/全局平均
    pred2 = predictor.predict("attack_e_code", record=None)
    if pred2["source"] == "predicted_suffix_variant":
        print(f"❌ predict 错误地把无关方法识别为同后缀变体: source={pred2['source']}")
        return 1

    print("✅ predict 同后缀变体兜底通过")
    return 0


def test_predict_base_variant_fallback() -> int:
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
    pred = predictor.predict("attack_code", record=None)
    if pred["source"] != "predicted_variant":
        print(f"❌ predict 未回退到同基底变体: source={pred['source']}")
        return 1
    expected = (1800.0 + 1200.0) / 2
    if abs(pred["elo"] - expected) > 1e-6:
        print(f"❌ predict 同基底预测错误: elo={pred['elo']}, expected={expected}")
        return 1

    print("✅ predict 同基底变体兜底通过")
    return 0


def test_convergence_resists_false_positive() -> int:
    """Elo 标准差不满足时不应判收敛。"""
    tracker = ELOTracker()
    defender = "test-model"

    # 模拟多轮，防御方 Elo 大幅波动（不满足 std < 10）
    tracker._round_defender_elos[defender] = [1500.0, 1550.0, 1480.0]
    tracker.defender_ratings[defender] = 1480.0

    # 构造足够的方法数以满足覆盖率
    for i in range(50):
        tracker.attacker_ratings[f"method_{i}"] = 700.0
    for i in range(15):
        tracker.ground_truth_methods.add(f"method_{i}")

    conv = tracker.check_convergence(defender, total_methods=50)
    if conv["converged"]:
        print(f"❌ 假收敛未被拦截: std={conv['std']}, coverage={conv['coverage']}")
        return 1
    print("✅ 抗假阳性收敛判定通过")
    return 0


def test_convergence_true_positive() -> int:
    """标准差小、覆盖率足够时应判收敛。"""
    tracker = ELOTracker()
    defender = "test-model"

    # 让防御方 Elo 稳定在 1500 附近
    tracker.defender_ratings[defender] = 1500.0
    tracker._round_defender_elos[defender] = [1495.0, 1502.0, 1498.0]

    # 总方法 50，已测 15 => 覆盖率 30%
    for i in range(50):
        tracker.attacker_ratings[f"method_{i}"] = 1500.0
    for i in range(15):
        tracker.ground_truth_methods.add(f"method_{i}")

    conv = tracker.check_convergence(defender, total_methods=50)
    if not conv["converged"]:
        print(f"❌ 真收敛未通过: {conv}")
        return 1
    print("✅ 真收敛判定通过")
    return 0


def main() -> int:
    tests = [
        test_strip_variant_suffix,
        test_extract_variant_suffix,
        test_predict_suffix_variant_fallback,
        test_predict_base_variant_fallback,
        test_convergence_resists_false_positive,
        test_convergence_true_positive,
    ]
    for t in tests:
        if t() != 0:
            return 1
    print("\n✅ 所有 Elo 收敛/预测回归测试通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
