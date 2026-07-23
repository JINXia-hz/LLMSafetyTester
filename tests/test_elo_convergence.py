#!/usr/bin/env python3
"""
回归测试：Elo 收敛判定与固定簇预测。

验证：
1. predict 对同基底变体优先使用变体 ground truth，而不是整个簇的平均。
2. check_convergence 在成功率偏离 50% 时不判收敛（抗假阳性）。
3. check_convergence 在标准差小、成功率接近 50%、覆盖率足够时判收敛。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llmsec.evaluation.elo import ELOTracker
from llmsec.evaluation.elo_cluster import ClusterEloPredictor, _strip_variant_suffix


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


def test_predict_variant_fallback() -> int:
    """predict 应优先用同基底变体的 ground truth。"""
    predictor = ClusterEloPredictor()
    predictor.ground_truth = {
        "attack_rot13": {"elo": 1800.0},
        "attack_b64": {"elo": 1200.0},
        "other_method": {"elo": 1500.0},
    }
    # 伪造 artifacts：attack_code 与 attack_rot13/attack_b64 同簇
    predictor.artifacts = {
        "labels": {
            "attack_rot13": 0,
            "attack_b64": 0,
            "attack_code": 0,
            "other_method": 1,
        },
        "features": {},  # 无特征时回退简单平均
        "weights": (0.15, 0.45, 0.25, 0.0),
    }

    # 预测同基底的新变体 attack_code：应接近变体平均 (1800+1200)/2 = 1500，但来源是 variant
    # 不传入 record，避免触发 embedding 网络请求；测试只验证变体兜底逻辑
    pred = predictor.predict("attack_code", record=None)
    if pred["source"] != "predicted_variant":
        print(f"❌ predict 未优先使用变体兜底: source={pred['source']}")
        return 1
    expected = (1800.0 + 1200.0) / 2
    if abs(pred["elo"] - expected) > 1e-6:
        print(f"❌ predict 变体预测错误: elo={pred['elo']}, expected={expected}")
        return 1

    # 预测没有同基底变体的 other_new：应使用簇内/全局平均
    pred2 = predictor.predict("other_new", record=None)
    if pred2["source"] == "predicted_variant":
        print(f"❌ predict 错误地把无关方法识别为变体: source={pred2['source']}")
        return 1

    print("✅ predict 变体兜底通过")
    return 0


def test_convergence_resists_false_positive() -> int:
    """最近成功率偏离 50% 时不应判收敛。"""
    tracker = ELOTracker()
    defender = "test-model"

    # 模拟 5 轮，防御方 Elo 小幅波动，但全部失败（成功率 0%）
    for _ in range(25):
        tracker.update("low_elo_attack", defender, -2.0)
    tracker.record_round_end(defender)

    for _ in range(5):
        tracker._round_defender_elos[defender].append(730.0)
    for _ in range(5):
        tracker._round_defender_elos[defender].append(731.0)
    for _ in range(5):
        tracker._round_defender_elos[defender].append(732.0)

    # 构造足够的方法数以满足覆盖率
    for i in range(50):
        tracker.attacker_ratings[f"method_{i}"] = 700.0
    for i in range(15):
        tracker.ground_truth_methods.add(f"method_{i}")

    conv = tracker.check_convergence(defender, total_methods=50)
    if conv["converged"]:
        print(f"❌ 假收敛未被拦截: success_rate={conv['recent_success_rate']}, coverage={conv['coverage']}")
        return 1
    print("✅ 抗假阳性收敛判定通过")
    return 0


def test_convergence_true_positive() -> int:
    """标准差小、成功率接近 50%、覆盖率足够时应判收敛。"""
    tracker = ELOTracker()
    defender = "test-model"

    # 让防御方 Elo 稳定在 1500 附近
    tracker.defender_ratings[defender] = 1500.0
    tracker._round_defender_elos[defender] = [1495.0, 1502.0, 1498.0]

    # 最近 15 次成功率约 50%
    for i in range(15):
        score = 2.0 if i % 2 == 0 else -2.0
        tracker.update(f"method_{i}", defender, score)
        tracker.ground_truth_methods.add(f"method_{i}")

    # 总方法 50，已测 15 => 覆盖率 30%
    for i in range(50):
        tracker.attacker_ratings[f"method_{i}"] = 1500.0

    conv = tracker.check_convergence(defender, total_methods=50)
    if not conv["converged"]:
        print(f"❌ 真收敛未通过: {conv}")
        return 1
    print("✅ 真收敛判定通过")
    return 0


def main() -> int:
    tests = [
        test_strip_variant_suffix,
        test_predict_variant_fallback,
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
