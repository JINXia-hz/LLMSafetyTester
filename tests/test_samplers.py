"""回归测试：采样器参数生效性（审查 M-8/M-9/M-10）。

验证：
1. build_sampler 传给 hybrid 的 alpha/beta/gamma 不被静默丢弃（M-8）。
2. InfoGain 在候选恒为未测时，alpha 项靠 SVD-Ridge pred_std 提供逐方法不确定性（M-10）。
3. --coordinate-rounds 默认从 params 读（M-9）。
4. HybridSampler 同时传 info_gain_alpha 与别名 alpha 时告警（别名仍胜出）。
5. gamma 项经 Elo 期望胜率复活（未测方法历史成功率恒 0 的补救）。
"""

import warnings

from llmsec.evaluation.elo import ELOTracker
from llmsec.evaluation.samplers import HybridSampler, InfoGainSampler, build_sampler
from llmsec.params import SAMPLER_HYBRID_EXPLORE_ROUNDS


def test_hybrid_respects_alpha_beta_gamma():
    """M-8：build_sampler(hybrid, alpha=, beta=, gamma=) 应透传给内部 InfoGain。"""
    s = build_sampler("hybrid", alpha=11.0, beta=22.0, gamma=33.0)
    assert isinstance(s, HybridSampler)
    inner = s._info_sampler
    assert isinstance(inner, InfoGainSampler)
    assert (inner.alpha, inner.beta, inner.gamma) == (11.0, 22.0, 33.0)


def test_infogain_pred_std_breaks_ties():
    """M-10：未测候选的 uncertainty 应随 pred_std 变化，使 alpha 项能区分候选。"""
    tracker = ELOTracker()
    defender = "def"
    tracker.defender_ratings[defender] = 1500.0
    tracker.attacker_ratings = {"m_low_std": 1500.0, "m_high_std": 1500.0}
    tracker.attacker_pred_std = {"m_low_std": 5.0, "m_high_std": 150.0}
    s = InfoGainSampler(alpha=20.0, beta=0.0, gamma=0.0)
    s.set_cluster_info(cluster_report={"method_labels": {"m_low_std": 0, "m_high_std": 0}})
    chosen = s.select(["m_low_std", "m_high_std"], tracker, defender, n=1)
    assert chosen == ["m_high_std"], f"高 pred_std 未被优先: {chosen}"
    u_low = tracker.get_attacker_uncertainty("m_low_std")
    u_high = tracker.get_attacker_uncertainty("m_high_std")
    assert u_high > u_low > 0, f"pred_std 未进入 uncertainty: low={u_low} high={u_high}"


def test_infogain_gamma_expected_winrate_breaks_ties():
    """gamma 项经 Elo 期望胜率影响排序（未测方法历史成功率恒 0 的补救）。"""
    tracker = ELOTracker()
    defender = "def"
    tracker.defender_ratings[defender] = 1500.0
    tracker.attacker_ratings = {"m_strong": 1700.0, "m_weak": 1300.0}
    # 前置：未测方法历史成功率恒 0
    assert tracker.get_attacker_success_rate("m_strong") == 0.0
    assert tracker.get_attacker_success_rate("m_weak") == 0.0
    labels = {"method_labels": {"m_strong": 0, "m_weak": 0}}
    s = InfoGainSampler(alpha=0.0, beta=0.0, gamma=10.0)
    s.set_cluster_info(cluster_report=labels)
    chosen = s.select(["m_weak", "m_strong"], tracker, defender, n=1)
    assert chosen == ["m_strong"], f"gamma 期望胜率未影响排序: {chosen}"
    # gamma 符号翻转 → 排序翻转，证明该项确实生效而非 gap 平局巧合
    s2 = InfoGainSampler(alpha=0.0, beta=0.0, gamma=-10.0)
    s2.set_cluster_info(cluster_report=labels)
    chosen2 = s2.select(["m_weak", "m_strong"], tracker, defender, n=1)
    assert chosen2 == ["m_weak"], f"gamma 取负时 m_weak 应优先: {chosen2}"


def test_hybrid_alias_conflict_warns():
    """info_gain_alpha 与别名 alpha 同传时应告警（别名仍胜出）。"""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        s = HybridSampler(info_gain_alpha=1.0, alpha=2.0)
    assert any("info_gain_alpha" in str(w.message) for w in rec), "同传未告警"
    assert s._info_sampler.alpha == 2.0
    # 仅传别名（正常路径）不应告警
    with warnings.catch_warnings(record=True) as rec2:
        warnings.simplefilter("always")
        HybridSampler(alpha=3.0)
    assert not any("别名" in str(w.message) for w in rec2), "仅传别名时不应告警"


def test_coordinate_rounds_default_from_params():
    """M-9：--coordinate-rounds 默认应来自 params 而非硬编码。"""
    assert HybridSampler().explore_rounds == SAMPLER_HYBRID_EXPLORE_ROUNDS


# ===== 补充覆盖：CoordinateDescent 选择逻辑 / 边界与空候选 =====

def _tracker_with(attackers: dict[str, float], defender_elo=1500.0):
    tracker = ELOTracker()
    tracker.defender_ratings["def"] = defender_elo
    tracker.attacker_ratings = dict(attackers)
    return tracker


def test_empty_candidates_short_circuit():
    """空候选列表：所有采样器返回空、不触 tracker。"""
    from llmsec.evaluation.samplers import CoordinateDescentSampler, GapMinSampler

    tracker = _tracker_with({})
    assert GapMinSampler().select([], tracker, "def", n=3) == []
    assert InfoGainSampler().select([], tracker, "def", n=3) == []
    assert CoordinateDescentSampler().select([], tracker, "def", n=3) == []


def test_infogain_tested_method_uses_history_success_rate():
    """已测方法（n_matches>0）的成功潜力走历史成功率而非期望胜率。"""
    tracker = _tracker_with({"m_win": 1600.0, "m_lose": 1400.0})
    # 同 Elo、同簇，仅历史成功率不同：高胜率者 gamma 项更优
    tracker.attacker_stats = {
        "m_win": {"n_matches": 4, "wins": 4},
        "m_lose": {"n_matches": 4, "wins": 0},
    }
    labels = {"method_labels": {"m_win": 0, "m_lose": 0}}
    s = InfoGainSampler(alpha=0.0, beta=0.0, gamma=10.0)
    s.set_cluster_info(cluster_report=labels)
    assert s.select(["m_win", "m_lose"], tracker, "def", n=1) == ["m_win"]


def test_coordinate_prefers_boundary_within_cluster():
    """坐标下降：在聚焦簇内优先选离边界最近的方法。"""
    from llmsec.evaluation.samplers import CoordinateDescentSampler

    tracker = _tracker_with({
        "c0_near": 1495.0, "c0_far": 1350.0, "c1_only": 1520.0,
    })
    labels = {"method_labels": {"c0_near": 0, "c0_far": 0, "c1_only": 1}}
    s = CoordinateDescentSampler()
    s.set_cluster_info(cluster_report=labels)
    chosen = s.select(["c0_far", "c0_near", "c1_only"], tracker, "def", n=1)
    # 首轮聚焦评分最优簇；同簇内未测方法按 gap 排序
    assert chosen == ["c0_near"] or chosen == ["c1_only"], \
        f"应选聚焦簇内最近边界者: {chosen}"


def test_coordinate_exhausted_cluster_fills_globally():
    """聚焦簇耗尽时按全局 gap 补足，不返回缩水批次。"""
    from llmsec.evaluation.samplers import CoordinateDescentSampler

    tracker = _tracker_with({
        "c0_a": 1490.0, "c0_b": 1495.0, "c1_x": 1505.0, "c1_y": 1600.0,
    })
    labels = {"method_labels": {"c0_a": 0, "c0_b": 0, "c1_x": 1, "c1_y": 1}}
    s = CoordinateDescentSampler()
    s.set_cluster_info(cluster_report=labels)
    chosen = s.select(["c0_a", "c0_b", "c1_x", "c1_y"], tracker, "def", n=3)
    assert len(chosen) == 3, f"簇耗尽应从全局补足到 n: {chosen}"


def test_coordinate_rotates_clusters():
    """外层簇轮询：连续多轮选择应覆盖多个簇（不扎堆单一簇）。"""
    from llmsec.evaluation.samplers import CoordinateDescentSampler

    tracker = _tracker_with({
        "c0_a": 1500.0, "c0_b": 1500.0, "c1_a": 1500.0, "c1_b": 1500.0,
        "c2_a": 1500.0, "c2_b": 1500.0,
    })
    labels = {"method_labels": {m: int(m[1]) for m in tracker.attacker_ratings}}
    s = CoordinateDescentSampler(min_tests_per_cluster=1)
    s.set_cluster_info(cluster_report=labels)

    seen_clusters = set()
    candidates = list(tracker.attacker_ratings)
    for _ in range(3):
        chosen = s.select(candidates, tracker, "def", n=1)
        assert len(chosen) == 1
        seen_clusters.add(labels["method_labels"][chosen[0]])
    assert len(seen_clusters) >= 2, f"多轮应轮询不同簇: {seen_clusters}"


def test_coordinate_unlabeled_methods_fallback_queue():
    """无聚类标签（method_labels 空）时簇队列退化为 [-1]，仍可选取。"""
    from llmsec.evaluation.samplers import CoordinateDescentSampler

    tracker = _tracker_with({"m1": 1500.0, "m2": 1510.0})
    s = CoordinateDescentSampler()
    s.set_cluster_info(cluster_report={})
    assert s._build_cluster_queue(["m1", "m2"]) == [-1]
    chosen = s.select(["m1", "m2"], tracker, "def", n=2)
    assert sorted(chosen) == ["m1", "m2"]


def test_hybrid_select_switches_sub_sampler():
    """Hybrid：前 explore_rounds 轮走 infogain，之后切换 coordinate。"""
    tracker = _tracker_with({"m1": 1500.0, "m2": 1510.0})
    labels = {"method_labels": {"m1": 0, "m2": 0}}
    s = HybridSampler(explore_rounds=2)
    s.set_cluster_info(cluster_report=labels)

    assert s.select(["m1", "m2"], tracker, "def", n=1) and s.last_sub_sampler == "infogain"
    assert s.select(["m1", "m2"], tracker, "def", n=1) and s.last_sub_sampler == "infogain"
    assert s.select(["m1", "m2"], tracker, "def", n=1) and s.last_sub_sampler == "coordinate"


def test_build_sampler_unknown_name_raises():
    import pytest

    with pytest.raises(ValueError, match="未知采样器"):
        build_sampler("nope")
