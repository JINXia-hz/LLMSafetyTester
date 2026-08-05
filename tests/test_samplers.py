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
