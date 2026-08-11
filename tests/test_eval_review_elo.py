"""评审修复回归：elo.py / elo_convergence.py（B 组）。

覆盖：
  5. _recent_success_rate 按方法去重——重复测试的方法不再被重复计入窗口。
  6. compute_security_boundary 缺省防御方：多防御方未显式指定时 raise ValueError。
"""

import pytest

from llmsec.evaluation.elo import ELOTracker


def test_recent_success_rate_dedupes_retested_methods():
    """同一方法重复测试时，窗口按" distinct 方法"计数，各取最近一次结果。"""
    tr = ELOTracker()
    # 时间序：m1 胜 → m2 胜 → m1 重测败（m1 最近一次为败）
    tr.update_round("def", [("m1", 3.0)])
    tr.update_round("def", [("m2", 3.0)])
    tr.update_round("def", [("m1", -1.0)])

    rate = tr._recent_success_rate(window_methods=15)
    # 去重口径：{m1: 最近=败, m2: 胜} → 1/2 = 0.5（旧口径数场次 = 2/3）
    assert rate == pytest.approx(0.5), f"应按方法去重取最近一次结果，得 {rate}"


def test_recent_success_rate_window_counts_distinct_methods():
    """window_methods 限的是"最近 N 个 distinct 方法"而非最近 N 场。"""
    tr = ELOTracker()
    # m1 连测 5 场全败，之后 m2 一场胜
    for _ in range(5):
        tr.update_round("def", [("m1", -1.0)])
    tr.update_round("def", [("m2", 3.0)])

    # 窗口 2：最近 2 个 distinct 方法 = {m2(胜), m1(败)} → 0.5
    assert tr._recent_success_rate(window_methods=2) == pytest.approx(0.5)
    # 窗口 1：最近 1 个 distinct 方法 = {m2(胜)} → 1.0（旧口径取最近 1 场也是 m2，但语义不同）
    assert tr._recent_success_rate(window_methods=1) == pytest.approx(1.0)


def test_security_boundary_raises_on_multiple_defenders_without_name():
    """多于一个防御方且未显式指定时 raise ValueError（不再任意取插入序第一个）。"""
    tr = ELOTracker()
    tr.update_round("def_a", [("m1", 3.0)])
    tr.update_round("def_b", [("m1", -1.0)])

    with pytest.raises(ValueError, match="defender_name"):
        tr.compute_security_boundary()

    # 显式指定则正常
    b = tr.compute_security_boundary("def_a")
    assert b["defender"] == "def_a"


def test_security_boundary_single_defender_default_still_works():
    """恰有一个防御方时缺省取它（唯一选择，无歧义）。"""
    tr = ELOTracker()
    tr.update_round("def_a", [("m1", 3.0)])

    b = tr.compute_security_boundary()
    assert b["defender"] == "def_a"

    # 无防御方时仍是早退 dict（不 raise）
    empty = ELOTracker().compute_security_boundary()
    assert empty["converged"] is False
