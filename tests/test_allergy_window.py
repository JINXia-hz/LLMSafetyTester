"""
回归测试：过敏检测候选选取（select_twin_candidates）。

针对 runner.run_allergy_phase 曾有的两个缺陷：
1. 一侧不足不补齐——自适应窗口 9 只选出 5（兜底分支形同虚设）；
2. 上方取错侧——ranking 按 Elo 降序，above[:k] 取到离边界最远的强攻击。
"""
from pathlib import Path
from llmsec.pipeline.runner import select_twin_candidates

def _ranking(elos: list[float]):
    """构造与 ELOTracker.get_attacker_ranking 相同的降序结构。"""
    ranking = [{'method': f'm{i:03d}', 'elo': e} for i, e in enumerate(elos)]
    ranking.sort(key=lambda x: x['elo'], reverse=True)
    return ranking

def test_both_sides_sufficient():
    elos = [1480, 1485, 1490, 1492, 1495, 1499, 1501, 1505, 1510, 1515, 1520, 1530]
    sel = select_twin_candidates(_ranking(elos), 1500.0, 9)
    sel_elos = [r['elo'] for r in sel]
    assert len(sel) == 9, '两侧充足：选满窗口 9'
    below = [e for e in sel_elos if e < 1500]
    above = [e for e in sel_elos if e >= 1500]
    assert len(below) == 4 and len(above) == 5, '下方 4 + 上方 5 各约一半'
    assert set(above) == {1501, 1505, 1510, 1515, 1520}, '上方取的是离边界最近的 5 个（回归：取错侧）'
    assert set(below) == {1499, 1495, 1492, 1490}, '下方取的是离边界最近的 4 个'

def test_one_side_short_backfill():
    elos = [1600, 1610, 1620, 1630, 1635, 1638, 1640, 1642, 1644, 1650]
    sel = select_twin_candidates(_ranking(elos), 1645.0, 9)
    sel_elos = [r['elo'] for r in sel]
    assert len(sel) == 9, '上方仅 1 个：仍选满窗口 9（回归：不补齐）'
    assert 1650 in sel_elos, '上方唯一方法被选中'
    assert set(sel_elos) == {1650, 1644, 1642, 1640, 1638, 1635, 1630, 1620, 1610}, '缺口按距离边界最近优先从下方补齐'

def test_one_side_empty():
    elos = [1400, 1450, 1480, 1490, 1495]
    sel = select_twin_candidates(_ranking(elos), 1500.0, 4)
    assert len(sel) == 4, '上方为空：全从下方取满'
    assert [r['elo'] for r in sel] == [1495, 1490, 1480, 1450], '按距离升序取最近的 4 个'

def test_window_exceeds_ranking():
    elos = [1490, 1510, 1520]
    sel = select_twin_candidates(_ranking(elos), 1500.0, 10)
    assert len(sel) == 3, '窗口超过方法数：截断为全部方法，不报错'
    assert select_twin_candidates([], 1500.0, 5) == [], '空 ranking 返回空'
    assert select_twin_candidates(_ranking(elos), 1500.0, 0) == [], '窗口 0 返回空'
