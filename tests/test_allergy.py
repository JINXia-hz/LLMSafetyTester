"""
回归测试：过敏检测候选选取（select_twin_candidates）。

针对 runner.run_allergy_phase 曾有的两个缺陷：
1. 一侧不足不补齐——自适应窗口 9 只选出 5（兜底分支形同虚设）；
2. 上方取错侧——ranking 按 Elo 降序，above[:k] 取到离边界最远的强攻击。
"""
from llmsec.pipeline.allergy_phase import select_twin_candidates


def _ranking(elos: list[float]):
    """构造与 ELOTracker.get_attacker_ranking 相同的降序结构（unit 键）。"""
    ranking = [{'unit': f'm{i:03d}', 'elo': e} for i, e in enumerate(elos)]
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


# ===== from test_eval_review_allergy.py（评审修复回归：D 组）=====
# 1. 过敏判定 OR 口径：judge 未判拒但关键词命中仍算过敏（漏判会低估 FPR）。
# 2. safe_twin.generate_all_twins 缺键防护：缺可选键填默认、缺 id/prompt 跳过该条。
import json

import llmsec.evaluation.safe_twin as safe_twin
import llmsec.pipeline.allergy_phase as allergy_phase


class _FakeTracker:
    # S3 守卫直接读 defender_ratings：恰一个防御方（defender_name=None 口径）才算有数据
    defender_ratings = {"fake_def": 1500.0}

    def get_attacker_ranking(self):
        return [{"unit": "m1", "elo": 1490.0}]

    def compute_security_boundary(self, _name):
        return {"boundary_elo": 1500.0}

    def get_attacker_elo(self, _m):
        return 1490.0


class _FakeJudge:
    """judge 认为不构成拒绝（compliance A）。"""

    def evaluate(self, _prompt, _content):
        return {"is_refusal": False, "compliance_level": "A"}


def _run_phase(monkeypatch, tmp_path, content):
    monkeypatch.setattr(allergy_phase, "get_or_create_twin",
                        lambda method, rec, cache, client: "安全孪生prompt")
    # call_target / API_DELAY 是 allergy_phase 的模块顶层引用，patch 落在真实消费方
    monkeypatch.setattr(allergy_phase, "call_target",
                        lambda prompt: {"error": None, "content": content, "meta": {}})
    monkeypatch.setattr(allergy_phase, "API_DELAY", 0)
    monkeypatch.setattr(allergy_phase, "SAFE_TWINS_FILE", tmp_path / "safe_twins.jsonl")
    return allergy_phase.run_allergy_phase(
        {"m1": {"id": "x1", "prompt": "p", "category": "c", "method": "m1"}},
        twin_client=None, judge=_FakeJudge(), tracker=_FakeTracker(),
        n_window=1, allergy_file=tmp_path / "allergy.json", concurrency=0)


def test_keyword_hit_counts_allergic_even_if_judge_says_no(monkeypatch, tmp_path):
    """judge 未判拒绝、但回复含拒绝关键词 → 仍计为过敏（OR 口径）。"""
    summary = _run_phase(monkeypatch, tmp_path, "好的。不过抱歉，这个请求我无法完成。")
    assert summary["allergic"] == 1
    assert summary["fpr"] == 1.0


def test_clean_reply_not_allergic(monkeypatch, tmp_path):
    """judge 未判拒绝且无关键词命中 → 不过敏（OR 不误伤正常回复）。"""
    summary = _run_phase(monkeypatch, tmp_path, "制作生日蛋糕的步骤如下：先准备面粉……")
    assert summary["allergic"] == 0
    assert summary["fpr"] == 0.0


def _setup_twin_files(monkeypatch, tmp_path, records):
    attack_file = tmp_path / "attacks.jsonl"
    attack_file.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records),
        encoding="utf-8",
    )
    twin_file = tmp_path / "safe_twins.jsonl"
    monkeypatch.setattr(safe_twin, "INPUT_FILE", attack_file)
    monkeypatch.setattr(safe_twin, "SAFE_TWINS_FILE", twin_file)
    monkeypatch.setattr(safe_twin, "API_DELAY", 0)
    monkeypatch.setattr(safe_twin, "create_openai_client", lambda **kw: object())
    monkeypatch.setattr(safe_twin, "generate_safe_twin",
                        lambda prompt, client: {"safe_prompt": "s", "replacement": "r"})
    return twin_file


def test_generate_all_twins_missing_optional_keys(monkeypatch, tmp_path):
    """缺 category/harm_type/method 的记录填默认值并正常落盘（harm_type 归一化为 other）。"""
    twin_file = _setup_twin_files(monkeypatch, tmp_path, [
        {"id": "a1", "prompt": "p1", "method": "m1"},
    ])
    safe_twin.generate_all_twins()
    rows = [json.loads(x) for x in twin_file.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["category"] == "unknown"
    assert rows[0]["harm_type"] == "other"
    assert rows[0]["method"] == "m1"


def test_generate_all_twins_missing_id_or_prompt_skipped(monkeypatch, tmp_path):
    """缺 id/prompt 的记录跳过且不抛 KeyError，其余记录照常生成。"""
    twin_file = _setup_twin_files(monkeypatch, tmp_path, [
        {"prompt": "p-no-id", "method": "m0"},
        {"id": "b1", "method": "m1"},
        {"id": "b2", "prompt": "p2", "method": "m2", "category": "c", "harm_type": "h"},
    ])
    safe_twin.generate_all_twins()
    rows = [json.loads(x) for x in twin_file.read_text(encoding="utf-8").splitlines()]
    assert [r["original_id"] for r in rows] == ["b2"]
