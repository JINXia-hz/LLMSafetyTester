"""
审查修复回归（D 组）：
1. allergy_phase 过敏判定 OR 口径（M-33 对齐 safe_twin）：judge 未判拒绝但
   ≥1 条 FAST_REFUSAL_PATTERNS 关键词命中时仍应算过敏，漏判会低估 FPR。
2. safe_twin.generate_all_twins 缺键防护：记录缺 category/harm_type/method
   时填 "unknown"，缺 id/prompt 时跳过该条而不是 KeyError 中断整批。
"""
import json

import llmsec.evaluation.safe_twin as safe_twin
import llmsec.pipeline.allergy_phase as allergy_phase

# ---------- 1. allergy_phase OR 口径 ----------

class _FakeTracker:
    # S3 守卫直接读 defender_ratings：恰一个防御方（defender_name=None 口径）才算有数据
    defender_ratings = {"fake_def": 1500.0}

    def get_attacker_ranking(self):
        return [{"method": "m1", "elo": 1490.0}]

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
    # 孪生缓存重定向到 tmp，避免读全局 output/state/safe_twins.jsonl
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


# ---------- 2. safe_twin 缺键不崩 ----------

def _setup_twin_files(monkeypatch, tmp_path, records):
    attack_file = tmp_path / "attacks.jsonl"
    attack_file.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records),
        encoding="utf-8",
    )
    twin_file = tmp_path / "safe_twins.jsonl"
    monkeypatch.setattr(safe_twin, "INPUT_FILE", attack_file)
    # TWIN_FILE 别名已删，落盘路径统一为 core.config.SAFE_TWINS_FILE（safe_twin 顶层导入）
    monkeypatch.setattr(safe_twin, "SAFE_TWINS_FILE", twin_file)
    monkeypatch.setattr(safe_twin, "API_DELAY", 0)
    monkeypatch.setattr(safe_twin, "create_openai_client", lambda **kw: object())
    monkeypatch.setattr(safe_twin, "generate_safe_twin",
                        lambda prompt, client: {"safe_prompt": "s", "replacement": "r"})
    return twin_file


def test_generate_all_twins_missing_optional_keys(monkeypatch, tmp_path):
    """缺 category/harm_type/method 的记录填 unknown 并正常落盘。"""
    twin_file = _setup_twin_files(monkeypatch, tmp_path, [
        {"id": "a1", "prompt": "p1", "method": "m1"},
    ])
    safe_twin.generate_all_twins()
    rows = [json.loads(x) for x in twin_file.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["category"] == "unknown"
    assert rows[0]["harm_type"] == "unknown"
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
