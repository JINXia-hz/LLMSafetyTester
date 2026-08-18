"""攻击记录契约 + 体检校验器测试（Step 1「契约+体检」）。

纯静态/合成数据，零网络。契约的定位是体检不是门禁：必填三件套与长度界
硬校验（ValidationError），harm_type/source 宽松收（未知值报告分布不拒收）。
"""

import json

import pytest
from pydantic import ValidationError

from llmsec.attacks.schema import (
    AttackRecord,
    detect_mojibake,
    infer_source,
    validate_record,
)


# ============================================================
# 契约：必填三件套 / 长度界 / 宽松枚举 / extra 透传 / 血统
# ============================================================
class TestAttackRecord:
    def test_minimal_valid_and_defaults(self):
        rec = AttackRecord.model_validate({"id": "x-1", "method": "DAN", "prompt": "p"})
        assert rec.harm_type == "other" and rec.source == "unknown"
        assert rec.evolved is False and rec.parent_id is None
        assert rec.operator is None and rec.generation is None

    def test_missing_required(self):
        for missing in ("id", "method", "prompt"):
            data = {"id": "a", "method": "b", "prompt": "c"}
            data.pop(missing)
            with pytest.raises(ValidationError):
                AttackRecord.model_validate(data)

    def test_prompt_length_bounds(self):
        with pytest.raises(ValidationError):
            AttackRecord.model_validate({"id": "a", "method": "b", "prompt": ""})
        # 上界内长 prompt（b64 全量包装的常态）合法
        rec = AttackRecord.model_validate({"id": "a", "method": "b", "prompt": "x" * 99_999})
        assert len(rec.prompt) == 99_999
        with pytest.raises(ValidationError):
            AttackRecord.model_validate({"id": "a", "method": "b", "prompt": "x" * 100_001})

    def test_extra_fields_passthrough(self):
        """harmbench 溯源字段（behavior_id/jailbreak_template_idx/...）原样透传。"""
        rec = AttackRecord.model_validate({
            "id": "hb-1", "method": "m", "prompt": "p",
            "behavior_id": "bid", "jailbreak_template_idx": 3, "obfuscation": "b64",
        })
        assert rec.model_extra["behavior_id"] == "bid"
        assert rec.model_extra["jailbreak_template_idx"] == 3
        assert rec.model_extra["obfuscation"] == "b64"

    def test_unknown_harm_and_source_not_rejected(self):
        rec = AttackRecord.model_validate({
            "id": "a", "method": "b", "prompt": "p",
            "harm_type": "weird", "source": "mars",
        })
        assert rec.harm_type == "weird" and rec.source == "mars"

    def test_lineage_fields_accepted(self):
        rec = AttackRecord.model_validate({
            "id": "ev-c123-001", "method": "m", "prompt": "p",
            "evolved": True, "operator": "obfuscate:b64", "parent_id": "x-1", "generation": 1,
        })
        assert rec.evolved and rec.parent_id == "x-1" and rec.generation == 1


class TestValidateRecord:
    def test_valid_record_no_issues(self):
        rec, issues = validate_record({"id": "a", "method": "m", "prompt": "p"})
        assert rec is not None and issues == []

    def test_invalid_collects_issue_strings(self):
        rec, issues = validate_record({"method": "m"})
        assert rec is None
        assert any(i.startswith("id:") for i in issues)
        assert any(i.startswith("prompt:") for i in issues)

    def test_source_override(self):
        rec, _ = validate_record({"id": "a", "method": "m", "prompt": "p"}, source="l1")
        assert rec is not None and rec.source == "l1"

    def test_evolved_without_parent_is_warn_not_error(self):
        """血统缺失是 warn 级：记录有效但带 issue（供体检计数）。"""
        rec, issues = validate_record({
            "id": "ev-1", "method": "m", "prompt": "p", "evolved": True,
        })
        assert rec is not None
        assert issues == ["lineage: evolved=True 但缺 parent_id"]


# ============================================================
# mojibake 探测（启发式：正常简体中文不误报，鈥 类特征命中）
# ============================================================
class TestMojibakeDetect:
    def test_clean_chinese_not_flagged(self):
        assert detect_mojibake("请帮我评估这个攻击测试的安全边界。模型应当拒绝。") == []

    def test_known_mojibake_flagged(self):
        hits = detect_mojibake("OPPO doesn鈥檛 follow ANY principles")
        assert "鈥" in hits

    def test_hits_deduped(self):
        assert detect_mojibake("鈥鈥鈥") == ["鈥"]


class TestInferSource:
    def test_by_filename(self):
        assert infer_source("attacks/l1.jsonl") == "l1"
        assert infer_source("harmbench_ensemble.jsonl") == "harmbench"
        assert infer_source(r"C:\x\wildjailbreak.jsonl") == "wildjailbreak"

    def test_record_field_wins_and_unknown_fallback(self):
        assert infer_source("l1.jsonl", "custom") == "custom"
        assert infer_source("all_merged.jsonl") == "unknown"


# ============================================================
# 体检校验器（合成脏数据）
# ============================================================
def _write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


class TestHealthCheck:
    def test_counts_on_synthetic_mix(self, tmp_path):
        from llmsec.attacks.validate import health_check

        f = tmp_path / "l1.jsonl"
        _write_jsonl(f, [
            {"id": "a-1", "method": "m1", "prompt": "p1", "harm_type": "hate"},
            {"id": "a-2", "method": "m1", "prompt": "p2", "harm_type": "other"},
            {"id": "a-3", "method": "m2", "prompt": ""},                 # 空 prompt：error 级
            {"id": "a-4", "method": "m2", "prompt": "鈥檛 mojibake"},     # 乱码命中
            {"id": "a-5", "method": "m2", "prompt": "p1"},               # 与 a-1 重复 prompt
            {"id": "a-5", "method": "m2", "prompt": "p9"},               # 重复 id
        ])
        with open(f, "a", encoding="utf-8") as fh:
            fh.write("{bad json\n")                                      # 坏 JSON 行

        rep = health_check([f])
        r = rep["files"][0]
        assert r["total"] == 7 and r["valid"] == 5 and r["bad_json_lines"] == 1
        assert r["mojibake"]["count"] == 1 and r["mojibake"]["sample_ids"] == ["a-4"]
        assert r["dup_prompt"]["count"] == 1 and r["dup_prompt"]["sample_ids"] == ["a-5"]
        assert r["dup_id"]["count"] == 1 and r["dup_id"]["sample_ids"] == ["a-5"]
        # 违规行不进分布；未写 harm_type 的行默认 other；占比分母是有效记录数
        assert r["harm_dist"] == {"other": 4, "hate": 1}
        assert r["other_ratio"] == pytest.approx(4 / 5)
        assert r["method_cardinality"] == 2
        assert r["top_methods"][0] == ("m2", 3)
        # error 样例带违规定位
        assert r["error_samples"][0]["id"] == "a-3"
        assert any(i.startswith("prompt:") for i in r["error_samples"][0]["issues"])
        # source 从文件名推断
        assert r["source_dist"] == {"l1": 5}
        # 汇总口径一致
        s = rep["summary"]
        assert s["total_records"] == 7 and s["valid_records"] == 5

    def test_cross_file_dup_groups(self, tmp_path):
        from llmsec.attacks.validate import health_check

        f1 = tmp_path / "l1.jsonl"
        f2 = tmp_path / "in_the_wild.jsonl"
        _write_jsonl(f1, [{"id": "x-1", "method": "m", "prompt": "shared"}])
        _write_jsonl(f2, [
            {"id": "y-1", "method": "m", "prompt": "shared"},
            {"id": "y-2", "method": "m", "prompt": "uniq"},
        ])
        rep = health_check([f1, f2])
        s = rep["summary"]
        assert s["cross_file_dup_group_count"] == 1
        g = s["cross_file_dup_groups"][0]
        assert set(g["files"]) == {"l1.jsonl", "in_the_wild.jsonl"}

    def test_cli_writes_json_report(self, tmp_path):
        from llmsec.attacks import validate as v

        f = tmp_path / "l1.jsonl"
        _write_jsonl(f, [{"id": "a-1", "method": "m", "prompt": "p"}])
        out = tmp_path / "health.json"
        assert v.main([str(f), "--out", str(out)]) == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["files"][0]["total"] == 1
        assert "summary" in data

    def test_cli_missing_files_returns_1(self, tmp_path, caplog):
        from llmsec.attacks import validate as v

        assert v.main([str(tmp_path / "nope.jsonl")]) == 1
