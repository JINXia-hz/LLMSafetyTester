"""tests/test_taxonomy — harm_type 归一化层测试。"""
import pytest

from llmsec.core.taxonomy import (
    CANONICAL_HARM_TYPES,
    normalize_harm_type,
    normalize_harm_types,
)


class TestNormalizeHarmType:
    """normalize_harm_type 单值归一化。"""

    @pytest.mark.parametrize("raw", CANONICAL_HARM_TYPES)
    def test_canonical_passes_through(self, raw):
        """标准词表值原样返回。"""
        assert normalize_harm_type(raw) == raw

    def test_empty_returns_other(self):
        assert normalize_harm_type("") == "other"
        assert normalize_harm_type(None) == "other"

    def test_unknown_returns_other(self):
        assert normalize_harm_type("totally_unknown_type") == "other"
        assert normalize_harm_type("random_garbage_xyz") == "other"

    def test_case_insensitive(self):
        """大小写不敏感：Violence / VIOLENCE → violence。"""
        assert normalize_harm_type("Violence") == "violence"
        assert normalize_harm_type("VIOLENCE") == "violence"
        assert normalize_harm_type("Cyber") == "cyber"
        assert normalize_harm_type("COPYRIGHT") == "copyright"

    def test_whitespace_trimmed(self):
        assert normalize_harm_type("  violence  ") == "violence"
        assert normalize_harm_type("\tself_harm\n") == "self_harm"

    # ---- HarmBench 原生 SemanticCategory 映射 ----
    @pytest.mark.parametrize("raw,expected", [
        ("chemical_biological", "weapon_drugs"),
        ("chemical_weapons", "weapon_drugs"),
        ("cybercrime_intrusion", "cyber"),
        ("cybercrime", "cyber"),
        ("misinformation_disinformation", "misinfo"),
        ("misinformation", "misinfo"),
        ("harassment_bullying", "hate"),
        ("harassment", "hate"),
        ("harmful", "other"),
    ])
    def test_harmbench_aliases(self, raw, expected):
        assert normalize_harm_type(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("暴力", "violence"),
        ("仇恨", "hate"),
        ("欺诈", "fraud"),
        ("隐私", "privacy"),
        ("自残", "self_harm"),
        ("违法", "illegal"),
        ("色情", "sexual"),
        ("版权", "copyright"),
    ])
    def test_chinese_aliases(self, raw, expected):
        assert normalize_harm_type(raw) == expected

    def test_real_data_coverage(self):
        """验证实际数据中出现的所有 16 个唯一 harm_type 值都能归一化。"""
        real_values = [
            "other", "hate", "violence", "illegal", "cyber", "weapon_drugs",
            "fraud", "privacy", "self_harm", "sexual", "misinfo", "copyright",
            "cybercrime_intrusion", "chemical_biological",
            "misinformation_disinformation", "harassment_bullying", "harmful",
        ]
        normalized = {normalize_harm_type(v) for v in real_values}
        # 全部应落入标准词表
        assert normalized.issubset(set(CANONICAL_HARM_TYPES))
        # HarmBench 的值不应原样保留（必须被映射）
        for hb_raw in ["cybercrime_intrusion", "chemical_biological",
                        "misinformation_disinformation", "harassment_bullying", "harmful"]:
            assert normalize_harm_type(hb_raw) not in {
                "cybercrime_intrusion", "chemical_biological",
                "misinformation_disinformation", "harassment_bullying", "harmful",
            }


class TestNormalizeHarmTypes:
    """normalize_harm_types 批量归一化。"""

    def test_dedup_preserves_order(self):
        result = normalize_harm_types(["violence", "Violence", "hate", "hate", "fraud"])
        assert result == ["violence", "hate", "fraud"]

    def test_empty_input(self):
        assert normalize_harm_types([]) == []

    def test_mixed_vocab_collapses(self):
        """混合词表归一化后坍缩到同一标准值。"""
        result = normalize_harm_types([
            "chemical_biological",    # → weapon_drugs
            "weapon_drugs",           # → weapon_drugs (dup)
            "cybercrime_intrusion",   # → cyber
            "cyber",                  # → cyber (dup)
        ])
        assert result == ["weapon_drugs", "cyber"]
