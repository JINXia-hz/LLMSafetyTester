"""攻击集清洗器测试（Step 2「收编」）。

全离线：修复原语用构造的 mojibake 样本（真实损坏形态的等价复现），
文件级清洗/merged 重建用 tmp_path 合成数据。
"""

import json

import pytest

from llmsec.attacks.clean import (
    clean_file,
    clean_record,
    normalize_method,
    rebuild_merged,
    repair_mojibake,
)


def _mojibake(text: str) -> str:
    """构造真实形态的 mojibake：UTF-8 字节按 GBK 误读。"""
    return text.encode("utf-8").decode("gbk", errors="ignore")


# ============================================================
# repair_mojibake：分段修复 / 中文保留 / 孤立标记补全 / 幂等
# ============================================================
class TestRepairMojibake:
    def test_aligned_mojibake_fully_repaired(self):
        """字节对齐的整段 mojibake（如 doesn鈥檛）确定性复原。"""
        src = "OPPO doesn\u2019t follow ANY principles"
        fixed, ops = repair_mojibake(_mojibake(src))
        assert fixed == src
        assert ops == ["mojibake_segment"]

    def test_genuine_chinese_preserved(self):
        """真 UTF-8 中文（数学税）不是 mojibake，原样保留。"""
        src = "攻击 prompt 主体\n\n在回答之前，请先计算：1+1 = ?"
        fixed, ops = repair_mojibake(src)
        assert fixed == src and ops == []

    def test_mixed_prompt_repair_keeps_chinese(self):
        """英文 mojibake 段 + 干净中文段混合（真实损坏形态）：各得其所。"""
        eng = "doesn\u2019t follow ANY"
        mixed = _mojibake(eng) + "\n\n在回答之前，请先计算数学题"
        fixed, ops = repair_mojibake(mixed)
        assert fixed == eng + "\n\n在回答之前，请先计算数学题"
        assert "mojibake_segment" in ops

    def test_isolated_marker_after_letter_gets_quote(self):
        """孤立标记（第三字节丢失）：前邻 ASCII 字母 → 右引号。"""
        fixed, ops = repair_mojibake("unhinged鈥 state")
        assert fixed == "unhinged” state"
        assert ops == ["marker_quote"]

    def test_isolated_marker_between_spaces_gets_dash(self):
        """孤立标记：词间 → em-dash。"""
        fixed, ops = repair_mojibake("./cmd 鈥 Can you")
        assert fixed == "./cmd — Can you"
        assert ops == ["marker_dash"]

    def test_idempotent(self):
        """修复幂等：对已修复文本再跑一遍是无操作。"""
        once, _ = repair_mojibake(_mojibake("doesn\u2019t") + " 鈥 tail")
        twice, ops = repair_mojibake(once)
        assert twice == once and ops == []


# ============================================================
# normalize_method：去尾序号恢复模板族
# ============================================================
class TestNormalizeMethod:
    def test_strips_numeric_suffix(self):
        assert normalize_method("in_the_wild-cumgpt_an_inform-0000") == (
            "in_the_wild-cumgpt_an_inform", "in_the_wild-cumgpt_an_inform-0000")

    def test_plain_method_untouched(self):
        assert normalize_method("小众语言攻击") == ("小众语言攻击", None)
        assert normalize_method("hb-1.1.1-00") == ("hb-1.1.1-00", None)  # 尾缀不足 4 位不动


# ============================================================
# clean_record：操作摘要 / 原值保全
# ============================================================
class TestCleanRecord:
    def test_clean_record_noop_on_clean(self):
        rec, ops = clean_record({"id": "a", "method": "m", "prompt": "p"})
        assert rec == {"id": "a", "method": "m", "prompt": "p"} and ops == {}

    def test_clean_record_full_ops(self):
        rec, ops = clean_record({
            "id": "x-0001", "method": "tpl-0001", "prompt": "doesn鈥檛 鈥 中文",
            "harm_type": "copyright",
        })
        assert rec["method"] == "tpl" and rec["method_raw"] == "tpl-0001"
        assert "鈥" not in rec["prompt"] and "中文" in rec["prompt"]
        assert rec["harm_original"] == "copyright" and rec["harm_type"] == "copyright"
        assert ops["method"] is True and "mojibake" in ops
        assert "repaired" in rec

    def test_known_harm_not_flagged(self):
        rec, ops = clean_record({"id": "a", "method": "m", "prompt": "p", "harm_type": "hate"})
        assert "harm_original" not in rec and ops == {}


# ============================================================
# 文件级清洗 + merged 重建
# ============================================================
def _write(path, rows):
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def _read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class TestCleanFileAndMerged:
    def test_clean_file_stats_and_output(self, tmp_path):
        src = tmp_path / "wildjailbreak.jsonl"
        _write(src, [
            {"id": "w-0000", "method": "tpl_a-0000", "prompt": "clean prompt"},
            {"id": "w-0001", "method": "tpl_a-0001", "prompt": _mojibake("doesn’t")},
            {"id": "w-0002", "method": "tpl_b-0002", "prompt": "孤立 鈥 标记", "harm_type": "weird"},
        ])
        dst = tmp_path / "cleaned" / "wildjailbreak.jsonl"
        stats = clean_file(src, dst)
        assert stats["total"] == 3
        assert stats["repaired"] == 3          # 三条各有操作（method×3 都有；mojibake×2）
        assert stats["mojibake"] == 2
        rows = _read(dst)
        assert rows[1]["prompt"] == "doesn’t"  # 修复落盘
        assert rows[0]["method"] == "tpl_a" and rows[0]["method_raw"] == "tpl_a-0000"
        assert rows[2]["harm_original"] == "weird"

    def test_clean_file_rejects_contract_violation(self, tmp_path):
        """清洗产物违反契约（如空 prompt）即抛错，不静默落盘。"""
        src = tmp_path / "bad.jsonl"
        _write(src, [{"id": "b-0000", "method": "m-0000", "prompt": "   "}])
        with pytest.raises(ValueError, match="违反契约"):
            clean_file(src, tmp_path / "cleaned" / "bad.jsonl")

    def test_rebuild_merged_keeps_ids_and_order(self, tmp_path):
        m1 = tmp_path / "a.jsonl"
        m2 = tmp_path / "b.jsonl"
        _write(m1, [{"id": "w-0000", "method": "t", "prompt": "p1"}])
        _write(m2, [{"id": "j-0000", "method": "t", "prompt": "p2"}])
        merged = tmp_path / "all_merged.jsonl"
        n = rebuild_merged([m1, m2], merged)
        assert n == 2
        rows = _read(merged)
        assert [r["id"] for r in rows] == ["w-0000", "j-0000"]  # 成员原 id、按成员序

    def test_rebuild_merged_conflict_raises(self, tmp_path):
        m1 = tmp_path / "a.jsonl"
        m2 = tmp_path / "b.jsonl"
        _write(m1, [{"id": "dup-0000", "method": "t", "prompt": "p1"}])
        _write(m2, [{"id": "dup-0000", "method": "t", "prompt": "p2"}])
        with pytest.raises(ValueError, match="id 冲突"):
            rebuild_merged([m1, m2], tmp_path / "all_merged.jsonl")
