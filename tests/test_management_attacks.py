"""attacks 导入通道与生成器契约自检测试（全离线，tmp_path 隔离 id 空间）。"""

import json

import pytest

from llmsec.attacks.base import ensure_contract


def _write(path, rows):
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def _read(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ============================================================
# ensure_contract：生成器自检
# ============================================================
class TestEnsureContract:
    def test_valid_entries_pass(self):
        ensure_contract([{"id": "a", "method": "m", "prompt": "p"}], where="t")

    def test_violation_raises_with_where(self):
        with pytest.raises(ValueError, match=r"generate\.py 1\.1\.1.*prompt"):
            ensure_contract(
                [{"id": "a", "method": "m", "prompt": ""}, {"id": "b", "method": "m", "prompt": "ok"}],
                where="generate.py 1.1.1",
            )


# ============================================================
# attacks import：校验 / 冲突 / dry-run / 追加
# ============================================================
class TestCmdImport:
    def _run(self, monkeypatch, tmp_path, rows, *, source="l1", yes=False, existing=None):
        """构造隔离的 ATTACKS_DIR 并执行 cmd_import。"""
        from llmsec.management import attacks as m

        attacks_dir = tmp_path / "attacks"
        attacks_dir.mkdir()
        if existing:
            _write(attacks_dir / "existing.jsonl", existing)
        src = tmp_path / "incoming.jsonl"
        _write(src, rows)
        monkeypatch.setattr("llmsec.core.config.ATTACKS_DIR", attacks_dir)
        return m.cmd_import(str(src), source, yes=yes), attacks_dir

    def test_dry_run_no_write(self, monkeypatch, tmp_path):
        rc, attacks_dir = self._run(monkeypatch, tmp_path, [
            {"id": "l1-9001", "method": "m", "prompt": "p", "harm_type": "hate"},
        ])
        assert rc == 0
        assert not (attacks_dir / "imported").exists() or not list((attacks_dir / "imported").glob("*.jsonl"))

    def test_yes_writes_with_source_registered(self, monkeypatch, tmp_path):
        rc, attacks_dir = self._run(monkeypatch, tmp_path, [
            {"id": "l1-9001", "method": "m", "prompt": "p"},
        ], yes=True)
        assert rc == 0
        rows = _read(attacks_dir / "imported" / "l1.jsonl")
        assert rows[0]["id"] == "l1-9001" and rows[0]["source"] == "l1"  # source 已登记

    def test_unknown_source_rejected(self, monkeypatch, tmp_path):
        rc, _ = self._run(monkeypatch, tmp_path, [
            {"id": "x-1", "method": "m", "prompt": "p"},
        ], source="mars")
        assert rc == 1

    def test_contract_violation_aborts(self, monkeypatch, tmp_path):
        rc, attacks_dir = self._run(monkeypatch, tmp_path, [
            {"id": "x-1", "method": "m", "prompt": "   "},   # 空白 prompt
        ], yes=True)
        assert rc == 1
        assert not (attacks_dir / "imported" / "l1.jsonl").exists()

    def test_id_conflict_with_existing_space(self, monkeypatch, tmp_path):
        rc, _ = self._run(monkeypatch, tmp_path, [
            {"id": "l1-0001", "method": "m", "prompt": "p"},
        ], yes=True, existing=[{"id": "l1-0001", "method": "m", "prompt": "old"}])
        assert rc == 1  # 与 attacks/ 现有 id 空间冲突，拒绝

    def test_reimport_same_id_rejected(self, monkeypatch, tmp_path):
        """同 id 二次导入被拒（首轮走落盘，二轮被 imported/ 空间冲突检测拦截）。"""
        rc, attacks_dir = self._run(monkeypatch, tmp_path, [
            {"id": "l1-9001", "method": "m", "prompt": "p"},
        ], yes=True)
        assert rc == 0
        from llmsec.management import attacks as m

        src2 = attacks_dir.parent / "incoming2.jsonl"
        _write(src2, [{"id": "l1-9001", "method": "m", "prompt": "p"}])
        rc2 = m.cmd_import(str(src2), "l1", yes=True)
        assert rc2 == 1
        assert len(_read(attacks_dir / "imported" / "l1.jsonl")) == 1
