"""第 7 轮审计回归——存储与并发（M-4 / M-7 / L-4 / L-7）。

  - M-4: read_json 区分"内容损坏"与"瞬时 IO 错误"——PermissionError 不得被
    误判为 CorruptedFileError。（原"load 与 save 共用文件锁"已随 P2 锁家族
    退役：R 走 SQLite 事务，elo_cache 表化。）
  - M-7: confirm._gc 必须持锁调用（锁外迭代 + 持锁 pop 并发会 RuntimeError）。
  - L-4: elo_cache 的 RMW 纳入文件锁。
  - L-7: write_json 的 tmp 名带 pid/tid（同进程并发写同一文件不互踩）。
"""

from __future__ import annotations

import builtins
import threading
import time

import pytest

# ============================================================
# M-4: read_json 异常分类 + ResultsMatrix.load
# ============================================================

class TestReadJsonClassification:
    def test_garbage_json_strict_raises_corrupted(self, tmp_path):
        from llmsec.core.io import CorruptedFileError, read_json

        f = tmp_path / "bad.json"
        f.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(CorruptedFileError):
            read_json(f, strict=True)

    def test_garbage_json_lenient_returns_default(self, tmp_path):
        from llmsec.core.io import read_json

        f = tmp_path / "bad.json"
        f.write_text("{not valid json", encoding="utf-8")
        assert read_json(f, default={"d": 1}) == {"d": 1}

    def test_transient_permission_error_retried_then_succeeds(self, tmp_path, monkeypatch):
        """瞬时 PermissionError（杀软/索引器占用）应退避重试后读到内容。"""
        import llmsec.core.io as io_mod

        f = tmp_path / "data.json"
        f.write_text('{"ok": true}', encoding="utf-8")

        real_open = builtins.open
        calls = {"n": 0}

        def flaky_open(file, *a, **kw):
            if str(file) == str(f):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise PermissionError(5, "Access is denied (simulated AV hold)")
            return real_open(file, *a, **kw)

        monkeypatch.setattr(io_mod, "open", flaky_open, raising=False)
        assert io_mod.read_json(f, strict=True) == {"ok": True}
        assert calls["n"] >= 2, "第一次失败后应重试"

    def test_persistent_permission_error_not_misdiagnosed(self, tmp_path, monkeypatch):
        """持续 PermissionError 上抛原异常（strict），不伪装成 CorruptedFileError。"""
        import llmsec.core.io as io_mod

        f = tmp_path / "data.json"
        f.write_text('{"ok": true}', encoding="utf-8")

        real_open = builtins.open

        def denied_open(file, *a, **kw):
            if str(file) == str(f):
                raise PermissionError(5, "Access is denied (simulated)")
            return real_open(file, *a, **kw)

        monkeypatch.setattr(io_mod, "open", denied_open, raising=False)
        with pytest.raises(PermissionError):
            io_mod.read_json(f, strict=True)


class TestResultsMatrixLoad:
    def test_load_takes_file_lock(self, tmp_path, monkeypatch):
        """阶段 2 语义：db 真相路径不再用文件锁——改为完整性快检。

        原 M-4"读路径与 save 共用锁（防读-替换竞态）"由 SQLite 事务接管；
        本测试改钉 quick_check：损坏的 db 文件 load 时显式失败（不静默返空）。
        """

        from llmsec.core.results import ResultsMatrix

        dbp = tmp_path / "catalog.db"
        dbp.write_bytes(b"this is definitely not a sqlite database")
        with pytest.raises(RuntimeError, match="完整性校验失败"):
            ResultsMatrix.load(dbp)


# ============================================================
# M-7: confirm._gc 持锁
# ============================================================

class TestConfirmGcThreadSafety:
    def test_concurrent_issue_confirm_no_crash(self, monkeypatch):
        """高频 issue/confirm 并发 + TTL 极短强制 GC 路径，不得抛异常。

        （r7 清理轮删除了无生产调用方的 peek，压测改用双 confirmer；
        _gc 的持锁路径经 issue/confirm 内部调用覆盖。）
        """
        import random

        import llmsec.mcp.confirm as confirm_mod

        monkeypatch.setattr(confirm_mod, "_TTL_SECONDS", 0.001)
        confirm_mod.clear()

        errors: list[Exception] = []
        stop = threading.Event()
        tokens: list[str] = []
        tk_lock = threading.Lock()

        def issuer():
            try:
                while not stop.is_set():
                    t = confirm_mod.issue("act", {"x": 1}, lambda: "done")
                    with tk_lock:
                        tokens.append(t)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        def confirmer(seed):
            try:
                rnd = random.Random(seed)
                while not stop.is_set():
                    with tk_lock:
                        t = rnd.choice(tokens) if tokens else "nope"
                    confirm_mod.confirm(t)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=issuer),
                   threading.Thread(target=confirmer, args=(7,)),
                   threading.Thread(target=confirmer, args=(8,))]
        for th in threads:
            th.start()
        time.sleep(1.0)
        stop.set()
        for th in threads:
            th.join(timeout=5)

        assert not errors, f"并发压测出现异常（M-7：_gc 锁外迭代竞态）: {errors[:3]}"
        confirm_mod.clear()


# ============================================================
# L-4: elo_cache RMW 加锁
# ============================================================

class TestEloCacheTable:
    def test_cache_rows_written_and_both_entries_survive(self, tmp_path, monkeypatch):
        """P2：elo_cache 表化——两个模型先后派生，两行都在（事务 upsert 取代锁 RMW）。"""
        import llmsec.core.config as cfg
        import llmsec.evaluation.elo_access as ea
        from llmsec.core.results import ResultsMatrix

        monkeypatch.setattr(cfg, "CATALOG_DB", tmp_path / "catalog.db")

        mat = ResultsMatrix(units=["u1", "u2"], models=["m1", "m2"])
        mat.upsert("r1", "m1", 2.0, status="fully_compliant", extra={"unit": "u1"})
        mat.upsert("r2", "m2", -1.0, status="refused", extra={"unit": "u2"})
        mat.save(cfg.CATALOG_DB)

        from llmsec.storage import rstore

        ea.elo_state_for("m1")
        ea.elo_state_for("m2")
        row1 = rstore.get_elo_cache("m1")
        row2 = rstore.get_elo_cache("m2")
        assert row1 is not None and row2 is not None
        assert "attacker_ratings" in row1[1]


# ============================================================
# L-7: write_json 并发安全 tmp 名
# ============================================================

class TestWriteJsonConcurrentTmp:
    def test_concurrent_writes_same_file_valid_json(self, tmp_path):
        """8 线程并发 write_json 同一文件，最终文件必须是合法 JSON。"""
        import json

        from llmsec.core.io import write_json

        target = tmp_path / "shared.json"
        write_json(target, {"init": True})

        def worker(tid):
            for i in range(25):
                write_json(target, {"tid": tid, "i": i})

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        with open(target, encoding="utf-8") as f:
            data = json.load(f)  # 解析失败即测试失败（半写/互踩）
        assert "tid" in data
        # 不应残留互踩产生的中间 tmp
        leftovers = [p.name for p in tmp_path.iterdir() if ".tmp." in p.name]
        assert not leftovers, f"残留 tmp 文件: {leftovers}"
