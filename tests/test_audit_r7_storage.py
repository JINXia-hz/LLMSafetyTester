"""第 7 轮审计回归——存储与并发（M-4 / M-7 / L-4 / L-7）。

  - M-4: read_json 区分"内容损坏"与"瞬时 IO 错误"——PermissionError 不得被
    误判为 CorruptedFileError（否则完好 results.json 被备份成 .corrupt.bak
    并回退旧 .bak 数据）；ResultsMatrix.load 读路径与 save 共用文件锁。
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
    def _valid_matrix_file(self, path):
        from llmsec.core.results import ResultsMatrix

        mat = ResultsMatrix(units=["u1"], models=["m1"])
        mat.upsert("r1", "m1", 2.0, status="fully_compliant", extra={"unit": "u1"})
        mat.save(path)   # 首次落盘（此时无旧文件，不产生 .bak）
        mat.save(path)   # 二次落盘 → .bak 生成（load 的恢复源）
        return mat

    def test_corrupt_main_recovers_from_bak(self, tmp_path):
        """主文件损坏 → .corrupt.bak 取证 + 从 .bak 恢复（F1 行为保持）。"""
        from llmsec.core.results import ResultsMatrix

        main = tmp_path / "results.json"
        self._valid_matrix_file(main)
        main.write_text("{broken", encoding="utf-8")

        mat = ResultsMatrix.load(main)
        assert (tmp_path / "results.json.bak").exists()
        assert (tmp_path / "results.json.corrupt.bak").exists(), "残文件应被备份供取证"
        assert mat.n_for_model("m1") > 0, "应从 .bak 恢复出模型列"

    def test_permission_error_does_not_trigger_corrupt_bak(self, tmp_path, monkeypatch):
        """M-4 核心：瞬时/持续占用不得把完好文件判成损坏回退旧数据。"""
        import llmsec.core.io as io_mod
        from llmsec.core.results import ResultsMatrix

        main = tmp_path / "results.json"
        self._valid_matrix_file(main)
        # 先造一个"旧"备份（若误判会回退到这份旧数据）
        (tmp_path / "results.json.bak").write_text(
            '{"version": 2, "units": [], "models": [], "results": {}}', encoding="utf-8")

        real_open = builtins.open

        def denied_open(file, *a, **kw):
            if str(file) == str(main):
                raise PermissionError(5, "Access is denied (simulated)")
            return real_open(file, *a, **kw)

        monkeypatch.setattr(io_mod, "open", denied_open, raising=False)
        with pytest.raises(PermissionError):
            ResultsMatrix.load(main)
        assert not (tmp_path / "results.json.corrupt.bak").exists(), (
            "PermissionError 不是损坏：不得把完好的新数据备份成 .corrupt.bak 并回退旧 .bak")

    def test_load_takes_file_lock(self, tmp_path, monkeypatch):
        """load 读路径与 save 共用锁（防读-替换竞态）。"""
        import llmsec.core.results as results_mod
        from llmsec.core.results import ResultsMatrix

        main = tmp_path / "results.json"
        self._valid_matrix_file(main)

        real_lock = results_mod._file_lock
        lock_paths = []

        def spy_lock(filepath, *a, **kw):
            lock_paths.append(str(filepath))
            return real_lock(filepath, *a, **kw)

        monkeypatch.setattr(results_mod, "_file_lock", spy_lock)
        ResultsMatrix.load(main)
        assert any(p == str(main) for p in lock_paths), "load 应持锁读取"

    def test_load_inside_held_lock_is_instant(self, tmp_path):
        """r8/P0 回归：持锁临界区内嵌套 load()/save() 不得卡满锁超时。

        原 _file_lock 无线程内重入：merge/delete_runs 持锁调 load() 在 Windows
        上精确卡 10 秒并记伪 ERROR（msvcrt 锁按句柄生效，同线程二次抢锁必失败）。
        """
        import time

        from llmsec.core.results import ResultsMatrix, _file_lock

        main = tmp_path / "results.json"
        self._valid_matrix_file(main)

        t0 = time.time()
        with _file_lock(main, strict=True):
            R = ResultsMatrix.load(main)   # merge.py / runs.py 的真实调用形态
            assert R.n_for_model("m1") > 0
            R.save(main)                   # publish_tracker 的嵌套 save 形态
        dt = time.time() - t0
        assert dt < 2.0, f"锁内嵌套 load/save 必须即时（重入计数生效），实际 {dt:.2f}s"

    def test_file_lock_still_excludes_other_threads(self, tmp_path):
        """重入只对本线程生效：另一线程仍被锁排除（strict 超时抛 LockTimeout）。"""
        import threading
        import time

        from llmsec.core.results import LockTimeout, _file_lock

        target = tmp_path / "results.json"
        target.write_text("{}", encoding="utf-8")

        with _file_lock(target, strict=True):
            errs = []

            def rival():
                try:
                    with _file_lock(target, timeout=0.3, strict=True):
                        pass
                except LockTimeout:
                    errs.append("timeout")

            th = threading.Thread(target=rival)
            th.start()
            th.join(timeout=3)
            time.sleep(0.05)
        assert errs == ["timeout"], "跨线程互斥必须保持（重入不得退化为全局放行）"


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

class TestEloCacheRmwLock:
    def test_cache_commit_under_lock_and_both_entries_survive(self, tmp_path, monkeypatch):
        import llmsec.core.config as cfg
        import llmsec.evaluation.elo_access as ea
        from llmsec.core.results import ResultsMatrix

        monkeypatch.setattr(cfg, "RESULTS_FILE", tmp_path / "results.json")
        monkeypatch.setattr(cfg, "ELO_CACHE_FILE", tmp_path / "elo_cache.json")

        mat = ResultsMatrix(units=["u1", "u2"], models=["m1", "m2"])
        mat.upsert("r1", "m1", 2.0, status="fully_compliant", extra={"unit": "u1"})
        mat.upsert("r2", "m2", -1.0, status="refused", extra={"unit": "u2"})
        mat.save(tmp_path / "results.json")

        # 记录锁调用（确定性验证 L-4 的锁确实加上）
        real_lock = ea._file_lock
        lock_targets: list[str] = []

        def spy_lock(filepath, *a, **kw):
            lock_targets.append(str(filepath))
            return real_lock(filepath, *a, **kw)

        monkeypatch.setattr(ea, "_file_lock", spy_lock)

        ea.elo_state_for("m1")
        assert any(str(cfg.ELO_CACHE_FILE) in p for p in lock_targets), (
            "缓存 RMW 提交必须在 ELO_CACHE_FILE 文件锁内")

        # 功能面：两个模型先后派生，缓存里两条都在
        ea.elo_state_for("m2")
        cache = ea._load_cache()
        assert "m1" in cache and "m2" in cache


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
