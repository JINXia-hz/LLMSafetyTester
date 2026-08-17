"""rstore（R 矩阵 SQLite 后端）阶段 2 回归测试。

核心承诺：并发写零丢失（H5/H6 场景——原文件锁 RMW 的丢更新路径）、
遗留 json 自迁移的幂等与防复活、指纹保真（elo_cache/predictor 缓存键稳定）、
删除列后不被旧 json 复活。
"""

from __future__ import annotations

import threading

import pytest

import llmsec.core.config as cfg
from llmsec.core.results import MatchResult, ResultsMatrix
from llmsec.storage import db as storage_db
from llmsec.storage import rstore


@pytest.fixture()
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "RESULTS_DB", tmp_path / "results.db")
    monkeypatch.setattr(cfg, "RESULTS_FILE", tmp_path / "results.json")
    storage_db.close()
    yield tmp_path
    storage_db.close()


def _mk_items(model: str, n: int, start: int = 0) -> list[MatchResult]:
    return [
        MatchResult(record=f"r{start + i}", model=model, eval_score=1.0 * (i + start),
                    status="fully_compliant", ts=i + 1, extra={"unit": f"u{i % 3}"})
        for i in range(n)
    ]


# ============================================================
# 并发写零丢失（换 SQLite 的根本理由）
# ============================================================

def test_concurrent_upserts_no_lost_updates(iso):
    """4 线程并发 publish 各 25 条（同模型不同 record）→ 全部 100 条在库。

    原实现是"文件锁 load→modify→save"：锁超时放行（publish 非 strict）时
    后写覆盖先写、静默丢观测（H5/H6）。事务化后各自串行提交，零丢失。
    """
    errors: list[Exception] = []

    def worker(w: int) -> None:
        try:
            for batch in range(5):
                rstore.upsert_observations(_mk_items(f"m{w}", 5, start=batch * 5))
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    R = ResultsMatrix.load(cfg.RESULTS_DB)
    for w in range(4):
        assert R.n_for_model(f"m{w}") == 25


def test_upsert_overwrite_same_record_keeps_value(iso):
    """同 (record, model) 重复 upsert：末值为准（旧 upsert 语义）。"""
    rstore.upsert_observations([MatchResult("r1", "m", 1.0, ts=1)])
    rstore.upsert_observations([MatchResult("r1", "m", 3.0, ts=2)])
    R = ResultsMatrix.load(cfg.RESULTS_DB)
    assert R.get("r1", "m").eval_score == 3.0
    assert R.n_for_model("m") == 1


# ============================================================
# 指纹与类型保真
# ============================================================

def test_ts_type_fidelity(iso):
    """int 5 与 float 5.0 在 column_payload 指纹里不同——类型必须逐位保真。"""
    mat = ResultsMatrix()
    mat.upsert("a", "m", 1.0, ts=5)          # int
    mat.upsert("b", "m", 2.0, ts=5.0)        # float
    mat.upsert("c", "m", 3.0, ts="t5")       # str
    mat.upsert("e", "m", 5.0, ts=5, extra={"unit": "u", "round": 2})
    fp_before = mat.column_payload("m", extra_fields=("unit", "round"))
    mat.save(cfg.RESULTS_DB)
    back = ResultsMatrix.load(cfg.RESULTS_DB)
    assert back.column_payload("m", extra_fields=("unit", "round")) == fp_before
    assert type(back.get("a", "m").ts) is int
    assert type(back.get("b", "m").ts) is float
    assert back.get("c", "m").ts == "t5"
    # ts=None 只经遗留 json 路径存在（upsert 对 None 自动编号）；from_store 保真
    m_none = MatchResult.from_store("z", "m", {"eval_score": 1.0})
    assert m_none.ts is None


def test_full_round_trip_fingerprint_equal(iso):
    """save→load 全量回路：逐模型指纹相等（elo_cache/predictor 键不漂移）。"""
    mat = ResultsMatrix(units=["u1", "u2"])
    mat.upsert("r1", "mA", 1.0, ts=1, status="x", extra={"unit": "u1", "round": 1})
    mat.upsert("r2", "mA", -2.0, ts=2, extra={"unit": "u2"})
    mat.upsert("r1", "mB", 0.0, ts=3, extra={"unit": "u1", "round": 4})
    fps = {m: mat.column_payload(m, extra_fields=("unit", "round")) for m in mat.all_models()}
    seqs = {m: [(r.record, r.eval_score, repr(r.ts)) for r in mat.ordered_results(m)]
            for m in mat.all_models()}
    mat.save(cfg.RESULTS_DB)
    back = ResultsMatrix.load(cfg.RESULTS_DB)
    for m in fps:
        assert back.column_payload(m, extra_fields=("unit", "round")) == fps[m]
        assert [(r.record, r.eval_score, repr(r.ts)) for r in back.ordered_results(m)] == seqs[m]
    assert back.all_units() == ["u1", "u2"]


# ============================================================
# 遗留 json 自迁移
# ============================================================

def test_no_implicit_json_migration(iso):
    """P3：隐式 json 自迁移已删——db 缺失时 load 返回空，旁边的 json 不再被吃。

    遗留 json 的显式工具：matrix_from_legacy_json（读）/ export_legacy_json（写）。
    """
    mat = ResultsMatrix()
    mat.upsert("r1", "m", 1.0, ts=1)
    from llmsec.core.io import write_json
    write_json(cfg.RESULTS_FILE, mat.to_store_dict())  # 手写遗留 json（db 仍缺）
    assert cfg.RESULTS_FILE.exists() and not cfg.RESULTS_DB.exists()

    R = ResultsMatrix.load()  # db 缺 → 空，不隐式导入
    assert R.all_models() == []

    # 显式读取：matrix_from_legacy_json
    R2 = rstore.matrix_from_legacy_json(cfg.RESULTS_FILE)
    assert R2.n_for_model("m") == 1


# ============================================================
# 损坏处置（替代 .bak 机器）
# ============================================================

def test_corrupt_db_raises_quick_check(iso):
    cfg.RESULTS_DB.write_bytes(b"not a sqlite file at all")
    with pytest.raises(RuntimeError, match="完整性校验失败"):
        ResultsMatrix.load(cfg.RESULTS_DB)


def test_backup_and_restore(iso):
    mat = ResultsMatrix()
    mat.upsert("r1", "m", 1.0, ts=1)
    mat.save(cfg.RESULTS_DB)
    dest = rstore.backup(iso / "backup.db")
    assert dest.exists()
    storage_db.close()  # 释放引擎句柄（Windows 上 unlink 需要）
    cfg.RESULTS_DB.unlink()
    restored = rstore.load_matrix(dest)
    assert restored.n_for_model("m") == 1
