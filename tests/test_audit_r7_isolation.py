"""第 7 轮审计回归——work-dir 隔离一致性（M-1 / M-2 / L-9）。

  - M-1: allergy_phase 的冻结导入 SAFE_TWINS_FILE 必须被重绑（读写不再分裂）。
  - M-2: CLUSTER_MATRIX_FILE 必须被重绑（config + clustering.pipeline 冻结导入），
         _export_matrix 不再穿透隔离写全局 output/。
  - L-9: 日志文件 handler 与告警文件落 work-dir（兑现"全局 output/ 零写入"）。
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest


@pytest.fixture
def restore_r7_isolation():
    """保存/恢复本轮涉及的全部被重绑属性 + root logger 的文件 handler。

    r9/P3-4：消费方已迁移动态读，cpipe/ap 模块不再持有路径常量——只恢复 config。
    """
    import llmsec.core.config as cfg

    saved = {
        "cfg.CLUSTER_MATRIX_FILE": cfg.CLUSTER_MATRIX_FILE,
        "cfg.LOG_FILE": cfg.LOG_FILE,
        "cfg.ALERTS_FILE": cfg.ALERTS_FILE,
        "cfg.CATALOG_DB": cfg.CATALOG_DB,
    }
    import llmsec.core.logging as logging_mod
    saved_root_configured = logging_mod._root_configured
    root = logging.getLogger("llmsec")
    saved_handlers = list(root.handlers)
    yield
    cfg.CLUSTER_MATRIX_FILE = saved["cfg.CLUSTER_MATRIX_FILE"]
    cfg.LOG_FILE = saved["cfg.LOG_FILE"]
    cfg.ALERTS_FILE = saved["cfg.ALERTS_FILE"]
    cfg.CATALOG_DB = saved["cfg.CATALOG_DB"]
    # 引擎缓存按库路径键控，重绑后旧卫星库引擎随手丢弃
    from llmsec.storage import db as storage_db
    storage_db.close()
    logging_mod._root_configured = saved_root_configured
    # 恢复 handler 快照（rebind_log_file 会关闭旧文件 handler 并挂新的）
    for h in list(root.handlers):
        if h not in saved_handlers:
            h.close()
            root.removeHandler(h)
    for h in saved_handlers:
        if h not in root.handlers:
            root.addHandler(h)


class TestRebindCoverage:
    def test_new_paths_rebound(self, tmp_path, restore_r7_isolation, monkeypatch):
        """M-1/M-2 语义在动态读模式下成立：矩阵/孪生路径经 config 重绑即全局生效。"""
        import llmsec.core.config as cfg
        from llmsec.core.isolation import rebind_to_workdir

        monkeypatch.delenv("LLMSEC_LOG_FILE", raising=False)
        wd = tmp_path / "wd"
        rebind_to_workdir(wd)

        assert wd / "cluster" / "cluster_matrix.csv" == cfg.CLUSTER_MATRIX_FILE
        assert wd / "state" / "safe_twins.jsonl" == cfg.SAFE_TWINS_FILE
        assert wd / "logs" / "llmsec.log" == cfg.LOG_FILE
        assert wd / "alerts.jsonl" == cfg.ALERTS_FILE
        assert wd / "catalog.db" == cfg.CATALOG_DB  # 目录库卫星化（storage 重构）

    def test_export_matrix_writes_workdir(self, tmp_path, restore_r7_isolation):
        """M-2 功能面：_export_matrix 的 CSV 落 work-dir，不穿透到全局路径。"""
        import llmsec.clustering.pipeline as cpipe
        from llmsec.core.isolation import rebind_to_workdir

        wd = tmp_path / "wd"
        rebind_to_workdir(wd)

        cpipe._export_matrix(
            {"m1": 0, "m2": 0},
            {"m1": {"textual": [0.1]}, "m2": {"textual": [0.2]}},
            {"method_names": ["m1", "m2"]},
        )
        assert (wd / "cluster" / "cluster_matrix.csv").exists(), "矩阵 CSV 必须落在 work-dir"

    def test_allergy_phase_reads_workdir_twins(self, tmp_path, restore_r7_isolation, monkeypatch):
        """M-1 功能面：Phase 2 预载读 work-dir 孪生库（命中缓存则不触发生成）。"""
        import llmsec.pipeline.allergy_phase as alp
        from llmsec.core.io import write_jsonl
        from llmsec.core.isolation import rebind_to_workdir
        from llmsec.evaluation.elo import ELOTracker

        wd = tmp_path / "wd"
        rebind_to_workdir(wd)

        # 预置 work-dir 孪生：method 键 = 排行榜里的 unit 名（unit 键空间）
        twin_file = wd / "state" / "safe_twins.jsonl"
        write_jsonl(str(twin_file), [{"method": "u1", "safe_prompt": "cached twin",
                                      "key_space": "unit"}])

        tracker = ELOTracker()
        tracker.update_round("def", [("u1", 3.0)])

        generated = []
        monkeypatch.setattr(alp, "generate_safe_twin",
                            lambda p, c: generated.append(p) or {"safe_prompt": "NEW"})
        monkeypatch.setattr(alp, "call_target",
                            lambda p: {"error": None, "content": "benign reply",
                                       "target_refused": False})
        monkeypatch.setattr(alp, "judge_allergic",
                            lambda judge, sp, c: (False, False,
                                                  {"compliance_level": "A", "is_refusal": False}))

        summary = alp.run_allergy_phase(
            {"u1": {"method": "u1", "prompt": "p1", "id": "u1"}},
            twin_client=None, judge=None, tracker=tracker,
            n_window=6, allergy_file=tmp_path / "allergy.json", defender_name="def",
        )
        assert summary["total_tested"] == 1
        assert not generated, "预载命中 work-dir 孪生库时不得重新生成"

    def test_log_file_and_alerts_in_workdir(self, tmp_path, restore_r7_isolation, monkeypatch):
        """L-9：日志文件 handler 与告警文件写 work-dir。"""
        from llmsec.core.isolation import rebind_to_workdir

        monkeypatch.delenv("LLMSEC_LOG_FILE", raising=False)
        wd = tmp_path / "wd"
        rebind_to_workdir(wd)

        # 日志：root 的文件 handler 已切到 work-dir
        root = logging.getLogger("llmsec")
        file_handlers = [h for h in root.handlers if isinstance(h, RotatingFileHandler)]
        assert file_handlers, "应存在文件 handler"
        assert all(Path(h.baseFilename).is_relative_to(wd) for h in file_handlers), (
            "文件 handler 必须全部指向 work-dir（import 期挂载的全局句柄要被替换）")
        root.info("r7 isolation log line")
        for h in file_handlers:
            h.flush()
        assert (wd / "logs" / "llmsec.log").exists()

        # 告警：_write_event_file 落 work-dir
        from llmsec.core import monitoring
        monitoring._write_event_file({"title": "r7", "level": "warning"})
        assert (wd / "alerts.jsonl").exists()

    def test_explicit_log_file_env_respected(self, tmp_path, restore_r7_isolation, monkeypatch):
        """用户显式设置 LLMSEC_LOG_FILE 时隔离不抢夺日志路径。"""
        from llmsec.core.isolation import rebind_to_workdir

        user_log = tmp_path / "user.log"
        monkeypatch.setenv("LLMSEC_LOG_FILE", str(user_log))
        # 模拟真实进程时序：env 先于 get_logger 存在（handler 按显式路径挂载）
        import llmsec.core.logging as logging_mod
        root = logging.getLogger("llmsec")
        for h in list(root.handlers):
            h.close()
            root.removeHandler(h)
        logging_mod._root_configured = False
        logging_mod.get_logger("llmsec.core.config")
        assert any(
            isinstance(h, RotatingFileHandler) and Path(h.baseFilename) == user_log
            for h in root.handlers
        ), "前置：显式 env 下 handler 应挂到用户路径"

        wd = tmp_path / "wd"
        rebind_to_workdir(wd)

        file_handlers = [h for h in root.handlers if isinstance(h, RotatingFileHandler)]
        assert file_handlers and all(
            Path(h.baseFilename) == user_log for h in file_handlers
        ), "显式 LLMSEC_LOG_FILE 应保持不被隔离切换"
