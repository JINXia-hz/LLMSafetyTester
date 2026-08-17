"""core.isolation 测试：work-dir 全量重绑（单元化的基础）。

r9/P3-4 之后：全部消费方调用期动态读 `_config.X`，重绑只落在 config 一处；
派生路径（probes.json / prescreen_model.joblib）经各模块的 `_xxx()` 助手
动态派生。本文件验证 config 重绑完备 + 派生助手跟随。
"""

from __future__ import annotations

import pytest


@pytest.fixture
def restore_modules():
    """测试后恢复所有被重绑的 config 属性（isolation 是进程内副作用，须清理）。"""
    import llmsec.core.config as cfg
    keys = [
        "RESULTS_FILE", "FEATURE_CACHE_FILE",
        "CLUSTER_RESULT_FILE", "CLUSTER_REPORT_FILE", "CLUSTER_MATRIX_FILE",
        "EMBEDDING_CACHE_FILE", "PREDICTORS_DIR", "STATE_DIR",
        "SAFE_TWINS_FILE", "TWIN_RESULT_FILE", "LOG_FILE", "ALERTS_FILE",
    ]
    saved = {k: getattr(cfg, k) for k in keys}
    yield
    for k in keys:
        setattr(cfg, k, saved[k])


class TestRebindToWorkdir:
    def test_rebinds_all_paths_to_workdir(self, tmp_path, restore_modules):
        """重绑后，config 的全部产物路径都在 work-dir 下。"""
        import llmsec.core.config as cfg
        from llmsec.core.isolation import rebind_to_workdir

        wd = tmp_path / "wd"
        rebind_to_workdir(wd)

        # 权威/派生存储
        assert wd / "results.json" == cfg.RESULTS_FILE
        # 聚类/特征（2026-08 storage 重构：统一归 cluster/ 子目录）
        assert wd / "cluster" == cfg.CLUSTER_DIR
        assert wd / "cluster" / "feature_cache.pkl" == cfg.FEATURE_CACHE_FILE
        assert wd / "cluster" / "cluster_result.pkl" == cfg.CLUSTER_RESULT_FILE
        assert wd / "cluster" / "cluster_report.json" == cfg.CLUSTER_REPORT_FILE
        assert wd / "cluster" / "cluster_matrix.csv" == cfg.CLUSTER_MATRIX_FILE
        assert wd / "cluster" / "embedding_cache.pkl" == cfg.EMBEDDING_CACHE_FILE
        # 预测器 / 指纹 / 预筛
        assert wd / "predictors" == cfg.PREDICTORS_DIR
        assert wd / "state" == cfg.STATE_DIR
        # 安全孪生
        assert wd / "state" / "safe_twins.jsonl" == cfg.SAFE_TWINS_FILE
        assert wd / "allergy_results.jsonl" == cfg.TWIN_RESULT_FILE
        # 日志/告警
        assert wd / "logs" / "llmsec.log" == cfg.LOG_FILE
        assert wd / "alerts.jsonl" == cfg.ALERTS_FILE

    def test_derived_paths_follow_config(self, tmp_path, restore_modules):
        """派生路径助手调用期读 config——重绑后自动指向 work-dir。"""
        from llmsec.core.isolation import rebind_to_workdir
        from llmsec.evaluation.predictors.fingerprint import _probes_file
        from llmsec.evaluation.prescreen_ml import _model_path

        wd = tmp_path / "wd"
        rebind_to_workdir(wd)

        assert wd / "state" / "probes.json" == _probes_file()
        assert wd / "state" / "prescreen_model.joblib" == _model_path()

    def test_creates_state_and_predictors_subdirs(self, tmp_path, restore_modules):
        from llmsec.core.isolation import rebind_to_workdir
        wd = tmp_path / "wd"
        rebind_to_workdir(wd)
        assert (wd / "state").is_dir()
        assert (wd / "predictors").is_dir()

    def test_no_path_remains_under_global_output(self, tmp_path, restore_modules):
        """重绑后，没有任何被重绑的路径仍指向全局 output（隔离的核心断言）。"""
        import llmsec.core.config as cfg
        from llmsec.core.config import OUTPUT_DIR
        from llmsec.core.isolation import rebind_to_workdir

        rebind_to_workdir(tmp_path / "wd")

        wd = tmp_path / "wd"
        checked = [
            cfg.RESULTS_FILE, cfg.FEATURE_CACHE_FILE,
            cfg.CLUSTER_RESULT_FILE, cfg.CLUSTER_REPORT_FILE, cfg.PREDICTORS_DIR,
            cfg.SAFE_TWINS_FILE, cfg.TWIN_RESULT_FILE,
        ]
        for p in checked:
            # 路径要么在 wd 下，要么不在全局 OUTPUT_DIR 下
            assert str(wd) in str(p) or not str(p).startswith(str(OUTPUT_DIR)), \
                f"路径 {p} 仍指向全局 output（隔离漏洞）"
