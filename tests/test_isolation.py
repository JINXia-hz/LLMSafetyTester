"""core.isolation 测试：work-dir 全量重绑（单元化的基础）。

验证 rebind_to_workdir 把全部 9 个产物路径（含原 5 个泄漏点）都重绑到 work-dir，
使消费模块读到的是 work-dir 内的路径。
"""

from __future__ import annotations

import pytest


@pytest.fixture
def restore_modules():
    """测试后恢复所有被重绑的模块属性（isolation 是进程内副作用，须清理）。"""
    import llmsec.core.config as cfg
    import llmsec.core.results as res
    import llmsec.evaluation.predictors.blend as blend
    import llmsec.evaluation.predictors.fingerprint as fp
    import llmsec.evaluation.prescreen_ml as ps
    import llmsec.evaluation.safe_twin as st
    saved = {
        "res.RESULTS_FILE": res.RESULTS_FILE,
        "cfg.RESULTS_FILE": cfg.RESULTS_FILE,
        "cfg.ELO_CACHE_FILE": cfg.ELO_CACHE_FILE,
        "cfg.FEATURE_CACHE_FILE": cfg.FEATURE_CACHE_FILE,
        "cfg.CLUSTER_RESULT_FILE": cfg.CLUSTER_RESULT_FILE,
        "cfg.CLUSTER_REPORT_FILE": cfg.CLUSTER_REPORT_FILE,
        "cfg.PREDICTORS_DIR": cfg.PREDICTORS_DIR,
        "cfg.SAFE_TWINS_FILE": cfg.SAFE_TWINS_FILE,
        "cfg.TWIN_RESULT_FILE": cfg.TWIN_RESULT_FILE,
        "fp.PROBES_FILE": fp.PROBES_FILE,
        "blend.PREDICTORS_DIR": blend.PREDICTORS_DIR,
        "ps.MODEL_PATH": ps.MODEL_PATH,
        "st.SAFE_TWINS_FILE": st.SAFE_TWINS_FILE,
        "st.TWIN_RESULT_FILE": st.TWIN_RESULT_FILE,
    }
    yield
    res.RESULTS_FILE = saved["res.RESULTS_FILE"]
    cfg.RESULTS_FILE = saved["cfg.RESULTS_FILE"]
    cfg.ELO_CACHE_FILE = saved["cfg.ELO_CACHE_FILE"]
    cfg.FEATURE_CACHE_FILE = saved["cfg.FEATURE_CACHE_FILE"]
    cfg.CLUSTER_RESULT_FILE = saved["cfg.CLUSTER_RESULT_FILE"]
    cfg.CLUSTER_REPORT_FILE = saved["cfg.CLUSTER_REPORT_FILE"]
    cfg.PREDICTORS_DIR = saved["cfg.PREDICTORS_DIR"]
    cfg.SAFE_TWINS_FILE = saved["cfg.SAFE_TWINS_FILE"]
    cfg.TWIN_RESULT_FILE = saved["cfg.TWIN_RESULT_FILE"]
    fp.PROBES_FILE = saved["fp.PROBES_FILE"]
    blend.PREDICTORS_DIR = saved["blend.PREDICTORS_DIR"]
    ps.MODEL_PATH = saved["ps.MODEL_PATH"]
    st.SAFE_TWINS_FILE = saved["st.SAFE_TWINS_FILE"]
    st.TWIN_RESULT_FILE = saved["st.TWIN_RESULT_FILE"]


class TestRebindToWorkdir:
    def test_rebinds_all_9_paths_to_workdir(self, tmp_path, restore_modules):
        """重绑后，所有消费模块读到的路径都在 work-dir 下。"""
        import llmsec.core.config as cfg
        import llmsec.core.results as res
        import llmsec.evaluation.predictors.blend as blend
        import llmsec.evaluation.predictors.fingerprint as fp
        import llmsec.evaluation.prescreen_ml as ps
        import llmsec.evaluation.safe_twin as st
        from llmsec.core.isolation import rebind_to_workdir

        wd = tmp_path / "wd"
        rebind_to_workdir(wd)

        # 权威/派生存储
        assert wd / "results.json" == res.RESULTS_FILE
        assert wd / "results.json" == cfg.RESULTS_FILE
        assert wd / "elo_cache.json" == cfg.ELO_CACHE_FILE
        # 聚类/特征
        assert wd / "feature_cache.pkl" == cfg.FEATURE_CACHE_FILE
        assert wd / "cluster_result.pkl" == cfg.CLUSTER_RESULT_FILE
        assert wd / "cluster_report.json" == cfg.CLUSTER_REPORT_FILE
        # 预测器（config + blend 两处）
        assert wd / "predictors" == cfg.PREDICTORS_DIR
        assert wd / "predictors" == blend.PREDICTORS_DIR
        # 指纹（原漏洞 1）
        assert wd / "state" / "probes.json" == fp.PROBES_FILE
        # 预筛模型（原漏洞 2）
        assert wd / "state" / "prescreen_model.joblib" == ps.MODEL_PATH
        # 安全孪生（原漏洞 5）
        assert wd / "state" / "safe_twins.jsonl" == cfg.SAFE_TWINS_FILE
        assert wd / "state" / "safe_twins.jsonl" == st.SAFE_TWINS_FILE
        assert wd / "allergy_results.jsonl" == cfg.TWIN_RESULT_FILE
        assert wd / "allergy_results.jsonl" == st.TWIN_RESULT_FILE

    def test_creates_state_and_predictors_subdirs(self, tmp_path, restore_modules):
        from llmsec.core.isolation import rebind_to_workdir
        wd = tmp_path / "wd"
        rebind_to_workdir(wd)
        assert (wd / "state").is_dir()
        assert (wd / "predictors").is_dir()

    def test_no_path_remains_under_global_output(self, tmp_path, restore_modules):
        """重绑后，没有任何被重绑的路径仍指向全局 output（隔离的核心断言）。"""
        import llmsec.core.config as cfg
        import llmsec.core.results as res
        import llmsec.evaluation.predictors.blend as blend
        import llmsec.evaluation.predictors.fingerprint as fp
        import llmsec.evaluation.prescreen_ml as ps
        from llmsec.core.config import OUTPUT_DIR
        from llmsec.core.isolation import rebind_to_workdir

        rebind_to_workdir(tmp_path / "wd")

        # 所有重绑后的路径都不应在全局 OUTPUT_DIR 下（除非 work-dir 恰好在 output 下，这里 wd 在 tmp）
        wd = tmp_path / "wd"
        checked = [
            res.RESULTS_FILE, cfg.ELO_CACHE_FILE, cfg.FEATURE_CACHE_FILE,
            cfg.CLUSTER_RESULT_FILE, cfg.CLUSTER_REPORT_FILE, cfg.PREDICTORS_DIR,
            blend.PREDICTORS_DIR, fp.PROBES_FILE, ps.MODEL_PATH,
            cfg.SAFE_TWINS_FILE, cfg.TWIN_RESULT_FILE,
        ]
        for p in checked:
            # 路径要么在 wd 下，要么不在全局 OUTPUT_DIR 下
            assert str(wd) in str(p) or not str(p).startswith(str(OUTPUT_DIR)), \
                f"路径 {p} 仍指向全局 output（隔离漏洞）"
