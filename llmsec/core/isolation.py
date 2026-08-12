"""core.isolation — work-dir 全量路径重绑（单元化的基础原语）。

把所有「会写到 output/ 全局」的路径常量集中重绑到一个 work-dir，使 runner 在
work-dir 模式下**真正隔离**（堵住 probes/prescreen/blend/cluster_report/safe_twins
等 5 个原隔离漏洞）。

为什么需要这个模块：
  - 消费路径的模块读法不一：有的动态读 ``config.X``（cold_start），有的 import 期冻结
    到自己模块命名空间（safe_twin/blend/fingerprint/prescreen_ml/results）。
  - 冻结的模块，重绑 ``config.X`` 不生效，必须改它自己的模块属性。
  - 集中在这里一次性重绑全部（含冻结模块），runner / experiments / 未来调用方复用同一份。

调用：
    from llmsec.core.isolation import rebind_to_workdir
    rebind_to_workdir(Path("/path/to/workdir"))
"""

from __future__ import annotations

from pathlib import Path

from llmsec.core.logging import get_logger

logger = get_logger(__name__)


def rebind_to_workdir(wd: Path) -> None:
    """把全部产物路径重绑到 work-dir（在调用进程内生效）。

    wd 下会创建 state/ 和 predictors/ 子目录。重绑后所有写操作落 work-dir，
    全局 output/ 零写入。

    覆盖的路径（10 个 + 2 个子目录创建）：
      权威/派生存储:  RESULTS_FILE, ELO_CACHE_FILE
      聚类/特征:      FEATURE_CACHE_FILE, CLUSTER_RESULT_FILE, CLUSTER_REPORT_FILE,
                      EMBEDDING_CACHE_FILE
      预测器:         PREDICTORS_DIR (config + blend 模块两处)
      指纹:           PROBES_FILE (fingerprint 模块)
      预筛模型:       MODEL_PATH (prescreen_ml 模块)
      安全孪生:       SAFE_TWINS_FILE, TWIN_RESULT_FILE
    """
    wd = Path(wd)
    state_dir = wd / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (wd / "predictors").mkdir(parents=True, exist_ok=True)

    # 1) config 模块属性（动态读 config 的消费方生效：cold_start/elo_access 等）
    import llmsec.core.config as _cfg
    import llmsec.core.results as _res
    _res.RESULTS_FILE = wd / "results.json"
    _cfg.RESULTS_FILE = wd / "results.json"
    _cfg.ELO_CACHE_FILE = wd / "elo_cache.json"
    _cfg.FEATURE_CACHE_FILE = wd / "feature_cache.pkl"
    _cfg.CLUSTER_RESULT_FILE = wd / "cluster_result.pkl"
    _cfg.EMBEDDING_CACHE_FILE = wd / "embedding_cache.pkl"
    _cfg.CLUSTER_REPORT_FILE = wd / "cluster_report.json"
    _cfg.PREDICTORS_DIR = wd / "predictors"
    _cfg.SAFE_TWINS_FILE = state_dir / "safe_twins.jsonl"
    _cfg.TWIN_RESULT_FILE = wd / "allergy_results.jsonl"

    # 2) import 期冻结的消费模块（必须改它们自己的模块属性）
    import llmsec.evaluation.predictors.blend as _blend
    import llmsec.evaluation.predictors.fingerprint as _fp
    import llmsec.evaluation.prescreen_ml as _ps
    import llmsec.evaluation.safe_twin as _st
    _fp.PROBES_FILE = state_dir / "probes.json"
    _blend.PREDICTORS_DIR = wd / "predictors"
    _ps.MODEL_PATH = state_dir / "prescreen_model.joblib"
    _st.SAFE_TWINS_FILE = state_dir / "safe_twins.jsonl"
    _st.TWIN_RESULT_FILE = wd / "allergy_results.jsonl"

    logger.info("🧪 work-dir 隔离已启用: %s（全局 output/ 零写入）", wd)
