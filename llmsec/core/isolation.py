"""core.isolation — work-dir 全量路径重绑（单元化的基础原语）。

把所有「会写到 output/ 全局」的路径常量集中重绑到一个 work-dir，使 runner 在
work-dir 模式下**真正隔离**。

r9/P3-4 之后的工作方式：全部消费方已迁移为 `_config.X` 调用期动态读
（新增顶层 `from llmsec.core.config import <路径常量>` 会被
tests/test_audit_r9_guard.py 的 AST 守卫拦截），因此本模块只需重绑
config 一处，不再维护"冻结模块逐个改属性"的清单。

调用：
    from llmsec.core.isolation import rebind_to_workdir
    rebind_to_workdir(Path("/path/to/workdir"))
"""

from __future__ import annotations

from pathlib import Path

from llmsec.core.logging import get_logger

logger = get_logger(__name__)


# work-dir 隔离实际重绑的路径常量集（单一来源）：
#  - isolation 重绑的就是这些；
#  - tests/test_audit_r9_guard 的冻结导入守卫只拦这个集合（PROJECT_ROOT/
#    OUTPUT_DIR 等静态锚点不拦——它们永不重绑，冻结导入无害）。
REBOUND_PATHS = frozenset({
    "CATALOG_DB",
    "CLUSTER_DIR", "FEATURE_CACHE_FILE", "CLUSTER_RESULT_FILE",
    "EMBEDDING_CACHE_FILE", "CLUSTER_REPORT_FILE", "CLUSTER_MATRIX_FILE",
    "PREDICTORS_DIR", "STATE_DIR", "SAFE_TWINS_FILE", "TWIN_RESULT_FILE",
    "LOG_FILE", "TASK_LOG_DIR",
})


def rebind_to_workdir(wd: Path) -> None:
    """把全部产物路径重绑到 work-dir（在调用进程内生效）。

    wd 下会创建 state/ 和 predictors/ 子目录。重绑后所有写操作落 work-dir，
    全局 output/ 零写入。

    r9/P3-4：消费方已全部迁移为 `_config.X` 调用期动态读（冻结导入守卫见
    tests/test_audit_r9_guard.py），本函数只重绑 config 一处——原先
    "冻结模块逐个改属性"的清单（blend/fingerprint/prescreen_ml/safe_twin/
    allergy_phase/clustering.pipeline/results 共 8 处）已删除。

    覆盖的路径（13 个 + 4 个子目录创建）：
      统一库:          CATALOG_DB（R 观测 + 目录登记 + control 表 + elo/probes，P7/P9）
      聚类/特征:      FEATURE_CACHE_FILE, CLUSTER_RESULT_FILE, CLUSTER_REPORT_FILE,
                      CLUSTER_MATRIX_FILE, EMBEDDING_CACHE_FILE
      预测器:         PREDICTORS_DIR
      指纹/预筛:      STATE_DIR（prescreen_model.joblib；指纹已表化，随 CATALOG_DB 走）
      安全孪生:       SAFE_TWINS_FILE, TWIN_RESULT_FILE
      日志:           LOG_FILE（含已挂载文件 handler 的切换；告警走 logger，P9 无独立文件）
      任务进度:       TASK_LOG_DIR（progress.jsonl 落 <wd>/tasks/——C-8：此前未重绑，
                      work-dir 子进程带 LLMSEC_TASK_ID 时进度会写进全局 output/tasks/，
                      违反"全局 output/ 零写入"承诺）
    """
    wd = Path(wd)
    state_dir = wd / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (wd / "predictors").mkdir(parents=True, exist_ok=True)
    logs_dir = wd / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    tasks_dir = wd / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    # 唯一重绑点：config 模块属性（全部消费方调用期动态读）
    import llmsec.core.config as _cfg
    _cfg.CATALOG_DB = wd / "catalog.db"
    _cfg.CLUSTER_DIR = wd / "cluster"
    _cfg.FEATURE_CACHE_FILE = wd / "cluster" / "feature_cache.pkl"
    _cfg.CLUSTER_RESULT_FILE = wd / "cluster" / "cluster_result.pkl"
    _cfg.EMBEDDING_CACHE_FILE = wd / "cluster" / "embedding_cache.pkl"
    _cfg.CLUSTER_REPORT_FILE = wd / "cluster" / "cluster_report.json"
    _cfg.CLUSTER_MATRIX_FILE = wd / "cluster" / "cluster_matrix.csv"
    _cfg.PREDICTORS_DIR = wd / "predictors"
    _cfg.STATE_DIR = state_dir
    _cfg.SAFE_TWINS_FILE = state_dir / "safe_twins.jsonl"
    _cfg.TWIN_RESULT_FILE = wd / "allergy_results.jsonl"
    _cfg.LOG_FILE = logs_dir / "llmsec.log"
    _cfg.TASK_LOG_DIR = tasks_dir

    # 日志文件 handler：get_logger 在 import 期已打开全局 output/logs 的句柄，
    # 重绑 config.LOG_FILE 不影响已挂载的 handler，必须显式切换。
    # 用户显式设置 LLMSEC_LOG_FILE 时尊重用户选择，不切换。
    import os as _os
    if _os.getenv("LLMSEC_LOG_FILE") is None:
        from llmsec.core.logging import rebind_log_file
        rebind_log_file(logs_dir / "llmsec.log")

    logger.info("🧪 work-dir 隔离已启用: %s（全局 output/ 零写入）", wd)
