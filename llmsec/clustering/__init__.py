"""
llmsec.clustering — 攻击方法聚类子包

  - features.py：5 维攻击特征提取
  - space.py：阻尼白化（轻量马氏）特征空间
  - tree.py：Ward 层次树聚类 + 多指标拐点 auto-k + 递归 DBSCAN 密度视图
  - pipeline.py：聚类工具（DBSCAN / 自动命名 / 画像 / 导出）

常用符号再导出，方便 `from llmsec.clustering import run_tree_clustering, ...`。
"""

from llmsec.clustering.features import (
    CROSS_MODEL_FEATURE_NAMES,
    DEFENSE_FEATURE_NAMES,
    INTENT_FEATURE_NAMES,
    TECHNIQUE_LABELS,
    TEXTUAL_FEATURE_NAMES,
    extract_all_features,
    extract_defense_features,
    extract_intent_features,
    extract_technique_labels,
    extract_text_embeddings,
    extract_textual_features,
    load_and_extract,
)
from llmsec.clustering.pipeline import (
    auto_name_clusters,
    build_cluster_profiles,
    euclidean_distance_matrix,
    knee_eps,
    run_dbscan,
    run_dbscan_recursive,
)
from llmsec.clustering.space import build_whitened_space, transform_to_space
from llmsec.clustering.tree import (
    build_tree,
    candidate_ks,
    cut_tree,
    log_growth_k0,
    run_final_tree_clustering,
    run_tree_clustering,
    select_knee,
    sweep_candidates,
)
from llmsec.core.config import (
    CLUSTER_ARTIFACTS_FILE,
    CLUSTER_FEATURES_FILE,
    CLUSTER_MATRIX_FILE,
    CLUSTER_REPORT_FILE,
)

__all__ = [
    # features
    "extract_all_features", "load_and_extract",
    "extract_textual_features", "extract_text_embeddings",
    "extract_technique_labels", "extract_intent_features",
    "extract_defense_features",
    "TEXTUAL_FEATURE_NAMES", "TECHNIQUE_LABELS", "INTENT_FEATURE_NAMES",
    "DEFENSE_FEATURE_NAMES", "CROSS_MODEL_FEATURE_NAMES",
    # pipeline（工具）
    "run_dbscan", "run_dbscan_recursive", "knee_eps",
    "euclidean_distance_matrix",
    "auto_name_clusters", "build_cluster_profiles",
    # space / tree
    "build_whitened_space", "transform_to_space",
    "run_tree_clustering", "run_final_tree_clustering",
    "build_tree", "cut_tree",
    "candidate_ks", "sweep_candidates", "select_knee", "log_growth_k0",
    # 路径常量
    "CLUSTER_REPORT_FILE", "CLUSTER_MATRIX_FILE", "CLUSTER_FEATURES_FILE",
    "CLUSTER_ARTIFACTS_FILE",
]
