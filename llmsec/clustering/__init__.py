"""
llmsec.clustering — 攻击方法聚类子包

  - features.py：5 维攻击特征提取（原根目录 features.py）
  - pipeline.py：复合距离聚类主流程（原根目录 clustering.py，
    模块改名以避免与子包名冲突）

常用符号再导出，方便 `from llmsec.clustering import run_clustering_pipeline, ...`。
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
    CLUSTER_ARTIFACTS_FILE,
    CLUSTER_FEATURES_FILE,
    CLUSTER_MATRIX_FILE,
    CLUSTER_REPORT_FILE,
    auto_name_clusters,
    build_cluster_profiles,
    build_composite_distance,
    compute_external_validation,
    cosine_distance_matrix,
    euclidean_distance_matrix,
    jaccard_distance_matrix,
    run_clustering_pipeline,
    run_dbscan,
    run_final_clustering,
    run_hdbscan,
    run_hierarchical,
    run_kmeans,
    run_pre_clustering,
)

__all__ = [
    # features
    "extract_all_features", "load_and_extract",
    "extract_textual_features", "extract_text_embeddings",
    "extract_technique_labels", "extract_intent_features",
    "extract_defense_features",
    "TEXTUAL_FEATURE_NAMES", "TECHNIQUE_LABELS", "INTENT_FEATURE_NAMES",
    "DEFENSE_FEATURE_NAMES", "CROSS_MODEL_FEATURE_NAMES",
    # pipeline
    "run_clustering_pipeline", "run_hdbscan", "run_kmeans", "run_hierarchical",
    "run_dbscan", "run_pre_clustering", "run_final_clustering",
    "build_composite_distance", "build_cluster_profiles",
    "auto_name_clusters", "compute_external_validation",
    "cosine_distance_matrix", "jaccard_distance_matrix",
    "euclidean_distance_matrix",
    "CLUSTER_REPORT_FILE", "CLUSTER_MATRIX_FILE", "CLUSTER_FEATURES_FILE",
    "CLUSTER_ARTIFACTS_FILE",
]
