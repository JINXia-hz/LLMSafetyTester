"""
llmsec.clustering — 攻击方法聚类子包

  - features.py：5 维攻击特征提取
  - space.py：阻尼白化（轻量马氏）特征空间 + 弱监督特征权重
  - hdb.py：HDBSCAN 聚类主管线（post-test）+ Ward 树关键层；
    compute_cluster_labels 为无命名/落盘的标签核心，供运行开始的预聚类复用
  - tree.py：层次树层选择工具（算法无关）
  - posterior.py：后验统计（机器反应 / 弱监督 / ANOVA 簇效验证）
  - pipeline.py：聚类工具（自动命名 / 画像 / 导出）

常用符号再导出，方便 `from llmsec.clustering import run_hdbscan_clustering, ...`。
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
from llmsec.clustering.hdb import compute_cluster_labels, run_hdbscan_clustering
from llmsec.clustering.pipeline import (
    auto_name_clusters,
    build_cluster_profiles,
)
from llmsec.clustering.posterior import (
    compute_method_reactions,
    learn_supervised_weights,
    reaction_validation,
)
from llmsec.clustering.space import build_feature_matrix, build_whitened_space
from llmsec.clustering.tree import (
    candidate_ks,
    cut_tree,
    log_growth_k0,
    select_knee,
    sweep_candidates,
)


def parse_cluster_id(cid) -> int:
    """把簇标签解析为 int，非法值（None/字符串/空）归一为 -1（噪声簇）。

    替代 samplers.py / cluster_analysis.py 各自重复的 `try: int(cid) except: -1`
    （M-36）。聚类产物 method_labels 的值理论上恒为 int，但 JSON 往返 / 手工编辑
    可能产生字符串，下游按 int 索引会 KeyError，故统一在此兜底。
    """
    try:
        return int(cid)
    except (TypeError, ValueError):
        return -1


__all__ = [
    # features
    "extract_all_features", "load_and_extract",
    "extract_textual_features", "extract_text_embeddings",
    "extract_technique_labels", "extract_intent_features",
    "extract_defense_features",
    "TEXTUAL_FEATURE_NAMES", "TECHNIQUE_LABELS", "INTENT_FEATURE_NAMES",
    "DEFENSE_FEATURE_NAMES", "CROSS_MODEL_FEATURE_NAMES",
    # hdb（主管线）
    "run_hdbscan_clustering", "compute_cluster_labels",
    # pipeline（工具）
    "auto_name_clusters", "build_cluster_profiles",
    # posterior
    "compute_method_reactions", "learn_supervised_weights", "reaction_validation",
    # space / tree
    "build_feature_matrix", "build_whitened_space",
    "cut_tree", "candidate_ks", "sweep_candidates", "select_knee", "log_growth_k0",
    # 簇 ID 解析
    "parse_cluster_id",
    # 路径常量
    "CLUSTER_REPORT_FILE", "CLUSTER_MATRIX_FILE",
    "FEATURE_CACHE_FILE", "CLUSTER_RESULT_FILE",
]
