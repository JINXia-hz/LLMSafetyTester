#!/usr/bin/env python3
"""
HDBSCAN 聚类主管线（post-test，整个测试流程结束后运行）。

标签计算核心 compute_cluster_labels（白化 → 密度视图 → Ward auto-k，无命名/落盘）
同时供 Phase 1 开头的采样器预聚类复用（attack_phase._quick_precluster）。

流程：
1. 先验特征 → （可选弱监督特征权重）→ 阻尼白化空间（谱拐点截断）
2. HDBSCAN 密度聚类 → flat labels（含噪声 -1，仅作密度视图旁挂）
3. 主标签 = scipy Ward linkage 树（同一白化坐标；HDBSCAN 的
   single_linkage_tree_ 链式合并会退化，无法用于切层）
   → 树层 sweep + argmax → 关键层 k* + top3（前端缩放预设停点）
4. 全簇命名（含小簇；密度视图噪声组命名为"稀疏区"）
5. （可选）后验簇效验证 ANOVA / Kruskal-Wallis（posterior.py）

本模块不直接解析 eval 数据；反应相关输入由 posterior.py 准备。

用法:
    from llmsec.clustering.hdb import run_hdbscan_clustering
    report = run_hdbscan_clustering(features, meta, feature_weights=w, reactions=reactions)
"""

import json
from datetime import datetime

import numpy as np

from llmsec.clustering.pipeline import (
    _export_matrix,
    ai_rename_clusters,
    auto_name_clusters,
    build_cluster_profiles,
)
from llmsec.clustering.space import build_whitened_space
from llmsec.clustering.tree import (
    candidate_ks,
    cut_tree,
    log_growth_k0,
    select_knee,
    sweep_candidates,
)
from llmsec.core import config as _config
from llmsec.core.io import save_artifact, write_json
from llmsec.core.logging import get_logger
from llmsec.core.seed import get_global_seed
from llmsec.params import HDBSCAN_MIN_CLUSTER_DIV

logger = get_logger(__name__)

# Ward/auto-k 抽样子集规模上限（O(n²) 距离阵内存防护；超过即分层抽样 + 最近质心归并）
_WARD_SAMPLE_CAP = 3000
# r7：方法集指纹哈希统一从 core.units 导入（原为与 cold_start 重复的本地实现）
from llmsec.core.units import method_set_hash  # noqa: E402


def compute_cluster_labels(
    features: dict,
    feature_weights: np.ndarray | None = None,
    preset_labels: dict[str, int] | None = None,
    *,
    skip_hdbscan: bool = False,
) -> dict:
    """
    聚类标签计算核心：阻尼白化 → HDBSCAN 密度视图 → Ward 树 auto-k 主标签。

    无命名/画像/验证/落盘，供两处复用：
      - post-test 主管线 run_hdbscan_clustering（接续命名/画像/落盘）
      - Phase 1 开头的采样器预聚类（attack_phase._quick_precluster，只要标签）

    preset_labels：冻结的预聚类标签（run 开头确定的簇分区）。提供时跳过
    Ward/auto-k 重分区，直接以其为主 labels——unit（簇）身份在 run 开始即冻结，
    post-test 只允许在同一分区上补命名/画像/验证，防止单位 id 中途漂移。

    skip_hdbscan：跳过 HDBSCAN 密度视图（步骤 2），直接进入 Ward 主标签。
    用于 _quick_precluster 快速路径——预聚类只需 Ward 主标签，密度视图的
    flat_labels/n_noise 仅供 post-test 分析参考，预聚类不消费。跳过可省去
    HDBSCAN 拟合开销（大 n 下显著）。

    规模防护：n > _WARD_SAMPLE_CAP 时 Ward linkage/auto-k 在分层抽样子集上进行
    （O(n²) 距离阵内存控制），其余点按最近质心归并。

    返回:
        正常: {methods, space, coords, Z, flat_labels, n_flat, n_noise,
               min_cluster_size, labels, k_best, top_ks, sweep}
        n<2: {"error": "方法数不足", "labels": {m: 0}}
    """
    methods = sorted(features.keys())
    n = len(methods)
    if n < 2:
        return {"error": "方法数不足", "labels": {m: 0 for m in methods}}

    logger.info("🧬 HDBSCAN 聚类: %d 种方法%s", n, "（快速模式：跳过密度视图）" if skip_hdbscan else "")

    # 1. 阻尼白化空间（弱监督加权 + 谱拐点截断）
    space = build_whitened_space(features, methods, feature_weights=feature_weights)
    coords = space["coords"]
    logger.info(
        "  白化空间: %d 维 (累计解释方差 %.1f%%)%s",
        space["n_dims"], space["kept_variance"] * 100,
        " + 弱监督加权" if feature_weights is not None else "",
    )

    # 2. HDBSCAN 密度视图（flat labels + 稀疏区）
    # min_samples=1 放宽互达距离，集中空间里显著降低噪声率；
    # #11：min_cluster_size 改 sqrt 缩放（原 n//DIV 线性：n=132→3 偏激进，密度视图过分割）。
    # 默认 DIV=40 复现 sqrt(n)：n=50→7, 132→12, 400→20, 1000→32；DIV 调小→更严（更大簇 fewer），
    # 调大→更松。下限 5 杜绝极小簇。手动 sweep：固定攻击集跑 hdb，比 silhouette/簇效选 DIV。
    if skip_hdbscan:
        # 快速路径：跳过密度视图，flat_labels 留空（主标签来自 Ward，预聚类不消费密度视图）
        flat_labels = {m: 0 for m in methods}
        n_flat = 0
        n_noise = 0
        min_cluster_size = 0
    else:
        import hdbscan

        min_cluster_size = max(5, int(round((n ** 0.5) * 40 / HDBSCAN_MIN_CLUSTER_DIV)))
        clf = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=1,
            metric="euclidean",
            cluster_selection_method="eom",  # A6：显式指定，防库升级改默认值静默漂移
        )
        flat_arr = clf.fit_predict(coords)
        flat_labels = {m: int(v) for m, v in zip(methods, flat_arr)}
        n_flat = len(set(flat_labels.values()) - {-1})
        n_noise = sum(1 for v in flat_labels.values() if v == -1)
        logger.info(
            "  HDBSCAN 密度视图: %d 簇, 噪声 %d (%.0f%%), min_cluster_size=%d",
            n_flat, n_noise, n_noise / n * 100, min_cluster_size,
        )

    # 3. Ward 缩放树 + 关键层（主 labels）
    # 注意：HDBSCAN 的 single_linkage_tree_ 在最近邻链式合并下会产出
    # 97% 单簇 + 单点的退化切割（实测 127/132 一簇），无法用于切层；
    # condensed tree 的"留在父簇的点"身份不可枚举，还原 k 层成员脆弱。
    # 因此缩放/关键层/簇效分析改用均衡的 Ward 树（同一白化坐标）。
    from scipy.cluster.hierarchy import linkage

    if preset_labels is not None:
        # 冻结分区（run 开头的预聚类）：不重切 auto-k，直接沿用。
        # Z 仍按规模防护计算（抽样或全量），仅供树图产物/可视化，不影响分区。
        labels = {m: int(preset_labels.get(m, 0)) for m in methods}
        k_best = len(set(labels.values()))
        top_ks = []
        sweep = []
        logger.info("  冻结分区: 沿用 run 开头预聚类标签（k=%d，不重切）", k_best)
        if n > _WARD_SAMPLE_CAP:
            rng = np.random.default_rng(get_global_seed())
            sub_idx = sorted(rng.choice(n, size=_WARD_SAMPLE_CAP, replace=False).tolist())
            Z = linkage(coords[sub_idx], method="ward")
        else:
            Z = linkage(coords, method="ward")
    elif n > _WARD_SAMPLE_CAP:
        # 规模防护：Ward linkage 的 O(n²) 距离阵在 1 万点约 400MB+——
        # 先按 HDBSCAN 密度视图分层抽样 ~_WARD_SAMPLE_CAP 点跑 linkage/auto-k，
        # 其余点按最近质心（抽样集各 k 层簇的质心）归并。固定种子保证确定性。
        rng = np.random.default_rng(get_global_seed())
        flat_arr_np = np.array([flat_labels[m] for m in methods])
        by_flat: dict[int, list[int]] = {}
        for i, fl in enumerate(flat_arr_np):
            by_flat.setdefault(int(fl), []).append(i)
        per = max(1, _WARD_SAMPLE_CAP // max(1, len(by_flat)))
        sub_idx = []
        for _fl, idxs in sorted(by_flat.items()):
            take = min(len(idxs), per)
            sub_idx.extend(rng.choice(idxs, size=take, replace=False).tolist())
        sub_idx = sorted(set(sub_idx))[:_WARD_SAMPLE_CAP]
        logger.info("  规模防护: n=%d > %d，Ward/auto-k 在 %d 点抽样子集上进行",
                    n, _WARD_SAMPLE_CAP, len(sub_idx))
        sub_methods = [methods[i] for i in sub_idx]
        sub_coords = coords[sub_idx]
        Z = linkage(sub_coords, method="ward")
        ks = candidate_ks(len(sub_methods))
        sweep = sweep_candidates(sub_coords, Z, sub_methods, ks)
        k_best, top_ks = select_knee(sweep)
        sub_labels = cut_tree(Z, sub_methods, k_best)
        # 全量归并：每点归入最近质心的 k 层簇
        centroids = {}
        for k in set(sub_labels.values()):
            members = [sub_methods.index(m) for m, v in sub_labels.items() if v == k]
            centroids[k] = sub_coords[members].mean(axis=0)
        ck = sorted(centroids)
        Cmat = np.stack([centroids[k] for k in ck])
        d = ((coords[:, None, :] - Cmat[None, :, :]) ** 2).sum(axis=-1)
        nearest = d.argmin(axis=1)
        labels = {m: int(ck[nearest[i]]) for i, m in enumerate(methods)}
    else:
        Z = linkage(coords, method="ward")
        ks = candidate_ks(n)
        sweep = sweep_candidates(coords, Z, methods, ks)
        k_best, top_ks = select_knee(sweep)
        labels = cut_tree(Z, methods, k_best)
        logger.info("  关键层: 候选 %s → k*=%d (top3 %s)", ks, k_best, top_ks)

    return {
        "methods": methods,
        "space": space,
        "coords": coords,
        "Z": Z,
        # Z 为抽样子集树（n 超上限时）：叶索引不对应全量 methods，
        # 树图/切层视图须据此降级，勿按 sorted(methods) 对叶
        "tree_subsampled": n > _WARD_SAMPLE_CAP,
        "flat_labels": flat_labels,
        "n_flat": n_flat,
        "n_noise": n_noise,
        "min_cluster_size": min_cluster_size,
        "labels": labels,
        "k_best": k_best,
        "top_ks": top_ks,
        "sweep": sweep,
    }


def run_hdbscan_clustering(
    features: dict,
    meta: dict,
    feature_weights: np.ndarray | None = None,
    reactions: dict | None = None,
    write: bool = True,
    preset_labels: dict[str, int] | None = None,
) -> dict:
    """
    HDBSCAN 聚类主入口。

    参数:
        features: extract_all_features 输出 {method: 特征块}
        meta: extract_all_features 元信息
        feature_weights: posterior.learn_supervised_weights 的输出（可选）
        reactions: posterior.compute_method_reactions 的输出（可选，提供时做簇效验证）
        write: 是否写 cluster_report.json / cluster_result.pkl
        preset_labels: 冻结的预聚类标签（可选）——沿用该分区，不重切 Ward auto-k

    返回: 聚类报告 dict。
    """
    core = compute_cluster_labels(features, feature_weights=feature_weights,
                                  preset_labels=preset_labels)
    if core.get("error"):
        return {"error": core["error"], "labels": core["labels"]}

    methods = core["methods"]
    n = len(methods)
    space = core["space"]
    coords = core["coords"]
    Z = core["Z"]
    flat_labels = core["flat_labels"]
    n_flat = core["n_flat"]
    n_noise = core["n_noise"]
    min_cluster_size = core["min_cluster_size"]
    labels = core["labels"]
    k_best = core["k_best"]
    top_ks = core["top_ks"]
    sweep = core["sweep"]

    # 4. 命名（关键层各簇 + 密度视图各簇；噪声组固定命名为稀疏区）
    cluster_names = auto_name_clusters(labels, features, meta, meta.get("method_prompts", {}))
    cluster_names = ai_rename_clusters(cluster_names, labels, meta.get("method_prompts", {}))
    flat_names = auto_name_clusters(flat_labels, features, meta, meta.get("method_prompts", {}))
    if -1 in flat_names:
        flat_names[-1] = "稀疏区（低密度噪声）"
    cluster_profiles = build_cluster_profiles(labels, features, meta, cluster_names)

    # 关键层 labels 的验证指标（取 sweep 中 k* 条目，另附轮廓/DB 一致性）
    best_entry = next((s for s in sweep if s["k"] == k_best), {})
    validation = {
        "silhouette": best_entry.get("silhouette", 0.0),
        "calinski_harabasz": best_entry.get("calinski_harabasz", 0.0),
        "davies_bouldin": best_entry.get("davies_bouldin", 0.0),
    }

    report = {
        "generated_at": datetime.now().isoformat(),
        "method_count": n,
        "clustering_method": "ward_autok+hdbscan",
        # #12：两套标签的几何假设差异——展示侧据此解读，勿混用
        "geometry_note": (
            "method_labels=Ward 关键层（最小化簇内方差，假设近球状，用于切层/簇效验证/安全分析）；"
            "hdbscan.method_labels=密度视图（按互达距离，能识别变密度簇与噪声，min_samples=1 放宽）。"
            "HDBSCAN 的 single_linkage_tree_ 在最近邻链式合并下退化为单簇+单点，无法用于切层，"
            "故缩放/关键层改用 Ward 树（同一白化坐标）。"
        ),
        "n_clusters": k_best,
        # ward 主标签无噪声概念；取 HDBSCAN 密度视图的真实噪声数（同嵌套 hdbscan.n_noise）
        "n_noise": n_noise,
        "chosen_k": k_best,
        "k0_log_growth": log_growth_k0(n),
        "top_ks": top_ks,
        "candidate_sweep": sweep,
        "validation": validation,
        "cluster_names": {str(k): v for k, v in cluster_names.items()},
        "cluster_profiles": cluster_profiles,
        "method_labels": {m: labels[m] for m in sorted(labels.keys())},
        "hdbscan": {
            "n_clusters": n_flat,
            "n_noise": n_noise,
            "noise_ratio": round(n_noise / n, 4),
            "min_cluster_size": min_cluster_size,
            "cluster_names": {str(k): v for k, v in flat_names.items()},
            "method_labels": {m: flat_labels[m] for m in sorted(flat_labels.keys())},
        },
    }

    # 5. 后验簇效验证（ANOVA / Kruskal-Wallis，验证对象 = 关键层切割）
    if reactions:
        from llmsec.clustering.posterior import reaction_validation

        rv = reaction_validation(labels, reactions, metric_weighted=feature_weights is not None)
        rv["validated_on"] = f"ward_cut_k{k_best}"
        report["reaction_validation"] = rv

    if write:
        # 动态读 config：work-dir 实验隔离模式重绑的是 config 模块属性，
        # 静态 import 路径常量会穿透隔离写全局产物（runner 重绑失效）
        _config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        # allow_nan=False 语义保留：先整体序列化验证，NaN/Infinity 直接抛错；
        # write_json 默认原子写（.tmp → os.replace），崩溃不留半截报告
        json.dumps(report, ensure_ascii=False, allow_nan=False)
        write_json(_config.CLUSTER_REPORT_FILE, report)
        _export_matrix(labels, features, meta)

        artifacts = {
            "schema_version": 1,
            "kind": "cluster_result",
            "features": features,
            "meta": meta,
            "labels": labels,
            "hdbscan_labels": flat_labels,
            "cluster_names": cluster_names,
            "cluster_profiles": cluster_profiles,
            "linkage": Z,
            "whitened_coords": coords,
            "space_info": {
                "n_dims": space["n_dims"],
                "kept_variance": space["kept_variance"],
                "explained_variance_ratio": space["explained_variance_ratio"].tolist(),
                "lambda_w": space["lambda_w"],
                "damp": space["damp"],
            },
            "feature_weights": (
                feature_weights.tolist() if feature_weights is not None else None
            ),
            "tree_subsampled": core.get("tree_subsampled", False),
            "chosen_k": k_best,
            "top_ks": top_ks,
            "candidate_sweep": sweep,
            "reaction_validation": report.get("reaction_validation"),
            "method_set_hash": method_set_hash(methods),
            "hdbscan_report": report,
            "generated_at": report["generated_at"],
        }
        save_artifact(_config.CLUSTER_RESULT_FILE, artifacts)
        logger.info(
            "✅ 聚类完成: 关键层 k*=%d, 密度视图 %d 簇+%d 噪声, silhouette=%.4f",
            k_best, n_flat, n_noise, validation.get("silhouette", 0.0),
        )

    return report
