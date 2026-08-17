#!/usr/bin/env python3
from llmsec.core.logging import get_logger

"""
攻击聚类分析 CLI 入口（post-test 设计：聚类在测试流程结束后运行）

流程：features 提取 → （可选弱监督特征权重）→ 阻尼白化 → HDBSCAN 密度视图
→ Ward 树关键层 auto-k（主标签）→ 全簇命名 → ANOVA 簇效验证。

用法:
    python -m llmsec.clustering.cli                     # HDBSCAN 聚类
    python -m llmsec.clustering.cli --result-file X.jsonl  # 带评估结果（启用弱监督 + 簇效验证）
    python -m llmsec.clustering.cli --dump-features     # 仅导出特征（不聚类）
"""

import argparse
import json
import sys

from llmsec.clustering import (
    compute_method_reactions,
    learn_supervised_weights,
    load_and_extract,
    run_hdbscan_clustering,
)
from llmsec.clustering.space import build_feature_matrix
from llmsec.core import config as _config  # 路径调用期动态读（work-dir 隔离兼容）
from llmsec.core.config import OUTPUT_DIR
from llmsec.core.io import read_jsonl
from llmsec.core.logging import setup_console

logger = get_logger(__name__)
setup_console()


def main():
    parser = argparse.ArgumentParser(description="攻击方法聚类分析（HDBSCAN + 关键层 auto-k）")
    parser.add_argument("--input", type=str, default="attacks/l1.jsonl",
                        help="攻击集输入文件")
    parser.add_argument("--result-file", type=str, default=None,
                        help="评估结果文件（提供时启用弱监督加权与 ANOVA 簇效验证）")
    parser.add_argument("--dump-features", action="store_true",
                        help="仅提取特征并导出 JSON，不聚类")
    args = parser.parse_args()

    logger.info(f"📂 加载数据: {args.input}")
    # M-32：result_file 路径解析必须与 load_and_extract 一致（相对 OUTPUT_DIR）。原实现
    # cli 侧按 CWD 读、内部按 OUTPUT_DIR 读，传 runs/<ts>/attack_results.jsonl 相对路径时
    # cli 读到 0 条 → 弱监督加权与 ANOVA 静默失效，而 defense 特征却成功提取（两个数据视图）。
    eval_results = []
    resolved_result = args.result_file
    if args.result_file:
        from pathlib import Path as _Path

        rp = _Path(args.result_file)
        if not rp.is_absolute():
            rp = OUTPUT_DIR / args.result_file
        resolved_result = str(rp)
        eval_results = read_jsonl(resolved_result)
        logger.info(f"📂 评估结果: {resolved_result} ({len(eval_results)} 条)")
        if not eval_results:
            # 读回 0 条时弱监督加权与 ANOVA 簇效验证会静默失效——显式提醒，防"以为带了评估"
            logger.warning("  ⚠ 评估结果读回 0 条（文件缺失或全为空行）——弱监督加权与簇效验证将被跳过")
    features, meta = load_and_extract(
        attack_file=args.input,
        result_file=resolved_result,
    )

    methods = meta["method_names"]
    logger.info(f"   共 {len(methods)} 种攻击方法")

    feat_dims = {}
    for m in methods[:1]:
        for block_name, block_data in features[m].items():
            dim = len(block_data) if hasattr(block_data, "__len__") else 1
            feat_dims[block_name] = dim
    logger.info(f"   特征维度: {feat_dims}")

    if args.dump_features:
        out_path = OUTPUT_DIR / "extracted_features.json"
        serializable = {}
        for m in methods:
            serializable[m] = {}
            for block_name, block_data in features[m].items():
                if hasattr(block_data, "tolist"):
                    serializable[m][block_name] = block_data.tolist()
                else:
                    serializable[m][block_name] = list(block_data) if hasattr(block_data, "__iter__") else float(block_data)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2, allow_nan=False)
        logger.info(f"\n📁 特征导出: {out_path}")
        return

    # 弱监督权重（有评估结果时）
    weights = None
    reactions = None
    if eval_results:
        reactions = compute_method_reactions(eval_results)
        X = build_feature_matrix(features, methods)
        y = {m: reactions[m]["mean_score"] for m in methods if m in reactions}
        weights = learn_supervised_weights(X, methods, y)
        logger.info(f"   弱监督: {len(y)} 个已测方法参与特征加权")

    logger.info("\n⏳ HDBSCAN 聚类")
    report = run_hdbscan_clustering(
        features, meta, feature_weights=weights, reactions=reactions,
    )

    if "error" in report:
        logger.error(f"\n❌ {report['error']}")
        sys.exit(1)

    logger.info(f"\n{'='*60}")
    logger.info("📊 聚类分析结果")
    logger.info(f"{'='*60}")
    logger.info(f"  簇数: {report['n_clusters']} (+ {report['n_noise']} 噪声)")
    logger.info(f"  关键层: k*={report.get('chosen_k')} (top3 {report.get('top_ks', [])})")

    val = report.get("validation", {})
    logger.info(f"  轮廓系数: {val.get('silhouette', 0):.4f}")
    logger.info(f"  Calinski-Harabasz: {val.get('calinski_harabasz', 0):.2f}")
    logger.info(f"  Davies-Bouldin: {val.get('davies_bouldin', 0):.4f}")

    rv = report.get("reaction_validation")
    if rv and rv.get("available"):
        logger.info(f"\n  簇效验证: {rv['verdict']}")
        logger.info(f"    p_anova={rv['p_anova']}, p_kruskal={rv['p_kruskal']}, "
              f"eta²={rv['eta2']}, ε²={rv['epsilon2']}")

    logger.info("\n  簇命名:")
    for cid, name in sorted(report.get("cluster_names", {}).items(), key=lambda x: int(x[0])):
        tag = "🟡 稀疏区" if cid == "-1" else f"簇{cid}"
        members = [m for m, c in report.get("method_labels", {}).items() if str(c) == str(cid)]
        logger.info(f"    {tag} ({len(members)} 种方法): {name}")
        if len(members) <= 8:
            logger.info(f"      → {', '.join(members)}")

    logger.info(f"\n  📁 报告: {_config.CLUSTER_REPORT_FILE}")
    logger.info(f"  📁 矩阵: {_config.CLUSTER_MATRIX_FILE}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
