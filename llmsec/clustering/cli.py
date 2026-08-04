#!/usr/bin/env python3
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
import os
import sys

from llmsec.core.config import OUTPUT_DIR
from llmsec.core.logging import setup_console
from llmsec.clustering import (
    CLUSTER_MATRIX_FILE,
    CLUSTER_REPORT_FILE,
    compute_method_reactions,
    learn_supervised_weights,
    load_and_extract,
    run_hdbscan_clustering,
)
from llmsec.clustering.space import build_feature_matrix
from llmsec.core.io import read_jsonl

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

    print(f"📂 加载数据: {args.input}")
    eval_results = []
    if args.result_file:
        eval_results = read_jsonl(args.result_file)
        print(f"📂 评估结果: {args.result_file} ({len(eval_results)} 条)")
    features, meta = load_and_extract(
        attack_file=args.input,
        result_file=args.result_file,
    )

    methods = meta["method_names"]
    print(f"   共 {len(methods)} 种攻击方法")

    feat_dims = {}
    for m in methods[:1]:
        for block_name, block_data in features[m].items():
            dim = len(block_data) if hasattr(block_data, "__len__") else 1
            feat_dims[block_name] = dim
    print(f"   特征维度: {feat_dims}")

    if args.dump_features:
        out_path = os.path.join(OUTPUT_DIR, "extracted_features.json")
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
        print(f"\n📁 特征导出: {out_path}")
        return

    # 弱监督权重（有评估结果时）
    weights = None
    reactions = None
    if eval_results:
        reactions = compute_method_reactions(eval_results)
        X = build_feature_matrix(features, methods)
        y = {m: reactions[m]["mean_score"] for m in methods if m in reactions}
        weights = learn_supervised_weights(X, methods, y)
        print(f"   弱监督: {len(y)} 个已测方法参与特征加权")

    print(f"\n⏳ HDBSCAN 聚类")
    report = run_hdbscan_clustering(
        features, meta, feature_weights=weights, reactions=reactions,
    )

    if "error" in report:
        print(f"\n❌ {report['error']}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"📊 聚类分析结果")
    print(f"{'='*60}")
    print(f"  簇数: {report['n_clusters']} (+ {report['n_noise']} 噪声)")
    print(f"  关键层: k*={report.get('chosen_k')} (top3 {report.get('top_ks', [])})")

    val = report.get("validation", {})
    print(f"  轮廓系数: {val.get('silhouette', 0):.4f}")
    print(f"  Calinski-Harabasz: {val.get('calinski_harabasz', 0):.2f}")
    print(f"  Davies-Bouldin: {val.get('davies_bouldin', 0):.4f}")

    rv = report.get("reaction_validation")
    if rv and rv.get("available"):
        print(f"\n  簇效验证: {rv['verdict']}")
        print(f"    p_anova={rv['p_anova']}, p_kruskal={rv['p_kruskal']}, "
              f"eta²={rv['eta2']}, ε²={rv['epsilon2']}")

    print(f"\n  簇命名:")
    for cid, name in sorted(report.get("cluster_names", {}).items(), key=lambda x: int(x[0])):
        tag = "🟡 稀疏区" if cid == "-1" else f"簇{cid}"
        members = [m for m, c in report.get("method_labels", {}).items() if str(c) == str(cid)]
        print(f"    {tag} ({len(members)} 种方法): {name}")
        if len(members) <= 8:
            print(f"      → {', '.join(members)}")

    print(f"\n  📁 报告: {CLUSTER_REPORT_FILE}")
    print(f"  📁 矩阵: {CLUSTER_MATRIX_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
