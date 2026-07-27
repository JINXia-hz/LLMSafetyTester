#!/usr/bin/env python3
"""
攻击聚类分析 CLI 入口

流程：features 提取 5 维特征 → space 阻尼白化 → tree Ward 层次聚类 + 拐点 auto-k。
--final 时附加白化空间递归 DBSCAN 密度视图（含小簇命名）。

用法:
    python -m llmsec.clustering.cli                     # 树聚类（auto-k）
    python -m llmsec.clustering.cli --final             # 树聚类 + DBSCAN 密度视图
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
    load_and_extract,
    run_final_tree_clustering,
    run_tree_clustering,
)

setup_console()


def main():
    parser = argparse.ArgumentParser(description="攻击方法聚类分析（树聚类 + 拐点 auto-k）")
    parser.add_argument("--final", action="store_true",
                        help="附加递归 DBSCAN 密度视图（最终聚类，含小簇命名）")
    parser.add_argument("--input", type=str, default="攻击集_L1.jsonl",
                        help="攻击集输入文件")
    parser.add_argument("--result-file", type=str, default=None,
                        help="评估结果文件 (默认自动查找)")
    parser.add_argument("--dump-features", action="store_true",
                        help="仅提取特征并导出 JSON，不聚类")
    args = parser.parse_args()

    print(f"📂 加载数据: {args.input}")
    features, meta = load_and_extract(
        attack_file=args.input,
        result_file=args.result_file,
    )

    methods = meta["method_names"]
    print(f"   共 {len(methods)} 种攻击方法")
    if meta["has_eval_data"]:
        print(f"   含评估数据: 是 (防御交互特征仅用于画像，不进入度量)")
    else:
        print(f"   含评估数据: 否")

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
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        print(f"\n📁 特征导出: {out_path}")
        return

    print(f"\n⏳ 树聚类（Ward + 拐点 auto-k{' + 递归DBSCAN' if args.final else ''}）")
    if args.final:
        report = run_final_tree_clustering(features, meta)
    else:
        report = run_tree_clustering(features, meta)

    if "error" in report:
        print(f"\n❌ {report['error']}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"📊 聚类分析结果")
    print(f"{'='*60}")
    print(f"  簇数: {report['n_clusters']} (top3 候选: {report.get('top_ks', [])})")

    val = report.get("validation", {})
    print(f"  轮廓系数: {val.get('silhouette', 0):.4f}")
    print(f"  Calinski-Harabasz: {val.get('calinski_harabasz', 0):.2f}")
    print(f"  Davies-Bouldin: {val.get('davies_bouldin', 0):.4f}")

    print(f"\n  簇命名:")
    for cid, name in sorted(report.get("cluster_names", {}).items(), key=lambda x: int(x[0])):
        members = [m for m, c in report.get("method_labels", {}).items() if str(c) == str(cid)]
        print(f"    簇{cid} ({len(members)} 种方法): {name}")
        if len(members) <= 8:
            print(f"      → {', '.join(members)}")

    dbscan = report.get("dbscan")
    if dbscan:
        print(f"\n  DBSCAN 密度视图: 核心簇 {dbscan['n_core_clusters']} 个, 噪声 {dbscan['n_noise']} 点")

    print(f"\n  📁 报告: {CLUSTER_REPORT_FILE}")
    print(f"  📁 矩阵: {CLUSTER_MATRIX_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
