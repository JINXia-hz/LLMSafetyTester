#!/usr/bin/env python3
"""
攻击聚类分析 CLI 入口（原根目录 cluster.py 的 CLI 逻辑）

调用 llmsec.clustering.features 提取 5 维特征 + llmsec.clustering.pipeline 进行复合距离聚类。

聚类维度：
  维1: 文本结构与语义 (Textual + Embedding) → 余弦距离
  维2: 攻击技术多标签 (Technique Labels)         → Jaccard 距离
  维3: 意图与对抗强度 (Intent)                   → 欧氏距离
  维4: 防御交互行为 (Defense Interaction)        → 欧氏距离
  维5: 跨模型指纹 (占位)

聚类方法:
  hdbscan (默认) — 自动选簇 + 噪声点识别
  kmeans   — 传统 K-Means，需指定 --k
  hierarchical — 层次聚类

用法:
    python cluster.py                              # HDBSCAN 自动聚类
    python cluster.py --method kmeans --k 5        # K-Means K=5
    python cluster.py --method hierarchical --k 6  # 层次聚类
    python cluster.py --weights 0.4,0.2,0.2,0.2   # 自定义距离权重
    python cluster.py --dump-features              # 仅导出特征（不聚类）
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
    run_clustering_pipeline,
)

setup_console()


def parse_weights(s: str) -> tuple[float, float, float, float]:
    """解析 "0.35,0.25,0.10,0.30" 格式的权重字符串。"""
    parts = s.split(",")
    if len(parts) != 4:
        raise ValueError("权重必须为 4 个逗号分隔的浮点数")
    w = [float(p) for p in parts]
    # 归一化
    total = sum(w)
    if total <= 0:
        raise ValueError("权重之和必须 > 0")
    return tuple(x / total for x in w)


def main():
    parser = argparse.ArgumentParser(description="攻击方法聚类分析")
    parser.add_argument("--method", type=str, default="hdbscan",
                        choices=["hdbscan", "kmeans", "hierarchical"],
                        help="聚类算法 (默认 hdbscan)")
    parser.add_argument("--k", type=int, default=None,
                        help="簇数 (kmeans/hierarchical 使用，hdbscan 忽略)")
    parser.add_argument("--min-cluster-size", type=int, default=3,
                        help="HDBSCAN 最小簇大小 (默认 3)")
    parser.add_argument("--weights", type=str, default="0.35,0.25,0.10,0.30",
                        help="复合距离权重: emb,tech,intent,defense (默认 0.35,0.25,0.10,0.30)")
    parser.add_argument("--input", type=str, default="攻击集_L1.jsonl",
                        help="攻击集输入文件")
    parser.add_argument("--result-file", type=str, default=None,
                        help="评估结果文件 (默认自动查找)")
    parser.add_argument("--dump-features", action="store_true",
                        help="仅提取特征并导出 JSON，不聚类")
    args = parser.parse_args()

    # 解析权重
    try:
        weights = parse_weights(args.weights)
    except ValueError as e:
        print(f"❌ 权重格式错误: {e}")
        sys.exit(1)

    print(f"📂 加载数据: {args.input}")
    features, meta = load_and_extract(
        attack_file=args.input,
        result_file=args.result_file,
    )

    methods = meta["method_names"]
    print(f"   共 {len(methods)} 种攻击方法")
    if meta["has_eval_data"]:
        print(f"   含评估数据: 是 (防御交互特征启用)")
    else:
        print(f"   含评估数据: 否 (防御交互特征为零)")

    # 输出特征维度
    feat_dims = {}
    for m in methods[:1]:
        for block_name, block_data in features[m].items():
            dim = len(block_data) if hasattr(block_data, "__len__") else 1
            feat_dims[block_name] = dim
    print(f"   特征维度: {feat_dims}")

    if args.dump_features:
        # 仅导出特征
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

    # 运行聚类
    print(f"\n⏳ 聚类算法: {args.method.upper()}")
    print(f"   权重: emb={weights[0]:.2f} tech={weights[1]:.2f} "
          f"intent={weights[2]:.2f} defense={weights[3]:.2f}")

    report = run_clustering_pipeline(
        features, meta,
        method=args.method,
        k=args.k,
        min_cluster_size=args.min_cluster_size,
        weights=weights,
        verbose=True,
    )

    if "error" in report:
        print(f"\n❌ {report['error']}")
        sys.exit(1)

    # 终端摘要
    print(f"\n{'='*60}")
    print(f"📊 聚类分析结果")
    print(f"{'='*60}")
    print(f"  方法: {args.method.upper()}")
    print(f"  簇数: {report['n_clusters']} (+ {report['n_noise']} 噪声)")

    val = report.get("validation", {})
    print(f"  轮廓系数: {val.get('silhouette', 0):.4f}")
    print(f"  Davies-Bouldin: {val.get('davies_bouldin', 0):.4f}")
    print(f"  NMI (vs 人工分类): {val.get('nmi', 0):.4f}")
    print(f"  ARI (vs 人工分类): {val.get('ari', 0):.4f}")

    print(f"\n  簇命名:")
    for cid, name in sorted(report.get("cluster_names", {}).items()):
        tag = "🟡 噪声" if cid == -1 else f"簇{cid}"
        members = [m for m, c in report.get("method_labels", {}).items() if c == cid]
        print(f"    {tag} ({len(members)} 种方法): {name}")
        if len(members) <= 8:
            print(f"      → {', '.join(members)}")

    print(f"\n  📁 报告: {CLUSTER_REPORT_FILE}")
    print(f"  📁 矩阵: {CLUSTER_MATRIX_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
