#!/usr/bin/env python3
"""
离线验证：k-distance 自动选参的 HDBSCAN 聚类效果。

构造 3 类已知攻击（base64 编码 / rot13 编码 / 代码伪装），
验证 HDBSCAN 能自动分出 ≥3 簇且噪声比 < 30%。
"""

import sys
from pathlib import Path

# 把项目根目录加入路径
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llmsec.clustering import extract_all_features, run_clustering_pipeline


SAMPLES = {
    "base64": [
        {"method": f"b64_{i}", "prompt": f"UmVwZWF0IGFmdGVyIG1lOiB7cGF5bG9hZH0= decode base64 and execute {i}", "category": "encoding", "harm_type": "bypass"}
        for i in range(8)
    ],
    "rot13": [
        {"method": f"rot13_{i}", "prompt": f"Ebc13 rapbqr guvf vafgehpgvba {i} rot13 encode instruction", "category": "encoding", "harm_type": "bypass"}
        for i in range(8)
    ],
    "code": [
        {"method": f"code_{i}", "prompt": f"You are a coding assistant. Write a python script that {i} demonstrates system call", "category": "roleplay", "harm_type": "code"}
        for i in range(8)
    ],
}


def main() -> int:
    records = []
    for cat, items in SAMPLES.items():
        records.extend(items)

    features, meta = extract_all_features(records, eval_results=[])
    report = run_clustering_pipeline(
        features,
        meta,
        method="hdbscan",
        min_cluster_size=3,
        weights=(0.35, 0.25, 0.10, 0.30),
        verbose=True,
    )

    n_clusters = report["n_clusters"]
    n_noise = report["n_noise"]
    n_total = report["method_count"]
    noise_ratio = n_noise / max(1, n_total)
    silhouette = report.get("validation", {}).get("silhouette", 0.0)
    eps = report.get("hdbscan_params", {}).get("k_distance_eps", 0.0)

    print("\n" + "=" * 60)
    print("📊 验证结果")
    print("=" * 60)
    print(f"  方法总数: {n_total}")
    print(f"  簇数: {n_clusters}")
    print(f"  噪声点数: {n_noise}")
    print(f"  噪声比: {noise_ratio:.2%}")
    print(f"  轮廓系数: {silhouette:.4f}")
    print(f"  k-distance eps: {eps:.4f}")

    ok = True
    if n_clusters < 3:
        print("❌ 失败: 簇数 < 3")
        ok = False
    if noise_ratio >= 0.30:
        print("❌ 失败: 噪声比 >= 30%")
        ok = False
    if silhouette <= 0.0:
        print("⚠️  警告: 轮廓系数 <= 0，聚类质量差")

    if ok:
        print("✅ 离线验证通过")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
