#!/usr/bin/env python3
"""
离线验证：预聚类（Agglomerative）与最终聚类（DBSCAN + Agglomerative）。

构造 3 类已知攻击（base64 编码 / rot13 编码 / 代码伪装），
验证：
1. 预聚类在无 defense 特征时仍能分出 ≥3 簇且噪声比 <30%
2. 最终聚类能分出 ≥3 簇
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llmsec.clustering import extract_all_features, run_final_clustering, run_pre_clustering


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

    # ---- 预聚类（无 eval 数据）----
    print("\n" + "=" * 60)
    print("🧊 预聚类（Agglomerative，无 defense 特征）")
    print("=" * 60)
    features, meta = extract_all_features(records, eval_results=[])
    pre_report = run_pre_clustering(features, meta, weights=(0.35, 0.25, 0.10, 0.30))

    n_clusters_pre = pre_report["n_clusters"]
    n_noise_pre = pre_report["n_noise"]
    noise_ratio_pre = n_noise_pre / max(1, pre_report["method_count"])
    silhouette_pre = pre_report.get("validation", {}).get("silhouette", 0.0)

    print(f"  方法总数: {pre_report['method_count']}")
    print(f"  目标簇数: {pre_report['target_k']}")
    print(f"  实际簇数: {n_clusters_pre}")
    print(f"  噪声点数: {n_noise_pre}")
    print(f"  噪声比: {noise_ratio_pre:.2%}")
    print(f"  轮廓系数: {silhouette_pre:.4f}")

    # ---- 最终聚类（有 eval 数据）----
    print("\n" + "=" * 60)
    print("🏁 最终聚类（DBSCAN + Agglomerative）")
    print("=" * 60)
    eval_results = [
        {
            "method": r["method"],
            "judge_harmfulness": 1,
            "judge_specificity": 1,
            "judge_dangerousness": 1,
            "compliance_level": "D",
            "math_score": 1,
            "status": "refused",
            "response_preview": "refused",
            "tokens_prompt": 50,
        }
        for r in records
    ]
    features_final, meta_final = extract_all_features(records, eval_results=eval_results)
    final_report = run_final_clustering(features_final, meta_final, weights=(0.35, 0.25, 0.10, 0.30))

    n_clusters_final = final_report["n_clusters"]
    n_noise_final = final_report["n_noise"]
    noise_ratio_final = n_noise_final / max(1, final_report["method_count"])
    silhouette_final = final_report.get("validation", {}).get("silhouette", 0.0)

    print(f"  方法总数: {final_report['method_count']}")
    print(f"  目标簇数: {final_report['target_k']}")
    print(f"  实际簇数: {n_clusters_final}")
    print(f"  噪声点数: {n_noise_final}")
    print(f"  噪声比: {noise_ratio_final:.2%}")
    print(f"  轮廓系数: {silhouette_final:.4f}")

    ok = True
    if n_clusters_pre < 3:
        print("❌ 预聚类失败: 簇数 < 3")
        ok = False
    if noise_ratio_pre >= 0.30:
        print("❌ 预聚类失败: 噪声比 >= 30%")
        ok = False
    if n_clusters_final < 3:
        print("❌ 最终聚类失败: 簇数 < 3")
        ok = False

    if ok:
        print("\n✅ 离线验证通过")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
