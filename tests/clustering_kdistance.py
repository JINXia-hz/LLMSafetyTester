#!/usr/bin/env python3
"""
离线验证：前置树聚类（Ward + 拐点 auto-k）与最终聚类（树 + 递归 DBSCAN）。

构造 3 类已知攻击（base64 编码 / rot13 编码 / 代码伪装），
验证：
1. 前置树聚类在无 defense 特征时仍能分出 ≥3 簇
2. 最终聚类能分出 ≥3 簇，且 DBSCAN 密度视图所有簇（含小簇）都有名称
注：write=False，不污染真实 artifacts。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from llmsec.clustering import (
    extract_all_features,
    run_final_tree_clustering,
    run_tree_clustering,
)


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

    # ---- 前置树聚类（无 eval 数据）----
    print("\n" + "=" * 60)
    print("🌲 前置树聚类（Ward + 拐点 auto-k，无 defense 特征）")
    print("=" * 60)
    features, meta = extract_all_features(records, eval_results=[])
    pre_report = run_tree_clustering(features, meta, write=False)

    n_clusters_pre = pre_report["n_clusters"]
    silhouette_pre = pre_report.get("validation", {}).get("silhouette", 0.0)

    print(f"  方法总数: {pre_report['method_count']}")
    print(f"  auto-k: {n_clusters_pre} (top3 {pre_report.get('top_ks', [])})")
    print(f"  轮廓系数: {silhouette_pre:.4f}")

    # ---- 最终聚类（有 eval 数据，树 + 递归 DBSCAN）----
    print("\n" + "=" * 60)
    print("🏁 最终聚类（树 + 递归 DBSCAN 密度视图）")
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
    final_report = run_final_tree_clustering(features_final, meta_final, write=False)

    n_clusters_final = final_report["n_clusters"]
    silhouette_final = final_report.get("validation", {}).get("silhouette", 0.0)
    dbscan = final_report.get("dbscan", {})

    print(f"  方法总数: {final_report['method_count']}")
    print(f"  auto-k: {n_clusters_final}")
    print(f"  DBSCAN: 核心簇 {dbscan.get('n_core_clusters', 0)}, 噪声 {dbscan.get('n_noise', 0)}")
    print(f"  轮廓系数: {silhouette_final:.4f}")

    ok = True
    if n_clusters_pre < 3:
        print("❌ 前置树聚类失败: 簇数 < 3")
        ok = False
    if n_clusters_final < 3:
        print("❌ 最终聚类失败: 簇数 < 3")
        ok = False

    # 点5：DBSCAN 所有簇（含小簇）都必须有非默认名称
    dbscan_names = dbscan.get("cluster_names", {})
    dbscan_labels = dbscan.get("method_labels", {})
    dbscan_cids = set(dbscan_labels.values()) - {-1}
    unnamed = [cid for cid in dbscan_cids if str(cid) not in dbscan_names]
    default_named = [
        cid for cid in dbscan_cids
        if str(cid) in dbscan_names and dbscan_names[str(cid)] == f"簇{cid}"
    ]
    if unnamed or default_named:
        print(f"❌ DBSCAN 存在未命名/默认名小簇: unnamed={unnamed}, default={default_named}")
        ok = False

    if ok:
        print("\n✅ 离线验证通过")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
