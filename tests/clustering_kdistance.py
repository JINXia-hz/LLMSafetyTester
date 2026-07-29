#!/usr/bin/env python3
"""
离线验证：HDBSCAN 聚类主管线（post-test 设计）。

构造 3 类已知攻击（base64 编码 / rot13 编码 / 代码伪装），
验证：
1. 无评估数据时 HDBSCAN 仍能分出 ≥2 簇且噪声比 <50%
2. 有评估数据时（弱监督加权 + ANOVA）簇效验证显著
3. 所有簇（含小簇）都有非默认名称
注：write=False，不污染真实 artifacts。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from llmsec.clustering import (
    compute_method_reactions,
    extract_all_features,
    learn_supervised_weights,
    run_hdbscan_clustering,
)
from llmsec.clustering.space import build_feature_matrix


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

    # ---- 无 eval 数据 ----
    print("\n" + "=" * 60)
    print("🧬 HDBSCAN 聚类（无 eval 数据）")
    print("=" * 60)
    features, meta = extract_all_features(records, eval_results=[])
    pre_report = run_hdbscan_clustering(features, meta, write=False)

    n_clusters_pre = pre_report["n_clusters"]
    noise_ratio_pre = pre_report["hdbscan"]["noise_ratio"]

    print(f"  方法总数: {pre_report['method_count']}")
    print(f"  簇数: {n_clusters_pre}, 噪声比: {noise_ratio_pre:.2%}")
    print(f"  关键层 k*: {pre_report.get('chosen_k')} (top3 {pre_report.get('top_ks', [])})")

    # ---- 有 eval 数据（弱监督 + ANOVA）----
    print("\n" + "=" * 60)
    print("🏁 HDBSCAN 聚类（弱监督 + 簇效验证）")
    print("=" * 60)
    # 反应与类别强相关：b64 全部成功、code 全部失败、rot13 居中
    score_by_prefix = {"b64": 3.0, "rot13": 0.5, "code": -2.0}
    eval_results = [
        {
            "method": r["method"],
            "eval_score": score_by_prefix[r["method"].split("_")[0]],
            "judge_harmfulness": 1,
            "judge_specificity": 1,
            "judge_dangerousness": 1,
            "compliance_level": "D",
            "math_score": 1,
            "status": "refused",
            "response_preview": "x",
            "tokens_prompt": 50,
        }
        for r in records
    ]
    features_f, meta_f = extract_all_features(records, eval_results=eval_results)
    reactions = compute_method_reactions(eval_results)
    methods = sorted(features_f.keys())
    X = build_feature_matrix(features_f, methods)
    y = {m: reactions[m]["mean_score"] for m in methods}
    weights = learn_supervised_weights(X, methods, y)
    final_report = run_hdbscan_clustering(
        features_f, meta_f, feature_weights=weights, reactions=reactions, write=False,
    )

    rv = final_report.get("reaction_validation", {})
    print(f"  簇数: {final_report['n_clusters']}, 密度视图噪声比: {final_report['hdbscan']['noise_ratio']:.2%}")
    print(f"  簇效: {rv.get('verdict')} (p={rv.get('p_anova')}, eta²={rv.get('eta2')})")

    ok = True
    if n_clusters_pre < 2:
        print("❌ 无 eval 聚类失败: 簇数 < 2")
        ok = False
    if noise_ratio_pre >= 0.5:
        print("❌ 无 eval 聚类失败: 噪声比 >= 50%")
        ok = False
    if final_report["n_clusters"] < 2:
        print("❌ 弱监督聚类失败: 簇数 < 2")
        ok = False
    if not rv.get("available") or (rv["p_anova"] > 0.05 and rv["p_kruskal"] > 0.05):
        print("❌ 强相关反应下簇效应应显著")
        ok = False

    # 所有簇（含小簇）都必须有非默认名称
    names = final_report.get("cluster_names", {})
    labels = final_report.get("method_labels", {})
    cids = set(labels.values()) - {-1}
    unnamed = [c for c in cids if str(c) not in names]
    default_named = [c for c in cids if names.get(str(c)) == f"簇{c}"]
    if unnamed or default_named:
        print(f"❌ 存在未命名/默认名簇: unnamed={unnamed}, default={default_named}")
        ok = False

    if ok:
        print("\n✅ 离线验证通过")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
