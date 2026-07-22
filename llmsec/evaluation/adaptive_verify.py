#!/usr/bin/env python3
"""
ELO 意外盲区自适应验证

核心逻辑：
  1. 从 ELO history 中找出“低 ELO 攻击却成功”的意外事件（upsets）。
  2. 对 top-N 意外方法按现有聚类结果分组。
  3. 在每个簇内挑选与意外方法 ELO 相近、但测试次数较少的其他方法，
     作为二次验证候选，确认模型在该方向是否确实存在短板。

用法:
    python -m llmsec.evaluation.adaptive_verify \
        --attack-file attacks/harmbench_ensemble.jsonl \
        --top-n 10 --per-cluster 3 --elo-window 50

输出:
    output/verify_plan.json   — 验证计划（含候选方法及其代表 prompt）
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from llmsec.core.config import ELO_FILE, LEGACY_ELO_FILE, OUTPUT_DIR, resolve_existing
from llmsec.core.io import iter_jsonl
from llmsec.evaluation.elo import ELOTracker


def load_tracker() -> ELOTracker | None:
    """加载 ELO 状态。"""
    elo_file = resolve_existing(ELO_FILE, LEGACY_ELO_FILE)
    if not elo_file.exists():
        return None
    tracker = ELOTracker()
    tracker.load(elo_file)
    return tracker


def load_cluster_report(path: Path | None = None) -> dict:
    """加载聚类报告。"""
    if path is None:
        path = OUTPUT_DIR / "cluster_report.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_prompts_by_method(attack_file: Path) -> dict[str, list[dict]]:
    """按方法分组加载攻击 prompt（取第一条作为代表）。"""
    by_method: dict[str, list[dict]] = defaultdict(list)
    if not attack_file.exists():
        return by_method
    for record in iter_jsonl(attack_file):
        method = record.get("method", "unknown")
        by_method[method].append(record)
    return by_method


def test_counts_from_tracker(tracker: ELOTracker) -> dict[str, int]:
    """统计每个攻击方法在 ELO history 中的测试次数。"""
    counts: dict[str, int] = defaultdict(int)
    for h in tracker.history:
        counts[h["attacker"]] += 1
    return counts


def build_upset_candidates(
    tracker: ELOTracker,
    cluster_report: dict,
    prompts_by_method: dict[str, list[dict]],
    top_n: int = 10,
    per_cluster: int = 3,
    elo_window: float = 50.0,
) -> list[dict]:
    """
    生成自适应验证候选。

    返回每个 upset 方法一个条目，包含同簇内 ELO 相近的候选方法。
    """
    method_labels = cluster_report.get("method_labels", {})
    cluster_names = cluster_report.get("cluster_names", {})
    ratings = tracker.attacker_ratings
    test_counts = test_counts_from_tracker(tracker)

    # 1. 聚合意外事件到方法级，取最大分差
    method_upset_gap: dict[str, float] = defaultdict(float)
    method_upset_events: dict[str, list[dict]] = defaultdict(list)
    for ev in tracker.find_upsets(min_elo_gap=0):
        m = ev["attacker"]
        method_upset_events[m].append(ev)
        method_upset_gap[m] = max(method_upset_gap[m], ev["elo_gap"])

    # 2. 按最大分差排序，取 top-N
    top_methods = sorted(method_upset_gap.items(), key=lambda x: x[1], reverse=True)[:top_n]

    plans = []
    seen_clusters: set[str] = set()

    for method, max_gap in top_methods:
        cluster_id = str(method_labels.get(method, -1))
        cluster_name = cluster_names.get(cluster_id, f"簇{cluster_id}")
        method_elo = ratings.get(method, 1500.0)

        # 同簇候选：ELO 在 [method_elo - window, method_elo + window] 内，
        # 排除自己，按测试次数少 -> ELO 接近排序
        # 若该方法不在聚类报告中（簇 -1），则退化到全局 ELO 邻近搜索
        candidates = []
        search_pool = (
            {other: other for other in ratings}
            if cluster_id == "-1" or not method_labels
            else method_labels
        )
        for other in search_pool:
            if other == method:
                continue
            # 有聚类报告时只考虑同簇；无聚类报告时全局搜索
            if method_labels and str(method_labels.get(other, -1)) != cluster_id:
                continue
            other_elo = ratings.get(other, 1500.0)
            if abs(other_elo - method_elo) > elo_window:
                continue
            candidates.append({
                "method": other,
                "elo": round(other_elo, 1),
                "test_count": test_counts.get(other, 0),
                "elo_distance": round(abs(other_elo - method_elo), 1),
            })

        # 优先选测试次数少、ELO 近的
        candidates.sort(key=lambda x: (x["test_count"], x["elo_distance"]))
        selected = candidates[:per_cluster]

        # 补充代表 prompt
        for c in selected:
            reps = prompts_by_method.get(c["method"], [])
            if reps:
                rep = reps[0]
                c["prompt_id"] = rep.get("id", rep.get("original_id", ""))
                c["prompt_preview"] = rep.get("prompt", "")[:200]
            else:
                c["prompt_id"] = ""
                c["prompt_preview"] = ""

        plans.append({
            "upset_method": method,
            "upset_elo": round(method_elo, 1),
            "max_weakness_gap": round(max_gap, 1),
            "weakness_events": len(method_upset_events[method]),
            "cluster_id": cluster_id,
            "cluster_name": cluster_name,
            "candidates": selected,
        })
        seen_clusters.add(cluster_id)

    return plans


def main():
    parser = argparse.ArgumentParser(description="ELO 意外盲区自适应验证")
    parser.add_argument("--attack-file", type=str, default="attacks/harmbench_ensemble.jsonl",
                        help="攻击集文件路径（相对 output/ 或绝对路径）")
    parser.add_argument("--cluster-report", type=str, default=None,
                        help="聚类报告路径（默认 output/cluster_report.json）")
    parser.add_argument("--top-n", type=int, default=10,
                        help="取前 N 个最大 ELO 分差的意外方法")
    parser.add_argument("--per-cluster", type=int, default=3,
                        help="每个簇内挑选的候选方法数")
    parser.add_argument("--elo-window", type=float, default=50.0,
                        help="候选方法与意外方法的 ELO 差值上限")
    parser.add_argument("--output", type=str, default="verify_plan.json",
                        help="输出计划文件名（相对 output/）")
    args = parser.parse_args()

    tracker = load_tracker()
    if tracker is None:
        print("❌ 未找到 ELO 状态文件，请先运行评估。")
        return

    cluster_report = load_cluster_report(
        Path(args.cluster_report) if args.cluster_report else None
    )
    if not cluster_report:
        print("⚠ 未找到聚类报告，将仅按 ELO 相近推荐候选。")

    attack_path = Path(args.attack_file)
    if not attack_path.is_absolute():
        attack_path = OUTPUT_DIR / attack_path
    prompts_by_method = load_prompts_by_method(attack_path)

    plans = build_upset_candidates(
        tracker,
        cluster_report,
        prompts_by_method,
        top_n=args.top_n,
        per_cluster=args.per_cluster,
        elo_window=args.elo_window,
    )

    output_path = OUTPUT_DIR / args.output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": __import__("datetime").datetime.now().isoformat(),
            "params": {
                "attack_file": str(attack_path),
                "top_n": args.top_n,
                "per_cluster": args.per_cluster,
                "elo_window": args.elo_window,
            },
            "plans": plans,
        }, f, ensure_ascii=False, indent=2)

    print(f"✅ 验证计划已保存: {output_path}")
    print(f"   共 {len(plans)} 个意外方法，推荐候选 {sum(len(p['candidates']) for p in plans)} 条")
    for p in plans:
        print(f"\n  🎯 {p['upset_method']} (ELO={p['upset_elo']}, gap={p['max_weakness_gap']})")
        print(f"     簇 {p['cluster_id']}: {p['cluster_name']}")
        for c in p["candidates"]:
            print(f"       → {c['method']} (ELO={c['elo']}, tests={c['test_count']})")


if __name__ == "__main__":
    main()
