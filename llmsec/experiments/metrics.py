"""
experiments.metrics — 从 trial 产物提取科学度量。

核心度量 conv_rounds（HPO 目标）：优先读 runner_report.elo.conv_rounds；
缺失则从 state.json 的轮次轨迹回放 check_convergence 重算（含未收敛惩罚）。
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def load_json(p: Path):
    if p is None:
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        import logging
        logging.getLogger("llmsec.experiments.metrics").warning("load_json(%s) 失败: %s", p, e)
        return None


def _find_artifact(work_dir: Path, filename: str) -> Path | None:
    """定位 trial 产物文件。

    两种落盘布局：
      • 普通模式：runner 写到 work_dir/<filename>
      • work-dir 隔离模式（HPO trial）：runner 写到 work_dir/<target>/<filename>（runs_dir=work_dir，
        run_dir=runs_dir/target）。HPO trial 单目标，故恰好一个 target 子目录。
    """
    root = work_dir / filename
    if root.exists():
        return root
    matches = list(work_dir.glob(f"*/{filename}"))
    return matches[0] if matches else None


def extract_metrics(work_dir: Path, max_rounds: int | None = None) -> dict:
    """
    从一个 trial 的 work_dir 提取度量字典。

    返回（至少含 conv_rounds、defender_elo、asr、ci_half、coverage、fpr、tested）。
    """
    report = load_json(_find_artifact(work_dir, "runner_report.json"))
    metrics: dict = {"work_dir": str(work_dir)}

    if report:
        elo = report.get("elo", {}) or {}
        atk = report.get("attack_phase", {}) or {}
        alg = report.get("allergy", {}) or {}
        metrics.update({
            "defender_elo": elo.get("boundary_elo"),
            "confidence": elo.get("boundary_confidence"),
            "ci_half": elo.get("ci_half"),
            "drift": elo.get("drift"),
            "converged": elo.get("converged"),
            "coverage": elo.get("coverage"),
            "conv_rounds": elo.get("conv_rounds"),
            "asr": atk.get("asr"),
            "rounds_run": atk.get("rounds"),
            "tested": atk.get("total_tested"),
            "fpr": alg.get("fpr"),
        })

    # conv_rounds 兜底：从 state 回放重算（runner_report 缺失或为 None 时）
    if metrics.get("conv_rounds") is None:
        cr = _conv_rounds_from_state(work_dir, max_rounds)
        if cr is not None:
            metrics["conv_rounds"] = cr
    return metrics


def _conv_rounds_from_state(work_dir: Path, max_rounds: int | None) -> int | float | None:
    """从 state.json 轮次轨迹回放 check_convergence，返回首个收敛轮数；未收敛给惩罚值。"""
    state_path = _find_artifact(work_dir, "state.json")
    state = load_json(state_path)
    if not state or state_path is None:
        return None
    import sys
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from llmsec.evaluation.elo import ELOTracker
        from llmsec.params import CONV_CI_TARGET
        tracker = ELOTracker()
        tracker.load(str(state_path))
        defender = state.get("defender_ratings") and next(iter(state["defender_ratings"]))
        if not defender:
            return None
        total = max(1, len(state.get("attacker_ratings", {})))
        n_gt = len(state.get("ground_truth", {}))
        rounds = state.get("round_defender_elos", {}).get(defender, [])
        mr = max_rounds or (len(rounds) or 1)
        for r in range(1, len(rounds) + 1):
            saved = tracker._round_defender_elos[defender]
            tracker._round_defender_elos[defender] = rounds[:r]
            conv = tracker.check_convergence(defender, total_methods=total, tested_count=n_gt)
            tracker._round_defender_elos[defender] = saved
            if conv.get("converged"):
                return r
        # 未收敛 → 惩罚：mr + (ci_half/target)，近收敛者排名更高
        final = tracker.check_convergence(defender, total_methods=total, tested_count=n_gt)
        ci = final.get("ci_half") or CONV_CI_TARGET
        return mr + float(ci) / float(CONV_CI_TARGET)
    except (ValueError, KeyError, AttributeError, TypeError) as e:
        import logging
        logging.getLogger("llmsec.experiments.metrics").error(
            "_conv_rounds_from_state 失败（trial 评分将缺失）: %s", e, exc_info=True)
        return None


def aggregate(values: list[float], mode: str) -> float:
    """跨 repeats 聚合目标值。

    mode:
      - "mean"：均值
      - "mean_plus_std"（风险厌恶，最小化方向推荐）：返回 mean + std。
        最小化方向下"低且稳"优先——同均值时 std 越大（越抖）越被惩罚。
    """
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return float("inf")
    mean = sum(vals) / len(vals)
    if mode == "mean_plus_std":
        if len(vals) >= 2:
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
        else:
            std = 0.0
        return mean + std
    return mean
