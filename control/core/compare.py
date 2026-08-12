"""control.core.compare — 历史对比（跨工作单元横截面分析）。

控制层消费 llmsec 的公开输出产物（runner_report.json / security_tree.json），
做服务端聚合对比。**只读文件，不 import llmsec，不碰 R 矩阵。**

对比维度：
  - 指标表：asr / fpr / boundary_elo / coverage / security_level / conv_rounds
  - 威胁树 diff：security_tree.json 的 dimensions（by_harm_type / by_attack_category）按类目对齐
  - 趋势：跨 run 的指标时序

替代 llmsec 看板现有的纯前端两两 diff（sections.js renderCompare），且支持 >2 个 run。
"""

from __future__ import annotations

import json
from pathlib import Path

from control.config import RUNS_DIR, WORKSPACES_DIR


def _load_report(run_dir: Path) -> dict | None:
    p = run_dir / "runner_report.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _load_tree(run_dir: Path) -> dict | None:
    p = run_dir / "security_tree.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _resolve_run_dir(run_name: str) -> Path | None:
    """run_name → 含 runner_report.json 的目录。支持三种来源：

      'ts/target' 或 'ts'           → output/runs/<name>（历史 run）
      'ws:<name>'                    → output/workspaces/<name>/ 下找含报告的 <target>/ 子目录
      'ws:<name>/<target>'           → output/workspaces/<name>/<target>/（指定 target）

    workspace 模式下 runner 产物在 <ws>/<target>/runner_report.json（runner.py 把
    runs_dir 重绑到 work-dir，per-target 子目录即 target 名）。
    """
    # workspace 来源
    if run_name.startswith("ws:"):
        rest = run_name[3:]
        parts = rest.split("/", 1)
        ws_name = parts[0]
        ws_dir = WORKSPACES_DIR / ws_name
        if not ws_dir.is_dir():
            return None
        # 指定了 target：ws:<name>/<target>
        if len(parts) == 2:
            d = ws_dir / parts[1]
            return d if (d / "runner_report.json").exists() else None
        # 未指定 target：扫子目录找第一个含报告的
        for sub in ws_dir.iterdir():
            if sub.is_dir() and (sub / "runner_report.json").exists():
                return sub
        return None
    # 历史来源：output/runs/<name>
    d = RUNS_DIR / run_name
    return d if d.is_dir() else None


# ============================================================
# 单 run 指标提取
# ============================================================
def run_metrics(run_name: str) -> dict | None:
    """提取单个 run 的对比指标（轻量，只读 runner_report.json）。"""
    run_dir = _resolve_run_dir(run_name)
    if run_dir is None:
        return None
    rep = _load_report(run_dir)
    if not rep:
        return None
    attack = rep.get("attack_phase", {}) or {}
    elo = rep.get("elo", {}) or {}
    allergy = rep.get("allergy", {}) or {}
    tax = attack.get("jailbreak_tax", {}) or {}
    return {
        "run": run_name,
        "target_model": rep.get("target_model"),
        "security_level": rep.get("security_level", "inconclusive"),
        "asr": attack.get("asr"),
        "fpr": allergy.get("fpr"),
        "boundary_elo": elo.get("boundary_elo"),
        "boundary_confidence": elo.get("boundary_confidence"),
        "coverage": elo.get("coverage"),
        "conv_rounds": elo.get("conv_rounds"),
        "converged": elo.get("converged"),
        "ci_half": elo.get("ci_half"),
        "total_methods": elo.get("total_methods"),
        "methods_above_boundary": elo.get("methods_above_boundary"),
        "total_tested": attack.get("total_tested"),
        "rounds": attack.get("rounds"),
        "tax_probed": tax.get("probed"),
    }


def discover_workspace_runs() -> list[dict]:
    """列出所有 workspace 内含报告的 run（供 list_runs tool 补充历史 run 列表）。

    每个 workspace 可能有多个 target 子目录（每个是一个独立 run）。
    返回 [{name, workspace, target, ...metrics}]，name 形如 'ws:<ws>/<target>'。
    """
    out = []
    if not WORKSPACES_DIR.exists():
        return out
    for ws_dir in sorted(WORKSPACES_DIR.iterdir()):
        if not ws_dir.is_dir() or ws_dir.name == "_index.json":
            continue
        for sub in ws_dir.iterdir():
            if not sub.is_dir() or not (sub / "runner_report.json").exists():
                continue
            rep = _load_report(sub)
            if not rep:
                continue
            attack = rep.get("attack_phase", {}) or {}
            elo = rep.get("elo", {}) or {}
            out.append({
                "name": f"ws:{ws_dir.name}/{sub.name}",
                "workspace": ws_dir.name,
                "target": sub.name,
                "target_model": rep.get("target_model", sub.name),
                "security_level": rep.get("security_level", "inconclusive"),
                "asr": attack.get("asr"),
                "boundary_elo": elo.get("boundary_elo"),
                "has_report": True,
            })
    return out


# ============================================================
# 多 run 对比
# ============================================================
def compare(run_names: list[str]) -> dict:
    """对比多个 run，返回结构化对比报告。

    Returns:
        {
          runs: [{run, target, level, asr, fpr, elo, ...}, ...],  # 指标表
          metrics: {metric: {run: value, ...}},                    # 按指标透视
          threat_diff: {dimension: {category: {run: value, ...}}}, # 威胁树类目对齐
          missing: [run_names 中无报告的],
        }
    """
    rows = []
    missing = []
    for name in run_names:
        m = run_metrics(name)
        if m is None:
            missing.append(name)
        else:
            rows.append(m)

    # 指标透视：metric → {run: value}
    metric_keys = [
        "asr", "fpr", "boundary_elo", "boundary_confidence", "coverage",
        "conv_rounds", "ci_half", "total_methods", "methods_above_boundary",
    ]
    metrics_pivot: dict[str, dict] = {}
    for mk in metric_keys:
        metrics_pivot[mk] = {r["run"]: r.get(mk) for r in rows}

    # 威胁树 diff
    threat_diff = _threat_tree_diff(run_names)

    return {
        "runs": rows,
        "metrics": metrics_pivot,
        "threat_diff": threat_diff,
        "missing": missing,
    }


def _threat_tree_diff(run_names: list[str]) -> dict:
    """对齐多 run 的 security_tree.dimensions，按类目透视 asr。

    返回 {dimension: {category: {run: asr}}}，便于发现「某 run 在某类目上特别弱」。
    """
    out: dict[str, dict[str, dict]] = {}
    for name in run_names:
        run_dir = _resolve_run_dir(name)
        if run_dir is None:
            continue
        tree = _load_tree(run_dir)
        if not tree:
            continue
        dims = tree.get("dimensions", {}) or {}
        for dim_name, dim_data in dims.items():
            out.setdefault(dim_name, {})
            # dim_data 可能是 {category: {asr, total, ...}} 或 {category: number}
            if not isinstance(dim_data, dict):
                continue
            for cat, val in dim_data.items():
                asr = _extract_asr(val)
                if asr is not None:
                    out[dim_name].setdefault(cat, {})[name] = asr
    return out


def _extract_asr(val) -> float | None:
    """从威胁树类目值提取 asr（可能是 dict 或裸数字）。"""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, dict):
        for k in ("asr", "attack_success_rate", "success_rate"):
            v = val.get(k)
            if isinstance(v, (int, float)):
                return float(v)
    return None
