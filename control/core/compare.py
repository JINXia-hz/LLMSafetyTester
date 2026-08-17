"""control.core.compare — 历史对比（跨工作单元横截面分析）。

控制层消费 llmsec 的公开输出产物（runner_report.json / security_tree.json），
做服务端聚合对比。**run 发现/解析经 control.core.storage 薄契约走目录库
（llmsec.storage 单一实现），指标提取用统一 extract_report_metrics；报告
本体仍只读文件，不碰 R 矩阵。**

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
from control.core.paths import safe_component, safe_subpath
from control.core.storage import (
    RUN_NAME_RE,
    extract_report_metrics,
    get_run,
    query_runs,
    reconcile_runs,
)


def _load_json_named(run_dir: Path, filename: str) -> dict | None:
    """读 run_dir 下的命名 JSON 产物（runner_report.json / security_tree.json）。

    不存在或解析失败返回 None。_load_report / _load_tree 共用此实现。
    """
    p = run_dir / filename
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _load_report(run_dir: Path) -> dict | None:
    return _load_json_named(run_dir, "runner_report.json")


def _load_tree(run_dir: Path) -> dict | None:
    return _load_json_named(run_dir, "security_tree.json")


def _resolve_run_dir(run_name: str) -> Path | None:
    """run_name → 含 runner_report.json 的目录（目录库解析，storage 单一实现）。

    支持三种来源：
      'ts/target' 或 'ts'           → output/runs/<name>（历史 run）
      'ws:<name>'                    → output/workspaces/<name>/ 下找含报告的 <target>/ 子目录
      'ws:<name>/<target>'           → output/workspaces/<name>/<target>/（指定 target）

    外部 run_name 先形状校验（RUN_NAME_RE / safe_component）防路径穿越——
    非法名称视为目录不存在（返回 None）；dir_path 取自目录库登记行（登记来源
    已经过扫描校验），不再手工拼路径。
    """
    # workspace 来源：卫星目录库（<ws>/catalog.db，query_runs 自带对账）
    if run_name.startswith("ws:"):
        rest = run_name[3:]
        parts = rest.split("/", 1)
        try:
            ws_dir = safe_component(WORKSPACES_DIR, parts[0])
        except ValueError:
            return None
        if not ws_dir.is_dir():
            return None
        rows = query_runs(runs_root=ws_dir)
        if len(parts) == 2:
            row = next((r for r in rows if r.target == parts[1] and r.has_report), None)
        else:
            row = next((r for r in rows if r.has_report), None)
        return Path(row.dir_path) if row else None
    # 历史来源：'ts' 或 'ts/target'
    parts = run_name.split("/", 1)
    if not parts or not RUN_NAME_RE.match(parts[0]):
        return None
    try:
        safe_subpath(RUNS_DIR, *parts)
    except ValueError:
        return None
    reconcile_runs(runs_root=RUNS_DIR)
    row = get_run(run_name, runs_root=RUNS_DIR) or get_run(parts[0], runs_root=RUNS_DIR)
    return Path(row.dir_path) if row is not None else None


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
    tax = attack.get("jailbreak_tax", {}) or {}
    return {
        "run": run_name,
        "target_model": rep.get("target_model"),
        "security_level": rep.get("security_level", "inconclusive"),
        **extract_report_metrics(rep),
        "total_methods": elo.get("total_methods"),
        "methods_above_boundary": elo.get("methods_above_boundary"),
        "rounds": attack.get("rounds"),
        "tax_probed": tax.get("probed"),
    }


def discover_workspace_runs() -> list[dict]:
    """列出所有 workspace 内含报告的 run（供 list_runs tool 补充历史 run 列表）。

    每个 workspace 可能有多个 target 子目录（每个是一个独立 run）——经各自
    卫星目录库查询（query_runs 自带对账，旧 workspace 无库时首查自动建册）。
    返回 [{name, workspace, target, ...metrics}]，name 形如 'ws:<ws>/<target>'。
    """
    out = []
    if not WORKSPACES_DIR.exists():
        return out
    for ws_dir in sorted(WORKSPACES_DIR.iterdir()):
        if not ws_dir.is_dir():
            continue
        for row in query_runs(runs_root=ws_dir):
            if not row.has_report:
                continue
            out.append({
                "name": f"ws:{ws_dir.name}/{row.target}",
                "workspace": ws_dir.name,
                "target": row.target,
                "target_model": row.target_model or row.target,
                "security_level": row.security_level or "inconclusive",
                "asr": (row.metrics or {}).get("asr"),
                "boundary_elo": (row.metrics or {}).get("boundary_elo"),
                "has_report": True,
            })
    return out


def list_all_runs(*, target: str | None = None, since: str | None = None,
                  junk_only: bool = False, include_workspaces: bool = True) -> list[dict]:
    """列出历史 run +（可选）workspace 分支内的 run。

    中书省 tools._do_list_runs 与尚书省 capabilities._h_list_runs 的统一口径
    （此前两份实现行为分叉：一版对 workspace runs 按 target 过滤、一版不过滤，
    同名能力经两省执行结果不一致）。
    """
    from control.core.invoker import list_runs as inv_list_runs
    runs = inv_list_runs(target=target, since=since, junk_only=junk_only)
    if include_workspaces:
        ws_runs = discover_workspace_runs()
        if target:
            ws_runs = [r for r in ws_runs if r.get("target") == target
                       or r.get("target_model") == target]
        runs = runs + ws_runs
    return runs


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
