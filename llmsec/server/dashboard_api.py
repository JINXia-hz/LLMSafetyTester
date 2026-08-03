#!/usr/bin/env python3
"""
LLMSEC 安全评估 Web 面板（FastAPI + 原生 HTML/JS）

功能：
- 只读数据 API：总览（雷达图）、威胁看板、ELO 排名与收敛曲线、
  Markdown 报告、聚类分析、SVD-Ridge 预测模型诊断
- 操作 API：图形化触发生成攻击集 / 自适应评估 / 聚类分析（子进程任务 + 状态轮询）

启动:
    cd C:/Users/LPF/Desktop/LLM攻击测试
    .venv/Scripts/uvicorn llmsec.server.dashboard_api:app --host 127.0.0.1 --port 8080

访问:
    http://localhost:8080
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from llmsec.core.config import CLUSTER_ARTIFACTS_FILE, OUTPUT_DIR, RUNS_DIR, STATE_FILE
from llmsec.params import ADAPTIVE_BATCH_MAX

# ============================================================
# 路径
# ============================================================
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SERVER_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = SERVER_DIR / "templates"
STATIC_DIR = SERVER_DIR / "static"
ATTACKS_DIR = OUTPUT_DIR / "attacks"
TASK_LOG_DIR = OUTPUT_DIR / "tasks"

RUN_NAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}$")

app = FastAPI(title="LLMSEC Dashboard")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ============================================================
# 通用工具
# ============================================================
def load_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _validate_run(run: str) -> str:
    if not RUN_NAME_RE.match(run):
        raise HTTPException(status_code=400, detail=f"非法 run 参数: {run!r}")
    return run


def _run_dir(run: str | None) -> Path | None:
    """解析 run 参数为目录；缺省取最新一个有报告的批次；无可用目录返回 None。"""
    if run:
        _validate_run(run)
        d = RUNS_DIR / run
        return d if d.is_dir() else None
    runs = _discover_runs()
    for r in runs:
        if r["has_report"]:
            return RUNS_DIR / r["name"]
    return RUNS_DIR / runs[0]["name"] if runs else None


def _discover_runs() -> list[dict]:
    if not RUNS_DIR.exists():
        return []
    runs = []
    for d in RUNS_DIR.iterdir():
        if not d.is_dir() or not RUN_NAME_RE.match(d.name):
            continue
        runs.append({
            "name": d.name,
            "has_report": (d / "runner_report.json").exists(),
            "has_md": (d / "security_report.md").exists(),
            "has_tree": (d / "security_tree.json").exists(),
            "has_cluster_analysis": (d / "cluster_security_analysis.json").exists(),
            "mtime": datetime.fromtimestamp(d.stat().st_mtime).isoformat(),
        })
    runs.sort(key=lambda x: x["name"], reverse=True)
    return runs


def _load_state(run: str | None = None) -> dict:
    """加载 Elo state。指定 run 时优先读 run 目录内的快照（runner 结束时保存），
    避免全局 state 漂移导致历史批次的实测/预测标记错配；无快照则回退全局。"""
    run_dir = _run_dir(run)
    if run_dir is not None:
        snapshot = load_json(run_dir / "state.json")
        if snapshot:
            return snapshot
    return load_json(STATE_FILE)


def _convergence_score(state: dict) -> float | None:
    """由 state.json 的 round_defender_elos 计算收敛稳定度：std 越小越接近 1。"""
    round_elos = state.get("round_defender_elos", {})
    elos = None
    for v in round_elos.values():
        if v:
            elos = v
            break
    if not elos:
        return None
    recent = elos[-3:]
    if len(recent) >= 2:
        mean = sum(recent) / len(recent)
        var = sum((x - mean) ** 2 for x in recent) / len(recent)
        std = var ** 0.5
    else:
        std = 10.0  # 单点按阈值保守处理（与 check_convergence 一致）
    return round(max(0.0, 1.0 - std / 20.0), 4)


# ============================================================
# 页面
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


# ============================================================
# 数据 API
# ============================================================
@app.get("/api/runs")
async def api_runs():
    return {"runs": _discover_runs()}


@app.get("/api/overview")
async def api_overview(run: str | None = None):
    run_dir = _run_dir(run)
    if run_dir is None:
        return {"available": False, "runs": []}

    report = load_json(run_dir / "runner_report.json")
    tree = load_json(run_dir / "security_tree.json")
    overall = tree.get("overall", {})
    state = _load_state(run)

    attack = report.get("attack_phase", {})
    elo = report.get("elo", {})
    allergy = report.get("allergy", {})

    asr = overall.get("asr", attack.get("asr", 0))
    fpr = overall.get("fpr", allergy.get("fpr", 0))
    confidence = overall.get("elo_confidence", elo.get("boundary_confidence", 0))
    total_methods = max(elo.get("total_methods", overall.get("total_methods", 0)), 1)
    total_tested = attack.get("total_tested", overall.get("total_tests", 0))
    coverage = min(total_tested / total_methods, 1.0)
    conv_score = _convergence_score(state)

    # 越狱税：优先 runner_report 的聚合块（新 run），回退 security_tree.overall（旧 run）
    tax_info = attack.get("jailbreak_tax") or overall.get("jailbreak_tax") or {}
    tax_mean = tax_info.get("tax_mean")
    if tax_mean is None:
        tax_mean = overall.get("jailbreak_tax_mean")
    tax_high_ratio = tax_info.get("high_tax_ratio")
    tax_probed = tax_info.get("probed")

    radar = {
        "labels": ["防线强度", "低误杀", "边界置信度", "测试覆盖率", "收敛稳定"],
        "values": [
            round(1 - asr, 4),
            round(1 - fpr, 4),
            round(confidence, 4),
            round(coverage, 4),
            conv_score if conv_score is not None else 0.0,
        ],
    }

    harm_type_asr = {
        k: v.get("asr", 0)
        for k, v in tree.get("dimensions", {}).get("by_harm_type", {}).items()
    }

    return {
        "available": True,
        "run": run_dir.name,
        "generated_at": report.get("generated_at"),
        "target_model": report.get("target_model"),
        "overall_verdict": report.get("overall_verdict"),
        "security_level": report.get("security_level", "inconclusive"),
        "recommendation": report.get("recommendation"),
        "asr": round(asr, 4),
        "fpr": round(fpr, 4),
        "rounds": attack.get("rounds", 0),
        "total_tested": total_tested,
        "successful": attack.get("successful", 0),
        "boundary_elo": overall.get("elo_boundary", elo.get("boundary_elo")),
        "boundary_confidence": round(confidence, 4),
        "methods_above_boundary": elo.get("methods_above_boundary", 0),
        "tested_above_boundary": elo.get("tested_above_boundary", 0),
        "predicted_above_boundary": elo.get("predicted_above_boundary", 0),
        "total_methods": total_methods,
        "allergy_tested": allergy.get("total_tested", 0),
        "allergic_count": allergy.get("allergic_count", 0),
        "jailbreak_tax_mean": tax_mean,
        "jailbreak_tax_high_ratio": tax_high_ratio,
        "jailbreak_tax_probed": tax_probed,
        "jailbreak_tax_attack_accuracy": tax_info.get("attack_accuracy"),
        "jailbreak_tax_baseline_accuracy": tax_info.get("baseline_accuracy"),
        "jailbreak_tax_drop": tax_info.get("accuracy_drop"),
        "radar": radar,
        "harm_type_asr": harm_type_asr,
    }


@app.get("/api/threats")
async def api_threats(run: str | None = None):
    run_dir = _run_dir(run)
    if run_dir is None:
        return {"available": False}

    tree = load_json(run_dir / "security_tree.json")
    state = _load_state(run)
    ratings = state.get("attacker_ratings", {})
    pred_std = state.get("attacker_pred_std", {})
    ground_truth = set(state.get("ground_truth", {}).keys())

    def _enrich(item: dict) -> dict:
        method = item.get("method", "")
        elo = round(ratings.get(method, item.get("elo", 1500.0)), 1)
        std = pred_std.get(method)
        tested = method in ground_truth
        return {
            **item,
            "elo": elo,
            "tested": tested,
            "source": "ground_truth" if tested else "svd_ridge",
            "pred_std": round(std, 1) if std is not None else None,
            "ci95": (
                [round(elo - 1.96 * std, 1), round(elo + 1.96 * std, 1)]
                if std is not None and not tested
                else None
            ),
        }

    return {
        "available": True,
        "run": run_dir.name,
        "top_threats": [_enrich(t) for t in tree.get("top_threats", [])],
        "strong_defenses": [_enrich(t) for t in tree.get("strong_defenses", [])],
        "upsets": tree.get("upsets", {}),
    }


@app.get("/api/elo")
async def api_elo(run: str | None = None):
    state = _load_state(run)
    ratings = state.get("attacker_ratings", {})
    pred_std = state.get("attacker_pred_std", {})
    ground_truth = set(state.get("ground_truth", {}).keys())

    ranking = [
        {
            "method": m,
            "elo": round(e, 1),
            "tested": m in ground_truth,
            "pred_std": round(pred_std[m], 1) if m in pred_std else None,
        }
        for m, e in ratings.items()
    ]
    ranking.sort(key=lambda x: x["elo"], reverse=True)

    defenders = [
        {"model": k, "elo": round(v, 1)}
        for k, v in state.get("defender_ratings", {}).items()
    ]
    round_elos = state.get("round_defender_elos", {})

    return {
        "total": len(ranking),
        "ranking": ranking,
        "defenders": defenders,
        "round_defender_elos": round_elos,
        "ground_truth_count": len(ground_truth),
    }


@app.get("/api/report-md")
async def api_report_md(run: str | None = None):
    run_dir = _run_dir(run)
    if run_dir is None:
        return {"available": False}
    md_path = run_dir / "security_report.md"
    if not md_path.exists():
        return {"available": False, "run": run_dir.name}
    return {
        "available": True,
        "run": run_dir.name,
        "markdown": md_path.read_text(encoding="utf-8"),
    }


@app.get("/api/clusters")
async def api_clusters(run: str | None = None):
    run_dir = _run_dir(run)
    analysis = load_json(run_dir / "cluster_security_analysis.json") if run_dir else {}
    # validation/簇效验证/密度视图优先读 run 内 cluster_report 快照（runner 结束时保存），
    # 无快照回退全局最近产物（旧 run 行为不变）
    report = load_json(run_dir / "cluster_report.json") if run_dir else {}
    if not report:
        report = load_json(OUTPUT_DIR / "cluster_report.json")

    if not analysis and not report:
        return {"available": False}

    clusters = []
    for cid, detail in (analysis.get("clusters") or {}).items():
        clusters.append({
            "id": cid,
            "name": detail.get("name", f"簇{cid}"),
            "size": detail.get("size", 0),
            "test_coverage": detail.get("test_coverage", 0),
            "mean_elo": detail.get("mean_elo"),
            "elo_std": detail.get("elo_std"),
            "mean_success_rate": detail.get("mean_success_rate", 0),
            "asr": detail.get("asr", 0),
            "members": detail.get("members", []),
            "tested_members": detail.get("tested_members", []),
        })
    clusters.sort(key=lambda c: c["size"], reverse=True)

    return {
        "available": True,
        "run": run_dir.name if run_dir else None,
        "defender_name": analysis.get("defender_name"),
        "defender_elo": analysis.get("defender_elo"),
        "n_methods": analysis.get("n_methods", report.get("method_count", 0)),
        "n_clusters": analysis.get("n_clusters", report.get("n_clusters", 0)),
        "n_noise": analysis.get("n_noise", report.get("n_noise", 0)),
        "validation": report.get("validation", {}),
        "reaction_validation": report.get("reaction_validation"),
        "hdbscan": report.get("hdbscan"),
        "clusters": clusters,
        "high_risk_clusters": analysis.get("high_risk_clusters", []),
        "blind_spot_clusters": analysis.get("blind_spot_clusters", []),
        "stable_clusters": analysis.get("stable_clusters", []),
    }


@app.get("/api/model")
async def api_model(run: str | None = None):
    run_dir = _run_dir(run)
    analysis = load_json(run_dir / "cluster_security_analysis.json") if run_dir else {}
    svd = analysis.get("svd_ridge")
    if not svd:
        return {"available": False, "run": run_dir.name if run_dir else None}
    return {"available": True, "run": run_dir.name, "svd_ridge": svd}


@app.get("/api/attack-sets")
async def api_attack_sets():
    if not ATTACKS_DIR.exists():
        return {"files": []}
    files = sorted(p.name for p in ATTACKS_DIR.glob("*.jsonl"))
    return {"files": files}


# ============================================================
# 聚类特征空间投影（PCA / t-SNE，按需计算 + 缓存）
# ============================================================
# 缓存大小上限：超出时按插入顺序淘汰最旧条目，防长期运行内存单调增长
_CACHE_MAX_SIZE = 64

_PROJECTION_CACHE: dict[tuple[str, float], dict] = {}
_PROJECTION_BLOCKS = ("textual", "embedding", "technique", "intent")


def _cache_put(cache: dict, key, value) -> None:
    """写入缓存并维护 _CACHE_MAX_SIZE 上限（dict 保序，弹掉头一个即最旧条目）。"""
    cache[key] = value
    while len(cache) > _CACHE_MAX_SIZE:
        cache.pop(next(iter(cache)))


def _build_feature_matrix(features: dict, methods: list[str]):
    """拼接 textual+embedding+technique+intent 特征块为矩阵（块维度不一致零填充）。"""
    import numpy as np

    dims = {b: 0 for b in _PROJECTION_BLOCKS}
    vecs = {}
    for m in methods:
        feat = features.get(m, {})
        v = {}
        for b in _PROJECTION_BLOCKS:
            vec = np.atleast_1d(np.asarray(feat.get(b, np.zeros(0)), dtype=np.float64))
            v[b] = vec
            dims[b] = max(dims[b], vec.shape[0])
        vecs[m] = v

    rows = []
    for m in methods:
        parts = []
        for b in _PROJECTION_BLOCKS:
            vec = vecs[m][b]
            if vec.shape[0] < dims[b]:
                vec = np.pad(vec, (0, dims[b] - vec.shape[0]))
            parts.append(vec)
        rows.append(np.concatenate(parts))
    return np.array(rows, dtype=np.float64)


@app.get("/api/cluster-projection")
async def api_cluster_projection(method: str = "pca"):
    """
    对聚类 artifacts 中的高维特征做 2D 投影（PCA / t-SNE），供分布散点图使用。
    结果按 (method, artifacts mtime) 缓存。
    """
    import joblib
    import numpy as np

    if method not in ("pca", "tsne"):
        raise HTTPException(status_code=400, detail=f"不支持的投影方法: {method!r}")
    if not CLUSTER_ARTIFACTS_FILE.exists():
        return {"available": False}

    mtime = CLUSTER_ARTIFACTS_FILE.stat().st_mtime
    cache_key = (method, mtime)
    if cache_key in _PROJECTION_CACHE:
        return _PROJECTION_CACHE[cache_key]

    try:
        artifacts = joblib.load(CLUSTER_ARTIFACTS_FILE)
    except Exception:
        return {"available": False}

    features = artifacts.get("features", {})
    labels = artifacts.get("labels", {})
    cluster_names = artifacts.get("cluster_names", {})
    gt_methods = set(artifacts.get("ground_truth_methods", []))
    if not features:
        return {"available": False}

    methods = sorted(features.keys())
    n = len(methods)
    X = _build_feature_matrix(features, methods)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    result: dict = {"available": True, "method": method, "n": n}
    if n < 2:
        coords = np.zeros((n, 2))
    elif method == "pca":
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X)
        result["explained_variance"] = [
            round(float(r), 4) for r in pca.explained_variance_ratio_
        ]
    else:
        from sklearn.manifold import TSNE

        # sklearn 要求 perplexity < n，小样本自适应收缩
        perplexity = max(1, min(30, (n - 1) // 3))
        tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=42)
        coords = tsne.fit_transform(X)
        result["perplexity"] = perplexity

    state = _load_state()
    ratings = state.get("attacker_ratings", {})

    points = []
    for i, m in enumerate(methods):
        cid = labels.get(m, -1)
        try:
            cid = int(cid)
        except (TypeError, ValueError):
            cid = -1
        points.append({
            "method": m,
            "x": round(float(coords[i, 0]), 4),
            "y": round(float(coords[i, 1]), 4),
            "cluster": cid,
            "cluster_name": cluster_names.get(str(cid), f"簇{cid}"),
            "tested": m in gt_methods,
            "elo": round(ratings[m], 1) if m in ratings else None,
        })

    result["points"] = points
    _cache_put(_PROJECTION_CACHE, cache_key, result)
    return result


# ============================================================
# 聚类层次树（树图 + 任意层切割）
# ============================================================
_CUT_CACHE: dict[tuple[int, float], dict] = {}


def _load_tree_artifacts() -> dict | None:
    """加载含 linkage 的聚类 artifacts；不存在或缺 linkage 时返回 None。"""
    import joblib

    if not CLUSTER_ARTIFACTS_FILE.exists():
        return None
    try:
        artifacts = joblib.load(CLUSTER_ARTIFACTS_FILE)
    except Exception:
        return None
    if artifacts.get("linkage") is None:
        return None
    return artifacts


@app.get("/api/cluster-tree")
async def api_cluster_tree():
    """返回层次树的树图坐标（scipy dendrogram 的 icoord/dcoord）与 auto-k 信息。"""
    artifacts = _load_tree_artifacts()
    if artifacts is None:
        return {"available": False}

    from scipy.cluster.hierarchy import dendrogram

    Z = artifacts["linkage"]
    labels = artifacts.get("labels", {})
    n = len(labels)
    dd = dendrogram(Z, no_plot=True)

    # maxclust=k 对应的切割高度：第 n-k 与 n-k+1 次合并高度之间
    heights = sorted(float(h) for h in Z[:, 2])

    def cut_height(k: int) -> float:
        if k <= 1:
            return heights[-1] * 1.05 if heights else 1.0
        if k >= n:
            return 0.0
        return (heights[n - k - 1] + heights[n - k]) / 2

    chosen_k = artifacts.get("chosen_k") or len(set(labels.values()) - {-1})
    return {
        "available": True,
        "n": n,
        "icoord": dd["icoord"],
        "dcoord": dd["dcoord"],
        "merge_heights": heights,
        "chosen_k": chosen_k,
        "chosen_height": cut_height(chosen_k),
        "top_ks": artifacts.get("top_ks", [chosen_k]),
        "candidate_sweep": artifacts.get("candidate_sweep", []),
        "max_height": heights[-1] if heights else 1.0,
    }


@app.get("/api/cluster-cut")
async def api_cluster_cut(k: int):
    """在层次树上切出 k 个簇（fcluster O(n)），返回该层簇结构与命名。"""
    import joblib

    artifacts = _load_tree_artifacts()
    if artifacts is None:
        return {"available": False}

    labels = artifacts.get("labels", {})
    n = len(labels)
    if k < 2 or k > n:
        raise HTTPException(status_code=400, detail=f"k 必须在 [2, {n}] 内")

    mtime = CLUSTER_ARTIFACTS_FILE.stat().st_mtime
    cache_key = (k, mtime)
    if cache_key in _CUT_CACHE:
        return _CUT_CACHE[cache_key]

    from scipy.cluster.hierarchy import fcluster

    from llmsec.clustering.pipeline import auto_name_clusters

    Z = artifacts["linkage"]
    methods = sorted(labels.keys())
    raw = fcluster(Z, t=k, criterion="maxclust")
    cut_labels = {m: int(c) - 1 for m, c in zip(methods, raw)}

    names = auto_name_clusters(
        cut_labels,
        artifacts.get("features", {}),
        artifacts.get("meta", {}),
        artifacts.get("meta", {}).get("method_prompts", {}),
    )

    clusters: dict[int, list[str]] = {}
    for m, cid in cut_labels.items():
        clusters.setdefault(cid, []).append(m)

    state = _load_state()
    ratings = state.get("attacker_ratings", {})

    result = {
        "available": True,
        "k": k,
        "clusters": [
            {
                "id": cid,
                "name": names.get(cid, f"簇{cid}"),
                "size": len(members),
                "members": sorted(members),
                "mean_elo": (
                    round(sum(ratings.get(m, 1500.0) for m in members) / len(members), 1)
                    if members else None
                ),
            }
            for cid, members in sorted(clusters.items())
        ],
    }
    _cache_put(_CUT_CACHE, cache_key, result)
    return result


# ============================================================
# 操作 API（子进程任务）
# ============================================================
TASKS: dict[str, dict] = {}


class EvaluateRequest(BaseModel):
    phase: str = Field(default="all", pattern="^(all|1|2)$")
    input: str = "l1.jsonl"
    # runner._adaptive_batch_size 会把 batch 压到 [ADAPTIVE_BATCH_MIN, ADAPTIVE_BATCH_MAX] 内，
    # 上限与 runner 对齐，避免用户传 >ADAPTIVE_BATCH_MAX 时被静默压回；默认值随上限自适应
    batch_size: int = Field(default=min(10, ADAPTIVE_BATCH_MAX), ge=1, le=ADAPTIVE_BATCH_MAX)
    max_rounds: int = Field(default=5, ge=1, le=50)
    sampler: str = Field(default="hybrid", pattern="^(gap|infogain|coordinate|hybrid)$")


def _refresh_task_status(t: dict) -> None:
    """刷新任务状态：子进程已结束但 status 仍为 running 时更新为 success/failed，
    并关闭 log_file 句柄（置 None）。

    子进程可能崩溃且无人轮询接口，若只在 _task_view 里更新状态，
    TASKS 中会残留永久 running 的任务（导致 _start_task 的 409 检查误拒同类新任务），
    log_file 句柄也随 TASKS 常驻泄漏。
    """
    if t["status"] != "running":
        return
    proc: subprocess.Popen = t["proc"]
    rc = proc.poll()
    if rc is None:
        return
    t["status"] = "success" if rc == 0 else "failed"
    t["returncode"] = rc
    log_file = t.get("log_file")
    if log_file is not None:
        log_file.close()
        t["log_file"] = None


def _task_view(task_id: str, t: dict) -> dict:
    _refresh_task_status(t)
    status = t["status"]
    rc = t.get("returncode")
    log_tail = ""
    log_path: Path = t["log_path"]
    if log_path.exists():
        try:
            log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        except OSError:
            pass
    return {
        "id": task_id,
        "kind": t["kind"],
        "cmd": t["cmd"],
        "status": status,
        "returncode": rc,
        "started_at": t["started_at"],
        "log_tail": log_tail,
    }


def _start_task(kind: str, argv: list[str]) -> dict:
    # 先刷新所有 running 任务的真实状态，避免子进程崩溃后无人轮询
    # 导致 status 永久 running（409 误拒同类新任务）与 log_file 句柄泄漏
    for t in TASKS.values():
        _refresh_task_status(t)
    for tid, t in TASKS.items():
        if t["kind"] == kind and t["status"] == "running":
            raise HTTPException(status_code=409, detail=f"{kind} 任务正在运行中 (id={tid})")

    TASK_LOG_DIR.mkdir(parents=True, exist_ok=True)
    task_id = f"{kind}-{datetime.now().strftime('%H%M%S')}-{uuid.uuid4().hex[:6]}"
    log_path = TASK_LOG_DIR / f"{task_id}.log"
    log_file = open(log_path, "w", encoding="utf-8")

    try:
        proc = subprocess.Popen(
            [sys.executable, *argv],
            cwd=WORKSPACE_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
    except OSError as e:
        log_file.close()
        raise HTTPException(status_code=500, detail=f"任务启动失败: {e}")

    TASKS[task_id] = {
        "kind": kind,
        "cmd": " ".join(argv),
        "proc": proc,
        "log_path": log_path,
        "log_file": log_file,
        "status": "running",
        "started_at": datetime.now().isoformat(),
    }
    return _task_view(task_id, TASKS[task_id])


@app.post("/api/run/generate")
async def api_run_generate():
    return _start_task("generate", ["-m", "llmsec.attacks.generate"])


@app.post("/api/run/evaluate")
async def api_run_evaluate(req: EvaluateRequest):
    # input 只允许 output/attacks/ 下的 jsonl 文件名，防路径穿越
    input_name = Path(req.input).name
    if not input_name.endswith(".jsonl"):
        raise HTTPException(status_code=400, detail="input 必须是 .jsonl 文件名")
    if not (ATTACKS_DIR / input_name).exists():
        raise HTTPException(status_code=404, detail=f"攻击集不存在: attacks/{input_name}")

    argv = [
        "-m", "llmsec.pipeline.runner",
        "--phase", req.phase,
        "--input", f"attacks/{input_name}",
        "--batch-size", str(req.batch_size),
        "--max-rounds", str(req.max_rounds),
        "--sampler", req.sampler,
    ]
    return _start_task("evaluate", argv)


@app.post("/api/run/cluster-analysis")
async def api_run_cluster_analysis():
    return _start_task("cluster-analysis", ["-m", "llmsec.evaluation.cluster_analysis"])


@app.get("/api/tasks")
async def api_tasks():
    return {"tasks": [_task_view(tid, t) for tid, t in sorted(TASKS.items(), reverse=True)]}


@app.get("/api/tasks/{task_id}")
async def api_task(task_id: str):
    t = TASKS.get(task_id)
    if t is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return _task_view(task_id, t)
