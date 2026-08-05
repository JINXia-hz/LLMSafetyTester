#!/usr/bin/env python3
"""
LLMSEC 安全评估 Web 面板（FastAPI + 原生 HTML/JS）

功能：
- 只读数据 API：总览（雷达图）、威胁看板、ELO 排名与收敛曲线、
  Markdown 报告、聚类分析、SVD-Ridge 预测模型诊断
- 操作 API：图形化触发生成攻击集 / 自适应评估 / 聚类分析（子进程任务 + 状态轮询）

启动（在仓库根目录下执行）:
    .venv/Scripts/uvicorn llmsec.server.dashboard_api:app --host 127.0.0.1 --port 8080

访问:
    http://localhost:8080
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from llmsec.core.config import (
    ATTACKS_DIR,
    CLUSTER_RESULT_FILE,
    OUTPUT_DIR,
    RUNS_DIR,
    TASK_LOG_DIR,
)
from llmsec.core.io import read_json
from llmsec.core.logging import get_logger
from llmsec.core.seed import get_global_seed as _get_seed

_SEED = _get_seed()
from llmsec.params import ADAPTIVE_BATCH_MAX

logger = get_logger(__name__)

# ============================================================
# 路径
# ============================================================
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SERVER_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = SERVER_DIR / "templates"
STATIC_DIR = SERVER_DIR / "static"
# ATTACKS_DIR / TASK_LOG_DIR 由 core.config 统一定义（见 import）

RUN_NAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}$")

app = FastAPI(title="LLMSEC Dashboard")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ============================================================
# 通用工具
# ============================================================
def load_json(path: Path | None) -> dict:
    """加载 JSON 文件（委托 core.io.read_json，缺失/坏文件返回 {}）。"""
    return read_json(path, default={}) if path is not None else {}


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
            # M-35：多目标 run 写 multi_target_report.json（+ canonical runner_report.json），
            # 两者任一存在即视为有报告，避免多目标 run 对看板不可见
            "has_report": (d / "runner_report.json").exists() or (d / "multi_target_report.json").exists(),
            "has_md": (d / "security_report.md").exists(),
            "has_tree": (d / "security_tree.json").exists(),
            "has_cluster_analysis": (d / "cluster_security_analysis.json").exists(),
            "mtime": datetime.fromtimestamp(d.stat().st_mtime).isoformat(),
        })
    runs.sort(key=lambda x: x["name"], reverse=True)
    return runs


def _run_time(run_name: str) -> str | None:
    """运行目录名 YYYY-MM-DD_HHMMSS → ISO 时间戳；非法/解析失败返回 None。"""
    if not RUN_NAME_RE.match(run_name):
        return None
    try:
        return datetime.strptime(run_name, "%Y-%m-%d_%H%M%S").isoformat()
    except ValueError:
        return None


def _run_summary(run_dir: Path) -> dict | None:
    """读取单批次 runner_report.json，抽取趋势/批次富化所需的轻量字段。

    只读 runner_report.json（不碰 state/tree），供 /api/trend 与 /api/runs 复用，
    单次服务端循环即可替代前端逐批次拉 /api/overview。
    """
    report = load_json(run_dir / "runner_report.json")
    if not report:
        return None
    attack = report.get("attack_phase", {}) or {}
    allergy = report.get("allergy", {}) or {}
    elo = report.get("elo", {}) or {}
    tax = attack.get("jailbreak_tax") or {}
    return {
        "run": run_dir.name,
        "time": _run_time(run_dir.name) or report.get("generated_at"),
        "target": report.get("target_model"),
        "asr": attack.get("asr"),
        "fpr": allergy.get("fpr"),
        "elo": elo.get("boundary_elo"),
        "level": report.get("security_level", "inconclusive"),
        "tax_probed": tax.get("probed"),
    }


# /api/runs 富化缓存：按 (run 名, 目录 mtime) 失效，避免每次下拉都重解析报告
_RUN_META_CACHE: dict[str, tuple[float, dict]] = {}


def _run_meta(run_dir: Path) -> dict:
    """批次富化信息（target_model/security_level/asr），按目录 mtime 缓存。"""
    name = run_dir.name
    try:
        mtime = run_dir.stat().st_mtime
    except OSError:
        return {}
    cached = _RUN_META_CACHE.get(name)
    if cached and cached[0] == mtime:
        return cached[1]
    meta: dict = {}
    if (run_dir / "runner_report.json").exists():
        summ = _run_summary(run_dir)
        if summ:
            meta = {
                "target_model": summ["target"],
                "security_level": summ["level"],
                "asr": summ["asr"],
            }
    _RUN_META_CACHE[name] = (mtime, meta)
    return meta


def _load_state(run: str | None = None) -> dict:
    """加载 Elo state。指定 run 时优先读 run 目录内的快照（runner 结束时保存），
    避免全局 state 漂移导致历史批次的实测/预测标记错配。

    R-cutover：无 run 快照时不再读易漂移的全局 state.json，而是从结果矩阵 R
    派生活跃模型的 Elo（经 elo_access 缓存）；R 亦空时才回退全局 state.json。"""
    run_dir = _run_dir(run)
    if run_dir is not None:
        snapshot = load_json(run_dir / "state.json")
        if snapshot:
            return snapshot
    # 无 run 快照：R 为唯一真相（不再回退全局 state.json）
    try:
        from llmsec.evaluation.elo_access import active_model, elo_state_for
        model = active_model()
        if model is not None:
            state = elo_state_for(model)
            if state:
                return state
    except Exception as _e:
        logger.warning("降级: %s", _e)
        pass
    return {}


def _gt_set(state: dict) -> set:
    """从 state 取 ground_truth 方法集合，兼容 dict（state.json 形态）与 list（派生缓存形态）。"""
    gt = state.get("ground_truth", {})
    return set(gt.keys() if isinstance(gt, dict) else gt)


def _convergence_score(state: dict) -> float | None:
    """由 state 经 ELOTracker.check_convergence 计算收敛稳定度（与 runner/metrics 同源）。

    复用权威判据，消除原 std 近似与硬编码 3/20.0/10.0 魔法数字：
      - 旧实现取末 3 轮原始 std（含趋势、未去趋势），单点回退 std=10.0 给出虚假 0.5 分；
      - 现走 check_convergence：全轨迹 OLS 去趋势残差 → t₀.₉₇₅(m−2)·noise = ci_half，
        与 runner 收敛判据、compute_security_boundary 的 confidence 完全同源。
    返回 [0, 0.99]：ci_half→0 满分，ci_half≥CONV_CI_TARGET 归零；m<3 不可估计时返回 0.0。
    """
    try:
        from llmsec.evaluation.elo import ELOTracker
        from llmsec.params import CONV_CI_TARGET

        defender_ratings = state.get("defender_ratings") or {}
        if not defender_ratings:
            return None
        defender = next(iter(defender_ratings))

        round_elos = (state.get("round_defender_elos") or {}).get(defender) or []
        if not round_elos:
            return None

        tracker = ELOTracker()
        tracker._round_defender_elos[defender] = list(round_elos)
        tracker.defender_ratings[defender] = defender_ratings[defender]

        total = max(1, len(state.get("attacker_ratings", {})))
        n_gt = len(state.get("ground_truth", {}) or [])

        conv = tracker.check_convergence(
            defender, total_methods=total, tested_count=n_gt
        )
        ci_half = conv.get("ci_half")
        if ci_half is None:
            return 0.0
        score = max(0.0, 1.0 - ci_half / CONV_CI_TARGET)
        return round(min(score, 0.99), 4)
    except Exception as _e:
        logger.warning("降级: %s", _e)
        return None


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
    runs = _discover_runs()
    for r in runs:
        # 富化：带报告的批次附 target_model/security_level/asr，
        # 供批次下拉渲染成"带等级印章的列表"（安/警/伤小方印排在批次名前）
        if r.get("has_report"):
            r.update(_run_meta(RUNS_DIR / r["name"]))
    return {"runs": runs}


@app.get("/api/trend")
async def api_trend(target: str | None = None):
    """跨批次安全趋势：返回所有带报告批次的 {time,target,asr,fpr,elo,level} 序列。

    单次服务端循环替代前端 N 次 /api/overview 轮询，供"安全边界随时间/批次的
    漂移曲线"使用。可选 ?target= 过滤成单目标单条线。

    targets 列表始终返回全部可用目标（不受 ?target= 过滤影响），让前端 chips
    不会因选中某个模型后消失。local_sim 测试目标被过滤掉。
    """
    _LOCAL_SIM_RE = re.compile(r"local.*sim", re.I)
    points: list[dict] = []
    targets_seen: list[str] = []
    for r in _discover_runs():
        if not r.get("has_report"):
            continue
        summ = _run_summary(RUNS_DIR / r["name"])
        if summ is None:
            continue
        tg = summ["target"]
        # 过滤 local_sim 冒烟测试目标（内部测试产物，不是真实评估）
        if tg and _LOCAL_SIM_RE.search(tg):
            continue
        # targets 列表收集在过滤之前：确保 chips 始终有全部真实目标
        if tg and tg not in targets_seen:
            targets_seen.append(tg)
        # ?target= 只影响 points（曲线数据），不影响 targets 列表
        if target and tg != target:
            continue
        points.append({
            "run": summ["run"],
            "time": summ["time"],
            "target": tg,
            "asr": summ["asr"],
            "fpr": summ["fpr"],
            "elo": summ["elo"],
            "level": summ["level"],
        })
    # 按批次名（=时间）升序，使曲线为正确的时间序列
    points.sort(key=lambda x: x["run"])
    return {"trend": points, "targets": sorted(targets_seen)}


@app.get("/api/overview")
async def api_overview(run: str | None = None):
    run_dir = _run_dir(run)
    if run_dir is None:
        return {
            "available": False,
            "reason": "no_runs",
            "message": "尚未进行任何评估运行",
            "runs": [],
        }

    report = load_json(run_dir / "runner_report.json")
    if not report:
        return {
            "available": False,
            "reason": "run_no_report",
            "message": f"批次 {run_dir.name} 未生成报告（评估未完成或失败）",
            "run": run_dir.name,
        }
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

    # stale 检测：仅当存在"更新的且带报告的批次"时提示（批次名 = 时间戳，字典序即时间序），
    # 不再按 mtime 差判定——mtime 只看新旧，历史批次恒被误报"看旧了"
    reason = None
    message = None
    newer = next(
        (r["name"] for r in _discover_runs()
         if r["has_report"] and r["name"] > run_dir.name),
        None,
    )
    if newer:
        reason = "stale_report"
        message = f"存在更新批次 {newer}（可在批次下拉切换到最新）"

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
        "reason": reason,
        "message": message,
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
    ground_truth = _gt_set(state)

    def _enrich(item: dict) -> dict:
        method = item.get("method", "")
        elo = round(ratings.get(method, item.get("elo", 1500.0)), 1)
        std = pred_std.get(method)
        tested = method in ground_truth
        # 未测方法的徽标来源：缓存/派生态里带真实来源（如 tree 条目自带 source）则用真实值，
        # 否则标中性的 'predicted'——不再硬编码 svd_ridge 误导徽标
        source = "ground_truth" if tested else (item.get("source") or "predicted")
        return {
            **item,
            "elo": elo,
            "tested": tested,
            "source": source,
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
    ground_truth = _gt_set(state)

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


@app.get("/api/report/download")
async def api_report_download(run: str | None = None, format: str = "md"):
    """报告下载：带 Content-Disposition 的 .md 附件。

    PDF 暂不支持服务端渲染（避免引入重依赖），前端用浏览器打印对话框存 PDF。
    """
    run_dir = _run_dir(run)
    if run_dir is None:
        raise HTTPException(status_code=404, detail="无可用批次")
    md_path = run_dir / "security_report.md"
    if not md_path.exists():
        raise HTTPException(status_code=404, detail=f"批次 {run_dir.name} 无 security_report.md")
    if format.lower() != "md":
        raise HTTPException(status_code=400, detail="当前仅支持 format=md（PDF 请用浏览器打印）")
    content = md_path.read_text(encoding="utf-8")
    fname = f"security_report_{run_dir.name}.md"
    return PlainTextResponse(
        content,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


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
        return {"available": False, "reason": "no_cluster",
                "message": "尚未运行聚类分析"}

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
        msg = "该批次无 SVD-Ridge 预测模型诊断数据（需用新版 pipeline 运行后生成）"
        if analysis.get("svd_ridge_skipped"):
            msg += f"（{analysis['svd_ridge_skipped']}）"
        elif analysis.get("svd_ridge_error"):
            msg += f"（生成时出错: {analysis['svd_ridge_error']}）"
        return {"available": False, "run": run_dir.name if run_dir else None,
                "reason": "no_model", "message": msg}
    return {"available": True, "run": run_dir.name, "svd_ridge": svd}


@app.get("/api/attack-sets")
async def api_attack_sets():
    if not ATTACKS_DIR.exists():
        return {"files": []}
    files = sorted(p.name for p in ATTACKS_DIR.glob("*.jsonl"))
    return {"files": files}


@app.get("/api/targets")
async def api_targets():
    """运行控制页「目标模型」下拉：列出 .env TARGETS 配置的目标名与模型。

    只返回 name / model 两个展示字段，API key 与 base_url 绝不出后端。
    .env 缺失或解析失败时返回空列表（前端回退 ".env 默认"）。
    """
    from llmsec.core.config import load_targets
    try:
        targets = [{"name": name, "model": cfg.model} for name, cfg in load_targets().items()]
    except Exception:
        targets = []
    return {"targets": targets}


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
    if not CLUSTER_RESULT_FILE.exists():
        return {"available": False}

    mtime = CLUSTER_RESULT_FILE.stat().st_mtime
    cache_key = (method, mtime)
    if cache_key in _PROJECTION_CACHE:
        return _PROJECTION_CACHE[cache_key]

    try:
        artifacts = joblib.load(CLUSTER_RESULT_FILE)
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

        pca = PCA(n_components=2, random_state=_SEED)
        coords = pca.fit_transform(X)
        result["explained_variance"] = [
            round(float(r), 4) for r in pca.explained_variance_ratio_
        ]
    else:
        from sklearn.manifold import TSNE

        # sklearn 要求 perplexity < n，小样本自适应收缩
        perplexity = max(1, min(30, (n - 1) // 3))
        tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=_SEED)
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

    if not CLUSTER_RESULT_FILE.exists():
        return None
    try:
        artifacts = joblib.load(CLUSTER_RESULT_FILE)
    except Exception as _e:
        logger.warning("降级: %s", _e)
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
    methods = sorted(labels.keys())
    n = len(labels)
    dd = dendrogram(Z, no_plot=True)
    # 叶节点方法名（左→右顺序）：dendrogram 的 leaves 是原始观测索引，对应 sorted(labels)
    leaf_order = dd.get("leaves", [])
    leaves = [methods[i] for i in leaf_order if isinstance(i, int) and 0 <= i < len(methods)]

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
        "leaves": leaves,
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

    artifacts = _load_tree_artifacts()
    if artifacts is None:
        return {"available": False}

    labels = artifacts.get("labels", {})
    n = len(labels)
    if k < 2 or k > n:
        raise HTTPException(status_code=400, detail=f"k 必须在 [2, {n}] 内")

    mtime = CLUSTER_RESULT_FILE.stat().st_mtime
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

# TASKS 上限：新任务入列时淘汰最旧的终态任务（running 不淘汰），防长期运行内存/句柄堆积
_TASKS_MAX = 64
_TERMINAL_STATUSES = ("success", "failed", "cancelled")


def _evict_tasks() -> None:
    """TASKS 超 _TASKS_MAX 时按插入序淘汰最旧的终态任务，并确保其日志句柄关闭。"""
    while len(TASKS) > _TASKS_MAX:
        victim = next(
            (tid for tid, t in TASKS.items() if t["status"] in _TERMINAL_STATUSES),
            None,
        )
        if victim is None:
            break  # 全是 running，不淘汰
        t = TASKS.pop(victim)
        log_file = t.get("log_file")
        if log_file is not None:
            log_file.close()


class EvaluateRequest(BaseModel):
    phase: str = Field(default="all", pattern="^(all|1|2)$")
    input: str = "l1.jsonl"
    # runner._adaptive_batch_size 会把 batch 压到 [ADAPTIVE_BATCH_MIN, ADAPTIVE_BATCH_MAX] 内，
    # 上限与 runner 对齐，避免用户传 >ADAPTIVE_BATCH_MAX 时被静默压回；默认值随上限自适应
    batch_size: int = Field(default=min(10, ADAPTIVE_BATCH_MAX), ge=1, le=ADAPTIVE_BATCH_MAX)
    max_rounds: int = Field(default=5, ge=1, le=50)
    sampler: str = Field(default="hybrid", pattern="^(gap|infogain|coordinate|hybrid)$")
    # 目标模型（.env TARGETS 中声明的名字）；None = .env 默认目标。
    # pattern 防异常字符（argv 以列表传递不走 shell，仍做白名单校验）
    target: str | None = Field(default=None, pattern=r"^[\w.\-:]+$")


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
    _evict_tasks()
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
    if req.target:
        # 目标须在 .env TARGETS 已声明，否则 400（静默丢弃会张冠李戴）；
        # load_targets 失败/为空时无法校验，放行交由 runner 自身报错
        from llmsec.core.config import load_targets
        try:
            declared = load_targets()
        except Exception:
            declared = {}
        if declared and req.target not in declared:
            raise HTTPException(status_code=400, detail=f"目标未在 TARGETS 中声明: {req.target!r}")
        argv += ["--target", req.target]
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


@app.get("/api/tasks/{task_id}/log")
async def api_task_log(task_id: str, download: bool = False):
    """完整任务日志（log_tail 只有尾部 4KB；任务失败后看完整上下文用此接口）。

    ?download=1 时以 text/plain + Content-Disposition 返回，便于直接下载 .log。
    """
    t = TASKS.get(task_id)
    if t is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    log_path: Path = t["log_path"]
    text = ""
    if log_path.exists():
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
    if download:
        return PlainTextResponse(
            text,
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{task_id}.log"'},
        )
    return {"id": task_id, "log": text}


@app.post("/api/tasks/{task_id}/cancel")
async def api_task_cancel(task_id: str):
    """取消运行中的任务：SIGTERM → 5s 宽限 → SIGKILL，置 cancelled 状态。

    Windows 无 SIGTERM 语义：Popen.terminate 即 TerminateProcess 强杀，宽限期仅对 POSIX 有效。
    proc.wait 经 asyncio.to_thread 包裹，避免同步等待阻塞事件循环。
    runner 每场攻击实时 upsert 进 R，故取消后已观测的结果保留在结果矩阵中。
    已结束的任务返回 409。
    """
    t = TASKS.get(task_id)
    if t is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    _refresh_task_status(t)
    if t["status"] != "running":
        raise HTTPException(status_code=409, detail=f"任务已结束（{t['status']}），无法取消")
    proc: subprocess.Popen = t["proc"]
    proc.terminate()
    try:
        await asyncio.to_thread(proc.wait, timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        await asyncio.to_thread(proc.wait)
    t["status"] = "cancelled"
    t["returncode"] = proc.returncode
    log_file = t.get("log_file")
    if log_file is not None:
        log_file.close()
        t["log_file"] = None
    return _task_view(task_id, t)


@app.get("/api/tasks/{task_id}/stream")
async def api_task_stream(task_id: str):
    """SSE 实时日志流：连接时先吐尾部 2KB 上下文，之后跟随新增字节（直播）。

    子进程结束时发一个 event:done（携带 status/returncode）再关闭，前端据此
    停止跟随并刷新数据。取代运行控制页 2~3s 轮询 log_tail 的"看监控录像"体验。
    """
    t = TASKS.get(task_id)
    if t is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    log_path: Path = t["log_path"]

    async def event_gen():
        buf = ""

        def take_lines(text: str) -> list[str]:
            """把文本切成完整行返回，末尾不完整段留作 buf 等下次拼齐。"""
            nonlocal buf
            buf += text
            parts = buf.split("\n")
            buf = parts.pop()  # 最后一段可能不完整，保留
            return parts

        offset = 0
        # 连接初始上下文：尾部 2KB（起始半行丢弃，避免半行噪音）
        if log_path.exists():
            try:
                size = log_path.stat().st_size
                head = max(0, size - 2048)
                with open(log_path, encoding="utf-8", errors="replace") as f:
                    f.seek(head)
                    if head > 0:
                        f.readline()  # 丢弃起始半行
                    init = f.read()
                offset = size
            except OSError:
                init = ""
            for line in take_lines(init):
                yield f"data: {line}\n\n"

        while True:
            if log_path.exists():
                try:
                    size = log_path.stat().st_size
                except OSError:
                    size = offset
                if size > offset:
                    try:
                        with open(log_path, encoding="utf-8", errors="replace") as f:
                            f.seek(offset)
                            chunk = f.read(size - offset)
                        offset = size
                    except OSError:
                        chunk = ""
                    for line in take_lines(chunk):
                        yield f"data: {line}\n\n"
                elif size < offset:
                    # 文件被截断/轮转，重置偏移跟随新内容
                    offset = size
            _refresh_task_status(t)
            if t["status"] != "running":
                # 刷出残留 buffer 后发结束事件
                if buf:
                    yield f"data: {buf}\n\n"
                    buf = ""
                yield (
                    "event: done\ndata: "
                    + json.dumps(
                        {"status": t["status"], "returncode": t.get("returncode")},
                        ensure_ascii=False,
                    )
                    + "\n\n"
                )
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
