#!/usr/bin/env python3
"""
LLMSEC 安全评估 Web 面板（FastAPI + 原生 HTML/JS）

提供只读可视化界面，同时预留后端 API 便于后续扩展为操作面板
（触发评估、聚类、生成报告等）。

启动:
    cd C:/Users/LPF/Desktop/LLM攻击测试
    .venv/Scripts/uvicorn llmsec.server.dashboard_api:app --reload --port 8080

访问:
    http://localhost:8080
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ============================================================
# 路径
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "llmsec" / "output"
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

app = FastAPI(title="LLMSEC Dashboard")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

# ============================================================
# 数据加载（与 dashboard.py 保持一致）
# ============================================================
def discover_result_files() -> list[Path]:
    if not OUTPUT_DIR.exists():
        return []
    return sorted(OUTPUT_DIR.glob("*_结果.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)


def discover_summary_files() -> list[Path]:
    if not OUTPUT_DIR.exists():
        return []
    return sorted(OUTPUT_DIR.glob("*_汇总.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def load_jsonl(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def derive_obfuscation(method: str) -> str:
    if method.endswith("_b64"):
        return "b64"
    if method.endswith("_rot13"):
        return "rot13"
    if method.endswith("_code"):
        return "code"
    if method.endswith("_story"):
        return "story"
    return "raw"


def resolve_summary(result_file: str) -> Path | None:
    result_path = OUTPUT_DIR / result_file
    prefix = result_path.stem.replace("_结果", "")
    for p in discover_summary_files():
        if p.stem.startswith(prefix):
            return p
    return None


def build_results_df(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    for col in ["is_harmful", "is_refusal", "eval_score", "latency_ms", "jailbreak_tax", "token_ratio"]:
        if col not in df.columns:
            df[col] = None
    if "method" not in df.columns:
        df["method"] = "unknown"
    df["obfuscation"] = df["method"].apply(derive_obfuscation)
    df["behavior"] = df.get("id", "").apply(lambda x: str(x).rsplit("-", 1)[0] if "-" in str(x) else x)
    return df


def build_method_stats(df: pd.DataFrame) -> list[dict]:
    if df.empty or "method" not in df.columns:
        return []
    rows = []
    for method, g in df.groupby("method"):
        total = len(g)
        harmful = int(g["is_harmful"].sum())
        taxes = g[g["jailbreak_tax"].notna()]["jailbreak_tax"]
        rows.append({
            "method": method,
            "total": total,
            "harmful": harmful,
            "asr": round(harmful / total, 4) if total else 0,
            "obfuscation": derive_obfuscation(method),
            "mean_tax": round(float(taxes.mean()), 2) if not taxes.empty else 0.0,
        })
    rows.sort(key=lambda x: x["asr"], reverse=True)
    return rows


def build_behavior_stats(df: pd.DataFrame) -> list[dict]:
    if df.empty or "behavior" not in df.columns:
        return []
    rows = []
    for behavior, g in df.groupby("behavior"):
        total = len(g)
        harmful = int(g["is_harmful"].sum())
        rows.append({
            "behavior": behavior,
            "total": total,
            "success": harmful,
            "asr": round(harmful / total, 4) if total else 0,
            "any_success": harmful > 0,
        })
    rows.sort(key=lambda x: x["asr"], reverse=True)
    return rows


def load_elo() -> dict[str, float]:
    data = load_json(OUTPUT_DIR / "state" / "elo.json")
    return data.get("attacker_ratings") or data.get("ratings") or {}


# ============================================================
# API 路由
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "dashboard.html", {})


@app.get("/api/results")
async def api_results():
    files = [p.name for p in discover_result_files()]
    return {"results": files}


@app.get("/api/summary")
async def api_summary(result: str):
    result_path = OUTPUT_DIR / result
    summary_path = resolve_summary(result)
    records = load_jsonl(result_path)
    summary = load_json(summary_path)
    df = build_results_df(records)

    core = summary.get("core_metrics", {})
    total = core.get("total_runs", len(df))
    harmful = core.get("harmful_count", int(df["is_harmful"].sum()) if not df.empty else 0)
    refusal = core.get("refusal_count", int(df["is_refusal"].sum()) if not df.empty else 0)
    asr = core.get("asr", harmful / total if total else 0)

    elo_boundary = summary.get("elo", {}).get("security_boundary", {})
    allergy_summary = summary.get("allergy_summary", {})
    fpr = allergy_summary.get("false_positive_rate")
    if fpr is None:
        fpr = summary.get("fpr", 0)

    harm_asr = summary.get("cross_category", {}).get("harm_type_asr", {})
    if not harm_asr and not df.empty and "harm_type" in df.columns:
        harm_asr = df.groupby("harm_type")["is_harmful"].mean().to_dict()

    obs_asr = {}
    if not df.empty and "obfuscation" in df.columns:
        obs_asr = df.groupby("obfuscation")["is_harmful"].mean().to_dict()

    return {
        "result_file": result,
        "summary_file": summary_path.name if summary_path else None,
        "total": total,
        "harmful": harmful,
        "refusal": refusal,
        "asr": round(asr, 4),
        "latency_mean_ms": round(core.get("latency_mean_ms", df["latency_ms"].mean() if not df.empty else 0), 1),
        "token_inflation_ratio": round(core.get("token_inflation_ratio", df["token_ratio"].mean() if not df.empty else 0), 2),
        "boundary_elo": elo_boundary.get("boundary_elo"),
        "boundary_confidence": elo_boundary.get("confidence", 0),
        "fpr": fpr,
        "harm_type_asr": harm_asr,
        "obfuscation_asr": obs_asr,
    }


@app.get("/api/methods")
async def api_methods(result: str):
    result_path = OUTPUT_DIR / result
    records = load_jsonl(result_path)
    df = build_results_df(records)
    method_stats = build_method_stats(df)
    elo = load_elo()
    for row in method_stats:
        row["elo"] = round(elo.get(row["method"], 1500.0), 1)
    return {"methods": method_stats}


@app.get("/api/behaviors")
async def api_behaviors(result: str):
    result_path = OUTPUT_DIR / result
    records = load_jsonl(result_path)
    df = build_results_df(records)
    return {"behaviors": build_behavior_stats(df)}


@app.get("/api/clusters")
async def api_clusters():
    report = load_json(OUTPUT_DIR / "cluster_report.json")
    if not report:
        return {"clusters": []}
    profiles = report.get("cluster_profiles", {})
    names = report.get("cluster_names", {})
    clusters = []
    for cid, prof in profiles.items():
        clusters.append({
            "id": cid,
            "name": names.get(cid, prof.get("name", f"簇{cid}")),
            "size": prof.get("size", 0),
            "members": prof.get("members", []),
        })
    return {
        "validation": report.get("validation", {}),
        "method_count": report.get("method_count", 0),
        "n_clusters": report.get("n_clusters", 0),
        "clusters": clusters,
    }


@app.get("/api/elo")
async def api_elo():
    elo = load_elo()
    ranking = sorted([{"method": k, "elo": round(v, 1)} for k, v in elo.items()],
                     key=lambda x: x["elo"], reverse=True)
    return {"total": len(ranking), "ranking": ranking}
