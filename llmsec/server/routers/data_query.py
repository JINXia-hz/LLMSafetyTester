"""数据查询路由（只读）：总览 / 趋势 / 威胁 / ELO / 报告 / 聚类摘要 / 模型诊断等。"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from llmsec.core.caches import SigCache
from llmsec.core.config import ATTACKS_DIR, CLUSTER_RESULT_FILE, OUTPUT_DIR
from llmsec.core.io import read_json
from llmsec.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()

from llmsec.storage.contract import RUN_NAME_RE  # noqa: E402 -- 命名契约单一来源


def _runs_dir() -> Path:
    """RUNS_DIR 经 dashboard_api 间接读取。

    测试通过 monkeypatch dashboard_api.RUNS_DIR 重定向运行目录，若本模块在
    import 期绑定 RUNS_DIR 值则 patch 不生效；故每次调用现取 dashboard_api 上的
    当前值。
    """
    from llmsec.server import dashboard_api

    return dashboard_api.RUNS_DIR


# ============================================================
# 通用工具
# ============================================================
def load_json(path: Path | None) -> dict:
    """加载 JSON 文件（委托 core.io.read_json，缺失/坏文件返回 {}）。"""
    return read_json(path, default={}) if path is not None else {}


def _validate_run(run: str) -> str:
    """run 可以是 'YYYY-MM-DD_HHMMSS'（旧格式）或 'YYYY-MM-DD_HHMMSS/target'（新格式）。

    逐段校验防路径穿越——此前只校验首段，'2026-01-01_000000/../../x'
    可通过后拼出越界路径（对齐 management/runs.py 的 safe_subpath 做法）。
    """
    parts = run.split("/")
    if not parts or not RUN_NAME_RE.match(parts[0]):
        raise HTTPException(status_code=400, detail=f"非法 run 参数: {run!r}")
    from llmsec.core.paths import safe_subpath
    try:
        safe_subpath(_runs_dir(), *parts)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"非法 run 参数: {run!r}")
    return run


def _run_dir(run: str | None) -> Path | None:
    """解析 run 参数为目录；支持 'ts/target' 和 'ts' 两种格式；缺省取最新。"""
    runs_dir = _runs_dir()
    if run:
        _validate_run(run)
        d = runs_dir / run
        return d if d.is_dir() else None
    runs = _discover_runs()
    for r in runs:
        if r["has_report"]:
            return runs_dir / r["name"]
    return runs_dir / runs[0]["name"] if runs else None



def _run_name(run_dir: Path | None, run: str | None = None) -> str:
    """返回完整的 run 名（如 'ts/target'）。run 参数有值时直接用，否则从路径推导。"""
    if run:
        return run
    if run_dir is None:
        return ""
    from llmsec.server.dashboard_api import RUNS_DIR
    try:
        return str(run_dir.relative_to(RUNS_DIR)).replace("\\", "/")
    except ValueError:
        return run_dir.name


def _discover_runs() -> list[dict]:
    """列出 run（dashboard 口径：只认有 runner_report.json 的）。

    storage 重构：目录扫描收敛为 storage.catalog 单一实现；本函数退化为口径
    适配层。增量对账（reconcile）按 run 目录 mtime 感知新增/变更——包括 batch
    内部新增 target 子目录（原 H4 的缓存盲区）与同秒撞名 `_2` 后缀目录
    （原 RUN_NAME_RE 带 $ 锚的不可见裂缝）。
    """
    from llmsec.storage import contract as _storage

    return [r.as_dict() for r in _storage.query_runs(runs_root=_runs_dir(), has_report=True)]


def _run_time(run_name: str) -> str | None:
    """运行目录名 YYYY-MM-DD_HHMMSS → ISO 时间戳；非法/解析失败返回 None。"""
    if not RUN_NAME_RE.match(run_name):
        return None
    try:
        return datetime.strptime(run_name, "%Y-%m-%d_%H%M%S").isoformat()
    except ValueError:
        return None


def _row_summary(row) -> dict | None:
    """从目录库登记行构造趋势摘要（库行 metrics 即 extract_report_metrics 的落库结果）。

    取代旧 _run_summary/_run_meta/_RUN_META_CACHE 三件：重解析 runner_report.json
    与按 (mtime,size) 的缓存签名字号全部冗余——目录库对账本身就是持久化缓存。
    """
    if not row.has_report or not row.metrics:
        return None
    m = row.metrics
    d = row.as_dict()
    return {
        "run": row.name,
        "time": _run_time(row.batch) or d["mtime"],
        "target": row.target_model or row.target,
        "asr": m.get("asr"),
        "fpr": m.get("fpr"),
        "elo": m.get("boundary_elo"),
        "level": row.security_level or "inconclusive",
        "tax_probed": m.get("tax_probed"),
    }


def _load_state(run: str | None = None) -> dict:
    """加载 Elo state。指定 run 时优先读 run 目录内的快照（runner 结束时保存），
    避免全局 state 漂移导致历史批次的实测/预测标记错配。

    无 run 快照时从结果矩阵 R 派生活跃模型的 Elo（唯一真相，经 elo_access 缓存）。"""
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
    """从 state 取 ground_truth 方法集合（dict 形态：{method: {...}}）。"""
    gt = state.get("ground_truth", {})
    return set(gt.keys())


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


def _load_tree_artifacts() -> dict | None:
    """加载含 linkage 的聚类 artifacts；不存在、缺 linkage 或 linkage 为抽样子集树
    （叶索引不对应全量方法）时返回 None——树图/切层视图对这些情形降级。"""
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
    if artifacts.get("tree_subsampled"):
        return None
    return artifacts


# ============================================================
# 数据 API
# ============================================================
@router.get("/api/runs")
async def api_runs():
    runs = _discover_runs()
    # 进行中标注：有 evaluate 任务在跑时，批次 ts ≥ 任务 started_at 的 run 标 active，
    # 前端据此渲染 ⏳（多目标运行时先完成的目标报告已落盘，需与"已完成"区分）
    from llmsec.server.task_manager import TASKS
    active_since: str | None = None
    for t in TASKS.values():
        if t.get("kind") == "evaluate" and t.get("status") in ("running", "queued"):
            ts = (t.get("started_at") or "")[:19].replace("T", "_").replace(":", "")
            if ts and (active_since is None or ts < active_since):
                active_since = ts
    for r in runs:
        if active_since and r.get("batch", "") >= active_since:
            r["active"] = True
    # target_model/security_level/asr 已随目录库登记行的 as_dict() 返回
    # （旧 _run_meta 富化与库行重复，已删）
    return {"runs": runs, "total": len(runs)}


@router.get("/api/trend")
async def api_trend(target: str | None = None):
    """跨批次安全趋势：返回所有带报告批次的 {time,target,asr,fpr,elo,level} 序列。

    单次服务端循环替代前端 N 次 /api/overview 轮询，供"安全边界随时间/批次的
    漂移曲线"使用。可选 ?target= 过滤成单目标单条线。

    targets 列表始终返回全部可用目标（不受 ?target= 过滤影响），让前端 chips
    不会因选中某个模型后消失。local_sim 测试目标被过滤掉。
    """
    runs_dir = _runs_dir()
    _LOCAL_SIM_RE = re.compile(r"local.*sim", re.I)
    points: list[dict] = []
    targets_seen: list[str] = []
    from llmsec.storage import contract as _storage

    for row in _storage.query_runs(runs_root=runs_dir, has_report=True):
        summ = _row_summary(row)
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


@router.get("/api/overview")
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
            "message": f"批次 {_run_name(run_dir, run)} 未生成报告（评估未完成或失败）",
            "run": _run_name(run_dir, run),
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
            round(1 - asr, 4) if asr is not None else 0.0,
            round(1 - fpr, 4) if fpr is not None else 0.0,
            round(confidence, 4) if confidence is not None else 0.0,
            round(coverage, 4) if coverage is not None else 0.0,
            conv_score if conv_score is not None else 0.0,
        ],
    }

    harm_type_asr = {
        k: v.get("asr", 0)
        for k, v in tree.get("dimensions", {}).get("by_harm_type", {}).items()
    }

    # stale 检测：仅当存在"更新的且带报告的批次"时提示。
    # 批次名格式为 "时间戳/目标"（如 2026-08-11_151938/gemma-4-12B-it）。原实现按完整批次名
    # 做字典序比较，导致同一时间戳目录下的不同目标互判"更新"（gemma < minimax 字典序 →
    # 看 gemma 时误报"存在更新批次 minimax"）。正确语义是只比较时间戳目录部分：
    # 同一 run 目录下的多目标评估是并行的，不存在先后关系。
    reason = None
    message = None
    cur_name = _run_name(run_dir, run)
    cur_batch = cur_name.split("/", 1)[0]
    newer = next(
        (r["name"] for r in _discover_runs()
         if r["has_report"] and r["name"].split("/", 1)[0] > cur_batch),
        None,
    )
    if newer:
        reason = "stale_report"
        message = f"存在更新批次 {newer}（可在批次下拉切换到最新）"

    return {
        "available": True,
        "run": _run_name(run_dir, run),
        "generated_at": report.get("generated_at"),
        "target_model": report.get("target_model"),
        "overall_verdict": report.get("overall_verdict"),
        "security_level": report.get("security_level", "inconclusive"),
        "recommendation": report.get("recommendation"),
        "asr": round(asr, 4) if asr is not None else None,
        "fpr": round(fpr, 4) if fpr is not None else None,
        "rounds": attack.get("rounds", 0),
        "total_tested": total_tested,
        "successful": attack.get("successful", 0),
        "boundary_elo": overall.get("elo_boundary", elo.get("boundary_elo")),
        "boundary_confidence": round(confidence, 4) if confidence is not None else None,
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


@router.get("/api/threats")
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
        # tree 条目的评级键 = unit（簇指纹）；method 字段已是簇展示名
        key = item.get("unit") or item.get("method", "")
        elo = round(ratings.get(key, item.get("elo", 1500.0)), 1)
        std = pred_std.get(key)
        tested = key in ground_truth
        # 未测单位的徽标来源：缓存/派生态里带真实来源（如 tree 条目自带 source）则用真实值，
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

    # 意外事件的 attacker = unit_id：附簇名供前端直接展示
    # （独立于 _load_tree_artifacts——树图降级不影响 units 查询）
    units: dict = {}
    try:
        import joblib
        if CLUSTER_RESULT_FILE.exists():
            units = (joblib.load(CLUSTER_RESULT_FILE).get("units") or {})
    except Exception as _e:
        logger.warning("降级: %s", _e)
    upsets = tree.get("upsets", {})
    if isinstance(upsets, dict):
        for side in ("weakness", "strength"):
            for ev in upsets.get(side, []):
                uid = ev.get("attacker")
                if uid in units:
                    ev["name"] = units[uid].get("name")

    return {
        "available": True,
        "run": _run_name(run_dir, run),
        "top_threats": [_enrich(t) for t in tree.get("top_threats", [])],
        "strong_defenses": [_enrich(t) for t in tree.get("strong_defenses", [])],
        "upsets": upsets,
    }


@router.get("/api/elo")
async def api_elo(run: str | None = None):
    state = _load_state(run)
    ratings = state.get("attacker_ratings", {})
    pred_std = state.get("attacker_pred_std", {})
    ground_truth = _gt_set(state)

    # 评级键 = unit_id（簇指纹）；簇名/规模取自聚类产物的 units 表
    # （独立于 _load_tree_artifacts——树图降级不影响 units 查询）
    # r7/M-11：joblib.load 可能反序列化全量特征矩阵，放 to_thread 不阻塞事件循环
    units: dict = {}

    def _load_units() -> dict:
        try:
            import joblib
            if CLUSTER_RESULT_FILE.exists():
                return joblib.load(CLUSTER_RESULT_FILE).get("units") or {}
        except Exception as _e:
            logger.warning("降级: %s", _e)
        return {}

    import asyncio

    units = await asyncio.to_thread(_load_units)

    ranking = [
        {
            "unit": m,
            "name": (units.get(m) or {}).get("name", m),
            "size": (units.get(m) or {}).get("size"),
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


@router.get("/api/report-md")
async def api_report_md(run: str | None = None):
    run_dir = _run_dir(run)
    if run_dir is None:
        return {"available": False}
    md_path = run_dir / "security_report.md"
    if not md_path.exists():
        return {"available": False, "run": _run_name(run_dir, run)}
    return {
        "available": True,
        "run": _run_name(run_dir, run),
        "markdown": md_path.read_text(encoding="utf-8"),
    }


@router.get("/api/report/download")
async def api_report_download(run: str | None = None):
    """报告下载：带 Content-Disposition 的 .md 附件。

    PDF 暂不支持服务端渲染（避免引入重依赖），前端用浏览器打印对话框存 PDF。
    """
    run_dir = _run_dir(run)
    if run_dir is None:
        raise HTTPException(status_code=404, detail="无可用批次")
    md_path = run_dir / "security_report.md"
    if not md_path.exists():
        raise HTTPException(status_code=404, detail=f"批次 {_run_name(run_dir, run)} 无 security_report.md")
    content = md_path.read_text(encoding="utf-8")
    fname = 'security_report_' + _run_name(run_dir, run).replace('/', '_') + '.md'
    return PlainTextResponse(
        content,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@router.get("/api/clusters")
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
        "run": _run_name(run_dir, run) if run_dir else None,
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


@router.get("/api/model")
async def api_model(run: str | None = None):
    run_dir = _run_dir(run)
    analysis = load_json(run_dir / "cluster_security_analysis.json") if run_dir else {}
    svd = analysis.get("svd_ridge")
    blend = analysis.get("blend_predictor")
    if not svd and not blend:
        msg = "该批次无预测模型诊断数据（需用新版 pipeline 运行后生成）"
        if analysis.get("svd_ridge_skipped"):
            msg += f"（{analysis['svd_ridge_skipped']}）"
        elif analysis.get("svd_ridge_error"):
            msg += f"（生成时出错: {analysis['svd_ridge_error']}）"
        return {"available": False, "run": _run_name(run_dir, run) if run_dir else None,
                "reason": "no_model", "message": msg}
    return {"available": True, "run": _run_name(run_dir, run) if run_dir else None,
            "svd_ridge": svd, "blend_predictor": blend}


# /api/attack-sets 行数缓存：按 (文件名, mtime+size) 失效。
# 攻击集是用户上传的静态文件（读多写少），逐行计数开销大（实测 attacks/ 共 37MB），
# 文件不变时复用上次计数。（r9/P3-5：SigCache 统一实现）
_ATTACK_SET_CACHE = SigCache(maxsize=128)


def _attack_set_records(p: Path) -> int:
    """攻击集非空行数，按文件 (mtime, size) 签名缓存。"""
    try:
        st = p.stat()
        sig = (st.st_mtime, st.st_size)
    except OSError:
        return 0
    n_records = 0
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n_records += 1
    except OSError:
        pass
    return _ATTACK_SET_CACHE.get(p.name, sig, lambda: n_records)


@router.get("/api/attack-sets")
async def api_attack_sets():
    """列出可用攻击集，含元信息（大小、修改时间、记录数）。"""

    if not ATTACKS_DIR.exists():
        return {"files": []}
    result = []
    for p in sorted(ATTACKS_DIR.glob("*.jsonl"), key=lambda x: x.name):
        size_kb = round(p.stat().st_size / 1024, 1)
        mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        n_records = _attack_set_records(p)
        result.append({"name": p.name, "size_kb": size_kb, "mtime": mtime, "n_records": n_records})
    return {"files": result}


@router.get("/api/targets")
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


class AddTargetRequest(BaseModel):
    name: str
    model: str
    base_url: str
    api_key: str


def _env_lock():
    """跨进程 .env 写锁（r7/M-10）。

    看板 / control 层 env_snapshot / MCP actions 可能并发读-改-写同一 .env：
    无锁时两个并发 RMW 互相覆盖丢 key、并发 add 算出相同 TARGET_<N> 索引互覆。
    基于 filelock（项目核心依赖），与 control.core.locks.cross_process_lock 同机制。
    """
    from filelock import FileLock

    from llmsec.core.config import PROJECT_ROOT
    return FileLock(str(PROJECT_ROOT / ".env.lock"), timeout=10.0)


def _env_quote(value: str) -> str:
    """dotenv 值引号规则（r7/M-10，与 control/core/env_snapshot._serialize_env 一致）。

    含空格/#/换行或为空时加双引号，否则裸写——不加引号的 `#` 会被 dotenv 当注释
    截断、空值会被当"未设置"。
    """
    value = value.strip()
    if any(c in value for c in (" ", "#", "\n")) or not value:
        return f'"{value}"'
    return value


def _backup_env_pre_write(env_path) -> None:
    """r7/M-10：真正的写前备份（旧内容，回滚用）→ .env 同目录 .env.bak。

    原实现唯一的"备份"发生在覆写**之后**（备的是新内容），旧内容永不可恢复。
    另保留写后 OUTPUT_DIR/.env.bak 拷贝（docker entrypoint 从该处恢复最新配置，
    语义不同不能合并）。
    """
    import shutil

    if env_path.exists():
        try:
            shutil.copy2(env_path, env_path.with_name(env_path.name + ".bak"))
        except OSError:
            pass


@router.post("/api/targets/add")
async def api_targets_add(req: AddTargetRequest):
    """「+」加目标：把新目标追加到 .env（TARGET_<N>_* 四件套 + 加入 TARGETS 列表）。

    r7/M-10：整段读-改-写持跨进程锁（防并发 add 算出相同 TARGET_<N> 互覆）；
    写前备份旧内容到 .env.bak（回滚用）；值按 dotenv 规则加引号；原子写
    （保留原有注释/格式，只更新 TARGETS 行 + 追加四件套）。
    api_key 写入 .env（gitignored），响应绝不回显明文。
    """
    import re
    import shutil

    from filelock import Timeout as FileLockTimeout

    from llmsec.core.config import PROJECT_ROOT
    name = req.name.strip()
    if not name or not req.base_url.strip():
        raise HTTPException(400, "name 与 base_url 不能为空")
    env_path = PROJECT_ROOT / ".env"
    if env_path.is_dir():
        raise HTTPException(500, ".env 是目录而非文件——通常是 Docker 挂载了不存在的 .env 导致（请先在宿主机 cp .env.example .env）")

    try:
        with _env_lock():
            # 读（.env 不存在 → 自动新建）
            if env_path.exists():
                try:
                    lines = env_path.read_text(encoding="utf-8").splitlines()
                except OSError as e:
                    raise HTTPException(500, f"读取 .env 失败: {e}")
            else:
                lines = []

            # 找已用最大 TARGET_<N>_ 索引 + 现有 TARGETS 列表
            used_n = []
            targets_line_idx = None
            targets_names = []
            for i, ln in enumerate(lines):
                s = ln.strip()
                if s.startswith("TARGETS="):
                    targets_line_idx = i
                    targets_names = [x.strip() for x in s[len("TARGETS="):].split(",") if x.strip()]
                m = re.match(r"^TARGET_(\d+)_NAME\s*=", ln)
                if m:
                    used_n.append(int(m.group(1)))
            if name in targets_names:
                raise HTTPException(400, f"目标名 {name!r} 已存在于 TARGETS")
            next_n = (max(used_n) + 1) if used_n else 1
            model = (req.model.strip() or name)

            # 更新 TARGETS 行
            new_targets_val = ",".join(targets_names + [name])
            if targets_line_idx is not None:
                lines[targets_line_idx] = f"TARGETS={new_targets_val}"
            else:
                lines.append(f"TARGETS={new_targets_val}")

            # 追加四件套（值按 dotenv 规则加引号）
            block = [
                "",
                f"# ---- 看板新增目标 {name}（TARGET_{next_n}） ----",
                f"TARGET_{next_n}_NAME={_env_quote(name)}",
                f"TARGET_{next_n}_MODEL={_env_quote(model)}",
                f"TARGET_{next_n}_BASE_URL={_env_quote(req.base_url)}",
                f"TARGET_{next_n}_API_KEY={_env_quote(req.api_key)}",
            ]
            lines.extend(block)

            # 写前备份旧内容（回滚用），然后原子写。
            # 注意 .env 是点文件（suffix 为空串），with_suffix 会生成 ".env.env.tmp"，
            # 必须用 with_name 拼接（与 _update_env_vars 的写法一致）。
            _backup_env_pre_write(env_path)
            tmp = env_path.with_name(env_path.name + ".tmp")
            try:
                tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
                tmp.replace(env_path)
            except OSError as e:
                raise HTTPException(500, f"写入 .env 失败: {e}")
    except FileLockTimeout:
        raise HTTPException(503, "另一个 .env 写入正在进行，请稍后重试")

    # 重新加载 env 到当前进程（load_targets 会 load_env，但显式 set 更新本进程 os.environ）
    import os

    from llmsec.core.config import load_env
    os.environ["TARGETS"] = new_targets_val
    os.environ[f"TARGET_{next_n}_NAME"] = name
    os.environ[f"TARGET_{next_n}_MODEL"] = model
    os.environ[f"TARGET_{next_n}_BASE_URL"] = req.base_url.strip()
    os.environ[f"TARGET_{next_n}_API_KEY"] = req.api_key.strip()
    load_env()

    # 持久化到 output 卷（docker 重启后 entrypoint 从此恢复 .env——此处是
    # 新内容的持久化，与写前回滚备份是两个不同语义）
    try:
        from llmsec.core.config import OUTPUT_DIR
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(env_path, OUTPUT_DIR / ".env.bak")
    except OSError:
        pass

    return {"ok": True, "name": name, "model": model, "prefix": f"TARGET_{next_n}"}


# ============================================================
# 环境参数配置（连接配置：默认 TARGET / GENERATOR / JUDGE）
# ============================================================
def _update_env_vars(updates: dict) -> None:
    """更新 .env 中指定 KEY=VALUE（存在则替换，不存在则追加）。

    r7/M-10：整段读-改-写持跨进程锁；写前备份旧内容到 .env.bak（回滚用）；
    值按 dotenv 规则加引号；原子写。注意：只更新传入的 key，其余行/注释原样保留。
    """
    import os
    import shutil

    from filelock import Timeout as FileLockTimeout

    from llmsec.core.config import PROJECT_ROOT
    env_path = PROJECT_ROOT / ".env"
    if env_path.is_dir():
        raise HTTPException(500, ".env 是目录而非文件——通常是 Docker 挂载了不存在的 .env 导致（请先在宿主机 cp .env.example .env）")

    try:
        with _env_lock():
            if env_path.exists():
                try:
                    lines = env_path.read_text(encoding="utf-8").splitlines()
                except OSError as e:
                    raise HTTPException(500, f"读取 .env 失败: {e}")
            else:
                lines = []  # .env 不存在 → 自动新建（下方追加逻辑填充内容）

            keys = set(updates.keys())
            found: set[str] = set()
            for i, ln in enumerate(lines):
                s = ln.lstrip()
                for k in keys:
                    if s.startswith(k + "="):
                        lines[i] = f"{k}={_env_quote(updates[k])}"
                        found.add(k)
                        break
            for k in keys - found:
                lines.append(f"{k}={_env_quote(updates[k])}")

            _backup_env_pre_write(env_path)
            tmp = env_path.with_name(env_path.name + ".tmp")
            try:
                tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
                tmp.replace(env_path)
            except OSError as e:
                raise HTTPException(500, f"写入 .env 失败: {e}")
    except FileLockTimeout:
        raise HTTPException(503, "另一个 .env 写入正在进行，请稍后重试")

    for k, v in updates.items():
        os.environ[k] = v.strip()

    # 持久化到 output 卷（docker 重启后 entrypoint 从此恢复 .env——新内容）
    try:
        from llmsec.core.config import OUTPUT_DIR
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(env_path, OUTPUT_DIR / ".env.bak")
    except OSError:
        pass  # output 卷不可写（如 :ro 挂载）不阻塞功能


def _masked(key: str) -> str | None:
    import os

    v = os.getenv(key, "")
    if not v:
        return None
    if len(v) <= 6:
        return "****"
    return v[:3] + "****" + v[-3:]


@router.get("/api/env")
async def api_env():
    """返回当前连接配置（默认 TARGET / GENERATOR / JUDGE）；api_key 掩码显示。"""
    from llmsec.core.config import load_env
    load_env()
    import os
    return {
        "target": {"base_url": os.getenv("TARGET_BASE_URL", ""), "model": os.getenv("TARGET_MODEL", ""),
                   "api_key_masked": _masked("TARGET_API_KEY")},
        "generator": {"base_url": os.getenv("GENERATOR_BASE_URL", ""), "model": os.getenv("GENERATOR_MODEL", ""),
                      "api_key_masked": _masked("GENERATOR_API_KEY")},
        "judge_model": os.getenv("JUDGE_MODEL", ""),
    }


class EnvUpdate(BaseModel):
    target_base_url: str | None = None
    target_model: str | None = None
    target_api_key: str | None = None          # 留空=不改
    generator_base_url: str | None = None
    generator_model: str | None = None
    generator_api_key: str | None = None
    judge_model: str | None = None


@router.put("/api/env")
async def api_env_put(req: EnvUpdate):
    """更新连接配置到 .env（仅写入提供的字段；api_key 留空则不变）。"""
    mapping = {
        "target_base_url": "TARGET_BASE_URL", "target_model": "TARGET_MODEL", "target_api_key": "TARGET_API_KEY",
        "generator_base_url": "GENERATOR_BASE_URL", "generator_model": "GENERATOR_MODEL",
        "generator_api_key": "GENERATOR_API_KEY", "judge_model": "JUDGE_MODEL",
    }
    updates: dict[str, str] = {}
    for fld, envkey in mapping.items():
        v = getattr(req, fld)
        if v is not None and v.strip():
            updates[envkey] = v.strip()
    if not updates:
        raise HTTPException(400, "未提供任何更新字段")
    _update_env_vars(updates)
    return {"ok": True, "updated": sorted(updates.keys())}


@router.get("/api/targets/probe")
async def api_targets_probe(name: str | None = None):
    """探查全部模型的 API 可通性：目标模型 + generator + judge。

    ?name=xxx 时只探单个目标（用于添加/编辑后即时反馈），不探 services。
    对每个目标发送最轻量请求（OpenAI models.list 或 HTTP GET），5s 超时。
    models.list 非空时顺带校验配置模型名是否在列表（不在 → warning，不判不可达：
    部分端点 list 不全）。
    返回 {targets: [...], services: [...]}，条目 {name, model, reachable, latency_ms, error, warning}。
    api_key / base_url 绝不出后端。
    """

    from llmsec.core.config import (
        GeneratorConfig,
        JudgeConfig,
        load_targets,
    )

    try:
        targets_cfg = load_targets()
    except Exception:
        return {"targets": [], "services": []}

    if name:
        targets_cfg = {k: v for k, v in targets_cfg.items() if k == name}

    # 探活逻辑统一走 llmsec.core.probe（与 MCP probe_targets 同一实现）；
    # 此处只做 async 包装（阻塞 IO 放线程池）
    from llmsec.core.probe import probe_service as _probe_service_sync
    from llmsec.core.probe import probe_target as _probe_target_sync

    async def _probe_one(name, cfg):
        return await asyncio.to_thread(_probe_target_sync, name, cfg)

    async def _probe_service(svc_name: str, cfg, shared: dict):
        """generator / judge 探活（第一段结果跨 service 复用：judge 常与
        generator 同端点同 key，重复 models.list 纯属浪费）。"""
        key = (cfg.base_url or "", cfg.api_key or "")
        if key in shared:
            models_result = shared[key]
        else:
            def _do():
                from llmsec.core.probe import ModelsProbeResult, models_list
                try:
                    latency, ids = models_list(cfg.api_key, cfg.base_url, timeout=5.0)
                    return ModelsProbeResult(latency, ids, None)
                except Exception as e:
                    return ModelsProbeResult(None, None, str(e)[:120])
            models_result = await asyncio.to_thread(_do)
            shared[key] = models_result
        return await asyncio.to_thread(_probe_service_sync, svc_name, cfg, models_result)

    results = await asyncio.gather(*[_probe_one(n, c) for n, c in targets_cfg.items()])
    services: list[dict] = []
    if not name:
        # 全量探活：追加 generator / judge。顺序执行（不并进 gather）：
        # 并发下两个协程会在对方写入 shared 前同时检查缓存，复用失效
        shared: dict = {}
        services.append(await _probe_service("generator", GeneratorConfig.from_env(), shared))
        services.append(await _probe_service("judge", JudgeConfig.from_env(), shared))
    return {"targets": list(results), "services": services}
