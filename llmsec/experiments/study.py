"""
experiments.study — 实验编排：加载配置 → 顺序执行 trials → 断点续跑 → 聚合。

study 目录：output/experiments/<name>/
  study.yaml     配置副本
  trials.jsonl   append-only trial 记录（断点续跑的真相源）
  best.json      report 时写入的最佳 config
"""

from __future__ import annotations

import json
import math
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from llmsec.core.config import OUTPUT_DIR
from llmsec.core.logging import get_logger
from llmsec.core.progress import emit_progress
from llmsec.experiments.executor import run_trial
from llmsec.experiments.metrics import aggregate
from llmsec.experiments.schema import StudyConfig
from llmsec.experiments.search import build_search

logger = get_logger(__name__)
STUDIES_DIR = OUTPUT_DIR / "experiments"


def study_dir(name: str) -> Path:
    return STUDIES_DIR / name


def _fingerprint(params: dict) -> str:
    return json.dumps(params, sort_keys=True, ensure_ascii=False)


def _load_trials(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # 跳过损坏行（断点续跑容错）
    return out


def _effective_targets(config: StudyConfig, config_full: dict) -> list[str]:
    """多目标：config.targets；单目标：fixed.target。空则报错。"""
    if config.targets:
        return list(config.targets)
    t = config_full.get("target")
    return [t] if t else []


def _done_units(trials: list[dict], search_fp: str) -> set[tuple]:
    """该 search config 已成功的 (target, seed) 单元集合。"""
    return {(t.get("target"), t.get("seed")) for t in trials
            if t.get("status") == "success" and t.get("search_fp") == search_fp}


def run_study(config: StudyConfig) -> dict:
    """顺序执行整个 study，支持断点续跑。返回汇总。"""
    sdir = study_dir(config.name)
    sdir.mkdir(parents=True, exist_ok=True)
    trials_path = sdir / "trials.jsonl"

    # 把 study 配置落盘到 study 目录（复现 + report 命令读取）。
    # 每次 run 都刷新：源 yaml 改过后续跑仍用旧副本会导致 report 读到过期配置（S-7 残留）
    cfg_copy = sdir / "study.yaml"
    src_path = getattr(config, "_source_path", None)
    if src_path:
        try:
            import shutil

            shutil.copyfile(src_path, cfg_copy)
        except OSError as e:
            logger.warning(f"⚠ study.yaml 拷贝失败（{e}），report 命令可能读到旧配置")
    elif not cfg_copy.exists():
        logger.info("ℹ 配置来自 from_dict（无源 yaml），未落盘 study.yaml——report 命令将不可用")

    completed = _load_trials(trials_path)
    seeds = [config.seed_base + i for i in range(config.repeats)]
    trials_lock = threading.Lock()   # 并发 trial 写 trials.jsonl 的互斥

    # fail-fast：无目标可跑时直接报错，不再逐 config 跳过 + 误报"搜索空间已穷尽"空转
    if not _effective_targets(config, config.fixed):
        raise ValueError(f"study '{config.name}' 无有效目标：targets 与 fixed.target 均为空")

    # 已完成的独立 config 数（一个 config 需 targets×seeds 个成功单元）
    done_search_fps = {t.get("search_fp") for t in completed if t.get("status") == "success"}
    counted_fps: set[str] = set()
    configs_done = 0
    # 估算每 config 的单元总数（单目标时 targets 取 fixed.target）
    n_targets = len(config.targets) if config.targets else (1 if config.fixed.get("target") else 0)
    units_per_config = max(1, n_targets) * config.repeats
    for fp in done_search_fps:
        if len(_done_units(completed, fp)) >= units_per_config:
            configs_done += 1
            counted_fps.add(fp)

    logger.info(f"🧪 study='{config.name}'  strategy={config.strategy}  budget={config.budget_max_trials}  "
          f"repeats={config.repeats}  targets={n_targets}  并发={config.max_concurrent}  config并发={config.config_concurrency}  已完成 config={configs_done}")
    logger.info(f"   目标: {config.objective.direction} {config.objective.metric} ({config.objective.aggregate})"
          f"  {'跨目标均值' if n_targets > 1 else ''}")

    engine = build_search(config, _completed_for_engine(completed, config))

    trial_idx = max([t.get("trial", 0) for t in completed], default=0)
    session_trials = 0
    consecutive_failures = 0   # 连续失败/超时计数；>=3 中止（防系统性故障空转烧 API）
    _FAIL_ABORT = 3
    _stall_count = 0           # 连续"空批"计数：整批 config 全是已完成重提；>=3 中止
    _STALL_LIMIT = 3
    started_at = datetime.now()
    wall_cap_s = (config.budget_max_wall_minutes or 0) * 60
    _safety_cap = config.budget_max_trials * units_per_config * 3 + 10
    best_obj: float | None = None   # 截至已完成 config 的最佳目标值（看板进度展示）
    _trial_total_est = config.budget_max_trials * units_per_config  # trial 总数估算（看板分母）

    K = max(1, config.config_concurrency)
    while configs_done < config.budget_max_trials:
        if session_trials > _safety_cap:
            logger.warning(f"⚠ 安全保险触发：已跑 {session_trials} 个 trial 超上限 {_safety_cap}，中止（疑似完成判定 bug）")
            break
        if wall_cap_s and (datetime.now() - started_at).total_seconds() > wall_cap_s:
            logger.info(f"⏰ 墙钟上限触发：已运行 {config.budget_max_wall_minutes} 分钟，中止（已完成 {configs_done} config）")
            break

        configs_done_before = configs_done   # 本批起点（stall 检测用）

        # --- 询问本批 K 个 config（不超预算余量；grid 耗尽则提前收）---
        remaining = config.budget_max_trials - configs_done
        k_this = min(K, remaining) if remaining > 0 else 0
        batch_params: list[dict] = []
        for _ in range(k_this):
            sp = engine.ask()
            if sp is None:
                break
            batch_params.append(sp)
        if not batch_params:
            logger.info("   搜索空间耗尽（grid）或预算已满。")
            break

        # --- 为本批每个 config 构建 (target, seed) 单元 ---
        units = []   # (tgt, sd, tidx, cf, wd, search_fp, search_params)
        cfgs = []    # [{"search_fp","search_params"}]
        for search_params in batch_params:
            search_fp = _fingerprint(search_params)
            config_base = {**config.fixed, **search_params}
            _inp = config_base.get("input")
            if _inp and "/" not in str(_inp) and "\\" not in str(_inp):
                config_base["input"] = f"attacks/{_inp}"
            targets = _effective_targets(config, config_base)
            if not targets:
                logger.warning(f"⚠ 无目标（config {search_params}），跳过该 config")
                continue
            done = _done_units(completed, search_fp)
            pending = [(tgt, sd) for tgt in targets for sd in seeds if (tgt, sd) not in done]
            for (tgt, sd) in pending:
                trial_idx += 1
                session_trials += 1
                cf = dict(config_base)
                cf["target"] = tgt
                units.append((tgt, sd, trial_idx, cf, sdir / f"trial_{trial_idx}_{tgt}_{sd}",
                              search_fp, search_params))
            cfgs.append({"search_fp": search_fp, "search_params": search_params})

        ran_any = False
        abort_study = False
        if units:
            cfg_range = f"#{configs_done + 1}" if len(cfgs) == 1 else f"#{configs_done + 1}..{configs_done + len(cfgs)}"
            # 并发池 = K × max_concurrent（不同 config 的单元混跑，吃满 GPU 富余吞吐；探针证实 .95:8000
            # 连续批处理，吞吐随并发上升、零 429，故不必限流）。封顶 len(units) 免空线程
            workers = max(1, min(K * config.max_concurrent, len(units)))
            logger.info(f"\n▶ config {cfg_range}  (batch={len(cfgs)}, 单元={len(units)}, 并发池={workers})"
                  f"  search={batch_params[0]}{' …' if len(cfgs) > 1 else ''}")
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(run_trial, cf, sd, wd, config.name, tidx,
                                       config.trial_timeout_minutes): (tgt, sd, tidx, cf, search_fp, search_params)
                           for (tgt, sd, tidx, cf, wd, search_fp, search_params) in units}
                for fut in as_completed(futures):
                    tgt, sd, tidx, cf, search_fp, search_params = futures[fut]
                    try:
                        rec = fut.result()
                    except Exception as e:
                        rec = {"params": cf, "status": "error",
                               "error": f"{type(e).__name__}: {e}", "metrics": {}}
                    # 补登记 target/seed/search_fp（run_trial 不知情多目标语义）
                    rec["target"] = tgt
                    rec["seed"] = sd
                    rec["trial"] = tidx
                    rec["search_fp"] = search_fp
                    rec["search_params"] = search_params
                    with trials_lock:
                        with open(trials_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                        completed.append(rec)
                        ran_any = True
                        if rec.get("status") in ("failed", "timeout", "error"):
                            consecutive_failures += 1
                        else:
                            consecutive_failures = 0
                    logger.info(f"   [{tgt}/seed{sd}] {rec.get('status')} "
                          f"{config.objective.metric}={(rec.get('metrics') or {}).get(config.objective.metric)}")
                    # 进度落盘：每个 trial 完成即汇报（看板实时 trial 计数 + 已知最佳 + 逐 trial 明细）
                    emit_progress({
                        "phase": "hpo",
                        "trial_done": len(completed),
                        "trial_total_est": _trial_total_est,
                        "configs_done": configs_done,
                        "configs_total": config.budget_max_trials,
                        "best_metric": best_obj,
                        "metric_name": config.objective.metric,
                        "direction": config.objective.direction,
                        "last": {"target": tgt, "seed": sd, "status": rec.get("status"),
                                 "value": (rec.get("metrics") or {}).get(config.objective.metric),
                                 "params": search_params},
                    })
                    if consecutive_failures >= _FAIL_ABORT:
                        logger.warning(f"⚠ 连续 {consecutive_failures} 个 trial 失败/超时，中止 study（疑似系统性故障）")
                        # 告警（监控设施故障不影响 study 中止逻辑）
                        try:
                            from llmsec.core.monitoring import alert_study_aborted

                            alert_study_aborted(
                                study_name=config.name,
                                consecutive_failures=consecutive_failures,
                                detail=f"连续 {consecutive_failures} 个 trial 失败/超时，study '{config.name}' 已中止（疑似系统性故障）。",
                            )
                        except Exception:
                            pass
                        abort_study = True
                    # wall-clock 检查（repeat 单元内部）：原仅在 config 边界检查，
                    # max_concurrent=1 顺序 repeats 时可能在下一个边界检查前已超时仍空转烧 API
                    if wall_cap_s and (datetime.now() - started_at).total_seconds() > wall_cap_s:
                        logger.info(f"⏰ 墙钟上限触发：已运行 {config.budget_max_wall_minutes} 分钟，"
                              f"中止（已完成 {configs_done} config，当前 batch 部分完成）")
                        abort_study = True
                    # 中止时取消尚未开始的 pending futures（已提交但排队中的）。
                    # 不 break：仍在运行的 trial 结果照常收集，防在途 API 成本白花、续跑重跑
                    if abort_study:
                        for f in futures:
                            f.cancel()

        # --- tell 本批每个 config + 计数（成功判定与串行版一致）---
        for cfg in cfgs:
            search_fp = cfg["search_fp"]
            search_params = cfg["search_params"]
            vals = [(t.get("metrics") or {}).get(config.objective.metric)
                    for t in completed
                    if t.get("search_fp") == search_fp and t.get("status") == "success"]
            vals = [float(v) for v in vals if isinstance(v, (int, float))]
            if not vals:
                logger.warning(f"   ⚠ 无有效目标值，跳过该 config 的 tell ({search_fp[:40]}…)")
                obj = float("inf") if config.objective.direction == "minimize" else float("-inf")
            else:
                obj = aggregate(vals, config.objective.aggregate)
            # optuna 对非有限目标值敏感（部分采样器拒收 inf）：用有界大数哨兵
            # 保持"最差"排序语义，避免 tell 抛错中断整个 study
            tell_obj = obj
            if not math.isfinite(tell_obj):
                tell_obj = 1e18 if tell_obj > 0 else -1e18
            engine.tell(search_params, tell_obj)
            logger.info(f"   ⇒ {config.objective.metric}={obj:.3f} (跨 {len(vals)} 单元)")
            # 更新最佳目标值（仅在有有效单元时；inf/-inf 兜底不计入）
            if vals:
                if best_obj is None or (
                    (config.objective.direction == "maximize") == (obj > best_obj)
                ):
                    best_obj = obj
            if len(_done_units(completed, search_fp)) >= units_per_config or not ran_any:
                if search_fp not in counted_fps:
                    counted_fps.add(search_fp)
                    configs_done += 1

        # stall 检测：整批全是已完成的重提（搜索空间不同 combo 数 < budget，引擎无法再产生新 config）。
        # bayesian/random 的 ask() 永不返回 None，grid 靠 ask 返回 None 终止故不受影响。
        # 触发条件：本批没跑任何单元(ran_any=False)且 configs_done 未增——即所有建议 config 均已计数过。
        if not ran_any and configs_done == configs_done_before:
            _stall_count += 1
            if _stall_count >= _STALL_LIMIT:
                logger.info(f"ℹ 搜索空间已穷尽（连续 {_STALL_LIMIT} 批全部重提已完成 config，无法产生新 config），"
                      f"停止：已完成 {configs_done}/{config.budget_max_trials}（budget 大于空间去重后规模属正常）")
                break
        else:
            _stall_count = 0

        if abort_study or (wall_cap_s and (datetime.now() - started_at).total_seconds() > wall_cap_s):
            break

    logger.info(f"\n✅ study 完成：{configs_done} 个 config × {units_per_config} 单元（targets×seeds）")
    return summarize(config)


def _completed_for_engine(trials: list[dict], config: StudyConfig) -> list[dict]:
    """喂给 bayesian 的已完成 (search_params, objective) 列表（每 search config 取代表值）。"""
    by_fp: dict[str, list] = {}
    params_by_fp: dict[str, dict] = {}
    for t in trials:
        if t.get("status") != "success":
            continue
        fp = t.get("search_fp") or _fingerprint(t.get("params", {}))
        mv = (t.get("metrics") or {}).get(config.objective.metric)
        if mv is not None:
            by_fp.setdefault(fp, []).append(float(mv))
            params_by_fp.setdefault(fp, t.get("search_params") or t.get("params", {}))
    out = []
    for fp, vals in by_fp.items():
        out.append({"params": params_by_fp[fp], "objective": aggregate(vals, config.objective.aggregate)})
    return out


def summarize(config: StudyConfig) -> dict:
    """聚合所有 trial，按 search config 分组，排名；多目标时附每目标拆分。"""
    trials = _load_trials(study_dir(config.name) / "trials.jsonl")
    by_fp: dict[str, list[dict]] = {}
    for t in trials:
        fp = t.get("search_fp") or _fingerprint(t.get("params", {}))
        by_fp.setdefault(fp, []).append(t)

    rows = []
    for _fp, ts in by_fp.items():
        succ = [t for t in ts if t.get("status") == "success"]
        if not succ:
            continue
        row = {"params": succ[0].get("search_params") or succ[0]["params"], "n_success": len(succ)}
        # 各指标 mean±std（跨目标+seed 聚合）
        metric_names = {k for t in succ for k in (t.get("metrics") or {})}
        for m in metric_names:
            vals = [(t.get("metrics") or {}).get(m) for t in succ]
            vals = [v for v in vals if isinstance(v, (int, float))]
            if vals:
                row[f"{m}_mean"] = statistics.mean(vals)
                row[f"{m}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        # 每目标拆分（多目标时高亮异类模型）
        per_target: dict[str, dict] = {}
        for t in succ:
            tgt = t.get("target", "?")
            mv = (t.get("metrics") or {}).get(config.objective.metric)
            if isinstance(mv, (int, float)):
                per_target.setdefault(tgt, []).append(float(mv))
        if per_target:
            row["per_target"] = {tgt: round(statistics.mean(vs), 3) for tgt, vs in per_target.items()}
        rows.append(row)

    rev = config.objective.direction == "maximize"
    metric_mean = f"{config.objective.metric}_mean"
    rows.sort(key=lambda r: (r.get(metric_mean) if r.get(metric_mean) is not None
                             else (float("-inf") if rev else float("inf"))), reverse=rev)

    best = rows[0] if rows else None
    summary = {"name": config.name, "objective": config.objective.metric,
               "direction": config.objective.direction, "best": best, "rows": rows}
    (study_dir(config.name) / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary
