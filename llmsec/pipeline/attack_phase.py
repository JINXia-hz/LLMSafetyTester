"""pipeline.attack_phase — Phase 1 自适应攻击测试（ELO 二分搜索）。

从 runner.py 拆出，包含：

  - _inject_predicted_elos
  - _compute_method_set_hash
  - _dedup_attack_results
  - _should_refresh_features
  - _adaptive_batch_size
  - _quick_precluster
  - run_attack_phase

为避免与 runner 形成循环导入，本模块对 runner.py 的模块级运行时依赖
（DEFENDER_NAME / logger / write_jsonl / read_jsonl / evaluate_single 等）
统一在函数体内延迟导入（``from llmsec.pipeline.runner import X``）；对 params.py
常量与 evaluation / cluster_analysis / samplers / tax 等模块在顶层正常导入。
runner.py 底部的兼容性 re-export 区会把这几个名字重新导出，保证
``from llmsec.pipeline.runner import run_attack_phase`` 等历史用法仍然可用。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llmsec.evaluation.elo import ELOTracker
    from llmsec.evaluation.judge import Judge

import hashlib
import time
from pathlib import Path

from llmsec.core.config import INITIAL_ELO
from llmsec.core.seed import get_global_seed
from llmsec.evaluation.cluster_analysis import analyze_clusters, save_cluster_analysis
from llmsec.evaluation.samplers import build_sampler
from llmsec.params import (
    ADAPTIVE_BATCH_MAX,
    ADAPTIVE_BATCH_MIN,
    API_DELAY,
    CONV_CI_TARGET,
    SAMPLER_COORD_MIN_PER_CLUSTER,
    SAMPLER_HYBRID_EXPLORE_ROUNDS,
    SAMPLER_INFOGAIN_ALPHA,
    SAMPLER_INFOGAIN_BETA,
    SAMPLER_INFOGAIN_GAMMA,
    SEED_MIN_COUNT,
)
from llmsec.pipeline.tax import format_tax_line, summarize_jailbreak_tax


# ============================================================
# Phase 1 辅助函数
# ============================================================
def _inject_predicted_elos(tracker: ELOTracker, method_records: dict[str, dict],
                           defender_name: str):
    """
    为所有尚未真实评估的方法注入预测初始 Elo。

    优先 **BlendPredictor**（跨模型：sim-加权 universal + 每模型层，发现层 D+A——
    冷启动从相似 donor 借先验，多模型场景的核心）；特征不可用/失败 → 回退
    ClusterEloPredictor（单模型 SVD-Ridge + 变体兜底）。已真实评估的方法 Elo 不变。
    """
    untested = {
        m: r for m, r in method_records.items()
        if m not in tracker.ground_truth_methods
    }
    if not untested:
        return

    # 优先 BlendPredictor：跨模型 sim-加权（发现层），单模型场景无 donor 自动退化为仅模型层
    predictions: dict | None = None
    artifacts = getattr(tracker.predictor, "artifacts", None) or {}
    feats = artifacts.get("features")
    if feats:
        try:
            from llmsec.core.results import ResultsMatrix
            from llmsec.evaluation.blend_predictor import load_or_fit_blend_predictor

            R = ResultsMatrix.load()
            catalog = list(method_records.keys())
            bp = load_or_fit_blend_predictor(R, feats, method_catalog=catalog)
            predictions = {m: bp.predict(m, defender_name) for m in untested}
        except Exception as e:
            try:
                from llmsec.pipeline.runner import logger
                logger.warning(f"BlendPredictor 预测失败，回退 ClusterEloPredictor: {e}")
            except Exception:
                pass
            predictions = None

    # 回退：ClusterEloPredictor（单模型 SVD-Ridge + 同后缀/同基底变体平均）
    if predictions is None:
        predictions = tracker.predictor.predict_batch(untested)

    for method, pred in predictions.items():
        tracker.attacker_ratings[method] = pred["elo"]
        if pred.get("std") is not None:
            tracker.attacker_pred_std[method] = pred["std"]



def _compute_method_set_hash(methods: list[str]) -> str:
    """计算方法集合的指纹 hash，用于判断攻击集是否发生变化。"""
    content = ",".join(sorted(set(methods)))
    return hashlib.md5(content.encode("utf-8")).hexdigest()



def _dedup_attack_results(rows: list[dict]) -> list[dict]:
    """按 (id, method) 去重攻击明细：同键保留后出现的记录（新结果覆盖旧记录）。

    续跑（resume）预载历史明细后，若某方法因 state 丢失被重测，会产生同键记录，
    此处保证落盘与 ASR 统计口径不重复计数。
    """
    merged: dict[tuple, dict] = {}
    order: list[tuple] = []
    for row in rows:
        # 缺 id/method 的记录用 id(row) 兜底防 (None, method) 错误合并丢数据
        key = (row.get("id", id(row)), row.get("method", id(row)))
        if key not in merged:
            order.append(key)
        merged[key] = row
    return [merged[k] for k in order]



def _should_refresh_features(
    predictor,
    method_records: dict[str, dict],
    force: bool = False,
) -> bool:
    """
    判断启动时是否需要重新提取特征缓存（供 SVD-Ridge / D-optimality）。
    聚类只在测试结束后进行，此处只维护特征缓存。

    触发条件：force=True、无可用 artifacts/features、攻击集方法列表发生变化、
    特征配置指纹（embedding source/model + PCA dim + 特征代码版本）不一致（M-6，
    老缓存无 hash 时刷新一次，刷新后 fit_features 会写入 hash）。
    """
    if force:
        return True
    if predictor.artifacts is None or "features" not in predictor.artifacts:
        return True
    current_hash = _compute_method_set_hash(list(method_records.keys()))
    if predictor.artifacts.get("method_set_hash") != current_hash:
        return True
    # M-6：EMBEDDING_PCA_DIM/EMBEDDING_MODEL 等特征配置变更后旧缓存必须失效，
    # 不能只看方法名。current_feature_config_hash 由 elo_cluster 暴露（动态反映当前配置）。
    try:
        from llmsec.evaluation import elo_cluster as _ec
        current_cfg_hash = getattr(_ec, "current_feature_config_hash", lambda: None)()
    except Exception:
        current_cfg_hash = None
    if current_cfg_hash is not None:
        cached_cfg_hash = (predictor.artifacts.get("meta") or {}).get("feature_config_hash")
        if cached_cfg_hash != current_cfg_hash:
            return True
    return False



def _adaptive_batch_size(
    current_batch: int,
    min_batch: int = ADAPTIVE_BATCH_MIN,
    max_batch: int = ADAPTIVE_BATCH_MAX,
) -> tuple[int, str]:
    """
    返回下一轮 batch_size。

    设计变更：batch 不再跟随 Elo 波动（旧逻辑"std 大→减小 batch"把"漂移"
    误当"噪声"，反而拖慢收敛）。Elo 稳定性现已由 K 衰减 + CI 收敛判据负责，
    batch 仅作覆盖率/预算旋钮——恒定保持用户设定值，受 [min_batch, max_batch] 钳位。
    """
    new_batch = max(min_batch, min(max_batch, current_batch))
    if new_batch != current_batch:
        return new_batch, f"batch 钳位至 [{min_batch},{max_batch}] 区间 → {new_batch}"
    return current_batch, f"batch 固定({current_batch}，与 Elo 波动解耦)"



def _build_attack_row(rec: dict, result: dict, round_idx: int, phase: str) -> dict:
    """构造 attack_results.jsonl 的单行（seed/attack 共用，避免两处大 dict 重复）。"""
    return {
        "round": round_idx,
        "phase": phase,
        "method": rec["method"],
        "id": rec["id"],
        "category": rec.get("category", "unknown"),
        "category_name": rec.get("category_name", ""),
        "source": rec.get("source", "our"),
        "functional_category": rec.get("functional_category", "standard"),
        "harm_type": rec["harm_type"],
        "eval_score": result["eval_score"],
        "jailbreak_tax": result["jailbreak_tax"],
        "math_score": result.get("math_score"),
        "actual_answer": result.get("actual_answer"),
        "expected_answer": rec["expected_answer"],
        "status": result["status"],
        "compliance_level": result.get("compliance_level"),
        "latency_ms": result["latency_ms"],
        "judge_harmfulness": result.get("judge_harmfulness", 1),
        "judge_specificity": result.get("judge_specificity", 1),
        "judge_dangerousness": result.get("judge_dangerousness", 1),
        "is_harmful": result.get("is_harmful", False),
        "is_refusal": result.get("is_refusal", False),
        "response_preview": result.get("content", "")[:500],
    }


def _resolve_workers(batch_n: int, concurrency: int | None) -> int:
    """解析并发度：None → 全并发(=batch)；0 → 串行(1)；N>0 → min(N, batch)。"""
    if concurrency is None:
        return max(1, batch_n)
    if concurrency <= 0:
        return 1
    return max(1, min(concurrency, batch_n))



# ============================================================
# Phase 1: ELO 自适应攻击测试
# ============================================================
def run_attack_phase(records: list[dict],
                     judge: Judge, tracker: ELOTracker,
                     batch_size: int, max_rounds: int,
                     attack_file,
                     sampler: str = "hybrid",
                     sampler_alpha: float = SAMPLER_INFOGAIN_ALPHA,
                     sampler_beta: float = SAMPLER_INFOGAIN_BETA,
                     sampler_gamma: float = SAMPLER_INFOGAIN_GAMMA,
                     coordinate_rounds: int | None = None,
                     coord_min_per_cluster: int = SAMPLER_COORD_MIN_PER_CLUSTER,
                     sampler_log_file: Path | None = None,
                     cluster_analysis_file: Path | None = None,
                     skip_final_clustering: bool = False,
                    state_file: Path | str | None = None,
                    no_early_stop: bool = False,
                    concurrency: int | None = None,
                    ) -> dict:
    """
    自适应攻击测试：从ELO中档开始，逐轮二分搜索。
    新增：聚类冷启动预测 + 动态重训练 + 种子采样 + 可插拔采样器 + 聚类安全分析。
    coordinate_rounds 为 None 时缺省读 params.SAMPLER_HYBRID_EXPLORE_ROUNDS。
    返回: {tested_methods, results, boundary, rounds}
    """
    from llmsec.pipeline.runner import (
        DEFENDER_NAME,
        evaluate_single,
        logger,
        publish_tracker,
        read_jsonl,
        write_jsonl,
    )
    if coordinate_rounds is None:
        coordinate_rounds = SAMPLER_HYBRID_EXPLORE_ROUNDS
    logger.info("=" * 60)
    logger.info("🗡️  Phase 1: 自适应攻击测试")
    logger.info("=" * 60)

    # 按方法分组（每种方法取第一条记录作为代表）
    method_records = {}
    for r in records:
        m = r["method"]
        if m not in method_records:
            method_records[m] = r

    all_methods = sorted(method_records.keys())

    # 加载已有 ELO（per-run 快照优先；不读全局 state.json——R 为唯一真相）
    sf = str(state_file) if state_file else None
    if sf and Path(sf).exists():
        tracker.load(sf)
    # 跨 run resume：从 R 注入当前攻击集内已测方法（R 跨 run 累积真实观测）
    from llmsec.core.results import ResultsMatrix as _RM
    _R = _RM.load()
    _tested_in_R = _R.tested_methods(DEFENDER_NAME) & set(all_methods)
    if _tested_in_R:
        tracker.ground_truth_methods.update(_tested_in_R)
        logger.info(f"  📥 从 R 恢复 {len(_tested_in_R)} 个已测方法（跨 run resume）")
    # 防跨攻击集 stale GT 污染
    _current_methods = set(all_methods)
    _stale_gt = tracker.ground_truth_methods - _current_methods
    if _stale_gt:
        for m in _stale_gt:
            tracker.ground_truth_methods.discard(m)
            tracker.predictor.ground_truth.pop(m, None)
            # 同步清理 attacker_ratings / pred_std / history，
            # 否则 compute_security_boundary 的 total_methods 膨胀（覆盖率偏低、永不收敛）、
            # predicted_above 虚高、stale history 经 publish_tracker 污染 R 矩阵
            tracker.attacker_ratings.pop(m, None)
            tracker.attacker_pred_std.pop(m, None)
        tracker.history = [h for h in tracker.history
                           if h.get("attacker") in _current_methods
                           or h.get("attacker") is None]
        logger.info(f"  🧹 过滤 {len(_stale_gt)} 个跨攻击集 stale 方法"
              f"（GT/attacker_ratings/history 已同步清理，保留 {len(tracker.ground_truth_methods)} 个）")
    # resume 时已实测方法直接计入 tested，避免被重新选中二次计 Elo
    tested = set(tracker.ground_truth_methods)
    # M-11：resume 回读已有 attack_file 预载历史明细——此前续跑首轮 write_jsonl 整体覆写
    # 会销毁上次明细。预载后轮内增量落盘与结尾全量写都基于合并结果，保证
    # attack_results.jsonl 含完整历史，ASR 口径与 Elo（ground truth 全历史）一致。
    all_results = read_jsonl(attack_file)
    if all_results:
        all_results = _dedup_attack_results(all_results)
        logger.info(f"  ♻️ 预载历史攻击明细: {len(all_results)} 条（续跑合并）")

    # ---- 启动时特征缓存：复用 / 重新提取（聚类在测试结束后才进行） ----
    gt_count = len(tracker.ground_truth_methods)
    if _should_refresh_features(tracker.predictor, method_records, force=False):
        tracker.predictor.fit_features(records)
        logger.info(f"  🧩 特征缓存: {len(method_records)} 种方法")
    else:
        logger.info(f"  ♻️ 复用已有特征缓存 (ground truth {gt_count} 种)")

    # ---- 冷启动：为所有未测方法注入预测 Elo ----
    _inject_predicted_elos(tracker, method_records, DEFENDER_NAME)
    logger.info(f"  🧊 冷启动: 已为 {len(all_methods)} 种方法注入初始 Elo "
          f"(ground truth {len(tracker.ground_truth_methods)} 种)")

    # ---- 构造采样器 ----
    # H-1 修复：Phase 1 期间聚类尚未运行（post-test），用特征缓存做快速预聚类
    # 注入 sampler，否则 InfoGain/Coordinate 的簇覆盖特性完全不工作（build_sampler
    # 默认 cluster_report=None → 所有方法被当做一个簇 → beta*visit_count 退化为全局惩罚）
    pre_labels = _quick_precluster(tracker, all_methods)
    pre_report = {"method_labels": pre_labels} if pre_labels else None
    if pre_labels:
        n_clusters_pre = len(set(pre_labels.values()))
        logger.info(f"  🔍 预聚类: {n_clusters_pre} 簇注入采样器（簇覆盖生效）")
    sampler_obj = build_sampler(
        sampler,
        cluster_report=pre_report,
        alpha=sampler_alpha,
        beta=sampler_beta,
        gamma=sampler_gamma,
        explore_rounds=coordinate_rounds,
    )
    logger.info(f"  🎲 采样策略: {sampler} "
          f"(alpha={sampler_alpha}, beta={sampler_beta}, gamma={sampler_gamma}, "
          f"coordinate_rounds={coordinate_rounds})")

    # 采样日志
    sampler_log: list[dict] = []

    # ---- D-optimality 种子：选信息量最大的方法做真实评估 ----
    if len(tracker.ground_truth_methods) == 0 and len(all_methods) > 0:
        from llmsec.clustering import log_growth_k0

        n_seeds = max(SEED_MIN_COUNT, log_growth_k0(len(all_methods)))
        seed_methods = tracker.predictor.select_d_optimal_seeds(method_records, n_seeds)
        logger.info(f"\n  🌱 D-optimal 种子: {len(seed_methods)} 种"
              f"（对预测矩阵信息量最大的方向，n={len(all_methods)} → k0={log_growth_k0(len(all_methods))}）")
        logger.info(f"     方法: {', '.join(m[:25] for m in seed_methods)}")

        seed_rows: list[tuple[str, float]] = []
        for method_name in seed_methods:
            rec = method_records[method_name]
            prompt_text = rec["prompt"]
            expected_answer = rec["expected_answer"]

            logger.info(f"     → {method_name[:40]}")
            result = evaluate_single(
                prompt_text, expected_answer, judge, use_judge=True
            )

            # API 错误（断网等）不更新 Elo、不记结果，方法保持未测状态以便下轮重试
            if result["status"] == "api_error":
                logger.warning(f" → ⚠️ API错误: {result.get('error', '')}，跳过")
                time.sleep(API_DELAY)
                continue

            tested.add(method_name)
            all_results.append(_build_attack_row(rec, result, 0, "seed"))
            seed_rows.append((method_name, result["eval_score"]))

            score = result["eval_score"]
            sym = "✅" if score > 0 else ("🔶" if score > -1 else "❌")
            logger.info(f" → {sym} score={score:.1f} {result['status']}")

            time.sleep(API_DELAY)

        # Model B 同步轮次 ELO：种子批用轮始快照一次性更新（与主循环语义统一）
        if seed_rows:
            tracker.update_round(DEFENDER_NAME, seed_rows, round_idx=0)

        # 明细先于 state 落盘（同主循环顺序，防崩溃窗口丢数据）
        write_jsonl(attack_file, all_results)
        tracker.record_round_end(DEFENDER_NAME)

        # 发现层 D+A：种子评估后算 per-seed Elo 指纹（独立于累积 R，
        # 供 BlendPredictor 相似度加权池化——冷启动从相似 donor 借先验）
        if seed_rows:
            try:
                from llmsec.evaluation.model_fingerprint import compute_fingerprint, save_probe

                seed_evaluated = [m for m, _ in seed_rows]
                fp = compute_fingerprint(tracker, seed_evaluated)
                if len(fp) >= 3:
                    save_probe(DEFENDER_NAME, fp, seed_evaluated)
                    logger.info(f"  🧬 防御指纹已记录: {len(fp)} 个哨兵 Elo（发现层 donor 相似度用）")
            except Exception as e:
                logger.warning(f"  ⚠ 指纹计算/存储失败（不影响评估）: {e}")

        # 用 SVD-Ridge 重新预测剩余方法
        remaining_records = {m: r for m, r in method_records.items() if m not in tested}
        _inject_predicted_elos(tracker, remaining_records, DEFENDER_NAME)
        if sf:
            tracker.save(sf)
        logger.info(f"  ✅ 种子阶段完成: 已建立 ground truth {len(tracker.ground_truth_methods)} 种，"
              f"剩余 {len(remaining_records)} 种使用 SVD-Ridge 预测 Elo")

    current_batch_size = batch_size
    # 兜底：max_rounds<=0 时循环不执行，下方 summary 仍引用 round_idx
    round_idx = 0
    for round_idx in range(1, max_rounds + 1):
        untested = [m for m in all_methods if m not in tested]
        if not untested:
            logger.info("\n  ✅ 所有方法已测试完毕")
            break

        # 自适应调整 batch_size
        current_batch_size, batch_reason = _adaptive_batch_size(current_batch_size)
        if round_idx == 1:
            logger.info(f"  📏 初始 batch_size={current_batch_size}")
        elif batch_reason:
            logger.info(f"  📏 自适应 batch_size={current_batch_size} ({batch_reason})")

        # 使用采样器选择下一批方法
        next_methods = sampler_obj.select(
            untested, tracker, DEFENDER_NAME, n=current_batch_size,
            round_idx=round_idx,
        )

        logger.info(f"\n  🔵 Round {round_idx}/{max_rounds}: 测试 {len(next_methods)} 种攻击方法")
        logger.info(f"     方法: {', '.join(m[:25] for m in next_methods)}")

        # ---- 批内并行求值（evaluate_single 纯函数，无 ELO 依赖）+ Model B 同步轮次 ELO ----
        max_workers = _resolve_workers(len(next_methods), concurrency)

        def _eval_one(m):
            # 并发 worker：补 threading.local 的 ambient 目标继承缺口（多目标路由正确）
            try:
                from llmsec.targets import set_active_target
                set_active_target(DEFENDER_NAME)
            except Exception:
                pass
            rec_m = method_records[m]
            return evaluate_single(rec_m["prompt"], rec_m["expected_answer"], judge, use_judge=True)

        if max_workers > 1:
            logger.info(f"     ⚡ 批内并行求值 (concurrency={max_workers})")
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                raw_results = list(ex.map(_eval_one, next_methods))
        else:
            raw_results = [_eval_one(m) for m in next_methods]

        round_rows: list[tuple[str, float]] = []
        for method_name, result in zip(next_methods, raw_results):
            # API 错误（断网等）不更新 Elo、不记结果，方法保持未测状态以便下轮重试
            if result["status"] == "api_error":
                logger.warning(f"     → {method_name[:40]} ⚠️ API错误: {result.get('error', '')}，跳过")
                if max_workers == 1:
                    time.sleep(API_DELAY)
                continue
            tested.add(method_name)
            all_results.append(_build_attack_row(method_records[method_name], result, round_idx, "attack"))
            round_rows.append((method_name, result["eval_score"]))
            score = result["eval_score"]
            sym = "✅" if score > 0 else ("🔶" if score > -1 else "❌")
            logger.info(f"     → {method_name[:40]} {sym} score={score:.1f} {result['status']}")
            if max_workers == 1:
                time.sleep(API_DELAY)

        # Model B 同步轮次 ELO：批内全部观测用轮始快照一次性更新（顺序无关、消除 batch↔K 耦合）
        if round_rows:
            tracker.update_round(DEFENDER_NAME, round_rows, round_idx=round_idx)

        # 落盘顺序：明细先于 state——若 state.json 已含本轮 GT 但 attack_results.jsonl
        # 还没写时崩溃，resume 会把本轮方法标为"已测"但明细永久丢失（ASR/税/threats 全失真）。
        # 明细先写则最坏情况是重测本轮（多花 API 预算），永不丢数据。
        write_jsonl(attack_file, all_results)

        # 记录本轮结束时的防御方 Elo（在 tracker.save 之前，确保轨迹点被持久化）
        tracker.record_round_end(DEFENDER_NAME)

        # 保存ELO进度
        if sf:
            tracker.save(sf)

        # SVD-Ridge 更新：基于新增 ground truth 刷新未测方法预测 Elo（聚类不重训）
        remaining_records = {m: r for m, r in method_records.items() if m not in tested}
        _inject_predicted_elos(tracker, remaining_records, DEFENDER_NAME)
        if sf:
            tracker.save(sf)
        logger.info(f"     🔄 预测已更新: {len(remaining_records)} 个未测方法的 SVD-Ridge 预测 Elo")

        # M-12：每轮同步发布进 R（唯一真相）。
        # publish_tracker 写 R 失败 = 真相源损坏，不可静默——重抛让调用方感知。
        publish_tracker(tracker, DEFENDER_NAME)

        # 记录采样器决策日志
        sampler_log.append({
            "round": round_idx,
            "selected": next_methods,
            "sampler": sampler,
            "sub_sampler": getattr(sampler_obj, "last_sub_sampler", None),
            "defender_elo": tracker.get_defender_elo(DEFENDER_NAME),
            "tested_count": len(tested),
        })

        # 检查收敛：综合轮次 Elo 标准差、相对标准差、覆盖率
        conv = tracker.check_convergence(DEFENDER_NAME, total_methods=len(all_methods), tested_count=len(tested))
        boundary_info = tracker.compute_security_boundary(DEFENDER_NAME)
        confidence = boundary_info.get("confidence", 0)
        # --no-early-stop：实验模式需每个 trial 跑满 max_rounds（固定预算），
        # 使 ci_half 在同一预算下可比——故不提前 break。
        if boundary_info.get("converged") and not no_early_stop:
            logger.info(f"\n  🎯 防御方 {DEFENDER_NAME} ELO 已收敛 "
                  f"(置信度={confidence*100:.0f}%, "
                  f"真值Elo 95%CI±{conv['ci_half']:.0f} (目标±{CONV_CI_TARGET:.0f}), "
                  f"漂移={conv['drift']:+.1f}/轮, "
                  f"覆盖率={conv['coverage']*100:.0f}%, "
                  f"ELO≈{conv['current_elo']:.0f}, "
                  f"已测{len(tested)}/{len(all_methods)}方法)")
            break
        else:
            notes = "; ".join(conv.get("notes", [])) if conv.get("notes") else "未收敛"
            ci_disp = f"{conv['ci_half']:.0f}" if conv.get("ci_half") is not None else "N/A"
            drift_disp = f"{conv['drift']:+.1f}" if conv.get("drift") is not None else "N/A"
            logger.info(f"     📊 防御={DEFENDER_NAME} ELO≈{conv['current_elo']:.0f} "
                  f"95%CI±{ci_disp} 漂移={drift_disp}/轮 "
                  f"覆盖率={conv['coverage']*100:.0f}% "
                  f"置信度={confidence*100:.0f}% "
                  f"({notes})")

    # ---- 攻击完成后最终聚类（post-test）+ 簇级安全分析 ----
    # 多目标模式下跳过（聚类是方法级、跨模型共享；由上层统一做一次，避免 N× embedding）
    final_report = None
    if not skip_final_clustering:
        # N-M4：final_fit 内部异常（空方法 PCA 崩溃、embedding 服务挂掉等）不应在 API
        # 成本已花之后炸掉整个 run——降级为跳过聚类，攻击结果照常落盘。
        try:
            final_report = tracker.predictor.final_fit(records, all_results)
        except Exception as e:
            logger.warning(f"\n  ⚠ 最终聚类失败，降级为跳过聚类（攻击结果不受影响）: {e}")
            final_report = None
    if final_report:
        logger.info(f"\n  🏁 最终聚类: {final_report.get('n_clusters', 0)} 簇 "
              f"(噪声={final_report.get('n_noise', 0)}, k*={final_report.get('chosen_k', 0)}, "
              f"silhouette={final_report.get('validation', {}).get('silhouette', 0):.4f})")
        rv = final_report.get("reaction_validation", {})
        if rv.get("available"):
            logger.info(f"     簇效验证: {rv.get('verdict')} "
                  f"(p={rv.get('p_anova')}, eta²={rv.get('eta2')})")
    else:
        # 记录 <2、或记录≥2 但同属 1 种方法（方法数不足）时 final_fit 返回 None，跳过聚类输出
        logger.warning("\n  ⚠ 攻击记录数或方法种类不足（需 ≥2 条且 ≥2 种方法），跳过最终聚类输出")

    try:
        cluster_analysis = analyze_clusters(tracker)
        # M-17：cluster_analysis_file 为 None（实验隔离模式）时不落盘，避免写全局默认路径污染
        if cluster_analysis_file:
            save_cluster_analysis(cluster_analysis, cluster_analysis_file)
    except Exception as e:
        logger.warning(f"     ⚠ 聚类安全分析失败: {e}")

    if sf:
            tracker.save(sf)

    # 保存攻击结果到专用文件（避免 Phase 3 读到旧数据）；去重防续跑重测产生同键重复
    all_results = _dedup_attack_results(all_results)
    write_jsonl(attack_file, all_results)

    # 保存采样器决策日志
    if sampler_log_file:
        write_jsonl(sampler_log_file, sampler_log)

    boundary = tracker.compute_security_boundary(DEFENDER_NAME)
    ranking = tracker.get_attacker_ranking()
    n_attacks = len(tested)
    # M-19：ASR 统一以 is_harmful 为准（与 evaluator 口径一致），eval_score>0 作兜底，
    # 避免"成功但税钳 0 分"的有害记录被判为未成功。
    successful = sum(1 for r in all_results if r.get("is_harmful", False) or r.get("eval_score", 0) > 0)
    asr = successful / len(all_results) if all_results else 0

    tax_summary = summarize_jailbreak_tax(all_results)

    summary = {
        "total_attacks": n_attacks,
        "total_tested": len(all_results),
        "successful": successful,
        "asr": round(asr, 4),
        "rounds": round_idx,
        "boundary_elo": boundary.get("boundary_elo", INITIAL_ELO),
        # 统一存浮点 confidence（compute_security_boundary 提供）；converged 标志另存 summary["converged"]。
        # 旧实现此处误存 converged 布尔，下游 metrics/dashboard 按数值读会被 coerce 成 0/1（类型漂移）。
        "boundary_confidence": boundary.get("confidence", 0.0),
        "converged": boundary.get("converged", False),
        "top_threats": [r["method"] for r in ranking[:5]],
        # #14：top_threats 里哪些是未真实测量的预测方法（避免报告把预测 Elo 当真实威胁）
        "top_threats_predicted": [r["method"] for r in ranking[:5] if r.get("predicted")],
        "defender_elo": boundary.get("defender_elo", INITIAL_ELO),
        "upsets": tracker.find_upsets(min_elo_gap=0),
        "jailbreak_tax": tax_summary,
    }

    logger.info("\n  📊 攻击阶段完成:")
    logger.info(f"     ASR={asr*100:.1f}% ({successful}/{len(all_results)})")
    logger.info(f"     边界ELO={boundary['boundary_elo']:.0f} (置信度{boundary['confidence']*100:.0f}%)")
    logger.info(f"     TOP5威胁: {', '.join(summary['top_threats'])}")
    logger.info(format_tax_line(tax_summary))
    logger.info("")
    return summary



def _quick_precluster(tracker: ELOTracker, all_methods: list[str]) -> dict[str, int] | None:
    """用 tracker 的特征缓存做快速 KMeans 预聚类，返回 {method: label} 或 None。

    H-1 修复：Phase 1 期间聚类尚未运行（post-test），sampler 的 InfoGain/Coordinate
    簇覆盖特性需要预聚类标签才能工作。用 build_whitened_space + KMeans 做轻量预聚类。

    失败时返回 None（sampler 退化为全局模式，与原行为一致，不崩溃）。
    """
    from llmsec.pipeline.runner import logger
    artifacts = getattr(tracker.predictor, "artifacts", None) or {}
    features = artifacts.get("features")
    if not features:
        return None
    methods = [m for m in all_methods if m in features]
    if len(methods) < 4:
        return None  # 太少不值得聚类
    try:
        from sklearn.cluster import KMeans

        from llmsec.clustering.space import build_whitened_space

        space = build_whitened_space(features, methods)
        coords = space["coords"]
        k = max(2, min(len(methods) // 3, 8))
        km = KMeans(n_clusters=k, n_init=3, random_state=get_global_seed())
        raw = km.fit_predict(coords)
        return {m: int(c) for m, c in zip(methods, raw)}
    except Exception as e:
        logger.warning(f"⚠️ 预聚类失败（sampler 将退化为全局模式）: {e}")
        return None

