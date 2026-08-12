"""pipeline.attack_phase — Phase 1 自适应攻击测试（ELO 二分搜索）。

从 runner.py 拆出，包含：

  - _inject_predicted_elos
  - _compute_method_set_hash
  - _dedup_attack_results
  - _should_refresh_features
  - _adaptive_batch_size
  - _quick_precluster
  - run_attack_phase

本模块的运行时依赖（logger / write_jsonl / read_jsonl / evaluate_single 等）
一律从源模块顶层导入（core.logging / core.io / evaluation.evaluator 等），
不经 runner 命名空间中转——这些名字本就只在源模块定义，runner 底部
re-export 区已删除，runner 命名空间下不再存在这些名字。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llmsec.evaluation.elo import ELOTracker
    from llmsec.evaluation.judge import Judge

import time
from pathlib import Path

from llmsec.core.config import INITIAL_ELO
from llmsec.core.io import read_jsonl, write_jsonl
from llmsec.core.logging import get_logger
from llmsec.core.progress import emit_progress
from llmsec.core.seed import get_global_seed
from llmsec.evaluation.cluster_analysis import analyze_clusters, save_cluster_analysis
from llmsec.evaluation.evaluator import evaluate_single
from llmsec.evaluation.predictors.cold_start import (
    _compute_method_set_hash,  # 与 cold_start.py 原为逐字节重复实现，统一从定义处导入
    current_feature_config_hash,
)
from llmsec.evaluation.samplers import build_sampler
from llmsec.evaluation.scoring import measure_math_baseline
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
from llmsec.targets import set_active_target

logger = get_logger(__name__)


# ============================================================
# Phase 1 辅助函数
# ============================================================
def _inject_predicted_elos(tracker: ELOTracker, method_records: dict[str, dict],
                           defender_name: str, r_snapshot=None,
                           full_method_records: dict[str, dict] | None = None):
    """
    为所有尚未真实评估的方法注入预测初始 Elo。

    优先 **BlendPredictor**（跨模型：sim-加权 universal + 每模型层，发现层 D+A——
    冷启动从相似 donor 借先验，多模型场景的核心）；特征不可用/失败 → 回退
    ColdStartPredictor（单模型 SVD-Ridge + 变体兜底）。已真实评估的方法 Elo 不变。

    r_snapshot：运行前获取的 R 快照（不可变）。评估期间不读活 R——传入快照保证
    并发安全 + 隔离（目标 B 不会读到目标 A 刚 publish 的半截数据）。

    full_method_records：完整攻击集方法清单（含已测）。BlendPredictor 缓存键含
    catalog 指纹（blend.cache_key 的 cat_fp）——method_records 逐轮传入的是
    remaining 子集，若直接拿它当 catalog，缓存键每轮变、永远 miss（每轮每目标
    全量重 fit + 孤儿 pkl 泄漏）。故 catalog 恒用完整清单，predict 仍只对
    untested。None 时回退 method_records（兼容独立调用/单测）。
    """
    untested = {
        m: r for m, r in method_records.items()
        if m not in tracker.ground_truth_methods
    }
    if not untested:
        return

    # 优先 BlendPredictor：跨模型 sim-加权（发现层）——仅 R 有 ≥2 模型时启用（单模型无 donor、
    # 无跨模型意义，走 ColdStartPredictor 单训练路径避免冗余）
    predictions: dict | None = None
    artifacts = getattr(tracker.predictor, "artifacts", None) or {}
    feats = artifacts.get("features")
    if feats:
        try:
            from llmsec.core.results import ResultsMatrix
            from llmsec.evaluation.predictors.blend import load_or_fit_blend_predictor

            # r_snapshot=None 的回退仅兜底单测/独立调用——生产路径由 run_attack_phase
            # 传入运行前快照，评估期间不读活 R（并发安全 + 跨目标隔离）
            R = r_snapshot if r_snapshot is not None else ResultsMatrix.load()
            if len(R.all_models()) >= 2:
                # catalog 恒用完整清单：缓存键 cat_fp 才跨轮稳定（见 docstring）
                catalog_src = full_method_records if full_method_records is not None else method_records
                catalog = list(catalog_src.keys())
                bp = load_or_fit_blend_predictor(R, feats, method_catalog=catalog)
                predictions = {m: bp.predict(m, defender_name) for m in untested}
        except Exception as e:
            logger.warning(f"BlendPredictor 预测失败，回退 ColdStartPredictor: {e}")
            predictions = None

    # 回退：ColdStartPredictor（单模型 SVD-Ridge + 同后缀/同基底变体平均）
    if predictions is None:
        predictions = tracker.predictor.predict_batch(untested)

    for method, pred in predictions.items():
        tracker.attacker_ratings[method] = pred["elo"]
        if pred.get("std") is not None:
            tracker.attacker_pred_std[method] = pred["std"]
        # S-0：透传预测来源标记，使下游 boundary/ranking 可区分严格预测与启发式兜底
        tracker.attacker_pred_source[method] = pred.get("source", "unknown")



def _dedup_attack_results(rows: list[dict]) -> list[dict]:
    """按 (id, method) 去重攻击明细：同键保留后出现的记录（新结果覆盖旧记录）。

    续跑（resume）预载历史明细后，若某方法因 state 丢失被重测，会产生同键记录，
    此处保证落盘与 ASR 统计口径不重复计数。
    """
    merged: dict[tuple, dict] = {}
    order: list[tuple] = []
    for row in rows:
        # id(row) 兜底仅对缺 id/method 的畸形行生效：防 (None, method) 错误合并丢数据。
        # 正常记录两字段齐全不会落入 fallback；畸形行各自 id(row) 唯一，不会互相合并。
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
    # 不能只看方法名。current_feature_config_hash 由 predictors.cold_start 暴露（动态反映当前配置）。
    cached_cfg_hash = (predictor.artifacts.get("meta") or {}).get("feature_config_hash")
    if cached_cfg_hash != current_feature_config_hash():
        return True
    return False



def _adaptive_batch_size(
    base_batch: int,
    ci_half: float | None = None,
    min_batch: int = ADAPTIVE_BATCH_MIN,
    max_batch: int = ADAPTIVE_BATCH_MAX,
) -> tuple[int, str]:
    """
    收敛距离驱动的自适应 batch_size。

    逻辑：ratio = ci_half / CONV_CI_TARGET（当前真值 Elo 95%CI 半宽 / 目标半宽）。
      - ratio > 1（远离收敛）→ 放大 batch，快覆盖、快降 CI
      - ratio < 1（接近收敛）→ 缩小 batch，精修边界
      - ci_half=None（首轮/轨迹不足未估）→ 用 base_batch（用户 --batch-size）
    batch = clamp(round(base_batch × ratio), min_batch, max_batch)。

    旧逻辑"batch 跟 Elo 波动"已废（K衰减+CI判据负责稳定性）；本函数把 batch
    与**收敛进度**耦合——远离收敛时多花预算快推、接近收敛时少花精修。
    """
    if ci_half is None or ci_half <= 0:
        return base_batch, f"batch={base_batch}（基准，暂无 CI 估计）"
    ratio = ci_half / CONV_CI_TARGET
    new_batch = max(min_batch, min(max_batch, round(base_batch * ratio)))
    return new_batch, f"batch={new_batch}（ci_half={ci_half:.0f}/目标{CONV_CI_TARGET:.0f}={ratio:.2f}×基准{base_batch}）"



def _build_attack_row(rec: dict, result: dict, round_idx: int, phase: str,
                      unit: str | None = None) -> dict:
    """构造 attack_results.jsonl 的单行（seed/attack 共用，避免两处大 dict 重复）。"""
    from llmsec.core.taxonomy import normalize_harm_type

    return {
        "round": round_idx,
        "phase": phase,
        "unit": unit,               # 评级单位（簇指纹）；分析/展示的聚合键
        "method": rec["method"],
        "id": rec["id"],
        "category": rec.get("category", "unknown"),
        "category_name": rec.get("category_name", ""),
        "source": rec.get("source", "our"),
        "functional_category": rec.get("functional_category", "standard"),
        "harm_type": normalize_harm_type(rec.get("harm_type", "other")),
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
        # 预筛可观测性：透传 prescreen_result（refusal/empty/None）与 Judge 调用数。
        # 否则 attack_results.jsonl 看不出预筛省了多少 API，prescreen_hit_rate 恒为 0。
        "prescreen_result": result.get("prescreen_result"),
        "judge_calls": result.get("judge_calls", 0),
    }


def _resolve_workers(batch_n: int, concurrency: int | None) -> int:
    """解析并发度：None → 全并发(=batch)；0 → 串行(1)；N>0 → min(N, batch)。"""
    if concurrency is None:
        return max(1, batch_n)
    if concurrency <= 0:
        return 1
    return max(1, min(concurrency, batch_n))



def _emit_round_progress(defender_name: str, round_idx: int, max_rounds: int,
                         conv: dict, prev_elo: float | None, tested: int, total: int) -> float | None:
    """落盘单轮攻击进度（无 LLMSEC_TASK_ID 时 no-op）。

    复用每轮已算好的 conv（current_elo/ci_half/coverage/converged），与收敛判据同源。
    progress_pct 与 dashboard _convergence_score 同口径：ci_half→0 满分、≥目标归零。
    返回本轮 elo，供下一轮计算 delta（箭头/幅度）。
    """
    elo = conv.get("current_elo")
    if elo is None:
        return prev_elo
    ci_half = conv.get("ci_half")
    progress_pct = (
        round(max(0.0, min(0.99, 1 - ci_half / CONV_CI_TARGET)) * 100)
        if ci_half is not None else None
    )
    delta = round(elo - prev_elo, 1) if prev_elo is not None else None
    emit_progress({
        "phase": "attack", "target": defender_name, "round": round_idx,
        "max_rounds": max_rounds, "elo": round(elo, 1),
        "prev_elo": (round(prev_elo, 1) if prev_elo is not None else None),
        "delta": delta, "ci_half": ci_half, "coverage": conv.get("coverage"),
        "tested": tested, "total": total,
        "converged": bool(conv.get("converged")), "progress_pct": progress_pct,
    })
    return round(elo, 1)



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
                     force_refresh: bool = False,
                     concurrency: int | None = None,
                     defender_name: str = "",
                     r_snapshot=None,
                     units: dict | None = None,
                     ) -> dict:
    """
    自适应攻击测试：以聚类簇（unit）为评级/采样单位，逐轮二分搜索。

    单位语义（core.units）：unit = Ward 关键层簇，unit_id = 成员指纹；一次实测 =
    从 unit 的未测记录池取一条 prompt 发送，unit 的 Elo 随多次观测累积。
    聚类/预测器输入仍是 method 级特征（all_merged 下 method ≡ prompt，天然一致）。

    units：调用方（runner）预算好的共享 unit 表（多目标算一次）；None 时本函数
    自聚类自建（单目标/独立调用路径）。
    返回: {tested_methods, results, boundary, rounds}
    """
    if coordinate_rounds is None:
        coordinate_rounds = SAMPLER_HYBRID_EXPLORE_ROUNDS
    logger.info("=" * 60)
    logger.info("🗡️  Phase 1: 自适应攻击测试")
    logger.info("=" * 60)

    # 按方法分组：代表记录（特征口径：首条）+ 全量记录池（unit 内 prompt 轮换用）。
    # 缺 id 的记录补稳定 id（unit 粒度下 R 行键 = 记录 id，必须非空唯一）
    method_records: dict[str, dict] = {}
    method_pool: dict[str, list[dict]] = {}
    for r in records:
        m = r["method"]
        if not r.get("id"):
            r["id"] = f"{m}#{len(method_pool.get(m, []))}"
        method_pool.setdefault(m, []).append(r)
        if m not in method_records:
            method_records[m] = r

    # ---- 启动时特征缓存：复用 / 重新提取（单位化之前先保证特征就绪） ----
    if _should_refresh_features(tracker.predictor, method_records, force=force_refresh):
        tracker.predictor.fit_features(records)
        logger.info(f"  🧩 特征缓存: {len(method_records)} 种方法")
    else:
        logger.info(f"  ♻️ 复用已有特征缓存 (ground truth {len(tracker.ground_truth_methods)} 条)")

    # ---- 单位（簇）装配：共享传入或本地聚类自建 ----
    if units is None:
        from llmsec.core.units import assemble_units

        pre_labels = _quick_precluster(tracker, sorted(method_records))
        if not pre_labels:
            # 聚类完全不可用。方法数大时每 method 自成一簇会导致
            # CoordinateDescentSampler O(n²) 冻结（n=10498 尤其致命），
            # 故紧急 KMeans 兜底；方法数小时保持每 method 一簇的确定性退化。
            if len(method_records) > 100:
                pre_labels = _emergency_cluster(method_records, tracker.predictor.artifacts)
                if pre_labels:
                    logger.info(f"  🆘 紧急 KMeans 兜底聚类: {len(set(pre_labels.values()))} 簇")
            if not pre_labels:
                pre_labels = {m: i for i, m in enumerate(sorted(method_records))}
        units = assemble_units(pre_labels, method_records, method_pool,
                               tracker.predictor.artifacts)
    unit_ids = sorted(units.keys())
    n_units = len(unit_ids)
    logger.info(f"  🧭 评级单位: {n_units} 簇（覆盖 {len(method_records)} 种方法 / {len(records)} 条 prompt）")

    from llmsec.core.units import build_unit_features, build_unit_proxy_records
    unit_proxies = build_unit_proxy_records(units)
    # unit 级特征 = 成员质心；每目标独立视图（并发目标共享底层 artifacts，
    # 替换 features/labels 键必须拷贝，不得原地改共享对象）
    _shared_art = tracker.predictor.artifacts or {}
    _unit_feats = build_unit_features(_shared_art.get("features") or {}, units)
    _unit_labels = {uid: units[uid]["label"] for uid in unit_ids}
    tracker.predictor.artifacts = {**_shared_art,
                                   "features": _unit_feats, "labels": _unit_labels}

    # 记录 id → unit 反查（resume/明细去重/实测状态）
    _rec_to_unit: dict[str, str] = {}
    for uid, u in units.items():
        for _m, r in u["pool"]:
            _rec_to_unit[str(r["id"])] = uid

    # 每 unit 已测记录集合（R 恢复 + 本 run 实测累积）
    tested_recs: dict[str, set[str]] = {uid: set() for uid in unit_ids}

    def _pick_record(uid: str):
        """unit 内选下一条待测记录：实测最少的成员优先（宽度优先），medoid 最先。"""
        u = units[uid]
        tset = tested_recs[uid]
        by_member: dict[str, list[dict]] = {}
        for m, r in u["pool"]:
            by_member.setdefault(m, []).append(r)
        members = sorted(
            by_member,
            key=lambda m: (sum(1 for r in by_member[m] if str(r["id"]) in tset),
                           m != u["medoid"], m),
        )
        for m in members:
            for r in by_member[m]:
                if str(r["id"]) not in tset:
                    return r
        return None

    # 加载已有 ELO（per-run 快照优先；不读全局 state.json——R 为唯一真相）
    sf = str(state_file) if state_file else None
    if sf and Path(sf).exists():
        tracker.load(sf)
    # 跨 run resume：从 R 注入当前攻击集内已测单位（R 跨 run 累积真实观测；
    # R 行键 = 记录 id，按 extra.unit 聚合成已测单位）。
    # 优先用调用方传入的运行前快照（评估期间不读活 R）；None 仅兜底单测/独立调用。
    from llmsec.core.results import ResultsMatrix as _RM
    _R = r_snapshot if r_snapshot is not None else _RM.load()
    _tested_in_R = _R.tested_units(defender_name) & set(unit_ids)
    if _tested_in_R:
        # 只恢复 GT 集合不够——全量 resume 时防御 Elo 停在默认 INITIAL_ELO、CI=None，
        # 且 predictor.ground_truth 缺已测单位（cluster_analysis 按需训练分支会把它们
        # 当未测预测）。用 derive_elo 从 R 回放重建派生态，字段级并入本 run tracker：
        # 评分/场次/收敛轨迹/历史/predictor GT 一并恢复，本 run 后续 update 不受影响
        #（K 衰减场次从 0 重累计是项目认可口径，同 elo_access.publish_tracker）。
        from llmsec.evaluation.elo import derive_elo
        _derived = derive_elo(_R, defender_name, unit_catalog=unit_ids)
        for m in _derived.ground_truth_methods:
            tracker.attacker_ratings[m] = _derived.attacker_ratings[m]
        tracker.defender_ratings.update(_derived.defender_ratings)
        tracker._defender_match_count.update(_derived._defender_match_count)
        tracker._round_defender_elos.update(_derived._round_defender_elos)
        # 历史：R 回放是该 defender 的累计真相——先剔除本 tracker 同 defender 旧条目
        #（state 与 R 同源时防重复计数），再并入回放历史；其他 defender 条目保留
        tracker.history = [h for h in tracker.history if h.get("defender") != defender_name]
        tracker.history.extend(_derived.history)
        tracker.ground_truth_methods.update(_tested_in_R)
        tracker.predictor.ground_truth.update(_derived.predictor.ground_truth)
        logger.info(f"  📥 从 R 恢复 {len(_tested_in_R)} 个已测单位"
              f"（跨 run resume，评分/历史/predictor GT 已回放）")
    # R 中该模型已测记录 → 标记对应 unit 的记录池（防同 prompt 重测）
    for rid, _res in _R.model_column(defender_name).items():
        uid = _rec_to_unit.get(str(rid))
        if uid is not None:
            tested_recs[uid].add(str(rid))
    # 防跨攻击集 stale GT 污染（单位指纹跨攻击集不复用，换攻击集即全部 stale）
    _current_units = set(unit_ids)
    _stale_gt = tracker.ground_truth_methods - _current_units
    if _stale_gt:
        for m in _stale_gt:
            tracker.ground_truth_methods.discard(m)
            tracker.predictor.ground_truth.pop(m, None)
            # 同步清理 attacker_ratings / pred_std / pred_source / history，
            # 否则 compute_security_boundary 的 total_methods 膨胀（覆盖率偏低、永不收敛）、
            # predicted_above 虚高、stale history 经 publish_tracker 污染 R 矩阵
            tracker.attacker_ratings.pop(m, None)
            tracker.attacker_pred_std.pop(m, None)
            tracker.attacker_pred_source.pop(m, None)
        tracker.history = [h for h in tracker.history
                           if h.get("attacker") in _current_units
                           or h.get("attacker") is None]
        logger.info(f"  🧹 过滤 {len(_stale_gt)} 个跨攻击集 stale 单位"
              f"（GT/attacker_ratings/history 已同步清理，保留 {len(tracker.ground_truth_methods)} 个）")
    # resume 时已实测单位直接计入 tested，避免被重新选中二次计 Elo
    tested = set(tracker.ground_truth_methods)
    # resume 带进来的已测数（含 R 恢复）——summary 的 this_run_tested 以此为基线
    _resumed_tested = len(tested)
    # M-11：resume 回读已有 attack_file 预载历史明细——此前续跑首轮 write_jsonl 整体覆写
    # 会销毁上次明细。预载后轮内增量落盘与结尾全量写都基于合并结果，保证
    # attack_results.jsonl 含完整历史，ASR 口径与 Elo（ground truth 全历史）一致。
    all_results = read_jsonl(attack_file)
    if all_results:
        all_results = _dedup_attack_results(all_results)
        logger.info(f"  ♻️ 预载历史攻击明细: {len(all_results)} 条（续跑合并）")
        # 历史明细中的已测记录同步进记录池状态（R 缺失时的兜底）
        for row in all_results:
            uid = _rec_to_unit.get(str(row.get("id")))
            if uid is not None:
                tested_recs[uid].add(str(row.get("id")))

    # ---- 冷启动：为所有未测单位注入预测 Elo ----
    _inject_predicted_elos(tracker, unit_proxies, defender_name, r_snapshot=r_snapshot,
                           full_method_records=unit_proxies)
    logger.info(f"  🧊 冷启动: 已为 {n_units} 个单位注入初始 Elo "
          f"(ground truth {len(tracker.ground_truth_methods)} 个)")

    # ---- 构造采样器 ----
    # unit 即簇：采样器的簇覆盖机制直接以 unit 为簇标签（坐标下降天然适配"簇轮询 +
    # 簇内选点"，内层选点即本函数的 _pick_record 记录轮换）
    pre_report = {"method_labels": _unit_labels}
    # coord_min_per_cluster 映射到 samplers.py 两个真实参数名：CoordinateSampler 是
    # min_tests_per_cluster、HybridSampler 是 coordinate_min_tests_per_cluster——两个都传，
    # 不适用的那个沿子类 **kwargs 链落到 AttackSampler 基类 sink（有意吞掉，见 samplers.py）。
    sampler_obj = build_sampler(
        sampler,
        cluster_report=pre_report,
        alpha=sampler_alpha,
        beta=sampler_beta,
        gamma=sampler_gamma,
        explore_rounds=coordinate_rounds,
        min_tests_per_cluster=coord_min_per_cluster,
        coordinate_min_tests_per_cluster=coord_min_per_cluster,
    )
    logger.info(f"  🎲 采样策略: {sampler} "
          f"(alpha={sampler_alpha}, beta={sampler_beta}, gamma={sampler_gamma}, "
          f"coordinate_rounds={coordinate_rounds})")

    # 采样日志（同 M-11 attack_file 口径：resume 预载已有日志再 append，
    # 否则结尾整体覆写会销毁上次运行的采样历史）
    sampler_log: list[dict] = read_jsonl(sampler_log_file) if sampler_log_file else []

    # 上一轮防御方 ELO，用于本轮 delta（看板进度箭头/幅度）；首轮/seed 前 None
    prev_elo: float | None = None

    # ---- D-optimality 种子：选信息量最大的单位，测其 medoid 记录 ----
    if len(tracker.ground_truth_methods) == 0 and n_units > 0:
        from llmsec.clustering import log_growth_k0

        n_seeds = max(SEED_MIN_COUNT, log_growth_k0(n_units))
        seed_units = tracker.predictor.select_d_optimal_seeds(unit_proxies, n_seeds)
        logger.info(f"\n  🌱 D-optimal 种子: {len(seed_units)} 个单位"
              f"（对预测矩阵信息量最大的方向，n={n_units} → k0={log_growth_k0(n_units)}）")
        logger.info(f"     单位: {', '.join(units[u]['name'][:12] for u in seed_units)}")

        seed_rows: list[tuple[str, float]] = []
        seed_statuses: list[str] = []
        seed_rec_ids: list[str] = []
        for uid in seed_units:
            rec = _pick_record(uid)
            if rec is None:
                continue
            prompt_text = rec["prompt"]
            expected_answer = rec["expected_answer"]

            logger.info(f"     → {units[uid]['name'][:24]}（{uid}）")
            result = evaluate_single(
                prompt_text, expected_answer, judge, use_judge=True
            )

            # API 错误（断网等）不更新 Elo、不记结果，单位保持未测状态以便下轮重试
            if result["status"] == "api_error":
                logger.warning(f" → ⚠️ API错误: {result.get('error', '')}，跳过")
                time.sleep(API_DELAY)
                continue

            tested.add(uid)
            tested_recs[uid].add(str(rec["id"]))
            all_results.append(_build_attack_row(rec, result, 0, "seed", unit=uid))
            seed_rows.append((uid, result["eval_score"]))
            seed_statuses.append(result.get("status", ""))
            seed_rec_ids.append(str(rec["id"]))

            score = result["eval_score"]
            sym = "✅" if score > 0 else ("🔶" if score > -1 else "❌")
            logger.info(f" → {sym} score={score:.1f} {result['status']}")

            time.sleep(API_DELAY)

        # Model B 同步轮次 ELO：种子批用轮始快照一次性更新（与主循环语义统一）
        if seed_rows:
            tracker.update_round(defender_name, seed_rows, round_idx=0,
                                 statuses=seed_statuses, record_ids=seed_rec_ids)

        # 明细先于 state 落盘（同主循环顺序，防崩溃窗口丢数据）
        write_jsonl(attack_file, all_results)
        tracker.record_round_end(defender_name)

        # 进度落盘：seed 轮（round=0）。delta=None（首轮无前值）
        _seed_conv = tracker.check_convergence(
            defender_name, total_methods=n_units,
            tested_count=len(tracker.ground_truth_methods))
        prev_elo = _emit_round_progress(
            defender_name, 0, max_rounds, _seed_conv, prev_elo,
            len(tracker.ground_truth_methods), n_units)

        # 发现层 D+A：种子评估后算 per-seed Elo 指纹（独立于累积 R，
        # 供 BlendPredictor 相似度加权池化——冷启动从相似 donor 借先验）
        if seed_rows:
            try:
                from llmsec.evaluation.predictors.fingerprint import compute_fingerprint, save_probe

                seed_evaluated = [m for m, _ in seed_rows]
                fp = compute_fingerprint(tracker, seed_evaluated)
                if len(fp) >= 3:
                    save_probe(defender_name, fp, seed_evaluated)
                    logger.info(f"  🧬 防御指纹已记录: {len(fp)} 个哨兵 Elo（发现层 donor 相似度用）")
            except Exception as e:
                logger.warning(f"  ⚠ 指纹计算/存储失败（不影响评估）: {e}")

        # 用 SVD-Ridge 重新预测剩余单位
        remaining_proxies = {u: r for u, r in unit_proxies.items() if u not in tested}
        _inject_predicted_elos(tracker, remaining_proxies, defender_name, r_snapshot=r_snapshot,
                               full_method_records=unit_proxies)
        if sf:
            tracker.save(sf)
        logger.info(f"  ✅ 种子阶段完成: 已建立 ground truth {len(tracker.ground_truth_methods)} 个单位，"
              f"剩余 {len(remaining_proxies)} 个使用 SVD-Ridge 预测 Elo")

    base_batch = batch_size  # 用户 --batch-size（nominal，自适应缩放的基准）
    current_batch_size = batch_size
    prev_ci_half: float | None = None  # 上一轮收敛 CI（首轮 None→用 base）
    # 兜底：max_rounds<=0 时循环不执行，下方 summary/进度落盘仍引用 round_idx 与 conv
    round_idx = 0
    conv: dict = {}
    for round_idx in range(1, max_rounds + 1):
        # 候选 = 记录池未耗尽的单位（已测单位仍可被选——同簇换 prompt 复测，
        # 这正是簇粒度评级的统计强度来源）
        candidates = [u for u in unit_ids if _pick_record(u) is not None]
        if not candidates:
            logger.info("\n  ✅ 所有单位的记录池已测尽")
            break

        # 自适应调整 batch_size（收敛距离驱动：远离收敛→大批覆盖，接近→小批精修）
        current_batch_size, batch_reason = _adaptive_batch_size(base_batch, prev_ci_half)
        if round_idx == 1:
            logger.info(f"  📏 初始 batch_size={current_batch_size}")
        elif batch_reason:
            logger.info(f"  📏 自适应 batch_size={current_batch_size} ({batch_reason})")

        # 使用采样器选择下一批单位
        next_units = sampler_obj.select(
            candidates, tracker, defender_name, n=current_batch_size,
            round_idx=round_idx,
        )

        logger.info(f"\n  🔵 Round {round_idx}/{max_rounds}: 测试 {len(next_units)} 个攻击单位")
        logger.info(f"     单位: {', '.join(units[u]['name'][:12] for u in next_units)}")

        # ---- 批内并行求值（evaluate_single 纯函数，无 ELO 依赖）+ Model B 同步轮次 ELO ----
        # 先定记录再并行：_pick_record 读共享 tested_recs，须在主线程定稿
        picked = [(u, _pick_record(u)) for u in next_units]
        picked = [(u, r) for u, r in picked if r is not None]
        max_workers = _resolve_workers(len(picked), concurrency)

        def _eval_one(rec):
            # 并发 worker：补 threading.local 的 ambient 目标继承缺口（多目标路由正确）
            try:
                set_active_target(defender_name)
            except Exception as e:
                logger.warning(f"     ⚠ worker 设置活动目标失败（多目标路由可能串扰）: {e}")
            return evaluate_single(rec["prompt"], rec["expected_answer"], judge, use_judge=True)

        if max_workers > 1:
            logger.info(f"     ⚡ 批内并行求值 (concurrency={max_workers})")
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                raw_results = list(ex.map(lambda pr: _eval_one(pr[1]), picked))
        else:
            raw_results = [_eval_one(r) for _u, r in picked]

        round_rows: list[tuple[str, float]] = []
        round_statuses: list[str] = []
        round_rec_ids: list[str] = []
        for (uid, rec), result in zip(picked, raw_results):
            # API 错误（断网等）不更新 Elo、不记结果，记录保持未测状态以便下轮重试
            if result["status"] == "api_error":
                logger.warning(f"     → {units[uid]['name'][:24]} ⚠️ API错误: {result.get('error', '')}，跳过")
                if max_workers == 1:
                    time.sleep(API_DELAY)
                continue
            tested.add(uid)
            tested_recs[uid].add(str(rec["id"]))
            all_results.append(_build_attack_row(rec, result, round_idx, "attack", unit=uid))
            round_rows.append((uid, result["eval_score"]))
            round_statuses.append(result.get("status", ""))
            round_rec_ids.append(str(rec["id"]))
            score = result["eval_score"]
            sym = "✅" if score > 0 else ("🔶" if score > -1 else "❌")
            logger.info(f"     → {units[uid]['name'][:24]}（{rec['method'][:20]}） {sym} score={score:.1f} {result['status']}")
            if max_workers == 1:
                time.sleep(API_DELAY)

        # Model B 同步轮次 ELO：批内全部观测用轮始快照一次性更新（顺序无关、消除 batch↔K 耦合）
        if round_rows:
            tracker.update_round(defender_name, round_rows, round_idx=round_idx,
                                 statuses=round_statuses, record_ids=round_rec_ids)

        # 落盘顺序：明细先于 state——若 state.json 已含本轮 GT 但 attack_results.jsonl
        # 还没写时崩溃，resume 会把本轮方法标为"已测"但明细永久丢失（ASR/税/threats 全失真）。
        # 明细先写则最坏情况是重测本轮（多花 API 预算），永不丢数据。
        write_jsonl(attack_file, all_results)

        # 记录本轮结束时的防御方 Elo（在 tracker.save 之前，确保轨迹点被持久化）
        tracker.record_round_end(defender_name)

        # SVD-Ridge 更新：基于新增 ground truth 刷新未测单位预测 Elo（聚类不重训）
        remaining_proxies = {u: r for u, r in unit_proxies.items() if u not in tested}
        _inject_predicted_elos(tracker, remaining_proxies, defender_name, r_snapshot=r_snapshot,
                               full_method_records=unit_proxies)
        # 保存ELO进度（每轮一次：轨迹点与刷新后的预测一并持久化）
        if sf:
            tracker.save(sf)
        logger.info(f"     🔄 预测已更新: {len(remaining_proxies)} 个未测单位的 SVD-Ridge 预测 Elo")

        # 不在此处 publish_tracker——R 快照模型下，评估期间不写 R，
        # 合并由调用方（runner main）在评估结束后统一执行

        # 记录采样器决策日志
        sampler_log.append({
            "round": round_idx,
            "selected": next_units,
            "sampler": sampler,
            "sub_sampler": getattr(sampler_obj, "last_sub_sampler", None),
            "defender_elo": tracker.get_defender_elo(defender_name),
            "tested_count": len(tested),
        })

        # 检查收敛：综合轮次 Elo 标准差、相对标准差、覆盖率（单位口径）
        conv = tracker.check_convergence(defender_name, total_methods=n_units, tested_count=len(tested))
        prev_ci_half = conv.get("ci_half")  # 供下一轮 batch 自适应（收敛距离驱动）
        boundary_info = tracker.compute_security_boundary(defender_name)
        confidence = boundary_info.get("confidence", 0)

        # 进度落盘：本轮（含 delta=本轮−上轮）。无 LLMSEC_TASK_ID 时 no-op
        prev_elo = _emit_round_progress(
            defender_name, round_idx, max_rounds, conv, prev_elo,
            len(tested), n_units)
        # --no-early-stop：实验模式需每个 trial 跑满 max_rounds（固定预算），
        # 使 ci_half 在同一预算下可比——故不提前 break。
        if boundary_info.get("converged") and not no_early_stop:
            logger.info(f"\n  🎯 防御方 {defender_name} ELO 已收敛 "
                  f"(置信度={confidence*100:.0f}%, "
                  f"真值Elo 95%CI±{conv['ci_half']:.0f} (目标±{CONV_CI_TARGET:.0f}), "
                  f"漂移={conv['drift']:+.1f}/轮, "
                  f"覆盖率={conv['coverage']*100:.0f}%, "
                  f"ELO≈{conv['current_elo']:.0f}, "
                  f"已测{len(tested)}/{n_units}单位)")
            break
        else:
            notes = "; ".join(conv.get("notes", [])) if conv.get("notes") else "未收敛"
            ci_disp = f"{conv['ci_half']:.0f}" if conv.get("ci_half") is not None else "N/A"
            drift_disp = f"{conv['drift']:+.1f}" if conv.get("drift") is not None else "N/A"
            logger.info(f"     📊 防御={defender_name} ELO≈{conv['current_elo']:.0f} "
                  f"95%CI±{ci_disp} 漂移={drift_disp}/轮 "
                  f"覆盖率={conv['coverage']*100:.0f}% "
                  f"置信度={confidence*100:.0f}% "
                  f"({notes})")

    # 进度落盘：该目标攻击阶段结束（收敛/跑满/单位测尽）。看板据此把行标灰
    # conv 主循环前已显式初始化为 {}，循环未进入时各进度字段自然为 None
    emit_progress({
        "phase": "attack_done", "target": defender_name,
        "round": round_idx, "max_rounds": max_rounds,
        "elo": conv.get("current_elo"),
        "ci_half": conv.get("ci_half"),
        "coverage": conv.get("coverage"),
        "tested": len(tested), "total": n_units,
        "converged": bool(conv.get("converged")),
        "progress_pct": (
            round(max(0.0, min(0.99, 1 - conv["ci_half"] / CONV_CI_TARGET)) * 100)
            if conv.get("ci_half") is not None else None
        ),
    })

    # ---- 攻击完成后最终聚类（post-test）+ 簇级安全分析 ----
    # 冻结分区：沿用 run 开头的预聚类标签（unit 身份不变），只补命名/画像/簇效验证
    # 多目标模式下跳过（聚类是方法级、跨模型共享；由上层统一做一次，避免 N× embedding）
    final_report = None
    if not skip_final_clustering:
        # N-M4：final_fit 内部异常（空方法 PCA 崩溃、embedding 服务挂掉等）不应在 API
        # 成本已花之后炸掉整个 run——降级为跳过聚类，攻击结果照常落盘。
        try:
            _frozen = {m: units[uid]["label"] for uid in unit_ids for m in units[uid]["members"]}
            final_report = tracker.predictor.final_fit(
                records, all_results, preset_labels=_frozen, units=units)
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
    elif skip_final_clustering:
        # 多目标并发主动跳过（聚类跨模型共享，由上层统一落盘一次）——非异常，不打 warning
        logger.info("\n  ⏭ 多目标共享聚类文件，本目标跳过最终聚类落盘")
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

    boundary = tracker.compute_security_boundary(defender_name)
    ranking = tracker.get_attacker_ranking()
    n_attacks = len(tested)
    # M-19：ASR 统一以 is_harmful 为准（与 evaluator 口径一致），eval_score>0 作兜底，
    # 避免"成功但税钳 0 分"的有害记录被判为未成功。
    successful = sum(1 for r in all_results if r.get("is_harmful", False) or (r.get("eval_score") or 0) > 0)
    asr_n = len(all_results)
    if not all_results and _tested_in_R:
        # 全量 resume（本轮 0 新测、明细为空）：从 R 列合成累计 ASR，否则汇总恒报
        # ASR=0/0。R 无 is_harmful 字段，用 eval_score>0（M-19 的兜底口径）计成功。
        _col = {rid: r for rid, r in _R.model_column(defender_name).items()
                if str(rid) in _rec_to_unit}
        successful = sum(1 for r in _col.values() if r.eval_score > 0)
        asr_n = len(_col)
    asr = successful / asr_n if asr_n else 0

    # S5：攻击集含数学探针时补测裸数学基线——baseline_accuracy/accuracy_drop 的数据源
    # （reporting/report.py 与前端 sections.js 消费）。先 set_active_target 保证多目标
    # 路由正确；测量抛异常或 accuracy 为 None 时记 warning 并降级为无基线输出。
    baseline = None
    if any(r.get("math_score") is not None for r in all_results):
        try:
            if defender_name:
                set_active_target(defender_name)
            baseline = measure_math_baseline()
            if baseline.get("accuracy") is None:
                logger.warning("  ⚠ 越狱税基线测量无有效探针，降级为无基线输出")
                baseline = None
        except Exception as e:
            logger.warning(f"  ⚠ 越狱税基线测量失败，降级为无基线输出: {e}")
            baseline = None

    tax_summary = summarize_jailbreak_tax(all_results, baseline=baseline or None)

    summary = {
        "total_attacks": n_attacks,
        "total_tested": len(all_results),
        # 本轮新测的单位数（不含 resume 带进来的已测）——区分全量 resume（=0）与真实新测
        "this_run_tested": len(tested) - _resumed_tested,
        "successful": successful,
        "asr": round(asr, 4),
        "rounds": round_idx,
        "n_units": n_units,
        "boundary_elo": boundary.get("boundary_elo", INITIAL_ELO),
        # 统一存浮点 confidence（compute_security_boundary 提供）；converged 标志另存 summary["converged"]。
        # 旧实现此处误存 converged 布尔，下游 metrics/dashboard 按数值读会被 coerce 成 0/1（类型漂移）。
        "boundary_confidence": boundary.get("confidence", 0.0),
        "converged": boundary.get("converged", False),
        # top_threats 为 unit_id（评级单位）；top_threat_names 为簇名（展示用）
        "top_threats": [r["unit"] for r in ranking[:5]],
        "top_threat_names": [units.get(r["unit"], {}).get("name", r["unit"]) for r in ranking[:5]],
        # #14：top_threats 里哪些是未真实测量的预测单位（避免报告把预测 Elo 当真实威胁）
        "top_threats_predicted": [r["unit"] for r in ranking[:5] if r.get("predicted")],
        "defender_elo": boundary.get("defender_elo", INITIAL_ELO),
        "upsets": tracker.find_upsets(min_elo_gap=0),
        "jailbreak_tax": tax_summary,
    }

    logger.info("\n  📊 攻击阶段完成:")
    logger.info(f"     ASR={asr*100:.1f}% ({successful}/{asr_n})")
    logger.info(f"     边界ELO={boundary['boundary_elo']:.0f} (置信度{boundary['confidence']*100:.0f}%)")
    logger.info(f"     TOP5威胁簇: {', '.join(summary['top_threat_names'])}")
    logger.info(format_tax_line(tax_summary))
    logger.info("")
    return summary



def _quick_precluster(tracker: ELOTracker, all_methods: list[str]) -> dict[str, int] | None:
    """用 tracker 的特征缓存做预聚类，返回 {method: label} 或 None。

    H-1 修复：Phase 1 期间聚类尚未运行（post-test），sampler 的 InfoGain/Coordinate
    簇覆盖特性需要预聚类标签才能工作。与 post-test 聚类同口径复用
    clustering.hdb.compute_cluster_labels（阻尼白化 → HDBSCAN 密度视图 → Ward auto-k
    主标签）；此时无 GT 反应，弱监督特征权重不存在，故用未加权白化空间，
    簇划分允许与最终聚类存在差异。

    hdbscan 是可选依赖，不可用/核心失败时回退 KMeans；再失败返回 None
    （sampler 退化为全局模式，与原行为一致，不崩溃）。
    """
    artifacts = getattr(tracker.predictor, "artifacts", None) or {}
    features = artifacts.get("features")
    if not features:
        return None
    methods = [m for m in all_methods if m in features]
    if len(methods) < 4:
        return None  # 太少不值得聚类
    try:
        from llmsec.clustering.hdb import compute_cluster_labels

        # skip_hdbscan=True：预聚类只需 Ward 主标签，跳过密度视图省 HDBSCAN 拟合开销
        core = compute_cluster_labels({m: features[m] for m in methods}, skip_hdbscan=True)
        if not core.get("error"):
            return core["labels"]
        logger.warning(f"⚠️ 预聚类核心返回错误（{core['error']}），回退 KMeans")
    except ImportError:
        logger.warning("⚠️ hdbscan 未安装，预聚类回退 KMeans")
    except Exception as e:
        logger.warning(f"⚠️ 预聚类(HDBSCAN/Ward)失败（{e}），回退 KMeans")
    try:
        from sklearn.cluster import KMeans

        from llmsec.clustering import log_growth_k0
        from llmsec.clustering.space import build_whitened_space

        space = build_whitened_space(features, methods)
        coords = space["coords"]
        # 簇数用 log_growth_k0（与正式聚类同口径），而非硬编码 8——
        # n=10498 时 k≈14，而非 8（否则每簇 ~1312 method，粒度严重退化）
        k = log_growth_k0(len(methods))
        km = KMeans(n_clusters=k, n_init=3, random_state=get_global_seed())
        raw = km.fit_predict(coords)
        return {m: int(c) for m, c in zip(methods, raw)}
    except Exception as e:
        logger.warning(f"⚠️ 预聚类失败（sampler 将退化为全局模式）: {e}")
        return None


def _emergency_cluster(method_records: dict[str, dict], artifacts: dict | None) -> dict[str, int] | None:
    """聚类完全不可用时的紧急 KMeans 兜底。

    _quick_precluster 返回 None 时调用。从 artifacts 取特征做 KMeans，
    簇数用 log_growth_k0（n=10498 → k≈14）。避免每 method 自成一簇
    导致 CoordinateDescentSampler O(n²) 冻结。

    返回 {method: label} 或 None（无特征 / sklearn 不可用时）。
    """
    features = (artifacts or {}).get("features")
    if not features:
        return None
    methods = sorted(m for m in method_records if m in features)
    if len(methods) < 4:
        return None
    try:
        from sklearn.cluster import KMeans

        from llmsec.clustering import log_growth_k0
        from llmsec.clustering.space import build_whitened_space

        space = build_whitened_space(features, methods)
        coords = space["coords"]
        k = log_growth_k0(len(methods))
        km = KMeans(n_clusters=k, n_init=3, random_state=get_global_seed())
        raw = km.fit_predict(coords)
        return {m: int(c) for m, c in zip(methods, raw)}
    except Exception as e:
        logger.warning(f"⚠️ 紧急 KMeans 兜底也失败: {e}")
        return None

