"""
evaluation.elo_access — Elo 派生访问层（R-cutover 的读取统一入口）

设计（R 为唯一真相）：
  结果矩阵 R[method][model] 是评估的唯一可变真相（原始观测）。
  Elo 不是存储，而是从 R **派生**的缓存（见 evaluation.elo.derive_elo）：
    R → derive_elo(R, model) → ELOTracker（ratings / ground truth / 收敛轨迹）

本模块负责：
  1. elo_state_for(model)：返回某模型的派生 Elo 状态（命中 ELO_CACHE_FILE 且指纹
     一致则直接返回；否则从 R 重算并写缓存）。report / dashboard 经此读取，不再
     直读 state.json。
  2. publish_tracker(tracker, model)：评估结束后把 live tracker 的结果写入 R，
     并把**完整**派生状态（含 live run 的收敛轨迹）发布到缓存。runner / evaluator
     经此写入，state.json 退化为可选的快照备份。

缓存失效以"模型列内容指纹"为准——R 中该模型列变动即作废对应缓存项。
"""

from __future__ import annotations

import hashlib

from llmsec.core import config
from llmsec.core.io import read_json, write_json
from llmsec.core.results import ResultsMatrix, _coarse_status, _file_lock
from llmsec.evaluation.elo import ELOTracker, derive_elo

_CACHE_VERSION = 3  # v3：簇粒度——ratings/GT 键为 unit_id（簇），R 行键为记录 id


# ============================================================
# 指纹 / 缓存底层
# ============================================================
def _model_fingerprint(R: ResultsMatrix, model: str) -> str | None:
    """该模型列的内容指纹（方法 + 分数 + ts + round 的确定性哈希）。无结果返回 None。"""
    # M-37：复用 ResultsMatrix.column_payload，替代内联拼接
    payload = R.column_payload(model, extra_fields=("round",))
    if payload is None:
        return None
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _load_cache() -> dict:
    cache = read_json(config.ELO_CACHE_FILE, default={})
    if not isinstance(cache, dict):
        return {}
    # 缓存 schema 漂移即整体作废（指纹不变时旧 schema 项仍会命中，须按版本拦截）
    if cache.get("_version") != _CACHE_VERSION:
        return {}
    return cache


def _save_cache(cache: dict) -> None:
    cache.setdefault("_version", _CACHE_VERSION)
    write_json(config.ELO_CACHE_FILE, cache, allow_nan=False)  # M12：派生缓存也禁 NaN


# ============================================================
# 读取（派生 + 缓存）
# ============================================================
def elo_state_for(model: str) -> dict:
    """返回某模型的派生 Elo 状态 dict。

    结构（与旧 state.json 子集兼容，供 report/dashboard 直接消费）：
      {fingerprint, attacker_ratings, defender_ratings, ground_truth,
       round_defender_elos, defender_match_count, attacker_pred_std, n}
      ground_truth 与 state.json 同构：{method: {elo: ...}}。

    该模型在 R 中无结果时返回 {}。冷派生（无 live 轨迹）时 round_defender_elos
    可能为空——收敛曲线类视图应优先读 run 快照（见 dashboard _load_state）。
    """
    R = ResultsMatrix.load()
    fp = _model_fingerprint(R, model)
    if fp is None:
        return {}

    cache = _load_cache()
    entry = cache.get(model)
    if entry and entry.get("fingerprint") == fp:
        return entry

    # 缓存未命中或过期：从 R 重算
    tracker = derive_elo(R, model)
    entry = {
        "fingerprint": fp,
        "attacker_ratings": dict(tracker.attacker_ratings),
        "defender_ratings": dict(tracker.defender_ratings),
        "ground_truth": {m: dict(g) for m, g in tracker.predictor.ground_truth.items()},
        "round_defender_elos": {k: v for k, v in tracker._round_defender_elos.items()},
        "defender_match_count": {k: v for k, v in tracker._defender_match_count.items()},
        "attacker_pred_std": dict(tracker.attacker_pred_std),
        "n": len(tracker.ground_truth_methods),
    }
    cache[model] = entry
    _save_cache(cache)
    return entry


def attacker_ratings_for(model: str) -> dict:
    """便捷：某模型的 method→elo 映射。"""
    return elo_state_for(model).get("attacker_ratings", {})


def active_model() -> str | None:
    """R 中最新活跃的模型（按其结果最大 ts）。R 空→None。"""
    R = ResultsMatrix.load()
    best, best_ts = None, None
    for m in R.all_models():
        ordered = R.ordered_results(m)
        if ordered:
            ts = ordered[-1].ts
            if best_ts is None or _ts_key(ts) > _ts_key(best_ts):
                best_ts, best = ts, m
    return best


def _ts_key(ts):
    """ts 可为 int 或 str，统一转成可比较的 key。"""
    try:
        return (0, float(ts))
    except (TypeError, ValueError):
        return (1, str(ts))


# ============================================================
# 写入（live tracker → R + 缓存）
# ============================================================
def publish_tracker(tracker: ELOTracker, model: str) -> None:
    """评估结束后发布：把 live tracker 的结果写入 R，并把派生状态发布到缓存。

    语义：R 的行键是实测记录 id（原始观测），同一评级单位（簇）的多条观测各自成行；
    derive_elo 按 extra.unit 聚合回放——故缓存项的 ratings/ground_truth 取
    derive_elo(R, model) 的派生态（M-2），使缓存恒等于 R 重算结果，
    不与 elo_state_for 冷派生产生分歧。

    收敛轨迹 / 场次 / pred_std 仍取 live tracker（缓存即真相呈现该 run）；
    其中收敛轨迹自 #10 起 R 也带 round，derive_elo 可按轮分组重建
    _round_defender_elos（仅当 R 记录全部带 round；旧数据回退逐条回放）。

    防御：tracker 缺少真实 ELOTracker 接口（如测试 stub）时静默跳过，不污染 R。
    """
    if not hasattr(tracker, "attacker_ratings") or not hasattr(tracker, "history"):
        return

    # H5/H6 修复：整段 load→modify→save 纳入文件锁，防并发 publish 丢更新（TOCTOU）。
    # 原 save() 内部锁只护字节级写、不护 RMW 临界区——两个并发 publish 各自 load 同一旧 R、
    # 各自 upsert 自己的子集、各自 save → 后写者覆盖先写者。
    with _file_lock(config.RESULTS_FILE):
        R = ResultsMatrix.load()
        # 镜像 history → R（按 defender 归属，防跨模型错记）。
        # 行键 = 实测记录 id（h["record"]，原始观测粒度）；评级单位（簇）写进 extra.unit，
        # derive_elo 回放时按它聚合——同一 unit 的多条 prompt 观测各自成行、不再互相覆盖。
        for h in tracker.history:
            if h.get("defender") == model:
                # #10：round 经 extra 持久化进 R，使 derive_elo 能按轮分组重建收敛轨迹
                extra = {"unit": h["attacker"]}
                if h.get("round") is not None:
                    extra["round"] = h["round"]
                # F2 修复：透传 live tracker 的原始 status（fully_compliant/safe_redirect/…），
                # 不用 _coarse_status 二次改写（原实现把 safe_redirect→irrelevant、
                # partially_compliant→fully_compliant，语义丢失）。status 缺失时才兜底。
                raw_status = h.get("status") or _coarse_status(h["eval_score"])
                R.upsert(
                    h.get("record") or h["attacker"], model, h["eval_score"],
                    status=raw_status,
                    extra=extra,
                )
        R.save(_locked=True)  # 已在锁内，跳过 save 的内部锁防重入死锁

    fp = _model_fingerprint(R, model)
    # M-2：ratings/ground_truth 用 R 派生态（同键多次观测以末值为准），
    # 不直接拷 live tracker 的累积态
    derived = derive_elo(R, model)
    cache = _load_cache()
    cache[model] = {
        "fingerprint": fp,
        "attacker_ratings": dict(derived.attacker_ratings),
        "defender_ratings": dict(derived.defender_ratings),
        "ground_truth": {m: dict(g) for m, g in derived.predictor.ground_truth.items()},
        "round_defender_elos": {k: v for k, v in tracker._round_defender_elos.items()},
        "defender_match_count": {k: v for k, v in tracker._defender_match_count.items()},
        "attacker_pred_std": dict(tracker.attacker_pred_std),
        "n": len(derived.ground_truth_methods),
    }
    _save_cache(cache)

