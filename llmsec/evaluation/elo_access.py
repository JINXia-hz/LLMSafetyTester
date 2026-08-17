"""
evaluation.elo_access — Elo 派生访问层（R-cutover 的读取统一入口）

设计（R 为唯一真相）：
  结果矩阵 R[method][model] 是评估的唯一可变真相（原始观测）。
  Elo 不是存储，而是从 R **派生**的缓存（见 evaluation.elo.derive_elo）：
    R → derive_elo(R, model) → ELOTracker（ratings / ground truth / 收敛轨迹）

本模块负责：
  1. elo_state_for(model)：返回某模型的派生 Elo 状态（命中 results.db 的
     elo_cache 行且指纹一致则直接返回；否则从 R 重算并写行）。report /
     dashboard 经此读取。
  2. publish_tracker(tracker, model)：评估结束后把 live tracker 的结果写入 R，
     并把**完整**派生状态（含 live run 的收敛轨迹）发布到缓存行。

缓存失效以"模型列内容指纹"为准——R 中该模型列变动即作废对应行。
P2：缓存自 elo_cache.json 迁入 results.db 的 elo_cache 表（rstore 事务 upsert，
文件锁 RMW / _load_cache / _save_cache 退役）。
"""

from __future__ import annotations

import hashlib

from llmsec.core.results import MatchResult, ResultsMatrix, _coarse_status
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


def _cache_hit(model: str, fp: str) -> dict | None:
    """命中判定：elo_cache 行存在、指纹一致、payload schema 版本未漂移。"""
    from llmsec.storage import rstore

    row = rstore.get_elo_cache(model)
    if row is None:
        return None
    row_fp, payload = row
    if row_fp != fp or payload.get("_version") != _CACHE_VERSION:
        return None
    return payload


def _cache_store(model: str, fp: str, entry: dict) -> None:
    from llmsec.storage import rstore

    payload = dict(entry)
    payload["_version"] = _CACHE_VERSION
    payload.setdefault("fingerprint", fp)  # 形状统一：读侧免特判（列与 payload 都有）
    rstore.upsert_elo_cache(model, fp, payload)


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

    entry = _cache_hit(model, fp)
    if entry is not None:
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
    _cache_store(model, fp, entry)  # P2：事务 upsert 行（锁 RMW 退役）
    return entry


def attacker_ratings_for(model: str) -> dict:
    """便捷：某模型的 method→elo 映射。"""
    return elo_state_for(model).get("attacker_ratings", {})


# 进程内 tracker memoize：按 (model, R 列指纹) 缓存 derive_elo 的完整 ELOTracker。
# 同一进程内多次派生同模型（典型：MCP agent 连续调 ranking + boundary + surprises）
# 只全量 derive 一次，后续命中直接返回缓存的 tracker 对象。
#
# 设计取舍：不用 elo_cache（磁盘序列化 dict）重建 tracker——elo_cache 不存 history
# 等 find_surprises 依赖的字段，重建易缺字段出错。这里缓存的是完整 tracker 对象，
# 三个方法（get_attacker_ranking/compute_security_boundary/find_surprises）都能服务。
#
# 内存量级（实测深尺寸）：一个 tracker ≈ 0.3 KB × 历史 entry 数（5轮×50方法 ≈ 0.3MB，
# 30轮×800方法 ≈ 18MB）。条目按 model 键替换（指纹变即覆盖）、从不淘汰，占用上界
# = 查询过的模型数 × 各自规模——消费者仅 MCP 查询工具，模型数天然有限，可接受；
# 进程重启即清零（与 _RUN_META_CACHE 同性质）。
#
# 失效：R 中该模型列变动 → 指纹变 → 自动重 derive。
_TRACKER_CACHE: dict[str, tuple[str | None, ELOTracker]] = {}


def elo_tracker_for(model: str) -> ELOTracker | None:
    """返回某模型的派生 ELOTracker（进程内缓存，按列指纹失效）。

    与 elo_state_for 的关系：elo_state_for 返回扁平 dict（供 report/dashboard），
    本函数返回完整 tracker 对象（供需要调 tracker 方法的场景，如 MCP 的
    elo_ranking/elo_security_boundary/elo_find_surprises）。

    该模型在 R 中无结果时返回 None。
    """
    R = ResultsMatrix.load()
    fp = _model_fingerprint(R, model)
    if fp is None:
        return None
    cached = _TRACKER_CACHE.get(model)
    if cached and cached[0] == fp:
        return cached[1]
    tracker = derive_elo(R, model)
    _TRACKER_CACHE[model] = (fp, tracker)
    return tracker


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
    """ts 可为 int 或 str，统一转成可比较的 key。

    语义说明：字符串 ts 只应来自旧迁移/手工编辑的数据（现行写入方——upsert 自增、
    publish_tracker/snapshot 的 round、merge 透传——全部产数字）。本函数的 (1, str)
    分支与 ResultsMatrix.ordered_results 的排序兜底同构，只为混型比较不抛
    TypeError 的确定性兜底，**不代表"字符串时间戳更新"的时间语义**——存在字符串
    ts 时 active_model 的"最新"判定本就无真值，两侧保持一致即可。
    """
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

    # 阶段 2（数据库重构）：H5/H6 的"文件锁 load→modify→save RMW"由
    # rstore.upsert_observations 的单事务（BEGIN IMMEDIATE）取代——两个并发
    # publish 各自事务串行提交，不存在后写覆盖先写的丢更新路径。原 B1 的
    # strict 权衡（超时放行 vs 中断评估）随之消失：busy_timeout 排队即可。
    # 镜像 history → R（按 defender 归属，防跨模型错记）。
    # 行键 = 实测记录 id（h["record"]，原始观测粒度）；评级单位（簇）写进 extra.unit，
    # derive_elo 回放时按它聚合——同一 unit 的多条 prompt 观测各自成行、不再互相覆盖。
    items = []
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
            items.append(MatchResult(
                record=h.get("record") or h["attacker"], model=model,
                eval_score=h["eval_score"], status=raw_status, ts=None, extra=extra,
            ))
    from llmsec.storage import rstore
    rstore.upsert_observations(items)
    R = ResultsMatrix.load()  # 读回完整矩阵（指纹/派生口径不变）

    fp = _model_fingerprint(R, model)
    # M-2：ratings/ground_truth 用 R 派生态（同键多次观测以末值为准），
    # 不直接拷 live tracker 的累积态
    derived = derive_elo(R, model)
    # M-2：ratings/ground_truth 用 R 派生态（同键多次观测以末值为准）
    _cache_store(model, fp, {
        "attacker_ratings": dict(derived.attacker_ratings),
        "defender_ratings": dict(derived.defender_ratings),
        "ground_truth": {m: dict(g) for m, g in derived.predictor.ground_truth.items()},
        "round_defender_elos": {k: v for k, v in tracker._round_defender_elos.items()},
        "defender_match_count": {k: v for k, v in tracker._defender_match_count.items()},
        "attacker_pred_std": dict(tracker.attacker_pred_std),
        "n": len(derived.ground_truth_methods),
    })

