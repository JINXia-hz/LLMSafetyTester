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
  3. maybe_migrate_legacy()：首次升级时把旧 state.json 的 history 一次性迁进 R。

缓存失效以"模型列内容指纹"为准——R 中该模型列变动即作废对应缓存项。
"""

from __future__ import annotations

import hashlib
from typing import Optional

from llmsec.core.config import ELO_CACHE_FILE, STATE_FILE
from llmsec.core.io import read_json, write_json
from llmsec.core.results import ResultsMatrix, _coarse_status
from llmsec.evaluation.elo import ELOTracker, derive_elo

_CACHE_VERSION = 1


# ============================================================
# 指纹 / 缓存底层
# ============================================================
def _model_fingerprint(R: ResultsMatrix, model: str) -> Optional[str]:
    """该模型列的内容指纹（方法 + 分数 + ts 的确定性哈希）。无结果返回 None。"""
    col = R.model_column(model)
    if not col:
        return None
    payload = ",".join(
        f"{m}:{r.eval_score}:{r.ts}" for m, r in sorted(col.items())
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _load_cache() -> dict:
    cache = read_json(ELO_CACHE_FILE, default={})
    if not isinstance(cache, dict):
        return {}
    return cache


def _save_cache(cache: dict) -> None:
    cache.setdefault("_version", _CACHE_VERSION)
    write_json(ELO_CACHE_FILE, cache)


# ============================================================
# 读取（派生 + 缓存）
# ============================================================
def elo_state_for(model: str) -> dict:
    """返回某模型的派生 Elo 状态 dict。

    结构（与旧 state.json 子集兼容，供 report/dashboard 直接消费）：
      {fingerprint, attacker_ratings, defender_ratings, ground_truth,
       round_defender_elos, defender_match_count, n}

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
        "ground_truth": {m: {} for m in tracker.ground_truth_methods},
        "round_defender_elos": {k: v for k, v in tracker._round_defender_elos.items()},
        "defender_match_count": {k: v for k, v in tracker._defender_match_count.items()},
        "n": len(tracker.ground_truth_methods),
    }
    cache[model] = entry
    _save_cache(cache)
    return entry


def attacker_ratings_for(model: str) -> dict:
    """便捷：某模型的 method→elo 映射。"""
    return elo_state_for(model).get("attacker_ratings", {})


def active_model() -> Optional[str]:
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
    """评估结束后发布：把 live tracker 的结果写入 R，并把完整派生状态发布到缓存。

    live tracker 含真实收敛轨迹（record_round_end 累积），故缓存项的
    round_defender_elos 完整——dashboard 收敛曲线经缓存即可正确呈现该 run。

    防御：tracker 缺少真实 ELOTracker 接口（如测试 stub）时静默跳过，不污染 R。
    """
    if not hasattr(tracker, "attacker_ratings") or not hasattr(tracker, "history"):
        return

    R = ResultsMatrix.load()
    # 镜像 history → R（按 defender 归属，防跨模型错记）
    for h in tracker.history:
        if h.get("defender") == model:
            R.upsert(
                h["attacker"], model, h["eval_score"],
                status=_coarse_status(h["eval_score"]),
            )
    R.save()

    fp = _model_fingerprint(R, model)
    cache = _load_cache()
    cache[model] = {
        "fingerprint": fp,
        "attacker_ratings": dict(tracker.attacker_ratings),
        "defender_ratings": dict(tracker.defender_ratings),
        "ground_truth": {m: {} for m in tracker.ground_truth_methods},
        "round_defender_elos": {k: v for k, v in tracker._round_defender_elos.items()},
        "defender_match_count": {k: v for k, v in tracker._defender_match_count.items()},
        "n": len(tracker.ground_truth_methods),
    }
    _save_cache(cache)


def invalidate(model: Optional[str] = None) -> None:
    """作废派生缓存。R 被外部改动后可手动调用；publish_tracker 已自动维护。"""
    if model is None:
        write_json(ELO_CACHE_FILE, {})
        return
    cache = _load_cache()
    cache.pop(model, None)
    _save_cache(cache)


# ============================================================
# 一次性迁移（旧 state.json → R）
# ============================================================
def maybe_migrate_legacy(force: bool = False) -> bool:
    """R 为空且存在旧 state.json（含 history）时，一次性迁移历史结果进 R。

    幂等：R 已有数据则跳过（除非 force=True，此时在 R 基础上补迁）。
    返回是否执行了迁移。state.json 不被删除（保留为只读 legacy 备份）。
    """
    R = ResultsMatrix.load()
    if R.all_models() and not force:
        return False
    if not STATE_FILE.exists():
        return False

    legacy = read_json(STATE_FILE)
    if not legacy or not legacy.get("history"):
        return False

    # 单防御方旧数据：defender 取 history 中的 defender，或回退 attacker_ratings 时期
    mat = ResultsMatrix.migrate_from_legacy_state(STATE_FILE)
    if mat.all_models():
        # 保留既有 R 内容（force 补迁场景），合并迁移结果
        for method, col in mat._r.items():
            for model, res in col.items():
                if R.get(method, model) is None:
                    R.upsert(method, model, res.eval_score, status=res.status, ts=res.ts)
        R.save()
        invalidate()
    return True
