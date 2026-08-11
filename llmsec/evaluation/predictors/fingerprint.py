"""evaluation.predictors.fingerprint — 模型防御指纹（发现层 D+A）

每个模型冷启动时跑 D-optimal 哨兵种子（特征驱动、矩阵独立），种子评估后的
per-seed Elo 向量即该模型的"防御指纹"。两模型指纹的相关系数量化行为相似度，
供 BlendPredictor 做相似度加权池化（取代弱 universal 均匀平均）。

指纹独立于累积 R（仅种子结果派生），符合"发现测试不依赖过去的矩阵"。
冷启动时 D-optimal 种子对各模型一致（GT 空、特征驱动）→ per-seed Elo 向量
同维度直接可比。

用法:
    from llmsec.evaluation.predictors.fingerprint import compute_fingerprint, save_probe, donor_similarities
    fp = compute_fingerprint(tracker, seed_methods)
    save_probe(model, fp, seed_methods)
    sims = donor_similarities(model)   # {donor: 相关系数}
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

from llmsec.core.config import STATE_DIR
from llmsec.core.logging import get_logger

logger = get_logger(__name__)

PROBES_FILE = STATE_DIR / "probes.json"
MIN_COMMON = 3  # 计算相关系数所需的最少公共种子方法


def compute_fingerprint(tracker, seed_methods: list[str]) -> dict:
    """从 tracker 当前 Elo 抽取 per-seed 指纹 {method: elo}（仅种子方法）。

    在种子 update_round + record_round_end 之后调用——此时 tracker 的
    attacker_ratings 已含种子方法的真实 Elo。
    """
    return {m: float(tracker.get_attacker_elo(m)) for m in seed_methods if m}


def model_similarity(fp_a: dict, fp_b: dict) -> float | None:
    """两模型指纹的相关系数（仅取双方都有的种子方法）。

    公共方法 < MIN_COMMON 或某方零方差 → None（不可比，调用方应排除）。
    用相关系数（而非余弦）自动忽略"某模型系统性更强但模式相似"的偏移。
    """
    common = [m for m in fp_a if m in fp_b]
    if len(common) < MIN_COMMON:
        return None
    a = np.array([fp_a[m] for m in common], dtype=np.float64)
    b = np.array([fp_b[m] for m in common], dtype=np.float64)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return None  # 一方指纹无变异，相关无意义
    corr = float(np.corrcoef(a, b)[0, 1])
    return corr if np.isfinite(corr) else None


def load_probes(path: Path | str | None = None) -> dict:
    """加载 probes.json 的 {model: entry}；缺失/损坏返回 {}。"""
    p = Path(path) if path else PROBES_FILE
    try:
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            return data.get("models", {}) if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        logger.warning("加载 %s 失败（指纹迁移将忽略历史 donor）: %s", p.name, e)
    return {}


# P9：save_probe 的 read-modify-write 无锁时，并发目标（线程）会互相覆盖指纹。
# 模块级锁保证同进程线程安全（本次运行是线程并发）；跨进程并发写仍有竞争风险。
_save_lock = threading.Lock()


def save_probe(
    model: str,
    fingerprint: dict,
    seed_methods: list[str],
    path: Path | str | None = None,
) -> None:
    """原子合并：追加/更新一个模型的指纹到 probes.json（同进程线程安全）。"""
    from llmsec.core.io import write_json

    p = Path(path) if path else PROBES_FILE
    with _save_lock:
        models = load_probes(p)
        models[model] = {
            "fingerprint": {m: round(float(e), 2) for m, e in fingerprint.items()},
            "seed_methods": list(seed_methods),
            "n": len(fingerprint),
            "computed_at": datetime.now().isoformat(),
        }
        p.parent.mkdir(parents=True, exist_ok=True)
        write_json(p, {"version": 1, "models": models})


def donor_similarities(
    target: str,
    probes: dict | None = None,
    min_sim: float = 0.0,
) -> dict[str, float]:
    """target 与所有有指纹的 donor 的相似度 {donor: sim}。

    排除 target 自身、无指纹者、相关不可算者。min_sim 以下裁掉（默认>0才借）。
    """
    probes = probes if probes is not None else load_probes()
    target_fp = (probes.get(target) or {}).get("fingerprint")
    if not target_fp:
        return {}
    sims: dict[str, float] = {}
    for donor, entry in probes.items():
        if donor == target:
            continue
        fp = (entry or {}).get("fingerprint")
        if not fp:
            continue
        sim = model_similarity(target_fp, fp)
        if sim is not None and sim > min_sim:
            sims[donor] = sim
    return sims
