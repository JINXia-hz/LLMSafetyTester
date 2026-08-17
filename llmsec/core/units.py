"""
core.units — 聚类簇作为评级/采样单位（unit）

设计背景：攻击集方法标签可能退化（all_merged：10157 行 = 10157 个"方法"，每行一条），
"method=稳定可评级技术"不再成立。体系的评级/采样单位改为**聚类簇（unit）**：

  - 聚类点 = method（其代表 prompt 的特征；all_merged 下 method ≡ prompt，天然一致）
  - unit = Ward 关键层切割的一个簇（主 labels 全覆盖，无噪声点遗留问题）
  - unit_id = 成员指纹 "c_<md5>"：同攻击集 + 同特征配置 + 确定性聚类 → 跨 run/跨模型
    稳定重现，作为 R 矩阵与 resume 的持久 key
  - method 保留为记录元数据（来源/展示），不再作为任何评级 key

一次实测 = 从 unit 的未测记录池中取一条 prompt 发送；unit 的 Elo 随多次实测累积。
"""
from __future__ import annotations

import hashlib

import numpy as np


def unit_fingerprint(members: list[str]) -> str:
    """簇成员指纹 → 稳定 unit_id。"""
    digest = hashlib.md5(",".join(sorted(members)).encode("utf-8")).hexdigest()
    return f"c_{digest[:10]}"


def method_set_hash(methods: list[str]) -> str:
    """方法集合指纹 hash（判断攻击集是否变化）。

    r7：原 hdb._method_set_hash 与 cold_start._compute_method_set_hash 为
    逐字节重复实现，统一到本定义处（改哈希口径只动这一处）。
    """
    return hashlib.md5(",".join(sorted(set(methods))).encode("utf-8")).hexdigest()


def build_units(
    labels: dict[str, int],
    method_records: dict[str, dict],
    *,
    method_pool: dict[str, list[dict]] | None = None,
    coords: dict[str, np.ndarray] | None = None,
    cluster_names: dict[int, str] | None = None,
) -> dict[str, dict]:
    """按簇标签构建 unit 表。

    参数:
        labels: {method: 簇标签}（Ward 关键层主标签，全覆盖）
        method_records: {method: 代表记录}（特征提取口径：首条记录）
        method_pool: {method: [全部记录]}（unit 内 prompt 轮换池；None 时各 method 仅代表记录）
        coords: {method: 白化坐标}（求 medoid；可空 → 取首成员）
        cluster_names: {label: 簇名}（可空 → "簇{label}"）

    返回: {unit_id: {"unit_id", "label", "name", "members", "size",
                     "medoid", "pool": [(method, record), ...]}}
    """
    by_label: dict[int, list[str]] = {}
    for m, lab in labels.items():
        if m in method_records:
            by_label.setdefault(int(lab), []).append(m)

    units: dict[str, dict] = {}
    for lab, members in sorted(by_label.items()):
        members = sorted(members)
        uid = unit_fingerprint(members)
        medoid = members[0]
        if coords:
            vecs = [np.asarray(coords[m], dtype=float) for m in members if m in coords]
            if len(vecs) == len(members) and vecs:
                centroid = np.mean(vecs, axis=0)
                d = [float(np.linalg.norm(v - centroid)) for v in vecs]
                medoid = members[int(np.argmin(d))]
        pool: list[tuple[str, dict]] = []
        for m in members:
            recs = (method_pool or {}).get(m) or [method_records[m]]
            pool.extend((m, r) for r in recs)
        units[uid] = {
            "unit_id": uid,
            "label": lab,
            "name": (cluster_names or {}).get(lab) or f"簇{lab}",
            "members": members,
            "size": len(members),
            "medoid": medoid,
            "pool": pool,
        }
    return units


def build_unit_features(features: dict[str, dict], units: dict[str, dict]) -> dict[str, dict]:
    """unit 级特征 = 成员各特征块逐块均值（质心）。

    供 SVD-Ridge / Blend / D-optimal 以 unit 为键工作——预测器对键不可知，
    只需把 {method: 特征块} 换成 {unit_id: 均值特征块}。
    """
    out: dict[str, dict] = {}
    for uid, u in units.items():
        member_feats = [features[m] for m in u["members"] if m in features]
        if not member_feats:
            continue
        agg: dict[str, np.ndarray] = {}
        blocks = {b for f in member_feats for b in f}
        for b in blocks:
            vecs = [np.atleast_1d(np.asarray(f[b], dtype=float)) for f in member_feats if b in f]
            if vecs:
                agg[b] = np.mean(vecs, axis=0)
        out[uid] = agg
    return out


def build_unit_proxy_records(units: dict[str, dict]) -> dict[str, dict]:
    """unit → 代理记录（medoid 记录的副本，method/id 改写为 unit_id）。

    供预测器 prior 特征（build_prior_features 需要 record）与 D-optimal 种子等
    "按记录取值"的既有接口以 unit 为键工作。代理记录的 prompt 只用于先验特征，
    实际实测的 prompt 由 pool 轮换决定。
    """
    proxies: dict[str, dict] = {}
    for uid, u in units.items():
        rec = None
        for m, r in u["pool"]:
            if m == u["medoid"]:
                rec = r
                break
        if rec is None:
            rec = u["pool"][0][1]
        proxy = dict(rec)
        proxy["method"] = uid
        proxy["id"] = uid
        proxies[uid] = proxy
    return proxies


def assemble_units(
    labels: dict[str, int],
    method_records: dict[str, dict],
    method_pool: dict[str, list[dict]],
    artifacts: dict | None,
) -> dict[str, dict]:
    """从簇标签 + 特征 artifacts 装配完整 unit 表（medoid 坐标 + 确定性命名）。

    runner 层调用一次、多目标共享（输入同一份 features/labels，结果确定）。
    命名用确定性 auto_name_clusters（不调 LLM；AI 重命名留给 post-test 最终聚类）。
    """
    from llmsec.clustering.pipeline import auto_name_clusters
    from llmsec.clustering.space import build_whitened_space

    features = (artifacts or {}).get("features") or {}
    meta = (artifacts or {}).get("meta") or {}
    methods = sorted(labels.keys())

    coords: dict[str, np.ndarray] | None = None
    feat_avail = [m for m in methods if m in features]
    if len(feat_avail) >= 2:
        try:
            space = build_whitened_space(features, feat_avail)
            coords = {m: space["coords"][i] for i, m in enumerate(feat_avail)}
        except Exception:
            coords = None  # medoid 退化为首成员，不影响正确性

    try:
        names = auto_name_clusters(labels, features, meta, meta.get("method_prompts", {}))
    except Exception:
        names = {}
    return build_units(labels, method_records, method_pool=method_pool,
                       coords=coords, cluster_names=names)

