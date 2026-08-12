"""聚类可视化路由：特征空间 2D 投影（PCA/t-SNE）与层次树切割。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from llmsec.core.config import CLUSTER_RESULT_FILE
from llmsec.core.logging import get_logger
from llmsec.core.seed import get_global_seed as _get_seed
from llmsec.server.routers.data_query import _load_state, _load_tree_artifacts

logger = get_logger(__name__)

_SEED = _get_seed()

router = APIRouter()

# ============================================================
# 聚类特征空间投影（PCA / t-SNE，按需计算 + 缓存）
# ============================================================
# 缓存大小上限：超出时按插入顺序淘汰最旧条目，防长期运行内存单调增长
_CACHE_MAX_SIZE = 64

_PROJECTION_CACHE: dict[tuple[str, float], dict] = {}
_PROJECTION_BLOCKS = ("textual", "embedding", "technique", "intent")


def _cache_put(cache: dict, key, value) -> None:
    """写入缓存并维护 _CACHE_MAX_SIZE 上限（dict 保序，弹掉头一个即最旧条目）。"""
    cache[key] = value
    while len(cache) > _CACHE_MAX_SIZE:
        cache.pop(next(iter(cache)))


def _build_feature_matrix(features: dict, methods: list[str]):
    """拼接 textual+embedding+technique+intent 特征块为矩阵（块维度不一致零填充）。"""
    import numpy as np

    dims = {b: 0 for b in _PROJECTION_BLOCKS}
    vecs = {}
    for m in methods:
        feat = features.get(m, {})
        v = {}
        for b in _PROJECTION_BLOCKS:
            vec = np.atleast_1d(np.asarray(feat.get(b, np.zeros(0)), dtype=np.float64))
            v[b] = vec
            dims[b] = max(dims[b], vec.shape[0])
        vecs[m] = v

    rows = []
    for m in methods:
        parts = []
        for b in _PROJECTION_BLOCKS:
            vec = vecs[m][b]
            if vec.shape[0] < dims[b]:
                vec = np.pad(vec, (0, dims[b] - vec.shape[0]))
            parts.append(vec)
        rows.append(np.concatenate(parts))
    return np.array(rows, dtype=np.float64)


@router.get("/api/cluster-projection")
async def api_cluster_projection(method: str = "pca"):
    """
    对聚类 artifacts 中的高维特征做 2D 投影（PCA / t-SNE），供分布散点图使用。
    结果按 (method, artifacts mtime) 缓存。
    """
    import joblib
    import numpy as np

    if method not in ("pca", "tsne"):
        raise HTTPException(status_code=400, detail=f"不支持的投影方法: {method!r}")
    if not CLUSTER_RESULT_FILE.exists():
        return {"available": False}

    mtime = CLUSTER_RESULT_FILE.stat().st_mtime
    cache_key = (method, mtime)
    if cache_key in _PROJECTION_CACHE:
        return _PROJECTION_CACHE[cache_key]

    try:
        artifacts = joblib.load(CLUSTER_RESULT_FILE)
    except Exception:
        return {"available": False}

    features = artifacts.get("features", {})
    labels = artifacts.get("labels", {})
    cluster_names = artifacts.get("cluster_names", {})
    gt_methods = set(artifacts.get("ground_truth_methods", []))
    if not features:
        return {"available": False}

    methods = sorted(features.keys())
    n = len(methods)
    X = _build_feature_matrix(features, methods)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    result: dict = {"available": True, "method": method, "n": n}
    if n < 2:
        coords = np.zeros((n, 2))
    elif method == "pca":
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=_SEED)
        coords = pca.fit_transform(X)
        result["explained_variance"] = [
            round(float(r), 4) for r in pca.explained_variance_ratio_
        ]
    else:
        from sklearn.manifold import TSNE

        # sklearn 要求 perplexity < n，小样本自适应收缩
        perplexity = max(1, min(30, (n - 1) // 3))
        tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=_SEED)
        coords = tsne.fit_transform(X)
        result["perplexity"] = perplexity

    state = _load_state()
    ratings = state.get("attacker_ratings", {})

    # 评级键 = unit_id：method → unit 反查（units 表随聚类产物持久化）
    method_to_unit = {}
    for uid, u in (artifacts.get("units") or {}).items():
        for mem in u.get("members", []):
            method_to_unit[mem] = uid

    points = []
    for i, m in enumerate(methods):
        cid = labels.get(m, -1)
        try:
            cid = int(cid)
        except (TypeError, ValueError):
            cid = -1
        uid = method_to_unit.get(m)
        points.append({
            "method": m,
            "unit": uid,
            "x": round(float(coords[i, 0]), 4),
            "y": round(float(coords[i, 1]), 4),
            "cluster": cid,
            "cluster_name": cluster_names.get(str(cid), f"簇{cid}"),
            "tested": uid in gt_methods if uid else False,
            "elo": round(ratings[uid], 1) if uid and uid in ratings else None,
        })

    result["points"] = points
    _cache_put(_PROJECTION_CACHE, cache_key, result)
    return result


# ============================================================
# 聚类层次树（树图 + 任意层切割）
# ============================================================
_CUT_CACHE: dict[tuple[int, float], dict] = {}


@router.get("/api/cluster-tree")
async def api_cluster_tree():
    """返回层次树的树图坐标（scipy dendrogram 的 icoord/dcoord）与 auto-k 信息。"""
    artifacts = _load_tree_artifacts()
    if artifacts is None:
        return {"available": False}

    from scipy.cluster.hierarchy import dendrogram

    Z = artifacts["linkage"]
    labels = artifacts.get("labels", {})
    methods = sorted(labels.keys())
    n = len(labels)
    dd = dendrogram(Z, no_plot=True)
    # 叶节点方法名（左→右顺序）：dendrogram 的 leaves 是原始观测索引，对应 sorted(labels)
    leaf_order = dd.get("leaves", [])
    leaves = [methods[i] for i in leaf_order if isinstance(i, int) and 0 <= i < len(methods)]

    # maxclust=k 对应的切割高度：第 n-k 与 n-k+1 次合并高度之间
    heights = sorted(float(h) for h in Z[:, 2])

    def cut_height(k: int) -> float:
        if k <= 1:
            return heights[-1] * 1.05 if heights else 1.0
        if k >= n:
            return 0.0
        return (heights[n - k - 1] + heights[n - k]) / 2

    chosen_k = artifacts.get("chosen_k") or len(set(labels.values()) - {-1})
    return {
        "available": True,
        "n": n,
        "leaves": leaves,
        "icoord": dd["icoord"],
        "dcoord": dd["dcoord"],
        "merge_heights": heights,
        "chosen_k": chosen_k,
        "chosen_height": cut_height(chosen_k),
        "top_ks": artifacts.get("top_ks", [chosen_k]),
        "candidate_sweep": artifacts.get("candidate_sweep", []),
        "max_height": heights[-1] if heights else 1.0,
    }


@router.get("/api/cluster-cut")
async def api_cluster_cut(k: int):
    """在层次树上切出 k 个簇（fcluster O(n)），返回该层簇结构与命名。"""

    artifacts = _load_tree_artifacts()
    if artifacts is None:
        return {"available": False}

    labels = artifacts.get("labels", {})
    n = len(labels)
    if k < 2 or k > n:
        raise HTTPException(status_code=400, detail=f"k 必须在 [2, {n}] 内")

    mtime = CLUSTER_RESULT_FILE.stat().st_mtime
    cache_key = (k, mtime)
    if cache_key in _CUT_CACHE:
        return _CUT_CACHE[cache_key]

    from scipy.cluster.hierarchy import fcluster

    from llmsec.clustering.pipeline import auto_name_clusters

    Z = artifacts["linkage"]
    methods = sorted(labels.keys())
    raw = fcluster(Z, t=k, criterion="maxclust")
    cut_labels = {m: int(c) - 1 for m, c in zip(methods, raw)}

    names = auto_name_clusters(
        cut_labels,
        artifacts.get("features", {}),
        artifacts.get("meta", {}),
        artifacts.get("meta", {}).get("method_prompts", {}),
    )

    clusters: dict[int, list[str]] = {}
    for m, cid in cut_labels.items():
        clusters.setdefault(cid, []).append(m)

    state = _load_state()
    ratings = state.get("attacker_ratings", {})
    # 评级键 = unit_id：method → unit 反查（任意 k 层切割的成员仍是 method）
    method_to_unit = {}
    for uid, u in (artifacts.get("units") or {}).items():
        for mem in u.get("members", []):
            method_to_unit[mem] = uid

    result = {
        "available": True,
        "k": k,
        "clusters": [
            {
                "id": cid,
                "name": names.get(cid, f"簇{cid}"),
                "size": len(members),
                "members": sorted(members),
                "mean_elo": (
                    round(sum(ratings.get(method_to_unit.get(m), 1500.0) for m in members) / len(members), 1)
                    if members else None
                ),
            }
            for cid, members in sorted(clusters.items())
        ],
    }
    _cache_put(_CUT_CACHE, cache_key, result)
    return result
