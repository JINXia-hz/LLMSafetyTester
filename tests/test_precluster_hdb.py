"""预聚类复用 HDBSCAN 核心的回归测试（tests/test_precluster_hdb.py）。

背景：_quick_precluster 原用 KMeans（k=n//3 启发式），现与 post-test 聚类
同口径复用 clustering.hdb.compute_cluster_labels（阻尼白化 → HDBSCAN 密度视图
→ Ward auto-k 主标签）。本文件锁定：
  - 复用核心而非整条 run_hdbscan_clustering（不做命名/画像/落盘）
  - 簇结构恢复 + 完全确定性
  - hdbscan 缺失/核心异常时回退 KMeans，再失败返回 None
"""
from types import SimpleNamespace

import numpy as np

from llmsec.pipeline.attack_phase import _quick_precluster


def _make_blob_features(n_per_cluster: int = 4, dim: int = 6, seed: int = 42):
    """造 3 个分离度极高的团簇（textual 特征块），返回 (features, true_labels)。"""
    rng = np.random.default_rng(seed)
    centers = [np.zeros(dim), np.full(dim, 10.0), np.full(dim, 20.0)]
    features, true_labels = {}, {}
    for ci, center in enumerate(centers):
        for j in range(n_per_cluster):
            m = f"method_c{ci}_{j}"
            features[m] = {"textual": center + rng.normal(0, 0.05, dim)}
            true_labels[m] = ci
    return features, true_labels


def _make_tracker(features: dict | None):
    """_quick_precluster 只需要 tracker.predictor.artifacts['features']。"""
    artifacts = {"features": features} if features else {}
    return SimpleNamespace(predictor=SimpleNamespace(artifacts=artifacts))


def test_precluster_recovers_blob_structure():
    """3 个明显团簇 → Ward auto-k 恢复的划分应与真实划分近乎一致。"""
    from sklearn.metrics import adjusted_rand_score

    features, true_labels = _make_blob_features()
    methods = sorted(features.keys())
    labels = _quick_precluster(_make_tracker(features), methods)
    assert labels is not None and set(labels.keys()) == set(methods)
    ari = adjusted_rand_score([true_labels[m] for m in methods],
                              [labels[m] for m in methods])
    assert ari >= 0.9, f"簇结构恢复失败: ARI={ari}"


def test_precluster_deterministic():
    """HDBSCAN/Ward 均无随机性：两次调用标签必须完全一致。"""
    features, _ = _make_blob_features()
    methods = sorted(features.keys())
    a = _quick_precluster(_make_tracker(features), methods)
    b = _quick_precluster(_make_tracker(features), methods)
    assert a == b


def test_precluster_uses_core_not_full_pipeline(monkeypatch):
    """预聚类只走 compute_cluster_labels 核心——命名/画像（潜在 LLM 调用）不得触发。"""
    import llmsec.clustering.hdb as hdb

    def _forbidden(*args, **kwargs):
        raise RuntimeError("预聚类不应触发命名/画像（整条 run_hdbscan_clustering 路径）")

    monkeypatch.setattr(hdb, "auto_name_clusters", _forbidden)
    monkeypatch.setattr(hdb, "ai_rename_clusters", _forbidden)
    monkeypatch.setattr(hdb, "build_cluster_profiles", _forbidden)

    features, _ = _make_blob_features()
    labels = _quick_precluster(_make_tracker(features), sorted(features.keys()))
    assert labels is not None


def test_precluster_kmeans_fallback_on_import_error(monkeypatch):
    """hdbscan 不可用（ImportError）→ 回退 KMeans 仍返回完整标签。"""
    import llmsec.clustering.hdb as hdb

    def _raise_import_error(features, feature_weights=None):
        raise ImportError("No module named 'hdbscan'")

    monkeypatch.setattr(hdb, "compute_cluster_labels", _raise_import_error)

    features, _ = _make_blob_features()
    methods = sorted(features.keys())
    labels = _quick_precluster(_make_tracker(features), methods)
    assert labels is not None and set(labels.keys()) == set(methods)


def test_precluster_kmeans_fallback_on_core_error(monkeypatch):
    """核心返回 error 分支 → 同样回退 KMeans。"""
    import llmsec.clustering.hdb as hdb

    monkeypatch.setattr(hdb, "compute_cluster_labels",
                        lambda features, feature_weights=None: {"error": "方法数不足", "labels": {}})

    features, _ = _make_blob_features()
    methods = sorted(features.keys())
    labels = _quick_precluster(_make_tracker(features), methods)
    assert labels is not None and set(labels.keys()) == set(methods)


def test_precluster_both_engines_fail_returns_none(monkeypatch):
    """HDBSCAN 核心与 KMeans 回退都失败 → None（sampler 退化全局模式，不崩溃）。"""
    import llmsec.clustering.hdb as hdb

    monkeypatch.setattr(hdb, "compute_cluster_labels",
                        lambda features, feature_weights=None: 1 / 0)

    features, _ = _make_blob_features()
    # KMeans 回退路径的 build_whitened_space 也打爆
    import llmsec.clustering.space as space_mod

    monkeypatch.setattr(space_mod, "build_whitened_space",
                        lambda *a, **k: 1 / 0)
    assert _quick_precluster(_make_tracker(features), sorted(features.keys())) is None


def test_precluster_too_few_methods():
    """方法数 <4 直接 None（不值得聚类）。"""
    features, _ = _make_blob_features(n_per_cluster=1)  # 3 个方法
    assert _quick_precluster(_make_tracker(features), sorted(features.keys())) is None


def test_precluster_no_features():
    """特征缓存为空 → None。"""
    assert _quick_precluster(_make_tracker(None), ["a", "b", "c", "d"]) is None


def test_compute_cluster_labels_n_lt_2():
    """核心守卫：方法数 <2 返回 error 分支（labels 全 0）。"""
    from llmsec.clustering.hdb import compute_cluster_labels

    core = compute_cluster_labels({"only_one": {"textual": np.zeros(4)}})
    assert core.get("error") and core["labels"] == {"only_one": 0}


def test_run_hdbscan_clustering_n_lt_2_shape():
    """主管线 n<2 返回结构保持原样（重构回归）。"""
    from llmsec.clustering.hdb import run_hdbscan_clustering

    report = run_hdbscan_clustering({"only_one": {"textual": np.zeros(4)}}, meta={})
    assert report.get("error") and report["labels"] == {"only_one": 0}
