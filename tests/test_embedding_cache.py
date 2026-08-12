"""tests/test_embedding_cache — embedding 磁盘缓存测试。

验证：
  1. 首次调用编码全部 prompt 并写入缓存文件
  2. 第二次调用全部命中缓存，不触发 model.encode
  3. 部分命中时只编码未命中的 prompt
  4. work-dir 隔离：EMBEDDING_CACHE_FILE 被 rebind 后读写到隔离路径
"""
import numpy as np
import pytest

import llmsec.clustering.features as features


class _FakeModel:
    """记录 encode 调用次数的假 embedding 模型。"""

    def __init__(self, dim=8):
        self.dim = dim
        self.encode_calls = 0
        self.encoded_prompts: list = []

    def encode(self, prompts, show_progress_bar=False, batch_size=None, **kwargs):
        self.encode_calls += 1
        self.encoded_prompts.extend(prompts)
        # 确定性伪 embedding：每条 prompt 的 hash → 固定向量
        import hashlib
        rows = []
        for p in prompts:
            h = hashlib.sha256(p.encode()).digest()
            vec = np.array([(h[i % len(h)] / 255.0) for i in range(self.dim)], dtype=np.float64)
            rows.append(vec)
        return np.array(rows)


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """隔离 embedding 缓存到 tmp_path。"""
    cache_file = tmp_path / "embedding_cache.pkl"
    monkeypatch.setattr(
        "llmsec.core.config.EMBEDDING_CACHE_FILE", cache_file,
    )
    # 清除模块级缓存状态
    monkeypatch.setattr(features, "_embedding_source", "cache")
    monkeypatch.setattr(features, "_embedding_available", True)
    return cache_file


def test_cache_hit_on_second_call(isolated_cache, monkeypatch):
    """第二次调用全部命中缓存，encode 调用数为 0。"""
    fake = _FakeModel()
    prompts = [f"attack prompt {i}" for i in range(5)]

    # 首次：全 encode
    result1 = features._cached_encode(prompts, fake)
    assert fake.encode_calls == 1
    assert result1.shape == (5, 8)
    assert not isolated_cache.exists() or isolated_cache.stat().st_size > 0

    # 第二次：全命中
    fake2 = _FakeModel()
    result2 = features._cached_encode(prompts, fake2)
    assert fake2.encode_calls == 0
    np.testing.assert_array_almost_equal(result1, result2)


def test_partial_cache_hit(isolated_cache, monkeypatch):
    """部分命中时只编码未命中的 prompt。"""
    fake = _FakeModel()
    prompts_a = [f"prompt_a_{i}" for i in range(3)]
    prompts_b = [f"prompt_b_{i}" for i in range(2)]

    # 首批 3 条
    features._cached_encode(prompts_a, fake)
    assert fake.encode_calls == 1

    # 追加 2 新 + 1 旧 → 只 encode 2 新
    mixed = prompts_b + [prompts_a[0]]
    fake2 = _FakeModel()
    features._cached_encode(mixed, fake2)
    assert fake2.encode_calls == 1
    assert len(fake2.encoded_prompts) == 2  # 只编码了 2 条新 prompt


def test_empty_prompts(isolated_cache):
    """空 prompt 列表不崩溃。"""
    fake = _FakeModel()
    result = features._cached_encode([], fake)
    assert result.shape == (0, 0)
    assert fake.encode_calls == 0


def test_cache_key_changes_with_source(monkeypatch):
    """不同 embedding source 产生不同缓存键。"""
    monkeypatch.setattr(features, "_embedding_source", "api")
    monkeypatch.setenv("EMBEDDING_API_MODEL", "text-embedding-3-small")
    key1 = features._embedding_cache_key()

    monkeypatch.setattr(features, "_embedding_source", "cache")
    monkeypatch.setenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    key2 = features._embedding_cache_key()

    assert key1 != key2
