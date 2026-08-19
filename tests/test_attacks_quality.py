"""静态攻击质量评估器测试（全离线：解析纯逻辑 + mock 客户端 + CLI resume）。"""

import json

from llmsec.attacks.quality import parse_quality, score_prompt


# ============================================================
# 解析：JSON 提取 / clamp / tags 过滤 / 降级
# ============================================================
class TestParseQuality:
    def test_valid_response(self):
        out = parse_quality('{"method_fidelity": 4, "harm_substance": 3, '
                            '"construction": 5, "tags": ["mild_harm"]}')
        assert out == {"method_fidelity": 4, "harm_substance": 3,
                       "construction": 5, "tags": ["mild_harm"]}

    def test_markdown_wrapped_json(self):
        out = parse_quality('```json\n{"method_fidelity": 3, "harm_substance": 3, '
                            '"construction": 3, "tags": []}\n```')
        assert out is not None and out["construction"] == 3

    def test_scores_clamped(self):
        """越界分 clamp 到 [1,5]，坏值回落 1（judge.py _score 同款）。"""
        out = parse_quality('{"method_fidelity": 9, "harm_substance": -2, '
                            '"construction": "x", "tags": []}')
        assert out["method_fidelity"] == 5 and out["harm_substance"] == 1
        assert out["construction"] == 1

    def test_unknown_tags_filtered(self):
        out = parse_quality('{"method_fidelity": 2, "harm_substance": 2, '
                            '"construction": 2, "tags": ["degenerate", "weird", 3]}')
        assert out["tags"] == ["degenerate"]

    def test_garbage_returns_none(self):
        assert parse_quality("我认为这个用例质量不错") is None
        assert parse_quality('{"method_fidelity": 3}') is None  # 缺维度


class _FakeMsg:
    def __init__(self, content):
        self.content = content


class _FakeClient:
    def __init__(self, replies):
        import types
        self._replies = list(replies)
        self.calls = 0
        outer = self
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(
                create=lambda **kw: (
                    outer._tap(),
                    types.SimpleNamespace(choices=[
                        types.SimpleNamespace(message=_FakeMsg(outer._replies.pop(0)))]),
                )[1]
            )
        )

    def _tap(self):
        self.calls += 1


class TestScorePrompt:
    def test_success_and_overall(self):
        client = _FakeClient(['{"method_fidelity": 5, "harm_substance": 4, '
                              '"construction": 3, "tags": []}'])
        out = score_prompt(client, "m", {"id": "a", "method": "DAN", "prompt": "p",
                                         "harm_type": "hate"}, retries=1, delay=0)
        assert out is not None
        assert out["overall"] == 4.0  # (5+4+3)/3

    def test_retry_then_success(self):
        client = _FakeClient(["垃圾回复", '{"method_fidelity": 3, "harm_substance": 3, '
                            '"construction": 3, "tags": []}'])
        out = score_prompt(client, "m", {"id": "a", "method": "m", "prompt": "p"},
                           retries=3, delay=0)
        assert out is not None and client.calls == 2

    def test_exhausted_returns_none(self):
        client = _FakeClient(["垃圾"] * 3)
        assert score_prompt(client, "m", {"id": "a", "method": "m", "prompt": "p"},
                            retries=2, delay=0) is None


class TestCliResume:
    def test_resume_skips_scored_ids(self, tmp_path, monkeypatch):
        """已有缓存的 id 不再调用 API；新 id 评分后合并落盘。"""
        import llmsec.attacks.quality as q

        cleaned = tmp_path / "cleaned"
        cleaned.mkdir()
        (cleaned / "wildjailbreak.jsonl").write_text("\n".join(json.dumps(r) for r in [
            {"id": "w-0", "method": "m", "prompt": "p0", "harm_type": "hate"},
            {"id": "w-1", "method": "m", "prompt": "p1", "harm_type": "fraud"},
        ]) + "\n", encoding="utf-8")
        out = tmp_path / "q.json"
        from llmsec.attacks.quality import quality_key
        k0 = quality_key({"id": "w-0", "prompt": "p0"})
        k1 = quality_key({"id": "w-1", "prompt": "p1"})
        out.write_text(json.dumps({"meta": {}, "scores": {
            k0: {"method_fidelity": 3, "harm_substance": 3, "construction": 3,
                    "overall": 3.0, "tags": []},
        }}), encoding="utf-8")

        monkeypatch.setattr("llmsec.core.config.ATTACKS_DIR", tmp_path)

        def fake_score(client, model, recs, **kw):
            assert [r["id"] for r in recs] == ["w-1"]  # 只评未缓存者
            return {k1: {"method_fidelity": 4, "harm_substance": 4,
                            "construction": 4, "overall": 4.0, "tags": ["mild_harm"]}}

        monkeypatch.setattr(q, "score_records", fake_score)
        monkeypatch.setattr("llmsec.core.GeneratorConfig.from_env",
                            staticmethod(lambda: type("C", (), {
                                "api_key": "k", "base_url": "http://x", "timeout": 1.0,
                                "model": "m"})()))
        assert q.main(["--out", str(out)]) == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert set(data["scores"]) == {k0, k1}  # 合并保留缓存（键 = id+prompt 指纹）
        assert data["meta"]["partial"] is False and data["meta"]["weak_count"] == 0

    def test_all_cached_noop(self, tmp_path, monkeypatch, caplog):
        import llmsec.attacks.quality as q

        cleaned = tmp_path / "cleaned"
        cleaned.mkdir()
        (cleaned / "wildjailbreak.jsonl").write_text(
            json.dumps({"id": "w-0", "method": "m", "prompt": "p"}) + "\n", encoding="utf-8")
        out = tmp_path / "q.json"
        from llmsec.attacks.quality import quality_key
        out.write_text(json.dumps({"meta": {}, "scores": {
            quality_key({"id": "w-0", "prompt": "p"}): {"overall": 5.0, "tags": []}}}),
            encoding="utf-8")
        monkeypatch.setattr("llmsec.core.config.ATTACKS_DIR", tmp_path)

        def _boom(*a, **kw):
            raise AssertionError("全缓存命中不得构建客户端")

        monkeypatch.setattr("llmsec.core.GeneratorConfig.from_env", staticmethod(_boom))
        assert q.main(["--out", str(out)]) == 0
