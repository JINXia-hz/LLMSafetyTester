"""harm_type 抽样重标校准器测试（全离线：抽样纯逻辑 + LLM 归类 mock，零网络）。"""

import json

import pytest

from llmsec.attacks.relabel import classify_prompt, parse_label, stratified_sample


# ============================================================
# 标签解析：全等之外的宽容提取（推理模型结论在末尾）
# ============================================================
class TestParseLabel:
    def test_bare_label(self):
        assert parse_label("violence") == "violence"

    def test_explained_label(self):
        assert parse_label("经过分析，该 prompt 的主要危害是制造武器，属于 violence") == "violence"

    def test_multiple_mentions_take_last(self):
        assert parse_label("这不是 fraud；综合判断应归为 privacy") == "privacy"

    def test_no_match_returns_none(self):
        assert parse_label("无法判断该内容类别") is None


# ============================================================
# 分层抽样：确定性 / 比例 / 截断
# ============================================================
class TestStratifiedSample:
    def _records(self):
        return ([{"id": f"w-{i}", "source": "wildjailbreak"} for i in range(90)]
                + [{"id": f"r-{i}", "source": "rubend18"} for i in range(10)])

    def test_proportional_quota(self):
        picked = stratified_sample(self._records(), 10, seed=42)
        srcs = [r["source"] for r in picked]
        assert srcs.count("wildjailbreak") == 9 and srcs.count("rubend18") == 1

    def test_deterministic_with_same_seed(self):
        a = stratified_sample(self._records(), 20, seed=7)
        b = stratified_sample(self._records(), 20, seed=7)
        assert [r["id"] for r in a] == [r["id"] for r in b]

    def test_sample_ge_total_returns_all(self):
        picked = stratified_sample(self._records()[:5], 10, seed=1)
        assert len(picked) == 5

    def test_small_source_gets_seat_when_share_sufficient(self):
        """小源份额足够（50 大 + 2 小，抽 26 → 小源配额恰 1）时拿到席位。"""
        recs = ([{"id": f"a-{i}", "source": "big"} for i in range(50)]
                + [{"id": f"b-{i}", "source": "small"} for i in range(2)])
        picked = stratified_sample(recs, 26, seed=3)
        assert len(picked) == 26
        assert sum(1 for r in picked if r["source"] == "small") == 1


# ============================================================
# LLM 归类（mock 客户端）
# ============================================================
class _FakeMsg:
    def __init__(self, content):
        self.content = content


class _FakeResp:
    def __init__(self, content):
        import types
        self.choices = [types.SimpleNamespace(message=_FakeMsg(content))]


class _FakeClient:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0
        self.chat = self

    def completions_create(self, **kw):
        raise NotImplementedError

    @property
    def completions(self):
        outer = self

        class _C:
            def create(self, **kw):
                outer.calls += 1
                reply = outer._replies.pop(0)
                if isinstance(reply, Exception):
                    raise reply
                return _FakeResp(reply)

        return _C()


class TestClassifyPrompt:
    def test_exact_label_parsed(self):
        client = _FakeClient(["  violence。\n"])
        assert classify_prompt(client, "m", "some attack text", retries=1, delay=0) == "violence"

    def test_garbage_then_valid_retries(self):
        from llmsec.attacks.relabel import _UnparseableLabel

        client = _FakeClient([_UnparseableLabel("我认为是暴力相关"), "hate"])
        assert classify_prompt(client, "m", "text", retries=3, delay=0) == "hate"
        assert client.calls == 2

    def test_exhausted_retries_raises(self):
        from llmsec.attacks.relabel import _UnparseableLabel

        client = _FakeClient([_UnparseableLabel("x")] * 3)
        with pytest.raises(_UnparseableLabel):  # 重试耗尽后最后一个异常上抛
            classify_prompt(client, "m", "text", retries=2, delay=0)
        assert client.calls == 2  # 首调 + 1 次重试


class TestCliDryRun:
    def test_dry_run_samples_without_api(self, tmp_path, monkeypatch):
        """dry-run：只抽样落清单，不构建 API 客户端。"""
        import llmsec.attacks.relabel as rl

        cleaned = tmp_path / "cleaned"
        cleaned.mkdir()
        rows = [{"id": f"w-{i}", "source": "wildjailbreak", "harm_type": "other",
                 "method": "m", "prompt": f"p{i}"} for i in range(30)]
        rows += [{"id": "x-1", "source": "wildjailbreak", "harm_type": "hate",
                  "method": "m", "prompt": "skip"}]  # 非 other 不入池
        (cleaned / "wildjailbreak.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        for name in ("in_the_wild", "rubend18", "jailbreakv28k", "jailbreakdb"):
            (cleaned / f"{name}.jsonl").write_text("", encoding="utf-8")

        monkeypatch.setattr("llmsec.core.config.ATTACKS_DIR", tmp_path)

        def _boom():
            raise AssertionError("dry-run 不得构建 API 客户端")

        monkeypatch.setattr("llmsec.core.GeneratorConfig.from_env", staticmethod(_boom))
        out = tmp_path / "sample.json"
        assert rl.main(["--sample", "10", "--seed", "1", "--dry-run",
                        "--out", str(out)]) == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["meta"]["dry_run"] is True and len(data["records"]) == 10
        assert all(r["id"].startswith("w-") for r in data["records"])


# ============================================================
# 回写：保守规则 + 溯源字段
# ============================================================
class TestApplyLabels:
    def _setup(self, tmp_path):
        cleaned = tmp_path / "cleaned"
        cleaned.mkdir()
        rows = [
            {"id": "a-1", "method": "m", "prompt": "p", "harm_type": "other"},
            {"id": "a-2", "method": "m", "prompt": "p", "harm_type": "other",
             "harm_original": "weird"},                       # 已有溯源不覆盖
            {"id": "a-3", "method": "m", "prompt": "p", "harm_type": "hate"},  # 非 other 不动
            {"id": "a-4", "method": "m", "prompt": "p", "harm_type": "other"},  # 预测仍 other
            {"id": "a-5", "method": "m", "prompt": "p", "harm_type": "other"},  # 报告未含
        ]
        (cleaned / "wildjailbreak.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        report = tmp_path / "report.json"
        report.write_text(json.dumps({
            "records": [
                {"id": "a-1", "predicted": "fraud"},
                {"id": "a-2", "predicted": "violence"},
                {"id": "a-3", "predicted": "fraud"},   # 原 hate：不回写
                {"id": "a-4", "predicted": "other"},
            ],
        }), encoding="utf-8")
        return cleaned / "wildjailbreak.jsonl", report

    def test_conservative_apply_rules(self, tmp_path):
        from llmsec.attacks.relabel import apply_labels

        data_file, report = self._setup(tmp_path)
        stats = apply_labels(report, [data_file])
        assert stats == {"matched": 4, "relabeled": 2, "kept_other": 1}
        rows = {r["id"]: r for r in (
            json.loads(l) for l in data_file.read_text(encoding="utf-8").splitlines() if l.strip())}
        assert rows["a-1"]["harm_type"] == "fraud"
        assert rows["a-1"]["harm_original"] == "other" and rows["a-1"]["repaired"]["relabel"] is True
        assert rows["a-2"]["harm_original"] == "weird"          # 既有溯源保留
        assert rows["a-3"]["harm_type"] == "hate"               # 非 other 不动
        assert rows["a-4"]["harm_type"] == "other"              # 预测 other 保持
        assert rows["a-5"]["harm_type"] == "other"              # 报告未含不动
