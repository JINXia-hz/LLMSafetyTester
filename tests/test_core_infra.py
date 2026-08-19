#!/usr/bin/env python3
"""core 基础设施测试：io 错误路径（损坏兜底 / 原子写失败 / tmp 清理）+ monitoring 告警评估。

全部离线：monitoring 的 webhook 通道一律 mock（_post_webhook / _get_executor），
绝不发真实网络请求；告警断言走 caplog（P9：事件文件通道已删，日志是主通道）。
"""

from __future__ import annotations

import pytest


# ============================================================
# io：read_json 兜底与损坏
# ============================================================
def test_read_json_missing_returns_default(tmp_path):
    """文件不存在 → 返回 default（None 或调用方指定值），strict 与否都一样。"""
    from llmsec.core.io import read_json

    assert read_json(tmp_path / "none.json") is None, "❌1 缺省 default 应为 None"
    assert read_json(tmp_path / "none.json", {"d": 1}) == {"d": 1}, "❌2 应返回指定 default"
    assert read_json(tmp_path / "none.json", strict=True) is None, "❌3 strict 也不影响缺失分支"


def test_read_json_corrupted_lenient_vs_strict(tmp_path):
    """损坏 JSON：非 strict 静默返回 default；strict 抛 CorruptedFileError（带路径与原因）。"""
    from llmsec.core.io import CorruptedFileError, read_json

    p = tmp_path / "bad.json"
    p.write_text("{not-json", encoding="utf-8")
    assert read_json(p, "fallback") == "fallback", "❌1 非 strict 应静默返回 default"
    with pytest.raises(CorruptedFileError) as ei:
        read_json(p, strict=True)
    assert str(p) == ei.value.path, "❌2 异常应携带文件路径"
    assert ei.value.cause is not None, "❌3 异常应携带底层原因"


def test_read_json_directory_oserror_path(tmp_path):
    """路径是目录：open 抛 OSError → 非 strict 返回 default，strict 原样上抛 OSError。

    r7/M-4 契约变更：IO 错误（权限/占用/目录）不再伪装成 CorruptedFileError——
    "损坏"专指内容解析失败（半写/非法 JSON），混同会让完好文件被误判触发
    .corrupt.bak 回退旧数据。详见 tests/test_audit_r7_storage.py。
    """
    from llmsec.core.io import read_json

    d = tmp_path / "dir.json"
    d.mkdir()
    assert read_json(d, "fb") == "fb", "❌1 目录路径非 strict 应静默返回 default"
    with pytest.raises(OSError):
        read_json(d, strict=True)


# ============================================================
# io：JSONL 读写与异常路径
# ============================================================
def test_jsonl_skips_bad_and_blank_lines(tmp_path):
    """iter/read_jsonl：坏行跳过、空行忽略；文件不存在产出空。"""
    from llmsec.core.io import read_jsonl

    p = tmp_path / "x.jsonl"
    p.write_text('{"a":1}\n\nnot-json\n{"a":2}\n   \n', encoding="utf-8")
    assert read_jsonl(p) == [{"a": 1}, {"a": 2}], "❌1 应只保留可解析的非空行"
    assert read_jsonl(tmp_path / "missing.jsonl") == [], "❌2 缺失文件应为空列表"


def test_write_jsonl_atomic_replace_retry_and_cleanup(tmp_path, monkeypatch):
    """write_jsonl：正常覆写；os.replace 持续 PermissionError → 重试后抛出，tmp 清理、原文件不损。"""
    from llmsec.core import io as io_mod
    from llmsec.core.io import read_jsonl, write_jsonl

    p = tmp_path / "out.jsonl"
    write_jsonl(p, [{"id": 1}, {"id": 2}])
    assert read_jsonl(p) == [{"id": 1}, {"id": 2}], "❌1 覆写后应可读回"

    def _deny(*a, **k):
        raise PermissionError("WinError 5 模拟并发占用")

    monkeypatch.setattr(io_mod.os, "replace", _deny)
    with pytest.raises(PermissionError):
        write_jsonl(p, [{"id": 3}])
    # 原子性：目标文件保持旧内容，无 .tmp 残留
    assert read_jsonl(p) == [{"id": 1}, {"id": 2}], "❌2 失败后目标应保持原内容"
    leftovers = [f.name for f in tmp_path.iterdir() if ".tmp." in f.name]
    assert leftovers == [], f"❌3 tmp 应被清理，残留: {leftovers}"


def test_append_jsonl_appends_lines(tmp_path):
    """append_jsonl：自动建父目录，逐次追加不覆盖。"""
    from llmsec.core.io import append_jsonl, read_jsonl

    p = tmp_path / "sub" / "log.jsonl"
    append_jsonl(p, {"n": 1})
    append_jsonl(p, {"n": 2})
    assert read_jsonl(p) == [{"n": 1}, {"n": 2}], "❌1 两次追加应都保留"


def test_load_done_ids_collects_keys(tmp_path):
    """load_done_ids：提取 key 字段集合，缺 key 行跳过，文件缺失返回空集。"""
    from llmsec.core.io import load_done_ids, write_jsonl

    p = tmp_path / "done.jsonl"
    write_jsonl(p, [{"id": "a"}, {"no_id": 1}, {"id": "b"}, {"id": "a"}])
    assert load_done_ids(p) == {"a", "b"}, "❌1 应去重收集 id"
    assert load_done_ids(tmp_path / "ghost.jsonl") == set(), "❌2 缺失文件应为空集"


# ============================================================
# io：write_json 各变体（indent / 非原子 / 备份 / NaN / numpy）
# ============================================================
def test_write_json_indent_and_non_atomic(tmp_path):
    """write_json：indent 生效；atomic=False 直写不走 replace。"""
    from llmsec.core.io import read_json, write_json

    p = tmp_path / "a.json"
    write_json(p, {"x": 1}, indent=4, atomic=False)
    text = p.read_text(encoding="utf-8")
    assert text.startswith('{\n    "x": 1'), f"❌1 indent=4 未生效: {text!r}"
    assert read_json(p) == {"x": 1}, "❌2 直写内容应可读回"


def test_write_json_backup_keeps_previous(tmp_path):
    """write_json(backup=True)：写前把现有文件复制为 .bak，旧内容可找回。"""
    from llmsec.core.io import read_json, write_json

    p = tmp_path / "b.json"
    write_json(p, {"v": 1})
    write_json(p, {"v": 2}, backup=True)
    assert read_json(p) == {"v": 2}, "❌1 新内容应生效"
    assert read_json(p.with_name(p.name + ".bak")) == {"v": 1}, "❌2 .bak 应保留旧内容"


def test_write_json_atomic_failure_cleans_tmp(tmp_path, monkeypatch):
    """write_json 原子写 replace 失败 → 抛出且不留 .tmp 残留。"""
    from llmsec.core import io as io_mod
    from llmsec.core.io import write_json

    def _deny(*a, **k):
        raise PermissionError("replace 被拒")

    monkeypatch.setattr(io_mod.os, "replace", _deny)
    p = tmp_path / "c.json"
    with pytest.raises(PermissionError):
        write_json(p, {"x": 1})
    assert not p.exists(), "❌1 失败后目标不应存在"
    leftovers = [f.name for f in tmp_path.iterdir() if f.name.endswith(".tmp")]
    assert leftovers == [], f"❌2 tmp 应被清理，残留: {leftovers}"


def test_write_json_rejects_nan_and_cleans_tmp(tmp_path):
    """write_json(allow_nan=False)：NaN → ValueError（防非法 JSON 字面量落盘）且清理 tmp。"""
    from llmsec.core.io import write_json

    p = tmp_path / "d.json"
    with pytest.raises(ValueError):
        write_json(p, {"v": float("nan")}, allow_nan=False)
    leftovers = [f.name for f in tmp_path.iterdir() if f.name.endswith(".tmp")]
    assert leftovers == [], f"❌1 序列化失败也应清理 tmp，残留: {leftovers}"


def test_write_json_serialization_error_cleans_tmp(tmp_path):
    """不可序列化对象 → default 抛 TypeError → 传播且无 .tmp 残留。"""
    from llmsec.core.io import write_json

    p = tmp_path / "e.json"
    with pytest.raises(TypeError):
        write_json(p, {"s": {1, 2}})
    leftovers = [f.name for f in tmp_path.iterdir() if f.name.endswith(".tmp")]
    assert leftovers == [], f"❌1 tmp 应被清理，残留: {leftovers}"


def test_write_json_numpy_types_roundtrip(tmp_path):
    """numpy 标量/数组/布尔经 default 自动转原生类型，读回等值。"""
    import numpy as np

    from llmsec.core.io import read_json, write_json

    p = tmp_path / "np.json"
    write_json(p, {"v": np.float32(1.5), "i": np.int64(7),
                   "arr": np.array([1, 2]), "flag": np.bool_(True)})
    assert read_json(p) == {"v": 1.5, "i": 7, "arr": [1, 2], "flag": True}, \
        "❌1 numpy 类型应转为原生等值"


# ============================================================
# io：二进制 artifacts 与 CSV
# ============================================================
def test_artifacts_roundtrip_backup_and_corruption(tmp_path):
    """save/load_artifact：roundtrip、backup、损坏/缺失的 strict 与非 strict 行为。"""
    from llmsec.core.io import CorruptedFileError, load_artifact, save_artifact

    p = tmp_path / "m.pkl"
    save_artifact(p, {"k": [1, 2, 3]})
    assert load_artifact(p) == {"k": [1, 2, 3]}, "❌1 应无损读回"
    save_artifact(p, {"k": [9]}, backup=True)
    assert load_artifact(p.with_name(p.name + ".bak")) == {"k": [1, 2, 3]}, "❌2 .bak 应存旧值"

    bad = tmp_path / "bad.pkl"
    bad.write_bytes(b"\x00\x01garbage")
    assert load_artifact(bad, "fb") == "fb", "❌3 非 strict 损坏文件应返回 default"
    with pytest.raises(CorruptedFileError):
        load_artifact(bad, strict=True)
    assert load_artifact(tmp_path / "ghost.pkl") is None, "❌4 缺失文件应返回 default"


def test_save_artifact_atomic_failure_cleans_tmp(tmp_path, monkeypatch):
    """save_artifact 原子写 replace 失败 → 重试后抛出，原 artifact 完好、tmp 清理。"""
    from llmsec.core import io as io_mod
    from llmsec.core.io import load_artifact, save_artifact

    p = tmp_path / "s.pkl"
    save_artifact(p, {"a": 1})

    def _deny(*a, **k):
        raise PermissionError("WinError 5 模拟并发占用")

    monkeypatch.setattr(io_mod.os, "replace", _deny)
    with pytest.raises(PermissionError):
        save_artifact(p, {"a": 2})
    assert load_artifact(p) == {"a": 1}, "❌1 失败后原 artifact 应完好"
    leftovers = [f.name for f in tmp_path.iterdir() if ".tmp." in f.name]
    assert leftovers == [], f"❌2 带唯一后缀的 tmp 应被清理，残留: {leftovers}"


def test_write_csv_empty_and_rows(tmp_path):
    """write_csv：空 rows 仍创建空文件；有数据时首行为表头。"""
    from llmsec.core.io import write_csv

    p = tmp_path / "t.csv"
    write_csv(p, [])
    assert p.exists() and p.read_text(encoding="utf-8") == "", "❌1 空 rows 应产出空文件"
    write_csv(p, [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "a,b", f"❌2 表头应取首行字段名，实际 {lines[0]!r}"
    assert len(lines) == 3, f"❌3 应 1 表头 + 2 数据行，实际 {len(lines)}"


# ============================================================
# monitoring：告警评估（级别过滤 / 去抖 / 双通道 / 不抛保证）
# ============================================================
@pytest.fixture
def mon(monkeypatch, tmp_path):
    """隔离 monitoring 全局状态：清去抖表、关闭 webhook（P9：告警走 logger+webhook，无事件文件）。"""
    import llmsec.core.monitoring as monitoring

    monkeypatch.delenv("LLMSEC_ALERT_WEBHOOK", raising=False)
    monkeypatch.delenv("LLMSEC_ALERT_LEVEL", raising=False)
    monkeypatch.setattr(monitoring, "_dedup", {})
    monkeypatch.setattr(monitoring, "_executor", None)
    return monitoring


def test_emit_alert_logs_warning(mon, caplog):
    """emit_alert(force) → logger.warning 留痕（P9：事件文件通道已删，日志是主通道）。"""
    import logging

    ok = mon.emit_alert("error", "磁盘将满", "detail-1", {"task_id": "t-1"}, force=True)
    assert ok is True, "❌1 force 告警应实际发出"
    recs = [r for r in caplog.records if r.levelno >= logging.WARNING and "磁盘将满" in r.message]
    assert recs, "❌2 告警应写入日志"
    assert "t-1" in recs[0].message, "❌3 context 标识应进日志"
    assert len(mon._dedup) == 1, f"❌4 去抖表应恰好 1 条，实际 {len(mon._dedup)}"


def test_emit_alert_dedup_by_identity(mon):
    """同 title+同一对象在窗口内去抖；不同对象不去抖；force 跳过去抖。"""
    assert mon.emit_alert("warning", "T", context={"task_id": "a"}, force=True) is True, "❌1 首发"
    assert mon.emit_alert("warning", "T", context={"task_id": "a"}) is False, "❌2 同对象应去抖"
    assert mon.emit_alert("warning", "T", context={"task_id": "b"}, force=True) is True, "❌3 不同对象"
    assert mon.emit_alert("warning", "T", context={"task_id": "a"}, force=True) is True, "❌4 force 跳过"


def test_title_hash_uses_identity_fields_only(mon):
    """_title_hash：标识字段（task_id 等）进键，非标识字段不影响键。"""
    h1 = mon._title_hash("T", {"task_id": "a", "other": "x"})
    h2 = mon._title_hash("T", {"task_id": "b", "other": "x"})
    h3 = mon._title_hash("T", {"task_id": "a", "other": "z"})
    assert h1 != h2, "❌1 不同对象应不同键"
    assert h1 == h3, "❌2 非标识字段不应影响键"


def test_emit_alert_level_filter(mon, monkeypatch):
    """级别过滤：低于 LLMSEC_ALERT_LEVEL 的丢弃；非法级别按 warning 处理。"""
    monkeypatch.setenv("LLMSEC_ALERT_LEVEL", "error")
    assert mon.emit_alert("info", "i-1", force=True) is False, "❌1 info 低于 error 应丢"
    assert mon.emit_alert("warning", "w-1", force=True) is False, "❌2 warning 低于 error 应丢"
    assert mon.emit_alert("error", "e-1", force=True) is True, "❌3 error 应放行"
    monkeypatch.setenv("LLMSEC_ALERT_LEVEL", "warning")
    assert mon.emit_alert("bogus-level", "x-1", force=True) is True, "❌4 非法级别按 warning 放行"


def test_emit_alert_never_raises(mon, monkeypatch):
    """内部任何异常都被兜底：emit_alert 返回 False，绝不向调用方抛。"""
    def _boom(*a, **k):
        raise RuntimeError("模拟内部故障")

    monkeypatch.setattr(mon, "_build_payload", _boom)
    assert mon.emit_alert("error", "t", force=True) is False, "❌1 内部故障应返回 False 而非抛出"


def test_emit_alert_webhook_channel_mocked(mon, monkeypatch, capsys):
    """webhook 通道：配置了 URL 才提交；payload 含 source/level/title（_post_webhook 已 mock，离线）。"""
    posted: list[tuple[str, dict]] = []

    class _SyncExec:
        def submit(self, fn, *a):
            fn(*a)

            class _F:
                @staticmethod
                def result():
                    return None
            return _F()

    monkeypatch.setattr(mon, "_get_executor", lambda: _SyncExec())
    monkeypatch.setattr(mon, "_post_webhook",
                        lambda url, payload: posted.append((url, payload)))
    # 未配置 URL → 不提交
    assert mon.emit_alert("error", "no-url", force=True) is True
    assert posted == [], "❌1 未配置 webhook 不应提交"
    # 配置 URL → 提交且 payload 结构完整
    monkeypatch.setenv("LLMSEC_ALERT_WEBHOOK", "http://example.invalid/hook")
    assert mon.emit_alert("error", "with-url", force=True) is True
    assert len(posted) == 1, "❌2 应提交一次"
    url, payload = posted[0]
    assert url == "http://example.invalid/hook", "❌3 URL 应原样透传"
    assert payload["source"] == "llmsec" and payload["title"] == "with-url", "❌4 payload 字段"

    # 提交本身抛异常 → 只写 stderr，主流程（事件文件/返回值）不受影响
    class _BoomExec:
        def submit(self, fn, *a):
            raise RuntimeError("线程池已关闭")

    monkeypatch.setattr(mon, "_get_executor", lambda: _BoomExec())
    assert mon.emit_alert("error", "boom-url", force=True) is True, "❌5 提交失败不应影响返回值"
    assert "提交失败" in capsys.readouterr().err, "❌6 提交失败应留痕 stderr"


def test_post_webhook_non_2xx_only_stderr(mon, monkeypatch, capsys):
    """_post_webhook：非 2xx / URLError / 其它异常都只写 stderr，不抛（离线，urlopen 全 mock）。

    非 2xx 的真实形态是 urlopen 抛 HTTPError（URLError 子类）——urlopen 只在
    2xx（重定向已自动跟随）时返回响应对象，原"返回后查 status>=300"分支不可达已删。
    """
    def _raise_http(req, timeout=None):
        raise mon.urllib.error.HTTPError("http://x/hook", 500, "Server Error", None, None)

    monkeypatch.setattr(mon.urllib.request, "urlopen", _raise_http)
    mon._post_webhook("http://x/hook", {"a": 1})  # 5xx（HTTPError）→ stderr
    assert "请求失败" in capsys.readouterr().err, "❌1 非 2xx 应写 stderr"

    def _raise_url(req, timeout=None):
        raise mon.urllib.error.URLError("connection refused")

    monkeypatch.setattr(mon.urllib.request, "urlopen", _raise_url)
    mon._post_webhook("http://x/hook", {"a": 1})  # URLError → stderr
    assert "请求失败" in capsys.readouterr().err, "❌2 URLError 应写 stderr"

    def _raise(req, timeout=None):
        raise ValueError("坏 URL")

    monkeypatch.setattr(mon.urllib.request, "urlopen", _raise)
    mon._post_webhook("http://x/hook", {"a": 1})  # 非 URLError 异常 → stderr
    assert "webhook" in capsys.readouterr().err, "❌3 通用异常也应只写 stderr"


def test_should_emit_prunes_stale_entries(mon):
    """去抖表超 200 条时顺手清理过期条目（防内存无限增长）。"""
    import time

    mon._dedup.update({f"k{i}": time.time() - 3600 for i in range(300)})  # 全部已过期
    assert mon._should_emit("fresh-key") is True, "❌1 未发过的 key 应放行"
    assert "fresh-key" in mon._dedup, "❌2 放行后应记录时间戳"
    assert len(mon._dedup) == 1, f"❌3 过期条目应被清理，实际剩 {len(mon._dedup)}"


def test_get_executor_lazy_singleton(mon):
    """线程池惰性创建且进程内单例（重复调用返回同一实例）。"""
    ex1 = mon._get_executor()
    ex2 = mon._get_executor()
    assert ex1 is ex2, "❌1 应复用同一 ThreadPoolExecutor"
    ex1.shutdown(wait=False)


def test_alert_helpers_delegate_to_emit(mon, monkeypatch):
    """三个标准告警包装：透传正确的 level/title/context 给 emit_alert。"""
    calls: list[dict] = []

    def _fake_emit(level, title, detail="", context=None, *, force=False):
        calls.append({"level": level, "title": title, "detail": detail,
                      "context": dict(context or {})})
        return True

    monkeypatch.setattr(mon, "emit_alert", _fake_emit)
    mon.alert_task_failed("tk-1", "evaluate", "llmsec ...", "log.txt", 3)
    assert calls[0]["level"] == "error" and calls[0]["title"] == "任务失败: evaluate", "❌1 task_failed"
    assert calls[0]["context"]["returncode"] == 3 and calls[0]["context"]["task_id"] == "tk-1", "❌2 上下文"

    mon.alert_zombie_task("tk-2", "evaluate", "cmd", 90.0)
    assert calls[1]["title"] == "僵尸任务: evaluate", "❌3 zombie"
    assert calls[1]["context"]["running_minutes"] == 90.0, "❌4 运行时长"

    mon.alert_study_aborted("st-1", 5)
    assert calls[2]["title"] == "Study 熔断: st-1", "❌5 study"
    assert "连续 5" in calls[2]["detail"], f"❌6 默认 detail 应含连续失败数: {calls[2]['detail']}"
