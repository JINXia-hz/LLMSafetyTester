"""代码审查第 2 轮修复的回归测试：control/ 逻辑隐患。

覆盖：
  M1  make_plan_from_llm 校验重复步骤 id / 悬空依赖（此前静默吞掉）
  M2  fallback JSON 直调的 call_tool 执行异常不再上抛（API 不再 500）
  M3  session 滑窗按完整对话单元裁剪（tool_calls/tool 配对不被拆散）
  M4  收尾总结的合成指令不再写入 session 历史
  M5  LLM 回路中断后悬空 tool_call_id 被补齐（session 不报废）
  M6  store.load 对损坏索引自愈隔离
  M7  block._TICKETS 持锁（并发迭代不崩）
  M8  门下省阈值缓存 TTL 过期重取（此前永不过期）
  M9  invoker manage 调用带超时（挂起子进程不再卡死 Plan）
  M10 eval 默认 work-dir 名唯一 + 相对路径判定用 is_relative_to
  M11 llm.status_code=None 不再 TypeError 进错误重试路径
  M12 env merge_to_global 备份名唯一 + 全程持锁
  M13 fork 快照导出异常时的错误带上下文
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest


# ============================================================
# M1：Plan 拟案校验
# ============================================================
def test_r2_plan_duplicate_step_id_rejected():
    from control.agent.shangshu.plan import make_plan_from_llm
    with pytest.raises(ValueError, match="重复"):
        make_plan_from_llm("意图", [
            {"id": "s1", "capability": "list_runs"},
            {"id": "s1", "capability": "list_workspaces"},
        ])


def test_r2_plan_dangling_dep_rejected():
    from control.agent.shangshu.plan import make_plan_from_llm
    with pytest.raises(ValueError, match="不存在"):
        make_plan_from_llm("意图", [
            {"id": "s1", "capability": "list_runs"},
            {"id": "s2", "capability": "compare_runs", "depends_on": ["sX"]},
        ])


def test_r2_plan_valid_passes():
    from control.agent.shangshu.plan import make_plan_from_llm
    p = make_plan_from_llm("意图", [
        {"id": "s1", "capability": "list_runs"},
        {"id": "s2", "capability": "compare_runs", "depends_on": ["s1"]},
    ])
    assert [s.id for s in p.steps] == ["s1", "s2"]


# ============================================================
# M2：fallback JSON 直调异常
# ============================================================
def test_r2_fallback_json_tool_exception_not_raised(monkeypatch):
    from control.agent.zhongshu import fallback

    def _boom(tool, args):
        raise RuntimeError("fork 重名")

    monkeypatch.setattr(fallback, "call_tool", _boom)
    out = fallback.chat_one('{"tool": "fork_workspace", "args": {"name": "x"}}')
    assert out.startswith("❌") and "RuntimeError" in out, \
        "M2: call_tool 执行异常应转为错误文案（上抛会让 API 500）"


# ============================================================
# M3/M5：session 配对不变量
# ============================================================
def _assert_pairing(messages):
    pending: set[str] = set()
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            assert not pending, f"嵌套 tool_calls: {pending}"
            pending = {tc["id"] for tc in m["tool_calls"]}
        elif m["role"] == "tool":
            assert m.get("tool_call_id") in pending, f"孤儿 tool 应答: {m}"
            pending.discard(m["tool_call_id"])
        elif m["role"] in ("user", "assistant"):
            assert not pending, f"assistant(tool_calls) 后未跟 tool 应答: {pending}"
    assert not pending, f"末尾悬空 tool_calls: {pending}"


def test_r2_session_trim_keeps_tool_pairing():
    from control.agent.zhongshu import session as sess

    sid = f"r2-trim-{time.time_ns()}"
    _, msgs = sess.get_or_create(sid)
    # 15 轮完整对话单元（60 条）超过 _HISTORY_MAX=40，滑窗反复触发
    for i in range(15):
        sess.append(sid, "user", f"u{i}")
        sess.append_message(sid, {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": f"c{i}", "type": "function",
                            "function": {"name": "t", "arguments": "{}"}}]})
        sess.append_message(sid, {"role": "tool", "tool_call_id": f"c{i}", "content": "r"})
        sess.append(sid, "assistant", f"a{i}")
    assert len(msgs) <= sess._HISTORY_MAX
    _assert_pairing(msgs)


def test_r2_patch_dangling_tool_calls():
    from control.agent.zhongshu import dialogue
    from control.agent.zhongshu import session as sess

    sid = f"r2-dangling-{time.time_ns()}"
    _, msgs = sess.get_or_create(sid)
    sess.append(sid, "user", "q")
    sess.append_message(sid, {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "c1", "type": "function",
                        "function": {"name": "t", "arguments": "{}"}},
                       {"id": "c2", "type": "function",
                        "function": {"name": "t", "arguments": "{}"}}]})
    sess.append_message(sid, {"role": "tool", "tool_call_id": "c1", "content": "r1"})

    dialogue._patch_dangling_tool_calls(sid, msgs)
    assert msgs[-1]["role"] == "tool" and msgs[-1]["tool_call_id"] == "c2", \
        "M5: 悬空的 c2 应被补一条中断应答"
    _assert_pairing(msgs)


# ============================================================
# M4：合成总结指令不落 session 历史
# ============================================================
class _Fn:
    def __init__(self, name):
        self.name = name
        self.arguments = "{}"


class _TC:
    def __init__(self, i, name):
        self.id = f"tc{i}"
        self.function = _Fn(name)


class _Msg:
    def __init__(self, content, tcs=None):
        self.content = content
        self.tool_calls = tcs


class _Resp:
    def __init__(self, msg):
        self.choices = [type("Ch", (), {"message": msg})()]


def test_r2_final_summary_prompt_not_persisted(monkeypatch):
    from control.agent.zhongshu import dialogue
    from control.agent.zhongshu import session as sess

    monkeypatch.setattr(dialogue, "is_llm_configured", lambda: True)
    calls = {"n": 0}

    def _fake_chat(messages, *, tools=None, **kw):
        calls["n"] += 1
        if tools is not None:
            return _Resp(_Msg(None, [_TC(calls["n"], "search_history")]))
        return _Resp(_Msg("总结回复"))

    monkeypatch.setattr(dialogue, "chat_with_tools", _fake_chat)
    monkeypatch.setattr(dialogue, "_do_search_history", lambda args: "ok")

    sid = f"r2-final-{time.time_ns()}"
    result = dialogue.handle_message("测试指令", session_id=sid)
    assert result["reply"] == "总结回复"

    _, msgs = sess.get_or_create(sid)
    contents = [m.get("content") for m in msgs]
    assert "请基于已有信息总结回复。" not in contents, \
        "M4: 收尾总结的合成指令不得写入 session 历史（永久污染后续上下文）"
    assert "总结回复" in contents


# （M6 store.load 损坏自愈：AtomicIndexStore 已随 P5 库化退役——
#  索引健壮性由 SQLite 事务承接，本测试移除）


# ============================================================
# ============================================================
# M7：block._TICKETS 并发（r7：list_pending_blocks 无生产调用方已删，
#     本测试随之移除——封驳令的并发安全由 issue/approve/clear 的锁覆盖）
# ============================================================


# （M8 阈值缓存 TTL：CLI+TTLCache+fallback 机器已随 P5 直读化删除）



# ============================================================
# M9：invoker manage 调用超时
# ============================================================
def test_r2_invoker_manage_calls_have_timeout(monkeypatch):
    from control.core import invoker

    seen: list[tuple[str, float | None]] = []

    def _fake_run(argv, *, timeout=None, **kw):
        seen.append((" ".join(argv[3:5]), timeout))
        return invoker.InvokeResult(argv=argv, returncode=0, json={})

    monkeypatch.setattr(invoker, "_run", _fake_run)
    invoker.list_runs()
    invoker.list_runs()
    invoker.delete_runs(["x"])
    timeouts = dict(seen)
    assert timeouts.get("runs list") == 120, "M9: list_runs 须带 120s 超时"
    assert timeouts.get("runs list") == 120, "M9: subprocess 调用须带超时"
    assert timeouts.get("runs delete") == 600, "M9: delete_runs 须带 600s 超时"


# ============================================================
# M10：eval 默认目录唯一 + 相对路径
# ============================================================
def test_r2_eval_default_dir_unique_and_relative(monkeypatch, tmp_path):
    import control.config as cfg
    from control.agent.shangshu import capabilities as caps
    from control.core import invoker as inv

    monkeypatch.setattr(cfg, "OUTPUT_DIR", tmp_path)
    captured: list[Path] = []

    def _fake_run_runner(work_dir, **kw):
        captured.append(Path(work_dir))
        return inv.InvokeResult(argv=[], returncode=0)

    monkeypatch.setattr(inv, "run_runner", _fake_run_runner)

    r1 = caps._h_run_evaluation({})
    caps._h_run_evaluation({})
    assert captured and captured[0] != captured[1], "M10: 同秒并发 eval 不得共用默认 work_dir"
    assert r1["work_dir"].startswith("eval_runs/") and "\\" not in r1["work_dir"], \
        "M10: OUTPUT_DIR 内的 work_dir 应输出为正斜杠相对路径"


# ============================================================
# M11：status_code=None
# ============================================================
def test_r2_llm_status_code_none_no_typeerror(monkeypatch):
    from control.agent import llm as llm_mod

    class _ErrWithNoneStatus(Exception):
        status_code = None

    class _FakeCompletions:
        def create(self, **kw):
            raise _ErrWithNoneStatus("boom")

    class _FakeChat:
        def __init__(self):
            self.completions = _FakeCompletions()

    class _FakeClient:
        def __init__(self):
            self.chat = _FakeChat()

    monkeypatch.setattr(llm_mod, "_client", _FakeClient())
    monkeypatch.setattr(llm_mod, "get_model", lambda: "m")
    monkeypatch.setattr("time.sleep", lambda s: None)

    with pytest.raises(_ErrWithNoneStatus):
        llm_mod.chat_with_tools([{"role": "user", "content": "x"}], max_retries=1)
    # 若 400<=None 抛 TypeError，上面捕到的会是 TypeError 而非原异常 → 用例失败


# ============================================================
# M12：env merge 备份唯一
# ============================================================
def test_r2_env_backup_names_unique(monkeypatch, tmp_path):
    from control.core import env_snapshot as es

    case = tmp_path / "case"
    case.mkdir()
    ge = case / ".env"
    ge.write_text("A=1\n", encoding="utf-8")

    monkeypatch.setattr(es, "_GLOBAL_ENV", ge)
    monkeypatch.setattr(es, "ENV_SNAPSHOTS_DIR", tmp_path / "snaps")
    monkeypatch.setattr(es, "load_env_dict", lambda name: {"K": "V"})
    monkeypatch.setattr(es, "_read_global_env", lambda: {"A": "1"})

    es.merge_to_global("s")
    es.merge_to_global("s")

    baks = list(case.glob(".env.bak.*"))
    assert len(baks) == 2, "M12: 两次 merge 各留一份备份"
    assert len({b.name for b in baks}) == 2, "M12: 同秒备份不得互相覆盖"
    assert "K=V" in ge.read_text(encoding="utf-8")


# ============================================================
# M13：fork 快照缺失的错误上下文
# ============================================================
def test_r2_fork_missing_source_raises(monkeypatch, tmp_path):
    """fork 源异常（如全局 R 库缺失）向上传播——库级 clone 无快照握手可丢上下文。"""
    from control.core import storage as cstorage
    from control.core import workspace as ws

    monkeypatch.setattr(ws, "WORKSPACES_DIR", tmp_path / "ws")
    monkeypatch.setattr(ws, "ensure_workspaces_dir", lambda: None)

    def boom(dest):
        raise RuntimeError("R 库不存在")

    monkeypatch.setattr(cstorage, "backup", boom)
    with pytest.raises(RuntimeError, match="R 库不存在"):
        ws.fork("w", source="global")
