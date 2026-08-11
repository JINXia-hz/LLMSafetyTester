"""core.progress 并发/容错 + attack_phase 每轮进度落盘测试。"""

import json
import threading


def test_emit_progress_concurrent(tmp_path, monkeypatch):
    """多线程并发 emit_progress：Lock 保证 20 行完整、不丢不交错。"""
    from llmsec.core import progress as P

    monkeypatch.setattr(P, "TASK_LOG_DIR", tmp_path)
    monkeypatch.setenv("LLMSEC_TASK_ID", "conc-ut")

    def worker(i):
        P.emit_progress({"phase": "attack", "target": "t", "round": i})

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    p = tmp_path / "conc-ut.progress.jsonl"
    assert p.exists()
    lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 20, "20 线程应写 20 行"
    rounds = sorted(json.loads(l)["round"] for l in lines)
    assert rounds == list(range(20)), "每条记录完整可 parse"
    print("✅ emit_progress 并发写通过")


def test_emit_progress_swallows_oserror(tmp_path, monkeypatch):
    """IO 失败时 emit_progress 必须静默（进度是辅助可见性，不拖垮主流程）。"""
    from llmsec.core import progress as P

    # TASK_LOG_DIR 指向"文件之下"的路径 → mkdir 抛 NotADirectoryError(OSError 子类)
    (tmp_path / "blocker").write_text("x")
    monkeypatch.setattr(P, "TASK_LOG_DIR", tmp_path / "blocker" / "sub")
    monkeypatch.setenv("LLMSEC_TASK_ID", "err-ut")
    P.emit_progress({"phase": "attack"})  # 不应抛
    print("✅ emit_progress OSError 静默通过")


def test_emit_round_progress_writes_file(tmp_path, monkeypatch):
    """_emit_round_progress 落盘字段正确：delta=本轮-prev，progress_pct 与 _convergence_score 同口径。"""
    from llmsec.core import progress as P
    from llmsec.params import CONV_CI_TARGET
    from llmsec.pipeline.attack_phase import _emit_round_progress

    monkeypatch.setattr(P, "TASK_LOG_DIR", tmp_path)
    monkeypatch.setenv("LLMSEC_TASK_ID", "round-ut")

    conv = {"current_elo": 1500.0, "ci_half": 12.0, "coverage": 0.1, "converged": False}
    new_prev = _emit_round_progress("deepseek", 2, 10, conv, 1490.0, 5, 50)

    p = tmp_path / "round-ut.progress.jsonl"
    rec = json.loads(p.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert rec["target"] == "deepseek" and rec["round"] == 2 and rec["max_rounds"] == 10
    assert rec["elo"] == 1500.0 and rec["delta"] == 10.0, "delta = 本轮 elo − prev_elo"
    exp_pct = round(max(0.0, min(0.99, 1 - 12.0 / CONV_CI_TARGET)) * 100)
    assert rec["progress_pct"] == exp_pct, "progress_pct 口径须与 dashboard _convergence_score 一致"
    assert new_prev == 1500.0, "返回新 prev_elo 供下轮 delta"
    print("✅ _emit_round_progress 落盘 + delta/pct 通过")
