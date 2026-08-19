"""tests for llmsec.tui.render — 纯字符渲染层（从 web 端 run-control.js 移植）。

只测纯逻辑（盲文条/OLS 平滑/sparkline/行格式/状态回放），不依赖 textual。
"""

from __future__ import annotations

from llmsec.tui.render import (
    C_GOLD,
    EvalProgressState,
    TargetProgressTracker,
    braille_bar,
    fmt_num,
    hpo_lines,
    pad_cells,
    progress_lines,
    sparkline,
    target_row,
)


# ============================================================
# braille_bar
# ============================================================
def _bar_text(pct) -> str:
    return str(braille_bar(pct).plain)


class TestBrailleBar:
    def test_zero_and_none_all_empty(self):
        assert _bar_text(0) == "[" + "⣀" * 14 + "]"
        assert _bar_text(None) == "[" + "⣀" * 14 + "]"

    def test_full_100(self):
        assert _bar_text(100) == "[" + "⣿" * 14 + "]"

    def test_overflow_clamped(self):
        assert _bar_text(150) == "[" + "⣿" * 14 + "]"
        assert _bar_text(-5) == "[" + "⣀" * 14 + "]"

    def test_half_transition_char(self):
        # 14 格 50% = 7 格整 → 无过渡字符；50%+半格（约 53.6%）→ 第 8 格 ⣦
        assert "⣦" not in _bar_text(50)
        assert _bar_text(50) == "[" + "⣿" * 7 + "⣀" * 7 + "]"
        assert _bar_text(53.6) == "[" + "⣿" * 7 + "⣦" + "⣀" * 6 + "]"

    def test_custom_width(self):
        assert braille_bar(50, width=2).plain == "[⣿⣀]"

    def test_filled_gold_empty_dim(self):
        bar = braille_bar(50, width=2)
        segments = [(bar.plain[s.start:s.end], s.style or "") for s in bar.spans]
        styles = dict(segments)
        assert C_GOLD in styles["⣿"]  # 填充描金
        assert "dim" in styles["⣀"]     # 空槽压暗


# ============================================================
# sparkline
# ============================================================
class TestSparkline:
    def test_min_two_points(self):
        assert sparkline([]).plain == ""
        assert sparkline([1.0]).plain == ""

    def test_normalize_levels(self):
        # 8 个点从 0→7 级：▁▂▃▄▅▆▇█
        s = sparkline([float(i) for i in range(8)], "maximize")
        assert s.plain == "▁▂▃▄▅▆▇█"

    def test_best_char_gold_by_direction(self):
        up = sparkline([1.0, 5.0, 3.0], "maximize")
        assert C_GOLD in (up.spans[1].style or "")
        down = sparkline([1.0, 5.0, 3.0], "minimize")
        assert C_GOLD in (down.spans[0].style or "")

    def test_flat_series_middle_level(self):
        assert sparkline([2.0, 2.0, 2.0]).plain == "▄▄▄"


# ============================================================
# TargetProgressTracker（OLS 平滑，移植 _recomputeDisp）
# ============================================================
class TestTracker:
    def test_single_point(self):
        tr = TargetProgressTracker()
        assert tr.update({"round": 1, "progress_pct": 40}, 5) == 40

    def test_monotonic_high_water(self):
        tr = TargetProgressTracker()
        a = tr.update({"round": 1, "progress_pct": 60}, 5)
        b = tr.update({"round": 2, "progress_pct": 0}, 5)  # 噪声骤降不倒退
        assert b >= a

    def test_linear_floor_from_round(self):
        # progress_pct 全 0（早期常见）：地板 round/max 保证条上升
        tr = TargetProgressTracker()
        for r in (1, 2, 3):
            pct = tr.update({"round": r, "progress_pct": 0}, 6)
        assert pct >= 50  # 3/6 地板

    def test_terminal_caps_100(self):
        tr = TargetProgressTracker()
        tr.update({"round": 2, "progress_pct": 10}, 5)
        assert tr.update({"round": 2, "phase": "attack_done", "progress_pct": 10}, 5) == 100

    def test_clamped_0_100(self):
        tr = TargetProgressTracker()
        tr.update({"round": 5, "progress_pct": 500}, 5)
        assert tr.disp_pct == 100

    def test_round_upsert_dedup(self):
        tr = TargetProgressTracker()
        tr.update({"round": 1, "progress_pct": 10}, 5)
        tr.update({"round": 1, "progress_pct": 20}, 5)
        assert len(tr.hist) == 1
        assert tr.hist[0] == (1.0, 20.0)


# ============================================================
# EvalProgressState（回放 + done/active 判定）
# ============================================================
def _ev_rec(tg, rnd, ts="2026-08-15T10:00:00", **kw):
    return {"phase": "attack", "target": tg, "round": rnd, "max_rounds": 5,
            "elo": 1500 + rnd, "delta": 12.0, "ci_half": 40.0, "progress_pct": 20 * rnd,
            "converged": False, "ts": ts, **kw}


class TestEvalProgressState:
    def test_replay_evaluate(self):
        st = EvalProgressState("t-1")
        st.apply_record(_ev_rec("模型A", 1))
        st.apply_record(_ev_rec("模型B", 1, ts="2026-08-15T10:00:01"))
        st.set_running(True)
        assert st.order == ["模型A", "模型B"]
        assert st.active_target == "模型B"  # ts 最新且未完成
        assert st.max_rounds == 5

    def test_done_detection(self):
        st = EvalProgressState("t-1")
        st.apply_record(_ev_rec("A", 5))  # round >= max_rounds → done
        st.set_running(True)
        assert "A" in st.done
        assert st.active_target is None

    def test_attack_done_marker(self):
        st = EvalProgressState("t-1")
        st.apply_record(_ev_rec("A", 2))
        st.apply_record(_ev_rec("A", 2, phase="attack_done", converged=True))
        assert "A" in st.done
        assert st.disp_pct("A") == 100

    def test_stopped_task_no_active(self):
        st = EvalProgressState("t-1")
        st.apply_record(_ev_rec("A", 1))
        st.set_running(False)  # 任务已结束 → 全灰无 active
        assert st.active_target is None

    def test_declare_targets_placeholder(self):
        st = EvalProgressState("t-1")
        st.declare_targets(["A", "B"], max_rounds=5)
        st.set_running(True)
        assert st.order == ["A", "B"]
        assert "等待中" in target_row("A", st.targets["A"], st).plain
        assert st.overall_pct() is None  # 占位目标无进度历史，不计入汇总
        # 记录到达后不重复声明
        st.apply_record(_ev_rec("A", 1))
        assert st.order == ["A", "B"]

    def test_hpo_replay_and_dedup(self):
        st = EvalProgressState("t-1")
        last = {"target": "A", "seed": 1, "status": "success", "value": 3.5, "params": {"K_FACTOR": 16}}
        st.apply_record({"phase": "hpo", "trial_done": 1, "trial_total_est": 10,
                         "configs_done": 1, "configs_total": 20, "best_metric": 3.5,
                         "metric_name": "conv_rounds", "direction": "minimize", "last": last})
        st.apply_record({"phase": "hpo", "trial_done": 1, "configs_done": 1, "configs_total": 20,
                         "last": last})  # 轮询重放同一条 → 去重
        assert st.kind == "hpo"
        assert len(st.hpo_trials) == 1
        assert st.overall_pct() == 5  # 1/20

    def test_hpo_lines_structure(self):
        st = EvalProgressState("t-1")
        st.apply_record({"phase": "hpo", "trial_done": 2, "configs_done": 2, "configs_total": 10,
                         "best_metric": 4.0, "metric_name": "conv_rounds", "direction": "minimize",
                         "last": {"target": "A", "seed": 0, "status": "success", "value": 4.0, "params": {}}})
        st.apply_record({"phase": "hpo", "trial_done": 3, "configs_done": 3, "configs_total": 10,
                         "best_metric": 3.2, "metric_name": "conv_rounds", "direction": "minimize",
                         "last": {"target": "A", "seed": 1, "status": "failed", "params": {}}})
        lines = hpo_lines(st)
        joined = "\n".join(x.plain for x in lines)
        assert "config 3/10" in joined
        assert "trial 3/" in joined
        assert "✓" in joined and "✗" in joined

    def test_overall_pct_evaluate(self):
        st = EvalProgressState("t-1")
        st.apply_record(_ev_rec("A", 1, progress_pct=20))
        st.apply_record(_ev_rec("B", 1, progress_pct=60))
        # A 地板 1/5=20、B 60 → 均值 40
        assert st.overall_pct() == 40


# ============================================================
# 行渲染
# ============================================================
class TestTargetRow:
    def test_active_row_format(self):
        st = EvalProgressState("t-1")
        st.apply_record(_ev_rec("模型A", 3, elo=1600, delta=45.0, ci_half=45.0, progress_pct=60))
        st.set_running(True)
        plain = target_row("模型A", st.targets["模型A"], st).plain
        # ❯ 名字 R3/5  ELO 1600 ↑45  CI±45  [bar] ..%
        assert plain.startswith("❯ 模型A")
        assert "R3/5" in plain
        assert "ELO 1600" in plain
        assert "↑45" in plain
        assert "CI±45" in plain
        assert "运行中" in plain
        assert "[" in plain and "]" in plain

    def test_done_row_no_bar(self):
        st = EvalProgressState("t-1")
        st.apply_record(_ev_rec("A", 5, converged=True))
        st.set_running(True)
        plain = target_row("A", st.targets["A"], st).plain
        assert plain.startswith("✓")
        assert "已收敛" in plain
        assert "⣿" not in plain

    def test_waiting_row(self):
        st = EvalProgressState("t-1")
        st.apply_record(_ev_rec("A", 1))
        st.set_running(True)
        # 未收到记录的目标（手动构造空 rec）→ 等待中
        plain = target_row("B", {}, st).plain
        assert "等待中" in plain

    def test_cjk_padding_display_width(self):
        assert pad_cells("模型A", 6) == "模型A  "  # 显示宽 5 < 6 → 补 1 格 + 尾空格
        assert pad_cells("abc", 3) == "abc "
        assert pad_cells("abcdefghijklmno", 14) == "abcdefghijklmno "  # 超宽不截断只留尾空格

    def test_progress_lines_cursor_on_active(self):
        st = EvalProgressState("t-1")
        st.apply_record(_ev_rec("A", 1))
        st.set_running(True)
        on = progress_lines(st, cursor_on=True)
        off = progress_lines(st, cursor_on=False)
        assert on[0].plain.endswith("▋")
        assert not off[0].plain.endswith("▋")

    def test_delta_down_colored(self):
        st = EvalProgressState("t-1")
        st.apply_record(_ev_rec("A", 2, delta=-30.0))
        st.set_running(True)
        plain = target_row("A", st.targets["A"], st).plain
        assert "↓30" in plain


# ============================================================
# fmt_num
# ============================================================
def test_fmt_num():
    assert fmt_num(None) is None
    assert fmt_num(1599.6) == "1600"
    assert fmt_num(3.14159, 3) == "3.142"
    assert fmt_num(-0.0, 0) == "0"
