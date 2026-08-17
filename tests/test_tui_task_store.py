"""tests for llmsec.tui.task_store — 任务状态层（磁盘扫描 + 增量回放）。

用临时目录伪造 .log/.progress.jsonl；本进程 TASKS 为空（测试进程未启动过任务），
全部快照走 detached 路径。start_hpo 用 monkeypatch 拦截 start_task 不真正起子进程。
"""

from __future__ import annotations

import json

import llmsec.core.config as cfg  # P9: TASK_LOG_DIR 动态读后统一 patch cfg
from llmsec.tui.task_store import EXTERNAL, TaskStore, attack_files, study_yamls

# TASKS 隔离由 conftest 的 autouse _hermetic_tasks 统一提供。


def _write(path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _ev_line(tg, rnd, **kw):
    return json.dumps(
        {
            "ts": f"2026-08-15T10:00:{rnd:02d}",
            "phase": "attack",
            "target": tg,
            "round": rnd,
            "max_rounds": 5,
            "elo": 1500 + rnd,
            "delta": 10.0,
            "ci_half": 40.0,
            "progress_pct": 20 * rnd,
            "converged": False,
            **kw,
        },
        ensure_ascii=False,
    )


class TestRefreshDetached:
    def test_scan_evaluate_task(self, tmp_path):
        _write(
            tmp_path / "evaluate-101010-ab12cd.progress.jsonl",
            _ev_line("模型A", 1) + "\n" + _ev_line("模型A", 2) + "\n",
        )
        _write(tmp_path / "evaluate-101010-ab12cd.log", "line1\nline2\n")
        store = TaskStore(log_dir=tmp_path)
        snaps, dirty = store.refresh()
        assert not dirty
        assert len(snaps) == 1
        s = snaps[0]
        assert s.id == "evaluate-101010-ab12cd"
        assert s.kind == "evaluate"
        assert s.status == EXTERNAL
        assert not s.owned
        assert "line2" in s.log_tail
        # 回放：两轮记录都已应用
        assert s.state.targets["模型A"]["round"] == 2
        assert s.state.max_rounds == 5

    def test_incremental_replay(self, tmp_path):
        pf = tmp_path / "evaluate-101010-ab12cd.progress.jsonl"
        _write(pf, _ev_line("A", 1) + "\n")
        store = TaskStore(log_dir=tmp_path)
        snaps, _ = store.refresh()
        assert snaps[0].state.targets["A"]["round"] == 1
        # 追加一轮 → 只解析新行，状态增量更新
        with open(pf, "a", encoding="utf-8") as f:
            f.write(_ev_line("A", 2) + "\n")
        snaps, _ = store.refresh()
        assert snaps[0].state.targets["A"]["round"] == 2

    def test_partial_line_held_until_complete(self, tmp_path):
        pf = tmp_path / "evaluate-101010-ab12cd.progress.jsonl"
        _write(pf, _ev_line("A", 1) + "\n")
        with open(pf, "a", encoding="utf-8") as f:
            f.write(_ev_line("A", 2)[:20])  # 半行（写入中）
        store = TaskStore(log_dir=tmp_path)
        snaps, _ = store.refresh()
        assert snaps[0].state.targets["A"]["round"] == 1  # 半行未应用
        with open(pf, "a", encoding="utf-8") as f:
            f.write(_ev_line("A", 2)[20:] + "\n")  # 补全
        snaps, _ = store.refresh()
        assert snaps[0].state.targets["A"]["round"] == 2

    def test_corrupt_line_skipped(self, tmp_path):
        _write(tmp_path / "evaluate-101010-ab12cd.progress.jsonl", "not json\n" + _ev_line("A", 1) + "\n")
        store = TaskStore(log_dir=tmp_path)
        snaps, _ = store.refresh()
        assert "A" in snaps[0].state.targets

    def test_hpo_task_replay(self, tmp_path):
        rec = {
            "phase": "hpo",
            "trial_done": 1,
            "trial_total_est": 10,
            "configs_done": 1,
            "configs_total": 4,
            "best_metric": 5.0,
            "metric_name": "conv_rounds",
            "direction": "minimize",
            "last": {"target": "A", "seed": 0, "status": "success", "value": 5.0, "params": {}},
        }
        _write(tmp_path / "hpo-111111-aabbcc.progress.jsonl", json.dumps(rec) + "\n")
        store = TaskStore(log_dir=tmp_path)
        snaps, _ = store.refresh()
        s = snaps[0]
        assert s.kind == "hpo"
        assert s.state.kind == "hpo"
        assert len(s.state.hpo_trials) == 1

    def test_max_20_external_tasks(self, tmp_path):
        for i in range(25):
            _write(tmp_path / f"evaluate-1000{i:02d}-aaaaaa.progress.jsonl", _ev_line("A", 1) + "\n")
        store = TaskStore(log_dir=tmp_path)
        snaps, _ = store.refresh()
        assert len(snaps) == 20

    def test_stray_files_ignored(self, tmp_path):
        _write(tmp_path / "readme.txt", "x")
        _write(tmp_path / "noDash.progress.jsonl", "{}\n")
        store = TaskStore(log_dir=tmp_path)
        assert store.refresh()[0] == []

    def test_empty_dir(self, tmp_path):
        store = TaskStore(log_dir=tmp_path)
        assert store.refresh()[0] == []

    def test_missing_dir(self, tmp_path):
        store = TaskStore(log_dir=tmp_path / "nonexistent")
        assert store.refresh()[0] == []


class TestOwnedMeta:
    def test_meta_declares_placeholder_targets(self, tmp_path):
        """本进程任务携带 launch 层 meta：progress 记录到达前先渲染「等待中」占位行。"""
        import llmsec.server.task_manager as tm

        tid = "evaluate-101010-meta01"
        tm.TASKS[tid] = {
            "kind": "evaluate",
            "cmd": "",
            "argv": [],
            "env_override": None,
            "meta": {"targets": ["模型A", "模型B"], "max_rounds": 5},
            "proc": None,
            "log_path": tmp_path / f"{tid}.log",
            "log_file": None,
            "status": "queued",
            "started_at": "2026-08-15T10:00:00",
            "_task_id": tid,
        }
        store = TaskStore(log_dir=tmp_path)
        snaps, _ = store.refresh()
        assert len(snaps) == 1
        s = snaps[0]
        assert s.owned and s.meta["targets"] == ["模型A", "模型B"]
        assert s.state is not None and s.state.order == ["模型A", "模型B"]
        assert s.state.max_rounds == 5


class TestStartHpo:
    def _patch_start(self, monkeypatch):
        import llmsec.server.task_manager as tm

        calls = []
        monkeypatch.setattr(
            tm, "start_task", lambda kind, argv, **kw: calls.append((kind, argv)) or {"id": "hpo-x", "status": "queued"}
        )
        return calls

    def test_nonexistent_file(self, tmp_path, monkeypatch):
        calls = self._patch_start(monkeypatch)
        store = TaskStore(log_dir=tmp_path)
        out = store.start_hpo(str(tmp_path / "nope.yaml"))
        assert "error" in out
        assert calls == []

    def test_outside_repo_rejected(self, tmp_path, monkeypatch):
        import llmsec.core.config as config

        calls = self._patch_start(monkeypatch)
        f = tmp_path / "study.yaml"
        _write(f, "name: x\n")
        # PROJECT_ROOT 指向别处 → 文件在“仓库外”
        monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path / "elsewhere")
        store = TaskStore(log_dir=tmp_path)
        out = store.start_hpo(str(f))
        assert "error" in out
        assert calls == []

    def test_valid_starts_task(self, tmp_path, monkeypatch):
        import llmsec.core.config as config

        calls = self._patch_start(monkeypatch)
        monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
        f = tmp_path / "experiments" / "study.yaml"
        f.parent.mkdir(exist_ok=True)
        _write(f, "name: x\n")
        store = TaskStore(log_dir=tmp_path)
        out = store.start_hpo(str(f))
        assert out == {"id": "hpo-x", "status": "queued"}
        assert calls == [("hpo", ["-m", "llmsec.experiments", "run", str(f)])]

    def test_relative_path_resolved(self, tmp_path, monkeypatch):
        import llmsec.core.config as config

        calls = self._patch_start(monkeypatch)
        monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
        _write(tmp_path / "s.yaml", "name: x\n")
        store = TaskStore(log_dir=tmp_path)
        store.start_hpo("s.yaml")
        assert calls and calls[0][1][-1].endswith("s.yaml")


class TestCancelAndLog:
    def test_cancel_unknown(self, tmp_path):
        store = TaskStore(log_dir=tmp_path)
        out = store.cancel("evaluate-000000-zzzzzz")
        assert "error" in out

    def test_full_log_falls_back_to_file(self, tmp_path):
        _write(tmp_path / "evaluate-101010-ab12cd.log", "hello log\n")
        store = TaskStore(log_dir=tmp_path)
        assert "hello log" in store.full_log("evaluate-101010-ab12cd")


class TestFormSources:
    def test_attack_files_missing_dir(self, tmp_path, monkeypatch):
        import llmsec.core.config as config

        monkeypatch.setattr(config, "ATTACKS_DIR", tmp_path / "none")
        assert attack_files() == []

    def test_study_yamls(self, tmp_path, monkeypatch):
        import llmsec.core.config as config

        root = tmp_path
        (root / "experiments").mkdir()
        (root / "output" / "experiments").mkdir(parents=True)
        _write(root / "experiments" / "a.yaml", "x: 1\n")
        _write(root / "output" / "experiments" / "_dashboard_b.yaml", "x: 1\n")
        _write(root / "experiments" / "ignore.txt", "x")
        monkeypatch.setattr(config, "PROJECT_ROOT", root)
        monkeypatch.setattr(config, "OUTPUT_DIR", root / "output")
        out = study_yamls()
        assert sorted(out) == ["experiments/a.yaml", "output/experiments/_dashboard_b.yaml"]


class TestExternalMeta:
    """外部任务可见性：库行（P4 真相）+ legacy meta.json 经对账吸收 / PID 探活 / 跨进程取消。"""

    def _write_meta(self, tmp_path, tid, **over):
        import json as _json

        meta = {
            "id": tid,
            "kind": "evaluate",
            "cmd": "-m x",
            "argv": ["-m", "x"],
            "meta": {"targets": ["模型A"], "max_rounds": 5},
            "started_at": "2026-08-15T10:00:00",
            "pid": None,
            "status": "running",
        }
        meta.update(over)
        (tmp_path / f"{tid}.meta.json").write_text(_json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        return meta

    def test_meta_running_alive_pid(self, tmp_path):
        import os

        self._write_meta(tmp_path, "evaluate-101010-aaa", pid=os.getpid())  # 本测试进程，必活
        (tmp_path / "evaluate-101010-aaa.log").write_text("x", encoding="utf-8")
        snaps, _ = TaskStore(log_dir=tmp_path).refresh()
        s = snaps[0]
        assert not s.owned and s.status == "running" and s.pid == os.getpid()
        assert s.kind == "evaluate" and s.meta == {"targets": ["模型A"], "max_rounds": 5}

    def test_meta_running_dead_pid_becomes_ended(self, tmp_path):
        self._write_meta(tmp_path, "evaluate-101010-bbb", pid=999999999)  # 不存在的 PID
        snaps, _ = TaskStore(log_dir=tmp_path).refresh()
        assert snaps[0].status == "ended", "持有进程消失且无终态回写 → 已结束"

    def test_meta_terminal_status_passed_through(self, tmp_path):
        self._write_meta(tmp_path, "hpo-111111-ccc", kind="hpo", pid=None, status="success")
        snaps, _ = TaskStore(log_dir=tmp_path).refresh()
        assert snaps[0].status == "success" and snaps[0].kind == "hpo"

    def test_no_meta_falls_back_external(self, tmp_path):
        (tmp_path / "evaluate-101010-ddd.log").write_text("x", encoding="utf-8")
        snaps, _ = TaskStore(log_dir=tmp_path).refresh()
        assert snaps[0].status == EXTERNAL and snaps[0].pid is None

    def test_cancel_external_alive_pid(self, tmp_path, monkeypatch):
        import os

        import llmsec.tui.task_store as ts

        tid = "evaluate-101010-eee"
        self._write_meta(tmp_path, tid, pid=os.getpid())
        killed = []
        monkeypatch.setattr(ts, "_kill_pid", lambda pid: killed.append(pid) or True)
        out = TaskStore(log_dir=tmp_path).cancel(tid)
        assert out.get("status") == "cancelled" and out.get("killed_pid") == os.getpid()
        assert killed == [os.getpid()]
        # 取消状态已回写目录库行（P1：回写从 meta.json 改库行；P4 起 meta.json 退役）
        from llmsec.storage import contract as _storage

        row = _storage.get_task(tid, tasks_dir=tmp_path)
        assert row is not None and row.status == "cancelled"

    def test_cancel_external_dead_pid_errors(self, tmp_path):
        self._write_meta(tmp_path, "evaluate-101010-fff", pid=999999999)
        out = TaskStore(log_dir=tmp_path).cancel("evaluate-101010-fff")
        assert "error" in out

    def test_task_manager_persists_task_lifecycle(self, tmp_path, monkeypatch):
        """task_manager 落库（P4：库行即真相）：running（带 pid）→ 终态回写。"""
        import time as _time

        import llmsec.server.task_manager as tm
        from llmsec.storage import contract as _storage

        monkeypatch.setattr(cfg, "TASK_LOG_DIR", tmp_path)
        # 子进程须活过 start_task 返回（首次含 ORM import/建引擎固定开销）
        view = tm.start_task("metauto", ["-c", "import time; time.sleep(0.5)"], meta={"targets": ["A"]})
        tid = view["id"]
        row = _storage.get_task(tid, db_path=cfg.CATALOG_DB)  # 显式 db_path：跳过对账
        assert row.status == "running" and isinstance(row.pid, int)
        assert row.meta == {"targets": ["A"]} and row.kind == "metauto"
        deadline = _time.time() + 15
        while _time.time() < deadline:
            tm.list_tasks()  # 驱动 _refresh_task_status（生产中由 TUI 2s 轮询驱动）
            row = _storage.get_task(tid, db_path=cfg.CATALOG_DB)
            if row.status == "success":
                break
            _time.sleep(0.1)
        assert row.status == "success", f"终态未回写: {row.status}"
