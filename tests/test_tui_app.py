"""e2e smoke: llmsec.tui 无头冒烟（textual App.run_test）。

测试函数内 asyncio.run 驱动（项目无 pytest-asyncio，textual 的 run_test 本身无头），
按项目惯例标 e2e——默认套件不跑，手动 pytest -m e2e tests/test_tui_app.py 触发。
"""

from __future__ import annotations

import asyncio
import json

import pytest

# 可选依赖：textual 属 [tui] extra，未安装环境整体跳过（沿用 hdbscan 惯例）
pytest.importorskip("textual")

from textual.widgets import DataTable

from llmsec.tui.app import LlmsecTUI
from llmsec.tui.panels.hpo_panel import HpoPanel
from llmsec.tui.panels.runs_panel import RunsPanel
from llmsec.tui.panels.tasks_panel import LaunchScreen, TasksPanel
from llmsec.tui.task_store import TaskStore

pytestmark = pytest.mark.e2e


def _fake_tasks(tmp_path) -> None:
    """伪造一个外部 evaluate 任务 + 一个 hpo 任务的磁盘痕迹。"""
    ev = tmp_path / "evaluate-101010-ab12cd.progress.jsonl"
    ev.write_text(
        json.dumps({"ts": "2026-08-15T10:00:01", "phase": "attack", "target": "模型A",
                    "round": 2, "max_rounds": 5, "elo": 1520.0, "delta": 20.0,
                    "ci_half": 30.0, "progress_pct": 40, "converged": False})
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "evaluate-101010-ab12cd.log").write_text("fake log\n", encoding="utf-8")
    hpo = tmp_path / "hpo-111111-aabbcc.progress.jsonl"
    hpo.write_text(
        json.dumps({"ts": "2026-08-15T11:00:01", "phase": "hpo", "trial_done": 2,
                    "trial_total_est": 10, "configs_done": 1, "configs_total": 8,
                    "best_metric": 5.5, "metric_name": "conv_rounds", "direction": "minimize",
                    "last": {"target": "模型A", "seed": 0, "status": "success", "value": 5.5,
                             "params": {"K_FACTOR": 16}}})
        + "\n",
        encoding="utf-8",
    )


def test_tui_smoke(tmp_path):
    _fake_tasks(tmp_path)

    async def _run() -> None:
        app = LlmsecTUI(store=TaskStore(log_dir=tmp_path))
        async with app.run_test(size=(120, 40)) as pilot:
            # 轮询线程首轮 refresh → TasksUpdated → 任务表出现外部任务
            await asyncio.sleep(0.8)
            await pilot.pause()
            table = app.query_one("#task-table", DataTable)
            assert table.row_count == 2  # evaluate + hpo 各一行

            # 面板切换：显示/隐藏正确
            app.action_panel("hpo")
            await pilot.pause()
            assert not app.query_one(HpoPanel).has_class("hidden")
            assert app.query_one(TasksPanel).has_class("hidden")
            hpo_table = app.query_one("#hpo-table", DataTable)
            assert hpo_table.row_count == 1  # 只列 hpo 任务

            app.action_panel("runs")
            await pilot.pause()
            assert not app.query_one(RunsPanel).has_class("hidden")
            assert app.query_one(HpoPanel).has_class("hidden")

            app.action_panel("tasks")
            await pilot.pause()
            assert not app.query_one(TasksPanel).has_class("hidden")

            # 发起评估表单：可打开、escape 关闭
            app.query_one(TasksPanel).action_new_eval()
            await pilot.pause()
            assert isinstance(app.screen, LaunchScreen)
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, LaunchScreen)

    asyncio.run(_run())
