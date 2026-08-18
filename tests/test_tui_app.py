"""e2e smoke: llmsec.tui 无头冒烟（textual App.run_test）——控制台范式。

测试函数内 asyncio.run 驱动（项目无 pytest-asyncio，textual 的 run_test 本身无头），
按项目惯例标 e2e——默认套件不跑，手动 pytest -m e2e tests/test_tui_app.py 触发。
"""

from __future__ import annotations

import asyncio
import json

import pytest

# 可选依赖：textual 属 [tui] extra，未安装环境整体跳过（沿用 hdbscan 惯例）
pytest.importorskip("textual")

from textual.widgets import DataTable, Input

from llmsec.tui.app import LlmsecTUI
from llmsec.tui.console import ConsoleScreen
from llmsec.tui.task_store import TaskStore
from llmsec.tui.views import TaskLiveScreen

pytestmark = pytest.mark.e2e


def _fake_tasks(tmp_path) -> None:
    """伪造一个外部 evaluate 任务 + 一个 hpo 任务的磁盘痕迹。"""
    ev = tmp_path / "evaluate-101010-ab12cd.progress.jsonl"
    ev.write_text(
        json.dumps(
            {
                "ts": "2026-08-15T10:00:01",
                "phase": "attack",
                "target": "模型A",
                "round": 2,
                "max_rounds": 5,
                "elo": 1520.0,
                "delta": 20.0,
                "ci_half": 30.0,
                "progress_pct": 40,
                "converged": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "evaluate-101010-ab12cd.log").write_text("fake log\n", encoding="utf-8")
    hpo = tmp_path / "hpo-111111-aabbcc.progress.jsonl"
    hpo.write_text(
        json.dumps(
            {
                "ts": "2026-08-15T11:00:01",
                "phase": "hpo",
                "trial_done": 2,
                "trial_total_est": 10,
                "configs_done": 1,
                "configs_total": 8,
                "best_metric": 5.5,
                "metric_name": "conv_rounds",
                "direction": "minimize",
                "last": {"target": "模型A", "seed": 0, "status": "success", "value": 5.5, "params": {"K_FACTOR": 16}},
            }
        )
        + "\n",
        encoding="utf-8",
    )


from tests.utils import wait_until as _wait_until


def test_tui_smoke(tmp_path):
    _fake_tasks(tmp_path)

    async def _run() -> None:
        app = LlmsecTUI(store=TaskStore(log_dir=tmp_path))
        async with app.run_test(size=(120, 40)) as pilot:
            # 历史文件隔离（默认写仓库 STATE_DIR，测试不得污染真实历史）
            app._console._hist_path = lambda: tmp_path / "hist.txt"  # noqa: SLF001
            await asyncio.sleep(0.8)
            await pilot.pause()
            # 默认屏 = 控制台（无常驻可视化区域）
            assert isinstance(app.screen, ConsoleScreen)
            # 轮询线程首轮 refresh → 控制台快照发现 2 个外部任务
            assert len(app._console._snaps) == 2

            # top 唤起直播视图（真实轮询泵驱动），q 返回控制台
            inp = app.query_one("#cmd-bar", Input)
            inp.value = "top"
            await pilot.pause()
            await pilot.press("enter")
            await _wait_until(pilot, lambda: isinstance(app.screen, TaskLiveScreen))
            await asyncio.sleep(2.2)  # 下一轮 TasksUpdated（2s 周期）驱动视图更新
            await pilot.pause()
            assert app.screen.query_one("#live-table", DataTable).row_count == 2
            await pilot.press("q")
            await _wait_until(pilot, lambda: isinstance(app.screen, ConsoleScreen))

            # top hpo 只看 hpo 任务（直接驱动一次数据泵验证 kind 过滤）
            inp.value = "top hpo"
            await pilot.pause()
            await pilot.press("enter")
            await _wait_until(pilot, lambda: isinstance(app.screen, TaskLiveScreen))
            app.screen.update_tasks(app._console._snaps)
            await pilot.pause()
            assert app.screen.query_one("#live-table", DataTable).row_count == 1

    asyncio.run(_run())
