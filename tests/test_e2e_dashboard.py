"""端到端全流程测试 —— 看板触发评估 → 子进程 → 进度轮询 → 报告产出。

默认**不跑**（`pytest tests/` 的 addopts 带 `-m 'not e2e'`）。
手动触发：`pytest -m e2e tests/test_e2e_dashboard.py -v -n 0`

覆盖的链路（与 tasks.py 生产路径完全相同）：
  POST /api/run/evaluate
    → _start_task（排队 + spawn subprocess）
      → subprocess: python -m llmsec.pipeline.runner --phase 1 ...
        → 真实 call_target（.env 目标模型）+ 真实 Judge（.env GENERATOR_*）
      → output/tasks/<id>.progress.jsonl（逐轮进度）
    → GET /api/tasks/{id}（状态轮询）
    → GET /api/tasks/{id}/progress（进度快照）
    → POST /api/tasks/{id}/cancel（取消）

特性：
  - 产生真实 API 费用（最小攻击集 + batch=2 + rounds=1，约 2~4 次目标调用 + 2~4 次 Judge）。
  - 凭证复用 .env；缺失时 require_real_api fixture skip。
  - 全局 R 零污染：patch task_manager.start_task 把 --publish-global 改写为 --work-dir <tmp>。
  - 建议串行跑（`-n 0`，覆盖 addopts 的 `-n 4`），子进程轮询不适合 xdist worker。
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# ============================================================
# 参数：用最小代价跑完整链路（控费用/时间）
# ============================================================
E2E_INPUT = "_unit_smoke.jsonl"   # attacks/ 下最小攻击集（200 条，batch=2 只取 2 条）
# 该攻击集未入库（attacks/ 除 example.jsonl 外均 gitignore）：fresh clone 上
# 手动跑 -m e2e 时优雅跳过，而非启动子进程后才 404
_INPUT_PATH = Path(__file__).resolve().parents[1] / "output" / "attacks" / E2E_INPUT
pytestmark = [pytest.mark.e2e,
              pytest.mark.skipif(not _INPUT_PATH.exists(),
                                 reason=f"攻击集 {E2E_INPUT} 未入库（本地生成后可跑）")]
E2E_BATCH = 2                     # 每轮 2 条攻击
E2E_ROUNDS = 1                    # 单轮（不收敛、不自适应）
E2E_TIMEOUT = 300                 # 任务终态轮询超时（秒）


# ============================================================
# 辅助
# ============================================================
def _client() -> TestClient:
    from llmsec.server.dashboard_api import app
    return TestClient(app)


def _wait_task_terminal(client: TestClient, task_id: str, timeout: float = E2E_TIMEOUT) -> dict:
    """轮询 GET /api/tasks/{id} 直到终态（success/failed/cancelled），返回最终 view。

    _refresh_task_status 在每次 _task_view 时检查 proc.poll()，故 TestClient 无需
    额外触发后台任务即可感知子进程结束。
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/tasks/{task_id}")
        assert r.status_code == 200, f"轮询任务状态失败: HTTP {r.status_code}"
        view = r.json()
        if view["status"] in ("success", "failed", "cancelled"):
            return view
        time.sleep(2)
    pytest.fail(f"⏰ 任务 {task_id} 超时未结束（{E2E_TIMEOUT}s）")


@pytest.fixture
def isolated_tasks(monkeypatch, tmp_path):
    """把看板评估的 --publish-global 改写为 --work-dir <tmp>，全局 R 零污染。

    tasks.api_run_evaluate 在 argv 末尾硬加 --publish-global（L232）。
    我们 patch task_manager.start_task：spawn 前扫描 argv，把 --publish-global 替换为
    --work-dir <tmp>/wd，让 runner 的 rebind_to_workdir 把全部产物重绑到 tmp。
    这样 e2e 测试不会往全局 output/state/results.json 写任何东西。
    """
    from llmsec.server import task_manager

    work_dir = tmp_path / "wd"
    work_dir.mkdir()
    orig_start_task = task_manager.start_task

    def _patched_start_task(kind: str, argv: list[str]) -> dict:
        # 改写 argv：去 --publish-global，加 --work-dir（隔离模式）
        cleaned = [a for a in argv if a != "--publish-global"]
        cleaned += ["--work-dir", str(work_dir)]
        return orig_start_task(kind, cleaned)

    monkeypatch.setattr(task_manager, "start_task", _patched_start_task)
    return work_dir


# ============================================================
# 全流程：触发 → 轮询 → 终态校验
# ============================================================
class TestDashboardEvaluateFlow:
    """看板评估全流程：POST /api/run/evaluate → 子进程跑完 → success。"""

    def test_evaluate_full_pipeline_success(self, require_real_api, isolated_tasks):
        """🎯 完整评估链路：触发 → 排队/运行 → success。

        走真实 runner 子进程（与 tasks.py 生产路径相同），target 真实 API。
        全局 R 经 isolated_tasks fixture 隔离到 tmp，零污染。
        """
        client = _client()
        r = client.post("/api/run/evaluate", json={
            "phase": "1",              # 只跑攻击阶段，省过敏阶段费用
            "input": E2E_INPUT,
            "batch_size": E2E_BATCH,
            "max_rounds": E2E_ROUNDS,
            "sampler": "gap",
        })
        assert r.status_code == 200, f"❌ 触发评估失败: HTTP {r.status_code} {r.text}"

        task = r.json()
        task_id = task["id"]
        assert task["status"] in ("queued", "running"), f"任务初始状态异常: {task['status']}"

        view = _wait_task_terminal(client, task_id)
        assert view["status"] == "success", (
            f"❌ 评估子进程未成功结束（status={view['status']}, rc={view.get('returncode')}）:\n"
            f"{view.get('log_tail', '')}"
        )

    def test_evaluate_produces_progress_records(self, require_real_api, isolated_tasks):
        """📊 评估运行期间产出 progress.jsonl 进度记录（GET /api/tasks/{id}/progress 可读）。

        即使任务很快结束，progress 文件也应至少有一行（每轮/每目标一条）。
        """
        client = _client()
        r = client.post("/api/run/evaluate", json={
            "phase": "1",
            "input": E2E_INPUT,
            "batch_size": E2E_BATCH,
            "max_rounds": E2E_ROUNDS,
            "sampler": "gap",
        })
        task_id = r.json()["id"]

        # 等任务跑完
        _wait_task_terminal(client, task_id)

        # 查进度快照
        pr = client.get(f"/api/tasks/{task_id}/progress")
        assert pr.status_code == 200
        body = pr.json()
        assert body["kind"] == "evaluate"
        # progress 字段应非空（至少有一条记录）
        progress = body.get("progress", {})
        assert progress, (
            f"❌ progress 为空，子进程可能未落进度记录。log_tail:\n"
            f"{client.get(f'/api/tasks/{task_id}').json().get('log_tail', '')}"
        )


# ============================================================
# 任务取消：queued / running → cancelled
# ============================================================
class TestDashboardTaskCancel:
    """POST /api/tasks/{id}/cancel 能终止运行中/排队中的任务。"""

    def test_cancel_running_task(self, require_real_api, isolated_tasks):
        """🛑 运行中的任务可被 cancel 终止（SIGTERM → 5s → SIGKILL）。

        触发一个评估任务，立即尝试 cancel。任务可能已结束（小任务很快），
        此时 cancel 返回 409——这是合法状态，不算失败。
        """
        client = _client()
        r = client.post("/api/run/evaluate", json={
            "phase": "1",
            "input": E2E_INPUT,
            "batch_size": E2E_BATCH,
            "max_rounds": E2E_ROUNDS,
            "sampler": "gap",
        })
        task_id = r.json()["id"]

        # 立即尝试取消（任务大概率还在 running）
        cr = client.post(f"/api/tasks/{task_id}/cancel")
        if cr.status_code == 409:
            # 任务已自然结束（小 batch + 单轮很快），cancel 无意义——跳过
            pytest.skip("任务在 cancel 前已自然结束（小任务跑得太快）")
        assert cr.status_code == 200, f"cancel 失败: HTTP {cr.status_code} {cr.text}"

        view = cr.json()
        assert view["status"] == "cancelled", f"取消后状态应为 cancelled，实际: {view['status']}"
