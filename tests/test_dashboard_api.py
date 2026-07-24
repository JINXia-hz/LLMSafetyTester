#!/usr/bin/env python3
"""
冒烟测试：LLMSEC Web 面板 API。

验证：
1. 首页与全部数据 API 返回 200 且结构正确。
2. 非法 run 参数被 400 拦截（防路径穿越）。
3. /api/model 在缺少 svd_ridge 数据时优雅降级（available=False）。
4. 评估任务对不存在的攻击集返回 404。
5. 任务运行器能启动/跟踪/完成一个轻量子进程。
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Windows GBK 控制台兼容：允许输出 ✅/❌
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from fastapi.testclient import TestClient

from llmsec.server.dashboard_api import TASKS, _start_task, app

client = TestClient(app)


def _check(cond: bool, msg: str) -> int:
    if not cond:
        print(f"❌ {msg}")
        return 1
    return 0


def test_index_and_data_apis() -> int:
    rc = 0
    r = client.get("/")
    rc |= _check(r.status_code == 200 and "LLMSEC" in r.text, "首页 200 且包含标题")

    r = client.get("/api/runs")
    rc |= _check(r.status_code == 200 and "runs" in r.json(), "/api/runs 结构")

    r = client.get("/api/overview")
    rc |= _check(r.status_code == 200, "/api/overview 200")
    d = r.json()
    if d.get("available"):
        rc |= _check(len(d.get("radar", {}).get("labels", [])) == 5, "雷达图五维")
        rc |= _check(len(d.get("radar", {}).get("values", [])) == 5, "雷达图五值")
        rc |= _check(all(0 <= v <= 1 for v in d["radar"]["values"]), "雷达值域 [0,1]")

    for path in ["/api/threats", "/api/elo", "/api/report-md", "/api/clusters",
                 "/api/model", "/api/attack-sets", "/api/tasks"]:
        r = client.get(path)
        rc |= _check(r.status_code == 200, f"{path} 200")

    print("✅ 首页与数据 API 通过")
    return rc


def test_run_param_validation() -> int:
    rc = 0
    r = client.get("/api/overview?run=../../etc")
    rc |= _check(r.status_code == 400, "路径穿越被 400 拦截")
    r = client.get("/api/overview?run=2026-01-01_000000")
    rc |= _check(r.status_code == 200, "合法但不存在的 run 不报错（available=False 或空）")
    print("✅ run 参数校验通过")
    return rc


def test_model_fallback() -> int:
    r = client.get("/api/model")
    d = r.json()
    rc = _check("available" in d, "/api/model 含 available 字段")
    if not d["available"]:
        rc |= _check("run" in d, "无 svd_ridge 时优雅降级")
    print("✅ /api/model 容错通过")
    return rc


def test_evaluate_validation() -> int:
    rc = 0
    r = client.post("/api/run/evaluate", json={"input": "../../etc/passwd"})
    rc |= _check(r.status_code in (400, 404), "非法 input 被拦截")
    r = client.post("/api/run/evaluate", json={"input": "not_exists.jsonl"})
    rc |= _check(r.status_code == 404, "不存在的攻击集 404")
    r = client.post("/api/run/evaluate", json={"input": "l1.jsonl", "phase": "bogus"})
    rc |= _check(r.status_code == 422, "非法 phase 被 pydantic 拦截")
    print("✅ 评估参数校验通过")
    return rc


def test_task_lifecycle() -> int:
    view = _start_task("smoke", ["-c", "print('smoke-ok')"])
    task_id = view["id"]
    if task_id not in TASKS:
        print("❌ 任务未注册")
        return 1
    deadline = time.time() + 30
    status = view["status"]
    while time.time() < deadline:
        r = client.get(f"/api/tasks/{task_id}")
        status = r.json()["status"]
        if status != "running":
            break
        time.sleep(0.3)
    if status != "success":
        print(f"❌ 任务未成功结束: {status}")
        return 1
    r = client.get(f"/api/tasks/{task_id}")
    if "smoke-ok" not in r.json().get("log_tail", ""):
        print("❌ 日志尾缺少子进程输出")
        return 1
    r = client.get("/api/tasks/nonexistent")
    if r.status_code != 404:
        print("❌ 不存在任务应 404")
        return 1
    print("✅ 任务生命周期通过")
    return 0


def main() -> int:
    tests = [
        test_index_and_data_apis,
        test_run_param_validation,
        test_model_fallback,
        test_evaluate_validation,
        test_task_lifecycle,
    ]
    for t in tests:
        if t() != 0:
            return 1
    print("\n✅ 所有 Web 面板冒烟测试通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
