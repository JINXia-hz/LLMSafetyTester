"""control.core.orchestrator — 批量并行编排（fork N 个工作单元并行跑）。

场景：A/B 对比实验、参数扫描。每个工作单元 = 一个隔离 work-dir 的 llmsec run。

模式（参照 llmsec/experiments/study.py 的并发，但控制层不依赖 experiments）：
  - 每个 spec 一个独立 workspace（fork 出来）
  - ThreadPoolExecutor 并行起 runner 子进程
  - 收集结果，可选自动 compare

spec = {name, source, target, max_rounds, seed, ...} 的列表。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime

from control.core.workspace import fork_and_run


@dataclass
class RunSpec:
    """单个并行工作单元的规格。"""
    name: str                       # workspace 名（唯一）
    source: str = "global"          # fork 来源
    target: str | None = None       # 目标模型
    input_file: str = "attacks/l1.jsonl"
    max_rounds: int = 5
    seed: int | None = None
    note: str = ""
    extra_argv: list[str] = field(default_factory=list)


def orchestrate(
    specs: list[RunSpec],
    *,
    max_workers: int = 2,
    timeout: float | None = None,
    compare_after: bool = True,
) -> dict:
    """并行 fork + run 多个工作单元。

    Args:
        specs: RunSpec 列表
        max_workers: 并行度（每个起一个 runner 子进程）
        timeout: 单个 run 超时秒
        compare_after: 全部完成后是否自动跑 compare（基于各 workspace 的 run 产物）

    Returns:
        {started, workers, results: [...], summary: {...}}
    """
    started = datetime.now().isoformat(timespec="seconds")
    results: list[dict] = []

    def _one(spec: RunSpec) -> dict:
        try:
            r = fork_and_run(
                spec.name, source=spec.source, target=spec.target,
                input_file=spec.input_file, max_rounds=spec.max_rounds,
                seed=spec.seed, note=spec.note or spec.name, timeout=timeout,
            )
            r["spec"] = asdict(spec)
            r["status"] = "success" if r["run"]["ok"] else "failed"
            return r
        except Exception as e:
            return {
                "spec": asdict(spec),
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
            }

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_one, s): s for s in specs}
        for fut in as_completed(futures):
            results.append(fut.result())

    # 汇总
    n_ok = sum(1 for r in results if r.get("status") == "success")
    summary = {
        "total": len(results),
        "success": n_ok,
        "failed": len(results) - n_ok,
    }

    report: dict = {
        "started": started,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "max_workers": max_workers,
        "results": results,
        "summary": summary,
    }

    # 可选：自动对比（读各 workspace 内的 runner_report.json）
    if compare_after and n_ok > 0:
        report["comparison"] = _compare_workspaces([r for r in results if r.get("status") == "success"])

    # 门下省事后审查：对每个成功的 workspace 自动生成审查摘要
    if n_ok > 0:
        report["reviews"] = _auto_review([r for r in results if r.get("status") == "success"])

    return report


def _auto_review(success_results: list[dict]) -> list[dict]:
    """对成功的 workspace 自动跑门下省审查（规则版，不用 LLM，快速呈递）。"""
    from control.agent.menxia import review_run
    reviews = []
    for r in success_results:
        ws = r.get("workspace", {})
        ws_name = ws.get("name", "")
        if not ws_name:
            continue
        try:
            rev = review_run(f"ws:{ws_name}", use_llm=False)
            if "error" not in rev:
                reviews.append({"workspace": ws_name, "summary": rev["summary"],
                                "n_findings": len(rev["findings"])})
        except Exception:
            pass  # 审查失败不影响主流程
    return reviews


def _compare_workspaces(success_results: list[dict]) -> dict:
    """对成功的 workspace 做 contrast 对比（复用 compare.run_metrics 的 ws: 解析）。"""
    from control.core.compare import run_metrics
    rows = []
    for r in success_results:
        ws = r.get("workspace", {})
        ws_name = ws.get("name", "")
        # run_metrics 经 _resolve_run_dir 识别 ws:<name> 前缀，自动定位 <ws>/<target>/runner_report.json
        m = run_metrics(f"ws:{ws_name}")
        if m is None:
            continue
        rows.append({
            "workspace": ws_name,
            "target": m.get("target_model"),
            "asr": m.get("asr"),
            "fpr": m.get("fpr"),
            "boundary_elo": m.get("boundary_elo"),
            "conv_rounds": m.get("conv_rounds"),
            "security_level": m.get("security_level"),
        })
    return {"runs": rows}
