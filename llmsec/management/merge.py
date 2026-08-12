"""management.merge — 通用 R 矩阵合并（单元化的显式统一动作）。

runner 默认不再自动 publish 到全局 R（单元化原则）；要把一个或多个 work-dir /
workspace / run 的观测合并进全局 R 或另一工作区，用本模块显式触发。

语义：
    sources (多个) → target (一个)
    sources: "global" | "ws:<name>" | 任意含 results.json 的目录路径
    target:  "global" | "ws:<name>"
    --models: 只合并指定 model 列（默认 source 的全部 model）

实现复用 ResultsMatrix.load/save + _file_lock + upsert，不新增 R 操作原语。
默认 dry-run（预览将合并多少条、影响哪些 model）+ --yes 执行。
"""

from __future__ import annotations

from pathlib import Path

from llmsec.core.config import OUTPUT_DIR, RESULTS_FILE
from llmsec.core.logging import get_logger
from llmsec.core.results import ResultsMatrix, _file_lock
from llmsec.management.common import Plan, emit, fmt_size, print_table

logger = get_logger(__name__)

WORKSPACES_DIR = OUTPUT_DIR / "workspaces"


# ============================================================
# 源/目标解析
# ============================================================
def _resolve_results_path(spec: str) -> Path:
    """把源/目标描述符解析为 results.json 的绝对路径。

    spec 形式：
      "global"         → output/state/results.json
      "ws:<name>"      → output/workspaces/<name>/results.json
      其他             → 视为目录路径，取其下 results.json
    """
    if spec == "global":
        return RESULTS_FILE
    if spec.startswith("ws:"):
        return WORKSPACES_DIR / spec[3:] / "results.json"
    p = Path(spec)
    # 若直接指向文件就用它，否则视为目录取 results.json
    if p.suffix == ".json":
        return p
    return p / "results.json"


def _load_R(path: Path) -> ResultsMatrix:
    """从指定 results.json 加载 R；文件不存在视为空矩阵。"""
    if not path.exists():
        logger.warning("源 results.json 不存在（视为空）: %s", path)
        return ResultsMatrix()
    return ResultsMatrix.load(path)


# ============================================================
# plan / execute
# ============================================================
def plan_merge(
    sources: list[str],
    target: str,
    *,
    models: list[str] | None = None,
) -> Plan:
    """构造合并预览（dry-run）。不执行写操作。

    合并语义：对每个 source 的每个 model（或 --models 指定的子集），把该列全部观测
    upsert 到 target R（同 record+model 覆盖，不同 record 累加）。
    """
    plan = Plan(action="merge", dry_run=True)
    plan.extra["target"] = target
    plan.extra["sources"] = sources
    plan.extra["models_filter"] = models

    target_path = _resolve_results_path(target)
    plan.extra["target_path"] = str(target_path)
    if not target_path.exists():
        plan.add(target_path, size=0, kind="target_new", detail="目标 R 不存在，将新建")

    # 聚合各 source 的 model 列
    source_models: dict[str, set[str]] = {}
    for src in sources:
        src_path = _resolve_results_path(src)
        if not src_path.exists():
            plan.add(src_path, size=0, kind="source_missing", detail=f"源 {src} 不存在，跳过")
            continue
        src_R = _load_R(src_path)
        src_models = src_R.all_models()
        if models:
            src_models = [m for m in src_models if m in models]
        source_models[src] = set(src_models)
        plan.add(src_path, size=src_path.stat().st_size if src_path.exists() else 0,
                 kind="source", detail=f"源 {src}：{len(src_R.all_models())} 模型 "
                       f"({', '.join(src_models) or '空'})，{sum(len(c) for c in src_R._r.values())} 条观测")

    # 预览：每个待合并 model 的记录数（target 已有 vs source 新增）
    target_R = _load_R(target_path)
    preview: dict[str, dict] = {}
    all_models = set().union(*source_models.values()) if source_models else set()
    for model in sorted(all_models):
        target_n = target_R.n_for_model(model)
        # 合并后该 model 的记录数上界（target 已有 + 各 source 新增，去重按 record）
        source_records: set[str] = set()
        for src in sources:
            src_R = _load_R(_resolve_results_path(src))
            source_records |= set(src_R.model_column(model).keys())
        new_records = source_records - set(target_R.model_column(model).keys())
        preview[model] = {
            "target_existing": target_n,
            "source_records": len(source_records),
            "new_to_target": len(new_records),
        }
    plan.extra["per_model"] = preview
    plan.extra["total_new"] = sum(p["new_to_target"] for p in preview.values())
    return plan


def execute_merge(
    sources: list[str],
    target: str,
    *,
    models: list[str] | None = None,
) -> Plan:
    """执行合并。target R 经 _file_lock + save(backup=True) 原子写。"""
    target_path = _resolve_results_path(target)
    # load target（在锁内 RMW）
    with _file_lock(target_path):
        target_R = _load_R(target_path)
        merged_counts: dict[str, int] = {}
        for src in sources:
            src_path = _resolve_results_path(src)
            if not src_path.exists():
                continue
            src_R = _load_R(src_path)
            for model in src_R.all_models():
                if models and model not in models:
                    continue
                col = src_R.model_column(model)
                for record, res in col.items():
                    target_R.upsert(record, model, res.eval_score, res.status, res.ts, dict(res.extra))
                merged_counts[model] = merged_counts.get(model, 0) + len(col)
        target_R.save(target_path, _locked=True)

    done = Plan(action="merge", dry_run=False)
    done.extra["target"] = target
    done.extra["merged_counts"] = merged_counts
    done.extra["total_merged"] = sum(merged_counts.values())
    done.add(target_path, size=target_path.stat().st_size if target_path.exists() else 0,
             kind="target_written", detail=f"合并 {sum(merged_counts.values())} 条观测，涉及 {len(merged_counts)} 模型")
    logger.info("merge 完成 → %s：%d 条观测（models=%s）",
                target, done.extra["total_merged"], sorted(merged_counts))
    return done


# ============================================================
# CLI 命令
# ============================================================
def cmd_merge(
    sources: list[str],
    target: str,
    *,
    models: list[str] | None = None,
    yes: bool = False,
    json_mode: bool = False,
) -> int:
    plan = plan_merge(sources, target, models=models)
    if json_mode:
        if not yes:
            emit(plan.to_dict(), json_mode=True, title="merge (dry-run)")
            return 0
        done = execute_merge(sources, target, models=models)
        emit(done.to_dict(), json_mode=True, title="merge (executed)")
        return 0
    # 人可读
    print(f"操作: merge  模式: {'dry-run' if plan.dry_run else 'executed'}")
    print(f"目标: {target} ({plan.extra['target_path']})")
    rows = [[it.kind, it.path, fmt_size(it.size), it.detail] for it in plan.items]
    print_table(rows, headers=["kind", "path", "size", "detail"])
    # per-model 预览
    pm = plan.extra.get("per_model", {})
    if pm:
        print("\n按模型预览（target 已有 / source 记录 / 新增到 target）：")
        pm_rows = [[m, str(d["target_existing"]), str(d["source_records"]), str(d["new_to_target"])]
                   for m, d in pm.items()]
        print_table(pm_rows, headers=["model", "target现有", "source记录", "新增"])
    print(f"\n合计将新增 {plan.extra.get('total_new', 0)} 条观测到 target")
    if not yes:
        print("（dry-run 预览）确认执行请加 --yes")
        return 0
    print("\n执行中...")
    done = execute_merge(sources, target, models=models)
    print(f"✓ 完成：合并 {done.extra['total_merged']} 条观测 → {target}")
    return 0
