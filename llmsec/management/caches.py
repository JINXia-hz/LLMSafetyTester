"""management.caches — 列出 / 清理派生缓存。

缓存按「可删可重建」性质分类，clean 前展示各类占用与可重建性，避免误删权威存储。

类别：
  predictors        output/predictors/*.pkl            删了自动重建（load_or_fit 重训）
  （elo_cache 已表化进 catalog.db 的 elo_cache 表，指纹自动失效，无需清理类别）
  feature_cluster   output/feature_cache.pkl + cluster_result.pkl + embedding_cache.pkl
                                                       feature/embedding 可重建 / cluster 需重跑
  model_state       output/state/probes.json + prescreen_model.joblib  可重算/重训

绝不清：catalog.db（runs/trials/tasks/R 观测的统一库）。
"""

from __future__ import annotations

from pathlib import Path

from llmsec.core import config as _config  # 重绑常量调期动态读（work-dir 兼容）
from llmsec.core.config import OUTPUT_DIR
from llmsec.core.logging import get_logger
from llmsec.management.common import (
    Plan,
    dir_size,
    emit,
    fmt_size,
    print_plan,
    print_table,
    soft_remove,
)

logger = get_logger(__name__)


# 类别元数据：name → (paths 生成器, 可重建性, 描述)
# paths 生成器返回 list[(path, is_dir, detail)]
CACHE_CATEGORIES: dict[str, dict] = {
    "predictors": {
        "rebuildable": "automatic",
        "desc": "混合预测器 pkl，删了由 load_or_fit 重训",
        "paths": lambda: _predictor_paths(),
    },
    "feature_cluster": {
        "rebuildable": "feature/embedding 自动重建 / cluster 需重跑",
        "desc": "特征缓存 + embedding 缓存 + 聚类产物",
        "paths": lambda: [
            (_config.FEATURE_CACHE_FILE, False, "feature_cache.pkl"),
            (_config.EMBEDDING_CACHE_FILE, False, "embedding_cache.pkl"),
            (_config.CLUSTER_RESULT_FILE, False, "cluster_result.pkl"),
        ],
    },
    "model_state": {
        "rebuildable": "automatic",
        "desc": "模型指纹探测 + 预筛 ML 模型（可重算/重训）",
        "paths": lambda: [
            (_config.STATE_DIR / "probes.json", False, "probes.json"),
            (_config.STATE_DIR / "prescreen_model.joblib", False, "prescreen_model.joblib"),
        ],
    },
}


def _predictor_paths() -> list[tuple[Path, bool, str]]:
    if not _config.PREDICTORS_DIR.exists():
        return []
    return [(p, False, p.name) for p in _config.PREDICTORS_DIR.glob("*.pkl")]


def category_summary(name: str) -> dict:
    """单个类别的占用汇总（只计实际存在的文件——固定路径类别缺文件不计）。"""
    meta = CACHE_CATEGORIES[name]
    entries = meta["paths"]()
    files = [e for e in entries if not e[1] and e[0].exists()]
    dirs = [e for e in entries if e[1] and e[0].exists()]
    size = sum(dir_size(p) for p, _, _ in files) + sum(dir_size(p) for p, _, _ in dirs)
    return {
        "name": name,
        "desc": meta["desc"],
        "rebuildable": meta["rebuildable"],
        "file_count": len(files),
        "size": size,
    }


def all_category_summaries() -> list[dict]:
    return [category_summary(n) for n in CACHE_CATEGORIES]


# ============================================================
# list 子命令
# ============================================================
def cmd_list(json_mode: bool = False) -> int:
    summaries = all_category_summaries()
    if json_mode:
        emit({"categories": summaries, "count": len(summaries)}, json_mode=True, title="caches")
    else:
        rows = []
        for s in summaries:
            rows.append([
                s["name"],
                s["rebuildable"],
                str(s["file_count"]),
                fmt_size(s["size"]),
                s["desc"],
            ])
        print_table(rows, headers=["category", "rebuildable", "files", "size", "desc"])
        total = sum(s["size"] for s in summaries)
        print(f"\n合计可清理: {fmt_size(total)}")
        print("\n注: catalog.db（runs/trials/tasks/R 观测的统一库）不在清理范围，绝不清。")
    return 0


# ============================================================
# clean 子命令
# ============================================================
def plan_clean(categories: list[str]) -> Plan:
    """构造清理预览（dry-run）。"""
    plan = Plan(action="clean", dry_run=True)
    for cat in categories:
        if cat not in CACHE_CATEGORIES:
            plan.add(OUTPUT_DIR / cat, size=0, kind="unknown_category", detail=f"未知类别: {cat}")
            continue
        meta = CACHE_CATEGORIES[cat]
        for path, is_dir, detail in meta["paths"]():
            size = dir_size(path)
            kind = "cache_dir" if is_dir else "cache_file"
            plan.add(path, size=size, kind=kind,
                     detail=f"[{cat}] {detail}（{meta['rebuildable']}）")
    return plan


def execute_clean(plan: Plan) -> Plan:
    """执行清理（软删除）。"""
    done = Plan(action="clean", dry_run=False)
    for item in plan.items:
        if item.kind not in ("cache_file", "cache_dir"):
            done.add(item.path, size=0, kind=item.kind, detail="跳过")
            continue
        src = OUTPUT_DIR / item.path
        if src.exists():
            # cache_dir 用 rmtree，cache_file 用 remove
            from llmsec.management.common import soft_rmtree
            dest = soft_rmtree(src) if Path(src).is_dir() else soft_remove(src)
            done.add(src, size=item.size, kind=item.kind,
                     detail=f"已移到 {dest}" if dest else "失败")
        else:
            done.add(src, size=0, kind="missing", detail="已不存在")
    return done


def cmd_clean(categories: list[str], *, yes: bool = False, json_mode: bool = False) -> int:
    # 校验类别
    valid = [c for c in categories if c in CACHE_CATEGORIES]
    invalid = [c for c in categories if c not in CACHE_CATEGORIES]
    plan = plan_clean(valid)
    if invalid:
        for c in invalid:
            plan.add(c, size=0, kind="unknown_category", detail=f"未知类别（可选: {', '.join(CACHE_CATEGORIES)}）")
    if json_mode:
        if not yes:
            emit(plan.to_dict(), json_mode=True, title="clean (dry-run)")
            return 0
        done = execute_clean(plan)
        emit(done.to_dict(), json_mode=True, title="clean (executed)")
        return 0
    # 人可读
    print_plan(plan)
    if not yes:
        print(f"\n（dry-run 预览，{len([i for i in plan.items if i.kind in ('cache_file','cache_dir')])} 项"
              f"，将释放 {fmt_size(plan.total_size)}）\n确认执行请加 --yes")
        return 0
    print("\n执行中...")
    done = execute_clean(plan)
    cleaned = [i for i in done.items if i.kind in ("cache_file", "cache_dir")]
    print(f"✓ 完成：清理 {len(cleaned)} 项，释放 {fmt_size(done.total_size)}")
    return 0


# ============================================================
# prune 子命令（predictors LRU 修剪）
# ============================================================
def plan_prune_predictors(max_n: int) -> Plan:
    """构造 predictors LRU 修剪预览：按 predictor_cache.last_hit（P8 真 LRU，
    blend.py load_or_fit 的命中登记）保留最近 max_n 个，其余软删。

    先对账（reconcile_predictors：blob 文件是真相，登记行是派生索引，
    缺行文件按 mtime 补建）再取淘汰键；文件软删后，行由下次对账清除。
    审计设定的"数百再议"阈值已触发（3 天 142→398 个）——修剪是唯一能
    自动控制 predictors 体积的手段；被误删的活缓存代价 = 下次重训。
    """
    from llmsec.storage.contract import lru_evict_keys, reconcile_predictors

    plan = Plan(action="prune", dry_run=True)
    pkls = list(_config.PREDICTORS_DIR.glob("*.pkl")) if _config.PREDICTORS_DIR.exists() else []
    evict: set[str] = set()
    if pkls:
        reconcile_predictors()
        evict = set(lru_evict_keys(max_n))
    for p in pkls:
        if p.name[: -len(".pkl")] in evict:
            plan.add(p, size=dir_size(p), kind="cache_file", detail=f"LRU 淘汰（保留最近 {max_n}）")
    plan.extra["kept"] = min(len(pkls), max_n)
    plan.extra["total"] = len(pkls)
    return plan


def cmd_prune(max_n: int, *, yes: bool = False, json_mode: bool = False) -> int:
    plan = plan_prune_predictors(max_n)
    if json_mode:
        if not yes:
            emit(plan.to_dict(), json_mode=True, title="prune (dry-run)")
            return 0
        done = execute_clean(plan)  # 软删逻辑与 clean 共用
        emit(done.to_dict(), json_mode=True, title="prune (executed)")
        return 0
    print_plan(plan)
    if not yes:
        print(f"\n（dry-run 预览，共 {plan.extra['total']} 个 predictor，保留最新 {max_n}，"
              f"将淘汰 {len(plan.items)} 个 / {fmt_size(plan.total_size)}）\n确认执行请加 --yes")
        return 0
    print("\n执行中...")
    done = execute_clean(plan)
    print(f"✓ 完成：淘汰 {len(done.items)} 个 predictor，释放 {fmt_size(done.total_size)}")
    return 0
