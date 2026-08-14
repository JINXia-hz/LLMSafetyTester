"""llmsec.mcp.tools.actions — Tier 3 写操作工具（带两步确认）。

危险操作（delete/clean）走 preview → confirm 两步模式：
  1. agent 调 *_preview 获取影响摘要 + confirm_token
  2. agent（或用户）审阅后调 *_confirm(token) 才真执行

低风险操作（fork/snapshot export）直接执行。

暴露的工具：
  - delete_runs_preview / delete_runs_confirm   删除 run（软删到 .trash/）
  - clean_caches_preview / clean_caches_confirm 清理可重建缓存
  - fork_workspace                               fork 隔离工作区（直接执行）
  - export_snapshot                              导出 R 矩阵快照（直接执行）
"""

from __future__ import annotations

from typing import Any

from llmsec.mcp import confirm as confirm_mod


# ============================================================
# 辅助
# ============================================================
def _try(fn, *, error_hint: str = "") -> Any:
    try:
        return fn()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "hint": error_hint}


def _validate_merge_spec(spec: str, *, is_target: bool = False) -> str:
    """校验 merge 的 source/target 描述符，防 LLM 传外部路径穿越。

    允许的形式：
      "global"        — 全局 R
      "ws:<name>"     — 工作区（name 由 management 层 safe_component 再校验）
      绝对/相对路径   — 须落在 OUTPUT_DIR 内（合并已导出快照的合法场景）

    target 额外约束：仅允许 "global" / "ws:<name>"（写目标不能是裸路径）。
    """
    from pathlib import Path

    from llmsec.core.config import OUTPUT_DIR

    if spec == "global" or spec.startswith("ws:"):
        return spec
    if is_target:
        raise ValueError(f"target 仅支持 'global' 或 'ws:<name>'，收到 {spec!r}")
    # source 裸路径：须落在 OUTPUT_DIR 内
    p = Path(spec)
    if any(part == ".." for part in p.parts):
        raise ValueError(f"source 路径含穿越段（..）: {spec!r}")
    pr = p.resolve() if p.is_absolute() else (OUTPUT_DIR / p).resolve()
    out_r = OUTPUT_DIR.resolve()
    if pr != out_r and out_r not in pr.parents:
        raise ValueError(f"source 路径越界，须在 output/ 内: {spec!r}")
    return spec


# ============================================================
# delete_runs — 两步确认
# ============================================================
def delete_runs_preview(names: list[str], delete_r: bool = False) -> dict[str, Any]:
    """预览删除评估 run 的影响（不执行任何写操作）。

    返回影响摘要和 confirm_token。审阅后调用 delete_runs_confirm(token) 才真删除。
    删除是软删除（移到 .trash/，可恢复）。

    Args:
        names:    要删除的 run 名称列表（用 list_runs 查可用 run）。
        delete_r: 是否同时从 R 矩阵移除这些 run 对应模型的观测列。

    Returns:
        {action, summary, total_size_human, confirm_token, ttl_seconds}
    """
    from llmsec.management.runs import plan_delete

    def _do() -> dict[str, Any]:
        plan = plan_delete(names, delete_r=delete_r)
        plan_dict = plan.to_dict()
        r_models = plan.extra.get("r_models_affected", [])
        token = confirm_mod.issue(
            action="delete_runs",
            summary=plan_dict,
            execute_fn=lambda: _execute_delete(names, delete_r),
            args_repr=f"names={names}, delete_r={delete_r}",
        )
        return {
            "action": "delete_runs",
            "summary": plan_dict,
            "impact_note": (
                f"将删除 {len(plan_dict['items'])} 项，释放 {plan_dict['total_size_human']}"
                + (f"，并从 R 矩阵移除模型 {r_models} 的观测列" if delete_r and r_models else "")
            ),
            "confirm_token": token,
            "ttl_seconds": 300,
            "next_step": "审阅后调用 delete_runs_confirm(token) 执行删除",
        }

    return _try(_do, error_hint="run 名不存在？用 list_runs 查可用 run。")


def _execute_delete(names: list[str], delete_r: bool) -> dict[str, Any]:
    """实际执行删除（由 confirm 机制触发）。"""
    from llmsec.management.runs import execute_delete, plan_delete

    plan = plan_delete(names, delete_r=delete_r)
    done = execute_delete(plan, delete_r=delete_r)
    return done.to_dict()


def delete_runs_confirm(token: str) -> dict[str, Any]:
    """用 confirm_token 执行已预览的 run 删除操作。

    Args:
        token: delete_runs_preview 返回的 confirm_token。

    Returns:
        {status, result}。status 为 "executed" 或 "expired_or_already_confirmed"。
    """
    return confirm_mod.confirm(token)


# ============================================================
# clean_caches — 两步确认
# ============================================================
def clean_caches_preview(categories: list[str]) -> dict[str, Any]:
    """预览清理可重建缓存的影响（不执行删除）。

    所有缓存类别都是可重建的（elo_cache / predictors 自动重建，task_logs 一次性），
    但清理会释放磁盘空间。

    Args:
        categories: 要清理的缓存类别列表，可选值：
            "elo_cache"      — Elo 派生缓存（删后从 R 重算）
            "predictors"     — 混合预测器 pkl（删后重训）
            "feature_cluster"— 特征缓存 + 聚类产物（特征自动重建/聚类需重跑）
            "task_logs"      — 已完成任务的日志（一次性，不可恢复）

    Returns:
        {action, summary, confirm_token, ttl_seconds}
    """
    from llmsec.management.caches import plan_clean

    def _do() -> dict[str, Any]:
        plan = plan_clean(categories)
        plan_dict = plan.to_dict()
        token = confirm_mod.issue(
            action="clean_caches",
            summary=plan_dict,
            execute_fn=lambda: _execute_clean(categories),
            args_repr=f"categories={categories}",
        )
        return {
            "action": "clean_caches",
            "summary": plan_dict,
            "impact_note": f"将清理 {len(plan_dict['items'])} 项，释放 {plan_dict['total_size_human']}",
            "confirm_token": token,
            "ttl_seconds": 300,
            "next_step": "审阅后调用 clean_caches_confirm(token) 执行清理",
        }

    return _try(_do)


def _execute_clean(categories: list[str]) -> dict[str, Any]:
    from llmsec.management.caches import execute_clean, plan_clean

    plan = plan_clean(categories)
    done = execute_clean(plan)
    return done.to_dict()


def clean_caches_confirm(token: str) -> dict[str, Any]:
    """用 confirm_token 执行已预览的缓存清理操作。

    Args:
        token: clean_caches_preview 返回的 confirm_token。

    Returns:
        {status, result}。
    """
    return confirm_mod.confirm(token)


# ============================================================
# fork_workspace — 低风险，直接执行
# ============================================================
def fork_workspace(name: str, source: str = "global", note: str = "") -> dict[str, Any]:
    """fork 一个隔离工作区：从全局（或指定 run）复制一份 R 矩阵副本。

    工作区允许在不影响全局数据的前提下跑实验。实验完成后可 merge 回全局。

    Args:
        name:   工作区名（唯一标识）。
        source: 数据来源："global"（全局 R 矩阵）或 "run:<run_name>"（某次 run 的 state）。
        note:   备注说明（记入索引）。

    Returns:
        工作区信息 dict（name/path/source/models/records 等）。
    """
    from control.core.workspace import fork

    return _try(
        lambda: fork(name, source=source, note=note),
        error_hint=f"工作区 '{name}' 可能已存在，或源 '{source}' 无效",
    )


# ============================================================
# export_snapshot — 低风险，直接执行
# ============================================================
def export_snapshot(
    source: str = "global",
    out: str | None = None,
    include_elo_cache: bool = True,
) -> dict[str, Any]:
    """导出 R 矩阵快照到 output/snapshots/（或指定路径）。

    快照包含 results.json（和可选的 elo_cache.json），用于备份或迁移。

    Args:
        source:            "global" 或 "run:<name>"。
        out:               输出路径（目录或 .tar.gz）；None 则默认到 output/snapshots/<时间戳>/。
        include_elo_cache: 是否一并导出 elo_cache.json（仅 global 源有效）。

    Returns:
        快照元信息 dict（含 snapshot 路径、models、records）。
    """
    from pathlib import Path

    from llmsec.management.snapshot import export_snapshot as _es

    return _try(
        lambda: _es(source=source, out=Path(out) if out else None, include_elo_cache=include_elo_cache),
        error_hint=f"源 '{source}' 无效（用 'global' 或 'run:<name>'），或 R 矩阵不存在",
    )


# ============================================================
# env_snapshot — 连接配置的隔离快照（CRUD）
# ============================================================
# env_snapshot 用于隔离一次评估的连接配置（TARGET_*/GENERATOR_*/JUDGE_* 等），
# 不碰全局 .env。典型流程：
#   create(source="blank") → edit_key 写入 API key 等 → run_evaluation(env_snapshot=...)
# 快照存储在 output/env_snapshots/<name>/.env，是纯文本 KEY=VALUE 文件。

def create_env_snapshot(
    name: str,
    source: str = "global",
    note: str = "",
) -> dict[str, Any]:
    """创建一个 .env 配置快照（隔离的连接配置副本）。

    快照让你为一次评估指定独立的模型列表 / API key / judge 配置，不碰全局 .env。
    创建后可用 edit_env_snapshot 修改里面的 key，再用 run_evaluation(env_snapshot=...) 使用。

    Args:
        name:   快照名（唯一标识）。
        source: 配置来源：
            "global" — 从全局 .env 复制当前配置（默认）
            "blank"  — 创建空快照，之后用 edit_env_snapshot 逐条写入
            也可指定另一个快照名，基于它创建副本。
        note:   备注说明。

    Returns:
        快照信息 dict（name/path/source/keys/note）。
    """
    from control.core.env_snapshot import create

    return _try(
        lambda: create(name, source=source, note=note),
        error_hint=f"快照 '{name}' 可能已存在，或源 '{source}' 无效",
    )


def edit_env_snapshot(name: str, key: str, value: str) -> dict[str, Any]:
    """修改快照里的某个配置项（写入或更新一个 KEY=VALUE）。

    受管理的 key 前缀（只有这些能写入，防止乱写）：
      TARGETS / TARGET_*  — 目标模型配置（如 TARGET_x_BASE_URL, TARGET_x_API_KEY）
      GENERATOR_*         — 生成器（攻击 prompt 生成）配置
      JUDGE_* / JUDGE_API_KEY — Judge（评判）配置
      CONTROL_*           — 控制层 LLM 配置
      LLMSEC_PARAM_*      — params.py 运行时参数覆写

    Args:
        name:  快照名。
        key:   .env key（须在受管理前缀范围内）。
        value: 新值。

    Returns:
        {name, key, value, keys}（keys 为快照当前所有 key 列表）。
    """
    from control.core.env_snapshot import edit_key

    return _try(
        lambda: edit_key(name, key, value),
        error_hint=f"快照 '{name}' 不存在，或 key '{key}' 不在受管理前缀范围内",
    )


def list_env_snapshots() -> list[dict[str, Any]]:
    """列出所有 .env 配置快照（按创建时间倒序）。

    Returns:
        快照信息列表，每条含 name/source/keys/note/created。
    """
    from control.core.env_snapshot import list_snapshots

    return _try(list_snapshots, error_hint="env_snapshots 目录可能不存在")


def get_env_config() -> dict[str, Any]:
    """读取当前全局 .env 的连接配置（脱敏：API key 只显示前 8 位）。

    用于了解当前已配置了哪些连接（哪些 key 已填、哪些还缺）。

    Returns:
        {configured: {...}, missing: [...], raw_keys: [...]}
    """

    from control.core.env_snapshot import _read_global_env

    def _do() -> dict[str, Any]:
        env = _read_global_env()
        # 脱敏：API_KEY 类只显示前 8 位
        masked = {}
        for k, v in env.items():
            if "KEY" in k.upper() and len(v) > 8:
                masked[k] = v[:8] + "***"
            else:
                masked[k] = v
        # 检查关键配置是否齐全
        essential = ["GENERATOR_API_KEY", "GENERATOR_BASE_URL"]
        missing = [k for k in essential if not env.get(k)]
        return {
            "configured": masked,
            "missing_essential": missing,
            "total_keys": len(env),
        }

    return _try(_do)


def delete_env_snapshot(name: str) -> dict[str, Any]:
    """删除一个 .env 配置快照。

    Args:
        name: 要删除的快照名。

    Returns:
        {deleted: name, info: {...}}。
    """
    from control.core.env_snapshot import delete

    return _try(
        lambda: delete(name),
        error_hint=f"快照 '{name}' 不存在",
    )


# ============================================================
# merge_workspaces — R 矩阵合并（两步确认，critical 级）
# ============================================================
def merge_workspaces_preview(
    sources: list[str],
    target: str = "global",
    models: list[str] | None = None,
) -> dict[str, Any]:
    """预览把多个源 R 矩阵合并到目标 R 的影响（不执行写操作）。

    合并语义：对每个源的每个模型（或 models 指定的子集），把该列全部观测
    upsert 到目标 R（同 record+model 覆盖，不同 record 累加）。

    source/target 描述符格式：
      "global"    → output/state/results.json（全局 R 矩阵）
      "ws:<name>" → output/workspaces/<name>/results.json（某工作区的 R）
      其他        → 视为目录路径，取其下 results.json

    典型场景：fork 工作区跑完实验后，把 ws:xxx 合并回 global。

    Args:
        sources: 源 R 矩阵描述符列表（如 ["ws:exp1", "ws:exp2"]）。
        target:  目标 R 矩阵描述符（默认 "global"）。
        models:  只合并指定模型（None = 全部模型）。

    Returns:
        {action, summary, confirm_token, ttl_seconds}。
    """
    from llmsec.management.merge import plan_merge

    def _do() -> dict[str, Any]:
        # source/target 走入口校验，防 LLM 传外部路径穿越
        for src in sources:
            _validate_merge_spec(src, is_target=False)
        _validate_merge_spec(target, is_target=True)
        plan = plan_merge(sources, target, models=models)
        plan_dict = plan.to_dict()
        token = confirm_mod.issue(
            action="merge_workspaces",
            summary=plan_dict,
            execute_fn=lambda: _execute_merge(sources, target, models),
            args_repr=f"sources={sources}, target={target}, models={models}",
        )
        total_new = plan.extra.get("total_new", 0)
        return {
            "action": "merge_workspaces",
            "summary": plan_dict,
            "impact_note": f"将向 {target} 合并 {total_new} 条新观测（来自 {sources}）",
            "confirm_token": token,
            "ttl_seconds": 300,
            "next_step": "审阅后调用 merge_workspaces_confirm(token) 执行合并",
        }

    return _try(_do, error_hint="源/目标描述符无效，或 results.json 不存在")


def _execute_merge(sources: list[str], target: str, models: list[str] | None) -> dict[str, Any]:
    from llmsec.management.merge import execute_merge

    done = execute_merge(sources, target, models=models)
    return done.to_dict()


def merge_workspaces_confirm(token: str) -> dict[str, Any]:
    """用 confirm_token 执行已预览的 R 矩阵合并操作。

    Args:
        token: merge_workspaces_preview 返回的 confirm_token。

    Returns:
        {status, result}。
    """
    return confirm_mod.confirm(token)


# ============================================================
# merge_env_snapshot_to_global — 快照写回全局 .env（两步确认，critical）
# ============================================================
def merge_env_snapshot_to_global_preview(name: str) -> dict[str, Any]:
    """预览把 env 快照写回全局 .env 的影响（不执行写操作）。

    语义：快照里有的 key 覆盖全局同名 key；快照里没有的不动。
    全局 .env 会先备份到 .env.bak.<timestamp>。

    Args:
        name: 快照名。

    Returns:
        {action, will_change_keys, confirm_token, ttl_seconds}。
    """
    from control.core.env_snapshot import _read_global_env, load_env_dict

    def _do() -> dict[str, Any]:
        snap_keys = load_env_dict(name)
        global_keys = _read_global_env()
        will_change = [k for k, v in snap_keys.items() if global_keys.get(k) != v]
        token = confirm_mod.issue(
            action="merge_env_to_global",
            summary={"snapshot": name, "will_change_keys": will_change},
            execute_fn=lambda: _execute_merge_env(name),
            args_repr=f"name={name}",
        )
        return {
            "action": "merge_env_to_global",
            "snapshot": name,
            "will_change_keys": will_change,
            "total_keys_in_snapshot": len(snap_keys),
            "confirm_token": token,
            "ttl_seconds": 300,
            "next_step": "审阅后调用 merge_env_snapshot_to_global_confirm(token) 执行",
        }

    return _try(_do, error_hint=f"快照 '{name}' 不存在")


def _execute_merge_env(name: str) -> dict[str, Any]:
    from control.core.env_snapshot import merge_to_global

    return merge_to_global(name)


def merge_env_snapshot_to_global_confirm(token: str) -> dict[str, Any]:
    """用 confirm_token 执行快照写回全局 .env。

    Args:
        token: merge_env_snapshot_to_global_preview 返回的 confirm_token。

    Returns:
        {status, result}。
    """
    return confirm_mod.confirm(token)


# ============================================================
# delete_workspace — 低风险，直接执行
# ============================================================
def delete_workspace(name: str) -> dict[str, Any]:
    """删除一个 fork 工作区（仅删隔离副本，不影响全局 R 矩阵）。

    Args:
        name: 要删除的工作区名。

    Returns:
        {deleted: name} 或错误信息。
    """
    from control.core.workspace import delete_workspace as _dw

    return _try(
        lambda: _dw(name),
        error_hint=f"工作区 '{name}' 不存在",
    )


# ============================================================
# gc_merged_workspaces — 低风险，直接执行
# ============================================================
def gc_merged_workspaces(older_than_days: int = 7) -> dict[str, Any]:
    """清理已 merge 且超期的工作区目录，释放空间（延迟 GC）。

    merge 后的工作区不会立即删除（orchestrator 的对比、历史记录仍可能引用其目录），
    而是按 merged_at 时间戳延迟清理。被清理的工作区合并去向记入审计日志，不丢失。

    Args:
        older_than_days: merged_at 距今超过该天数才清理（默认 7 天）。

    Returns:
        {cleaned: [{name, size}], skipped_fresh: N, gc_log_size: N}。
    """
    from control.core.workspace import gc_merged_workspaces as _gc

    return _try(
        lambda: _gc(older_than_days=older_than_days),
        error_hint="工作区索引读取失败",
    )


# ============================================================
# 注册
# ============================================================
def register(mcp: Any) -> None:
    """把本模块所有工具注册到 FastMCP server。"""
    mcp.tool(delete_runs_preview)
    mcp.tool(delete_runs_confirm)
    mcp.tool(clean_caches_preview)
    mcp.tool(clean_caches_confirm)
    mcp.tool(fork_workspace)
    mcp.tool(export_snapshot)
    mcp.tool(create_env_snapshot)
    mcp.tool(edit_env_snapshot)
    mcp.tool(list_env_snapshots)
    mcp.tool(get_env_config)
    mcp.tool(delete_env_snapshot)
    mcp.tool(merge_workspaces_preview)
    mcp.tool(merge_workspaces_confirm)
    mcp.tool(merge_env_snapshot_to_global_preview)
    mcp.tool(merge_env_snapshot_to_global_confirm)
    mcp.tool(delete_workspace)
    mcp.tool(gc_merged_workspaces)
