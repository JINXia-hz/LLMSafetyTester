"""control.core.workspace — fork 编排（控制层核心能力）。

fork = 「以某个状态为起点，起一个新的隔离 llmsec 工作单元」。

流程（P3 库级 clone + P5 表化索引）：
  1. 经薄契约 rstore.backup / clone_from_run → workspaces/<name>/catalog.db
  2. （可选）run_runner(work_dir=<该 workspace>) → 起一个隔离工作单元
  3. work-dir 模式下 runner 把库路径重绑到该目录，全局零污染

索引在目录库 ctl_workspaces 表（P5：原 _index.json + AtomicIndexStore 退役）；
"""

from __future__ import annotations

import shutil
from datetime import datetime, timedelta

from control.config import LLMSEC_REPO, WORKSPACES_DIR, ensure_workspaces_dir
from control.core.invoker import run_runner
from control.core.paths import safe_component
from control.core.storage import (
    delete_workspace_row,
    get_workspace,
    save_workspace,
)
from control.core.storage import (
    list_workspaces as _list_rows,
)


# ============================================================
# fork
# ============================================================
def fork(
    name: str,
    *,
    source: str = "global",
    note: str = "",
) -> dict:
    """fork 一个新工作区：库级 clone 全局 R（或从 run 重建）到 workspaces/<name>/。

    Args:
        name: 工作区名（唯一）
        source: "global" 或 "run:<run_name>"
        note: 备注（记入索引行）

    Returns:
        工作区信息 dict（name/path/source/created/models/records）
    """
    ensure_workspaces_dir()
    ws_dir = safe_component(WORKSPACES_DIR, name)
    if ws_dir.exists():
        raise FileExistsError(f"工作区已存在: {name}（{ws_dir}）")

    # 库级 clone（P3：db→json→子进程→复制→json→db 的六步握手删除——经薄契约
    # 直调 rstore.backup / clone_from_run，WAL 安全整库复制含 elo_cache 表）
    from control.core.storage import backup, clone_from_run, results_stats
    ws_dir.mkdir(parents=True)
    dst = ws_dir / "catalog.db"
    if source == "global":
        backup(dst)
        stats = results_stats(dst)
    elif source.startswith("run:"):
        stats = clone_from_run(source[4:], dst)
    else:
        raise ValueError(f"未知 source: {source!r}（用 'global' 或 'run:<name>'）")

    info = {
        "name": name,
        "path": str(ws_dir.relative_to(LLMSEC_REPO)).replace("\\", "/"),
        "source": source,
        "note": note,
        "created": datetime.now().isoformat(timespec="seconds"),
        "models": stats["models"],
        "records": stats["records"],
        "merged": False,            # 是否已 merge 回全局/其他目标（merge 后置 True）
        "merged_at": None,
        "merged_to": None,
    }
    save_workspace(info)
    return info


def fork_and_run(
    name: str,
    *,
    source: str = "global",
    target: str | None = None,
    input_file: str = "attacks/l1.jsonl",
    max_rounds: int = 5,
    seed: int | None = None,
    note: str = "",
    timeout: float | None = None,
    env_override: dict[str, str] | None = None,
) -> dict:
    """fork 后立即在该工作区起一个 llmsec run（隔离）。

    返回 {workspace, run}，run 含 returncode/elapsed。
    """
    info = fork(name, source=source, note=note)
    ws_dir = safe_component(WORKSPACES_DIR, name)
    log_file = ws_dir / "runner.log"

    res = run_runner(
        ws_dir, target=target, input_file=input_file,
        max_rounds=max_rounds, seed=seed, timeout=timeout, log_file=log_file,
        env_override=env_override,
    )
    return {
        "workspace": info,
        "run": {
            "returncode": res.returncode,
            "ok": res.ok,
            "elapsed_s": res.elapsed_s,
            "log": str(log_file.relative_to(LLMSEC_REPO)).replace("\\", "/"),
        },
    }


# ============================================================
# list / get / delete
# ============================================================
def list_workspaces() -> list[dict]:
    """列出所有工作区（按创建时间倒序，库行直查 + 补目录 size）。"""
    ws = _list_rows()
    for w in ws:
        d = LLMSEC_REPO / w.get("path", "")
        w["size"] = _dir_size(d) if w.get("path") and d.exists() else 0
    ws.sort(key=lambda x: x.get("created", ""), reverse=True)
    return ws


def mark_merged(name: str, target: str) -> bool:
    """标记工作区已 merge 到某目标（merge tool 执行后调）。

    容错：行更新失败不抛异常（merge 本身已成功不可逆，不应因索引更新失败而报错）。
    返回 True=成功更新，False=更新失败（调用方可提示用户但 merge 已生效）。
    """
    try:
        info = get_workspace(name)
        if info is None:
            return False
        info["merged"] = True
        info["merged_at"] = datetime.now().isoformat(timespec="seconds")
        info["merged_to"] = target
        save_workspace(info)
        return True
    except Exception:
        return False


def delete_workspace(name: str) -> dict:
    """删除工作区（目录 + 索引行）。不碰全局 R（工作区本就是隔离副本）。"""
    info = get_workspace(name)
    if info is None:
        raise KeyError(f"工作区不存在: {name}")
    ws_dir = safe_component(WORKSPACES_DIR, name)
    if ws_dir.exists():
        # Windows：merge/fork 可能打开过该库的引擎句柄，持句柄 rmtree 会 500——先释放
        try:
            from control.core.storage import close_db
            close_db(ws_dir / "catalog.db")
        except Exception:
            pass
        shutil.rmtree(ws_dir)
    delete_workspace_row(name)
    return {"deleted": name, "info": info}


def gc_merged_workspaces(older_than_days: int = 7) -> dict:
    """清理已 merge 且超期的 workspace 目录，释放空间。

    延迟 GC 设计（非 merge 后立即删）：
      orchestrator 的 compare_after、discover_workspace_runs、gazette 历史记录在
      merge 之后仍可能引用 ws 目录（compare 直接 iterdir 不读索引的 merged 标记），
      立即删会造成悬空引用。故 mark_merged 只标记，物理清理由此入口按 merged_at 超期延迟执行。

    Args:
        older_than_days: merged_at 距今超过该天数的 workspace 才清理。默认 7 天。

    Returns:
        {cleaned: [...], skipped_fresh: N, older_than_days: N}
    """
    cutoff = datetime.now() - timedelta(days=older_than_days)
    cleaned = []
    skipped_fresh = 0
    for info in _list_rows():
        name = info["name"]
        if not info.get("merged") or not info.get("merged_at"):
            continue
        try:
            merged_at = datetime.fromisoformat(info["merged_at"])
        except (ValueError, TypeError):
            continue
        if merged_at > cutoff:
            skipped_fresh += 1
            continue
        # 超期：物理删除目录（name 来自库行，但仍走校验防被污染的行穿越）
        try:
            ws_dir = safe_component(WORKSPACES_DIR, name)
        except ValueError:
            continue
        size = _dir_size(ws_dir) if ws_dir.exists() else 0
        if ws_dir.exists():
            try:
                from control.core.storage import close_db
                close_db(ws_dir / "catalog.db")
            except Exception:
                pass
            shutil.rmtree(ws_dir, ignore_errors=True)
        # P9：gc 审计链（__gc_log__ 哨兵行）删除——只被 len() 消费的写入型
        # 遥测；合并去向仍可从 ctl_events 的 merge 事件追溯
        delete_workspace_row(name)
        cleaned.append({"name": name, "size": size})
    return {
        "cleaned": cleaned,
        "skipped_fresh": skipped_fresh,
        "older_than_days": older_than_days,
    }


# r7：与 llmsec/management/common.dir_size 的重复实现收敛为单源导入
# （control 依赖 llmsec 共享层的先例见 control/core/paths.py）
from llmsec.management.common import dir_size as _dir_size  # noqa: E402
