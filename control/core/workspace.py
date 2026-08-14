"""control.core.workspace — fork 编排（控制层核心能力）。

fork = 「以某个状态为起点，起一个新的隔离 llmsec 工作单元」。

流程（经 invoker，绝不 import llmsec 内部）：
  1. snapshot export --source <global|run:name>   → 拿到自包含快照（results.json）
  2. 复制快照 results.json → output/workspaces/<name>/results.json
  3. （可选）run_runner(work_dir=<该 workspace>)    → 起一个隔离工作单元
  4. work-dir 模式下 runner 把 state/results 重绑到该目录，全局零污染

workspaces 由控制层独立管理（output/workspaces/），维护 _index.json 记录来源/时间/备注。
切换/列出/删除都是纯文件操作。
"""

from __future__ import annotations

import shutil
from pathlib import Path

from control.config import LLMSEC_REPO, WORKSPACES_DIR, ensure_workspaces_dir
from control.core.invoker import export_snapshot, run_runner
from control.core.paths import safe_component
from control.core.store import AtomicIndexStore

# workspace 索引存储（原子读写 + Windows PermissionError 重试 + 并发锁）
# base_dir 传 lambda：测试期 monkeypatch WORKSPACES_DIR 后能动态生效
_store = AtomicIndexStore(lambda: WORKSPACES_DIR, "workspaces")


# ============================================================
# fork
# ============================================================
def fork(
    name: str,
    *,
    source: str = "global",
    note: str = "",
) -> dict:
    """fork 一个新工作区：导出快照 → 复制到 workspaces/<name>/。

    Args:
        name: 工作区名（唯一）
        source: "global" 或 "run:<run_name>"
        note: 备注（记入索引）

    Returns:
        工作区信息 dict（name/path/source/created/models/records）
    """
    ensure_workspaces_dir()
    ws_dir = safe_component(WORKSPACES_DIR, name)
    if ws_dir.exists():
        raise FileExistsError(f"工作区已存在: {name}（{ws_dir}）")

    # 1. 导出快照（控制层调 llmsec-manage，不碰 R 内部）
    snap = export_snapshot(source=source)
    # snap["snapshot"] 是相对 OUTPUT_DIR 的路径（llmsec-manage 用 relative_to(OUTPUT_DIR) 存）
    from control.config import OUTPUT_DIR
    snap_path = OUTPUT_DIR / snap["snapshot"]
    results_src = snap_path / "results.json"
    if not results_src.exists():
        raise FileNotFoundError(f"快照缺 results.json: {results_src}")

    # 2. 复制到工作区
    ws_dir.mkdir(parents=True)
    shutil.copy2(results_src, ws_dir / "results.json")
    # elo_cache 可选复制（global 源才有）
    elo_src = snap_path / "elo_cache.json"
    if elo_src.exists():
        shutil.copy2(elo_src, ws_dir / "elo_cache.json")

    # 3. 清理临时快照目录（快照已内化进工作区）
    shutil.rmtree(snap_path, ignore_errors=True)

    info = {
        "name": name,
        "path": str(ws_dir.relative_to(LLMSEC_REPO)).replace("\\", "/"),
        "source": source,
        "note": note,
        "created": _store.now(),
        "models": snap.get("models", []),
        "records": snap.get("records", 0),
        "merged": False,            # 是否已 merge 回全局/其他目标（merge 后置 True）
        "merged_at": None,
        "merged_to": None,
    }

    # 4. 记入索引（加锁防并发 fork 丢更新）
    def _record(idx):
        idx["workspaces"][name] = info
        _store.save(idx)
    _store.update(_record)
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
    """列出所有工作区（按创建时间倒序）。"""
    idx = _store.load()
    ws = list(idx.get("workspaces", {}).values())
    # 补 size
    for w in ws:
        d = LLMSEC_REPO / w["path"]
        w["size"] = _dir_size(d) if d.exists() else 0
    ws.sort(key=lambda x: x.get("created", ""), reverse=True)
    return ws


def mark_merged(name: str, target: str) -> bool:
    """标记工作区已 merge 到某目标（merge tool 执行后调）。

    容错：_index.json 写失败不抛异常（merge 本身已成功不可逆，不应因索引更新失败而报错）。
    返回 True=成功更新索引，False=更新失败（调用方可提示用户但 merge 已生效）。
    """
    try:
        def _mark(idx):
            if name in idx.get("workspaces", {}):
                idx["workspaces"][name]["merged"] = True
                idx["workspaces"][name]["merged_at"] = _store.now()
                idx["workspaces"][name]["merged_to"] = target
                _store.save(idx)
        _store.update(_mark)
        return True
    except Exception:
        return False


def delete_workspace(name: str) -> dict:
    """删除工作区（目录 + 索引项）。不碰全局 R（工作区本就是隔离副本）。"""
    def _delete(idx):
        if name not in idx.get("workspaces", {}):
            raise KeyError(f"工作区不存在: {name}")
        ws_dir = safe_component(WORKSPACES_DIR, name)
        if ws_dir.exists():
            shutil.rmtree(ws_dir)
        info = idx["workspaces"].pop(name)
        _store.save(idx)
        return info
    info = _store.update(_delete)
    return {"deleted": name, "info": info}


def gc_merged_workspaces(older_than_days: int = 7) -> dict:
    """清理已 merge 且超期的 workspace 目录，释放空间。

    延迟 GC 设计（非 merge 后立即删）：
      orchestrator 的 compare_after、discover_workspace_runs、gazette 历史记录在
      merge 之后仍可能引用 ws 目录（compare.py:120 直接 iterdir 不读索引的 merged 标记），
      立即删会造成悬空引用。故 mark_merged 只标记，物理清理由此入口按 merged_at 超期延迟执行。

    审计链保留：被 GC 的 workspace 信息（含 merged_to）记入 _index.json 的 "gc_log"，
    供事后追溯，不因物理删除而丢失合并去向。

    Args:
        older_than_days: merged_at 距今超过该天数的 workspace 才清理。默认 7 天。

    Returns:
        {cleaned: [...], skipped_fresh: N, gc_log_size: N}
    """
    from datetime import datetime, timedelta

    cutoff = datetime.now() - timedelta(days=older_than_days)

    def _gc(idx):
        ws_map = idx.get("workspaces", {})
        gc_log = idx.setdefault("gc_log", [])
        cleaned = []
        skipped_fresh = 0
        for name, info in list(ws_map.items()):
            if not info.get("merged"):
                continue
            merged_at_str = info.get("merged_at")
            if not merged_at_str:
                continue
            try:
                merged_at = datetime.fromisoformat(merged_at_str)
            except (ValueError, TypeError):
                continue
            if merged_at > cutoff:
                skipped_fresh += 1
                continue
            # 超期：物理删除目录（name 来自索引，但仍走校验防被污染的索引项穿越）
            try:
                ws_dir = safe_component(WORKSPACES_DIR, name)
            except ValueError:
                continue  # 索引项名称非法，跳过不删
            size = _dir_size(ws_dir) if ws_dir.exists() else 0
            if ws_dir.exists():
                shutil.rmtree(ws_dir, ignore_errors=True)
            # 审计：保留精简记录（含 merged_to 合并去向），从主索引移除
            gc_log.append({
                "name": name,
                "merged_at": merged_at_str,
                "merged_to": info.get("merged_to"),
                "source": info.get("source"),
                "gc_at": _store.now(),
                "size": size,
            })
            del ws_map[name]
            cleaned.append({"name": name, "size": size})
        if cleaned:
            _store.save(idx)
        return cleaned, skipped_fresh, len(gc_log)

    cleaned, skipped_fresh, gc_log_size = _store.update(_gc)
    return {
        "cleaned": cleaned,
        "skipped_fresh": skipped_fresh,
        "gc_log_size": gc_log_size,
        "older_than_days": older_than_days,
    }


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink())
