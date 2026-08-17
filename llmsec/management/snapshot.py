"""management.snapshot — 导出 R 库快照（人工备份 / 分发用）。

P3（json 链清除）后：
  - 输出是 **results.db 的整库副本**（sqlite3 backup API，WAL 安全）+ manifest.json
    ——控制层 fork 已不走本模块（workspace.fork 经薄契约直调 rstore backup/clone），
    快照只剩人工备份/分发用途。
  - run:<name> 源从 state.json 重建（rstore.clone_from_run）。
  - 人读 JSON 导出用 ``storage backup-r`` 之外的 ``rstore.export_legacy_json``（按需调用）。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from llmsec.core.config import OUTPUT_DIR
from llmsec.core.io import write_json
from llmsec.core.logging import get_logger
from llmsec.management.common import emit, print_table
from llmsec.storage import rstore

logger = get_logger(__name__)

SNAPSHOT_DIR = OUTPUT_DIR / "snapshots"


def export_snapshot(
    source: str = "global",
    *,
    out: Path | None = None,
) -> dict:
    """导出 R 库快照。返回快照元信息 dict（供 --json 输出）。

    Args:
        source: "global" 或 "run:<name>"
        out: 输出目录；None 则默认到 output/snapshots/<ts>/
    """
    if out is None:
        out_dir = SNAPSHOT_DIR / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    else:
        out = Path(out)
        # 相对路径统一锚到 OUTPUT_DIR：校验与写盘必须同锚点（M 修复保留）。
        if not out.is_absolute():
            out = OUTPUT_DIR / out
        # out 外部可控（MCP/CLI 传入），约束在 OUTPUT_DIR 子树内防穿越写出
        out_r = out.resolve()
        out_root = OUTPUT_DIR.resolve()
        if out_r != out_root and out_root not in out_r.parents:
            raise ValueError(f"输出路径越界，须在 output/ 内: {out}")
        out_dir = out
    out_dir.mkdir(parents=True, exist_ok=True)

    dst = out_dir / "results.db"
    if source == "global":
        rstore.backup(dst)
        stats = rstore.results_stats(dst)
        source_desc = "global R (results.db)"
    elif source.startswith("run:"):
        stats = rstore.clone_from_run(source[4:], dst)
        source_desc = f"run:{source[4:]} (state.json 重建)"
    else:
        raise ValueError(f"未知 source: {source!r}（用 'global' 或 'run:<name>'）")

    manifest = {
        "source": source,
        "source_desc": source_desc,
        "exported_at": datetime.now().isoformat(),
        "results": {
            "path": str(dst.relative_to(OUTPUT_DIR)),
            **stats,
        },
    }
    write_json(out_dir / "manifest.json", manifest)

    info = {
        "snapshot": str(out_dir.relative_to(OUTPUT_DIR)),
        "source": source,
        "models": stats["models"],
        "records": stats["records"],
        "results_total": stats["observations"],
    }
    logger.info("快照已导出: %s（来源 %s，%d 模型 %d 记录）",
                info["snapshot"], source, len(info["models"]), info["records"])
    return info


# ============================================================
# export 子命令
# ============================================================
def cmd_export(
    source: str = "global",
    *,
    out: str | None = None,
    json_mode: bool = False,
) -> int:
    try:
        info = export_snapshot(source, out=Path(out) if out else None)
    except (ValueError, FileNotFoundError) as e:
        logger.error("导出失败: %s", e)
        return 1
    if json_mode:
        emit(info, json_mode=True, title="snapshot export")
    else:
        rows = [
            ["snapshot", info["snapshot"]],
            ["source", info["source"]],
            ["models", ", ".join(info["models"]) or "(无)"],
            ["records", str(info["records"])],
            ["results_total", str(info["results_total"])],
        ]
        print_table(rows, headers=["field", "value"])
        print("\n控制层 fork 已直调库级 clone（workspace.fork），本命令用于人工备份/分发。")
    return 0
