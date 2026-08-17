"""management.snapshot — 导出快照（控制层 fork 的握手点）。

快照 = 一个自包含的 R 矩阵副本，控制层复制它到新 work-dir 后
``llmsec ... --work-dir <new>`` 即可零污染全局 R，实现 fork。

来源：
  global       当前全局 R（results.db；导出为 results.json 快照格式）
  run:<name>   从指定历史 run 的 state.json 重建一份 R（无 state.json 则报错）

本模块只「导出文件」，不做 fork 决策（那是控制层职责）。

输出格式：
  --out 指定目录 → 写入 <out>/results.json + manifest.json
  --out 指定 .tar.gz → 打包上述文件
  不指定 → 默认 output/snapshots/<时间戳>/
"""

from __future__ import annotations

import tarfile
from datetime import datetime
from pathlib import Path

from llmsec.core.config import OUTPUT_DIR, RUNS_DIR
from llmsec.core.io import read_json, write_json
from llmsec.core.logging import get_logger
from llmsec.core.paths import safe_subpath
from llmsec.core.results import ResultsMatrix
from llmsec.management.common import emit, print_table

logger = get_logger(__name__)

SNAPSHOT_DIR = OUTPUT_DIR / "snapshots"


def export_snapshot(
    source: str = "global",
    *,
    out: Path | None = None,
) -> dict:
    """导出快照。返回快照元信息 dict（供 --json 输出）。

    Args:
        source: "global" 或 "run:<name>"
        out: 输出目录或 .tar.gz 文件；None 则默认到 output/snapshots/<ts>/
    """
    # 解析来源，得到一个 ResultsMatrix
    if source == "global":
        R = ResultsMatrix.load()
        source_desc = "global R (results.db)"
    elif source.startswith("run:"):
        run_name = source[4:]
        R = _R_from_run(run_name)
        source_desc = f"run:{run_name} (state.json 重建)"
    else:
        raise ValueError(f"未知 source: {source!r}（用 'global' 或 'run:<name>'）")

    # 确定输出目标
    if out is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out_dir = SNAPSHOT_DIR / ts
        archive = None
    else:
        out = Path(out)
        # 相对路径统一锚到 OUTPUT_DIR：校验与写盘必须同锚点。原实现校验按
        # OUTPUT_DIR 解析、写盘却按 CWD 解析——相对 out 会先把文件写到 output/
        # 之外，再在 manifest 的 relative_to(OUTPUT_DIR) 处崩溃，残留文件逃出约束。
        if not out.is_absolute():
            out = OUTPUT_DIR / out
        # out 外部可控（MCP/CLI 传入），约束在 OUTPUT_DIR 子树内防穿越写出
        out_r = out.resolve()
        out_root = OUTPUT_DIR.resolve()
        if out_r != out_root and out_root not in out_r.parents:
            raise ValueError(f"输出路径越界，须在 output/ 内: {out}")
        if out.suffix == ".gz" and ".tar" in out.name:
            out_dir = OUTPUT_DIR / ".snapshot_staging" / out.stem
            archive = out
        else:
            out_dir = out
            archive = None
    out_dir.mkdir(parents=True, exist_ok=True)

    # 写 R
    results_path = out_dir / "results.json"
    R.save(results_path)

    # 写 manifest（来源/时间/规模，供 agent 解析）
    manifest = {
        "source": source,
        "source_desc": source_desc,
        "exported_at": datetime.now().isoformat(),
        "results": {
            "path": str(results_path.relative_to(OUTPUT_DIR)),
            "models": R.all_models(),
            "records": len(R._r),
            "results_total": sum(len(col) for col in R._r.values()),
        },
    }
    write_json(out_dir / "manifest.json", manifest)

    # 打包（若 out 是 .tar.gz）
    if archive:
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(out_dir, arcname=archive.stem)
        # 清理 staging
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)
        result_path = str(archive.relative_to(OUTPUT_DIR)) if _rel(archive) else str(archive)
    else:
        result_path = str(out_dir.relative_to(OUTPUT_DIR))

    info = {
        "snapshot": result_path,
        "source": source,
        "models": R.all_models(),
        "records": len(R._r),
        "results_total": sum(len(col) for col in R._r.values()),
    }
    logger.info("快照已导出: %s（来源 %s，%d 模型 %d 记录）",
                result_path, source, len(info["models"]), info["records"])
    return info


def _R_from_run(run_name: str) -> ResultsMatrix:
    """从 run 的 state.json 重建一份 R。

    state.json 是 ELOTracker 快照（history 含每场对局），重建为 record→model→MatchResult。
    若 run 目录无 state.json，报错。
    """
    # run_name 外部可控（source[4:]），走 safe_subpath 逐段校验防穿越
    parts = run_name.split("/")
    run_dir = safe_subpath(RUNS_DIR, *parts)
    state_path = run_dir / "state.json"
    if not state_path.exists():
        # 旧布局
        state_path = safe_subpath(RUNS_DIR, parts[0]) / "state.json"
    if not state_path.exists():
        raise FileNotFoundError(
            f"run {run_name!r} 无 state.json，无法重建 R。"
            "（仅 global 源或含 state.json 的 run 可导出）"
        )
    state = read_json(state_path) or {}
    history = state.get("history", [])
    R = ResultsMatrix()
    # ELOTracker.save 不写 defender_name 键，模型名恒从 runner_report 取
    model = read_json(run_dir / "runner_report.json", {}).get("target_model") if run_dir.exists() else None
    for h in history:
        rec = h.get("record")
        def_ = h.get("defender") or model
        if not rec or not def_:
            continue
        R.upsert(
            record=rec, model=def_,
            eval_score=float(h.get("eval_score", 0.0)),
            status=h.get("status", ""),
            ts=h.get("round"),
            extra={"unit": h.get("unit"), "round": h.get("round")},
        )
    return R


def _rel(p: Path) -> bool:
    try:
        p.relative_to(OUTPUT_DIR)
        return True
    except ValueError:
        return False


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
        print("\n控制层 fork 用法：复制快照 results.json 到新 work-dir，"
              "再 ``llmsec ... --work-dir <new>``")
    return 0
