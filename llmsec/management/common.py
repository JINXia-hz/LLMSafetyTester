"""management.common — 共用原语：目录大小、软删除、dry-run 结果、表格/JSON 输出。

这些原语被 runs.py / caches.py / snapshot.py 复用，保证四条机器友好契约：
  - ``dir_size`` 递归计算目录占用（填补代码库空白，_discover_runs 只取 mtime）。
  - ``soft_remove`` / ``soft_rmtree`` 软删除到 output/.trash/，保留相对结构可恢复。
  - ``Plan`` / ``PlanItem`` 统一 dry-run 预览数据结构，可序列化为 JSON 供 agent 解析。
  - ``print_table`` / ``emit`` 人/机双输出。
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from llmsec.core.config import OUTPUT_DIR
from llmsec.core.logging import get_logger

logger = get_logger(__name__)

# 软删除回收站根目录：output/.trash/<时间戳>/...
TRASH_DIR = OUTPUT_DIR / ".trash"


# ============================================================
# 目录大小
# ============================================================
def dir_size(path: Path) -> int:
    """递归计算目录占用字节数（符号链接不计）。文件不存在返回 0。"""
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file() and not p.is_symlink():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def fmt_size(n: float) -> str:
    """字节数 → 人类可读（KB/MB/GB）。"""
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    if n < 1024 ** 3:
        return f"{n / 1024 ** 2:.1f} MB"
    return f"{n / 1024 ** 3:.2f} GB"


# ============================================================
# 软删除
# ============================================================
def _trash_subdir() -> Path:
    """本次操作的回收站子目录（按时间戳隔离，避免跨操作混淆）。"""
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    d = TRASH_DIR / ts
    d.mkdir(parents=True, exist_ok=True)
    return d


def _soft_move(path: Path, kind: str) -> Path | None:
    """软删除底层实现：把文件或目录移到 trash，保留相对 OUTPUT_DIR 的结构。

    kind 仅用于日志文案（"文件"/"目录"）。shutil.move 对文件和目录都适用。
    """
    path = Path(path)
    if not path.exists() and not path.is_symlink():
        return None
    trash = _trash_subdir()
    try:
        rel = path.relative_to(OUTPUT_DIR)
    except ValueError:
        rel = Path(path.name)
    dest = trash / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest = dest.with_name(f"{dest.name}.{datetime.now().strftime('%H%M%S_%f')}")
    shutil.move(str(path), str(dest))
    logger.info("软删除%s %s → %s", kind, path, dest)
    return dest


def soft_remove(path: Path) -> Path | None:
    """软删除单个文件 → 移到 trash，返回 trash 内的新路径。

    保留相对 OUTPUT_DIR 的结构以便恢复。文件不存在返回 None。
    """
    return _soft_move(path, "文件")


def soft_rmtree(path: Path) -> Path | None:
    """软删除整个目录 → 移到 trash，返回 trash 内的新路径。

    目录不存在返回 None。比逐文件 soft_remove 快（整体 move）。
    """
    return _soft_move(path, "目录")


# ============================================================
# dry-run 预览数据结构
# ============================================================
@dataclass
class PlanItem:
    """单个将被处理的文件/目录条目。"""
    path: str               # 原始路径（相对 OUTPUT_DIR 或绝对）
    size: int = 0           # 占用字节
    kind: str = ""          # run_dir / file / cache / ...
    detail: str = ""        # 人可读补充说明


@dataclass
class Plan:
    """一次写操作的 dry-run 预览。可序列化为 JSON 供 agent 解析。"""
    action: str                                   # delete / clean / export
    items: list[PlanItem] = field(default_factory=list)
    total_size: int = 0                           # 将释放的总字节
    extra: dict = field(default_factory=dict)     # 额外结构化信息（如 r_rows_affected）
    dry_run: bool = True

    def add(self, path: Path | str, *, size: int = 0, kind: str = "", detail: str = "") -> None:
        try:
            rel = str(Path(path).relative_to(OUTPUT_DIR)).replace("\\", "/")
        except ValueError:
            rel = str(path)
        self.items.append(PlanItem(path=rel, size=size, kind=kind, detail=detail))
        self.total_size += size

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total_size_human"] = fmt_size(self.total_size)
        return d


# ============================================================
# 输出：人/机双模式
# ============================================================
def emit(obj: dict | list, *, json_mode: bool, title: str = "") -> None:
    """统一输出：json_mode 输出结构化 JSON（供 agent 解析）；否则什么都不做
    （表格由调用方自行 print）。本函数仅负责 JSON 模式。
    """
    if json_mode:
        if title:
            obj = {"_title": title, **obj} if isinstance(obj, dict) else {"_title": title, "items": obj}
        print(json.dumps(obj, ensure_ascii=False, indent=2))


def print_table(rows: list[list[str]], headers: list[str], *, widths: list[int] | None = None) -> None:
    """简易对齐表格打印（人可读）。rows/homogeneous 列数 = len(headers)。"""
    if not rows:
        print("（无）")
        return
    if widths is None:
        widths = []
        for i, h in enumerate(headers):
            w = len(h)
            for r in rows:
                if i < len(r):
                    w = max(w, len(r[i]))
            widths.append(w)
    # 表头
    header_line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for r in rows:
        cells = [r[i].ljust(widths[i]) if i < len(r) else "" for i in range(len(headers))]
        print("  ".join(cells))
