"""
core.io — 统一文件 I/O 工具层

所有业务模块的文件读写都应经此模块，禁止散落地 open()+json.dump/load。

提供：
  - JSONL：read_jsonl / iter_jsonl / write_jsonl / append_jsonl / load_done_ids
  - JSON （单对象）：read_json / write_json
  - 二进制 artifacts（joblib/pickle）：load_artifact / save_artifact
  - CSV：read_csv / write_csv

所有文本操作统一 utf-8 编码；写操作自动创建父目录。

数据完整性约定：
  - 权威存储（results.json / state.json）应使用 read_json(strict=True) +
    write_json(atomic=True, backup=True)，避免崩溃/并发导致静默数据丢失。
  - 可丢弃缓存（elo_cache / feature_cache）用默认 strict=False 即可。
"""

import csv
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


class CorruptedFileError(Exception):
    """权威存储文件存在但解析失败（损坏/半写），不应静默吞掉。

    由 read_json(strict=True) 抛出。调用方应：备份残文件 + 警告 + 决定是否重置。
    """

    def __init__(self, path, cause):
        self.path = str(path)
        self.cause = cause
        super().__init__(f"File corrupted: {self.path} ({cause})")


def read_jsonl(path) -> list[dict]:
    """读取整个 JSONL 文件为 dict 列表。文件不存在返回空列表；坏行跳过。"""
    return list(iter_jsonl(path))


def iter_jsonl(path) -> Iterator[dict]:
    """逐行迭代 JSONL，坏行（JSON 解析失败）跳过并记 warning。文件不存在时不产出任何行。"""
    path = Path(path)
    if not path.exists():
        return
    bad_count = 0
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                bad_count += 1
                continue
    if bad_count:
        logger.warning("跳过 %s 中的 %d 行坏 JSONL（解析失败）", path, bad_count)


def write_jsonl(path, rows) -> None:
    """整体覆写 JSONL 文件（自动创建父目录）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path, row: dict) -> None:
    """追加一行 JSONL（自动创建父目录）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_done_ids(path, key: str = "id") -> set:
    """
    断点续传：读取已有结果文件，提取已完成记录的 key 字段集合。
    文件不存在或行解析失败均不报错，返回已收集到的集合。
    """
    done = set()
    for row in iter_jsonl(path):
        if key in row:
            done.add(row[key])
    return done


# ============================================================
# 单对象 JSON
# ============================================================
def read_json(path, default=None, *, strict: bool = False):
    """读取单个 JSON 对象。

    - 文件不存在 → 返回 default（无论 strict）
    - 解析失败：
      - strict=False（默认，向后兼容；适用于可丢弃缓存）：返回 default
      - strict=True（权威存储用）：抛 CorruptedFileError，让调用方备份+告警
    """
    path = Path(path)
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        if strict:
            raise CorruptedFileError(path, e) from e
        logger.warning("读取 %s 失败（静默返回 default）: %s", path, e)
        return default


def write_json(path, obj, indent: int = 2, *, atomic: bool = True, backup: bool = False) -> None:
    """写入单个 JSON 对象（自动创建父目录，ensure_ascii=False）。

    atomic=True（默认）：写 <path>.tmp → flush+fsync → os.replace。
        os.replace 在 Windows/Linux 上均为原子操作，崩溃中途不会留下半截文件。
    backup=True：写前把现有文件复制为 <path>.bak（权威存储的最后一道兜底）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if backup and path.exists():
        bak = path.with_suffix(path.suffix + ".bak")
        try:
            shutil.copy2(path, bak)
        except OSError as e:
            logger.warning("备份 %s -> %s 失败: %s", path, bak, e)
    if atomic:
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=indent)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except OSError:
            # 清理残留 tmp（os.replace 失败时）
            try:
                tmp.unlink()
            except OSError:
                pass
            raise
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=indent)


# ============================================================
# 二进制 artifacts（joblib）
# ============================================================
def load_artifact(path, default=None, *, strict: bool = False):
    """joblib.load 一个 artifact。

    - 文件不存在 → 返回 default
    - 加载失败：strict=False 返回 default；strict=True 抛 CorruptedFileError
    """
    path = Path(path)
    if not path.exists():
        return default
    try:
        import joblib
        return joblib.load(path)
    except Exception as e:
        if strict:
            raise CorruptedFileError(path, e) from e
        logger.warning("加载 artifact %s 失败（静默返回 default）: %s", path, e)
        return default


def save_artifact(path, obj, *, atomic: bool = True, backup: bool = False) -> None:
    """joblib.dump 一个 artifact（自动创建父目录）。

    atomic=True（默认）：写 .tmp → os.replace（崩溃安全）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    if backup and path.exists():
        bak = path.with_suffix(path.suffix + ".bak")
        try:
            shutil.copy2(path, bak)
        except OSError as e:
            logger.warning("备份 %s -> %s 失败: %s", path, bak, e)
    if atomic:
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            joblib.dump(obj, tmp)
            os.replace(tmp, path)
        except OSError:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise
    else:
        joblib.dump(obj, path)


# ============================================================
# CSV
# ============================================================
def read_csv(path) -> list[dict]:
    """读取整个 CSV 为 dict 行列表（utf-8）。文件不存在返回空列表。"""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows: list[dict]) -> None:
    """写入 CSV（首行取字段名，自动创建父目录）。空 rows 不写表头以外的内容。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # 仍创建空文件
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
