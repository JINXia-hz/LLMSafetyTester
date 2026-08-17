"""
core.io — 统一文件 I/O 工具层

所有业务模块的文件读写都应经此模块，禁止散落地 open()+json.dump/load。

提供：
  - JSONL：read_jsonl / iter_jsonl / write_jsonl / append_jsonl / load_done_ids
  - JSON （单对象）：read_json / write_json
  - 二进制 artifacts（joblib/pickle）：load_artifact / save_artifact
  - CSV：write_csv

所有文本操作统一 utf-8 编码；写操作自动创建父目录。

数据完整性约定：
  - 权威存储（results.json / state.json）应使用 read_json(strict=True) +
    write_json(atomic=True, backup=True)，避免崩溃/并发导致静默数据丢失。
  - 可丢弃缓存（elo_cache / feature_cache）用默认 strict=False 即可。
"""

import csv
import json
import os
import shutil
import threading
import time
from collections.abc import Iterator
from pathlib import Path

from llmsec.core.logging import get_logger


def _replace_with_retry(tmp: Path, path: Path, attempts: int = 8) -> None:
    """os.replace + Windows 瞬时占用重试（write_jsonl / write_json / save_artifact 共用）。

    Windows 上并发 replace 同一目标会抛 PermissionError（WinError 5，目标被另一
    线程/进程的 replace 或杀软/索引器瞬时占用）。指数退避 20ms→640ms 上限，
    8 次尝试共 7 次睡眠（总预算 ~1.9s）——实测高负载并发套件下固定 5×50ms
    （250ms）仍可能耗尽。非瞬时原因重试耗尽后照常抛出，由调用方清理 tmp。
    """
    delay = 0.02
    for attempt in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.64)

logger = get_logger(__name__)


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
    with open(path, encoding="utf-8") as f:
        for _lineno, line in enumerate(f, 1):
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
    """整体覆写 JSONL 文件（自动创建父目录）。

    原子写（仿 write_json）：写 <path>.tmp.<pid>.<tid> → flush+fsync → os.replace，
    中断不会留下截断的 JSONL。tmp 名带 pid/tid 后缀：同进程两线程并发覆写同一
    文件时固定名会互踩（save_artifact 的 P9 同类修复）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        _replace_with_retry(tmp, path)
    except OSError:
        # 清理残留 tmp（os.replace 失败时）
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


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
    - 解析失败（JSONDecodeError，内容损坏/半写）：
      - strict=False（默认，向后兼容；适用于可丢弃缓存）：返回 default
      - strict=True（权威存储用）：抛 CorruptedFileError，让调用方备份+告警
    - PermissionError（Windows 瞬时占用：杀软/索引器/并发 replace）：
      指数退避重试；耗尽后 strict 上抛、非 strict 返回 default
    - 其他 OSError（磁盘/权限模式等）：**不是文件损坏**——strict 上抛原异常
      （调用方不得把完好文件备份成 .corrupt.bak 回退旧数据），非 strict 返回 default
    """
    path = Path(path)
    if not path.exists():
        return default
    delay = 0.02
    for attempt in range(6):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            if strict:
                raise CorruptedFileError(path, e) from e
            logger.warning("读取 %s 解析失败（静默返回 default）: %s", path, e)
            return default
        except PermissionError as e:
            if attempt == 5:
                if strict:
                    raise
                logger.warning("读取 %s 失败（静默返回 default）: %s", path, e)
                return default
            time.sleep(delay)
            delay = min(delay * 2, 0.32)
        except OSError as e:
            if strict:
                raise
            logger.warning("读取 %s 失败（静默返回 default）: %s", path, e)
            return default
    return default  # 不可达（循环必 return/raise），为类型检查器保留


def _json_numpy_default(obj):
    """json.dump 的 default 处理器：numpy 标量/数组 → Python 原生类型（M12）。

    防止任何路径漏转 numpy 类型（如 np.float64 混进 ratings dict）时 json.dump 抛
    TypeError；对权威存储 results.json 尤为关键。
    """
    import numpy as _np
    if isinstance(obj, _np.integer):
        return int(obj)
    if isinstance(obj, _np.floating):
        return float(obj)
    if isinstance(obj, _np.bool_):
        return bool(obj)
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(
    path,
    obj,
    indent: int = 2,
    *,
    atomic: bool = True,
    backup: bool = False,
    allow_nan: bool = True,
) -> None:
    """写入单个 JSON 对象（自动创建父目录，ensure_ascii=False）。

    atomic=True（默认）：写 <path>.tmp → flush+fsync → os.replace。
        os.replace 在 Windows/Linux 上均为原子操作，崩溃中途不会留下半截文件。
    backup=True：写前把现有文件复制为 <path>.bak（权威存储的最后一道兜底）。
    allow_nan=False（M12）：权威存储应设 False——NaN/Infinity 会写出非法 JSON 字面量，
        Python json.load 能读但浏览器 JSON.parse 报 SyntaxError。设 False 时遇 NaN 直接抛错。
    default=_json_numpy_default（M12）：numpy 标量/数组自动转原生，防漏转 TypeError。
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
        # tmp 名带 pid/tid 后缀（对齐 write_jsonl/save_artifact）：同进程两线程
        # 并发写同一文件（如 MCP 线程池并发触发的 elo_cache 写）固定名互踩。
        # with_name 而非 with_suffix：兼容 ".env" 这类空后缀点文件
        tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=indent,
                          allow_nan=allow_nan, default=_json_numpy_default)
                f.flush()
                os.fsync(f.fileno())
            _replace_with_retry(tmp, path)
        except Exception:
            # 清理残留 tmp（os.replace 失败或 json.dump 序列化错误时）
            try:
                tmp.unlink()
            except OSError:
                pass
            raise
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=indent,
                      allow_nan=allow_nan, default=_json_numpy_default)


# ============================================================
# 二进制 artifacts（joblib）
# ============================================================
def load_artifact(path, default=None, *, strict: bool = False):
    """joblib.load 一个 artifact。

    - 文件不存在 → 返回 default
    - OSError（权限/占用/磁盘等瞬时 IO 错误）：**不是损坏**——strict 上抛原异常、
      非 strict 返回 default（r8/病根4：与 read_json 的 M-4 分类口径对齐，
      IO 错误不得伪装成 CorruptedFileError 触发"损坏"处置路径）
    - 其他异常（UnpicklingError/EOFError 等内容损坏）：strict 抛 CorruptedFileError；
      非 strict 返回 default
    """
    path = Path(path)
    if not path.exists():
        return default
    try:
        import joblib
        return joblib.load(path)
    except OSError as e:
        if strict:
            raise
        logger.warning("加载 artifact %s 失败（IO 错误，静默返回 default）: %s", path, e)
        return default
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
        # P9：tmp 名加进程/线程唯一后缀——并发写同一 cache key 时两线程共用
        # <path>.tmp 会互相截断/损坏；os.replace 成功后 tmp 已不存在，无需额外清理
        tmp = Path(f"{path}.tmp.{os.getpid()}.{threading.get_ident()}")
        try:
            joblib.dump(obj, tmp)
            _replace_with_retry(tmp, path)
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
