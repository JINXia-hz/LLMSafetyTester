"""control.core.fsig — 文件变更签名（mtime 族缓存的统一指纹）。

项目里有数份手写的"按 mtime 失效"逻辑（data_query 的 run 发现缓存、run 元信息
缓存，zhongshu 的文牍文本缓存）——共享同一套语义与同一已知局限：粗粒度 mtime
文件系统（如 FAT 的 2s 粒度）上同刻写入可能吃到旧缓存；NTFS/ext4 无虞。
control 侧统一用本模块（llmsec 侧的 data_query 有同构实现，因隔离边界不能共用）。
（r7：dir_sig 无生产调用方已删——目录签名消费方 data_query 用的是自己的
_dir_sig 副本。）
"""
from __future__ import annotations

from pathlib import Path


def file_sig(path: Path) -> tuple[float, int] | None:
    """单文件签名 (mtime, size)；不存在/不可 stat 返回 None。"""
    try:
        st = Path(path).stat()
        return (st.st_mtime, st.st_size)
    except OSError:
        return None

