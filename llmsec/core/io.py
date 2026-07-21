"""
core.io — JSONL 读写与断点续传工具

替代原 10+ 处重复的「逐行 strip → 非空 → json.loads」读循环，
以及 4 处重复的断点续传 done_ids 收集逻辑（参照 evaluate.py:381-393）。
所有文件操作统一 utf-8 编码。
"""

import json
import os
from pathlib import Path
from typing import Iterator


def read_jsonl(path) -> list[dict]:
    """读取整个 JSONL 文件为 dict 列表。文件不存在返回空列表；坏行跳过。"""
    return list(iter_jsonl(path))


def iter_jsonl(path) -> Iterator[dict]:
    """逐行迭代 JSONL，坏行（JSON 解析失败）跳过。文件不存在时不产出任何行。"""
    path = Path(path)
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


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
