"""control.core.store — 原子索引存储（control 层共享原语）。

workspace / env_snapshot / gazette 三个模块各自管理一组「具名子目录 +
_index.json 索引」，索引读写逻辑（load / 原子 save / 时间戳）完全相同。
本模块抽出 AtomicIndexStore 统一封装，消除三份重复实现。

设计：
  - load()：读 JSON 索引，不存在返回 {top_key: {}}
  - save(idx)：原子写（.tmp → os.replace），Windows 下 PermissionError 重试
  - now()：ISO 时间戳（timespec=seconds），建索引项时统一用
  - update(mutator)：加锁 RMW（read → mutate → write），并发安全

线程安全：每个 store 自带 threading.Lock，orchestrator 多线程 fork 并发写索引不丢更新。
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path


class AtomicIndexStore:
    """具名子目录的原子索引存储。

    Args:
        base_dir: 索引文件所在目录（Path 或返回 Path 的 0 参可调用对象）。
                  传可调用对象时每次访问动态求值——供测试 monkeypatch 目录常量。
        top_key:  索引 JSON 的顶层键（如 "workspaces" / "snapshots" / "plans"）
    """

    def __init__(self, base_dir, top_key: str):
        self._base_dir = base_dir
        self.top_key = top_key
        self._lock = threading.Lock()

    @property
    def base_dir(self) -> Path:
        """实际目录（可调用对象则惰性求值，兼容测试期 monkeypatch）。"""
        return self._base_dir() if callable(self._base_dir) else self._base_dir

    @property
    def path(self) -> Path:
        """索引文件路径 <base_dir>/_index.json。"""
        return self.base_dir / "_index.json"

    def ensure_dir(self) -> Path:
        """确保 base_dir 存在。"""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        return self.base_dir

    def load(self) -> dict:
        """读索引；不存在返回 {top_key: {}}。"""
        p = self.path
        if not p.exists():
            return {self.top_key: {}}
        return json.loads(p.read_text(encoding="utf-8"))

    def save(self, idx: dict) -> None:
        """原子写索引（调用方持 self._lock 或走 update()）。

        Windows 下 os.replace 偶有锁竞争（杀软/编辑器占用），重试 3 次。
        最后一次仍失败则忽略——索引是缓存，丢了下次 append 会重建。
        """
        self.ensure_dir()
        p = self.path
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")
        for _attempt in range(3):
            try:
                os.replace(str(tmp), str(p))
                return
            except PermissionError:
                time.sleep(0.05)
        # 最后一次仍失败则强制写
        try:
            os.replace(str(tmp), str(p))
        except PermissionError:
            pass

    def now(self) -> str:
        """ISO 时间戳（秒精度）。"""
        return datetime.now().isoformat(timespec="seconds")

    def update(self, mutator) -> dict:
        """加锁 RMW：读索引 → mutator(idx) 改写 → 原子写回。返回 mutator 的返回值。

        mutator 接收 idx dict，可直接修改并返回结果。
        """
        with self._lock:
            idx = self.load()
            return mutator(idx)
