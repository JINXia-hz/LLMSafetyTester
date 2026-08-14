"""llmsec.core.paths — 外部名称到文件系统路径的安全拼接。

MCP 工具参数 / HTTP 请求体 / CLI 参数里的 name、plan_id、run_name 等是
LLM/用户可控的外部数据，直接 ``BASE / name`` 会遭路径穿越（``name="../../etc"``
逃出 base）。本模块提供两个校验函数，供 llmsec 内部及 control 层统一复用：

  - safe_component(base, name)   单段拼接（workspace 名 / 快照名 / plan_id）
  - safe_subpath(base, *parts)   多段拼接（ws:<name>/<target>、ts/target）

防御策略（黑名单 + resolve 双保险）：
  1. 拒绝含路径分隔符（/ \\）、空、. .. 的 name——堵掉绝大多数穿越向量；
  2. resolve() 后断言结果仍在 base 内——堵漏网（如符号链接、盘符 ``C:`` 等）。

抛 ValueError，与现有 fork 的 FileExistsError/ValueError 风格一致；
HTTP router 已有 ``except ValueError → 400`` 捕获。
"""

from __future__ import annotations

from pathlib import Path


def safe_component(base: Path, name: str) -> Path:
    """把外部 name 作为 base 下的**单段**路径安全拼接，拒绝穿越。

    Args:
        base: 不可信名称应落在其下的基目录（如 WORKSPACES_DIR）。
        name: 外部可控的单段名称（workspace 名 / 快照名 / plan_id 等）。

    Returns:
        校验通过后的绝对路径。

    Raises:
        ValueError: name 含路径分隔符 / 为空 / 为 . 或 .. / resolve 后越出 base。
    """
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError(f"非法名称: {name!r}")
    base_r = base.resolve()
    p = (base / name).resolve()
    # 结果必须严格落在 base 内（不允许等于 base 之外）
    if p != base_r and base_r not in p.parents:
        raise ValueError(f"路径越界: {name!r}")
    return p


def safe_subpath(base: Path, *parts: str) -> Path:
    """把多个外部段逐段校验后拼接到 base 下（如 ws:<name>/<target>）。

    每段都走 safe_component 的黑名单 + resolve 校验，链式收窄——
    任一段恶意（含 / 或 ..）即抛 ValueError。

    Args:
        base:  基目录。
        parts: 依次下钻的名称段（如 "2024-01-01_120000", "target-a"）。

    Returns:
        校验通过后的绝对路径。

    Raises:
        ValueError: 任一段非法或越界。
    """
    p = base.resolve()
    for part in parts:
        p = safe_component(p, part)
    return p
