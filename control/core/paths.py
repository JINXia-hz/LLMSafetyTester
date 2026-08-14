"""control.core.paths — 外部名称到文件系统路径的安全拼接（薄封装）。

权威实现见 ``llmsec.core.paths``（llmsec 是更底层的共享层，control 依赖它）。
本模块仅重新导出，使 control 内部代码可从 ``control.core.paths`` 导入而保持
与 llmsec 侧同一套校验口径，避免逻辑漂移。

详见 ``llmsec/core/paths.py`` 的 docstring 与防御策略说明。
"""

from __future__ import annotations

from llmsec.core.paths import safe_component, safe_subpath

__all__ = ["safe_component", "safe_subpath"]
