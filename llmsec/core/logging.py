"""
core.logging — 日志与控制台编码设施

  - setup_console()：win32 下 stdout/stderr UTF-8 reconfigure（幂等），
    替代原 13 处重复的内联修复。
  - get_logger(name)：统一格式的 logging.Logger，handler 只配置一次。

业务代码后续从 print 迁移到 logger；本模块只提供设施。
"""

import logging
import sys

_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"

_console_ready = False
_root_configured = False


def setup_console() -> None:
    """win32 下把 stdout/stderr 重配为 UTF-8（幂等）。其他平台为 no-op。"""
    global _console_ready
    if _console_ready:
        return
    _console_ready = True
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            reconfigure = getattr(stream, "reconfigure", None)
            if reconfigure is not None:
                reconfigure(encoding="utf-8", errors="replace")


def get_logger(name: str) -> logging.Logger:
    """
    获取统一格式的 logger。
    首次调用时在 root logger 上挂一个 StreamHandler（只挂一次），
    默认级别 INFO，可用环境变量 LLMSEC_LOG_LEVEL 覆盖。
    """
    global _root_configured
    if not _root_configured:
        import os

        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        root = logging.getLogger("llmsec")
        root.addHandler(handler)
        root.setLevel(os.getenv("LLMSEC_LOG_LEVEL", "INFO").upper())
        root.propagate = False
        _root_configured = True
    return logging.getLogger(name if name.startswith("llmsec") else f"llmsec.{name}")
