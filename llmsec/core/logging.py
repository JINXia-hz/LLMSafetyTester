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
    并可选挂一个 RotatingFileHandler 落盘到 output/logs/llmsec.log。

    环境变量：
      LLMSEC_LOG_LEVEL  日志级别（默认 INFO）
      LLMSEC_LOG_FILE   落盘路径（默认 output/logs/llmsec.log）；
                        置空字符串则不落盘（仅 stdout，CLI 短任务可关）
    """
    global _root_configured
    if not _root_configured:
        import os
        from logging.handlers import RotatingFileHandler

        formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)
        root = logging.getLogger("llmsec")
        root.setLevel(os.getenv("LLMSEC_LOG_LEVEL", "INFO").upper())
        root.propagate = False

        # 控制台 handler（始终挂载）
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)

        # 文件 handler（LLMSEC_LOG_FILE 非空时挂载；默认落盘 output/logs/llmsec.log）
        log_file = os.getenv("LLMSEC_LOG_FILE")
        if log_file is None:
            # 未显式设置时用默认路径；显式置空字符串则跳过落盘
            try:
                from llmsec.core.config import LOG_FILE
                log_file = str(LOG_FILE)
            except Exception:
                log_file = ""
        if log_file:
            try:
                from pathlib import Path

                Path(log_file).parent.mkdir(parents=True, exist_ok=True)
                file_handler = RotatingFileHandler(
                    log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
                )
                file_handler.setFormatter(formatter)
                root.addHandler(file_handler)
            except Exception:
                # 落盘失败不阻断控制台日志（只输出 stderr 警告）
                import sys

                print(f"[logging] RotatingFileHandler 挂载失败 ({log_file})，仅 stdout 输出",
                      file=sys.stderr)

        _root_configured = True
    return logging.getLogger(name if name.startswith("llmsec") else f"llmsec.{name}")
