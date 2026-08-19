"""
core.logging — 日志与控制台编码设施

  - setup_console()：win32 下 stdout/stderr UTF-8 reconfigure（幂等），
    替代原 13 处重复的内联修复。
  - get_logger(name)：统一格式的 logging.Logger，handler 只配置一次。

业务代码后续从 print 迁移到 logger；本模块只提供设施。
"""

import logging
import sys
import threading

_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"

_console_ready = False
_root_configured = False
_init_lock = threading.Lock()  # A-9：root logger 初始化的检查-置位竞态锁


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


def _make_file_handler(path):
    """构造统一的滚动文件 handler（R8：get_logger 初始化与 rebind_log_file
    两处逐字重复的构造参数单源——maxBytes/backupCount 只在这一处定义）。"""
    from logging.handlers import RotatingFileHandler
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(
        path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    return fh


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
        # A-9：检查-置位持锁——MCP 线程池/多 worker 首次并发导入时，无锁会让两个
        # 线程都通过检查、重复挂 console/file handler（日志成倍重复输出）
        with _init_lock:
            if not _root_configured:
                import os

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
                        root.addHandler(_make_file_handler(log_file))
                    except Exception:
                        # 落盘失败不阻断控制台日志（只输出 stderr 警告）
                        import sys

                        print(f"[logging] RotatingFileHandler 挂载失败 ({log_file})，仅 stdout 输出",
                              file=sys.stderr)

                _root_configured = True
    return logging.getLogger(name if name.startswith("llmsec") else f"llmsec.{name}")


def rebind_log_file(new_path) -> None:
    """把已挂载的文件 handler 切换到 new_path（work-dir 隔离用）。

    get_logger 在首次调用（通常是模块 import 期）就打开全局 output/logs 的
    RotatingFileHandler——事后重绑 config.LOG_FILE 不影响已打开的句柄。
    本函数关闭并移除现有文件 handler，改挂 new_path（格式/滚动策略不变）。
    """
    import logging
    from logging.handlers import RotatingFileHandler

    # A-9 附注：rebind 若先于任何 get_logger 调用发生（isolation 在构造函数
    # import 期执行时可能），root 尚未配置 level/propagate——先强制完成初始化，
    # 避免挂出"孤儿"文件 handler 后 get_logger 再补挂第二个（同路径双写）。
    get_logger("llmsec")

    root = logging.getLogger("llmsec")
    for h in list(root.handlers):
        if isinstance(h, RotatingFileHandler):
            h.close()
            root.removeHandler(h)
    try:
        root.addHandler(_make_file_handler(new_path))
    except Exception:
        import sys

        print(f"[logging] 日志文件切换失败 ({new_path})，保持仅控制台输出", file=sys.stderr)
