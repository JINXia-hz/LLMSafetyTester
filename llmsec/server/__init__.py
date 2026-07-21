"""
llmsec.server — 本地服务子包

  - local_model_server：本地模拟小模型服务器（OpenAI 兼容 API，
    原根目录 local_model_server.py）

子模块采用惰性导入，避免 `python -m llmsec.server.xxx` 时的 runpy 重复加载告警。
"""

__all__ = ["local_model_server"]


def __getattr__(name: str):
    if name in __all__:
        import importlib

        return importlib.import_module(f"llmsec.server.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
