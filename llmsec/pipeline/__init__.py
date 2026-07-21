"""
llmsec.pipeline — 编排流水线子包

  - runner：三阶段自适应安全评估编排器（原根目录 runner.py）
  - launcher：交互式启动器（原根目录 launcher.py）
  - probe：受害者 PCAP Judge API 探测脚本（原根目录 probe_victim.py）

子模块采用惰性导入，避免 `python -m llmsec.pipeline.xxx` 时的 runpy 重复加载告警。
"""

__all__ = ["runner", "launcher", "probe"]


def __getattr__(name: str):
    if name in __all__:
        import importlib

        return importlib.import_module(f"llmsec.pipeline.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
