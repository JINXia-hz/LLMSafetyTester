"""
llmsec.attacks — 攻击集生成

  - generate：解析 攻击分析.md，两轮 LLM 生成 L1 攻击 prompt（嵌入数学题）。
  - harmbench：内置 HarmBench 行为库 × 人工越狱模板 → 攻击集 JSONL（示范用，数据见 llmsec/data/）。

子模块采用惰性导入，避免 `python -m llmsec.attacks.xxx` 时的 runpy 重复加载告警。
"""

__all__ = ["generate", "harmbench"]


def __getattr__(name: str):
    if name in __all__:
        import importlib

        return importlib.import_module(f"llmsec.attacks.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
