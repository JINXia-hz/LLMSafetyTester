"""
targets.base — 目标客户端抽象基类

统一标准返回格式（与原 targets.call_target 完全一致）：
  {
      "content": str,              # 目标模型输出文本
      "latency_ms": float,         # 单次成功调用的耗时（毫秒）
      "tokens_prompt": int,
      "tokens_completion": int,
      "error": str | None,         # 最终失败时的错误描述
      "target_refused": bool,      # 后端可判定的拒绝信号（目前仅 pcap_judge 使用）
      "meta": dict,                # 后端附加信息，至少含 "backend" 键
  }

注意：call() 不抛异常——所有失败在重试耗尽后以 error 字段返回（保留现行行为）。
"""

from abc import ABC, abstractmethod


class TargetClient(ABC):
    """目标模型后端抽象。子类实现 call()。"""

    #: meta["backend"] 标识
    backend_name: str = "unknown"

    @abstractmethod
    def call(self, prompt: str) -> dict:
        """调用目标模型，返回标准格式 dict（见模块 docstring）。"""

    @staticmethod
    def _error_result(error: str, meta: dict) -> dict:
        """构造失败结果（各后端共用）。"""
        return {
            "content": "",
            "latency_ms": 0,
            "tokens_prompt": 0,
            "tokens_completion": 0,
            "error": error,
            "target_refused": False,
            "meta": meta,
        }
