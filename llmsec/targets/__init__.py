"""
targets — 目标模型后端包

工厂入口：
    from llmsec.targets import create_target_client, call_target

后端路由（与原 targets.py 一致，由环境变量 TARGET_TYPE 决定）：
  - openai      : 标准 OpenAI /v1/chat/completions（默认）
  - local_sim   : 本地模拟模型服务器 (local_model_server.py)
  - pcap_judge  : 受害者 PCAP 判读 API

call_target(prompt) 为顶层便捷函数（惰性单例），保持原 targets.py 的调用签名，
供现行 runner.py / evaluate.py 直接替换 `from targets import call_target`。
"""

import os

from llmsec.core.config import TargetConfig, load_env
from llmsec.targets.base import TargetClient
from llmsec.targets.local_sim import LocalSimTargetClient
from llmsec.targets.openai_backend import OpenAITargetClient
from llmsec.targets.pcap import (
    BASE_PAYLOAD,
    PCAP_JUDGE_URL,
    PCAP_MODEL_VERSION,
    PCAP_PROMPT_KEY,
    PcapJudgeTargetClient,
    build_pcap_log,
    build_pcap_payload,
)

__all__ = [
    "TargetClient",
    "OpenAITargetClient",
    "LocalSimTargetClient",
    "PcapJudgeTargetClient",
    "create_target_client",
    "call_target",
    # pcap 复用件（供 probe_victim 等使用）
    "BASE_PAYLOAD",
    "PCAP_JUDGE_URL",
    "PCAP_MODEL_VERSION",
    "PCAP_PROMPT_KEY",
    "build_pcap_log",
    "build_pcap_payload",
]

# backend 名 → client 类
_BACKENDS = {
    "openai": OpenAITargetClient,
    "local_sim": LocalSimTargetClient,
    "pcap_judge": PcapJudgeTargetClient,
}


def create_target_client(
    config: TargetConfig | None = None,
    backend: str | None = None,
) -> TargetClient:
    """
    按后端名创建目标客户端。
    backend 缺省时读取环境变量 TARGET_TYPE（默认 "openai"，与原 targets.py 一致）。
    config 仅对 openai/local_sim 后端有效；pcap_judge 使用自身 env 配置。
    """
    load_env()
    if backend is None:
        backend = os.getenv("TARGET_TYPE", "openai")
    if backend == "pcap_judge":
        return PcapJudgeTargetClient()
    cls = _BACKENDS.get(backend)
    if cls is None:
        # 与原 targets.call_target 一致：未知类型回退 openai
        cls = OpenAITargetClient
    return cls(config)


# ------------------------------------------------------------
# 顶层便捷函数（惰性单例，保持原 targets.call_target 签名）
# ------------------------------------------------------------
_default_client: TargetClient | None = None


def call_target(prompt: str) -> dict:
    """按 TARGET_TYPE 路由调用目标模型，返回标准格式 dict（见 base 模块）。"""
    global _default_client
    if _default_client is None:
        _default_client = create_target_client()
    return _default_client.call(prompt)
