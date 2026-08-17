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
import threading

from llmsec.core.config import TargetConfig, load_env, load_targets, target_backend
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
    pcap_judge_url,
    pcap_model_version,
    pcap_prompt_key,
)

__all__ = [
    "TargetClient",
    "OpenAITargetClient",
    "LocalSimTargetClient",
    "PcapJudgeTargetClient",
    "create_target_client",
    "create_named_target_client",
    "call_target",
    "call_target_named",
    "available_targets",
    "set_active_target",
    "get_active_target",
    # pcap 复用件（供 probe_victim 等使用）
    "BASE_PAYLOAD",
    "PCAP_JUDGE_URL",
    "PCAP_MODEL_VERSION",
    "PCAP_PROMPT_KEY",
    "pcap_judge_url",
    "pcap_model_version",
    "pcap_prompt_key",
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
_default_client_lock = threading.Lock()


# ------------------------------------------------------------
# 多目标：按 defender 名分发（{name: client} 字典，惰性缓存）
# ------------------------------------------------------------
_named_clients: dict[str, TargetClient] = {}
_named_clients_lock = threading.Lock()


def available_targets() -> dict[str, TargetConfig]:
    """返回 .env 声明的全部目标 {name: TargetConfig}（来自 load_targets）。"""
    return load_targets()


def create_named_target_client(name: str) -> TargetClient:
    """
    为指定名称的目标创建/复用客户端。
    backend 类型取 target_backend(name)（每目标可独立，缺失继承全局 TARGET_TYPE）。
    """
    cfg = load_targets().get(name)
    if cfg is None:
        raise KeyError(f"未在 .env 声明的目标: {name}（可用: {list(load_targets())}）")
    return create_target_client(cfg, backend=target_backend(name))


def call_target_named(prompt: str, target_name: str) -> dict:
    """按目标名路由调用（客户端惰性缓存，线程安全）。"""
    client = _named_clients.get(target_name)
    if client is None:
        with _named_clients_lock:
            client = _named_clients.get(target_name)
            if client is None:
                client = create_named_target_client(target_name)
                _named_clients[target_name] = client
    return client.call(prompt)


def call_target(prompt: str, target_name: str | None = None) -> dict:
    """
    调用目标模型，返回标准格式 dict（见 base 模块）。

    - target_name 显式给定 → 路由到该命名目标。
    - 否则若有 ambient 活动目标（set_active_target 设置）→ 用它。
    - 都没有 → 回退旧单例（TARGET_TYPE/TARGET_* 路由，向后兼容）。

    ambient 机制让 evaluator 内核的 call_target(prompt) 调用无需改动即可
    在多目标循环中自动路由到当前激活目标。
    """
    if target_name is not None:
        return call_target_named(prompt, target_name)
    ambient = get_active_target()
    if ambient is not None:
        return call_target_named(prompt, ambient)
    global _default_client
    if _default_client is None:
        with _default_client_lock:
            if _default_client is None:
                _default_client = create_target_client()
    return _default_client.call(prompt)


# ------------------------------------------------------------
# ambient 活动目标（多目标循环用：runner 每轮 set_active_target(name)）
# ------------------------------------------------------------
# H-5 修复：用 threading.local 替代模块级全局变量。
# 原代码是进程级全局（set 加锁但 get 不加锁），docstring 撒谎称"当前线程"。
# threading.local 保证每线程独立，未来线程化并发评估不会串扰。
_active_local = threading.local()


def set_active_target(name: str | None) -> None:
    """设置当前线程的"活动目标"名；call_target(prompt) 无显式名时路由到它。"""
    _active_local.name = name


def get_active_target() -> str | None:
    return getattr(_active_local, "name", None)
