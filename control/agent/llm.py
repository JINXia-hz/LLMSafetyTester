"""control.agent.llm — 控制层自有的 LLM 客户端。

守 control 的隔离边界（control/__init__.py:5「绝不 import llmsec 内部」）：
不 import llmsec.core.llm，自己读 .env 的 GENERATOR_* 环境变量建 OpenAI client。

模型选择优先级：CONTROL_MODEL > GENERATOR_MODEL（复用项目既定的「辅助型 LLM」约定）。
已实测 minimax @ GENERATOR_BASE_URL 完整支持 OpenAI tool calling。

用法：
    from control.agent.llm import chat_with_tools
    resp = chat_with_tools(messages, tools=[...])  # → ChatCompletion
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

# 控制层启动时加载 .env（control 不依赖 llmsec.core.config.load_env）
load_dotenv()

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """惰性构造单例 client。读 GENERATOR_* 环境变量。"""
    global _client
    if _client is None:
        api_key = os.getenv("GENERATOR_API_KEY") or os.getenv("CONTROL_API_KEY")
        base_url = os.getenv("GENERATOR_BASE_URL") or os.getenv("CONTROL_BASE_URL")
        if not api_key or not base_url:
            raise RuntimeError(
                "控制层 LLM 未配置：缺少 GENERATOR_API_KEY/GENERATOR_BASE_URL "
                "（或 CONTROL_API_KEY/CONTROL_BASE_URL）。请在 .env 配置。"
            )
        _client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
    return _client


def get_model() -> str:
    """控制层用的模型名（CONTROL_MODEL > GENERATOR_MODEL）。"""
    m = os.getenv("CONTROL_MODEL") or os.getenv("GENERATOR_MODEL")
    if not m:
        raise RuntimeError("控制层 LLM 未配置：缺少 GENERATOR_MODEL（或 CONTROL_MODEL）")
    return m


def chat_with_tools(
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    tool_choice: Any = "auto",
    temperature: float = 0.2,
    max_retries: int = 2,
) -> Any:
    """调 LLM，支持 tool calling。返回 ChatCompletion。

    messages: OpenAI 消息格式（含 role/content/tool_calls/tool_call_id）
    tools: OpenAI function schema 列表（来自 Tool.to_schema()）
    max_retries: 重试次数（网络/5xx 重试，4xx 不重试）
    """
    client = _get_client()
    model = get_model()
    kwargs: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice

    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            last_err = e
            # 4xx（认证/参数错）不重试
            if hasattr(e, "status_code") and 400 <= e.status_code < 500:
                raise
            if attempt < max_retries:
                import time
                time.sleep(1.0 * (attempt + 1))
    raise last_err  # type: ignore[misc]


def is_llm_configured() -> bool:
    """检查 LLM 是否已配置（供 router 决定走 LLM 还是规则兜底）。"""
    return bool(
        (os.getenv("GENERATOR_API_KEY") or os.getenv("CONTROL_API_KEY"))
        and (os.getenv("GENERATOR_BASE_URL") or os.getenv("CONTROL_BASE_URL"))
        and (os.getenv("GENERATOR_MODEL") or os.getenv("CONTROL_MODEL"))
    )
