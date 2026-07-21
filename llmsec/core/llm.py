"""
core.llm — OpenAI 兼容客户端工厂与统一重试封装

替代原 6 处各自为政的 OpenAI(...) 创建（超时各异）。
重试模式参照 judge.py 的 _call_judge：失败后固定间隔 sleep，最后一次抛出异常。
"""

import time

from openai import OpenAI


def create_openai_client(
    api_key: str | None,
    base_url: str | None,
    timeout: float = 60.0,
) -> OpenAI:
    """创建 OpenAI 兼容客户端。timeout 默认 60s（原 runner/evaluate 约定）。"""
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def chat_with_retry(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict],
    max_retries: int = 3,
    delay: float = 1.0,
    **kwargs,
):
    """
    带重试的 chat.completions.create 封装。

    失败时 sleep(delay) 后重试；最后一次尝试仍失败则抛出原异常。
    额外参数（temperature、max_tokens 等）经 **kwargs 透传。
    返回 openai 的 ChatCompletion 响应对象。
    """
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs,
            )
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(delay)
    raise last_error
