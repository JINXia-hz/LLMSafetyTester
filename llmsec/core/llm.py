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


def retry_call(
    func,
    *,
    retries: int,
    delay: float,
    retry_on=None,
    on_retry=None,
):
    """
    通用重试辅助（统一原各处内联重试循环）。

    - func: 无参可调用，成功时原样返回其结果。
    - retries: 总尝试次数；全部失败时抛出最后一次异常。
    - delay: 重试前的固定 sleep 秒数。
    - retry_on: 谓词 (e) -> bool，判定异常是否可重试；
      None 表示所有 Exception 均可重试。判定为不可重试的异常立即抛出，
      不 sleep、不消耗后续尝试。
    - on_retry: 回调 (attempt, e) -> float | None，仅在真正重试前调用
      （最后一次失败不调用）；返回数值时覆盖本次 sleep 间隔
      （用于 429/限流等特殊间隔），返回 None 时使用 delay。
      也可用于打印重试日志。
    """
    last_error: Exception | None = None
    # 至少一次尝试：retries<1（误传 0）时也调用一次，否则 range 为空 → raise last_error(None)
    # 抛 TypeError: exceptions must derive from BaseException
    for attempt in range(1, max(1, retries) + 1):
        try:
            return func()
        except Exception as e:
            if retry_on is not None and not retry_on(e):
                raise
            last_error = e
            if attempt < retries:
                wait = delay
                if on_retry is not None:
                    override = on_retry(attempt, e)
                    if override is not None:
                        wait = override
                time.sleep(wait)
    raise last_error


def is_retryable_error(e: Exception) -> bool:
    """判定异常是否值得重试。

    H-7 修复：排除 4xx 确定性错误（认证失败/参数错误/权限拒绝/资源不存在），
    保留 429 限流、5xx 服务端错误、网络错误。原代码对所有异常重试，
    错误 API Key 跑 500 条 prompt 会白等 75 分钟。

    用 status_code 属性判定（不依赖具体异常类，跨 OpenAI 版本健壮）。
    """
    status = getattr(e, "status_code", None)
    if status is not None and 400 <= status < 500 and status != 429:
        return False  # 4xx（非 429）= 确定性错误，重试无意义
    return True


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
    H-7 修复：4xx 确定性错误（认证/参数/权限/404）不重试，直接抛出。
    额外参数（temperature、max_tokens 等）经 **kwargs 透传。
    返回 openai 的 ChatCompletion 响应对象。
    """
    return retry_call(
        lambda: client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
        ),
        retries=max_retries,
        delay=delay,
        retry_on=is_retryable_error,
    )
