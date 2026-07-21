"""
targets.openai_backend — 标准 OpenAI /v1/chat/completions 后端

行为与原 targets.py 的 _call_openai 完全一致：
  temperature=0.0, max_tokens=1024, 最多 3 次重试、间隔 3s，timeout 默认 90s。
"""

import time

from llmsec.core.config import TargetConfig
from llmsec.core.llm import create_openai_client
from llmsec.targets.base import TargetClient


class OpenAITargetClient(TargetClient):
    """标准 OpenAI 兼容后端（DeepSeek 等）。"""

    backend_name = "openai"

    def __init__(self, config: TargetConfig | None = None):
        self.config = config or TargetConfig.from_env()
        self.client = create_openai_client(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )

    def call(self, prompt: str) -> dict:
        cfg = self.config
        for attempt in range(1, cfg.max_retries + 1):
            try:
                t0 = time.perf_counter()
                response = self.client.chat.completions.create(
                    model=cfg.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                )
                latency = (time.perf_counter() - t0) * 1000

                content = response.choices[0].message.content or ""
                usage = response.usage

                return {
                    "content": content,
                    "latency_ms": round(latency, 1),
                    "tokens_prompt": usage.prompt_tokens if usage else 0,
                    "tokens_completion": usage.completion_tokens if usage else 0,
                    "error": None,
                    "target_refused": False,
                    "meta": {"backend": self.backend_name, "model": cfg.model},
                }
            except Exception as e:
                if attempt >= cfg.max_retries:
                    return self._error_result(
                        str(e),
                        {"backend": self.backend_name, "attempts": attempt},
                    )
                time.sleep(3)
