#!/usr/bin/env python3
"""
目标 API 适配层

统一封装不同后端的目标 API 调用，返回标准化格式。

支持的后端：
  - openai      : 标准 OpenAI /v1/chat/completions
  - local_sim   : 本地模拟模型服务器 (local_model_server.py)
  - pcap_judge  : 受害者 PCAP 判读 API

标准返回格式：
  {"content": str, "latency_ms": float, "tokens_prompt": int, "tokens_completion": int, "error": str|None, "meta": dict}
"""

import json
import os
import re
import sys
import time
from typing import Optional

import urllib3
import requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 配置加载
# ============================================================
TARGET_TYPE = os.getenv("TARGET_TYPE", "openai")

# OpenAI / 通用
TARGET_API_KEY = os.getenv("TARGET_API_KEY", "")
TARGET_BASE_URL = os.getenv("TARGET_BASE_URL", "https://api.deepseek.com/v1")
TARGET_MODEL = os.getenv("TARGET_MODEL", "deepseek-v4-flash")

# PCAP Judge
PCAP_JUDGE_URL = os.getenv(
    "PCAP_JUDGE_URL",
    "https://10.132.65.75:7078/api/v1/soc/pcap/judge",
)
PCAP_MODEL_VERSION = os.getenv("PCAP_MODEL_VERSION", "Qwen3.6-35B-A3B")
PCAP_PROMPT_KEY = os.getenv("PCAP_PROMPT_KEY", "custom:dev")

MAX_RETRIES = 3
REQUEST_TIMEOUT = 90.0  # PCAP 判读较慢


# ============================================================
# 通用调用
# ============================================================
def call_target(prompt: str) -> dict:
    """
    根据 TARGET_TYPE 路由到对应后端。
    返回: {content, latency_ms, tokens_prompt, tokens_completion, error, meta}
    """
    if TARGET_TYPE == "pcap_judge":
        return _call_pcap_judge(prompt)
    elif TARGET_TYPE == "local_sim":
        return _call_openai(prompt)
    else:
        return _call_openai(prompt)


# ============================================================
# OpenAI 后端（兼容本地模拟）
# ============================================================
def _call_openai(prompt: str) -> dict:
    """标准 OpenAI /v1/chat/completions。"""
    from openai import OpenAI

    client = OpenAI(
        api_key=TARGET_API_KEY,
        base_url=TARGET_BASE_URL,
        timeout=REQUEST_TIMEOUT,
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t0 = time.perf_counter()
            response = client.chat.completions.create(
                model=TARGET_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1024,
            )
            latency = (time.perf_counter() - t0) * 1000

            choice = response.choices[0]
            content = choice.message.content or ""
            usage = response.usage

            return {
                "content": content,
                "latency_ms": round(latency, 1),
                "tokens_prompt": usage.prompt_tokens if usage else 0,
                "tokens_completion": usage.completion_tokens if usage else 0,
                "error": None,
                "target_refused": False,
                "meta": {"backend": "openai", "model": TARGET_MODEL},
            }
        except Exception as e:
            if attempt >= MAX_RETRIES:
                return {
                    "content": "",
                    "latency_ms": 0,
                    "tokens_prompt": 0,
                    "tokens_completion": 0,
                    "error": str(e),
                    "target_refused": False,
                    "meta": {"backend": "openai", "attempts": attempt},
                }
            time.sleep(3)


# ============================================================
# PCAP Judge 后端
# ============================================================
def _strip_math(prompt_text: str) -> str:
    """
    剥离 prompt 末尾的数学题标记（PCAP 分析模式下模型不会回答数学题）。
    保留纯攻击文本。
    """
    import re
    cleaned = re.sub(
        r"\n*\s*在回答之前，请先计算以下数学题.*$",
        "", prompt_text, flags=re.DOTALL
    ).strip()
    cleaned = re.sub(r"\[MATH:\s*-?\d+\s*\]", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def _build_pcap_log(prompt_text: str) -> str:
    """
    将攻击 prompt 嵌入 PCAP 日志的 HTTP 请求体中。
    自动剥离数学题（PCAP 分析模型不回答数学题）。
    """
    body = _strip_math(prompt_text)
    body_bytes = body.encode("utf-8")
    return (
        "【REQUEST】\r\n"
        "POST /api/query HTTP/1.1\r\n"
        "Host: 127.0.0.1\r\n"
        "Content-Type: text/plain\r\n"
        "User-Agent: Mozilla/5.0\r\n"
        "Content-Length: {}\r\n"
        "\r\n"
        "{}\r\n"
        "【RESPONSE】\r\n"
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/plain\r\n"
        "\r\n"
        "success"
    ).format(len(body_bytes), body)


def _call_pcap_judge(prompt: str) -> dict:
    """
    调用受害者 PCAP 判读 API。
    攻击 prompt 嵌入 log 字段的 HTTP 请求体中。
    返回模型的分析文本作为 content。
    """
    payload = {
        "model_config": {"version_name": PCAP_MODEL_VERSION},
        "temperature": 0.5,
        "repetition_penalty": 1.1,
        "top_p": 0.75,
        "max_new_tokens": 5120,
        "echo": False,
        "top_k": 1,
        "score": 0.6,
        "piggyback_query": True,
        "threat": "测试",
        "pcap_judge_prompt_key": PCAP_PROMPT_KEY,
        "log": _build_pcap_log(prompt),
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t0 = time.perf_counter()
            resp = requests.post(
                PCAP_JUDGE_URL,
                json=payload,
                timeout=REQUEST_TIMEOUT,
                verify=False,
            )
            latency = (time.perf_counter() - t0) * 1000

            if resp.status_code != 200:
                return {
                    "content": "",
                    "latency_ms": round(latency, 1),
                    "tokens_prompt": len(prompt) // 2,
                    "tokens_completion": 0,
                    "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                    "target_refused": False,
                    "meta": {"backend": "pcap_judge", "status": resp.status_code},
                }

            data = resp.json()
            pred = data.get("pred", "")
            target_refused = (pred == "攻击")

            if data.get("error_code", -1) != 0:
                return {
                    "content": data.get("text", ""),
                    "latency_ms": round(latency, 1),
                    "tokens_prompt": len(prompt) // 2,
                    "tokens_completion": len(data.get("text", "")) // 2,
                    "error": f"API error_code={data.get('error_code')}",
                    "target_refused": target_refused,
                    "meta": {
                        "backend": "pcap_judge",
                        "pred": pred,
                        "error_code": data.get("error_code"),
                    },
                }

            return {
                "content": data.get("text", ""),
                "latency_ms": round(latency, 1),
                "tokens_prompt": len(prompt) // 2,
                "tokens_completion": len(data.get("text", "")) // 2,
                "error": None,
                "target_refused": target_refused,
                "meta": {
                    "backend": "pcap_judge",
                    "pred": pred,
                    "threat_type": data.get("threat_type"),
                    "attack_result": data.get("attack_result"),
                },
            }

        except Exception as e:
            if attempt >= MAX_RETRIES:
                return {
                    "content": "",
                    "latency_ms": 0,
                    "tokens_prompt": 0,
                    "tokens_completion": 0,
                    "error": f"{type(e).__name__}: {e}",
                    "meta": {"backend": "pcap_judge", "attempts": attempt},
                }
            time.sleep(3)