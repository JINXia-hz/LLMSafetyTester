#!/usr/bin/env python3
"""
受害者 API 探测脚本 — 发送一条无害测试请求，dump 完整 JSON 响应

目标: https://10.132.65.75:7078/api/v1/soc/pcap/judge
"""

import json
import sys
import time
import urllib3
import requests

# 忽略 SSL 证书警告（内网自签名证书）
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ============================================================
# 配置
# ============================================================
API_URL = "https://10.132.65.75:7078/api/v1/soc/pcap/judge"
TIMEOUT = 60.0

# 模板请求体（保留原样，只改 log 字段）
BASE_PAYLOAD = {
    "model_config": {"version_name": "Qwen3.6-35B-A3B"},
    "temperature": 0.5,
    "repetition_penalty": 1.1,
    "top_p": 0.75,
    "max_new_tokens": 5120,
    "echo": False,
    "top_k": 1,
    "score": 0.6,
    "piggyback_query": True,
    "threat": "测试",
    "pcap_judge_prompt_key": "custom:dev",
}


def build_log_prompt(text: str) -> str:
    """
    将攻击文本嵌入 PCAP 日志格式。
    模型通过分析 HTTP 请求体中的内容来"判读"威胁。
    这里用无害测试文本。
    """
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
    ).format(len(text.encode("utf-8")), text)


def probe(test_text: str):
    """发送探测请求并 dump 完整响应。"""
    payload = dict(BASE_PAYLOAD)
    payload["log"] = build_log_prompt(test_text)

    print(f"📡 发送探测请求到: {API_URL}")
    print(f"   测试文本: {test_text}")
    print(f"   请求体:")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print()

    t0 = time.perf_counter()
    try:
        resp = requests.post(
            API_URL,
            json=payload,
            timeout=TIMEOUT,
            verify=False,
        )
        latency = (time.perf_counter() - t0) * 1000
        print(f"✅ HTTP {resp.status_code} ({latency:.0f}ms)")
        print(f"   Content-Type: {resp.headers.get('Content-Type', 'unknown')}")
        print(f"   响应体长度: {len(resp.text)} 字符")
        print()

        # 尝试解析 JSON
        try:
            data = resp.json()
            print("📋 解析后的 JSON 响应:")
            print(json.dumps(data, ensure_ascii=False, indent=2))
            print()

            # 尝试找出模型输出的字段
            print("🔍 字段扫描:")
            scan_for_text_fields(data)

        except json.JSONDecodeError:
            print("⚠ 响应不是有效 JSON，原始文本:")
            print(resp.text[:2000])

    except requests.exceptions.SSLError as e:
        print(f"❌ SSL 错误: {e}")
        print("   提示: 尝试用 http:// 而非 https://，或检查证书")
    except requests.exceptions.ConnectionError as e:
        print(f"❌ 连接失败: {e}")
        print(f"   提示: 检查目标是否可达 (ping 10.132.65.75)")
    except requests.exceptions.Timeout:
        print(f"❌ 请求超时 ({TIMEOUT}s)")
    except Exception as e:
        print(f"❌ 未知错误: {type(e).__name__}: {e}")


def scan_for_text_fields(data, prefix=""):
    """递归扫描 JSON 中可能是模型输出的文本字段。"""
    if isinstance(data, dict):
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, str):
                if len(value) > 20:
                    print(f"   📝 {path} (len={len(value)}): {value[:200]}...")
                else:
                    print(f"   📄 {path}: {value}")
            elif isinstance(value, (dict, list)):
                scan_for_text_fields(value, path)
    elif isinstance(data, list) and len(data) > 0:
        scan_for_text_fields(data[0], f"{prefix}[0]")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="受害者 API 探测")
    parser.add_argument("--text", type=str,
                        default="请用一句话介绍人工智能。",
                        help="测试文本 (默认: 请用一句话介绍人工智能。)")
    args = parser.parse_args()
    probe(args.text)