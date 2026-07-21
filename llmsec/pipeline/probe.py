#!/usr/bin/env python3
"""
受害者 API 探测脚本（原根目录 probe_victim.py）

发送一条无害测试请求到 PCAP Judge API，dump 完整 JSON 响应。
payload 模板与日志构造复用 llmsec.targets.pcap（原脚本内的重复实现已删除）。
"""

import argparse
import json
import time

import requests
import urllib3

from llmsec.core.logging import setup_console
from llmsec.targets.pcap import PCAP_JUDGE_URL, build_pcap_payload

setup_console()

# 忽略 SSL 证书警告（内网自签名证书）
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

TIMEOUT = 60.0


def probe(test_text: str):
    """发送探测请求并 dump 完整响应。"""
    # strip_math=False：探测文本原样嵌入，不做数学题越狱税剥离
    payload = build_pcap_payload(test_text, strip_math=False)

    print(f"📡 发送探测请求到: {PCAP_JUDGE_URL}")
    print(f"   测试文本: {test_text}")
    print(f"   请求体:")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print()

    t0 = time.perf_counter()
    try:
        resp = requests.post(
            PCAP_JUDGE_URL,
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


def main():
    parser = argparse.ArgumentParser(description="受害者 API 探测")
    parser.add_argument("--text", type=str,
                        default="请用一句话介绍人工智能。",
                        help="测试文本 (默认: 请用一句话介绍人工智能。)")
    args = parser.parse_args()
    probe(args.text)


if __name__ == "__main__":
    main()
