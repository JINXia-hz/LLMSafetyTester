#!/usr/bin/env python3
"""
目标 API 探测脚本（原根目录 probe_victim.py）

按 TARGET_TYPE 路由探测当前配置的目标后端：
  - openai / local_sim：经 llmsec.targets.call_target 发送一条测试 prompt，
    打印后端、延迟、响应内容（用于确认 openai 类后端连通性与模型行为）。
  - pcap_judge：发送一条无害测试请求到 PCAP Judge API，dump 完整 JSON 响应
    （payload 模板与日志构造复用 llmsec.targets.pcap）。
"""

import argparse
import json
import os
import time

import requests
import urllib3

from llmsec.core.logging import get_logger, setup_console
from llmsec.targets import call_target
from llmsec.targets.pcap import PCAP_JUDGE_URL, build_pcap_payload

logger = get_logger(__name__)
setup_console()

# 忽略 SSL 证书警告（内网自签名证书）
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

TIMEOUT = 60.0


def probe(test_text: str):
    """按 TARGET_TYPE 路由探测目标后端。"""
    # .env 已在 import llmsec 时由 llmsec/__init__.py 加载，无需重复 load_env
    target_type = os.getenv("TARGET_TYPE", "openai")
    if target_type == "pcap_judge":
        probe_pcap(test_text)
    else:
        probe_openai(test_text, target_type)


def probe_openai(test_text: str, target_type: str):
    """openai / local_sim 后端探测：走统一路由，打印响应概要。"""
    logger.info(f"📡 探测目标后端: TARGET_TYPE={target_type}")
    logger.info(f"   测试文本: {test_text}")
    logger.info("")

    result = call_target(test_text)

    meta = result.get("meta", {})
    logger.info(f"   backend : {meta.get('backend', target_type)}")
    logger.info(f"   latency : {result['latency_ms']:.0f}ms")
    logger.info(f"   tokens  : prompt={result['tokens_prompt']}, completion={result['tokens_completion']}")
    if result["error"]:
        logger.error(f"❌ 调用失败: {result['error']}")
        logger.info("   提示: 检查 TARGET_BASE_URL / TARGET_API_KEY / TARGET_MODEL 配置与目标可达性")
        return
    if result.get("target_refused"):
        logger.warning("   ⚠ 目标侧主动拒绝了该请求（target_refused=True）")
    logger.info("")
    logger.info("✅ 连接成功，响应内容:")
    logger.info(result["content"][:2000])


def probe_pcap(test_text: str):
    """发送探测请求到 PCAP Judge API 并 dump 完整响应。"""
    # strip_math=False：探测文本原样嵌入，不做数学题越狱税剥离
    payload = build_pcap_payload(test_text, strip_math=False)

    logger.info(f"📡 发送探测请求到: {PCAP_JUDGE_URL}")
    logger.info(f"   测试文本: {test_text}")
    logger.info("   请求体:")
    logger.info(json.dumps(payload, ensure_ascii=False, indent=2))
    logger.info("")

    t0 = time.perf_counter()
    try:
        resp = requests.post(
            PCAP_JUDGE_URL,
            json=payload,
            timeout=TIMEOUT,
            verify=False,
        )
        latency = (time.perf_counter() - t0) * 1000
        logger.info(f"✅ HTTP {resp.status_code} ({latency:.0f}ms)")
        logger.info(f"   Content-Type: {resp.headers.get('Content-Type', 'unknown')}")
        logger.info(f"   响应体长度: {len(resp.text)} 字符")
        logger.info("")

        # 尝试解析 JSON
        try:
            data = resp.json()
            logger.info("📋 解析后的 JSON 响应:")
            logger.info(json.dumps(data, ensure_ascii=False, indent=2))
            logger.info("")

            # 尝试找出模型输出的字段
            logger.info("🔍 字段扫描:")
            scan_for_text_fields(data)

        except ValueError:  # 兼容 simplejson 环境（其 JSONDecodeError 不一定是 json.JSONDecodeError，但都是 ValueError 子类）
            logger.warning("⚠ 响应不是有效 JSON，原始文本:")
            logger.info(resp.text[:2000])

    except requests.exceptions.SSLError as e:
        logger.error(f"❌ SSL 错误: {e}")
        logger.info("   提示: 尝试用 http:// 而非 https://，或检查证书")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"❌ 连接失败: {e}")
        logger.info("   提示: 检查 PCAP_JUDGE_URL 指向的目标是否可达")
    except requests.exceptions.Timeout:
        logger.error(f"❌ 请求超时 ({TIMEOUT}s)")
    except Exception as e:
        logger.error(f"❌ 未知错误: {type(e).__name__}: {e}")


def scan_for_text_fields(data, prefix=""):
    """递归扫描 JSON 中可能是模型输出的文本字段。"""
    if isinstance(data, dict):
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, str):
                if len(value) > 20:
                    logger.info(f"   📝 {path} (len={len(value)}): {value[:200]}...")
                else:
                    logger.info(f"   📄 {path}: {value}")
            elif isinstance(value, (dict, list)):
                scan_for_text_fields(value, path)
    elif isinstance(data, list) and len(data) > 0:
        # 启发式取舍：列表只下钻第一个元素——响应里的同类列表（choices/candidates）
        # 结构通常一致，扫 [0] 足以定位文本字段，避免大列表刷屏
        scan_for_text_fields(data[0], f"{prefix}[0]")


def main():
    parser = argparse.ArgumentParser(description="目标 API 探测（按 TARGET_TYPE 路由）")
    parser.add_argument("--text", type=str,
                        default="请用一句话介绍人工智能。",
                        help="测试文本 (默认: 请用一句话介绍人工智能。)")
    args = parser.parse_args()
    probe(args.text)


if __name__ == "__main__":
    main()
