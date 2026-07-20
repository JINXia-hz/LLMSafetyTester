#!/usr/bin/env python3
"""
本地模拟小模型服务器（OpenAI 兼容 API）
用于在没有真实 LLM 的情况下测试安全评估框架。

模拟行为：
  - 对明显有害请求 → 安全拒绝
  - 对数学题 → 尝试回答
  - 对一般请求 → 简短回复
  - 有一定概率"越狱成功"（模拟小模型安全能力弱）

用法：
    python local_model_server.py                    # 默认端口 8000
    python local_model_server.py --port 8000 --refusal-rate 0.6  # 60%拒绝率

然后用 evaluate.py：
    python evaluate.py --max-samples 10 --no-judge
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import math

# ============================================================
# 参数
# ============================================================
parser = argparse.ArgumentParser(description="本地模拟小模型服务器")
parser.add_argument("--port", type=int, default=8000, help="监听端口")
parser.add_argument("--refusal-rate", type=float, default=0.70,
                    help="对有害请求的拒绝率 (0-1)，默认 0.70，模拟弱安全模型")
parser.add_argument("--math-accuracy", type=float, default=0.55,
                    help="数学题正确率 (0-1)，默认 0.55，模拟越狱后推理退化")
args = parser.parse_args()

REFUSAL_RATE = max(0.0, min(1.0, args.refusal_rate))
MATH_ACCURACY = max(0.0, min(1.0, args.math_accuracy))

# ============================================================
# 拒绝/有害相关词汇表
# ============================================================
HARMFUL_KEYWORDS = {
    "暴力": 0.9, "杀人": 0.95, "炸弹": 0.95, "武器": 0.8, "伤害": 0.85,
    "毒品": 0.95, "制毒": 0.98, "海洛因": 0.95, "冰毒": 0.95,
    "黑客": 0.7, "入侵": 0.85, "病毒": 0.65, "木马": 0.8, "攻击": 0.6,
    "诈骗": 0.9, "假钞": 0.95, "洗钱": 0.95,
    "自杀": 0.95, "自残": 0.98,
    "歧视": 0.85, "种族": 0.75,
    "色情": 0.9, "裸体": 0.9,
    "制造": 0.5, "制作": 0.5, "怎么": 0.4, "如何": 0.4,
    "explosive": 0.95, "hack": 0.8, "kill": 0.95, "drug": 0.9,
    "bomb": 0.95, "weapon": 0.85, "poison": 0.9,
    "suicide": 0.95, "porn": 0.9, "illegal": 0.8,
    "how to make": 0.7, "how to create": 0.7,
}

DANGEROUS_PATTERNS = [
    r"如何(制造|制作|合成|提取|获取)",
    r"(制造|制作)(炸药|炸弹|枪支|毒药)",
    r"怎么(杀人|放火|偷|抢|骗|黑)",
    r"(黑客|入侵|破解).{0,5}(系统|账号|密码|服务器|网站)",
    r"how\s+to\s+(make|build|create|hack|steal|kill|manufacture)",
    r"(step|steps|guide|tutorial).{0,20}(hack|bomb|drug|weapon|attack)",
]

SAFE_REFUSAL_TEMPLATES = [
    "抱歉，我无法提供这类信息。这涉及不安全的内容。",
    "对不起，我是一个AI助手，不能回答这个问题。",
    "I'm sorry, but I cannot assist with this request as it may involve harmful content.",
    "I cannot provide information on this topic as it goes against my safety guidelines.",
    "作为一个AI助手，我无法提供可能造成伤害的建议。",
    "Sorry, I'm not able to help with that. Let me know if you have another question.",
    "这个问题涉及到不安全的领域，我无法回答。请提出其他问题。",
    "I apologize, but I cannot generate content that might be used for harmful purposes.",
    "我不能提供这类信息。请考虑其他合法的替代方案。",
    "作为安全负责任的AI，我无法协助这个请求。",
]

SAFE_HELP_TEMPLATES = [
    "当然！这是关于{}的一些信息：",
    "好的，让我来帮你解答这个问题。",
    "Sure! Here's some information about that:",
    "Here's what I can tell you:",
    "Good question! Let me explain.",
    "没问题，以下是相关信息：",
]

HARMFUL_COMPLIANCE_TEMPLATES = [
    "好的，以下是关于{}的详细步骤：\n\n第一步，你需要准备以下材料...\n第二步...\n第三步...",
    "Sure, here's a detailed guide on how to do that:\n\nStep 1: Gather the necessary materials...\nStep 2: ...\nStep 3: ...",
    "以下是制作{}的方法：\n\n首先，获取以下原料...\n然后，按照以下步骤操作...",
    "I can help explain that. First, you'll need to...",
    "关于这个问题，我给你详细说明一下：\n1. 准备工作...\n2. 操作步骤...\n3. 注意事项...",
    "Here's what you're looking for:\n\n- Materials needed: ...\n- Procedure: ...\n- Notes: ...",
]


# ============================================================
# 有害度评估
# ============================================================
def evaluate_harmfulness(text: str) -> float:
    """计算文本的有害度分数 (0~1)。"""
    text_lower = text.lower()
    score = 0.0
    matched = 0

    # 关键词匹配
    for keyword, weight in HARMFUL_KEYWORDS.items():
        if keyword.lower() in text_lower:
            score = max(score, weight)
            matched += 1

    # 危险模式匹配
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, text_lower):
            score = max(score, 0.85)
            matched += 1

    # 多关键词惩罚
    if matched >= 3:
        score = min(1.0, score + 0.1)
    if matched >= 5:
        score = min(1.0, score + 0.15)

    # 中文"谢谢"/"你好"等安全词汇降低分数
    safe_signals = ["谢谢", "你好", "请问", "help", "thanks", "hi", "hello", "what is"]
    safe_count = sum(1 for s in safe_signals if s in text_lower)
    if safe_count >= 2 and matched < 2:
        score *= 0.3

    return round(score, 4)


def detect_math_question(text: str) -> tuple[bool, str, int | None]:
    """
    检测数学题并提取答案格式。
    返回 (is_math, expression, expected_answer)。
    """
    # 检测 [MATH:数字] 格式
    m = re.search(r"\[MATH:\s*(-?\d+)\s*\]", text, re.IGNORECASE)
    if m:
        return True, "", int(m.group(1))

    # 检测自然语言数学题
    math_patterns = [
        r"(\d+)\s*\+\s*(\d+)\s*=\s*\?",
        r"(\d+)\s*\-\s*(\d+)\s*=\s*\?",
        r"(\d+)\s*\*\s*(\d+)\s*=\s*\?",
        r"(\d+)\s*\+\s*(\d+)等于",
        r"计算[:：]\s*(\d+)\s*\+\s*(\d+)",
        r"计算[:：]\s*(\d+)\s*\-\s*(\d+)",
    ]
    for pattern in math_patterns:
        m = re.search(pattern, text)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if "+" in pattern or "加" in pattern:
                return True, f"{a}+{b}", a + b
            elif "-" in pattern or "减" in pattern:
                return True, f"{a}-{b}", a - b
            elif "*" in pattern or "乘" in pattern:
                return True, f"{a}×{b}", a * b

    return False, "", None


# ============================================================
# 响应生成
# ============================================================
def generate_response(user_message: str) -> str:
    """
    模拟一个小型安全 LLM 的响应。
    响应策略：
      1. 检测数学题 → 回答（带退化概率）
      2. 评估有害度 → 高于阈值按 REFUSAL_RATE 拒绝
      3. 否则正常回复
    """
    harm_score = evaluate_harmfulness(user_message)
    is_math, math_expr, expected = detect_math_question(user_message)

    # ---- 数学题处理 ----
    if is_math and expected is not None:
        # 模拟越狱税：正确答案概率随 harm_score 增加而降低
        effective_accuracy = MATH_ACCURACY * (1.0 - harm_score * 0.5)
        if random.random() < effective_accuracy:
            return f"计算结果为：{expected}\n\n[MATH:{expected}]"
        else:
            wrong = expected + random.choice([-2, -1, 1, 2, 3])
            return f"计算结果为：{wrong}\n\n[MATH:{wrong}]"

    # ---- 安全拒绝 ----
    if harm_score > 0.5:
        if random.random() < REFUSAL_RATE:
            return random.choice(SAFE_REFUSAL_TEMPLATES)

        # 越狱成功：返回详细有害内容 + 可能带数学题答案
        topic = _extract_topic(user_message)
        response = random.choice(HARMFUL_COMPLIANCE_TEMPLATES).format(topic or "这个")
        # 如果有数学题嵌入，处理它
        if expected is not None:
            if random.random() < MATH_ACCURACY:
                response += f"\n\n[MATH:{expected}]"
            else:
                response += f"\n\n[MATH:{expected + random.choice([-1, 1])}]"
        return response

    # ---- 低有害度：正常回复 ----
    if harm_score < 0.2:
        return _generate_safe_response(user_message)

    # ---- 模糊区域：随机 ----
    if random.random() < 0.5:
        return random.choice(SAFE_REFUSAL_TEMPLATES)
    else:
        return _generate_safe_response(user_message)


def _extract_topic(text: str) -> str:
    """从文本中提取主题词。"""
    # 简单提取：取"如何"或"how to"后面的内容
    m = re.search(r"(?:如何|怎么|how\s+to)\s*(.+?)(?:[？?\n]|$)", text, re.IGNORECASE)
    if m:
        return m.group(1)[:20]
    # 取前几个词
    words = text.replace("\n", " ").split()
    return " ".join(words[:3]) if words else "这个"


def _generate_safe_response(text: str) -> str:
    """生成安全的正常回复。"""
    text_lower = text.lower()

    greetings = {
        "hi": "Hello! How can I help you today?",
        "hello": "Hi there! What can I do for you?",
        "你好": "你好！有什么我可以帮助你的吗？",
        "hey": "Hey! What's up?",
    }
    for g, reply in greetings.items():
        if text_lower.startswith(g):
            return reply

    # 一般回复
    general_replies = [
        "这是一个有趣的问题。根据我的理解...",
        "好的，让我来帮你分析一下这个问题。",
        "That's an interesting topic. Here's what I can share...",
        "I understand your question. Let me provide some useful information.",
        "关于这个问题，我的理解是：",
        "Sure, I can help with that. Here's some general information:",
    ]
    return random.choice(general_replies) + "\n\n" + _generate_generic_content(text)


def _generate_generic_content(text: str) -> str:
    """生成通用无害内容。"""
    generic = [
        "这是一个需要多方面考虑的话题。从不同角度来看，情况可能会有所不同。",
        "This topic involves several aspects that are worth considering carefully.",
        "根据一般知识，这个问题可以从以下几个角度来理解：\n1. 基本概念\n2. 实际应用\n3. 注意事项",
        "Here are some general points to consider:\n- Point 1\n- Point 2\n- Point 3",
    ]
    return random.choice(generic)


# ============================================================
# 统计
# ============================================================
request_count = 0
refusal_count = 0
comply_count = 0


# ============================================================
# FastAPI 服务
# ============================================================
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Local Simulated LLM", version="1.0")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "local-model"
    messages: list[Message]
    temperature: float = 0.0
    max_tokens: int = 512
    stream: bool = False


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "local-simulated-model", "object": "model", "owned_by": "local"}],
    }


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(req: ChatRequest):
    global request_count, refusal_count, comply_count

    # 获取最后一条用户消息
    user_messages = [m.content for m in req.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")
    user_content = user_messages[-1]

    # 生成响应
    request_count += 1
    response_text = generate_response(user_content)

    # 统计
    is_refusal = any(
        phrase in response_text
        for phrase in ["抱歉", "对不起", "I'm sorry", "I cannot", "无法提供", "不能提供"]
    )
    if is_refusal:
        refusal_count += 1
    else:
        comply_count += 1

    prompt_tokens = sum(len(m.content) for m in req.messages) // 2
    completion_tokens = len(response_text) // 2

    return ChatResponse(
        id=f"chatcmpl-{int(time.time() * 1000)}",
        created=int(time.time()),
        model=req.model,
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=response_text),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "local-simulated-model",
        "refusal_rate": REFUSAL_RATE,
        "math_accuracy": MATH_ACCURACY,
        "stats": {
            "requests": request_count,
            "refusals": refusal_count,
            "complied": comply_count,
        },
    }


@app.get("/stats")
async def stats():
    return {
        "requests": request_count,
        "refusals": refusal_count,
        "complied": comply_count,
        "refusal_rate_actual": refusal_count / max(request_count, 1),
        "config_refusal_rate": REFUSAL_RATE,
        "config_math_accuracy": MATH_ACCURACY,
    }


# ============================================================
# 启动
# ============================================================
if __name__ == "__main__":
    print(f"""
{'='*60}
  本地模拟小模型服务器
{'='*60}
  端口:     {args.port}
  拒绝率:   {REFUSAL_RATE:.0%}（模拟弱安全模型）
  数学正确率: {MATH_ACCURACY:.0%}（模拟越狱后推理退化）
  
  API 端点: http://localhost:{args.port}/v1/chat/completions
  健康检查: http://localhost:{args.port}/health
  统计信息: http://localhost:{args.port}/stats
{'='*60}

启动后，在另一个终端运行：
  python evaluate.py --max-samples 10 --no-judge

按 Ctrl+C 停止
""")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")