"""llmsec.core.probe — 目标/服务探活的共享核心（同步原语）。

此前 dashboard（routers/data_query.py 的 async 版）与 MCP（mcp/tools/query.py
的同步版）各自维护一份几乎逐行重复的两段式探活实现，文案与行为已开始漂移。
本模块抽出**同步核心**：
  - probe_target(name, cfg)：目标两段式探活（models.list + chat smoke，
    pcap judge 走 HTTP GET；chat 401/403 判不可达）
  - probe_service(svc_name, cfg, models_result=None)：generator/judge 探活
    （models_result 允许调用方跨 service 复用第一段结果）
  - models_list() / chat_smoke()：单段原语
异步调用方（dashboard）用 asyncio.to_thread 包一层；同步调用方（MCP）直接调。
"""
from __future__ import annotations

import time
from typing import NamedTuple


class ModelsProbeResult(NamedTuple):
    """models.list 第一段探活的结果（跨 service 复用时的传递单元）。"""
    latency_ms: float | None
    model_ids: list[str] | None
    error: str | None


def models_list(api_key: str, base_url: str, timeout: float = 5.0):
    """第一段：models.list（OpenAI 兼容端点的最轻量 GET，不耗 token）。

    Returns:
        (latency_ms, [model_id, ...])；失败向上抛（调用方决定判不可达/降级）。
    """
    from llmsec.core.llm import create_openai_client

    t0 = time.time()
    client = create_openai_client(api_key, base_url, timeout=timeout)
    ids = [m.id for m in client.models.list()]
    return round((time.time() - t0) * 1000), ids


def chat_smoke(api_key: str, base_url: str, model: str,
               timeout: float = 12.0, max_tokens: int = 64):
    """第二段：最小 chat（max_tokens=64——部分模型需 ~64 tokens 才开始输出 content，
    预算太小会被截断成 content=None 制造假警报）。

    Returns:
        (content, reasoning_content, finish_reason)；失败向上抛。
    """
    from llmsec.core.llm import create_openai_client

    client = create_openai_client(api_key, base_url, timeout=timeout)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=max_tokens,
    )
    msg = resp.choices[0].message
    return (getattr(msg, "content", None),
            getattr(msg, "reasoning_content", None),
            resp.choices[0].finish_reason)


def _model_warning(model_ids, model: str) -> str | None:
    """models.list 非空且配置模型不在其中 → warning（不判不可达）。"""
    if model_ids and model and model not in model_ids:
        return f"模型 {model} 不在端点列表"
    return None


def _empty_content_warning(content, reasoning, finish) -> str | None:
    """chat 返回空 content 的三种情况分类（均良性提示，不判不可达）。"""
    if content is not None:
        return None
    if reasoning:
        return "推理模型：content 为空但 reasoning_content 有内容（已自动回退读取，评估正常）"
    if finish == "length":
        return "chat 探活预算不足被截断（content 为空），真实业务请求不受影响"
    return "chat 返回空 content 且无 reasoning_content（疑似配置/鉴权问题，需确认）"


def probe_target(name: str, cfg) -> dict:
    """目标探活（两段式，同步）。

    - pcap_judge 后端：非 OpenAI 兼容，HTTP GET 探通即可（无第二段）。
    - OpenAI 兼容：models.list 通 → 再发 chat smoke。chat 鉴权失败（401/403）
      判不可达（历史教训：探活亮绿灯、运行全线 401 → ASR=0 假阴性）；
      其他 chat 异常（限流/5xx/截断/空 content）仅 warning 不阻塞。
    """
    from llmsec.targets import target_backend

    backend = target_backend(name)
    ids = None
    try:
        if backend == "pcap_judge":
            # r7/M-3：pcap 后端实际评估端点是 PCAP_JUDGE_URL（create_target_client
            # 对 pcap 完全忽略 cfg），探活必须探同一端点——探 cfg.base_url
            # （OpenAI 型地址）时探活结论与评估所用端点根本不是同一个
            from llmsec.targets.pcap import pcap_judge_url, pcap_verify_tls, suppress_insecure_warning
            _pcap_url = pcap_judge_url()
            if not _pcap_url:
                return {"name": name, "model": cfg.model, "reachable": False,
                        "latency_ms": None,
                        "error": "PCAP_JUDGE_URL 未配置", "warning": None}
            import requests
            _verify = pcap_verify_tls()
            if not _verify:
                suppress_insecure_warning()
            t0 = time.time()
            r = requests.get(_pcap_url, timeout=5, verify=_verify)
            r.raise_for_status()
            latency = round((time.time() - t0) * 1000)
        else:
            latency, ids = models_list(cfg.api_key, cfg.base_url, timeout=5.0)
    except Exception as e:
        return {"name": name, "model": cfg.model, "reachable": False,
                "latency_ms": None, "error": str(e)[:120], "warning": None}

    warnings = []
    w = _model_warning(ids, cfg.model)
    if w:
        warnings.append(w)
    if backend != "pcap_judge":
        try:
            content, reasoning, finish = chat_smoke(cfg.api_key, cfg.base_url, cfg.model)
            w = _empty_content_warning(content, reasoning, finish)
            if w:
                warnings.append(w)
        except Exception as e:
            status = getattr(e, "status_code", None)
            if status in (401, 403):
                # 鉴权失败：chat 一定全线阵亡，判不可达让前端拦截，防白跑
                return {"name": name, "model": cfg.model, "reachable": False,
                        "latency_ms": latency,
                        "error": f"chat 鉴权失败({status}): {str(e)[:120]}",
                        "warning": None}
            # 非 401/403（限流/5xx/超时等）：models.list 已通，不阻塞，仅 warning
            warnings.append(f"chat 探测失败（不阻塞）: {str(e)[:120]}")
    return {"name": name, "model": cfg.model, "reachable": True,
            "latency_ms": latency, "error": None,
            "warning": "；".join(warnings) or None}


def probe_service(svc_name: str, cfg, models_result: ModelsProbeResult | None = None) -> dict:
    """generator/judge 探活（两段式，同步）。

    models_result：调用方可预先算好的第一段结果（跨 service 复用同端点的
    models.list）；None 则现算。chat 错误不判不可达（models.list 已通，
    偶发抖动不应阻塞），仅 warning。
    """
    if models_result is not None:
        latency, ids, err = models_result.latency_ms, models_result.model_ids, models_result.error
    else:
        try:
            latency, ids = models_list(cfg.api_key, cfg.base_url, timeout=5.0)
            err = None
        except Exception as e:
            latency, ids, err = None, None, str(e)[:120]
    if err is not None:
        return {"name": svc_name, "model": cfg.model, "reachable": False,
                "latency_ms": None, "error": err, "warning": None}

    warnings = []
    w = _model_warning(ids, cfg.model)
    if w:
        warnings.append(w)
    try:
        content, reasoning, finish = chat_smoke(cfg.api_key, cfg.base_url, cfg.model)
        w = _empty_content_warning(content, reasoning, finish)
        if w:
            warnings.append(w)
    except Exception as e:
        warnings.append(f"chat 探测失败：{str(e)[:120]}")
    return {"name": svc_name, "model": cfg.model, "reachable": True,
            "latency_ms": latency, "error": None,
            "warning": "；".join(warnings) or None}
