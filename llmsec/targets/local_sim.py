"""
targets.local_sim — 本地模拟模型服务器后端（local_model_server.py）

走本地 OpenAI 兼容服务（由 TARGET_BASE_URL 指向，如 http://127.0.0.1:8000/v1），
调用逻辑与 OpenAI 后端完全相同——原 targets.py 中 local_sim 分支
与默认分支逐字相同，故这里直接继承。

与原实现的唯一差异：meta["backend"] 标记为 "local_sim"（原实现标为 "openai"）。
下游仅判断 backend == "pcap_judge"，此差异无行为影响。
"""

from llmsec.targets.openai_backend import OpenAITargetClient


class LocalSimTargetClient(OpenAITargetClient):
    """本地模拟模型后端（local_model_server.py，默认端口 8000）。"""

    backend_name = "local_sim"
