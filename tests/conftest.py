"""共享测试引导：路径注入 + Windows 控制台 UTF-8 + 网络隔离。

各 test 模块不再各自 sys.path.insert / reconfigure / setup_console——统一在此完成。
pytest 收集任何 test 模块前会先加载本文件。
"""

import os
import sys
from pathlib import Path

import pytest

# 让 tests/ 能 import llmsec 包（项目根 = tests/ 的父目录）
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Windows 控制台 UTF-8：emoji / 中文断言与失败信息可正确打印（幂等）
from llmsec.core.logging import setup_console

setup_console()

# 预先把 .env 灌进 os.environ。config.load_env() 有 _ENV_LOADED 幂等保护，
# 这里先调一次：把仓库 .env 的内网地址提前注入，后续测试内再调 from_env() 时
# load_env() 会直接返回、不会重新注入——这样下面的 _isolate_network_env fixture
# 才能可靠地 pop 掉它们（否则 fixture 跑在 load_env() 之前，pop 个空，测试函数
# 内部 from_env() 一调又把内网地址灌回来了）。
from llmsec.core.config import load_env

load_env()


# 触发真实网络调用的环境变量前缀/名单。
# 这些变量在 .env 里通常配的是内网服务地址（embedding 服务 / 生成模型 / judge），
# 测试环境不可达时会让隐式走网络路径的测试（如 run_attack_phase 内部的 embedding
# 探活、ai_rename_clusters 的 LLM 调用）卡满 HTTP 超时（~60-120s/次）才降级。
# 测试期间统一清空，让对应代码走"未配置→早退"分支，保持套件离线、秒级。
_NETWORK_ENV_KEYS = (
    "EMBEDDING_API_BASE", "EMBEDDING_API_KEY", "EMBEDDING_API_MODEL",
    "GENERATOR_API_KEY", "GENERATOR_BASE_URL",
    "JUDGE_API_KEY",
)


@pytest.fixture(autouse=True)
def _isolate_network_env():
    """每个测试期间清空网络相关环境变量，结束后恢复，避免 .env 内网地址泄漏进测试。"""
    saved = {k: os.environ.pop(k) for k in _NETWORK_ENV_KEYS if k in os.environ}
    yield
    os.environ.update(saved)
