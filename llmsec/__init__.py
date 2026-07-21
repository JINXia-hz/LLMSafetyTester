"""
llmsec — LLM 红队攻击测试工具包

由原根目录平铺脚本重构而来的统一包：
  - core/     基础设施：配置、日志、JSONL IO、LLM 客户端、文本工具
  - targets/  目标模型后端：openai / local_sim / pcap_judge
  - （后续阶段）evaluation/ attacks/ clustering/ reporting/ pipeline/ server/

import llmsec 即自动加载项目根目录的 .env（幂等）。
"""

from llmsec.core.config import load_env as _load_env

__version__ = "0.1.0"

# 保证任何模块 `import llmsec.xxx` 时 .env 已加载
_load_env()
