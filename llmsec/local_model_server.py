"""本地模拟 LLM 服务（OpenAI 兼容）。实现已迁移至 llmsec.server.local_model_server，本文件仅为兼容入口。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llmsec.server.local_model_server import main

if __name__ == "__main__":
    main()
