"""攻击聚类分析 CLI 入口。实现已迁移至 llmsec.clustering.cli，本文件仅为兼容入口。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llmsec.clustering.cli import main

if __name__ == "__main__":
    main()
