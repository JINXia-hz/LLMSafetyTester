"""全量手动评估器。实现已迁移至 llmsec.evaluation.evaluator，本文件仅为兼容入口。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llmsec.evaluation.evaluator import main

if __name__ == "__main__":
    main()
