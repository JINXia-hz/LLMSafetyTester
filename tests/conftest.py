"""共享测试引导：路径注入 + Windows 控制台 UTF-8。

各 test 模块不再各自 sys.path.insert / reconfigure / setup_console——统一在此完成。
pytest 收集任何 test 模块前会先加载本文件。
"""

import sys
from pathlib import Path

# 让 tests/ 能 import llmsec 包（项目根 = tests/ 的父目录）
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Windows 控制台 UTF-8：emoji / 中文断言与失败信息可正确打印（幂等）
from llmsec.core.logging import setup_console

setup_console()
