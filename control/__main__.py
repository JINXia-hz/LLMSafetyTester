"""允许 ``python -m control ...`` 调用控制层 CLI。"""

import sys

from control.cli import main

if __name__ == "__main__":
    sys.exit(main())
