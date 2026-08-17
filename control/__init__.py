"""control — llmsec 的元控制层（编排 / fork / 对比 / agent）。

本包是「代码外面」的控制层：把 llmsec 当作独立工作单元，通过 subprocess 调用
llmsec CLI（``llmsec`` / ``llmsec-manage``），主要交互经「文件 + CLI」边界。
（r7 修正：并非绝对不 import llmsec——共享的纯工具层除外，如
``core/paths.py`` 复用 ``llmsec.core.paths`` 的路径校验、``core/workspace.py``
复用 ``llmsec.management.common.dir_size``；业务数据操作仍全部走 CLI/文件。）

分层（与 llmsec 本体解耦）：
  - core/invoker.py    subprocess 调 llmsec CLI 的封装（跑 run / 导出快照 / 列 run）
  - core/workspace.py  fork 编排（快照→复制→work-dir→起 run）
  - core/compare.py    历史对比（读 runner_report.json 聚合）
  - core/orchestrator.py  批量并行编排
  - agent/zhongshu/    中书省（对话入口 + 意图分流 + 最小对话循环 fallback）
  - agent/shangshu/    尚书省（拟案 + 执行调度，capabilities.py 为能力清单）
  - agent/menxia/      门下省（封驳 + 审查）
  - cli.py             命令入口
  - config.py          定位 llmsec（python 解释器、仓库根、output 路径）

设计原则：
  - 控制层只调度/审视，不做 llmsec 内部数据操作（删 R 列等留给 llmsec-manage）
  - 所有对 llmsec 的调用经 invoker，便于 mock 与观测
  - 输出结构化（dict / JSON），供 agent 与人双消费
"""

__version__ = "0.1.0"
