"""control — llmsec 的元控制层（编排 / fork / 对比 / agent）。

本包是「代码外面」的控制层：把 llmsec 当作独立工作单元，通过 subprocess 调用
llmsec CLI（``llmsec`` / ``llmsec-manage``），只读 llmsec 的公开输出产物文件。
**绝不 import llmsec 内部 API**——隔离边界落在「文件 + CLI」。

分层（与 llmsec 本体解耦）：
  - core/invoker.py    subprocess 调 llmsec CLI 的封装（跑 run / 导出快照 / 列 run）
  - core/workspace.py  fork 编排（快照→复制→work-dir→起 run）
  - core/compare.py    历史对比（读 runner_report.json 聚合）
  - core/orchestrator.py  批量并行编排
  - agent/tools.py     tool schema 定义（供 agent / 外部调用）
  - agent/loop.py      最小对话循环
  - cli.py             命令入口
  - config.py          定位 llmsec（python 解释器、仓库根、output 路径）

设计原则：
  - 控制层只调度/审视，不做 llmsec 内部数据操作（删 R 列等留给 llmsec-manage）
  - 所有对 llmsec 的调用经 invoker，便于 mock 与观测
  - 输出结构化（dict / JSON），供 agent 与人双消费
"""

__version__ = "0.1.0"
