"""management — llmsec 自我维护能力（信息管理 / 清理 / 快照）。

本包是「代码功能」层：操作 llmsec 自己的数据与产物（run 历史、派生缓存、
结果矩阵 R），暴露为 `python -m llmsec.management` CLI。

设计契约（为未来 agent 化的控制层预留机器友好接口）：
  - 所有 list/export 支持 ``--json`` 结构化输出，默认输出人可读表格。
  - 所有写操作（delete / clean）默认 dry-run 预览，``--yes`` 才真执行。
  - 删除一律走软删除（移到 output/.trash/），可恢复。

不属本层（留给控制层）：fork 新测试环境、历史对比、对话式编排。
"""
