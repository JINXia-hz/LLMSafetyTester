"""llmsec.storage — 统一存储层（DAO 收口，SQLite/SQLModel）。

2026-08 数据库重构引入。此前"发现 run"有 3 份目录扫描实现、字段提取 2 套、
原子写 2 套、锁 2 套、路径配置 3 处——本包是收口后的唯一数据访问层：

  db.py       引擎/会话/事务基建（WAL + BEGIN IMMEDIATE；唯一握 ORM 的地方）
  models.py   表模型（SQLModel：模型即 schema）+ 旧 dict 口径的 as_dict()
  catalog.py  runs/trials/tasks 三张表的读写实现 + 增量对账（reconcile）
  contract.py 唯一公开契约（service 层与 control 层都从这里进）

分层约定（"严谨"落在机器上，不靠命名自觉）：
  - service 层只 import llmsec.storage.contract，不碰 SQL/引擎/连接；
  - AST 守卫禁止包外 import sqlite3/sqlalchemy/sqlmodel（见 test_audit_r9_guard）；
  - 目录库是**可重建的派生索引**（真相在文件），删库后 `storage reindex` 重建。

阶段 2（rstore.py：R 矩阵 observations 表，ResultsMatrix 后端切换）另行落地。
"""

from llmsec.storage import catalog, contract, db, models

__all__ = ["catalog", "contract", "db", "models"]
