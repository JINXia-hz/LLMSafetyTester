"""dashboard_api 路由模块（APIRouter 拆分）。

按职责拆为：
- data_query：只读数据查询 API
- cluster_viz：聚类可视化（投影 / 层次树）API
- tasks：子进程任务管理 API

各模块自持 `router = APIRouter()`，由 dashboard_api 统一 include_router 注册。
"""
