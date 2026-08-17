"""control.agent.menxia — 门下省（全局监听 + 封驳 + 审查）。

三省中唯一常态挂起、全局监听的部门。三项职能：
  - 事前封驳：监听 step_start，按 capability.block_message 判据封驳危险步骤
  - 事后审查：plan_done 时自动审查产生的 run，呈递安全简报
  - 异常呈递：step_failed 时主动报告

子模块：
  block.py    — 封驳令管理（BlockTicket + issue/approve/clear）
  review.py   — 事后审查（读报告 → findings → 摘要）
  listener.py — 总线订阅回调

对外接口（本 __init__ re-export）：
  init_menxia() / reinit_menxia()    初始化总线订阅
  assess_step(cap, args)              封驳审查
  review_run(run_name)                审查某 run
  issue_block / approve_block / ...   封驳令管理
  BlockTicket                         封驳令类型
"""

from control.agent.menxia.block import (
    BlockTicket,
    approve_block,
    clear_all_for_plan,
    clear_block,
    get_block,
    issue_block,
    reset_blocks,
)
from control.agent.menxia.listener import (
    assess_step,
    init_menxia,
    reinit_menxia,
)
from control.agent.menxia.review import (
    assess_findings,
    get_thresholds,
    read_report,
    render_digest,
    review_run,
)

__all__ = [
    # 初始化
    "init_menxia", "reinit_menxia",
    # 封驳审查
    "assess_step",
    # 封驳令管理
    "BlockTicket", "issue_block", "get_block", "clear_block",
    "clear_all_for_plan", "approve_block", "reset_blocks",
    # 事后审查
    "review_run", "read_report", "assess_findings", "render_digest", "get_thresholds",
]
