"""control.agent.gatekeeper — 门下省（事前封驳）。

职责：在尚书省执行前审查，对不可逆/高影响操作（merge 到全局 R、删 R 列）
发劝谏要求用户二次确认。用户确认后发「令牌」放行；拒绝或超时则封驳。

流程：
  1. LLM 决定调某工具 → gatekeeper 检查该工具+参数是否危险
  2. 危险 → 不执行，返回 {blocked, summary, detail, token} 给前端
  3. 前端展示劝谏卡片（「即将 merge 到全局 R，影响 N 条观测，确认？」）
  4. 用户确认 → 前端带 token 再调一次 → gatekeeper 验证 token → 放行执行
  5. 用户拒绝 → 清除 pending_confirm → 不执行

危险操作判据（可配置）：
  - merge 且 target == "global"：合并到全局 R（不可逆地改变全局真相）
  - 任何含 --delete-r / remove_model 的操作：删全局 R 观测（最危险）
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass


@dataclass
class ConfirmTicket:
    """门下省发出的确认令牌。"""
    token: str
    action: str          # "merge" / "delete_r" / ...
    tool_name: str
    tool_args: dict
    summary: str         # 一句话劝谏标题
    detail: str          # 详细影响说明
    created: float

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# 危险操作识别
# ============================================================
def assess(tool_name: str, args: dict) -> dict | None:
    """审查一个工具调用是否危险。

    返回 None = 放行（不危险）；
    返回 dict（含 summary/detail，无 token）= 需要确认（调用方据此发 ConfirmTicket）。
    """
    if tool_name == "merge":
        target = args.get("target", "")
        if target == "global":
            sources = args.get("sources", [])
            models = args.get("models")
            src_str = ", ".join(str(s) for s in sources) if sources else "（未指定）"
            model_str = f"，仅模型 [{', '.join(models)}]" if models else "（全部模型）"
            return {
                "action": "merge_to_global",
                "summary": f"即将把 {src_str} 的观测合并到全局 R 矩阵",
                "detail": (
                    f"目标：全局 R（output/state/results.json，唯一真相）\n"
                    f"来源：{src_str}\n"
                    f"范围{model_str}\n"
                    f"影响：全局 R 将永久累加这些观测，derive_elo 回放会纳入它们。\n"
                    f"这是不可逆的全局状态变更——合并后无法按来源精确剔除。"
                ),
            }

    # delete_runs --delete-r（经 invoker 调 llmsec-manage）——极危险
    if tool_name == "delete_runs":
        if args.get("delete_r"):
            names = args.get("names", [])
            return {
                "action": "delete_r_column",
                "summary": f"即将删除 R 矩阵中 {names} 的观测列",
                "detail": (
                    f"这将永久删除全局 R 中这些模型的全部观测（{names}）。\n"
                    f"derive_elo 对这些模型的回放将失效，相关 Elo/预测器缓存作废。\n"
                    f"属于最危险的操作——删了无法从 R 重算恢复（只能从 .trash 恢复 run 目录，"
                    f"但 R 列本身没有独立备份）。"
                ),
            }

    # clean_cache ——警告级（缓存可重建，但影响性能 + 可能丢失未落盘预测器）
    if tool_name == "clean_cache":
        categories = args.get("categories", [])
        cat_str = ", ".join(categories) if categories else "全部"
        return {
            "action": "clean_cache",
            "summary": f"即将清理缓存类别：{cat_str}",
            "detail": (
                f"这些缓存（{cat_str}）会在下次评估时自动重建，但：\n"
                f"- elo_cache 清除后下次查询需从 R 重算（几秒）\n"
                f"- predictors 清除后下次需重训预测器（几十秒）\n"
                f"- feature_cluster 清除后需重跑特征提取/聚类\n"
                f"删除是软删除（移到 .trash/），可恢复。确认执行？"
            ),
        }

    return None  # 放行


def issue_ticket(tool_name: str, args: dict, assessment: dict) -> ConfirmTicket:
    """对一个危险操作发确认令牌（门下省「封驳待确认」）。"""
    return ConfirmTicket(
        token=uuid.uuid4().hex[:12],
        action=assessment["action"],
        tool_name=tool_name,
        tool_args=args,
        summary=assessment["summary"],
        detail=assessment["detail"],
        created=time.time(),
    )


def is_confirmed(ticket: ConfirmTicket | None, token: str | None) -> bool:
    """验证用户回传的 token 是否匹配待确认令牌。"""
    if ticket is None or token is None:
        return False
    return ticket.token == token
