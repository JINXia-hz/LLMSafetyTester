"""control.agent.menxia — 门下省（监听者 + 封驳 + 审查）。

重构后的门下省不再是被动调用的函数，而是**总线订阅者**：
  - 常态挂起，监听三省所有动作。
  - step_start 时审查：dangerous capability → 发 KIND_BLOCK 消息（附 ticket）。
  - plan_done 时自动审查：对产生的 run 跑 review，呈递简报经总线推到中书省面板。
  - step_failed 时呈递异常简报。

封驳粒度（经用户确认的设计）：
  - 只挡该步，不依赖它的步骤继续执行（executor._propagate_blockage 处理依赖链）。
  - 用户准奏 → 清该步的 ticket → executor 重入时重试该步。

封驳判据：按 capability + risk_level 判断，覆盖 merge 到全局 / 删 R 列 / clean_cache /
改全局 .env / 跑评估等场景。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass

from control.agent.bus import (
    ALL,
    KIND_BLOCK,
    KIND_PLAN_DONE,
    KIND_REVIEW,
    KIND_STEP_FAILED,
    KIND_STEP_START,
    MENXIA,
    ZHONGSHU,
    BusMessage,
    get_bus,
)


@dataclass
class BlockTicket:
    """门下省封驳令（附在某步骤上，待用户确认）。"""
    token: str
    plan_id: str
    step_id: str
    capability: str
    risk_level: str
    summary: str         # 一句话劝谏标题
    detail: str          # 详细影响说明
    created: float

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# 危险判据（基于 capability + risk_level + 具体参数）
# ============================================================
def assess_step(capability: str, args: dict, risk_level: str) -> dict | None:
    """审查一个步骤是否需要封驳。

    返回 None = 放行；返回 dict（含 summary/detail）= 需封驳。
    判据：risk_level >= high 的都封驳；某些 low/medium 在特定参数下也封驳。
    """
    # critical 级必封驳
    if risk_level == "critical":
        return _critical_assessment(capability, args)

    # high 级封驳（跑评估/删 run）
    if risk_level == "high":
        return _high_assessment(capability, args)

    # medium 级（清缓存）封驳提示
    if risk_level == "medium":
        return _medium_assessment(capability, args)

    return None  # low 放行


def _critical_assessment(capability: str, args: dict) -> dict:
    if capability == "merge_results" and args.get("target") == "global":
        sources = args.get("sources", [])
        src_str = ", ".join(str(s) for s in sources) if sources else "（未指定）"
        models = args.get("models")
        model_str = f"，仅模型 [{', '.join(models)}]" if models else "（全部模型）"
        return {
            "summary": f"即将把 {src_str} 的观测合并到全局 R 矩阵",
            "detail": (
                f"目标：全局 R（output/state/results.json，唯一真相）\n"
                f"来源：{src_str}\n范围{model_str}\n"
                f"全局 R 将永久累加这些观测，不可按来源精确剔除。"
                f"这是不可逆的全局状态变更。"
            ),
        }
    if capability == "merge_results":
        # target=ws:<name>：分支融合，降到 high 级提示
        target = args.get("target", "")
        sources = args.get("sources", [])
        return {
            "summary": f"即将把 {sources} 合并到 {target}",
            "detail": f"分支融合：{sources} → {target}。目标工作区的 R 将被覆盖更新。",
        }
    if capability == "delete_runs" and args.get("delete_r"):
        names = args.get("names", [])
        return {
            "summary": f"即将删除 R 矩阵中 {names} 的观测列",
            "detail": (
                f"永久删除全局 R 中这些模型的全部观测（{names}）。\n"
                f"derive_elo 对这些模型的回放将失效。属于最危险操作——不可从 R 重算恢复。"
            ),
        }
    if capability == "merge_env_to_global":
        name = args.get("name", "")
        return {
            "summary": f"即将把 .env 快照「{name}」写回全局 .env",
            "detail": (
                "全局 .env 会被覆盖（先备份到 .env.bak.<ts>）。"
                "改全局连接配置会让所有后续 run 受影响——目标模型/judge/参数全部变更。"
                "快照里有的 key 覆盖全局同名 key；快照里没有的不动。"
            ),
        }
    # 其他 critical 兜底
    return {
        "summary": f"即将执行 critical 级操作：{capability}",
        "detail": f"该操作（{capability}）被标记为 critical，风险极高。请确认。",
    }


def _high_assessment(capability: str, args: dict) -> dict:
    if capability == "run_evaluation":
        targets = args.get("targets") or [args.get("target", "?")]
        max_rounds = args.get("max_rounds", 5)
        return {
            "summary": f"即将对 {targets} 跑评估（{max_rounds} 轮自适应）",
            "detail": (
                f"将消耗 API 额度（调目标模型 + judge），{max_rounds} 轮 × batch_size 次。"
                f"产物落在隔离 work-dir，不污染全局。确认执行？"
            ),
        }
    if capability == "run_batch_experiment":
        specs = args.get("specs", [])
        workers = args.get("max_workers", 2)
        return {
            "summary": f"即将批量跑 {len(specs)} 个实验（并行度 {workers}）",
            "detail": (
                f"{len(specs)} 个 spec 各起隔离 workspace + runner，是 API 开销大头。"
                f"完成后自动对比 + 审查。确认执行？"
            ),
        }
    if capability == "delete_runs":
        names = args.get("names", [])
        delete_r = args.get("delete_r", False)
        if delete_r:
            # delete_r=True 升级为 critical 判据（删全局 R 观测不可恢复）
            return _critical_assessment(capability, args)
        return {
            "summary": f"即将删除 {len(names)} 个 run",
            "detail": f"软删除到 .trash/（可恢复）。目标：{names}。",
        }
    # merge_results target=ws 也在 high
    if capability == "merge_results":
        return _critical_assessment(capability, args)  # 复用上面的 ws 分支
    return {
        "summary": f"即将执行 high 级操作：{capability}",
        "detail": f"该操作（{capability}）有一定风险，请确认。",
    }


def _medium_assessment(capability: str, args: dict) -> dict:
    if capability == "clean_cache":
        cats = args.get("categories", [])
        cat_str = ", ".join(cats) if cats else "全部"
        return {
            "summary": f"即将清理缓存：{cat_str}",
            "detail": (
                f"这些缓存（{cat_str}）下次评估时自动重建，但：\n"
                f"- elo_cache 清后下次查询需从 R 重算（几秒）\n"
                f"- predictors 清后下次需重训预测器（几十秒）\n"
                f"软删除到 .trash/，可恢复。"
            ),
        }
    if capability == "edit_env_snapshot":
        name = args.get("name", "")
        key = args.get("key", "")
        return {
            "summary": f"即将改 .env 快照「{name}」的 {key}",
            "detail": f"修改隔离快照内的 {key}。不影响全局 .env，只影响引用此快照的 run。",
        }
    return {
        "summary": f"即将执行 medium 级操作：{capability}",
        "detail": f"该操作（{capability}）有轻微副作用，请确认。",
    }


# ============================================================
# 封驳令牌管理（内存，按 plan_id+step_id 索引）
# ============================================================
# _TICKETS: key=(plan_id, step_id) → BlockTicket
_TICKETS: dict[tuple[str, str], BlockTicket] = {}


def issue_block(plan_id: str, step_id: str, capability: str, risk_level: str,
                assessment: dict) -> BlockTicket:
    """对一个危险步骤发封驳令。"""
    ticket = BlockTicket(
        token=uuid.uuid4().hex[:12],
        plan_id=plan_id, step_id=step_id,
        capability=capability, risk_level=risk_level,
        summary=assessment["summary"], detail=assessment["detail"],
        created=time.time(),
    )
    _TICKETS[(plan_id, step_id)] = ticket
    return ticket


def get_block(plan_id: str, step_id: str) -> BlockTicket | None:
    return _TICKETS.get((plan_id, step_id))


def clear_block(plan_id: str, step_id: str) -> bool:
    """用户准奏后清除封驳（让 executor 重试该步）。返回是否清除成功。"""
    return _TICKETS.pop((plan_id, step_id), None) is not None


def clear_all_for_plan(plan_id: str) -> int:
    """清除某 Plan 的所有封驳（plan 驳回/重置时）。"""
    keys = [k for k in _TICKETS if k[0] == plan_id]
    for k in keys:
        del _TICKETS[k]
    return len(keys)


def approve_block(plan_id: str, step_id: str) -> bool:
    """用户准奏某步的封驳（= clear_block，executor 重入时重试）。"""
    return clear_block(plan_id, step_id)


def list_pending_blocks() -> list[dict]:
    """列出所有待确认封驳（供前端展示）。"""
    return [t.to_dict() for t in _TICKETS.values()]


def reset_blocks() -> None:
    """清空所有封驳（测试用）。"""
    _TICKETS.clear()


# ============================================================
# 总线订阅（门下省挂起监听）
# ============================================================
_INITIALIZED = False


def init_menxia() -> None:
    """初始化门下省：订阅总线消息。进程启动时调一次。

    幂等：重复调用安全（已初始化则跳过）。
    测试用 reset_bus() 后需调 reinit_menxia() 重新订阅新总线。

    订阅：
      - step_start → 封驳审查（dangerous → 发 KIND_BLOCK）
      - plan_done → 自动审查呈递简报
      - step_failed → 异常呈递
    """
    global _INITIALIZED
    if _INITIALIZED:
        return
    bus = get_bus()
    bus.subscribe(MENXIA, [KIND_STEP_START], _on_step_start)
    bus.subscribe(MENXIA, [KIND_PLAN_DONE], _on_plan_done)
    bus.subscribe(MENXIA, [KIND_STEP_FAILED], _on_step_failed)
    _INITIALIZED = True


def reinit_menxia() -> None:
    """强制重新订阅（测试 reset_bus 后用）。"""
    global _INITIALIZED
    _INITIALIZED = False
    init_menxia()


def _on_step_start(msg: BusMessage) -> None:
    """步骤开始前审查。dangerous → 发封驳令 + 发 KIND_BLOCK 消息。"""
    payload = msg.payload
    capability = payload.get("capability", "")
    risk_level = payload.get("risk_level", "low")
    plan_id = payload.get("plan_id", "")
    step_id = payload.get("step_id", "")
    args = payload.get("args", {})

    assessment = assess_step(capability, args, risk_level)
    if assessment is None:
        return  # 放行

    # 已有封驳令且未被清除 → 保持封驳
    existing = get_block(plan_id, step_id)
    if existing is not None:
        # 尚未准奏，重发 block 让 executor 看到
        bus = get_bus()
        bus.publish(BusMessage(
            from_dept=MENXIA, to_dept=ALL, kind=KIND_BLOCK,
            payload={"plan_id": plan_id, "step_id": step_id,
                     "ticket": existing.to_dict()},
        ))
        return

    ticket = issue_block(plan_id, step_id, capability, risk_level, assessment)
    bus = get_bus()
    bus.publish(BusMessage(
        from_dept=MENXIA, to_dept=ALL, kind=KIND_BLOCK,
        payload={"plan_id": plan_id, "step_id": step_id, "ticket": ticket.to_dict()},
    ))


def _on_plan_done(msg: BusMessage) -> None:
    """Plan 执行完，自动审查所有产生的 run，呈递简报到中书省面板。"""
    payload = msg.payload
    plan_id = payload.get("plan_id", "")

    # 找出 run_evaluation / run_batch_experiment 产生的 run
    # 简化：对每个 done 的 run_evaluation 步骤尝试审查
    from control.agent.shangshu.plan import load_plan
    plan = load_plan(plan_id)
    if plan is None:
        return

    reviews = []
    for s in plan.steps:
        if s.status != "done":
            continue
        if s.capability in ("run_evaluation", "run_batch_experiment"):
            review = _try_review_step(s)
            if review:
                reviews.append({"step_id": s.id, "review": review})

    if not reviews:
        return

    bus = get_bus()
    bus.publish(BusMessage(
        from_dept=MENXIA, to_dept=ZHONGSHU, kind=KIND_REVIEW,
        payload={"plan_id": plan_id, "reviews": reviews},
    ))


def _try_review_step(step) -> dict | None:
    """尝试对一个执行步的结果跑审查。失败静默返回 None。"""
    try:
        from control.agent.review import review_run
        result = step.result or {}
        # run_evaluation 的结果里有 work_dir，尝试构造 run 名
        work_dir = result.get("work_dir", "")
        if not work_dir:
            return None
        # work_dir 形如 "eval_runs/eval_xxx" 或 "workspaces/<name>"
        # review_run 支持 'ts/target' 和 'ws:<name>/<target>' 格式
        # 这里用 work_dir 的最后一段作为 run 名尝试
        # 简化：若 work_dir 在 workspaces 下，用 ws: 前缀
        if "workspaces" in work_dir:
            parts = work_dir.split("/")
            if len(parts) >= 2:
                ws_name = parts[-1]
                # 需要知道 target，从 step.args 取
                target = step.args.get("target") or (step.args.get("targets", [""])[0] if step.args.get("targets") else "")
                if target:
                    run_name = f"ws:{ws_name}/{target}"
                    return review_run(run_name)
        return None
    except Exception:
        return None


def _on_step_failed(msg: BusMessage) -> None:
    """步骤失败，呈递异常简报到中书省面板。"""
    payload = msg.payload
    bus = get_bus()
    bus.publish(BusMessage(
        from_dept=MENXIA, to_dept=ZHONGSHU, kind=KIND_REVIEW,
        payload={
            "plan_id": payload.get("plan_id", ""),
            "step_id": payload.get("step_id", ""),
            "capability": payload.get("capability", ""),
            "error": payload.get("error", ""),
            "type": "failure_report",
        },
    ))
