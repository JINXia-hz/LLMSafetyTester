"""control.agent.shangshu.capabilities — 尚书省能力清单（固定）。

每个 capability = {name, description, parameters(JSON schema), handler, risk_level, doc}。
尚书省 LLM 在此清单内组合步骤拟 Plan。

risk_level（门下省封驳判据）：
  - low:      无副作用或只读，不封驳
  - medium:   有副作用但可恢复（清缓存），封驳提示
  - high:     耗资源/有副作用（跑评估、删 run），封驳确认
  - critical: 不可逆全局变更（merge 到全局 R、删 R 列、改全局 .env），必封驳

handler 签名：handler(args: dict) -> dict（结果结构化，供 Plan 记录 + 门下省审查）
所有 handler 经 control.core.* 调底层，绝不 import llmsec 内部。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class Capability:
    """尚书省的一项能力。"""
    name: str
    description: str
    parameters: dict                  # JSON schema（给 LLM 看）
    handler: Callable[[dict], Any]    # 执行函数
    risk_level: str = "low"           # low / medium / high / critical
    doc: str = ""                     # 详细文档（参数含义/约束/示例/失败模式）

    def to_schema(self) -> dict:
        """OpenAI function-calling 兼容 schema。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ============================================================
# handler 实现
# ============================================================
def _h_run_evaluation(args: dict) -> dict:
    """跑一次评估。经 invoker.run_runner 起 runner（work-dir 隔离）。"""
    from control.config import OUTPUT_DIR, WORKSPACES_DIR
    from control.core import env_snapshot
    from control.core.invoker import run_runner

    # 确定 work-dir：指定 workspace 用它，否则创建临时 work-dir
    ws = args.get("workspace")
    if ws:
        work_dir = WORKSPACES_DIR / ws
        if not work_dir.exists():
            raise FileNotFoundError(f"工作区不存在: {ws}")
    else:
        # 临时 work-dir（用 run 名或自动命名）
        import time
        wname = args.get("work_dir_name") or f"eval_{int(time.time())}"
        work_dir = OUTPUT_DIR / "eval_runs" / wname
        work_dir.mkdir(parents=True, exist_ok=True)

    # env_snapshot 注入
    env_override = None
    snap = args.get("env_snapshot")
    if snap:
        env_override = env_snapshot.load_env_dict(snap)

    # 参数注入（LLMSEC_PARAM_*）
    if args.get("param_overrides"):
        env_override = dict(env_override or {})
        for k, v in args["param_overrides"].items():
            env_override[f"LLMSEC_PARAM_{k}"] = str(v)

    res = run_runner(
        work_dir,
        target=args.get("target"),
        targets=args.get("targets"),
        input_file=args.get("input_file", "attacks/l1.jsonl"),
        max_rounds=args.get("max_rounds", 5),
        phase=args.get("phase", "all"),
        seed=args.get("seed"),
        env_override=env_override,
        timeout=args.get("timeout"),
    )
    return {
        "returncode": res.returncode,
        "ok": res.ok,
        "elapsed_s": res.elapsed_s,
        "work_dir": str(work_dir.relative_to(OUTPUT_DIR)).replace("\\", "/") if str(work_dir).startswith(str(OUTPUT_DIR)) else str(work_dir),
        "stdout_tail": (res.stdout or "")[-800:] if res.stdout else "",
        "stderr_tail": (res.stderr or "")[-800:] if res.stderr else "",
    }


def _h_run_batch_experiment(args: dict) -> dict:
    """批量并行实验。包装 orchestrator.orchestrate。"""
    from control.core import orchestrator as orch_mod
    specs = [orch_mod.RunSpec(**s) for s in args["specs"]]
    return orch_mod.orchestrate(
        specs,
        max_workers=args.get("max_workers", 2),
        compare_after=args.get("compare_after", True),
    )


def _h_fork_workspace(args: dict) -> dict:
    from control.core import workspace as ws_mod
    return ws_mod.fork(args["name"], source=args.get("source", "global"), note=args.get("note", ""))


def _h_merge_results(args: dict) -> dict:
    """合并 R 矩阵。经 invoker 调 llmsec-manage merge。"""
    from control.core.invoker import _manage_argv, _run
    from control.core.workspace import mark_merged
    sub = ["merge", "--sources", *args["sources"], "--target", args["target"], "--json"]
    if args.get("models"):
        sub += ["--models", *args["models"]]
    if args.get("confirm"):
        sub.append("--yes")
    res = _run(_manage_argv(sub))
    res.require_ok()
    result = res.json or {}
    if args.get("confirm") and result.get("dry_run") is False:
        target = args["target"]
        for src in args["sources"]:
            if src.startswith("ws:"):
                mark_merged(src[3:], target)
    return result


def _h_delete_runs(args: dict) -> dict:
    from control.core.invoker import delete_runs
    return delete_runs(args["names"], delete_r=args.get("delete_r", False))


def _h_clean_cache(args: dict) -> dict:
    from control.core.invoker import clean_caches
    return clean_caches(args["categories"])


def _h_list_runs(args: dict) -> list[dict]:
    from control.core.invoker import list_runs as inv_list_runs
    runs = inv_list_runs(target=args.get("target"), since=args.get("since"),
                        junk_only=args.get("junk_only", False))
    if args.get("include_workspaces", True):
        from control.core.compare import discover_workspace_runs
        runs = runs + discover_workspace_runs()
    return runs


def _h_compare_runs(args: dict) -> dict:
    from control.core import compare as compare_mod
    return compare_mod.compare(args["runs"])


def _h_list_workspaces(args: dict) -> list[dict]:
    from control.core import workspace as ws_mod
    return ws_mod.list_workspaces()


def _h_delete_workspace(args: dict) -> dict:
    from control.core import workspace as ws_mod
    return ws_mod.delete_workspace(args["name"])


def _h_create_env_snapshot(args: dict) -> dict:
    from control.core import env_snapshot
    return env_snapshot.create(args["name"], source=args.get("source", "global"),
                               note=args.get("note", ""))


def _h_edit_env_snapshot(args: dict) -> dict:
    from control.core import env_snapshot
    return env_snapshot.edit_key(args["name"], args["key"], str(args["value"]))


def _h_list_env_snapshots(args: dict) -> list[dict]:
    from control.core import env_snapshot
    return env_snapshot.list_snapshots()


def _h_delete_env_snapshot(args: dict) -> dict:
    from control.core import env_snapshot
    return env_snapshot.delete(args["name"])


def _h_request_review(args: dict) -> dict:
    """请门下省审查某 run。返回审查摘要。"""
    from control.agent.review import review_run
    return review_run(args["run"])


def _h_merge_env_to_global(args: dict) -> dict:
    """把 .env 快照写回全局 .env（critical）。"""
    from control.core import env_snapshot
    return env_snapshot.merge_to_global(args["name"])


# ============================================================
# 能力清单（固定，尚书省 LLM 在此范围内拟案）
# ============================================================
_REGISTRY: list[Capability] | None = None


def all_capabilities() -> list[Capability]:
    """返回全部能力（首次调用构建，之后复用）。"""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build()
    return _REGISTRY


def capability_by_name(name: str) -> Capability | None:
    for c in all_capabilities():
        if c.name == name:
            return c
    return None


def call(name: str, args: dict) -> Any:
    """执行某能力，返回结果。能力不存在抛 KeyError。"""
    c = capability_by_name(name)
    if c is None:
        raise KeyError(f"未知能力: {name}")
    return c.handler(args)


def _build() -> list[Capability]:
    return [
        # ---------- 执行类 ----------
        Capability(
            name="run_evaluation",
            description=(
                "跑一次 llmsec 安全评估（自适应攻击流水线）。起一个隔离 work-dir runner。"
                "支持指定目标模型、攻击集、轮数、phase、随机种子。"
                "可用 env_snapshot 注入隔离配置，用 param_overrides 覆写实验参数。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "目标模型名（与 targets 二选一）"},
                    "targets": {"type": "array", "items": {"type": "string"}, "description": "多目标（与 target 二选一）"},
                    "input_file": {"type": "string", "description": "攻击集文件", "default": "attacks/l1.jsonl"},
                    "max_rounds": {"type": "integer", "description": "自适应最大轮数", "default": 5},
                    "phase": {"type": "string", "enum": ["all", "1", "2"], "description": "评估阶段", "default": "all"},
                    "seed": {"type": "integer", "description": "随机种子（可复现）"},
                    "workspace": {"type": "string", "description": "在指定 workspace 内跑（复用其 R 矩阵起点）"},
                    "work_dir_name": {"type": "string", "description": "自定义 work-dir 名（不指定 workspace 时用）"},
                    "env_snapshot": {"type": "string", "description": ".env 快照名，用其隔离配置起 runner"},
                    "param_overrides": {
                        "type": "object",
                        "description": "覆写 params.py 常量（key 不带 LLMSEC_PARAM_ 前缀），如 {\"BATCH_SIZE\": 8}",
                    },
                    "timeout": {"type": "number", "description": "超时秒数（可选）"},
                },
            },
            handler=_h_run_evaluation,
            risk_level="high",
            doc=(
                "跑一次评估会消耗 API 额度（调目标模型 + judge）。max_rounds 越大开销越大。"
                "work-dir 模式全局零污染——产物落在 output/eval_runs/<name>/ 或指定 workspace 内。"
                "env_snapshot 让 runner 用隔离的模型列表/judge 配置，不碰全局 .env。"
                "param_overrides 临时覆写实验参数（如 BATCH_SIZE/JUDGE_K_FACTOR），只在本次 run 生效。"
                "失败模式：目标模型不可达 → returncode!=0，看 stderr_tail。"
            ),
        ),
        Capability(
            name="run_batch_experiment",
            description=(
                "批量并行 fork + run（A/B 对比实验）。每个 spec 起一个隔离 workspace + runner。"
                "完成后自动 compare_after 对比 + 规则审查。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "specs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "source": {"type": "string", "default": "global"},
                                "target": {"type": "string"},
                                "max_rounds": {"type": "integer", "default": 5},
                                "seed": {"type": "integer"},
                            },
                            "required": ["name"],
                        },
                    },
                    "max_workers": {"type": "integer", "default": 2},
                    "compare_after": {"type": "boolean", "default": True},
                },
                "required": ["specs"],
            },
            handler=_h_run_batch_experiment,
            risk_level="high",
            doc=(
                "批量实验是 API 开销大头——N 个 spec × max_rounds 轮 × batch_size。"
                "max_workers 控制并行度（默认 2，太高会触发目标模型限流）。"
                "每个 spec 独立 fork workspace，互不污染；完成后自动对比 + 审查。"
            ),
        ),
        Capability(
            name="fork_workspace",
            description="fork 一个隔离测试环境（以全局或指定 run 的 R 矩阵为起点）。不污染全局。",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "source": {"type": "string", "default": "global"},
                    "note": {"type": "string"},
                },
                "required": ["name"],
            },
            handler=_h_fork_workspace,
            risk_level="low",
        ),
        Capability(
            name="merge_results",
            description=(
                "把一个或多个源的 R 矩阵观测合并到目标（global 或 ws:<name>）。"
                "默认 dry-run（confirm=False），confirm=True 执行。"
                "target=global 是不可逆全局变更。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "sources": {"type": "array", "items": {"type": "string"}},
                    "target": {"type": "string"},
                    "models": {"type": "array", "items": {"type": "string"}},
                    "confirm": {"type": "boolean", "default": False},
                },
                "required": ["sources", "target"],
            },
            handler=_h_merge_results,
            risk_level="critical",
            doc=(
                "target=global：全局 R 永久累加，不可按来源精确剔除（critical 级，门下省必封驳）。"
                "target=ws:<name>：合并到另一工作区（分支融合，high 级）。"
                "默认 dry-run 预览将合并多少条；confirm=True 才真正执行。"
            ),
        ),
        Capability(
            name="delete_runs",
            description="删除评测 run 历史（软删除到 .trash/ 可恢复）。delete_r=True 同时删 R 矩阵列（极危险）。",
            parameters={
                "type": "object",
                "properties": {
                    "names": {"type": "array", "items": {"type": "string"}},
                    "delete_r": {"type": "boolean", "default": False},
                },
                "required": ["names"],
            },
            handler=_h_delete_runs,
            risk_level="high",
            doc="delete_r=True 时升级为 critical（删全局 R 观测不可恢复）。",
        ),
        Capability(
            name="clean_cache",
            description="清理派生缓存（elo_cache/predictors/feature_cluster/task_logs）。可重建。",
            parameters={
                "type": "object",
                "properties": {
                    "categories": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["categories"],
            },
            handler=_h_clean_cache,
            risk_level="medium",
        ),
        # ---------- .env 快照 ----------
        Capability(
            name="create_env_snapshot",
            description="创建 .env 快照（从全局 .env 复制，或基于另一快照）。用于隔离配置跑实验。",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "source": {"type": "string", "default": "global", "description": "global / blank / 另一快照名"},
                    "note": {"type": "string"},
                },
                "required": ["name"],
            },
            handler=_h_create_env_snapshot,
            risk_level="low",
        ),
        Capability(
            name="edit_env_snapshot",
            description=(
                "编辑 .env 快照内某个 key。受管理 key 前缀：TARGETS/TARGET_/JUDGE_/GENERATOR_/"
                "CONTROL_/LLMSEC_PARAM_。例：加模型设 TARGETS=X,Y,Z；改 judge 设 JUDGE_MODEL=W。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["name", "key", "value"],
            },
            handler=_h_edit_env_snapshot,
            risk_level="medium",
            doc=(
                "key 必须在受管理前缀内（防乱写）。value 统一存字符串。"
                "改 TARGETS 时记得同时设各 TARGET_<N>_NAME/API_KEY/BASE_URL/MODEL。"
                "改 JUDGE_MODEL 时若 judge 在不同服务商，还要设 JUDGE_API_KEY/JUDGE_BASE_URL。"
                "改实验参数用 LLMSEC_PARAM_ 前缀的 key（如 LLMSEC_PARAM_BATCH_SIZE=8）。"
            ),
        ),
        Capability(
            name="list_env_snapshots",
            description="列出所有 .env 快照。",
            parameters={"type": "object", "properties": {}},
            handler=_h_list_env_snapshots,
            risk_level="low",
        ),
        Capability(
            name="delete_env_snapshot",
            description="删除 .env 快照（仅删隔离副本，不影响全局 .env）。",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            handler=_h_delete_env_snapshot,
            risk_level="low",
        ),
        Capability(
            name="merge_env_to_global",
            description="把 .env 快照的 key 写回全局 .env（critical，备份后覆盖）。",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            handler=_h_merge_env_to_global,
            risk_level="critical",
            doc="改全局连接配置会让所有后续 run 受影响。先备份 .env.bak.<ts> 再写。",
        ),
        # ---------- 查询类 ----------
        Capability(
            name="list_runs",
            description="列出评测 run 历史（含 workspace 分支内的 run）。",
            parameters={
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "since": {"type": "string"},
                    "junk_only": {"type": "boolean", "default": False},
                    "include_workspaces": {"type": "boolean", "default": True},
                },
            },
            handler=_h_list_runs,
            risk_level="low",
        ),
        Capability(
            name="compare_runs",
            description="对比多个 run 的安全指标（至少 2 个）。",
            parameters={
                "type": "object",
                "properties": {"runs": {"type": "array", "items": {"type": "string"}}},
                "required": ["runs"],
            },
            handler=_h_compare_runs,
            risk_level="low",
        ),
        Capability(
            name="list_workspaces",
            description="列出所有 fork 工作区。",
            parameters={"type": "object", "properties": {}},
            handler=_h_list_workspaces,
            risk_level="low",
        ),
        Capability(
            name="delete_workspace",
            description="删除 fork 工作区（仅删隔离副本）。",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            handler=_h_delete_workspace,
            risk_level="low",
        ),
        Capability(
            name="request_review",
            description="请门下省审查某 run，读报告识别异常 + 呈递摘要。",
            parameters={
                "type": "object",
                "properties": {"run": {"type": "string"}},
                "required": ["run"],
            },
            handler=_h_request_review,
            risk_level="low",
        ),
    ]


def reset_registry() -> None:
    """重置注册表（测试用）。"""
    global _REGISTRY
    _REGISTRY = None
