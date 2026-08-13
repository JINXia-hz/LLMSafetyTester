"""control.agent.shangshu.capabilities — 尚书省能力清单（固定）。

每个 capability = {name, description, parameters, handler, risk_level, doc,
                   block_message?, extract_review_target?}。

封驳判据**数据化**：每个 capability 自带 block_message(args) → {summary, detail} | None。
门下省不再用 switch 判断哪个 capability 危险——由 capability 自己声明。
- block_message 为 None：该能力永不封驳
- block_message(args) 返回 None：本次调用放行
- block_message(args) 返回 dict：封驳，附 summary/detail 文案

extract_review_target(result, args) → str | None：从执行结果提取可审查的 run 名，
供门下省事后审查。为 None 表示该能力不产出可审查的 run。

handler 签名：handler(args: dict) -> dict
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
    risk_level: str = "low"           # low / medium / high / critical（文档 + 前端展示）
    doc: str = ""                     # 详细文档（参数含义/约束/示例/失败模式）
    block_message: Callable[[dict], dict | None] | None = None       # 封驳判据（None=永不封驳）
    extract_review_target: Callable[[dict, dict], str | None] | None = None  # 事后审查的 run 名提取

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

    ws = args.get("workspace")
    if ws:
        work_dir = WORKSPACES_DIR / ws
        if not work_dir.exists():
            raise FileNotFoundError(f"工作区不存在: {ws}")
    else:
        import time
        wname = args.get("work_dir_name") or f"eval_{int(time.time())}"
        work_dir = OUTPUT_DIR / "eval_runs" / wname
        work_dir.mkdir(parents=True, exist_ok=True)

    env_override = None
    snap = args.get("env_snapshot")
    if snap:
        env_override = env_snapshot.load_env_dict(snap)

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
    from control.agent.menxia import review_run
    return review_run(args["run"])


def _h_merge_env_to_global(args: dict) -> dict:
    """把 .env 快照写回全局 .env（critical）。"""
    from control.core import env_snapshot
    return env_snapshot.merge_to_global(args["name"])


def _h_get_env_config(args: dict) -> dict:
    """只读查询当前 .env 的关键配置（脱敏 API key）。供尚书省拟案时了解项目现状。"""
    from control.core.env_snapshot import _read_global_env
    keys_raw = _read_global_env()
    # 只返回关键配置 key，API key 脱敏
    safe = {}
    for k, v in keys_raw.items():
        if "API_KEY" in k or "SECRET" in k:
            safe[k] = v[:4] + "****" if len(v) > 4 else "****"
        elif k in ("TARGETS", "JUDGE_MODEL", "GENERATOR_MODEL", "TARGET_MODEL",
                    "JUDGE_BASE_URL", "GENERATOR_BASE_URL", "TARGET_BASE_URL",
                    "TARGET_TYPE") or k.startswith("TARGET_"):
            safe[k] = v
    return safe


# ============================================================
# block_message 判据函数（封驳文案由 capability 自己提供）
# ============================================================
def _blk_run_evaluation(args: dict) -> dict | None:
    targets = args.get("targets") or [args.get("target", "?")]
    max_rounds = args.get("max_rounds", 5)
    return {
        "summary": f"即将对 {targets} 跑评估（{max_rounds} 轮自适应）",
        "detail": (
            f"将消耗 API 额度（调目标模型 + judge），{max_rounds} 轮 × batch_size 次。"
            f"产物落在隔离 work-dir，不污染全局。确认执行？"
        ),
    }


def _blk_run_batch_experiment(args: dict) -> dict | None:
    specs = args.get("specs", [])
    workers = args.get("max_workers", 2)
    return {
        "summary": f"即将批量跑 {len(specs)} 个实验（并行度 {workers}）",
        "detail": (
            f"{len(specs)} 个 spec 各起隔离 workspace + runner，是 API 开销大头。"
            f"完成后自动对比 + 审查。确认执行？"
        ),
    }


def _blk_merge_results(args: dict) -> dict | None:
    """target=global 必封；target=ws 封驳提示。"""
    sources = args.get("sources", [])
    src_str = ", ".join(str(s) for s in sources) if sources else "（未指定）"
    target = args.get("target", "")
    if target == "global":
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
    # target=ws:<name>：分支融合也封驳（影响目标工作区 R）
    return {
        "summary": f"即将把 {src_str} 合并到 {target}",
        "detail": f"分支融合：{src_str} → {target}。目标工作区的 R 将被覆盖更新。",
    }


def _blk_delete_runs(args: dict) -> dict | None:
    names = args.get("names", [])
    if args.get("delete_r"):
        return {
            "summary": f"即将删除 R 矩阵中 {names} 的观测列",
            "detail": (
                f"永久删除全局 R 中这些模型的全部观测（{names}）。\n"
                f"derive_elo 对这些模型的回放将失效。属于最危险操作——不可从 R 重算恢复。"
            ),
        }
    return {
        "summary": f"即将删除 {len(names)} 个 run",
        "detail": f"软删除到 .trash/（可恢复）。目标：{names}。",
    }


def _blk_clean_cache(args: dict) -> dict | None:
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


def _blk_edit_env_snapshot(args: dict) -> dict | None:
    name = args.get("name", "")
    key = args.get("key", "")
    return {
        "summary": f"即将改 .env 快照「{name}」的 {key}",
        "detail": f"修改隔离快照内的 {key}。不影响全局 .env，只影响引用此快照的 run。",
    }


def _blk_merge_env_to_global(args: dict) -> dict | None:
    name = args.get("name", "")
    return {
        "summary": f"即将把 .env 快照「{name}」写回全局 .env",
        "detail": (
            "全局 .env 会被覆盖（先备份到 .env.bak.<ts>）。"
            "改全局连接配置会让所有后续 run 受影响——目标模型/judge/参数全部变更。"
            "快照里有的 key 覆盖全局同名 key；快照里没有的不动。"
        ),
    }


# ============================================================
# extract_review_target 函数（从执行结果提取可审查的 run 名）
# ============================================================
def _review_target_run_eval(result: dict, args: dict) -> str | None:
    """run_evaluation 产出：从 work_dir 推导 run 名。"""
    work_dir = (result or {}).get("work_dir", "")
    if not work_dir:
        return None
    target = args.get("target") or (
        args.get("targets", [""])[0] if args.get("targets") else "")
    if not target:
        return None
    if "workspaces" in work_dir:
        parts = work_dir.split("/")
        return f"ws:{parts[-1]}/{target}"
    return None  # 临时 eval_runs 目前无报告可审


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
            block_message=_blk_run_evaluation,
            extract_review_target=_review_target_run_eval,
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
            block_message=_blk_run_batch_experiment,
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
            block_message=_blk_merge_results,
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
            block_message=_blk_delete_runs,
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
            block_message=_blk_clean_cache,
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
            block_message=_blk_edit_env_snapshot,
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
            block_message=_blk_merge_env_to_global,
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
        Capability(
            name="get_env_config",
            description="只读查询当前 .env 的关键配置（TARGETS/JUDGE_MODEL/GENERATOR_MODEL 等，API key 脱敏）。了解项目现状用。",
            parameters={"type": "object", "properties": {}},
            handler=_h_get_env_config,
            risk_level="low",
        ),
    ]


def reset_registry() -> None:
    """重置注册表（测试用）。"""
    global _REGISTRY
    _REGISTRY = None
