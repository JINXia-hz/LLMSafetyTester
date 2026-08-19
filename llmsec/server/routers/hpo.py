"""HPO 配置台路由：参数清单 / 搜索空间预览 / 启动 HPO study（作为任务）。

把"手动配置环境参数和关键参数的 HPO"搬进看板——因子选择器读 key params 清单，
预览搜索空间规模与预估成本，启动后作为 hpo 任务进任务列表（复用 _start_task）。
"""
from __future__ import annotations

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from llmsec import params as P
from llmsec.core.config import OUTPUT_DIR
from llmsec.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

# HPO 可调 key params（分组 + 类型 + 当前值 + 建议范围），驱动前端因子选择器。
# 全部经 LLMSEC_PARAM_<NAME> 注入子进程（params._apply_env_overrides）。
_KEY_PARAMS = [
    {"name": "K_FACTOR", "group": "Elo/K", "type": "float", "current": P.K_FACTOR, "low": 12.0, "high": 20.0, "step": 4.0},
    {"name": "SCORE_PERF_TAU", "group": "Elo/K", "type": "float", "current": P.SCORE_PERF_TAU, "low": 1.0, "high": 4.0, "step": 0.5},
    {"name": "K_DEF_DECAY_N0", "group": "Elo/K", "type": "float", "current": P.K_DEF_DECAY_N0, "low": 5.0, "high": 20.0, "step": 5.0},
    {"name": "CONV_CI_TARGET", "group": "收敛", "type": "float", "current": P.CONV_CI_TARGET, "low": 10.0, "high": 40.0, "step": 5.0},
    {"name": "CONV_DRIFT_TARGET", "group": "收敛", "type": "float", "current": P.CONV_DRIFT_TARGET, "low": 2.0, "high": 10.0, "step": 1.0},
    {"name": "SAMPLER_INFOGAIN_ALPHA", "group": "采样器", "type": "float", "current": P.SAMPLER_INFOGAIN_ALPHA, "low": 0.2, "high": 3.0, "step": 0.2, "log": True},
    {"name": "SAMPLER_INFOGAIN_BETA", "group": "采样器", "type": "float", "current": P.SAMPLER_INFOGAIN_BETA, "low": 0.0, "high": 1.5, "step": 0.1},
    {"name": "SAMPLER_INFOGAIN_GAMMA", "group": "采样器", "type": "float", "current": P.SAMPLER_INFOGAIN_GAMMA, "low": 0.2, "high": 3.0, "step": 0.2, "log": True},
    {"name": "SAMPLER_COORD_MIN_PER_CLUSTER", "group": "采样器", "type": "int", "current": P.SAMPLER_COORD_MIN_PER_CLUSTER, "low": 1, "high": 8, "step": 1},
    {"name": "BLEND_PRIOR_K", "group": "预测器", "type": "float", "current": P.BLEND_PRIOR_K, "low": 2.0, "high": 30.0, "step": 2.0, "log": True},
    {"name": "RIDGE_N_FOLDS", "group": "预测器", "type": "int", "current": P.RIDGE_N_FOLDS, "choices": [3, 5, 7, 10]},
    {"name": "sampler", "group": "采样器(CLI)", "type": "categorical", "choices": ["hybrid", "infogain", "gap", "coordinate"]},
    {"name": "batch_size", "group": "规模(CLI)", "type": "int", "current": P.DEFAULT_BATCH_SIZE, "low": 3, "high": 12, "step": 1},
]


@router.get("/api/hpo/params")
async def api_hpo_params():
    """返回 HPO 可调 key params 清单（名/类型/当前值/建议范围/分组），驱动因子选择器。"""
    return {"params": _KEY_PARAMS}


# ---- pydantic 模型 ----
class _FactorSpec(BaseModel):
    type: str = "float"
    low: float | None = None
    high: float | None = None
    step: float | None = None
    log: bool = False
    choices: list | None = None


class _Objective(BaseModel):
    metric: str = "conv_rounds"
    direction: str = "minimize"
    aggregate: str = "mean"


class HpoRequest(BaseModel):
    name: str
    objective: _Objective = _Objective()
    strategy: str = "bayesian"
    max_trials: int = Field(default=20, ge=1, le=500)
    max_wall_minutes: int = Field(default=0, ge=0)
    trial_timeout_minutes: int = Field(default=30, ge=1)
    repeats: int = Field(default=1, ge=1, le=5)
    seed_base: int = 0
    space: dict[str, _FactorSpec] = Field(default_factory=dict)
    fixed: dict = Field(default_factory=dict)
    targets: list[str] = Field(default_factory=list)
    max_concurrent: int = Field(default=1, ge=1, le=8)
    # 预估成本用：每 trial 约评估多少方法（batch_size × max_rounds 近似）
    est_methods_per_trial: int = Field(default=50, ge=1)


def _factor_cardinality(f: _FactorSpec) -> int:
    """单个因子在 grid 策略下的取值数。"""
    if f.choices:
        return max(1, len(f.choices))
    if f.low is not None and f.high is not None:
        step = f.step if f.step and f.step > 0 else 1.0
        return max(1, int(round((f.high - f.low) / step)) + 1)
    return 1


@router.post("/api/hpo/preview")
async def api_hpo_preview(req: HpoRequest):
    """预览搜索空间：configs 数（grid=笛卡尔积；random/bayesian=max_trials）、
    总 trial 数（×repeats ×targets）、预估方法调用数。"""
    warnings = []
    if not req.space:
        warnings.append("未选择任何因子——study 不会搜索，只会按 fixed 跑 repeats 次")
    if req.strategy == "grid":
        n_configs = 1
        for f in req.space.values():
            n_configs *= _factor_cardinality(f)
        if n_configs > req.max_trials:
            warnings.append(f"grid 笛卡尔积 {n_configs} > max_trials {req.max_trials}，将截断到 {req.max_trials}")
            n_configs = req.max_trials
    else:  # random / bayesian
        n_configs = req.max_trials
    n_targets = max(1, len(req.targets)) if req.targets else 1
    n_trials = n_configs * req.repeats * n_targets
    est_calls = n_trials * max(1, req.est_methods_per_trial)
    return {
        "n_configs": n_configs,
        "n_trials": n_trials,
        "est_method_calls": est_calls,
        "warnings": warnings,
    }


@router.post("/api/run/hpo")
async def api_run_hpo(req: HpoRequest):
    """写临时 study.yaml 并作为 hpo 任务启动（进任务列表）。

    任务启动统一走 llmsec.server.launch（与 TUI 同链路）；本端点只负责
    把 HpoRequest 落成 study.yaml。
    """
    from llmsec.server.launch import LaunchError, launch_hpo_study

    if not req.name.strip():
        raise HTTPException(400, "study 名不能为空")
    if not req.targets and not req.fixed.get("target"):
        raise HTTPException(400, "未选择目标模型（targets 为空且 fixed 无 target）——study 无目标可跑")
    # name 进文件路径前须做安全组件校验（"../../evil" 可把 yaml 写到 experiments 之外）
    from llmsec.core.paths import safe_component
    try:
        safe_name = safe_component(OUTPUT_DIR / "experiments", req.name.strip()).name
    except ValueError as e:
        raise HTTPException(400, f"study 名非法: {e}")
    # 构造 StudyConfig 兼容的 dict（schema.StudyConfig.from_dict 解析）。
    # name 用已过 safe_component 校验的 safe_name——原始 req.name 带穿越串时
    # 虽过不了文件名校验，但纵深防御要求 yaml 内容与文件名同源
    cfg_dict = {
        "name": safe_name,
        "objective": {"metric": req.objective.metric, "direction": req.objective.direction,
                      "aggregate": req.objective.aggregate},
        "strategy": req.strategy,
        "repeats": req.repeats,
        "seed_base": req.seed_base,
        "space": {k: v.model_dump(exclude_none=True) for k, v in req.space.items()},
        "fixed": req.fixed,
        "targets": req.targets,
        "max_concurrent": req.max_concurrent,
        "budget": {"max_trials": req.max_trials, "max_wall_minutes": req.max_wall_minutes,
                   "trial_timeout_minutes": req.trial_timeout_minutes},
    }
    studies_dir = OUTPUT_DIR / "experiments"
    studies_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = studies_dir / f"_dashboard_{safe_name}.yaml"
    try:
        yaml_path.write_text(yaml.safe_dump(cfg_dict, allow_unicode=True, sort_keys=False), encoding="utf-8")
    except Exception as e:
        raise HTTPException(500, f"study.yaml 写入失败: {e}")
    try:
        return launch_hpo_study(yaml_path)
    except LaunchError as e:
        raise HTTPException(404 if e.reason == "not_found" else 400, str(e)) from None
