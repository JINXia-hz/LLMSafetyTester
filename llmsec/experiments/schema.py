"""
experiments.schema — 实验配置（study.yaml）的数据模型与因子解析。

因子分两类注入 trial 子进程：
  • CLI 因子（sampler/batch_size/...）→ runner argv 旗标
  • params 因子（K_FACTOR/SCORE_PERF_TAU/...）→ LLMSEC_PARAM_<NAME> 环境变量
    （由 params.py 的 _apply_env_overrides 在 import 时生效）
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# 已知的 CLI 旗标因子（名 → runner argv 旗标）。未列出的因子名一律按 params 环境变量注入。
CLI_FACTORS: dict[str, str] = {
    "sampler": "--sampler",
    "batch_size": "--batch-size",
    "max_rounds": "--max-rounds",
    "sampler_alpha": "--sampler-alpha",
    "sampler_beta": "--sampler-beta",
    "sampler_gamma": "--sampler-gamma",
    "coordinate_rounds": "--coordinate-rounds",
    "twin_window": "--twin-window",
    "target": "--target",
    "input": "--input",
}


@dataclass
class FactorSpec:
    """单个搜索因子的定义。"""

    type: str                       # "int" | "float" | "categorical"
    low: float | None = None
    high: float | None = None
    step: float | None = None
    log: bool = False
    choices: list | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "FactorSpec":
        t = str(d.get("type", "float")).lower()
        # 容错：写成 {int, 16..64} 这种简写时 type 可能被吃成 "int16..64"——这里只取字母前缀
        pure = "".join(c for c in t if c.isalpha())
        return cls(
            type=pure or "float",
            low=d.get("low"),
            high=d.get("high"),
            step=d.get("step"),
            log=bool(d.get("log", False)),
            choices=list(d["choices"]) if "choices" in d else None,
        )


@dataclass
class ObjectiveSpec:
    metric: str = "conv_rounds"
    direction: str = "minimize"     # "minimize" | "maximize"
    aggregate: str = "mean"         # 跨 repeats 聚合：mean | mean_minus_std

    @classmethod
    def from_dict(cls, d: dict | None) -> "ObjectiveSpec":
        d = d or {}
        return cls(metric=d.get("metric", "conv_rounds"),
                   direction=d.get("direction", "minimize"),
                   aggregate=d.get("aggregate", "mean"))


@dataclass
class StudyConfig:
    name: str
    objective: ObjectiveSpec
    budget_max_trials: int
    strategy: str                   # grid | random | bayesian
    repeats: int
    seed_base: int
    space: dict[str, FactorSpec]
    fixed: dict                     # 锁定维度（target/input/CONV_* 等）
    description: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "StudyConfig":
        budget = d.get("budget", {}) or {}
        space_raw = d.get("space", {}) or {}
        # 兼容简写 {type, ...} 或裸 {low, high}（默认 float）
        space: dict[str, FactorSpec] = {}
        for k, v in space_raw.items():
            v = dict(v) if isinstance(v, dict) else {}
            if "type" not in v and ("low" in v or "choices" in v):
                v["type"] = "categorical" if "choices" in v else "float"
            space[k] = FactorSpec.from_dict(v)
        return cls(
            name=d["name"],
            objective=ObjectiveSpec.from_dict(d.get("objective")),
            budget_max_trials=int(budget.get("max_trials", 30)),
            strategy=d.get("strategy", "bayesian"),
            repeats=int(d.get("repeats", 1)),
            seed_base=int(d.get("seed_base", 0)),
            space=space,
            fixed=dict(d.get("fixed", {}) or {}),
            description=d.get("description", ""),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "StudyConfig":
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))


def resolve_trial(config: dict) -> tuple[list[str], dict[str, str]]:
    """
    把一组 {factor_name: value}（搜索值 + fixed）解析为 runner 的 (argv_extra, env_extra)。

    CLI 因子 → argv；其余 → LLMSEC_PARAM_<NAME> 环境变量。
    """
    argv: list[str] = []
    env: dict[str, str] = {}
    for name, val in config.items():
        if name in CLI_FACTORS:
            argv += [CLI_FACTORS[name], str(val)]
        else:
            env[f"LLMSEC_PARAM_{name.upper()}"] = str(val)
    return argv, env
