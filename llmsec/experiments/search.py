"""
experiments.search — 搜索引擎（grid / random / bayesian）。

统一接口：
  engine = build_search(study_config, completed_trials)
  params = engine.ask()            # 建议下一组超参（dict），或 None 表示预算/空间耗尽
  engine.tell(params, objective_value)   # 回报该 config 的目标值（仅 bayesian 需要）

bayesian 用 optuna TPE；grid/random 纯标准库。completed_trials 用于断点续跑时
把已有结果喂回（bayesian 重建研究状态）。
"""

from __future__ import annotations

import itertools
import math
import random

from llmsec.experiments.schema import FactorSpec, StudyConfig


class SearchEngine:
    def __init__(self, config: StudyConfig):
        self.config = config

    def ask(self) -> dict | None:
        raise NotImplementedError

    def tell(self, params: dict, value: float) -> None:
        """默认空实现（grid/random 不需要回报）。"""
        pass


class GridSearch(SearchEngine):
    """笛卡尔积穷举。每个因子按 step/choices 展开为有限取值集。"""

    def __init__(self, config: StudyConfig):
        super().__init__(config)
        axis: dict[str, list] = {}
        for name, spec in config.space.items():
            axis[name] = _factor_values(spec)
        self._combos = [dict(zip(axis, c)) for c in itertools.product(*axis.values())]
        self._idx = 0

    def ask(self) -> dict | None:
        if self._idx >= len(self._combos):
            return None
        c = self._combos[self._idx]
        self._idx += 1
        return c


class RandomSearch(SearchEngine):
    """均匀随机采样（float/log 按对数均匀，int 离散，categorical 等概率）。"""

    def __init__(self, config: StudyConfig, rng: random.Random):
        super().__init__(config)
        self._rng = rng

    def ask(self) -> dict | None:
        return {name: _sample(self.config.space[name], self._rng) for name in self.config.space}


class BayesianSearch(SearchEngine):
    """optuna TPE 贝叶斯优化。tell() 把已完成 config 的目标值喂回。"""

    def __init__(self, config: StudyConfig, completed: list[dict]):
        super().__init__(config)
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self._study = optuna.create_study(
            direction=config.objective.direction,
            sampler=optuna.samplers.TPESampler(seed=config.seed_base),
        )
        self._param_order = list(config.space.keys())
        # 把已完成 trial 灌入（断点续跑时复用历史）
        for t in completed:
            params, value = t.get("params"), t.get("objective")
            if params is None or value is None or not math.isfinite(value):
                continue
            try:
                self._study.enqueue_trial({k: params[k] for k in self._param_order if k in params})
            except Exception:
                pass

    def ask(self) -> dict | None:
        # 用固定 trial 对象构造，避免 ask/tell 接口在多线程下的复杂性
        trial = self._study.ask()
        params = {}
        for name in self._param_order:
            params[name] = _suggest(self._study, trial, name, self.config.space[name])
        self._pending = (trial, params)  # 暂存，tell 时提交
        return params

    def tell(self, params: dict, value: float) -> None:
        trial = getattr(self, "_pending", None)
        if trial is None:
            return
        trial_obj, _ = trial
        try:
            self._study.tell(trial_obj, float(value))
        except Exception:
            pass
        self._pending = None


def build_search(config: StudyConfig, completed: list[dict] | None = None,
                 rng: random.Random | None = None) -> SearchEngine:
    completed = completed or []
    s = config.strategy.lower()
    if s == "grid":
        return GridSearch(config)
    if s == "random":
        return RandomSearch(config, rng or random.Random(config.seed_base))
    if s == "bayesian":
        return BayesianSearch(config, completed)
    raise ValueError(f"未知 search 策略: {config.strategy}")


# ---------- 因子取值/采样工具 ----------
def _factor_values(spec: FactorSpec) -> list:
    """grid 展开用的有限取值集。"""
    if spec.type == "categorical":
        return list(spec.choices or [])
    lo, hi, step = spec.low, spec.high, spec.step
    if lo is None or hi is None:
        return []
    if spec.type == "int":
        step = int(step or 1)
        return list(range(int(lo), int(hi) + 1, step))
    step = step or (hi - lo)
    n = int(round((hi - lo) / step)) + 1
    return [lo + i * step for i in range(n)]


def _sample(spec: FactorSpec, rng: random.Random):
    if spec.type == "categorical":
        return rng.choice(spec.choices or [None])
    lo, hi = spec.low, spec.high
    if spec.type == "int":
        return rng.randint(int(lo), int(hi))
    if spec.log and lo and hi and lo > 0:
        return math.exp(rng.uniform(math.log(lo), math.log(hi)))
    return rng.uniform(lo, hi)


def _suggest(study, trial, name: str, spec: FactorSpec):
    if spec.type == "categorical":
        return trial.suggest_categorical(name, spec.choices or [None])
    if spec.type == "int":
        return trial.suggest_int(name, int(spec.low), int(spec.high), step=int(spec.step or 1))
    if spec.log:
        return trial.suggest_float(name, spec.low, spec.high, log=True)
    return trial.suggest_float(name, spec.low, spec.high)
