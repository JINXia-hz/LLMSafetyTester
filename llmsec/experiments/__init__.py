"""llmsec.experiments — 科学实验自动化框架（HPO + 复现 + 聚合）。"""

from llmsec.experiments.schema import StudyConfig, FactorSpec, ObjectiveSpec, resolve_trial

__all__ = ["StudyConfig", "FactorSpec", "ObjectiveSpec", "resolve_trial"]
