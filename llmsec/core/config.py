"""
core.config — 统一配置入口

职责：
  1. 幂等加载项目根目录 .env（替代原 7 处逐字重复的 load_dotenv 调用）
  2. 集中管理路径常量（历史三套路径已统一裁决为 output/state/、output/attacks/，
     即 runner.py 现行约定；旧路径仅作读取兼容回退，写入只写新路径）
  3. TargetConfig / GeneratorConfig / JudgeConfig dataclass + from_env()
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# ============================================================
# .env 加载（幂等）
# ============================================================
_ENV_LOADED = False


def load_env() -> bool:
    """
    幂等加载项目根目录的 .env。
    项目根 = llmsec 包所在目录。
    返回是否找到了 .env 文件。
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return (PROJECT_ROOT / ".env").exists()
    _ENV_LOADED = True
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        return True
    load_dotenv()  # 回退：从 cwd 向上查找
    return False


# ============================================================
# 路径常量
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "output"
STATE_DIR = OUTPUT_DIR / "state"
ATTACKS_DIR = OUTPUT_DIR / "attacks"
RUNS_DIR = OUTPUT_DIR / "runs"

# 状态文件（新约定：output/state/）
ELO_FILE = STATE_DIR / "elo.json"
SAFE_TWINS_FILE = STATE_DIR / "safe_twins.jsonl"
GROUND_TRUTH_ELO_FILE = STATE_DIR / "ground_truth_elo.json"
PREDICTED_ELO_FILE = STATE_DIR / "predicted_elo.json"

# 攻击集（新约定：output/attacks/）
ATTACK_SET_L1_FILE = ATTACKS_DIR / "l1.jsonl"

# 聚类产物（按 clustering.py 现行约定：直接落在 output/ 下）
CLUSTER_REPORT_FILE = OUTPUT_DIR / "cluster_report.json"
CLUSTER_MATRIX_FILE = OUTPUT_DIR / "cluster_matrix.csv"
CLUSTER_FEATURES_FILE = OUTPUT_DIR / "cluster_features.json"
CLUSTER_ARTIFACTS_FILE = OUTPUT_DIR / "cluster_artifacts.pkl"

# ------------------------------------------------------------
# 旧路径（仅用于读取兼容回退，写入一律走上面的新路径）
# ------------------------------------------------------------
LEGACY_ELO_FILE = OUTPUT_DIR / "elo.json"
LEGACY_SAFE_TWINS_FILE = OUTPUT_DIR / "safe_twins.jsonl"
LEGACY_ATTACK_SET_L1_FILE = OUTPUT_DIR / "攻击集_L1.jsonl"


def resolve_existing(primary, *legacy_candidates):
    """
    兼容回退读取：primary 存在则返回 primary，
    否则依次返回第一个存在的旧路径；都不存在时返回 primary。
    参数与返回值均为 pathlib.Path。
    """
    primary = Path(primary)
    if primary.exists():
        return primary
    for candidate in legacy_candidates:
        candidate = Path(candidate)
        if candidate.exists():
            return candidate
    return primary


# ============================================================
# 模型配置 dataclass
# ============================================================
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-v4-flash"

# ELO 默认参数
INITIAL_ELO = 1500


@dataclass
class TargetConfig:
    """目标模型（被攻击方）配置。环境变量前缀 TARGET_*。"""

    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout: float = 90.0          # targets.py 现行 REQUEST_TIMEOUT
    max_retries: int = 3
    temperature: float = 0.0
    max_tokens: int = 1024

    @classmethod
    def from_env(cls) -> "TargetConfig":
        load_env()
        return cls(
            api_key=os.getenv("TARGET_API_KEY", ""),
            base_url=os.getenv("TARGET_BASE_URL", DEFAULT_BASE_URL),
            model=os.getenv("TARGET_MODEL", DEFAULT_MODEL),
        )


@dataclass
class GeneratorConfig:
    """攻击生成模型配置。环境变量前缀 GENERATOR_*。

    注意：与现行 generate_attacks.py 一致，三个字段均无默认值（None），
    由调用方决定是否报错或回退。
    """

    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    timeout: float = 60.0
    max_retries: int = 3
    temperature: float = 0.9
    max_tokens: int = 4096

    @classmethod
    def from_env(cls) -> "GeneratorConfig":
        load_env()
        return cls(
            api_key=os.getenv("GENERATOR_API_KEY"),
            base_url=os.getenv("GENERATOR_BASE_URL"),
            model=os.getenv("GENERATOR_MODEL"),
        )


@dataclass
class JudgeConfig:
    """LLM-as-Judge 配置。

    与 judge.py 现行读取逻辑一致：
      api_key  = GENERATOR_API_KEY or JUDGE_API_KEY or ""
      base_url = GENERATOR_BASE_URL，默认 DEFAULT_BASE_URL
      model    = JUDGE_MODEL，默认 DEFAULT_MODEL
    """

    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout: float = 30.0          # judge.py 现行 JUDGE_TIMEOUT
    max_retries: int = 2           # judge.py 现行 JUDGE_MAX_RETRIES
    temperature: float = 0.0
    max_tokens: int = 512

    @classmethod
    def from_env(cls) -> "JudgeConfig":
        load_env()
        return cls(
            api_key=os.getenv("GENERATOR_API_KEY", os.getenv("JUDGE_API_KEY", "")),
            base_url=os.getenv("GENERATOR_BASE_URL", DEFAULT_BASE_URL),
            model=os.getenv("JUDGE_MODEL", DEFAULT_MODEL),
        )
