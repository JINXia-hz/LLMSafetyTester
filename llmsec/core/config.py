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

# 状态文件（统一持久化：output/state/state.json）
STATE_FILE = STATE_DIR / "state.json"
SAFE_TWINS_FILE = STATE_DIR / "safe_twins.jsonl"

# 结果矩阵（多模型唯一真相）+ 派生缓存目录
RESULTS_FILE = STATE_DIR / "results.json"          # R[method][model] 主存储
PREDICTORS_DIR = OUTPUT_DIR / "predictors"          # 统一/每模型 ridge 预测器
ELO_CACHE_FILE = STATE_DIR / "elo_cache.json"       # Elo 派生缓存（可删可重建）

# 攻击集（新约定：output/attacks/）
ATTACK_SET_L1_FILE = ATTACKS_DIR / "l1.jsonl"

# 过敏检测产物（规范存储，按模型隔离见 W4）
TWIN_RESULT_FILE = OUTPUT_DIR / "allergy_results.jsonl"
ALLERGY_REPORT_FILE = OUTPUT_DIR / "allergy_report.json"

# 报告产物（reporting 写、dashboard 读）
TREE_FILE = OUTPUT_DIR / "security_tree.json"
REPORT_FILE = OUTPUT_DIR / "security_report.md"
METHOD_REGISTRY_FILE = OUTPUT_DIR / "method_registry.json"
CLUSTER_SECURITY_ANALYSIS_FILE = OUTPUT_DIR / "cluster_security_analysis.json"

# 后台任务日志目录（dashboard 子进程任务）
TASK_LOG_DIR = OUTPUT_DIR / "tasks"

# 内置静态数据目录（HarmBench 行为库 + 越狱模板，出处见 data/Explication.md）
DATA_DIR = PROJECT_ROOT / "data"

# 聚类产物（按 clustering.py 现行约定：直接落在 output/ 下）
CLUSTER_REPORT_FILE = OUTPUT_DIR / "cluster_report.json"
CLUSTER_MATRIX_FILE = OUTPUT_DIR / "cluster_matrix.csv"
CLUSTER_FEATURES_FILE = OUTPUT_DIR / "cluster_features.json"
# 聚类 artifacts 已按写者拆分为两个文件（原 cluster_artifacts.pkl 由两个写者混写不同
# schema，后写覆盖会导致下游 KeyError）：
#   - feature_cache.pkl：先验特征缓存，仅 elo_cluster.fit_features 写
#   - cluster_result.pkl：完整聚类产物，hdb 写、final_fit 增补
FEATURE_CACHE_FILE = OUTPUT_DIR / "feature_cache.pkl"
CLUSTER_RESULT_FILE = OUTPUT_DIR / "cluster_result.pkl"
# 已废弃别名（指向 cluster_result.pkl），仅为未迁移的外部引用保留；新代码用上面两个。
CLUSTER_ARTIFACTS_FILE = CLUSTER_RESULT_FILE


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


# ============================================================
# 多目标扫描（.env: TARGETS + TARGET_<N>_* ；兼容旧 TARGET_*）
# ============================================================
def _target_from_prefixed(prefix: str, name: str) -> "TargetConfig":
    """从 TARGET_<N>_* 一组变量构造一个目标（prefix 形如 'TARGET_1'）。"""
    return TargetConfig(
        api_key=os.getenv(f"{prefix}_API_KEY", ""),
        base_url=os.getenv(f"{prefix}_BASE_URL", DEFAULT_BASE_URL),
        model=os.getenv(f"{prefix}_MODEL", DEFAULT_MODEL),
        # backend 类型下沉为每目标属性（缺失则继承全局 TARGET_TYPE/openai）
    )


def _resolve_target_prefixes(names: list[str]) -> "dict[str, str]":
    """解析 TARGETS 列表，返回 {declared_name: 实际使用的前缀}。

    load_targets 与 target_backend **必须共用本函数**，保证二者对同一 name
    返回一致的配置/类型（F-4 修复）。

    解析策略（不依赖列表位置 idx，消除中间缺口导致的错位）：
      1. 预扫描所有 ``TARGET_<N>_NAME`` 环境变量，建立 {declared_name: "TARGET_<N>"}。
         编号是稳定标识符——即使用户配置有缺口（TARGET_1, TARGET_3 但无 TARGET_2），
         也能正确识别，不会因 idx 错位而漏认/错路由。
      2. 对 TARGETS 列表中的每个 name：
         a. 若 name 作为某 TARGET_<N>_NAME 的 declared 值出现 → 用该编号前缀
         b. 否则若有 TARGET_<name.lower>_* 配置 → 用 name 前缀
         c. 都无 → 跳过（不凭空造目标）
    """
    import re

    load_env()
    # 预扫描编号前缀：{declared_name: num_prefix}
    numbered: dict[str, str] = {}
    for key in os.environ:
        m = re.match(r"^TARGET_(\d+)_NAME$", key)
        if m:
            declared = os.environ[key].strip()
            if declared:
                numbered[declared] = f"TARGET_{m.group(1)}"

    result: dict[str, str] = {}
    names_set = set(names)
    # 编号前缀优先：declared name 在 TARGETS 列表中 → 记录
    for declared, prefix in numbered.items():
        if declared in names_set and declared not in result:
            result[declared] = prefix
    # name 前缀回退：TARGET_<name.lower>_*（针对未被编号覆盖的列表 name）
    for nm in names:
        if nm in result:
            continue
        key = nm.lower()
        if os.getenv(f"TARGET_{key}_BASE_URL") or os.getenv(f"TARGET_{key}_API_KEY"):
            result[nm] = f"TARGET_{key}"
    return result


def load_targets() -> dict[str, "TargetConfig"]:
    """
    扫描 .env，返回 {defender_name: TargetConfig}。

    优先级：
      1. 定义了 TARGETS=name1,name2 → 逐个读 TARGET_<N>_* 四件套
         （TARGET_<N>_NAME 必填；TYPE/API_KEY/BASE_URL/MODEL 缺失则回退默认）
      2. 否则回退旧单目标：读 TARGET_* 一组，defender_name = TARGET_MODEL

    所有目标混合 backend：每目标的 type 取 TARGET_<N>_TYPE，缺失取全局 TARGET_TYPE。
    type 不放 TargetConfig（它是协议路由，非连接参数），由调用方另行读取——
    此处只负责"连接配置 + 名称"。
    """
    load_env()
    targets: dict[str, TargetConfig] = {}

    raw_list = os.getenv("TARGETS", "").strip()
    if raw_list:
        names = [n.strip() for n in raw_list.split(",") if n.strip()]
        # F-4 修复：复用共享映射，保证与 target_backend 一致
        for declared, prefix in _resolve_target_prefixes(names).items():
            targets[declared] = _target_from_prefixed(prefix, declared)

    if not targets:
        # 回退：旧单目标 TARGET_*
        legacy = TargetConfig.from_env()
        if legacy.api_key or legacy.base_url != DEFAULT_BASE_URL or os.getenv("TARGET_MODEL"):
            targets[os.getenv("TARGET_MODEL", DEFAULT_MODEL)] = legacy

    return targets


def target_backend(name: str) -> str:
    """读取某目标的 backend 类型（openai / local_sim / pcap_judge）。

    多目标下取 TARGET_<name>_TYPE（或编号映射），缺失取全局 TARGET_TYPE。

    F-4 修复：复用 _resolve_target_prefixes 与 load_targets 保持映射一致。
    """
    load_env()
    raw_list = os.getenv("TARGETS", "").strip()
    if raw_list:
        names = [n.strip() for n in raw_list.split(",") if n.strip()]
        prefix_map = _resolve_target_prefixes(names)
        if name in prefix_map:
            t = os.getenv(f"{prefix_map[name]}_TYPE")
            if t:
                return t
    return os.getenv("TARGET_TYPE", "openai")


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
