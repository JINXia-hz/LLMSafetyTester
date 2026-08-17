"""
core.config — 统一配置入口

职责：
  1. 幂等加载项目根目录 .env（替代原 7 处逐字重复的 load_dotenv 调用）
  2. 集中管理路径常量（历史三套路径已统一裁决为 output/state/、attacks/，
     即 runner.py 现行约定；旧路径仅作读取兼容回退，写入只写新路径）
  3. TargetConfig / GeneratorConfig / JudgeConfig dataclass + from_env()
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# ============================================================
# .env 加载（幂等）
# ============================================================
_ENV_LOADED = False


def load_env() -> bool:
    """
    幂等加载仓库根目录的 .env。
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
# 路径常量（全部锚定仓库根，不落进 llmsec/ 包内部）
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "output"
STATE_DIR = OUTPUT_DIR / "state"
ATTACKS_DIR = PROJECT_ROOT / "attacks"
RUNS_DIR = OUTPUT_DIR / "runs"

# R 矩阵为唯一真相（results.json）；全局 state.json 已废弃。
# per-run 快照在 runs/<ts>/state.json，per-target 快照在 state__<name>.json。
SAFE_TWINS_FILE = STATE_DIR / "safe_twins.jsonl"

# 结果矩阵（多模型唯一真相）+ 派生缓存目录
# R 真相 = results.db（SQLite，storage.rstore 后端；阶段 2 起 results.json 只是
# 导出快照格式）。work-dir 隔离时重绑到 <wd>/results.db（见 core/isolation.py）。
RESULTS_DB = STATE_DIR / "results.db"
RESULTS_FILE = STATE_DIR / "results.json"          # 遗留快照格式（导出/对账用）
PREDICTORS_DIR = OUTPUT_DIR / "predictors"          # 统一/每模型 ridge 预测器
ELO_CACHE_FILE = STATE_DIR / "elo_cache.json"       # Elo 派生缓存（可删可重建）

# 目录库（SQLite：runs/trials/tasks 登记索引，llmsec/storage/ 唯一读写方）。
# 可重建的派生索引——删库后 `llmsec-manage storage reindex` 全量重建。
# work-dir 隔离时重绑到 <wd>/catalog.db 卫星库（见 core/isolation.py）。
CATALOG_DB = STATE_DIR / "catalog.db"

# 攻击集（仓库根 attacks/ 目录——用户可见，支持拖拽上传）
ATTACK_SET_L1_FILE = ATTACKS_DIR / "l1.jsonl"

# 过敏检测产物（规范存储，按模型隔离见 W4）
TWIN_RESULT_FILE = OUTPUT_DIR / "allergy_results.jsonl"

# 报告产物（reporting 写、dashboard 读；具体文件名在 report.py 内以字面量使用）
CLUSTER_SECURITY_ANALYSIS_FILE = OUTPUT_DIR / "cluster_security_analysis.json"

# 后台任务日志目录（dashboard 子进程任务）
TASK_LOG_DIR = OUTPUT_DIR / "tasks"

# 持久化日志目录（RotatingFileHandler 落盘，monitoring 监控设施）
LOG_DIR = OUTPUT_DIR / "logs"
LOG_FILE = LOG_DIR / "llmsec.log"

# 告警事件文件（append-only JSONL，webhook + 事件文件双通道）
ALERTS_FILE = OUTPUT_DIR / "alerts.jsonl"

# 内置静态数据目录（HarmBench 行为库 + 越狱模板，出处见 data/Explication.md）
DATA_DIR = PROJECT_ROOT / "data"

# 聚类/特征产物统一归 output/cluster/（2026-08 storage 重构前散落 output/ 根目录）
# 聚类 artifacts 已按写者拆分为两个文件（原 cluster_artifacts.pkl 由两个写者混写不同
# schema，后写覆盖会导致下游 KeyError）：
#   - feature_cache.pkl：先验特征缓存，仅 predictors/cold_start.py fit_features 写
#   - cluster_result.pkl：完整聚类产物，hdb 写、final_fit 增补
CLUSTER_DIR = OUTPUT_DIR / "cluster"
CLUSTER_REPORT_FILE = CLUSTER_DIR / "cluster_report.json"
CLUSTER_MATRIX_FILE = CLUSTER_DIR / "cluster_matrix.csv"
FEATURE_CACHE_FILE = CLUSTER_DIR / "feature_cache.pkl"
CLUSTER_RESULT_FILE = CLUSTER_DIR / "cluster_result.pkl"
# Embedding 磁盘缓存：按 (feature_config_hash, prompt_sha256) 键存原始 embedding 向量，
# 冷启动时只 encode 缓存未命中的 prompt。features.py 动态读此常量（不 import 期冻结）。
EMBEDDING_CACHE_FILE = CLUSTER_DIR / "embedding_cache.pkl"


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
def _target_from_prefixed(prefix: str) -> "TargetConfig":
    """从 TARGET_<N>_* 一组变量构造一个目标（prefix 形如 'TARGET_1'）。"""
    return TargetConfig(
        api_key=os.getenv(f"{prefix}_API_KEY", ""),
        base_url=os.getenv(f"{prefix}_BASE_URL", DEFAULT_BASE_URL),
        model=os.getenv(f"{prefix}_MODEL", DEFAULT_MODEL),
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
            targets[declared] = _target_from_prefixed(prefix)

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


def resolve_defender_name(target_model: str) -> str:
    """解析防御方（被攻击模型）名称：pcap 模式用 PCAP_MODEL_VERSION，其它用 target_model。

    替代 evaluator.py / safe_twin.py / runner.py 重复的三元表达式（M-35）。
    R 矩阵的结果列、过敏结果、画像 ASR 都按此名索引——三处必须口径一致，
    否则 pcap 模式下 FPR 与 ASR 会查不同列导致画像错配。

    PCAP_MODEL_VERSION 从 llmsec.targets 惰性读取（避免 core.config ↔ targets 循环导入），
    取值与原各模块顶层 `from llmsec.targets import PCAP_MODEL_VERSION` 一致（import 期冻结）。
    """
    if os.getenv("TARGET_TYPE", "openai") == "pcap_judge":
        from llmsec.targets.pcap import pcap_model_version
        return pcap_model_version()
    return target_model


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
            timeout=float(os.getenv("GENERATOR_TIMEOUT", "60.0")),
            max_tokens=int(os.getenv("GENERATOR_MAX_TOKENS", "4096")),
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
    timeout: float = 90.0       # 推理模型（minimax 等）首 token 常 >30s，30s 系统性超时
    max_retries: int = 2
    temperature: float = 0.0
    max_tokens: int = 1024      # 推理模型把 token 预算耗在 reasoning，512 会截断最终 JSON

    @classmethod
    def from_env(cls) -> "JudgeConfig":
        load_env()
        # M-23：JUDGE_MODEL 缺省回退 GENERATOR_MODEL（与 README 一致），最后才回退 DEFAULT_MODEL。
        # 原实现回退硬编码 DEFAULT_MODEL=deepseek-v4-flash，GENERATOR 配到非 deepseek 服务商
        # 且不设 JUDGE_MODEL 时，Judge 用对方 base_url 请求 "deepseek-v4-flash" → 404。
        model = os.getenv("JUDGE_MODEL") or os.getenv("GENERATOR_MODEL") or DEFAULT_MODEL
        return cls(
            api_key=os.getenv("GENERATOR_API_KEY", os.getenv("JUDGE_API_KEY", "")),
            base_url=os.getenv("GENERATOR_BASE_URL", DEFAULT_BASE_URL),
            model=model,
            timeout=float(os.getenv("JUDGE_TIMEOUT", "90.0")),
            max_tokens=int(os.getenv("JUDGE_MAX_TOKENS", "1024")),
        )
