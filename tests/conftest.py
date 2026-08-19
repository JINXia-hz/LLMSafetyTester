"""共享测试引导：路径注入 + Windows 控制台 UTF-8 + 网络隔离。

各 test 模块不再各自 sys.path.insert / reconfigure / setup_console——统一在此完成。
pytest 收集任何 test 模块前会先加载本文件。
"""

import os
import sys
from pathlib import Path

import pytest

# 让 tests/ 能 import llmsec 包（项目根 = tests/ 的父目录）
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Windows 控制台 UTF-8：emoji / 中文断言与失败信息可正确打印（幂等）
from llmsec.core.logging import setup_console

setup_console()

# 预先把 .env 灌进 os.environ。config.load_env() 有 _ENV_LOADED 幂等保护，
# 这里先调一次：把仓库 .env 的内网地址提前注入，后续测试内再调 from_env() 时
# load_env() 会直接返回、不会重新注入——这样下面的 _isolate_network_env fixture
# 才能可靠地 pop 掉它们（否则 fixture 跑在 load_env() 之前，pop 个空，测试函数
# 内部 from_env() 一调又把内网地址灌回来了）。
from llmsec.core.config import load_env

load_env()


# 触发真实网络调用的环境变量前缀/名单。
# 这些变量在 .env 里通常配的是内网服务地址（embedding 服务 / 生成模型 / judge），
# 测试环境不可达时会让隐式走网络路径的测试（如 run_attack_phase 内部的 embedding
# 探活、ai_rename_clusters 的 LLM 调用）卡满 HTTP 超时（~60-120s/次）才降级。
# 测试期间统一清空，让对应代码走"未配置→早退"分支，保持套件离线、秒级。
_NETWORK_ENV_KEYS = (
    "EMBEDDING_API_BASE", "EMBEDDING_API_KEY", "EMBEDDING_API_MODEL",
    "GENERATOR_API_KEY", "GENERATOR_BASE_URL",
    "JUDGE_API_KEY", "JUDGE_BASE_URL",
)


@pytest.fixture(autouse=True)
def _isolate_network_env(request):
    """每个测试期间清空网络相关环境变量，结束后恢复，避免 .env 内网地址泄漏进测试。

    恢复用完整快照 diff（新增键也一并清除）——此前只恢复被 pop 的键，
    测试期间新注入的同名前缀键会残留给同 worker 的后续测试。

    标了 real_api / e2e marker 的用例**不清**——它们需要 .env 注入的真实网络凭证
    去打外部 API，清掉会让代码走"未配置→早退"分支而失去测试意义。
    """
    if request.node.get_closest_marker("real_api") or request.node.get_closest_marker("e2e"):
        # 真实 API / 端到端测试：保留 .env 凭证，不清
        yield
        return
    saved = {k: os.environ.pop(k) for k in _NETWORK_ENV_KEYS if k in os.environ}
    before_keys = set(os.environ)
    yield
    os.environ.update(saved)
    # 清掉测试期间新增的键（含同名前缀的新注入），还原到进入前的键集
    for k in set(os.environ) - before_keys:
        os.environ.pop(k, None)


@pytest.fixture(autouse=True)
def _hermetic_catalog(tmp_path):
    """目录库与 R 真相库按测试隔离（全 suite autouse）。

    背景：task_manager._persist_meta 的目录库镜像、runner 的 register_run、
    ResultsMatrix.save()（阶段 2 起默认写 results.db）等写入口会在不经意间
    触碰**真实**库文件（测试只重定向了 TASK_LOG_DIR / runs 根时，常量仍指向
    真实 output）。任何测试都不该往真实库写行/观测。

    显式需要真实库的用例（当前无）可用 marker 豁免。teardown 顺手释放引擎，
    防止 tmp 库的连接句柄滞留到后续测试（Windows 上会锁住 pytest 的 tmp 清理）。
    """
    import llmsec.core.config as cfg
    from llmsec.storage import db as storage_db

    saved_catalog = cfg.CATALOG_DB
    cfg.CATALOG_DB = tmp_path / "catalog.db"
    # P9 补遗：派生缓存/报告路径一并隔离。此前只隔离库文件——runner 全局模式
    # 测试（_should_refresh_features 判定小方法集 ≠ 真实缓存 → 重算覆写）把
    # 真实 output/cluster 四件套打成测试规模数据；safe_twin 的 allergy__<模型>
    # 报告与 .env.bak 同族泄漏（OUTPUT_DIR 动态读，重绑即生效）。
    _saved_paths = {k: getattr(cfg, k) for k in (
        "OUTPUT_DIR", "TWIN_RESULT_FILE", "SAFE_TWINS_FILE",
        "CLUSTER_DIR", "FEATURE_CACHE_FILE", "CLUSTER_RESULT_FILE",
        "CLUSTER_REPORT_FILE", "CLUSTER_MATRIX_FILE", "EMBEDDING_CACHE_FILE",
    )}
    out = tmp_path / "output"
    cluster = out / "cluster"
    state = out / "state"
    for k, v in {
        "OUTPUT_DIR": out, "TWIN_RESULT_FILE": out / "allergy_results.jsonl",
        "SAFE_TWINS_FILE": state / "safe_twins.jsonl",
        "CLUSTER_DIR": cluster, "FEATURE_CACHE_FILE": cluster / "feature_cache.pkl",
        "CLUSTER_RESULT_FILE": cluster / "cluster_result.pkl",
        "CLUSTER_REPORT_FILE": cluster / "cluster_report.json",
        "CLUSTER_MATRIX_FILE": cluster / "cluster_matrix.csv",
        "EMBEDDING_CACHE_FILE": cluster / "embedding_cache.pkl",
    }.items():
        setattr(cfg, k, v)
    try:
        yield
    finally:
        cfg.CATALOG_DB = saved_catalog
        for k, v in _saved_paths.items():
            setattr(cfg, k, v)
        storage_db.close()


@pytest.fixture(autouse=True)
def _hermetic_tasks(tmp_path):
    """进程级 TASKS 注册表按测试隔离（全 suite autouse）。

    背景：xdist 把多个测试文件分到同一 worker，某文件的真实 start_task 残留任务
    会被其它文件的读取方（如 TUI 的 TaskStore.refresh）并进视图，造成跨文件污染
    （曾实际表现为任务表多出无关行）。原先 test_tui_task_store / test_tui_panels
    各自复制了局部 fixture，现统一收编。

    实现注意：必须 clear/restore **同一 dict 对象**而非替换引用——routers/tasks.py
    等消费方 `from task_manager import TASKS` 绑定的是原对象，替换引用会造成读写
    分裂（即审计 H3 修过的 bug 形态）。teardown 直接 terminate 残留子进程而不用
    cancel_task（避免 _advance_queue 在清理中途再拉起新任务），并关闭日志句柄。
    """
    import llmsec.core.config as cfg
    from llmsec.server import task_manager as tm

    # 任务日志目录一并隔离（P9：堵 .log/.progress.jsonl 写进真实 output/tasks
    # ——此前只隔离库行，测试起的任务把日志漏在真目录里；TASK_LOG_DIR 已
    # 全线改为调用期动态读，此处重绑即全局生效）
    saved_tasklog = cfg.TASK_LOG_DIR
    cfg.TASK_LOG_DIR = tmp_path / "tasks"
    saved = dict(tm.TASKS)
    tm.TASKS.clear()
    try:
        yield
    finally:
        for tid, t in list(tm.TASKS.items()):
            proc = t.get("proc")
            if t.get("status") == "running" and proc is not None:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    pass
            log_file = t.get("log_file")
            if log_file is not None:
                try:
                    log_file.close()
                except Exception:
                    pass
        tm.TASKS.clear()
        tm.TASKS.update(saved)
        cfg.TASK_LOG_DIR = saved_tasklog
        # Plan 队列 worker 收敛：worker 若活过测试（slow/blocking execute 的
        # 用例），teardown dispose 引擎池时会与它迟来的 finish_queue_item
        # 落库竞态（CI: Set changed size during iteration）。只在本测试进程
        # 已加载过 queue 模块时等待（否则不为未用过的队列付 import + 读库）。
        import sys as _sys
        import time as _time
        if "control.agent.shangshu.queue" in _sys.modules:
            from control.agent.shangshu.queue import get_queue as _get_queue

            q = _get_queue()
            deadline = _time.time() + 5
            while _time.time() < deadline:
                st = q.status()
                if st["running"] is None and not st["queued"]:
                    break
                _time.sleep(0.05)


# ============================================================
# 真实 API / 端到端测试的凭证门控
# ============================================================
# .env.example 里的占位符示例值（sk-yyy / sk-xxx），用于识别"未真正配置"的 .env，
# 避免误把模板当成真实凭证放行真实 API 测试。
_PLACEHOLDER_KEYS = ("sk-yyy", "sk-xxx")


def _has_real_credentials() -> tuple[bool, str]:
    """检测 .env 是否配了可用的真实 API 凭证。返回 (ok, reason)。

    判定标准：TARGET_API_KEY / TARGET_BASE_URL 非空且非占位符，且 GENERATOR_API_KEY
    非空且非占位符（Judge 未配独立凭证时也回退借用它）。任一缺失 → 视为未配置。
    """
    target_key = os.getenv("TARGET_API_KEY", "").strip()
    target_url = os.getenv("TARGET_BASE_URL", "").strip()
    if not target_key or any(target_key.startswith(p) for p in _PLACEHOLDER_KEYS) or not target_url:
        return False, "TARGET_API_KEY/TARGET_BASE_URL 未配置或为占位符"
    gen_key = os.getenv("GENERATOR_API_KEY", "").strip()
    if not gen_key or any(gen_key.startswith(p) for p in _PLACEHOLDER_KEYS):
        return False, "GENERATOR_API_KEY 未配置或为占位符（生成/Judge 缺省都依赖它）"
    return True, "ok"


@pytest.fixture
def require_real_api(request):
    """real_api / e2e 用例的凭证门控：无真实凭证时优雅 skip 并给出配置提示。

    用法：在标了 @pytest.mark.real_api 或 @pytest.mark.e2e 的测试函数参数里
    加 `require_real_api`。无凭证时该用例 skip（非失败），有凭证时正常执行。
    """
    ok, reason = _has_real_credentials()
    if not ok:
        pytest.skip(f"跳过真实 API 测试：{reason}（配置 .env 后可启用）")
    yield
