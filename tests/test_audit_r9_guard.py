"""r9 整理轮守卫测试。

  - P3-1: 预览截断常量单源（params.PREVIEW_*）+ 训练/预筛耦合口径。
  - P3-2: 路径常量冻结导入的 AST 守卫——work-dir 隔离靠 isolation 重绑"冻结
    消费方"的模块属性生效，任何新的顶层 `from llmsec.core.config import
    <路径常量>` 都会静默绕过隔离（M-1/M-2/L-9 的病根模式）。
    守卫只允许白名单内的既有模块；P3-4 迁移一个、白名单收缩一个，
    新增违例立即红。
"""

from __future__ import annotations

import ast
from pathlib import Path

# ---- P3-1：常量存在且口径被钉住 ----

def test_preview_constants_pinned():
    from llmsec.params import PREVIEW_LOG, PREVIEW_PROMPT, PREVIEW_RESPONSE

    assert PREVIEW_RESPONSE == 500
    assert PREVIEW_PROMPT == 300
    assert PREVIEW_LOG == 200


def test_prescreen_input_matches_training_constant(monkeypatch):
    """judge.evaluate 的预筛输入截断 = 训练侧 response_preview 截断（同一常量）。

    L-3 的教训：两处各自硬编码 500/全文时必然再次漂移。
    """
    import llmsec.evaluation.judge as jd
    import llmsec.evaluation.prescreen_ml as pml
    from llmsec.params import PREVIEW_RESPONSE

    seen: list[str] = []
    monkeypatch.setattr(pml, "predict",
                        lambda text: seen.append(text) or "refusal")

    judge = jd.Judge(None)
    judge.evaluate("safe prompt", "x" * (PREVIEW_RESPONSE + 5000))

    assert seen and all(len(t) <= PREVIEW_RESPONSE for t in seen)


# ---- P3-2：冻结导入 AST 守卫（P6：拦截集 = isolation.REBOUND_PATHS，白名单清零）----

# 单一来源：isolation 实际重绑的常量集 == 守卫拦截集，两处不再漂移。
# 静态锚点（PROJECT_ROOT/OUTPUT_DIR/RUNS_DIR/TASK_LOG_DIR/ATTACKS_DIR...）永不
# 重绑，冻结导入无害、不拦。
from llmsec.core.isolation import REBOUND_PATHS as _REBOUND_PATHS


def _frozen_path_imports(tree: ast.Ast, rel: str) -> list[str]:
    """返回该模块顶层 import 的冻结重绑常量名。"""
    names: list[str] = []
    for node in tree.body:  # 只查模块顶层（函数内调用期导入 = 动态读，合法）
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        is_config = (
            module in ("llmsec.core.config", "llmsec.core", "llmsec.core.config")
            or (module == "config" and rel.startswith("llmsec/core/"))
            or (module == "core" and rel.startswith("llmsec/"))
        )
        if not is_config:
            continue
        for alias in node.names:
            if alias.name in _REBOUND_PATHS:
                names.append(alias.name)
    return names


def test_no_frozen_rebound_path_imports():
    """重绑常量（REBOUND_PATHS）禁止任何顶层冻结导入——白名单已清零。

    全部消费方已迁移 `import llmsec.core.config as _config` + 调期动态读。
    """
    root = Path(__file__).resolve().parent.parent / "llmsec"
    offenders: list[str] = []
    for py in sorted(root.rglob("*.py")):
        rel = py.relative_to(root.parent).as_posix()
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            offenders.append(f"{rel}: <syntax error>")
            continue
        frozen = _frozen_path_imports(tree, rel)
        if frozen:
            offenders.append(f"{rel}: {sorted(set(frozen))}")
    assert not offenders, (
        "发现重绑路径常量的顶层冻结导入（work-dir 隔离将被静默绕过）。\n"
        "改法：消费处改为 `import llmsec.core.config as _config` + 调期 "
        "`_config.XXX_FILE`。\n" + "\n".join(offenders))


# ============================================================
# storage 重构守卫：DAO 边界（SQL/ORM 只存在于 llmsec/storage/ 内）
# ============================================================

# storage 包外禁止 import 的数据访问模块（含驱动 sqlite3——control 若直连库，
# 薄契约边界即被绕过；经 control.core.storage 的 re-export 是唯一许可路径）
_DB_IMPORT_MODULES = ("sqlite3", "sqlalchemy", "sqlmodel", "aiosqlite")


def _db_driver_imports(tree: ast.Ast) -> list[str]:
    """返回该模块顶层 import 的 DB 驱动/ORM 模块名（含 from X import 形式）。"""
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names if a.name.split(".")[0] in _DB_IMPORT_MODULES]
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod in _DB_IMPORT_MODULES:
                names.append(node.module or mod)
    return names


def test_db_imports_confined_to_storage_package():
    """sqlite3/sqlalchemy/sqlmodel 只允许出现在 llmsec/storage/ 与 tests/。

    这是"DAO 收口"的机器强制：service 层（management/server/tui/mcp/experiments）
    与 control 层只能经 llmsec.storage.contract（或 control.core.storage 薄
    re-export）访问目录库——SQL/引擎/连接细节对业务代码不可见。
    """
    repo = Path(__file__).resolve().parent.parent
    offenders: list[str] = []
    for pkg in ("llmsec", "control"):
        root = repo / pkg
        for py in sorted(root.rglob("*.py")):
            rel = py.relative_to(repo).as_posix()
            if rel.startswith("llmsec/storage/"):
                continue
            try:
                tree = ast.parse(py.read_text(encoding="utf-8"))
            except SyntaxError:
                offenders.append(f"{rel}: <syntax error>")
                continue
            hits = _db_driver_imports(tree)
            if hits:
                offenders.append(f"{rel}: {sorted(set(hits))}")
    assert not offenders, (
        "storage 包外出现 DB 驱动/ORM import（DAO 边界被绕过）。\n"
        "改法：数据访问统一走 `from llmsec.storage import contract`（control 侧经 "
        "control.core.storage），SQL/引擎只存在于 llmsec/storage/ 内。\n"
        + "\n".join(offenders))


# ============================================================
# P3-3: pcap 三件套函数化（运行期 env 生效；常量只是 import 期快照）
# ============================================================

def test_pcap_functions_runtime_sensitive(monkeypatch):

    from llmsec.targets.pcap import (
        pcap_judge_url,
        pcap_model_version,
        pcap_prompt_key,
    )

    monkeypatch.setenv("PCAP_JUDGE_URL", "https://runtime.local/judge")
    monkeypatch.setenv("PCAP_MODEL_VERSION", "RuntimeModel-Y")
    monkeypatch.setenv("PCAP_PROMPT_KEY", "custom:runtime2")

    assert pcap_judge_url() == "https://runtime.local/judge"
    assert pcap_model_version() == "RuntimeModel-Y"
    assert pcap_prompt_key() == "custom:runtime2"


def test_resolve_defender_name_runtime_sensitive(monkeypatch):
    """pcap 模式防御方名 = PCAP_MODEL_VERSION（运行期改 env 即生效）。"""
    from llmsec.core.config import resolve_defender_name

    monkeypatch.setenv("TARGET_TYPE", "pcap_judge")
    monkeypatch.setenv("PCAP_MODEL_VERSION", "LateModel-Z")
    assert resolve_defender_name("whatever") == "LateModel-Z"


# ============================================================
# P3-5: 缓存助手行为
# ============================================================

def test_sigcache_hit_invalidate_and_evict():
    from llmsec.core.caches import SigCache

    cache = SigCache(maxsize=2)
    loads = []

    def loader(v):
        loads.append(v)
        return {"v": v}

    assert cache.get("k", 1, lambda: loader("a")) == {"v": "a"}
    assert cache.get("k", 1, lambda: loader("b")) == {"v": "a"}   # sig 相同命中，不重载
    assert cache.get("k", 2, lambda: loader("c")) == {"v": "c"}   # sig 变化失效
    assert loads == ["a", "c"]

    cache.get("x", 1, lambda: 1)
    cache.get("y", 1, lambda: 2)
    cache.get("z", 1, lambda: 3)                                  # 超上限淘汰最旧
    assert len(cache._data) == 2
    cache.clear()
    assert not cache._data


def test_ttlcache_expiry_and_clear():
    import time

    from llmsec.core.caches import TTLCache

    cache = TTLCache(ttl=0.05)
    loads = []
    assert cache.get(lambda: loads.append(1) or "v1") == "v1"
    assert cache.get(lambda: loads.append(1) or "v2") == "v1"     # 未过期命中
    time.sleep(0.08)
    assert cache.get(lambda: loads.append(1) or "v3") == "v3"     # 过期重载
    assert loads == [1, 1], "仅首次与过期后各加载一次，命中不加载"
    cache.clear()
    assert cache.get(lambda: "v4") == "v4"


# ============================================================
# P3-6: task_manager TaskSpec 类型化
# ============================================================

def test_task_spec_schema_and_dict_bridge(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import llmsec.server.task_manager as tm

    spec = tm.TaskSpec(task_id="q1", kind="r9", cmd="x", argv=["x"],
                       log_path=tmp_path / "q1.log")
    # dict 桥：存量读/写点与 .get 兼容
    assert spec["status"] == "queued"
    assert spec.get("returncode") is None
    spec["status"] = "running"
    assert spec.status == "running"
    assert spec["_task_id"] == "q1"          # 告警/僵尸检测的既有引用名

    # start_task 产出的必须是 TaskSpec（schema 单一定义处）
    tm.TASKS.clear()
    import llmsec.core.config as cfg
    monkeypatch.setattr(cfg, "TASK_LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(tm.subprocess, "Popen",
                        lambda *a, **kw: SimpleNamespace(
                            pid=1, poll=lambda: None, returncode=None,
                            terminate=lambda: None, wait=lambda timeout=None: None))
    try:
        view = tm.start_task("r9b", ["-c", "pass"])
        tid = view["id"]
        assert isinstance(tm.TASKS[tid], tm.TaskSpec)
        assert tm.TASKS[tid].argv == ["-c", "pass"]
    finally:
        t = next(iter(tm.TASKS.values()), None)
        if t is not None and getattr(t, "log_file", None) is not None:
            t["log_file"].close()
        tm.TASKS.clear()


# ============================================================
# CI 修复回归：TF-IDF 降级的同质语料防护
# ============================================================

def test_tfidf_fallback_homogeneous_corpus(monkeypatch):
    """无 embedding 通道 + 同质语料（模板化攻击集）时，TF-IDF 词项全被
    max_df=0.9 剪除 → 原 ValueError 崩掉特征提取。降级路径必须放宽重拟合。

    CI 复现路径：runner 离线测试的 8 条 "attack prompt i"（attack/prompt
    在全部文档出现，df=1.0 > 0.9，数字被默认 token_pattern 丢弃）。
    """
    import llmsec.clustering.features as feats

    monkeypatch.setattr(feats, "_get_embedding_model", lambda: None)
    prompts = [f"attack prompt {i}" for i in range(8)]
    emb, vec, _ = feats.extract_text_embeddings(prompts)
    assert emb.shape[0] == 8, "全部样本都应有特征向量"
    assert emb.shape[1] >= 1, "同质语料放宽 max_df 后必须有存活词项"
    assert vec is not None


# ---- P1 修复守卫：env 覆盖入口必须在全部常量定义之后 ----

def test_env_override_call_after_all_constants():
    """_apply_env_overrides() 调用点必须晚于所有顶层常量赋值。

    此前调用点在 §9 末尾，其后定义的 §10-§13 常量（SAMPLERS/PREVIEW_*/
    ATTACK_*/RECTIFY_*）覆盖静默失效——HPO 调参白跑。AST 结构守卫防止
    未来在文件中部重新插入调用点。
    """
    import llmsec.params as p

    tree = ast.parse(Path(p.__file__).read_text(encoding="utf-8"))
    call_lines = []
    last_const_line = 0
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(n.isupper() for n in names):
                last_const_line = max(last_const_line, node.end_lineno)
        elif (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
              and isinstance(node.value.func, ast.Name)
              and node.value.func.id == "_apply_env_overrides"):
            call_lines.append(node.end_lineno)
    assert call_lines, "params.py 必须在模块级调用 _apply_env_overrides()"
    assert min(call_lines) > last_const_line, (
        f"_apply_env_overrides() 调用（行{call_lines}）必须晚于最后一个常量定义"
        f"（行{last_const_line}）——否则其后定义的常量 env 覆盖静默失效")


def test_env_override_covers_trailing_constants():
    """功能性验证：文件尾部的 §11 常量（PREVIEW_LOG）可被子进程 env 覆盖。"""
    import subprocess
    import sys

    code = ("import os; os.environ['LLMSEC_PARAM_PREVIEW_LOG'] = '77'; "
            "import llmsec.params as p; "
            "assert p.PREVIEW_LOG == 77, p.PREVIEW_LOG")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, (
        f"PREVIEW_LOG 覆盖失效（P1 回归）: {r.stderr[-300:]}")
