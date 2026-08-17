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


# ---- P3-2：冻结导入 AST 守卫 ----

# 既有冻结消费方白名单（= 当前隔离模型覆盖/容忍的集合）。
# P3-4 逐模块迁移为 `_config.X` 动态读后从此清单删除；清空之日即
# isolation 的"冻结模块重绑清单"可整体删除之时。
_FROZEN_IMPORT_ALLOWLIST = {
    "llmsec/core/__init__.py",
    "llmsec/core/progress.py",
    "llmsec/clustering/__init__.py",
    "llmsec/clustering/cli.py",
    "llmsec/clustering/features.py",
    "llmsec/attacks/generate.py",
    "llmsec/attacks/harmbench.py",
    "llmsec/evaluation/cli.py",
    "llmsec/evaluation/cluster_analysis.py",
    "llmsec/experiments/study.py",
    "llmsec/management/caches.py",
    "llmsec/management/merge.py",
    "llmsec/management/runs.py",
    "llmsec/management/snapshot.py",
    "llmsec/management/common.py",
    "llmsec/pipeline/runner.py",
    "llmsec/server/dashboard_api.py",
    "llmsec/server/task_manager.py",
    "llmsec/server/routers/cluster_viz.py",
    "llmsec/server/routers/hpo.py",
    "llmsec/server/routers/tasks.py",
    "llmsec/server/routers/data_query.py",
    "llmsec/tui/task_store.py",
}

# 路径常量命名约定：以 _FILE / _DIR / _ROOT 结尾。
_PATH_NAME_SUFFIXES = ("_FILE", "_DIR", "_ROOT")


def _frozen_path_imports(tree: ast.Ast, rel: str) -> list[str]:
    """返回该模块顶层 import 的冻结路径常量名。"""
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
            if alias.name.endswith(_PATH_NAME_SUFFIXES):
                names.append(alias.name)
    return names


def test_no_new_frozen_path_constant_imports():
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
        if frozen and rel not in _FROZEN_IMPORT_ALLOWLIST:
            offenders.append(f"{rel}: {sorted(set(frozen))}")
    assert not offenders, (
        "发现白名单之外的路径常量冻结导入（work-dir 隔离将被静默绕过）。\n"
        "改法：消费处改为 `import llmsec.core.config as _config` + 调期 "
        "`_config.XXX_FILE`（hdb.py/cold_start.py 已示范），并把该模块从本测试"
        "白名单中移除。\n" + "\n".join(offenders))


def test_frozen_allowlist_has_no_stale_entries():
    """P3-4 迁移推进的伴生守卫：模块迁完就应从白名单删除，防清单腐化。

    （迁移期间的中间态——模块已改动态读但白名单未删——会在下次全量跑时
    被本测试点名，提示收缩。）
    """
    root = Path(__file__).resolve().parent.parent / "llmsec"
    stale: list[str] = []
    for rel in _FROZEN_IMPORT_ALLOWLIST:
        p = root.parent / rel
        if not p.exists():
            stale.append(f"{rel} (文件不存在)")
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"))
        if not _frozen_path_imports(tree, rel):
            stale.append(f"{rel} (已无冻结导入，应从白名单删除)")
    assert not stale, "冻结导入白名单存在过期条目：\n" + "\n".join(stale)


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
    monkeypatch.setattr(tm, "TASK_LOG_DIR", tmp_path / "logs")
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
