"""第 7 轮审计回归——清理轮（死代码/无用参数/冗余单源）。

- 冗余收敛后的行为等价与单源一致性（method_set_hash / _try / SAMPLERS /
  RUN_NAME_RE / dir_size）。
- 死代码移除的回归护栏（不得复活）。
"""

from __future__ import annotations

# ============================================================
# 冗余收敛：method_set_hash 单源
# ============================================================


def test_method_set_hash_single_source():
    from llmsec.core.units import method_set_hash
    from llmsec.evaluation.predictors.cold_start import _compute_method_set_hash

    methods = ["b", "a", "c", "a"]
    assert method_set_hash(methods) == _compute_method_set_hash(methods), (
        "cold_start 别名必须与 core.units 的单源实现完全一致"
    )
    assert method_set_hash(["a", "b"]) == method_set_hash(["b", "a"])  # 顺序无关
    assert method_set_hash(["a"]) != method_set_hash(["b"])


def test_hdb_uses_shared_hash():
    import llmsec.clustering.hdb as hdb
    from llmsec.core.units import method_set_hash

    assert not hasattr(hdb, "_method_set_hash"), "本地重复实现应已删除"
    assert hdb.method_set_hash is method_set_hash


# ============================================================
# 冗余收敛：MCP _try 单源
# ============================================================


def test_mcp_try_single_source():
    import llmsec.mcp.tools as tools
    import llmsec.mcp.tools.actions as actions
    import llmsec.mcp.tools.query as query
    import llmsec.mcp.tools.tasks as tasks

    assert actions._try is tools._try
    assert tasks._try is tools._try
    assert query._try is tools._try

    # 行为：异常转结构化 dict
    r = tools._try(lambda: 1 / 0, error_hint="demo")
    assert isinstance(r, dict) and "ZeroDivisionError" in r["error"] and r["hint"] == "demo"
    assert tools._try(lambda: 42) == 42


# ============================================================
# 冗余收敛：SAMPLERS 单源
# ============================================================


def test_samplers_single_source():
    import llmsec.server.launch as launch
    import llmsec.tui.console as tui_console
    from llmsec.params import SAMPLERS

    assert launch._SAMPLERS == SAMPLERS
    assert set(tui_console._sampler_names()) == set(SAMPLERS)
    assert tui_console._sampler_names()[0] == "hybrid", "TUI 补全保持 hybrid 默认置顶"

    # 任务路由校验从单源派生：接受全部成员、拒绝清单外值
    import pytest
    from pydantic import ValidationError

    from llmsec.server.routers.tasks import EvaluateRequest

    for s in SAMPLERS:
        EvaluateRequest(sampler=s)
    with pytest.raises(ValidationError):
        EvaluateRequest(sampler="nope")


# ============================================================
# 冗余收敛：RUN_NAME_RE / dir_size 单源
# ============================================================


def test_run_name_re_single_source():
    """命名契约单源：data_query 与 management 都指向 storage.contract 的同一对象。"""
    import llmsec.server.routers.data_query as dq
    from llmsec.storage import contract

    assert dq.RUN_NAME_RE is contract.RUN_NAME_RE


def test_dir_size_single_source():
    import control.core.workspace as ws
    from llmsec.management.common import dir_size

    assert ws._dir_size is dir_size


# ============================================================
# 死代码移除护栏
# ============================================================


def test_dead_code_stays_removed():
    import control.agent.menxia.block as block
    import llmsec.clustering as clustering_pkg
    import llmsec.clustering.space as space
    import llmsec.mcp.confirm as confirm

    assert not hasattr(confirm, "peek")
    assert not hasattr(block, "list_pending_blocks")
    # fsig 模块整体已随 P5 删除（无消费者）
    assert not hasattr(space, "transform_to_space")
    assert "transform_to_space" not in clustering_pkg.__all__


# ============================================================
# 清理后的接口仍工作（签名收窄不破坏正常调用）
# ============================================================


def test_build_tree_and_registry_narrowed_signatures(tmp_path):
    """build_tree/build_method_registry 收窄签名后仍正常产出。"""
    from llmsec.reporting.report import build_method_registry, build_method_stats, build_tree

    results = [{"method": f"m{i}", "is_harmful": False, "harm_type": "t", "category": "c"} for i in range(6)]
    method_stats = build_method_stats(results, {}, {})
    tree = build_tree(method_stats, {"summary": {"false_positive_rate": 0.0}})
    assert "overall" in tree

    registry = build_method_registry(method_stats, {"m0": 1600.0}, results)
    assert registry["m0"]["elo"] == 1600.0
