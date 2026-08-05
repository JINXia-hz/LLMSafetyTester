#!/usr/bin/env python3
"""
回归测试：结果矩阵 R（多模型唯一真相）+ 多目标扫描 + Elo 派生。

验证 P1 数据模型：
1. ResultsMatrix: upsert/get/列访问/时序/round-trip 持久化。
2. load_targets: 多目标编号前缀 + legacy 单目标回退。
3. derive_elo: R 是唯一真相（可重算且幂等）、胜场越多 Elo 越高。
"""

import os
import tempfile
from pathlib import Path

from llmsec.core.results import ResultsMatrix
from llmsec.evaluation.elo import derive_elo


def test_results_matrix_basics():
    """upsert/get/列访问/覆盖率/时序。"""
    mat = ResultsMatrix()
    mat.upsert("DAN", "qwen", 3.0, status="fully_compliant", ts=2)
    mat.upsert("DAN", "gpt", -2.0, status="refused", ts=1)
    mat.upsert("rot13", "qwen", 1.5, ts=3)

    if mat.get("DAN", "qwen").eval_score != 3.0:
        print("❌ get 取值错误"); return 1
    if set(mat.model_column("qwen").keys()) != {"DAN", "rot13"}:
        print("❌ model_column 错误"); return 1
    if set(mat.method_row("DAN").keys()) != {"qwen", "gpt"}:
        print("❌ method_row 错误"); return 1
    if mat.tested_methods("gpt") != {"DAN"} or mat.n_for_model("gpt") != 1:
        print("❌ 覆盖率统计错误"); return 1
    # 时序：gpt 列只有 DAN(ts=1)；qwen 列应按 ts 升序
    qwen_order = [r.method for r in mat.ordered_results("qwen")]
    if qwen_order != ["DAN", "rot13"]:
        print(f"❌ ordered_results 时序错误: {qwen_order}"); return 1
    if sorted(mat.all_models()) != ["gpt", "qwen"]:
        print("❌ all_models 错误"); return 1


def test_results_matrix_roundtrip():
    """save → load 幂等。"""
    mat = ResultsMatrix(methods=["DAN", "rot13"])
    mat.upsert("DAN", "qwen", 3.0, status="fully_compliant", ts=1, extra={"len": 42})
    mat.upsert("rot13", "qwen", -1.0, ts=2)
    with tempfile.TemporaryDirectory() as d:
        p = mat.save(Path(d) / "r.json")
        mat2 = ResultsMatrix.load(p)
    if mat2.get("DAN", "qwen").eval_score != 3.0:
        print("❌ round-trip eval_score 丢失"); return 1
    if mat2.get("DAN", "qwen").status != "fully_compliant":
        print("❌ round-trip status 丢失"); return 1
    if mat2.get("DAN", "qwen").extra.get("len") != 42:
        print("❌ round-trip extra 丢失"); return 1
    if mat2.get("rot13", "qwen").eval_score != -1.0:
        print("❌ round-trip 第二条丢失"); return 1


def test_load_targets_multi_and_legacy():
    """多目标编号前缀 + legacy 单目标回退。"""
    from llmsec.core.config import load_targets

    saved = {k: os.environ.get(k) for k in os.environ}
    try:
        # 清掉可能干扰的键
        for k in ("TARGETS", "TARGET_TYPE", "TARGET_API_KEY", "TARGET_BASE_URL", "TARGET_MODEL",
                  "TARGET_1_NAME", "TARGET_1_BASE_URL", "TARGET_1_API_KEY", "TARGET_1_MODEL", "TARGET_1_TYPE",
                  "TARGET_2_NAME", "TARGET_2_BASE_URL", "TARGET_2_API_KEY", "TARGET_2_MODEL"):
            os.environ.pop(k, None)

        # 场景 A：多目标编号前缀
        os.environ["TARGETS"] = "qwen9b,gpt4o"
        os.environ["TARGET_1_NAME"] = "qwen9b"
        os.environ["TARGET_1_BASE_URL"] = "http://h1/v1"
        os.environ["TARGET_1_API_KEY"] = "k1"
        os.environ["TARGET_1_MODEL"] = "Qwen3.5-9B"
        os.environ["TARGET_2_NAME"] = "gpt4o"
        os.environ["TARGET_2_BASE_URL"] = "http://h2/v1"
        os.environ["TARGET_2_API_KEY"] = "k2"
        os.environ["TARGET_2_MODEL"] = "gpt-4o"
        t = load_targets()
        if set(t.keys()) != {"qwen9b", "gpt4o"}:
            print(f"❌ 多目标扫描错误: {list(t.keys())}"); return 1
        if t["qwen9b"].base_url != "http://h1/v1" or t["gpt4o"].model != "gpt-4o":
            print("❌ 多目标字段映射错误"); return 1

        # 场景 B：legacy 单目标回退（无 TARGETS）
        for k in ("TARGETS", "TARGET_1_NAME", "TARGET_1_BASE_URL", "TARGET_1_API_KEY", "TARGET_1_MODEL",
                  "TARGET_2_NAME", "TARGET_2_BASE_URL", "TARGET_2_API_KEY", "TARGET_2_MODEL"):
            os.environ.pop(k, None)
        os.environ["TARGET_MODEL"] = "LegacyModel"
        os.environ["TARGET_BASE_URL"] = "http://legacy/v1"
        os.environ["TARGET_API_KEY"] = "lk"
        t2 = load_targets()
        if list(t2.keys()) != ["LegacyModel"] or t2["LegacyModel"].base_url != "http://legacy/v1":
            print(f"❌ legacy 回退错误: {list(t2.keys())}"); return 1
    finally:
        # 还原
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_derive_elo_deterministic_and_monotone():
    """R 是唯一真相：两次派生幂等；胜场越多 Elo 越高；不跨模型。"""
    mat = ResultsMatrix()
    # winner 对 qwen 全胜，loser 对 qwen 全败
    for ts in range(1, 6):
        mat.upsert("winner", "qwen", 3.0, ts=ts)
    for ts in range(6, 11):
        mat.upsert("loser", "qwen", -2.0, ts=ts)

    t1 = derive_elo(mat, "qwen")
    t2 = derive_elo(mat, "qwen")
    # 幂等：R 不变 → Elo 完全一致
    if t1.get_attacker_elo("winner") != t2.get_attacker_elo("winner"):
        print("❌ derive_elo 非幂等（R 未变但 Elo 不同）"); return 1
    # 单调：winner Elo > loser Elo
    if t1.get_attacker_elo("winner") <= t1.get_attacker_elo("loser"):
        print("❌ 胜场多者 Elo 未更高"); return 1
    # 不跨模型：gpt 列无结果 → 派生时 winner/loser 保持初始 Elo
    tg = derive_elo(mat, "gpt")
    if tg.get_attacker_elo("winner") != 1500.0:
        print("❌ Elo 跨模型泄漏（gpt 列空却变了 Elo）"); return 1


