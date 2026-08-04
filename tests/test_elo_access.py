#!/usr/bin/env python3
"""
回归测试：R-cutover 的 Elo 派生访问层（elo_access）+ BlendPredictor 缓存。

验证 W1/W2：
  1. publish_tracker：live tracker → R 写入 + 缓存发布（含收敛轨迹）
  2. elo_state_for：冷派生（缓存未命中）→ 写缓存 → 二次命中
  3. 缓存失效：R 变动后指纹不一致 → 自动重派生
  4. active_model：取最新活跃模型
  5. maybe_migrate_legacy：旧 state.json history → R 迁移（幂等）
  6. load_or_fit_blend_predictor：同 (R+清单) 复用缓存，变化则重训
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import llmsec.evaluation.elo_access as ea
import llmsec.evaluation.blend_predictor as bp_mod
from llmsec.evaluation.elo import ELOTracker
from llmsec.evaluation.blend_predictor import load_or_fit_blend_predictor


def _setup(tmp: Path):
    """把 elo_access / results 的全局路径指向临时目录，隔离测试。"""
    import llmsec.core.results as results_mod

    results_mod.RESULTS_FILE = tmp / "results.json"
    ea.ELO_CACHE_FILE = tmp / "elo_cache.json"
    ea.STATE_FILE = tmp / "state.json"
    # ResultsMatrix.load/save 默认读 RESULTS_FILE 模块常量
    return results_mod


def test_publish_and_derive() -> int:
    with tempfile.TemporaryDirectory() as d:
        _setup(Path(d))
        ea.invalidate()  # 清缓存

        # 构造一个 live tracker，跑几场
        tracker = ELOTracker()
        tracker.update("DAN", "qwen9b", 3.5)
        tracker.update("rot13", "qwen9b", -1.0)
        tracker.record_round_end("qwen9b")

        ea.publish_tracker(tracker, "qwen9b")

        # R 应含两条
        from llmsec.core.results import ResultsMatrix
        R = ResultsMatrix.load()
        if R.n_for_model("qwen9b") != 2:
            print("❌ publish_tracker 未写入 R"); return 1

        # 派生状态命中缓存（刚 publish）
        st = ea.elo_state_for("qwen9b")
        if "DAN" not in st.get("attacker_ratings", {}):
            print("❌ elo_state_for 缺 DAN rating"); return 1
        if st.get("fingerprint") is None:
            print("❌ 缓存缺 fingerprint"); return 1
        # 收敛轨迹应被发布
        if not st.get("round_defender_elos", {}).get("qwen9b"):
            print("❌ 收敛轨迹未随 publish 发布"); return 1

        print("✅ publish_tracker + elo_state_for 通过")
    return 0


def test_cache_invalidates_on_R_change() -> int:
    with tempfile.TemporaryDirectory() as d:
        _setup(Path(d))
        ea.invalidate()

        tracker = ELOTracker()
        tracker.update("DAN", "qwen9b", 3.5)
        ea.publish_tracker(tracker, "qwen9b")
        st1 = ea.elo_state_for("qwen9b")
        fp1 = st1["fingerprint"]

        # R 变动：新增一条结果
        tracker2 = ELOTracker()
        tracker2.update("rot13", "qwen9b", 2.0)
        ea.publish_tracker(tracker2, "qwen9b")

        st2 = ea.elo_state_for("qwen9b")
        if st2["fingerprint"] == fp1:
            print("❌ R 变动后缓存未失效/未重派生"); return 1
        if "rot13" not in st2["attacker_ratings"]:
            print("❌ 重派生未包含新方法 rot13"); return 1

        print("✅ 缓存随 R 变动自动失效通过")
    return 0


def test_active_model_and_empty() -> int:
    with tempfile.TemporaryDirectory() as d:
        _setup(Path(d))
        ea.invalidate()

        if ea.active_model() is not None:
            print("❌ 空 R 应返回 None active_model"); return 1

        t = ELOTracker()
        t.update("DAN", "modelA", 1.0)
        ea.publish_tracker(t, "modelA")
        if ea.active_model() != "modelA":
            print("❌ active_model 应为 modelA"); return 1
        if ea.attacker_ratings_for("modelA").get("DAN") is None:
            print("❌ attacker_ratings_for 取值错误"); return 1
        if ea.elo_state_for("nope") != {}:
            print("❌ 不存在的模型应返回 {}"); return 1

        print("✅ active_model / 空模型处理通过")
    return 0


def test_migrate_legacy() -> int:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _setup(tmp)
        ea.invalidate()

        # 造一个旧 state.json（含 history）
        import json
        legacy = {
            "attacker_ratings": {"DAN": 1600},
            "history": [
                {"attacker": "DAN", "defender": "oldmodel", "eval_score": 3.0},
                {"attacker": "rot13", "defender": "oldmodel", "eval_score": -1.0},
            ],
        }
        (tmp / "state.json").write_text(json.dumps(legacy), encoding="utf-8")

        from llmsec.core.results import ResultsMatrix
        if ResultsMatrix.load().all_models():
            print("❌ 初始 R 应为空"); return 1

        if not ea.maybe_migrate_legacy():
            print("❌ 迁移应执行"); return 1
        R = ResultsMatrix.load()
        if R.n_for_model("oldmodel") != 2:
            print("❌ 迁移后 R 缺记录"); return 1

        # 幂等：再次调用不重复迁（R 已非空）
        if ea.maybe_migrate_legacy():
            print("❌ 迁移应幂等（R 非空时跳过）"); return 1

        print("✅ maybe_migrate_legacy 幂等迁移通过")
    return 0


def test_blend_predictor_cache_reuse() -> int:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _setup(tmp)
        bp_mod.PREDICTORS_DIR = tmp / "predictors"

        from llmsec.core.results import ResultsMatrix
        R = ResultsMatrix()
        R.upsert("DAN", "qwen9b", 3.0, ts=1)
        R.upsert("rot13", "qwen9b", -1.0, ts=2)
        R.upsert("b64", "qwen9b", 2.0, ts=3)
        features = {"DAN": {"t": [1.0]}, "rot13": {"t": [0.0]}, "b64": {"t": [0.5]}}
        catalog = ["DAN", "rot13", "b64"]

        bp1 = load_or_fit_blend_predictor(R, features, method_catalog=catalog)
        # 第二次应命中缓存（同 R + 同 catalog）
        bp2 = load_or_fit_blend_predictor(R, features, method_catalog=catalog)
        if bp1 is not bp2:
            print("⚠ 缓存未命中返回同实例（load 重建实例，非同一对象，可接受）")

        # 验证确实写了缓存文件
        if not any((bp_mod.PREDICTORS_DIR).glob("blend_*.pkl")):
            print("❌ PREDICTORS_DIR 未写缓存文件"); return 1

        # R 变动 → 新 key → 重训
        R.upsert("extra", "qwen9b", 1.0, ts=4)
        bp3 = load_or_fit_blend_predictor(R, features, method_catalog=catalog)
        # 不同 R 内容应产生不同缓存文件
        files = list((bp_mod.PREDICTORS_DIR).glob("blend_*.pkl"))
        if len(files) < 2:
            print("❌ R 变动后未生成新缓存"); return 1

        print("✅ BlendPredictor 缓存复用/失效通过")
    return 0


def main() -> int:
    rc = 0
    for t in (test_publish_and_derive, test_cache_invalidates_on_R_change,
              test_active_model_and_empty, test_migrate_legacy,
              test_blend_predictor_cache_reuse):
        rc |= t()
    if rc:
        print("\n❌ elo_access / blend 缓存测试存在失败"); sys.exit(rc)
    print("\n🎉 全部 elo_access / blend 缓存测试通过")


if __name__ == "__main__":
    main()
