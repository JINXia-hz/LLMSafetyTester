"""回归测试：R-cutover 的 Elo 派生访问层（elo_access）+ BlendPredictor 缓存。

验证：
  1. publish_tracker：live tracker → R 写入 + 缓存发布（含收敛轨迹）。
  2. elo_state_for：冷派生（缓存未命中）→ 写缓存 → 二次命中。
  3. 缓存失效：R 变动后指纹不一致 → 自动重派生。
  4. active_model：取最新活跃模型。
  5. load_or_fit_blend_predictor：同 (R+清单) 复用缓存，变化则重训。
"""

import llmsec.core.config as cfg
import llmsec.core.results as results_mod
from llmsec.core.results import ResultsMatrix
from llmsec.evaluation.blend_predictor import load_or_fit_blend_predictor
from llmsec.evaluation.elo import ELOTracker
from llmsec.evaluation import elo_access as ea


def _setup(tmp_path, monkeypatch):
    monkeypatch.setattr(results_mod, "RESULTS_FILE", tmp_path / "results.json")
    monkeypatch.setattr(cfg, "ELO_CACHE_FILE", tmp_path / "elo_cache.json")


def test_publish_and_derive(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    tracker = ELOTracker()
    tracker.update("DAN", "qwen9b", 3.5)
    tracker.update("rot13", "qwen9b", -1.0)
    tracker.record_round_end("qwen9b")
    ea.publish_tracker(tracker, "qwen9b")

    R = ResultsMatrix.load()
    assert R.n_for_model("qwen9b") == 2
    st = ea.elo_state_for("qwen9b")
    assert "DAN" in st["attacker_ratings"]
    assert st["fingerprint"] is not None
    assert st.get("round_defender_elos", {}).get("qwen9b")


def test_cache_invalidates_on_R_change(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    t = ELOTracker()
    t.update("DAN", "qwen9b", 3.5)
    ea.publish_tracker(t, "qwen9b")
    fp1 = ea.elo_state_for("qwen9b")["fingerprint"]

    t2 = ELOTracker()
    t2.update("rot13", "qwen9b", 2.0)
    ea.publish_tracker(t2, "qwen9b")
    st2 = ea.elo_state_for("qwen9b")
    assert st2["fingerprint"] != fp1
    assert "rot13" in st2["attacker_ratings"]


def test_active_model_and_empty(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    assert ea.active_model() is None
    t = ELOTracker()
    t.update("DAN", "modelA", 1.0)
    ea.publish_tracker(t, "modelA")
    assert ea.active_model() == "modelA"
    assert ea.attacker_ratings_for("modelA").get("DAN") is not None
    assert ea.elo_state_for("nope") == {}


def test_blend_predictor_cache_reuse(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    import llmsec.evaluation.blend_predictor as bp_mod
    monkeypatch.setattr(bp_mod, "PREDICTORS_DIR", tmp_path / "predictors")
    R = ResultsMatrix()
    R.upsert("DAN", "qwen9b", 3.0, ts=1)
    R.upsert("rot13", "qwen9b", -1.0, ts=2)
    R.upsert("b64", "qwen9b", 2.0, ts=3)
    features = {"DAN": {"t": [1.0]}, "rot13": {"t": [0.0]}, "b64": {"t": [0.5]}}
    catalog = ["DAN", "rot13", "b64"]

    load_or_fit_blend_predictor(R, features, method_catalog=catalog)
    load_or_fit_blend_predictor(R, features, method_catalog=catalog)  # 二次命中缓存
    from pathlib import Path
    assert any((tmp_path / "predictors").glob("blend_*.pkl"))

    R.upsert("extra", "qwen9b", 1.0, ts=4)  # R 变动 → 新缓存
    load_or_fit_blend_predictor(R, features, method_catalog=catalog)
    files = list((tmp_path / "predictors").glob("blend_*.pkl"))
    assert len(files) >= 2
