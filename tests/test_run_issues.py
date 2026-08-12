"""
本轮运行问题修复的回归测试（D 组收尾）。

覆盖：
  P1  跨 run 全量 resume：从 R 回放重建 tracker（防御 Elo 非默认 1500、
      predictor.ground_truth 含 R 已测方法），summary this_run_tested==0，
      ASR 为 R 合成的累计口径（非 0/0）
  P2  部分 resume：predict_batch 视角 GT 数 = R 恢复数 + 本轮新测，
      已测方法不被当未测重复预测/重测
  P3  连续两轮 _inject_predicted_elos（untested 收缩）blend 缓存键一致，
      第二轮命中缓存不再 fit
  P4  skip_final_clustering=True 且无记录 → 无"记录不足"warning，有 info
  P5  attack_summary.this_run_tested==0 → generate_reports 不调
      generate_narrative，直接用 generate_fallback_report
  P6  纯噪声特征 → λ* 顶格 + no_signal=True + predict variance ≥ y_std²；
      有信号数据 no_signal=False
  P8  得分在最大候选仍上升 → sweep_candidates 自动外扩（k* 不被 2*k0 截断）；
      max_expansions=0 保持旧截断行为
  P9  多线程并发 save_artifact 同 key → 文件可加载且无残留 tmp；
      多线程 save_probe → 条目不丢

全部 mock/合成数据，禁止真实 API/网络；文件一律落 tmp_path，不碰真实 output/。
"""
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import llmsec.pipeline.attack_phase as ap
from llmsec.core.io import load_artifact, save_artifact
from llmsec.core.results import ResultsMatrix
from llmsec.evaluation.elo import ELOTracker


# ============================================================
# 公共构造
# ============================================================
def _records(methods):
    """最小攻击集记录（与 test_pipeline_review.py 同构）。"""
    return [{
        "id": m, "method": m, "prompt": f"attack prompt {m}",
        "expected_answer": 0, "category": "test", "harm_type": "test",
    } for m in methods]


def _eval_ok(prompt, ea, judge, use_judge=True):
    """evaluate_single 替身：全部越狱成功（eval_score=3.0）。"""
    return {
        "content": "ok", "is_refusal": False, "is_harmful": True,
        "eval_score": 3.0, "jailbreak_tax": 0.5, "status": "fully_compliant",
        "compliance_level": "A", "latency_ms": 100,
    }


def _isolate_feature_cache(tmp_path, monkeypatch):
    """特征缓存隔离到 tmp——fit_features 会原子写 FEATURE_CACHE_FILE，
    不隔离会抢写全局 output/feature_cache.pkl。"""
    import llmsec.core.config as cfg
    monkeypatch.setattr(cfg, "FEATURE_CACHE_FILE", tmp_path / "feature_cache.pkl")


def _preseed_features(tracker, methods):
    """预置合成特征缓存，使 _should_refresh_features 判复用——
    避免 fit_features 走真实特征提取（可能触达 embedding 服务）。"""
    from llmsec.evaluation.predictors.cold_start import (
        _compute_method_set_hash,
        current_feature_config_hash,
    )
    tracker.predictor.artifacts = {
        "features": {
            m: {"textual": [1.0, 0.0], "embedding": [0.1 * i, 0.2]}
            for i, m in enumerate(methods)
        },
        "method_set_hash": _compute_method_set_hash(sorted(methods)),
        "meta": {"feature_config_hash": current_feature_config_hash()},
    }


# ============================================================
# P1：全量 resume —— R 回放重建 + ASR 合成 + this_run_tested==0
# （簇粒度：R 行键 = 记录 id，extra.unit = 簇指纹；预聚类打叉为 None → 每方法一簇，确定性）
# ============================================================
def _one_method_units(monkeypatch, methods):
    """强制每方法自成一个 unit（_quick_precluster 返回 None 走退化路径），
    返回 {method: unit_id}——测试侧据此构造与 run 内完全一致的 unit 指纹。"""
    from llmsec.core.units import unit_fingerprint

    monkeypatch.setattr(ap, "_quick_precluster", lambda *a, **k: None)
    return {m: unit_fingerprint([m]) for m in methods}


def _r_with_unit_results(defender, scores: dict, uid_of):
    """构造簇粒度 R：行键 = 记录 id（=方法名），extra.unit = 对应 unit 指纹。"""
    R = ResultsMatrix()
    for i, (m, s) in enumerate(scores.items()):
        R.upsert(m, defender, s,
                 status="fully_compliant" if s > 0 else "refused",
                 ts=i + 1, extra={"round": 1, "unit": uid_of[m]})
    return R


def test_p1_full_resume_replays_elo_from_r(tmp_path, monkeypatch):
    _isolate_feature_cache(tmp_path, monkeypatch)
    monkeypatch.setattr(ap, "analyze_clusters", lambda tracker: {})
    eval_calls = []
    monkeypatch.setattr(ap, "evaluate_single",
                        lambda *a, **k: eval_calls.append(1) or _eval_ok(*a, **k))

    methods = ["m0", "m1", "m2", "m3"]
    uid_of = _one_method_units(monkeypatch, methods)
    # R：m0/m1/m3 成功（score>0），m2 被拒 —— 合成 ASR 应为 3/4
    R = _r_with_unit_results("def-x", {"m0": 3.0, "m1": 3.0, "m2": -2.0, "m3": 3.0}, uid_of)

    tracker = ELOTracker()
    _preseed_features(tracker, methods)
    summary = ap.run_attack_phase(
        _records(methods), judge=None, tracker=tracker,
        batch_size=4, max_rounds=2, attack_file=tmp_path / "attack.jsonl",
        sampler="gap", coordinate_rounds=1,
        state_file=str(tmp_path / "state.json"),
        defender_name="def-x", r_snapshot=R,
    )

    assert eval_calls == [], "P1: 全量 resume 不应发起任何新评估"
    # R 回放重建：防御 Elo 不再停在默认 INITIAL_ELO=1500
    assert tracker.get_defender_elo("def-x") != pytest.approx(1500.0)
    # predictor.ground_truth 含 R 已测单位（4 个单方法簇）
    assert set(tracker.predictor.ground_truth) == set(uid_of.values())
    assert tracker.ground_truth_methods == set(uid_of.values())
    # 本轮无新测；ASR 为 R 合成的累计口径（非 0/0）
    assert summary["this_run_tested"] == 0
    assert summary["successful"] == 3
    assert summary["asr"] == pytest.approx(0.75)


# ============================================================
# P2：部分 resume —— GT 数 = R 恢复 + 本轮新测；已测单位不重测不重复预测
# ============================================================
def test_p2_partial_resume_gt_count_and_no_retest(tmp_path, monkeypatch):
    _isolate_feature_cache(tmp_path, monkeypatch)
    monkeypatch.setattr(ap, "analyze_clusters", lambda tracker: {})
    evaluated: list[str] = []

    def _eval_spy(prompt, ea, judge, use_judge=True):
        evaluated.append(prompt)
        return _eval_ok(prompt, ea, judge, use_judge=use_judge)

    monkeypatch.setattr(ap, "evaluate_single", _eval_spy)

    resumed = ["m0", "m1", "m2", "m3"]
    fresh = ["m4", "m5", "m6", "m7"]
    uid_of = _one_method_units(monkeypatch, resumed + fresh)
    R = _r_with_unit_results("def-y", {m: 3.0 for m in resumed}, uid_of)

    tracker = ELOTracker()
    _preseed_features(tracker, resumed + fresh)
    summary = ap.run_attack_phase(
        _records(resumed + fresh), judge=None, tracker=tracker,
        batch_size=8, max_rounds=1, attack_file=tmp_path / "attack.jsonl",
        sampler="gap", coordinate_rounds=1,
        state_file=str(tmp_path / "state.json"),
        defender_name="def-y", r_snapshot=R,
    )

    # R 已测的 4 个单位不被重测，本轮只新测剩余 4 个（各簇唯一记录即该方法 prompt）
    assert {p.replace("attack prompt ", "") for p in evaluated} == set(fresh)
    assert summary["this_run_tested"] == 4
    # predict_batch 视角：GT 数 = R 恢复数(4) + 本轮新测(4)
    assert tracker.predictor.ground_truth_count() == 8
    # 已测单位（R 恢复 + 新测）不被当未测预测
    remaining = tracker.predictor.predict_batch(
        {uid: {"method": uid} for uid in uid_of.values()})
    assert remaining == {}, "P2: 全部已测后 predict_batch 不应再产出预测"


# ============================================================
# P3：连续两轮 _inject_predicted_elos —— blend 缓存键跨轮稳定
# ============================================================
def _blend_ready_tracker_and_r():
    """tracker 带特征缓存 + 双模型 R（≥2 模型才走 BlendPredictor 路径）。"""
    features = {
        f"m{i}": {"textual": [float(i), 1.0], "embedding": [0.1 * i, 0.2]}
        for i in range(6)
    }
    tracker = ELOTracker()
    tracker.predictor.artifacts = {"features": features}
    R = ResultsMatrix()
    for mdl in ("modelA", "modelB"):
        for i in range(3):
            R.upsert(f"m{i}", mdl, 2.0 + i, ts=i + 1, extra={"round": 1})
    return tracker, R, features


def test_p3_inject_uses_stable_full_catalog(tmp_path, monkeypatch):
    """第二轮 untested 收缩时，传给 blend 的 method_catalog 仍是完整清单。"""
    import llmsec.evaluation.predictors.blend as blend_mod

    tracker, R, _ = _blend_ready_tracker_and_r()
    catalogs: list[list[str]] = []

    class _StubBP:
        def predict(self, method, model):
            return {"elo": 1500.0, "std": 10.0, "source": "blend"}

    def _spy(results, feats, method_catalog=None):
        catalogs.append(list(method_catalog))
        return _StubBP()

    monkeypatch.setattr(blend_mod, "load_or_fit_blend_predictor", _spy)

    full = {f"m{i}": {"method": f"m{i}"} for i in range(6)}
    ap._inject_predicted_elos(tracker, full, "def-z",
                              r_snapshot=R, full_method_records=full)
    # 第二轮：m0/m1 已测，untested 收缩为 4 个
    tracker.ground_truth_methods.update({"m0", "m1"})
    remaining = {m: r for m, r in full.items() if m not in tracker.ground_truth_methods}
    ap._inject_predicted_elos(tracker, remaining, "def-z",
                              r_snapshot=R, full_method_records=full)

    assert len(catalogs) == 2, "P3: 两轮都应走 BlendPredictor 路径"
    assert catalogs[0] == catalogs[1] == list(full.keys()), \
        "P3: catalog 应恒为完整清单（缓存键跨轮稳定）"


def test_p3_blend_cache_hit_skips_refit(tmp_path, monkeypatch):
    """真实缓存路径：同 R/features/catalog 第二次调用命中 pkl，不再 fit。"""
    import llmsec.evaluation.predictors.blend as blend_mod

    monkeypatch.setattr(blend_mod, "PREDICTORS_DIR", tmp_path)

    _, R, features = _blend_ready_tracker_and_r()
    catalog = sorted(features.keys())
    fit_calls = []
    orig_fit = blend_mod.BlendPredictor.fit

    def _counting_fit(self, *a, **k):
        fit_calls.append(1)
        return orig_fit(self, *a, **k)

    monkeypatch.setattr(blend_mod.BlendPredictor, "fit", _counting_fit)

    bp1 = blend_mod.load_or_fit_blend_predictor(R, features, method_catalog=catalog)
    bp2 = blend_mod.load_or_fit_blend_predictor(R, features, method_catalog=catalog)

    assert len(fit_calls) == 1, "P3: 第二轮应命中缓存，不重复 fit"
    assert isinstance(bp2, blend_mod.BlendPredictor)
    assert bp2._catalog == bp1._catalog == catalog
    # 缓存键带模型 schema 版本盐（v2：EloPredictorModel.no_signal 引入后旧 pkl 失效）
    key = blend_mod.BlendPredictor.cache_key(R, catalog, features)
    assert key.startswith("blend_v2_")


# ============================================================
# P4：skip_final_clustering=True 且无记录 → info 而非"记录不足"warning
# ============================================================
def test_p4_skip_final_clustering_logs_info_not_warning(tmp_path, monkeypatch, caplog):
    _isolate_feature_cache(tmp_path, monkeypatch)
    monkeypatch.setattr(ap, "analyze_clusters", lambda tracker: {})

    # llmsec logger propagate=False（core/logging.py），caplog.handler 需直接挂上
    target_logger = logging.getLogger("llmsec.pipeline.attack_phase")
    caplog.handler.setLevel(logging.INFO)
    target_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.INFO, logger="llmsec.pipeline.attack_phase"):
            ap.run_attack_phase(
                [], judge=None, tracker=ELOTracker(),
                batch_size=2, max_rounds=1,
                attack_file=tmp_path / "attack.jsonl",
                sampler="gap", coordinate_rounds=1,
                skip_final_clustering=True,
                state_file=str(tmp_path / "state.json"),
                defender_name="def-w", r_snapshot=ResultsMatrix(),
            )
    finally:
        target_logger.removeHandler(caplog.handler)

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any("跳过最终聚类输出" in r.getMessage() for r in warnings), \
        "P4: skip_final_clustering 主动跳过不应再打'记录不足'warning"
    assert any("跳过最终聚类落盘" in r.getMessage()
               for r in caplog.records if r.levelno == logging.INFO), \
        "P4: 应有 info 日志说明主动跳过最终聚类"


# ============================================================
# P5：this_run_tested==0 → 跳过 LLM 叙事，直接 fallback 报告
# ============================================================
def _run_generate_reports(tmp_path, monkeypatch, attack_summary):
    import llmsec.reporting.final_report as fr
    import llmsec.reporting.report as rep

    calls = {"narrative": 0, "fallback": 0}
    monkeypatch.setattr(rep, "build_method_stats", lambda *a, **k: {})
    monkeypatch.setattr(rep, "build_tree", lambda *a, **k: {"tree": "stub"})
    monkeypatch.setattr(
        rep, "generate_narrative",
        lambda *a, **k: calls.__setitem__("narrative", calls["narrative"] + 1) or "llm-md")
    monkeypatch.setattr(
        rep, "generate_fallback_report",
        lambda *a, **k: calls.__setitem__("fallback", calls["fallback"] + 1) or "fallback-md")

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fr.generate_reports(
        run_dir, ELOTracker(), "def-v", attack_summary,
        allergy_summary={}, total_methods=10,
    )
    return calls, (run_dir / "security_report.md").read_text(encoding="utf-8")


def test_p5_zero_new_tests_skips_narrative(tmp_path, monkeypatch):
    summary = {"this_run_tested": 0, "total_tested": 0, "asr": 0.0,
               "total_attacks": 0, "successful": 0, "rounds": 0}
    calls, md = _run_generate_reports(tmp_path, monkeypatch, summary)
    assert calls["narrative"] == 0, "P5: 本轮无新测试不应调 generate_narrative"
    assert calls["fallback"] == 1
    assert md == "fallback-md"


def test_p5_with_new_tests_calls_narrative(tmp_path, monkeypatch):
    """对照：本轮有新测时仍走 LLM 叙事。"""
    summary = {"this_run_tested": 3, "total_tested": 3, "asr": 0.5,
               "total_attacks": 3, "successful": 1, "rounds": 1}
    calls, md = _run_generate_reports(tmp_path, monkeypatch, summary)
    assert calls["narrative"] == 1
    assert calls["fallback"] == 0
    assert md == "llm-md"


# ============================================================
# P6：no_signal —— 纯噪声 λ* 顶格、variance ≥ y_std²；有信号对照
# ============================================================
def _toy_features(n, dim, rng):
    return {
        f"m{i}": {
            "textual": rng.normal(size=4).tolist(),
            "embedding": rng.normal(size=dim).tolist(),
        } for i in range(n)
    }


def test_p6_noise_features_flag_no_signal():
    from llmsec.evaluation.predictors.svd_ridge import EloPredictorModel

    rng = np.random.default_rng(0)
    n = 30
    features = _toy_features(n, 16, rng)
    # y 与特征无关（纯噪声标签）
    gt = {m: {"elo": float(v)} for m, v in
          zip(features, rng.normal(1500, 100, n))}

    model = EloPredictorModel().fit(features, gt)
    assert model.lambda_opt == pytest.approx(float(np.max(model.lambda_candidates)))
    assert model.no_signal is True

    methods = sorted(features)[:5]
    _, variances = model.predict(features, methods)
    assert np.all(variances >= model.y_std ** 2), \
        "P6: no_signal 时 predict 方差下限应提到 GT 边际方差 y_std²"


def test_p6_signal_features_not_flagged():
    from llmsec.evaluation.predictors.svd_ridge import EloPredictorModel

    rng = np.random.default_rng(1)
    n = 40
    features = _toy_features(n, 8, rng)
    # y 由 embedding 前两维线性决定 + 小噪声 → 有真实信号
    gt = {}
    for m, f in features.items():
        e = f["embedding"]
        gt[m] = {"elo": 1500 + 80 * e[0] - 60 * e[1] + float(rng.normal(0, 5))}

    model = EloPredictorModel().fit(features, gt)
    assert model.no_signal is False
    assert model.lambda_opt < float(np.max(model.lambda_candidates))


# ============================================================
# P8：sweep_candidates 边界上升自动外扩；max_expansions=0 恢复旧行为
# ============================================================
def _p8_sweep(monkeypatch, n, max_expansions):
    """合成"得分随 k 严格单调上升"场景，跑 sweep_candidates。

    指标只让 silhouette 携带信息（=k），CH/DB 恒定 → 归一化后 score 随 k
    线性上升；geomspace 候选间距递增 → 末尾 delta 恒为最大 → 触发边界外扩。
    """
    import llmsec.clustering.tree as tree_mod

    monkeypatch.setattr(
        tree_mod, "cut_tree",
        lambda Z, methods, k: {m: i % k for i, m in enumerate(methods)})
    monkeypatch.setattr(
        tree_mod, "_evaluate_cut",
        lambda coords, labels, methods: {
            # 用当前簇数当 silhouette——labels 里簇数即 k
            "silhouette": float(len(set(labels.values()))),
            "calinski_harabasz": 0.0,
            "davies_bouldin": 1e6,
        })

    methods = [f"m{i}" for i in range(n)]
    Z = np.zeros((n - 1, 4))
    coords = np.zeros((n, 2))
    return tree_mod.sweep_candidates(coords, Z, methods,
                                     max_expansions=max_expansions)


def test_p8_boundary_rising_expands_candidates(monkeypatch):
    from llmsec.clustering.tree import log_growth_k0

    n = 100
    entries = _p8_sweep(monkeypatch, n, max_expansions=2)
    ks = [e["k"] for e in entries]
    # 返回严格按 k 升序去重
    assert ks == sorted(set(ks))
    # n=100 → k0=7，旧候选 hi=2*k0=14；外扩后最大 k 应突破 14（不被截断）
    hi_old = 2 * log_growth_k0(n)
    assert ks[-1] > hi_old, "P8: 边界仍上升时应自动外扩候选"


def test_p8_max_expansions_zero_keeps_old_behavior(monkeypatch):
    from llmsec.clustering.tree import candidate_ks

    n = 100
    entries = _p8_sweep(monkeypatch, n, max_expansions=0)
    ks = [e["k"] for e in entries]
    # max_expansions=0：不外扩，候选与 candidate_ks 一致（旧截断行为）
    assert ks == candidate_ks(n)


# ============================================================
# P9：并发写安全 —— save_artifact 无残留 tmp；save_probe 条目不丢
# ============================================================
def test_p9_concurrent_save_artifact_no_tmp_leftover(tmp_path):
    path = tmp_path / "art.pkl"

    def _work(i):
        for _ in range(20):
            save_artifact(path, {"writer": i, "data": list(range(50))})

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(_work, range(8)))

    obj = load_artifact(path)
    assert obj is not None and 0 <= obj["writer"] < 8 and len(obj["data"]) == 50
    assert not list(tmp_path.glob("art.pkl.tmp.*")), \
        "P9: tmp 名带 pid/tid 后缀，os.replace 成功后不应有残留"


def test_p9_concurrent_save_probe_no_lost_entries(tmp_path):
    import llmsec.evaluation.predictors.fingerprint as fp_mod

    probes_file = tmp_path / "probes.json"  # 隔离：不碰真实 output/state/probes.json
    n_models = 16

    def _work(i):
        fp_mod.save_probe(
            f"model{i}",
            {f"s{j}": 1500.0 + i + j for j in range(4)},
            [f"s{j}" for j in range(4)],
            path=probes_file,
        )

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(_work, range(n_models)))

    models = fp_mod.load_probes(probes_file)
    assert set(models) == {f"model{i}" for i in range(n_models)}, \
        "P9: 持锁 read-modify-write 后并发 save_probe 不应丢条目"
