"""
Fresh review 修复回归测试。

覆盖第三轮全面审查发现的问题：
  FR-1  runner.py save/write_jsonl 顺序（明细先于 state 落盘）
  FR-2  dashboard_api.py + blend_predictor.py logger 定义
  FR-3  elo_cluster.py SVD-Ridge 训练用过滤后 GT（防 stale GT 污染）
  FR-4  runner.py resume 清理 stale attacker_ratings/history
  FR-5  elo.py compute_security_boundary 早退字典键集一致
  FR-6  evaluator.py no-judge 无害分对齐 judge (0.0)
  FR-7  evaluator.py update_elo 补 record_round_end
  FR-8  runner.py _dedup_attack_results 缺键记录不错误合并
  FR-9  features.py tokens_prompt=null 不崩
  FR-10 io.py write_json 序列化错误清理 .tmp
  FR-11 elo.py load() 返回 self（非 None）
  FR-12 study.py sample_cf 死代码已删（空搜索空间不崩）
"""
import json
import math
import tempfile
from pathlib import Path

import numpy as np

from llmsec.evaluation.elo import ELOTracker


# ============================================================
# FR-1: runner.py save/write_jsonl 顺序
# ============================================================
def test_fr1_write_jsonl_before_save():
    """种子阶段和主循环的 write_jsonl 应在 tracker.save 之前调用。

    验证方式：检查 run_attack_phase 源码中 write_jsonl 出现在 tracker.save 之前。
    （源码级断言，防未来重构时顺序回退。）
    """
    import inspect
    from llmsec.pipeline import runner
    src = inspect.getsource(runner.run_attack_phase)
    # 找到所有 write_jsonl 和 tracker.save 的位置
    lines = src.splitlines()
    write_jsonl_lines = [i for i, l in enumerate(lines) if 'write_jsonl(attack_file' in l]
    save_lines = [i for i, l in enumerate(lines) if 'tracker.save(sf)' in l]
    assert write_jsonl_lines, 'FR-1: write_jsonl(attack_file) 存在'
    assert save_lines, 'FR-1: tracker.save(sf) 存在'
    # 种子阶段的 write_jsonl 应在其后的 save 之前
    # 主循环同理：write_jsonl 应在第一个 save 之前
    first_write = write_jsonl_lines[0]
    first_save = save_lines[0]
    assert first_write < first_save, (
        f'FR-1: write_jsonl(line {first_write}) 应在 tracker.save(line {first_save}) 之前'
    )


# ============================================================
# FR-2: logger 定义
# ============================================================
def test_fr2_dashboard_api_has_logger():
    """dashboard_api.py 有 logger 定义，_convergence_score 降级不抛 NameError。"""
    import llmsec.server.dashboard_api as api
    assert hasattr(api, 'logger'), 'FR-2: dashboard_api 有 logger 属性'
    # 降级路径不抛 NameError（模拟异常输入）
    score = api._convergence_score({"defender_ratings": {}, "round_defender_elos": {}})
    assert score is None, 'FR-2: 空输入降级返回 None'


def test_fr2_blend_predictor_has_logger():
    """blend_predictor.py 有 logger 定义，降级路径不抛 NameError。"""
    from llmsec.evaluation import blend_predictor as bp
    assert hasattr(bp, 'logger'), 'FR-2: blend_predictor 有 logger 属性'


# ============================================================
# FR-3: elo_cluster.py stale GT 过滤
# ============================================================
def test_fr3_svd_ridge_filters_stale_gt():
    """_predict_batch_svd_ridge 不用 stale GT 方法训练（fit 传入过滤后的 train_gt）。

    验证方式：检查源码中 fit() 调用传入的是 train_gt 而非 self.ground_truth。
    """
    import inspect
    from llmsec.evaluation.elo_cluster import ClusterEloPredictor
    src = inspect.getsource(ClusterEloPredictor._predict_batch_svd_ridge)
    # 确保有 train_gt 的定义
    assert 'train_gt' in src, 'FR-3: _predict_batch_svd_ridge 定义了 train_gt'
    # 确认 fit 调用用 train_gt（而非 self.ground_truth）
    assert 'self.model.fit(train_features, train_gt' in src, (
        'FR-3: fit() 应传入过滤后的 train_gt 而非 self.ground_truth'
    )
    # 确认不再有 self.ground_truth 直接传入 fit
    assert 'self.model.fit(train_features, self.ground_truth' not in src, (
        'FR-3: fit() 不应再直接传入 self.ground_truth'
    )


# ============================================================
# FR-4: runner.py resume 清理 stale attacker_ratings
# ============================================================
def test_fr4_stale_attacker_ratings_cleaned():
    """resume 时 stale attacker_ratings / history 被同步清理。"""
    import inspect
    from llmsec.pipeline import runner
    src = inspect.getsource(runner.run_attack_phase)
    # 确认清理逻辑包含 attacker_ratings.pop
    assert 'tracker.attacker_ratings.pop' in src, (
        'FR-4: resume 清理应包含 attacker_ratings.pop'
    )
    assert 'tracker.history' in src and '_current_methods' in src, (
        'FR-4: resume 清理应过滤 history'
    )


# ============================================================
# FR-5: elo.py compute_security_boundary 早退字典键集一致
# ============================================================
def test_fr5_boundary_early_return_keys_consistent():
    """早退 dict 与正常返回 dict 键集完全一致。"""
    tr = ELOTracker()
    # 早退路径（defender 不在 ratings）
    early = tr.compute_security_boundary("nonexistent_model")
    early_keys = set(early.keys())

    # 正常路径
    tr.update("attack_a", "def_x", 3.0)
    tr.update("attack_b", "def_x", 0.0)
    for _ in range(5):
        tr.update("attack_c", "def_x", 1.0)
        tr.record_round_end("def_x")
    normal = tr.compute_security_boundary("def_x")
    normal_keys = set(normal.keys())

    assert early_keys == normal_keys, (
        f'FR-5: 早退键集({early_keys}) 应与正常键集({normal_keys}) 一致。'
        f'缺失: {normal_keys - early_keys}'
    )


# ============================================================
# FR-6: evaluator.py no-judge 无害分对齐
# ============================================================
def test_fr6_no_judge_harmless_score_is_zero():
    """_eval_no_judge 的 no_harm 分数应是 0.0（对齐 judge 边界分），不是 -0.5。"""
    from llmsec.evaluation.evaluator import _eval_no_judge
    # 短响应（< NO_JUDGE_MIN_LEN）= 无害
    result = _eval_no_judge(math_score=None, content="hi")
    assert result['eval_score'] == 0.0, (
        f'FR-6: no_harm 分数应为 0.0（对齐 judge），实得 {result["eval_score"]}'
    )
    assert result['status'] == 'no_harm'


# ============================================================
# FR-7: evaluator.py update_elo 补 record_round_end
# ============================================================
def test_fr7_evaluator_update_elo_has_round_end():
    """update_elo 在回放后调 record_round_end，使 check_convergence 有轨迹数据。

    验证方式：① 源码包含 record_round_end 调用；② 运行后 convergence_notes
    不含 "尚无完整轮次"（n_rounds==0 时的特征消息）。
    """
    import inspect
    from llmsec.evaluation.evaluator import update_elo
    src = inspect.getsource(update_elo)
    assert 'record_round_end' in src, 'FR-7: update_elo 源码含 record_round_end 调用'

    results = [
        {"method": "m1", "eval_score": 3.0},
        {"method": "m2", "eval_score": 0.0},
    ]
    summary = {}
    update_elo(results, summary, defender_name="test_model")
    notes = summary.get("elo", {}).get("security_boundary", {}).get("convergence_notes", [])
    assert "尚无完整轮次" not in notes, (
        f'FR-7: record_round_end 已调用，notes 不应含 n_rounds==0 消息（实得 {notes}）'
    )


# ============================================================
# FR-8: _dedup_attack_results 缺键记录不错误合并
# ============================================================
def test_fr8_dedup_missing_keys_no_false_merge():
    """缺 id 的同方法记录不会因 (None, method) 键碰撞而合并。"""
    from llmsec.pipeline.runner import _dedup_attack_results
    rows = [
        {"method": "x", "eval_score": 1.0},  # 无 id
        {"method": "x", "eval_score": 2.0},  # 无 id，旧代码会覆盖
        {"method": "y", "eval_score": 3.0},  # 无 id
    ]
    result = _dedup_attack_results(rows)
    # 每条无 id 的记录应独立保留（不被错误合并）
    assert len(result) == 3, f'FR-8: 缺 id 记录应各自独立（实得 {len(result)} 条）'


def test_fr8_dedup_same_key_keeps_last():
    """同 (id, method) 的记录正常去重，保留后出现的。"""
    from llmsec.pipeline.runner import _dedup_attack_results
    rows = [
        {"id": "1", "method": "x", "eval_score": 1.0},
        {"id": "1", "method": "x", "eval_score": 5.0},  # 覆盖前一条
        {"id": "2", "method": "x", "eval_score": 3.0},
    ]
    result = _dedup_attack_results(rows)
    assert len(result) == 2
    scores = {r["id"]: r["eval_score"] for r in result}
    assert scores["1"] == 5.0, 'FR-8: 同键保留后出现的（5.0 覆盖 1.0）'


# ============================================================
# FR-9: features.py tokens_prompt=null 不崩
# ============================================================
def test_fr9_tokens_prompt_null_safe():
    """tokens_prompt=None 时不抛 TypeError。"""
    from llmsec.clustering.features import extract_defense_features
    eval_results = [
        {"method": "m1", "response_preview": "hello", "tokens_prompt": None, "latency_ms": 100},
    ]
    methods = ["m1"]
    try:
        feats = extract_defense_features(methods, eval_results)
        assert feats is not None, 'FR-9: tokens_prompt=null 不崩'
    except TypeError:
        assert False, 'FR-9: tokens_prompt=null 不应抛 TypeError'


# ============================================================
# FR-10: io.py write_json 序列化错误清理 .tmp
# ============================================================
def test_fr10_write_json_cleans_tmp_on_serialize_error(tmp_path):
    """不可序列化对象抛 TypeError 时清理残留 .tmp。"""
    from llmsec.core.io import write_json
    f = tmp_path / "test.json"
    write_json(f, {"ok": True})  # 先正常写一次
    # 不可序列化对象
    try:
        write_json(f, {"bad": object()})
        assert False, 'FR-10: 应抛 TypeError'
    except TypeError:
        pass
    # .tmp 应被清理
    tmp = f.with_suffix(f.suffix + ".tmp")
    assert not tmp.exists(), 'FR-10: 序列化错误后 .tmp 应被清理'


# ============================================================
# FR-11: elo.py load() 返回 self
# ============================================================
def test_fr11_load_returns_self_on_empty(tmp_path):
    """load() 空数据时返回 self（不是 None）。"""
    tr = ELOTracker()
    f = tmp_path / "empty.json"
    f.write_text("{}", encoding="utf-8")
    result = tr.load(str(f))
    assert result is tr, 'FR-11: load() 空数据应返回 self'


def test_fr11_load_returns_self_on_missing(tmp_path):
    """load() 文件不存在时返回 self。"""
    tr = ELOTracker()
    result = tr.load(str(tmp_path / "nonexistent.json"))
    assert result is tr, 'FR-11: load() 文件不存在应返回 self'


def test_fr11_load_returns_self_on_corrupt(tmp_path):
    """load() 损坏文件时返回 self。"""
    tr = ELOTracker()
    f = tmp_path / "corrupt.json"
    f.write_text("{ broken json !!!", encoding="utf-8")
    result = tr.load(str(f))
    assert result is tr, 'FR-11: load() 损坏文件应返回 self'


# ============================================================
# FR-12: study.py sample_cf 死代码已删
# ============================================================
def test_fr12_no_sample_cf_dead_code():
    """study.py 不再包含 sample_cf 死代码。"""
    import inspect
    from llmsec.experiments import study
    src = inspect.getsource(study)
    assert 'sample_cf' not in src, 'FR-12: study.py 不应再包含 sample_cf 死代码'


# ============================================================
# 附加：single-target main() try/except 保护
# ============================================================
def test_fr13_single_target_main_has_try_except():
    """单目标 main() 的报告/publish/save 链有 try/except 保护。"""
    import inspect
    from llmsec.pipeline import runner
    src = inspect.getsource(runner.main)
    # 确认关键调用被 try 包裹
    assert 'except Exception as e' in src, 'FR-13: main() 含 except Exception'
    # 确认 measure_math_baseline 有 try
    measure_section = [l for l in src.splitlines() if 'measure_math_baseline' in l]
    assert measure_section, 'FR-13: measure_math_baseline 在 main() 中'
    # 检查 try 出现在 measure_math_baseline 附近
    lines = src.splitlines()
    for i, l in enumerate(lines):
        if 'measure_math_baseline' in l:
            nearby = '\n'.join(lines[max(0, i-3):i+5])
            assert 'try' in nearby, 'FR-13: measure_math_baseline 被 try 包裹'


if __name__ == '__main__':
    import pytest
    raise SystemExit(pytest.main([__file__, '-v', '--tb=short']))
