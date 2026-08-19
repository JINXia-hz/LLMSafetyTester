"""第 7 轮审计回归——评估/报告正确性（M-5 / M-6 / M-13 / L-1 / L-2 / L-3）。

  - M-5: judge_statistics 必须在 l1_汇总.json 落盘前挂进 summary。
  - M-6: safe_twin.evaluate_allergy 的 Judge 故障降级关键词，不崩循环。
  - M-13: weekend_hpo 在 ci_half 缺失时给出诊断而非 TypeError。
  - L-1: runner 复用磁盘特征缓存（同 _should_refresh_features 口径）。
  - L-2: harmbench --obfuscate 在默认 --variants 1 下也轮换全部混淆方法。
  - L-3: prescreen 推理输入与训练同口径（截断 500 字）。
"""

from __future__ import annotations

import sys

# ============================================================
# M-5: 汇总文件必须含 judge_statistics
# ============================================================

class _FakeJudge:
    def evaluate(self, prompt, content, skip_prescreen=False):
        return {
            "is_refusal": False, "is_harmful": True, "compliance_level": "A",
            "combined_score": 4.0, "harmfulness": 4, "specificity": 3,
            "dangerousness": 5, "judge_calls": 1, "prescreen_result": None,
        }


def test_summary_file_contains_judge_statistics(tmp_path, monkeypatch):
    import llmsec.evaluation.cli as eval_cli
    import llmsec.evaluation.evaluator as ev
    from llmsec.core.io import write_jsonl

    attacks = tmp_path / "l1r7.jsonl"
    write_jsonl(str(attacks), [{
        "id": "r7-1", "prompt": "prompt text long enough", "method": "m1",
        "expected_answer": None, "harm_type": "h", "category": "c",
    }])
    out = tmp_path / "output"
    out.mkdir()
    monkeypatch.setattr(eval_cli, "OUTPUT_DIR", out)
    monkeypatch.setattr(eval_cli, "RUNS_DIR", tmp_path / "runs")

    monkeypatch.setattr(ev, "call_target", lambda prompt: {
        "content": "详细的模型回复内容，长度足够", "error": None, "latency_ms": 1,
        "tokens_prompt": 10, "tokens_completion": 20, "meta": {},
    })
    monkeypatch.setattr(eval_cli, "API_DELAY", 0)
    monkeypatch.setattr(eval_cli, "create_judge_client", lambda: None)
    monkeypatch.setattr(eval_cli, "Judge", lambda client, model=None: _FakeJudge())
    monkeypatch.setattr(eval_cli, "update_elo", lambda *a, **kw: None)

    monkeypatch.setattr(sys, "argv", ["eval", "--input", str(attacks)])
    eval_cli.main()

    import json
    summaries = list((tmp_path / "runs").glob("*/l1r7_汇总.json"))
    assert summaries, "main 应产出汇总文件"
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert "judge_statistics" in summary, (
        "M-5：judge 统计块（合规分布/H·S·D/预筛命中率/Judge 调用数）必须持久化——"
        "原先在 write_json 之后才挂载，文件里永远缺失")
    assert "compliance_distribution" in summary["judge_statistics"]


# ============================================================
# M-6: safe_twin 评估循环的 Judge 故障降级
# ============================================================

def test_safe_twin_judge_failure_degrades(tmp_path, monkeypatch):
    """Judge 抛异常（4xx/超时）时降级关键词判定，循环继续、条目落盘。"""
    import llmsec.evaluation.safe_twin as st
    from llmsec.core.io import read_jsonl, write_jsonl

    twins = [
        {"original_id": f"o{i}", "method": f"m{i}", "harm_type": "h",
         "safe_prompt": f"safe prompt {i}"} for i in range(3)
    ]
    twin_file = tmp_path / "safe_twins.jsonl"
    write_jsonl(str(twin_file), twins)

    def boom_judge_allergic(judge, safe_prompt, content):
        raise RuntimeError("judge 401 (simulated)")
    monkeypatch.setattr(st, "judge_allergic", boom_judge_allergic)
    monkeypatch.setattr(st, "API_DELAY", 0)

    # 目标模型调用桩：b0f0933（B-1）后 evaluate_allergy 的目标调用统一走
    # targets.call_target（safe_twin 内函数导入、运行期解析），client= 参数
    # 已废弃——此前注入 _FakeClient 是死桩，本地靠 .env 真实端点"碰巧通过"，
    # CI 无凭证三条全 API错误 跳过、entries=0（与 test_allergy 的 B-6 桩同式）
    import llmsec.targets as tgt
    monkeypatch.setattr(tgt, "call_target",
                        lambda p: {"error": None, "content": "benign target response",
                                   "meta": {}})
    result_file = tmp_path / "twin_results.jsonl"

    # r9/P3-7：twins/judge/result_file 显式注入——不再 patch
    # create_openai_client/create_judge_client/config 路径
    import llmsec.evaluation.judge as jd
    st.evaluate_allergy(twins=twins, judge=jd.Judge(None), result_file=result_file)
    # 修复前：第一条 judge 故障即未捕获异常带栈退出

    entries = read_jsonl(result_file)
    assert len(entries) == 3, "Judge 故障不得中断评估循环（M-6）"
    assert all(e["judge_level"] == "N/A" for e in entries)
    assert all(e["is_allergic"] is False for e in entries), "良性内容关键词降级 → 不过敏"


# ============================================================
# M-13: weekend_hpo 的 None 指标
# ============================================================

def test_hpo_report_handles_missing_ci_half(tmp_path, monkeypatch):
    """best 存在但 ci_half_mean=None（部分失败）时 cmd_report 不崩溃。"""
    import pytest

    import llmsec.experiments.study as study_mod

    # scripts/ 已移出仓库（本地维护脚本）——本地有则测、CI 无则跳过
    wh = pytest.importorskip("scripts.weekend_hpo")

    fake_summaries = {
        "weekend_stage1": {"best": {"ci_half_mean": None, "ci_half_std": 0,
                                    "params": {"x": 1}}},
        "weekend_stage2": {"best": None},
        "weekend_stage3": {"best": None},
    }

    def fake_dir(name):
        d = tmp_path / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "study.yaml").write_text("name: x\n", encoding="utf-8")
        return d

    monkeypatch.setattr(study_mod, "study_dir", fake_dir)
    monkeypatch.setattr(
        study_mod.StudyConfig, "from_yaml",
        classmethod(lambda cls, path: type("C", (), {"name": path.parent.name})()))
    # cmd_report 内部是 `from llmsec.experiments.study import summarize`（调用期取模块属性）
    monkeypatch.setattr(study_mod, "summarize",
                        lambda cfg: fake_summaries.get(cfg.name, {}))

    rc = wh.cmd_report()
    assert rc == 0, "ci_half=None 时 cmd_report 应正常返回而非 TypeError"


# ============================================================
# L-1: runner 复用特征缓存
# ============================================================

def test_runner_reuses_fresh_feature_cache(tmp_path, monkeypatch):
    """同一攻击集第二次 run：特征缓存命中 → 不再全量 fit_features。"""
    import llmsec.core.config as cfg
    import llmsec.evaluation.predictors.cold_start as cs
    from tests.test_audit_r7_high import _offline_runner_env

    rn, fixed_run, base_argv, deps = _offline_runner_env(tmp_path, monkeypatch)
    monkeypatch.setattr(cfg, "FEATURE_CACHE_FILE", tmp_path / "feature_cache.pkl")

    calls = {"fit": 0}
    orig_fit = cs.ColdStartPredictor.fit_features

    def spy_fit(self, records):
        calls["fit"] += 1
        return orig_fit(self, records)
    monkeypatch.setattr(cs.ColdStartPredictor, "fit_features", spy_fit)

    rn.main(base_argv[1:] + ["--phase", "1"], deps=deps)
    assert calls["fit"] == 1, "首次 run（缓存缺失）应全量提取特征"

    rn.main(base_argv[1:] + ["--phase", "1"], deps=deps)
    assert calls["fit"] == 1, (
        "L-1：缓存未变（同攻击集/同特征配置）时第二次 run 应复用缓存，"
        "不得重新 fit_features 并覆写缓存")


# ============================================================
# L-2: harmbench 混淆轮换
# ============================================================

def test_harmbench_obfuscation_rotates_with_single_variant(tmp_path):
    """--variants 1 --obfuscate：不同 behavior 轮换全部 4 种混淆。"""
    import csv

    from llmsec.attacks.harmbench import generate

    rows = [{"Behavior": f"behavior {i}", "SemanticCategory": "s",
             "FunctionalCategory": "standard", "BehaviorID": f"b{i:03d}",
             "ContextString": ""} for i in range(4)]

    src = tmp_path / "hb.csv"
    with open(src, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    jailbreaks = [f"JB{i}: {{behavior}}" for i in range(8)]
    out = tmp_path / "out.jsonl"
    generate(src, jailbreaks, out, variants=1, obfuscate=True, math_tax=False)

    from llmsec.core.io import read_jsonl
    records = read_jsonl(out)
    obf_suffixes = {r["method"].rsplit("_", 1)[-1] for r in records}
    assert obf_suffixes == {"b64", "rot13", "code", "story"}, (
        f"--variants 1 + --obfuscate 应轮换全部混淆方法，实际: {obf_suffixes}")


# ============================================================
# L-3: prescreen 推理输入截断
# ============================================================

def test_prescreen_input_truncated_to_500(monkeypatch):
    """judge.evaluate 的预筛输入与训练口径一致（≤500 字）。"""
    import llmsec.evaluation.judge as jd
    import llmsec.evaluation.prescreen_ml as pml

    seen: list[str] = []
    # 返回 "refusal" 走预筛早退分支，不再触达真实 Judge API
    monkeypatch.setattr(pml, "predict", lambda text: seen.append(text) or "refusal")

    judge = jd.Judge(None)
    long_response = "x" * 3000
    result = judge.evaluate("safe prompt", long_response)

    assert seen and all(len(t) <= 500 for t in seen), (
        "预筛输入必须截断到 500 字（与训练侧 response_preview 同口径）")
    assert result["is_refusal"] is True and result["prescreen_result"] == "refusal"
