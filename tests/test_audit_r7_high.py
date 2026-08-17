"""第 7 轮审计回归——高危修复。

  - H-1: `--phase 2` 独立运行（work-dir/固定 runs 目录恢复 state.json）时，
    units 装配不再依赖 do_phase1，过敏候选键空间与 state.json 的 unit_id 对齐，
    FPR 不再恒为"未测"。
  - H-2: approve_plan 的总线通知必须广播——门下省（dept=MENXIA）的准奏阶段
    风险评估订阅在生产定向下（to_dept=SHANGSHU）永不派发。
"""

from __future__ import annotations

from types import SimpleNamespace as NS

# ============================================================
# H-1: phase 2 独立运行的过敏检测键空间
# ============================================================

def _offline_runner_env(tmp_path, monkeypatch):
    """把 runner main() 全程搬到 tmp + 离线桩（无网络、无全局 output 写入）。"""
    import llmsec.evaluation.judge as jd
    import llmsec.pipeline.allergy_phase as alp
    import llmsec.pipeline.attack_phase as ap
    import llmsec.pipeline.runner as rn
    import llmsec.targets as tgt
    from llmsec.core.io import write_jsonl

    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    monkeypatch.setattr(rn, "RUNS_DIR", runs_root)
    # 固定 run 目录名：模拟 work-dir 模式的"phase 1 与 phase 2 复用同一目录"
    # （全局时间戳目录每次分配新目录，--phase 2 找不到 state.json 属另一报错路径）
    fixed_run = runs_root / "fixed"
    monkeypatch.setattr(rn, "_allocate_runs_dir", lambda base, name: fixed_run)

    # R 矩阵指向 tmp（缺省文件 → 空矩阵），不读/写仓库全局 output
    # r9/P3-4：results.py 已改为调用期读 _config.RESULTS_FILE，patch 落 config
    import llmsec.core.config as results_cfg
    monkeypatch.setattr(results_cfg, "RESULTS_FILE", tmp_path / "results.json")

    # r9/P3-7：judge/twin_client/reporter 经 deps 注入（不再 patch 构造函数）
    monkeypatch.setattr(tgt, "available_targets", lambda: {
        "t1": NS(model="t1-model", api_key="k", base_url="http://t1"),
    })
    monkeypatch.setattr(tgt, "set_active_target", lambda name: None)

    # Phase 1 评估桩：全部成功且有害（攻击方胜）
    monkeypatch.setattr(ap, "evaluate_single",
                        lambda prompt, ea, judge, use_judge=True: {
                            "content": "ok", "math_score": None, "actual_answer": None,
                            "is_refusal": False, "is_harmful": True, "eval_score": 3.0,
                            "jailbreak_tax": None, "status": "fully_compliant",
                            "compliance_level": "A", "latency_ms": 100,
                            "tokens_prompt": 50, "tokens_completion": 10,
                        })
    # Phase 2 桩：孪生即时生成 + 目标良性回复
    monkeypatch.setattr(alp, "get_or_create_twin",
                        lambda name, rec, cache, client: f"safe twin of {name}")
    monkeypatch.setattr(alp, "call_target",
                        lambda prompt: {"error": None, "content": "a benign safe response",
                                        "target_refused": False})
    # Judge 故障 → M-15 关键词降级（良性内容不命中拒绝词 → 不过敏）
    def _boom_evaluate(self, prompt, response, skip_prescreen=False):
        raise RuntimeError("judge offline in test")
    monkeypatch.setattr(jd.Judge, "evaluate", _boom_evaluate)

    # 攻击集：8 个方法（≥ 预聚类最小方法数）
    records = [{
        "id": f"m{i}", "method": f"m{i}", "prompt": f"attack prompt {i}",
        "expected_answer": 0, "category": "test", "harm_type": "test",
    } for i in range(8)]
    attacks = tmp_path / "attacks.jsonl"
    write_jsonl(str(attacks), records)

    base_argv = ["runner", "--input", str(attacks), "--target", "t1",
                 "--batch-size", "4", "--max-rounds", "1",
                 "--concurrency", "0", "--twin-window", "6"]
    deps = NS(judge=jd.Judge(None), twin_client=NS(), reporter=lambda **kw: None)
    return rn, fixed_run, base_argv, deps


def test_phase2_standalone_allergy_keyspace(tmp_path, monkeypatch):
    """phase 1 落 state.json 后，独立 phase 2 的过敏检测必须测到样本（修复前恒 0）。"""
    rn, fixed_run, base_argv, deps = _offline_runner_env(tmp_path, monkeypatch)
    rn.main(base_argv[1:] + ["--phase", "1"], deps=deps)
    assert (fixed_run / "t1" / "state.json").exists(), "phase 1 应落 state.json"

    res = rn.main(base_argv[1:] + ["--phase", "2"], deps=deps)

    info = res["per_target"]["t1"]
    assert info.get("fpr") is not None, (
        "phase 2 独立运行必须测到过敏样本（修复前：method 名键空间查 unit_id 恒 miss，"
        "total=0 → fpr=None）")
    allergy = (fixed_run / "t1" / "allergy.json")
    assert allergy.exists()
    import json
    summary = json.loads(allergy.read_text(encoding="utf-8"))["summary"]
    assert summary["total"] > 0
    assert summary["false_positive_rate"] == 0.0  # 良性回复 + 关键词降级 → 无过敏


# ============================================================
# H-2: 准奏通知必须到达门下省
# ============================================================

def test_approve_plan_notifies_menxia(tmp_path, monkeypatch):
    """approve_plan 的 KIND_PLAN_APPROVED 必须能被门下省（dept=MENXIA）订阅收到。"""
    from control.agent import bus as bus_mod
    from control.agent import gazette
    from control.agent.bus import KIND_PLAN_APPROVED, reset_bus
    from control.agent.menxia.listener import reinit_menxia

    monkeypatch.setattr(gazette, "_GAZETTE_DIR", tmp_path / "gazette")
    from control.agent.shangshu import plan as plan_mod
    monkeypatch.setattr(plan_mod, "_PLANS_DIR", tmp_path / "plans")
    plan_mod._PLANS_DIR.mkdir(parents=True, exist_ok=True)
    plan_mod.reset_plans()

    reset_bus()
    reinit_menxia()

    got = []
    # 与生产订阅完全同形：门下省按 (MENXIA, [KIND_PLAN_APPROVED]) 注册
    bus_mod.get_bus().subscribe(bus_mod.MENXIA, [KIND_PLAN_APPROVED],
                                lambda m: got.append(m))

    from control.agent.shangshu import approve_plan
    from control.agent.shangshu.plan import P_DRAFTED, Plan, Step, save_plan

    plan = Plan(intent="audit-r7", steps=[Step(id="s1", capability="list_runs", args={})],
                status=P_DRAFTED)
    save_plan(plan)
    approve_plan(plan.id)

    assert got, ("准奏通知必须广播到门下省订阅——修复前 to_dept=SHANGSHU 与 "
                 "bus 过滤 to_dept in (dept, ALL) 永不相交，准奏风险评估整段静默失效")
