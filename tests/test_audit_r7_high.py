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

    # CI 无 sentence-transformers/本地模型缓存 → 特征走 TF-IDF 降级；本机有模型时
    # 走真 embedding。固定走降级路径，保证本机与 CI 行为一致（也覆盖该路径）。
    import llmsec.clustering.features as feats_mod
    monkeypatch.setattr(feats_mod, "_get_embedding_model", lambda: None)

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
    # Judge 正常判不拒绝（良性内容 → 不过敏 → FPR=0 但测到样本）。
    # C-2 之后 Judge 故障行不计 FPR 分子分母——"judge 全崩"场景下 fpr 恒 None，
    # 会掩盖本组测试真正要验的键空间/units 覆盖；降级路径的覆盖在
    # tests/test_audit_r10_final.py 的过敏组。
    def _ok_evaluate(self, prompt, response, skip_prescreen=False):
        return {"is_refusal": False, "is_harmful": False, "compliance_level": "A",
                "combined_score": 0.0}
    monkeypatch.setattr(jd.Judge, "evaluate", _ok_evaluate)

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
    from control.agent.bus import KIND_PLAN_APPROVED, reset_bus
    from control.agent.menxia.listener import reinit_menxia
    from control.agent.shangshu import plan as plan_mod
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


# ============================================================
# P0 回归：--publish-global 分支 declared 重绑 set 崩溃（09a1007 引入）
# ============================================================

def test_publish_global_branch_publishes_declared_targets(tmp_path, monkeypatch):
    """--publish-global：守卫放行的目标必须真的 publish 进全局 R。

    修复前 publish 分支把 declared（dict）重绑为 set，随后 declared[name].model
    在 publish 循环与汇总日志两处下标访问必 TypeError——看板/MCP 默认评估通路
    （launch.py publish_global=True）全部以失败告终，全局 R 一条都写不进。
    """
    rn, fixed_run, base_argv, deps = _offline_runner_env(tmp_path, monkeypatch)

    published = []
    monkeypatch.setattr(rn, "publish_tracker", lambda tr, d: published.append(d))
    # t1 声明进 .env 目标集 → 防注入守卫放行（覆盖 declared[name].model 下标路径；
    # 即便守卫全跳过，汇总日志的 declared[name].model 同样会崩）
    import llmsec.core.config as cfg
    monkeypatch.setattr(cfg, "load_targets",
                        lambda: {"t1": NS(model="t1-model", api_key="k",
                                          base_url="http://t1")})

    rn.main(base_argv[1:] + ["--publish-global"], deps=deps)

    assert published == ["t1-model"], (
        "守卫放行目标必须经 publish_tracker 写入——修复前此路径在写入前即 "
        "TypeError: 'set' object is not subscriptable（runner.py:627/651）")
