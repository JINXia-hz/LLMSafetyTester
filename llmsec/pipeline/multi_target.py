"""pipeline.multi_target — 多目标编排（--targets）。

从 runner.py 拆出，包含：

  - run_multi_target_phase

为避免与 runner 形成循环导入，并保持运行时语义一致，本模块对 runner.py 的模块级
依赖（DEFENDER_NAME / logger / ELOTracker / run_attack_phase / run_allergy_phase /
publish_tracker / adaptive_twin_window / STATE_DIR / write_json / read_jsonl 等）
统一在函数体内延迟导入（``from llmsec.pipeline.runner import X``）。

特别说明：``run_multi_target_phase`` 原在 runner.py 中通过 ``global DEFENDER_NAME``
对其写入，拆出后为保持“写入 runner 模块全局”的语义，改用
``import llmsec.pipeline.runner as _runner`` 后赋值 ``_runner.DEFENDER_NAME``，
等价于原 ``global`` 写入——这样后续在 runner 模块内（及经 runner 延迟导入的
run_attack_phase / run_allergy_phase）读到的仍是最新值。

对非 runner 模块的依赖（core.results.ResultsMatrix / targets 可用目标切换 /
evaluation.blend_predictor 混合预测器等）在顶层正常导入。
runner.py 底部的兼容性 re-export 区会重新导出 run_multi_target_phase，保证
``from llmsec.pipeline.runner import run_multi_target_phase`` 历史用法仍然可用。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from llmsec.core.results import ResultsMatrix
from llmsec.evaluation.blend_predictor import load_or_fit_blend_predictor


def run_multi_target_phase(
    args,
    records: list[dict],
    method_records: dict[str, dict],
    runs_dir: Path,
    judge,
    twin_client=None,
) -> dict:
    """
    多目标攻击编排。

    对每个选定目标：set_active_target → 复用既有 run_attack_phase 跑 Phase 1 →
    把该目标 tracker.history 镜像进结果矩阵 R。全部跑完后从 R 派生每模型 Elo
    （derive_elo，不跨模型）、训练统一/模型双层混合预测器，输出跨模型汇总。

    R 是唯一真相；STATE_FILE 仅保留最后一个目标的 legacy 视图（次要）。
    """
    import llmsec.pipeline.runner as _runner
    from llmsec.pipeline.runner import (
        STATE_DIR,
        ELOTracker,
        adaptive_twin_window,
        logger,
        publish_tracker,
        read_jsonl,
        run_allergy_phase,
        run_attack_phase,
        write_json,
    )

    # available_targets / set_active_target 延迟导入：原 runner 实现即在函数内
    # ``from llmsec.targets import ...``，且测试经 monkeypatch llmsec.targets 属性
    # 注入桩；顶层导入会在 import 期固化引用而漏掉 monkeypatch，故保持函数内延迟导入。
    from llmsec.targets import available_targets, set_active_target

    declared = available_targets()
    names = [n.strip() for n in args.targets.split(",") if n.strip()]
    invalid = [n for n in names if n not in declared]
    if invalid:
        logger.error(f"❌ 未声明的目标: {invalid}（可用: {sorted(declared)}）")
        sys.exit(1)
    logger.info(f"\n🌐 多目标模式: {len(names)} 个目标 → {names}")

    # 方法特征（方法级、跨模型共享）——提取一次复用
    feat_tracker = ELOTracker()
    feat_tracker.predictor.fit_features(records)
    features = feat_tracker.predictor.artifacts.get("features", {})
    catalog = list(method_records.keys())

    # 载入/初始化结果矩阵 R（唯一真相）
    R = ResultsMatrix.load()
    R.set_method_catalog(catalog)

    per_target: dict[str, dict] = {}
    per_target_attack_files: dict[str, Path] = {}
    trackers: dict[str, ELOTracker] = {}
    do_phase1 = args.phase in ("all", "1")
    do_phase2 = args.phase in ("all", "2")

    # ---------------- Phase 1：逐目标自适应攻击 ----------------
    for idx, name in enumerate(names, 1):
        logger.info(f"\n{'='*60}\n  🎯 目标 [{idx}/{len(names)}]: {name} (model={declared[name].model})\n{'='*60}")
        set_active_target(name)
        _runner.DEFENDER_NAME = name

        tracker = ELOTracker()
        tracker.predictor.ridge_refit_threshold = args.ridge_refit_threshold
        if features:
            tracker.predictor.artifacts = feat_tracker.predictor.artifacts

        if do_phase1:
            attack_file = runs_dir / f"attack_results__{name}.jsonl"
            per_target_attack_files[name] = attack_file
            try:
                run_attack_phase(
                    records, judge, tracker,
                    batch_size=args.batch_size, max_rounds=args.max_rounds,
                    attack_file=attack_file,
                    sampler=args.sampler,
                    sampler_alpha=args.sampler_alpha,
                    sampler_beta=args.sampler_beta,
                    sampler_gamma=args.sampler_gamma,
                    coordinate_rounds=args.coordinate_rounds,
                    sampler_log_file=runs_dir / f"sampler_log__{name}.jsonl",
                    cluster_analysis_file=None,
                    skip_final_clustering=True,
                    state_file=STATE_DIR / f"state__{name}.json",
                    concurrency=getattr(args, "concurrency", None),
                )
            except Exception as e:
                logger.warning(f"  ⚠ 目标 {name} 攻击阶段失败: {e}")
                per_target[name] = {"error": str(e)}
                continue
            # R 唯一真相：把 live tracker 的结果发布进 R + Elo 派生缓存（含收敛轨迹）
            publish_tracker(tracker, name)
        else:
            # 仅 Phase 2：从 per-target state 恢复 tracker（含 Elo/边界，供过敏窗口选取）
            tracker.load(str(STATE_DIR / f"state__{name}.json"))

        trackers[name] = tracker
        live_conv = tracker.check_convergence(
            name, total_methods=len(catalog), tested_count=len(tracker.ground_truth_methods))
        per_target[name] = {
            "defender_elo": round(tracker.get_defender_elo(name), 1),
            "this_run_tested": len(tracker.ground_truth_methods),
            "converged": live_conv["converged"],
            "ci_half": live_conv["ci_half"],
            "drift": live_conv["drift"],
            # fpr 由 Phase 2 填写；缺省 None（canonical runner_report 按此键读取）
            "fpr": None,
            "attack_file": str(per_target_attack_files.get(name, "")),
        }
        if do_phase1:
            # publish_tracker 内部重载并存盘 R，此处刷新本地 R 视图供后续读取
            R = ResultsMatrix.load()
            R.set_method_catalog(catalog)
            logger.info(f"  💾 已写入 R 矩阵: {name} 本次 {len(tracker.ground_truth_methods)} 条，"
                  f"R 累计 {R.n_for_model(name)} 条")

    # ---------------- Phase 2：逐目标过敏检测（FPR）----------------
    if do_phase2 and twin_client is not None:
        logger.info(f"\n{'='*60}\n  🤧 Phase 2: 多目标过敏检测\n{'='*60}")
        for name in names:
            tracker = trackers.get(name)
            if tracker is None or not tracker.attacker_ratings:
                logger.warning(f"  ⚠ {name}: 无 Elo 数据，跳过过敏检测")
                continue
            set_active_target(name)
            _runner.DEFENDER_NAME = name
            boundary_info = tracker.compute_security_boundary(name)
            n_window = adaptive_twin_window(
                boundary_info, len(method_records),
                allergy_summary=None, user_window=args.twin_window)
            allergy_file = runs_dir / f"allergy__{name}.json"
            try:
                asm = run_allergy_phase(
                    method_records, twin_client, judge, tracker,
                    n_window=n_window, allergy_file=allergy_file,
                    concurrency=getattr(args, "concurrency", None))
            except Exception as e:
                logger.warning(f"  ⚠ {name} 过敏检测失败: {e}")
                asm = {"error": str(e)}
            per_target.setdefault(name, {}).update({
                "fpr": asm.get("fpr") if isinstance(asm, dict) else None,
                "allergic": asm.get("allergic") if isinstance(asm, dict) else None,
                "allergy_file": str(allergy_file),
            })
            fpr = asm.get("fpr") if isinstance(asm, dict) else None
            logger.info(f"  {name:28s} FPR={fpr}  过敏={asm.get('allergic') if isinstance(asm,dict) else '?'}"
                  f"/{asm.get('total_tested') if isinstance(asm,dict) else '?'}")

    set_active_target(None)

    # ---- 跨模型汇总（Elo/收敛来自 live tracker；覆盖率按当前攻击集 catalog）----
    logger.info(f"\n{'='*60}\n  📊 跨模型汇总\n{'='*60}")
    catalog_set = set(catalog)
    for name in names:
        info = per_target.get(name, {})
        if not info or "error" in info:
            logger.info(f"  {name}: 失败/无结果 ({info.get('error', '')})")
            continue
        n_catalog = len(R.tested_methods(name) & catalog_set)  # 当前攻击集内的覆盖
        n_total = R.n_for_model(name)                          # R 全量（含历史迁移）
        logger.info(f"  {name:28s} ELO≈{info['defender_elo']:6.0f}  "
              f"本次覆盖{info['this_run_tested']}/{len(catalog)}  R累计{n_total}  "
              f"CI±{info['ci_half']}  drift={info['drift']}/轮  "
              f"收敛={'是' if info['converged'] else '否'}"
              + (f"  FPR={info['fpr']}" if info.get("fpr") is not None else ""))
        info["coverage_in_catalog"] = n_catalog
        info["total_in_R"] = n_total

    # ---- 混合预测器（统一 + 模型，自适应权重）----
    bp_summary = {}
    if features:
        try:
            bp = load_or_fit_blend_predictor(R, features, method_catalog=catalog)
            bp_summary = bp.summary()
            logger.info(f"\n  🧠 混合预测器: unified={bp_summary['unified_trained']}  "
                  f"models={bp_summary['models_trained']}")
            for m, w in bp_summary["weights_per_model"].items():
                logger.info(f"     {m:28s} w_model={w['w_model']:.2f} w_unified={w['w_unified']:.2f}")
        except Exception as e:
            logger.warning(f"  ⚠ 混合预测器训练失败: {e}")

    report = {"mode": "multi_target", "targets": names, "per_target": per_target,
              "blend_predictor": bp_summary}
    report_file = runs_dir / "multi_target_report.json"
    write_json(report_file, report)

    # M-35：多目标 run 也写一份 canonical runner_report.json（取首个成功目标作代表）。
    # 否则 dashboard _discover_runs / report.py 只认单目标 runner_report.json → 多目标 run
    # 对 Web 看板和独立报告完全不可见，load_all_results 回退到更旧的单目标数据。
    try:
        primary = next((n for n in names if "error" not in per_target.get(n, {})), None)
        if primary:
            pinfo = per_target[primary]
            asr_val = None
            af = pinfo.get("attack_file")
            if af:
                try:
                    rows = read_jsonl(af)
                    if rows:
                        succ = sum(1 for r in rows
                                   if r.get("is_harmful", False) or r.get("eval_score", 0) > 0)
                        asr_val = round(succ / len(rows), 4)
                except (json.JSONDecodeError, OSError) as e:
                    # B5：ASR 读失败不可静默写 0（成功 run 被记成 ASR=0 = 数据损坏）
                    logger.warning(f"  ⚠ canonical ASR 计算失败（attack_file={af}）: {e}")
            write_json(runs_dir / "runner_report.json", {
                "generated_at": datetime.now().isoformat(),
                "target_model": primary,
                "mode": "multi_target",
                "security_level": "inconclusive",
                "attack_phase": {"asr": asr_val if asr_val is not None else None,
                                 "total_tested": pinfo.get("this_run_tested", 0)},
                "elo": {"boundary_elo": pinfo.get("defender_elo"),
                        "ci_half": pinfo.get("ci_half"),
                        "converged": pinfo.get("converged")},
                "allergy": {"fpr": pinfo.get("fpr")},
            })
    except OSError as e:
        # B6：canonical 报告写失败 = 多目标 run 对看板不可见，记 ERROR
        logger.error(f"  ❌ canonical runner_report 写入失败: {e}")

    logger.info(f"\n  📝 多目标报告: {report_file}")
    return report
