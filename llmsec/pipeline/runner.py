#!/usr/bin/env python3
from llmsec.core.logging import get_logger

"""
统一编排器 — 自适应安全评估流水线（原根目录 runner.py）

将评估和过敏检测串联为三阶段自适应测试：

Phase 1: 攻击阶段（ELO自适应）
  1. 加载攻击集，初始化 ELO
  2. 从 ELO 中档采样初始 batch → 发送 → Judge 评分 → 实时更新 ELO
  3. 根据 ELO 边界二分搜索，每次推荐下一批攻击
  4. 直到置信度收敛或达到最大轮次

Phase 2: 过敏检测
  5. 取当前 ELO 边界上下 N 个攻击方法
  6. 查找已有安全孪生 → 缺失则按需生成 → 发给目标
  7. 统计 FPR

Phase 3: 综合评判
  8. ASR + FPR → 2D 安全画像
  9. ELO 边界 + 置信度 → 量化安全等级
  10. 输出统一报告 → output/runs/<时间戳>/runner_report.json

用法:
    python runner.py                                    # 全流程
    python runner.py --phase 1                          # 仅攻击阶段
    python runner.py --phase 2                          # 仅过敏阶段
    python runner.py --max-rounds 3 --batch-size 10     # 自定义参数
"""

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from llmsec.core.config import (
    OUTPUT_DIR,
    RUNS_DIR,
    SAFE_TWINS_FILE,  # noqa: F401 — allergy_phase 经 runner 命名空间 lazy import
    STATE_DIR,
    GeneratorConfig,
    TargetConfig,
)
from llmsec.core.io import (
    append_jsonl,  # noqa: F401 — allergy_phase lazy import
    iter_jsonl,  # noqa: F401 — allergy_phase lazy import
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,  # noqa: F401 — attack_phase lazy import + tests monkeypatch
)
from llmsec.core.logging import setup_console
from llmsec.core.text import strip_math_tax  # noqa: F401 — allergy_phase lazy import
from llmsec.evaluation import (
    FAST_REFUSAL_PATTERNS,  # noqa: F401 — allergy_phase lazy import
    ELOTracker,
    Judge,
    create_judge_client,
    evaluate_single,  # noqa: F401 — attack_phase lazy import + tests monkeypatch
    generate_safe_twin,  # noqa: F401 — allergy_phase lazy import
    measure_math_baseline,
    publish_tracker,
)
from llmsec.params import (
    API_DELAY,  # noqa: F401 — allergy_phase lazy import
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_ROUNDS,
    MAX_TWIN_WINDOW,  # noqa: F401 — allergy_phase lazy import
    MIN_TWIN_WINDOW,  # noqa: F401 — allergy_phase lazy import
    SAMPLER_HYBRID_EXPLORE_ROUNDS,
    SAMPLER_INFOGAIN_ALPHA,
    SAMPLER_INFOGAIN_BETA,
    SAMPLER_INFOGAIN_GAMMA,
)
from llmsec.reporting import (
    build_method_stats,
    build_tree,
    generate_narrative,
    load_elo,
    load_prompt_metadata,
)
from llmsec.targets import PCAP_JUDGE_URL, PCAP_MODEL_VERSION, call_target  # noqa: F401

logger = get_logger(__name__)
setup_console()

# ============================================================
# 配置（惰性 from_env：改 env 后新建进程即生效，不再 import 期固化）
# ============================================================
_tcfg = TargetConfig.from_env()
_gcfg = GeneratorConfig.from_env()
TARGET_API_KEY = _tcfg.api_key
TARGET_BASE_URL = _tcfg.base_url
TARGET_MODEL = _tcfg.model
GENERATOR_API_KEY = _gcfg.api_key
GENERATOR_BASE_URL = _gcfg.base_url
GENERATOR_MODEL = _gcfg.model

# 目标后端类型（路由协议，非连接配置）
TARGET_TYPE = os.getenv("TARGET_TYPE", "openai")

# 防御方（目标模型）名称：PCAP 模式使用 PCAP_MODEL_VERSION，其它模式使用 TARGET_MODEL
if TARGET_TYPE == "pcap_judge":
    DEFENDER_NAME = PCAP_MODEL_VERSION
else:
    DEFENDER_NAME = TARGET_MODEL


# ============================================================
# 辅助函数
# ============================================================
def load_prompt_records(filepath) -> list[dict]:
    """加载攻击prompt的JSONL文件（委托 core.io.read_jsonl）。"""
    return read_jsonl(filepath)


# ============================================================
# Phase 2: 过敏检测
# ============================================================
# 以下函数已拆至 llmsec/pipeline/allergy_phase.py，底部兼容性 re-export
# 重新绑定回 runner 命名空间，历史 ``from llmsec.pipeline.runner import X`` 用法不变：
#   - select_twin_candidates
#   - run_allergy_phase
# （compute_min_twin_sample_size / adaptive_twin_window / get_or_create_twin
#   见文件顶部，同样已拆出。）


# ============================================================
# 多目标编排（--targets）
# ============================================================
# 以下函数已拆至 llmsec/pipeline/multi_target.py，底部兼容性 re-export
# 重新绑定回 runner 命名空间，历史 `from llmsec.pipeline.runner import X` 用法不变：
#   - run_multi_target_phase


# ============================================================
# 主流程
# ============================================================
def _positive_int(value: str) -> int:
    """argparse 类型：要求 >=1 的整数（用于 --max-rounds），非法值抛 argparse 错误。"""
    try:
        iv = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"无效整数: {value!r}")
    if iv < 1:
        raise argparse.ArgumentTypeError(f"必须 >= 1，当前为 {iv}")
    return iv


def _allocate_runs_dir(base_dir: Path, name: str) -> Path:
    """返回不冲突的 run 目录路径：name 已存在时追加 _2/_3 后缀。

    run 目录名为秒级时间戳，同一秒内启动两个 run 会撞名——本函数检测到冲突时
    追加递增后缀（name_2、name_3…），确保同秒多 run 产物不互相覆盖。
    """
    candidate = base_dir / name
    suffix = 2
    while candidate.exists():
        candidate = base_dir / f"{name}_{suffix}"
        suffix += 1
    return candidate


def main():
    parser = argparse.ArgumentParser(description="统一编排器 — 自适应安全评估流水线")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "1", "2"],
                        help="运行阶段: all/1(攻击)/2(过敏)")
    parser.add_argument("--input", type=str, default="attacks/l1.jsonl",
                        help="攻击集输入文件")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"每轮测试的攻击数（默认{DEFAULT_BATCH_SIZE}）")
    parser.add_argument("--max-rounds", type=_positive_int, default=DEFAULT_MAX_ROUNDS,
                        help=f"最大自适应轮次（默认{DEFAULT_MAX_ROUNDS}，必须 >= 1）")
    parser.add_argument("--twin-window", type=int, default=None,
                        help="过敏检测方法数上限；未指定时按ELO边界置信度自适应（置信度越低窗口越大）")
    parser.add_argument("--ridge-refit-threshold", type=int, default=10,
                        help="新增 ground truth 方法数达到多少时触发 SVD-Ridge 重跑 K-Fold（默认 10）；"
                             "未达阈值则用现有 λ* 快速 refit")
    parser.add_argument("--refresh-features", action="store_true",
                        help="强制在本次运行开始时重建特征缓存（攻击集/特征未变时本会跳过）")
    parser.add_argument("--sampler", type=str, default="hybrid",
                        choices=["gap", "infogain", "coordinate", "hybrid"],
                        help="Phase 1 采样策略（默认 hybrid）")
    parser.add_argument("--sampler-alpha", type=float, default=SAMPLER_INFOGAIN_ALPHA,
                        help=f"InfoGain 不确定性权重（默认 {SAMPLER_INFOGAIN_ALPHA}）")
    parser.add_argument("--sampler-beta", type=float, default=SAMPLER_INFOGAIN_BETA,
                        help=f"InfoGain 簇覆盖权重（默认 {SAMPLER_INFOGAIN_BETA}）")
    parser.add_argument("--sampler-gamma", type=float, default=SAMPLER_INFOGAIN_GAMMA,
                        help=f"InfoGain 成功潜力权重（默认 {SAMPLER_INFOGAIN_GAMMA}）")
    parser.add_argument("--coordinate-rounds", type=int, default=SAMPLER_HYBRID_EXPLORE_ROUNDS,
                        help=f"Hybrid 模式下前多少轮使用 InfoGain 探索（默认 {SAMPLER_HYBRID_EXPLORE_ROUNDS}）")
    parser.add_argument("--targets", type=str, default=None,
                        help="多目标：逗号分隔的目标名子集（取自 .env TARGETS）；"
                             "指定后 Phase 1 逐目标攻击，结果写入 results 矩阵 R，"
                             "结束后派生每模型 Elo + 训练混合预测器。缺省=旧单目标流程")
    parser.add_argument("--target", type=str, default=None,
                        help="单目标：按名称选择一个 .env 声明的目标进行常规评估"
                             "（走单目标流程，结果写入 R 矩阵）。与 --targets 互斥")
    parser.add_argument("--seed", type=int, default=42,
                        help="全局随机种子，贯穿 K-Fold/D-optimal/PCA（实验复现用，默认 42）")
    parser.add_argument("--work-dir", type=str, default=None,
                        help="实验隔离模式：state/results 写入该目录，不碰全局 R/"
                             "elo_cache，且跳过聚类落盘。HPO trial 用")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="跑满 max_rounds 不提前收敛停（实验 ci_half@固定预算可比性所需）")
    args = parser.parse_args()

    # 实验隔离模式默认跑满预算（ci_half@预算目标要求每个 trial 同预算）；CLI 显式可覆盖
    if args.work_dir:
        args.no_early_stop = True

    # 全局种子注入（实验复现）
    from llmsec.core.seed import set_global_seed
    set_global_seed(args.seed)

    # 实验隔离模式：重绑 results/elo_cache/state 路径到 work-dir，全局零污染（M-17）。
    # elo_access 经 config 模块动态读取这些路径，故重绑模块属性即生效。
    if args.work_dir:
        from pathlib import Path
        wd = Path(args.work_dir)
        wd.mkdir(parents=True, exist_ok=True)
        import llmsec.core.config as _cfg
        import llmsec.core.results as _res
        _res.RESULTS_FILE = wd / "results.json"
        _cfg.ELO_CACHE_FILE = wd / "elo_cache.json"
        # M-17：特征缓存/聚类产物同样隔离——elo_cluster 动态读 core.config 的这两个
        # 路径（仿 elo_access），重绑后 fit_features/_should_refresh_features 读写均落 work-dir
        _cfg.FEATURE_CACHE_FILE = wd / "feature_cache.pkl"
        _cfg.CLUSTER_RESULT_FILE = wd / "cluster_result.pkl"
        logger.info(f"🧪 实验隔离模式: work-dir={wd}（全局 state/results/elo_cache 不被触碰）")

    # 本次运行目录（原模块级 datetime.now() import 副作用移入 main）；
    # 秒级时间戳撞名时追加 _2/_3 后缀，避免同秒两个 run 互相覆盖产物
    runs_dir = _allocate_runs_dir(RUNS_DIR, datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    # 实验隔离模式：所有 per-run 产物（report/attack_results/state快照/...）直接写 work-dir
    if args.work_dir:
        runs_dir = Path(args.work_dir)
    runner_report_file = runs_dir / "runner_report.json"
    runner_attack_file = runs_dir / "attack_results.jsonl"
    runner_allergy_file = runs_dir / "allergy.json"
    runner_sampler_log_file = runs_dir / "sampler_log.jsonl"
    runner_cluster_analysis_file = runs_dir / "cluster_security_analysis.json"

    # 加载攻击集
    input_path = os.path.join(OUTPUT_DIR, args.input) if not os.path.isabs(args.input) else args.input
    if not Path(input_path).exists():
        logger.error(f"❌ 攻击集不存在: {input_path}")
        logger.info("   提示: python -m llmsec.attacks.generate 或 python -m llmsec.attacks.harmbench")
        sys.exit(1)

    records = load_prompt_records(input_path)

    # 按方法分组
    method_records = {}
    for r in records:
        m = r["method"]
        if m not in method_records:
            method_records[m] = r

    target_desc = {
        "pcap_judge": f"PCAP Judge @ {PCAP_JUDGE_URL} (模型: {PCAP_MODEL_VERSION})",
        "local_sim": f"本地模拟 @ {TARGET_BASE_URL} (模型: {TARGET_MODEL})",
        "openai": f"OpenAI @ {TARGET_BASE_URL} (模型: {TARGET_MODEL})",
    }.get(TARGET_TYPE, f"{TARGET_TYPE} @ {TARGET_BASE_URL} (模型: {TARGET_MODEL})")

    logger.info(f"📂 加载 {len(records)} 条攻击prompt，涵盖 {len(method_records)} 种攻击方法")
    logger.info(f"🎯 攻击目标: {target_desc}")
    logger.info(f"   模式: {TARGET_TYPE}")
    logger.info("")

    # 初始化客户端
    # 注意：不再创建 target_client——evaluate_single 忽略该参数，实际走 call_target 路由
    twin_client = OpenAI(api_key=GENERATOR_API_KEY, base_url=GENERATOR_BASE_URL)
    judge_client = create_judge_client()
    judge = Judge(judge_client)
    tracker = ELOTracker()

    # 将 CLI 聚类参数同步给 predictor
    tracker.predictor.ridge_refit_threshold = args.ridge_refit_threshold

    os.makedirs(runs_dir, exist_ok=True)

    # ---- 多目标分支：--targets 指定时逐目标攻击，结果入 R 矩阵 ----
    if args.targets:
        if args.target:
            logger.error("❌ --target 与 --targets 互斥")
            sys.exit(1)
        return run_multi_target_phase(args, records, method_records, runs_dir, judge, twin_client)

    # ---- 单目标命名分支：--target <name> 时切换 DEFENDER + ambient 路由 ----
    # 走常规单目标流程（写 STATE_FILE 供看板展示该模型），call_target 经 ambient 自动路由
    if args.target:
        from llmsec.targets import available_targets, set_active_target
        declared = available_targets()
        if args.target not in declared:
            logger.error(f"❌ 未声明的目标: {args.target}（可用: {sorted(declared)}）")
            sys.exit(1)
        global DEFENDER_NAME
        DEFENDER_NAME = args.target
        set_active_target(args.target)
        logger.info(f"🎯 已选择目标: {args.target} (model={declared[args.target].model} @ {declared[args.target].base_url})")

    # ---- Phase 1 ----
    attack_summary = {}
    if args.phase in ("all", "1"):
        # 如用户要求强制重训练，先重建特征缓存再进入 Phase 1
        if args.refresh_features:
            logger.info("  🔄 强制重建特征缓存 ...")
            tracker.predictor.fit_features(records)
            _inject_predicted_elos(tracker, method_records)
            logger.info("  ✅ 强制重建完成，已更新所有方法预测 Elo")

        attack_summary = run_attack_phase(
            records, None, judge, tracker,
            batch_size=args.batch_size, max_rounds=args.max_rounds,
            attack_file=runner_attack_file,
            sampler=args.sampler,
            sampler_alpha=args.sampler_alpha,
            sampler_beta=args.sampler_beta,
            sampler_gamma=args.sampler_gamma,
            coordinate_rounds=args.coordinate_rounds,
            sampler_log_file=runner_sampler_log_file,
            cluster_analysis_file=(None if args.work_dir else runner_cluster_analysis_file),
            skip_final_clustering=bool(args.work_dir),  # 隔离模式跳过聚类落盘
            state_file=(str(Path(args.work_dir) / "state.json") if args.work_dir
                        else (str(STATE_DIR / f"state__{args.target}.json") if args.target
                              else str(runs_dir / "state.json"))),  # per-run 快照（不再写全局 STATE_FILE）
            no_early_stop=args.no_early_stop,
        )
        # publish_tracker 在 run_attack_phase 每轮已调用（写 R + elo_cache），
        # main() 末尾再次 publish 做最终同步——此处不再重复镜像 R。
    else:
        # 仅过敏阶段：从 per-run/per-target 快照或 R 派生加载 ELO。
        if args.work_dir:
            tracker.load(str(Path(args.work_dir) / "state.json"))
        elif args.target:
            tracker.load(str(STATE_DIR / f"state__{args.target}.json"))
        else:
            # 从 R 派生（唯一真相），不再读全局 state.json
            from llmsec.core.results import ResultsMatrix as _RM
            from llmsec.evaluation.elo import derive_elo as _de

            _R = _RM.load()
            if _R.n_for_model(DEFENDER_NAME) > 0:
                _derived = _de(_R, DEFENDER_NAME, method_catalog=list(method_records.keys()))
                tracker.attacker_ratings = _derived.attacker_ratings
                tracker.defender_ratings = _derived.defender_ratings
                tracker.ground_truth_methods = _derived.ground_truth_methods
                tracker._round_defender_elos = _derived._round_defender_elos
                tracker._defender_match_count = _derived._defender_match_count
        if not tracker.attacker_ratings:
            logger.warning("⚠ 无ELO数据，请先运行 Phase 1")
            sys.exit(1)

    # ---- Phase 2 ----
    allergy_summary = {}
    if args.phase in ("all", "2"):
        boundary_info = tracker.compute_security_boundary(DEFENDER_NAME)
        n_window = adaptive_twin_window(
            boundary_info, len(method_records),
            allergy_summary=allergy_summary, user_window=args.twin_window
        )
        logger.info(f"  📏 本次过敏检测窗口：{n_window} 个方法 "
              f"(ELO边界置信度={boundary_info.get('confidence', 0)*100:.0f}%)")
        allergy_summary = run_allergy_phase(
            method_records, None, twin_client, judge, tracker,
            n_window=n_window,
            allergy_file=runner_allergy_file,
        )

    # ---- Phase 3 ----
    # 越狱税基线测量：攻击集带探针时，用裸数学探针测正常正确率作对照
    tax_block = attack_summary.get("jailbreak_tax", {})
    if tax_block.get("probed", 0) > 0:
        logger.info("  📐 测量越狱税基线（裸数学探针对照）...")
        try:
            baseline = measure_math_baseline()
            if baseline.get("accuracy") is not None and tax_block.get("attack_accuracy") is not None:
                tax_block["baseline_accuracy"] = baseline["accuracy"]
                tax_block["accuracy_drop"] = round(
                    baseline["accuracy"] - tax_block["attack_accuracy"], 4)
                tax_block["baseline"] = baseline
        except Exception as e:
            logger.warning(f"  ⚠ 越狱税基线测量失败（跳过基线对照）: {e}")

    # ---- 生成报告 + 发布 ----
    # 单目标 main 的报告/publish/save 链各自独立 try，某一步失败不阻止后续产物落盘
    try:
        report = generate_final_report(attack_summary, allergy_summary, tracker,
                                       report_file=runner_report_file)
        write_json(runner_report_file, report)
    except Exception as e:
        logger.warning(f"  ⚠ 最终报告生成失败: {e}")

    try:
        # R-cutover：把本次 live tracker 的结果发布进 R（唯一真相）+ Elo 派生缓存。
        publish_tracker(tracker, DEFENDER_NAME)
    except Exception as e:
        logger.warning(f"  ⚠ publish_tracker（写 R 矩阵）失败: {e}")

    try:
        # run 内 state 快照：dashboard 按 run 查看历史时优先读快照
        tracker.save(runs_dir / "state.json")
    except Exception as e:
        logger.warning(f"  ⚠ state 快照保存失败: {e}")

    # cluster_report.json 快照
    global_cluster_report = OUTPUT_DIR / "cluster_report.json"
    if global_cluster_report.exists():
        try:
            shutil.copy2(global_cluster_report, runs_dir / "cluster_report.json")
        except Exception as e:
            logger.warning(f"  ⚠ cluster_report 快照失败: {e}")

    # ---- 生成树形 + 叙事报告（仅使用 runner 自己的数据） ----
    results = read_jsonl(runner_attack_file)
    elo_data = load_elo(OUTPUT_DIR)
    allergy_data = read_json(runner_allergy_file, default={})
    metadata = load_prompt_metadata()

    generated_files = [runner_report_file, runs_dir / "state.json"]

    if results:
        try:
            logger.info("🌳 生成层级安全报告...")
            ms = build_method_stats(results, elo_data, metadata)
            tree = build_tree(ms, allergy_data, elo_data,
                              tax_info=attack_summary.get("jailbreak_tax"))
            tree_path = runs_dir / "security_tree.json"
            write_json(tree_path, tree)
            generated_files.append(tree_path)

            markdown = generate_narrative(tree, OUTPUT_DIR)
            md_path = runs_dir / "security_report.md"
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(markdown, encoding="utf-8")
            generated_files.append(md_path)
        except Exception as e:
            logger.warning(f"  ⚠ 树形/叙事报告生成失败: {e}")

    if Path(runner_attack_file).exists():
        generated_files.append(runner_attack_file)
    if Path(runner_allergy_file).exists():
        generated_files.append(runner_allergy_file)
    if Path(runner_sampler_log_file).exists():
        generated_files.append(runner_sampler_log_file)
    if Path(runner_cluster_analysis_file).exists():
        generated_files.append(runner_cluster_analysis_file)

    # ---- 清晰的文件清单 ----
    generated_files = [str(f) for f in generated_files]
    logger.info("")
    logger.info("=" * 60)
    logger.info("  📋 输出文件")
    logger.info("=" * 60)
    # 按类别分组
    reports = [f for f in generated_files if f.endswith(".md") or "runner_report" in f]
    data = [f for f in generated_files if f.endswith(".json") and "state" not in f.lower() and "allergy" not in f.lower() and "tree" not in f.lower() and "runner_report" not in f]
    jsonl_files = [f for f in generated_files if f.endswith(".jsonl") and "attack_results" not in f]
    allergy = [f for f in generated_files if "allergy" in f.lower()]
    state = [f for f in generated_files if "state" in f.lower()]
    tree_files = [f for f in generated_files if "tree" in f.lower()]
    detail = [f for f in generated_files if ("攻击结果" in f or "attack_results" in f)]

    if reports:
        logger.info("  人类可读报告:")
        for f in reports:
            logger.info(f"    📄 {Path(f).name}")
    if data:
        logger.info("  结构数据:")
        for f in data:
            logger.info(f"    📊 {Path(f).name}")
    if jsonl_files:
        logger.info("  日志数据:")
        for f in jsonl_files:
            logger.info(f"    📜 {Path(f).name}")
    if detail:
        logger.info("  攻击详情（含响应原文，可人工复核）:")
        for f in detail:
            logger.info(f"    🗡️  {Path(f).name}")
    if allergy:
        logger.info("  过敏检测详情:")
        for f in allergy:
            logger.info(f"    🤧 {Path(f).name}")
    if state:
        logger.info("  运行状态:")
        for f in state:
            logger.info(f"    📁 {Path(f).name}")
    if tree_files:
        logger.info("  树形数据:")
        for f in tree_files:
            logger.info(f"    🌳 {Path(f).name}")
    logger.info("\n  💡 想快速看结论 → 打开 security_report.md")
    logger.info("  💡 想看原始数据 → 打开 runner_report.json")
    logger.info("=" * 60)


# ============================================================
# 兼容性 re-export（拆分后保持 from llmsec.pipeline.runner import X 可用）
# F401=unused import（re-export 本身就是"未使用"，必须保留供外部 import）
# ============================================================
from llmsec.pipeline.allergy_phase import (  # noqa: E402, F401
    adaptive_twin_window,
    compute_min_twin_sample_size,
    get_or_create_twin,
    run_allergy_phase,
    select_twin_candidates,
)
from llmsec.pipeline.attack_phase import (  # noqa: E402, F401
    _adaptive_batch_size,
    _compute_method_set_hash,
    _dedup_attack_results,
    _inject_predicted_elos,
    _quick_precluster,
    _should_refresh_features,
    run_attack_phase,
)
from llmsec.pipeline.multi_target import run_multi_target_phase  # noqa: E402, F401
from llmsec.pipeline.tax import format_tax_line, summarize_jailbreak_tax  # noqa: E402, F401
from llmsec.reporting.final_report import (  # noqa: E402, F401
    _compute_conv_rounds,
    generate_final_report,
    generate_recommendation,
)

if __name__ == "__main__":
    # 优先使用项目根目录下的 .venv，避免系统 Python 缺少依赖。
    # 注意：必须在 __main__ 内而非模块顶层，否则 import 本模块（如测试）会被杀进程。
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    _VENV_PYTHON = _PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if _VENV_PYTHON.exists() and sys.executable != str(_VENV_PYTHON):
        _proc = subprocess.run(
            [str(_VENV_PYTHON), "-m", "llmsec.pipeline.runner"] + sys.argv[1:],
            cwd=_PROJECT_ROOT,
        )
        # 透传子进程退出码，避免失败被吞成 0
        sys.exit(_proc.returncode)
    main()
