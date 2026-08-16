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
    python -m llmsec.pipeline.runner                                    # 全流程
    python -m llmsec.pipeline.runner --phase 1                          # 仅攻击阶段
    python -m llmsec.pipeline.runner --phase 2                          # 仅过敏阶段
    python -m llmsec.pipeline.runner --max-rounds 3 --batch-size 10     # 自定义参数
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from llmsec.core.config import (
    PROJECT_ROOT,
    RUNS_DIR,
    GeneratorConfig,
    TargetConfig,
)
from llmsec.core.io import read_jsonl, write_json
from llmsec.core.logging import setup_console
from llmsec.core.seed import get_global_seed
from llmsec.evaluation import ELOTracker, Judge, create_judge_client
from llmsec.evaluation.elo_access import publish_tracker
from llmsec.params import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_ROUNDS,
    DEFAULT_SAMPLER,
    RIDGE_REFIT_THRESHOLD,
    SAMPLER_COORD_MIN_PER_CLUSTER,
    SAMPLER_HYBRID_EXPLORE_ROUNDS,
    SAMPLER_INFOGAIN_ALPHA,
    SAMPLER_INFOGAIN_BETA,
    SAMPLER_INFOGAIN_GAMMA,
)
from llmsec.pipeline.allergy_phase import run_allergy_phase
from llmsec.pipeline.attack_phase import _quick_precluster, run_attack_phase
from llmsec.targets import PCAP_JUDGE_URL, PCAP_MODEL_VERSION

logger = get_logger(__name__)
setup_console()

# ============================================================
# 配置（import 期 from_env 固化：改 env 后须重启进程才生效）
# ============================================================
_tcfg = TargetConfig.from_env()
_gcfg = GeneratorConfig.from_env()
TARGET_BASE_URL = _tcfg.base_url
TARGET_MODEL = _tcfg.model
GENERATOR_API_KEY = _gcfg.api_key
GENERATOR_BASE_URL = _gcfg.base_url

# 目标后端类型（路由协议，非连接配置）
TARGET_TYPE = os.getenv("TARGET_TYPE", "openai")


# ============================================================
# Phase 2: 过敏检测
# ============================================================
# 过敏检测逻辑见 llmsec/pipeline/allergy_phase.py。


# ============================================================
# 主流程
# ============================================================
def _positive_int(value: str) -> int:
    """argparse 类型：要求 >=1 的整数，非法值抛 argparse 错误。"""
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


def partition_publish_names(names: list[str], declared: set[str]) -> tuple[list[str], list[str]]:
    """--publish-global 的目标过滤守卫（纯函数，可单测）。

    全局 R 防注入：只接受 .env TARGETS 声明的目标。历史上测试用 test_model/t1/t2
    跑评估时误 publish 进全局 R，污染 BlendPredictor（samples_per_model 出现假目标）。
    declared 为空集（load_targets 失败/未配置）时不校验，全部放行。

    Returns:
        (allowed, skipped)：允许写全局 R 的目标 / 被拒绝的目标。
    """
    if not declared:
        return list(names), []
    allowed = [n for n in names if n in declared]
    skipped = [n for n in names if n not in declared]
    return allowed, skipped


def main():
    parser = argparse.ArgumentParser(description="统一编排器 — 自适应安全评估流水线")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "1", "2"],
                        help="运行阶段: all/1(攻击)/2(过敏)")
    parser.add_argument("--input", type=str, default="attacks/l1.jsonl",
                        help="攻击集输入文件")
    parser.add_argument("--batch-size", type=_positive_int, default=DEFAULT_BATCH_SIZE,
                        help=f"每轮测试的攻击数（默认{DEFAULT_BATCH_SIZE}，必须 >= 1）")
    parser.add_argument("--max-rounds", type=_positive_int, default=DEFAULT_MAX_ROUNDS,
                        help=f"最大自适应轮次（默认{DEFAULT_MAX_ROUNDS}，必须 >= 1）")
    parser.add_argument("--twin-window", type=_positive_int, default=None,
                        help="过敏检测方法数上限（必须 >= 1）；未指定时按ELO边界置信度自适应（置信度越低窗口越大）")
    parser.add_argument("--ridge-refit-threshold", type=int, default=RIDGE_REFIT_THRESHOLD,
                        help=f"新增 ground truth 方法数达到多少时触发 SVD-Ridge 重跑 K-Fold（默认 {RIDGE_REFIT_THRESHOLD}）；"
                             "未达阈值则用现有 λ* 快速 refit")
    parser.add_argument("--refresh-features", action="store_true",
                        help="强制在本次运行开始时重建特征缓存（攻击集/特征未变时本会跳过）")
    parser.add_argument("--sampler", type=str, default=DEFAULT_SAMPLER,
                        choices=["gap", "infogain", "coordinate", "hybrid"],
                        help=f"Phase 1 采样策略（默认 {DEFAULT_SAMPLER}）")
    parser.add_argument("--sampler-alpha", type=float, default=SAMPLER_INFOGAIN_ALPHA,
                        help=f"InfoGain 不确定性权重（默认 {SAMPLER_INFOGAIN_ALPHA}）")
    parser.add_argument("--sampler-beta", type=float, default=SAMPLER_INFOGAIN_BETA,
                        help=f"InfoGain 簇覆盖权重（默认 {SAMPLER_INFOGAIN_BETA}）")
    parser.add_argument("--sampler-gamma", type=float, default=SAMPLER_INFOGAIN_GAMMA,
                        help=f"InfoGain 成功潜力权重（默认 {SAMPLER_INFOGAIN_GAMMA}）")
    parser.add_argument("--coordinate-rounds", type=_positive_int, default=SAMPLER_HYBRID_EXPLORE_ROUNDS,
                        help=f"Hybrid 模式下前多少轮使用 InfoGain 探索（默认 {SAMPLER_HYBRID_EXPLORE_ROUNDS}，必须 >= 1）")
    parser.add_argument("--coord-min-per-cluster", type=_positive_int, default=SAMPLER_COORD_MIN_PER_CLUSTER,
                        help=f"坐标下降采样器每簇最少实测数（默认 {SAMPLER_COORD_MIN_PER_CLUSTER}，必须 >= 1）")
    parser.add_argument("--targets", type=str, default=None,
                        help="多目标：逗号分隔的目标名子集（取自 .env TARGETS）；"
                             "指定后 Phase 1 逐目标攻击，结果写入 results 矩阵 R，"
                             "结束后派生每模型 Elo + 训练混合预测器。缺省=跑全部声明目标")
    parser.add_argument("--target", type=str, default=None,
                        help="单目标：按名称选择一个 .env 声明的目标进行常规评估"
                             "（走单目标流程，结果写入 R 矩阵）。与 --targets 互斥")
    parser.add_argument("--seed", type=int, default=get_global_seed(),
                        help=f"全局随机种子，贯穿 K-Fold/D-optimal/PCA（实验复现用，默认 {get_global_seed()}）")
    parser.add_argument("--work-dir", type=str, default=None,
                        help="实验隔离模式：所有产物（results/elo_cache/probes/prescreen/blend/"
                             "cluster_report/safe_twins 等全部）写入该目录，全局 output/ 零写入。"
                             "fork/HPO trial 用。")
    parser.add_argument("--publish-global", action="store_true",
                        help="全局模式（无 --work-dir）下，结束时把本次观测 publish 进全局 R 矩阵。"
                             "默认关（单元化原则）：评估产物只在 run_dir，更新全局 R 用 "
                             "`llmsec-manage merge`。work-dir 模式忽略此开关（本就隔离）。")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="跑满 max_rounds 不提前收敛停（实验 ci_half@固定预算可比性所需）")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="批内并行求值并发度：不传=全并发(=batch_size)；0/--no-parallel=串行；N>0=限 N。"
                             "评估纯函数并行、ELO 串行(Model B 同步轮次)，结果与并发度无关")
    parser.add_argument("--no-parallel", action="store_true",
                        help="禁用批内并行求值（等价 --concurrency 0，串行）")
    parser.add_argument("--target-concurrency", type=_positive_int, default=1,
                        help="多目标并发数（默认 1=串行，必须 >= 1；总并发 API = target_concurrency × concurrency）")
    args = parser.parse_args()

    # work_dir 实验模式强制跑满预算：ci_half@固定预算可比性要求每个 trial 同预算
    if args.work_dir:
        args.no_early_stop = True
    if args.no_parallel:
        args.concurrency = 0

    # 全局种子注入（实验复现）
    from llmsec.core.seed import set_global_seed
    set_global_seed(args.seed)

    # 实验隔离模式：经 core.isolation 集中重绑全部产物路径到 work-dir，全局 output/ 零写入。
    # 覆盖 results/elo_cache/feature/cluster/predictors/probes/prescreen/safe_twins 全部 9 个
    # 写入点（原 M-17 只重绑 4 个，probes/prescreen/blend/cluster_report/safe_twins 会泄漏全局）。
    if args.work_dir:
        from llmsec.core.isolation import rebind_to_workdir
        rebind_to_workdir(Path(args.work_dir))

    # 本次运行目录（原模块级 datetime.now() import 副作用移入 main）；
    # 秒级时间戳撞名时追加 _2/_3 后缀，避免同秒两个 run 互相覆盖产物。
    # 实验隔离模式：所有 per-run 产物（report/attack_results/state快照/...）直接写 work-dir
    # （work-dir 已在上面 mkdir，不再走时间戳分配）
    if args.work_dir:
        runs_dir = Path(args.work_dir)
    else:
        runs_dir = _allocate_runs_dir(RUNS_DIR, datetime.now().strftime("%Y-%m-%d_%H%M%S"))
        runs_dir.mkdir(parents=True, exist_ok=True)

    # 加载攻击集（相对路径锚到仓库根：attacks/l1.jsonl → repo_root/attacks/l1.jsonl）
    input_path = os.path.join(PROJECT_ROOT, args.input) if not os.path.isabs(args.input) else args.input
    if not Path(input_path).exists():
        logger.error(f"❌ 攻击集不存在: {input_path}")
        logger.info("   提示: python -m llmsec.attacks.generate 或 python -m llmsec.attacks.harmbench")
        sys.exit(1)

    records = read_jsonl(input_path)

    # 按方法分组
    method_records = {}
    for r in records:
        m = r["method"]
        if m not in method_records:
            method_records[m] = r

    logger.info(f"📂 加载 {len(records)} 条攻击prompt，涵盖 {len(method_records)} 种攻击方法")
    logger.info("")

    # 初始化客户端（twin_client 仅 Phase 2 过敏检测使用，不跑 Phase 2 就不构造）
    do_phase1 = args.phase in ("all", "1")
    do_phase2 = args.phase in ("all", "2")
    twin_client = OpenAI(api_key=GENERATOR_API_KEY, base_url=GENERATOR_BASE_URL) if do_phase2 else None
    judge_client = create_judge_client()
    judge = Judge(judge_client)

    # ---- 确定目标列表 ----
    from llmsec.targets import available_targets

    declared = available_targets()
    if args.target and args.targets:
        logger.error("❌ --target 与 --targets 互斥")
        sys.exit(1)
    if args.target:
        if args.target not in declared:
            logger.error(f"❌ 未声明的目标: {args.target}（可用: {sorted(declared)}）")
            sys.exit(1)
        args.targets = args.target
    elif not args.targets:
        if not declared:
            logger.error("❌ 无可用目标，请在 .env 中声明 TARGETS 或通过看板配置")
            sys.exit(1)
        args.targets = ",".join(declared.keys())

    names = [n.strip() for n in args.targets.split(",") if n.strip()]
    if not names:
        logger.error(f"❌ --targets 解析为空: {args.targets!r}（请给出逗号分隔的目标名）")
        sys.exit(1)
    invalid = [n for n in names if n not in declared]
    if invalid:
        logger.error(f"❌ 未声明的目标: {invalid}（可用: {sorted(declared)}）")
        sys.exit(1)

    # 多目标时各目标后端配置取自 .env TARGETS、彼此可能不同，单目标才按 TARGET_TYPE 描述
    if len(names) > 1:
        logger.info(f"🎯 攻击目标: {len(names)} 个（{', '.join(names)}），逐目标评估")
    else:
        # 单目标：优先用该目标自己的声明配置（--target X 或 TARGETS 中唯一项），
        # 而非全局 TARGET_MODEL/TARGET_BASE_URL——后者只是兜底默认，可能与实际单目标不符
        # （如 --target minimax 但全局 TARGET_MODEL=gemma 时，原日志误导性地显示 gemma）。
        single = names[0]
        try:
            from llmsec.core.config import load_targets as _load_targets
            declared_targets = _load_targets()
        except Exception:
            declared_targets = {}
        tcfg = declared_targets.get(single)
        t_url = tcfg.base_url if tcfg else TARGET_BASE_URL
        t_model = tcfg.model if tcfg else TARGET_MODEL
        target_desc = {
            "pcap_judge": f"PCAP Judge @ {PCAP_JUDGE_URL} (模型: {PCAP_MODEL_VERSION})",
            "local_sim": f"本地模拟 @ {t_url} (模型: {t_model})",
            "openai": f"OpenAI @ {t_url} (模型: {t_model})",
        }.get(TARGET_TYPE, f"{TARGET_TYPE} @ {t_url} (模型: {t_model})")
        logger.info(f"🎯 攻击目标: {single} → {target_desc}")
        logger.info(f"   模式: {TARGET_TYPE}")

    # ---- 1. 快照：R + 特征（运行前一次性获取，运行期间不读不写 R）----
    from llmsec.core.results import ResultsMatrix

    catalog = list(method_records.keys())
    R_snapshot = ResultsMatrix.load()
    _feat_tracker = ELOTracker()
    _feat_tracker.predictor.fit_features(records)
    features = _feat_tracker.predictor.artifacts.get("features", {})

    # 评级单位（簇）在 runner 层只装配一次（输入同一份 features，结果确定），
    # 多目标共享——避免每个目标的 run_attack_phase 内各自 embedding/聚类。
    # 必须在目标线程启动前完成（主线程串行预计算）。
    units = None
    if do_phase1:
        from llmsec.core.units import assemble_units

        pre_labels = _quick_precluster(_feat_tracker, sorted(catalog))
        if not pre_labels:
            # 聚类不可用：每方法自成一个 unit（粒度退化但流程一致）
            pre_labels = {m: i for i, m in enumerate(sorted(catalog))}
        method_pool: dict[str, list[dict]] = {}
        for r in records:
            method_pool.setdefault(r["method"], []).append(r)
        units = assemble_units(pre_labels, method_records, method_pool,
                               _feat_tracker.predictor.artifacts)
        logger.info(f"  🧭 评级单位: {len(units)} 簇（一次装配，多目标共享）")
    R_snapshot.set_unit_catalog(sorted(units.keys()) if units else catalog)

    concurrency = args.concurrency
    target_concurrency = min(args.target_concurrency, len(names))

    # ---- 2. 核心评估函数（per-target，完全独立，不读不写 R）----
    def _eval_one_target(name: str):
        """运行单个目标：评估→报告。返回 (tracker, info)。不写 R。"""
        from llmsec.targets import set_active_target

        run_dir = runs_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)

        tracker = ELOTracker()
        tracker.predictor.ridge_refit_threshold = args.ridge_refit_threshold
        if features:
            tracker.predictor.artifacts = _feat_tracker.predictor.artifacts

        info: dict = {"target": name, "model": declared[name].model}

        # Phase 1：攻击
        # M6：多目标共享全局 CLUSTER_RESULT_FILE，各目标 final_fit 写同一文件会互相覆盖——
        # 串行时仅 names 顺序最后一个目标落盘；并发时无安全落盘方，全部跳过。
        # work_dir 实验模式重绑了隔离的 CLUSTER_RESULT_FILE，不受影响。
        skip_final_clustering = (
            not args.work_dir
            and (target_concurrency > 1 or name != names[-1])
        )
        attack_summary: dict = {}
        if do_phase1:
            attack_file = run_dir / "attack_results.jsonl"
            try:
                attack_summary = run_attack_phase(
                    records, judge, tracker,
                    batch_size=args.batch_size, max_rounds=args.max_rounds,
                    attack_file=attack_file,
                    sampler=args.sampler,
                    sampler_alpha=args.sampler_alpha,
                    sampler_beta=args.sampler_beta,
                    sampler_gamma=args.sampler_gamma,
                    coordinate_rounds=args.coordinate_rounds,
                    coord_min_per_cluster=args.coord_min_per_cluster,
                    sampler_log_file=run_dir / "sampler_log.jsonl",
                    cluster_analysis_file=run_dir / "cluster_security_analysis.json",
                    skip_final_clustering=skip_final_clustering,
                    state_file=str(run_dir / "state.json"),
                    no_early_stop=args.no_early_stop,
                    force_refresh=args.refresh_features,
                    concurrency=concurrency,
                    defender_name=name,
                    r_snapshot=R_snapshot,
                    units=units,
                )
            except Exception as e:
                logger.warning(f"  ⚠ {name} 攻击失败: {e}", exc_info=True)
                # 攻击中途失败不应丢掉已测 ground truth——先尽力落盘再返回
                try:
                    tracker.save(run_dir / "state.json")
                except Exception:
                    logger.warning(f"  ⚠ {name} 攻击失败后 state 落盘也失败", exc_info=True)
                return None, {**info, "error": str(e)}
        else:
            state_path = run_dir / "state.json"
            if not state_path.exists():
                # 不能 sys.exit：并发模式下本函数跑在 worker 线程里，SystemExit
                # 只终止该线程被 fut.result() 吞掉，与串行模式的"杀整个进程"
                # 语义不一致。统一 raise：并发模式记为该目标失败、其余照跑；
                # 串行模式向上传播终止进程。
                raise RuntimeError(
                    f"{name}: --phase 2 需要 Phase 1 的 state.json（{state_path} 不存在）"
                    "——请先运行 --phase all 或 --phase 1")
            tracker.load(str(state_path))

        conv = tracker.check_convergence(
            name, total_methods=len(units) if units else len(catalog),
            tested_count=len(tracker.ground_truth_methods))
        boundary = tracker.compute_security_boundary(name)
        info.update({
            "defender_elo": round(tracker.get_defender_elo(name), 1),
            "this_run_tested": len(tracker.ground_truth_methods),
            "converged": conv["converged"], "ci_half": conv["ci_half"],
            "drift": conv["drift"], "confidence": boundary.get("confidence", 0),
        })

        # Phase 2：过敏
        allergy_smmry: dict = {}
        if do_phase2 and tracker.attacker_ratings:
            set_active_target(name)
            from llmsec.core.units import build_unit_proxy_records
            from llmsec.pipeline.allergy_phase import adaptive_twin_window
            # 过敏检测同样以簇为单位：候选取自 unit 排行榜，孪生 prompt 用 unit 代理
            # 记录（medoid prompt）
            _allergy_recs = build_unit_proxy_records(units) if units else method_records
            n_window = adaptive_twin_window(
                boundary, len(_allergy_recs), user_window=args.twin_window)
            allergy_file = run_dir / "allergy.json"
            try:
                allergy_smmry = run_allergy_phase(
                    _allergy_recs, twin_client, judge, tracker,
                    n_window=n_window, allergy_file=allergy_file,
                    concurrency=concurrency, defender_name=name)
            except Exception as e:
                logger.warning(f"  ⚠ {name} 过敏检测失败: {e}")
                allergy_smmry = {}
            info["fpr"] = allergy_smmry.get("fpr")
            info["allergic"] = allergy_smmry.get("allergic")

        # ---- 3. 报告：先于写 R（报告从 local tracker 生成，R 此刻还是干净的）----
        # reporting 负责生成全部产物（runner_report.json 完整版 + security_tree.json + security_report.md）
        try:
            tracker.save(run_dir / "state.json")
        except Exception:
            logger.warning(f"  ⚠ {name} state 落盘失败", exc_info=True)

        try:
            from llmsec.reporting.final_report import generate_reports
            generate_reports(
                run_dir=run_dir,
                tracker=tracker,
                defender_name=name,
                attack_summary=attack_summary,
                allergy_summary=allergy_smmry,
                total_methods=len(units) if units else len(catalog),
                units=units,
            )
        except Exception as e:
            logger.warning(f"  报告生成失败（回退精简版）: {e}")
            # 精简兜底
            write_json(run_dir / "runner_report.json", {
                "generated_at": datetime.now().isoformat(),
                "target_model": name,
                "security_level": "inconclusive",
                "attack_phase": {"asr": None, "total_tested": info.get("this_run_tested", 0)},
                "elo": {"boundary_elo": info.get("defender_elo"),
                        "ci_half": info.get("ci_half"),
                        "converged": info.get("converged")},
                "allergy": {"fpr": info.get("fpr")},
            })

        return tracker, info

    # ============================================================
    # Phase A+B：评估 + 报告（并发，不写 R）
    # ============================================================
    results: dict[str, dict] = {}
    trackers: dict[str, ELOTracker | None] = {}

    def _eval_and_report(name: str) -> tuple[str, ELOTracker | None, dict]:
        """评估单个目标 + 生成报告。不写 R。返回 (name, tracker, info)。"""
        logger.info(f"\n{'='*60}\n  🎯 {name}\n{'='*60}")
        tracker, info = _eval_one_target(name)
        if tracker:
            logger.info(f"  📄 {name}: ELO≈{info.get('defender_elo',0):.0f}  "
                  f"CI±{info.get('ci_half','?')}  "
                  f"{'✓收敛' if info.get('converged') else '⚠未收敛'}"
                  + (f"  FPR={info.get('fpr')}" if info.get("fpr") is not None else ""))
        else:
            logger.info(f"  ❌ {name}: {info.get('error', '失败')}")
        return name, tracker, info

    if target_concurrency <= 1:
        for name in names:
            n, tr, info = _eval_and_report(name)
            results[n] = info
            trackers[n] = tr
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        logger.info(f"  ⚡ 并发度: {target_concurrency}")
        if not args.work_dir:
            logger.info("  多目标并发：final 聚类落盘全部跳过（无安全落盘方）")
        with ThreadPoolExecutor(max_workers=target_concurrency) as ex:
            futures = {ex.submit(_eval_and_report, name): name for name in names}
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    n, tr, info = fut.result()
                    results[n] = info
                    trackers[n] = tr
                except Exception as e:
                    results[name] = {"target": name, "error": str(e)}
                    trackers[name] = None

    # ============================================================
    # Phase C：写入 R（主线程串行，所有评估+报告完成后）
    # 单元化原则：work-dir 模式 publish 到 work-dir（隔离）；全局模式默认不 publish
    # （评估产物只在 run_dir），需更新全局 R 用 `llmsec-manage merge`。
    # --publish-global 显式恢复旧行为（全局模式下直接 publish 进全局 R）。
    # ============================================================
    if args.work_dir:
        logger.info(f"\n{'='*60}\n  💾 写入 work-dir R（隔离）\n{'='*60}")
        for name in names:
            tracker = trackers.get(name)
            if tracker is None:
                continue
            try:
                publish_tracker(tracker, name)
                logger.info(f"  {name}: {len(tracker.ground_truth_methods)} 条 → work-dir R")
            except Exception as e:
                logger.error(f"  ❌ {name} 写入 R 失败: {e}")
    elif args.publish_global:
        logger.info(f"\n{'='*60}\n  💾 写入全局 R（--publish-global）\n{'='*60}")
        # 全局 R 防注入：只接受 .env TARGETS 声明的目标（守卫逻辑见
        # partition_publish_names——此前内联在编排流程里不可单测）。
        # work-dir 模式不受此限（隔离实验可任意命名）。load_targets 失败时
        # 不校验（放行交由既有逻辑）。
        try:
            from llmsec.core.config import load_targets as _load_targets
            declared = set(_load_targets().keys())
        except Exception:
            declared = set()
        allowed, skipped = partition_publish_names(names, declared)
        for name in skipped:
            logger.warning(
                f"  ⏭️ {name}: 未在 .env TARGETS 中声明，跳过全局 R 写入（防测试目标污染生产数据）。"
                f" 如需隔离测试，请用 --work-dir。"
            )
        for name in allowed:
            tracker = trackers.get(name)
            if tracker is None:
                continue
            try:
                publish_tracker(tracker, name)
                logger.info(f"  {name}: {len(tracker.ground_truth_methods)} 条 → 全局 R")
            except Exception as e:
                logger.error(f"  ❌ {name} 写入 R 失败: {e}")
    else:
        logger.info(f"\n{'='*60}\n  ℹ️ 未写全局 R（单元化默认）\n{'='*60}"
                    f"\n  评估产物在 {runs_dir}/；更新全局 R 用: llmsec-manage merge")

    # ============================================================
    # Phase D：清理（trackers 离开作用域 → GC）
    # ============================================================
    trackers.clear()

    # ---- 汇总日志 ----
    logger.info(f"\n{'='*60}\n  📊 汇总\n{'='*60}")
    R_final = ResultsMatrix.load()
    for name in names:
        info = results.get(name, {})
        if "error" in info:
            logger.info(f"  {name}: ❌ {info['error']}")
        else:
            logger.info(f"  {name:28s} ELO≈{info.get('defender_elo',0):6.0f}  "
                  f"R累计{R_final.n_for_model(name)}  "
                  f"CI±{info.get('ci_half','?')}  "
                  f"{'✓' if info.get('converged') else '⚠'}"
                  + (f"  FPR={info.get('fpr')}" if info.get("fpr") is not None else ""))

    logger.info(f"\n  📁 产出: {runs_dir}/")
    for name in names:
        if (runs_dir / name / "runner_report.json").exists():
            logger.info(f"    {name}/runner_report.json")

    # 自动重训 ML 预筛模型（<1s，不阻塞；数据不足时静默跳过）。
    # 单元化原则：work-dir 模式下重训（MODEL_PATH 已隔离到 work-dir）；全局模式默认跳过
    # （避免单次 run 覆盖全局预筛模型）。
    if args.work_dir:
        try:
            from llmsec.evaluation.prescreen_ml import train as _retrain
            result = _retrain()
            if result.get("trained"):
                logger.info(f"  🧠 ML 预筛模型已更新（work-dir）: {result['n_samples']} 条, "
                      f"CV accuracy={result['cv_accuracy']:.3f}")
        except Exception as e:
            logger.debug(f"ML 预筛自动重训跳过: {e}")
    else:
        logger.debug("全局模式：跳过 ML 预筛自动重训（单元化；用 llmsec-manage 触发）")

    return {"targets": names, "per_target": results}


if __name__ == "__main__":
    # 优先使用项目根目录下的 .venv，避免系统 Python 缺少依赖。
    # 注意：必须在 __main__ 内而非模块顶层，否则 import 本模块（如测试）会被杀进程。
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    _VENV_PYTHON = _PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if _VENV_PYTHON.exists() and os.path.realpath(sys.executable) != os.path.realpath(str(_VENV_PYTHON)):
        _proc = subprocess.run(
            [str(_VENV_PYTHON), "-m", "llmsec.pipeline.runner"] + sys.argv[1:],
            cwd=_PROJECT_ROOT,
        )
        # 透传子进程退出码，避免失败被吞成 0
        sys.exit(_proc.returncode)
    main()
