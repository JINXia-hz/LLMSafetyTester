#!/usr/bin/env python3
"""
evaluation.cli — 评估器命令行入口（参数解析 + 主循环编排 + 终端输出）。

从 evaluator.py 拆出（M-43）：评估核心（evaluate_single / build_summary / update_elo）
留在 evaluator.py，本文件只含 CLI 编排层。依赖方向：cli → evaluator → scoring。

用法：
    python -m llmsec.evaluation.cli                     # 默认1轮
    python -m llmsec.evaluation.cli --repeat 3          # 每条prompt重复3次
    python -m llmsec.evaluation.cli --start-from 1.3.1  # 断点续传
    python -m llmsec.evaluation.cli --only 1.1.1        # 仅评估指定方法
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from llmsec.core.config import ATTACK_SET_L1_FILE, OUTPUT_DIR, PROJECT_ROOT, RUNS_DIR, JudgeConfig
from llmsec.core.io import append_jsonl, read_jsonl, write_json
from llmsec.core.logging import get_logger, setup_console
from llmsec.evaluation.evaluator import (
    DEFENDER_NAME,
    build_summary,
    evaluate_single,
    filter_results_for_model,
    load_records,
    update_elo,
)
from llmsec.evaluation.judge import Judge, create_judge_client
from llmsec.params import API_DELAY, PREVIEW_RESPONSE

logger = get_logger(__name__)
setup_console()


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM攻击评估器")
    parser.add_argument("--repeat", type=int, default=1,
                        help="每条prompt重复测试次数（默认1）")
    parser.add_argument("--only", type=str, default=None,
                        help="仅评估指定方法ID，如 --only 1.1.1")
    parser.add_argument("--start-from", type=str, default=None,
                        help="从指定方法ID开始评估")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="最多评估N条（用于快速测试）")
    parser.add_argument("--no-judge", action="store_true",
                        help="禁用LLM-as-Judge，回退到旧版关键词检测")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Judge使用的模型（默认同GENERATOR_MODEL）")
    parser.add_argument("--skip-judge-prescreen", action="store_true",
                        help="跳过Judge预筛，所有案例都经Judge判断")
    parser.add_argument("--input", type=str, default=None,
                        help="指定输入文件（默认 attacks/l1.jsonl），如 --input harmbench_prompts.jsonl")
    return parser.parse_args(argv)


def resolve_input_file(args: argparse.Namespace):
    """确定输入文件：--input 优先，否则默认 attacks/l1.jsonl。"""
    if args.input:
        p = args.input
        return p if Path(p).is_absolute() else PROJECT_ROOT / p
    return ATTACK_SET_L1_FILE


def init_judge(args: argparse.Namespace, use_judge: bool) -> Judge | None:
    """按 CLI 参数初始化 Judge（--no-judge 时返回 None）。"""
    if not use_judge:
        return None
    judge_client = create_judge_client()
    # M-23：走 JudgeConfig.from_env() 的 or 链（JUDGE_MODEL → GENERATOR_MODEL → DEFAULT_MODEL），
    # 杜绝硬编码 deepseek 模型名与空串 env 传空 model。
    judge_model = args.judge_model or JudgeConfig.from_env().model
    judge = Judge(judge_client, model=judge_model)
    logger.info(f"🧑‍⚖️  Judge模型: {judge_model}\n")
    return judge


def run_evaluation(records: list[dict], args: argparse.Namespace,
                   result_file, done_ids: set,
                   judge: Judge | None, use_judge: bool) -> dict:
    """
    逐条评估主循环：调用 evaluate_single（唯一评估核心），
    追加写入结果文件，支持断点续传。返回计数统计。
    """
    total = len(records) * args.repeat
    done_count = 0
    success_count = 0
    fail_count = 0
    skipped_count = 0
    incomplete_count = 0

    for rec in records:
        # 缺关键字段的记录跳过并计数警告，不让一条坏数据崩掉整批（审查轻微项）。
        # expected_answer 允许为 None（无税探针哨兵），不在必检之列。
        mid = rec.get("id")
        prompt_text = rec.get("prompt")
        method_name = rec.get("method")
        if mid is None or prompt_text is None or method_name is None:
            incomplete_count += 1
            continue
        expected_answer = rec.get("expected_answer")
        from llmsec.core.taxonomy import normalize_harm_type
        harm_type = normalize_harm_type(rec.get("harm_type", "other"))
        category = rec.get("category")

        for round_i in range(args.repeat):
            run_id = f"{mid}-r{round_i}" if args.repeat > 1 else mid

            # 断点续传：跳过的也计入进度，续跑时 [done/total] 不从 1 重计
            if run_id in done_ids:
                skipped_count += 1
                done_count += 1
                continue

            # 单条评估（核心逻辑统一在 evaluate_single）
            result = evaluate_single(
                prompt_text, expected_answer, judge,
                use_judge=use_judge,
                skip_prescreen=args.skip_judge_prescreen,
            )

            if result["status"] == "api_error":
                fail_count += 1
                logger.error(f"❌ {run_id} API错误: {result['error']}")
                continue

            latency = result["latency_ms"]
            tp = result["tokens_prompt"]
            tc = result["tokens_completion"]

            result_entry = {
                "id": run_id,
                "original_id": mid,
                "model": DEFENDER_NAME,  # N-S2：按模型隔离，done_ids/汇总/Elo 据此过滤
                "category": category,
                "method": method_name,
                "harm_type": harm_type,
                "round": round_i,
                "expected_answer": expected_answer,
                "actual_answer": result["actual_answer"],
                "math_score": result["math_score"],
                "is_refusal": result["is_refusal"],
                "is_harmful": result["is_harmful"],
                "eval_score": result["eval_score"],
                "jailbreak_tax": result["jailbreak_tax"],
                "status": result["status"],
                "compliance_level": result["compliance_level"],
                "latency_ms": latency,
                "tokens_prompt": tp,
                "tokens_completion": tc,
                # tp=0 表示 API 未返回 prompt usage，无法计算膨胀比——置 None 而非
                # 退化为 completion 绝对值（爆表），build_summary 统计时排除 None
                "token_ratio": round(tc / tp, 2) if tp > 0 else None,
                "response_preview": result["content"][:PREVIEW_RESPONSE],
                "timestamp": datetime.now().isoformat(),
            }
            # Judge 特有字段（judge_mode 两分支都产生：judge/no_judge/fallback_keyword
            # 三模式标签，供下游过滤异质混合数据；其余仅 Judge 分支产生）
            for key in ("judge_harmfulness", "judge_specificity",
                        "judge_dangerousness", "judge_calls", "prescreen_result",
                        "judge_mode"):
                if key in result:
                    result_entry[key] = result[key]

            append_jsonl(result_file, result_entry)
            done_ids.add(run_id)
            success_count += 1
            done_count += 1

            # 简洁进度
            eval_score = result["eval_score"]
            sym = "✅" if eval_score > 0 else ("🔶" if eval_score == 0 else "❌")
            logger.info(f"{sym} [{done_count}/{total}] {run_id} "
                  f"math={result['math_score']} tax={result['jailbreak_tax']} "
                  f"status={result['status']} "
                  f"lat={latency:.0f}ms")

            time.sleep(API_DELAY)

    if incomplete_count:
        logger.warning(f"⚠️ 跳过 {incomplete_count} 条缺字段记录（缺 id/prompt/method）")

    return {
        "total": total,
        "done_count": done_count,
        "success": success_count,
        "fail": fail_count,
        "skipped": skipped_count,
        "incomplete": incomplete_count,
    }


def print_summary(summary: dict, judge_stats: dict | None,
                  result_file, summary_file) -> None:
    """终端打印汇总（输出格式与原 evaluate.py 一致）。"""
    core = summary["core_metrics"]
    total_runs = core["total_runs"]
    harmful_count = core["harmful_count"]
    refusal_count = core["refusal_count"]
    asr = core["asr"]

    logger.info(f"\n{'='*60}")
    logger.info("📊 评估汇总")
    logger.info(f"{'='*60}")
    logger.info(f"  总运行: {total_runs} 次")
    logger.info(f"  有害产出: {harmful_count} ({asr*100:.1f}%)")
    logger.info(f"  拒绝: {refusal_count} ({refusal_count/total_runs*100:.1f}%)")
    logger.info(f"  ASR: {asr*100:.2f}%")
    logger.info(f"  越狱税均值: {core['jailbreak_tax_mean']:.4f}（仅成功案例）"
                if core['jailbreak_tax_mean'] is not None else
                "  越狱税均值: N/A（无成功案例）")
    logger.info(f"  高税比例 (JT>1): {core['jailbreak_tax_high_ratio']*100:.1f}%")
    logger.info(f"  格式丧失率: {core['format_loss_rate']*100:.1f}%")
    logger.info(f"  平均延迟: {core['latency_mean_ms']:.0f}ms")
    logger.info(f"  Token膨胀比: {core['token_inflation_ratio']:.2f}")
    logger.info(f"  跨类别ASR标准差: {summary['cross_category']['cross_category_std']:.4f}")
    multi_round_stability = summary["multi_round_stability"]
    if multi_round_stability:
        logger.info(f"  多轮一致性: {multi_round_stability['consistent_ratio']*100:.1f}%")
    # Judge额外输出
    if judge_stats:
        logger.info("\n  🧑‍⚖️ Judge 统计:")
        logger.info(f"    合规分布: {judge_stats['compliance_distribution']}")
        logger.info(f"    有害度均值: H={judge_stats['harmfulness_mean']} S={judge_stats['specificity_mean']} D={judge_stats['dangerousness_mean']}")
        logger.info(f"    预筛命中率: {judge_stats['prescreen_hit_rate']*100:.1f}%")
        logger.info(f"    Judge API调用: {judge_stats['total_judge_api_calls']} 次")
        # 注：judge_statistics 的持久化在 main() 落盘前完成（r7/M-5），此处只管终端输出

    logger.info("\n  按有害类别ASR:")
    harm_type_asr = summary["cross_category"]["harm_type_asr"]
    for ht in sorted(harm_type_asr):
        logger.info(f"    {ht}: {harm_type_asr[ht]*100:.1f}%")
    # ELO汇总输出
    if "elo" in summary:
        elo_s = summary["elo"]["summary"]
        elo_b = summary["elo"]["security_boundary"]
        upsets = summary["elo"].get("upsets", [])
        logger.info("\n  🎯 ELO 评分:")
        # 读嵌套 attackers 结构（get_summary 不再提供平铺的 total_methods/min_elo 等旧键）
        att = elo_s.get("attackers", {})
        logger.info(f"    单位数: {elo_s.get('total_attackers', 0)}")
        logger.info(f"    ELO范围: {att.get('min_elo', 0)} ~ {att.get('max_elo', 0)}")
        logger.info(f"    TOP5攻击方: {', '.join(t['unit'] for t in att.get('top_threats', []))}")
        if elo_b.get("boundary_elo") is not None:
            logger.info(f"    安全边界: {elo_b['boundary_elo']} (置信度 {elo_b['confidence']*100:.0f}%)")
            logger.info(f"    边界以上威胁: {elo_b.get('methods_above_boundary', 0)} 种")
        if upsets:
            logger.warning("\n  ⚠️ 意外盲区（低 ELO 成功）TOP5:")
            for u in upsets[:5]:
                logger.info(f"      {u['attacker']} (ELO={u['att_elo']}) 击败 {u['defender']} (ELO={u['def_elo']}) gap={u['elo_gap']}")

    logger.info(f"\n  📁 详细结果: {result_file}")
    logger.info(f"  📁 汇总报告: {summary_file}")
    # r7/L-6：R-cutover 后 state.json 不再由本路径写盘，旧 saved_to 键无写入方（死键）
    logger.info("  📁 ELO状态: R 矩阵")
    logger.info(f"{'='*60}")


def main():
    args = parse_args()

    # 确定输入文件，并据此派生结果文件（不同数据集不同输出，避免覆盖）
    input_file = resolve_input_file(args)

    if not Path(input_file).exists():
        logger.error(f"❌ 输入文件不存在: {input_file}")
        logger.info("   提示: python -m llmsec.attacks.harmbench 或 python -m llmsec.attacks.generate")
        sys.exit(1)

    # M-13：结果文件用稳定的 per-input 路径（非每次新建时间戳目录），使 load_done_ids
    # 能命中上次结果实现真正的断点续传（旧实现每次新 run_dir → done_ids 恒空 → 全部重测，
    # API 成本翻倍）。README 输出布局亦按此 {输入名}_结果.jsonl 口径。汇总仍落时间戳目录。
    base_name = Path(input_file).stem  # e.g. "l1" or "harmbench_jailbreak"
    result_file = OUTPUT_DIR / f"{base_name}_结果.jsonl"

    # 撞名分配（storage.catalog 单一实现）+ 写入口登记：同秒撞名加 _2 后缀且
    # 创建即入目录库（旧实现 mkdir(exist_ok=True) 同秒会共用目录互相覆盖产物）。
    from llmsec.storage import contract as _storage
    run_dir = _storage.allocate_runs_dir(
        RUNS_DIR, datetime.now().strftime(_storage.RUN_TS_FORMAT))
    try:
        _storage.register_run(run_dir, batch=run_dir.name, target=None)
    except Exception as e:
        logger.warning("目录库登记失败（不影响评估，稍后对账自愈）: %s", e)
    summary_file = run_dir / f"{base_name}_汇总.json"

    use_judge = not args.no_judge
    logger.info(f"📂 输入: {Path(input_file).name}")
    logger.info(f"📂 输出: {Path(result_file).name} / {Path(summary_file).name}")
    logger.info("")

    # ---- 加载攻击集 ----
    records = load_records(input_file, args)

    logger.info(f"📋 将评估 {len(records)} 条攻击prompt × {args.repeat} 轮 = {len(records) * args.repeat} 次API调用")
    if use_judge:
        logger.info(f"🧑‍⚖️  使用 LLM-as-Judge 评分 (预筛: {'关闭' if args.skip_judge_prescreen else '开启'})")
    else:
        logger.warning("⚠️  使用旧版关键词检测")
    logger.info("")

    # ---- 加载已有结果（断点续传）----
    # N-S2：按模型隔离（同 safe_twin 的 S-3 修法）。结果文件跨模型共用，
    # done_ids 只取当前 DEFENDER_NAME 的记录；历史无 model 字段的记录视为
    # 不属于任何模型 → 换模型重跑同一输入会真实重测而不是全跳过+张冠李戴。
    done_ids = {r["id"] for r in filter_results_for_model(read_jsonl(result_file))
                if "id" in r}
    if done_ids:
        logger.info(f"📋 已有 {len(done_ids)} 个测试用例已完成，将跳过\n")

    # ---- 初始化Judge ----
    judge = init_judge(args, use_judge)

    # ---- 逐条评估 ----
    counts = run_evaluation(records, args, result_file, done_ids, judge, use_judge)

    # ============================================================
    # 生成汇总报告
    # ============================================================
    logger.info("\n📊 生成汇总报告...")

    # N-S2：全量回读同样按模型过滤——汇总与 update_elo 只回放当前模型的记录，
    # 避免他模型攻击记录被 upsert 进 R 的当前模型列（污染 R 观测）。
    all_results = filter_results_for_model(read_jsonl(result_file))

    if not all_results:
        logger.warning("⚠ 无结果可汇总")
        logger.info(f"\n✅ 评估完成: {counts['success']} 成功, {counts['fail']} 失败")
        return

    summary, judge_stats = build_summary(records, all_results, args, use_judge)

    # r7/M-5：judge 统计块必须在落盘前挂进 summary——原先在 print_summary 内
    # （write_json 之后）才赋值，l1_汇总.json 永远缺整个 Judge 统计区块
    if judge_stats:
        summary["judge_statistics"] = judge_stats
    write_json(summary_file, summary)

    # ---- ELO更新（始终更新；elo 区块仅挂到内存中的 summary，与原版一致） ----
    update_elo(all_results, summary)

    # ---- 终端输出 ----
    print_summary(summary, judge_stats, result_file, summary_file)


if __name__ == "__main__":
    main()
