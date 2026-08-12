"""pipeline.allergy_phase — Phase 2 过敏检测（安全孪生 FPR 测试）。

从 runner.py 拆出，包含：

  - compute_min_twin_sample_size
  - adaptive_twin_window
  - get_or_create_twin
  - select_twin_candidates
  - run_allergy_phase

依赖（core.config / core.io / core.text / evaluation.safe_twin / evaluation.judge /
params / targets）均为顶层直接导入；这些模块不反向导入 pipeline，无循环依赖。
defender_name 是 run_allergy_phase 的函数参数，由调用方（runner）按目标传入。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI

    from llmsec.evaluation.elo import ELOTracker
    from llmsec.evaluation.judge import Judge

import math
import time
from concurrent.futures import ThreadPoolExecutor

from llmsec.core.config import SAFE_TWINS_FILE
from llmsec.core.io import iter_jsonl, write_json
from llmsec.core.logging import get_logger
from llmsec.core.text import strip_math_tax
from llmsec.evaluation.judge import FAST_REFUSAL_PATTERNS
from llmsec.evaluation.safe_twin import (
    append_twin_entry,
    generate_safe_twin,
    judge_allergic,
    make_twin_entry,
)
from llmsec.params import API_DELAY, MAX_TWIN_WINDOW, MIN_TWIN_WINDOW
from llmsec.targets import call_target, set_active_target

logger = get_logger(__name__)


def compute_min_twin_sample_size(
    observed_refusals: int,
    observed_total: int,
) -> int:
    """
    用正态近似（n = z²·p(1-p)/e²，95% 置信取 z=1.96，允许误差 e=0.05）
    估计 FPR 所需的最小样本量。

    返回:
        最小需要的总样本数；没有任何观测时返回保守默认值 MIN_TWIN_WINDOW。
    """
    if observed_total == 0:
        # 没有任何观测时，返回保守默认值
        return MIN_TWIN_WINDOW

    p = observed_refusals / observed_total
    n_required = (1.96 ** 2 * p * (1 - p)) / (0.05 ** 2)
    n_required = max(n_required, observed_total)  # 至少测到当前已观测数
    return int(math.ceil(n_required))


def adaptive_twin_window(
    boundary_info: dict,
    max_methods: int,
    user_window: int | None = None,
) -> int:
    """
    根据 ELO 边界的置信度决定过敏检测样本量。

    思路：边界置信度越低，说明模型表现越不稳定（好坏方法难以区分），
    需要更多安全孪生样本来可靠估计 FPR。

    映射：confidence 0.8 → ~10，0.5 → ~14，0.2 → ~18，
    最终 clamp 在 [MIN_TWIN_WINDOW, min(MAX_TWIN_WINDOW, max_methods)]。
    """
    if user_window is not None:
        # 用户显式窗口同样 clamp 到 [MIN_TWIN_WINDOW, min(MAX_TWIN_WINDOW, max_methods)]，
        # 与 docstring 承诺的最终范围一致
        return min(max(user_window, MIN_TWIN_WINDOW), min(MAX_TWIN_WINDOW, max_methods))

    confidence = boundary_info.get("confidence", 0)
    n = int(round(8 + 12 * (1 - confidence)))
    return min(max(n, MIN_TWIN_WINDOW), min(MAX_TWIN_WINDOW, max_methods))


def get_or_create_twin(method_name: str, rec: dict, twin_cache: dict,
                       twin_client: OpenAI) -> str | None:
    """
    获取或按需生成安全孪生。
    twin_cache: {method_name: safe_prompt}，由 run_allergy_phase 在并行前从
    SAFE_TWINS_FILE 一次性预载；worker 内不再扫文件（M9：core.io 无锁，
    并发 扫文件+append 会产生半写行/重复生成）。
    """
    if method_name in twin_cache:
        return twin_cache[method_name]

    # rec 来自 method_records（runner 按 r["method"] 分组的攻击集原始记录），
    # method 是 dict 键、prompt 是攻击集必填字段，硬索引安全（M-36 的 .get 只针对
    # category/harm_type 这类 README 标注的可选键）
    clean_prompt = strip_math_tax(rec["prompt"])

    twin = generate_safe_twin(clean_prompt, twin_client)
    if twin is None:
        return None

    twin_cache[method_name] = twin["safe_prompt"]

    # 追加写入孪生文件（append_twin_entry 带锁）；落盘失败不拖垮整个 phase——
    # 孪生已在内存缓存，本次检测照常进行
    try:
        append_twin_entry(make_twin_entry(rec, rec.get("id", method_name), clean_prompt, twin))
    except OSError as e:
        logger.warning(f"  ⚠ {method_name[:30]} 孪生落盘失败: {e}")

    return twin["safe_prompt"]


def select_twin_candidates(ranking: list[dict], boundary_elo: float,
                           n_window: int) -> list[dict]:
    """
    在 ELO 边界附近选 n_window 个方法做过敏检测。

    规则：以 |elo - boundary| 距离升序为基准，边界下方取一半、上方取一半；
    一侧不足时按距离用另一侧补齐（不再静默缩水窗口）。

    修复历史缺陷：
      - 旧实现 above[:k] 在降序 ranking 上取的是离边界**最远**的强攻击；
      - 旧兜底 if not candidates 只在两侧全空才触发，一侧不足时窗口被吞。
    """
    if not ranking or n_window <= 0:
        return []
    n_window = min(n_window, len(ranking))

    sorted_by_dist = sorted(ranking, key=lambda x: abs(x["elo"] - boundary_elo))
    below = [r for r in sorted_by_dist if r["elo"] < boundary_elo]
    above = [r for r in sorted_by_dist if r["elo"] >= boundary_elo]

    half = n_window // 2
    candidates = below[:half] + above[:n_window - half]

    # 一侧不足：按距离边界最近优先，从剩余单位补齐
    if len(candidates) < n_window:
        chosen = {r["unit"] for r in candidates}
        for r in sorted_by_dist:
            if len(candidates) >= n_window:
                break
            if r["unit"] not in chosen:
                candidates.append(r)
                chosen.add(r["unit"])

    return candidates


def run_allergy_phase(method_records: dict[str, dict],
                      twin_client: OpenAI, judge: Judge,
                      tracker: ELOTracker, n_window: int,
                      allergy_file,
                      concurrency: int | None = None,
                      defender_name: str | None = None) -> dict:
    """
    在ELO边界上下取N个方法，用安全孪生测试过敏。

    并发：过敏检测无 Elo/共享态（每方法的孪生生成+target调用+judge 完全独立），
    批内整段并行求值，计数后汇总。concurrency=None 全并发；0 串行；N>0 限 N。
    """
    logger.info("=" * 60)
    logger.info("🤧 Phase 2: 过敏检测")
    logger.info("=" * 60)

    # S3：无数据守卫。compute_security_boundary 在无该防御方数据时返回
    # INITIAL_ELO=1500 伪边界（elo.py 早退分支），判 boundary_elo 是死分支；
    # 必须直接查 defender_ratings，否则拿伪边界跑完整检测会产出虚构 FPR。
    # defender_name=None 时与 compute_security_boundary 同口径：恰一个防御方才算有数据。
    if defender_name is not None:
        has_elo_data = defender_name in tracker.defender_ratings
    else:
        has_elo_data = len(tracker.defender_ratings) == 1
    if not has_elo_data:
        logger.warning("  ⚠ 无ELO数据，跳过过敏检测")
        # S6 修复：返 fpr=None（未测）而非 fpr=0（伪"完美无过敏"）
        return {"total_tested": 0, "allergic": 0, "fpr": None}

    ranking = tracker.get_attacker_ranking()
    boundary = tracker.compute_security_boundary(defender_name)
    boundary_elo = boundary["boundary_elo"]

    # 取ELO边界附近的 n_window 个方法（一侧不足按距离补齐，上方取最近侧）
    candidates = select_twin_candidates(ranking, boundary_elo, n_window)

    twin_methods = [r["unit"] for r in candidates]
    logger.info(f"  ELO边界={boundary_elo:.0f}，选取 {len(twin_methods)} 个单位做过敏检测 (窗口={n_window})")
    logger.info(f"  单位: {', '.join(m[:25] for m in twin_methods)}")

    # M9：并行前主线程一次性预载已有孪生（worker 内并发扫文件+append 有竞态）
    twin_cache = {}
    for t in iter_jsonl(SAFE_TWINS_FILE):
        if t.get("method") and t.get("safe_prompt"):
            twin_cache[t["method"]] = t["safe_prompt"]
    allergy_results = []

    # ---- 批内并行求值（过敏检测无 Elo/共享态，每方法整段独立；计数后汇总）----
    max_workers = (len(twin_methods) if concurrency is None
                   else 1 if concurrency <= 0
                   else max(1, min(concurrency, len(twin_methods))))

    def _eval_allergy(method_name):
        # 并发 worker：补 threading.local 的 ambient 目标继承缺口（多目标路由正确）
        if defender_name:
            try:
                set_active_target(defender_name)
            except Exception as e:
                logger.warning(f"     ⚠ 设置活动目标 {defender_name} 失败: {e}")
        rec = method_records.get(method_name)
        if not rec:
            return None
        safe_prompt = get_or_create_twin(method_name, rec, twin_cache, twin_client)
        if safe_prompt is None:
            logger.error(f"     ❌ {method_name[:30]} 孪生生成失败")
            return None
        # H-2：total（FPR 分母）只在 API 成功获取 content 后计数（见下方汇总）
        try:
            api_result = call_target(safe_prompt)
            if api_result["error"]:
                logger.error(f"     ❌ {method_name[:30]} API错误: {api_result['error']}")
                return None
            content = api_result["content"]
        except Exception as e:
            logger.error(f"     ❌ {method_name[:30]} API错误: {e}")
            return None
        # S4：限流紧跟 API 调用（串行路径）；并行模式由各 worker 自然错开，不强制 sleep
        if max_workers == 1:
            time.sleep(API_DELAY)
        # 判定过敏：target_refused 信号优先，否则走 Judge
        target_refused = api_result.get("target_refused", False)
        if target_refused:
            is_allergic = True
            judge_level = api_result.get("meta", {}).get("pred", "N/A")
            judge_failed = False
        else:
            # M-15：Judge 故障降级关键词（对照 evaluate_single 的 H-6）
            try:
                # M-33：与 safe_twin.py 同一判定口径（judge 判拒绝 ∨ 关键词命中，见 judge_allergic）
                is_allergic, _keyword_refused, jr = judge_allergic(judge, safe_prompt, content)
                judge_level = jr["compliance_level"]
            except Exception as _e:
                # S5：打标 judge_failed，下游汇总时可据此判 inconclusive
                logger.warning(f"     ⚠ {method_name[:30]} Judge故障，降级关键词: {_e}")
                is_allergic = any(p.lower() in content.lower() for p in FAST_REFUSAL_PATTERNS)
                judge_level = "N/A"
                judge_failed = True
            else:
                judge_failed = False
        return (method_name, safe_prompt, content, is_allergic, judge_level, judge_failed)

    if max_workers > 1:
        logger.info(f"     ⚡ 批内并行求值 (concurrency={max_workers})")
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            raw = list(ex.map(_eval_allergy, twin_methods))
    else:
        raw = [_eval_allergy(m) for m in twin_methods]

    refused_count = 0
    total = 0
    judge_failed_count = 0
    for res in raw:
        if res is None:
            continue
        method_name, safe_prompt, content, is_allergic, judge_level, judge_failed = res
        total += 1
        if judge_failed:
            judge_failed_count += 1
        if is_allergic:
            refused_count += 1
        allergy_results.append({
            "method": method_name,
            "elo": tracker.get_attacker_elo(method_name),
            "safe_prompt": safe_prompt[:200],
            "is_allergic": is_allergic,
            "judge_level": judge_level,
            "judge_failed": judge_failed,
            "response_preview": content[:500],
        })
        sym = "🤧" if is_allergic else "✅"
        logger.info(f"     {sym} {method_name[:35]} (ELO={tracker.get_attacker_elo(method_name):.0f}) "
              f"algy={is_allergic} level={judge_level}")

    # M8：total==0 时 fpr=None（未测），不伪造 0（与上方 S6 早退口径一致）
    fpr = refused_count / total if total > 0 else None
    fpr_rounded = round(fpr, 4) if fpr is not None else None

    # 保存过敏结果到专用文件（消费方 reporting/report.py 只读 false_positive_rate，
    # 不再重复落同值的 fpr 键）
    write_json(allergy_file, {
        "results": allergy_results,
        "summary": {
            "total": total, "allergic": refused_count,
            "false_positive_rate": fpr_rounded,
            "judge_failed_count": judge_failed_count,
        },
    })

    summary = {
        "total_tested": total,
        "allergic": refused_count,
        "fpr": fpr_rounded,
        "boundary_elo": boundary_elo,
        "methods_tested": twin_methods,
        "judge_failed_count": judge_failed_count,
    }
    if judge_failed_count > 0:
        logger.warning(f"  ⚠ {judge_failed_count}/{total} 条过敏检测用了关键词降级，FPR 可能不准")
    if fpr is not None:
        logger.info(f"\n  📊 过敏检测完成: FPR={fpr*100:.1f}% ({refused_count}/{total})")
    else:
        logger.info("\n  📊 过敏检测完成: 无有效样本，FPR 未测")
    logger.info("")
    return summary
