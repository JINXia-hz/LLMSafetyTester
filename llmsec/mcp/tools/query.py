"""llmsec.mcp.tools.query — Tier 2 只读查询工具（幂等、无副作用）。

复用 llmsec.management / control.core / control.agent.menxia 的只读函数，
绕过 control/core 的 subprocess 胶水层（MCP 与 llmsec 同进程，直接调 Python API）。

暴露的工具：
  - list_runs                列出所有评估 run（含度量摘要）
  - compare_runs             对比多个 run 的指标
  - read_run_report          读单个 run 的完整报告 + 安全树
  - assess_run_findings      用阈值规则审查 run 的异常发现
  - get_results_summary      R 矩阵（统一库 catalog.db）概要
  - elo_ranking              某模型的攻击方 Elo 排名
  - elo_security_boundary    某模型的安全边界（含收敛/置信度）
  - elo_find_surprises       双向意外（防御短板 / 强项）
  - list_workspaces          列出 fork 工作区
"""

from __future__ import annotations

from typing import Any

from llmsec.mcp.tools import _try


# ============================================================
# 工具函数
# ============================================================
def list_runs(
    target: str | None = None,
    since: str | None = None,
    junk_only: bool = False,
    level: str | None = None,
    has_report: bool | None = None,
    min_size: int | None = None,
) -> list[dict[str, Any]]:
    """列出所有评估 run，返回每个 run 的元数据与度量摘要（时间倒序）。

    支持多维过滤。每个 run 条目含：name, target, security_level, asr,
    boundary_elo, has_report, mtime, size 等字段。

    Args:
        target:     只列指定目标模型的 run。
        since:      起始时间（ISO 格式或 yyyy-mm-dd）。
        junk_only:  只列无报告的垃圾/失败 run。
        level:      按安全等级过滤（safe/allergic/vulnerable/broken/inconclusive）。
        has_report: 只列有/无 runner_report.json 的 run。
        min_size:   最小字节数过滤。

    Returns:
        run 元数据 dict 列表（时间倒序）。
    """
    from llmsec.management.runs import discover_runs, filter_runs

    runs = discover_runs()
    if any([target, since, level is not None, has_report is not None, min_size is not None]):
        runs = filter_runs(
            runs,
            target=target or None,
            since=since or None,
            level=level,
            has_report=has_report,
            min_size=min_size or 0,
        )
    if junk_only:
        from llmsec.management.runs import detect_junk

        # 复用已扫描的 runs（修复前这里重复调 discover_runs() 多扫一次目录树）。
        # detect_junk 是纯内存过滤（[r for r in runs if not r["has_report"]]），无需重扫。
        junk_names = {j["name"] for j in detect_junk(runs)}
        runs = [r for r in runs if r["name"] in junk_names]
    return runs


def compare_runs(run_names: list[str]) -> dict[str, Any]:
    """对比多个 run 的评估指标，返回结构化对比报告。

    对比维度包括：安全等级、ASR、Elo 边界、收敛情况、覆盖率等。

    Args:
        run_names: 要对比的 run 名称列表（至少 2 个）。

    Returns:
        对比报告 dict；若某 run 不存在会在报告中标注。
    """
    from control.core.compare import compare

    return _try(lambda: compare(run_names), error_hint="run 名不存在或无报告？用 list_runs 查可用 run。")


def read_run_report(run_name: str) -> dict[str, Any] | None:
    """读取单个 run 的完整评估报告（runner_report.json + security_tree.json）。

    Args:
        run_name: run 名称（格式 "batch/target"，用 list_runs 查）。

    Returns:
        {report, tree, run_dir, run_name}；run 不存在返回 None。
    """
    from control.agent.menxia.review import read_report

    return _try(lambda: read_report(run_name), error_hint="run 名格式 'batch/target'，用 list_runs 查。")


def assess_run_findings(run_name: str) -> dict[str, Any]:
    """用内置阈值规则审查某个 run，产出异常发现列表（findings）。

    自动读取 run 报告并判定：样本量不足、未收敛、FPR 过高、安全等级存疑等。
    这是纯规则判定（不调 LLM），用于快速筛查。

    Args:
        run_name: 要审查的 run 名称。

    Returns:
        {run_name, findings: [...], thresholds: {...}}，每个 finding 含
        severity / metric / value / threshold / interpretation。
    """
    from control.agent.menxia.review import assess_findings, get_thresholds, read_report

    def _do() -> dict[str, Any]:
        data = read_report(run_name)
        if data is None:
            return {"run_name": run_name, "findings": [], "error": "run 不存在或无报告"}
        findings = assess_findings(data["report"], data.get("tree"))
        return {
            "run_name": run_name,
            "findings": findings,
            "thresholds": get_thresholds(),
        }

    return _try(_do)


def get_results_summary() -> dict[str, Any]:
    """读取 R 矩阵（统一库 catalog.db）的概要信息。

    所有模型的评估观测（统一库 observations 表）。

    Returns:
        {models, records, total_observations} 概要；R 不存在或空时返回相应提示。
    """
    from llmsec.core.results import ResultsMatrix
    from llmsec.storage.contract import catalog_db as _catalog_db

    def _do() -> dict[str, Any]:
        results_db = _catalog_db()  # 调期解析（work-dir/测试重绑兼容）
        if not results_db.exists():
            return {"models": [], "records": 0, "total_observations": 0, "note": "R 库不存在，尚无评估数据"}
        R = ResultsMatrix.load()
        models = R.all_models()
        n_records = len(R._r)  # record → model → MatchResult
        total = sum(len(col) for col in R._r.values())
        return {
            "models": sorted(models),
            "records": n_records,
            "total_observations": total,
            "results_db": str(results_db),
        }

    return _try(_do, error_hint="统一库可能损坏，检查 output/state/catalog.db")


def elo_ranking(model: str) -> list[dict[str, Any]]:
    """从 R 矩阵派生指定模型的攻击方 Elo 排名（降序：高 Elo = 强攻击）。

    Elo 从 R 矩阵纯函数回放派生（可随时重算）。进程内按列指纹
    缓存派生的 tracker（elo_access.elo_tracker_for），同一 MCP 会话连续调用不重复
    全量 derive_elo。

    Args:
        model: 目标模型名（R 矩阵中的一列）。

    Returns:
        攻击方排名 dict 列表，每条含 unit/elo/predicted 字段
        （predicted=True 表示该 Elo 是未实测的预测值）。
    """
    return _elo_derive(model, lambda tracker: tracker.get_attacker_ranking())


def elo_security_boundary(model: str) -> dict[str, Any]:
    """从 R 矩阵派生指定模型的安全边界（含收敛状态与置信度）。

    安全边界综合反映模型的防御强度：boundary_elo、是否收敛、置信区间宽度、
    边界上下的攻击方法数等。

    Args:
        model: 目标模型名。

    Returns:
        安全边界 dict，含 boundary_elo / converged / confidence / ci_half /
        methods_above_boundary 等字段。
    """
    return _elo_derive(model, lambda tracker: tracker.compute_security_boundary())


def elo_find_surprises(model: str, min_elo_gap: float = 0.0) -> dict[str, list[dict[str, Any]]]:
    """从 R 矩阵派生指定模型的双向"意外"事件。

    - weakness: 低 Elo 攻击成功 → 模型防御短板（需重点关注）
    - strength: 高 Elo 攻击失败 → 模型防御强项

    Args:
        model:       目标模型名。
        min_elo_gap: 最小 Elo 差距阈值，过滤微小差距的噪声事件。

    Returns:
        {"weakness": [...], "strength": [...]}，每条含 attacker/elo_gap/eval_score。
    """
    return _elo_derive(model, lambda tracker: tracker.find_surprises(min_elo_gap))


def _elo_derive(model: str, extract_fn) -> Any:
    """公共：取派生 tracker（进程内缓存）→ 提取结果。

    经 elo_access.elo_tracker_for 获取按列指纹缓存的 ELOTracker，避免每次工具调用
    都全量 ResultsMatrix.load() + derive_elo。回退保证：缓存层异常时回退到直接派生。
    """
    from llmsec.storage.contract import catalog_db as _catalog_db

    def _do() -> Any:
        if not _catalog_db().exists():
            return {"error": "R 库不存在，尚无评估数据", "model": model}
        tracker = None
        try:
            from llmsec.evaluation.elo_access import elo_tracker_for
            tracker = elo_tracker_for(model)
        except Exception:
            tracker = None  # 缓存层异常则回退到直接派生
        if tracker is None:
            from llmsec.core.results import ResultsMatrix
            from llmsec.evaluation.elo import derive_elo
            R = ResultsMatrix.load()
            tracker = derive_elo(R, model)
        return extract_fn(tracker)

    return _try(_do, error_hint=f"模型 '{model}' 可能不在 R 矩阵中。用 get_results_summary 查可用模型。")


def list_workspaces() -> list[dict[str, Any]]:
    """列出所有 fork 工作区。

    工作区是 llmsec 的隔离实验环境——从全局 R 矩阵 fork 出一份独立副本，
    可以在里面跑实验而不影响全局数据。

    Returns:
        工作区元数据列表，每条含 name/source/note/created 等字段。
    """
    from control.core.workspace import list_workspaces as _lw

    return _try(_lw, error_hint="workspaces 目录可能不存在")


# ============================================================
# 完整审查 + 阈值 + 过敏报告 + 目标模型
# ============================================================
def review_run(run_name: str, use_llm: bool = True) -> dict[str, Any]:
    """对单个 run 执行完整审查：读报告 → 规则判定 → 生成中文叙事摘要。

    比 assess_run_findings 更完整——后者只给规则判定的 findings 列表，
    本工具额外生成含安全等级、严重/警告计数、中文叙事摘要的完整审查报告。

    Args:
        run_name: 要审查的 run 名称。
        use_llm:  是否用 LLM 润色叙事摘要（True 需配 GENERATOR_*；
                  False 用规则模板兜底，离线可用）。

    Returns:
        {run_name, summary, findings, digest, metrics}。
    """
    from control.agent.menxia.review import review_run as _rr

    return _try(lambda: _rr(run_name, use_llm=use_llm))


def get_thresholds() -> dict[str, Any]:
    """读取安全审查的阈值常量（来自 params.py）。

    这些阈值用于判定 run 是否达标：最小测试量、ASR 安全线、FPR 安全线、收敛目标等。

    Returns:
        阈值名 → 值的 dict，如 {PORTRAIT_MIN_TESTED: 5, PORTRAIT_ASR_SAFE: 0.3, ...}。
    """
    from control.agent.menxia.review import get_thresholds as _gt

    return _try(_gt)


def get_allergy_report() -> dict[str, Any]:
    """读取过敏检测（false positive rate）报告。

    过敏检测是安全画像的另一半维度——攻击侧（Elo/ASR）评估"能不能攻破"，
    过敏侧（FPR）评估"会不会误伤"（安全模型对无害请求过度拒绝）。

    Returns:
        过敏报告 dict，含 summary.false_positive_rate 等字段。
    """
    from llmsec.core.config import OUTPUT_DIR
    from llmsec.reporting.report import load_allergy

    return _try(lambda: load_allergy(OUTPUT_DIR))


def list_targets() -> list[dict[str, Any]]:
    """列出 .env 中声明的全部目标模型（可评估的对象）。

    这是 run_evaluation 的自然前置——先查有哪些合法 target 名，再提交评估。
    API key 等敏感信息会脱敏。

    Returns:
        目标模型列表，每条含 name/base_url/model/api_key(脱敏)。
    """
    from llmsec.core.config import load_targets

    def _do() -> list[dict[str, Any]]:
        targets = load_targets()
        out = []
        for name, cfg in targets.items():
            entry: dict[str, Any] = {"name": name}
            if hasattr(cfg, "base_url"):
                entry["base_url"] = cfg.base_url
            if hasattr(cfg, "model"):
                entry["model"] = cfg.model
            if hasattr(cfg, "api_key"):
                k = cfg.api_key or ""
                entry["api_key"] = k[:8] + "***" if len(k) > 8 else "***"
            out.append(entry)
        return out

    return _try(_do, error_hint="检查 .env 是否配置了 TARGETS")


# ============================================================
# 目标模型探活
# ============================================================
def probe_targets(name: str | None = None) -> dict[str, Any]:
    """探测目标模型和服务的 API 连通性（快速健康检查）。

    对每个目标模型发送最轻量请求（models.list + chat smoke），返回是否可达、
    延迟、错误信息。全量模式下还探测 generator 和 judge 服务。

    **强烈建议在 run_evaluation 前先探测**——如果某模型不可达（鉴权失败/网络不通），
    跑完整评估只会得到全 ASR=0 的假阴性结果，浪费 API 额度。

    两阶段探测：
      1. models.list（GET）—— 校验端点连通，不消耗 token
      2. chat smoke（max_tokens=64）—— 校验鉴权（401/403 判不可达）

    Args:
        name: 只探测指定目标模型（不探 services）。None 探测全部目标 + generator + judge。

    Returns:
        {targets: [{name, model, reachable, latency_ms, error, warning}],
         services: [{name, model, reachable, latency_ms, error, warning}]}
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _do() -> dict[str, Any]:
        from llmsec.core.config import (
            GeneratorConfig,
            JudgeConfig,
            load_targets,
        )

        try:
            targets_cfg = load_targets()
        except Exception:
            return {"targets": [], "services": [], "error": "load_targets 失败，检查 .env"}

        if name:
            if name not in targets_cfg:
                return {"targets": [], "services": [],
                        "error": f"目标 {name!r} 不存在（可用: {', '.join(targets_cfg) or '无'}）"}
            targets_cfg = {k: v for k, v in targets_cfg.items() if k == name}

        def _probe_one(n: str, cfg) -> dict[str, Any]:
            """探测单个目标模型（统一走 llmsec.core.probe，与 dashboard 同一实现）。"""
            from llmsec.core.probe import probe_target
            return probe_target(n, cfg)

        def _probe_service(svc_name: str, cfg) -> dict[str, Any]:
            """探测 generator/judge 服务（统一走 llmsec.core.probe）。"""
            from llmsec.core.probe import probe_service
            return probe_service(svc_name, cfg)

        # 并行探测所有目标
        target_results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=min(8, len(targets_cfg) or 1)) as pool:
            futures = {pool.submit(_probe_one, n, c): n for n, c in targets_cfg.items()}
            for fut in as_completed(futures):
                target_results.append(fut.result())
        target_results.sort(key=lambda x: list(targets_cfg.keys()).index(x["name"]))

        services: list[dict[str, Any]] = []
        if not name:
            services.append(_probe_service("generator", GeneratorConfig.from_env()))
            services.append(_probe_service("judge", JudgeConfig.from_env()))

        return {"targets": target_results, "services": services}

    return _try(_do, error_hint="探活失败，检查 .env 连接配置是否正确")
def elo_suggest_next_pairing(
    model: str,
    n: int = 5,
) -> list[dict[str, Any]]:
    """从 R 矩阵派生下一批测试配对建议（选 Elo 差距最小的配对）。

    策略：选 |攻击Elo - 防御Elo| 最小的 n 对——差距最小意味着不确定性最大，
    测试获益最高。用于主动学习/下一轮采样决策。

    Args:
        model: 目标模型名。
        n:     返回配对数（默认 5）。

    Returns:
        配对建议列表，每条为 {attacker, defender}（按信息增益排序的二元组）。
    """
    def _extract(tracker):

        # attackers = tracker 里的全部单位；defender = model 自身
        attackers = list(tracker.attacker_ratings.keys())
        defenders = [model]
        pairs = tracker.suggest_next_pairing(attackers, defenders, n=n)
        return [{"attacker": a, "defender": d} for a, d in pairs]

    return _elo_derive(model, _extract)


# ============================================================
# Plan / Gazette 只读查询（磁盘持久化）
# ============================================================
def list_plans(recent: int = 20) -> list[dict[str, Any]]:
    """列出最近的 Plan（编排计划，按创建时间倒序）。

    Plan 是三省制（尚书省）的结构化多步执行计划。即使 MCP 不走三省制工作流，
    也能通过此工具查看历史 Plan 记录。

    Args:
        recent: 最多返回条数（默认 20）。

    Returns:
        Plan 摘要列表，每条含 id/intent/status/created。
    """
    from control.agent.shangshu.plan import list_plans as _lp

    return _try(lambda: _lp(recent=recent))


def get_plan(plan_id: str) -> dict[str, Any] | None:
    """读取单个 Plan 的完整详情（含步骤、状态、依赖关系）。

    Args:
        plan_id: Plan ID。

    Returns:
        Plan 详情 dict（含 steps/topological_layers/status）；不存在返回 None。
    """
    from control.agent.shangshu.plan import load_plan

    def _do():
        plan = load_plan(plan_id)
        if plan is None:
            return None
        return plan.to_dict()

    return _try(_do)


def list_gazettes(recent: int = 20) -> list[dict[str, Any]]:
    """列出最近的文牍（事件流索引，按时间倒序）。

    文牍是三省制的执行历史记录——每个 Plan 的每一步执行都会产生事件。
    这是了解"过去做过什么"的入口。

    Args:
        recent: 最多返回条数（默认 20）。

    Returns:
        文牍索引列表，每条含 plan_id/intent/created/events_count。
    """
    from control.agent.gazette import list_gazettes as _lg

    return _try(lambda: _lg(recent=recent))


def get_plan_context(plan_id: str) -> dict[str, Any] | None:
    """从文牍事件流重建 Plan 的上下文快照。

    聚合散落的事件为可读视图：意图、各步骤状态、封驳记录、审查记录、事件总数。
    这是查看某个 Plan"执行到哪了、出了什么问题"最有价值的工具。

    Args:
        plan_id: Plan ID。

    Returns:
        上下文快照 dict；Plan 不存在返回 None。
    """
    from control.agent.gazette import read_plan_context

    return _try(lambda: read_plan_context(plan_id))


def read_plan_events(plan_id: str) -> list[dict[str, Any]]:
    """读取某 Plan 的完整事件流（按时间排序）。

    比 get_plan_context 更细——返回每条原始事件，而非聚合视图。
    用于审计/调试某个 Plan 的完整执行时间线。

    Args:
        plan_id: Plan ID。

    Returns:
        事件列表，每条含 ts/kind/dept/detail。
    """
    from control.agent.gazette import read_events

    def _do():
        events = read_events(plan_id)
        return [e.to_dict() if hasattr(e, "to_dict") else e for e in events]

    return _try(_do)


# ============================================================
# Workspace 内 run 发现 + 聚类报告 + 能力自省
# ============================================================
def list_workspace_runs() -> list[dict[str, Any]]:
    """列出所有 fork 工作区内的 run（含报告的）。

    与 list_runs（只扫 output/runs/）互补——本工具扫描 output/workspaces/
    下的分支 run。每个 workspace 可能有多个 target 子目录（各是一个独立 run）。

    Returns:
        run 列表，每条含 name/workspace/target/security_level/asr/boundary_elo。
    """
    from control.core.compare import discover_workspace_runs

    return _try(discover_workspace_runs)


def get_cluster_report() -> dict[str, Any] | None:
    """读取聚类分析报告（cluster_report.json）。

    聚类在评估后把攻击方法按特征相似度分组，用于发现攻击模式。
    需要 [cluster] extras 安装（含 sentence-transformers）。

    Returns:
        聚类报告 dict；未跑过聚类返回 None。
    """
    from llmsec.evaluation.cluster_analysis import load_cluster_report

    return _try(load_cluster_report)


def get_params(category: str | None = None) -> dict[str, Any]:
    """读取 llmsec 的全部行为调参参数（params.py），含当前值、类型和注释。

    这些是控制评估行为的"旋钮"——Elo K 因子、收敛阈值、采样器权重、Judge 评分映射、
    聚类参数等。用 run_evaluation(param_overrides={...}) 可临时覆写（只影响本次评估）。

    参数分 9 组：pipeline（流水线）、elo（Elo 评分与收敛）、ridge（SVD-Ridge 预测）、
    sampler（采样器）、judge（评判与评分）、cluster（聚类与特征）、blend（Blend 预测器）、
    twin（安全双胞胎/过敏）、report（报告）、sim（本地模拟）。

    Args:
        category: 只返回指定分组的参数（如 "elo" / "sampler" / "judge"）。
                  None 返回全部分组。

    Returns:
        {分组名: {参数名: {value, type, description}}}。
    """
    import ast
    import re

    from llmsec import params as params_mod

    def _do() -> dict[str, Any]:
        src_path = params_mod.__file__
        with open(src_path, encoding="utf-8") as f:
            source = f.read()
        source_lines = source.splitlines()
        tree = ast.parse(source)

        # 预扫描：确定每个行号属于哪个分组。
        # 分组标题格式：# N. 标题（括注），两边是 # === 行
        cat_title_pattern = re.compile(r"#\s*(\d+\w?)\.\s*(.+)")
        line_to_cat: dict[int, str] = {}
        current_cat = "other"
        for i, line in enumerate(source_lines):
            stripped = line.strip()
            m = cat_title_pattern.match(stripped)
            if m and i > 0 and source_lines[i - 1].strip().startswith("# ==="):
                # 这是一个分组标题行
                current_cat = m.group(2).strip().split("（")[0].strip()
            line_to_cat[i] = current_cat

        categories: dict[str, dict[str, Any]] = {}

        # 解析每个赋值语句 + 关联注释
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                name = target.id
                if name.startswith("_"):
                    continue
                if not hasattr(params_mod, name):
                    continue
                val = getattr(params_mod, name)
                cat = line_to_cat.get(node.lineno - 1, "other")
                if cat not in categories:
                    categories[cat] = {}
                # 收集同行尾注释 + 下方注释行（描述参数含义）
                desc_parts = []
                line_text = source_lines[node.lineno - 1]
                inline = re.search(r"#\s*(.+)$", line_text)
                if inline:
                    desc_parts.append(inline.group(1).strip())
                for ln2 in range(node.lineno, min(len(source_lines), node.lineno + 5)):
                    nl = source_lines[ln2].strip()
                    if nl.startswith("# ") and not cat_title_pattern.match(nl):
                        desc_parts.append(nl.lstrip("# ").strip())
                    elif nl and not nl.startswith("#"):
                        break
                categories[cat][name] = {
                    "value": val if not isinstance(val, tuple) else list(val),
                    "type": type(val).__name__,
                    "description": " ".join(desc_parts)[:300] if desc_parts else "",
                }

        if category:
            cat_lower = category.lower()
            matched = {k: v for k, v in categories.items() if cat_lower in k.lower()}
            return matched if matched else {
                "error": f"未找到匹配 '{category}' 的分组",
                "available": list(categories.keys()),
            }
        return categories

    return _try(_do)


# ============================================================
# 注册
# ============================================================
def register(mcp: Any) -> None:
    """把本模块所有工具注册到 FastMCP server。"""
    mcp.tool(list_runs)
    mcp.tool(compare_runs)
    mcp.tool(read_run_report)
    mcp.tool(assess_run_findings)
    mcp.tool(review_run)
    mcp.tool(get_thresholds)
    mcp.tool(get_results_summary)
    mcp.tool(elo_ranking)
    mcp.tool(elo_security_boundary)
    mcp.tool(elo_find_surprises)
    mcp.tool(elo_suggest_next_pairing)
    mcp.tool(get_allergy_report)
    mcp.tool(list_targets)
    mcp.tool(probe_targets)
    mcp.tool(list_workspaces)
    mcp.tool(list_workspace_runs)
    mcp.tool(get_cluster_report)
    mcp.tool(get_params)
    mcp.tool(list_plans)
    mcp.tool(get_plan)
    mcp.tool(list_gazettes)
    mcp.tool(get_plan_context)
    mcp.tool(read_plan_events)
