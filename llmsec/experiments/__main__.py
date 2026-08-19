"""
experiments — CLI 入口

用法：
  python -m llmsec.experiments run <study.yaml>      运行/续跑 study
  python -m llmsec.experiments report <name>         打印最佳 config + 对比表
  python -m llmsec.experiments trials <name>         列出全部 trial
"""

from __future__ import annotations

import sys

from llmsec.core.logging import get_logger
from llmsec.experiments.schema import StudyConfig

logger = get_logger(__name__)


def _cmd_run(study_yaml: str) -> int:
    cfg = StudyConfig.from_yaml(study_yaml)
    from llmsec.experiments.study import run_study
    summary = run_study(cfg)
    _print_summary(summary, cfg)
    return 0


def _cmd_report(name: str) -> int:

    from llmsec.experiments.study import study_dir, summarize
    sd = study_dir(name)
    cfg_path = sd / "study.yaml"
    if not cfg_path.exists():
        logger.error(f"❌ 未找到 study: {name} (无 {cfg_path})")
        return 1
    cfg = StudyConfig.from_yaml(cfg_path)
    summary = summarize(cfg)
    _print_summary(summary, cfg)
    return 0


def _cmd_trials(name: str) -> int:
    from llmsec.experiments.study import load_trial_records

    trials = load_trial_records(name)
    if not trials:
        logger.info(f"(study '{name}' 无 trial 记录)")
        return 0
    logger.info(f"study '{name}': {len(trials)} 个 trial")
    for t in trials:
        m = t.get("metrics") or {}
        logger.info(f"  trial#{t.get('idx')} seed={t.get('seed')} {t.get('status')}  "
              f"conv_rounds={m.get('conv_rounds')} elo={m.get('defender_elo')}  "
              f"params={t.get('params')}"[:160])
    return 0


def _print_summary(summary: dict, cfg: StudyConfig) -> None:
    best = summary.get("best")
    logger.info("\n" + "=" * 60)
    logger.info(f"📊 study='{summary.get('name')}' 汇总 ({len(summary.get('rows', []))} 个 config)")
    logger.info("=" * 60)
    metric = cfg.objective.metric
    if best:
        logger.info(f"🏆 最佳 {metric}: {best.get(metric + '_mean'):.3f}"
              f" ± {best.get(metric + '_std', 0):.3f}")
        logger.info(f"   params: {best['params']}")
        if best.get("per_target"):
            logger.info(f"   每目标 {metric}: {best['per_target']}")
    logger.info(f"\n{'config':>6}  {metric:>10}  {'elo':>7}  params  | 每目标拆分")
    for i, r in enumerate(summary.get("rows", [])[:20], 1):
        mv = r.get(metric + "_mean")
        mv_s = f"{mv:.3f}" if isinstance(mv, (int, float)) else "-"
        elo = r.get("defender_elo_mean")
        elo_s = f"{elo:.0f}" if isinstance(elo, (int, float)) else "-"
        p = {k: v for k, v in r["params"].items() if k not in ("input", "target", "max_rounds")}
        pt = r.get("per_target") or ""
        logger.info(f"{i:>6}  {mv_s:>10}  {elo_s:>7}  {p}  | {pt}")


def main() -> int:
    args = sys.argv[1:]
    if not args:
        logger.info(__doc__)
        return 0
    cmd = args[0]
    if cmd == "run" and len(args) >= 2:
        return _cmd_run(args[1])
    if cmd == "report" and len(args) >= 2:
        return _cmd_report(args[1])
    if cmd == "trials" and len(args) >= 2:
        return _cmd_trials(args[1])
    logger.info(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
