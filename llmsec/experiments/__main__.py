"""
experiments — CLI 入口

用法：
  python -m llmsec.experiments run <study.yaml>      运行/续跑 study
  python -m llmsec.experiments report <name>         打印最佳 config + 对比表
  python -m llmsec.experiments trials <name>         列出全部 trial
"""

from __future__ import annotations

import sys

from llmsec.experiments.schema import StudyConfig


def _cmd_run(study_yaml: str) -> int:
    cfg = StudyConfig.from_yaml(study_yaml)
    from llmsec.experiments.study import run_study
    summary = run_study(cfg)
    _print_summary(summary, cfg)
    return 0


def _cmd_report(name: str) -> int:
    from llmsec.experiments.study import study_dir, summarize
    import json
    from pathlib import Path
    sd = study_dir(name)
    cfg_path = sd / "study.yaml"
    if not cfg_path.exists():
        print(f"❌ 未找到 study: {name} (无 {cfg_path})")
        return 1
    cfg = StudyConfig.from_yaml(cfg_path)
    summary = summarize(cfg)
    _print_summary(summary, cfg)
    return 0


def _cmd_trials(name: str) -> int:
    from llmsec.experiments.study import study_dir, _load_trials
    trials = _load_trials(study_dir(name) / "trials.jsonl")
    if not trials:
        print(f"(study '{name}' 无 trial 记录)")
        return 0
    print(f"study '{name}': {len(trials)} 个 trial")
    for t in trials:
        m = t.get("metrics") or {}
        print(f"  trial#{t.get('trial')} seed={t.get('seed')} {t.get('status')}  "
              f"conv_rounds={m.get('conv_rounds')} elo={m.get('defender_elo')}  "
              f"params={t.get('params')}"[:160])
    return 0


def _print_summary(summary: dict, cfg: StudyConfig) -> None:
    best = summary.get("best")
    print("\n" + "=" * 60)
    print(f"📊 study='{summary.get('name')}' 汇总 ({len(summary.get('rows', []))} 个 config)")
    print("=" * 60)
    if best:
        print(f"🏆 最佳 {cfg.objective.metric}: {best.get(cfg.objective.metric + '_mean'):.3f}"
              f" ± {best.get(cfg.objective.metric + '_std', 0):.3f}")
        print(f"   params: {best['params']}")
    print(f"\n{'config':>6}  {'conv_rounds':>11}  {'elo':>7}  {'asr':>6}  params")
    for i, r in enumerate(summary.get("rows", [])[:20], 1):
        cr = r.get("conv_rounds_mean")
        cr_s = f"{cr:.2f}" if isinstance(cr, (int, float)) else "-"
        elo = r.get("defender_elo_mean")
        elo_s = f"{elo:.0f}" if isinstance(elo, (int, float)) else "-"
        asr = r.get("asr_mean")
        asr_s = f"{asr:.2f}" if isinstance(asr, (int, float)) else "-"
        p = {k: v for k, v in r["params"].items() if k not in ("input", "target", "max_rounds")}
        print(f"{i:>6}  {cr_s:>11}  {elo_s:>7}  {asr_s:>6}  {p}")


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 0
    cmd = args[0]
    if cmd == "run" and len(args) >= 2:
        return _cmd_run(args[1])
    if cmd == "report" and len(args) >= 2:
        return _cmd_report(args[1])
    if cmd == "trials" and len(args) >= 2:
        return _cmd_trials(args[1])
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
