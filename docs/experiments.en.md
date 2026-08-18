# Experiment Framework (HPO)

> English | [中文](实验框架说明.md)

`llmsec/experiments/` is a hyperparameter-optimization (HPO) framework for systematically tuning the evaluation's behavioral parameters and sampling strategies — the goal is to **converge on a reliable Elo security boundary with the fewest test rounds (API calls)**.

---

## 1. Why it exists

Evaluation quality depends on many coupled knobs: Elo's K factor and τ, convergence thresholds, sampler weights and strategy, batch size… Tuning them by hand is slow and irreproducible. The framework treats them as searchable **factors** and automatically:

- Enumerates/samples config combinations over the search space
- Repeats each config with multiple random seeds (`repeats`) to wash out noise
- Extracts scientific metrics, aggregates them by the objective, and reports the "best config"
- Stays fully reproducible and resumable

The core optimization target is **`conv_rounds`** (the round at which convergence is first reached — lower is better): fewer rounds = fewer real API calls = cheaper evaluation.

---

## 2. Architecture

```mermaid
graph LR
    S[study.yaml] --> R[Search engine<br/>grid/random/bayesian]
    R --> T[Trial subprocess isolation]
    T --> M[Metric extraction conv_rounds]
    M --> R
    M --> A[Aggregate across repeats]
    A --> B[Best config + comparison table]
```

- **study**: one experiment run, defined by a `study.yaml` (search space, objective, budget).
- **trial**: one config × one seed = one runner invocation. Each trial runs runner in an **isolated work-dir** as a subprocess and **never touches the global `state`/`results`**, so trials don't pollute each other.
- **manifest**: each trial writes `manifest.json` (git version / params snapshot / argv / attack-set sha1 / seed / library versions / redacted `.env` keys) so the "best config" can be reproduced exactly.

---

## 3. study.yaml format

```yaml
name: sampler-tuning          # study name (determines output dir output/experiments/<name>/)
description: Tune sampler and Elo K
strategy: bayesian            # grid | random | bayesian(optuna TPE)
repeats: 3                    # number of seeds per config (washes out noise)
seed_base: 0                  # seed = seed_base + i
budget:
  max_trials: 30              # total trial budget cap

objective:                    # optimization target
  metric: conv_rounds         # metric name (see §5)
  direction: minimize         # minimize | maximize
  aggregate: mean             # aggregate across repeats: mean | mean_plus_std

# Locked dimensions: fixed values used by every trial
fixed:
  input: l1.jsonl             # attack set (relative to output/, or with attacks/ prefix)
  target: deepseek-v4-flash   # model under test
  max_rounds: 12

# Search space: factors to tune
space:
  sampler:                    # categorical factor
    choices: [hybrid, infogain, coordinate]
  batch_size:                 # int factor
    type: int
    low: 5
    high: 15
    step: 5
  K_FACTOR:                   # params factor (injected as LLMSEC_PARAM_K_FACTOR)
    type: int
    low: 16
    high: 48
    step: 8
  SCORE_PERF_TAU:             # float factor (sampled in log space)
    type: float
    low: 1.0
    high: 4.0
    log: true
```

### Two factor classes (see `schema.py: resolve_trial`)

| Factor class | Examples | Injection |
|---|---|---|
| **CLI factors** | `sampler` / `batch_size` / `max_rounds` / `sampler_alpha` / `coordinate_rounds` / `target` / `input` … | Translated into runner argv flags (`--sampler`, …) |
| **params factors** | `K_FACTOR` / `SCORE_PERF_TAU` / `CONV_CI_TARGET` … any `params.py` constant | Injected as env var `LLMSEC_PARAM_<NAME>`, applied by `params.py` at import time |

> In other words: **any** behavioral parameter in `params.py` can be searched as a factor — the subprocess sets `LLMSEC_PARAM_<NAME>=value` before importing `llmsec.params`, so the binding takes effect with no code changes.

Factor types: `int` / `float` (optional `log: true` for log-space sampling) / `categorical` (`choices` list).

---

## 4. Search strategies

| Strategy | Description | Use when |
|---|---|---|
| `grid` | Full cartesian product (discrete value set per factor) | Small space, want a complete comparison table |
| `random` | Uniform random sampling until budget exhausted | Medium space, quick scouting |
| `bayesian` | optuna TPE sequential model-based optimization; uses completed results to guide the next config | Large space, budget-saving (default) |

Resumable runs: the `trials` table in the unified DB is the source of truth (sole write path since P4; legacy trials.jsonl is import-only). Restarting a study skips already-completed (config × seed) pairs; `bayesian` additionally feeds prior results back into TPE to rebuild the study state.

---

## 5. Metrics and objective

Each trial extracts metrics from its work-dir's `runner_report.json` (`metrics.py: extract_metrics`):

| Metric | Source | Meaning |
|---|---|---|
| **`conv_rounds`** ★ | `elo.conv_rounds` | First round satisfying the convergence criteria (default HPO target) |
| `defender_elo` | `elo.boundary_elo` | Defender's security-boundary Elo at convergence |
| `ci_half` | `elo.ci_half` | Half-width of the 95%CI for the true Elo |
| `asr` | `attack_phase.asr` | Attack success rate |
| `coverage` | `elo.coverage` | Fraction of methods tested |
| `fpr` | `allergy.fpr` | False-positive rate (if the allergy phase ran) |

`conv_rounds` fallback: if `runner_report.json` is missing or the field is empty, it is recomputed by replaying `check_convergence` over the `state.json` round trajectory; **non-converged trials get a penalty** `max_rounds + ci_half/target`, so "nearly converged" still ranks above "never converged".

Aggregation across repeats (`aggregate`):
- `mean`: simple mean.
- `mean_plus_std`: mean + std (under minimization, prefers configs that are **low and stable**, penalizing jittery ones). `mean_minus_std` is a legacy alias with the same behavior.

---

## 6. CLI

```bash
python -m llmsec.experiments run <study.yaml>      # run / resume the whole study
python -m llmsec.experiments report <name>         # print best config + comparison table
python -m llmsec.experiments trials <name>         # list all trial details
```

Example `report` output:

```
📊 study='sampler-tuning' summary (12 configs)
🏆 best conv_rounds: 5.667 ± 0.471
   params: {'sampler': 'hybrid', 'batch_size': 10, 'K_FACTOR': 32, ...}

 config  conv_rounds      elo     asr  params
      1         5.67     1512    0.42  {'sampler': 'hybrid', 'batch_size': 10, ...}
      2         6.33     1505    0.45  {'sampler': 'infogain', ...}
      ...
```

---

## 7. Output layout

```
output/experiments/<name>/
├── study.yaml          # config copy (reused by report)
│                       # (trial records live in the unified DB `trials` table)
├── best.json           # best config, written by report
└── <trial_idx>/        # each trial's isolated work-dir
    ├── manifest.json   #   reproducibility manifest (git/params/argv/attack-set hash/seed/libs)
    ├── runner.log      #   full runner subprocess log
    ├── runner_report.json
    └── state.json / catalog.db     # that trial's isolated state & satellite DB
```

---

## 8. Isolation and safety

- Each trial runs `runner --phase 1` in an isolated work-dir (experiments only care about attack-phase convergence metrics) and **never writes the global `output/state/`**, avoiding pollution of production data.
- `manifest.json` captures the git commit / dirty flag / redacted `.env` keys so the best config is exactly reproducible.
- Subprocesses run with `PYTHONUNBUFFERED=1` so logs stream in real time.

> 📚 For the theory behind runner / Elo / convergence criteria / samplers, see the project README ("Core concepts") and [attack-features-clustering.en.md](attack-features-clustering.en.md).
