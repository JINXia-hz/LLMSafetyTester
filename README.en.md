# LLM API Security Evaluation System

> English | [中文](README.md)

A systematic black-box LLM security evaluation framework: adaptive attack testing → ELO threat ranking → SVD-Ridge / Blend batch prediction → over-sensitivity detection → cluster analysis → human-readable Markdown security report.

> What this project evaluates is the **security-evaluation pipeline itself** (adaptive sampling, threat ranking, capability prediction, over-sensitivity detection, cluster analysis). Attack sets are merely input consumables.

---

## What it does

```mermaid
graph LR
    A[Attack set JSONL] --> B[Phase 1 Adaptive attack]
    B --> C[ELO-driven round-by-round sampling]
    C --> D[Phase 2 Over-sensitivity detection]
    D --> E[Phase 3 Synthesis report]
    E --> F[security_report.md<br/>+ Web dashboard]
```

- **Phase 1 — Attack**: round-by-round attack-method selection driven by reverse ELO; the Judge scores responses and updates ELO live, until the defender's true-Elo 95% confidence interval converges.
- **Phase 2 — Over-sensitivity**: near the ELO boundary, "safe twins" (semantically safe, structurally similar prompts) test whether the model over-refuses legitimate requests (FPR).
- **Phase 3 — Synthesis**: combines ASR (attack success rate) + FPR (false-positive rate) + ELO boundary into a quantified security profile.

---

## Core architecture: R is the single source of truth

The whole evaluation system is built around the **results matrix R** (`core/results.py`). R is the only non-recomputable raw observation; everything else is a derived cache:

```
R[method][model] = MatchResult         ← single source of truth (raw observation)
        │
        ├── derive_elo(R, model)       → ELOTracker (ratings / trajectory / convergence)
        │     (evaluation/elo.py; replays a model's column in time order; pure function, recomputable anytime)
        │
        ├── BlendPredictor(R, X)       → unified + model two-layer prediction (cold-start Elo)
        │     (evaluation/blend_predictor.py)
        │
        └── clustering / report / dashboard  → read R-derived state, never read state.json directly
              (via the unified entry point evaluation/elo_access.py)
```

This guarantees:

1. **No cross-model Elo contamination** — each model's Elo is replayed solely from its own column, never borrowing another model's.
2. **Native multi-model support** — R's second dimension is the model; the `TARGETS` env var can scan multiple targets at once.
3. **Rebuildable** — Elo, predictors, and convergence can all be fully recomputed from R + method-feature matrix X; losing the cache is harmless.

Storage layout (`output/state/`): `results.json` (R, authoritative) + `elo_cache.json` (derived cache, deletable/rebuildable). `state.json` degrades to an optional snapshot backup.

---

## Core concepts

### Reverse ELO + K dynamics

Attack methods are the offense, the target model is the defense. A successful attack raises the method's ELO; a blocked attack lowers it. The defender's ELO is the "security boundary" — higher-ELO attack methods are more threatening.

- **Continuous score mapping**: `perf = score/(score+τ)` (saturating); the score magnitude goes into the outcome term rather than the K factor, curing early Elo jitter.
- **K decay**: attackers use full K (each method is usually tested only 1–2 times); the defender plays every match, so `K_def = K / sqrt(max(1, n/N0))` — more matches ⇒ a more stable rating.

### Single-CI convergence criterion

Convergence is no longer a weighted blend of metrics. Instead the defender's Elo trajectory is fit with OLS, separating **drift** (movement toward the true value) from **noise** (random jitter), combined into the "half-width of the defender true-Elo 95%CI". Convergence holds iff:

```
ci_half < CONV_CI_TARGET   ∧   |drift| < CONV_DRIFT_TARGET   ∧   coverage sufficient   ∧   enough rounds
```

A single, interpretable, false-positive-resistant stopping rule (`evaluation/elo.py:check_convergence`).

### Blend two-layer prediction (cold start)

Unmeasured methods' initial Elo comes from a predictor that adaptively blends two layers (`evaluation/blend_predictor.py`):

- **Unified prediction P_u**: trained pooled across all models, capturing "intrinsic method threat" (a strong jailbreak is strong against most models). The only source of priors when cold-starting a new model.
- **Model prediction P_m**: trained on that model's column only, capturing model-specific weaknesses.
- **Blend**: `pred = w_u·P_u + w_m·P_m`, where `w_m = n_model/(n_model + K)`: few samples ⇒ rely fully on unified (shrink toward the group mean); many samples ⇒ trust itself. This is empirical Bayes shrinkage — the weight grows with evidence, no manual tuning.

Under the hood is **SVD-Ridge**: train Ridge on measured methods' feature matrix X and derived Elo y, SVD-decompose X, and one forward pass yields all unmeasured predictions; K-Fold selects the optimal λ on the regularization path; each prediction carries MAP uncertainty (variance + 95%CI). Unchanged ground truth ⇒ reuse weights for a pure matrix prediction; growth ⇒ fast refit with the existing λ; large growth ⇒ rerun K-Fold.

### Unified prior metric space + clustering (post-test)

The distance metric for clustering and prediction **uses only prior features** (available for any method: textual structure + semantic embedding + attack technique + intent + name priors); posterior features (defense interaction, all-zero for unmeasured points) never enter the metric. First z-score standardize, then SVD reduce + truncate the noise tail at the spectral knee.

After testing, real machine reactions drive **weakly-supervised feature weighting** (amplify relevant directions, suppress irrelevant ones). HDBSCAN (EOM) density clustering cuts candidate k values from one shared scalable tree (`k0 = ceil(log2(n))` as the center), and silhouette/CH/DB are combined to take the global argmax as the key layer. After clustering, ANOVA / Kruskal-Wallis post-hoc tests verify whether cluster effects are significant.

### D-Optimality active learning

Cold-start seeds use greedy D-optimality: repeatedly pick the method maximizing `xᵀ(XᵀX + λI)⁻¹x` (the direction most informative to the prediction matrix), updating the information matrix each time via Sherman–Morrison rank-1 (`evaluation/active_learning.py`). This shares the same origin as Ridge's MAP variance. When GT is empty it automatically degrades to the max-leverage point, naturally covering the feature space.

### Samplers

Pluggable attack-method sampling strategies (`evaluation/samplers.py`) aiming to converge on a reliable boundary in the fewest tests:

- `gap`: pick by smallest |attack ELO − defense ELO|
- `infogain`: global information gain (gap + uncertainty + cluster coverage + success potential)
- `coordinate`: cluster coordinate descent (outer loop polls clusters, inner loop picks boundary-near methods)
- `hybrid` (default): InfoGain for the first few rounds to build coverage, then switch to Coordinate for fine search

### ASR + FPR two-dimensional profile

- **ASR** (attack success rate): measures line strength
- **FPR** (false-positive rate): uses "safe twins" to test whether the model over-refuses legitimate requests

### Jailbreak Tax

A math problem is embedded in the attack prompt (answer on the last line as `[MATH:answer]`). If math reasoning degrades after a successful jailbreak, the model paid a capability cost to "comply".

- **Value lies in comparison**: each run measures the target's normal accuracy with a bare math probe (no attack content) as a baseline, then compares it to accuracy under attack; `accuracy_drop = baseline − under-attack` is the real jailbreak degradation. Reports and the dashboard present it as `baseline → under-attack (degraded x%)`.
- **Sentinel convention**: a record with `expected_answer: 0` means **this entry does not test the jailbreak tax** — it's recorded as `null` and doesn't affect scoring. For attack sets you bring yourself without a math problem, just keep `expected_answer: 0`.
- **Scoring**: `math_score` has three tiers (2=correct, 1=wrong, 0=missing format); on a successful jailbreak with a probe, `tax/2` is deducted from eval_score.
- **Note**: the probe is static text injected at generation time; after changing problem difficulty or the template you **must** regenerate the attack set (baseline measurement is live).

### HPO experiment framework (`experiments/`)

study.yaml-driven hyperparameter search that treats any knob in `params.py` as a factor (injected into the subprocess via the `LLMSEC_PARAM_<NAME>` env var):

- **Strategies**: grid / random / bayesian (optuna TPE)
- **Resumable**: `trials.jsonl` is the source of truth; completed configs are skipped automatically
- **Metric**: optimizes `conv_rounds` by default (rounds to convergence, lower is better), configurable

```bash
python -m llmsec.experiments run <study.yaml>     # run / resume
python -m llmsec.experiments report <name>        # best config + comparison table
python -m llmsec.experiments trials <name>        # list all trials
```

> 📚 Full details (study.yaml format, factor types, search strategies, metrics, isolation & reproducibility) in [docs/experiments.en.md](docs/experiments.en.md).

---

## Quick start

### 1. Install dependencies

```bash
pip install -r llmsec/requirements.txt
```

Python 3.11. `hdbscan`, `sentence-transformers`, and `tiktoken` are optional/lazy dependencies of the clustering module.

### 2. Configure the environment

Copy `llmsec/.env.example` to `llmsec/.env` and fill in the target and generator model config.

### 3. Three steps to run

```bash
# Step 1: generate an attack set (extracts L1 methods from llmsec/攻击分析.md)
python -m llmsec.attacks.generate --output output/attacks/l1.jsonl

# Step 2: adaptive attack + over-sensitivity + synthesis report (main entry)
python -m llmsec.pipeline.runner --input attacks/l1.jsonl --max-rounds 10 --batch-size 10

# Step 3: read the report
cat llmsec/output/runs/<timestamp>/security_report.md
```

**Offline test without a real LLM**:

```bash
# Terminal 1: start a local simulated model
python -m llmsec.server.local_model_server --port 8000

# Terminal 2: run runner in local_sim mode
TARGET_TYPE=local_sim TARGET_BASE_URL=http://127.0.0.1:8000/v1 \
  python -m llmsec.pipeline.runner --input attacks/l1.jsonl --max-rounds 5
```

---

## Attack sets

**Attack sets are just input consumables.** The generators under `llmsec/attacks/` (the `攻击分析.md` parser, the built-in HarmBench data wrapper) are only sample sources for testing and demonstration — you can bring your own attack set from any source.

> 📚 HarmBench attribution and license in [llmsec/data/Explication.en.md](llmsec/data/Explication.en.md).

Your own attack set only needs to follow the standard JSONL format (one entry per line):

```json
{"id": "unique-id", "method": "method-name", "category": "category", "harm_type": "harm-type", "prompt": "attack-text", "expected_answer": 0, "source": "custom-source", "functional_category": "standard"}
```

Field notes:

- `id`: unique identifier (suggest `method-name-index`)
- `method`: attack method name (the key tracked by clustering and ELO; for variants use `base_suffix` naming like `dan_style_rot13` — same base/suffix borrow each other's predictions)
- `category` / `category_name`: attack category (optional, default `unknown`)
- `harm_type`: harm type (e.g. `cybercrime`, `fraud`, `chemical_biological`)
- `prompt`: full attack text
- `expected_answer`: the jailbreak-tax math answer; set `0` when not using the tax
- `source`: source tag (optional, default `our`)
- `functional_category`: functional category (optional, default `standard`)

Drop it into `output/attacks/` and run directly:

```bash
python -m llmsec.pipeline.runner --input attacks/<your-file>.jsonl
```

---

## Command reference

### Attack generation

```bash
python -m llmsec.attacks.generate [--dry-run] [--only ID] [--start-from ID] [--output PATH]
    # Parses L1 attack methods from llmsec/攻击分析.md

python -m llmsec.attacks.harmbench [--max N] [--seed N] [--variants N] [--obfuscate]
                                   [--no-math-tax]
    # Generates a demo attack set from built-in HarmBench data;
    # default output output/attacks/harmbench_jailbreak.jsonl
    # Injects the jailbreak-tax math probe by default; --no-math-tax disables it
    # (for backends like PCAP replay that don't answer in format)
```

### Adaptive evaluation (main entry)

```bash
python -m llmsec.pipeline.runner [--phase {all,1,2}] [--input FILE] [--batch-size N]
                                 [--max-rounds N] [--twin-window N]
                                 [--sampler {gap,infogain,coordinate,hybrid}]
                                 [--sampler-alpha A] [--sampler-beta B] [--sampler-gamma G]
                                 [--coordinate-rounds R] [--target NAME]
```

- `--phase`: `all` (attack + sensitivity), `1` (attack only), `2` (sensitivity only)
- `--input`: attack-set path, relative to the `output/` directory
- `--target`: target model name (for multi-target scans)
- `--twin-window`: number of methods for sensitivity detection; auto-adapts to ELO-boundary confidence if unset
- `--sampler`: sampling strategy (see above)

### Experiment framework

```bash
python -m llmsec.experiments run <study.yaml>      # run / resume a study
python -m llmsec.experiments report <name>         # print best config + comparison table
python -m llmsec.experiments trials <name>         # list all trials
```

### Auxiliary commands

```bash
python -m llmsec.evaluation.evaluator [--input attacks/l1.jsonl] [--max-samples N] [--repeat N]
                                      [--only ID] [--start-from ID] [--no-judge]
    # Full evaluation: send each → Judge score → update ELO (no adaptive sampling)

python -m llmsec.evaluation.cluster_analysis [--defender NAME] [--output PATH]
    # Cluster-level security analysis from current ELO and clustering results
    # Includes SVD-Ridge model diagnostics: regularization path, optimal λ,
    # PCA, feature importance, prediction confidence intervals

python -m llmsec.evaluation.elo_cluster --status
    # Inspect the cluster-ELO predictor state

python -m llmsec.evaluation.safe_twin [--generate|--evaluate|--all]
    # Safe-twin generation and over-sensitivity (FPR) detection

python -m llmsec.clustering.cli [--input FILE] [--result-file FILE] [--dump-features]
    # Attack-method cluster analysis (HDBSCAN + key-layer auto-k);
    # --result-file provides eval results to enable weakly-supervised
    # feature weighting and ANOVA cluster-effect validation

python -m llmsec.reporting.report [--output-dir DIR]
    # Standalone report: scans *_结果.jsonl and the latest runs/ attack_results.jsonl

python -m llmsec.pipeline.launcher
    # Interactive launcher: pick an attack set and mode, then guide execution

python -m llmsec.pipeline.probe [--text "test text"]
    # Target API connectivity probe (routes by TARGET_TYPE)
```

### Tests

```bash
python -m tests.clustering_kdistance      # offline clustering-effect verification
python -m tests.test_whitened_tree        # damped whitened space / auto-k / D-optimal seed coverage
python -m tests.test_elo_convergence      # predict variant fallback & check_convergence anti-false-positive
python -m tests.test_svd_ridge            # SVD-Ridge batch-prediction accuracy / fallback / K-Fold / cache
python -m tests.test_blend_predictor      # Blend two-layer prediction + Bayesian shrinkage
python -m tests.test_elo_access           # R → derived-Elo cache fingerprint invalidation
python -m tests.test_p2_correctness       # ResultsMatrix correctness
python -m tests.test_p2_data_integrity    # data integrity (atomic write / corruption recovery)
python -m tests.test_dashboard_api        # Web panel API / task lifecycle
python -m tests.test_jailbreak_tax        # jailbreak-tax injection/scoring/sentinel guard
```

---

## Web panel (graphical workbench)

```bash
.venv/Scripts/uvicorn llmsec.server.dashboard_api:app --host 127.0.0.1 --port 8080
# Open http://localhost:8080
```

Six sections in the sidebar:

- **Overview**: security-level banner, ASR/FPR/boundary-ELO/confidence metric cards, jailbreak-tax mean card, five-dimensional security-profile radar, ASR by harm category
- **Threat board**: Top 10 threats, high-threat method table (measured/predicted badges + 95% CI + jailbreak-tax column), defender ELO convergence curve, unexpected blind spots
- **Report**: `security_report.md` rendered in sections, with in-section navigation
- **Cluster analysis**: validation metrics, PCA/t-SNE feature-space scatter (switchable), hierarchical cluster dendrogram (zoom to cut any layer, top-3 k presets), high-risk/blind-spot/stable cluster cards
- **Prediction model**: SVD-Ridge diagnostics — regularization path, PCA explained variance, feature importance, prediction Elo confidence intervals
- **Run control**: buttons to trigger attack-set generation / adaptive evaluation (param form) / cluster analysis, with task status and live log polling

---

## Configuration

### Behavioral parameters

> 🎛️ **To tune behavioral parameters (Elo K, convergence thresholds, sampler weights, clustering params, scoring weights, simulation params, …) edit `llmsec/params.py`** — the project's single tuning entry point, grouped by module, each param annotated with its purpose and review notes.
>
> You can also override any parameter via an env var: `LLMSEC_PARAM_<NAME>=value` (the HPO framework's injection point; supports bool/int/float/str type inference).

### Connection config (env vars)

| Variable | Description | Default |
|---|---|---|
| `TARGET_TYPE` | Target backend: `openai` / `local_sim` / `pcap_judge` | `openai` |
| `TARGET_API_KEY` | Target model API key | - |
| `TARGET_BASE_URL` | Target model URL | `https://api.deepseek.com/v1` |
| `TARGET_MODEL` | Target model name (under `pcap_judge` the defender name auto-uses `PCAP_MODEL_VERSION`) | `deepseek-v4-flash` |
| `TARGETS` | **Multi-target scan**: comma-separated names, with `TARGET_<N>_*` quadruples (NAME/TYPE/API_KEY/BASE_URL/MODEL) to mix multiple backends | - |
| `GENERATOR_API_KEY` | API key for attack-gen / safe-twin / report narrative | - |
| `GENERATOR_BASE_URL` | Generator model URL | `https://api.deepseek.com/v1` |
| `GENERATOR_MODEL` | Generator model name | `deepseek-v4-flash` |
| `JUDGE_MODEL` | Judge model name (falls back to GENERATOR) | `deepseek-v4-flash` |
| `EMBEDDING_MODEL` | Clustering semantic-embedding model | `all-MiniLM-L6-v2` |
| `HF_ENDPOINT` | HF mirror URL (when huggingface.co is unreachable) | `https://hf-mirror.com` |
| `SENTENCE_TRANSFORMERS_HOME` | embedding model cache dir (in-project; download once, work offline) | `llmsec/.models` |
| `EMBEDDING_API_BASE/KEY/MODEL` | Optional: OpenAI-compatible API embedding fallback | - |
| `PCAP_JUDGE_URL` | PCAP Judge URL (when TARGET_TYPE=pcap_judge) | - |

Full template in `llmsec/.env.example` (copy to `.env`).

Embedding fallback chain: local cache → HF mirror → API embedding → TF-IDF. After the model downloads once via the mirror it's cached under `llmsec/.models/` and works fully offline.

---

## Directory structure

```
llmsec/
├── params.py     # unified tuning entry (LLMSEC_PARAM_* env override)
├── core/         # config(.env/paths/multi-target) / results(R matrix) / io(atomic write) /
│                 # llm(retry) / text(jailbreak tax) / logging / seed
├── targets/      # target backend routing: openai / local_sim / pcap_judge (TARGET_TYPE)
├── evaluation/   # evaluator(full eval) / judge(LLM scoring) / elo(two-sided ELO + CI convergence + derive_elo)
│                 # elo_cluster(SVD-Ridge) / blend_predictor(unified+model Bayesian shrinkage)
│                 # elo_access(R-derived Elo cache layer) / active_learning(D-optimal seeds)
│                 # samplers(gap/infogain/coordinate/hybrid) / safe_twin(sensitivity detection)
│                 # cluster_analysis(cluster-level analysis + model diagnostics)
├── attacks/      # demo attack-set generation (optional, non-core): generate(L1) / harmbench(built-in data)
├── data/         # built-in attack data (HarmBench behavior library + jailbreak templates; see Explication)
├── pipeline/     # runner(adaptive orchestration) / launcher(interactive) / probe(connectivity)
├── reporting/    # report(five-dimensional tree profile + LLM narrative + method registry)
├── clustering/   # space(whitened metric space) / hdb(HDBSCAN) / tree(key-layer auto-k) /
│                 # features / posterior / pipeline / cli
├── experiments/  # HPO framework: study/executor/search(grid/random/bayesian)/metrics/schema
└── server/       # dashboard_api(web panel) / local_model_server(local simulated model, OpenAI-compatible)
```

---

## Output layout

```
llmsec/output/
├── attacks/                # attack sets (l1.jsonl, harmbench_jailbreak.jsonl)
├── state/                  # persistent state
│   ├── results.json        #   ★ R matrix (single source of truth, multi-model)
│   ├── elo_cache.json      #   derived Elo cache (deletable/rebuildable; invalidates per model-column fingerprint)
│   ├── state.json          #   legacy snapshot backup (optional; migrated into R on upgrade)
│   └── safe_twins.jsonl    #   safe-twin set
├── predictors/             # unified/per-model Ridge predictors (BlendPredictor-derived)
├── runs/<timestamp>/       # single runner-run artifacts
│   ├── attack_results.jsonl      # attack details (incl. raw responses)
│   ├── runner_report.json        # synthesis report
│   ├── allergy.json              # sensitivity report + 2D profile
│   ├── sampler_log.jsonl         # per-round sampler decision log
│   ├── cluster_security_analysis.json  # cluster-level security analysis + SVD-Ridge model diagnostics
│   ├── security_tree.json        # five-dimensional tree profile
│   └── security_report.md        # LLM narrative report (final deliverable)
├── experiments/<name>/     # HPO study: study.yaml / trials.jsonl / best.json
├── feature_cache.pkl       # prior-feature cache (written by elo_cluster)
├── cluster_result.pkl      # full clustering artifacts (written by hdb)
├── {input-name}_结果.jsonl  # evaluator per-entry results
├── {input-name}_汇总.json   # evaluator summary stats
├── method_registry.json    # method registry (ELO + cluster label + prompt list)
├── cluster_report.json     # clustering report
├── cluster_matrix.csv      # method × feature matrix
└── cluster_features.json   # features exported by --dump-features
```

---

## License

GPL
