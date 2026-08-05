# Attack Features & Clustering — Concise English Summary

> English (concise summary) | [中文 (完整版)](攻击特征与聚类深度研究报告.md)

> This is a condensed English summary of the full Chinese report. The full version (with formulas, audit tables, and real-run data) is authoritative; in case of any discrepancy, the **code** is the ground truth. Updated 2026-08-04 to add the R-matrix architecture, Blend two-layer predictor, K-dynamics, and the CI convergence criterion.

---

## 1. Four first principles

The whole system is built around four principles; every design trade-off traces back to them:

1. **R-matrix is the single source of truth**: the result matrix `R[method][model] = MatchResult` is the only non-recomputable observation (paid for in API cost). Elo, predictors, and convergence are all *derived* caches from R + method features X, fully recomputable at any time — losing a cache is harmless. This guarantees: Elo doesn't cross-contaminate across models (each model's Elo is replayed from its own column in R), multi-model is native (R's 2nd dimension is the model), and full rebuildability.
2. **Prior metrics, posterior forbidden**: the metric space for clustering and prediction uses only *prior* features (information available from the attack text itself). Posterior features (how the machine responded) never enter any distance computation or model training — they are used only for cluster profiling and cluster-effect validation.
3. **Clustering is post-hoc**: clustering runs only after the entire test flow. There is no "pre-clustering" during testing — it wastes compute and forces immature structure onto sampling.
4. **Matrixized batching**: Elo prediction for unmeasured methods goes from "per-method distance calc" to "one matrix multiply" (SVD-Ridge); training cost is incurred only when the ground truth changes.

---

## 2. Feature system (7 blocks)

Extracted by `clustering/features.py: extract_all_features()`, outputting `{method: {textual, embedding, technique, intent, prior, defense, cross_model}}`. **5 blocks (the `PRIOR_BLOCKS`: textual, embedding, technique, intent, prior) enter the clustering metric; defense and cross_model do not.**

| Block | Dim | Type | Notes |
|---|---|---|---|
| `textual` | 12 | prior | Text-structure stats (length, token count, base64/hex flags, multilingual, roleplay, output-format control, …). Math-tax tail is stripped first. |
| `embedding` | PCA-reduced | prior | `all-MiniLM-L6-v2` → PCA to `min(50, n//3, n-1)`; offline fallback to TF-IDF (200-d, 1-2gram) → same PCA dims. |
| `technique` | dynamic | prior | 28 regex technique labels (low-resource language, encoding obfuscation, role-play, DAN family, …) + dynamic `harm:<type>` / `cat:<category>` tags. Binary. |
| `intent` | 3 | prior | `semantic_drift`, `typo_ratio`, `filler_ratio`. |
| `prior` | 9 | prior | Derivable with zero evaluation: method-name length, variant-suffix one-hot (rot13/b64/code/story/numeric), math-tax flag, prompt line-count log. |
| `defense` | 14 | **posterior** | Judge dims, compliance-grade distribution, status distribution, response length, … All-zero when no eval data. **Only used for cluster-profile means & CSV export.** |
| `cross_model` | — | placeholder | Empty; reserved for multi-model evaluation. |

---

## 3. Test-time training pipeline

### 3.1 Cold start: feature cache + D-Optimality seeds

- **Feature cache** (`ClusterEloPredictor.fit_features`): extract features for all attack methods. **No clustering here** — it's just the data basis for Ridge and D-optimal. Skipped entirely when the attack set is unchanged (`method_set_hash`).
- **D-Optimality seeds** (`evaluation/active_learning.py`): with empty ground truth, frame seed selection as optimal experimental design. Ridge's information matrix `M = X_gtᵀX_gt + λI`; a candidate's prediction variance (leverage) is `xᵀM⁻¹x`. Greedily pick the max-leverage point, then rank-1 update the inverse via Sherman–Morrison (exact, O(d²) per step, no re-inversion). With empty GT, `M = λI` ⇒ degrades to picking the largest, most spread-out points — natural feature-space coverage. Seed count `max(5, ceil(log2 n))`.

### 3.2 SVD-Ridge prediction matrix

`evaluation/elo_cluster.py: EloPredictorModel`:

- **Features**: the 5 prior blocks concatenated for GT methods.
- **Standardize + intercept**: column-wise Z-score on X; **center y and store `y_mean`** (a past severe bug — without the intercept, predicted means collapsed to ≈0).
- **Degenerate-column mask** (`RIDGE_DEGENERATE_COL_EPS=1e-4`): a column near-constant within the GT subset (std hitting the 1e-8 floor) would produce ~1e8 standardized values for unmeasured methods, exploding the MAP variance. The `col_keep` mask zeroes these columns after standardization (mean unaffected: w=0 in that direction). The threshold is far below any legitimate embedding column's std (≥0.017).
- **SVD solve**: `X = UΣVᵀ`, `w(λ) = V·diag(σᵢ/(σᵢ²+λ))·Uᵀy_c` (near-zero singular values truncated).
- **λ selection**: K-Fold (K=5) over `logspace(-3, 4, 24)` picks the min-CV-error λ; σ² taken from CV residuals at λ* (out-of-sample, more honest than training residuals). **H-9 fix**: fold remainders are now balanced across the first r folds (the old code dumped all leftovers into the last fold, so n=9,k=5 gave fold sizes 1,1,1,1,5 — badly unbalanced CV that biased λ* high on small GT).
- **Three-level cache**: GT fingerprint unchanged ⇒ reuse `w`, pure matrix multiply; GT growth < threshold(10) ⇒ single SVD refit with the existing λ* (σ² retained from last CV, avoiding the false confidence of in-sample residuals → 0); growth ≥ threshold ⇒ rerun K-Fold.
- **MAP uncertainty**: Ridge ≡ Gaussian-prior Bayesian MAP (λ=σ²/τ²); predicted mean `E = y_mean + X_test·w`, variance `σ²·(1 + diag(X_test(XᵀX+λI)⁻¹X_testᵀ))` — irreducible noise σ² plus parameter uncertainty (the leverage `x'(XᵀX+λI)⁻¹x`, same quantity as D-optimality). 95% CI persisted; std is capped at `min(3·GT Elo std, 200)` to protect all downstream from variance anomalies. Wide-CI predictions show as low-confidence badges on the dashboard.

> Note: the full Chinese report covers the training pipeline in detail (§3.3–3.6).

### 3.3 Blend two-layer prediction (unified + per-model, Bayesian shrinkage)

`evaluation/blend_predictor.py: BlendPredictor` — stacks a "unified + per-model" two-layer structure on top of single-layer SVD-Ridge to solve the cold-start problem for new models. Both layers are `EloPredictorModel` (SVD-Ridge); they differ only in training data:

| Layer | Training data | Captures |
|---|---|---|
| **Unified Pu** | Pools **all models'** (method, elo) as independent samples (synthetic key `method#model`, same method features) | "Intrinsic method threat" — strong jailbreaks are strong against most models. **The only prior source for a brand-new model's cold start.** |
| **Per-model Pm** | Only that model's column (Elo derived via `derive_elo(R, model)`) | Model-specific weaknesses |

**Adaptive blend weight = empirical Bayes shrinkage**: `pred = w_u·Pu + w_m·Pm`, where `w_m = n_model/(n_model + K)`, `K = 10`. Few samples (n→0) ⇒ w_m→0, rely entirely on the unified prior (shrink toward the group mean); many samples (n→∞) ⇒ w_m→1, trust the model's own prediction. This implements "0.5+0.5 → 0.7+0.3" naturally with no manual tuning — the weight grows with evidence. Prediction variance is weighted by squared weights: `var = w_u²·var_u + w_m²·var_m`.

**Triple-fingerprint cache** (`load_or_fit_blend_predictor`): reuse requires all three unchanged — R content fingerprint + method-catalog fingerprint + **features-structure signature** (H-11 fix: detects embedding swap / TF-IDF fallback / feature-code changes; stale features pickled into the cache would silently use an old feature space). Cached under `output/predictors/blend_<fp>.pkl`.

### 3.4 Elo scoring: K-dynamics + continuous performance mapping

`evaluation/elo.py: ELOTracker.update` — the Elo update magnitude is not a fixed constant but adapts to the participant's role and the score magnitude ("K-dynamics"), designed to cure early Elo oscillation.

**Continuous performance mapping** (`SCORE_PERF_TAU=2.0`): moves the score magnitude out of the K-factor and into the "result term" via the saturating function `perf = score/(score+τ)` (when score>0; else perf=0). score=1→0.33, 2→0.50, 3→0.60, 5→0.71 — monotonic and bounded, replacing the old `K·(1+score/2)` amplification (which sent K soaring on strong scores, causing oscillation). The update becomes `delta = K·(perf − E)`; since `perf∈[0,1)`, single-match delta is naturally capped by K.

**Defender K decay** (`K_DEF_DECAY_N0=10`): in evaluation, each attacker method is measured only 1–2 times (keep full K=32 to locate its level fast), while the defender plays every match (match count accumulates rapidly). Defender K decays with cumulative matches: `K_def = K / sqrt(max(1, n_def/N0))`. The first N0 matches are a "warm-up period" (K_def=K, no decay); afterward: n=10→32, n=40→16, n=90→10.7. Using 1/√n (not 1/n) balances "increasingly stable" with "still responsive to new-method shocks" (the classic standard-error rate). Paired with a `MAX_DELTA_PER_UPDATE=40` hard cap as a safety net.

### 3.5 Convergence criterion: drift + noise decomposition → single CI

`evaluation/elo.py: check_convergence` — **replaces the legacy "std/rel_std thresholds + 4-weight confidence" system**. Core idea: separate the defender Elo trajectory's "drift" (systematic movement toward the true value — a good thing) from "noise" (random jitter).

Each round, `record_round_end` logs the defender Elo, forming a per-round trajectory. `_trajectory_stats` fits an OLS line:

- **drift** (slope, Elo/round): systematic trend;
- **noise**: std of the detrended residuals;
- **Autocorrelation-corrected effective sample size**: `k_eff = m·(1−ρ)/(1+ρ)` (Bartlett, ρ = residual lag-1 autocorrelation); stronger positive correlation ⇒ fewer effective samples ⇒ wider CI (conservative);
- **Synthesized true-Elo 95% CI half-width**: `ci_half = 1.96·noise/√k_eff`.

Converged iff **all four hold** (`rounds_sufficient and ci_ok and drift_ok and coverage_ok`):

| Condition | Threshold | Meaning |
|---|---|---|
| `rounds_sufficient` | `n_rounds ≥ CONV_WINDOW_MIN = 4` | Enough rounds to judge |
| `ci_ok` | `ci_half < CONV_CI_TARGET = 20.0` Elo | True-level estimate precise enough |
| `drift_ok` | `abs(drift) < CONV_DRIFT_TARGET = 5.0` Elo/round | No longer moving systematically |
| `coverage_ok` | `coverage ≥ 0.20` or `tested ≥ 20` | Tested enough — independent of Elo stability |

**Key detail — drift uses a recent window, noise uses the full trajectory**: drift takes the local slope of the last `CONV_WINDOW_MIN` rounds (the full-window slope is permanently dragged up by the early rapid rise, misjudging a stable trajectory as still drifting); noise/ci_half still uses the full trajectory (more data ⇒ stabler noise estimate).

False-positive resistance: ① drift/noise separation (systematic convergence trend is not penalized); ② recent-window drift (avoids early-rise bias); ③ autocorrelation correction (conservative); ④ coverage as an independent gate; ⑤ multi-condition AND.

---

## 4. Post-test clustering pipeline

Entry: `ClusterEloPredictor.final_fit(...)` → `clustering/hdb.py`.

### 4.1 Weakly-supervised feature weighting

`clustering/posterior.py`: `w_j = |pearson(X_j, y)|` (normalized, clipped to [0.2, 5]). `y` = each measured method's real reaction. **Computed only on ground truth** — letting ridge *predictions* of unmeasured methods participate would create circular reasoning (predictions are a linear combo of these features). Relevant directions are amplified, pulling same-reaction points together.

### 4.2 Metric space: z-score + SVD truncation (no whitening)

`clustering/space.py: build_whitened_space` (name kept; default `damp=0`): Z-score standardize → SVD → take the top-k PCs (95% variance ∧ spectral-knee × 2 ∧ 50 dims) → `coords = U_k · σ/√n` (standard PCA projection).

**Why no whitening** (2026-07-27 correction, overturning an earlier "damped whitening damp=0.5" conclusion): whitening doesn't delete signal — it redistributes each dimension's *voting weight* in Euclidean distance (∝ that dimension's variance). Strong cluster separation necessarily creates large total variance in high-variance directions, so "high variance" is almost a proxy for "contains cluster signal"; whitening averages that weight over to noise directions. Whitening/Mahalanobis targets outlier detection (assumes all variance is background scatter); clustering wants the opposite. Measured: silhouette 0.028 (damp=0.5) vs **0.076 (damp=0)**.

### 4.3 Dual-tree division of labor

Real-data lesson: HDBSCAN's `single_linkage_tree_` produced a degenerate cut (127/132 single-cluster + singletons) under nearest-neighbor chaining. So a dual-tree split is used:

- **Density view (HDBSCAN flat)**: a few tight density clusters + honestly-labeled "sparse-region" noise.
- **Scaling tree (Ward linkage)**: the same whitened coords → Ward hierarchical tree. Scaling / key-layer / cluster-effect analysis all run on this tree. Main `method_labels` = the Ward cut at the key layer k*.

**Key-layer auto-k** (`clustering/tree.py`, algorithm-agnostic): candidate k centered on `k0 = clamp(ceil(log2 n), 4, 20)` at log spacing; for each k cut the tree and compute silhouette / Calinski-Harabasz / Davies-Bouldin, normalize (DB inverted), combine, take the **global argmax**; top-3 k kept as frontend presets.

### 4.4 Cluster-effect validation (ANOVA / Kruskal-Wallis)

`posterior.reaction_validation` answers "did our features capture what the machine cares about?": per-cluster mean eval_score (GT only) → `f_oneway` + `kruskal` + effect size eta²/ε². Three-level verdict: effective (p<0.05 ∧ effect>0.1) / weakly correlated / "feature abstraction needs an upgrade". All posterior stats are isolated in `posterior.py` (code hygiene).

---

## 5. Posterior-leakage audit — conclusion: no leakage

defense (posterior) is "stored but not measured" — present only in 4 non-metric positions: feature extraction/dict, cluster-profile means, CSV export columns, and zero-padding for unmeasured methods. Every metric/training/projection path is verified to use only `PRIOR_BLOCKS`. Consistent with the "posterior is for profiling & validation only" architectural contract.

---

## 6. Real-run highlights & limitations

- **Run 2026-07-27** (132 methods, PCAP Judge Qwen3.5-9B, TF-IDF fallback): 8 D-optimal seeds → 18-round convergence (ASR 32.8%, FPR 12.5%, boundary Elo 1534, verdict BROKEN). Cluster validation: p_anova=0.20, eta²=0.01 ⇒ "features don't match what the machine cares about".
- **TF-IDF vs semantic embedding** (same data): embedding lifted silhouette ~0.05 → **0.376** and eta² 0.012 → **0.151** — embedding quality was the main ANOVA bottleneck, not the feature design.
- **Metric-space fix** (prior injection + damp=0, on the same embedding data): silhouette 0.023 → **0.079**, ANOVA p/eta² 0.060/0.16 → **0.0/0.545** — the verdict flipped from "needs upgrade" to **"features effective"**.
- **Known limits**: λ* fluctuates across retraining during GT growth (normal); family-correlated batches can alias the drift/noise estimates (the current CI criterion with its recent-window drift + full-trajectory noise is more robust to this; mitigable by enforcing suffix diversity within a batch); density clustering has a noise-rate floor in concentrated spaces — use the frontend tree scaling as the structural truth.

---

## 7. Improvement directions

1. Stronger embedding (API embedding or a larger model) to push p_anova past 0.05.
2. If ANOVA stays non-significant under stronger embedding, that's the answer about feature abstraction itself: use reaction-related signals (judge dims, response behavior) as semi-supervised embedding targets, or train a "reaction-aware" text encoder via contrastive learning.
3. Convergence & family aliasing: enforce suffix/family diversity within a batch (the CI criterion is already more robust).
