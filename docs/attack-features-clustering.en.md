# Attack Features & Clustering — Concise English Summary

> English (concise summary) | [中文 (完整版)](攻击特征与聚类深度研究报告.md)

> This is a condensed English summary of the full Chinese report. The full version (with formulas, audit tables, and real-run data) is authoritative; in case of any discrepancy, the **code** is the ground truth (baseline `c55b678`).

---

## 1. Three first principles

The whole system is built around three principles; every design trade-off traces back to them:

1. **Prior metrics, posterior forbidden**: the metric space for clustering and prediction uses only *prior* features (information available from the attack text itself). Posterior features (how the machine responded) never enter any distance computation or model training — they are used only for cluster profiling and cluster-effect validation.
2. **Clustering is post-hoc**: clustering runs only after the entire test flow. There is no "pre-clustering" during testing — it wastes compute and forces immature structure onto sampling.
3. **Matrixized batching**: Elo prediction for unmeasured methods goes from "per-method distance calc" to "one matrix multiply" (SVD-Ridge); training cost is incurred only when the ground truth changes.

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
- **SVD solve**: `X = UΣVᵀ`, `w(λ) = V·diag(σᵢ/(σᵢ²+λ))·Uᵀy_c` (near-zero singular values truncated).
- **λ selection**: K-Fold (K=5) over `logspace(-3, 4, 24)` picks the min-CV-error λ; σ² taken from CV residuals at λ* (out-of-sample, more honest than training residuals).
- **Three-level cache**: GT fingerprint unchanged ⇒ reuse `w`, pure matrix multiply; GT growth < threshold(10) ⇒ single SVD refit with the existing λ*; growth ≥ threshold ⇒ rerun K-Fold.
- **MAP uncertainty**: Ridge ≡ Gaussian-prior Bayesian MAP; predicted mean `E = y_mean + X_test·w`, variance `σ²·diag(X_test(XᵀX+λI)⁻¹X_testᵀ)`, 95% CI persisted. Wide-CI predictions show as low-confidence badges on the dashboard.

> Note: the full Chinese report's §3.3 describes the *legacy* 4-weight convergence system; the code has since moved to the single-CI criterion described in the README. (Per the report's own note: the code is ground truth.)

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
- **Known limits**: λ* fluctuates across retraining during GT growth (normal); family-correlated batches can alias convergence (mitigable by raising the window or enforcing suffix diversity in batches); density clustering has a noise-rate floor in concentrated spaces — use the frontend tree scaling as the structural truth.

---

## 7. Improvement directions

1. Stronger embedding (API embedding or a larger model) to push p_anova past 0.05.
2. If ANOVA stays non-significant under stronger embedding, that's the answer about feature abstraction itself: use reaction-related signals (judge dims, response behavior) as semi-supervised embedding targets, or train a "reaction-aware" text encoder via contrastive learning.
3. Convergence window & family aliasing: enforce suffix/family diversity within a batch, or raise the window to 5.
