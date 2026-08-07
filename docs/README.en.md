# LLM API Security Assessment System

> [中文](../README.md) | English

A systematic black-box LLM security assessment framework: adaptive attack testing → ELO threat ranking → SVD-Ridge / Blend prediction → allergy detection → cluster analysis → human-readable Markdown security report.

> This project evaluates **the assessment pipeline itself** (adaptive sampling, threat ranking, capability prediction, allergy detection, cluster analysis). Attack sets are consumable inputs.

---

## What It Does

```mermaid
graph LR
    A[Attack Set JSONL] --> B[Phase 1 Adaptive Attack]
    B --> C[ELO-driven Per-round Sampling]
    C --> D[Phase 2 Allergy Detection]
    D --> E[Phase 3 Comprehensive Report]
    E --> F[security_report.md<br/>+ Web Dashboard]
```

- **Phase 1 Attack**: Reverse ELO drives per-round attack method selection. Judge scores update ELO in real-time until the defender's true Elo 95% CI converges.
- **Phase 2 Allergy Detection**: Near the ELO boundary, "safe twins" (semantically safe, structurally similar prompts) test whether the model over-blocks legitimate requests (FPR).
- **Phase 3 Report**: Combines ASR (attack success rate) + FPR (false positive rate) + ELO boundary into a quantitative security profile.

---

## Quick Start

### Docker (one-line startup, zero install, zero config)

```bash
# Full version (clustering + pre-cached embedding model, ~3GB)
docker run -p 8080:8080 -v llmsec-data:/app/output jinxiahz/llmsec

# Slim version (no clustering/torch, attack evaluation only, ~500MB)
docker run -p 8080:8080 -v llmsec-data:/app/output jinxiahz/llmsec:slim
```

Open `http://localhost:8080` in your browser and configure via the dashboard — no manual `.env` editing needed.

### pip install

```bash
pip install -e .              # Core (no clustering, no torch)
pip install -e ".[cluster]"   # Full (clustering + embedding model)
pip install -e ".[dev]"       # Development (tests + lint)
```

Python 3.11 required.

### Run

```bash
# Generate attack set
python -m llmsec.attacks.generate --output attacks/l1.jsonl

# Adaptive evaluation (main entry point)
python -m llmsec.pipeline.runner --input attacks/l1.jsonl --max-rounds 10

# View report
cat output/runs/<timestamp>/security_report.md
```

---

## Web Dashboard

```bash
# Docker: already running at localhost:8080
# pip: 
python -m uvicorn llmsec.server.dashboard_api:app --host 127.0.0.1 --port 8080
```

Sidebar sections: Overview, Threat Board, Report, Clustering, Prediction Model, Run Control (adaptive evaluation / HPO / target management / environment config).

---

## Core Concepts

### Reverse ELO + K Dynamics

Attack methods are the offense; target models are the defense. Successful attacks raise the method's ELO; blocked attacks lower it. The defender ELO is the "security boundary".

- **Continuous score mapping**: `perf = score/(score+τ)` (saturating)
- **K decay**: Attackers use full K; defenders decay `K_def = K / sqrt(max(1, n/N0))`
- **Synchronous round update (Model B)**: All attacks in a round use the round-start snapshot; defender aggregates with √N scaling

### CI-based Convergence

Defender Elo trajectory decomposed into drift (toward true value) and noise (random jitter). Convergence = CI half-width + drift + coverage all within targets. Drift threshold auto-relaxes when CI is very tight.

### Blend Dual-Layer Prediction (Cold Start)

- **Universal predictor P_u**: Pooled across all models, captures "method intrinsic threat"
- **Model predictor P_m**: Per-model column, captures model-specific weaknesses
- **Blend**: `pred = w_u·P_u + w_m·P_m`, where `w_m = n/(n+K)` (Bayesian shrinkage)

### Discovery Layer: Probe Fingerprint + Similarity Transfer

Cold-start D-optimal sentinel seeds → per-seed Elo vector = model "defense fingerprint". Cross-model correlation → similarity-weighted pooling (borrow from similar donors, not uniform average).

### Data Storage & Reproducibility

The result matrix **R** (`R[method][model] = MatchResult`) is the single source of truth. All derived data (Elo, predictors, convergence) can be fully recomputed from R + method features X.

---

## Configuration

Key environment variables (see `.env.example` for full template):

| Variable | Description | Default |
|---|---|---|
| `TARGET_TYPE` | Target backend: `openai` / `local_sim` / `pcap_judge` | `openai` |
| `TARGET_API_KEY` | Target model API Key | - |
| `TARGET_BASE_URL` | Target model URL | `https://api.deepseek.com/v1` |
| `TARGET_MODEL` | Target model name | `deepseek-v4-flash` |
| `TARGETS` | Multi-target scan: comma-separated names + `TARGET_<N>_*` | - |
| `GENERATOR_API_KEY` | Attack generation / safe twin / report API Key | - |
| `GENERATOR_MODEL` | Generator model name | `deepseek-v4-flash` |

Behavior parameters: `llmsec/params.py` (override via `LLMSEC_PARAM_<NAME>=value`).

---

## License

GPL v3
