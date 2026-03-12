# Instagram Reach Intelligence Engine
## Technical Architecture Specification

---

## 1. System Architecture

```
Synthetic Engine  (accounts · clusters · posts · velocity)
      ↓
Feature Pipeline  (baseline · account features · content features)
      ↓
Stage-1 Survival Model  (pre-post prediction)
      ↓
Evaluation  (walk-forward · calibration · segments · thresholds)
      ↓
Velocity Simulation  (1h · 3h · 6h engagement curves)
      ↓
Stage-2 Velocity Correction Model  (posterior update)
      ↓
Prior vs Posterior Analysis
```

---

## 2. Synthetic Engine Design

### 2.1 Account Class

**Attributes:**

- `account_id`
- `follower_count`
- `baseline_reach_ratio`  ← random-walks daily in simulation
- `quality_mean`
- `quality_variance`
- `velocity_sensitivity`
- `decay_rate`
- `volatility_factor`  ← random-walks daily in simulation
- `cluster_distribution`

### 2.2 Cluster Class

**Attributes:**

- `cluster_id`
- `performance_multiplier`
- `engagement_rate_multipliers`
- `early_burst_bias`
- `tier`  ← strong / medium / weak (4 / 8 / 8 split)

### 2.3 Simulation Loop

Time-progressive iteration over all accounts per day:

```python
for day in simulation_days:
    step_baseline_and_volatility_random_walk()
    for account in accounts:
        if account_should_post():
            generate_post()
```

### 2.4 Reach Computation

```
reach = follower_count
      × current_baseline_ratio   (drifting)
      × content_quality
      × cluster.performance_multiplier
      × post_noise                (mean-corrected lognormal)
      × algorithmic_factor        (20% suppressed · 20% viral · 60% normal)
```

The **mean-corrected lognormal** ensures `E[post_noise] = 1` regardless of sigma, so variance increases without inflating the mean. The **discrete algorithmic events** model Instagram's irreducible unpredictability.

---

## 3. Rolling Baseline

Exponential decay weighting with global λ:

```
weight = exp(-λ * age_in_days)
baseline = weighted_median(past_reaches, weights)
```

Computed strictly over past posts — no future leakage. With daily baseline drift (σ=0.12) and λ=0.03 (half-life ≈ 23 days), the rolling median is always a lagging estimate of the true current threshold, introducing realistic prediction uncertainty.

---

## 4. Feature Engineering

### Account Features

- `rolling_weighted_median`  — exponential decay baseline (the survival threshold proxy)
- `rolling_volatility`  — std of recent reaches (noise level estimate)
- `posting_frequency`  — posts in the past 14 days
- `cluster_entropy`  — Shannon entropy of cluster usage over last 20 posts

### Content Features

- `cluster_id` (categorical)
- `posting_time_bucket`  — 0=night, 1=morning, 2=afternoon, 3=evening
- `content_quality`  — sampled per post from account's quality distribution

### Excluded (Phase 1)

- Early velocity features (likes/comments in first 1–3 hours) — reserved for Stage-2

---

## 5. Survival Label

```
survived = 1  if  reach_24h > rolling_weighted_median
survived = 0  otherwise
```

First post per account is dropped (no baseline available). Target balance: ~50/50.

---

## 6. Modeling

**Library:** LightGBM

**Stage-1 config:**

```python
objective = "binary"
metric = "binary_logloss"
num_leaves = 31
learning_rate = 0.05
n_estimators = 300
categorical_feature = ["cluster_id"]
early_stopping_rounds = 30
```

**Walk-forward config:** same but `n_estimators = 100` (no early stopping — no val set per window).

---

## 7. Data Splitting

### Primary split (Stage-1 training)

Chronological — no randomization:

| Split      | Proportion |
|------------|------------|
| Train      | 70%        |
| Validation | 15%        |
| Test       | 15%        |

### Walk-forward validation

Roll forward in 30-day windows with minimum 90 days of training history. Each window trains on all prior data and tests on the next 30 days. Tests temporal stability of model quality.

---

## 8. Phase 2: Velocity Simulation & Stage-2 Model

### 8.1 Velocity Simulation

For each post, early engagement at 1h, 3h, 6h is generated using a cumulative exponential model:

```
F(t) = 1 - exp(-λ × t)      where λ = burst_bias × 0.5
```

`burst_bias` is drawn per post from the cluster tier: strong clusters (0.6–0.9), medium (0.4–0.7), weak (0.2–0.5). Higher burst_bias = more front-loaded engagement. Per-observation lognormal noise is applied with decreasing sigma (0.25 → 0.18 → 0.12) as the sample stabilizes.

### 8.2 Stage-2 Features

| Feature | Description |
|---|---|
| `stage1_prior` | Stage-1 predicted survival probability |
| `norm_likes_1h/3h/6h` | Cumulative likes normalized by `rolling_weighted_median` |
| `like_velocity_1to3` | Average likes/hour from 1h to 3h, normalized |
| `like_velocity_3to6` | Average likes/hour from 3h to 6h, normalized |
| `burst_ratio` | `likes_1h / likes_3h` — how front-loaded is engagement |
| `comment_ratio_1h` | `comments_1h / likes_1h` — community depth signal |
| `on_track_score` | Implied 24h reach from 1h velocity vs. account baseline |

### 8.3 Stage-2 Data Split

Stage-2 is trained only on Stage-1 **out-of-sample** predictions to prevent the correction model from learning to trust an overfit prior:

| Set | Source | Posts |
|-----|--------|-------|
| Stage-2 train | First 70% of Stage-1 val | ~1,565 |
| Stage-2 val   | Last 30% of Stage-1 val  | ~671   |
| Stage-2 test  | Stage-1 test set         | ~2,237 |

---

## 9. Observed Metrics (Phase 1 + Phase 2)

**Stage-1 (pre-post, no velocity):**

| Metric                  | Observed     | Notes                                   |
|-------------------------|--------------|-----------------------------------------|
| ROC-AUC (test)          | 0.835        | Above 0.70–0.80 target; acceptable      |
| Log loss (test)         | 0.500        | Reasonable                              |
| Walk-forward AUC mean   | 0.851        | std=0.010 — temporally stable           |
| Survival rate           | 0.501        | Target ~0.50 ✓                          |
| ECE (calibration)       | 0.020        | Well-calibrated                         |

**Stage-2 (post-live, with velocity):**

| Metric                         | Observed | Notes |
|--------------------------------|----------|-------|
| ROC-AUC (test)                 | 0.987    | +0.152 AUC lift over Stage-1 |
| Log loss (test)                | 0.139    | Large improvement from velocity signal |
| Corrections (S1 wrong → S2 right) | 510   | Stage-2 catches Stage-1 errors |
| Regressions (S1 right → S2 wrong) | 83    | Small regression cost |

**Top Stage-2 features:** `stage1_prior` (dominant) → `norm_likes_6h` → velocity rates → `on_track_score`

**Feature importance order (actual):**

1. `rolling_volatility`
2. `rolling_weighted_median`
3. `content_quality`
4. `cluster_entropy`
5. `cluster_id`
6. `posting_frequency`
7. `posting_time_bucket`

**Note on AUC floor:** With `rolling_weighted_median` (the survival threshold) as a direct feature alongside `rolling_volatility` (the noise level proxy), the model has near-complete distributional information. This creates an information floor at ~0.83 AUC that cannot be broken without removing features from the spec. This is correct behavior — Phase 2 velocity features will add real-time signal that further improves predictions.

---

## 9. Deep Analysis Modules

### 9.1 Calibration

Expected Calibration Error (ECE): weighted mean absolute deviation between predicted probability bins and actual survival rates. ECE = 0.020 — well-calibrated (threshold: < 0.05).

### 9.2 Per-Segment AUC

Breakdown by account size (nano/micro/macro) and cluster tier (strong/medium/weak). Reveals whether model generalizes across creator types.

| Segment              | AUC    |
|----------------------|--------|
| Macro (>100k)        | 0.875  |
| Nano (<10k)          | 0.834  |
| Micro (10k–100k)     | 0.799  |
| Cluster weak         | 0.843  |
| Cluster medium       | 0.836  |
| Cluster strong       | 0.820  |

### 9.3 Threshold Analysis

Recommended operating point for a creator-facing product: **threshold = 0.35** (best F1 = 0.768, precision = 0.689, recall = 0.868).

For a high-precision internal tool: **threshold = 0.70** (precision = 0.834, recall = 0.542).

---

## 10. Code Structure

```
ml_engine/
│
├── synthetic/
│   ├── account.py              — Account dataclass + generator
│   ├── cluster.py              — Cluster dataclass + generator
│   ├── simulator.py            — Day-by-day simulation, reach computation
│   └── velocity_simulator.py  — Early engagement at 1h/3h/6h  [Phase 2]
│
├── features/
│   ├── baseline.py             — Rolling weighted median (no leakage)
│   ├── feature_pipeline.py     — Stage-1 feature set (7 features)
│   └── velocity_features.py   — Stage-2 velocity features (9 features)  [Phase 2]
│
├── models/
│   ├── stage1.py               — LightGBM Stage-1 training
│   ├── stage2.py              — LightGBM Stage-2 training + comparison  [Phase 2]
│   ├── evaluator.py            — Metrics, feature importance, diagnostics
│   ├── walk_forward.py         — Rolling walk-forward validation
│   └── analysis.py             — Calibration, segments, thresholds
│
├── outputs/                    — Generated on each run (git-ignored)
│   ├── model_stage1.txt
│   ├── model_stage2.txt
│   ├── simulation_data.csv
│   ├── feature_matrix.csv
│   └── run_report.json
│
├── config.py                   — All constants and hyperparameters
├── main.py                     — 9-step orchestrator (Phase 1 + Phase 2)
├── requirements.txt
└── README.md
```

---

## 11. Determinism Requirement

Global seed required — all random processes derive from a controlled RNG:

```python
np.random.seed(42)           # global numpy state
np.random.default_rng(42)    # simulator-local RNG
```

---

## 12. Engineering Principles

- Causality preserved in reach computation
- No data leakage (rolling features use strictly past posts)
- Robust baseline with exponential decay
- Time-aware validation (chronological split + walk-forward)
- Well-calibrated output probabilities (ECE = 0.020)
- Modular architecture — each layer independently testable
- Expandable to Phase 2 (Stage-2 velocity model)
