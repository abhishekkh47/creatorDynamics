# Instagram Reach Intelligence Engine
## Technical Architecture Specification

---

## 1. System Architecture

```
Synthetic Engine
      ↓
Feature Pipeline
      ↓
Stage-1 Survival Model
      ↓
Evaluation Module
      ↓
Walk-Forward Validation
      ↓
Deep Analysis (Calibration · Segments · Thresholds)
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

## 8. Observed Metrics (Phase 1)

| Metric                  | Observed     | Notes                                   |
|-------------------------|--------------|-----------------------------------------|
| ROC-AUC (test)          | 0.835        | Above 0.70–0.80 target; acceptable      |
| Log loss (test)         | 0.500        | Reasonable                              |
| Walk-forward AUC mean   | 0.851        | std=0.010 — temporally stable           |
| Survival rate           | 0.501        | Target ~0.50 ✓                          |
| ECE (calibration)       | 0.020        | Well-calibrated                         |

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
│   ├── account.py
│   ├── cluster.py
│   └── simulator.py
│
├── features/
│   ├── baseline.py
│   └── feature_pipeline.py
│
├── models/
│   ├── stage1.py
│   ├── evaluator.py
│   ├── walk_forward.py       ← Phase 1 deepening
│   └── analysis.py           ← Phase 1 deepening
│
├── outputs/
│   ├── model.txt
│   ├── simulation_data.csv
│   ├── feature_matrix.csv
│   └── run_report.json
│
├── config.py
└── main.py
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
