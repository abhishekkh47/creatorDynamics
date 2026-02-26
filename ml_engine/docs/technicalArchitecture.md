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
```

---

## 2. Synthetic Engine Design

### 2.1 Account Class

**Attributes:**

- `account_id`
- `follower_count`
- `baseline_reach_ratio`
- `quality_mean`
- `quality_variance`
- `velocity_sensitivity`
- `decay_rate`
- `volatility_factor`
- `cluster_distribution`

### 2.2 Cluster Class

**Attributes:**

- `cluster_id`
- `performance_multiplier`
- `engagement_rate_multipliers`
- `early_burst_bias`

### 2.3 Simulation Loop

Time-progressive iteration over all accounts per day:

```python
for day in simulation_days:
    for account in accounts:
        if account_should_post():
            generate_post()
```

---

## 3. Reach Computation Model

Causal structure:

```
Account + Cluster + Quality
          ↓
        Reach
          ↓
   Engagement Rates
          ↓
   Engagement Curve
```

---

## 4. Rolling Baseline

Exponential decay weighting:

```
weight = exp(-λ * age)
```

Compute weighted median over past posts only — no future leakage.

---

## 5. Feature Engineering

### Account Features

- `rolling_weighted_median`
- `rolling_volatility`
- `posting_frequency`
- `cluster_entropy`

### Content Features

- `cluster_id` (categorical)
- `posting_time_bucket`
- `content_quality`

### Excluded (Phase 1)

- Early velocity features

---

## 6. Modeling

**Library:** LightGBM

**Config:**

```python
objective = "binary"
metric = "binary_logloss"
categorical_feature = ["cluster_id"]
```

---

## 7. Data Splitting

Chronological split — no randomization:

| Split      | Proportion |
|------------|------------|
| Train      | 70%        |
| Validation | 15%        |
| Test       | 15%        |

---

## 8. Expected Metrics

A healthy synthetic design should produce:

- **ROC-AUC:** 0.70 – 0.80
- **Log loss:** reasonable

**Feature importance expectations:**

- Quality: strong signal
- Cluster: meaningful signal
- Baseline: meaningful signal

---

## 9. Code Structure

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
│   └── evaluator.py
│
├── config.py
└── main.py
```

---

## 10. Determinism Requirement

A global seed is required — all random processes must derive from a controlled RNG:

```python
np.random.seed(42)
```

---

## 11. Engineering Principles

- Causality preserved
- No data leakage
- Robust baseline computation
- Time-aware validation
- Modular architecture
- Expandable to Phase 2
