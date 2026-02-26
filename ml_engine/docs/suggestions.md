# Phase 1 Engineering Guidelines

> Follow this and the project stays clean. Ignore it and you'll be refactoring in 2–3 weeks.

---

## 1. Freeze the Scope of Phase 1

**You are NOT building (yet):**

- Regime shifts
- Follower growth loops
- Adaptive decay λ
- Walk-forward retraining
- Stage-2 velocity model

**Phase 1 objective:**

Validate that synthetic physics → produces learnable signal → Stage-1 survival classifier achieves realistic AUC.

That's it. If Stage 1 fails, everything else is irrelevant.

---

## 2. Determinism First

Set a global random seed before anything else:

```python
import numpy as np
np.random.seed(42)
```

If your synthetic engine isn't reproducible:

- Debugging becomes impossible
- Metrics shift randomly
- You won't know if a change improved things or just moved noise

Reproducibility is non-negotiable.

---

## 3. Build in Layers, Not a Monolith

Do NOT write one 800-line script. Implement in this exact order:

**Step 1 — Account generator**

Return a DataFrame of accounts. Inspect:

- Follower distribution
- Baseline ratio distribution

**Step 2 — Cluster generator**

Return 20 clusters with multipliers. Inspect:

- Distribution sanity

**Step 3 — Post simulator (without engagement curve)**

Generate: `timestamp`, `cluster`, `quality`, `reach_24h`. Inspect:

- Reach distribution
- Cluster vs. reach pattern
- Quality vs. reach pattern

**Step 4 — Rolling baseline logic**

Compute: weighted rolling median, survival label. Inspect:

- Survival balance (~50/50)
- Drift responsiveness

Only after these 4 steps pass sanity checks: add the engagement curve.

---

## 4. Print Diagnostics Early

Before modeling, verify:

- `correlation(quality, reach)`
- `correlation(cluster_multiplier, reach)`
- `correlation(reach, survival)`
- A few random account timelines

If the synthetic world looks fake, the model will either be too easy or impossible to train.

---

## 5. Target Metrics

After Stage-1 training, expect:

| Metric           | Target        |
|------------------|---------------|
| ROC-AUC          | 0.70 – 0.80   |
| Log loss         | Reasonable    |
| Cluster importance | Visible     |
| Quality importance | Strong      |

**Interpreting results:**

- AUC > 0.90 → synthetic is too deterministic
- AUC < 0.60 → synthetic is too noisy

Use this feedback to guide tuning of the simulation, not the model.

---

## 6. Do Not Tune Hyperparameters Yet

If the model fails, fix the synthetic design first. Do not start hyperparameter hunting.

---

## Final Mindset

You are not coding a toy. You are building:

- A causal synthetic simulation
- A feature pipeline
- A probabilistic survival system
- A two-stage predictive architecture

Build it methodically.
