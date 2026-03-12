# Experiment Log
## Instagram Reach Intelligence Engine

> This document is the canonical record of every meaningful milestone: what we were
> trying to achieve, what was built, what the numbers said, and what we decided next.
>
> Every future run that produces a meaningful result should add an entry following
> the template at the bottom of this file.

---

## North Star Goal

Build a two-stage predictive system that answers one question:

> **Will this Reel outperform the creator's own recent baseline — and can we know it within 1 hour of posting?**

**Stage-1** (pre-post): Given what we know about the account, cluster, and content before the post goes live — what is the survival probability?

**Stage-2** (1h post-post): Given the first hour of engagement velocity, how should we revise that prediction?

**End state (Phase 3+):** This runs on real Instagram data, exposed via a FastAPI backend, consumed by a creator-facing frontend that shows "this post is tracking above/below your baseline" within 60 minutes of publishing.

---

## Milestone 1 — Phase 1: First Working Pipeline

**Date:** Early February 2026  
**Status:** Complete

### Goal
Implement the full Phase 1 pipeline end-to-end from scratch:
- Synthetic account + cluster + post simulation
- Rolling weighted median baseline
- Stage-1 LightGBM survival classifier
- Basic evaluation (AUC, log loss, feature importance)

### Config
| Parameter | Value |
|---|---|
| Accounts | 200 |
| Clusters | 20 |
| Simulation days | 400 |
| Decay λ | 0.05 |
| Random seed | 42 |

### Results

| Metric | Value |
|---|---|
| Stage-1 test AUC | 0.988 |
| Survival rate | ~0.50 |

### Key Finding

**AUC was too high (0.988).** The synthetic world was too predictable — the model learned to almost perfectly distinguish survivors from non-survivors, which means the synthetic physics were too deterministic. A model this confident on synthetic data will fail on real data where chaos reigns.

Target was 0.70–0.80. At 0.988, we had no room for the unpredictability that makes Instagram reach genuinely hard to predict.

### Decision

Fix the synthetic engine to introduce realistic unpredictability before moving forward. Do not tune the model — fix the data generator.

---

## Milestone 2 — Phase 1: Synthetic Tuning (AUC Reduction)

**Date:** February 2026  
**Status:** Complete

### Goal
Reduce Stage-1 AUC from ~0.99 to a realistic 0.70–0.80 range by making the synthetic simulation noisier and less predictable.

### What Was Tuned

**In `synthetic/simulator.py` — `_compute_reach`:**
- Replaced fixed lognormal noise with a mean-corrected lognormal: `mean = -0.5 * σ²` (ensures noise has mean=1 regardless of σ, so reach doesn't systematically inflate or deflate)
- Added discrete algorithmic events (final probabilities: 30% suppressed, 30% viral, 40% normal)
  - Suppressed: multiplier 0.02–0.20 (algorithm buries the post)
  - Viral: multiplier 3.0–10.0 (algorithm boosts the post)
  - Normal: exponential(0.8) + 0.2 (organic variation)
- Implemented daily random walks for both `baseline_reach_ratio` and `volatility_factor` per account — making the rolling baseline a lagging estimate rather than a perfect predictor

**In `synthetic/account.py`:**
- Increased `volatility_factor` range to 0.8–1.3 (higher baseline noise)

**In `synthetic/cluster.py`:**
- Compressed cluster tier performance multipliers to reduce inter-cluster signal:
  - Strong: 1.2–1.7 (was 1.5–2.5)
  - Medium: 0.8–1.2 (was 0.8–1.4)
  - Weak: 0.5–0.85 (was 0.3–0.75)

**In `config.py`:**
- Reduced `DECAY_LAMBDA` from 0.05 → 0.03 (longer memory on rolling baseline = more lag = harder prediction problem)

### Config (Final)
| Parameter | Value |
|---|---|
| Accounts | 200 |
| Clusters | 20 |
| Simulation days | 400 |
| Decay λ | 0.03 |
| Volatility factor range | 0.8–1.3 |
| Algorithmic events | 30% suppressed / 30% viral / 40% normal |

### Results

| Metric | Value | Target |
|---|---|---|
| Stage-1 test AUC | 0.835 | 0.70–0.80 |
| Survival rate | 0.501 | ~0.50 |
| corr(quality, reach) | 0.027 | Low |
| corr(cluster, reach) | 0.013 | Low |
| corr(reach, survived) | 0.072 | Low |

### Key Finding

**AUC settled at a structural floor of ~0.835** — slightly above the 0.70–0.80 target. Further tuning would require degrading the simulation to the point where the signal is fake.

The reason: `rolling_weighted_median` is simultaneously the survival threshold (label definition) and a direct input feature. The model will always have near-complete distributional knowledge because the thing we're predicting against is also something we feed in. This is a known architectural constraint, not a bug. AUC ~0.835 is acceptable for Phase 1.

**Feature importance ordering confirms causality:**

```
rolling_volatility           ████████████████████ (669)
rolling_weighted_median      ███████████████████ (648)
content_quality              ████████████████ (544)
cluster_entropy              █████████████ (452)
cluster_id                   █████████ (324)
posting_frequency            █ (65)
posting_time_bucket            (28)
```

The model learned the right things — account-level volatility and baseline are the primary signals, followed by content quality and cluster, then posting pattern.

### Decision

AUC is acceptable. Proceed to Phase 1 Deepening before moving to Phase 2.

---

## Milestone 3 — Phase 1 Deepening

**Date:** February 2026  
**Status:** Complete

### Goal
Before building Phase 2, harden the Stage-1 evaluation with four additional analyses that answer: *can we actually trust this model?*

### What Was Built
- `models/walk_forward.py` — rolling window validation across all 10 available time windows
- `models/analysis.py` — calibration, per-segment AUC, threshold analysis
- Updated `main.py` to a 7-step pipeline

### Config
Same as Milestone 2.

### Results

**Walk-Forward Validation (10 windows, 30-day each)**

| Metric | Value |
|---|---|
| AUC mean | 0.851 |
| AUC std | 0.010 |
| AUC min | 0.836 |
| AUC max | 0.869 |
| Stable (std < 0.05) | ✓ |

The model does not degrade over time. AUC is consistent across the entire 400-day simulation.

**Calibration**

| Metric | Value |
|---|---|
| Expected Calibration Error (ECE) | 0.020 |
| Interpretation | Well-calibrated |

When Stage-1 says 70% survival probability, ~70% of those posts actually survived. The output probabilities can be trusted as confidence scores.

**Per-Segment AUC**

| Segment | AUC | Posts |
|---|---|---|
| Macro (>100k followers) | 0.875 | 60 |
| Nano (<10k followers) | 0.834 | 1,182 |
| Micro (10k–100k followers) | 0.799 | 995 |
| Cluster weak | 0.843 | 933 |
| Cluster medium | 0.836 | 863 |
| Cluster strong | 0.820 | 441 |

Micro creators are the weakest segment (AUC 0.799). This is the gap Phase 2 velocity features need to address.

**Threshold Analysis**

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.35 | 0.689 | 0.868 | 0.768 ← best F1 |
| 0.50 | 0.748 | 0.753 | 0.750 |
| 0.70 | 0.834 | 0.542 | 0.657 |

For a creator-facing product, threshold 0.35 maximizes F1. For a high-precision internal alert, use 0.70.

### Key Finding

Stage-1 is genuinely trustworthy: temporally stable, well-calibrated, and predictive across all creator sizes. The weakest point is micro creators — exactly what early engagement velocity should fix, since micro accounts have the most variable organic reach patterns.

### Decision

Phase 1 is validated. Build Phase 2: velocity simulation + Stage-2 correction model.

---

## Milestone 4 — Phase 2: Stage-2 Velocity Correction Model

**Date:** February 2026  
**Status:** Complete

### Goal
After a post has been live for 6 hours, use the early engagement velocity (likes and comments at 1h, 3h, 6h) to update Stage-1's pre-post prediction. This answers: *does early traction confirm or contradict what we expected?*

### What Was Built
- `synthetic/velocity_simulator.py` — simulates per-post engagement curves using exponential burst model; tier-aware (strong clusters burst faster)
- `features/velocity_features.py` — 9 velocity features, all normalized by `rolling_weighted_median` to be account-size-agnostic
- `models/stage2.py` — Stage-2 LightGBM classifier + prior vs posterior analysis
- Updated `main.py` to a 9-step pipeline

### Stage-2 Features

| Feature | What It Measures |
|---|---|
| `stage1_prior` | Stage-1 pre-post prediction (dominant) |
| `norm_likes_1h/3h/6h` | Cumulative likes, normalized by account baseline |
| `like_velocity_1to3` | Average likes/hour between 1h and 3h |
| `like_velocity_3to6` | Average likes/hour between 3h and 6h |
| `burst_ratio` | Fraction of 3h likes that arrived in first hour (front-loading) |
| `comment_ratio_1h` | Comments/likes at 1h (strong community signal) |
| `on_track_score` | Is 1h velocity on pace to beat the baseline by 24h? |

### Config
| Parameter | Value |
|---|---|
| Stage-2 train split | 70% of val set (chronological) |
| Stage-2 n_estimators | 200 |
| Stage-2 early stopping | 20 rounds |

### Results

**Stage-2 Metrics**

| Metric | Stage-1 | Stage-2 | Lift |
|---|---|---|---|
| Test AUC | 0.835 | 0.987 | +0.152 |
| Val AUC | 0.855 | 0.991 | +0.136 |

**Feature Importance (Stage-2)**

```
stage1_prior          ████████████████████ (1102)  — dominant
norm_likes_6h         ██████ (354)
like_velocity_3to6    █████ (315)
like_velocity_1to3    █████ (305)
on_track_score        ███ (220)
norm_likes_1h         ██ (147)
norm_likes_3h         █ (110)
burst_ratio           █ (95)
comment_ratio_1h        (23)
```

**Prior vs Posterior**

| Metric | Value |
|---|---|
| Corrections (Stage-1 wrong → Stage-2 right) | 510 |
| Regressions (Stage-1 right → Stage-2 wrong) | 83 |
| Net correction ratio | 6.1× more fixes than breaks |

### Key Finding

Stage-2 genuinely corrects Stage-1 rather than replacing it. The dominance of `stage1_prior` in Stage-2 feature importance confirms the two-stage architecture is working as designed: Stage-2 is a Bayesian update on the prior, not an independent classifier.

The 6.1× correction ratio (510 fixes vs 83 breaks) is strong evidence that early engagement velocity carries real information beyond what's available before the post goes live.

### Decision

Phase 2 validated. Before moving to the backend API, run Phase 2 deepening: add the same rigorous analysis that Phase 1 received (calibration, segments, thresholds + two new analyses: observation windows and uncertainty resolution).

---

## Milestone 5 — Phase 2 Deepening

**Date:** 12 March 2026  
**Status:** Complete

### Goal
Apply the same evaluation rigor to Stage-2 that Phase 1 received, plus two new analyses specific to the two-stage architecture:

1. **Observation window analysis** — at what time checkpoint (1h, 3h, 6h) is the prediction good enough to act on?
2. **Uncertainty resolution analysis** — where does Stage-2 help most: when Stage-1 is confident or uncertain?
3. **Stage-2 walk-forward** — is the lift consistent across all time windows, or is it a one-split artifact?
4. **Stage-2 calibration, segments, thresholds** — same as Phase 1 deepening

### What Was Built
- `features/velocity_features.py` — added `VELOCITY_FEATURES_1H` (4 features) and `VELOCITY_FEATURES_3H` (7 features) subsets
- `models/analysis.py` — `feature_cols` param added to all analysis functions; added `observation_window_analysis`, `uncertainty_resolution_analysis`, `print_stage2_deep_analysis`
- `models/walk_forward.py` — added `walk_forward_stage2`: retrains Stage-1 from scratch on each window, applies fixed Stage-2, records both AUCs and lift
- `main.py` — expanded to 11-step pipeline (Steps 10–11 are Stage-2 deepening)

### Config
Same as Milestone 4. Three additional Stage-2 models trained internally for the observation window analysis (1h, 3h, 6h feature subsets).

### Results

**Stage-2 Calibration**

| Metric | Value |
|---|---|
| ECE | 0.011 |
| Interpretation | Well-calibrated |

Stage-2 probabilities are reliable confidence scores. Most bins are tightly calibrated; minor gaps appear in mid-range bins (0.53–0.75), which is typical when velocity signals are bimodal.

**Stage-2 Per-Segment AUC**

| Segment | Stage-1 AUC | Stage-2 AUC | Lift |
|---|---|---|---|
| Macro (>100k) | 0.875 | 0.997 | +0.122 |
| Micro (10k–100k) | 0.799 | 0.992 | +0.193 |
| Nano (<10k) | 0.834 | 0.981 | +0.147 |
| Cluster strong | 0.820 | 0.983 | +0.163 |
| Cluster medium | 0.836 | 0.990 | +0.154 |
| Cluster weak | 0.843 | 0.987 | +0.144 |

Micro creators see the **largest lift (+0.193)** — exactly the segment where Stage-1 was weakest. The velocity signal resolves the ambiguity that pre-post features couldn't.

**Stage-2 Threshold Analysis**

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.35 | 0.944 | 0.936 | 0.940 |
| 0.55 | 0.972 | 0.915 | 0.943 ← best F1 |
| 0.70 | 0.978 | 0.907 | 0.941 |

Stage-2 is much more tolerant to threshold choice than Stage-1 — all thresholds from 0.35–0.70 produce F1 > 0.939. For the backend API, use **threshold = 0.55** for creator-facing notifications (high precision, still strong recall).

**Observation Window AUC — the most important finding**

| Checkpoint | AUC | Lift vs Stage-1 | Cumulative lift captured |
|---|---|---|---|
| 0h (Stage-1 only) | 0.835 | — | 0% |
| 1h | 0.978 | +0.142 | 93% |
| 3h | 0.987 | +0.152 | 99% |
| 6h | 0.987 | +0.152 | 100% |

**93% of Stage-2's total lift is available within 1 hour of posting.** The incremental value of waiting from 1h to 6h is only 0.001 AUC. A 4-feature, 1-hour model is nearly as powerful as the full 9-feature, 6-hour model.

**Uncertainty Resolution**

| Stage-1 confidence bucket | Posts | Stage-1 AUC | Stage-2 AUC | Lift |
|---|---|---|---|---|
| Uncertain (prob 0.35–0.65) | 550 | 0.590 | 0.976 | **+0.385** |
| Moderate (0.25–0.35 or 0.65–0.75) | 438 | 0.665 | 0.972 | +0.307 |
| Confident (<0.25 or >0.75) | 1,249 | 0.904 | 0.993 | +0.089 |

Stage-2 adds the most value precisely where Stage-1 is most unsure. This is the hallmark of a healthy two-stage Bayesian update — the correction is proportional to uncertainty.

**Stage-2 Walk-Forward (10 windows)**

| Metric | Value |
|---|---|
| Stage-1 mean AUC | 0.851 |
| Stage-2 mean AUC | 0.987 |
| Mean lift | +0.136 |
| Lift std | 0.010 |
| Minimum lift (worst window) | +0.120 |
| Consistent (all windows positive) | ✓ |

The lift is not a single-split artifact. Stage-2 reliably improves Stage-1 on every time window across the full 400-day simulation.

### Key Findings Summary

1. **Check at 1h, not 6h.** 93% of the lift is already there. The backend API should fire a creator notification at T+1h, not T+6h.
2. **Stage-2 is most valuable for micro creators.** The largest per-segment lift (+0.193) is for the 10k–100k follower range — the bulk of the creator economy.
3. **Threshold 0.55 for production.** Any threshold between 0.35–0.70 is nearly equivalent (F1 > 0.939). Use 0.55 for the API as a clean midpoint.
4. **The architecture is sound.** Stage-2 corrects uncertainty rather than replicating confidence — it behaves exactly like a principled Bayesian update should.

### Decision

Phase 2 deepening is complete. The full two-stage system is validated, robust, and ready to be exposed via an API. Next: **Phase 3 — FastAPI Backend.**

---

## Upcoming: Milestone 6 — Phase 3: FastAPI Backend

**Status:** Planned

### Goal
Expose the two trained models (Stage-1 pre-post, Stage-2 1h-post) via a REST API that can be consumed by the frontend and eventually by real Instagram webhook events.

### Planned Endpoints

| Endpoint | Input | Output |
|---|---|---|
| `POST /predict/stage1` | account metadata, content quality, cluster, posting time | survival probability (pre-post) |
| `POST /predict/stage2` | stage1_prior + 1h engagement velocity | corrected survival probability |
| `GET /health` | — | service status |

### Key Architecture Decisions to Make
- Model loading: load from `ml_engine/outputs/` at startup or keep in-process?
- Observation window: ship the 1h model only (4 features) or support 1h/3h/6h?
- Input validation: Pydantic schemas for all request/response types
- How does the backend get real Instagram engagement data? (Mock for now, real in Phase 4)

---

## Log Template (for future entries)

Copy this block when adding a new milestone:

```markdown
## Milestone N — [Phase Name]: [Short Description]

**Date:** [Date]
**Status:** Complete / In Progress

### Goal
[1–3 sentences: what question are we trying to answer?]

### What Was Built
- [file] — [what it does]

### Config
| Parameter | Value |
|---|---|

### Results

[tables / code blocks with metrics]

### Key Finding
[The most important insight. What did the numbers tell us that we didn't know before?]

### Decision
[What do we do next, and why?]
```
