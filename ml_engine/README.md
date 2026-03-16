# ML Engine

The predictive core of CreatorDynamix. Handles synthetic data simulation, feature engineering, model training, and evaluation — fully self-contained with no dependency on the backend or frontend.

---

## What It Does

1. **Simulates** 200 Instagram accounts posting over 400 days (~15,000 posts)
2. **Computes** a rolling weighted median baseline per account (no data leakage)
3. **Labels** each post: did it beat the account's recent baseline? (`survived = 1/0`)
4. **Extracts** 7 features per post (account history + content attributes)
5. **Trains** a LightGBM binary classifier (Stage-1 survival model)
6. **Validates** it with walk-forward rolling windows (10 × 30-day periods)
7. **Analyses** calibration, per-segment AUC, and optimal decision thresholds
8. **Saves** the trained model, datasets, and a full JSON report to `outputs/`

---

## Setup

**Requirements:** Python 3.10+

```bash
cd ml_engine

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running

```bash
# Make sure the venv is active
source .venv/bin/activate

# Run the full pipeline
python main.py
```

The pipeline runs all 7 steps and prints progress to the terminal. Typical runtime: ~10 seconds.

---

## Output

After a successful run, `outputs/` will contain:

| File | Description |
|------|-------------|
| `model.txt` | Trained LightGBM model (loadable for inference) |
| `simulation_data.csv` | All raw simulated posts with account, cluster, reach, engagement |
| `feature_matrix.csv` | Posts with engineered features + survival labels |
| `run_report.json` | Full structured report: metrics, walk-forward results, calibration, segments, thresholds |

---

## Configuration

All tunable constants live in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RANDOM_SEED` | 42 | Global seed — change to get a different simulation |
| `N_ACCOUNTS` | 200 | Number of synthetic accounts |
| `N_CLUSTERS` | 20 | Number of topic clusters |
| `SIMULATION_DAYS` | 400 | Days of simulated history |
| `DECAY_LAMBDA` | 0.03 | Exponential decay rate for rolling baseline (lower = longer memory) |
| `TRAIN_RATIO` | 0.70 | Chronological train split |
| `VAL_RATIO` | 0.15 | Validation split |

---

## Code Structure

```
ml_engine/
│
├── synthetic/                  — Data simulation layer
│   ├── account.py              — Account dataclass + generator
│   ├── cluster.py              — Cluster dataclass + generator (strong/medium/weak)
│   └── simulator.py            — Day-by-day simulation loop, reach computation
│
├── features/                   — Feature engineering layer
│   ├── baseline.py             — Rolling weighted median (no leakage)
│   └── feature_pipeline.py     — All 7 features + survival label
│
├── models/                     — Modeling and evaluation layer
│   ├── stage1.py               — Chronological split + LightGBM training
│   ├── evaluator.py            — ROC-AUC, log loss, feature importance, diagnostics
│   ├── walk_forward.py         — Rolling walk-forward validation
│   └── analysis.py             — Calibration, per-segment AUC, threshold analysis
│
├── docs/                       — Living documentation
│   ├── srs.md                  — Software Requirements Specification
│   ├── technicalArchitecture.md— Architecture decisions and observed metrics
│   └── suggestions.md          — Engineering guidelines and Phase 1 status
│
├── outputs/                    — Generated on each run (git-ignored)
│   ├── model.txt
│   ├── simulation_data.csv
│   ├── feature_matrix.csv
│   └── run_report.json
│
├── config.py                   — All constants and hyperparameters
├── main.py                     — Orchestrator (runs all 7 steps)
├── requirements.txt
└── README.md
```

---

## Feature Dictionary — Plain-English Reference

The 7 Stage-1 features and 4 Stage-2 features are defined here in plain English, alongside how they should be sourced in production. This table exists so that no developer mistakes these for "user inputs" — they are all computed or inferred automatically.

### Stage-1 Features (pre-post prediction)

| Feature name | Plain English | Production source |
|---|---|---|
| `rolling_weighted_median` | The account's typical reach across their last ~20 posts, with recent posts weighted more | Computed by `backend/feature_engine.py` from post history in the DB |
| `rolling_volatility` | How much the account's reach varies — a high number means unpredictable reach | Computed alongside the median from the same post history |
| `posting_frequency` | How often the account has posted in the last 14 days | Count of `Post` rows in the last 14 days for this account |
| `cluster_entropy` | How diverse the account's content topics are — a creator who mixes cooking and finance has high entropy | Computed from the distribution of `cluster_id` values across the account's post history |
| `cluster_id` | Which of the 20 topic clusters this specific post belongs to (0–19) | Auto-detected by `backend/ai_provider.py → detect_niche()` from caption + hashtags (OpenAI or keyword matching); user can override in the UI |
| `posting_time_bucket` | Broad time-of-day band: 0 = night, 1 = morning, 2 = afternoon, 3 = evening | Derived from the post's timestamp — not a user input |
| `content_quality` | A 0–1 quality score for the post's hook, caption, and hashtag combination | Auto-scored by `backend/ai_provider.py → score_content()` from caption + hashtags (OpenAI or heuristic); no manual input required |

### Stage-2 Features (1h velocity correction)

| Feature name | Plain English | Production source |
|---|---|---|
| `stage1_prior` | The Stage-1 survival probability — Stage-2 uses it as its starting point | Returned by the Stage-1 inference call; stored on the `Prediction` row in the backend |
| `rolling_weighted_median` | Same value used in Stage-1 | Passed through from the Stage-1 call; no re-computation needed |
| `likes_1h` | Raw like count approximately 60 minutes after the post went live | Entered by the user **or** fetched via Instagram Graph API webhook |
| `comments_1h` | Raw comment count at the same 1h checkpoint | Same as `likes_1h` |

> **Bottom line for product:** the only numbers an end user should ever type are `likes_1h` and `comments_1h`. Every other feature is derived from stored account history, AI-inferred from content, or passed automatically between API calls.

---

## Key Design Decisions

**Why synthetic data?**
Phase 1 validates that the modeling architecture and feature set work before touching real Instagram data. If the model can't learn on controlled synthetic data, it won't learn on noisy real data either.

**Why a rolling weighted median baseline?**
Each account has a very different follower count and engagement level. Raw reach is meaningless across accounts. The weighted median normalizes for account-level differences and adapts over time — a post that gets 10k reach is great for a 5k-follower account and poor for a 500k-follower account.

**Why chronological splitting?**
Real-world deployment means you train on the past and predict the future. Random train/test splits would leak future patterns into training. All splits here are strictly chronological.

**Why walk-forward validation?**
A single chronological split only tells you how the model performed on one slice of time. Walk-forward validation rolls the window forward month by month and checks whether performance is consistent — or whether the model degrades over time.

**Why LightGBM?**
Fast, handles categorical features natively (`cluster_id`), works well with tabular data, and produces well-calibrated probabilities without additional calibration steps. The Phase 1 model achieved ECE = 0.020 (well-calibrated).

---

## Loading the Model for Inference

```python
import lightgbm as lgb
import pandas as pd

# Load the saved model
booster = lgb.Booster(model_file="outputs/model.txt")

# Prepare a feature row (must match FEATURE_COLS order)
# ['rolling_weighted_median', 'rolling_volatility', 'posting_frequency',
#  'cluster_entropy', 'cluster_id', 'posting_time_bucket', 'content_quality']
features = pd.DataFrame([{
    "rolling_weighted_median": 4200.0,
    "rolling_volatility": 1800.0,
    "posting_frequency": 6.0,
    "cluster_entropy": 1.4,
    "cluster_id": 3,
    "posting_time_bucket": 2,
    "content_quality": 0.75,
}])

prob = booster.predict(features)[0]
print(f"Predicted survival probability: {prob:.3f}")
```

---

## Phase Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Synthetic simulation + Stage-1 survival model + deep evaluation | Complete ✓ |
| 2 | Stage-2 velocity correction model (early engagement signals) | Complete ✓ |
| 3 | Backend API serving predictions + real-data lifecycle (account → post → velocity → reach) | Complete ✓ |
| 4 | End-user frontend: AI content scoring, auto niche detection, plug-n-play OpenAI provider | Complete ✓ |
| 5 | Real Instagram Graph API ingestion (replace manual `likes_1h` / `comments_1h` entry) | Planned |
| 6 | Model retraining on real creator data + cluster reassignment | Planned |
| 7 | Deployment as hosted microservice | Planned |
