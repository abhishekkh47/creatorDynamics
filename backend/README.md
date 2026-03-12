# Backend — Reach Intelligence API

REST API that serves the two-stage ML model predictions. Built with FastAPI.

---

## Status

**Implemented (Phase 3 MVP).** Serving synthetic model artifacts. Ready for real data in Phase 4.

---

## Quickstart

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn app:app --reload --port 8000
```

The API will be live at `http://localhost:8000`.

**Prerequisite:** Run `ml_engine/main.py` first to generate the model artifact files in `ml_engine/outputs/`. The backend loads models from there at startup.

---

## Interactive Docs

FastAPI auto-generates interactive documentation:

| UI | URL |
|---|---|
| Swagger UI | `http://localhost:8000/docs` |
| ReDoc | `http://localhost:8000/redoc` |

Both show all endpoints, request schemas, response schemas, and let you make live requests.

---

## Endpoints

### `GET /health`

Returns load status of both models and the models directory path.

```json
{
  "status": "ok",
  "models": {
    "stage1":    { "loaded": true, "file": "model_stage1.txt" },
    "stage2_1h": { "loaded": true, "file": "model_stage2_1h.txt" }
  },
  "models_dir": "/path/to/ml_engine/outputs"
}
```

---

### `POST /predict/stage1`

**Pre-post prediction.** Call this before a Reel goes live.

Uses account history features to predict whether the post will outperform the creator's rolling baseline reach within 24 hours.

**Request:**
```json
{
  "rolling_weighted_median": 8500.0,
  "rolling_volatility": 1200.0,
  "posting_frequency": 5.0,
  "cluster_entropy": 1.8,
  "content_quality": 0.72,
  "cluster_id": 3,
  "hour_of_day": 14
}
```

| Field | Type | Description |
|---|---|---|
| `rolling_weighted_median` | float > 0 | Account's exponentially-weighted median reach (the survival baseline) |
| `rolling_volatility` | float ≥ 0 | Std of account's recent log-reach |
| `posting_frequency` | float ≥ 0 | Posts in the past 14 days |
| `cluster_entropy` | float ≥ 0 | Shannon entropy of account's topic cluster distribution |
| `content_quality` | float [0–1] | Content quality score |
| `cluster_id` | int ≥ 0 | Topic cluster ID (0–19) |
| `hour_of_day` | int [0–23] | Hour of posting; optional, defaults to morning |

**Response:**
```json
{
  "survival_probability": 0.71,
  "survives": true,
  "confidence": "high",
  "posting_time_bucket": 2,
  "model": "stage1"
}
```

**Decision threshold:** `0.35` (max-F1 from threshold analysis on test set).

---

### `POST /predict/stage2`

**1h-post velocity correction.** Call this ~60 minutes after a post goes live.

Takes the Stage-1 prior and first-hour engagement counts, then returns a corrected prediction. 93% of Stage-2's total AUC lift (+0.142 of +0.152) is available at 1 hour — no need to wait 6 hours.

**Request:**
```json
{
  "stage1_prior": 0.62,
  "rolling_weighted_median": 8500.0,
  "likes_1h": 340,
  "comments_1h": 18,
  "cluster_tier": "medium"
}
```

| Field | Type | Description |
|---|---|---|
| `stage1_prior` | float [0–1] | `survival_probability` from the Stage-1 call |
| `rolling_weighted_median` | float > 0 | Same value used in the Stage-1 call |
| `likes_1h` | int ≥ 0 | Raw like count at 1h |
| `comments_1h` | int ≥ 0 | Raw comment count at 1h |
| `cluster_tier` | `"strong"` / `"medium"` / `"weak"` | Topic cluster tier |

**Response:**
```json
{
  "survival_probability": 0.9878,
  "survives": true,
  "stage1_prior": 0.62,
  "correction": 0.3678,
  "confidence": "high",
  "velocity_features": {
    "norm_likes_1h": 0.04,
    "comment_ratio_1h": 0.0528,
    "on_track_score": 1.4286
  },
  "model": "stage2_1h"
}
```

| Response field | Description |
|---|---|
| `correction` | `stage2_prob − stage1_prior`. Positive = velocity revised prediction up. Negative = post underperforming pre-post expectation. |
| `velocity_features` | Normalized features computed from raw inputs. Useful for frontend display ("tracking 1.43× above baseline"). |

**Decision threshold:** `0.55` (max-F1 from Stage-2 threshold analysis).

---

## Code Structure

```
backend/
├── app.py          — FastAPI app, all routes, lifespan (model loading)
├── predictor.py    — Model store, inference functions, feature computation
├── schemas.py      — Pydantic request/response schemas
├── requirements.txt
└── README.md
```

**Design principle:** `app.py` handles HTTP concerns only. `predictor.py` handles all ML inference and feature engineering. `schemas.py` defines the contract. This keeps each file focused and independently testable.

---

## Relationship to ML Engine

The backend does **not** retrain models. It loads pre-trained LightGBM booster files:

```
ml_engine/outputs/model_stage1.txt     — Stage-1 pre-post model
ml_engine/outputs/model_stage2_1h.txt  — Stage-2 1h model (used by this API)
```

To retrain: run `ml_engine/main.py` and restart the backend. The new model files will be picked up automatically on the next startup.

---

## Model Details

| Model | AUC | Features | When to call |
|---|---|---|---|
| Stage-1 | 0.835 | 7 (account history + content) | Before posting |
| Stage-2 1h | 0.978 | 4 (stage1_prior + 1h velocity) | 60 min after posting |

The Stage-2 model corrects the Stage-1 prior most aggressively when Stage-1 was uncertain (prob 0.35–0.65). When Stage-1 is already confident, Stage-2 only adds small corrections.

---

## Phase 4 — Real Data

In Phase 4, the rolling features (`rolling_weighted_median`, `rolling_volatility`, etc.) will be computed from real Instagram post history stored in a database, and Stage-2 inputs will come from real engagement data via the Instagram API or webhooks. The prediction endpoints will remain unchanged — only the data sources change.
