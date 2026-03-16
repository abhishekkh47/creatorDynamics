# Backend — Reach Intelligence API

REST API that serves the two-stage ML model predictions. Built with FastAPI.

---

## Status

**Implemented (Phase 3 + Phase 4).** Serving synthetic model artifacts. Full real-data lifecycle — account registration, post ingestion, velocity updates, 24h reach closure — persisted to PostgreSQL. Schema migrations managed by Alembic.

---

## Quickstart

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Apply database migrations (always run before first start or after pulling changes)
alembic upgrade head

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

### `PATCH /predictions/{id}/outcome`

Record whether the post actually outperformed the creator's baseline at 24 hours. Closes the prediction lifecycle and records whether each stage's prediction was correct.

```json
{ "actual_survived": true }
```

Response:
```json
{
  "prediction_id": 1,
  "stage1_prob": 0.0378,
  "stage2_prob": 0.9404,
  "actual_survived": true,
  "stage1_correct": false,
  "stage2_correct": true
}
```

---

### `GET /predictions`

List all prediction records, newest first. Supports query params:
- `limit` — max records (default 50, max 500)
- `account_id` — filter by creator
- `has_outcome` — `true` = only rows with outcome recorded, `false` = only rows awaiting outcome

---

## Code Structure

```
backend/
├── app.py              — Entry point: lifespan, middleware, router registration only
├── routers/
│   ├── health.py       — GET  /health
│   ├── accounts.py     — POST /accounts, GET /accounts/{id}
│   ├── posts.py        — POST /accounts/{id}/posts, GET /posts/{id},
│   │                     PATCH /posts/{id}/velocity, PATCH /posts/{id}/reach
│   └── predictions.py  — POST /predict/stage1, POST /predict/stage2,
│                         PATCH /predictions/{id}/outcome, GET /predictions
├── predictor.py        — ModelStore, inference functions
├── schemas.py          — Pydantic request/response schemas
├── serializers.py      — ORM row → Pydantic response conversions
├── utils.py            — Shared primitive helpers (utcnow, fmt)
├── database.py         — SQLAlchemy engine + session setup (SQLite → Postgres via env var)
├── db_models.py        — ORM models: Account, Post, FeatureStore, Prediction
├── feature_engine.py   — Rolling feature computation from Post history
├── migrations/         — Alembic migration files (version-controlled schema changes)
│   ├── env.py          — Alembic runtime config (loads DATABASE_URL from .env)
│   └── versions/       — One .py file per schema change
├── alembic.ini         — Alembic configuration
├── .env                — DATABASE_URL and secrets (git-ignored)
├── .env.example        — Template for .env (committed)
├── data/               — SQLite database file lives here (git-ignored)
├── requirements.txt
└── README.md
```

**Design principles:**
- `app.py` registers routers — it does nothing else
- Each router owns one domain; adding a new domain means adding one file and one `include_router()` call
- `serializers.py` is the only place that maps ORM rows to Pydantic shapes — no router builds responses by hand
- `utils.py` has no app imports — safe to import from anywhere without circular risk

---

## Database Migrations

Schema changes are managed by **Alembic**. Never edit the database manually. The workflow for every schema change is:

1. **Edit `db_models.py`** — add/remove/rename a column or table in Python.
2. **Autogenerate a migration:**
   ```bash
   alembic revision --autogenerate -m "describe_what_changed"
   ```
   Alembic connects to the DB, compares it against your models, and generates a versioned migration file in `migrations/versions/`.
3. **Review the generated file** — always inspect it before applying. Autogenerate is ~95% correct but occasionally misses renames or index changes.
4. **Apply it:**
   ```bash
   alembic upgrade head
   ```
5. **Commit both** the model change and the migration file together in one git commit.

**Rolling back:**
```bash
alembic downgrade -1    # roll back one step
alembic downgrade base  # roll all the way back to empty DB
```

**Checking current state:**
```bash
alembic current         # which revision is the DB on?
alembic history         # full migration history
```

## Database

**Local development:** SQLite — no setup required. The database file is created automatically at `backend/data/predictions.db` on first startup.

**Production:** set the `DATABASE_URL` environment variable to a Postgres connection string:
```bash
export DATABASE_URL="postgresql://user:password@host:5432/creatorDynamix"
uvicorn app:app --port 8000
```
Everything else — the ORM models, queries, session management — works identically.

The `Prediction` table tracks the full lifecycle of every prediction:

```
POST /predict/stage1  →  creates row  (stage1_prob, input features, timestamp)
POST /predict/stage2  →  updates row  (stage2_prob, velocity features, correction)
PATCH /predictions/{id}/outcome  →  updates row  (actual_survived, stage1_correct, stage2_correct)
```

Over time, rows with `actual_survived` filled in become the retraining dataset for Phase 4.

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

## Feature Computation Reference — Where Each ML Input Comes From

The manual prediction endpoints (`POST /predict/stage1`, `POST /predict/stage2`) require callers to supply feature values. In the **account-based flow** (Phase 4), all of these are computed automatically. This table is the source of truth for what each feature is and where it originates — it exists so no developer ever exposes these to an end user again.

| Feature | What it is in plain English | Computed by | When |
|---|---|---|---|
| `rolling_weighted_median` | Typical reach for this account's recent posts | `feature_engine.py → compute_rolling_features()` | After every `PATCH /posts/{id}/reach` call |
| `rolling_volatility` | How consistent the account's reach is post-to-post | Same as above | Same as above |
| `posting_frequency` | Posts published in the last 14 days | Count of `Post` rows for this account in the last 14 days | Computed at prediction time |
| `cluster_entropy` | How varied the account's content topics are | `feature_engine.py` from per-post `cluster_id` history | After every `PATCH /posts/{id}/reach` call |
| `cluster_id` | Which topic cluster this specific post belongs to | Inferred from caption/hashtags using the topic model at post creation | `POST /accounts/{id}/posts` |
| `cluster_tier` | Whether this niche historically performs well | Precomputed lookup table keyed by `cluster_id` — set at account creation | `POST /accounts` onboarding |
| `content_quality` | Quality of hook, caption, hashtag combination (0–1) | Map from a 1–5 star rating supplied by the user; the **only** subjective input | `POST /accounts/{id}/posts` |
| `hour_of_day` | Hour of day the post goes live | Extracted from the post's `created_at` timestamp | `POST /accounts/{id}/posts` |
| `stage1_prior` | Stage-1 survival probability, passed into Stage-2 | Returned by `POST /predict/stage1`, stored on the `Prediction` row | Automatic in account flow |
| `likes_1h` | Raw like count ~60 minutes after posting | Entered by the user OR fetched from Instagram Graph API | `PATCH /posts/{id}/velocity` |
| `comments_1h` | Raw comment count ~60 minutes after posting | Same as `likes_1h` | `PATCH /posts/{id}/velocity` |

**Rule:** `likes_1h` and `comments_1h` are the only two values a real end user should ever type manually. Everything else is either computed from stored history, inferred from content, or returned by a previous API call.

---

## Real-Data Flow (Phase 4 — Complete)

The full real-data lifecycle is implemented and live:

```
POST /accounts                       — register creator
POST /accounts/{id}/posts (history)  — ingest past posts with known reach_24h
                                       → feature store auto-computed after each
POST /accounts/{id}/posts (new)      — new Reel goes live
                                       → Stage-1 prediction fires automatically
PATCH /posts/{id}/velocity           — T+1h: record likes_1h, comments_1h
                                       → Stage-2 prediction fires automatically
PATCH /posts/{id}/reach              — T+24h: record final reach
                                       → outcome recorded, feature store updated
```

Rolling features (`rolling_weighted_median`, `rolling_volatility`, etc.) are computed from real post history stored in the database. The next step is connecting Stage-2 input to real Instagram engagement data via the Instagram API or webhooks.
