# Backend — Reach Intelligence API

REST API that serves the two-stage ML model predictions. Built with FastAPI.

---

## Status

**Current (Phase 4 + AI layer complete).** Serving synthetic model artifacts. Full real-data lifecycle — account registration, post ingestion, velocity updates, 24h reach closure — persisted to PostgreSQL. Schema migrations managed by Alembic. AI provider layer added: content scoring and niche detection run via OpenAI (gpt-4o-mini) when an API key is present, or via the built-in heuristic provider when not.

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

## Environment Variables

Copy `.env.example` to `.env` and set:

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | No (defaults to SQLite) | PostgreSQL connection string for production |
| `OPENAI_API_KEY` | No | Enables OpenAI provider for content scoring and niche detection. Leave empty to use the offline heuristic provider. |

**Switching AI provider:**
```bash
# Turn OpenAI ON — add to .env, restart server
OPENAI_API_KEY=sk-...

# Turn OpenAI OFF — blank it or remove the line, restart server
OPENAI_API_KEY=
```

No code changes required. The server logs which provider is active on startup:
```
[ai] Provider: OpenAI (gpt-4o-mini) — with heuristic fallback
# or
[ai] Provider: Heuristic (OPENAI_API_KEY not set)
```

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

### `GET /meta/niches`

Returns the current model's cluster → niche mapping. The frontend fetches this at runtime to populate the niche dropdown. **Never hardcode cluster IDs or tiers in the frontend** — always fetch from this endpoint.

```json
[
  { "cluster_id": 7,  "label": "Comedy & Entertainment", "tier": "strong" },
  { "cluster_id": 0,  "label": "Fitness & Health",       "tier": "strong" },
  ...
]
```

To update after a model retrain: edit `cluster_config.py` and restart the server.

---

### `POST /meta/score-content`

Scores a Reel caption and hashtags on 5 content quality signals. Returns a `quality_score` (0–1) that maps directly to the `content_quality` ML feature.

Uses **OpenAI** when `OPENAI_API_KEY` is set, **heuristic scorer** otherwise. The response shape is identical either way.

**Request:**
```json
{ "caption": "Stop scrolling 👇 Here's the 1 thing nobody tells beginners...", "hashtags": "#fitness #gym" }
```

**Response:**
```json
{
  "quality_score": 0.83,
  "grade": "Excellent",
  "breakdown": {
    "hook_strength":      0.95,
    "cta_presence":       0.80,
    "hashtag_quality":    0.70,
    "caption_length":     0.85,
    "engagement_signals": 0.75
  },
  "tips": []
}
```

| Signal | Weight | What it measures |
|---|---|---|
| `hook_strength` | 30% | Opening line — questions, numbers, bold claims |
| `cta_presence` | 25% | Save / comment / share / follow / link in bio |
| `hashtag_quality` | 20% | 3–10 focused hashtags is the sweet spot |
| `caption_length` | 15% | 100–300 chars is the engagement-optimised range |
| `engagement_signals` | 10% | Emojis, in-body questions, exclamation marks |

---

### `POST /meta/detect-niche`

Detects the best-matching content niche from a caption and hashtags. Returns the `cluster_id` to use for the Stage-1 prediction.

Uses **OpenAI** when `OPENAI_API_KEY` is set, **keyword matching** otherwise.

**Request:**
```json
{ "caption": "My 6-month gym transformation 💪 what actually worked", "hashtags": "#fitness #gym" }
```

**Response:**
```json
{
  "cluster_id": 0,
  "confidence": 0.94,
  "reasoning": "Caption and hashtags strongly indicate Fitness & Health content."
}
```

The frontend uses this to auto-select the niche dropdown but keeps it editable.

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
│   ├── meta.py         — GET  /meta/niches
│   │                     POST /meta/score-content
│   │                     POST /meta/detect-niche
│   ├── accounts.py     — POST /accounts, GET /accounts/{id}
│   ├── posts.py        — POST /accounts/{id}/posts, GET /posts/{id},
│   │                     PATCH /posts/{id}/velocity, PATCH /posts/{id}/reach
│   └── predictions.py  — POST /predict/stage1, POST /predict/stage2,
│                         PATCH /predictions/{id}/outcome, GET /predictions
│
├── ai_provider.py      — Plug-n-play AI provider: ABC + HeuristicProvider +
│                         OpenAIProvider + get_provider() factory.
│                         Toggle: set/unset OPENAI_API_KEY in .env + restart.
├── content_scorer.py   — Rule-based content quality scorer (used by HeuristicProvider).
│                         Scores hook strength, CTA, hashtags, length, engagement signals.
├── cluster_config.py   — Cluster → niche mapping (single source of truth).
│                         Update here after every model retrain.
│
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
├── .env                — DATABASE_URL + OPENAI_API_KEY (git-ignored)
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
- `routers/meta.py` calls `get_provider()` — it has no direct dependency on OpenAI or the heuristic scorer

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

## AI Provider — Plug-n-Play Architecture

Two features — content scoring and niche detection — are powered by an AI provider layer that can be switched without any code changes.

```
get_provider()
     │
     ├── OPENAI_API_KEY set  →  OpenAIProvider (gpt-4o-mini)
     │                              └── any call fails → auto-fallback to heuristic
     │
     └── no key             →  HeuristicProvider (offline, zero cost, zero latency)
```

**`ai_provider.py`** is the only file that knows about this decision. Routers call `get_provider().score_content()` or `get_provider().detect_niche()` — they never import OpenAI or the heuristic scorer directly.

### Adding a new AI provider (e.g. Anthropic, Gemini)

1. Subclass `AIProvider` in `ai_provider.py` and implement `score_content()` and `detect_niche()`.
2. Add a detection branch in `get_provider()`.
3. Done — no router or schema changes needed.

### OpenAI call details

| Task | Model | Approx. tokens | Approx. cost |
|---|---|---|---|
| `score_content` | gpt-4o-mini | ~350 in / ~120 out | ~$0.00008 per call |
| `detect_niche` | gpt-4o-mini | ~250 in / ~40 out  | ~$0.00004 per call |

Both calls use `response_format={"type": "json_object"}` for reliable structured output.

---

## Feature Computation Reference — Where Each ML Input Comes From

The manual prediction endpoints (`POST /predict/stage1`, `POST /predict/stage2`) require callers to supply feature values. In the **account-based flow** (Phase 4), all of these are computed automatically. This table is the source of truth for what each feature is and where it originates — it exists so no developer ever exposes these to an end user again.

| Feature | What it is in plain English | Computed by | When |
|---|---|---|---|
| `rolling_weighted_median` | Typical reach for this account's recent posts | `feature_engine.py → compute_rolling_features()` | After every `PATCH /posts/{id}/reach` call |
| `rolling_volatility` | How consistent the account's reach is post-to-post | Same as above | Same as above |
| `posting_frequency` | Posts published in the last 14 days | Count of `Post` rows for this account in the last 14 days | Computed at prediction time |
| `cluster_entropy` | How varied the account's content topics are | `feature_engine.py` from per-post `cluster_id` history | After every `PATCH /posts/{id}/reach` call |
| `cluster_id` | Which topic cluster this specific post belongs to | `POST /meta/detect-niche` — AI detects from caption + hashtags; user can override in dropdown | `POST /accounts/{id}/posts` |
| `cluster_tier` | Whether this niche historically performs well | Derived from `cluster_id` via `cluster_config.py`; served by `GET /meta/niches` | Set at account onboarding |
| `content_quality` | Quality of hook, caption, hashtag combination (0–1) | `POST /meta/score-content` — AI scores automatically from caption + hashtags; no star rating | `POST /accounts/{id}/posts` |
| `hour_of_day` | Hour of day the post goes live | Extracted from the post's `created_at` timestamp | `POST /accounts/{id}/posts` |
| `stage1_prior` | Stage-1 survival probability, passed into Stage-2 | Returned by `POST /predict/stage1`, stored on the `Prediction` row | Automatic in account flow |
| `likes_1h` | Raw like count ~60 minutes after posting | Entered by the user OR fetched from Instagram Graph API | `PATCH /posts/{id}/velocity` |
| `comments_1h` | Raw comment count ~60 minutes after posting | Same as `likes_1h` | `PATCH /posts/{id}/velocity` |

**Rule:** `likes_1h` and `comments_1h` are the only two values a real end user should ever type manually. Everything else is computed from stored post history, AI-inferred from content, or returned by a previous API call.

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
