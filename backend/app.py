"""
CreatorDynamix Reach Intelligence API  v0.3.0

Entry point — app creation, lifespan, middleware, and router registration.

Route ownership:
  routers/health.py       → GET  /health
  routers/accounts.py     → POST /accounts, GET /accounts/{id}
  routers/posts.py        → POST /accounts/{id}/posts
                            GET  /posts/{id}
                            PATCH /posts/{id}/velocity
                            PATCH /posts/{id}/reach
  routers/predictions.py  → POST  /predict/stage1
                            POST  /predict/stage2
                            PATCH /predictions/{id}/outcome
                            GET   /predictions
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from predictor import model_store
from routers import accounts, health, posts, predictions


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Schema is managed by Alembic migrations — do NOT call create_all() here.
    # Run `alembic upgrade head` before starting the server for the first time
    # or after pulling new migrations.
    model_store.load()
    loaded  = [k for k, v in model_store.status.items() if v["loaded"]]
    missing = [k for k, v in model_store.status.items() if not v["loaded"]]
    if loaded:
        print(f"[startup] Models loaded: {', '.join(loaded)}")
    if missing:
        print(f"[startup] WARNING — models not found: {', '.join(missing)}")
    yield


app = FastAPI(
    title="CreatorDynamix Reach Intelligence API",
    description=(
        "Two-stage survival prediction for Instagram Reels.\n\n"
        "**Real-data flow** — register an account, ingest post history, "
        "then for every new Reel: ingest → T+1h velocity → T+24h reach.\n\n"
        "**Manual flow** — use `/predict/stage1` and `/predict/stage2` directly "
        "with precomputed features (for testing or external integrations)."
    ),
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(accounts.router)
app.include_router(posts.router)
app.include_router(predictions.router)
