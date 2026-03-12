"""
CreatorDynamics Reach Intelligence API

Two endpoints for the two-stage survival prediction pipeline:
  POST /predict/stage1  — pre-post prediction (no engagement data needed)
  POST /predict/stage2  — 1h-post correction (uses first-hour engagement velocity)

Run locally:
  uvicorn app:app --reload --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from predictor import model_store, predict_stage1, predict_stage2
from schemas import (
    HealthResponse,
    Stage1Request,
    Stage1Response,
    Stage2Request,
    Stage2Response,
)


# ---------------------------------------------------------------------------
# Lifespan: load models once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_store.load()
    loaded = [name for name, info in model_store.status.items() if info["loaded"]]
    missing = [name for name, info in model_store.status.items() if not info["loaded"]]
    if loaded:
        print(f"[startup] Models loaded: {', '.join(loaded)}")
    if missing:
        print(f"[startup] WARNING — models not found: {', '.join(missing)}")
        print(f"[startup] Run ml_engine/main.py first to generate model artifacts.")
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CreatorDynamics Reach Intelligence API",
    description=(
        "Two-stage survival prediction for Instagram Reels.\n\n"
        "**Stage-1** — pre-post: given account history and content attributes, "
        "what is the probability this post will outperform the creator's own recent baseline?\n\n"
        "**Stage-2** — 1h-post: given first-hour engagement velocity, "
        "how should we revise that prediction? (93% of correction lift available at 1h.)"
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health and model load status",
    tags=["Infrastructure"],
)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if model_store.all_loaded else "degraded",
        models=model_store.status,
        models_dir=str(model_store.models_dir),
    )


@app.post(
    "/predict/stage1",
    response_model=Stage1Response,
    summary="Pre-post survival prediction",
    description=(
        "Predict whether a Reel will outperform the creator's rolling baseline "
        "**before the post goes live**. Uses account history and content attributes only.\n\n"
        "Threshold: **0.35** (max-F1 from threshold analysis). "
        "Use this prediction to decide whether to post now or refine the content first."
    ),
    tags=["Predictions"],
)
def stage1_predict(req: Stage1Request) -> Stage1Response:
    if model_store.stage1 is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Stage-1 model not loaded. Run ml_engine/main.py to generate model artifacts.",
        )

    result = predict_stage1(
        rolling_weighted_median=req.rolling_weighted_median,
        rolling_volatility=req.rolling_volatility,
        posting_frequency=req.posting_frequency,
        cluster_entropy=req.cluster_entropy,
        content_quality=req.content_quality,
        cluster_id=req.cluster_id,
        hour_of_day=req.hour_of_day,
    )
    return Stage1Response(**result)


@app.post(
    "/predict/stage2",
    response_model=Stage2Response,
    summary="1h-post velocity correction",
    description=(
        "Update the Stage-1 prediction using first-hour engagement velocity.\n\n"
        "Call this **~60 minutes after a post goes live** with the raw like and "
        "comment counts. The model normalizes them by the account's rolling baseline "
        "and applies the 1h Stage-2 model (AUC: 0.978).\n\n"
        "Threshold: **0.55** (max-F1 from threshold analysis). "
        "A positive `correction` means early engagement is stronger than expected; "
        "negative means the post is underperforming its pre-post prediction."
    ),
    tags=["Predictions"],
)
def stage2_predict(req: Stage2Request) -> Stage2Response:
    if model_store.stage2_1h is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Stage-2 (1h) model not loaded. Run ml_engine/main.py to generate model artifacts.",
        )

    result = predict_stage2(
        stage1_prior=req.stage1_prior,
        rolling_weighted_median=req.rolling_weighted_median,
        likes_1h=req.likes_1h,
        comments_1h=req.comments_1h,
        cluster_tier=req.cluster_tier,
    )
    return Stage2Response(**result)
