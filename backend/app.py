"""
CreatorDynamix Reach Intelligence API

Endpoints:
  GET  /health                        — service health + model load status
  POST /predict/stage1                — pre-post survival prediction (creates DB row)
  POST /predict/stage2                — 1h-post velocity correction (updates DB row)
  PATCH /predictions/{id}/outcome     — record actual 24h outcome (closes the lifecycle)
  GET  /predictions                   — list all prediction records

Run locally:
  uvicorn app:app --reload --port 8000
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from database import Base, engine, get_db
from db_models import Prediction
from predictor import model_store, predict_stage1, predict_stage2
from schemas import (
    HealthResponse,
    OutcomeRequest,
    OutcomeResponse,
    PredictionSummary,
    Stage1Request,
    Stage1Response,
    Stage2Request,
    Stage2Response,
)


# ---------------------------------------------------------------------------
# Lifespan: create DB tables + load models
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create all tables if they don't exist yet (safe to call on every startup)
    Base.metadata.create_all(bind=engine)

    # Load ML model artifacts
    model_store.load()
    loaded  = [k for k, v in model_store.status.items() if v["loaded"]]
    missing = [k for k, v in model_store.status.items() if not v["loaded"]]
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
    title="CreatorDynamix Reach Intelligence API",
    description=(
        "Two-stage survival prediction for Instagram Reels.\n\n"
        "**Stage-1** — pre-post: predict whether a post will outperform the creator's "
        "own rolling baseline before it goes live.\n\n"
        "**Stage-2** — 1h-post: revise that prediction using first-hour engagement velocity. "
        "93% of correction lift is available within 1 hour of posting.\n\n"
        "Every prediction is persisted in a local SQLite database with a full lifecycle: "
        "Stage-1 → Stage-2 → actual outcome at 24h. This is the retraining dataset for Phase 4."
    ),
    version="0.2.0",
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
# Helper
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _format_dt(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


# ---------------------------------------------------------------------------
# Health
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


# ---------------------------------------------------------------------------
# Stage-1
# ---------------------------------------------------------------------------

@app.post(
    "/predict/stage1",
    response_model=Stage1Response,
    summary="Pre-post survival prediction",
    description=(
        "Predict whether a Reel will outperform the creator's rolling baseline "
        "**before the post goes live**.\n\n"
        "Creates a Prediction row in the database and returns a `prediction_id`. "
        "Pass that ID to `/predict/stage2` when the first-hour data is available."
    ),
    tags=["Predictions"],
)
def stage1_predict(req: Stage1Request, db: Session = Depends(get_db)) -> Stage1Response:
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

    # Persist the prediction
    row = Prediction(
        stage1_prob=result["survival_probability"],
        stage1_survives=result["survives"],
        stage1_confidence=result["confidence"],
        stage1_called_at=_utcnow(),
        feat_rolling_weighted_median=req.rolling_weighted_median,
        feat_rolling_volatility=req.rolling_volatility,
        feat_posting_frequency=req.posting_frequency,
        feat_cluster_entropy=req.cluster_entropy,
        feat_content_quality=req.content_quality,
        feat_cluster_id=req.cluster_id,
        feat_posting_time_bucket=result["posting_time_bucket"],
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    return Stage1Response(prediction_id=row.id, **result)


# ---------------------------------------------------------------------------
# Stage-2
# ---------------------------------------------------------------------------

@app.post(
    "/predict/stage2",
    response_model=Stage2Response,
    summary="1h-post velocity correction",
    description=(
        "Revise the Stage-1 prediction using first-hour engagement velocity.\n\n"
        "Call this ~60 minutes after a post goes live. "
        "Requires the `prediction_id` from the Stage-1 response to link the two predictions.\n\n"
        "Updates the existing Prediction row — it does not create a new one."
    ),
    tags=["Predictions"],
)
def stage2_predict(req: Stage2Request, db: Session = Depends(get_db)) -> Stage2Response:
    if model_store.stage2_1h is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Stage-2 (1h) model not loaded. Run ml_engine/main.py to generate model artifacts.",
        )

    row = db.get(Prediction, req.prediction_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No prediction found with id={req.prediction_id}. "
                   "Call /predict/stage1 first.",
        )

    result = predict_stage2(
        stage1_prior=req.stage1_prior,
        rolling_weighted_median=req.rolling_weighted_median,
        likes_1h=req.likes_1h,
        comments_1h=req.comments_1h,
        cluster_tier=req.cluster_tier,
    )

    # Update the existing row with Stage-2 results
    row.stage2_prob       = result["survival_probability"]
    row.stage2_survives   = result["survives"]
    row.stage2_correction = result["correction"]
    row.stage2_confidence = result["confidence"]
    row.stage2_called_at  = _utcnow()
    row.feat_likes_1h     = req.likes_1h
    row.feat_comments_1h  = req.comments_1h
    row.feat_cluster_tier = req.cluster_tier
    vel = result["velocity_features"]
    row.vel_norm_likes_1h    = vel["norm_likes_1h"]
    row.vel_comment_ratio_1h = vel["comment_ratio_1h"]
    row.vel_on_track_score   = vel["on_track_score"]

    db.commit()

    return Stage2Response(prediction_id=row.id, **result)


# ---------------------------------------------------------------------------
# Outcome recording
# ---------------------------------------------------------------------------

@app.patch(
    "/predictions/{prediction_id}/outcome",
    response_model=OutcomeResponse,
    summary="Record actual 24h outcome",
    description=(
        "Record whether the post actually outperformed the creator's baseline at 24h.\n\n"
        "Call this once per post, ~24 hours after it went live. "
        "This closes the prediction lifecycle and provides the ground truth label "
        "that will be used for retraining in Phase 4."
    ),
    tags=["Predictions"],
)
def record_outcome(
    prediction_id: int,
    req: OutcomeRequest,
    db: Session = Depends(get_db),
) -> OutcomeResponse:
    row = db.get(Prediction, prediction_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No prediction found with id={prediction_id}.",
        )

    row.actual_survived      = req.actual_survived
    row.outcome_recorded_at  = _utcnow()
    db.commit()

    # Evaluate whether each stage's binary prediction was correct
    s1_correct = None
    s2_correct = None

    if row.stage1_survives is not None:
        s1_correct = row.stage1_survives == req.actual_survived
    if row.stage2_survives is not None:
        s2_correct = row.stage2_survives == req.actual_survived

    return OutcomeResponse(
        prediction_id=row.id,
        stage1_prob=row.stage1_prob,
        stage2_prob=row.stage2_prob,
        actual_survived=req.actual_survived,
        stage1_correct=s1_correct,
        stage2_correct=s2_correct,
    )


# ---------------------------------------------------------------------------
# List predictions
# ---------------------------------------------------------------------------

@app.get(
    "/predictions",
    response_model=list[PredictionSummary],
    summary="List all prediction records",
    description=(
        "Returns prediction history, newest first. "
        "Filter by whether the outcome has been recorded, or by account."
    ),
    tags=["Predictions"],
)
def list_predictions(
    limit: int = Query(default=50, le=500, description="Max records to return"),
    account_id: Optional[str] = Query(default=None, description="Filter by account_id"),
    has_outcome: Optional[bool] = Query(
        default=None,
        description="True = only rows where actual outcome is recorded. "
                    "False = only rows still awaiting outcome.",
    ),
    db: Session = Depends(get_db),
) -> list[PredictionSummary]:
    q = db.query(Prediction)

    if account_id is not None:
        q = q.filter(Prediction.account_id == account_id)

    if has_outcome is True:
        q = q.filter(Prediction.actual_survived.isnot(None))
    elif has_outcome is False:
        q = q.filter(Prediction.actual_survived.is_(None))

    rows = q.order_by(Prediction.id.desc()).limit(limit).all()

    return [
        PredictionSummary(
            prediction_id=r.id,
            account_id=r.account_id,
            post_id=r.post_id,
            stage1_prob=r.stage1_prob,
            stage1_survives=r.stage1_survives,
            stage2_prob=r.stage2_prob,
            stage2_survives=r.stage2_survives,
            stage2_correction=r.stage2_correction,
            actual_survived=r.actual_survived,
            stage1_called_at=_format_dt(r.stage1_called_at),
            stage2_called_at=_format_dt(r.stage2_called_at),
            outcome_recorded_at=_format_dt(r.outcome_recorded_at),
        )
        for r in rows
    ]
