from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from database import get_db
from db_models import Prediction
from predictor import model_store, predict_stage1, predict_stage2
from schemas import (
    OutcomeRequest, OutcomeResponse,
    PredictionSummary,
    Stage1Request, Stage1Response,
    Stage2Request, Stage2Response,
)
from serializers import prediction_to_summary
from utils import utcnow

router = APIRouter(tags=["Manual Predictions"])


@router.post(
    "/predict/stage1",
    response_model=Stage1Response,
    summary="Manual Stage-1 prediction (pass precomputed features)",
)
def stage1_predict(req: Stage1Request, db: Session = Depends(get_db)) -> Stage1Response:
    if model_store.stage1 is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Stage-1 model not loaded.")
    result = predict_stage1(
        rolling_weighted_median=req.rolling_weighted_median,
        rolling_volatility=req.rolling_volatility,
        posting_frequency=req.posting_frequency,
        cluster_entropy=req.cluster_entropy,
        content_quality=req.content_quality,
        cluster_id=req.cluster_id,
        hour_of_day=req.hour_of_day,
    )
    row = Prediction(
        stage1_prob=result["survival_probability"],
        stage1_survives=result["survives"],
        stage1_confidence=result["confidence"],
        stage1_called_at=utcnow(),
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


@router.post(
    "/predict/stage2",
    response_model=Stage2Response,
    summary="Manual Stage-2 prediction",
)
def stage2_predict(req: Stage2Request, db: Session = Depends(get_db)) -> Stage2Response:
    if model_store.stage2_1h is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Stage-2 model not loaded.")
    row = db.get(Prediction, req.prediction_id)
    if row is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"No prediction with id={req.prediction_id}."
        )
    result = predict_stage2(
        stage1_prior=req.stage1_prior,
        rolling_weighted_median=req.rolling_weighted_median,
        likes_1h=req.likes_1h,
        comments_1h=req.comments_1h,
        cluster_tier=req.cluster_tier,
    )
    row.stage2_prob       = result["survival_probability"]
    row.stage2_survives   = result["survives"]
    row.stage2_correction = result["correction"]
    row.stage2_confidence = result["confidence"]
    row.stage2_called_at  = utcnow()
    vel = result["velocity_features"]
    row.vel_norm_likes_1h    = vel["norm_likes_1h"]
    row.vel_comment_ratio_1h = vel["comment_ratio_1h"]
    row.vel_on_track_score   = vel["on_track_score"]
    db.commit()
    return Stage2Response(prediction_id=row.id, **result)


@router.patch(
    "/predictions/{prediction_id}/outcome",
    response_model=OutcomeResponse,
    summary="Record outcome for a manual prediction",
)
def record_outcome(
    prediction_id: int,
    req: OutcomeRequest,
    db: Session = Depends(get_db),
) -> OutcomeResponse:
    row = db.get(Prediction, prediction_id)
    if row is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"Prediction {prediction_id} not found."
        )
    row.actual_survived     = req.actual_survived
    row.outcome_recorded_at = utcnow()
    db.commit()
    s1_correct = (row.stage1_survives == req.actual_survived) if row.stage1_survives is not None else None
    s2_correct = (row.stage2_survives == req.actual_survived) if row.stage2_survives is not None else None
    return OutcomeResponse(
        prediction_id=row.id,
        stage1_prob=row.stage1_prob,
        stage2_prob=row.stage2_prob,
        actual_survived=req.actual_survived,
        stage1_correct=s1_correct,
        stage2_correct=s2_correct,
    )


@router.get(
    "/predictions",
    response_model=list[PredictionSummary],
    summary="List prediction records",
)
def list_predictions(
    limit: int = Query(50, le=500),
    account_id: Optional[int] = Query(None),
    has_outcome: Optional[bool] = Query(None),
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
    return [prediction_to_summary(r) for r in rows]
