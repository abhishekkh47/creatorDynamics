"""
ORM → Pydantic serializers.

Converts SQLAlchemy row objects to Pydantic response models.
Kept separate from:
  - schemas.py   (defines the shape)
  - db_models.py (defines the tables)
so the mapping logic lives in one place and is reusable across all routers.
"""

from typing import Optional

from db_models import FeatureStore, Prediction
from schemas import PredictionSummary
from utils import fmt


def prediction_to_summary(p: Optional[Prediction]) -> Optional[PredictionSummary]:
    if p is None:
        return None

    s1_correct = s2_correct = None
    if p.actual_survived is not None:
        if p.stage1_survives is not None:
            s1_correct = p.stage1_survives == p.actual_survived
        if p.stage2_survives is not None:
            s2_correct = p.stage2_survives == p.actual_survived

    vel = None
    if p.vel_norm_likes_1h is not None:
        vel = {
            "norm_likes_1h":    p.vel_norm_likes_1h,
            "comment_ratio_1h": p.vel_comment_ratio_1h,
            "on_track_score":   p.vel_on_track_score,
        }

    return PredictionSummary(
        prediction_id=p.id,
        account_id=p.account_id,
        post_id=p.post_id,
        stage1_prob=p.stage1_prob,
        stage1_survives=p.stage1_survives,
        stage2_prob=p.stage2_prob,
        stage2_survives=p.stage2_survives,
        stage2_correction=p.stage2_correction,
        actual_survived=p.actual_survived,
        stage1_correct=s1_correct,
        stage2_correct=s2_correct,
        velocity_features=vel,
        stage1_called_at=fmt(p.stage1_called_at),
        stage2_called_at=fmt(p.stage2_called_at),
        outcome_recorded_at=fmt(p.outcome_recorded_at),
    )


def feature_dict(f: Optional[FeatureStore]) -> Optional[dict]:
    if f is None:
        return None
    return {
        "rolling_weighted_median": f.rolling_weighted_median,
        "rolling_volatility":      f.rolling_volatility,
        "posting_frequency":       f.posting_frequency,
        "cluster_entropy":         f.cluster_entropy,
        "post_count":              f.post_count,
        "computed_at":             fmt(f.computed_at),
    }
