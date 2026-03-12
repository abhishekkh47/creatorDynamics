"""
Pydantic request/response schemas for all API endpoints.
"""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Account
# ---------------------------------------------------------------------------

class AccountCreate(BaseModel):
    username: str = Field(..., description="Instagram username (unique).", example="john_creates")
    instagram_id: Optional[str] = Field(
        None, description="Instagram numeric account ID. Optional — fill in when available.",
        example="17841400008460056",
    )
    follower_count: int = Field(..., gt=0, description="Current follower count.", example=45000)
    cluster_tier: Literal["strong", "medium", "weak"] = Field(
        "medium",
        description="Performance tier of this account's primary niche. "
                    "Drives the burst expectation in Stage-2 prediction.",
    )

    model_config = {"json_schema_extra": {"example": {
        "username": "john_creates",
        "follower_count": 45000,
        "cluster_tier": "medium",
    }}}


class AccountResponse(BaseModel):
    id: int
    username: str
    instagram_id: Optional[str]
    follower_count: int
    cluster_tier: str
    created_at: str
    features: Optional[dict] = Field(
        None,
        description="Current rolling features from the feature store. "
                    "Null if fewer than 2 posts with known 24h reach exist.",
    )
    post_count: int = Field(
        0, description="Total posts ingested for this account."
    )


# ---------------------------------------------------------------------------
# Post ingestion
# ---------------------------------------------------------------------------

class PostIngest(BaseModel):
    """
    Data available at posting time. Stage-1 prediction fires automatically
    if the account's feature store has been populated.
    """
    instagram_post_id: Optional[str] = Field(
        None, description="Instagram post ID. Optional — fill in when available."
    )
    posted_at: datetime = Field(
        ..., description="Timestamp when the post went live (ISO 8601 with timezone).",
        example="2026-03-12T14:30:00+05:30",
    )
    content_quality: float = Field(
        ..., ge=0.0, le=1.0,
        description="Content quality score [0–1]. Estimated from hook strength, "
                    "caption quality, hashtag relevance, etc.",
        example=0.75,
    )
    cluster_id: int = Field(
        ..., ge=0,
        description="Topic cluster ID for this post (0–19).",
        example=4,
    )

    model_config = {"json_schema_extra": {"example": {
        "posted_at": "2026-03-12T14:30:00+05:30",
        "content_quality": 0.75,
        "cluster_id": 4,
    }}}


class PostResponse(BaseModel):
    id: int
    account_id: int
    instagram_post_id: Optional[str]
    posted_at: str
    content_quality: Optional[float]
    cluster_id: Optional[int]
    reach_24h: Optional[int]
    likes_1h: Optional[int]
    comments_1h: Optional[int]
    created_at: str
    prediction: Optional["PredictionSummary"] = Field(
        None, description="Linked prediction. Populated at ingest (Stage-1) "
                          "and updated at 1h velocity update (Stage-2)."
    )


# ---------------------------------------------------------------------------
# Velocity update  (T+1h)
# ---------------------------------------------------------------------------

class VelocityUpdate(BaseModel):
    """First-hour engagement counts. Triggers Stage-2 auto-prediction."""
    likes_1h: int = Field(..., ge=0, example=280)
    comments_1h: int = Field(..., ge=0, example=14)

    model_config = {"json_schema_extra": {"example": {"likes_1h": 280, "comments_1h": 14}}}


# ---------------------------------------------------------------------------
# Reach update  (T+24h)
# ---------------------------------------------------------------------------

class ReachUpdate(BaseModel):
    """
    24h reach. Closes the prediction lifecycle and triggers feature store
    recomputation so the account's next prediction uses fresh baselines.
    """
    reach_24h: int = Field(..., gt=0, example=12000)
    likes_24h: Optional[int] = Field(None, ge=0, example=890)
    comments_24h: Optional[int] = Field(None, ge=0, example=45)

    model_config = {"json_schema_extra": {"example": {
        "reach_24h": 12000, "likes_24h": 890, "comments_24h": 45,
    }}}


class ReachUpdateResponse(BaseModel):
    post_id: int
    reach_24h: int
    actual_survived: bool = Field(
        description="Did the post beat the account's rolling baseline?"
    )
    rolling_weighted_median_at_time: Optional[float] = Field(
        description="The baseline that was active when this post was predicted."
    )
    stage1_correct: Optional[bool]
    stage2_correct: Optional[bool]
    feature_store_updated: bool = Field(
        description="True if the feature store was successfully recomputed "
                    "with this post included."
    )


# ---------------------------------------------------------------------------
# Prediction summary  (embedded in PostResponse + list endpoints)
# ---------------------------------------------------------------------------

class PredictionSummary(BaseModel):
    prediction_id: int
    account_id: Optional[int]
    post_id: Optional[int]
    stage1_prob: Optional[float]
    stage1_survives: Optional[bool]
    stage2_prob: Optional[float]
    stage2_survives: Optional[bool]
    stage2_correction: Optional[float]
    actual_survived: Optional[bool]
    stage1_correct: Optional[bool]
    stage2_correct: Optional[bool]
    velocity_features: Optional[dict] = None
    stage1_called_at: Optional[str]
    stage2_called_at: Optional[str]
    outcome_recorded_at: Optional[str]


PostResponse.model_rebuild()


# ---------------------------------------------------------------------------
# Manual prediction endpoints  (backward-compatible — no DB required)
# ---------------------------------------------------------------------------

class Stage1Request(BaseModel):
    rolling_weighted_median: float = Field(..., gt=0, example=8500.0)
    rolling_volatility: float = Field(..., ge=0, example=1200.0)
    posting_frequency: float = Field(..., ge=0, example=5.0)
    cluster_entropy: float = Field(..., ge=0, example=1.8)
    content_quality: float = Field(..., ge=0.0, le=1.0, example=0.72)
    cluster_id: int = Field(..., ge=0, example=3)
    hour_of_day: Optional[int] = Field(None, ge=0, le=23, example=14)


class Stage1Response(BaseModel):
    prediction_id: int
    survival_probability: float
    survives: bool
    confidence: Literal["high", "medium", "low"]
    posting_time_bucket: int
    model: str = "stage1"


class Stage2Request(BaseModel):
    prediction_id: int = Field(..., example=1)
    stage1_prior: float = Field(..., ge=0.0, le=1.0, example=0.62)
    rolling_weighted_median: float = Field(..., gt=0, example=8500.0)
    likes_1h: int = Field(..., ge=0, example=340)
    comments_1h: int = Field(..., ge=0, example=18)
    cluster_tier: Literal["strong", "medium", "weak"] = Field(..., example="medium")


class Stage2Response(BaseModel):
    prediction_id: int
    survival_probability: float
    survives: bool
    stage1_prior: float
    correction: float
    confidence: Literal["high", "medium", "low"]
    velocity_features: dict
    model: str = "stage2_1h"


# ---------------------------------------------------------------------------
# Outcome recording  (manual, for predictions not linked to a real Post)
# ---------------------------------------------------------------------------

class OutcomeRequest(BaseModel):
    actual_survived: bool = Field(..., example=True)


class OutcomeResponse(BaseModel):
    prediction_id: int
    stage1_prob: Optional[float]
    stage2_prob: Optional[float]
    actual_survived: bool
    stage1_correct: Optional[bool]
    stage2_correct: Optional[bool]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    models: dict
    models_dir: str
