"""
Pydantic request/response schemas for all API endpoints.

Design principle: the API accepts the minimum raw inputs needed and computes
derived features internally. Callers should not need to know what
`norm_likes_1h` means — they just pass `likes_1h` and the backend normalizes.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Stage-1  (pre-post prediction)
# ---------------------------------------------------------------------------

class Stage1Request(BaseModel):
    """
    Features available before a post goes live.

    The rolling features (rolling_weighted_median, rolling_volatility,
    posting_frequency, cluster_entropy) are computed from the account's
    post history. In production these come from a feature store that is
    updated after every post.
    """

    # Account-level rolling features (from post history)
    rolling_weighted_median: float = Field(
        ..., gt=0,
        description="Account's exponentially-weighted median reach over recent posts. "
                    "Acts as the personal baseline — survival is defined as reach > this value.",
        example=8500.0,
    )
    rolling_volatility: float = Field(
        ..., ge=0,
        description="Standard deviation of the account's recent log-reach. "
                    "Higher = more unpredictable account.",
        example=1200.0,
    )
    posting_frequency: float = Field(
        ..., ge=0,
        description="Number of posts the account made in the past 14 days.",
        example=5.0,
    )
    cluster_entropy: float = Field(
        ..., ge=0,
        description="Shannon entropy of the account's topic cluster distribution "
                    "over recent posts. High entropy = diverse content.",
        example=1.8,
    )

    # Post-level features
    content_quality: float = Field(
        ..., ge=0.0, le=1.0,
        description="Content quality score [0–1]. Estimated from production signals "
                    "(e.g. hook strength, caption length, hashtag relevance).",
        example=0.72,
    )
    cluster_id: int = Field(
        ..., ge=0,
        description="Topic cluster ID for this post (0–19 for the 20-cluster model).",
        example=3,
    )
    hour_of_day: Optional[int] = Field(
        None, ge=0, le=23,
        description="Hour of posting in local time (0–23). Used to derive the "
                    "posting time bucket (night/morning/afternoon/evening). "
                    "If omitted, defaults to morning bucket.",
        example=14,
    )

    model_config = {"json_schema_extra": {"example": {
        "rolling_weighted_median": 8500.0,
        "rolling_volatility": 1200.0,
        "posting_frequency": 5.0,
        "cluster_entropy": 1.8,
        "content_quality": 0.72,
        "cluster_id": 3,
        "hour_of_day": 14,
    }}}


class Stage1Response(BaseModel):
    prediction_id: int = Field(
        description="Unique ID for this prediction row in the database. "
                    "Pass this as prediction_id in the /predict/stage2 request "
                    "to link the two predictions together."
    )
    survival_probability: float = Field(
        description="Predicted probability that this post will exceed the account's "
                    "rolling baseline reach within 24 hours. Range [0, 1]."
    )
    survives: bool = Field(
        description="Binary prediction at the recommended threshold (0.35). "
                    "True = model predicts the post will outperform baseline."
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence in the prediction. "
                    "High = probability far from threshold (≥0.25 away). "
                    "Low = probability near threshold (<0.10 away)."
    )
    posting_time_bucket: int = Field(
        description="Derived time bucket used as input: 0=night, 1=morning, "
                    "2=afternoon, 3=evening."
    )
    model: str = Field(default="stage1")


# ---------------------------------------------------------------------------
# Stage-2  (1h-post prediction)
# ---------------------------------------------------------------------------

class Stage2Request(BaseModel):
    """
    Stage-1 prior + first-hour engagement data.

    The backend uses the 1h model (AUC 0.978, trained with 4 features).
    93% of Stage-2's total lift is available at 1 hour — no need to wait 6h.
    """

    # Link back to the Stage-1 prediction row in the database
    prediction_id: int = Field(
        ...,
        description="The prediction_id returned by /predict/stage1. "
                    "Links this correction to the original pre-post prediction "
                    "so both are stored together in the database.",
        example=1,
    )

    # Stage-1 prior — the prediction to be revised
    stage1_prior: float = Field(
        ..., ge=0.0, le=1.0,
        description="Stage-1 survival probability from /predict/stage1. "
                    "This is the prior that Stage-2 will update.",
        example=0.62,
    )

    # Account baseline (needed to normalize velocity signals)
    rolling_weighted_median: float = Field(
        ..., gt=0,
        description="Same value used in the Stage-1 call. "
                    "Required to normalize raw engagement counts.",
        example=8500.0,
    )

    # 1h engagement counts (raw, not normalized — backend handles normalization)
    likes_1h: int = Field(
        ..., ge=0,
        description="Total likes received in the first hour after posting.",
        example=340,
    )
    comments_1h: int = Field(
        ..., ge=0,
        description="Total comments received in the first hour after posting.",
        example=18,
    )

    # Cluster tier — used to estimate expected burst fraction
    cluster_tier: Literal["strong", "medium", "weak"] = Field(
        ...,
        description="Performance tier of the post's topic cluster. "
                    "Used to estimate how front-loaded engagement should be "
                    "(strong clusters burst faster in the first hour).",
        example="medium",
    )

    model_config = {"json_schema_extra": {"example": {
        "stage1_prior": 0.62,
        "rolling_weighted_median": 8500.0,
        "likes_1h": 340,
        "comments_1h": 18,
        "cluster_tier": "medium",
    }}}


class Stage2Response(BaseModel):
    prediction_id: int = Field(
        description="The same prediction_id from the Stage-1 call. "
                    "The database row has been updated with Stage-2 results."
    )
    survival_probability: float = Field(
        description="Stage-2 corrected probability that this post will exceed "
                    "the account's baseline reach. Updated using 1h velocity."
    )
    survives: bool = Field(
        description="Binary prediction at the recommended threshold (0.55). "
                    "Higher threshold than Stage-1 because Stage-2 precision is much higher."
    )
    stage1_prior: float = Field(
        description="The Stage-1 prior that was revised."
    )
    correction: float = Field(
        description="stage2_probability − stage1_prior. "
                    "Positive = velocity evidence revised prediction upward. "
                    "Negative = early engagement is weaker than expected."
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence in the corrected prediction."
    )
    velocity_features: dict = Field(
        description="The normalized velocity features computed from raw inputs. "
                    "Useful for debugging and frontend display."
    )
    model: str = Field(default="stage2_1h")


# ---------------------------------------------------------------------------
# Outcome recording  (called 24h after posting)
# ---------------------------------------------------------------------------

class OutcomeRequest(BaseModel):
    """
    Record what actually happened 24h after the post went live.

    This is the ground truth that closes the prediction lifecycle.
    Over time, the collection of (prediction, actual_outcome) pairs becomes
    the retraining dataset for Phase 4.
    """
    actual_survived: bool = Field(
        ...,
        description="Did the post's 24h reach actually exceed the account's "
                    "rolling_weighted_median baseline? True = outperformed.",
        example=True,
    )


class OutcomeResponse(BaseModel):
    prediction_id: int
    stage1_prob: Optional[float]
    stage2_prob: Optional[float]
    actual_survived: bool
    stage1_correct: Optional[bool] = Field(
        description="Whether Stage-1's binary prediction matched the actual outcome."
    )
    stage2_correct: Optional[bool] = Field(
        description="Whether Stage-2's binary prediction matched the actual outcome. "
                    "Null if Stage-2 was never called for this prediction."
    )


# ---------------------------------------------------------------------------
# Prediction list  (GET /predictions)
# ---------------------------------------------------------------------------

class PredictionSummary(BaseModel):
    prediction_id: int
    account_id: Optional[str]
    post_id: Optional[str]
    stage1_prob: Optional[float]
    stage1_survives: Optional[bool]
    stage2_prob: Optional[float]
    stage2_survives: Optional[bool]
    stage2_correction: Optional[float]
    actual_survived: Optional[bool]
    stage1_called_at: Optional[str]
    stage2_called_at: Optional[str]
    outcome_recorded_at: Optional[str]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = Field(
        description="'ok' if all models are loaded. 'degraded' if any model failed to load."
    )
    models: dict = Field(
        description="Per-model load status."
    )
    models_dir: str = Field(
        description="Absolute path to the directory where model files are loaded from."
    )
