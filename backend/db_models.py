"""
Database table definitions (SQLAlchemy ORM).

One table: Prediction.

A Prediction row is created when /predict/stage1 is called.
It is updated when /predict/stage2 is called (linked via prediction_id).
It is updated again when the actual outcome is recorded via PATCH /predictions/{id}/outcome.

This gives us a complete lifecycle record for every prediction the system makes —
which is the retraining dataset for Phase 4+ (real data).

Row lifecycle:
    stage1_called_at set   →  stage2_called_at set   →  outcome_recorded_at set
         (at post time)         (60 min later)             (24h later)
"""

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Prediction(Base):
    __tablename__ = "predictions"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ---------------------------------------------------------------------------
    # Identifiers (optional now — populated when real data arrives in Phase 4)
    # ---------------------------------------------------------------------------
    account_id: Mapped[str | None] = mapped_column(
        String, nullable=True, index=True,
        comment="Creator account ID. Null in synthetic phase.",
    )
    post_id: Mapped[str | None] = mapped_column(
        String, nullable=True, index=True,
        comment="Post ID. Null in synthetic phase.",
    )

    # ---------------------------------------------------------------------------
    # Stage-1 prediction
    # ---------------------------------------------------------------------------
    stage1_prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    stage1_survives: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    stage1_confidence: Mapped[str | None] = mapped_column(String, nullable=True)
    stage1_called_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Stage-1 input features — snapshot for audit and future retraining
    feat_rolling_weighted_median: Mapped[float | None] = mapped_column(Float, nullable=True)
    feat_rolling_volatility: Mapped[float | None] = mapped_column(Float, nullable=True)
    feat_posting_frequency: Mapped[float | None] = mapped_column(Float, nullable=True)
    feat_cluster_entropy: Mapped[float | None] = mapped_column(Float, nullable=True)
    feat_content_quality: Mapped[float | None] = mapped_column(Float, nullable=True)
    feat_cluster_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    feat_posting_time_bucket: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # ---------------------------------------------------------------------------
    # Stage-2 prediction  (filled when /predict/stage2 is called)
    # ---------------------------------------------------------------------------
    stage2_prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    stage2_survives: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    stage2_correction: Mapped[float | None] = mapped_column(Float, nullable=True)
    stage2_confidence: Mapped[str | None] = mapped_column(String, nullable=True)
    stage2_called_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Stage-2 input features — raw engagement counts
    feat_likes_1h: Mapped[int | None] = mapped_column(Integer, nullable=True)
    feat_comments_1h: Mapped[int | None] = mapped_column(Integer, nullable=True)
    feat_cluster_tier: Mapped[str | None] = mapped_column(String, nullable=True)

    # Stage-2 derived velocity features — stored for frontend display and debugging
    vel_norm_likes_1h: Mapped[float | None] = mapped_column(Float, nullable=True)
    vel_comment_ratio_1h: Mapped[float | None] = mapped_column(Float, nullable=True)
    vel_on_track_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ---------------------------------------------------------------------------
    # Ground truth  (filled 24h after posting — the retraining signal)
    # ---------------------------------------------------------------------------
    actual_survived: Mapped[bool | None] = mapped_column(
        Boolean, nullable=True,
        comment="Did the post actually beat the rolling baseline at 24h? "
                "Null until outcome is recorded.",
    )
    outcome_recorded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    # ---------------------------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------------------------
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow,
    )

    def __repr__(self) -> str:
        return (
            f"<Prediction id={self.id} "
            f"s1={self.stage1_prob:.2f} "
            f"s2={self.stage2_prob:.2f if self.stage2_prob is not None else 'None'} "
            f"actual={self.actual_survived}>"
        )
