"""
Database ORM models — four tables, one lifecycle:

  Account  →  Post  →  FeatureStore  (updated after every 24h reach record)
                  ↓
              Prediction  (Stage-1 on ingest, Stage-2 after 1h velocity)
"""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base
from utils import utcnow


# ---------------------------------------------------------------------------
# Account
# ---------------------------------------------------------------------------

class Account(Base):
    """
    A registered Instagram creator account.

    Rolling features for this account are stored in FeatureStore and
    recomputed every time a new post's 24h reach is recorded.
    """
    __tablename__ = "accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String, unique=True, index=True)
    instagram_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, unique=True)
    follower_count: Mapped[int] = mapped_column(Integer)
    # Cluster tier drives burst expectation in Stage-2 (strong/medium/weak niche)
    cluster_tier: Mapped[str] = mapped_column(String, default="medium")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    posts: Mapped[List["Post"]] = relationship("Post", back_populates="account")
    feature: Mapped[Optional["FeatureStore"]] = relationship(
        "FeatureStore", back_populates="account", uselist=False
    )


# ---------------------------------------------------------------------------
# Post
# ---------------------------------------------------------------------------

class Post(Base):
    """
    A single Instagram Reel or post.

    Data arrives in three phases:
      1. At posting time — content_quality, cluster_id, posted_at
      2. At T+1h — likes_1h, comments_1h  (triggers Stage-2 prediction)
      3. At T+24h — reach_24h             (closes lifecycle + updates FeatureStore)
    """
    __tablename__ = "posts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(Integer, ForeignKey("accounts.id"), index=True)
    instagram_post_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    posted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    # Content attributes (known at posting time)
    content_quality: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cluster_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # 1h engagement (filled ~60 min after posting)
    likes_1h: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    comments_1h: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


    # 24h outcome (filled ~24h after posting)
    reach_24h: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    likes_24h: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    comments_24h: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    account: Mapped["Account"] = relationship("Account", back_populates="posts")
    prediction: Mapped[Optional["Prediction"]] = relationship(
        "Prediction", back_populates="post", uselist=False
    )


# ---------------------------------------------------------------------------
# FeatureStore
# ---------------------------------------------------------------------------

class FeatureStore(Base):
    """
    Latest rolling ML features for each account.

    One row per account (upserted after every 24h reach record).
    These are the features passed to Stage-1 when a new post is ingested.

    Mirrors ml_engine/features/feature_pipeline.py FEATURE_COLS, computed
    from the account's actual post history rather than synthetic data.
    """
    __tablename__ = "feature_store"
    __table_args__ = (UniqueConstraint("account_id", name="uq_feature_store_account"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("accounts.id"), index=True
    )
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    rolling_weighted_median: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rolling_volatility: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    posting_frequency: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cluster_entropy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    post_count: Mapped[int] = mapped_column(Integer, default=0,
        comment="Number of posts with known reach_24h used to compute these features")

    account: Mapped["Account"] = relationship("Account", back_populates="feature")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

class Prediction(Base):
    """
    Full prediction lifecycle for a single post.

    Created when Stage-1 runs (at post ingest time).
    Updated when Stage-2 runs (at T+1h velocity update).
    Closed when actual outcome is recorded (at T+24h reach update).
    """
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("accounts.id"), nullable=True, index=True
    )
    post_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("posts.id"), nullable=True, index=True
    )

    # Stage-1 results
    stage1_prob: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    stage1_survives: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    stage1_confidence: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    stage1_called_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Stage-1 input feature snapshot (for audit and future retraining)
    feat_rolling_weighted_median: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    feat_rolling_volatility: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    feat_posting_frequency: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    feat_cluster_entropy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    feat_content_quality: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    feat_cluster_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    feat_posting_time_bucket: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Stage-2 results (filled at T+1h)
    stage2_prob: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    stage2_survives: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    stage2_correction: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    stage2_confidence: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    stage2_called_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Stage-2 derived velocity features (for frontend display)
    vel_norm_likes_1h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    vel_comment_ratio_1h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    vel_on_track_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Ground truth (filled at T+24h)
    actual_survived: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    outcome_recorded_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    post: Mapped[Optional["Post"]] = relationship("Post", back_populates="prediction")
