"""
Rolling feature computation from real post history.

Ports the logic from ml_engine/features/ to operate on database records
instead of a synthetic pandas DataFrame.

Called every time a post's 24h reach is recorded, to keep the feature
store current for that account's next prediction.
"""

import math
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
from sqlalchemy.orm import Session

from db_models import FeatureStore, Post

# Matches ml_engine/config.py  DECAY_LAMBDA
DECAY_LAMBDA: float = 0.03


# ---------------------------------------------------------------------------
# Core feature computation
# ---------------------------------------------------------------------------

def _weighted_median(values: list[float], weights: list[float]) -> float:
    """
    Weighted median — ported from ml_engine/features/baseline.py.

    Sort (value, weight) pairs by value, accumulate weights, return the
    value at which cumulative weight first reaches half the total.
    """
    if not values:
        return 0.0

    pairs = sorted(zip(values, weights), key=lambda x: x[0])
    total = sum(weights)
    cumulative = 0.0
    for val, w in pairs:
        cumulative += w
        if cumulative >= total / 2:
            return val
    return pairs[-1][0]


def _age_days(post_time: datetime, reference: datetime) -> float:
    """Days between post_time and reference (reference is always later)."""
    delta = reference - post_time
    return max(delta.total_seconds() / 86400.0, 0.0)


def _cluster_entropy(cluster_ids: list[int]) -> float:
    """Shannon entropy of a cluster ID distribution — ported from feature_pipeline.py."""
    if not cluster_ids:
        return 0.0
    _, counts = np.unique(cluster_ids, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs + 1e-10)))


def compute_rolling_features(account_id: int, db: Session) -> Optional[dict]:
    """
    Compute current rolling features for an account from its post history.

    Only posts with known reach_24h are used (that's the signal the baseline
    needs — we can't include a post whose outcome we don't know yet).

    Returns None if the account has fewer than 2 posts with known reach
    (not enough history to compute meaningful features).
    """
    posts = (
        db.query(Post)
        .filter(Post.account_id == account_id, Post.reach_24h.isnot(None))
        .order_by(Post.posted_at.asc())
        .all()
    )

    if len(posts) < 2:
        return None

    reference_time = posts[-1].posted_at
    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=timezone.utc)

    # ------------------------------------------------------------------
    # rolling_weighted_median
    # ------------------------------------------------------------------
    # Compute for each post: the weighted median of ALL PRIOR posts' reach.
    # The feature for the next post uses all known history.
    reaches = [float(p.reach_24h) for p in posts]
    times   = [p.posted_at for p in posts]

    weights = []
    for t in times:
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        age = _age_days(t, reference_time)
        weights.append(math.exp(-DECAY_LAMBDA * age))

    rolling_weighted_median = _weighted_median(reaches, weights)

    # ------------------------------------------------------------------
    # rolling_volatility  (std of last 10 reach values)
    # ------------------------------------------------------------------
    recent = reaches[-10:]
    rolling_volatility = float(np.std(recent)) if len(recent) >= 2 else 0.0

    # ------------------------------------------------------------------
    # posting_frequency  (posts in last 14 days, not counting the last)
    # ------------------------------------------------------------------
    cutoff = reference_time - timedelta(days=14)
    posting_frequency = sum(
        1 for t in times[:-1]
        if (t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t) >= cutoff
    )

    # ------------------------------------------------------------------
    # cluster_entropy  (Shannon entropy over last 20 posts' cluster IDs)
    # ------------------------------------------------------------------
    recent_clusters = [
        p.cluster_id for p in posts[-20:] if p.cluster_id is not None
    ]
    cluster_entropy = _cluster_entropy(recent_clusters)

    return {
        "rolling_weighted_median": round(rolling_weighted_median, 2),
        "rolling_volatility":      round(rolling_volatility, 2),
        "posting_frequency":       float(posting_frequency),
        "cluster_entropy":         round(cluster_entropy, 4),
        "post_count":              len(posts),
    }


# ---------------------------------------------------------------------------
# Feature store upsert
# ---------------------------------------------------------------------------

def upsert_feature_store(account_id: int, db: Session) -> Optional[FeatureStore]:
    """
    Compute and save (or update) the feature store entry for an account.

    Called after every post's 24h reach is recorded.
    Returns the updated FeatureStore row, or None if not enough history.
    """
    features = compute_rolling_features(account_id, db)
    if features is None:
        return None

    row = db.query(FeatureStore).filter(FeatureStore.account_id == account_id).first()

    if row is None:
        row = FeatureStore(account_id=account_id)
        db.add(row)

    row.rolling_weighted_median = features["rolling_weighted_median"]
    row.rolling_volatility      = features["rolling_volatility"]
    row.posting_frequency       = features["posting_frequency"]
    row.cluster_entropy         = features["cluster_entropy"]
    row.post_count              = features["post_count"]
    row.computed_at             = datetime.now(timezone.utc)

    db.commit()
    db.refresh(row)
    return row
