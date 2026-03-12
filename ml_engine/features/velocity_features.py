import numpy as np
import pandas as pd

# Full Stage-2 feature set (all 6h data available)
VELOCITY_FEATURE_COLS = [
    "stage1_prior",
    "norm_likes_1h",
    "norm_likes_3h",
    "norm_likes_6h",
    "like_velocity_1to3",
    "like_velocity_3to6",
    "burst_ratio",
    "comment_ratio_1h",
    "on_track_score",
]

# Observation-window subsets: features available at each checkpoint
# Used by observation_window_analysis to show lift over time
VELOCITY_FEATURES_1H = [
    "stage1_prior",
    "norm_likes_1h",
    "comment_ratio_1h",
    "on_track_score",
]

VELOCITY_FEATURES_3H = [
    "stage1_prior",
    "norm_likes_1h",
    "norm_likes_3h",
    "like_velocity_1to3",
    "burst_ratio",
    # Comment signal: high comment/like ratio = strong community engagement
    "comment_ratio_1h",
    # On-track signal: where is 1h velocity relative to what we'd expect if the
    # post is heading toward the baseline?
    "on_track_score",
]


def build_velocity_features(df: pd.DataFrame, stage1_probs: np.ndarray) -> pd.DataFrame:
    """
    Build Stage-2 input features from early engagement velocity + Stage-1 prior.

    All velocity features are normalized by rolling_weighted_median so they
    carry the same semantic meaning across accounts of different sizes.
    A nano creator getting 50 likes in 1h is very different from a macro
    creator getting 50 likes in 1h — normalization captures this.

    Parameters
    ----------
    df : feature matrix with velocity columns added by simulate_velocity()
    stage1_probs : Stage-1 predicted probabilities for the same rows
    """
    df = df.copy()
    df["stage1_prior"] = stage1_probs

    # Avoid division by zero on rolling_weighted_median
    baseline = df["rolling_weighted_median"].clip(lower=1.0)

    # Normalized cumulative likes
    df["norm_likes_1h"] = df["likes_1h"] / baseline
    df["norm_likes_3h"] = df["likes_3h"] / baseline
    df["norm_likes_6h"] = df["likes_6h"] / baseline

    # Velocity rates (likes per hour in each inter-window period)
    df["like_velocity_1to3"] = (df["likes_3h"] - df["likes_1h"]) / 2.0 / baseline
    df["like_velocity_3to6"] = (df["likes_6h"] - df["likes_3h"]) / 3.0 / baseline

    # Burst ratio: fraction of 3h likes that arrived in the first hour
    df["burst_ratio"] = df["likes_1h"] / (df["likes_3h"] + 1)

    # Comment ratio at 1h: comments signal stronger intent than passive likes
    df["comment_ratio_1h"] = df["comments_1h"] / (df["likes_1h"] + 1)

    # On-track score: is 1h velocity on pace to beat the baseline by 24h?
    # If likes accumulate at the 1h rate for 24h, would total likes exceed baseline?
    # Uses like_rate assumption: likes ≈ 0.08 × reach (approximate average)
    implied_reach_24h = df["likes_1h"] / (0.08 * df["burst_bias"].clip(lower=0.1))
    df["on_track_score"] = implied_reach_24h / baseline

    return df
