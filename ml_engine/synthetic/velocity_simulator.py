import numpy as np
import pandas as pd

# Observation windows in hours
VELOCITY_HOURS = [1, 3, 6]

# Per-tier burst bias ranges: strong clusters front-load engagement faster
_TIER_BURST = {
    "strong": (0.60, 0.90),
    "medium": (0.40, 0.70),
    "weak":   (0.20, 0.50),
}


def _cumulative_fraction(t_hours: np.ndarray, lambda_: np.ndarray) -> np.ndarray:
    """
    Fraction of total engagement accumulated by t_hours post-publish.
    Models decay rate λ = burst_bias × 0.5.
    Higher burst_bias → engagement front-loaded into first few hours.
    """
    return 1.0 - np.exp(-lambda_ * t_hours)


def simulate_velocity(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    For each post, simulate cumulative engagement at 1h, 3h, 6h after publishing.

    Driven by:
      - Final reach (reach_24h → total_likes, total_comments)
      - Cluster tier (strong clusters get more burst-biased curves)
      - Per-post noise (even identical posts vary in their early trajectories)

    Returns df with added columns:
      likes_1h, likes_3h, likes_6h
      comments_1h, comments_3h, comments_6h
      burst_bias (for diagnostics)
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    df = df.copy().reset_index(drop=True)

    # --- Assign per-post burst_bias from cluster tier ---
    burst_bias = np.full(n, 0.5)
    tiers = df["cluster_tier"].values if "cluster_tier" in df.columns else np.full(n, "medium")
    for tier, (lo, hi) in _TIER_BURST.items():
        mask = tiers == tier
        if mask.any():
            burst_bias[mask] = rng.uniform(lo, hi, int(mask.sum()))

    lambda_ = burst_bias * 0.5

    # --- Cumulative fractions at each window ---
    f1 = _cumulative_fraction(1.0, lambda_)
    f3 = _cumulative_fraction(3.0, lambda_)
    f6 = _cumulative_fraction(6.0, lambda_)

    total_likes    = df["likes"].values.astype(float)
    total_comments = df["comments"].values.astype(float)

    # --- Likes (observations get progressively less noisy as sample size grows) ---
    likes_1h_raw = (total_likes * f1 * rng.lognormal(0.0, 0.25, n)).astype(int)
    likes_3h_raw = (total_likes * f3 * rng.lognormal(0.0, 0.18, n)).astype(int)
    likes_6h_raw = (total_likes * f6 * rng.lognormal(0.0, 0.12, n)).astype(int)

    # Enforce monotonically non-decreasing
    likes_1h = np.maximum(likes_1h_raw, 0)
    likes_3h = np.maximum(likes_3h_raw, likes_1h)
    likes_6h = np.maximum(likes_6h_raw, likes_3h)

    # --- Comments (noisier signal) ---
    comm_1h_raw = (total_comments * f1 * rng.lognormal(0.0, 0.35, n)).astype(int)
    comm_3h_raw = (total_comments * f3 * rng.lognormal(0.0, 0.25, n)).astype(int)
    comm_6h_raw = (total_comments * f6 * rng.lognormal(0.0, 0.18, n)).astype(int)

    comments_1h = np.maximum(comm_1h_raw, 0)
    comments_3h = np.maximum(comm_3h_raw, comments_1h)
    comments_6h = np.maximum(comm_6h_raw, comments_3h)

    df["likes_1h"]     = likes_1h
    df["likes_3h"]     = likes_3h
    df["likes_6h"]     = likes_6h
    df["comments_1h"]  = comments_1h
    df["comments_3h"]  = comments_3h
    df["comments_6h"]  = comments_6h
    df["burst_bias"]   = burst_bias

    return df
