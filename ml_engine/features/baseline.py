import numpy as np
import pandas as pd

from config import DECAY_LAMBDA


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    order = np.argsort(values)
    sorted_vals = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    midpoint = cumulative[-1] / 2.0
    idx = int(np.searchsorted(cumulative, midpoint))
    idx = np.clip(idx, 0, len(sorted_vals) - 1)
    return float(sorted_vals[idx])


def compute_rolling_baseline(df: pd.DataFrame, decay_lambda: float = DECAY_LAMBDA) -> pd.Series:
    """
    Per-account rolling weighted median of reach_24h.
    weight = exp(-λ * age_in_days), computed over strictly past posts only.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    baselines = np.full(len(df), np.nan)

    for _, group in df.groupby("account_id", sort=False):
        group = group.sort_values("timestamp")
        indices = group.index.to_numpy()
        timestamps = group["timestamp"].to_numpy()
        reaches = group["reach_24h"].to_numpy()

        for i in range(1, len(indices)):
            global_i = indices[i]
            current_ts = timestamps[i]
            past_reaches = reaches[:i]
            ages = (current_ts - timestamps[:i]) / np.timedelta64(1, "D")
            weights = np.exp(-decay_lambda * ages)
            baselines[global_i] = _weighted_median(past_reaches, weights)

    return pd.Series(baselines, index=df.index, name="rolling_weighted_median")
