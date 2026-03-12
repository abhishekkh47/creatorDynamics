import numpy as np
import pandas as pd

from features.baseline import compute_rolling_baseline

FEATURE_COLS = [
    "rolling_weighted_median",
    "rolling_volatility",
    "posting_frequency",
    "cluster_entropy",
    "cluster_id",
    "posting_time_bucket",
    "content_quality",
]

LABEL_COL = "survived"
CATEGORICAL_FEATURES = ["cluster_id"]


def _rolling_volatility(df: pd.DataFrame, window: int = 10) -> pd.Series:
    result = np.zeros(len(df))

    for _, group in df.groupby("account_id", sort=False):
        group = group.sort_values("timestamp")
        indices = group.index.to_numpy()
        reaches = group["reach_24h"].to_numpy()

        for i in range(len(indices)):
            if i < 2:
                result[indices[i]] = 0.0
            else:
                result[indices[i]] = float(np.std(reaches[max(0, i - window) : i]))

    return pd.Series(result, index=df.index, name="rolling_volatility")


def _posting_frequency(df: pd.DataFrame, window_days: int = 14) -> pd.Series:
    result = np.zeros(len(df))
    window = np.timedelta64(window_days, "D")

    for _, group in df.groupby("account_id", sort=False):
        group = group.sort_values("timestamp")
        indices = group.index.to_numpy()
        timestamps = group["timestamp"].to_numpy()

        for i in range(len(indices)):
            cutoff = timestamps[i] - window
            result[indices[i]] = float(np.sum(timestamps[:i] >= cutoff))

    return pd.Series(result, index=df.index, name="posting_frequency")


def _cluster_entropy(df: pd.DataFrame, window: int = 20) -> pd.Series:
    result = np.zeros(len(df))

    for _, group in df.groupby("account_id", sort=False):
        group = group.sort_values("timestamp")
        indices = group.index.to_numpy()
        cluster_ids = group["cluster_id"].to_numpy()

        for i in range(len(indices)):
            if i < 2:
                result[indices[i]] = 0.0
                continue
            past = cluster_ids[max(0, i - window) : i]
            _, counts = np.unique(past, return_counts=True)
            probs = counts / counts.sum()
            result[indices[i]] = float(-np.sum(probs * np.log(probs + 1e-10)))

    return pd.Series(result, index=df.index, name="cluster_entropy")


def _posting_time_bucket(df: pd.DataFrame) -> pd.Series:
    hours = df["timestamp"].dt.hour
    # 0=night(0-5), 1=morning(6-11), 2=afternoon(12-17), 3=evening(18-23)
    buckets = pd.cut(hours, bins=[0, 6, 12, 18, 24], labels=[0, 1, 2, 3], right=False)
    return buckets.astype(int).rename("posting_time_bucket")


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["rolling_weighted_median"] = compute_rolling_baseline(df)
    df["rolling_volatility"] = _rolling_volatility(df)
    df["posting_frequency"] = _posting_frequency(df)
    df["cluster_entropy"] = _cluster_entropy(df)
    df["posting_time_bucket"] = _posting_time_bucket(df)

    df[LABEL_COL] = (df["reach_24h"] > df["rolling_weighted_median"]).astype(int)

    # Drop first post per account — no baseline available
    df = df.dropna(subset=["rolling_weighted_median"]).reset_index(drop=True)

    return df
