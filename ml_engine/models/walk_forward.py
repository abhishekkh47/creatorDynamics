from typing import List

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from config import WF_LGBM_PARAMS
from features.feature_pipeline import CATEGORICAL_FEATURES, FEATURE_COLS, LABEL_COL


def walk_forward_validation(
    df: pd.DataFrame,
    min_train_days: int = 90,
    window_days: int = 30,
) -> List[dict]:
    """
    Roll a training window forward one month at a time.

    For each window:
      - Train on ALL data prior to window_start
      - Test on the next window_days of data
      - Record AUC and log loss

    Requires at least min_train_days of history before the first test window.
    This tests whether model quality is stable over time, not just on one split.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    start_ts = df["timestamp"].min()
    end_ts = df["timestamp"].max()

    results = []
    window_start = start_ts + pd.Timedelta(days=min_train_days)

    while window_start + pd.Timedelta(days=window_days) <= end_ts:
        window_end = window_start + pd.Timedelta(days=window_days)

        train = df[df["timestamp"] < window_start]
        test = df[(df["timestamp"] >= window_start) & (df["timestamp"] < window_end)]

        if len(train) < 200 or len(test) < 50:
            window_start = window_end
            continue

        if train[LABEL_COL].nunique() < 2 or test[LABEL_COL].nunique() < 2:
            window_start = window_end
            continue

        model = lgb.LGBMClassifier(**WF_LGBM_PARAMS)
        model.fit(
            train[FEATURE_COLS],
            train[LABEL_COL],
            categorical_feature=CATEGORICAL_FEATURES,
            callbacks=[lgb.log_evaluation(period=-1)],
        )

        probs = model.predict_proba(test[FEATURE_COLS])[:, 1]
        auc = roc_auc_score(test[LABEL_COL], probs)
        ll = log_loss(test[LABEL_COL], probs)

        results.append(
            {
                "window_start": window_start.strftime("%Y-%m-%d"),
                "window_end": window_end.strftime("%Y-%m-%d"),
                "train_size": len(train),
                "test_size": len(test),
                "roc_auc": round(float(auc), 4),
                "log_loss": round(float(ll), 4),
            }
        )

        window_start = window_end

    return results


def summarise_walk_forward(results: List[dict]) -> dict:
    if not results:
        return {}

    aucs = [r["roc_auc"] for r in results]
    losses = [r["log_loss"] for r in results]

    return {
        "n_windows": len(results),
        "auc_mean": round(float(np.mean(aucs)), 4),
        "auc_std": round(float(np.std(aucs)), 4),
        "auc_min": round(float(np.min(aucs)), 4),
        "auc_max": round(float(np.max(aucs)), 4),
        "log_loss_mean": round(float(np.mean(losses)), 4),
        "stable": bool(np.std(aucs) < 0.05),  # flag if variance is low
    }
