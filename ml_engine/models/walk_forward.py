from typing import List

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from config import WF_LGBM_PARAMS
from features.feature_pipeline import CATEGORICAL_FEATURES, FEATURE_COLS, LABEL_COL
from features.velocity_features import VELOCITY_FEATURE_COLS, build_velocity_features


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


def walk_forward_stage2(
    df_vel: pd.DataFrame,
    model_s2: lgb.LGBMClassifier,
    min_train_days: int = 90,
    window_days: int = 30,
) -> List[dict]:
    """
    Walk-forward for the full Stage-1 + Stage-2 pipeline.

    For each window:
      - Retrain Stage-1 on all prior data
      - Get Stage-1 out-of-sample predictions on the test window
      - Build velocity features using those OOS predictions
      - Apply the fixed Stage-2 model
      - Record Stage-1 AUC, Stage-2 AUC, and the lift

    Tests whether the lift Stage-2 provides is stable over time,
    not just on one held-out split.
    """
    df_vel = df_vel.sort_values("timestamp").reset_index(drop=True)
    start_ts = df_vel["timestamp"].min()
    end_ts   = df_vel["timestamp"].max()

    results = []
    window_start = start_ts + pd.Timedelta(days=min_train_days)

    while window_start + pd.Timedelta(days=window_days) <= end_ts:
        window_end = window_start + pd.Timedelta(days=window_days)

        train = df_vel[df_vel["timestamp"] < window_start]
        test  = df_vel[(df_vel["timestamp"] >= window_start) & (df_vel["timestamp"] < window_end)]

        if len(train) < 200 or len(test) < 50:
            window_start = window_end
            continue
        if train[LABEL_COL].nunique() < 2 or test[LABEL_COL].nunique() < 2:
            window_start = window_end
            continue

        # Retrain Stage-1 on prior data
        s1_model = lgb.LGBMClassifier(**WF_LGBM_PARAMS)
        s1_model.fit(
            train[FEATURE_COLS], train[LABEL_COL],
            categorical_feature=CATEGORICAL_FEATURES,
            callbacks=[lgb.log_evaluation(period=-1)],
        )

        s1_probs = s1_model.predict_proba(test[FEATURE_COLS])[:, 1]
        s1_auc   = roc_auc_score(test[LABEL_COL], s1_probs)

        # Build Stage-2 features and apply fixed Stage-2 model
        test_s2  = build_velocity_features(test, s1_probs)
        s2_probs = model_s2.predict_proba(test_s2[VELOCITY_FEATURE_COLS])[:, 1]
        s2_auc   = roc_auc_score(test[LABEL_COL], s2_probs)

        results.append({
            "window_start": window_start.strftime("%Y-%m-%d"),
            "window_end":   window_end.strftime("%Y-%m-%d"),
            "train_size":   len(train),
            "test_size":    len(test),
            "stage1_auc":   round(float(s1_auc), 4),
            "stage2_auc":   round(float(s2_auc), 4),
            "auc_lift":     round(float(s2_auc - s1_auc), 4),
        })

        window_start = window_end

    return results


def summarise_walk_forward_stage2(results: List[dict]) -> dict:
    if not results:
        return {}
    s1_aucs   = [r["stage1_auc"] for r in results]
    s2_aucs   = [r["stage2_auc"] for r in results]
    lifts     = [r["auc_lift"]   for r in results]
    return {
        "n_windows":       len(results),
        "stage1_auc_mean": round(float(np.mean(s1_aucs)), 4),
        "stage2_auc_mean": round(float(np.mean(s2_aucs)), 4),
        "lift_mean":       round(float(np.mean(lifts)),   4),
        "lift_std":        round(float(np.std(lifts)),    4),
        "lift_min":        round(float(np.min(lifts)),    4),
        "lift_consistent": bool(np.min(lifts) > 0),  # True if Stage-2 always helps
    }


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
