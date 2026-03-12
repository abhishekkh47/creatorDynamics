from typing import List

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score

from features.feature_pipeline import FEATURE_COLS, LABEL_COL


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibration_analysis(
    model: lgb.LGBMClassifier,
    df: pd.DataFrame,
    n_bins: int = 10,
) -> dict:
    """
    Check whether predicted probabilities match actual survival rates.

    A well-calibrated model predicts P=0.70 for posts that actually survive
    ~70% of the time. Poor calibration = the probability scores can't be
    trusted as confidence values in a creator-facing product.
    """
    probs = model.predict_proba(df[FEATURE_COLS])[:, 1]
    y = df[LABEL_COL].to_numpy()

    frac_positive, mean_predicted = calibration_curve(y, probs, n_bins=n_bins)

    # Expected Calibration Error: weighted mean absolute deviation
    counts, _ = np.histogram(probs, bins=n_bins, range=(0, 1))
    weights = counts / counts.sum()
    # align lengths (calibration_curve may return fewer bins if some are empty)
    n = min(len(frac_positive), len(weights))
    ece = float(np.sum(np.abs(frac_positive[:n] - mean_predicted[:n]) * weights[:n]))

    return {
        "expected_calibration_error": round(ece, 4),
        "interpretation": (
            "well-calibrated" if ece < 0.05
            else "moderate miscalibration" if ece < 0.10
            else "poorly calibrated"
        ),
        "bins": [
            {
                "predicted_prob": round(float(mp), 3),
                "actual_survival_rate": round(float(fp), 3),
                "gap": round(float(abs(mp - fp)), 3),
            }
            for mp, fp in zip(mean_predicted, frac_positive)
        ],
    }


# ---------------------------------------------------------------------------
# Per-segment breakdown
# ---------------------------------------------------------------------------

def _follower_segment(f: int) -> str:
    if f < 10_000:
        return "nano (<10k)"
    elif f < 100_000:
        return "micro (10k–100k)"
    else:
        return "macro (>100k)"


def segment_analysis(
    model: lgb.LGBMClassifier,
    df: pd.DataFrame,
) -> dict:
    """
    Break down AUC by account size and cluster tier.

    Reveals whether the model is strong only for certain creator types —
    important to know before shipping a creator-facing prediction.
    """
    df = df.copy()
    df["_prob"] = model.predict_proba(df[FEATURE_COLS])[:, 1]
    df["_size"] = df["follower_count"].apply(_follower_segment)

    results = {}

    for seg, group in df.groupby("_size", sort=False):
        if group[LABEL_COL].nunique() < 2 or len(group) < 30:
            continue
        results[f"account_size | {seg}"] = {
            "n_posts": len(group),
            "survival_rate": round(float(group[LABEL_COL].mean()), 3),
            "roc_auc": round(float(roc_auc_score(group[LABEL_COL], group["_prob"])), 4),
        }

    for tier, group in df.groupby("cluster_tier", sort=False):
        if group[LABEL_COL].nunique() < 2 or len(group) < 30:
            continue
        results[f"cluster_tier | {tier}"] = {
            "n_posts": len(group),
            "survival_rate": round(float(group[LABEL_COL].mean()), 3),
            "roc_auc": round(float(roc_auc_score(group[LABEL_COL], group["_prob"])), 4),
        }

    return results


# ---------------------------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------------------------

def threshold_analysis(
    model: lgb.LGBMClassifier,
    df: pd.DataFrame,
    thresholds: List[float] | None = None,
) -> List[dict]:
    """
    Compute precision, recall, F1, and positive prediction rate at each
    decision threshold.

    This is a product decision: a creator tool needs high precision
    (don't false-alarm) whereas an internal ranking tool might prefer
    high recall. The optimal threshold depends on the use case.
    """
    if thresholds is None:
        thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    probs = model.predict_proba(df[FEATURE_COLS])[:, 1]
    y = df[LABEL_COL].to_numpy()

    results = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = int(((preds == 1) & (y == 1)).sum())
        fp = int(((preds == 1) & (y == 0)).sum())
        fn = int(((preds == 0) & (y == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        results.append(
            {
                "threshold": t,
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
                "predicted_positive_rate": round(float(preds.mean()), 3),
            }
        )

    best_f1 = max(results, key=lambda x: x["f1"])
    best_precision = max(results, key=lambda x: x["precision"])
    best_recall = max(results, key=lambda x: x["recall"])

    return {
        "by_threshold": results,
        "recommended": {
            "max_f1": best_f1["threshold"],
            "max_precision": best_precision["threshold"],
            "max_recall": best_recall["threshold"],
        },
    }


# ---------------------------------------------------------------------------
# Console printer
# ---------------------------------------------------------------------------

def print_deep_analysis(
    calibration: dict,
    segments: dict,
    thresholds: dict,
) -> None:
    # Calibration
    print(f"\n  Calibration  (ECE = {calibration['expected_calibration_error']}  —  {calibration['interpretation']})")
    print("  " + "-" * 52)
    print(f"  {'Predicted':>12}  {'Actual':>8}  {'Gap':>6}")
    for b in calibration["bins"]:
        bar = "▓" * int(b["gap"] * 40)
        print(f"  {b['predicted_prob']:>12.2f}  {b['actual_survival_rate']:>8.2f}  {b['gap']:>6.3f}  {bar}")

    # Segments
    print(f"\n  Per-Segment AUC")
    print("  " + "-" * 52)
    for seg, info in segments.items():
        bar = "█" * int(info["roc_auc"] * 20)
        print(f"  {seg:<35} AUC {info['roc_auc']:.4f}  {bar}  (n={info['n_posts']:,})")

    # Thresholds
    print(f"\n  Threshold Analysis")
    print("  " + "-" * 52)
    print(f"  {'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  {'F1':>6}  {'Pos Rate':>9}")
    for row in thresholds["by_threshold"]:
        marker = "  ← best F1" if row["threshold"] == thresholds["recommended"]["max_f1"] else ""
        print(
            f"  {row['threshold']:>10.2f}  {row['precision']:>10.3f}  {row['recall']:>8.3f}"
            f"  {row['f1']:>6.3f}  {row['predicted_positive_rate']:>9.3f}{marker}"
        )
