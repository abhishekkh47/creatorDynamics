from typing import List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss, roc_auc_score

from config import STAGE2_LGBM_PARAMS
from features.feature_pipeline import FEATURE_COLS, LABEL_COL
from features.velocity_features import (
    VELOCITY_FEATURE_COLS,
    VELOCITY_FEATURES_1H,
    VELOCITY_FEATURES_3H,
)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibration_analysis(
    model: lgb.LGBMClassifier,
    df: pd.DataFrame,
    n_bins: int = 10,
    feature_cols: Optional[List[str]] = None,
) -> dict:
    """
    Check whether predicted probabilities match actual survival rates.

    A well-calibrated model predicts P=0.70 for posts that actually survive
    ~70% of the time. Poor calibration = the probability scores can't be
    trusted as confidence values in a creator-facing product.
    
    Works for both Stage-1 (feature_cols=FEATURE_COLS) and Stage-2 (feature_cols=VELOCITY_FEATURE_COLS).
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    probs = model.predict_proba(df[feature_cols])[:, 1]
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
    feature_cols: Optional[List[str]] = None,
) -> dict:
    """
    Break down AUC by account size and cluster tier.
    Works for both Stage-1 and Stage-2 via feature_cols parameter.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    df = df.copy()
    df["_prob"] = model.predict_proba(df[feature_cols])[:, 1]
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
    thresholds: Optional[List[float]] = None,
    feature_cols: Optional[List[str]] = None,
) -> dict:
    """
    Compute precision, recall, F1 at each decision threshold.
    This is a product decision: a creator tool needs high precision
    (don't false-alarm) whereas an internal ranking tool might prefer
    high recall. The optimal threshold depends on the use case.
    
    Works for both Stage-1 and Stage-2 via feature_cols parameter.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    if thresholds is None:
        thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    probs = model.predict_proba(df[feature_cols])[:, 1]
    y = df[LABEL_COL].to_numpy()

    results = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = int(((preds == 1) & (y == 1)).sum())
        fp = int(((preds == 1) & (y == 0)).sum())
        fn = int(((preds == 0) & (y == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        results.append({
            "threshold": t,
            "precision": round(precision, 3),
            "recall":    round(recall, 3),
            "f1":        round(f1, 3),
            "predicted_positive_rate": round(float(preds.mean()), 3),
        })

    best_f1        = max(results, key=lambda x: x["f1"])
    best_precision = max(results, key=lambda x: x["precision"])
    best_recall    = max(results, key=lambda x: x["recall"])

    return {
        "by_threshold": results,
        "recommended": {
            "max_f1":        best_f1["threshold"],
            "max_precision": best_precision["threshold"],
            "max_recall":    best_recall["threshold"],
        },
    }


# ---------------------------------------------------------------------------
# Observation window analysis  (Stage-2 specific)
# ---------------------------------------------------------------------------

def _train_window_model(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: List[str],
) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(**STAGE2_LGBM_PARAMS)
    model.fit(
        train[feature_cols], train[LABEL_COL],
        eval_set=[(val[feature_cols], val[LABEL_COL])],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    return model


def observation_window_analysis(
    s2_train: pd.DataFrame,
    s2_val: pd.DataFrame,
    s2_test: pd.DataFrame,
    stage1_test_auc: float,
) -> dict:
    """
    Train separate Stage-2 models at 1h, 3h, and 6h observation windows.

    Answers the product question: at what point after posting is the
    corrected prediction good enough to act on?

    Includes Stage-1 AUC as the 0h baseline for comparison.
    """
    results = {"0h (Stage-1 only)": {"auc": round(stage1_test_auc, 4), "n_features": 0}}

    for label, feature_cols in [
        ("1h", VELOCITY_FEATURES_1H),
        ("3h", VELOCITY_FEATURES_3H),
        ("6h", VELOCITY_FEATURE_COLS),
    ]:
        model = _train_window_model(s2_train, s2_val, feature_cols)
        probs = model.predict_proba(s2_test[feature_cols])[:, 1]
        auc = roc_auc_score(s2_test[LABEL_COL], probs)
        ll  = log_loss(s2_test[LABEL_COL], probs)

        results[label] = {
            "auc": round(float(auc), 4),
            "log_loss": round(float(ll), 4),
            "auc_lift_vs_stage1": round(float(auc) - round(stage1_test_auc, 4), 4),
            "n_features": len(feature_cols),
        }

    return results


# ---------------------------------------------------------------------------
# Uncertainty resolution analysis  (Stage-2 specific)
# ---------------------------------------------------------------------------

def uncertainty_resolution_analysis(
    stage1_probs: np.ndarray,
    stage2_probs: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """
    Bin Stage-1 predictions by confidence, then measure Stage-2 AUC in each bin.

    The most valuable correction happens when Stage-1 is uncertain (0.35–0.65).
    When Stage-1 is already very confident, velocity should add less value.
    """
    bins = {
        "very_uncertain (0.35–0.65)": (0.35, 0.65),
        "moderate (0.25–0.35 or 0.65–0.75)": None,
        "confident (<0.25 or >0.75)": None,
    }

    def get_mask(probs, lo, hi):
        return (probs >= lo) & (probs < hi)

    uncertain_mask = get_mask(stage1_probs, 0.35, 0.65)
    moderate_mask  = get_mask(stage1_probs, 0.25, 0.35) | get_mask(stage1_probs, 0.65, 0.75)
    confident_mask = (stage1_probs < 0.25) | (stage1_probs >= 0.75)

    results = {}
    for name, mask in [
        ("very_uncertain (S1 prob 0.35–0.65)", uncertain_mask),
        ("moderate       (S1 prob 0.25–0.35 or 0.65–0.75)", moderate_mask),
        ("confident      (S1 prob <0.25 or >0.75)", confident_mask),
    ]:
        if mask.sum() < 20 or len(np.unique(labels[mask])) < 2:
            continue
        s1_auc = roc_auc_score(labels[mask], stage1_probs[mask])
        s2_auc = roc_auc_score(labels[mask], stage2_probs[mask])
        results[name] = {
            "n_posts": int(mask.sum()),
            "stage1_auc": round(float(s1_auc), 4),
            "stage2_auc": round(float(s2_auc), 4),
            "lift": round(float(s2_auc - s1_auc), 4),
        }

    return results


# ---------------------------------------------------------------------------
# Console printers
# ---------------------------------------------------------------------------

def print_deep_analysis(
    calibration: dict,
    segments: dict,
    thresholds: dict,
    label: str = "Stage-1",
) -> None:
    print(f"\n  {label} Calibration  (ECE = {calibration['expected_calibration_error']}  —  {calibration['interpretation']})")
    print("  " + "-" * 52)
    print(f"  {'Predicted':>12}  {'Actual':>8}  {'Gap':>6}")
    for b in calibration["bins"]:
        bar = "▓" * int(b["gap"] * 40)
        print(f"  {b['predicted_prob']:>12.2f}  {b['actual_survival_rate']:>8.2f}  {b['gap']:>6.3f}  {bar}")

    print(f"\n  {label} Per-Segment AUC")
    print("  " + "-" * 52)
    for seg, info in segments.items():
        bar = "█" * int(info["roc_auc"] * 20)
        print(f"  {seg:<35} AUC {info['roc_auc']:.4f}  {bar}  (n={info['n_posts']:,})")

    print(f"\n  {label} Threshold Analysis")
    print("  " + "-" * 52)
    print(f"  {'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  {'F1':>6}  {'Pos Rate':>9}")
    for row in thresholds["by_threshold"]:
        marker = "  ← best F1" if row["threshold"] == thresholds["recommended"]["max_f1"] else ""
        print(
            f"  {row['threshold']:>10.2f}  {row['precision']:>10.3f}  {row['recall']:>8.3f}"
            f"  {row['f1']:>6.3f}  {row['predicted_positive_rate']:>9.3f}{marker}"
        )


def print_stage2_deep_analysis(
    calibration: dict,
    segments: dict,
    thresholds: dict,
    window_results: dict,
    uncertainty_results: dict,
) -> None:
    print_deep_analysis(calibration, segments, thresholds, label="Stage-2")

    # Observation window lift table
    print(f"\n  Observation Window AUC  (how much lift at each checkpoint)")
    print("  " + "-" * 52)
    print(f"  {'Window':<30}  {'AUC':>7}  {'Lift vs S1':>11}  {'Features':>9}")
    for window, info in window_results.items():
        lift_str = f"+{info.get('auc_lift_vs_stage1', 0):.4f}" if "auc_lift_vs_stage1" in info else "  baseline"
        n_feat   = info.get("n_features", "—")
        print(f"  {window:<30}  {info['auc']:>7.4f}  {lift_str:>11}  {str(n_feat):>9}")

    # Uncertainty resolution table
    print(f"\n  Uncertainty Resolution  (where does Stage-2 help most?)")
    print("  " + "-" * 52)
    print(f"  {'Bucket':<52}  {'n':>5}  {'S1 AUC':>7}  {'S2 AUC':>7}  {'Lift':>6}")
    for bucket, info in uncertainty_results.items():
        print(
            f"  {bucket:<52}  {info['n_posts']:>5,}  {info['stage1_auc']:>7.4f}"
            f"  {info['stage2_auc']:>7.4f}  +{info['lift']:.4f}"
        )
