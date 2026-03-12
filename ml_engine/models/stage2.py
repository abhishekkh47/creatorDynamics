from typing import Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from config import STAGE2_LGBM_PARAMS
from features.feature_pipeline import LABEL_COL
from features.velocity_features import VELOCITY_FEATURE_COLS


def train_stage2(
    train: pd.DataFrame,
    val: pd.DataFrame,
) -> lgb.LGBMClassifier:
    """
    Train Stage-2 velocity correction model.

    Inputs  : Stage-1 prior probability + 8 early velocity features
    Output  : corrected survival probability (posterior)

    Training data must use Stage-1 predictions that are OUT-OF-SAMPLE
    for those posts — otherwise Stage-2 learns to trust an overfit prior.
    """
    X_train, y_train = train[VELOCITY_FEATURE_COLS], train[LABEL_COL]
    X_val, y_val = val[VELOCITY_FEATURE_COLS], val[LABEL_COL]

    model = lgb.LGBMClassifier(**STAGE2_LGBM_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    return model


def evaluate_stage2(
    model: lgb.LGBMClassifier,
    df: pd.DataFrame,
    split_name: str = "test",
) -> dict:
    probs = model.predict_proba(df[VELOCITY_FEATURE_COLS])[:, 1]
    y = df[LABEL_COL]

    auc = roc_auc_score(y, probs)
    ll = log_loss(y, probs)

    print(f"  [{split_name.upper()}]  ROC-AUC: {auc:.4f}  |  Log Loss: {ll:.4f}")

    return {"split": split_name, "roc_auc": round(auc, 4), "log_loss": round(ll, 4)}


def print_stage2_feature_importance(model: lgb.LGBMClassifier) -> None:
    importance = (
        pd.Series(model.feature_importances_, index=VELOCITY_FEATURE_COLS)
        .sort_values(ascending=False)
    )

    print("\n  Stage-2 Feature Importance")
    print("  " + "-" * 44)
    for feat, imp in importance.items():
        bar = "█" * int(imp / importance.max() * 20)
        print(f"  {feat:<25} {bar}  ({imp})")


def prior_vs_posterior_analysis(
    stage1_probs: np.ndarray,
    stage2_probs: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """
    Compare Stage-1 and Stage-2 predictions on the same posts.
    Shows how Stage-2 corrects Stage-1 errors using velocity evidence.
    """
    stage1_auc = roc_auc_score(labels, stage1_probs)
    stage2_auc = roc_auc_score(labels, stage2_probs)
    auc_lift = stage2_auc - stage1_auc

    # Posts Stage-1 got wrong but Stage-2 corrected
    s1_pred = (stage1_probs >= 0.5).astype(int)
    s2_pred = (stage2_probs >= 0.5).astype(int)

    s1_wrong = s1_pred != labels
    s2_correct = s2_pred == labels
    corrections = int((s1_wrong & s2_correct).sum())
    regressions = int((~s1_wrong & ~s2_correct).sum())

    print(f"\n  Stage-1 vs Stage-2 Comparison")
    print("  " + "-" * 44)
    print(f"  Stage-1 AUC        : {stage1_auc:.4f}")
    print(f"  Stage-2 AUC        : {stage2_auc:.4f}")
    print(f"  AUC lift           : +{auc_lift:.4f}" if auc_lift >= 0 else f"  AUC lift           : {auc_lift:.4f}")
    print(f"  Corrections        : {corrections}  (Stage-1 wrong → Stage-2 right)")
    print(f"  Regressions        : {regressions}  (Stage-1 right → Stage-2 wrong)")

    return {
        "stage1_auc": round(stage1_auc, 4),
        "stage2_auc": round(stage2_auc, 4),
        "auc_lift": round(auc_lift, 4),
        "corrections": corrections,
        "regressions": regressions,
    }
