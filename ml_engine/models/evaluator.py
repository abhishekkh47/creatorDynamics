import lightgbm as lgb
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from features.feature_pipeline import FEATURE_COLS, LABEL_COL


def evaluate(model: lgb.LGBMClassifier, df: pd.DataFrame, split_name: str = "test") -> dict:
    X, y = df[FEATURE_COLS], df[LABEL_COL]
    probs = model.predict_proba(X)[:, 1]

    auc = roc_auc_score(y, probs)
    ll = log_loss(y, probs)

    print(f"  [{split_name.upper()}]  ROC-AUC: {auc:.4f}  |  Log Loss: {ll:.4f}")

    return {"split": split_name, "roc_auc": auc, "log_loss": ll}


def print_feature_importance(model: lgb.LGBMClassifier) -> None:
    importance = (
        pd.Series(model.feature_importances_, index=FEATURE_COLS)
        .sort_values(ascending=False)
    )

    print("\n  Feature Importance")
    print("  " + "-" * 40)
    for feat, imp in importance.items():
        bar = "█" * int(imp / importance.max() * 20)
        print(f"  {feat:<30} {bar}  ({imp})")


def print_diagnostics(df: pd.DataFrame) -> None:
    corr_quality_reach = df["content_quality"].corr(df["reach_24h"])
    corr_cluster_reach = df["cluster_multiplier"].corr(df["reach_24h"])
    corr_reach_survived = df["reach_24h"].corr(df[LABEL_COL])
    survival_rate = df[LABEL_COL].mean()

    print(f"\n  Total posts         : {len(df):,}")
    print(f"  Survival rate       : {survival_rate:.3f}  (target ~0.50)")
    print(f"  corr(quality, reach): {corr_quality_reach:.3f}")
    print(f"  corr(cluster, reach): {corr_cluster_reach:.3f}")
    print(f"  corr(reach, survived): {corr_reach_survived:.3f}")
