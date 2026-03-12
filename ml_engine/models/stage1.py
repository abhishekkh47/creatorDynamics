from typing import Tuple

import lightgbm as lgb
import pandas as pd

from config import LGBM_PARAMS, TRAIN_RATIO, VAL_RATIO
from features.feature_pipeline import CATEGORICAL_FEATURES, FEATURE_COLS, LABEL_COL


def chronological_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:val_end].copy(),
        df.iloc[val_end:].copy(),
    )


def train_stage1(
    train: pd.DataFrame,
    val: pd.DataFrame,
) -> lgb.LGBMClassifier:
    X_train, y_train = train[FEATURE_COLS], train[LABEL_COL]
    X_val, y_val = val[FEATURE_COLS], val[LABEL_COL]

    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        categorical_feature=CATEGORICAL_FEATURES,
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    return model
