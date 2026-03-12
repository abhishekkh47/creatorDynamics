"""
Model loading and inference logic.

Keeps all ML concern out of app.py. app.py handles HTTP; predictor.py handles inference.

Model files are loaded once at startup from ml_engine/outputs/ and held in a
module-level ModelStore instance. All prediction functions are pure — they take
a request and return a dict, with no side effects.
"""

from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np

# ---------------------------------------------------------------------------
# Feature ordering  (must match exactly what the models were trained on)
# ---------------------------------------------------------------------------

# From ml_engine/features/feature_pipeline.py  FEATURE_COLS
STAGE1_FEATURE_ORDER = [
    "rolling_weighted_median",
    "rolling_volatility",
    "posting_frequency",
    "cluster_entropy",
    "cluster_id",
    "posting_time_bucket",
    "content_quality",
]

# From ml_engine/features/velocity_features.py  VELOCITY_FEATURES_1H
STAGE2_1H_FEATURE_ORDER = [
    "stage1_prior",
    "norm_likes_1h",
    "comment_ratio_1h",
    "on_track_score",
]

# Expected burst fraction by cluster tier — mirrors velocity_simulator.py _TIER_BURST
_TIER_BURST: dict[str, float] = {
    "strong": 0.55,
    "medium": 0.35,
    "weak":   0.15,
}

# Decision thresholds chosen from threshold analysis in the experiment log
STAGE1_THRESHOLD = 0.35   # max-F1 threshold for Stage-1
STAGE2_THRESHOLD = 0.55   # max-F1 threshold for Stage-2 (1h model)


# ---------------------------------------------------------------------------
# Model store
# ---------------------------------------------------------------------------

class ModelStore:
    """Holds pre-trained LightGBM boosters loaded at application startup."""

    def __init__(self) -> None:
        self.stage1: Optional[lgb.Booster] = None
        self.stage2_1h: Optional[lgb.Booster] = None
        self.models_dir: Path = Path(__file__).parent.parent / "ml_engine" / "outputs"

    def load(self) -> None:
        """Load all model artifacts. Called once during FastAPI lifespan startup."""
        s1_path = self.models_dir / "model_stage1.txt"
        s2_path = self.models_dir / "model_stage2_1h.txt"

        if s1_path.exists():
            self.stage1 = lgb.Booster(model_file=str(s1_path))

        if s2_path.exists():
            self.stage2_1h = lgb.Booster(model_file=str(s2_path))

    @property
    def status(self) -> dict:
        return {
            "stage1":    {"loaded": self.stage1 is not None,    "file": "model_stage1.txt"},
            "stage2_1h": {"loaded": self.stage2_1h is not None, "file": "model_stage2_1h.txt"},
        }

    @property
    def all_loaded(self) -> bool:
        return self.stage1 is not None and self.stage2_1h is not None


# Module-level singleton — imported by app.py
model_store = ModelStore()


# ---------------------------------------------------------------------------
# Helper: posting time bucket
# ---------------------------------------------------------------------------

def _time_bucket(hour: Optional[int]) -> int:
    """
    Convert hour of day to the 4-bucket scheme used during training.
    0 = night (0–5h)
    1 = morning (6–11h)
    2 = afternoon (12–17h)
    3 = evening (18–23h)
    """
    if hour is None:
        return 1  # default to morning — the most common posting time
    return hour // 6


# ---------------------------------------------------------------------------
# Helper: prediction confidence
# ---------------------------------------------------------------------------

def _confidence(prob: float, threshold: float) -> str:
    distance = abs(prob - threshold)
    if distance >= 0.25:
        return "high"
    elif distance >= 0.10:
        return "medium"
    else:
        return "low"


# ---------------------------------------------------------------------------
# Stage-1 inference
# ---------------------------------------------------------------------------

def predict_stage1(
    rolling_weighted_median: float,
    rolling_volatility: float,
    posting_frequency: float,
    cluster_entropy: float,
    content_quality: float,
    cluster_id: int,
    hour_of_day: Optional[int],
) -> dict:
    """
    Run Stage-1 inference. All inputs are passed individually so callers
    are not coupled to internal feature naming conventions.
    """
    time_bucket = _time_bucket(hour_of_day)

    # Build feature vector in exact training order
    features = np.array([[
        rolling_weighted_median,
        rolling_volatility,
        posting_frequency,
        cluster_entropy,
        cluster_id,
        time_bucket,
        content_quality,
    ]], dtype=np.float64)

    prob = float(model_store.stage1.predict(features)[0])

    return {
        "survival_probability":  round(prob, 4),
        "survives":              prob >= STAGE1_THRESHOLD,
        "confidence":            _confidence(prob, STAGE1_THRESHOLD),
        "posting_time_bucket":   time_bucket,
        "model":                 "stage1",
    }


# ---------------------------------------------------------------------------
# Stage-2 inference
# ---------------------------------------------------------------------------

def predict_stage2(
    stage1_prior: float,
    rolling_weighted_median: float,
    likes_1h: int,
    comments_1h: int,
    cluster_tier: str,
) -> dict:
    """
    Run Stage-2 (1h) inference.

    Normalizes raw engagement counts using rolling_weighted_median,
    then applies the 1h model (AUC 0.978, trained on 4 features).
    """
    baseline    = max(rolling_weighted_median, 1.0)
    burst_bias  = _TIER_BURST.get(cluster_tier, 0.35)

    # Normalize engagement by account baseline (makes features account-agnostic)
    norm_likes_1h      = likes_1h / baseline
    comment_ratio_1h   = comments_1h / (likes_1h + 1)

    # On-track score: would the 1h rate, sustained for 24h, beat the baseline?
    # like_rate ≈ 0.08 × reach  (empirical approximation from simulation)
    implied_reach_24h  = likes_1h / (0.08 * max(burst_bias, 0.1))
    on_track_score     = implied_reach_24h / baseline

    features = np.array([[
        stage1_prior,
        norm_likes_1h,
        comment_ratio_1h,
        on_track_score,
    ]], dtype=np.float64)

    prob       = float(model_store.stage2_1h.predict(features)[0])
    correction = round(prob - stage1_prior, 4)

    return {
        "survival_probability": round(prob, 4),
        "survives":             prob >= STAGE2_THRESHOLD,
        "stage1_prior":         round(stage1_prior, 4),
        "correction":           correction,
        "confidence":           _confidence(prob, STAGE2_THRESHOLD),
        "velocity_features": {
            "norm_likes_1h":    round(norm_likes_1h, 4),
            "comment_ratio_1h": round(comment_ratio_1h, 4),
            "on_track_score":   round(on_track_score, 4),
        },
        "model": "stage2_1h",
    }
