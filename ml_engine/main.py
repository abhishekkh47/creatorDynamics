import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DECAY_LAMBDA, N_ACCOUNTS, N_CLUSTERS, RANDOM_SEED,
    SIMULATION_DAYS, STAGE2_TRAIN_RATIO,
)
from features.feature_pipeline import FEATURE_COLS, LABEL_COL, build_feature_matrix
from features.velocity_features import build_velocity_features
from models.analysis import (
    calibration_analysis,
    print_deep_analysis,
    segment_analysis,
    threshold_analysis,
)
from models.evaluator import evaluate, print_diagnostics, print_feature_importance
from models.stage1 import chronological_split, train_stage1
from models.stage2 import (
    evaluate_stage2,
    prior_vs_posterior_analysis,
    print_stage2_feature_importance,
    train_stage2,
)
from models.walk_forward import summarise_walk_forward, walk_forward_validation
from features.velocity_features import VELOCITY_FEATURE_COLS
from synthetic.account import generate_accounts
from synthetic.cluster import generate_clusters
from synthetic.simulator import run_simulation
from synthetic.velocity_simulator import simulate_velocity

OUTPUTS_DIR = Path(__file__).parent / "outputs"


def _save_outputs(report: dict, model_s1, model_s2, df_raw, df) -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)

    model_s1.booster_.save_model(str(OUTPUTS_DIR / "model_stage1.txt"))
    model_s2.booster_.save_model(str(OUTPUTS_DIR / "model_stage2.txt"))

    df_raw.to_csv(OUTPUTS_DIR / "simulation_data.csv", index=False)

    save_cols = (
        FEATURE_COLS
        + [LABEL_COL, "account_id", "follower_count", "timestamp",
           "reach_24h", "cluster_tier", "likes", "comments",
           "likes_1h", "likes_3h", "likes_6h",
           "comments_1h", "comments_3h", "comments_6h"]
    )
    df[[c for c in save_cols if c in df.columns]].to_csv(
        OUTPUTS_DIR / "feature_matrix.csv", index=False
    )

    with open(OUTPUTS_DIR / "run_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Saved to  {OUTPUTS_DIR}/")
    print(f"    model_stage1.txt     — Stage-1 pre-post model")
    print(f"    model_stage2.txt     — Stage-2 velocity correction model")
    print(f"    simulation_data.csv  — {report['simulation']['total_posts_raw']:,} raw posts")
    print(f"    feature_matrix.csv   — {report['simulation']['total_posts_after_pipeline']:,} posts with all features + velocity")
    print(f"    run_report.json      — full structured report")


def main() -> None:
    np.random.seed(RANDOM_SEED)

    print("=" * 56)
    print("  Instagram Reach Intelligence Engine  —  Phase 2")
    print("=" * 56)

    # -------------------------------------------------------------------------
    # Phase 1: Simulation + Feature Pipeline + Stage-1 Model
    # -------------------------------------------------------------------------

    print(f"\n[1/9] Generating {N_ACCOUNTS} accounts...")
    accounts = generate_accounts(N_ACCOUNTS, N_CLUSTERS)

    print(f"[2/9] Generating {N_CLUSTERS} topic clusters...")
    clusters = generate_clusters(N_CLUSTERS)
    tier_counts = {}
    for c in clusters:
        tier_counts[c.tier] = tier_counts.get(c.tier, 0) + 1
    print(f"      Tiers: {tier_counts}")

    print(f"[3/9] Running simulation over {SIMULATION_DAYS} days...")
    df_raw = run_simulation(accounts, clusters, SIMULATION_DAYS, seed=RANDOM_SEED)
    print(f"      Generated {len(df_raw):,} posts.")

    print(f"[4/9] Building feature matrix  (λ={DECAY_LAMBDA})...")
    df = build_feature_matrix(df_raw)
    print_diagnostics(df)

    print("\n[5/9] Training Stage-1 survival classifier...")
    train_s1, val_s1, test_s1 = chronological_split(df)
    print(f"      Train: {len(train_s1):,}  |  Val: {len(val_s1):,}  |  Test: {len(test_s1):,}")
    model_s1 = train_stage1(train_s1, val_s1)

    print("\n  Stage-1 Metrics")
    print("  " + "-" * 40)
    s1_val_metrics  = evaluate(model_s1, val_s1,  "val")
    s1_test_metrics = evaluate(model_s1, test_s1, "test")
    print_feature_importance(model_s1)

    print("\n[6/9] Walk-forward validation  (90-day min train, 30-day windows)...")
    wf_results = walk_forward_validation(df, min_train_days=90, window_days=30)
    wf_summary = summarise_walk_forward(wf_results)
    stability = "STABLE ✓" if wf_summary.get("stable") else "UNSTABLE"
    print(f"      AUC  mean={wf_summary.get('auc_mean')}  std={wf_summary.get('auc_std')}  "
          f"min={wf_summary.get('auc_min')}  max={wf_summary.get('auc_max')}  →  {stability}")

    print("\n[7/9] Deep analysis  (calibration · segments · thresholds)...")
    calibration = calibration_analysis(model_s1, test_s1)
    segments    = segment_analysis(model_s1, test_s1)
    thresholds  = threshold_analysis(model_s1, test_s1)
    print_deep_analysis(calibration, segments, thresholds)

    # -------------------------------------------------------------------------
    # Phase 2: Velocity Simulation + Stage-2 Model
    # -------------------------------------------------------------------------

    print("\n[8/9] Simulating early engagement velocity (1h · 3h · 6h)...")
    df_vel = simulate_velocity(df, seed=RANDOM_SEED)

    # Re-split with velocity columns attached
    _, val_vel, test_vel = chronological_split(df_vel)

    # Get Stage-1 out-of-sample probabilities for val and test
    val_s1_probs  = model_s1.predict_proba(val_vel[FEATURE_COLS])[:, 1]
    test_s1_probs = model_s1.predict_proba(test_vel[FEATURE_COLS])[:, 1]

    # Build Stage-2 feature frames
    val_s2  = build_velocity_features(val_vel,  val_s1_probs)
    test_s2 = build_velocity_features(test_vel, test_s1_probs)

    # Split val into Stage-2 train / Stage-2 val (chronologically)
    n_s2_train = int(len(val_s2) * STAGE2_TRAIN_RATIO)
    s2_train = val_s2.iloc[:n_s2_train]
    s2_val   = val_s2.iloc[n_s2_train:]
    print(f"      Stage-2  Train: {len(s2_train):,}  |  Val: {len(s2_val):,}  |  Test: {len(test_s2):,}")

    print("\n[9/9] Training Stage-2 velocity correction model...")
    model_s2 = train_stage2(s2_train, s2_val)

    print("\n  Stage-2 Metrics")
    print("  " + "-" * 40)
    s2_val_metrics  = evaluate_stage2(model_s2, s2_val,   "val")
    s2_test_metrics = evaluate_stage2(model_s2, test_s2,  "test")
    print_stage2_feature_importance(model_s2)

    # Prior vs posterior on the test set
    test_s2_probs = model_s2.predict_proba(test_s2[VELOCITY_FEATURE_COLS])[:, 1]
    comparison = prior_vs_posterior_analysis(test_s1_probs, test_s2_probs, test_s2[LABEL_COL].to_numpy())

    # -------------------------------------------------------------------------
    # Save everything
    # -------------------------------------------------------------------------
    s1_importance = (
        pd.Series(model_s1.feature_importances_, index=FEATURE_COLS)
        .sort_values(ascending=False).to_dict()
    )
    s2_importance = (
        pd.Series(model_s2.feature_importances_, index=VELOCITY_FEATURE_COLS)
        .sort_values(ascending=False).to_dict()
    )

    report = {
        "run_at": datetime.now().isoformat(),
        "config": {
            "random_seed": RANDOM_SEED,
            "n_accounts": N_ACCOUNTS,
            "n_clusters": N_CLUSTERS,
            "simulation_days": SIMULATION_DAYS,
            "decay_lambda": DECAY_LAMBDA,
        },
        "simulation": {
            "total_posts_raw": len(df_raw),
            "total_posts_after_pipeline": len(df),
            "survival_rate": round(float(df[LABEL_COL].mean()), 4),
            "corr_quality_reach": round(float(df["content_quality"].corr(df["reach_24h"])), 4),
            "corr_cluster_reach": round(float(df["cluster_multiplier"].corr(df["reach_24h"])), 4),
            "corr_reach_survived": round(float(df["reach_24h"].corr(df[LABEL_COL])), 4),
        },
        "stage1": {
            "metrics": {"val": s1_val_metrics, "test": s1_test_metrics},
            "feature_importance": {k: int(v) for k, v in s1_importance.items()},
            "walk_forward": {"summary": wf_summary, "windows": wf_results},
            "deep_analysis": {
                "calibration": calibration,
                "segments": segments,
                "thresholds": thresholds,
            },
        },
        "stage2": {
            "metrics": {"val": s2_val_metrics, "test": s2_test_metrics},
            "feature_importance": {k: int(v) for k, v in s2_importance.items()},
            "prior_vs_posterior": comparison,
        },
    }

    _save_outputs(report, model_s1, model_s2, df_raw, df_vel)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
