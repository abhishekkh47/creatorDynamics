import json
from datetime import datetime
from pathlib import Path

import numpy as np

from config import DECAY_LAMBDA, N_ACCOUNTS, N_CLUSTERS, RANDOM_SEED, SIMULATION_DAYS
from features.feature_pipeline import FEATURE_COLS, LABEL_COL, build_feature_matrix
from models.evaluator import evaluate, print_diagnostics, print_feature_importance
from models.stage1 import chronological_split, train_stage1
from synthetic.account import generate_accounts
from synthetic.cluster import generate_clusters
from synthetic.simulator import run_simulation

OUTPUTS_DIR = Path(__file__).parent / "outputs"


def _save_outputs(model, df_raw, df, val_metrics, test_metrics) -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # --- Model ---
    model.booster_.save_model(str(OUTPUTS_DIR / "model.txt"))

    # --- Datasets ---
    df_raw.to_csv(OUTPUTS_DIR / "simulation_data.csv", index=False)
    df[FEATURE_COLS + [LABEL_COL, "account_id", "timestamp", "reach_24h", "cluster_tier"]].to_csv(
        OUTPUTS_DIR / "feature_matrix.csv", index=False
    )

    # --- Run report ---
    import pandas as pd
    importance = (
        pd.Series(model.feature_importances_, index=FEATURE_COLS)
        .sort_values(ascending=False)
        .to_dict()
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
        "metrics": {
            "val": val_metrics,
            "test": test_metrics,
        },
        "feature_importance": {k: int(v) for k, v in importance.items()},
    }
    with open(OUTPUTS_DIR / "run_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Saved to  {OUTPUTS_DIR}/")
    print(f"    model.txt            — LightGBM model (loadable)")
    print(f"    simulation_data.csv  — {len(df_raw):,} raw simulated posts")
    print(f"    feature_matrix.csv   — {len(df):,} posts with features + labels")
    print(f"    run_report.json      — metrics, diagnostics, feature importance")


def main() -> None:
    np.random.seed(RANDOM_SEED)

    print("=" * 56)
    print("  Instagram Reach Intelligence Engine  —  Phase 1")
    print("=" * 56)

    # --- Step 1: Accounts ---
    print(f"\n[1/5] Generating {N_ACCOUNTS} accounts...")
    accounts = generate_accounts(N_ACCOUNTS, N_CLUSTERS)

    # --- Step 2: Clusters ---
    print(f"[2/5] Generating {N_CLUSTERS} topic clusters...")
    clusters = generate_clusters(N_CLUSTERS)

    tier_counts = {}
    for c in clusters:
        tier_counts[c.tier] = tier_counts.get(c.tier, 0) + 1
    print(f"      Tiers: {tier_counts}")

    # --- Step 3: Simulation ---
    print(f"[3/5] Running simulation over {SIMULATION_DAYS} days...")
    df_raw = run_simulation(accounts, clusters, SIMULATION_DAYS, seed=RANDOM_SEED)
    print(f"      Generated {len(df_raw):,} posts.")

    # --- Step 4: Feature pipeline ---
    print(f"[4/5] Building feature matrix  (λ={DECAY_LAMBDA})...")
    df = build_feature_matrix(df_raw)
    print_diagnostics(df)

    # --- Step 5: Train & evaluate ---
    print("\n[5/5] Training Stage-1 survival classifier...")
    train, val, test = chronological_split(df)
    print(f"      Train: {len(train):,}  |  Val: {len(val):,}  |  Test: {len(test):,}")

    model = train_stage1(train, val)

    print("\n  Metrics")
    print("  " + "-" * 40)
    val_metrics = evaluate(model, val, "val")
    test_metrics = evaluate(model, test, "test")

    print_feature_importance(model)

    # --- Save everything ---
    _save_outputs(model, df_raw, df, val_metrics, test_metrics)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
