from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

from config import SIMULATION_DAYS
from synthetic.account import Account
from synthetic.cluster import Cluster

_START_DATE = datetime(2024, 1, 1)
_POSTING_HOURS = [8, 10, 12, 14, 16, 18, 20, 22]


def _should_post(account: Account, rng: np.random.Generator) -> bool:
    prob = np.clip(0.15 + (np.log10(account.follower_count) - 2) * 0.02, 0.05, 0.50)
    return bool(rng.random() < prob)


def _compute_reach(
    account: Account,
    cluster: Cluster,
    quality: float,
    current_baseline_ratio: float,
    current_volatility: float,
    rng: np.random.Generator,
) -> float:
    # Mean-corrected so E[post_noise]=1 regardless of sigma; only variance changes
    sigma = current_volatility
    post_noise = rng.lognormal(mean=-0.5 * sigma ** 2, sigma=sigma)

    # Discrete algorithmic events: 30% suppressed, 30% viral, 40% normal
    event = rng.random()
    if event < 0.30:
        algorithmic_factor = rng.uniform(0.02, 0.20)
    elif event < 0.60:
        algorithmic_factor = rng.uniform(3.0, 10.0)
    else:
        algorithmic_factor = rng.exponential(scale=0.8) + 0.2  # E=1.0

    noise = post_noise * algorithmic_factor

    reach = (
        account.follower_count
        * current_baseline_ratio
        * quality
        * cluster.performance_multiplier
        * noise
    )
    return max(1.0, float(reach))


def _generate_engagement(
    reach: float,
    cluster: Cluster,
    rng: np.random.Generator,
) -> dict:
    er = cluster.engagement_rate_multipliers
    return {
        "likes": int(reach * er["like_rate"] * rng.uniform(0.8, 1.2)),
        "comments": int(reach * er["comment_rate"] * rng.uniform(0.8, 1.2)),
        "shares": int(reach * er["share_rate"] * rng.uniform(0.8, 1.2)),
    }


def run_simulation(
    accounts: List[Account],
    clusters: List[Cluster],
    simulation_days: int = SIMULATION_DAYS,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Both baseline_ratio and volatility drift daily via random walk.
    # rolling_weighted_median and rolling_volatility become lagging estimates,
    # introducing irreducible uncertainty the model cannot fully recover from.
    current_ratios: Dict[int, float] = {
        a.account_id: a.baseline_reach_ratio for a in accounts
    }
    current_vols: Dict[int, float] = {
        a.account_id: a.volatility_factor for a in accounts
    }

    records = []

    for day in range(simulation_days):
        date = _START_DATE + timedelta(days=day)
        hour = int(rng.choice(_POSTING_HOURS))

        # Step baseline and volatility forward via small independent random walks
        for account in accounts:
            aid = account.account_id
            current_ratios[aid] = float(
                np.clip(current_ratios[aid] * rng.lognormal(0.0, 0.12), 0.01, 1.0)
            )
            current_vols[aid] = float(
                np.clip(current_vols[aid] * rng.lognormal(0.0, 0.18), 0.2, 3.0)
            )

        for account in accounts:
            if not _should_post(account, rng):
                continue

            aid = account.account_id
            cluster_idx = int(rng.choice(len(clusters), p=account.cluster_distribution))
            cluster = clusters[cluster_idx]

            quality = float(np.clip(rng.normal(account.quality_mean, account.quality_variance), 0.01, 1.0))
            reach_24h = _compute_reach(
                account, cluster, quality,
                current_ratios[aid], current_vols[aid], rng,
            )
            engagement = _generate_engagement(reach_24h, cluster, rng)

            records.append(
                {
                    "account_id": account.account_id,
                    "follower_count": account.follower_count,
                    "timestamp": date.replace(hour=hour),
                    "day": day,
                    "cluster_id": cluster.cluster_id,
                    "cluster_tier": cluster.tier,
                    "cluster_multiplier": cluster.performance_multiplier,
                    "content_quality": quality,
                    "reach_24h": reach_24h,
                    "likes": engagement["likes"],
                    "comments": engagement["comments"],
                    "shares": engagement["shares"],
                }
            )

    return pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
