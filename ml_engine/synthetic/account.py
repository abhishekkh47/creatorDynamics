from dataclasses import dataclass
from typing import List

import numpy as np

from config import N_ACCOUNTS, N_CLUSTERS


@dataclass
class Account:
    account_id: int
    follower_count: int
    baseline_reach_ratio: float
    quality_mean: float
    quality_variance: float
    velocity_sensitivity: float
    decay_rate: float
    volatility_factor: float
    cluster_distribution: List[float]


def generate_accounts(
    n_accounts: int = N_ACCOUNTS,
    n_clusters: int = N_CLUSTERS,
) -> List[Account]:
    accounts = []

    for i in range(n_accounts):
        # Target realistic active Instagram creators: 2k–2M followers, median ~36k.
        # Previous distribution (mean=9, σ=1.5) produced median follower_count ≈ 8k
        # with most accounts below 5k, causing training rolling_weighted_median to
        # cluster around 193 — far below what real creators produce (2k–50k reach).
        follower_count = int(np.clip(np.random.lognormal(mean=10.5, sigma=1.2), 2_000, 5_000_000))

        # Reels can reach well beyond follower count via Explore/For-You distribution.
        # Beta(2,3) → mean 0.40, range ≈ 0.05–0.85 (vs old Beta(2,5) mean 0.29).
        baseline_reach_ratio = float(np.random.beta(2, 3))
        quality_mean = float(np.random.uniform(0.3, 0.9))
        quality_variance = float(np.random.uniform(0.15, 0.35))
        velocity_sensitivity = float(np.random.uniform(0.1, 0.5))
        decay_rate = float(np.random.uniform(0.01, 0.05))
        volatility_factor = float(np.random.uniform(0.8, 1.3))

        # Sparse Dirichlet: most accounts concentrate on a few clusters
        alpha = np.ones(n_clusters) * 0.3
        cluster_distribution = np.random.dirichlet(alpha).tolist()

        accounts.append(
            Account(
                account_id=i,
                follower_count=follower_count,
                baseline_reach_ratio=baseline_reach_ratio,
                quality_mean=quality_mean,
                quality_variance=quality_variance,
                velocity_sensitivity=velocity_sensitivity,
                decay_rate=decay_rate,
                volatility_factor=volatility_factor,
                cluster_distribution=cluster_distribution,
            )
        )

    return accounts
