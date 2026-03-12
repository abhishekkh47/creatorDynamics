from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from config import N_CLUSTERS


@dataclass
class Cluster:
    cluster_id: int
    performance_multiplier: float
    engagement_rate_multipliers: Dict[str, float]
    early_burst_bias: float
    tier: str


def generate_clusters(n_clusters: int = N_CLUSTERS) -> List[Cluster]:
    n_strong = 4
    n_medium = 8
    n_weak = n_clusters - n_strong - n_medium

    # Compressed tier spread: cluster is a meaningful but not dominant signal
    tier_specs = (
        [("strong", 1.2, 1.7)] * n_strong
        + [("medium", 0.8, 1.2)] * n_medium
        + [("weak", 0.5, 0.85)] * n_weak
    )
    np.random.shuffle(tier_specs)

    clusters = []
    for i, (tier, mult_low, mult_high) in enumerate(tier_specs):
        performance_multiplier = float(np.random.uniform(mult_low, mult_high))

        engagement_rate_multipliers = {
            "like_rate": float(np.clip(np.random.uniform(0.03, 0.12) * performance_multiplier, 0.01, 0.30)),
            "comment_rate": float(np.clip(np.random.uniform(0.005, 0.03) * performance_multiplier, 0.001, 0.10)),
            "share_rate": float(np.clip(np.random.uniform(0.002, 0.015) * performance_multiplier, 0.001, 0.05)),
        }

        clusters.append(
            Cluster(
                cluster_id=i,
                performance_multiplier=performance_multiplier,
                engagement_rate_multipliers=engagement_rate_multipliers,
                early_burst_bias=float(np.random.uniform(0.3, 0.9)),
                tier=tier,
            )
        )

    return clusters
