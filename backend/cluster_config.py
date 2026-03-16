"""
Cluster → niche mapping for the current model generation.

This is the single source of truth for what each cluster ID means and which
performance tier it belongs to. It is served to the frontend via GET /meta/niches
so the frontend never hardcodes these values.

--- How tiers are assigned ---
Tiers (strong / medium / weak) reflect per-cluster survival rates observed in
walk-forward validation. When the model is retrained, update this file to match
the new cluster assignments and re-check tier assignments from run_report.json
(per-segment AUC / survival rate by cluster).

--- How to update after a retrain ---
1. Run ml_engine/main.py to produce a new run_report.json.
2. Check the per-cluster survival rate or AUC in run_report.json.
3. Update CLUSTER_NICHES below to reflect any cluster reassignments or
   tier changes.
4. Restart the backend — the frontend will pick up the new mapping automatically
   on its next page load (no frontend deployment needed).
"""

from typing import Literal

Tier = Literal["strong", "medium", "weak"]


# Ordered: strong niches first (best default UX — most common niches at top).
CLUSTER_NICHES: list[dict] = [
    {"cluster_id": 7,  "label": "Comedy & Entertainment",      "tier": "strong"},
    {"cluster_id": 0,  "label": "Fitness & Health",            "tier": "strong"},
    {"cluster_id": 1,  "label": "Beauty & Makeup",             "tier": "strong"},
    {"cluster_id": 2,  "label": "Food & Cooking",              "tier": "strong"},
    {"cluster_id": 4,  "label": "Fashion & Style",             "tier": "strong"},
    {"cluster_id": 10, "label": "Music & Dance",               "tier": "strong"},
    {"cluster_id": 14, "label": "Sports",                      "tier": "strong"},
    {"cluster_id": 3,  "label": "Travel",                      "tier": "medium"},
    {"cluster_id": 5,  "label": "Finance & Investing",         "tier": "medium"},
    {"cluster_id": 6,  "label": "Tech & Gadgets",              "tier": "medium"},
    {"cluster_id": 8,  "label": "Lifestyle",                   "tier": "medium"},
    {"cluster_id": 9,  "label": "Education & How-to",          "tier": "medium"},
    {"cluster_id": 11, "label": "Gaming",                      "tier": "medium"},
    {"cluster_id": 12, "label": "Parenting & Family",          "tier": "medium"},
    {"cluster_id": 15, "label": "Business & Entrepreneurship", "tier": "medium"},
    {"cluster_id": 16, "label": "Pets & Animals",              "tier": "medium"},
    {"cluster_id": 13, "label": "Art & Creativity",            "tier": "weak"},
    {"cluster_id": 17, "label": "Mental Health & Wellness",    "tier": "weak"},
    {"cluster_id": 18, "label": "DIY & Home",                  "tier": "weak"},
    {"cluster_id": 19, "label": "News & Commentary",           "tier": "weak"},
]
