"""
Cluster → niche mapping for the current model generation.

This is the single source of truth for all cluster-related configuration:
  - CLUSTER_NICHES  — cluster_id / label / tier (served to the frontend)
  - NICHE_KEYWORDS  — keyword bank for offline (heuristic) niche detection

Both live here so that a model retrain only requires editing one file.

--- How tiers are assigned ---
Tiers (strong / medium / weak) reflect per-cluster survival rates observed in
walk-forward validation. When the model is retrained, update this file to match
the new cluster assignments and re-check tier assignments from run_report.json
(per-segment AUC / survival rate by cluster).

--- How to update after a retrain ---
1. Run ml_engine/main.py to produce a new run_report.json.
2. Check the per-cluster survival rate or AUC in run_report.json.
3. Update CLUSTER_NICHES to reflect any cluster reassignments or tier changes.
4. Update NICHE_KEYWORDS to add/remove terms that match the new cluster themes.
5. Restart the backend — the frontend picks up the new mapping automatically
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


# ---------------------------------------------------------------------------
# Keyword bank for offline (heuristic) niche detection
#
# Maps cluster_id → list of ~25 terms covering caption vocabulary, common
# hashtag roots (without the #), and topic-specific jargon.
#
# Longer phrases are intentional — they produce fewer false-positives than
# single generic words.
#
# Keep in sync with CLUSTER_NICHES above: every cluster_id that appears in
# CLUSTER_NICHES should have a matching entry here.
# ---------------------------------------------------------------------------

NICHE_KEYWORDS: dict[int, list[str]] = {
    0: [  # Fitness & Health
        "fitness", "workout", "gym", "exercise", "training", "muscle", "diet",
        "hiit", "cardio", "gains", "protein", "macros", "bulk", "cut", "shred",
        "squat", "deadlift", "bench press", "reps", "sets", "pr", "personal record",
        "fat loss", "weight loss", "transformation", "lean", "bodybuilding",
    ],
    1: [  # Beauty & Makeup
        "beauty", "makeup", "skincare", "cosmetic", "glow", "foundation", "lipstick",
        "serum", "blush", "eyeshadow", "contour", "highlight", "moisturizer", "spf",
        "sunscreen", "routine", "grwm", "get ready with me", "no makeup", "tutorial",
        "skin care", "hyaluronic", "retinol", "dewy", "glam",
    ],
    2: [  # Food & Cooking
        "food", "recipe", "cooking", "eat", "delicious", "kitchen", "meal", "dish",
        "bake", "cuisine", "homemade", "easy recipe", "quick meal", "meal prep",
        "healthy eating", "vegan", "restaurant", "foodie", "chef", "ingredients",
        "dinner", "lunch", "breakfast", "dessert", "snack",
    ],
    3: [  # Travel
        "travel", "adventure", "explore", "destination", "vacation", "trip",
        "wanderlust", "journey", "abroad", "flight", "hotel", "backpack", "roadtrip",
        "bucket list", "solo travel", "itinerary", "hidden gem", "tourist", "visa",
        "passport", "airbnb", "hostel", "beach", "mountains",
    ],
    4: [  # Fashion & Style
        "fashion", "style", "outfit", "clothing", "ootd", "wear", "wardrobe",
        "trend", "streetwear", "aesthetic", "fit check", "haul", "thrift", "vintage",
        "designer", "luxury", "capsule wardrobe", "styling", "looks", "collection",
        "seasonal", "inspo", "accessories", "shoes",
    ],
    5: [  # Finance & Investing
        "finance", "money", "invest", "stock", "wealth", "budget", "crypto",
        "trading", "savings", "passive income", "financial freedom", "debt",
        "emergency fund", "retirement", "401k", "index fund", "dividend",
        "net worth", "frugal", "side hustle", "income", "expenses", "portfolio",
        "compound interest", "inflation",
    ],
    6: [  # Tech & Gadgets
        "tech", "technology", "gadget", "software", "code", "app", "ai",
        "innovation", "programming", "developer", "startup tech", "iphone",
        "android", "laptop", "unboxing", "review tech", "cybersecurity", "cloud",
        "machine learning", "automation", "saas", "api", "open source", "python",
    ],
    7: [  # Comedy & Entertainment
        "funny", "comedy", "humor", "laugh", "joke", "meme", "entertainment",
        "viral", "trending", "skit", "prank", "relatable", "lol", "hilarious",
        "pov", "satire", "parody", "bloopers", "reaction", "roast", "stand up",
        "impersonation", "trending audio", "trend",
    ],
    8: [  # Lifestyle
        "lifestyle", "daily", "routine", "productivity", "life", "morning",
        "evening", "vlog", "everyday", "day in the life", "habits", "goals",
        "self improvement", "minimalism", "aesthetic life", "adulting", "balance",
        "gratitude", "journaling", "growth", "self", "personal",
    ],
    9: [  # Education & How-to
        "education", "learn", "tutorial", "how to", "tips", "knowledge", "study",
        "guide", "explained", "facts", "did you know", "lesson", "school",
        "university", "course", "skill", "hack", "trick", "mistake", "advice",
        "mistakes", "beginner", "step by step", "breakdown", "cheat sheet",
    ],
    10: [  # Music & Dance
        "music", "dance", "song", "artist", "perform", "sing", "choreography",
        "beat", "lyrics", "album", "playlist", "concert", "cover song", "original",
        "rap", "pop", "rnb", "hip hop", "freestyle", "remix", "producer",
        "musician", "studio", "release", "new music",
    ],
    11: [  # Gaming
        "gaming", "game", "play", "gamer", "esports", "stream", "twitch",
        "console", "fps", "rpg", "minecraft", "fortnite", "valorant", "gameplay",
        "clutch", "highlight", "speedrun", "controller", "pc gaming",
        "playstation", "xbox", "nintendo", "streamer", "loot",
    ],
    12: [  # Parenting & Family
        "parenting", "mom", "dad", "baby", "kids", "family", "children", "parent",
        "toddler", "newborn", "pregnancy", "breastfeeding", "nursery", "first time mom",
        "raising kids", "tantrum", "school age", "teenager", "motherhood",
        "fatherhood", "homeschool", "sibling", "grandparent", "daycare",
    ],
    13: [  # Art & Creativity
        "art", "creative", "paint", "draw", "design", "artist", "illustration",
        "craft", "portfolio", "sketch", "watercolor", "acrylic", "digital art",
        "procreate", "commission", "studio art", "gallery", "abstract",
        "calligraphy", "pottery", "sculpture", "printmaking", "pattern", "color",
    ],
    14: [  # Sports
        "sports", "athlete", "football", "basketball", "soccer", "cricket",
        "match", "team", "coaching", "training drill", "game day", "tournament",
        "league", "championship", "referee", "stadium", "fan", "jersey",
        "tennis", "golf", "swimming", "track", "marathon", "cycling",
    ],
    15: [  # Business & Entrepreneurship
        "business", "entrepreneur", "startup", "marketing", "brand", "strategy",
        "growth", "sales", "founder", "ceo", "revenue", "profit", "product launch",
        "client", "freelance", "agency", "e-commerce", "dropshipping", "seo",
        "content marketing", "funnel", "leads", "networking", "pitch",
    ],
    16: [  # Pets & Animals
        "pets", "dog", "cat", "animal", "puppy", "kitten", "paw", "fur",
        "rescue", "adopt", "breeder", "vet", "training dog", "grooming",
        "bird", "rabbit", "hamster", "reptile", "aquarium", "wildlife",
        "exotic", "shelter", "pet care", "treats",
    ],
    17: [  # Mental Health & Wellness
        "mental health", "wellness", "mindfulness", "meditation", "anxiety",
        "self care", "therapy", "healing", "burnout", "stress", "depression",
        "boundaries", "trauma", "inner child", "journaling therapy", "breathwork",
        "gratitude practice", "emotional", "vulnerability", "awareness", "coping",
    ],
    18: [  # DIY & Home
        "diy", "home", "decor", "garden", "renovation", "interior", "handmade",
        "upcycle", "thrift flip", "home improvement", "before after", "small space",
        "organization", "cleaning", "apartment", "furniture", "bedroom",
        "living room", "kitchen decor", "bathroom", "storage hack", "budget decor",
    ],
    19: [  # News & Commentary
        "news", "politics", "opinion", "commentary", "current events", "world",
        "update", "breaking", "election", "government", "policy", "economy",
        "social issues", "culture", "debate", "analysis", "explained news",
        "hot topic", "controversy", "viral news", "reaction news",
    ],
}
