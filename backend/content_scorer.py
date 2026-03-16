"""
Rule-based content quality scorer.

Scores a post's caption and hashtags on five signals that are strongly
correlated with early engagement on Instagram Reels.  Returns a 0–1 score
that maps directly to the `content_quality` ML feature.

No external APIs or LLMs required — all signals are deterministic, cheap
to compute, and easy to audit.

--- Scoring signals ---

| Signal            | Weight | What it measures                                   |
|-------------------|--------|----------------------------------------------------|
| hook_strength     | 0.30   | Is the opening line compelling enough to stop the  |
|                   |        | scroll? Questions, numbers, and bold claims score  |
|                   |        | higher.                                            |
| cta_presence      | 0.25   | Does the caption direct the viewer to DO something |
|                   |        | (save, comment, share, follow)?  Posts with an     |
|                   |        | explicit CTA get ~40% more comments on average.    |
| hashtag_quality   | 0.20   | 3–10 focused hashtags is the sweet spot.  Too few  |
|                   |        | limits discovery; 20+ looks spammy and Instagram   |
|                   |        | may suppress distribution.                         |
| caption_length    | 0.15   | 100–300 chars is the engagement-optimised range    |
|                   |        | for Reels captions.  Very short = no context; very |
|                   |        | long = people scroll past.                         |
| engagement_signals| 0.10   | Moderate emoji use and in-body questions drive     |
|                   |        | comment velocity.                                  |

--- Updating this scorer ---
When model retraining reveals that a different signal distribution leads to
better calibration between content_quality and actual survival rates, adjust
the weights or scoring functions here.  This file has no dependency on the
ML models and can be updated independently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# CTA keyword bank
# ---------------------------------------------------------------------------

_CTA_PHRASES = [
    "comment", "comments", "let me know", "tell me", "drop a", "reply",
    "save this", "save it", "bookmark", "share this", "share it",
    "tag a", "tag someone", "tag your",
    "follow", "follow for", "hit follow",
    "link in bio", "click the link", "check the link",
    "dm me", "dm us", "send me",
    "double tap", "like if",
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ScoreBreakdown:
    hook_strength:      float
    cta_presence:       float
    hashtag_quality:    float
    caption_length:     float
    engagement_signals: float


@dataclass
class ContentScoreResult:
    quality_score: float          # 0–1, fed directly into the ML model
    grade: str                    # "Excellent" / "Good" / "Average" / "Needs Work"
    breakdown: ScoreBreakdown
    tips: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"   # emoticons
    "\U0001F300-\U0001F5FF"   # symbols & pictographs
    "\U0001F680-\U0001F6FF"   # transport & map
    "\U0001F1E0-\U0001F1FF"   # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


def _score_hook(caption: str) -> tuple[float, list[str]]:
    """Score the first 120 characters as the hook."""
    tips: list[str] = []
    first_line = caption.strip().split("\n")[0][:120].lower()

    score = 0.40  # baseline

    if "?" in first_line:
        score += 0.30  # question hook — strongest signal

    # Number in the hook ("5 reasons", "3 mistakes", "1 thing")
    if re.search(r"\b\d+\b", first_line):
        score += 0.15

    # Power-word openers
    power_openers = ("stop ", "this ", "how ", "why ", "what ", "never ", "always ",
                     "the truth", "nobody", "everyone", "secret", "mistake")
    if any(first_line.startswith(w) or w in first_line[:40] for w in power_openers):
        score += 0.15

    score = min(score, 1.0)
    if score < 0.55:
        tips.append("Strengthen your opening line — start with a question or a bold claim (e.g. 'Stop doing this…', 'Here's why 90% fail…').")
    return round(score, 3), tips


def _score_cta(caption: str) -> tuple[float, list[str]]:
    tips: list[str] = []
    lower = caption.lower()
    has_cta = any(phrase in lower for phrase in _CTA_PHRASES)
    score = 1.0 if has_cta else 0.25
    if not has_cta:
        tips.append("Add a call-to-action — 'Save this', 'Comment your answer', or 'Share with a friend' can significantly lift engagement.")
    return score, tips


def _score_hashtags(hashtags: str) -> tuple[float, list[str]]:
    tips: list[str] = []
    tags = [t for t in hashtags.replace(",", " ").split() if t.startswith("#")]
    n = len(tags)

    if 3 <= n <= 10:
        score = 1.0
    elif 1 <= n < 3:
        score = 0.60
        tips.append(f"You have {n} hashtag(s) — using 3–10 focused hashtags improves discoverability without looking spammy.")
    elif 11 <= n <= 20:
        score = 0.70
        tips.append("Consider trimming to 3–10 highly relevant hashtags; quality beats quantity.")
    elif n > 20:
        score = 0.35
        tips.append(f"{n} hashtags is too many — Instagram may limit your reach. Use 3–10 focused ones instead.")
    else:
        score = 0.15
        tips.append("No hashtags found — add 3–10 relevant hashtags to help people discover your Reel.")

    return score, tips


def _score_length(caption: str) -> tuple[float, list[str]]:
    tips: list[str] = []
    n = len(caption.strip())

    if 100 <= n <= 300:
        score = 1.0
    elif 50 <= n < 100:
        score = 0.70
        tips.append("Your caption is a little short — 100–300 characters tends to drive better engagement.")
    elif 300 < n <= 600:
        score = 0.80
    elif n > 600:
        score = 0.50
        tips.append("Long captions work for storytelling but can lose readers on mobile. Consider trimming to under 300 characters or breaking with line breaks.")
    else:
        score = 0.25
        tips.append("Caption is very short — give viewers context or a reason to engage.")

    return round(score, 3), tips


def _score_engagement_signals(caption: str) -> tuple[float, list[str]]:
    emoji_count    = len(_EMOJI_RE.findall(caption))
    question_count = caption.count("?")
    exclaim_count  = caption.count("!")

    score = 0.30
    score += min(emoji_count  * 0.08, 0.30)   # emojis add up to 0.30
    score += min(question_count * 0.15, 0.25)  # questions add up to 0.25
    score += min(exclaim_count  * 0.05, 0.15)  # exclamation marks add up to 0.15
    return min(round(score, 3), 1.0), []


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

_WEIGHTS = {
    "hook_strength":      0.30,
    "cta_presence":       0.25,
    "hashtag_quality":    0.20,
    "caption_length":     0.15,
    "engagement_signals": 0.10,
}


def score_content(caption: str, hashtags: str = "") -> ContentScoreResult:
    """
    Score a post's content quality from 0 to 1.

    Args:
        caption:   The full Reel caption text.
        hashtags:  Hashtag string, e.g. '#fitness #workout' or comma-separated.
                   Can also be embedded inside the caption — the scorer handles both.

    Returns:
        ContentScoreResult with quality_score, grade, breakdown, and tips.
    """
    caption  = (caption  or "").strip()
    hashtags = (hashtags or "").strip()

    # If no separate hashtag field, extract tags from caption
    if not hashtags:
        hashtags = " ".join(w for w in caption.split() if w.startswith("#"))

    hook_score,  hook_tips  = _score_hook(caption)
    cta_score,   cta_tips   = _score_cta(caption)
    tag_score,   tag_tips   = _score_hashtags(hashtags)
    len_score,   len_tips   = _score_length(caption)
    eng_score,   _          = _score_engagement_signals(caption)

    breakdown = ScoreBreakdown(
        hook_strength      = hook_score,
        cta_presence       = cta_score,
        hashtag_quality    = tag_score,
        caption_length     = len_score,
        engagement_signals = eng_score,
    )

    quality_score = (
        hook_score  * _WEIGHTS["hook_strength"]      +
        cta_score   * _WEIGHTS["cta_presence"]       +
        tag_score   * _WEIGHTS["hashtag_quality"]    +
        len_score   * _WEIGHTS["caption_length"]     +
        eng_score   * _WEIGHTS["engagement_signals"]
    )
    quality_score = round(quality_score, 3)

    if quality_score >= 0.80:
        grade = "Excellent"
    elif quality_score >= 0.65:
        grade = "Good"
    elif quality_score >= 0.45:
        grade = "Average"
    else:
        grade = "Needs Work"

    # Collect tips, most impactful first (hook > cta > hashtags > length)
    all_tips = hook_tips + cta_tips + tag_tips + len_tips
    return ContentScoreResult(
        quality_score = quality_score,
        grade         = grade,
        breakdown     = breakdown,
        tips          = all_tips[:3],  # surface max 3 actionable tips
    )
