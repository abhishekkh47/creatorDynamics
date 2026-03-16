"""
Rule-based content quality scorer  (offline / heuristic provider).

Scores a Reel caption and hashtags on five signals correlated with early
engagement.  Returns a 0–1 quality_score fed directly into the ML model's
`content_quality` feature.

This file is the offline fallback — it runs with no external dependencies
when OpenAI is disabled.  It must stay calibrated with the OpenAI provider
so that switching between providers does not change the feature distribution
that the model was trained on.

--- Scoring signals and weights ---

| Signal            | Weight | Core question                                      |
|-------------------|--------|----------------------------------------------------|
| hook_strength     | 0.30   | Does the first line stop the scroll?               |
| cta_presence      | 0.25   | Does the caption direct the viewer to DO something?|
| hashtag_quality   | 0.20   | Are hashtags the right count and variety?          |
| caption_length    | 0.15   | Is the caption in the engagement-optimal range?    |
| engagement_signals| 0.10   | Structure, emojis, in-body questions               |

--- Calibration targets ---

These reference points align this scorer with the OpenAI provider.
When updating, verify that all three still hold:

  Poor:    "Gym workout 💪 #gym"            →  ~0.30 (Needs Work)
  Average: "Morning run done! What's yours? #running #fitness #motivation"
                                             →  ~0.55 (Average)
  Good:    strong hook + CTA + 5 hashtags + good length  →  ~0.75 (Good)
  Excellent: POV hook + urgency + explicit CTA + 5-10 hashtags + story body
                                             →  ~0.85 (Excellent)

--- Updating this scorer ---

Adjust the WEIGHTS dict or individual _score_* functions here.
This file has no dependency on the ML models and can be updated independently.
Run the calibration examples above manually to verify grade boundaries hold.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Hook pattern definitions
# Each tuple: (compiled_regex, score_boost, scope_chars)
# scope_chars = how many chars from the start of the caption to search in
# ---------------------------------------------------------------------------

_HOOK_PATTERNS: list[tuple[re.Pattern[str], float, int]] = [
    # Question in the hook — single strongest signal
    (re.compile(r"\?"),                                                   0.30, 120),

    # POV hook (dominant format on Reels)
    (re.compile(r"\bpov\b"),                                              0.25, 80),

    # Urgency / imperative openers
    (re.compile(r"\b(stop|quit|never|always|warning|listen up|attention)\b"), 0.22, 60),

    # Curiosity gap patterns
    (re.compile(
        r"(the truth about|nobody tells you|what they don.t tell|secret to|"
        r"the real reason|here.s why|why most|the one thing|what actually|"
        r"i.ll never|this is why|no one talks about)"
    ),                                                                    0.20, 120),

    # Story / transformation hook
    (re.compile(
        r"(how i |i lost |i gained |i went from|the day i |when i |"
        r"my journey|changed my life|this changed|it changed)"
    ),                                                                    0.18, 120),

    # Number-based hook ("5 mistakes", "3 reasons", "10k in 30 days", "90%")
    (re.compile(r"\b\d+[kK%]?\b"),                                        0.15, 120),

    # Pattern interrupt / contrarian openers
    (re.compile(
        r"(unpopular opinion|hot take|plot twist|am i the only|"
        r"hear me out|controversial|wait,|hold on,)"
    ),                                                                    0.15, 120),

    # Ellipsis / cliffhanger at end of first line
    (re.compile(r"(\.{3}|…)\s*$", re.MULTILINE),                         0.10, 200),
]

_HOOK_BASELINE = 0.28   # Score for a caption with no recognisable hook patterns


# ---------------------------------------------------------------------------
# CTA bank — split into strong (explicit) and soft (engagement-driving)
# ---------------------------------------------------------------------------

# Strong CTAs: direct, unambiguous instructions → full score
_CTA_STRONG = [
    "save this", "save it", "save for later", "bookmark this", "pin this", "pin for later",
    "comment below", "comment your", "drop a comment", "drop your", "leave a comment",
    "comment '", 'comment "',
    "share this", "share it", "share with", "send this to", "send to",
    "tag a", "tag someone", "tag your", "tag a friend",
    "follow me", "follow for more", "hit follow", "click follow",
    "link in bio", "click the link", "check the link", "head to the link",
    "dm me", "dm us", "send me a dm", "message me",
    "sign up", "subscribe", "join", "click here",
    "swipe up", "check out my",
]

# Soft CTAs: engagement questions that drive comments
_CTA_SOFT = [
    "what do you think", "what would you", "would you", "have you ever",
    "agree or disagree", "agree?", "thoughts?", "opinions?",
    "yes or no", "true or false", "am i right",
    "let me know", "tell me", "tell us", "what's yours",
    "which one", "which do you", "how about you",
    "can you relate", "does this resonate",
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
    quality_score: float           # 0–1, fed directly into the ML model
    grade:         str             # "Excellent" / "Good" / "Average" / "Needs Work"
    breakdown:     ScoreBreakdown
    tips:          list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def _score_hook(caption: str) -> tuple[float, list[str]]:
    """
    Score hook strength from the opening of the caption.

    Runs each pattern against the relevant scope (usually the first 120 chars).
    Boosts are cumulative but capped at 1.0 so multiple patterns don't
    artificially inflate the score.
    """
    tips: list[str] = []
    lower = caption.strip().lower()
    score = _HOOK_BASELINE

    for pattern, boost, scope in _HOOK_PATTERNS:
        window = lower[:scope]
        if pattern.search(window):
            score += boost

    score = min(round(score, 3), 1.0)

    if score < 0.50:
        tips.append(
            "Strengthen your opening line — try a POV hook, curiosity gap "
            "('The truth about…'), or start with a number ('3 mistakes…')."
        )
    return score, tips


def _score_cta(caption: str) -> tuple[float, list[str]]:
    """
    Score call-to-action strength.

    Strong CTAs (save, comment, share, DM, follow, link in bio) → 1.0
    Soft CTAs (engagement questions like 'Agree?', 'What's yours?') → 0.72
    No CTA detected → 0.20
    """
    tips: list[str] = []
    lower = caption.lower()

    has_strong = any(phrase in lower for phrase in _CTA_STRONG)
    if has_strong:
        return 1.0, []

    has_soft = any(phrase in lower for phrase in _CTA_SOFT)
    if has_soft:
        tips.append(
            "Good engagement question, but adding an explicit action "
            "('Save this', 'Comment below') usually lifts engagement further."
        )
        return 0.72, tips

    tips.append(
        "No call-to-action found. Add one — 'Save this for later', "
        "'Comment your answer below', or 'Share with someone who needs this'."
    )
    return 0.20, tips


def _score_hashtags(hashtags: str) -> tuple[float, list[str]]:
    """
    Score hashtag count.

    3–10 is the Instagram sweet spot for Reels:
    fewer limits discovery, more looks spammy and may reduce distribution.
    """
    tips: list[str] = []
    tags = [t for t in hashtags.replace(",", " ").split() if t.startswith("#")]
    n    = len(tags)

    if 3 <= n <= 10:
        return 1.0, []
    if n == 0:
        tips.append("No hashtags — add 3–10 relevant ones to help people discover your Reel.")
        return 0.10, tips
    if 1 <= n < 3:
        tips.append(f"Only {n} hashtag(s) — 3–10 focused ones improve reach without looking spammy.")
        return 0.55, tips
    if 11 <= n <= 20:
        tips.append("Good variety, but 3–10 targeted hashtags usually outperforms a long list.")
        return 0.70, tips
    # n > 20
    tips.append(
        f"{n} hashtags is too many — Instagram may suppress reach. "
        "Use 3–10 highly focused ones instead."
    )
    return 0.30, tips


def _score_length(caption: str) -> tuple[float, list[str]]:
    """
    Score caption length.

    100–300 chars: ideal for Reels — enough context without losing mobile readers.
    300–600 chars: storytelling range — still good.
    <100 chars:    too short, no context or value.
    >600 chars:    risk losing readers; needs good line-break structure to work.
    """
    tips: list[str] = []
    n    = len(caption.strip())

    if 100 <= n <= 300:
        return 1.0, []
    if 300 < n <= 600:
        return 0.82, []
    if 50 <= n < 100:
        tips.append(
            "Caption is a bit short (100–300 chars drives stronger engagement). "
            "Add context, a story sentence, or a CTA."
        )
        return 0.65, tips
    if n > 600:
        tips.append(
            "Long caption — make sure you're using line breaks to keep it readable on mobile. "
            "Trim to under 500 chars if possible."
        )
        return 0.50, tips
    # n < 50
    tips.append(
        "Caption is very short — give viewers a reason to engage or provide context."
    )
    return 0.20, tips


def _score_engagement_signals(caption: str) -> tuple[float, list[str]]:
    """
    Score engagement signals: emojis, in-body questions, exclamation marks,
    and structural readability (line breaks).

    Line breaks are rewarded because structured, scannable captions retain
    readers longer and correlate with higher comment velocity.
    """
    emoji_count    = len(_EMOJI_RE.findall(caption))
    question_count = caption.count("?")
    exclaim_count  = caption.count("!")
    line_breaks    = caption.count("\n")

    score = 0.22  # lower baseline — this signal should be genuinely earned
    score += min(emoji_count    * 0.07,  0.25)   # cap at 0.25
    score += min(question_count * 0.12,  0.20)   # cap at 0.20
    score += min(exclaim_count  * 0.05,  0.12)   # cap at 0.12
    score += min(line_breaks    * 0.06,  0.18)   # readability cap at 0.18

    return min(round(score, 3), 1.0), []


# ---------------------------------------------------------------------------
# Weights (must sum to 1.0)
# ---------------------------------------------------------------------------

_WEIGHTS: dict[str, float] = {
    "hook_strength":      0.30,
    "cta_presence":       0.25,
    "hashtag_quality":    0.20,
    "caption_length":     0.15,
    "engagement_signals": 0.10,
}

# Grade thresholds — must match the OpenAI provider prompt exactly
_GRADE_THRESHOLDS = [
    (0.80, "Excellent"),
    (0.65, "Good"),
    (0.45, "Average"),
    (0.00, "Needs Work"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_content(caption: str, hashtags: str = "") -> ContentScoreResult:
    """
    Score a post's content quality from 0 to 1.

    Args:
        caption:   The full Reel caption text.
        hashtags:  Hashtag string (space or comma separated).
                   If omitted, hashtags are extracted from the caption body.

    Returns:
        ContentScoreResult with quality_score, grade, breakdown, and tips.
    """
    caption  = (caption  or "").strip()
    hashtags = (hashtags or "").strip()

    # Extract hashtags from caption body if none supplied separately
    if not hashtags:
        hashtags = " ".join(w for w in caption.split() if w.startswith("#"))

    hook_score, hook_tips = _score_hook(caption)
    cta_score,  cta_tips  = _score_cta(caption)
    tag_score,  tag_tips  = _score_hashtags(hashtags)
    len_score,  len_tips  = _score_length(caption)
    eng_score,  _         = _score_engagement_signals(caption)

    quality_score = round(
        hook_score * _WEIGHTS["hook_strength"]      +
        cta_score  * _WEIGHTS["cta_presence"]       +
        tag_score  * _WEIGHTS["hashtag_quality"]    +
        len_score  * _WEIGHTS["caption_length"]     +
        eng_score  * _WEIGHTS["engagement_signals"],
        3,
    )

    grade = next(g for threshold, g in _GRADE_THRESHOLDS if quality_score >= threshold)

    # Surface the most impactful tips first (hook > cta > hashtags > length)
    all_tips = hook_tips + cta_tips + tag_tips + len_tips
    return ContentScoreResult(
        quality_score = quality_score,
        grade         = grade,
        breakdown     = ScoreBreakdown(
            hook_strength      = hook_score,
            cta_presence       = cta_score,
            hashtag_quality    = tag_score,
            caption_length     = len_score,
            engagement_signals = eng_score,
        ),
        tips = all_tips[:3],
    )
