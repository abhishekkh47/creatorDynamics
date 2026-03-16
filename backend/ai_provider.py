"""
AI Provider — plug-n-play strategy pattern.

The rest of the codebase calls get_provider() and uses the returned object.
Which implementation runs is controlled entirely by the environment:

    OPENAI_API_KEY set and non-empty  →  OpenAIProvider  (gpt-4o-mini)
    OPENAI_API_KEY missing or empty   →  HeuristicProvider (fully offline)

To switch off OpenAI at any time: remove or blank OPENAI_API_KEY in .env,
then restart the server.  No code changes are needed anywhere else.

--- Adding a new provider (e.g. Anthropic, Gemini) ---
1. Subclass AIProvider and implement score_content() and detect_niche().
2. Add a detection branch in get_provider().
3. That's it.  All routers and callers are unchanged.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Optional

from cluster_config import CLUSTER_NICHES
from content_scorer import (
    ContentScoreResult,
    ScoreBreakdown,
    score_content as _heuristic_score,
)


# ---------------------------------------------------------------------------
# Shared result type
# ---------------------------------------------------------------------------

class NicheDetectionResult:
    def __init__(self, cluster_id: int, confidence: float, reasoning: str) -> None:
        self.cluster_id = cluster_id
        self.confidence = confidence
        self.reasoning  = reasoning


# ---------------------------------------------------------------------------
# Abstract base — the contract every provider must fulfill
# ---------------------------------------------------------------------------

class AIProvider(ABC):
    @abstractmethod
    def score_content(self, caption: str, hashtags: str) -> ContentScoreResult:
        """Score a post's content quality.  Returns a 0-1 score + breakdown."""

    @abstractmethod
    def detect_niche(self, caption: str, hashtags: str) -> NicheDetectionResult:
        """Detect which niche cluster best matches the post content."""


# ---------------------------------------------------------------------------
# Heuristic provider — fully offline, zero external dependencies
# ---------------------------------------------------------------------------

# Keyword bank mapping cluster_id → relevant terms
_NICHE_KEYWORDS: dict[int, list[str]] = {
    0:  ["fitness", "workout", "gym", "health", "exercise", "training", "muscle", "diet", "hiit", "cardio"],
    1:  ["beauty", "makeup", "skincare", "cosmetic", "glow", "foundation", "lipstick", "serum", "blush"],
    2:  ["food", "recipe", "cooking", "eat", "delicious", "kitchen", "meal", "dish", "bake", "cuisine"],
    3:  ["travel", "adventure", "explore", "destination", "vacation", "trip", "wanderlust", "journey", "abroad"],
    4:  ["fashion", "style", "outfit", "clothing", "ootd", "wear", "wardrobe", "trend", "streetwear"],
    5:  ["finance", "money", "invest", "stock", "wealth", "budget", "crypto", "trading", "savings", "passive income"],
    6:  ["tech", "technology", "gadget", "software", "code", "app", "ai", "innovation", "programming", "developer"],
    7:  ["funny", "comedy", "humor", "laugh", "joke", "meme", "entertainment", "viral", "trending", "skit"],
    8:  ["lifestyle", "daily", "routine", "productivity", "life", "morning", "evening", "vlog", "everyday"],
    9:  ["education", "learn", "tutorial", "how to", "tips", "knowledge", "study", "guide", "explained", "facts"],
    10: ["music", "dance", "song", "artist", "perform", "sing", "choreography", "beat", "lyrics", "album"],
    11: ["gaming", "game", "play", "gamer", "esports", "stream", "twitch", "console", "fps", "rpg"],
    12: ["parenting", "mom", "dad", "baby", "kids", "family", "children", "parent", "toddler", "newborn"],
    13: ["art", "creative", "paint", "draw", "design", "artist", "illustration", "craft", "portfolio", "sketch"],
    14: ["sports", "athlete", "football", "basketball", "soccer", "cricket", "match", "team", "training", "coach"],
    15: ["business", "entrepreneur", "startup", "marketing", "brand", "strategy", "growth", "sales", "founder"],
    16: ["pets", "dog", "cat", "animal", "puppy", "kitten", "paw", "fur", "pet owner", "rescue"],
    17: ["mental health", "wellness", "mindfulness", "meditation", "anxiety", "self care", "therapy", "healing"],
    18: ["diy", "home", "decor", "garden", "renovation", "interior", "craft", "handmade", "upcycle", "thrift"],
    19: ["news", "politics", "opinion", "commentary", "current events", "world", "update", "breaking"],
}


class HeuristicProvider(AIProvider):
    """
    Fully offline provider.  Uses the existing rule-based content scorer and
    keyword matching for niche detection.  No latency, no cost, no API key.
    """

    def score_content(self, caption: str, hashtags: str) -> ContentScoreResult:
        return _heuristic_score(caption=caption, hashtags=hashtags)

    def detect_niche(self, caption: str, hashtags: str) -> NicheDetectionResult:
        text = (caption + " " + hashtags).lower()

        scores: dict[int, int] = {cid: 0 for cid in _NICHE_KEYWORDS}
        for cluster_id, keywords in _NICHE_KEYWORDS.items():
            scores[cluster_id] = sum(1 for kw in keywords if kw in text)

        best_id    = max(scores, key=lambda k: scores[k])
        best_count = scores[best_id]

        if best_count == 0:
            return NicheDetectionResult(
                cluster_id=8,
                confidence=0.25,
                reasoning="No strong niche signals detected — defaulted to Lifestyle.",
            )

        confidence  = min(0.40 + best_count * 0.12, 0.82)
        niche_label = next(
            (n["label"] for n in CLUSTER_NICHES if n["cluster_id"] == best_id),
            "Unknown",
        )
        return NicheDetectionResult(
            cluster_id=best_id,
            confidence=round(confidence, 2),
            reasoning=f"Matched {best_count} keyword(s) for {niche_label}.",
        )


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

class OpenAIProvider(AIProvider):
    """
    OpenAI-powered provider using gpt-4o-mini.

    Falls back gracefully per-call: if an OpenAI call fails for any reason
    (rate limit, network error, malformed response), the heuristic provider
    is used as a fallback so the endpoint never errors out.
    """

    _MODEL = "gpt-4o-mini"

    def __init__(self, api_key: str) -> None:
        from openai import OpenAI  # type: ignore[import-untyped]  # lazy import — only required when this provider is active
        self._client   = OpenAI(api_key=api_key)
        self._fallback = HeuristicProvider()

    def score_content(self, caption: str, hashtags: str) -> ContentScoreResult:
        try:
            return self._openai_score(caption, hashtags)
        except Exception as exc:
            print(f"[ai] OpenAI score_content failed ({exc}) — using heuristic fallback")
            return self._fallback.score_content(caption, hashtags)

    def detect_niche(self, caption: str, hashtags: str) -> NicheDetectionResult:
        try:
            return self._openai_detect(caption, hashtags)
        except Exception as exc:
            print(f"[ai] OpenAI detect_niche failed ({exc}) — using heuristic fallback")
            return self._fallback.detect_niche(caption, hashtags)

    # ------------------------------------------------------------------
    # Private OpenAI calls
    # ------------------------------------------------------------------

    def _openai_score(self, caption: str, hashtags: str) -> ContentScoreResult:
        prompt = (
            "You are an expert Instagram content analyst. "
            "Score this Reel caption and hashtags on 5 quality signals.\n\n"
            f"Caption:\n{caption}\n\n"
            f"Hashtags: {hashtags or '(none)'}\n\n"
            "Rate each signal 0.0–1.0:\n"
            "- hook_strength: Is the opening line compelling? "
            "Questions, numbers, curiosity gaps, bold claims score higher.\n"
            "- cta_presence: Is there a clear call-to-action "
            "(save, comment, share, follow, link in bio, DM)?\n"
            "- hashtag_quality: Are 3–10 focused hashtags used? "
            "Too few (<3) or too many (>15) reduces the score.\n"
            "- caption_length: Is caption 100–300 chars (ideal range)? "
            "Very short or very long reduces score.\n"
            "- engagement_signals: Emojis, in-body questions, exclamation marks.\n\n"
            "Also provide:\n"
            "- quality_score: weighted 0.0–1.0 overall (hook 30%, cta 25%, "
            "hashtag 20%, length 15%, engagement 10%)\n"
            "- grade: exactly one of: Excellent, Good, Average, Needs Work\n"
            "- tips: array of up to 3 short, specific, actionable tips "
            "(empty array [] if grade is Excellent)\n\n"
            "Return ONLY valid JSON:\n"
            '{"quality_score":0.82,"grade":"Good","breakdown":{"hook_strength":0.9,'
            '"cta_presence":0.8,"hashtag_quality":0.7,"caption_length":0.8,'
            '"engagement_signals":0.75},"tips":["Add a CTA like \'Save this\'"]}'
        )

        response = self._client.chat.completions.create(
            model=self._MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=400,
        )
        data = json.loads(response.choices[0].message.content)
        bd   = data["breakdown"]

        return ContentScoreResult(
            quality_score=round(float(data["quality_score"]), 3),
            grade=data["grade"],
            breakdown=ScoreBreakdown(
                hook_strength      = float(bd["hook_strength"]),
                cta_presence       = float(bd["cta_presence"]),
                hashtag_quality    = float(bd["hashtag_quality"]),
                caption_length     = float(bd["caption_length"]),
                engagement_signals = float(bd["engagement_signals"]),
            ),
            tips=data.get("tips", [])[:3],
        )

    def _openai_detect(self, caption: str, hashtags: str) -> NicheDetectionResult:
        niche_list = "\n".join(
            f"  cluster_id={n['cluster_id']}: {n['label']} ({n['tier']} tier)"
            for n in CLUSTER_NICHES
        )
        prompt = (
            "You are an Instagram content categorization expert.\n"
            "Given this Reel caption and hashtags, identify the single best-matching "
            "content niche from the list below.\n\n"
            f"Caption:\n{caption}\n\n"
            f"Hashtags: {hashtags or '(none)'}\n\n"
            f"Available niches:\n{niche_list}\n\n"
            "Pick the PRIMARY niche. If multiple fit, choose the most dominant one.\n"
            "confidence should reflect your certainty (0.0–1.0).\n"
            "reasoning should be one concise sentence.\n\n"
            "Return ONLY valid JSON:\n"
            '{"cluster_id":7,"confidence":0.92,"reasoning":"Caption is about comedy skits."}'
        )
        response = self._client.chat.completions.create(
            model=self._MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=120,
        )
        data = json.loads(response.choices[0].message.content)
        return NicheDetectionResult(
            cluster_id=int(data["cluster_id"]),
            confidence=round(float(data["confidence"]), 2),
            reasoning=data.get("reasoning", ""),
        )


# ---------------------------------------------------------------------------
# Factory — the ONLY function the rest of the codebase should call
# ---------------------------------------------------------------------------

_provider: Optional[AIProvider] = None


def get_provider() -> AIProvider:
    """
    Returns the active AI provider.  Cached after first call.
    Restart the server to pick up a changed OPENAI_API_KEY.

    Switching modes:
      Turn ON OpenAI  → set OPENAI_API_KEY=sk-... in .env, restart
      Turn OFF OpenAI → remove or blank OPENAI_API_KEY in .env, restart
    """
    global _provider
    if _provider is None:
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if key:
            try:
                _provider = OpenAIProvider(api_key=key)
                print("[ai] Provider: OpenAI (gpt-4o-mini) — with heuristic fallback")
            except Exception as exc:
                print(f"[ai] OpenAI provider failed to init ({exc}) — using heuristic")
                _provider = HeuristicProvider()
        else:
            _provider = HeuristicProvider()
            print("[ai] Provider: Heuristic (OPENAI_API_KEY not set)")
    return _provider
