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

from cluster_config import CLUSTER_NICHES, NICHE_KEYWORDS  # CLUSTER_NICHES used for niche label lookup
from content_scorer import (
    ContentScoreResult,
    ScoreBreakdown,
    score_content as _heuristic_score,
)
from prompts import build_detect_niche_prompt, build_score_content_prompt


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

class HeuristicProvider(AIProvider):
    """
    Fully offline provider.  Uses the existing rule-based content scorer and
    keyword matching for niche detection.  No latency, no cost, no API key.
    """

    def score_content(self, caption: str, hashtags: str) -> ContentScoreResult:
        return _heuristic_score(caption=caption, hashtags=hashtags)

    def detect_niche(self, caption: str, hashtags: str) -> NicheDetectionResult:
        text = (caption + " " + hashtags).lower()

        scores: dict[int, int] = {cid: 0 for cid in NICHE_KEYWORDS}
        for cluster_id, keywords in NICHE_KEYWORDS.items():
            scores[cluster_id] = sum(1 for kw in keywords if kw in text)

        best_id    = max(scores, key=lambda k: scores[k])
        best_count = scores[best_id]

        if best_count == 0:
            return NicheDetectionResult(
                cluster_id=8,
                confidence=0.28,
                reasoning="No strong niche signals detected — defaulted to Lifestyle.",
            )

        # Confidence calibrated to match OpenAI provider's scale:
        #   1 keyword match  → ~0.45 (weak signal)
        #   2 keyword matches → ~0.55
        #   3–4 matches      → ~0.65–0.70 (reasonable guess)
        #   5+ matches       → ~0.75 (strong signal, hard cap at 0.78 for heuristic)
        second_best = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0
        separation  = best_count - second_best   # how far ahead of the next best niche

        base_confidence = min(0.38 + best_count * 0.08, 0.72)
        if separation >= 3:
            base_confidence = min(base_confidence + 0.06, 0.78)

        niche_label = next(
            (n["label"] for n in CLUSTER_NICHES if n["cluster_id"] == best_id),
            "Unknown",
        )
        return NicheDetectionResult(
            cluster_id=best_id,
            confidence=round(base_confidence, 2),
            reasoning=(
                f"Matched {best_count} keyword(s) for {niche_label}"
                + (f" ({separation} ahead of next closest niche)." if separation > 0 else ".")
            ),
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
        prompt = build_score_content_prompt(caption, hashtags)

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
        prompt = build_detect_niche_prompt(caption, hashtags)
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
