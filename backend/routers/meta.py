from fastapi import APIRouter
from pydantic import BaseModel

from ai_provider import get_provider
from cluster_config import CLUSTER_NICHES

router = APIRouter(prefix="/meta", tags=["meta"])


# ---------------------------------------------------------------------------
# Niches
# ---------------------------------------------------------------------------

@router.get(
    "/niches",
    summary="List all niche clusters",
    description=(
        "Returns the current model's cluster → niche mapping. "
        "The frontend fetches this at runtime — never hardcode cluster IDs or tiers client-side. "
        "Updated by editing cluster_config.py after a model retrain."
    ),
)
def list_niches() -> list[dict]:
    return CLUSTER_NICHES


# ---------------------------------------------------------------------------
# Content scoring
# ---------------------------------------------------------------------------

class ContentScoreRequest(BaseModel):
    caption:  str
    hashtags: str = ""


class ScoreBreakdownOut(BaseModel):
    hook_strength:      float
    cta_presence:       float
    hashtag_quality:    float
    caption_length:     float
    engagement_signals: float


class ContentScoreResponse(BaseModel):
    quality_score: float
    grade:         str
    breakdown:     ScoreBreakdownOut
    tips:          list[str]


@router.post(
    "/score-content",
    response_model=ContentScoreResponse,
    summary="Score post content quality",
    description=(
        "Analyzes a Reel caption and hashtags. "
        "Uses OpenAI (gpt-4o-mini) when OPENAI_API_KEY is set, "
        "otherwise falls back to the built-in heuristic scorer. "
        "Returns a 0–1 quality_score that maps to the content_quality ML feature."
    ),
)
def score_post_content(body: ContentScoreRequest) -> ContentScoreResponse:
    result = get_provider().score_content(caption=body.caption, hashtags=body.hashtags)
    return ContentScoreResponse(
        quality_score=result.quality_score,
        grade=result.grade,
        breakdown=ScoreBreakdownOut(
            hook_strength      = result.breakdown.hook_strength,
            cta_presence       = result.breakdown.cta_presence,
            hashtag_quality    = result.breakdown.hashtag_quality,
            caption_length     = result.breakdown.caption_length,
            engagement_signals = result.breakdown.engagement_signals,
        ),
        tips=result.tips,
    )


# ---------------------------------------------------------------------------
# Niche detection
# ---------------------------------------------------------------------------

class NicheDetectRequest(BaseModel):
    caption:  str
    hashtags: str = ""


class NicheDetectResponse(BaseModel):
    cluster_id: int
    confidence: float   # 0–1
    reasoning:  str


@router.post(
    "/detect-niche",
    response_model=NicheDetectResponse,
    summary="Auto-detect niche from caption",
    description=(
        "Detects the best-matching content niche cluster from a Reel caption and hashtags. "
        "Uses OpenAI when available, keyword matching otherwise. "
        "The frontend should pre-fill the niche dropdown with the result "
        "but keep the dropdown editable so the user can override."
    ),
)
def detect_niche(body: NicheDetectRequest) -> NicheDetectResponse:
    result = get_provider().detect_niche(caption=body.caption, hashtags=body.hashtags)
    return NicheDetectResponse(
        cluster_id=result.cluster_id,
        confidence=result.confidence,
        reasoning=result.reasoning,
    )
