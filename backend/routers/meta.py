from fastapi import APIRouter
from pydantic import BaseModel

from cluster_config import CLUSTER_NICHES
from content_scorer import score_content

router = APIRouter(prefix="/meta", tags=["meta"])


@router.get(
    "/niches",
    summary="List all niche clusters",
    description=(
        "Returns the current model's cluster → niche mapping. "
        "The frontend must fetch this at runtime — never hardcode cluster IDs or tiers client-side. "
        "Updated by editing cluster_config.py after a model retrain."
    ),
)
def list_niches() -> list[dict]:
    return CLUSTER_NICHES


# ---------------------------------------------------------------------------
# Content scoring
# ---------------------------------------------------------------------------

class ContentScoreRequest(BaseModel):
    caption:   str
    hashtags:  str = ""


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
        "Analyzes a Reel caption and hashtags using rule-based signals "
        "(hook strength, CTA presence, hashtag count, caption length, engagement triggers). "
        "Returns a 0–1 quality_score that maps directly to the content_quality ML feature — "
        "no star rating required from the user."
    ),
)
def score_post_content(body: ContentScoreRequest) -> ContentScoreResponse:
    result = score_content(caption=body.caption, hashtags=body.hashtags)
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
