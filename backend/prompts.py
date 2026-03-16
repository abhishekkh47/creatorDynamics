"""
OpenAI prompt builders.

All prompts used by the OpenAI provider live here — separated from the
provider logic so they can be iterated on independently.

Design notes:
- Each builder is a plain function that takes dynamic inputs and returns a
  string.  No classes, no state.
- Prompt calibration comments live here alongside the prompt text so the
  person editing a prompt can see the intent without diving into ai_provider.py.
- Grade thresholds and signal weights defined here must stay in sync with the
  _WEIGHTS and _GRADE_THRESHOLDS constants in content_scorer.py.

--- Updating a prompt ---
1. Edit the relevant builder function below.
2. Restart the backend (prompts are not cached at the module level beyond the
   normal Python import cache — a restart is all that's needed).
3. Run a quick sanity check via POST /meta/score-content or /meta/detect-niche
   to confirm the new prompt still returns valid JSON.
"""

from __future__ import annotations

from cluster_config import CLUSTER_NICHES


# ---------------------------------------------------------------------------
# Content scoring prompt
# ---------------------------------------------------------------------------

def build_score_content_prompt(caption: str, hashtags: str) -> str:
    """
    Prompt for scoring a Reel caption and hashtags on 5 quality signals.

    Calibration targets (must align with content_scorer.py):
      Excellent  quality_score >= 0.80
      Good       quality_score >= 0.65
      Average    quality_score >= 0.45
      Needs Work quality_score <  0.45
    """
    return (
        "You are an expert Instagram Reels content analyst. "
        "Score the caption and hashtags below on exactly 5 signals.\n\n"
        f"Caption:\n{caption}\n\n"
        f"Hashtags: {hashtags or '(none)'}\n\n"
        "--- Scoring guide (each signal 0.0–1.0) ---\n\n"
        "hook_strength (opening ~120 chars):\n"
        "  0.28 = no recognisable hook (flat opener)\n"
        "  0.50 = mild hook (numbers, generic power word)\n"
        "  0.70 = strong hook (POV, curiosity gap, story opener, urgency word)\n"
        "  0.90 = excellent (question + curiosity gap, or POV + transformation)\n\n"
        "cta_presence:\n"
        "  0.20 = no CTA at all\n"
        "  0.72 = soft CTA (engagement question like 'Agree?', 'What's yours?')\n"
        "  1.00 = explicit CTA (save, comment, share, DM, follow, link in bio)\n\n"
        "hashtag_quality:\n"
        "  0.10 = no hashtags\n"
        "  0.55 = 1–2 hashtags\n"
        "  1.00 = 3–10 hashtags (ideal)\n"
        "  0.70 = 11–20 hashtags\n"
        "  0.30 = >20 hashtags\n\n"
        "caption_length:\n"
        "  0.20 = <50 chars\n"
        "  0.65 = 50–99 chars\n"
        "  1.00 = 100–300 chars (ideal)\n"
        "  0.82 = 301–600 chars\n"
        "  0.50 = >600 chars\n\n"
        "engagement_signals (emojis, questions in body, exclamation marks, line breaks):\n"
        "  0.22 = none\n"
        "  0.50 = moderate (a few emojis + one question)\n"
        "  0.80 = high (structured with line breaks, 3+ emojis, questions)\n\n"
        "--- Overall quality_score ---\n"
        "Apply EXACTLY these weights:\n"
        "  quality_score = hook*0.30 + cta*0.25 + hashtag*0.20 + length*0.15 + engagement*0.10\n\n"
        "--- Grade thresholds (apply strictly) ---\n"
        "  quality_score >= 0.80 → Excellent\n"
        "  quality_score >= 0.65 → Good\n"
        "  quality_score >= 0.45 → Average\n"
        "  quality_score <  0.45 → Needs Work\n\n"
        "tips: up to 3 short, specific, actionable tips ([] if grade is Excellent)\n\n"
        "Return ONLY valid JSON matching this exact shape:\n"
        '{"quality_score":0.74,"grade":"Good","breakdown":{"hook_strength":0.70,'
        '"cta_presence":1.0,"hashtag_quality":1.0,"caption_length":0.82,'
        '"engagement_signals":0.50},"tips":["Open with a curiosity gap to lift CTR."]}'
    )


# ---------------------------------------------------------------------------
# Niche detection prompt
# ---------------------------------------------------------------------------

def build_detect_niche_prompt(caption: str, hashtags: str) -> str:
    """
    Prompt for categorising a Reel into one of the known niche clusters.

    The available niche list is built from CLUSTER_NICHES at call time so
    it stays in sync with cluster_config.py automatically.

    Confidence calibration:
      0.90–1.00  topic is unambiguous (e.g. recipe post → Food & Cooking)
      0.70–0.89  strong signal but slight overlap possible
      0.50–0.69  reasonable guess, some ambiguity
      0.30–0.49  weak signal, low certainty
    """
    niche_list = "\n".join(
        f"  cluster_id={n['cluster_id']}: {n['label']} ({n['tier']} tier)"
        for n in CLUSTER_NICHES
    )
    return (
        "You are an Instagram content categorisation expert.\n"
        "Given the Reel caption and hashtags below, identify the single "
        "best-matching niche from the list.\n\n"
        f"Caption:\n{caption}\n\n"
        f"Hashtags: {hashtags or '(none)'}\n\n"
        f"Available niches:\n{niche_list}\n\n"
        "Rules:\n"
        "- Pick the PRIMARY niche; if multiple fit, choose the most dominant one.\n"
        "- Do NOT pick 'Lifestyle' (cluster_id=8) unless the content is genuinely "
        "generic daily-life content with no stronger niche signal.\n\n"
        "confidence calibration:\n"
        "  0.90–1.00 = topic is unambiguous (e.g. recipe post → Food & Cooking)\n"
        "  0.70–0.89 = strong signal but slight overlap possible\n"
        "  0.50–0.69 = reasonable guess, some ambiguity\n"
        "  0.30–0.49 = weak signal, low certainty\n\n"
        "reasoning: one concise sentence explaining the strongest signal.\n\n"
        "Return ONLY valid JSON:\n"
        '{"cluster_id":2,"confidence":0.91,"reasoning":"Caption describes a homemade recipe with cooking hashtags."}'
    )
