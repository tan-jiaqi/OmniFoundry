"""Quality tiers: how a candidate that failed only soft checks is kept instead of discarded.

A check is hard or soft (`checks.<name>.severity`, or the class default). A hard failure means the answer
itself is unreliable: the question is discarded. A soft failure means the answer stands but the question is
easier than intended (guessable from wording, or answerable from one modality). Those are kept at a lower
tier, with the reasons recorded, so training can weight or filter by tier:

    high     no soft failure and neither modality alone answers (av_strict)
    medium   one modality alone answers, but the other modality's clues were verified against the media,
             so they corroborate the answer; the thinking is written as corroboration, not as a chain
    low      guessable from the question alone, or neither modality is needed

Rationale: OmniQA (reference/OmniReasoning) discards nothing at this point, because its own shortcut test
only ever shows the model ONE random event segment (`verification/modality_gate.py:425-460`), so almost
everything passes. Our gates compare against the FULL single-modality captions and real muted / audio-only
media, which is much stricter; tiering keeps that strictness as information instead of as data loss.
"""

# Soft by default: everything that leaves the answer correct but makes the question easier.
SOFT_REASONS = {
    "blind_correct", "answer_length_leak",
    "audio_only_shortcut", "visual_only_shortcut", "video_only_shortcut",
    "both_modalities_shortcut", "without_clues_shortcut",
}
TIERS = ("high", "medium", "low")
LOW_REASONS = {"blind_correct", "both_modalities_shortcut", "without_clues_shortcut"}


def severity(ctx, check_name, reason):
    """'hard' (discard) or 'soft' (keep at a lower tier)."""
    configured = (ctx.get(f"checks.{check_name}", {}) or {}).get("severity")
    if configured in ("hard", "soft"):
        return configured
    return "soft" if reason in SOFT_REASONS else "hard"


def tier(soft_reasons, tags):
    """Quality tier from the soft failures a candidate collected and its modality tags."""
    reasons = set(soft_reasons)
    if reasons & LOW_REASONS:
        return "low"
    if reasons:
        return "medium"
    return "high" if "av_strict" in (tags or []) else "medium"


def corroborating_modality(tags):
    """For a medium question: the modality that supports the answer without being required.
    audio_sufficient -> the visuals corroborate; visual_sufficient -> the audio corroborates."""
    if "audio_sufficient" in (tags or []):
        return "visual"
    if "visual_sufficient" in (tags or []):
        return "audio"
    return None
