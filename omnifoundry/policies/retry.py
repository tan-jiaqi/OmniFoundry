"""RetryPolicy: after a candidate fails a check, decide whether and how to regenerate.

Signature: `decide(ctx, candidate, verdict, attempt) -> dict | None`
  None            -> give up on this task type for this video
  {"feedback": s} -> regenerate with this feedback appended (OmniQA behaviour)
Further keys a future policy may return (the generate stage honours them):
  "task_id": switch task type, "role": use another generator role (e.g. a stronger model for hard cases).
"""

# Failures that a regeneration with feedback can plausibly fix. Others (e.g. a video with no usable
# audio) are not worth another call.
FIXABLE = {
    "dag_invalid", "missing_modality", "quote_not_in_captions", "timestamp_not_in_captions",
    "simultaneity_gap", "music_anchor", "options_invalid", "answer_invalid",
    "solver_disagrees", "solver_unsupported", "blind_correct", "presupposition_unsupported",
}


def feedback_text(verdict):
    detail = verdict.get("detail", {})
    hint = detail.get("hint") or detail.get("message") or ""
    return f"[{verdict['check']}] {verdict['reason']}" + (f": {hint}" if hint else "")


def with_feedback(ctx, candidate, verdict, attempt):
    max_attempts = int(ctx.get("generate.max_attempts", 3))
    if attempt + 1 >= max_attempts or verdict["reason"] not in FIXABLE:
        return None
    return {"feedback": feedback_text(verdict)}


def never(ctx, candidate, verdict, attempt):
    return None


RETRY = {"with_feedback": with_feedback, "never": never}
