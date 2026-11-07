"""S2 generate (ROV, reasoning over video): CaptionBundle -> Candidates.

The AnchorMiner enumerates cross-modal anchors in the captions and the TaskSelector proposes task types;
each (anchor, task) pair is one generation attempt, so one video yields several questions instead of one
(JointAVBench fans out over segment x task: reference/JointAVBench/generation_pipeline/generate_qa.py:93-151).
The generator writes an MCQ with an evidence chain, or replies `[Unavailable]` when the anchor cannot carry
that task -- a free skip that costs no checks. The `generate.checks` chain then screens it; on failure the
RetryPolicy decides whether to regenerate with feedback, and a candidate that only failed soft checks is
kept at a lower quality tier (quality.py). Every attempt and verdict is written out.
"""
import random
import re
import sys

from ..checks import chain, chain_prompts, chain_roles, run_chain
from ..checks.gate import modality_tags
from ..jsonio import digest, parse_json_text, read_jsonl
from ..policies.anchor_miner import MINERS, anchor_text
from ..policies.retry import RETRY
from ..policies.selector import PROMPTS as SELECTOR_PROMPTS, SELECTORS
from ..quality import corroborating_modality, tier
from ..schema import AUDIO_MODALITIES, MODALITIES, Candidate, fmt_range, letters, parse_range
from .. import lineage
from .common import captions_text, lineage_extra

NAME, VERSION = "generate", 1
CONFIG_KEYS = ("generate", "tasks", "policies.selector", "policies.retry", "policies.anchors", "anchors")

USES_CATALOG = True


def _checks(ctx):
    return ctx.get("generate.checks", ["structure", "grounding", "solver", "blind"])


def extra_config_keys(ctx):
    return [f"checks.{n}" for n in _checks(ctx)]


def prompts(ctx):
    return ["generate"] + SELECTOR_PROMPTS[ctx.get("policies.selector", "llm")] + chain_prompts(_checks(ctx))


def roles(ctx):
    sel = ["selector"] if ctx.get("policies.selector", "llm") == "llm" else []
    return ["generator"] + sel + chain_roles(ctx, _checks(ctx))


def inputs(ctx, prev_dir):
    return read_jsonl(prev_dir / "out.jsonl")


# ---- normalization ------------------------------------------------------------------------------

def _clean_option(text):
    return re.sub(r"^\s*\(?[A-Ha-h][\.\):]\s*", "", str(text)).strip()


def shuffle_options(options, answer, seed):
    """Deterministic shuffle (seeded by content) so the gold letter is uniformly spread."""
    n = len(options)
    order = list(range(n))
    random.Random(seed).shuffle(order)
    ls = letters(n)
    bodies = [_clean_option(o) for o in options]
    new = [f"{ls[i]}. {bodies[j]}" for i, j in enumerate(order)]
    old_idx = ls.index(answer) if answer in ls else None
    new_answer = ls[order.index(old_idx)] if old_idx is not None else answer
    return new, new_answer


def normalize(raw, bundle, task_id, attempt, feedback, ctx):
    """Model JSON -> Candidate dict. Raises ValueError with a reason code on unusable output."""
    if not isinstance(raw, dict):
        raise ValueError("options_invalid")
    options = [str(o) for o in raw.get("options", [])]
    answer = str(raw.get("answer", "")).strip().upper()[:1]
    clues = []
    for i, c in enumerate(raw.get("clues", [])):
        mod = str(c.get("modality", "")).lower()
        if mod == "audio":
            mod = "sound"
        if mod not in MODALITIES:
            raise ValueError("dag_invalid")
        try:
            s, e = parse_range(c["timestamp"])
        except (KeyError, ValueError):
            raise ValueError("timestamp_not_in_captions")
        clues.append({"id": str(c.get("id") or f"c{i + 1}"), "modality": mod, "timestamp": fmt_range(s, e),
                      "start_s": s, "end_s": e, "quote": str(c.get("quote", "")), "fact": str(c.get("fact", "")),
                      "caption_ref": str(c.get("caption_ref", ""))})
    steps = []
    for i, s in enumerate(raw.get("reasoning_steps", [])):
        mod = str(s.get("modality", "")).lower()
        steps.append({"step_id": str(s.get("step_id") or f"s{i + 1}"),
                      "modality": "audio" if mod in AUDIO_MODALITIES else mod,
                      "timestamp": str(s.get("timestamp", "")), "content": str(s.get("content", "")),
                      "depends_on": [str(d) for d in s.get("depends_on", [])],
                      "clue_ids": [str(x) for x in s.get("clue_ids", [])]})
    question = str(raw.get("question", "")).strip()
    if not question:
        raise ValueError("options_invalid")
    cid = "q-" + digest(bundle["id"], task_id, attempt, question, options)[:16]
    if ctx.get("generate.shuffle_options", True) and len(options) > 1:
        options, answer = shuffle_options(options, answer, cid)
    return Candidate(
        id=cid, video_id=bundle["video_id"], route="rov", task_type=task_id, question=question, options=options,
        answer=answer, clues=clues, steps=steps, attempt=attempt, feedback=list(feedback),
        lineage=lineage.make(NAME, VERSION, [bundle["id"]], **lineage_extra(ctx, sys.modules[__name__])),
        extra={"single_modality_traps": raw.get("single_modality_traps", []),
               "presuppositions": raw.get("presuppositions", [])},
    ).to_dict()


# ---- process ------------------------------------------------------------------------------------

UNAVAILABLE = "[unavailable]"


def bridge_text(task):
    b = task.get("bridge")
    if not b:
        return ""
    return (f"Bridge: the {b['referent_modality']} identifies what the question is about; the answer must come from "
            f"the {b['answer_modality']}.\nRequired condition: {b['uniqueness']}\n"
            f"Reply [Unavailable] if: {b['unavailable_if']}")


def _generate_once(ctx, bundle, task, anchor, attempt, feedback, role):
    """Returns the parsed JSON, or None when the model declined this anchor with [Unavailable]."""
    fb = "\n".join(f"- {f}" for f in feedback) if feedback else "(none: first attempt)"
    text = ctx.prompts["generate"].render(
        task_id=task["id"], task_name=task["name"], task_kind=task.get("kind", "anchor"),
        task_definition=task["definition"].strip(), task_rules="\n".join(f"- {r}" for r in task.get("rules", [])) or "(none)",
        bridge=bridge_text(task), anchor=anchor_text(bundle, anchor) if anchor else "(no anchor given: choose the strongest cross-modal moment yourself)",
        num_options=ctx.get("checks.structure.num_options", 8), captions=captions_text(bundle), feedback=fb)
    res = ctx.models(role).call([{"type": "text", "text": text}], json=False, tag="generate", sample=attempt,
                                context={"bundle": bundle["id"], "task": task["id"], "anchor": (anchor or {}).get("audio_ref"),
                                         "attempt": attempt})
    body = (res.get("text") or "").strip()
    if body[:13].lower().startswith(UNAVAILABLE):
        return None
    return parse_json_text(body)


def fingerprint(cand):
    """Normalized question text, to drop near-duplicates mined from neighbouring anchors."""
    return re.sub(r"[^a-z0-9 ]", "", cand["question"].lower())[:160]


def work_items(ctx, bundle, picks):
    """(anchor, task_id) pairs to try, best anchor first, cycling through the selected task types.

    Cycling rather than grouping matters: the strongest anchors should be spread over task types, not all
    spent on whichever type the selector named first."""
    anchors = MINERS[ctx.get("policies.anchors", "default")](bundle, ctx.get("anchors", {}))
    if not anchors:
        return [(None, t) for t in picks]
    return [(a, picks[i % len(picks)]) for i, a in enumerate(anchors)]


def process(ctx, bundle):
    """Generate one question per (anchor, task) pair until the per-video quota is met.

    A hard failure discards the attempt. Soft failures (the answer stands, the question is just easier than
    intended) are retried once more for a better question, then kept: the best attempt so far is accepted at
    the quality tier its soft failures imply (omnifoundry/quality.py). `[Unavailable]` skips an anchor for
    free, without spending checks or a retry.
    """
    tasks = {t["id"]: t for t in ctx.catalog.for_route("rov")}
    picks = SELECTORS[ctx.get("policies.selector", "llm")](ctx, bundle, "rov", [])
    retry = RETRY[ctx.get("policies.retry", "with_feedback")]
    checks = chain(_checks(ctx))
    want = int(ctx.get("generate.questions_per_video", 1))
    budget = int(ctx.get("generate.max_total_attempts", 4 * want))
    keep_soft = bool(ctx.get("generate.keep_soft_failures", True))
    # One task type must not take every slot: temporal questions cite the most clues, so they both dominate
    # generation and dominate clue-verification losses (m2b: 196 of 222 attempts, 72 of 80 clue failures).
    max_per_task = int(ctx.get("generate.max_per_task", max(1, -(-want // 2))))
    out, rejected, verdicts, seen, spent, skipped = [], [], [], set(), 0, 0
    accepted_per_task = {}
    if not picks:
        return {"rejected": [{"id": bundle["id"], "video_id": bundle["video_id"], "reason": "no_task_selected"}]}

    def record(cand, vs, soft, anchor):
        """Accept a candidate at the tier its soft failures imply."""
        tags = modality_tags(vs)
        reasons = sorted({v["reason"] for v in soft})
        accepted_per_task[cand["task_type"]] = accepted_per_task.get(cand["task_type"], 0) + 1
        out.append({**cand, **tags, "quality": tier(reasons, tags["tags"]), "soft_failures": reasons,
                    "corroborating_modality": corroborating_modality(tags["tags"]),
                    "anchor": {k: anchor[k] for k in ("audio_ref", "score", "uniqueness", "leak")} if anchor else None})
        seen.add(fingerprint(cand))

    for anchor, task_id in work_items(ctx, bundle, picks):
        if len(out) >= want or spent >= budget:
            break
        if accepted_per_task.get(task_id, 0) >= max_per_task:
            continue
        task, feedback, role, attempt = tasks[task_id], [], "generator", 0
        best = None          # (candidate, verdicts, soft failures) with the fewest soft failures so far
        while True:
            spent += 1
            try:
                raw = _generate_once(ctx, bundle, task, anchor, attempt, feedback, role)
                if raw is None:                      # the model declined this anchor
                    skipped += 1
                    break
                cand = normalize(raw, bundle, task_id, attempt, feedback, ctx)
            except ValueError as exc:
                reason = str(exc) if str(exc) in ("options_invalid", "dag_invalid", "timestamp_not_in_captions") else "options_invalid"
                failure = {"kind": "verdict", "check": "parse", "candidate_id": f"{bundle['id']}/{task_id}/{attempt}", "passed": False,
                           "reason": reason, "detail": {"hint": "output did not follow the required JSON structure"}}
                verdicts.append(failure)
                cand = None
            else:
                if fingerprint(cand) in seen:
                    rejected.append({**cand, "rejected_by": "dedupe", "reason": "duplicate_question"})
                    break
                hard, vs, soft = run_chain(ctx, checks, cand, bundle)
                verdicts += vs
                if hard is None and not soft:
                    record(cand, vs, [], anchor)
                    best = None
                    break
                if hard is None:
                    if best is None or len(soft) < len(best[2]):
                        best = (cand, vs, soft)
                    rejected.append({**cand, "rejected_by": soft[0]["check"], "reason": soft[0]["reason"], "soft": True})
                    failure = soft[0]
                else:
                    rejected.append({**cand, "rejected_by": hard["check"], "reason": hard["reason"]})
                    failure = hard
            decision = retry(ctx, cand, failure, attempt)
            if decision is None or spent >= budget:
                break
            feedback = feedback + [decision["feedback"]]
            task = tasks.get(decision.get("task_id", task["id"]), task)
            role = decision.get("role", role)
            attempt += 1
        if best and keep_soft and len(out) < want and fingerprint(best[0]) not in seen:
            # No clean question came out of the retries: keep the best one, downgraded.
            cand, vs, soft = best
            record(cand, vs, soft, anchor)
    return {"out": out, "rejected": rejected, "verdicts": verdicts,
            "stats": {"attempts": spent, "unavailable": skipped}}
