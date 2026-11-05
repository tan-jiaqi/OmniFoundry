"""S2 generate (ROV, reasoning over video): CaptionBundle -> Candidates.

For each video: the TaskSelector policy proposes task types; for each, the generator writes an MCQ with
an evidence chain; the `generate.checks` chain screens it; on failure the RetryPolicy decides whether to
regenerate with feedback. Every attempt and verdict is kept (rejected attempts go to rejected.jsonl), so
later analysis can see where and why generation fails per task type.
"""
import random
import re
import sys

from ..checks import chain, chain_prompts, chain_roles, run_chain
from ..checks.gate import modality_tags
from ..jsonio import digest, read_jsonl
from ..policies.retry import RETRY
from ..policies.selector import PROMPTS as SELECTOR_PROMPTS, SELECTORS
from ..schema import AUDIO_MODALITIES, MODALITIES, Candidate, fmt_range, letters, parse_range
from .. import lineage
from .common import captions_text, lineage_extra

NAME, VERSION = "generate", 1
CONFIG_KEYS = ("generate", "tasks", "policies.selector", "policies.retry")
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

def _generate_once(ctx, bundle, task, attempt, feedback, role):
    fb = "\n".join(f"- {f}" for f in feedback) if feedback else "(none: first attempt)"
    text = ctx.prompts["generate"].render(
        task_id=task["id"], task_name=task["name"], task_kind=task.get("kind", "anchor"),
        task_definition=task["definition"].strip(), task_rules="\n".join(f"- {r}" for r in task.get("rules", [])) or "(none)",
        num_options=ctx.get("checks.structure.num_options", 8), captions=captions_text(bundle), feedback=fb)
    res = ctx.models(role).call([{"type": "text", "text": text}], tag="generate", sample=attempt,
                                context={"bundle": bundle["id"], "task": task["id"], "attempt": attempt})
    return res["json"]


def process(ctx, bundle):
    tasks = {t["id"]: t for t in ctx.catalog.for_route("rov")}
    picks = SELECTORS[ctx.get("policies.selector", "llm")](ctx, bundle, "rov", [])
    retry = RETRY[ctx.get("policies.retry", "with_feedback")]
    checks = chain(_checks(ctx))
    want = int(ctx.get("generate.questions_per_video", 1))
    out, rejected, verdicts = [], [], []
    if not picks:
        return {"rejected": [{"id": bundle["id"], "video_id": bundle["video_id"], "reason": "no_task_selected"}]}
    for task_id in picks:
        if len(out) >= want:
            break
        task, feedback, role, attempt = tasks[task_id], [], "generator", 0
        fallback = None   # (candidate, verdicts, index after the soft-failing check)
        accepted = False
        while True:
            try:
                cand = normalize(_generate_once(ctx, bundle, task, attempt, feedback, role), bundle, task_id, attempt, feedback, ctx)
            except ValueError as exc:
                reason = str(exc) if str(exc) in ("options_invalid", "dag_invalid", "timestamp_not_in_captions") else "options_invalid"
                v = {"kind": "verdict", "check": "parse", "candidate_id": f"{bundle['id']}/{task_id}/{attempt}", "passed": False,
                     "reason": reason, "detail": {"hint": "output did not follow the required JSON structure"}}
                verdicts.append(v)
                cand, failure = None, v
            else:
                passed, vs, failure = run_chain(ctx, checks, cand, bundle)
                verdicts += vs
                if passed:
                    out.append({**cand, **modality_tags(vs)})
                    accepted = True
                    break
                rejected.append({**cand, "rejected_by": failure["check"], "reason": failure["reason"]})
                if failure.get("detail", {}).get("soft"):
                    fallback = (cand, vs, len(vs))
            decision = retry(ctx, cand, failure, attempt)
            if decision is None:
                break
            feedback = feedback + [decision["feedback"]]
            task = tasks.get(decision.get("task_id", task["id"]), task)
            role = decision.get("role", role)
            attempt += 1
        if not accepted and fallback:
            # No better question came: accept the soft-failed one if it passes the remaining checks.
            cand, vs, done = fallback
            soft = {**vs[-1], "passed": True, "reason": "accepted_fallback"}
            passed, rest, failure = run_chain(ctx, checks[done:], cand, bundle)
            verdicts += [soft] + rest
            if passed:
                out.append({**cand, **modality_tags(vs[:-1] + [soft] + rest), "accepted_as_fallback": True})
            else:
                rejected.append({**cand, "rejected_by": failure["check"], "reason": failure["reason"]})
    return {"out": out, "rejected": rejected, "verdicts": verdicts}
