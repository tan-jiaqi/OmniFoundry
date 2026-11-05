"""Gate 2, modality shortcuts. A question is kept only if the joint condition answers it and no restricted
condition does.

`gate_text` (cheap, on captions):
    full          all captions             -> must be correct (majority of trials)
    audio_only    audio captions only      -> must NOT be correct (<= max_correct of trials)
    visual_only   visual captions only     -> must NOT be correct
    without_clues all captions minus every cited caption entry -> must NOT be correct
                  (this is the complement test OmniQA intended; theirs never removed anything)
`gate_media` (real media, role `gate_av`):
    av            the clue window with sound      -> must be correct
    video_only    the same window, muted          -> must NOT be correct
    audio_only    the same window, audio only     -> must NOT be correct
Conditions are configurable lists, so each one can be switched on or off for A/B runs.

Filtering vs tagging. `full` / `av` must be answered correctly: that is a correctness requirement and always
filters. For the single-modality conditions, `reject_if` decides what filters:
    any   reject if either single modality alone answers (strict)
    both  reject only if BOTH single modalities alone answer, i.e. the question needs no cross-modal
          evidence at all (lenient, in the spirit of OmniQA's gate)
    none  never reject for shortcuts
`without_clues` (when enabled) filters in modes any/both. `retry_on: [visual]` additionally marks a
visual-side shortcut as a *soft* failure: the generate stage asks for a better question with feedback and
falls back to this one only if none comes. Whatever the mode, every verdict records the
single-modality outcome and a dependency label, from which `modality_tags` derives the data-quality tags:
    av_strict          neither modality alone answers: the high-quality joint-reasoning subset
    audio_sufficient   the audio alone answers
    visual_sufficient  the visuals alone answer
"""
from ..schema import fmt_range
from ..stages.common import captions_text, mcq_text
from .clue import clip_window, cut_clip
from .model_checks import answer_letter
from . import Check, parallel, register

MUST_PASS = {"full", "av"}
AUDIO_SIDE = ("audio_only",)
VISUAL_SIDE = ("visual_only", "video_only")


HINTS = {
    "audio_only": "answerable from the audio alone: the speech or sound already gives the answer away",
    "visual_only": ("answerable from the visuals alone. Either only one thing in the video fits, or the audio cue as worded "
                    "has a visible counterpart: a word, letter, name or number in the cue is also shown on screen around "
                    "that moment, so a muted viewer can find the moment by looking. Describe the audio cue only by what is "
                    "heard, or choose an anchor whose words never appear on screen"),
    "video_only": "answerable from the muted video alone: only one thing in the video fits, so the audio cue is not needed",
    "without_clues": "answerable without the cited evidence: the question does not depend on it",
    "full": "not answerable even with all the evidence: the question or its answer is not supported",
    "av": "not answerable even from the video with sound: the question or its answer is not supported",
    "both_modalities": "answerable from the audio alone AND from the visuals alone: the question needs no cross-modal evidence",
}


def _evaluate(cand, results, max_correct, reject_if="any", retry_on=()):
    """results: {condition: [letters]}. Returns (failure reason or None, detail).

    detail[cond] = {correct, trials}; detail['shortcuts'] = single-modality conditions answered above
    max_correct; detail['dependency'] = 'joint' | 'audio_sufficient' | 'visual_sufficient' | 'either_sufficient'.
    """
    detail, shortcuts, wrong = {}, [], None
    for cond, picks in results.items():
        correct = sum(p == cand["answer"] for p in picks)
        detail[cond] = {"correct": correct, "trials": len(picks)}
        if cond in MUST_PASS:
            if correct * 2 <= len(picks) and wrong is None:
                wrong = f"{cond}_wrong"
        elif correct > max_correct:
            shortcuts.append(cond)
    audio_short = any(c in shortcuts for c in AUDIO_SIDE)
    visual_short = any(c in shortcuts for c in VISUAL_SIDE)
    detail["shortcuts"] = shortcuts
    detail["dependency"] = {(False, False): "joint", (True, False): "audio_sufficient",
                            (False, True): "visual_sufficient", (True, True): "either_sufficient"}[(audio_short, visual_short)]
    failed = wrong
    if failed is None and reject_if == "any" and shortcuts:
        failed = f"{shortcuts[0]}_shortcut"
    elif failed is None and reject_if == "both":
        if audio_short and visual_short:
            failed = "both_modalities_shortcut"
        elif "without_clues" in shortcuts:
            failed = "without_clues_shortcut"
    # Soft failure: the question is acceptable but a better one is wanted. The generate stage regenerates
    # with feedback and, if no better question comes, accepts this one (see generate_rov.process).
    if failed is None:
        side = [c for c, short in (("visual", visual_short), ("audio", audio_short)) if short and c in retry_on]
        if side:
            cond = next(c for c in shortcuts if c in (VISUAL_SIDE if side[0] == "visual" else AUDIO_SIDE))
            failed, detail["soft"] = f"{cond}_shortcut", True
    if failed:
        detail["hint"] = HINTS.get(failed.rsplit("_", 1)[0], failed)
    return failed, detail


def modality_tags(verdicts):
    """Data-quality tags from the most direct gate evidence available (real media over captions)."""
    by_check = {v["check"]: v for v in verdicts if v["check"] in ("gate_media", "gate_text")}
    v = by_check.get("gate_media") or by_check.get("gate_text")
    if v is None or "dependency" not in v.get("detail", {}):
        return {"tags": [], "modality_dependency": None}
    d = v["detail"]
    dep = d["dependency"]
    tags = {"joint": ["av_strict"], "audio_sufficient": ["audio_sufficient"], "visual_sufficient": ["visual_sufficient"],
            "either_sufficient": ["audio_sufficient", "visual_sufficient"]}[dep]
    single = {c: d[c] for c in (*AUDIO_SIDE, *VISUAL_SIDE) if c in d}
    return {"tags": tags, "modality_dependency": {"label": dep, "evidence": v["check"], **single}}


@register
class GateText(Check):
    name = "gate_text"
    prompts = ("gate_text",)
    roles = ("gate_text",)

    def run(self, ctx, cand, bundle):
        o = self.options(ctx)
        conds = o.get("conditions", ["full", "audio_only", "visual_only", "without_clues"])
        trials, temp = int(o.get("trials", 3)), float(o.get("temperature", 0.7))
        cited = [c["caption_ref"] for c in cand["clues"]]
        views = {
            "full": captions_text(bundle, with_refs=False),
            "audio_only": captions_text(bundle, visual=False, with_refs=False),
            "visual_only": captions_text(bundle, audio=False, with_refs=False),
            "without_clues": captions_text(bundle, exclude_refs=cited, with_refs=False),
        }
        client = ctx.models("gate_text")

        def draw(cond, i):
            text = ctx.prompts["gate_text"].render(captions=views[cond], mcq=mcq_text(cand))
            return lambda: answer_letter(client.call([{"type": "text", "text": text}], sample=i, temperature=temp,
                                                     tag=f"gate_text:{cond}", context={"candidate": cand["id"]}))

        picks = parallel(draw(cond, i) for cond in conds for i in range(trials))
        results = {cond: picks[k * trials:(k + 1) * trials] for k, cond in enumerate(conds)}
        samples = [{"condition": cond, "answers": results[cond]} for cond in conds]
        failed, detail = _evaluate(cand, results, int(o.get("max_correct", 1)), o.get("reject_if", "any"), tuple(o.get("retry_on", ())))
        return self.verdict(cand, failed is None, failed or "", samples=samples, detail=detail)


@register
class GateMedia(Check):
    name = "gate_media"
    prompts = ("gate_media",)
    roles = ("gate_av",)

    def run(self, ctx, cand, bundle):
        o = self.options(ctx)
        conds = o.get("conditions", ["av", "video_only", "audio_only"])
        trials, temp = int(o.get("trials", 3)), float(o.get("temperature", 0.7))
        duration = float(bundle["extra"].get("duration_s", 0))
        start = min(c["start_s"] for c in cand["clues"])
        end = max(c["end_s"] for c in cand["clues"])
        if o.get("window", "clues") == "full":
            s, e = 0.0, duration
        else:
            s, e = clip_window(start, end, duration, float(o.get("pad_s", 5)), float(o.get("min_len_s", 10)),
                               float(o.get("max_len_s", 600)))
        src = bundle["extra"]["video_path"]
        modes = {"av": ("av", "video"), "video_only": ("video", "video"), "audio_only": ("audio", "audio")}
        client = ctx.models("gate_av")
        descs = {"av": "a video with its audio", "video_only": "a video without audio",
                 "audio_only": "only the audio track of a video"}
        inputs = {}
        for cond in conds:                                  # cut clips first (ffmpeg, cached on disk)
            mode, kind = modes[cond]
            inputs[cond] = (kind, cut_clip(ctx, src, s, e, mode),
                            ctx.prompts["gate_media"].render(mcq=mcq_text(cand), clip_range=fmt_range(s, e), input_desc=descs[cond]))

        def draw(cond, i):
            kind, clip, text = inputs[cond]
            return lambda: answer_letter(client.call([{"type": "text", "text": text}, {"type": kind, "path": str(clip)}],
                                                     sample=i, temperature=temp, tag=f"gate_media:{cond}",
                                                     context={"candidate": cand["id"]}))

        picks = parallel(draw(cond, i) for cond in conds for i in range(trials))
        results = {cond: picks[k * trials:(k + 1) * trials] for k, cond in enumerate(conds)}
        samples = [{"condition": cond, "answers": results[cond]} for cond in conds]
        failed, detail = _evaluate(cand, results, int(o.get("max_correct", 1)), o.get("reject_if", "any"), tuple(o.get("retry_on", ())))
        detail["window"] = [round(s, 2), round(e, 2)]
        return self.verdict(cand, failed is None, failed or "", samples=samples, detail=detail)

