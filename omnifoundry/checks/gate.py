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
"""
from ..schema import fmt_range
from ..stages.common import captions_text, mcq_text
from .clue import clip_window, cut_clip
from .model_checks import answer_letter
from . import Check, register

MUST_PASS = {"full", "av"}


def _evaluate(cand, results, max_correct):
    """results: {condition: [letters]}. Returns verdict pieces."""
    detail, failed = {}, None
    for cond, picks in results.items():
        correct = sum(p == cand["answer"] for p in picks)
        detail[cond] = {"correct": correct, "trials": len(picks)}
        if cond in MUST_PASS:
            if correct * 2 <= len(picks) and failed is None:
                failed = f"{cond}_wrong"
        elif correct > max_correct and failed is None:
            failed = f"{cond}_shortcut"
    return failed, detail


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
        results, samples = {}, []
        for cond in conds:
            text = ctx.prompts["gate_text"].render(captions=views[cond], mcq=mcq_text(cand))
            picks = []
            for i in range(trials):
                res = ctx.models("gate_text").call([{"type": "text", "text": text}], sample=i, temperature=temp,
                                                   tag=f"gate_text:{cond}", context={"candidate": cand["id"]})
                picks.append(answer_letter(res))
            results[cond] = picks
            samples.append({"condition": cond, "answers": picks})
        failed, detail = _evaluate(cand, results, int(o.get("max_correct", 1)))
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
        results, samples = {}, []
        for cond in conds:
            mode, kind = modes[cond]
            clip = cut_clip(ctx, src, s, e, mode)
            text = ctx.prompts["gate_media"].render(mcq=mcq_text(cand), clip_range=fmt_range(s, e),
                                                    input_desc={"av": "a video with its audio", "video_only": "a video without audio",
                                                                "audio_only": "only the audio track of a video"}[cond])
            picks = []
            for i in range(trials):
                res = ctx.models("gate_av").call([{"type": "text", "text": text}, {"type": kind, "path": str(clip)}],
                                                 sample=i, temperature=temp, tag=f"gate_media:{cond}",
                                                 context={"candidate": cand["id"]})
                picks.append(answer_letter(res))
            results[cond] = picks
            samples.append({"condition": cond, "answers": picks})
        failed, detail = _evaluate(cand, results, int(o.get("max_correct", 1)))
        detail["window"] = [round(s, 2), round(e, 2)]
        return self.verdict(cand, failed is None, failed or "", samples=samples, detail=detail)

