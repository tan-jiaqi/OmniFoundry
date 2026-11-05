"""Gate 1, clue validity: each clue is checked against the real media around its timestamp.

Per clue modality, `checks.clue.routes` gives the media mode and model role:
    visual -> muted video clip -> role `clue_visual`
    speech / sound / music -> audio clip (or av clip) -> role `clue_audio`
Modalities not listed in `checks.clue.modalities` are skipped (recorded as skipped), which is how the
OmniQA behaviour (visual only) and the audio-verified variant are A/B-switched from config.

The claim sent is the clue's `quote`, its single-modality observation taken from the caption, not its
`fact`. The fact states the cross-modal link (for example "this clank is the chute flap"), which one
modality alone can never confirm. Gate 2 tests that link.
"""
from ..jsonio import digest
from ..schema import fmt_range
from . import Check, register

DEFAULT_ROUTES = {
    "visual": {"mode": "video", "role": "clue_visual"},
    "speech": {"mode": "audio", "role": "clue_audio"},
    "sound": {"mode": "audio", "role": "clue_audio"},
    "music": {"mode": "audio", "role": "clue_audio"},
}


def clip_window(start, end, duration, pad, min_len, max_len):
    s, e = start - pad, end + pad
    if e - s < min_len:
        mid = (s + e) / 2
        s, e = mid - min_len / 2, mid + min_len / 2
    if e - s > max_len:
        mid = (start + end) / 2
        s, e = mid - max_len / 2, mid + max_len / 2
    s = max(0.0, s)
    e = min(duration, e) if duration else e
    return s, max(e, s + 0.5)


def cut_clip(ctx, src, start, end, mode):
    suffix = ".mp3" if mode == "audio" else ".mp4"
    path = ctx.work_dir / "clips" / f"{digest(str(src), round(start, 2), round(end, 2), mode)[:24]}{suffix}"
    if not path.exists():
        ctx.media.cut(src, path, start, end, mode=mode)
    return path


@register
class Clue(Check):
    name = "clue"
    prompts = ("clue_verify",)

    def roles_for(self, ctx):
        o = self.options(ctx)
        routes = {**DEFAULT_ROUTES, **(o.get("routes") or {})}
        enabled = o.get("modalities", ["visual", "speech", "sound", "music"])
        return tuple(sorted({routes[m]["role"] for m in enabled}))

    def run(self, ctx, cand, bundle):
        o = self.options(ctx)
        enabled = set(o.get("modalities", ["visual", "speech", "sound", "music"]))
        routes = {**DEFAULT_ROUTES, **(o.get("routes") or {})}
        threshold = float(o.get("min_strength", 0.75))
        src = bundle["extra"]["video_path"]
        duration = float(bundle["extra"].get("duration_s", 0))
        samples, skipped, failed = [], [], None
        for c in cand["clues"]:
            if c["modality"] not in enabled:
                skipped.append(c["id"])
                continue
            route = routes[c["modality"]]
            s, e = clip_window(c["start_s"], c["end_s"], duration, float(o.get("pad_s", 2)), float(o.get("min_len_s", 6)),
                               float(o.get("max_len_s", 45)))
            clip = cut_clip(ctx, src, s, e, route["mode"])
            text = ctx.prompts["clue_verify"].render(
                modality=c["modality"], media_kind={"video": "a muted video clip", "audio": "an audio clip", "av": "a video clip with audio"}[route["mode"]],
                clip_range=fmt_range(s, e), clue_range=c["timestamp"], offset=f"{c['start_s'] - s:.0f}",
                claim=c.get("quote") or c.get("fact", ""))
            kind = "audio" if route["mode"] == "audio" else "video"
            res = ctx.models(route["role"]).call([{"type": "text", "text": text}, {"type": kind, "path": str(clip)}],
                                                 tag="clue_verify", context={"candidate": cand["id"], "clue": c["id"]})
            d = res.get("json") or {}
            strength = d.get("support_strength")
            ok = (d.get("pass") is True and isinstance(strength, (int, float)) and strength >= threshold
                  and not d.get("unsupported_claims"))
            samples.append({"clue": c["id"], "modality": c["modality"], "pass": ok, "strength": strength,
                            "unsupported": d.get("unsupported_claims", []), "observed": str(d.get("observed", ""))[:400], "window": [round(s, 2), round(e, 2)],
                            "model": res.get("model")})
            if not ok and failed is None:
                failed = c["id"]
                if o.get("short_circuit", True):
                    break
        detail = {"skipped": skipped}
        if failed:
            bad = next(x for x in samples if x["clue"] == failed)
            clue = next(c for c in cand["clues"] if c["id"] == failed)
            hint = (f"clue {failed} (caption {clue['caption_ref']}, {clue['timestamp']}) does not match the video: "
                    f"{bad['observed'][:250]}. That caption entry is unreliable; build the question on other evidence")
            return self.verdict(cand, False, "clue_unsupported", samples=samples, detail={**detail, "clue": failed, "hint": hint})
        return self.verdict(cand, True, samples=samples, detail=detail)
