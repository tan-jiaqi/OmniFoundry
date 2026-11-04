"""S1 caption: Video -> CaptionBundle.

1. Transcode with burned-in MM:SS timestamps (low fps / width): the model reads global time off the frame.
2. Plan events: `gemini` (planner model over the whole video), `scenedetect` (PySceneDetect, if installed)
   or `fixed` (equal windows; no model call). Events are capped at `max_event_s` and `max_events`.
3. Caption each event in order with the captioner model, passing earlier events' captions for
   reference resolution. `event_input: clip` sends the event's span cut from the overlay video (global
   timestamps stay visible); `full` sends the whole overlay video every time (OmniQA behaviour).
4. Normalize: parse timestamps, clamp to the event, assign stable refs (`e02.a3`, `e02.v1`).
"""
import sys

from ..jsonio import digest
from ..schema import AUDIO_TYPES, CaptionBundle, fmt_range, parse_range
from .. import lineage
from .common import lineage_extra
from .select import video_file_key

NAME, VERSION = "caption", 1
CONFIG_KEYS = ("caption", "media.overlay")


def prompts(ctx):
    names = ["caption_event"]
    if ctx.get("caption.planner", "gemini") == "gemini":
        names.append("caption_plan")
    return names


def roles(ctx):
    return ["captioner"] + (["planner"] if ctx.get("caption.planner", "gemini") == "gemini" else [])


def inputs(ctx, prev_dir):
    from ..jsonio import read_jsonl
    return read_jsonl(prev_dir / "out.jsonl")


def overlay_path(ctx, video):
    o = ctx.get("media.overlay", {}) or {}
    key = digest("overlay", video_file_key(video["path"]), o)[:24]
    path = ctx.work_dir / "overlay" / f"{key}.mp4"
    if not path.exists():
        ctx.media.overlay_transcode(video["path"], path, fps=o.get("fps", 2), width=o.get("width", 640))
    return path


def _plan_fixed(duration, opts):
    win = float(opts.get("fixed_window_s", 60))
    events, t, i = [], 0.0, 1
    while t < duration - 0.5:
        events.append({"event_id": f"e{i:02d}", "start_s": t, "end_s": min(duration, t + win), "summary": ""})
        t += win
        i += 1
    return events, "", ""


def _plan_scenedetect(path, duration, opts):
    try:
        from scenedetect import ContentDetector, detect
    except ImportError as exc:
        raise RuntimeError("caption.planner=scenedetect needs `pip install scenedetect[opencv]`") from exc
    scenes = detect(str(path), ContentDetector(threshold=opts.get("scene_threshold", 27),
                                                min_scene_len=int(opts.get("min_scene_s", 2) * 25)))
    spans = [(s.get_seconds(), e.get_seconds()) for s, e in scenes] or [(0.0, duration)]
    events = [{"event_id": f"e{i + 1:02d}", "start_s": s, "end_s": e, "summary": ""} for i, (s, e) in enumerate(spans)]
    return _merge_short(events, opts), "", ""


def _merge_short(events, opts):
    """Greedy: merge the shortest event into its shorter neighbour until within max_events."""
    max_events = int(opts.get("max_events", 10))
    events = [dict(e) for e in events]
    while len(events) > max_events:
        i = min(range(len(events)), key=lambda k: events[k]["end_s"] - events[k]["start_s"])
        j = i - 1 if i == len(events) - 1 or (i > 0 and (events[i - 1]["end_s"] - events[i - 1]["start_s"]) <=
                                             (events[i + 1]["end_s"] - events[i + 1]["start_s"])) else i + 1
        a, b = sorted((i, j))
        events[a] = {**events[a], "end_s": events[b]["end_s"], "summary": (events[a]["summary"] + " " + events[b]["summary"]).strip()}
        del events[b]
    for k, e in enumerate(events):
        e["event_id"] = f"e{k + 1:02d}"
    return events


def _plan_model(ctx, overlay, duration, opts):
    p = ctx.prompts["caption_plan"].render(max_events=opts.get("max_events", 10), max_event_s=int(opts.get("max_event_s", 180)),
                                           duration=fmt_range(0, duration).split("-")[1])
    res = ctx.models("planner").call([{"type": "text", "text": p}, {"type": "video", "path": str(overlay)}],
                                     tag="caption_plan")
    data = res["json"] or {}
    events = []
    for seg in data.get("event_segments", []):
        try:
            s, e = parse_range(f"{seg['start']}-{seg['end']}")
        except (KeyError, ValueError):
            continue
        s, e = max(0.0, s), min(duration, e)
        if e - s >= 1:
            events.append({"event_id": "", "start_s": s, "end_s": e, "summary": seg.get("summary", "")})
    events.sort(key=lambda x: x["start_s"])
    if not events:
        events, _, _ = _plan_fixed(duration, opts)
    return _merge_short(events, opts), data.get("video_summary", ""), data.get("video_type", "")


def _split_long(events, opts, duration):
    max_s = float(opts.get("max_event_s", 180))
    out = []
    for e in events:
        s = e["start_s"]
        while e["end_s"] - s > max_s + 1e-6:
            out.append({**e, "start_s": s, "end_s": s + max_s})
            s += max_s
        out.append({**e, "start_s": s})
    out = [e for e in out if e["start_s"] < duration]
    for k, e in enumerate(out):
        e["event_id"] = f"e{k + 1:02d}"
    return out


def _prior_context(done, limit):
    text = "\n".join(f"[{e['event_id']} {fmt_range(e['start_s'], e['end_s'])}] {e.get('global_caption', '')}" for e in done)
    return text[-limit:] if text else "(none: this is the first event)"


def _normalize_items(raw_items, event, kind, duration):
    out = []
    for item in raw_items or []:
        try:
            s, e = parse_range(item["timestamp"])
        except (KeyError, ValueError, TypeError):
            continue
        # Clamp to the event (small slack for boundary-crossing speech) and the video.
        s = max(event["start_s"] - 2, min(s, event["end_s"]))
        e = min(event["end_s"] + 2, max(e, s), duration)
        caption = str(item.get("caption", "")).strip()
        if not caption:
            continue
        rec = {"event_id": event["event_id"], "start_s": round(s, 2), "end_s": round(e, 2),
               "timestamp": fmt_range(s, e), "caption": caption}
        if kind == "a":
            t = str(item.get("audio_type", "")).lower().strip()
            rec["audio_type"] = t if t in AUDIO_TYPES else "unclear"
        out.append(rec)
    out.sort(key=lambda r: (r["start_s"], r["end_s"]))
    for i, r in enumerate(out):
        r["ref"] = f"{event['event_id']}.{kind}{i + 1}"
    return out


def process(ctx, video):
    opts = ctx.get("caption", {}) or {}
    duration = float(video["duration_s"])
    overlay = overlay_path(ctx, video)
    planner = opts.get("planner", "gemini")
    if planner == "gemini":
        events, summary, vtype = _plan_model(ctx, overlay, duration, opts)
    elif planner == "scenedetect":
        events, summary, vtype = _plan_scenedetect(video["path"], duration, opts)
    else:
        events, summary, vtype = _plan_fixed(duration, opts)
    events = _split_long(events, opts, duration)

    audio, visual, done = [], [], []
    template = ctx.prompts["caption_event"]
    pad = float(opts.get("clip_pad_s", 1.0))
    for ev in events:
        text = template.render(event_range=fmt_range(ev["start_s"], ev["end_s"]),
                               prior_context=_prior_context(done, int(opts.get("prior_context_chars", 12000))),
                               plan_summary=ev.get("summary", "") or "(none)")
        if opts.get("event_input", "clip") == "clip":
            clip = ctx.work_dir / "event_clips" / f"{digest(str(overlay), ev['start_s'], ev['end_s'], pad)[:24]}.mp4"
            if not clip.exists():
                ctx.media.cut(overlay, clip, ev["start_s"] - pad, ev["end_s"] + pad, mode="av",
                              width=(ctx.get("media.overlay", {}) or {}).get("width", 640))
            media_path = clip
        else:
            media_path = overlay
        res = ctx.models("captioner").call([{"type": "text", "text": text}, {"type": "video", "path": str(media_path)}],
                                           tag="caption_event", context={"video": video["id"], "event": ev["event_id"]})
        data = res["json"] or {}
        ev = {**ev, "global_caption": str(data.get("global_caption", "")).strip()}
        audio += _normalize_items(data.get("audio_captions"), ev, "a", duration)
        visual += _normalize_items(data.get("visual_captions"), ev, "v", duration)
        done.append(ev)

    bundle = CaptionBundle(
        id=f"cap-{video['id']}", video_id=video["id"], video_summary=summary, video_type=vtype,
        events=done, audio_captions=audio, visual_captions=visual,
        lineage=lineage.make(NAME, VERSION, [video["id"]], **lineage_extra(ctx, sys.modules[__name__])),
        extra={"video_path": video["path"], "overlay_path": str(overlay), "duration_s": duration},
    )
    if not audio or not visual:
        return {"rejected": [dict(bundle.to_dict(), reason="empty_modality_captions")]}
    return {"out": [bundle.to_dict()]}
