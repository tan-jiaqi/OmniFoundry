"""Offline stand-in for every model role. Dispatches on the role and reads what it needs from the prompt,
so the whole pipeline runs without network. First generation attempts deliberately cite a bad quote,
which exercises the check -> feedback -> regenerate path.
"""
import json
import re

CORRECT = "the correct joint answer"


def _text(parts):
    return "\n".join(p["text"] for p in parts if p["type"] == "text")


def _gold_letter(text):
    m = re.search(rf"^([A-H])\. {CORRECT}", text, re.M)
    return m.group(1) if m else "A"


def _wrong_letter(text):
    g = _gold_letter(text)
    return "B" if g == "A" else "A"


def respond(role, parts, options):
    text = _text(parts)
    if role == "captioner":
        m = re.search(r"Event range on the video clock: (\d+):(\d+)-(\d+):(\d+)", text)
        s = int(m[1]) * 60 + int(m[2])
        e = int(m[3]) * 60 + int(m[4])
        ts = f"{s // 60:02d}:{s % 60:02d}-{e // 60:02d}:{e % 60:02d}"
        return json.dumps({
            "global_timestamp": ts, "global_caption": f"Event {ts}: a colored screen with a steady tone.",
            "audio_captions": [{"timestamp": ts, "audio_type": "speech", "caption": f"The narrator says the secret code word for segment {s} clearly"}],
            "visual_captions": [{"timestamp": ts, "caption": f"A plain colored screen showing panel number {s} in the center"}]})
    if role == "selector":
        return json.dumps({"task_ids": ["attribute"], "reason": "mock"})
    if role == "generator":
        first = "first attempt" in text
        a = re.search(r"AUDIO \[(\S+)\] (\S+) \(speech\): (.+)", text)
        v = re.search(r"VISUAL \[(\S+)\] (\S+): (.+)", text)
        quote = "words that never appear anywhere at all in this video" if first else a[3]
        options_ = [f"{CORRECT}"] + [f"distractor number {i}" for i in range(1, 8)]
        return json.dumps({
            "question": "Which panel is on screen after the narrator says the code word?",
            "options": [f"{chr(65 + i)}. {o}" for i, o in enumerate(options_)], "answer": "A",
            "clues": [{"id": "c1", "modality": "speech", "timestamp": a[2], "quote": quote, "fact": "narrator says the code word", "caption_ref": a[1]},
                      {"id": "c2", "modality": "visual", "timestamp": v[2], "quote": v[3], "fact": "panel number visible", "caption_ref": v[1]}],
            "reasoning_steps": [{"step_id": "s1", "modality": "audio", "timestamp": a[2], "content": "hear code word", "depends_on": [], "clue_ids": ["c1"]},
                                {"step_id": "s2", "modality": "visual", "timestamp": v[2], "content": "read panel", "depends_on": ["s1"], "clue_ids": ["c2"]}]})
    if role == "solver":
        return json.dumps({"all_presuppositions_supported": True, "unsupported": [], "answer": _gold_letter(text), "reason": "mock"})
    if role == "blind":
        return json.dumps({"answer": _wrong_letter(text), "reason": "guess"})
    if role in ("clue_visual", "clue_audio"):
        return json.dumps({"pass": True, "support_strength": 0.9, "unsupported_claims": [], "observed": "mock"})
    if role == "gate_text":
        # Answerable only when both cited entries (the first audio and visual captions) are present.
        both = "code word for segment 0 " in text and "panel number 0 " in text
        if options.get("visual_leak") and "panel number 0 " in text and "AUDIO" not in text:
            both = True   # simulate a question a muted viewer can answer
        return json.dumps({"answer": _gold_letter(text) if both else _wrong_letter(text), "reason": "mock"})
    if role == "gate_av":
        full = "a video with its audio" in text
        return json.dumps({"answer": _gold_letter(text) if full else _wrong_letter(text), "reason": "mock"})
    if role == "thinker":
        ans = re.search(r"The correct option is ([A-H])", text)[1]
        t = re.search(r"\[audio\] (\d+:\d+)", text)[1]
        return f"Got it! I need to connect the narration with the panel.\nAt {t}, I hear the code word and then read the panel.\nThe others do not match.\nThis points to option {ans}."
    raise KeyError(f"mock has no response for role {role}")
