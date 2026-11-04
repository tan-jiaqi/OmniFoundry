"""Deterministic grounding check: every clue must point at a real caption entry and match it.

- caption_ref exists, and its modality agrees with the clue (audio clue -> audio caption).
- quote appears in that caption: verbatim after normalization, or >= `min_overlap` of its content words
  with at least `min_words` distinct words (OmniQA's rule).
- clue time overlaps the referenced caption's span (± `time_slack_s`).
- audio clues may not cite `forbid_audio_types` (default background/silence/unclear); music only for
  task types in `music_tasks`.
- simultaneity: if the question says "when / while / as ...", the audio and visual clues it relies on must
  be within `simultaneity_gap_s`.
"""
import re

from ..schema import AUDIO_MODALITIES
from ..stages.common import caption_index
from . import Check, register

_WORD = re.compile(r"[a-z0-9]+|[一-鿿]")
STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "is", "are", "was", "were", "it", "this", "that",
        "with", "for", "as", "by", "be", "his", "her", "their", "its", "from"}
SIMULTANEITY = re.compile(r"\b(when|while|as soon as|at the moment|at the same time|during|just as)\b|此时|同时|当.*时", re.I)


def words(text):
    return [w for w in _WORD.findall(text.lower()) if w not in STOP]


def quote_matches(quote, caption, min_overlap, min_words):
    q, c = " ".join(_WORD.findall(quote.lower())), " ".join(_WORD.findall(caption.lower()))
    if q and q in c:
        return True
    qw = set(words(quote))
    if len(qw) < min_words:
        return False
    return len(qw & set(words(caption))) / len(qw) >= min_overlap


@register
class Grounding(Check):
    name = "grounding"

    def run(self, ctx, cand, bundle):
        o = self.options(ctx)
        index = caption_index(bundle)
        slack = float(o.get("time_slack_s", 2))
        forbid = set(o.get("forbid_audio_types", ["background", "silence", "unclear"]))
        music_tasks = set(o.get("music_tasks", ["music"]))
        for c in cand["clues"]:
            ref = index.get(c.get("caption_ref"))
            if ref is None:
                return self.verdict(cand, False, "quote_not_in_captions",
                                    detail={"hint": f"clue {c['id']} cites unknown caption_ref {c.get('caption_ref')!r}", "clue": c["id"]})
            is_audio_ref = "audio_type" in ref
            if (c["modality"] in AUDIO_MODALITIES) != is_audio_ref:
                return self.verdict(cand, False, "quote_not_in_captions",
                                    detail={"hint": f"clue {c['id']} is {c['modality']} but cites a {'audio' if is_audio_ref else 'visual'} caption", "clue": c["id"]})
            if is_audio_ref:
                if ref["audio_type"] in forbid:
                    return self.verdict(cand, False, "music_anchor",
                                        detail={"hint": f"clue {c['id']} relies on {ref['audio_type']} audio, which is not a reliable anchor", "clue": c["id"]})
                if ref["audio_type"] == "music" and cand["task_type"] not in music_tasks:
                    return self.verdict(cand, False, "music_anchor",
                                        detail={"hint": f"clue {c['id']} uses music as evidence; only {sorted(music_tasks)} tasks may", "clue": c["id"]})
            if not quote_matches(c.get("quote", ""), ref["caption"], float(o.get("min_overlap", 0.6)), int(o.get("min_words", 4))):
                return self.verdict(cand, False, "quote_not_in_captions",
                                    detail={"hint": f"clue {c['id']} quote is not found in caption {ref['ref']}", "clue": c["id"]})
            if not (c["start_s"] <= ref["end_s"] + slack and c["end_s"] >= ref["start_s"] - slack):
                return self.verdict(cand, False, "timestamp_not_in_captions",
                                    detail={"hint": f"clue {c['id']} time {c['timestamp']} is outside caption {ref['ref']} ({ref['timestamp']})", "clue": c["id"]})
        if SIMULTANEITY.search(cand["question"]):
            gap = float(o.get("simultaneity_gap_s", 15))
            a = [c for c in cand["clues"] if c["modality"] in AUDIO_MODALITIES]
            v = [c for c in cand["clues"] if c["modality"] == "visual"]
            closest = min((max(0.0, max(x["start_s"], y["start_s"]) - min(x["end_s"], y["end_s"])) for x in a for y in v), default=0.0)
            if closest > gap:
                return self.verdict(cand, False, "simultaneity_gap",
                                    detail={"hint": f"question implies simultaneity but audio and visual clues are {closest:.0f}s apart"})
        return self.verdict(cand, True)
