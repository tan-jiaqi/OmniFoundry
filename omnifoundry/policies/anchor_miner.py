"""AnchorMiner: enumerate cross-modal evidence anchors in a caption bundle, deterministically.

An anchor is one audio caption entry that can point at a moment, together with the visual entries that
co-occur with it. Generation then writes a question *for a given anchor* instead of inventing one from the
whole video, which is what lets one video yield several questions (JointAVBench generates per
segment x task: reference/JointAVBench/generation_pipeline/generate_qa.py:93-151).

Scoring is deterministic, from the captions alone — no model calls — so the whole fan-out can be inspected
and tuned offline against an existing run's caption output.

A good anchor:
  - is speech / sound / music, not background or silence (those are not reliable anchors)
  - has at least one visual entry overlapping it, or starting within `lookahead_s`
  - is UNIQUELY identifiable by its own words: a question saying "when X says Y" must select one moment,
    so the entry's distinctive words must not recur in other audio entries
  - does not leak: its distinctive words should not already appear in the visual captions at that time,
    or a muted viewer can find the moment by looking (weak evidence on 31 reviewed items: 33% joint when
    leaky vs 50% when clean, n=6 vs 14 — kept as a cheap ranking signal, not a hard filter)
Selection then spreads anchors across events so one video's questions do not all come from one moment.
"""
import re

WORD = re.compile(r"[a-z0-9']+")
STOP = frozenset("""a an the and or but of to in on at is are was were be been it its this that these those with for
as by from he she they you i we his her their them him us our your not no nor so if then than there here what which
who whom when where why how all any both each few more most other some such only own same too very can will just
don should now says said say sings sing speaks speak voice heard hear audio sound sounds video visual screen shows
show shown seen see appears appear while during after before man woman person people character""".split())
AUDIO_KINDS = ("speech", "sound", "music")

DEFAULTS = {
    "lookahead_s": 3.0,        # a visual entry starting this soon after the audio still counts as co-occurring
    "pad_s": 1.0,              # tolerance on the other side
    "min_words": 3,            # an anchor needs this many distinctive words to be quotable
    "max_per_event": 2,        # diversity: at most this many anchors from one event
    "leak_penalty": 0.5,       # weight of the on-screen-leak penalty in the score
    "limit": 24,
}


def words(text):
    return {w for w in WORD.findall((text or "").lower()) if w not in STOP and len(w) > 2}


def _overlaps(a_start, a_end, v_start, v_end, pad, lookahead):
    return v_start <= a_end + lookahead and v_end >= a_start - pad


def mine(bundle, options=None):
    """Return anchors for one caption bundle, best first. Pure function of the bundle."""
    o = {**DEFAULTS, **(options or {})}
    audio = [a for a in bundle["audio_captions"] if a.get("audio_type") in AUDIO_KINDS]
    visual = bundle["visual_captions"]
    if not audio or not visual:
        return []

    # How often each word occurs across audio entries: a word in exactly one entry can identify it.
    counts = {}
    per_entry = {}
    for a in audio:
        w = words(a["caption"])
        per_entry[a["ref"]] = w
        for token in w:
            counts[token] = counts.get(token, 0) + 1

    anchors = []
    for a in audio:
        own = per_entry[a["ref"]]
        if len(own) < o["min_words"]:
            continue
        near = [v for v in visual if _overlaps(a["start_s"], a["end_s"], v["start_s"], v["end_s"], o["pad_s"], o["lookahead_s"])]
        if not near:
            continue
        unique_words = {w for w in own if counts[w] == 1}
        uniqueness = len(unique_words) / len(own)
        near_visual_words = set().union(*(words(v["caption"]) for v in near))
        leak = len(own & near_visual_words) / len(own)
        # Prefer anchors that are quotable and unique, and that a muted viewer cannot locate.
        score = uniqueness - o["leak_penalty"] * leak + 0.1 * min(len(near), 3)
        anchors.append({
            "audio_ref": a["ref"], "event_id": a["event_id"], "audio_type": a["audio_type"],
            "timestamp": a["timestamp"], "span": [a["start_s"], a["end_s"]],
            "visual_refs": [v["ref"] for v in near],
            "uniqueness": round(uniqueness, 3), "leak": round(leak, 3),
            "unique_words": sorted(unique_words)[:8], "score": round(score, 3),
        })

    anchors.sort(key=lambda x: -x["score"])
    # Spread across events, keeping the best few per event.
    kept, per_event = [], {}
    for a in anchors:
        n = per_event.get(a["event_id"], 0)
        if n >= o["max_per_event"]:
            continue
        per_event[a["event_id"]] = n + 1
        kept.append(a)
        if len(kept) >= o["limit"]:
            break
    return kept


def anchor_text(bundle, anchor):
    """The anchor rendered for the generation prompt."""
    index = {c["ref"]: c for c in bundle["audio_captions"] + bundle["visual_captions"]}
    a = index[anchor["audio_ref"]]
    lines = [f"AUDIO ANCHOR [{a['ref']}] {a['timestamp']} ({a['audio_type']}): {a['caption']}",
             "CO-OCCURRING VISUALS:"]
    lines += [f"  [{r}] {index[r]['timestamp']}: {index[r]['caption']}" for r in anchor["visual_refs"]]
    return "\n".join(lines)


MINERS = {"default": mine}
