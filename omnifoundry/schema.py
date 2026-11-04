"""Records passed between stages. Plain dicts on disk; these dataclasses document and build them.

Every record carries `id`, `kind`, `schema_version` and `lineage`. Free-form fields go in `extra`, and
`from_dict` moves unknown top-level keys there, so experiments can attach fields without migrations.
"""
from dataclasses import asdict, dataclass, field, fields
import re

SCHEMA_VERSION = 1

MODALITIES = ("speech", "sound", "music", "visual")
AUDIO_MODALITIES = ("speech", "sound", "music")
AUDIO_TYPES = ("speech", "music", "sound", "background", "silence", "unclear")


class Record:
    kind = "record"

    def to_dict(self):
        return {"kind": self.kind, "schema_version": SCHEMA_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        known = {k: v for k, v in d.items() if k in names}
        extra = {k: v for k, v in d.items() if k not in names and k not in ("kind", "schema_version")}
        if "extra" in names:
            known["extra"] = {**known.get("extra", {}), **extra}
        return cls(**known)


@dataclass
class Video(Record):
    kind = "video"
    id: str
    path: str
    duration_s: float
    has_audio: bool
    source: str = ""
    meta: dict = field(default_factory=dict)
    lineage: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


@dataclass
class CaptionBundle(Record):
    """Event-segmented captions. Each item in audio/visual lists has a stable `ref` like `e02.a3`."""
    kind = "caption_bundle"
    id: str
    video_id: str
    video_summary: str
    video_type: str
    events: list            # [{event_id, start_s, end_s, summary, global_caption}]
    audio_captions: list    # [{ref, event_id, start_s, end_s, timestamp, audio_type, caption}]
    visual_captions: list   # [{ref, event_id, start_s, end_s, timestamp, caption}]
    lineage: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


@dataclass
class Candidate(Record):
    """One generated question with its evidence chain.

    clues: [{id, modality, timestamp, start_s, end_s, quote, fact, caption_ref}]
    steps: [{step_id, modality, timestamp, content, depends_on, clue_ids}]
    """
    kind = "candidate"
    id: str
    video_id: str
    route: str              # "rov" | "rby"
    task_type: str
    question: str
    options: list           # ["A. ...", ...] for MCQ; [] for open-ended
    answer: str             # option letter for MCQ, text for open-ended
    clues: list
    steps: list
    attempt: int = 0
    feedback: list = field(default_factory=list)   # reasons from earlier attempts
    lineage: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


@dataclass
class Verdict(Record):
    """Result of one Check on one Candidate. `samples` keeps every model draw for difficulty signals."""
    kind = "verdict"
    check: str
    candidate_id: str
    passed: bool
    reason: str = ""        # short machine code, e.g. "quote_not_in_captions"
    detail: dict = field(default_factory=dict)
    samples: list = field(default_factory=list)
    cost: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


# ---- time helpers -------------------------------------------------------------------------------

_TS = re.compile(r"^\s*(\d+):(\d{1,2})(?::(\d{1,2}))?(?:\.(\d+))?\s*$")


def parse_time(text):
    """'MM:SS', 'H:MM:SS' or seconds -> float seconds."""
    if isinstance(text, (int, float)):
        return float(text)
    m = _TS.match(str(text))
    if not m:
        raise ValueError(f"bad timestamp: {text!r}")
    a, b, c, frac = m.groups()
    secs = int(a) * 60 + int(b) if c is None else int(a) * 3600 + int(b) * 60 + int(c)
    return secs + (float("0." + frac) if frac else 0.0)


def parse_range(text):
    """'MM:SS-MM:SS' -> (start_s, end_s)."""
    parts = re.split(r"\s*[-–—~]\s*(?=\d)", str(text).strip(), maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"bad range: {text!r}")
    start, end = parse_time(parts[0]), parse_time(parts[1])
    if end < start:
        raise ValueError(f"range ends before it starts: {text!r}")
    return start, end


def fmt_time(seconds):
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def fmt_range(start, end):
    return f"{fmt_time(start)}-{fmt_time(end)}"


def letters(n):
    return [chr(ord("A") + i) for i in range(n)]
