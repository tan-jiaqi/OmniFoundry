"""S4 think: verified candidate + captions + evidence chain -> first-person thinking that ends at the gold.

The thinker never sees the video; it expands the verified chain. `think_qc` is deterministic:
required opening, no pipeline vocabulary, only timestamps from the evidence, and a closing line that
names the gold option. Up to `think.max_attempts` draws; the cleanest is kept, failing rows are rejected.
"""
import re
import sys

from ..jsonio import read_jsonl
from ..schema import fmt_time
from .. import lineage
from .common import captions_text, lineage_extra, mcq_text

NAME, VERSION = "think", 1
CONFIG_KEYS = ("think",)

DEFAULT_FORBIDDEN = ["caption", "clue", "modality", "evidence chain", "transcript", "annotation", "gold answer",
                     "reasoning step", "the provided"]
_AT = re.compile(r"\bAt (\d{1,2}:\d{2})")


def prompts(ctx):
    return ["think", "think_corroborate"]


def roles(ctx):
    return ["thinker"]


def inputs(ctx, prev_dir):
    bundles = {b["video_id"]: b for b in read_jsonl(ctx.run_dir / "caption" / "out.jsonl")}
    return [{"id": c["id"], "candidate": c, "bundle": bundles[c["video_id"]]}
            for c in read_jsonl(prev_dir / "out.jsonl") if c["video_id"] in bundles]


def evidence_text(cand):
    clues = {c["id"]: c for c in cand["clues"]}
    lines = []
    for s in cand["steps"]:
        cites = [clues[i] for i in s.get("clue_ids", []) if i in clues]
        when = ", ".join(c["timestamp"] for c in cites) or s.get("timestamp", "")
        lines.append(f"{s['step_id']} [{s['modality']}] {when}: {s['content']} (uses {', '.join(s.get('depends_on', [])) or 'nothing earlier'})")
    return "\n".join(lines)


def allowed_times(cand):
    out = set()
    for c in cand["clues"]:
        out.update({fmt_time(c["start_s"]), fmt_time(c["end_s"])})
    return out


def qc(text, cand, opts):
    issues = []
    opening = opts.get("opening", "Got it!")
    if not text.strip().startswith(opening):
        issues.append("missing_opening")
    low = text.lower()
    for w in opts.get("forbidden", DEFAULT_FORBIDDEN):
        if w.lower() in low:
            issues.append(f"forbidden:{w}")
    ok_times = allowed_times(cand)
    for t in _AT.findall(text):
        if t.zfill(5) not in ok_times:
            issues.append(f"time:{t}")
    m = re.findall(r"points to option ([A-H])", text)
    if not m:
        issues.append("no_conclusion")
    elif m[-1] != cand["answer"]:
        issues.append("wrong_conclusion")
    return issues


def process(ctx, item):
    cand, bundle = item["candidate"], item["bundle"]
    opts = ctx.get("think", {}) or {}
    # A question one modality already answers gets a corroboration chain: reach the answer, then say the
    # other modality agrees. Writing it as if both were required would make the reasoning dishonest.
    corroborating = cand.get("corroborating_modality") if cand.get("quality") in ("medium", "low") else None
    name = "think_corroborate" if corroborating and opts.get("corroborate", True) else "think"
    fields = dict(captions=captions_text(bundle, with_refs=False)[: int(opts.get("caption_chars", 14000))],
                  mcq=mcq_text(cand), answer=cand["answer"], evidence=evidence_text(cand),
                  opening=opts.get("opening", "Got it!"))
    if name == "think_corroborate":
        fields["corroborating"] = "audio" if corroborating == "audio" else "visuals"
    text = ctx.prompts[name].render(**fields)
    best = None
    for i in range(int(opts.get("max_attempts", 2))):
        res = ctx.models("thinker").call([{"type": "text", "text": text}], json=False, sample=i, tag="think",
                                         context={"candidate": cand["id"]})
        body = res["text"].strip()
        if body.startswith("[UNGROUNDED]"):
            best = (["ungrounded"], body)
            break
        issues = qc(body, cand, opts)
        if best is None or len(issues) < len(best[0]):
            best = (issues, body)
        if not issues:
            break
    issues, body = best
    lin = lineage.make(NAME, VERSION, [cand["id"]], **lineage_extra(ctx, sys.modules[__name__]))
    rec = {**cand, "thinking": body, "think_qc": issues, "think_style": name, "think_lineage": lin}
    if issues:
        return {"rejected": [{**rec, "rejected_by": "think_qc", "reason": issues[0]}]}
    return {"out": [rec]}
