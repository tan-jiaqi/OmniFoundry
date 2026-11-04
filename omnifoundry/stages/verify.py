"""S3 verify: run the `verify.checks` chain (clue validity, modality gates) on generated candidates.

Items pair each candidate with its caption bundle, so a changed bundle re-verifies its candidates.
Accepted candidates gain a `verify` summary; every verdict (with all model draws) is written out.
"""
import sys

from ..checks import chain, chain_prompts, chain_roles, run_chain
from ..jsonio import read_jsonl
from .. import lineage
from .common import lineage_extra

NAME, VERSION = "verify", 1
CONFIG_KEYS = ("verify",)


def _checks(ctx):
    return ctx.get("verify.checks", ["clue", "gate_text", "gate_media"])


def extra_config_keys(ctx):
    return [f"checks.{n}" for n in _checks(ctx)]


def prompts(ctx):
    return chain_prompts(_checks(ctx))


def roles(ctx):
    return chain_roles(ctx, _checks(ctx))


def inputs(ctx, prev_dir):
    """prev_dir is the generate stage; bundles come from the caption stage of the same run."""
    bundles = {b["video_id"]: b for b in read_jsonl(ctx.run_dir / "caption" / "out.jsonl")}
    return [{"id": c["id"], "candidate": c, "bundle": bundles[c["video_id"]]}
            for c in read_jsonl(prev_dir / "out.jsonl") if c["video_id"] in bundles]


def process(ctx, item):
    cand, bundle = item["candidate"], item["bundle"]
    passed, verdicts, failure = run_chain(ctx, chain(_checks(ctx)), cand, bundle,
                                          short_circuit=ctx.get("verify.short_circuit", True))
    lin = lineage.make(NAME, VERSION, [cand["id"], bundle["id"]], **lineage_extra(ctx, sys.modules[__name__]))
    summary = {v["check"]: {"passed": v["passed"], "reason": v["reason"], "detail": v.get("detail", {})} for v in verdicts}
    rec = {**cand, "verify": summary, "verify_lineage": lin}
    if passed:
        return {"out": [rec], "verdicts": verdicts}
    return {"rejected": [{**rec, "rejected_by": failure["check"], "reason": failure["reason"]}], "verdicts": verdicts}
