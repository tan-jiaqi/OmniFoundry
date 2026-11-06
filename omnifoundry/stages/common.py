"""Shared stage machinery.

A stage is a module with:
    NAME, VERSION                 bump VERSION to force recomputation after a semantic code change
    CONFIG_KEYS                   config sections whose values affect output (hashed into the item key)
    extra_config_keys(ctx)        optional: further sections that depend on the config
    prompts(ctx) -> [names]       prompts used (their hashes join the item key and lineage)
    roles(ctx) -> [roles]         model roles used (their model ids join the item key)
    inputs(ctx, prev_dir) -> list of input records
    process(ctx, item) -> {"out": [...], "rejected": [...], "verdicts": [...]}

`run_stage` wraps `process` with the stage cache, a thread pool and per-item error capture, and writes
<run>/<stage>/{out,rejected,verdicts,errors}.jsonl. A change to a prompt or a config section changes the
item key of the stage that uses it, and therefore the content (hash) of its outputs, and therefore the
item keys of every downstream stage: exactly the affected work is recomputed, the rest is a cache hit.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import traceback

from ..config import section_hash
from ..jsonio import digest, write_jsonl
from ..schema import fmt_range


def config_keys(ctx, stage):
    """Config sections that affect a stage: static CONFIG_KEYS plus, if the stage defines
    `extra_config_keys(ctx)`, sections that depend on the config (e.g. only the checks in its own chain)."""
    extra = stage.extra_config_keys(ctx) if hasattr(stage, "extra_config_keys") else ()
    return tuple(stage.CONFIG_KEYS) + tuple(extra)


def _content(obj):
    """The item without provenance: drop `lineage` / `*_lineage` keys at any depth. Lineage carries the
    code version, which must not reach cache keys (refactors would otherwise re-run every stage)."""
    if isinstance(obj, dict):
        return {k: _content(v) for k, v in obj.items() if k != "lineage" and not k.endswith("_lineage")}
    if isinstance(obj, list):
        return [_content(v) for v in obj]
    return obj


def item_key(ctx, stage, item):
    return digest("stage", stage.NAME, stage.VERSION, _content(item),
                  section_hash(ctx.config, *config_keys(ctx, stage)),
                  ctx.prompts.hashes(*stage.prompts(ctx)),
                  ctx.models.identities(*stage.roles(ctx)),
                  ctx.catalog.hash if getattr(stage, "USES_CATALOG", False) else None)


def lineage_extra(ctx, stage):
    return {"prompts": ctx.prompts.hashes(*stage.prompts(ctx)),
            "models": ctx.models.model_ids(*stage.roles(ctx)),
            "config_hash": section_hash(ctx.config, *config_keys(ctx, stage))[:16]}


def run_stage(ctx, stage, items, log=print):
    out_dir = ctx.run_dir / stage.NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = int(ctx.get(f"{stage.NAME}.workers", ctx.get("run.workers", 4)))
    results, errors = {}, []
    hits = 0
    t0 = time.monotonic()

    def one(idx, item):
        key = item_key(ctx, stage, item)
        value, hit = ctx.cache.memo("stage", key, lambda: stage.process(ctx, item))
        return idx, value, hit

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(one, i, it): (i, it) for i, it in enumerate(items)}
        for fut in as_completed(futures):
            i, it = futures[fut]
            try:
                idx, value, hit = fut.result()
                results[idx] = value
                hits += hit
            except Exception as exc:  # noqa: BLE001 - one bad item must not stop the run
                errors.append({"item_id": it.get("id"), "error": f"{type(exc).__name__}: {exc}",
                               "traceback": traceback.format_exc()[-2000:]})

    ordered = [results[i] for i in sorted(results)]
    out = [r for v in ordered for r in v.get("out", [])]
    rejected = [r for v in ordered for r in v.get("rejected", [])]
    verdicts = [r for v in ordered for r in v.get("verdicts", [])]
    write_jsonl(out_dir / "out.jsonl", out)
    write_jsonl(out_dir / "rejected.jsonl", rejected)
    write_jsonl(out_dir / "verdicts.jsonl", verdicts)
    write_jsonl(out_dir / "errors.jsonl", errors)
    log(f"[{stage.NAME}] items={len(items)} cache_hits={hits} out={len(out)} rejected={len(rejected)} "
        f"errors={len(errors)} {time.monotonic() - t0:.1f}s")
    return out


# ---- prompt rendering helpers --------------------------------------------------------------------

def captions_text(bundle, *, audio=True, visual=True, exclude_refs=(), with_refs=True):
    """Captions as prompt text, grouped by event. `exclude_refs` drops cited entries (complement tests)."""
    lines = [f"Video type: {bundle.get('video_type', '')}", f"Summary: {bundle.get('video_summary', '')}"]
    excluded = set(exclude_refs)
    for ev in bundle["events"]:
        lines.append(f"\n## Event {ev['event_id']} [{fmt_range(ev['start_s'], ev['end_s'])}]")
        if audio:
            for c in bundle["audio_captions"]:
                if c["event_id"] == ev["event_id"] and c["ref"] not in excluded:
                    ref = f"[{c['ref']}] " if with_refs else ""
                    lines.append(f"AUDIO {ref}{c['timestamp']} ({c['audio_type']}): {c['caption']}")
        if visual:
            for c in bundle["visual_captions"]:
                if c["event_id"] == ev["event_id"] and c["ref"] not in excluded:
                    ref = f"[{c['ref']}] " if with_refs else ""
                    lines.append(f"VISUAL {ref}{c['timestamp']}: {c['caption']}")
    return "\n".join(lines)


def mcq_text(candidate):
    opts = "\n".join(candidate["options"])
    return f"Question: {candidate['question']}\nOptions:\n{opts}"


def caption_index(bundle):
    return {c["ref"]: c for c in bundle["audio_captions"] + bundle["visual_captions"]}
