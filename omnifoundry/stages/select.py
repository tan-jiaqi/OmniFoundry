"""S0 select: dataset manifest -> Video records.

Sources are pluggable loaders (`select.source`). `omnivideo` reads OmniVideo-100K's scripts.jsonl for
ids / durations / search tags; `manifest` reads an existing JSONL of {id, path, ...}. The sampler policy
picks the subset; only picked videos are probed (ffprobe), and videos without audio are dropped.
"""
import json
from pathlib import Path

from ..config import resolve_path
from ..policies.sampler import SAMPLERS
from ..schema import Video
from .. import lineage

NAME, VERSION = "select", 1
CONFIG_KEYS = ("select",)


def prompts(ctx):
    return []


def roles(ctx):
    return []


def load_omnivideo(ctx, opts):
    root = resolve_path(opts["root"])
    videos_dir = root / "extracted" / "videos"
    lo, hi = opts.get("min_duration_s", 0), opts.get("max_duration_s", 1e9)
    out = []
    with open(root / "scripts.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            vid, dur = row.get("video_id"), row.get("duration")
            path = videos_dir / f"{vid}.mp4"
            if not vid or dur is None or not (lo <= float(dur) <= hi) or not path.exists():
                continue
            out.append({"id": f"omnivideo-{vid}", "path": str(path), "duration_s": float(dur), "source": "omnivideo",
                        "meta": {"search_tag": row.get("search_tag", ""), "language": row.get("language", "")}})
    return out


def load_manifest(ctx, opts):
    out = []
    with open(resolve_path(opts["path"]), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out.append({"id": row["id"], "path": str(resolve_path(row["path"])), "source": row.get("source", "manifest"),
                            "duration_s": row.get("duration_s"), "meta": row.get("meta", {})})
    return out


SOURCES = {"omnivideo": load_omnivideo, "manifest": load_manifest}


def inputs(ctx, prev_dir=None):
    opts = ctx.get("select", {})
    pool = SOURCES[opts.get("source", "manifest")](ctx, opts)
    picked = SAMPLERS[ctx.get("policies.sampler", "stratified")](pool, opts)
    return picked


def process(ctx, item):
    info = ctx.media.probe(item["path"])
    lin = lineage.make(NAME, VERSION, [], source_path=item["path"])
    if not info["has_audio"] or not info["has_video"]:
        rec = dict(item, reason="no_audio" if not info["has_audio"] else "no_video", probe=info)
        return {"rejected": [rec]}
    v = Video(id=item["id"], path=item["path"], duration_s=info["duration_s"], has_audio=True,
              source=item.get("source", ""), meta={**item.get("meta", {}), "width": info["width"], "height": info["height"]},
              lineage=lin)
    return {"out": [v.to_dict()]}


def video_file_key(path):
    p = Path(path)
    st = p.stat()
    return [str(p.resolve()), st.st_size, int(st.st_mtime)]
