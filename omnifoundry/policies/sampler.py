"""VideoSampler: which videos enter a run.

Default `stratified`: shuffle with a seed, then round-robin over strata (a manifest field, e.g. the
dataset's search tag), so a small pilot still spans content types. Future samplers can read failure
statistics from earlier runs and prefer video types that feed weak task types.
"""
import random


def stratified(videos, options):
    n = options.get("n")
    key = options.get("stratify_by", "source")
    rng = random.Random(options.get("seed", 0))
    groups = {}
    for v in videos:
        groups.setdefault(str(v.get("meta", {}).get(key, v.get(key, ""))), []).append(v)
    for g in groups.values():
        rng.shuffle(g)
    order = sorted(groups)
    out = []
    while any(groups.values()) and (n is None or len(out) < n):
        for k in order:
            if groups[k] and (n is None or len(out) < n):
                out.append(groups[k].pop())
    return out


def first(videos, options):
    n = options.get("n")
    return list(videos)[:n] if n else list(videos)


SAMPLERS = {"stratified": stratified, "first": first}
