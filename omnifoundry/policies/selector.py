"""TaskSelector: which task type to generate for a video.

Every selector has the signature `select(ctx, bundle, route, attempt_history) -> list[str]` and returns
task ids in preference order (empty list = skip this video). The generate stage tries them in order.

- `llm`: the selector model reads the catalog and captions and picks (OmniQA behaviour), with the
  configured weights shown to it as a prior.
- `weighted`: sample by `tasks.weights` without a model call. A weakness-driven selector would compute
  these weights from student-model error rates per task type and plug in here.
"""
import random

from ..jsonio import digest


def _weights(ctx, route):
    tasks = ctx.catalog.for_route(route)
    w = ctx.get("tasks.weights", {}) or {}
    return {t["id"]: float(w.get(t["id"], 1.0)) for t in tasks}


def weighted(ctx, bundle, route, history):
    weights = {k: v for k, v in _weights(ctx, route).items() if v > 0}
    rng = random.Random(digest("select", bundle["id"], ctx.get("tasks.seed", 0)))
    order = []
    pool = dict(weights)
    while pool:
        total = sum(pool.values())
        r, acc = rng.random() * total, 0.0
        for k, v in pool.items():
            acc += v
            if acc >= r:
                order.append(k)
                del pool[k]
                break
    return order


def llm(ctx, bundle, route, history):
    from ..stages.common import captions_text
    prompt = ctx.prompts["select_task"]
    weights = _weights(ctx, route)
    prior = ", ".join(f"{k}={v:g}" for k, v in sorted(weights.items()))
    text = prompt.render(catalog=ctx.catalog.describe(route), captions=captions_text(bundle), weights=prior,
                         max_choices=ctx.get("tasks.max_choices", 2))
    res = ctx.models("selector").call([{"type": "text", "text": text}], tag="select_task",
                                      context={"bundle": bundle["id"]})
    picks = (res["json"] or {}).get("task_ids") or []
    valid = [p for p in picks if p in weights and weights[p] > 0]
    return valid


SELECTORS = {"llm": llm, "weighted": weighted}
PROMPTS = {"llm": ["select_task"], "weighted": []}
