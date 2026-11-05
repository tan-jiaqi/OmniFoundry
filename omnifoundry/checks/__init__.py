"""Checks: independent pass/fail units over a Candidate, composed into chains by config.

    class MyCheck(Check):
        name = "my_check"
        prompts = ("my_prompt",)       # prompt names used (hashed into stage keys / lineage)
        roles = ("my_role",)           # model roles used
        def run(self, ctx, cand, bundle) -> Verdict

Register with @register. Chains are lists of names in the config (`generate.checks`, `verify.checks`);
each check reads its options from `checks.<name>` in the config. Every model draw a check makes goes into
`Verdict.samples`, so pass rates / difficulty can be recomputed later without new calls.
"""
from concurrent.futures import ThreadPoolExecutor

from ..schema import Verdict

CHECKS = {}


def parallel(jobs, max_workers=16):
    """Run zero-argument callables concurrently; results come back in job order.

    Each job is one independent model request, so results and cache keys are the same as running them
    one after another. The per-model limiter in llm.py still caps how many are in flight at once."""
    jobs = list(jobs)
    if len(jobs) <= 1:
        return [j() for j in jobs]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(jobs))) as pool:
        return list(pool.map(lambda j: j(), jobs))


def register(cls):
    CHECKS[cls.name] = cls
    return cls


class Check:
    name = "check"
    prompts = ()
    roles = ()

    def roles_for(self, ctx):
        """Model roles this check uses under the current config (default: the static `roles`)."""
        return tuple(self.roles)

    def options(self, ctx):
        return ctx.get(f"checks.{self.name}", {}) or {}

    def verdict(self, cand, passed, reason="", **kw):
        return Verdict(check=self.name, candidate_id=cand["id"], passed=bool(passed), reason=reason if not passed else (reason or "ok"),
                       **kw).to_dict()

    def run(self, ctx, cand, bundle):
        raise NotImplementedError


def chain(names):
    missing = [n for n in names if n not in CHECKS]
    if missing:
        raise KeyError(f"unknown checks {missing}; known: {sorted(CHECKS)}")
    return [CHECKS[n]() for n in names]


def chain_prompts(names):
    return sorted({p for c in chain(names) for p in c.prompts})


def chain_roles(ctx, names):
    return sorted({r for c in chain(names) for r in c.roles_for(ctx)})


def run_chain(ctx, checks, cand, bundle, *, short_circuit=True):
    """Returns (passed, verdicts, first_failure)."""
    verdicts, failure = [], None
    for check in checks:
        v = check.run(ctx, cand, bundle)
        verdicts.append(v)
        if not v["passed"] and failure is None:
            failure = v
            if short_circuit:
                break
    return failure is None, verdicts, failure


# Import modules so their @register decorators run.
from . import structure, balance, grounding, model_checks, clue, gate  # noqa: E402,F401
