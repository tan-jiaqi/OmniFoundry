"""Funnel: where candidates are lost, per stage / check / reason / task type, plus model cost.

This is the shared measuring stick for every change to prompts, policies or checks (see compare.py).
`cost` counts what this run actually did: items served from the stage cache make no calls and do not appear.
"""
from collections import Counter, defaultdict

from ..jsonio import read_jsonl

STAGES = ("select", "caption", "generate", "verify", "think")


def funnel(run_dir):
    out = {"stages": {}, "checks": {}, "by_task": {}, "cost": {}}
    for st in STAGES:
        d = run_dir / st
        if not d.exists():
            continue
        out["stages"][st] = {"out": len(read_jsonl(d / "out.jsonl")), "rejected": len(read_jsonl(d / "rejected.jsonl")),
                             "errors": len(read_jsonl(d / "errors.jsonl"))}
    checks = defaultdict(Counter)
    for st in ("generate", "verify"):
        for v in read_jsonl(run_dir / st / "verdicts.jsonl"):
            checks[v["check"]]["passed" if v["passed"] else v["reason"]] += 1
    out["checks"] = {k: dict(v) for k, v in checks.items()}

    by_task = defaultdict(Counter)
    for c in read_jsonl(run_dir / "generate" / "rejected.jsonl"):
        if "task_type" in c:
            by_task[c["task_type"]]["generate_rejected_attempts"] += 1
    for st in ("generate", "verify", "think"):
        for c in read_jsonl(run_dir / st / "out.jsonl"):
            by_task[c["task_type"]][f"{st}_out"] += 1
    out["by_task"] = {k: dict(v) for k, v in sorted(by_task.items())}

    cost = defaultdict(lambda: Counter())
    for row in read_jsonl(run_dir / "calls.jsonl"):
        c = cost[row["role"]]
        if row.get("error"):
            c["errors"] += 1
            continue
        c["calls"] += 1
        c["cached"] += bool(row.get("cached"))
        u = row.get("usage") or {}
        c["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
        c["completion_tokens"] += int(u.get("completion_tokens") or 0)
    out["cost"] = {k: dict(v) for k, v in sorted(cost.items())}
    return out


def render(f):
    lines = ["stage      out  rejected  errors"]
    for st, v in f["stages"].items():
        lines.append(f"{st:9} {v['out']:4} {v['rejected']:9} {v['errors']:7}")
    lines.append("\ncheck results")
    for ch, counts in f["checks"].items():
        lines.append(f"  {ch:12} " + ", ".join(f"{k}={n}" for k, n in sorted(counts.items(), key=lambda x: -x[1])))
    lines.append("\nby task")
    for t, counts in f["by_task"].items():
        lines.append(f"  {t:14} " + ", ".join(f"{k}={n}" for k, n in sorted(counts.items())))
    lines.append("\nmodel calls")
    for role, c in f["cost"].items():
        lines.append(f"  {role:12} calls={c.get('calls', 0)} cached={c.get('cached', 0)} errors={c.get('errors', 0)} "
                     f"tokens={c.get('prompt_tokens', 0)}+{c.get('completion_tokens', 0)}")
    return "\n".join(lines)
