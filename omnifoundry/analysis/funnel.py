"""Funnel: where candidates are lost, per stage / check / reason / task type, plus model cost.

This is the shared measuring stick for every change to prompts, policies or checks (see compare.py).
`cost` counts what this run actually did: items served from the stage cache make no calls and do not appear.
"""
from collections import Counter, defaultdict

from ..jsonio import read_jsonl

STAGES = ("select", "caption", "generate", "verify", "think")
# Pipeline order of all checks; survival counts candidates that passed each, over generation attempts.
CHECK_ORDER = ("parse", "structure", "option_balance", "grounding", "solver", "blind", "gate_text", "clue", "gate_media")


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

    # Survival: of all generation attempts, how many passed each check (checks absent from a run are
    # skipped). A candidate's last check is final only if it passed everything and reached think.
    attempts = len(read_jsonl(run_dir / "generate" / "out.jsonl")) + sum(
        1 for c in read_jsonl(run_dir / "generate" / "rejected.jsonl") if "options" in c) + checks["parse"].total()
    passed = {k: v.get("passed", 0) for k, v in checks.items()}
    out["survival"] = {"attempts": attempts,
                       **{k: passed[k] for k in CHECK_ORDER if k in passed},
                       "final": len(read_jsonl(run_dir / "think" / "out.jsonl"))}

    reviews = read_jsonl(run_dir / "reviews.jsonl")
    if reviews:
        hr = defaultdict(Counter)
        for r in reviews:
            for tag in r.get("tags") or ["untagged"]:
                hr[tag]["reviewed"] += bool(r.get("verdict"))
                hr[tag][r.get("verdict") or "unjudged"] += 1
                hr[tag]["blind_correct"] += r.get("human_answer") == r.get("gold")
                hr[tag]["needs_both"] += r.get("needs") == "both"
        out["human_review"] = {k: dict(v) for k, v in hr.items()}
    final = read_jsonl(run_dir / "think" / "out.jsonl")
    out["final_tags"] = dict(Counter(t for c in final for t in c.get("tags", [])))
    out["final_quality"] = dict(Counter(c.get("quality") or "untiered" for c in final))
    out["soft_failures"] = dict(Counter(r for c in final for r in c.get("soft_failures", [])))

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
    sv = f.get("survival", {})
    if sv.get("attempts"):
        lines.append("\nsurvival (share of generation attempts passing each check, in pipeline order)")
        for k, n in sv.items():
            if k != "attempts":
                lines.append(f"  {k:14} {n:4}  {n / sv['attempts']:6.1%}")
    if f.get("final_quality"):
        lines.append("\nfinal data by quality: " + ", ".join(f"{k}={v}" for k, v in sorted(f["final_quality"].items())))
        if f.get("soft_failures"):
            lines.append("  downgraded because: " + ", ".join(f"{k}={v}" for k, v in sorted(f["soft_failures"].items())))
    if f.get("final_tags"):
        lines.append("\nfinal data by tag: " + ", ".join(f"{k}={v}" for k, v in sorted(f["final_tags"].items())))
    if f.get("human_review"):
        lines.append("\nhuman review by tag")
        for tag, c in sorted(f["human_review"].items()):
            lines.append(f"  {tag:18} reviewed={c.get('reviewed', 0)} correct={c.get('correct', 0)} wrong={c.get('wrong', 0)} "
                         f"ambiguous={c.get('ambiguous', 0)} blind_correct={c.get('blind_correct', 0)} needs_both={c.get('needs_both', 0)}")
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
