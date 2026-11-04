"""Compare two runs' funnels side by side: stage yields, check outcomes, per-task yields."""
from collections import Counter

from .funnel import funnel


def _rate(counts):
    total = sum(counts.values())
    return counts.get("passed", 0) / total if total else 0.0, total


def compare(run_a, run_b):
    fa, fb = funnel(run_a), funnel(run_b)
    lines = [f"A = {run_a}", f"B = {run_b}", "", "stage        A.out  B.out"]
    for st in sorted(set(fa["stages"]) | set(fb["stages"])):
        a, b = fa["stages"].get(st, {}), fb["stages"].get(st, {})
        lines.append(f"{st:10} {a.get('out', '-'):>6} {b.get('out', '-'):>6}")
    lines += ["", "check         A.pass_rate (n)   B.pass_rate (n)   top failure reasons A | B"]
    for ch in sorted(set(fa["checks"]) | set(fb["checks"])):
        ca, cb = Counter(fa["checks"].get(ch, {})), Counter(fb["checks"].get(ch, {}))
        (ra, na), (rb, nb) = _rate(ca), _rate(cb)
        top = lambda c: ", ".join(f"{k}={v}" for k, v in c.most_common(3) if k != "passed")  # noqa: E731
        lines.append(f"{ch:12} {ra:6.1%} ({na:3})     {rb:6.1%} ({nb:3})     {top(ca)} | {top(cb)}")
    lines += ["", "task           A.final  B.final"]
    for t in sorted(set(fa["by_task"]) | set(fb["by_task"])):
        a = fa["by_task"].get(t, {}).get("think_out", 0)
        b = fb["by_task"].get(t, {}).get("think_out", 0)
        lines.append(f"{t:14} {a:7} {b:7}")
    return "\n".join(lines)
