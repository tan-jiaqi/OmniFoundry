"""Option balance: the gold option must not stand out by length.

M1c: the gold was the longest of 8 options in 25% of candidates (chance 12.5%), and in 33% of those that
failed the blind test. A guesser picks the longest, most specific option, so this cheap deterministic
check runs before any model-based check. It fails when the gold is the strictly longest option and
longer than the median of the other options by more than `max_ratio`.
"""
import statistics

from . import Check, register


def _body(opt):
    return opt[3:].strip() if len(opt) > 2 and opt[1] == "." else opt.strip()


@register
class OptionBalance(Check):
    name = "option_balance"

    def run(self, ctx, cand, bundle):
        o = self.options(ctx)
        max_ratio = float(o.get("max_ratio", 1.25))
        lengths = {opt[0]: len(_body(opt)) for opt in cand["options"]}
        gold = lengths.get(cand["answer"])
        others = [n for k, n in lengths.items() if k != cand["answer"]]
        if gold is None or not others:
            return self.verdict(cand, True)
        ratio = gold / max(1, statistics.median(others))
        detail = {"gold_len": gold, "median_other_len": statistics.median(others), "ratio": round(ratio, 2)}
        if gold > max(others) and ratio > max_ratio:
            detail["hint"] = (f"the correct option is the longest ({gold} chars vs median {statistics.median(others):.0f}); "
                              f"make every distractor as long and as specific as the correct option, or shorten it")
            return self.verdict(cand, False, "answer_length_leak", detail=detail)
        return self.verdict(cand, True, detail=detail)
