"""Model-based text checks on the candidate: caption-conditioned solver and question-only blind test."""
import re

from ..stages.common import captions_text, mcq_text
from . import Check, register


def answer_letter(res):
    """Pull an option letter from a model reply: JSON {'answer': 'C'} or text. None if absent."""
    data = res.get("json") if isinstance(res, dict) else None
    raw = (data or {}).get("answer") if isinstance(data, dict) else None
    if raw is None:
        raw = res.get("text", "") if isinstance(res, dict) else str(res)
    m = re.match(r"^\s*\(?([A-H])\b", str(raw).strip())
    return m.group(1) if m else None


@register
class Solver(Check):
    """A solver sees all captions and the MCQ (no answer). Pass iff it picks the gold answer and reports
    every presupposition supported. Unparseable replies fail (OmniQA counted them as passes); the gold
    is never silently reassigned."""
    name = "solver"
    prompts = ("solver",)
    roles = ("solver",)

    def run(self, ctx, cand, bundle):
        text = ctx.prompts["solver"].render(captions=captions_text(bundle), mcq=mcq_text(cand))
        res = ctx.models("solver").call([{"type": "text", "text": text}], tag="solver", context={"candidate": cand["id"]})
        data = res.get("json") or {}
        pick = answer_letter(res)
        sample = {"answer": pick, "supported": data.get("all_presuppositions_supported"), "model": res.get("model")}
        if pick is None:
            return self.verdict(cand, False, "solver_unparseable", samples=[sample])
        if data.get("all_presuppositions_supported") is False:
            return self.verdict(cand, False, "presupposition_unsupported", samples=[sample],
                                detail={"hint": "; ".join(map(str, data.get("unsupported", [])))[:500]})
        if pick != cand["answer"]:
            return self.verdict(cand, False, "solver_disagrees", samples=[sample],
                                detail={"hint": f"solver chose {pick} instead of {cand['answer']}: {str(data.get('reason', ''))[:300]}"})
        return self.verdict(cand, True, samples=[sample])


@register
class Blind(Check):
    """Question-only shortcut test: `trials` draws see only the question and options; any correct pick
    (beyond `max_correct`) means the answer leaks through wording or priors."""
    name = "blind"
    prompts = ("blind",)
    roles = ("blind",)

    def run(self, ctx, cand, bundle):
        o = self.options(ctx)
        trials, temp = int(o.get("trials", 2)), float(o.get("temperature", 0.3))
        text = ctx.prompts["blind"].render(mcq=mcq_text(cand))
        samples = []
        for i in range(trials):
            res = ctx.models("blind").call([{"type": "text", "text": text}], sample=i, temperature=temp, tag="blind",
                                           context={"candidate": cand["id"]})
            samples.append({"answer": answer_letter(res), "reason": str((res.get("json") or {}).get("reason", ""))[:300]})
        correct = sum(s["answer"] == cand["answer"] for s in samples)
        if correct > int(o.get("max_correct", 0)):
            hint = next(s["reason"] for s in samples if s["answer"] == cand["answer"])
            return self.verdict(cand, False, "blind_correct", samples=samples,
                                detail={"hint": f"answerable without the video: {hint}", "correct": correct, "trials": trials})
        return self.verdict(cand, True, samples=samples, detail={"correct": correct, "trials": trials})
