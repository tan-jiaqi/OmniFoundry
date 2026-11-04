"""Deterministic structure check: options, answer, clues, and the reasoning-step DAG."""
import re

from ..schema import AUDIO_MODALITIES, MODALITIES, letters
from . import Check, register


@register
class Structure(Check):
    name = "structure"

    def run(self, ctx, cand, bundle):
        o = self.options(ctx)
        n_opts = int(o.get("num_options", 8))
        opts = cand["options"]
        if len(opts) != n_opts:
            return self.verdict(cand, False, "options_invalid", detail={"hint": f"need exactly {n_opts} options, got {len(opts)}"})
        expected = letters(n_opts)
        bodies = []
        for letter, opt in zip(expected, opts):
            if not re.match(rf"^{letter}\.\s+\S", opt):
                return self.verdict(cand, False, "options_invalid", detail={"hint": f"option must start with '{letter}. ': {opt[:40]!r}"})
            bodies.append(re.sub(r"\s+", " ", opt[3:].strip().lower()))
        if len(set(bodies)) != len(bodies):
            return self.verdict(cand, False, "options_invalid", detail={"hint": "duplicate option texts"})
        if cand["answer"] not in expected:
            return self.verdict(cand, False, "answer_invalid", detail={"hint": f"answer must be one of {expected}"})

        clues = cand["clues"]
        ids = [c.get("id") for c in clues]
        if len(set(ids)) != len(ids) or not all(ids):
            return self.verdict(cand, False, "dag_invalid", detail={"hint": "clue ids must be unique and non-empty"})
        mods = {c.get("modality") for c in clues}
        if not mods <= set(MODALITIES):
            return self.verdict(cand, False, "dag_invalid", detail={"hint": f"clue modality must be one of {MODALITIES}"})
        if "visual" not in mods or not mods & set(AUDIO_MODALITIES):
            return self.verdict(cand, False, "missing_modality", detail={"hint": "need at least one audio clue and one visual clue"})

        steps = cand["steps"]
        if len(steps) < int(o.get("min_steps", 2)):
            return self.verdict(cand, False, "dag_invalid", detail={"hint": f"need at least {o.get('min_steps', 2)} reasoning steps"})
        seen, clue_ids = set(), set(ids)
        step_mods = set()
        for s in steps:
            sid = s.get("step_id")
            if not sid or sid in seen:
                return self.verdict(cand, False, "dag_invalid", detail={"hint": f"bad or duplicate step_id {sid!r}"})
            for d in s.get("depends_on", []):
                if d not in seen:
                    return self.verdict(cand, False, "dag_invalid", detail={"hint": f"step {sid} depends on {d}, which is not an earlier step"})
            for c in s.get("clue_ids", []):
                if c not in clue_ids:
                    return self.verdict(cand, False, "dag_invalid", detail={"hint": f"step {sid} cites unknown clue {c}"})
            step_mods.add(s.get("modality"))
            seen.add(sid)
        if "visual" not in step_mods or not step_mods & {"audio", *AUDIO_MODALITIES}:
            return self.verdict(cand, False, "missing_modality", detail={"hint": "steps must include an audio step and a visual step"})
        # Every step except the last must feed a later step (no dangling evidence).
        used = {d for s in steps for d in s.get("depends_on", [])}
        dangling = [s["step_id"] for s in steps[:-1] if s["step_id"] not in used]
        if dangling and not o.get("allow_dangling", False):
            return self.verdict(cand, False, "dag_invalid", detail={"hint": f"steps {dangling} are not used by any later step"})
        return self.verdict(cand, True)
