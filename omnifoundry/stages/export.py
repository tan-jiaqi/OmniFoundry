"""S5 export: training files + funnel.

sft.jsonl  ChatML: user = video + question + options, assistant = "<think>…</think>\n\n{letter}"
rl.jsonl   question, options, answer, audio_clues, visual_clues (text, for clue-conditioned scoring
           such as MFSD), plus task type and difficulty signals collected by the gates.
Export is cheap and deterministic, so it is not cached.
"""
from ..jsonio import read_jsonl, write_json, write_jsonl
from ..schema import AUDIO_MODALITIES

NAME = "export"


def _clue_text(c):
    return f"[{c['timestamp']}] {c.get('fact') or c.get('quote')}"


def _signals(rec):
    v = rec.get("verify", {})
    return {k: v[k]["detail"] for k in ("gate_text", "gate_media") if k in v}


def sft_row(rec, opts):
    user = opts.get("user_template", "{question}\n{options}\nAnswer with the option letter.").format(
        question=rec["question"], options="\n".join(rec["options"]))
    return {"id": rec["id"], "video": rec["video_path"],
            "messages": [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": user}]},
                         {"role": "assistant", "content": f"<think>\n{rec['thinking']}\n</think>\n\n{rec['answer']}"}],
            "task_type": rec["task_type"], "route": rec["route"]}


def rl_row(rec):
    return {"id": rec["id"], "video_id": rec["video_id"], "video": rec["video_path"], "question": rec["question"],
            "options": rec["options"], "answer": rec["answer"], "task_type": rec["task_type"], "route": rec["route"],
            "audio_clues": [_clue_text(c) for c in rec["clues"] if c["modality"] in AUDIO_MODALITIES],
            "visual_clues": [_clue_text(c) for c in rec["clues"] if c["modality"] == "visual"],
            "signals": _signals(rec)}


def run(ctx, log=print):
    out_dir = ctx.run_dir / NAME
    bundles = {b["video_id"]: b for b in read_jsonl(ctx.run_dir / "caption" / "out.jsonl")}
    thought = read_jsonl(ctx.run_dir / "think" / "out.jsonl")
    verified = read_jsonl(ctx.run_dir / "verify" / "out.jsonl")
    for rec in thought + verified:
        rec["video_path"] = bundles[rec["video_id"]]["extra"]["video_path"]
    opts = ctx.get("export", {}) or {}
    sft = [sft_row(r, opts) for r in thought]
    rl = [rl_row(r) for r in verified]
    write_jsonl(out_dir / "sft.jsonl", sft)
    write_jsonl(out_dir / "rl.jsonl", rl)
    from ..analysis.funnel import funnel
    f = funnel(ctx.run_dir)
    write_json(ctx.run_dir / "funnel.json", f)
    log(f"[export] sft={len(sft)} rl={len(rl)} -> {out_dir}")
    return {"sft": len(sft), "rl": len(rl)}
