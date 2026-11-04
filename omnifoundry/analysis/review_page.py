"""Static HTML page for manual spot checks: question, options, gold, clue chain with playable clips, thinking.

Reviewers open the page, judge each item, and export their judgments as JSON (`reviews.json`), which is
how human accuracy per run and per task type gets measured.
"""
import html
import json
import random

from ..jsonio import read_jsonl


def _esc(x):
    return html.escape(str(x))


def build(run_dir, stage="think", n=20, seed=0):
    rows = read_jsonl(run_dir / stage / "out.jsonl")
    bundles = {b["video_id"]: b for b in read_jsonl(run_dir / "caption" / "out.jsonl")}
    random.Random(seed).shuffle(rows)
    rows = rows[:n]
    cards = []
    for r in rows:
        path = bundles.get(r["video_id"], {}).get("extra", {}).get("video_path", "")
        opts = "".join(f"<li class='{'gold' if o.startswith(r['answer'] + '.') else ''}'>{_esc(o)}</li>" for o in r["options"])
        clues = "".join(
            f"<tr><td>{_esc(c['modality'])}</td><td><a href='#' onclick=\"seek('{_esc(r['id'])}',{c['start_s']});return false\">{_esc(c['timestamp'])}</a></td>"
            f"<td>{_esc(c.get('fact') or c.get('quote'))}</td></tr>" for c in r["clues"])
        thinking = f"<details><summary>thinking</summary><pre>{_esc(r.get('thinking', ''))}</pre></details>" if r.get("thinking") else ""
        cards.append(
            f"<section id='{_esc(r['id'])}'><h3>{_esc(r['task_type'])} · {_esc(r['video_id'])}</h3>"
            f"<video controls preload='metadata' src='file://{_esc(path)}'></video>"
            f"<p class='q'>{_esc(r['question'])}</p><ol type='A' class='opts'>{opts}</ol>"
            f"<table>{clues}</table>{thinking}"
            f"<p class='judge'><label><input type='radio' name='{_esc(r['id'])}' value='correct'>correct</label> "
            f"<label><input type='radio' name='{_esc(r['id'])}' value='wrong'>wrong</label> "
            f"<input class='note' data-id='{_esc(r['id'])}' placeholder='error type / note'></p></section>")
    ids = json.dumps([r["id"] for r in rows])
    page = f"""<!doctype html><meta charset='utf-8'><title>OmniFoundry review</title><style>
body{{font:15px system-ui;max-width:860px;margin:2em auto;padding:0 16px}}section{{border-top:1px solid #ccc;padding:1em 0}}
video{{width:100%;max-height:360px;background:#000}}.q{{font-weight:600}}.opts{{list-style:none;padding:0}}.gold{{color:#0a7d32;font-weight:600}}
table{{border-collapse:collapse;width:100%;font-size:13px}}td{{border:1px solid #ddd;padding:4px;vertical-align:top}}.note{{width:50%}}pre{{white-space:pre-wrap}}
</style><h1>Review: {_esc(run_dir.name)} ({len(rows)} items)</h1>{''.join(cards)}
<button onclick='save()'>Download reviews.json</button><script>
const ids={ids};function seek(id,t){{const v=document.getElementById(id).querySelector('video');v.currentTime=t;v.play()}}
function save(){{const out=ids.map(id=>({{id,judgment:(document.querySelector(`input[name="${{id}}"]:checked`)||{{}}).value||null,
note:document.querySelector(`.note[data-id="${{id}}"]`).value}}));
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{{type:'application/json'}}));a.download='reviews.json';a.click()}}
</script>"""
    out = run_dir / "review.html"
    out.write_text(page, encoding="utf-8")
    return out
