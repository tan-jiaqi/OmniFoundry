"""Local review UI for one run: answer each question blind, then reveal the gold, the evidence and the thinking,
and record a judgment. Results go to <run>/reviews.jsonl and feed `omnifoundry funnel` (human accuracy by tag).

    python -m omnifoundry review-serve runs/m1e-001 [--stage think] [--port 8765]

Open http://127.0.0.1:8765 (from your own computer: ssh -L 8765:127.0.0.1:8765 <server>, then the same URL).
Standard library only; binds to 127.0.0.1; serves only the videos of the run's own questions (by item id).
Every save rewrites reviews.jsonl atomically and keeps the previous version as reviews.jsonl.bak.
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import shutil
import threading
import time
from urllib.parse import unquote

from ..jsonio import read_jsonl, write_jsonl

LOCK = threading.Lock()
VERDICTS = ("correct", "wrong", "ambiguous", "unsure")
NEEDS = ("both", "audio_only", "visual_only", "neither")
ISSUES = ("caption_error", "unclear_question", "answer_leak", "too_easy", "bad_distractor", "other")


def load_items(run_dir, stage):
    bundles = {b["video_id"]: b for b in read_jsonl(run_dir / "caption" / "out.jsonl")}
    items = []
    for r in read_jsonl(run_dir / stage / "out.jsonl"):
        b = bundles.get(r["video_id"], {})
        items.append({
            "id": r["id"], "video_id": r["video_id"], "task_type": r["task_type"], "tags": r.get("tags", []),
            "dependency": r.get("modality_dependency"), "question": r["question"], "options": r["options"],
            "answer": r["answer"], "clues": r["clues"], "steps": r["steps"], "thinking": r.get("thinking", ""),
            "video_path": b.get("extra", {}).get("video_path", ""), "duration_s": b.get("extra", {}).get("duration_s"),
        })
    return items


def load_reviews(path):
    return {r["id"]: r for r in read_jsonl(path)}


def summarize(items, reviews):
    """Human accuracy overall and per tag; agreement between the human modality judgment and the tags."""
    out = {"reviewed": 0, "blind_correct": 0, "verdicts": {}, "by_tag": {}}
    for it in items:
        r = reviews.get(it["id"])
        if not r or not r.get("verdict"):
            continue
        out["reviewed"] += 1
        out["blind_correct"] += r.get("human_answer") == it["answer"]
        out["verdicts"][r["verdict"]] = out["verdicts"].get(r["verdict"], 0) + 1
        for tag in it["tags"] or ["untagged"]:
            t = out["by_tag"].setdefault(tag, {"reviewed": 0, "correct": 0, "needs_both": 0})
            t["reviewed"] += 1
            t["correct"] += r["verdict"] == "correct"
            t["needs_both"] += r.get("needs") == "both"
    return out


def _send_file(handler, path):
    """File response with HTTP Range support; browsers cannot seek a <video> without 206 responses."""
    size = path.stat().st_size
    ctype = "video/mp4" if path.suffix == ".mp4" else "application/octet-stream"
    m = re.fullmatch(r"bytes=(\d*)-(\d*)", handler.headers.get("Range", ""))
    start, end = 0, size - 1
    if m and (m[1] or m[2]):
        if m[1]:
            start = int(m[1])
            end = min(int(m[2]), size - 1) if m[2] else size - 1
        else:
            start = max(0, size - int(m[2]))
        if start > end or start >= size:
            handler.send_response(416)
            handler.send_header("Content-Range", f"bytes */{size}")
            handler.end_headers()
            return
        handler.send_response(206)
        handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    else:
        handler.send_response(200)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Accept-Ranges", "bytes")
    handler.send_header("Content-Length", str(end - start + 1))
    handler.end_headers()
    with path.open("rb") as f:
        f.seek(start)
        left = end - start + 1
        while left > 0:
            chunk = f.read(min(1 << 20, left))
            if not chunk:
                break
            try:
                handler.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                return
            left -= len(chunk)


def make_handler(run_dir, items, reviews_path):
    by_id = {it["id"]: it for it in items}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _json(self, value, code=200):
            data = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = unquote(self.path.split("?", 1)[0])
            if path == "/":
                data = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif path == "/api/items":
                with LOCK:
                    reviews = load_reviews(reviews_path)
                self._json({"run": run_dir.name, "items": [{k: v for k, v in it.items() if k != "video_path"} for it in items],
                            "reviews": reviews, "summary": summarize(items, reviews)})
            elif path.startswith("/video/"):
                it = by_id.get(path[len("/video/"):])
                if not it or not Path(it["video_path"]).is_file():
                    self._json({"error": "unknown video"}, 404)
                    return
                _send_file(self, Path(it["video_path"]))
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            if self.path != "/api/review":
                self._json({"error": "not found"}, 404)
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                rid = body["id"]
                if rid not in by_id:
                    raise ValueError("unknown id")
                if body.get("verdict") not in VERDICTS + (None,) or body.get("needs") not in NEEDS + (None,):
                    raise ValueError("bad verdict or needs")
                if not set(body.get("issues", [])) <= set(ISSUES):
                    raise ValueError("bad issue")
                row = {"id": rid, "human_answer": body.get("human_answer"), "verdict": body.get("verdict"),
                       "needs": body.get("needs"), "issues": sorted(body.get("issues", [])),
                       "note": str(body.get("note", ""))[:2000], "gold": by_id[rid]["answer"],
                       "tags": by_id[rid]["tags"], "task_type": by_id[rid]["task_type"], "t": round(time.time())}
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, 400)
                return
            with LOCK:
                reviews = load_reviews(reviews_path)
                reviews[rid] = row
                if reviews_path.exists():
                    shutil.copyfile(reviews_path, reviews_path.with_suffix(".jsonl.bak"))
                write_jsonl(reviews_path, [reviews[i] for i in by_id if i in reviews])
                summary = summarize(items, reviews)
            self._json({"ok": True, "summary": summary})

    return Handler


def serve(run_dir, stage="think", port=8765, host="127.0.0.1"):
    run_dir = Path(run_dir)
    items = load_items(run_dir, stage)
    if not items:
        raise SystemExit(f"no items in {run_dir / stage / 'out.jsonl'}")
    reviews_path = run_dir / "reviews.jsonl"
    server = ThreadingHTTPServer((host, port), make_handler(run_dir, items, reviews_path))
    print(f"{len(items)} questions from {run_dir}/{stage}; reviews -> {reviews_path}")
    print(f"open http://{host}:{port}   (remote: ssh -L {port}:127.0.0.1:{port} <server>)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return server


PAGE = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OmniFoundry 人工核验</title><style>
:root{--bg:#fafafa;--panel:#fff;--fg:#1f2328;--mut:#6b7280;--line:#e5e7eb;--acc:#2563eb;--ok:#15803d;--bad:#b91c1c;--warn:#b45309;--sel:#dbeafe}
@media (prefers-color-scheme:dark){:root{--bg:#121214;--panel:#1b1b1f;--fg:#e5e7eb;--mut:#9ca3af;--line:#2e2e35;--acc:#60a5fa;--ok:#4ade80;--bad:#f87171;--warn:#fbbf24;--sel:#1e3a5f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.55 system-ui,"PingFang SC","Microsoft YaHei",sans-serif;display:flex;height:100vh}
#side{width:270px;flex:none;overflow:auto;border-right:1px solid var(--line);background:var(--panel)}
#sum{padding:10px 12px;border-bottom:1px solid var(--line);font-size:12px;color:var(--mut)}#sum b{color:var(--fg)}
#filter{margin:8px 12px;width:calc(100% - 24px)}
#list div{padding:6px 12px;cursor:pointer;font-size:12px;border-left:3px solid transparent}
#list div.cur{background:var(--sel);border-left-color:var(--acc)}
.st{display:inline-block;width:16px}.st.correct{color:var(--ok)}.st.wrong{color:var(--bad)}.st.ambiguous,.st.unsure{color:var(--warn)}
main{flex:1;overflow:auto;padding:16px 22px;max-width:1000px}
video{width:100%;max-height:46vh;background:#000;border-radius:6px}
.meta{color:var(--mut);font-size:12px;margin:6px 0}.tag{display:inline-block;padding:0 6px;border-radius:4px;border:1px solid var(--line);margin-right:4px;font-size:12px}
.tag.av_strict{border-color:var(--ok);color:var(--ok)}
.q{font-size:16px;font-weight:600;margin:12px 0 8px}
.opt{display:block;width:100%;text-align:left;padding:7px 10px;margin:4px 0;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--fg);cursor:pointer;font:inherit}
.opt.mine{border-color:var(--acc);background:var(--sel)}.opt.gold{border-color:var(--ok);box-shadow:inset 3px 0 0 var(--ok)}
.opt.mine.wrongpick{border-color:var(--bad);box-shadow:inset 3px 0 0 var(--bad)}
section{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 14px;margin:12px 0}
h4{margin:0 0 6px;font-size:13px;color:var(--mut);font-weight:600}
table{border-collapse:collapse;width:100%;font-size:13px}td{border-top:1px solid var(--line);padding:4px 6px;vertical-align:top}
a.ts{color:var(--acc);cursor:pointer;white-space:nowrap}pre{white-space:pre-wrap;font:13px/1.55 inherit;margin:0}
.row{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:6px 0}.row>span{width:96px;color:var(--mut);font-size:13px}
.chip{padding:4px 10px;border:1px solid var(--line);border-radius:999px;cursor:pointer;background:var(--panel);color:var(--fg);font:inherit;font-size:13px}
.chip.on{background:var(--acc);border-color:var(--acc);color:#fff}
textarea{width:100%;min-height:52px;background:var(--panel);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px;font:inherit}
.bar{display:flex;gap:8px;align-items:center;margin-top:8px}.btn{padding:6px 14px;border-radius:6px;border:1px solid var(--acc);background:var(--acc);color:#fff;cursor:pointer;font:inherit}
.btn.sec{background:transparent;color:var(--acc)}#saved{color:var(--ok);font-size:12px}.hidden{display:none}
.keys{color:var(--mut);font-size:12px}
@media (max-width:760px){body{flex-direction:column}#side{width:auto;max-height:30vh;border-right:0;border-bottom:1px solid var(--line)}main{padding:12px 16px}}
</style></head><body>
<div id="side"><div id="sum"></div>
<select id="filter"><option value="">全部题目</option><option value="todo">未审</option><option value="av_strict">av_strict</option><option value="audio_sufficient">audio_sufficient</option><option value="visual_sufficient">visual_sufficient</option></select>
<div id="list"></div></div>
<main id="main"></main>
<script>
const NEEDS={both:"需要音画结合",audio_only:"只听音频即可",visual_only:"只看画面即可",neither:"都不需要"};
const VERD={correct:"答案正确",wrong:"答案错误",ambiguous:"有歧义（多个选项可对）",unsure:"无法判断"};
const ISSUES={caption_error:"描述/线索错误",unclear_question:"题意不清",answer_leak:"题面泄题",too_easy:"太简单",bad_distractor:"干扰项不合理",other:"其他"};
let D,cur=0,state={};
const $=s=>document.querySelector(s),esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const fmt=s=>{s=Math.max(0,Math.floor(s));return String(Math.floor(s/60)).padStart(2,"0")+":"+String(s%60).padStart(2,"0")};
async function load(){D=await (await fetch("/api/items")).json();renderSide();const i=D.items.findIndex(x=>!D.reviews[x.id]?.verdict);show(i<0?0:i)}
function visible(){const f=$("#filter").value;return D.items.map((x,i)=>[x,i]).filter(([x])=>!f||(f==="todo"?!D.reviews[x.id]?.verdict:x.tags.includes(f)))}
function renderSide(){const s=D.summary,n=D.items.length;
 const tags=Object.entries(s.by_tag).map(([t,v])=>`${esc(t)} ${v.correct}/${v.reviewed}`).join("　");
 $("#sum").innerHTML=`<b>${esc(D.run)}</b><br>已审 <b>${s.reviewed}</b>/${n}　盲答对 <b>${s.blind_correct}</b>/${s.reviewed}<br>`+
  Object.entries(s.verdicts).map(([k,v])=>`${esc(VERD[k]||k)} ${v}`).join("，")+(tags?`<br>按标签（判为正确/已审）：${tags}`:"");
 $("#list").innerHTML=visible().map(([x,i])=>{const r=D.reviews[x.id];const v=r?.verdict||"";
  const mark={correct:"✓",wrong:"✗",ambiguous:"?",unsure:"?"}[v]||"·";
  return `<div data-i="${i}" class="${i===cur?"cur":""}"><span class="st ${v}">${mark}</span>${i+1}. ${esc(x.task_type)} · ${esc(x.tags.join(","))}</div>`}).join("");
 document.querySelectorAll("#list div").forEach(d=>d.onclick=()=>show(+d.dataset.i))}
function show(i){cur=i;const x=D.items[i],r=D.reviews[x.id]||{};
 state={human_answer:r.human_answer||null,verdict:r.verdict||null,needs:r.needs||null,issues:new Set(r.issues||[]),note:r.note||"",revealed:!!r.human_answer};
 const dep=x.dependency?`只听音频 ${x.dependency.audio_only?.correct??"-"}/${x.dependency.audio_only?.trials??"-"}，只看画面 ${(x.dependency.video_only||x.dependency.visual_only)?.correct??"-"}/${(x.dependency.video_only||x.dependency.visual_only)?.trials??"-"}（依据 ${esc(x.dependency.evidence)}）`:"";
 $("#main").innerHTML=`<div class="meta">${i+1}/${D.items.length} · ${esc(x.video_id)} · ${esc(x.task_type)} · ${x.tags.map(t=>`<span class="tag ${t}">${esc(t)}</span>`).join("")}</div>
 <video id="v" controls preload="metadata" src="/video/${encodeURIComponent(x.id)}"></video>
 <div class="q">${esc(x.question)}</div><div id="opts"></div>
 <div class="keys">先自己作答（A–H 或点击选项）再看答案　空格 播放/暂停　← → 跳 2 秒　N / P 下一题 / 上一题　R 直接显示答案</div>
 <div id="reveal" class="${state.revealed?"":"hidden"}">
 <section><h4>证据链（点击时间跳转）</h4><table>${x.clues.map(c=>`<tr><td><a class="ts" data-t="${c.start_s}">${esc(c.timestamp)}</a></td><td>${esc(c.modality)}</td><td>${esc(c.quote)}</td></tr>`).join("")}</table></section>
 <section><h4>推理步骤</h4><table>${x.steps.map(s=>`<tr><td>${esc(s.step_id)}</td><td>${esc(s.modality)}</td><td>${esc(s.content)}</td></tr>`).join("")}</table></section>
 <section><h4>自动标签依据</h4>${dep||"—"}</section>
 ${x.thinking?`<section><h4>合成的 thinking</h4><pre>${esc(x.thinking)}</pre></section>`:""}
 <section><h4>你的判断</h4>
 <div class="row"><span>答案</span>${Object.entries(VERD).map(([k,v])=>`<button class="chip" data-g="verdict" data-k="${k}">${v}</button>`).join("")}</div>
 <div class="row"><span>音画依赖</span>${Object.entries(NEEDS).map(([k,v])=>`<button class="chip" data-g="needs" data-k="${k}">${v}</button>`).join("")}</div>
 <div class="row"><span>问题</span>${Object.entries(ISSUES).map(([k,v])=>`<button class="chip" data-g="issues" data-k="${k}">${v}</button>`).join("")}</div>
 <textarea id="note" placeholder="备注（可选）">${esc(state.note)}</textarea>
 <div class="bar"><button class="btn" id="save">保存并下一题</button><button class="btn sec" id="saveonly">保存</button><span id="saved"></span></div></section></div>`;
 renderOpts();renderChips();renderSide();
 document.querySelectorAll("a.ts").forEach(a=>a.onclick=()=>{const v=$("#v");v.currentTime=+a.dataset.t;v.play()});
 document.querySelectorAll(".chip").forEach(b=>b.onclick=()=>{const g=b.dataset.g,k=b.dataset.k;
  if(g==="issues"){state.issues.has(k)?state.issues.delete(k):state.issues.add(k)}else state[g]=state[g]===k?null:k;renderChips()});
 $("#save").onclick=()=>save(true);$("#saveonly").onclick=()=>save(false)}
function renderOpts(){const x=D.items[cur];$("#opts").innerHTML=x.options.map(o=>{const L=o[0];const cls=["opt"];
  if(state.human_answer===L)cls.push("mine");if(state.revealed&&L===x.answer)cls.push("gold");if(state.revealed&&state.human_answer===L&&L!==x.answer)cls.push("wrongpick");
  return `<button class="${cls.join(" ")}" data-l="${L}">${esc(o)}</button>`}).join("")+(state.revealed?`<div class="meta">正确答案 ${x.answer}${state.human_answer?`，你选了 ${state.human_answer}`:""}</div>`:"");
 document.querySelectorAll(".opt").forEach(b=>b.onclick=()=>pick(b.dataset.l))}
function renderChips(){document.querySelectorAll(".chip").forEach(b=>{const g=b.dataset.g,k=b.dataset.k;
  b.classList.toggle("on",g==="issues"?state.issues.has(k):state[g]===k)})}
function pick(L){if(state.revealed)return;state.human_answer=L;reveal()}
function reveal(){state.revealed=true;$("#reveal").classList.remove("hidden");renderOpts()}
async function save(next){const x=D.items[cur];
 const body={id:x.id,human_answer:state.human_answer,verdict:state.verdict,needs:state.needs,issues:[...state.issues],note:$("#note").value};
 const r=await fetch("/api/review",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});const j=await r.json();
 if(!r.ok){$("#saved").textContent="保存失败："+(j.error||r.status);return}
 D.reviews[x.id]={...body,gold:x.answer};D.summary=j.summary;$("#saved").textContent="已保存";renderSide();
 if(next){const vis=visible().map(([,i])=>i);const k=vis.indexOf(cur);if(k>=0&&k<vis.length-1)show(vis[k+1])}}
document.addEventListener("keydown",e=>{if(e.target.tagName==="TEXTAREA")return;const v=$("#v");const k=e.key.toUpperCase();
 if(e.key===" "&&v){e.preventDefault();v.paused?v.play():v.pause()}
 else if(e.key==="ArrowLeft"&&v)v.currentTime=Math.max(0,v.currentTime-2);else if(e.key==="ArrowRight"&&v)v.currentTime+=2;
 else if(k==="N"){if(cur<D.items.length-1)show(cur+1)}else if(k==="P"){if(cur>0)show(cur-1)}
 else if(k==="R"){reveal()}else if(/^[A-H]$/.test(k)&&D.items[cur].options.some(o=>o[0]===k))pick(k)});
$("#filter").onchange=renderSide;load();
</script></body></html>"""
