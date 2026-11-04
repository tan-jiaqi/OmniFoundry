"""Which configured models exist on the gateway, and which media wire formats do they understand?

Sends tiny synthetic clips with known answers (RED screen + 440 Hz tone, then BLUE + 1200 Hz), never dataset media:
    env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy \
      python scripts/probe_gateway.py --config configs/base.yaml --out runs/_probe
"An answer that is right" is the test, since gateways may silently drop parts they cannot forward.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omnifoundry.cache import Cache  # noqa: E402
from omnifoundry.config import load_config, load_env_file  # noqa: E402
from omnifoundry.jsonio import write_json  # noqa: E402
from omnifoundry.llm import Client  # noqa: E402
from omnifoundry.media import Media  # noqa: E402

VIDEO_Q = ("What is the screen color at the start and at the end of this video? And is the tone higher in pitch at the start or at the end? "
           "Reply in the form: start=<color> end=<color> higher=<start|end>. If you did not receive video or audio, say which is missing.")
AUDIO_Q = "You hear a tone that changes once. Is it higher in pitch at the start or at the end? Reply: higher=<start|end>. If you received no audio, say NO_AUDIO."
TEXT_Q = "Reply with exactly the word: pong"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--out", default="runs/_probe")
    args = ap.parse_args()
    cfg = load_config(args.config)
    load_env_file(cfg.get("env_file", ".env"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    m = Media(cfg.get("media"))
    clip = m.synth(out / "clip.mp4", segments=[(0, 3, "red", 440), (3, 6, "blue", 1200)])
    mp3 = m.cut(clip, out / "clip.mp3", 0, 6, mode="audio")
    defaults = cfg["models"]["defaults"]
    models = sorted({v["model"] for k, v in cfg["models"].items() if k != "defaults"})
    media_models = sorted({cfg["models"][r]["model"] for r in ("captioner", "clue_visual", "clue_audio", "gate_av")})
    jobs = [(mod, "text", {}, [{"type": "text", "text": TEXT_Q}]) for mod in models]
    for mod in media_models:
        for fmt in ("video_url", "file", "image_url"):
            jobs.append((mod, f"video:{fmt}", {"video_format": fmt}, [{"type": "text", "text": VIDEO_Q}, {"type": "video", "path": str(clip)}]))
        for fmt in ("input_audio", "file"):
            jobs.append((mod, f"audio:{fmt}", {"audio_format": fmt}, [{"type": "text", "text": AUDIO_Q}, {"type": "audio", "path": str(mp3)}]))

    def one(job):
        mod, what, extra, parts = job
        c = Client("probe", {**defaults, "model": mod, "max_retries": 0, "timeout_s": 120, "max_tokens": 1000, **extra}, Cache(out / "cache", enabled=False))
        try:
            return {"model": mod, "what": what, "ok": True, "answer": c.call(parts, json=False)["text"].strip()[:160]}
        except Exception as exc:  # noqa: BLE001
            return {"model": mod, "what": what, "ok": False, "error": str(exc)[:160]}

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(one, jobs))
    write_json(out / "probe.json", results)
    for r in results:
        print(f"{r['model'][:26]:26} {r['what']:18} {'OK ' if r['ok'] else 'ERR'} {json.dumps(r.get('answer', r.get('error')), ensure_ascii=False)}")


if __name__ == "__main__":
    main()
