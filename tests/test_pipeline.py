"""End-to-end and unit tests. Offline: mock models, ffmpeg-synthesized media.

    /data1/tanjiaqi/miniconda3/envs/lmms-eval/bin/python -m unittest discover -s tests -t .
"""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
import unittest.mock

from omnifoundry.analysis.funnel import funnel
from omnifoundry.checks import CHECKS
from omnifoundry.checks.grounding import quote_matches
from omnifoundry.cli import main
from omnifoundry.config import deep_merge, load_config, load_env_file
from omnifoundry.jsonio import read_jsonl
from omnifoundry.media import Media
from omnifoundry.schema import parse_range
from omnifoundry.stages.common import captions_text
from omnifoundry.stages.generate_rov import shuffle_options

ROOT = Path(__file__).resolve().parents[1]


def bundle():
    return {"id": "cap", "video_id": "v", "video_summary": "", "video_type": "", "extra": {},
            "events": [{"event_id": "e01", "start_s": 0, "end_s": 20}],
            "audio_captions": [{"ref": "e01.a1", "event_id": "e01", "start_s": 0, "end_s": 5, "timestamp": "00:00-00:05",
                                "audio_type": "speech", "caption": "The host says welcome to the workshop today"},
                               {"ref": "e01.a2", "event_id": "e01", "start_s": 6, "end_s": 9, "timestamp": "00:06-00:09",
                                "audio_type": "background", "caption": "Soft ambient hum in the room"}],
            "visual_captions": [{"ref": "e01.v1", "event_id": "e01", "start_s": 0, "end_s": 5, "timestamp": "00:00-00:05",
                                 "caption": "A man in a teal jacket stands at a bench"}]}


def candidate(**over):
    c = {"id": "q1", "video_id": "v", "route": "rov", "task_type": "attribute", "question": "What is shown?", "answer": "A",
         "options": [f"{chr(65 + i)}. option {i}" for i in range(8)],
         "clues": [{"id": "c1", "modality": "speech", "timestamp": "00:00-00:05", "start_s": 0, "end_s": 5,
                    "quote": "welcome to the workshop", "fact": "", "caption_ref": "e01.a1"},
                   {"id": "c2", "modality": "visual", "timestamp": "00:00-00:05", "start_s": 0, "end_s": 5,
                    "quote": "A man in a teal jacket", "fact": "", "caption_ref": "e01.v1"}],
         "steps": [{"step_id": "s1", "modality": "audio", "depends_on": [], "clue_ids": ["c1"]},
                   {"step_id": "s2", "modality": "visual", "depends_on": ["s1"], "clue_ids": ["c2"]}]}
    return {**c, **over}


class Ctx:
    """Minimal context for deterministic checks."""
    config = {}

    def get(self, dotted, default=None):
        return default


class TestDeterministicChecks(unittest.TestCase):
    def run_check(self, name, cand):
        return CHECKS[name]().run(Ctx(), cand, bundle())

    def test_good_candidate_passes(self):
        self.assertTrue(self.run_check("structure", candidate())["passed"])
        self.assertTrue(self.run_check("grounding", candidate())["passed"])

    def test_structure_failures(self):
        self.assertEqual(self.run_check("structure", candidate(options=["A. x"]))["reason"], "options_invalid")
        self.assertEqual(self.run_check("structure", candidate(answer="Z"))["reason"], "answer_invalid")
        bad = candidate()
        bad["steps"] = [{"step_id": "s1", "modality": "audio", "depends_on": ["s9"], "clue_ids": []},
                        {"step_id": "s2", "modality": "visual", "depends_on": [], "clue_ids": []}]
        self.assertEqual(self.run_check("structure", bad)["reason"], "dag_invalid")
        visual_only = candidate()
        visual_only["clues"] = [visual_only["clues"][1]]
        self.assertEqual(self.run_check("structure", visual_only)["reason"], "missing_modality")

    def test_dangling_step_rejected(self):
        c = candidate()
        c["steps"] = [{"step_id": "s1", "modality": "audio", "depends_on": [], "clue_ids": []},
                      {"step_id": "s2", "modality": "visual", "depends_on": [], "clue_ids": []}]
        self.assertEqual(self.run_check("structure", c)["reason"], "dag_invalid")

    def test_grounding_failures(self):
        c = candidate()
        c["clues"][0] = {**c["clues"][0], "quote": "completely unrelated statement about planets orbiting"}
        self.assertEqual(self.run_check("grounding", c)["reason"], "quote_not_in_captions")
        c = candidate()
        c["clues"][0] = {**c["clues"][0], "caption_ref": "e01.v1"}
        self.assertEqual(self.run_check("grounding", c)["reason"], "quote_not_in_captions")
        c = candidate()
        c["clues"][0] = {**c["clues"][0], "caption_ref": "e01.a2", "quote": "Soft ambient hum", "start_s": 6, "end_s": 9}
        self.assertEqual(self.run_check("grounding", c)["reason"], "music_anchor")
        c = candidate()
        c["clues"][0] = {**c["clues"][0], "start_s": 100, "end_s": 105, "timestamp": "01:40-01:45"}
        self.assertEqual(self.run_check("grounding", c)["reason"], "timestamp_not_in_captions")

    def test_simultaneity(self):
        c = candidate(question="What is shown when the host speaks?")
        c["clues"][1] = {**c["clues"][1], "start_s": 0, "end_s": 5}
        self.assertTrue(self.run_check("grounding", c)["passed"])
        far = bundle()
        far["visual_captions"][0].update(start_s=60, end_s=65)
        c["clues"][1] = {**c["clues"][1], "start_s": 60, "end_s": 65}
        self.assertEqual(CHECKS["grounding"]().run(Ctx(), c, far)["reason"], "simultaneity_gap")

    def test_option_balance(self):
        long_gold = candidate(options=["A. a much longer and very specific correct option text"] + [f"{chr(66 + i)}. short {i}" for i in range(7)])
        self.assertEqual(self.run_check("option_balance", long_gold)["reason"], "answer_length_leak")
        self.assertTrue(self.run_check("option_balance", candidate())["passed"])

    def test_quote_matching(self):
        self.assertTrue(quote_matches("Welcome to the WORKSHOP!", "the host says welcome to the workshop today", .6, 4))
        self.assertFalse(quote_matches("rain", "heavy storm", .6, 4))


class TestGateModes(unittest.TestCase):
    def evaluate(self, audio, visual, mode, full=3):
        from omnifoundry.checks.gate import _evaluate
        res = {"full": ["A"] * full + ["B"] * (3 - full), "audio_only": ["A"] * audio + ["B"] * (3 - audio),
               "visual_only": ["A"] * visual + ["B"] * (3 - visual)}
        return _evaluate({"answer": "A"}, res, 1, mode)

    def test_modes(self):
        self.assertEqual(self.evaluate(0, 0, "any"), (None, self.evaluate(0, 0, "any")[1]))
        self.assertEqual(self.evaluate(2, 0, "any")[0], "audio_only_shortcut")
        self.assertIsNone(self.evaluate(2, 0, "both")[0])
        self.assertEqual(self.evaluate(2, 3, "both")[0], "both_modalities_shortcut")
        self.assertIsNone(self.evaluate(3, 3, "none")[0])
        self.assertEqual(self.evaluate(0, 0, "none", full=1)[0], "full_wrong")   # correctness always filters

    def test_tier_and_severity(self):
        from omnifoundry.quality import corroborating_modality, severity, tier
        self.assertEqual(severity(Ctx(), "gate_text", "visual_only_shortcut"), "soft")
        self.assertEqual(severity(Ctx(), "gate_media", "av_wrong"), "hard")
        self.assertEqual(severity(Ctx(), "clue", "clue_unsupported"), "hard")
        self.assertEqual(tier([], ["av_strict"]), "high")
        self.assertEqual(tier(["visual_only_shortcut"], ["visual_sufficient"]), "medium")
        self.assertEqual(tier([], ["visual_sufficient"]), "medium")
        self.assertEqual(tier(["blind_correct"], ["av_strict"]), "low")
        self.assertEqual(tier(["both_modalities_shortcut"], []), "low")
        self.assertEqual(corroborating_modality(["visual_sufficient"]), "audio")
        self.assertEqual(corroborating_modality(["audio_sufficient"]), "visual")

    def test_tags(self):
        from omnifoundry.checks.gate import modality_tags
        for (a, v), tags in {(0, 0): ["av_strict"], (2, 0): ["audio_sufficient"], (0, 3): ["visual_sufficient"]}.items():
            _, detail = self.evaluate(a, v, "none")
            self.assertEqual(modality_tags([{"check": "gate_text", "detail": detail}])["tags"], tags)
        _, text = self.evaluate(0, 0, "none")
        _, media = self.evaluate(3, 0, "none")
        out = modality_tags([{"check": "gate_text", "detail": text}, {"check": "gate_media", "detail": media}])
        self.assertEqual((out["tags"], out["modality_dependency"]["evidence"]), (["audio_sufficient"], "gate_media"))


class TestHelpers(unittest.TestCase):
    def test_complement_removes_cited_entries(self):
        text = captions_text(bundle(), exclude_refs=["e01.a1", "e01.v1"], with_refs=False)
        self.assertNotIn("welcome", text)
        self.assertNotIn("teal", text)
        self.assertIn("ambient", text)

    def test_shuffle_keeps_gold_body(self):
        opts = [f"{chr(65 + i)}. body {i}" for i in range(8)]
        for seed in range(20):
            new, ans = shuffle_options(opts, "C", f"s{seed}")
            self.assertTrue(next(o for o in new if o.startswith(ans + ".")).endswith("body 2"))
            self.assertEqual([o[0] for o in new], list("ABCDEFGH"))

    def test_parse_range(self):
        self.assertEqual(parse_range("01:05-01:10"), (65.0, 70.0))
        self.assertEqual(parse_range("1:00:00-1:00:05"), (3600.0, 3605.0))
        with self.assertRaises(ValueError):
            parse_range("nonsense")

    def test_env_file_accepts_export_prefix(self):
        with tempfile.TemporaryDirectory() as d, unittest.mock.patch.dict(os.environ, clear=False):
            os.environ.pop("OF_TEST_KEY", None)
            os.environ.pop("OF_TEST_PLAIN", None)
            path = Path(d) / ".env"
            path.write_text("# comment\nexport OF_TEST_KEY='abc'\nOF_TEST_PLAIN=def\n")
            load_env_file(path)
            self.assertEqual(os.environ["OF_TEST_KEY"], "abc")
            self.assertEqual(os.environ["OF_TEST_PLAIN"], "def")

    def test_deep_merge(self):
        self.assertEqual(deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 3}}), {"a": {"x": 1, "y": 3}})


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="omnifoundry-test-"))
        media = Media()
        rows = []
        for i in range(2):
            path = media.synth(cls.tmp / f"v{i}.mp4")
            rows.append(json.dumps({"id": f"v{i}", "path": str(path), "source": "synth"}))
        (cls.tmp / "manifest.jsonl").write_text("\n".join(rows) + "\n")
        cfg = load_config(ROOT / "configs" / "mock.yaml")
        self_cfg = deep_merge(cfg, {
            "cache": {"dir": str(cls.tmp / "cache")},
            "media": {"work_dir": str(cls.tmp / "media")},
            "select": {"path": str(cls.tmp / "manifest.jsonl"), "stratify_by": "source"},
            "verify": {"checks": ["clue", "gate_text", "gate_media"]},
            "checks": {"clue": {"modalities": ["visual", "speech", "sound", "music"]},
                       "gate_text": {"conditions": ["full", "audio_only", "visual_only", "without_clues"]}},
        })
        import yaml
        cls.config = cls.tmp / "config.yaml"
        cls.config.write_text(yaml.safe_dump(self_cfg))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_cli(self, name, *extra):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(["run", "--config", str(self.config), "--out", str(self.tmp / name), *extra])
        return buf.getvalue()

    def test_full_pipeline_then_caching(self):
        fresh = ["--set", f"cache.dir={self.tmp / 'fresh_cache'}"]   # independent of other tests' cache
        first = self.run_cli("run1", *fresh)
        run = self.tmp / "run1"
        for st in ("select", "caption", "generate", "verify", "think"):
            self.assertEqual(read_jsonl(run / st / "errors.jsonl"), [], st)
        sft = read_jsonl(run / "export" / "sft.jsonl")
        rl = read_jsonl(run / "export" / "rl.jsonl")
        self.assertEqual(len(sft), 2, first)
        self.assertEqual(len(rl), 2)

        # ChatML shape and answer consistency.
        reply = sft[0]["messages"][1]["content"]
        self.assertTrue(reply.startswith("<think>\nGot it!"))
        self.assertRegex(reply, r"</think>\n\n[A-H]$")
        self.assertEqual(reply[-1], reply.split("points to option ")[1][0])
        self.assertTrue(rl[0]["audio_clues"] and rl[0]["visual_clues"])
        self.assertIn("gate_media", rl[0]["signals"])
        self.assertEqual(rl[0]["tags"], ["av_strict"])
        self.assertEqual(sft[0]["modality_dependency"]["evidence"], "gate_media")

        # The first generation attempt fails grounding and is regenerated with feedback.
        rejected = read_jsonl(run / "generate" / "rejected.jsonl")
        self.assertEqual({r["reason"] for r in rejected}, {"quote_not_in_captions"})
        accepted = read_jsonl(run / "generate" / "out.jsonl")
        self.assertTrue(all(c["attempt"] == 1 and c["feedback"] for c in accepted))
        self.assertTrue(all(c["lineage"]["prompts"]["generate"] for c in accepted))

        # Funnel records every check with its reasons.
        f = funnel(run)
        self.assertEqual(f["stages"]["think"]["out"], 2)
        self.assertEqual(f["checks"]["grounding"]["quote_not_in_captions"], 2)
        self.assertIn("generator", f["cost"])

        # Rerun: everything is a stage-cache hit, no model call at all.
        second = self.run_cli("run2", *fresh)
        for st, n in (("select", 2), ("caption", 2), ("generate", 2), ("verify", 2), ("think", 2)):
            self.assertIn(f"[{st}] items={n} cache_hits={n}", second)
        self.assertFalse((self.tmp / "run2" / "calls.jsonl").exists())
        self.assertEqual(read_jsonl(self.tmp / "run2" / "export" / "sft.jsonl"), sft)

    def test_code_change_keeps_stage_cache(self):
        self.run_cli("code_a")
        from omnifoundry import lineage
        with unittest.mock.patch.object(lineage, "code_version", lambda: "different-code"):
            out = self.run_cli("code_b")
        for st in ("caption", "generate", "verify", "think"):
            self.assertIn(f"[{st}] items=2 cache_hits=2", out)

    def test_prompt_change_recomputes_only_downstream(self):
        self.run_cli("base")
        changed = self.tmp / "think_variant.md"
        changed.write_text((ROOT / "prompts" / "think.md").read_text() + "\nBe concise.\n")
        out = self.run_cli("variant", "--set", f"prompts.overrides={{think: {changed}}}")
        self.assertIn("[caption] items=2 cache_hits=2", out)
        self.assertIn("[generate] items=2 cache_hits=2", out)
        self.assertIn("[verify] items=2 cache_hits=2", out)
        self.assertIn("[think] items=2 cache_hits=0", out)
        calls = read_jsonl(self.tmp / "variant" / "calls.jsonl")
        self.assertEqual({c["role"] for c in calls}, {"thinker"})
        self.assertTrue(all(not c["cached"] for c in calls))

    def test_verify_check_config_does_not_invalidate_generate(self):
        self.run_cli("base2")
        out = self.run_cli("variant2", "--set", "checks.clue.min_strength=0.5")
        self.assertIn("[generate] items=2 cache_hits=2", out)
        self.assertIn("[verify] items=2 cache_hits=0", out)
        # A threshold change alters verify's lineage, so think re-runs, but it sends identical prompts:
        # every model call in this run is served from the LLM cache, at no cost.
        calls = read_jsonl(self.tmp / "variant2" / "calls.jsonl")
        self.assertTrue(calls and all(c["cached"] for c in calls))

    def test_soft_failure_kept_at_lower_tier(self):
        out = self.run_cli("soft", "--set", "models.gate_text.visual_leak=true", "--set", "verify.checks=[clue]",
                           "--set", "generate.checks=[structure, grounding, solver, blind, gate_text]")
        run = self.tmp / "soft"
        acc = read_jsonl(run / "generate" / "out.jsonl")
        self.assertEqual(len(acc), 2, out)
        for c in acc:
            self.assertEqual((c["quality"], c["tags"], c["soft_failures"], c["corroborating_modality"]),
                             ("medium", ["visual_sufficient"], ["visual_only_shortcut"], "audio"))
        # retried for a better question first, and the thinking is written as corroboration
        self.assertTrue([r for r in read_jsonl(run / "generate" / "rejected.jsonl") if r.get("soft")])
        thought = read_jsonl(run / "think" / "out.jsonl")
        self.assertTrue(all(t["think_style"] == "think_corroborate" for t in thought))
        sft = read_jsonl(run / "export" / "sft.jsonl")
        self.assertEqual([r["quality"] for r in sft], ["medium", "medium"])
        self.assertEqual(funnel(run)["final_quality"], {"medium": 2})

    def test_hard_failure_still_discards(self):
        out = self.run_cli("hard", "--set", "checks.clue.min_strength=0.95")
        self.assertIn("[verify] items=2", out)
        self.assertEqual(read_jsonl(self.tmp / "hard" / "export" / "sft.jsonl"), [])

    def test_clue_failure_rejects(self):
        out = self.run_cli("strict", "--set", "checks.clue.min_strength=0.95")
        self.assertIn("[verify] items=2", out)
        rej = read_jsonl(self.tmp / "strict" / "verify" / "rejected.jsonl")
        self.assertEqual({r["reason"] for r in rej}, {"clue_unsupported"})
        self.assertEqual(read_jsonl(self.tmp / "strict" / "export" / "sft.jsonl"), [])

    def test_overlay_and_cut_modes(self):
        m = Media()
        src = self.tmp / "v0.mp4"
        over = m.overlay_transcode(src, self.tmp / "over.mp4", fps=2, width=320)
        info = m.probe(over)
        self.assertTrue(info["has_audio"] and info["has_video"])
        self.assertAlmostEqual(info["duration_s"], 20, delta=1)
        muted = m.probe(m.cut(src, self.tmp / "m.mp4", 2, 8, mode="video"))
        self.assertFalse(muted["has_audio"])
        audio = m.probe(m.cut(src, self.tmp / "a.mp3", 2, 8, mode="audio"))
        self.assertFalse(audio["has_video"])
        self.assertTrue(audio["has_audio"])


if __name__ == "__main__":
    unittest.main()


class TestReviewServer(unittest.TestCase):
    def test_serve_save_and_range(self):
        import threading
        import urllib.request
        from omnifoundry.analysis import review_server as rs
        from omnifoundry.jsonio import write_jsonl
        tmp = Path(tempfile.mkdtemp(prefix="of-review-"))
        try:
            video = Media().synth(tmp / "v.mp4", duration=4, segments=[(0, 4, "red", 440)])
            write_jsonl(tmp / "caption" / "out.jsonl", [{**bundle(), "extra": {"video_path": str(video), "duration_s": 4}}])
            write_jsonl(tmp / "think" / "out.jsonl", [{**candidate(), "tags": ["av_strict"], "thinking": "Got it!"}])
            items = rs.load_items(tmp, "think")
            server = rs.ThreadingHTTPServer(("127.0.0.1", 0), rs.make_handler(tmp, items, tmp / "reviews.jsonl"))
            threading.Thread(target=server.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            data = json.loads(opener.open(base + "/api/items").read())
            self.assertEqual(data["items"][0]["id"], "q1")
            self.assertNotIn("video_path", data["items"][0])
            req = urllib.request.Request(base + "/video/q1", headers={"Range": "bytes=0-99"})
            resp = opener.open(req)
            self.assertEqual((resp.status, len(resp.read())), (206, 100))
            with self.assertRaises(urllib.error.HTTPError):
                opener.open(base + "/video/..%2F..%2Fetc%2Fpasswd")
            body = json.dumps({"id": "q1", "human_answer": "A", "verdict": "correct", "needs": "both", "issues": [], "note": "ok"}).encode()
            resp = json.loads(opener.open(urllib.request.Request(base + "/api/review", data=body, method="POST",
                                                                 headers={"Content-Type": "application/json"})).read())
            self.assertEqual(resp["summary"]["blind_correct"], 1)
            self.assertEqual(read_jsonl(tmp / "reviews.jsonl")[0]["verdict"], "correct")
            self.assertEqual(funnel(tmp)["human_review"]["av_strict"]["correct"], 1)
            bad = json.dumps({"id": "q1", "verdict": "great"}).encode()
            with self.assertRaises(urllib.error.HTTPError):
                opener.open(urllib.request.Request(base + "/api/review", data=bad, method="POST"))
            server.shutdown()
            server.server_close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
