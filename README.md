# OmniFoundry v3

Generates audio-visual joint-reasoning QA data for training omni models: hard multiple-choice questions that need both sound and picture, each with a timestamped evidence chain and synthesized thinking.

The method follows OmniQA from [OmniReasoning](reference/OmniReasoning) (arXiv 2609.39490). Earlier attempts are in `archive/`.

## Pipeline

```
select    OmniVideo-100K -> Video records (sampler policy)
caption   timestamp-overlay transcode -> event plan -> per-event Gemini captions (audio / visual, with refs)
generate  task selector -> MCQ + clues + reasoning DAG -> check chain -> regenerate with feedback
verify    clue validity on real clips -> modality-shortcut gates
think     captions + verified chain -> first-person thinking -> deterministic QC
export    sft.jsonl (ChatML), rl.jsonl (question + audio/visual clues + difficulty signals), funnel.json
```

## Running

```bash
PY=/data1/tanjiaqi/miniconda3/envs/lmms-eval/bin/python
NP="env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy"   # gateway needs a direct connection

$PY -m unittest discover -s tests -t .                                  # offline: mock models, synthetic media
$NP $PY scripts/probe_gateway.py                                        # which models / media formats really work
$NP $PY -m omnifoundry run --config configs/base.yaml --out runs/m1-001 # add --stages caption,generate to run a subset
$PY -m omnifoundry funnel runs/m1-001                                   # where candidates were lost, and cost
$PY -m omnifoundry compare runs/a runs/b                                # A/B any change
$PY -m omnifoundry review-serve runs/m1-001                             # interactive human review (see below)
```

### Human review

`review-serve` starts a local page on 127.0.0.1:8765. From your own computer, run `ssh -L 8765:127.0.0.1:8765 <server>` and open http://127.0.0.1:8765.

For each question:
1. Watch the video and answer it yourself first. The gold answer stays hidden until you pick an option, which measures human accuracy without anchoring.
2. The gold answer, the clue chain (click a timestamp to seek), the reasoning steps, the automatic tag evidence and the thinking are then shown.
3. Record a verdict, whether the question really needs both sound and picture, and any issues.

Judgments go to `<run>/reviews.jsonl`, with the previous version kept as `.bak`. `omnifoundry funnel` then reports human accuracy per tag. Shortcuts: A–H answer, Space play/pause, ←/→ 2 s, N/P next/previous, R reveal.

Override any setting with `--set dotted.key=value`, or write `configs/exp-x.yaml` with `extends: base.yaml`.

## Layout

| Path | Contents |
| --- | --- |
| `omnifoundry/stages/` | One module per stage, using the contract in `stages/common.py` |
| `omnifoundry/checks/` | Pass/fail units sharing `Check.run(ctx, cand, bundle) -> Verdict` |
| `omnifoundry/policies/` | Replaceable decisions: sampler, task selector, retry, model router |
| `omnifoundry/analysis/` | Funnel, run comparison, review page |
| `prompts/`, `catalog/`, `configs/` | Prompts, task types, settings: data rather than code |
| `runs/<name>/` | `run.json`, `<stage>/{out,rejected,verdicts,errors}.jsonl`, `calls.jsonl`, `funnel.json` |
| `cache/` | Model responses, stage results and derived media, shared by all runs |
| `archive/`, `reference/` | Earlier versions; OmniReasoning source |

## Making changes

Most changes need no pipeline code:

| To change | Do this |
| --- | --- |
| A prompt | Edit `prompts/x.md`, or set `prompts.overrides: {x: path}` for a variant |
| Task types or their mix | Edit `catalog/tasks.yaml`; set `tasks.weights`, `catalog.enabled` |
| A model for a role | Set `models.<role>` |
| Which checks run, and in what order | Set `generate.checks`, `verify.checks`, `checks.<name>.*` |
| Add a check | Subclass `Check` in `checks/`, add `@register`, list it in a config |
| Change how tasks, videos or retries are chosen | Add a function to the dict in `policies/`, select it in `policies.*` |

**Recomputation.** A stage's cache key covers its input records, its config sections, its prompt hashes and its model ids. Changing a prompt therefore recomputes that stage and the ones downstream, and nothing upstream. Model calls are cached separately by exact request, so a re-run that sends identical requests costs nothing. Code changes do not invalidate caches: bump the stage's `VERSION` after a change that alters its output.

**Signals for later experiments.** Every verdict keeps its individual model draws, and every attempt, rejection reason and retry is on disk. Per-task failure rates and per-question difficulty (for example how often restricted conditions answered correctly) can be computed from existing runs. A weakness-driven selector plugs into `policies/selector.py`; hard-case iteration plugs into `policies/retry.py`, whose decision can already switch task type or generator role.

## Differences from OmniQA

Same stages and approach. Deliberate differences, each switchable from config for A/B runs:

| OmniQA, as read in its code | Here |
| --- | --- |
| Audio clues are skipped by verify and treated as true by refine | `checks.clue.modalities` can include audio; audio clips go to `clue_audio` |
| The complement test depends on `segment_id`, which is never generated, so nothing is removed | `gate_text` condition `without_clues` removes the cited caption entries |
| Modality gates run on caption text | `gate_media` also tests muted video and audio-only input |
| An unparseable solver reply passes; a disagreeing solver silently changes the answer | Both reject, with a reason |
| Resume keys use the input row number | Keys are content hashes |
| Each event sends the whole video | `caption.event_input: clip` sends the event's span; the burned-in clock stays global |

Not implemented yet: the RBY (reasoning beyond video) route, the `refine` repair step, and open-ended questions.

## M1 experiment log (20 then 50 OmniVideo-100K videos, gemini-3.8-flash unless noted)

| Run | Change (config) | Final / attempts | Lesson |
| --- | --- | --- | --- |
| m1-001 | OmniQA baseline (`m1.yaml`) | 0 / 43 | Most questions had a single-modality shortcut |
| m1b-001 | anchor rules in prompt; text gate inside generate; full-video media gate (`m1b.yaml`) | 1 / 73 | Text-gate pass 0% to 29%; caption errors now dominate |
| m1c-001 | captions by gemini-3.1-pro; clue check inside generate (`m1c.yaml`) | 4 / 103 | Clue validity 33% to 89% |
| m1d-001 | 50 videos; prompt rule 9 + option_balance (`m1d.yaml`) | 3 / 300 | "True details as distractors" made answers ambiguous (solver 100% to 80%) |
| m1e-001 | correctness filters; shortcut tagging, reject only if both modalities suffice (`m1e.yaml`) | 12 / 208 | About 0.74M tokens per final question; 2 of 12 are `av_strict` |
| m1f-001 | prompt rule 10 (audio cue must not name on-screen words); visual shortcut triggers a soft retry (`m1f.yaml`) | 12 / 272 | No change in the tag mix: audio and visuals are redundant in the source videos (alphabet songs, captioned explainers), so selection matters more than prompt wording |

Human review of m1e-001 (12/12): every answer correct, but 5 questions were answerable with the sound off,
because the audio cue named a letter, word or name also shown on screen.

Data tags (in `sft.jsonl` / `rl.jsonl`): `av_strict` (neither audio alone nor visuals alone answers: the
high-quality subset), `audio_sufficient`, `visual_sufficient`, with the per-condition counts in `modality_dependency`.

## Known gateway behaviour

Probed 2026-11-04 with one synthetic clip per format. `video_format: video_url` is accepted, but the media is dropped and the model guesses, with no error. `file` and `image_url` answered correctly, and `base.yaml` uses `file`. Re-run `scripts/probe_gateway.py` when changing models or gateways.

At any concurrency a few requests hang without answering, and 503 outages lasting tens of minutes occur, so timeouts are short (180 s, 400 to 600 s for long-output or full-video roles) with 5 retries and exponential backoff. A run interrupted by an outage is recovered by re-running the same command: finished items are cache hits.
