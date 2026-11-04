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
$PY -m omnifoundry review runs/m1-001                                   # review.html for manual spot checks
```

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

## Known gateway behaviour

Probed 2026-11-04 with one synthetic clip per format. `video_format: video_url` is accepted, but the media is dropped and the model guesses, with no error. `file` and `image_url` answered correctly, and `base.yaml` uses `file`. Re-run `scripts/probe_gateway.py` when changing models or gateways.
