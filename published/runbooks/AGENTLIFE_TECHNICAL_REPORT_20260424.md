# AgentLife Technical Report — 2026-04-24

Status: draft for internal use. This report supersedes the pending April 18 draft as
our current headline benchmark snapshot.

## Summary

This report captures the current headline AgentLife block after the April 24
re-run and repair cycle.

The short version:

- Quaid's best current `AL-S` row is `93.64%` under Sonnet re-evaluation.
- Quaid's best current plain `AL-L` row is `88.52%` under Sonnet re-evaluation.
- Quaid's repaired `AL-L OBD` row is `88.69%` under Sonnet re-evaluation.
- Current FC anchors are:
  - `AL-S FC Sonnet`: `93.11%`
  - `AL-S FC Opus 4.7`: `92.76%`
  - `AL-L FC Sonnet`: `88.69%`
  - `AL-L FC Opus 4.7`: `89.40%`
- The major methodology lesson from this cycle is not just "better prompts" or
  "better recall". The bigger correction was benchmark validity:
  - the older pending draft correctly identified that harness-seeded project
    docs and future-state/stub-style doc surfaces were distorting eval
  - the follow-on temporal and OBD work showed that date-bounded docs recall is
    only trustworthy when source timestamps, `PROJECT.log` chronology, and docs
    indexing are all preserved end-to-end
- The multilingual-first Japanese run remains important, but it should still be
  treated as a preview rather than a headline claim. It proved non-Roman-script
  operation is viable, but it has not been scrutinized to the same level as the
  English headline block.

## Headline Block

Model setup for the current headline rows:

- ingest/deep runtime: `claude-sonnet-4-6`
- fast runtime: `claude-haiku-4-5-20251001`
- Sonnet re-eval rows: `claude-sonnet-4-6`
- judge: `gpt-4o-mini`
- embedding: `nomic-embed-text`

### Quaid and FC Headline Rows

| Surface | Run | Setup | Accuracy | Counts |
| --- | --- | --- | ---: | --- |
| `AL-S` full | `r1423` | `Sonnet/Haiku` | `89.75%` | `248C / 12P / 23W` |
| `AL-S` re-eval | `r1433 (r1423)` | `Sonnet/Sonnet` | `93.64%` | `261C / 8P / 14W` |
| `AL-S FC` | `r1434` | `Sonnet` | `93.11%` | `261C / 5P / 17W` |
| `AL-S FC` | `r1454 (r1427)` | `Opus 4.7` | `92.76%` | `262C / 1P / 20W` |
| `AL-L` full | `r1439` | `Sonnet/Haiku` | `84.63%` | `234C / 11P / 38W` |
| `AL-L` re-eval | `r1441 (r1439)` | `Sonnet/Sonnet` | `88.52%` | `248C / 5P / 30W` |
| `AL-L FC` | `r1455 (r1442)` | `Sonnet` | `88.69%` | `249C / 4P / 30W` |
| `AL-L FC` | `r1457` | `Opus 4.7` | `89.40%` | `251C / 4P / 28W` |
| `AL-L OBD` full | `r1451` | `Sonnet/Haiku` | `84.63%` | `233C / 13P / 37W` |
| `AL-L OBD` re-eval | `r1453 (r1451)` | `Sonnet/Sonnet` | `88.69%` | `248C / 6P / 29W` |

### Quick Read

- `AL-S` remains the strongest Quaid lane at `93.64%` under Sonnet re-eval.
- Plain `AL-L` and repaired `AL-L OBD` are now effectively tied under Sonnet
  re-eval (`88.52%` vs `88.69%`).
- On FC anchors:
  - `AL-S`: Sonnet is slightly ahead of Opus (`93.11%` vs `92.76%`).
  - `AL-L`: Opus is slightly ahead of Sonnet (`89.40%` vs `88.69%`).
- The repaired `AL-L OBD` row matters because it is no longer a tainted source.
  The earlier broken OBD line is not part of the headline set.

## Token Accounting And Preinject Methodology

This report changes how headline token usage is presented.

### Quaid Rows

For Quaid full/eval rows, the reported eval token number is:

```text
non_judge_eval_tokens = answer_model + tool_recall + preinject_recall
```

This excludes:

- judge tokens

This includes:

- answer-model tokens
- explicit recall/tool-call tokens
- preinject recall tokens

For Quaid rows we report:

- total non-judge eval tokens
- input-side non-judge eval tokens
- cache percentage on total tokens
- cache percentage on input tokens
- average preinject time

The preinject metric is important. Another platform can look cheaper or more
accurate if it hides a large unbounded preinject phase outside the visible eval
surface. Reporting average preinject time keeps that tradeoff visible.

### FC Rows

For FC rows, the tiny summary token file is not a fair headline metric. FC runs
load the whole transcript repeatedly, and in production that history would be
uncached message history.

So the public FC token number in this report is:

```text
full_fc_prompt_footprint = sum(raw llm-usage-events, excluding judge)
```

This includes cached and uncached history from the raw usage events so the
headline token figure stays true to prod behavior rather than the benchmark's
cost-saving cache shortcut.

Important caveat:

- FC Sonnet-vs-Opus token totals should be read as provider-reported billed
  model-token footprints, not as proof that different benchmark input corpora
  were sent.
- The harness uses the same FC transcript/context construction path for both
  models; only `answer_model` changes.
- We directly validated this with a same-prompt control probe through the same
  Anthropic call path: an identical cached prompt came back as `12,209` input
  tokens on `claude-sonnet-4-6` and `29,059` input tokens on
  `claude-opus-4-7`.
- The real FC benchmark rows show the same pattern with a stable ratio across
  lanes, which strongly suggests model-specific Anthropic token
  accounting/tokenization rather than different benchmark inputs.

Implication:

- FC token totals remain useful as true billed/model-footprint numbers.
- They are not a clean apples-to-apples measure of identical prompt size across
  Sonnet and Opus.
- Do not infer that Opus consumed a larger transcript set just because its
  provider-reported token count is higher.

### Headline Token And Preinject Table

| Run | Surface | Setup | Acc | Tokens | Input Tokens | Cache % Total | Cache % Input | Avg Preinject |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `r1423` | `AL-S` full | `Sonnet/Haiku` | `89.75%` | `6.96M` | `6.83M` | `80.1%` | `81.7%` | `1.03s` |
| `r1433` | `AL-S` re-eval | `Sonnet/Sonnet` | `93.64%` | `7.95M` | `7.79M` | `79.3%` | `80.9%` | `1.02s` |
| `r1434` | `AL-S FC` | `Sonnet` | `93.11%` | `29.83M` | `29.80M` | `99.5%` | `99.6%` | n/a |
| `r1454` | `AL-S FC` | `Opus 4.7` | `92.76%` | `40.77M` | `40.72M` | `99.5%` | `99.6%` | n/a |
| `r1439` | `AL-L` full | `Sonnet/Haiku` | `84.63%` | `8.98M` | `8.83M` | `81.9%` | `83.4%` | `1.08s` |
| `r1441` | `AL-L` re-eval | `Sonnet/Sonnet` | `88.52%` | `9.64M` | `9.46M` | `81.0%` | `82.6%` | `0.97s` |
| `r1455` | `AL-L FC` | `Sonnet` | `88.69%` | `26.50M` | `26.46M` | `98.5%` | `98.6%` | n/a |
| `r1457` | `AL-L FC` | `Opus 4.7` | `89.40%` | `36.22M` | `36.18M` | `98.5%` | `98.6%` | n/a |
| `r1451` | `AL-L OBD` full | `Sonnet/Haiku` | `84.63%` | `7.20M` | `7.05M` | `77.5%` | `79.2%` | `1.21s` |
| `r1453` | `AL-L OBD` re-eval | `Sonnet/Sonnet` | `88.69%` | `8.45M` | `8.27M` | `76.1%` | `77.8%` | `1.11s` |

## What We Learned From The Old Draft

The April 18 draft contained the important warning that the benchmark harness
had drifted away from a neutral fresh-production surface.

That warning was correct.

The concrete problem was not that Quaid needed benchmark-only tricks. The
problem was that eval quality and retrieval surfaces were being distorted by
artifacts that should not have been treated as a clean production signal.

Key corrections from that draft that still matter:

- do not seed future-state or benchmark-specific project docs into the workspace
- preserve runtime-managed project surfaces like `PROJECT.md` and `PROJECT.log`
- route project updates through the runtime updater rather than inventing a
  harness-only project-doc reasoning path
- keep benchmark methodology honest when the problem is setup leakage rather
  than true model/runtime behavior

That draft should also be remembered for a second lesson: some apparent recall
problems were really docs-surface problems.

In practice, that showed up in two ways:

- stale or benchmark-shaped doc surfaces could make eval look better or worse
  than real product behavior
- stub/current docs and broad project docs could crowd out the historical,
  date-bounded evidence the query actually needed

This is why the right direction was not "tune recall harder until the score
moves". The right direction was to make date-bounded docs recall actually obey
source chronology and project scope.

## Why Timestamp-Based Docs Search Became A Core Requirement

The most important technical lesson from the temporal work is that date-bounded
project recall is only valid when docs search is timestamp-aware.

The progression was:

1. We proved that `PROJECT.log` and other project history can carry the exact
   evidence needed for historical questions.
2. We then found multiple ways that this evidence could still be missed:
   - appends indexed too late or not at all
   - wall-clock timestamps instead of source/session timestamps
   - broad docs retrieval returning zero dated rows even when dated evidence was
     present on disk
   - stale or older same-day rows occupying the preserved docs slot ahead of the
     exact cutoff-day line
3. The repair path taught us that the docs system needs all of the following:
   - source/session timestamps, not write-time timestamps
   - indexed `PROJECT.log` rows inside docs RAG
   - date-bounded filtering over those rows
   - ranking that respects cutoff dates and same-day query relevance
   - no leakage from current `PROJECT.md` when the query is historical

This is the deeper lesson behind the earlier stub-docs/eval confusion: once the
benchmark asks true historical questions, docs retrieval cannot be a generic
semantic blob. It needs to be able to search by time as well as by meaning.

The repaired OBD line reinforced the same point from another angle. OBD did not
underperform because "OBD is weak" in the abstract. It underperformed when OBD
collapsed project facts and `PROJECT.log` lines onto a synthetic terminal day.
Once chronology was preserved through rolling OBD compaction, `AL-L OBD`
returned to a legitimate headline-quality row.

## Current Interpretation Of The Main Surfaces

### `AL-S`

`AL-S` remains Quaid's strongest current lane.

- `r1433`: `93.64%`
- strong at moderate token cost
- stable preinject profile near `1.0s`

This is the clearest headline row for current Quaid quality.

### Plain `AL-L`

Plain `AL-L` now lands at `88.52%` under Sonnet re-eval.

That is a strong refreshed row, but it should be presented honestly rather than
forced into an overclaim. It is a legitimate large-lane benchmark row, not a
marketing outlier.

### `AL-L OBD`

The repaired `AL-L OBD` line matters because it is now methodologically valid.

- old broken OBD lineage: not headline-usable
- repaired OBD lineage: `84.63%` full / `88.69%` Sonnet re-eval

That makes OBD useful again as a real benchmark surface rather than a tainted
comparison row.

### FC Anchors

The FC rows are now all filled in for the current block.

The most important comparison lesson is that FC token usage is much larger than
it looked when using the tiny summary files. Once full prompt footprint is used,
FC is clearly much more expensive in prompt volume than the comparable Quaid
rows.

There is one more nuance for FC comparisons:

- Sonnet and Opus FC token totals are not directly interchangeable as a
  same-input-size metric.
- In this report they should be interpreted as billed provider-token footprint
  per model.
- Cross-model accuracy comparisons are still valid, but cross-model token
  comparisons should be read with the model-specific Anthropic accounting caveat
  above.

## OpenClaw Neutral Rerun Status

The current Quaid headline block in this report is the new benchmark reference
surface. It is based on the neutralized harness path after the project-doc
stub/future-state cleanup and the timestamp-aware docs/chronology fixes.

That means older OpenClaw-native comparison rows are no longer a fully clean
match against this study, so the OC side needed a neutral rerun on the same
surface.

What changed on the Quaid side:

- benchmark project docs are no longer allowed to preload future-state or
  stub-style operator knowledge
- runtime-managed project surfaces are preserved and updated through the real
  updater path
- historical/date-bounded project recall now depends on timestamp-aware docs
  behavior rather than broad synthetic docs surfaces

What we validated on the OC rerun path:

- the neutral OC-native smoke now passes on the corrected harness path
- direct OpenAI auth for OC-native needs to be provisioned into
  `~/.openclaw/.env`, not only the agent auth-profile store
- guest-visible embeddings need a reachable VM-side endpoint; on this setup the
  harness used a proxy on `192.168.64.1:11435/v1` to expose the host Ollama
  embeddings service to the VM

Current scored-rerun status:

- the fresh neutral OC-native smoke succeeded end-to-end
- the first scored neutral `AL-S` OC-native rerun did not produce valid headline
  numbers because native indexing stalled after injection with the store still
  marked `dirty=true` and `files=0 / chunks=0`
- because the scored rerun did not complete cleanly, no fresh OC headline rows
  are inserted into this report yet

Implication:

- existing OC-native comparison rows remain provisional for now
- this report should still be read as the current Quaid headline study first
- fresh OC comparison rows should only be added once a neutral scored OC rerun
  completes successfully on the corrected harness path

## Multilingual Preview Status

The multilingual-first Japanese work remains worth keeping in the report, but it
should stay clearly labeled as a preview.

What we can say:

- Quaid successfully handled a full Japanese Kanji/Kana benchmark run without
  romaji transliteration.
- This validates that non-Roman-script operation is viable end-to-end.
- The run did not collapse; it exposed a real multilingual retrieval/docs gap
  rather than a basic language-support failure.

What we should not say yet:

- We should not treat the Japanese row as a final headline benchmark claim.
- We should not claim the multilingual planner/docs path is fully scrutinized.
- We should not over-interpret the first improvement passes as a settled win.

So the multilingual section is still a preview of direction, not a final
headline result.

## Publication Guidance

Use these claims:

- Quaid reaches `93.64%` on `AL-S` with Sonnet re-eval.
- Quaid reaches `88.52%` on plain `AL-L` with Sonnet re-eval.
- Quaid reaches `88.69%` on repaired `AL-L OBD` with Sonnet re-eval.
- FC anchors for the same block are now complete for Sonnet and Opus on both
  `AL-S` and `AL-L`.
- Public token totals now exclude judge tokens.
- FC public token totals now use full prompt footprint, not the old tiny
  summary token files.
- Average preinject time is reported alongside Quaid token usage to keep hidden
  latency tradeoffs visible.
- The benchmark/eval cleanup showed that timestamp-based docs search is a real
  product requirement, not a benchmark-only convenience.

Avoid these claims:

- Do not use the broken old OBD lineage as a real comparison row.
- Do not compare old undercounted FC token totals against the new corrected FC
  prompt-footprint numbers.
- Do not present the Japanese probe as a fully scrutinized final headline row.
