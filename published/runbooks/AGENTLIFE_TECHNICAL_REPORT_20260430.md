# AgentLife Technical Report — 2026-04-30

Status: draft for release review. This report supersedes the April 24 draft as
the current document to review before release.

Important scope note:

- This report intentionally freezes the last full validated matrix rather than
  folding in newer single-lane diagnostic runs.
- Post-April-24 narrow proofs, OC-gap studies, and eval-only lineage checks are
  useful internal diagnostics, but they are not a refreshed full benchmark
  block and therefore are not used here as headline publication rows.

## Summary

This report captures the release-review headline benchmark set using the last
full AgentLife matrix that was rerun and validated as a block.

The short version:

- Quaid's strongest current `AL-S` row remains `93.64%` under Sonnet
  re-evaluation.
- Quaid's strongest current plain `AL-L` row remains `88.52%` under Sonnet
  re-evaluation.
- Quaid's repaired `AL-L OBD` row remains `88.69%` under Sonnet re-evaluation.
- Current FC anchors remain:
  - `AL-S FC Sonnet`: `93.11%`
  - `AL-S FC Opus 4.7`: `92.76%`
  - `AL-L FC Sonnet`: `88.69%`
  - `AL-L FC Opus 4.7`: `89.40%`
- Current published OpenClaw surface anchors remain:
  - OpenClaw native `AL-S`: `26.49%`
  - OpenClaw native `AL-L`: `31.72%`
  - Quaid on OpenClaw `AL-S`: `80.97%`

Release-read:

- The clean direct Quaid headline block remains strong and stable enough for
  release framing.
- The bigger unresolved story is not core Quaid collapse; it is the OpenClaw
  execution tax between stored memory and final answer behavior.
- Because the newest OC investigations were single-lane diagnostics rather than
  a refreshed full matrix, they are excluded from the headline tables here and
  should be discussed separately if needed.

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
- Plain `AL-L` and repaired `AL-L OBD` remain effectively tied under Sonnet
  re-eval (`88.52%` vs `88.69%`).
- On FC anchors:
  - `AL-S`: Sonnet is slightly ahead of Opus (`93.11%` vs `92.76%`).
  - `AL-L`: Opus is slightly ahead of Sonnet (`89.40%` vs `88.69%`).
- The repaired `AL-L OBD` row matters because it is methodologically valid
  again. The older broken OBD lineage is not part of the headline set.

## Token Accounting And Preinject Methodology

The token accounting standard from the April 24 report remains in force.

For Quaid full/eval rows, the public eval token number is:

```text
non_judge_eval_tokens = answer_model + tool_recall + preinject_recall
```

This excludes:

- judge tokens

This includes:

- answer-model tokens
- explicit recall/tool-call tokens
- preinject recall tokens

For FC rows, public token totals use full prompt footprint from raw usage
events, excluding judge traffic.

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

## What Still Matters From The April Repair Cycle

The April 18 and April 24 work established the main methodological rules that
still govern release-quality benchmark interpretation:

- do not seed future-state or benchmark-shaped project docs into the workspace
- preserve runtime-managed project surfaces such as `PROJECT.md` and
  `PROJECT.log`
- route project updates through the runtime updater rather than through
  harness-only reasoning
- keep historical/docs recall date-bounded against true source chronology
- treat OBD and docs-heavy lanes as valid only when chronology survives end to
  end

This is why the current headline matrix is still trustworthy enough to carry
release framing: the strongest rows in this block were produced after those
benchmark-validity repairs, not before them.

## Current OpenClaw Surface Table

The last published OpenClaw surface table remains the correct release-review
reference until a new full OC block is rerun and reviewed.

| Surface | Run | Method | Score | Notes |
| --- | --- | --- | ---: | --- |
| OpenClaw native `AL-S` | `oc-native-als-20260426aaf` | fresh full, clean eval-isolated | `26.49%` | current native small-lane baseline |
| OpenClaw native `AL-L` | `oc-native-all-20260426aag` | fresh full, clean eval-isolated | `31.72%` | current native large-lane baseline |
| Quaid on OpenClaw `AL-S` | `quaid-ocvm-full-bridgefix-20260428-010749` | fresh full embedded-Quaid VM run | `80.97%` | current trustworthy OC-VM Quaid baseline |
| Quaid direct `AL-S` | `r1421` | fresh full direct harness | `88.62%` | clean direct baseline used for the OC gap comparison |
| Quaid direct `AL-S` Sonnet re-eval | `r1422 (r1421)` | eval-only stronger-answer recheck | `93.10%` | direct-path stronger-answer control |

### OpenClaw Read

- The meaningful OC-VM Quaid comparison is against the fresh direct Quaid
  baseline, not against the stronger Sonnet re-eval headline.
- On that apples-to-apples comparison, Quaid on OpenClaw remains down about
  `7.65pp` (`88.62%` to `80.97%`).
- That gap should still be read primarily as OpenClaw execution tax: noisier
  evidence surfacing, weaker freshness/current-state shaping, and answer-path
  distortion between stored memory and final prompt.
- The native OpenClaw rows (`26.49%`, `31.72%`) remain current native baselines
  and should not be mixed with the embedded-Quaid story.

## What Is Intentionally Excluded From This Report

The following are intentionally excluded from the headline tables here:

- newer single-lane `AL-S` direct reruns
- newer single-lane OpenClaw diagnostics
- eval-only lineage experiments that reuse stored data
- narrow three-query OC proofs

Those runs are useful for diagnosis and release risk analysis, but they are not
a refreshed full matrix. They should not displace the current full-set rows in
the release-facing headline section.

## Release Guidance

Use this report for release-review numbers unless and until a new full matrix is
rerun and reviewed.

The correct high-level framing is:

- direct Quaid headline quality remains strong
- the repaired `AL-L OBD` row is legitimate and publication-safe
- FC anchors remain competitive but much more expensive in prompt footprint
- the OpenClaw story is still the main caveat, and should be discussed as an
  execution-path tax rather than as evidence that core Quaid quality regressed

## Publication Guidance

- Use `93.64%` as the strongest current direct Quaid `AL-S` headline
  (`Sonnet` re-eval).
- Use `88.52%` as the strongest plain direct Quaid `AL-L` headline.
- Use `88.69%` as the repaired direct Quaid `AL-L OBD` headline.
- Use `93.11%` / `92.76%` (`AL-S`) and `88.69%` / `89.40%` (`AL-L`) as the FC
  anchors.
- Use `80.97%` as the current trustworthy Quaid-on-OpenClaw `AL-S` headline
  until a new full OC block supersedes it.
- Do not replace the headline block with newer single-lane diagnostics unless a
  new full set is intentionally being published.
