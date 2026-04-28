# AgentLife Technical Report — 2026-04-18 Draft

Status: draft. Final publication is waiting on two artifacts:

- Fresh neutral-surface `AL-S` after removing benchmark data leaks from seeded
  project/static context and wiring project updates through the runtime updater.
- OpenClaw-native `AL-L` refresh on the same `2026.4.11 (769908e)` base as the new `AL-S` row.
- Follow-up analysis on Japanese `AL-S` planner/docs gaps after the first
  validated JP review/filler run.

## Summary

This report refreshes the public AgentLife numbers after the April recall and
extraction tuning cycle.

The short version:

- Quaid's best refreshed `AL-S` row is `92.58%` under Sonnet re-evaluation.
- Quaid's refreshed `AL-L` row is `87.10%` under Sonnet re-evaluation.
- Quaid's refreshed `AL-L OBD` row is `88.69%` under Sonnet re-evaluation.
- These rows are slightly lower than the strongest prior "beats FC" framing on
  plain `AL-L`, but the tradeoff is intentional: language-dependent lexical
  planning has moved toward the LLM-owned layer, which gives Quaid a viable path
  to multilingual behavior instead of hardcoded English-only matching.
- The first Japanese `AL-S` run completed end-to-end on Kanji/Kana inputs. It
  scored below the matched English run, but it did not collapse; this validates
  that Quaid can operate on a non-Roman-script benchmark while exposing clear
  retrieval/planner gaps.
- Public token accounting is now corrected to include answer tokens,
  preinject-recall tokens, and recall tool-call tokens, while excluding judge
  tokens.

## Methodology Change: Token Accounting

Prior technical reports undercounted Quaid public token spend because they used
answer/eval accounting that did not include tokens returned through recall tool
calls or preinject recall. That made older rows useful for internal lineage
comparison, but not complete enough for public benchmark reporting.

Effective with this report, the public benchmark token metric is:

```text
public_eval_tokens = eval.total_tokens - eval.by_source.judge.total_tokens
```

This includes:

- answer model tokens
- preinject recall tokens
- explicit recall/tool-call tokens

This excludes:

- judge tokens

The new metric is intentionally more conservative and should be used for public
numbers going forward. Older token totals that did not include preinject/tool
recall are not directly comparable.

## Methodology Change: Run Setup Integrity

During the April 18 audit we found benchmark setup leakage in the harness-created
workspace. The problem was not a model/runtime behavior change; it was the
benchmark harness seeding more knowledge and instruction surface than a normal
fresh Quaid project should have.

Observed leak classes:

- Project docs contained benchmark/data-specific future state rather than only
  base project scaffolds and synced source artifacts.
- `USER.md` had unnatural setup/coaching language instead of looking like the
  normal production base user profile.
- The benchmark extraction prompt had drifted into a harness fork with
  AgentLife-specific examples instead of using the same extraction prompt as
  product runtime.
- Project file replay copied source snapshots directly, but did not call the
  project updater path that normally runs when the daemon detects dirty project
  files. This meant `PROJECT.md`/`PROJECT.log` behavior was not aligned with
  production.

New run-setup policy:

- Seed only natural base Quaid context files that match production templates.
- Seed project docs only as new-project scaffolds. Do not preload future project
  facts, benchmark-specific examples, evaluator hints, or special instructions.
- Project/code artifacts may be copied because they are the test artifacts.
- After project artifacts change, the harness must call Quaid's product project
  updater flow rather than doing project-doc reasoning in harness code.
- `PROJECT.md` and `PROJECT.log` are runtime-managed artifacts and must be
  preserved across project snapshot replay.
- Extraction must use the runtime extraction prompt plus runtime-style dynamic
  owner/domain/project blocks, not a benchmark-specific prompt fork.

Harness changes made for this report:

- `eval/extract_compact.py` now loads `modules/quaid/prompts/extraction.txt`
  from the checkpoint/runtime tree and appends the same dynamic blocks used by
  product runtime.
- `eval/run_production_benchmark.py` now seeds `recipe-app` and
  `portfolio-site` with base `PROJECT.md` scaffolds only, and no per-project
  `TOOLS.md` future-state docs.
- Project snapshot replay now preserves `PROJECT.md`, `PROJECT.log`, and
  `TOOLS.md`.
- After every benchmark project source sync, the harness queues a generic
  source-file-change event and invokes `datastore/docsdb/project_updater.py
  process-event`, followed by `refresh-project-md`.

Implication:

- Rows in this draft remain useful for internal lineage comparison, but the
  final public table should be refreshed from the neutral-surface harness. The
  immediate gate is a fresh `AL-S` run on the cleaned setup before publishing
  updated headline claims.

## Current Quaid Rows

Model setup for the refreshed rows:

- ingest/deep runtime: `claude-sonnet-4-6`
- fast runtime: `claude-haiku-4-5-20251001`
- Sonnet re-eval rows: `claude-sonnet-4-6`
- Haiku eval rows: `claude-haiku-4-5-20251001`
- judge: `gpt-4o-mini`
- embedding: `nomic-embed-text`

| Lane | Run | Eval | Combined | T1-T4 | Retrieval | Tier 5 | Public Eval Tokens |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `AL-S` | `r1298` | Haiku | `89.93%` | `89.93%` | `48.13%` | `80.00%` | `9,629,542` |
| `AL-S` | `r1301 (r1298)` | Sonnet | `92.58%` | `92.58%` | `48.51%` | `86.67%` | `8,759,805` |
| `AL-L OBD` | `r1299` | Haiku | `86.93%` | `86.93%` | `59.51%` | `80.00%` | `7,803,071` |
| `AL-L OBD` | `r1303 (r1299)` | Sonnet | `88.69%` | `88.81%` | `57.46%` | `86.67%` | `7,536,022` |
| `AL-L` | `r1302` | Haiku | `83.39%` | `84.10%` | `45.34%` | `70.00%` | `10,491,927` |
| `AL-L` | `r1304 (r1302)` | Sonnet | `87.10%` | `86.57%` | `41.42%` | `96.67%` | `9,230,495` |

Notes:

- `AL-S` Sonnet re-eval is the strongest current refreshed row at `92.58%`.
- `AL-L OBD` remains very strong at `88.69%`, but OBD is not the same
  methodology as plain `AL-L`.
- Plain `AL-L` lands at `87.10%`, slightly below the historical FC-Sonnet
  `87.70%` row. This should be presented honestly as a small fidelity tradeoff
  for a more multilingual-capable recall architecture.
- Sonnet re-eval materially lifts `AL-L` from the Haiku-eval source row
  (`83.39%` combined to `87.10%` combined), consistent with previous `AL-L`
  behavior where Haiku evaluation underestimates final Sonnet-answer quality.

## Why The Fidelity Changed

The major quality tradeoff in this cycle was moving language-dependent lexical
planning out of hardcoded recall logic and into LLM-owned planning/prompt layers.

The older English lexical matcher was doing real work. In A/B testing, restoring
the older deterministic lexical behavior produced strong `AL-S` scores quickly.
The downside is that hardcoded English matching creates structural multilingual
debt: English gets special recall behavior, while other languages either miss
the boost or risk being shaped by the wrong assumptions.

The current direction is:

- slow/full recall uses LLM-owned lexical planning so anchor selection can
  preserve the source language and script
- fast preinject may keep narrow guarded boosts only when they do not make
  non-English behavior worse than lexical-off behavior
- extraction and edge-review prompts carry relationship-role fidelity rules in
  the LLM layer rather than Python/TypeScript regexes

This is why a small drop from the strongest prior English-only lineage is
acceptable: the architecture now has a realistic path to Japanese, Spanish, and
other non-English benchmark coverage without cloning English regexes into each
language.

## Historical FC Baselines

Historical FC rows remain unchanged from prior reports:

| Lane | FC Run | FC Accuracy | FC Eval Tokens |
| --- | --- | ---: | ---: |
| `AL-S FC Sonnet` | `r606` | `92.90%` | `29,828,646` |
| `AL-L FC Sonnet` | `r857` | `87.70%` | `34,596,206` |
| `AL-S FC Haiku` | `r600` | `87.70%` | `29,855,754` |
| `AL-L FC Haiku` | `r607` | `83.60%` | `34,397,219` |

Comparison notes:

- `AL-S`: refreshed Quaid Sonnet re-eval is `92.58%`, within `0.32pp` of
  FC-Sonnet `92.90%`, at far lower public eval token spend.
- `AL-L`: refreshed Quaid Sonnet re-eval is `87.10%`, `0.60pp` below FC-Sonnet
  `87.70%`, again at far lower public eval token spend.
- `AL-L OBD`: refreshed Quaid Sonnet re-eval is `88.69%`; this is strong, but
  OBD is a one-day load/check methodology and should not be described as the
  same surface as plain `AL-L`.

## OpenClaw Native Baseline Update

OpenClaw-native rows use OpenClaw's native memory system rather than Quaid.

Current refreshed base metadata:

- OpenClaw version: `2026.4.11 (769908e)`
- answer model: `openai/gpt-5.4`
- auth/backend: `codex-oauth`
- judge: `haiku`
- host: `alfie.local` Tart VM
- embedding: `qwen3-embedding:8b`
- embedding endpoint: `http://192.168.64.1:11435/v1`

| Lane | Run | Score | Notes |
| --- | --- | ---: | --- |
| `AL-S` | `oc-native-als-codex-alfie-retry2-20260414-163950` | `70.15%` | Completed `268/268`; `188C/0P/80W`; much slower than older OC anchor. |
| `AL-L` | `ocnative-all-codex-alfie-20260418-185035` | pending | Running on the same OC `2026.4.11 (769908e)` base. |

OpenClaw lifecycle caveat:

- The current OC-native benchmark harness does not expose a deterministic
  "trigger dream now" control.
- `AL-S` and `AL-L` simulate roughly 60 days and would ideally require one dream
  cycle per simulated day to mimic production behavior.
- `AL-L OBD` is closer to a one-day check, so it is easier to approximate, but
  still not a perfect match without explicit dream-cycle control.
- Until the harness can trigger dreams deterministically, OC-native rows should
  be described as session-memory hook + forced index + memory-wiki bridge
  measurements, not full production dreaming measurements.

## Japanese AL-S Multilingual Probe

The Japanese translation set is available and validator-clean:

- arc sessions: `27/27`
- filler sessions: `259/259`
- query count: `268`
- Tier 5 count: `15`
- validator status: `Validation OK`

The first multilingual probe used `AL-S` with Japanese Kanji/Kana content and no
romaji transliteration as a first-pass symbolic-language test.

| Run | Dataset | Eval | Combined | T1-T4 | Retrieval | Tier 5 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `r1320` | Japanese | Haiku | `74.73%` | `73.88%` | `25.93%` | `90.00%` | First validated JP run; end-to-end non-Roman-script support confirmed. |
| `r1322 (r1320)` | Japanese | Haiku | `77.61%` | `77.61%` | `26.68%` | n/a | Eval-only first-pass multilingual planner A/B; no re-ingest. |
| `r1321` | English matched state | Haiku | `86.57%` | `86.57%` | `49.25%` | `86.67%` | Same current runtime/checkpoint state, canonical English dataset. |

Interpretation:

- Quaid successfully ingested, recalled, answered, and judged the Japanese
  Kanji/Kana dataset without romaji conversion. This is enough to say Quaid
  supports multilingual operation at a real benchmark level, not just isolated
  prompt probes.
- The JP run is not parity with English. Against the matched English state,
  JP is down `11.84pp` combined and `23.32pp` retrieval.
- The largest gap is retrieval quality, not basic language handling. JP answer
  behavior remains usable, but recall surfaces fewer correct memories before the
  answer model writes.
- A first long-recall planner A/B (`r1322`) improved JP answer accuracy from
  `73.88%` to `77.61%` on the same `r1320` ingest, but retrieval barely moved
  (`25.93%` to `26.68%`). This suggests planner wording can improve answer-time
  synthesis, but the deeper JP gap is still retrieval/document surfacing.
- In the first full JP run (`r1320`), strong categories include adversarial IDK (`100.00%`), negative questions
  (`100.00%`), inference (`92.31%`), multi-session synthesis (`92.86%`), and
  cross-reference (`92.31%`).
- Weak `r1320` categories include architecture comprehension (`45.45%`), contested
  fact (`50.00%`), factual recall (`58.62%`), stale fact (`58.33%`), and the
  retrieval-only metric (`25.93%`).
- Project-state answer accuracy is moderate (`70.83%`), but retrieval remains
  weak. This aligns with the broader project-docs finding: the runtime needs
  better generated project documentation and better multilingual query planning
  for Japanese questions over English-heavy source artifacts.

Follow-up work:

- Move multilingual query shaping into the long/full recall planner, where the
  LLM can preserve the source language while adding retrieval-only variants for
  English/code/document artifacts when useful.
- Keep fast preinject conservative; do not add Japanese-specific regexes,
  dictionaries, or transliteration rules in code.
- Re-run JP after the product project-docs updater is active, because many JP
  misses are in project and architecture categories where better generated docs
  should help.
- Do not treat `r1322` as a final planner win yet. It is a useful first pass,
  but the retrieval-only metric shows that more planner/docs work is needed
  before the JP path can be called competitive with English.

## Publication Guidance

Use these public claims:

- Quaid reaches `92.58%` on `AL-S` with Sonnet re-eval.
- Quaid reaches `87.10%` on `AL-L` with Sonnet re-eval.
- Quaid reaches `88.69%` on `AL-L OBD` with Sonnet re-eval.
- Public token totals now include preinject and recall tool-call spend and
  exclude judge spend.
- The small fidelity loss versus the strongest prior English-only path is the
  cost of moving language-dependent recall shaping toward LLM-owned multilingual
  behavior.

Avoid these claims:

- Do not say the refreshed plain `AL-L` row beats FC-Sonnet; it does not.
- Do not compare old undercounted token totals directly against new public token
  totals.
- Do not present OC-native rows as full production-dreaming results until the
  harness can trigger dream cycles deterministically.

## In-Flight Extension Block (2026-04-24)

This draft now has an explicit follow-on run block appended to the end of the
current study. It is being executed as a chained queue on Spark so that the
technical study reflects the live run order rather than an after-the-fact
reconstruction.

Important sequencing note:

- `r1423` was already in flight before the operator pivoted the auth/model
  sequence.
- Because of that, the live block starts with the already-running fresh
  `AL-S Sonnet/Haiku` line, then switches into the Solomon-auth segment, then
  switches back to Yuni for the remaining FC/`AL-L` work.

Current queued block:

| Order | Run | Surface | Models | Auth | Purpose |
| --- | --- | --- | --- | --- | --- |
| 1 | `r1423` | `AL-S` | `Sonnet/Haiku` | existing in-flight run | fresh small-lane source run already launched before the auth pivot |
| 2 | `r1424` | `AL-S FC` | `Sonnet` | Solomon OAuth | refreshed small-lane FC Sonnet anchor |
| 3 | `r1425 (r1423)` | `AL-S` | `Sonnet/Sonnet` | Solomon OAuth | eval-only Sonnet re-eval on the fresh `r1423` lineage |
| 4 | `r1426` | `AL-L OBD` | `Sonnet/Haiku` | Solomon OAuth | fresh long OBD source run |
| 5 | `r1427` | `AL-S FC` | `Opus 4.7` | Yuni OAuth | small-lane FC Opus 4.7 ceiling datapoint |
| 6 | `r1428` | `AL-L` | `Sonnet/Haiku` | Yuni OAuth | fresh long per-day source run |
| 7 | `r1429 (r1426)` | `AL-L OBD` | `Sonnet/Sonnet` | Yuni OAuth | eval-only Sonnet re-eval on the fresh `r1426` lineage |
| 8 | `r1430 (r1428)` | `AL-L` | `Sonnet/Sonnet` | Yuni OAuth | eval-only Sonnet re-eval on the fresh `r1428` lineage |
| 9 | `r1431` | `AL-L FC` | `Sonnet` | Yuni OAuth | refreshed large-lane FC Sonnet anchor |

Operational rules for this block:

- FC runs must use Anthropic prompt caching; they are too expensive to run
  uncached.
- If any source run fails (`r1423`, `r1426`, or `r1428`), downstream eval-only
  lineage runs do not continue from a tainted workspace.
- On source-run failure, the benchmark owner diagnoses the real failure,
  relaunches a fresh replacement source run, and resumes the chain from the new
  lineage.

## 2026-04-24 Queue Correction

- `r1425` (`AL-S` Sonnet/Sonnet eval-only from `r1423`) failed on a checkpoint-side explicit-store-plan timeout: the harness gave the recall subprocess 90s, but did not propagate that budget into explicit `stores=["vector","graph"]` deliberate recall config, so checkpoint `_run_recall_store_plan()` fail-hard timed out at its internal 30s default first.
- Harness fix landed locally as `9ea06e47` in `agentlife-benchmark`: deliberate recall now passes `timeout_ms` through to checkpoint config, and a new `fc-tier5` mode backfills Tier-5 onto an existing FC baseline without rerunning the FC answer sweep.
- Fresh replacement lineage is `r1433 (r1423)`.
- Inserted follow-up run is `r1434 (r1424)`: `AL-S` FC Sonnet Tier-5-only backfill on the preserved `r1424` FC baseline.
- Remaining queued order after `r1434`: `r1426`, `r1427`, `r1428`, `r1429`, `r1430`, `r1431`.

## 2026-04-28 Staged OpenClaw Update

This section supersedes the earlier pending OpenClaw notes in this draft.

The April 26-28 work split the OpenClaw study into two distinct surfaces:

- OpenClaw native memory (`oc-native`)
- Quaid embedded inside OpenClaw (`quaid-ocvm`)

That split matters. The fixes that raised Quaid-on-OC were mostly Quaid product
fixes in the bridge/docs/project-recall path, not harness-only tricks and not
generic OpenClaw-native improvements.

### Current OpenClaw Surface Table

| Surface | Run | Method | Score | Notes |
| --- | --- | --- | ---: | --- |
| OpenClaw native `AL-S` | `oc-native-als-20260426aaf` | fresh full, clean eval-isolated | `26.49%` | correct `alfie/test-openclaw` VM; eval contamination fixed; current native baseline |
| OpenClaw native `AL-L` | `oc-native-all-20260426aag` | fresh full, clean eval-isolated | `31.72%` | same native stack with filler sessions |
| Quaid on OpenClaw `AL-S` | `quaid-ocvm-full-bridgefix-20260428-010749` | fresh full embedded-Quaid VM run | `80.97%` | current best trustworthy OC-VM Quaid baseline |
| Quaid direct `AL-S` | `r1421` | fresh full direct harness | `88.62%` | clean direct baseline on the current valid set |
| Quaid direct `AL-S` Sonnet re-eval | `r1422 (r1421)` | eval-only stronger-answer recheck | `93.10%` | demonstrates direct-path lift from a stronger answer model |

Interpretation:

- The meaningful OC-VM Quaid gap is against the fresh direct Quaid baseline, not
  against the stronger eval-only Sonnet headline. On that apples-to-apples
  comparison, Quaid on OpenClaw is down about `7.65pp` (`88.62%` to `80.97%`).
- The stronger answer-model re-eval (`r1422`) moved the direct Quaid lineage up,
  not down. This is evidence against the claim that `openai/gpt-5.4` itself is
  the reason OpenClaw scores lower.
- The remaining loss is best read as OpenClaw pipeline tax: noisier evidence
  surfacing, weaker freshness/current-state shaping, and answer-path distortion
  between stored memory and the final model prompt.

### What Fixed Quaid-on-OC

The critical Quaid product fixes from this study were:

- instance-scoped bridge env preservation
- bridge docs-bundle materialization into real recall rows
- dated project-state auto-inject bounded to docs
- dated project-log recall focus
- structural graph date-bound enforcement
- structural freshness rerank / invented-anchor extraction guards

These changes are product/runtime fixes. They explain why the embedded Quaid
lane moved from broken numbers (`14.93%`) to stable numbers (`72.76%`, then
`80.97%`).

They do **not** transfer directly to OpenClaw native memory, because native OC
does not use Quaid's bridge/docs/project-log pipeline. For that reason, the
native `26.49%` / `31.72%` rows remain a fair current baseline for native
OpenClaw rather than an artifact of "Quaid got special benchmark handling."

### Remaining Quaid-on-OC Weaknesses

The current full Quaid-on-OC `AL-S` row is no longer failing on project-state
routing. The weaker families are now:

- `temporal_current`: `62.5%`
- `agent_retrieved`: `60.0%`
- `adversarial_false_attribution`: `40.0%`

Project-state is materially healthier:

- `project_state`: `83.3%`

This means the next improvement cycle should focus on answer-surface quality:
freshness, attribution, and exact assistant-retrieved detail preservation, not
on the older bridge/docs path failures.

### Dated Project-State Proof

Targeted proof run `quaid-ocvm-projectstate-proof13-20260428-093651` scored
`75.0%` (`3/4`) on the dated `recipe-app` project-state slice.

The remaining miss should not be treated as a runtime/product failure:

- `session-10-review-v2.txt` states that `dietary.test.js` and
  `mealplan.test.js` were added before `2026-03-18`
- `session-16-review-v2.txt` later hardcodes a `2026-05-08` ground truth that
  omits those suites

The stored `PROJECT.log` and the benchmark answer followed the earlier source
evidence. The recommendation is:

- treat the dated project-state path as product-pass / `PWN`
- queue dataset reconciliation post-ship

### Current Publication Guidance For OpenClaw

- Use `80.97%` as the current trustworthy Quaid-on-OpenClaw `AL-S` headline.
- Use `26.49%` (`AL-S`) and `31.72%` (`AL-L`) as the current clean native
  OpenClaw baselines.
- Do not compare the native `26.49%` row against the older `70.15%`
  `oc-native` eval-only number as if they were the same methodology; the older
  row is a useful historical reference, but not the current canonical native
  baseline.
- Do not blame the Quaid-on-OC gap primarily on `gpt-5.4`; the stronger direct
  Quaid re-eval shows the answer model can help when the evidence surface is
  clean.
- The next required OpenClaw datapoint is a fresh Quaid-on-OpenClaw `AL-L` run
  on a checkpoint cut from current `dev/main`.
