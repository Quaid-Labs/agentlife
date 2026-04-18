# AgentLife Technical Report — 2026-04-14

## Summary
Current AgentLife benchmark results support a clear product direction:
- Anthropic is the recommended provider for Quaid today.
- Codex/OpenAI OAuth is not recommended as the default provider.
- Harsher extraction prompts are not a net win as a default strategy; they compress output, but they do not improve the best end-to-end result and materially hurt Codex.
- Quaid is moving to a single-key model because users are more likely to want one reliable provider path than a fragmented multi-key system with weaker platforms and more operational overhead.
- Provider channels are not a reliable foundation because no platform gives a dependable stateless channel contract; auth-backed access is the only stable substrate.

## Headline Recommendation
Recommended default:
- Anthropic auth-backed lane

Not recommended as default:
- Codex/OpenAI OAuth lane

Current stance:
- We are open to revisiting this if benchmark numbers change materially.
- Based on the current evidence, the simpler and better-performing product choice is a single-key system centered on the strongest provider.

## Benchmark Evidence
### Best recent Anthropic baseline
Run:
- `r1195`
- `AL-S`
- backend `oauth`
- ingest `claude-haiku-4-5-20251001`
- eval `claude-haiku-4-5-20251001`
- judge `gpt-4o-mini`

Results:
- T1-T4: `87.13%`
- retrieval: `36.75%`
- Tier 5: `73.33%`
- weighted overall: `86.40%`

Runtime:
- total elapsed: `3656.366s` = `60.9m`
- ingest API calls: `282`
- ingest tokens: `906,053 in / 239,626 out`
- eval API calls: `1521`
- eval tokens: `11,308,494 in / 140,498 out`

Ingest shape:
- total facts extracted: `734`
- stored: `690 facts / 78 edges`
- rolling days: `3`

This is the restored clean Anthropic reference after the benchmark harness layout bug was fixed.

### Best recent Codex full run
Run:
- `r1221`
- `AL-S`
- backend `codex`
- deep/ingest `gpt-5.4`
- fast `gpt-5.4-mini`
- eval `gpt-5.4`
- judge `gpt-5.4-mini`

Results:
- T1-T4: `80.22%`
- retrieval: `79.85%`
- Tier 5: `83.33%`
- weighted overall: `80.39%`

Runtime:
- total elapsed: `13,658.743s` = `227.6m`
- ingest API calls: `443`
- ingest tokens: `767,432 in / 639,069 out`
- eval API calls: `1481`
- eval tokens: `6,325,686 in / 74,929 out`

Compared to `r1195`:
- weighted score delta: `-6.01pp`
- T1-T4 delta: `-6.91pp`
- elapsed time multiplier: `3.74x` slower

This is the strongest Codex lane tested recently. Even here, Anthropic remains clearly better and much faster.

### Codex mini-only result
Run:
- `r1218 (r1217)`
- eval-only lineage on Codex mini ingest
- backend `codex`
- deep/ingest `gpt-5.4-mini`
- eval `gpt-5.4-mini`
- judge `gpt-5.4-mini`

Results:
- T1-T4: `62.50%`
- retrieval: `67.91%`
- Tier 5: `90.0%`
- weighted overall: `63.96%`

This lane is not competitive.

## Harsher Extraction Prompt Experiments
## Goal
Test whether much stricter extraction guidance can reduce redundant fact spam and improve end-to-end benchmark quality.

### Prompt-shape slice experiments
We ran fixed 5-session ingest slices and compared baseline vs harsh prompt appendices.

Observed provider sensitivity:
- Codex `gpt-5.4` was the most sensitive to aggressive canonicalization.
- Haiku responded meaningfully but less dramatically.
- Sonnet was the least sensitive and was already more selective by default.

The main extraction anti-patterns were:
- one fact per sibling field or list item
- one fact per rejected option
- one fact per config key / dependency / endpoint
- repeated emotional reframings of the same concern
- fragmented logistics and care-planning facts

### Haiku harsh vs extra-harsh full runs
Earlier harsh run:
- `r1228`
- T1-T4 `83.96%`
- retrieval `33.21%`
- Tier 5 `80.0%`
- weighted overall `83.75%`
- elapsed `2918.650s` = `48.6m`

Extra-harsh run:
- `r1232`
- T1-T4 `85.63%`
- retrieval `37.13%`
- Tier 5 `76.67%`
- weighted overall `85.16%`
- elapsed `2539.807s` = `42.3m`

Interpretation:
- Extra-harsh recovered some quality relative to the first harsh attempt.
- But it still did not beat the restored clean Haiku baseline `r1195`.
- So harsher prompting did not produce a new best Anthropic result.

### Codex extra-harsh result
Ingest run:
- `r1233`
- backend `codex`
- deep `gpt-5.4`
- fast `gpt-5.4-mini`
- eval target `gpt-5.4-mini`
- extra-harsh extraction appendix

Ingest summary:
- extracted: `592 facts`
- stored: `576 facts / 120 edges`
- post-janitor DB: `508 nodes / 50 edges`
- ingest completed cleanly

Eval retry that completed:
- `r1236 (r1233)`
- eval parallelism `2`

Results:
- T1-T4: `75.75%`
- retrieval: `77.05%`
- Tier 5: `80.0%`
- weighted overall: `75.97%`
- elapsed `1201.219s` = `20.0m` for eval-only

Compared to the stronger non-extra-harsh Codex run `r1221`:
- weighted score delta: `-4.42pp`
- T1-T4 delta: `-4.47pp`

Interpretation:
- Extra-harsh prompting materially hurt Codex quality.
- Lowering eval parallelism solved Codex OAuth `429` failures, but it did not solve the quality problem.

## Final Read On Harsh Prompts
Current conclusion:
- Harsher extraction prompts reduce redundancy.
- But they are not a general quality win.
- On Haiku, extra-harsh is still below the clean baseline.
- On Codex, extra-harsh is clearly worse than the better non-extra-harsh setup.

Recommendation:
- Do not adopt harsher extraction prompts as the default product strategy.
- The better path is to use the stronger provider, not to overconstrain a weaker one.

## Why Quaid Is Moving To A Single-Key System
The benchmark results reinforce a product decision:
- Users are more likely to want one key that works well everywhere than multiple keys for multiple providers with uneven performance.
- Provider fragmentation increases setup cost, cognitive overhead, and operational failure modes.
- If one provider is clearly stronger, the simpler product is to standardize around it.

Current rationale:
- Anthropic materially outperforms Codex on `AL-S`.
- Anthropic is much faster on the same benchmark family.
- Codex requires more operational care around rate limits and configuration.
- Harsher prompt steering does not close the gap.

Therefore:
- Quaid is moving toward a single-key model centered on the strongest provider.
- We are open to reverting this if benchmark results materially change.
- Today, the simpler product is also the better-performing one.

## Why Provider Channels Are Not A Reliable Base
Another reason for the single-key direction is infrastructure realism.

We cannot rely on provider-owned channels as a durable product substrate because:
- there is no reliable stateless channel guarantee
- platform channels can accumulate hidden state or implicit history
- that breaks reproducibility and makes evaluation and runtime behavior less predictable
- channel semantics are provider-controlled, not product-controlled

So the stable foundation is:
- auth-backed provider access
- product-managed context and memory
- not channel-managed conversational state

This is not just a benchmark concern. It is a product constraint.

## Product Direction
Current direction:
1. recommend Anthropic as the default provider
2. keep Codex/OpenAI non-default and experimental
3. keep Quaid on a single-key model rather than fragmenting provider setup
4. rely on auth-backed access, not provider channels, as the operational substrate

This direction is based on measured benchmark performance, latency, and operational reliability, not preference.
