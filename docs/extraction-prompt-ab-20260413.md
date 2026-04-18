# Extraction Prompt A/B Notes — 2026-04-13

## Goal
Measure how much provider extraction shape can be changed through harness-only prompt pressure, without changing runtime logic, chunking, or datastore behavior.

## Method
Use the canonical benchmark harness on Spark with a fixed small ingest slice:
- command shape: `--mode ingest --max-sessions 5`
- embeddings: `nomic-embed-text` at `768`
- no runtime logic changes between A/Bs
- stop each run after the 5 cached-preextract chunk extraction lines are emitted
- compare:
  - total extraction latency
  - total output tokens
  - total fact count

This isolates extraction-shape behavior without paying for a full `AL-S` every iteration.

## Harness Hook
Harness-only appendix hook added via environment variable:
- `BENCHMARK_EXTRACTION_PROMPT_APPENDIX`

This is appended to the extraction prompt in `eval/extract_compact.py` and passed through by `scripts/launch-remote-benchmark.sh`.

## First Harsh Appendix
```text
HARSH CANONICALIZATION MODE:
- Output the minimum complete set of durable facts needed for future retrieval.
- Do not restate the same topic from multiple angles.
- If two candidate facts overlap substantially, keep only the single most specific searchable fact.
- Do not emit paraphrases, glosses, or narrative reformulations of a fact already captured.
- For project passages, keep one canonical fact per concrete implementation detail; do not restate the same feature, stack choice, or bug in alternate wording.
- Prefer omission over duplicate or weakly useful detail.
- Before finalizing, aggressively delete any fact whose retrieval value is mostly contained in another fact.
- Keep relationship wording canonical and singular; never emit alternate labels for the same relationship.
- Keep journal entries and snippets concise; do not mirror project logs inside journal_entries.
```

## A/B Results On The Same 5-Session Slice

### Codex `gpt-5.4`
Baseline:
- `650.5s`
- `35,019` output tokens
- `323` facts

Harsh:
- `493.8s`
- `26,582` output tokens
- `234` facts

Delta:
- latency: `-24.1%`
- output tokens: `-24.1%`
- facts: `-27.6%`

### Haiku `claude-haiku-4-5-20251001`
Baseline:
- `204.3s`
- `29,621` output tokens
- `158` facts

Harsh:
- `173.0s`
- `24,438` output tokens
- `127` facts

Delta:
- latency: `-15.3%`
- output tokens: `-17.5%`
- facts: `-19.6%`

### Sonnet `claude-sonnet-4-6`
Baseline:
- `386.9s`
- `25,515` output tokens
- `120` facts

Harsh:
- `348.5s`
- `21,582` output tokens
- `101` facts

Delta:
- latency: `-9.9%`
- output tokens: `-15.4%`
- facts: `-15.8%`

## What We Learned
1. Prompt pressure changes extraction shape on all providers.
2. Codex is the most sensitive to suppressive canonicalization guidance.
3. Haiku responds clearly, but less dramatically than Codex.
4. Sonnet is the most selective already; it moves the least under the same pressure.
5. The same base prompt is not provider-neutral in behavior.
6. A provider can improve materially under prompt pressure without becoming another provider.

## Provider Shape Observed
Baseline fact density on the same slice:
- Codex: `323`
- Haiku: `158`
- Sonnet: `120`

Harsh fact density:
- Codex: `234`
- Haiku: `127`
- Sonnet: `101`

Interpretation:
- Codex has a strong tendency to enumerate and restate unless constrained.
- Anthropic models are more selective by default, especially Sonnet.
- If quality correlates with lower redundancy, prompt suppression is a legitimate lever worth testing.

## Important Constraints
- This is harness-only prompt shaping, not runtime/product logic.
- These slice tests measure extraction shape, not end-to-end benchmark quality.
- Full `AL-S` runs are still required to know whether reduced redundancy helps or harms final score.

## Current Next Step
A full `AL-S` Haiku/Haiku run is in progress using an even harsher appendix (`r1228`) to test whether stronger canonicalization improves or degrades full-run quality.

## Revised Real-Usage Wording
For further iterations, avoid benchmark-facing wording. Use wording like:

```text
EXTREME CANONICALIZATION MODE:
- Emit only durable, retrieval-worthy facts. Omit ephemeral detail unless it is likely to matter for future recall or action.
- One canonical fact per underlying point. Never restate the same point with different wording, granularity, or framing.
- If two facts overlap, keep only the single most specific fact that subsumes the others.
- Prefer fewer facts. When unsure between keeping and dropping, drop.
- Do not emit narrative gloss, summary prose, commentary, sentiment framing, or explanatory paraphrase.
- For projects, keep only concrete implementation details, decisions, bugs, or requirements that would support future recall. Do not restate the same feature or stack choice in alternate wording.
- For people and relationships, use one canonical relationship label only. Do not emit alternate labels or mirrored variants.
- Journal entries and snippets must be concise. Never mirror project logs into journal entries.
- Before finalizing, aggressively delete any fact whose retrieval value is mostly contained in another fact.
- Target the minimum complete set, not an exhaustive transcript decomposition.
```

## Open Questions
1. Is there an optimal harshness level where redundancy drops without starving recall?
2. Does Haiku benefit from stronger suppression the way Codex does?
3. Are there query families where aggressive canonicalization hurts more than it helps?
4. Should provider-specific extraction appendices exist for experiments, even if runtime remains provider-agnostic?

## Stronger Real-Usage Appendix Draft
```text
CRITICAL: MINIMUM DURABLE MEMORY ONLY
- Output the minimum complete set of durable facts that would help future recall or action.
- Prefer omission over inclusion.
- Do not try to preserve everything said in the transcript.
- If a detail is ephemeral, weak, repetitive, or low-value, drop it.

CRITICAL: ONE FACT PER UNDERLYING POINT
- Emit exactly one canonical fact for each underlying point.
- If several candidate facts describe the same point, keep only the single best fact.
- Never emit paraphrases, glosses, alternate framings, mirrored restatements, or different-granularity versions of the same point.

CRITICAL: COLLAPSE SIBLING DETAILS
- When several items belong to the same parent object, decision, or situation, combine them into one fact unless an item is independently likely to matter later.
- Collapse simple field lists, ingredient-like lists, attribute lists, config key lists, endpoint lists, dependency lists, and exclusion lists.
- Do not emit one fact per item when one grouped fact would preserve the same retrieval value.

CRITICAL: BAN OVER-ATOMIC PATTERNS
- Do not emit one fact per recipe field, database column, package script, dependency, environment variable, endpoint, ingredient, or rejected option unless that individual item clearly has standalone retrieval value.
- Do not emit one fact for each emotion word if they all reflect the same concern.
- Do not emit one fact for each logistical step if they all support one plan.

CRITICAL: PREFER POSITIVE CANONICAL FACTS
- When several negative or exclusionary statements express one preference, emit one positive canonical fact if possible.
- Example: prefer one fact like "the project should stay simple with minimal dependencies" instead of separate facts banning Docker, PostgreSQL, and setup overhead.

IMPORTANT: PROJECT COMPRESSION
- Preserve concrete implementation details, requirements, decisions, bugs, and future planned changes.
- Group related implementation details into compact canonical facts.
- Do not restate the same project feature or architectural choice from multiple angles.

IMPORTANT: PEOPLE AND RELATIONSHIPS
- Use one canonical relationship label only.
- Merge nearby caregiving, travel, and support details when they are part of one plan or decision.
- Keep at most one durable emotional-state fact per underlying concern unless multiple clearly distinct concerns exist.

IMPORTANT: JOURNAL AND SNIPPETS
- Keep journal entries and snippets short, sparse, and non-redundant.
- Never mirror project logs, fact lists, or field-by-field summaries into journal entries.

FINAL CHECK: DELETE AGGRESSIVELY
- Delete any fact whose retrieval value is mostly contained in another fact.
- Delete any fact that exists only to restate, decorate, explain, or emotionally color another fact.
- The correct answer is the smallest durable memory set that still preserves future usefulness.
```
