# AgentLife Benchmark Methodology

**Version:** 8.0
**Status:** Active runbook snapshot
**Last updated:** 2026-03-25

This document now tracks the current benchmark runbook rather than preserving
 every historical experiment. Public benchmark claims should be grounded in the
 current clean run matrix and refreshed as new serial reruns land.

## 1. Scope

AgentLife evaluates an end-to-end memory system, not just isolated retrieval.
The scored Quaid lanes exercise:

- transcript ingest
- runtime rolling extraction
- janitor review and dedup
- project-document indexing
- tool-using evaluation on the resulting memory system

The benchmark currently keeps three Quaid production-faithful lanes:

- `AL-S`: 20-session lane
- `AL-L`: 279-session large-scale lane
- `AL-L OBD`: one-big-day synthetic rolling stress lane

Reference baselines are kept alongside Quaid:

- `FC`: full-context ceilings with no memory compression
- `native OpenClaw`: native OC memory behavior on the same corpus

## 2. Dataset

### AgentLife S

- `20` scripted arc sessions
- about `92k` transcript tokens
- no filler sessions

### AgentLife L

- `20` arc sessions plus `259` filler sessions
- `279` total sessions
- about `423k` transcript tokens
- same gold eval set as AgentLife S; filler sessions only add noise/scale

### AgentLife L OBD

- uses the same `279` sessions
- merges them into one synthetic operational day
- stresses the real runtime rolling extractor and final flush path directly

## 3. Scoring

Current canonical scoring uses:

- `268` mainline queries for Tiers 1-4
- separate `30`-query Tier 5 lane

The public headline score is the mainline `268`-query score. Tier 5 is tracked
separately and should not be folded into the mainline number.

### Mainline

The `268` mainline queries cover:

- factual recall
- temporal reasoning
- cross-reference and synthesis
- project-state understanding
- graph-shaped recall
- adversarial and negative controls
- non-question behavior
- architecture comprehension/planning

### Tier 5

Tier 5 is the separate emotional and boundary-awareness lane. It remains useful
for product work, but it is reported separately from the mainline matrix.

## 4. Canonical Quaid Methodology

Current clean Quaid runs follow these rules:

1. Use serial runs, not concurrent batches.
2. Use `oauth` backend for scored Quaid runs.
3. Keep eval answer model fixed on `claude-haiku-4-5-20251001`.
4. Keep judge fixed on `gpt-4o-mini`.
5. Vary ingest/deep model across `Haiku`, `Sonnet`, and `Opus`.
6. Preserve OC-style `1h` gap splitting for day-based lanes.
7. Use runtime rolling carryover when a capture exceeds the `8k` token window.
8. Treat `AL-L OBD` as a real rolling-ingest stress lane, not a cached shortcut.

### Ingest Semantics

`AL-S` and `AL-L`:

- group sessions by operational day
- preserve the OC-style `1h` timeout/gap splitter inside each day
- send oversized day captures through the same runtime rolling extractor used by
  production

`AL-L OBD`:

- merges the full large corpus into one synthetic operational day
- runs fully through the runtime rolling extractor
- finishes with the normal flush and janitor path

### Full-Context Baselines

FC runs skip ingest entirely:

- all transcript context is placed directly in the answer prompt
- no memory DB or janitor lifecycle is involved
- only the answer model changes in FC experiments

## 5. Current Public Runbook Snapshot

These are the current clean runbook-backed numbers as of 2026-03-25.

### Quaid

| Scale | Haiku | Sonnet | Opus |
| --- | ---: | ---: | ---: |
| `AL-S` | `83.39` (`r802`) | `90.11` (`r803`) | `84.45` (`r807`) |
| `AL-L` | `81.21` (`r804`) | `83.75` (`r796`) | `82.16` (`r808`) |
| `AL-L OBD` | `80.04` (`r805`) | `85.87` (`r806`) | `83.21` (`r812`) |

### Reference Baselines

| Baseline | Score |
| --- | ---: |
| `AL-S FC Haiku` | `87.70` (`r600`) |
| `AL-S FC Sonnet` | `92.90` (`r606`) |
| `AL-L FC Haiku` | `83.60` (`r607`) |
| `AL-S native OpenClaw` | `69.40` |
| `AL-L native OpenClaw` | `63.06` |

### Current Read

- `Sonnet` is the strongest practical Quaid line across all three lanes.
- `Opus` acts as a costly ceiling datapoint rather than the best value line.
- `AL-L OBD Sonnet` currently outperforms the full `AL-L` Sonnet lane.
- The current `8k` rolling window remains the best default; larger Sonnet probe
  windows did not improve quality enough to justify the extra split pressure.

## 6. Reproduction

The canonical launcher is:

```bash
./scripts/launch-remote-benchmark.sh --remote spark --scale <s|l> -- <flags>
```

Examples:

```bash
# AL-S Sonnet
./scripts/launch-remote-benchmark.sh --remote spark --scale s -- \
  --mode full \
  --backend oauth \
  --model claude-sonnet-4-6 \
  --judge gpt-4o-mini

# AL-L Sonnet
./scripts/launch-remote-benchmark.sh --remote spark --scale l -- \
  --mode full \
  --backend oauth \
  --model claude-sonnet-4-6 \
  --judge gpt-4o-mini

# AL-L OBD Sonnet
./scripts/launch-remote-benchmark.sh --remote spark --scale l -- \
  --mode full \
  --backend oauth \
  --schedule rolling-obd \
  --model claude-sonnet-4-6 \
  --judge gpt-4o-mini
```

## 7. Unscored Utilities

Rolling replay utilities exist for migration and stress testing:

- `scripts/export-imported-claude-history.py`
- `scripts/run-imported-claude-history.py`

These are useful for understanding scaling, migration, and daemon behavior, but
they are not leaderboard lanes.

## 8. Source of Truth

For the active matrix and richer per-run cost/token telemetry, use:

- `~/quaid/util/agents/codex-benchmark/RUN_COMPARISON_MATRIX_20260325.md`

This methodology file should stay aligned with that runbook snapshot rather
than drifting into a second competing benchmark narrative.
