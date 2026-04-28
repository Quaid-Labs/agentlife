# AgentLife Benchmark Methodology

**Version:** 8.2
**Status:** Release-prep methodology
**Last updated:** 2026-03-29

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
- about `100k` transcript tokens
- no filler sessions

### AgentLife L

- `20` arc sessions plus `259` filler sessions
- `279` total sessions
- about `200k` transcript tokens in current release methodology accounting
- same gold eval set as AgentLife S; filler sessions only add noise/scale
- FC baseline for AL-L uses compaction at the `160k` boundary (summary prefix + trailing raw context window)

### AgentLife L OBD

- uses the same `279` sessions
- merges them into one synthetic operational day
- stresses the real runtime rolling extractor and final flush path directly

## 3. Scoring

Current canonical scoring uses:

- `268` mainline queries for Tiers 1-4
- `15` Tier 5 emotional-intelligence queries

The canonical full query set is `283` (`268 + 15`). Tier 5 is scored with a
separate, looser rubric suitable for emotional and relational responses.

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

Tier 5 is the emotional and boundary-awareness lane. It remains useful for
product work and is included in the canonical `283` query set, while retaining
its own scoring rubric.

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

## 5. Native OpenClaw VM Methodology

The native OpenClaw lane is run through `eval/vm_benchmark.py --system
oc-native`, not through the Quaid production launcher. It uses a Tart macOS VM
with OpenClaw installed and no Quaid plugin dependency.

The current native OC memory stack under test is:

- bundled `memory` CLI surface retained for forced `openclaw memory index/status`
- `memory-core` with builtin memory backend
- native session transcript indexing
- bundled `session-memory` hook on `/new`
- `active-memory` blocking recall sub-agent for direct `main` sessions
- `memory-wiki` bridge/import/compile flow over memory-core public artifacts
- host-visible `nomic-embed-text` embeddings through
  `http://192.168.64.1:11435/v1`

Injection semantics:

- benchmark transcripts are written as real OpenClaw session JSONL files
- each synthetic session is closed with `/new` so the bundled
  `session-memory` hook can create workspace memory
- the harness forces `openclaw memory index --agent main --force`
- after indexing, the harness runs `openclaw wiki init`, `openclaw wiki bridge
  import`, and `openclaw wiki compile`
- evaluation sends each benchmark question through `openclaw agent` in an
  isolated eval session
- OC eval sessions are registered under a non-user hook-scoped session key and
  write transcripts under a sibling agent session tree
  (`~/.openclaw/agents/benchmark-eval/sessions`) so the active eval turn stays
  outside the indexed main-agent `sessions/` directory during answering
- each OC eval session sibling transcript/store entry is then removed from the
  guest immediately after its answer is captured so later eval queries cannot
  retrieve prior `eval-q*` material while the full OC memory stack remains
  enabled during each query
- benchmark startup now kills any lingering guest `openclaw-gateway` process
  before clearing main/sibling eval transcripts and session state, so aborted
  prior runs cannot repopulate stale `eval-q*` files into the next scored lane

Run `AL-S` with `--no-filler`. Run `AL-L` without `--no-filler`, using the same
arc sessions plus filler corpus. Restore the clean VM snapshot before each
scored lane.

## 6. Token Accounting

Token numbers have different precision depending on the lane and phase:

- Quaid ingest/janitor lanes record real runtime token usage where the runtime
  exposes usage accounting.
- VM session context metrics are simulated from benchmark transcript token
  counts and show context growth, compaction savings, and cache-aware effective
  token estimates.
- Eval rows record visible lower-bound token estimates: question tokens,
  prediction tokens, visible agent total, and judge prompt tokens.
- `scores.json` aggregates eval estimates under `eval_token_estimate`.
- Codex app-server/OpenAI-family lanes use Spark's logged-in Codex account
  rather than a raw `OPENAI_API_KEY`; provider-reported token counts are
  recorded when exposed, but OAuth/app-server billing spend is not treated as an
  exact dollar-cost source.
- For native OpenClaw, hidden prompt context, active-memory sub-agent calls,
  tool-call payloads, gateway provider overhead, and any unreported provider
  usage are not included unless OpenClaw exposes them in machine-readable usage
  logs.

Use eval token estimates for relative lower-bound comparisons and trend checks.
Do not present them as exact billed provider spend unless the run also has
provider-reported usage artifacts.

## 7. Public Runbook Policy

Exact public numbers are now kept out of this file.

Why:

- release numbers change more often than the stable lane definitions
- root docs should not lag a reviewed public runbook
- the repo now has a dedicated tracked home for released benchmark snapshots

Public released numbers should live in:

- `published/runbooks/`

Frozen public-supporting artifacts should live in:

- `published/checkpoints/`

This methodology file should describe:

- what the lanes mean
- how runs are launched
- how scores are interpreted

It should not become a second competing source of exact leaderboard numbers.

## 8. Reproduction

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
  --ingest-schedule rolling-obd \
  --model claude-sonnet-4-6 \
  --judge gpt-4o-mini
```

Native OpenClaw VM examples:

```bash
# AL-S OC native
python3 eval/vm_benchmark.py \
  --system oc-native \
  --vm-ip 192.168.64.3 \
  --tart-host alfie.local \
  --snapshot clean-openclaw \
  --results-dir data/results-vm-oc-native-current-als \
  --answer-model openai/gpt-5.4 \
  --judge-model gpt-4o-mini \
  --splitting timeout \
  --no-filler

# AL-L OC native
python3 eval/vm_benchmark.py \
  --system oc-native \
  --vm-ip 192.168.64.3 \
  --tart-host alfie.local \
  --snapshot clean-openclaw \
  --results-dir data/results-vm-oc-native-current-all \
  --answer-model openai/gpt-5.4 \
  --judge-model gpt-4o-mini \
  --splitting timeout
```

## 9. Unscored Utilities

Rolling replay utilities exist for migration and stress testing:

- `scripts/export-imported-claude-history.py`
- `scripts/run-imported-claude-history.py`

These are useful for understanding scaling, migration, and daemon behavior, but
they are not leaderboard lanes.

## 10. Source of Truth

For released/public numbers:

- `published/runbooks/`
- `published/checkpoints/`

For working/internal rerun matrices:

- benchmark operator runbooks outside this repo

This methodology file should stay aligned with the released runbooks without
duplicating their exact numeric tables inline.
