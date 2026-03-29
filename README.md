# AgentLife

A full-lifecycle benchmark for AI agent memory systems. AgentLife tests the
full production memory pipeline: extraction, rolling carryover, janitor review,
deduplication, project documentation, and tool-using recall.

## Why AgentLife?

Traditional memory benchmarks such as LoCoMo and LongMemEval are useful, but
they primarily score retrospective QA-style recall. AgentLife is designed for
agentic lifecycle behavior under real operating conditions:

- long-running multi-session interaction
- context resets between sessions
- retrieval and synthesis under context pressure
- conflicting/stale facts over time
- project-state continuity across sessions

AgentLife is the primary release-gate KPI in this repo; external memory benches
are supporting signals.

## Dataset

20 scripted conversation sessions between **Maya** (a product manager in Austin) and an AI assistant, spanning March–May 2026:

- **Track 1 (Personal):** Partner David, dog Biscuit, running training, mom Linda's diabetes, job transition from TechFlow to Stripe, house hunting
- **Track 2 (Project):** Recipe app development — Express/SQLite → dietary tags → SQL injection fix → GraphQL → Docker → JWT auth; portfolio site

All characters and events are fictional.

### Benchmark Variants

| Variant | Description |
| ------- | ----------- |
| **AL-S (AgentLife Short)** | Core corpus only (~100K tokens, 20 arc sessions). |
| **AL-L (AgentLife Long)** | AL-S plus filler sessions (~200K tokens) to force context-pressure behavior. |
| **AL-L OBD (One Big Day)** | AL-L data compressed into a one-day ingest path to stress heavy-session load. |
| **FC (Full Context)** | No memory system; answer model sees raw/compacted transcript each query. |
| **OC Native** | OpenClaw built-in memory baseline (memory-core/session-memory/session-index). |

### Eval Queries

| Tier | Count | What It Tests |
|------|-------|---------------|
| **T1-T4** | 268 | Factual recall, temporal reasoning, cross-reference synthesis, adversarial/stale facts, architecture/project-state |
| **T5** | 15 | Emotional intelligence and relational boundary handling (separate rubric) |

Each query includes ground truth, evidence sessions, query type, and recall difficulty.

## Launch Headline Results

Headline launch comparison (`Quaid Sonnet/Haiku` vs strongest FC Sonnet and OC native baselines):

| Lane | Quaid Sonnet/Haiku | FC Sonnet | OpenClaw Native |
| --- | ---: | ---: | ---: |
| AL-S | 87.69% (`r880`) | 92.90% (`r606`) | 69.40% (`oc-native-als-20260315d`) |
| AL-L | 85.82% (`r895`) | 87.70% (`r857`) | 63.06% (`oc-native-all-20260315d`) |

Additional launch note:

- On AL-L Sonnet-eval study, Quaid reaches **88.69%** (`r944`), above AL-L FC Sonnet at **87.70%** (`r857`).

Canonical docs for full tables and methodology:

- Public overview: [`docs/AGENTLIFE_PUBLIC.md`](docs/AGENTLIFE_PUBLIC.md)
- Technical report: [`published/runbooks/AGENTLIFE_TECHNICAL_REPORT.md`](published/runbooks/AGENTLIFE_TECHNICAL_REPORT.md)
- Hosted slugs:
  - `https://quaid.ai/benchmarks/agentlife`
  - `https://quaid.ai/benchmarks/agentlife/technical-report`

## Quick Start

### Branch Workflow

- Canonical branch for active development and publish is `main`.
- `agentlife` is retained only as historical compatibility context.
- For local migration from older branch state, run:

```bash
./scripts/adopt-main-workflow.sh origin
```

For canonical push flow to GitHub:

```bash
./scripts/push-main.sh origin
```

### Prerequisites

- Python 3.11+
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI
- OpenAI API key (for GPT-4o-mini judge)
- A memory system to evaluate

### Setup

```bash
git clone https://github.com/quaid-labs/agentlife.git
cd agentlife
cp .agentlife-benchmark.example.json .agentlife-benchmark.local.json
# Create secret files referenced by the local config
```

Canonical local setup is documented in
[docs/LOCAL-DEVELOPMENT.md](docs/LOCAL-DEVELOPMENT.md).

For quick local-only scripts that still read environment variables directly,
`.env.example` remains available as a convenience template.

### Benchmark OAuth Token

For benchmark direct Anthropic API runs, prefer an explicit benchmark token
path in `.agentlife-benchmark.local.json` instead of relying on local Claude
Code login state:

1. Generate a token with:
   ```bash
   claude setup-token
   ```
2. Put the token in a local secret file.
3. Point `auth.anthropic.primaryKeyPath` at that file in
   `.agentlife-benchmark.local.json`.

Notes:
- this is benchmark-only harness behavior
- the launcher prefers `.agentlife-benchmark.local.json`
- legacy fallback to `~/quaid/dev/.quaid-dev.local.json` still exists for local
  compatibility
- do not add automatic fallback across multiple Anthropic OAuth accounts/tokens
- secondary token switching is manual-only by operator action

### Run Evaluation

```bash
# Parse dataset and show query stats
python eval/dataset.py --stats

# Run evaluation against a memory system
python eval/evaluate.py \
    --sessions data/sessions/ \
    --queries all \
    --results-dir data/results/my-system/

# Score results
python eval/metrics.py data/results/my-system/
```

### Generate Filler Sessions (AgentLife L)

```bash
python eval/densify.py --count 259 --output data/filler-sessions/
```

### Rolling Replay Utilities

For long-transcript stress lanes that should exercise the real Quaid daemon path
instead of the normal per-day ingest path:

- `python scripts/export-imported-claude-history.py ...`
  - Export a raw Claude Code transcript into day-sliced JSONL plus a manifest.
- `python scripts/run-imported-claude-history.py ... --rolling`
  - Replay those exported days through Quaid workspace setup, rolling extraction,
    final flush, and janitor.

These utilities are intended for migration/stress work, not scored leaderboard
runs. See [docs/rolling-replay.md](docs/rolling-replay.md) for the manifest
schema, replay summary schema, and rolling telemetry surfaces.

## Repository Structure

```
agentlife/
├── README.md                  # Package overview
├── METHODOLOGY.md             # Benchmark methodology and lane definitions
├── SPEC.md                    # Original specification
├── LICENSE                    # MIT
├── .agentlife-benchmark.example.json
│   └── Local benchmark config template
├── .env.example               # Optional quick local env template
├── generate.py                # Session generation script
├── eval/                      # Benchmark scripts
│   ├── dataset.py             # Session parser + query collector
│   ├── evaluate.py            # Recall → answer → judge pipeline
│   ├── metrics.py             # Scoring and reporting
│   ├── extract_compact.py     # Extraction prompts + storage
│   ├── vm_benchmark.py        # Multi-system VM orchestrator
│   ├── densify.py             # Filler session generator
│   ├── session_splitter.py    # Timeout-based session splitting
│   └── ...                    # Analysis and utility scripts
├── scripts/                   # Launch, monitor, release, and utility scripts
├── docs/                      # Operational docs
├── published/                 # Release-ready runbooks and frozen public artifacts
├── data/
│   ├── sessions/              # 20 arc session transcripts
│   ├── filler-sessions/       # 259 filler sessions (AgentLife L)
│   └── timestamps/            # Session timing metadata
├── benchmark-assets/          # Session-scoped project asset snapshots used by eval/runtime
├── apps/                      # Reference project implementations
│   ├── recipe-app/            # Express.js + SQLite recipe app
│   └── portfolio-site/        # Static portfolio site
└── briefs/                    # Session generation briefs
```

## Benchmark Package Boundaries

This repo contains the benchmark harness package.

- canonical harness path: `eval/`
- canonical operational scripts: `scripts/`
- release-ready benchmark artifacts: `published/`
- packaged project snapshots for session-aware eval context: `benchmark-assets/`

Quaid runtime intelligence is benchmarked from a sibling checkout, not from
this repo.

There is also a legacy `agentlife/eval/` mirror in the tree for older
compatibility paths. It is not the primary harness entrypoint for release docs.

## Adding a New System

To benchmark your memory system:

1. **Implement an adapter** following one of the existing adapters in `eval/`
2. **Expose two functions:**
   - `ingest(session_text: str)` — Process a conversation session
   - `recall(query: str) -> str` — Answer a question using stored memories
3. **Run evaluation:** `python eval/evaluate.py --adapter your_adapter`
4. **Submit results** via PR to update the leaderboard

See [METHODOLOGY.md](METHODOLOGY.md) for complete details on scoring methodology and reproduction steps.

## Eval Query Format

Queries are embedded in session transcript files (Section 4):

```
Q1: What is Maya's partner's name?
Ground Truth: David. They've been together for about 3 years.
Evidence Sessions: [1, 6, 11]
Query Type: factual_recall
Recall Difficulty: Easy
```

## Citation

If you use AgentLife in your research:

```bibtex
@misc{steadman2026agentlife,
  title={AgentLife: A Full-Lifecycle Benchmark for AI Agent Memory Systems},
  author={Solomon Steadman},
  year={2026},
  url={https://github.com/quaid-labs/agentlife}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Notes

- Public benchmark claims in this repo should stay aligned with the release
  runbooks stored under `published/`.
- Historical experiment artifacts and local scratch runs may still exist in the
  working tree, but they should not leak into release docs or tarballs.
