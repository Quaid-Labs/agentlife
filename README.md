# AgentLife

A full-lifecycle benchmark for AI agent memory systems. AgentLife tests the
full production memory pipeline: extraction, rolling carryover, janitor review,
deduplication, project documentation, and tool-using recall.

## Why AgentLife?

The public docs in this repo now mirror the current benchmark runbook instead
of carrying a long narrative of older experiments. If a public number changes
here, the clean runbook matrix changed first.

**Key innovations:**
- Tests the **complete memory lifecycle**, not just store-and-retrieve
- Interleaves personal conversations with project development sessions
- Two scales: S (20 sessions, ~92K tokens) and L (279 sessions, ~423K tokens)
- Filler sessions force natural compaction behavior in large-scale lanes
- Current canonical scoring uses **268 mainline T1-T4 queries** plus a separate **30-query Tier 5** lane
- Cross-vendor GPT-4o-mini judge for unbiased scoring

## Dataset

20 scripted conversation sessions between **Maya** (a product manager in Austin) and an AI assistant, spanning March–May 2026:

- **Track 1 (Personal):** Partner David, dog Biscuit, running training, mom Linda's diabetes, job transition from TechFlow to Stripe, house hunting
- **Track 2 (Project):** Recipe app development — Express/SQLite → dietary tags → SQL injection fix → GraphQL → Docker → JWT auth; portfolio site

All characters and events are fictional.

### Scales

| Scale | Arc Sessions | Filler Sessions | Total | Tokens | Use Case |
|-------|-------------|-----------------|-------|--------|----------|
| **AgentLife S** | 20 | 0 | 20 | ~92K | Quick iteration, no compaction pressure |
| **AgentLife L** | 20 | 259 | 279 | ~423K | Production-realistic with compaction events |

### Eval Queries

| Tier | Count | What It Tests |
|------|-------|---------------|
| **Tier 1: Core** | part of 268 | Factual recall, temporal reasoning, cross-references, project state |
| **Tier 2: Adversarial** | part of 268 | Contested facts, stale info, speaker attribution, false premises, negative/control queries |
| **Tier 3: Non-Question** | part of 268 | "Hi", "Thanks" — should NOT trigger memory retrieval |
| **Tier 4: Architecture** | part of 268 | Project knowledge and implementation planning for development tasks |
| **Tier 5: Emotional Intelligence** | 30 | Boundary awareness, emotional context, self-awareness (scored separately) |

Each query includes ground truth, evidence sessions, query type, and recall difficulty.

## Results

Public release numbers no longer live inline in the root README because they
go stale too quickly during active rerun work.

Use these tracked locations instead:

- public release-ready summaries: [`published/runbooks/`](published/runbooks/)
- frozen public-supporting artifacts: [`published/checkpoints/`](published/checkpoints/)
- methodology and lane definitions: [`METHODOLOGY.md`](METHODOLOGY.md)

Internal working matrices may move faster than the released package. Only copy
numbers into `published/` after they have been reviewed for release.

## Quick Start

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
