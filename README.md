<p align="center">
  <img src="assets/agentlife-crop-feather.png" alt="AgentLife" width="500">
</p>

# AgentLife

A full-lifecycle benchmark for AI agent memory systems. AgentLife tests the
full agentic pipeline: days and months of data, projects, conversations,
evolving context, and a user whose story changes over time.

## Why AgentLife?

AgentLife is designed for agentic lifecycle behavior under real operating
conditions:

- long-running multi-session interaction
- context resets between sessions
- retrieval and synthesis under context pressure
- conflicting/stale facts over time
- project-state continuity across sessions
- assistant-style judgment about what to surface and what not to surface

AgentLife is the primary release-gate KPI for the Quaid project; external memory
benchmarks are supporting signals for the development of personalized agentic memory.

Full-context baselines are useful upper bounds for short-horizon tasks, but
they grow linearly in cost and do not persist state across session resets.
AgentLife is built to test the regime where persistent knowledge and continuity
matter.

It also covers categories that most memory benchmarks skip entirely: non-question
restraint, emotional intelligence, privacy boundaries, hallucination resistance,
and adversarial "don't make things up" behavior. That makes it a benchmark for
assistant-like agentic systems, not just a benchmark for retrieval. It measures
both what a system should share and what it should not.

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
| **FC (Full Context)** | No memory system; answer model sees raw/compacted transcript each query, but FC does not persist state across session resets. |
| **OC Native** | OpenClaw built-in memory baseline (memory-core/session-memory/session-index). |

### Eval Queries

The canonical query set is `283` scored prompts. Broken out by tier:

| Tier | Count | What It Tests |
|------|------:|---------------|
| **T1** | 201 | Personal memory and long-horizon continuity: fact retention, updates over time, synthesis, callbacks, relationship reasoning, and adversarial resistance on the user's changing life story. |
| **T2** | 34 | Project and tool-derived memory: project-state tracking plus facts the agent found during research/tool use. |
| **T3** | 16 | Non-question grounding and restraint on casual chat. |
| **T4** | 17 | Architecture comprehension and implementation planning. |
| **T5** | 15 | Emotional intelligence, privacy boundaries, relational context, and socially appropriate responses under memory pressure. |

#### Category Breakdown

| Tier | Category | Count | What It Tests |
|------|----------|------:|---------------|
| **T1** | **factual_recall** | 29 | Basic extraction and retention of directly stated facts. |
| **T1** | **temporal_current** | 16 | Current-state reasoning after updates, corrections, and time passing. |
| **T1** | **graph_traversal** | 14 | Multi-hop relationship recall through the memory graph. |
| **T1** | **multi_session_synthesis** | 14 | Combining information across many sessions into one answer. |
| **T1** | **cross_reference** | 13 | Linking facts across different arcs or domains. |
| **T1** | **inference** | 13 | Inferring the right answer from multiple stored facts. |
| **T1** | **stale_fact** | 12 | Replacing outdated memories with the latest valid state. |
| **T1** | **negative** | 12 | Hallucination resistance when the correct answer is no or not established. |
| **T1** | **evolution** | 12 | Tracking how a person, project, or situation changes over time. |
| **T1** | **adversarial_confirm** | 11 | Confirming true facts even when the phrasing is adversarial or trap-like. |
| **T1** | **surprise_callback** | 11 | Long-range recall of early details that return much later. |
| **T1** | **speaker_attribution** | 10 | Distinguishing who said, suggested, or did what. |
| **T1** | **contested_fact** | 10 | Facts with corrections, changing states, or multiple valid positions. |
| **T1** | **adversarial_false_attribution** | 10 | Resisting entity confusion and wrong-person/source attribution. |
| **T1** | **adversarial_idk** | 8 | Saying unknown when the corpus does not support an answer. |
| **T1** | **tangent_recall** | 6 | Pulling relevant facts out of side remarks and conversational noise. |
| **T2** | **project_state** | 24 | Current state of active projects, tools, and implementation work. |
| **T2** | **agent_retrieved** | 10 | Facts the agent found or established through tool use or research. |
| **T3** | **non_question** | 16 | Appropriate restraint on casual utterances without dumping memory. |
| **T4** | **arch_comprehension** | 11 | Architectural/codebase understanding from remembered project history. |
| **T4** | **arch_planning** | 6 | Implementation planning that depends on remembered architecture. |
| **T5** | **emotional_intelligence** | 15 | Empathy, privacy boundaries, relational context, and socially appropriate responses under memory pressure. |

Each query includes ground truth, evidence sessions, query type, and recall difficulty.

## Launch Headline Results

FC is included here as an upper-bound baseline, not as the target operating
model. The question is not "can memory beat raw transcript in every short
horizon case," but whether a persistent system can stay competitive while
surviving resets and reducing long-run token cost.

Headline launch summary:

| Metric | Quaid | FC Sonnet | OpenClaw Native |
| --- | ---: | ---: | ---: |
| AL-S | 87.69% | 92.90% | 69.40% |
| Tokens | 5.75M | 29.83M | unknown |
| AL-L | 87.10% | 87.70% | 63.06% |
| Tokens | 6.46M | 34.60M | unknown |
| AL-L OBD | 86.04% | 87.70% | unknown |
| Tokens | 6.08M | 34.60M | unknown |

Quaid was measured with Haiku fast, Sonnet deep, and a Sonnet agent running
eval. `AL-L` and `AL-L OBD` are chosen here as the best representation of real
use data; `AL-S` remains the cleaner, more idealized lane and a full Sonnet-eval
row for it is planned before public launch. `Sonnet/Haiku` remains the flagship
configuration on cleanliness and overall benchmark tradeoffs. `Opus` was
evaluated, but underperformed `Sonnet` overall and is not the recommended
launch configuration. On `AL-L` and `AL-L OBD`, FC is forced to compact, and
the drop in FC quality reflects that compaction plus the added noise in the
larger corpus. OpenClaw Native tokens remain unknown due to telemetry
restrictions. Token counts here are the minimum tokens used to answer the full
set of 283 eval questions.

Benchmark note:

- Results are measured on synthetic high-density conversations designed to stress memory systems.
- Public rows are single-run per lane/configuration; informal repeat variance on stable configs has typically been about `+-1pp`.

Canonical docs for full tables and methodology:

- Technical report: [`published/runbooks/AGENTLIFE_TECHNICAL_REPORT.md`](published/runbooks/AGENTLIFE_TECHNICAL_REPORT.md)

## Quick Start

### Branch Workflow

- Canonical branch for development and publish is `main`.
- If you are carrying an older local checkout from the previous branch model, run:

```bash
./scripts/adopt-main-workflow.sh origin
```

Canonical push flow to GitHub:

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
- no legacy config fallback is supported in launch mode
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

### Generate Additional Filler Sessions (Optional)

The repo already includes the canonical AgentLife L filler corpus in
`data/filler-sessions/`. You only need this if you want to generate more filler
sessions or build a variant large-scale lane.

```bash
python eval/densify.py --count 259 --output data/filler-sessions-extra/
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
