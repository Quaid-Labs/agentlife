<p align="center">
  <img src="assets/agentlife-crop-feather.png" alt="AgentLife" width="500">
</p>

# AgentLife

A full-lifecycle benchmark for AI agent memory systems and other agent
architectures that must operate over time. AgentLife tests the full agentic
pipeline: days and months of data, projects, conversations, evolving context,
and a user whose story changes over time.

## Benchmark Comparison

The tables below position AgentLife against two widely cited long-memory
benchmarks. The goal is to clarify benchmark structure and what each dataset
actually measures, not to argue from rhetoric.

### Dataset Scale

| Dimension | LoCoMo | LongMemEval-S | LongMemEval-M | AgentLife S | AgentLife L |
| --- | --- | --- | --- | --- | --- |
| Conversations | 10 | 500 (per-question haystacks) | 500 (per-question haystacks) | 1 (continuous narrative) | 1 (continuous narrative) |
| Sessions | 19–32 per conversation | ~30–50 per question | ~500 per question | 20 | 279 (20 arc + 259 filler) |
| Tokens | 9K–26K per conversation | ~115K per question | ~1.5M per question | ~92K | ~423K |
| Eval questions | 1,986 QA pairs across 10 conversations | 500 | 500 | 234 | 234 |
| Fits in context window? | Yes (200K) | Yes (200K) | No | Yes (200K) | No |

### What Each Benchmark Tests

Table semantics:
- `✅` = explicitly targeted by the benchmark's evaluated task design
- `❌` = absent as a first-class benchmark target, not a claim that no incidental example exists

| Capability | LoCoMo | LongMemEval | AgentLife |
| --- | --- | --- | --- |
| Single-fact recall | ✅ | ✅ | ✅ |
| Multi-hop reasoning | ✅ | ✅ (multi-session) | ✅ (cross-reference) |
| Temporal reasoning | ✅ | ✅ | ✅ |
| Knowledge updates / stale facts | ❌ | ✅ | ✅ (contested_fact, stale_fact tiers) |
| Contradiction detection | ❌ | ❌ | ✅ (facts evolve and contradict across sessions) |
| Adversarial / unanswerable | ✅ (446 questions) | ✅ (abstention) | ✅ (adversarial_idk, adversarial_confirm, false_attribution) |
| Project / technical state | ❌ | ❌ | ✅ (24 project_state questions, 15 architecture tier) |
| Speaker attribution | ❌ | ❌ | ✅ (who said what) |
| Non-question handling | ❌ | ❌ | ✅ (“Hi”, “Thanks” should not trigger recall) |
| Emotional intelligence | ❌ | ❌ | ✅ (Tier 5: boundary awareness, self-awareness) |
| Agent self-knowledge | ❌ | ❌ | ✅ (agent_retrieved: what did the agent do or suggest) |
| Compaction pressure | ❌ | ❌ (static haystack) | ✅ (AgentLife L forces compaction events) |
| Maintenance pipeline testing | ❌ | ❌ | ✅ (dedup, decay, journal distillation affect results) |
| Interleaved personal + project | ❌ | ❌ | ✅ (Track 1 personal, Track 2 project, interleaved) |

### Structural Limitations

| Limitation | LoCoMo | LongMemEval | AgentLife |
| --- | --- | --- | --- |
| Context window bypass | Entire conversation can be bypassed with full-context on modern 200K models | LongMemEval-S can be bypassed with full-context on modern 200K models | AgentLife S fits; AgentLife L does not, so long runs force real retrieval |
| Narrative coherence | 10 independent conversations with no cross-conversation continuity | Per-question isolated haystacks with no shared narrative | One continuous character arc across 20 sessions, where facts evolve, contradict, and resolve |
| Maintenance sensitivity | Maintenance pipeline effects are absent as a benchmark target | Maintenance pipeline effects are absent as a benchmark target | Simulated day cycles and timestamps exercise dedup, contradiction handling, decay, and temporal recall under changing state |
| What high scores prove | Strong contained conversational memory and reasoning | Strong long-haystack retrieval and reasoning | Lifecycle stability under changing state: extraction, maintenance, and recall all have to hold |
| Metric type | Token F1, exact match, ROUGE, LLM judge | LLM judge (GPT-4o) | LLM judge (GPT-4o-mini, cross-vendor) |

### Core Distinction

LoCoMo and LongMemEval primarily test store-and-retrieve behavior: can a system
recover information from a conversation history or a large haystack. That is an
important capability, and those benchmarks are useful for long-memory retrieval
and reasoning. But they do not by themselves test the maintenance operations
that change what a long-running agent remembers.

AgentLife is designed to test the full lifecycle: extraction, maintenance, and
recall. Facts evolve across sessions, stale information must be replaced,
contradictions must be resolved, and project-state memory is interleaved with
personal narrative. As a result, high performance on AgentLife says more than
"the model found the right passage." It says the memory pipeline stayed stable
while the remembered world changed over time.

That lifecycle pressure is not only conversational. AgentLife advances through
simulated day cycles and explicit timestamps, so maintenance logic and temporal
reasoning are exercised together. The benchmark therefore tests whether a
system can preserve the right facts, retire the wrong ones, and still answer
time-sensitive questions correctly after the world state has changed.

## Why AgentLife?

AgentLife is designed for agentic lifecycle behavior under real operating
conditions:

- long-running multi-session interaction
- context resets between sessions
- retrieval and synthesis under context pressure
- conflicting/stale facts over time
- project-state continuity across sessions
- assistant-style judgment about what to surface and what not to surface

AgentLife was created to evaluate systems like Quaid — persistent knowledge
layers that must survive resets, continuity breaks, and rising context cost —
but the benchmark itself is intended to evaluate any agent architecture
operating beyond a single context window.

For Quaid, AgentLife is the primary release-gate KPI. External memory
benchmarks remain supporting signals for the development of personalized
agentic memory.

Full-context baselines are useful short-horizon upper bounds, but they grow
linearly in cost and do not persist state across session resets. AgentLife is
built to test the regime where persistence, cost scaling, and cross-session
continuity decide whether an agent system actually holds up over time.

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

## Current Headline Results

FC is included here as an upper-bound baseline, not as the target operating
model. The question is not "can memory beat raw transcript in every short
horizon case," but whether a persistent system can stay competitive while
surviving resets and reducing long-run token cost.

These headline rows are Quaid's numbers on the clean benchmark harness. They
should be read as the current theoretical maximum for the same memory system
without platform harness execution-path noise.

| Surface | Quaid | FC Sonnet | Quaid Tokens | FC Tokens |
| --- | ---: | ---: | ---: | ---: |
| AgentLife Short | 93.64% | 93.11% | 7.95M | 29.83M |
| AgentLife Long | 88.52% | 88.69% | 9.64M | 26.50M |
| AgentLife Long OBD | 88.69% | 88.69% | 8.45M | 26.50M |

Quaid was measured with Haiku fast, Sonnet deep, and a Sonnet answer model on
the clean harness. AgentLife Short is the strongest current direct lane.
AgentLife Long and AgentLife Long OBD are the more realistic long-context
lanes. `Opus 4.7` was also evaluated, but `Sonnet` remains the cleaner
headline configuration for Quaid.

### OpenClaw Execution Results

These rows are not clean-harness theoretical ceilings. They are affected by the
OpenClaw execution path itself, and we are actively working to improve them.

| Surface | OpenClaw Native | Quaid on OpenClaw |
| --- | ---: | ---: |
| AgentLife Short | 26.49% | 80.97% |
| AgentLife Long | 31.72% | pending refresh |

The main point of this split is:

- the headline Quaid table above is the clean reference surface
- the OpenClaw table measures the extra execution-path tax imposed by OC
- we intentionally do not publish OC token numbers here because they are not
  yet measured cleanly enough to be trustworthy
- the native OC rows here were run with OpenClaw's built-in memory plugins
  enabled: `memory-core`, `session-memory`, and `session-index`
- the next required OC datapoint is a fresh Quaid-on-OpenClaw AgentLife Long
  lane so the OC table is complete on the same footing as the clean-harness
  block

### Multilingual Preview

On the first validated full Japanese AgentLife Short harness run, Quaid
handled a Kanji/Kana benchmark end-to-end at `74.73%` overall (`73.88%` on
`T1-T4`).

We are still actively improving multilingual retrieval and documentation
surfaces, so this should be read as a preview rather than a headline claim. But
it is already strong enough to show that non-Roman-script operation is viable
at real benchmark scale, not just in isolated probes.

Token accounting standard (public, effective April 18, 2026):

- Public token rows now use `eval_tokens_ex_judge` from `token_usage.json`:
  - `eval_tokens_ex_judge = eval.total_tokens - sum(by_source[*judge*].total_tokens)`
- This includes answer + preinject + tool-recall token spend, while excluding
  judge spend from headline token totals.
- Older historical rows that cite `evaluation_results.json` per-question
  `eval_tokens` are legacy answer-only accounting and are not directly
  comparable to the current standard.

Historical token-cost reference from the March 29 technical report / Sonnet study:
- `AL-S`: Quaid Sonnet/Haiku (`r880`, `r847`) reached `87.69%` at `5,753,673` eval tokens, versus FC Sonnet (`r606`) at `92.90%` and `29,828,646` eval tokens.
- `AL-L`: Quaid Sonnet/Haiku (`r895`, `r863`) reached `85.82%` at `5,917,209` eval tokens, versus FC Sonnet (`r857`) at `87.70%` and `34,596,206` eval tokens.
- Sonnet eval study: `AL-L` Haiku-ingest + Sonnet-eval (`r944`) reached `88.69%` at `8,382,952` eval tokens, versus the same FC Sonnet row at `34,596,206`.

Those rows are the citation basis for the claim that Quaid can stay near
full-context quality at roughly one-fifth FC token cost on long-form lanes.

Benchmark note:

- Results are measured on synthetic high-density conversations designed to stress memory systems.
- Public rows are single-run per lane/configuration; informal repeat variance on stable configs has typically been about `+-1pp`.

Canonical docs for full tables and methodology:

- Latest technical report:
  - [`published/runbooks/AGENTLIFE_TECHNICAL_REPORT_20260430.md`](published/runbooks/AGENTLIFE_TECHNICAL_REPORT_20260430.md)
- Runbooks folder:
  - [`published/runbooks/`](published/runbooks/)

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
