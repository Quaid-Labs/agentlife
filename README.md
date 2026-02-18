# AgentLife

A full-lifecycle benchmark for AI agent memory systems. Unlike extraction-only benchmarks, AgentLife tests the **complete production pipeline**: extraction, deduplication, contradiction resolution, core file evolution, project documentation, and agent-driven recall with tool use.

## Why AgentLife?

Existing memory benchmarks (LoCoMo, LongMemEval) test extraction and recall in isolation. But production memory systems have janitors, dedup pipelines, document updaters, and decay mechanisms that all affect what an agent actually remembers. AgentLife tests the whole chain.

**Key innovations:**
- Tests the **complete memory lifecycle**, not just store-and-retrieve
- Interleaves personal conversations with project development sessions
- Two scales: S (20 sessions, ~92K tokens) and L (279 sessions, ~423K tokens)
- Filler sessions force natural compaction behavior in context-window-based systems
- 234 eval queries across 5 tiers including emotional intelligence
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

### Eval Queries (234 total)

| Tier | Count | What It Tests |
|------|-------|---------------|
| **Tier 1: Core** | 120 | Factual recall, temporal reasoning, cross-references, project state |
| **Tier 2: Adversarial** | 72 | Contested facts, stale info, speaker attribution, false premises |
| **Tier 3: Non-Question** | 12 | "Hi", "Thanks" — should NOT trigger memory retrieval |
| **Tier 4: Architecture** | 15 | Project knowledge for development tasks |
| **Tier 5: Emotional Intelligence** | 15 | Boundary awareness, emotional context, self-awareness |

Each query includes ground truth, evidence sessions, query type, and recall difficulty.

## Results

### AgentLife S (20 sessions, no filler)

| System | Accuracy | Correct | Wrong | Notes |
|--------|----------|---------|-------|-------|
| **FC-Sonnet** | 90.0% | 197 | 22 | Full-context upper bound |
| **Quaid v12** | 80.8% | 177 | 42 | Per-day extraction, full janitor pipeline |
| **Mem0** | 47.0% | 103 | 116 | Per-message-pair, GPT-4o-mini extraction |

### AgentLife L (279 sessions, 259 filler) — In Progress

| System | Accuracy | Notes |
|--------|----------|-------|
| **Base** | 11.4% | Raw context only, 2 compactions |
| **Quaid** | Running | Timeout-based extraction, ~117 chunks |
| **Mem0** | Running | Per-message-pair, 279 sessions |
| **QMD** | Pending | OpenClaw built-in memory |

## Quick Start

### Prerequisites

- Python 3.11+
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI (for extraction via `claude -p`)
- OpenAI API key (for GPT-4o-mini judge and Mem0)
- A memory system to evaluate

### Setup

```bash
git clone https://github.com/Steadman-Labs/agentlife.git
cd agentlife
cp .env.example .env
# Edit .env with your API keys
```

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

### VM-Based Benchmark (Full Pipeline)

For systems that need a complete environment (gateway, plugins, compaction):

```bash
python eval/vm_benchmark.py \
    --system quaid \
    --splitting timeout \
    --vm-ip 192.168.64.3 \
    --vm-user admin
```

## Repository Structure

```
agentlife/
├── README.md                  # This file
├── METHODOLOGY.md             # Full methodology (publication-ready)
├── SPEC.md                    # Original specification
├── LICENSE                    # MIT
├── .env.example               # Configuration template
├── generate.py                # Session generation script
├── eval/                      # Benchmark scripts
│   ├── dataset.py             # Session parser + query collector
│   ├── evaluate.py            # Recall → answer → judge pipeline
│   ├── metrics.py             # Scoring and reporting
│   ├── extract_compact.py     # Extraction prompts + storage
│   ├── mem0_adapter.py        # Mem0 integration
│   ├── vm_benchmark.py        # Multi-system VM orchestrator
│   ├── densify.py             # Filler session generator
│   ├── session_splitter.py    # Timeout-based session splitting
│   └── ...                    # Analysis and utility scripts
├── data/
│   ├── sessions/              # 20 arc session transcripts
│   ├── filler-sessions/       # 259 filler sessions (AgentLife L)
│   └── timestamps/            # Session timing metadata
├── apps/                      # Reference project implementations
│   ├── recipe-app/            # Express.js + SQLite recipe app
│   └── portfolio-site/        # Static portfolio site
└── briefs/                    # Session generation briefs
```

## Adding a New System

To benchmark your memory system:

1. **Implement an adapter** following `eval/mem0_adapter.py` as a template
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
  url={https://github.com/Steadman-Labs/agentlife}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Related

- [Quaid](https://github.com/Steadman-Labs/quaid) — The memory system that motivated this benchmark
- [LoCoMo](https://github.com/snap-research/locomo) — Conversational memory benchmark (extraction+recall only)
- [LongMemEval](https://github.com/xiaowu0162/LongMemEval) — Long-term memory evaluation from ICLR 2025
