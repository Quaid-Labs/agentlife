# AgentLife Dataset

## Directory Structure

### `sessions/`
20 arc session transcripts (`session-XX-review-vY.txt`). Each file contains 4 sections:
1. **Session Brief** — Goal, turn count, learning objectives
2. **Generated Transcript** — Full MAYA/ASSISTANT conversation
3. **Leakage Check** — Verification no future info leaked
4. **Eval Queries** — Ground truth Q&A pairs for evaluation

### `filler-sessions/`
259 filler sessions for AgentLife L scale. Categories:
- **A (Quick):** Brief interactions (weather, reminders, calculations)
- **B (Work):** Professional discussions (meetings, reports, presentations)
- **C (Recipe-App):** Project-related but non-arc (generic coding questions)
- **D (Personal):** Life topics (health, hobbies, events)
- **E (Callbacks):** Reference arc events without adding new info

### `timestamps/`
Session timing metadata for realistic temporal simulation:
- `timestamps-S.json` — 20 arc sessions only
- `timestamps-L.json` — All 279 sessions (20 arc + 259 filler)

Used by `session_splitter.py` for timeout-based extraction chunking.

## Format

Session transcripts use a simple text format:
```
MAYA: [user message]
ASSISTANT: [assistant response]
```

Eval queries use structured format:
```
Q1: Question text?
Ground Truth: Expected answer
Evidence Sessions: [session numbers]
Query Type: factual_recall
Recall Difficulty: Easy
```
