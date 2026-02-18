#!/usr/bin/env python3
"""AgentLife Benchmark — Realistic Pipeline Injector.

Simulates real OpenClaw+Quaid usage by:
1. Writing pre-written conversations as session JSONL (both user + assistant)
2. Tracking token accumulation per session
3. Triggering compaction at realistic thresholds (80% context)
4. Running full janitor after compaction (review, dedup, snippets, journal)
5. Building up core markdown naturally (USER.md, SOUL.md equivalents)
6. Tracking cost per session (with and without compaction)

A/B Variants:
  natural   — Compact only when session hits 80% context threshold
  nightly   — Auto-compact after each day's sessions

Usage:
    # Natural compaction (realistic)
    python3 injector.py --mode natural --results-dir ../data/results-pipeline

    # Nightly compaction (proposed feature)
    python3 injector.py --mode nightly --results-dir ../data/results-pipeline-nightly

    # Compare cost curves
    python3 injector.py --mode cost-analysis
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_DIR = Path(__file__).resolve().parent
_WORKSPACE = Path(os.environ.get("CLAWDBOT_WORKSPACE", Path.home() / "clawd"))
_QUAID_DIR = _WORKSPACE / "plugins" / "quaid"

sys.path.insert(0, str(_DIR))
if str(_QUAID_DIR) not in sys.path:
    sys.path.insert(0, str(_QUAID_DIR))

from dataset import load_all_reviews, load_filler_reviews, merge_sessions_chronologically

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# OpenClaw defaults
CONTEXT_WINDOW = 200_000  # Haiku context
COMPACTION_THRESHOLD = 0.80  # 80% = 160K tokens
COMPACTION_TOKEN_LIMIT = int(CONTEXT_WINDOW * COMPACTION_THRESHOLD)

# Model pricing ($ per million tokens, as of Feb 2026)
MODEL_PRICING = {
    "haiku": {"input": 1.00, "output": 5.00},
    "sonnet": {"input": 3.00, "output": 15.00},
    "opus": {"input": 15.00, "output": 75.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}

# Token budget estimates per compaction event
EXTRACTION_INPUT_TOKENS = 5000   # Transcript + system prompt
EXTRACTION_OUTPUT_TOKENS = 2000  # Facts + snippets + journal

# Janitor token budget per run (review + dedup, ~2 Opus calls)
JANITOR_INPUT_TOKENS = 8000
JANITOR_OUTPUT_TOKENS = 3000


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    def count_tokens(text: str) -> int:
        return len(text.split()) * 4 // 3  # Rough estimate


# ---------------------------------------------------------------------------
# Session JSONL management
# ---------------------------------------------------------------------------

def transcript_to_messages(review) -> List[dict]:
    """Convert a SessionReview transcript into JSONL message format.

    Uses the structured transcript_turns (list of dicts with 'maya'/'agent' keys).
    """
    messages = []
    for turn in review.transcript_turns:
        if "maya" in turn and turn["maya"].strip():
            messages.append({"role": "user", "content": turn["maya"].strip()})
        if "agent" in turn and turn["agent"].strip():
            messages.append({"role": "assistant", "content": turn["agent"].strip()})
    return messages


def write_session_jsonl(messages: List[dict], path: Path):
    """Write messages in OpenClaw session JSONL format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for msg in messages:
            f.write(json.dumps({"role": msg["role"], "content": msg["content"]}) + "\n")


def session_token_count(messages: List[dict]) -> int:
    """Count total tokens across all messages (simulating context replay)."""
    return sum(count_tokens(m["content"]) for m in messages)


# ---------------------------------------------------------------------------
# Core markdown workspace setup
# ---------------------------------------------------------------------------

def setup_workspace(results_dir: Path, owner_name: str = "Maya Chen"):
    """Create core markdown files for the benchmark persona.

    These files start empty and build up through snippets/journal.
    """
    workspace = results_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    # Core markdown files (minimal starters)
    user_md = workspace / "USER.md"
    if not user_md.exists():
        user_md.write_text(f"# About {owner_name}\n\n")

    soul_md = workspace / "SOUL.md"
    if not soul_md.exists():
        soul_md.write_text("# Assistant Personality\n\nI am Maya's AI assistant.\n")

    memory_md = workspace / "MEMORY.md"
    if not memory_md.exists():
        memory_md.write_text("# Core Memories\n\n")

    # Snippet accumulation files
    for md_file in ["USER.md", "SOUL.md", "MEMORY.md"]:
        snippet_file = workspace / f"{md_file}.snippets.md"
        if not snippet_file.exists():
            snippet_file.write_text(f"# Snippets for {md_file}\n\n")

    # Journal directory
    journal_dir = workspace / "journal"
    journal_dir.mkdir(exist_ok=True)
    for md_file in ["USER.md", "SOUL.md", "MEMORY.md", "SPEAKERS.md"]:
        journal_file = journal_dir / f"{md_file.replace('.md', '')}.journal.md"
        if not journal_file.exists():
            journal_file.write_text(f"# Journal — {md_file}\n\n")

    # Journal archive
    (journal_dir / "archive").mkdir(exist_ok=True)

    # Projects directory
    (workspace / "projects").mkdir(exist_ok=True)

    return workspace


# ---------------------------------------------------------------------------
# Extraction via real Quaid code
# ---------------------------------------------------------------------------

def run_extraction(
    session_file: Path,
    results_dir: Path,
    db_path: Path,
    owner_id: str = "maya",
    session_date: str = "2026-03-01",
    session_id: str = "benchmark",
    extract_model: str = "sonnet",
) -> dict:
    """Run extraction using the real Quaid extraction pipeline.

    Calls the Python extraction code (same as index.ts calls).
    Returns stats dict.
    """
    from memory_graph import MemoryGraph, store, recall
    from ingest import extract_session, store_session_facts, _switch_to_db

    # Read messages and build transcript
    messages = []
    with open(session_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                messages.append(msg)
            except json.JSONDecodeError:
                continue

    # Build transcript in extraction format
    transcript_parts = []
    for msg in messages:
        role = "Maya" if msg["role"] == "user" else "AI Assistant"
        content = msg["content"] if isinstance(msg["content"], str) else str(msg["content"])
        transcript_parts.append(f"{role}: {content}")
    transcript = "\n\n".join(transcript_parts)

    if not transcript.strip():
        return {"facts_stored": 0, "edges_created": 0, "error": "empty transcript"}

    # Switch to the benchmark DB
    _switch_to_db(db_path, fresh=False)

    # Extract
    extraction, duration = extract_session(transcript, 0, extract_model)
    if extraction is None:
        return {"facts_stored": 0, "edges_created": 0, "error": "extraction failed"}

    # Store facts
    stats = store_session_facts(extraction, owner_id, session_date, 0)

    return {
        "facts_stored": stats["facts_stored"],
        "edges_created": stats["edges_created"],
        "extraction_duration_s": duration,
    }


def run_snippet_extraction(
    session_file: Path,
    workspace: Path,
    extract_model: str = "sonnet",
) -> dict:
    """Extract snippets and journal entries from a session.

    Uses the real soul_snippets.py extraction (or simulates via LLM call).
    Returns stats dict.
    """
    # Read messages
    messages = []
    with open(session_file) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    messages.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Build transcript
    transcript_parts = []
    for msg in messages:
        role = "Maya" if msg["role"] == "user" else "AI Assistant"
        content = msg["content"] if isinstance(msg["content"], str) else str(msg["content"])
        transcript_parts.append(f"{role}: {content}")
    transcript = "\n\n".join(transcript_parts)

    if not transcript.strip():
        return {"snippets_written": 0, "journal_entries": 0}

    # Use claude -p to extract snippets + journal in one call
    from claude_backend import call_claude

    prompt = f"""Extract personality snippets and journal reflections from this conversation between Maya and her AI assistant.

Return JSON with two keys:
1. "soul_snippets": object mapping filename to array of bullet-point snippets
   - "USER.md": facts about Maya (biographical, preferences, relationships)
   - "SOUL.md": observations about how you should interact with Maya
   - "MEMORY.md": critical facts to always remember
2. "journal_entries": object mapping filename to paragraph-form reflections
   - "USER.md": biographical narrative about Maya
   - "SOUL.md": personality/interaction reflections
   - "SPEAKERS.md": summary of who was discussed

Only include entries where there's genuinely new information. Skip if nothing notable.

Conversation:
{transcript[:50000]}"""

    response, duration = call_claude(
        prompt=prompt,
        model=extract_model,
        timeout=90,
    )

    if not response:
        return {"snippets_written": 0, "journal_entries": 0}

    # Parse response
    try:
        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            data = json.loads(json_match.group())
        else:
            return {"snippets_written": 0, "journal_entries": 0}
    except json.JSONDecodeError:
        return {"snippets_written": 0, "journal_entries": 0}

    snippets_written = 0
    journal_entries_written = 0
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Write snippets
    soul_snippets = data.get("soul_snippets", {})
    if isinstance(soul_snippets, dict):
        for filename, snippets in soul_snippets.items():
            if not isinstance(snippets, list) or not snippets:
                continue
            snippet_file = workspace / f"{filename}.snippets.md"
            if snippet_file.exists():
                existing = snippet_file.read_text()
                new_section = f"\n## Compaction — {now}\n"
                for s in snippets:
                    if isinstance(s, str) and s.strip():
                        new_section += f"- {s.strip()}\n"
                        snippets_written += 1
                snippet_file.write_text(existing + new_section)

    # Write journal entries
    journal_raw = data.get("journal_entries", {})
    if isinstance(journal_raw, dict):
        for filename, content in journal_raw.items():
            if not content:
                continue
            base = filename.replace(".md", "")
            journal_file = workspace / "journal" / f"{base}.journal.md"
            if journal_file.exists():
                existing = journal_file.read_text()
                text = content if isinstance(content, str) else " ".join(content)
                if text.strip():
                    new_entry = f"\n## {now} — Compaction\n{text.strip()}\n"
                    # Insert after header (newest at top)
                    lines = existing.split("\n", 1)
                    journal_file.write_text(lines[0] + "\n" + new_entry + (lines[1] if len(lines) > 1 else ""))
                    journal_entries_written += 1

    return {
        "snippets_written": snippets_written,
        "journal_entries": journal_entries_written,
        "duration_s": duration,
    }


# ---------------------------------------------------------------------------
# Janitor execution
# ---------------------------------------------------------------------------

def run_janitor(
    db_path: Path,
    workspace: Path,
    tasks: Optional[List[str]] = None,
    dry_run: bool = False,
) -> dict:
    """Run janitor tasks against the benchmark DB.

    Default tasks for post-compaction: embeddings, review, dedup, temporal, snippets
    """
    from ingest import _switch_to_db, run_janitor_pass

    _switch_to_db(db_path, fresh=False)

    tasks = tasks or ["embeddings", "review", "temporal", "duplicates"]
    results = {}

    for task in tasks:
        t0 = time.monotonic()
        try:
            run_janitor_pass([task])
            results[task] = {"status": "ok", "duration_s": round(time.monotonic() - t0, 1)}
        except Exception as e:
            results[task] = {"status": "error", "error": str(e), "duration_s": round(time.monotonic() - t0, 1)}

    return results


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

class CostTracker:
    """Track cumulative token spend for session replay and extraction.

    Tracks raw tokens — costs computed at summary time per model.
    """

    def __init__(self):
        self.session_tokens = 0           # Current session token accumulation
        self.total_replay_input_tokens = 0  # Cumulative input tokens for session replay
        self.compaction_count = 0
        self.messages_since_compaction = 0
        self.cost_curve = []              # Per-message token/cost snapshots

    def add_message(self, tokens: int, message_idx: int):
        """Track tokens for a message (simulating session replay)."""
        self.session_tokens += tokens
        self.messages_since_compaction += 1

        # Each message replays the entire session context as input
        self.total_replay_input_tokens += self.session_tokens

        self.cost_curve.append({
            "message_idx": message_idx,
            "session_tokens": self.session_tokens,
            "cumulative_replay_input_tokens": self.total_replay_input_tokens,
        })

    def add_compaction(self):
        """Track compaction and reset session."""
        self.compaction_count += 1
        self.session_tokens = 0
        self.messages_since_compaction = 0

    def summary(self) -> dict:
        """Return token counts and per-model cost breakdowns."""
        extraction_input = self.compaction_count * EXTRACTION_INPUT_TOKENS
        extraction_output = self.compaction_count * EXTRACTION_OUTPUT_TOKENS
        janitor_input = self.compaction_count * JANITOR_INPUT_TOKENS
        janitor_output = self.compaction_count * JANITOR_OUTPUT_TOKENS

        result = {
            "compaction_count": self.compaction_count,
            "final_session_tokens": self.session_tokens,
            "tokens": {
                "replay_input": self.total_replay_input_tokens,
                "extraction_input": extraction_input,
                "extraction_output": extraction_output,
                "janitor_input": janitor_input,
                "janitor_output": janitor_output,
            },
            "cost_by_model": {},
        }

        # Compute cost for each model combination
        # Session replay model × extraction/janitor model
        for replay_model in ["haiku", "sonnet", "opus"]:
            for extract_model in ["haiku", "sonnet", "opus"]:
                rp = MODEL_PRICING[replay_model]
                ep = MODEL_PRICING[extract_model]

                replay_cost = self.total_replay_input_tokens * rp["input"] / 1_000_000
                extract_cost = (
                    extraction_input * ep["input"] / 1_000_000
                    + extraction_output * ep["output"] / 1_000_000
                )
                janitor_cost = (
                    janitor_input * ep["input"] / 1_000_000
                    + janitor_output * ep["output"] / 1_000_000
                )
                total = replay_cost + extract_cost + janitor_cost

                key = f"replay={replay_model},extract={extract_model}"
                result["cost_by_model"][key] = {
                    "replay": round(replay_cost, 4),
                    "extraction": round(extract_cost, 4),
                    "janitor": round(janitor_cost, 4),
                    "total": round(total, 4),
                }

        return result


# ---------------------------------------------------------------------------
# Main injection loop
# ---------------------------------------------------------------------------

def inject_sessions(
    reviews: list,
    results_dir: Path,
    mode: str = "natural",  # "natural" or "nightly"
    extract_model: str = "sonnet",
    owner_id: str = "maya",
    do_extraction_flag: bool = True,
    run_janitor_flag: bool = True,
) -> dict:
    """Main injection loop.

    Processes all sessions chronologically, managing compaction timing.

    Args:
        reviews: Chronologically sorted session reviews
        results_dir: Output directory
        mode: "natural" (compact at 80% threshold) or "nightly" (compact after each day)
        extract_model: Model for extraction LLM calls
        owner_id: Owner ID for facts
        run_extraction_flag: Actually run extraction (False for cost-only analysis)
        run_janitor_flag: Actually run janitor tasks

    Returns:
        Stats dict with costs, compaction events, etc.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    db_path = results_dir / "memory.db"
    session_dir = results_dir / "sessions"
    session_dir.mkdir(exist_ok=True)

    # Setup workspace
    workspace = setup_workspace(results_dir, "Maya Chen")

    # Initialize DB if extraction enabled
    if do_extraction_flag:
        from ingest import _switch_to_db
        _switch_to_db(db_path, fresh=True)

    # Cost trackers for both modes (run both for comparison)
    tracker = CostTracker()

    # Session state
    current_session_messages = []
    current_session_id = 0
    current_day = None
    compaction_events = []
    total_facts = 0
    total_edges = 0
    total_snippets = 0
    total_journal = 0
    message_idx = 0

    t0 = time.monotonic()

    for review in reviews:
        snum = review.session_num
        # Determine the session date
        if snum < 0:
            from dataset import FILLER_DATES
            filler_id = f"F{abs(snum):03d}"
            date_str = FILLER_DATES.get(filler_id, "2026-03-15")
        else:
            from ingest import SESSION_DATES
            date_str = SESSION_DATES.get(snum, "2026-03-01")

        session_day = date_str.split(" ")[0] if " " in date_str else date_str

        # Convert transcript to messages
        messages = transcript_to_messages(review)
        if not messages:
            continue

        label = f"F{abs(snum):03d}" if snum < 0 else f"Session {snum}"
        print(f"\n--- {label} ({session_day}, {len(messages)} messages) ---")

        # Check for day boundary (for nightly mode)
        day_changed = current_day is not None and session_day != current_day

        # Nightly mode: compact at day boundary
        if mode == "nightly" and day_changed and current_session_messages:
            print(f"  [NIGHTLY COMPACT] Day changed {current_day} → {session_day}")
            compaction_event = _do_compaction(
                current_session_messages, current_session_id,
                session_dir, db_path, workspace,
                current_day, extract_model, owner_id,
                do_extraction_flag, run_janitor_flag, tracker,
            )
            compaction_events.append(compaction_event)
            total_facts += compaction_event.get("facts_stored", 0)
            total_edges += compaction_event.get("edges_created", 0)
            total_snippets += compaction_event.get("snippets_written", 0)
            total_journal += compaction_event.get("journal_entries", 0)

            # Reset session
            current_session_messages = []
            current_session_id += 1

        current_day = session_day

        # Inject messages one by one
        for msg in messages:
            msg_tokens = count_tokens(msg["content"])
            current_session_messages.append(msg)
            tracker.add_message(msg_tokens, message_idx)
            message_idx += 1

            # Natural mode: compact when threshold hit
            if mode == "natural" and tracker.session_tokens >= COMPACTION_TOKEN_LIMIT:
                print(f"  [NATURAL COMPACT] Session at {tracker.session_tokens:,} tokens (threshold: {COMPACTION_TOKEN_LIMIT:,})")
                compaction_event = _do_compaction(
                    current_session_messages, current_session_id,
                    session_dir, db_path, workspace,
                    session_day, extract_model, owner_id,
                    do_extraction_flag, run_janitor_flag, tracker,
                )
                compaction_events.append(compaction_event)
                total_facts += compaction_event.get("facts_stored", 0)
                total_edges += compaction_event.get("edges_created", 0)
                total_snippets += compaction_event.get("snippets_written", 0)
                total_journal += compaction_event.get("journal_entries", 0)

                # Reset session
                current_session_messages = []
                current_session_id += 1

    # Final compaction for remaining messages
    if current_session_messages:
        print(f"\n  [FINAL COMPACT] Remaining {len(current_session_messages)} messages")
        compaction_event = _do_compaction(
            current_session_messages, current_session_id,
            session_dir, db_path, workspace,
            current_day or "2026-05-01", extract_model, owner_id,
            do_extraction_flag, run_janitor_flag, tracker,
        )
        compaction_events.append(compaction_event)
        total_facts += compaction_event.get("facts_stored", 0)
        total_edges += compaction_event.get("edges_created", 0)
        total_snippets += compaction_event.get("snippets_written", 0)
        total_journal += compaction_event.get("journal_entries", 0)

    elapsed = round(time.monotonic() - t0, 1)

    # Run final janitor tasks (snippets fold, journal distill, RAG reindex)
    if run_janitor_flag:
        print("\n--- Final Janitor (snippets + journal + RAG) ---")
        run_janitor(db_path, workspace, ["snippets", "journal", "rag"])

    stats = {
        "mode": mode,
        "total_messages": message_idx,
        "total_sessions_injected": len(reviews),
        "compaction_count": len(compaction_events),
        "total_facts": total_facts,
        "total_edges": total_edges,
        "total_snippets": total_snippets,
        "total_journal": total_journal,
        "cost": tracker.summary(),
        "cost_curve": tracker.cost_curve,
        "compaction_events": compaction_events,
        "elapsed_s": elapsed,
    }

    # Save stats
    stats_path = results_dir / "injection_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"\nStats saved to {stats_path}")

    return stats


def _do_compaction(
    messages: List[dict],
    session_id: int,
    session_dir: Path,
    db_path: Path,
    workspace: Path,
    session_date: str,
    extract_model: str,
    owner_id: str,
    do_extraction: bool,
    run_janitor_flag: bool,
    tracker: CostTracker,
) -> dict:
    """Execute a compaction event."""
    # Write session JSONL
    session_file = session_dir / f"session-{session_id:04d}.jsonl"
    write_session_jsonl(messages, session_file)

    msg_tokens = sum(count_tokens(m["content"]) for m in messages)
    print(f"  Compaction: {len(messages)} messages, {msg_tokens:,} tokens")

    event = {
        "session_id": session_id,
        "date": session_date,
        "message_count": len(messages),
        "token_count": msg_tokens,
        "facts_stored": 0,
        "edges_created": 0,
        "snippets_written": 0,
        "journal_entries": 0,
    }

    if do_extraction:
        # Extract facts + edges
        ext_stats = run_extraction(
            session_file, db_path=db_path, results_dir=session_dir.parent,
            owner_id=owner_id, session_date=session_date,
            session_id=f"session-{session_id}",
            extract_model=extract_model,
        )
        event["facts_stored"] = ext_stats.get("facts_stored", 0)
        event["edges_created"] = ext_stats.get("edges_created", 0)
        print(f"  Extracted: {event['facts_stored']} facts, {event['edges_created']} edges")

        # Extract snippets + journal
        snippet_stats = run_snippet_extraction(session_file, workspace, extract_model)
        event["snippets_written"] = snippet_stats.get("snippets_written", 0)
        event["journal_entries"] = snippet_stats.get("journal_entries", 0)
        print(f"  Snippets: {event['snippets_written']}, Journal: {event['journal_entries']}")

    if run_janitor_flag:
        # Run post-compaction janitor (embeddings + review + dedup)
        janitor_stats = run_janitor(db_path, workspace)
        event["janitor"] = janitor_stats

    # Track cost
    tracker.add_compaction()

    return event


# ---------------------------------------------------------------------------
# Cost analysis (no LLM, just math)
# ---------------------------------------------------------------------------

def cost_analysis(reviews: list) -> dict:
    """Compare cost curves for natural vs nightly compaction.

    No LLM calls — pure token counting and cost math.
    Shows raw token spend + per-model cost breakdown.
    """
    print("=" * 70)
    print("COST ANALYSIS: Natural vs Nightly Compaction")
    print("=" * 70)

    natural_tracker = CostTracker()
    nightly_tracker = CostTracker()

    current_day = None
    message_idx = 0
    total_content_tokens = 0

    for review in reviews:
        snum = review.session_num
        if snum < 0:
            from dataset import FILLER_DATES
            filler_id = f"F{abs(snum):03d}"
            date_str = FILLER_DATES.get(filler_id, "2026-03-15")
        else:
            from ingest import SESSION_DATES
            date_str = SESSION_DATES.get(snum, "2026-03-01")

        session_day = date_str.split(" ")[0] if " " in date_str else date_str
        messages = transcript_to_messages(review)
        if not messages:
            continue

        # Day boundary check for nightly mode
        day_changed = current_day is not None and session_day != current_day
        if day_changed:
            nightly_tracker.add_compaction()

        current_day = session_day

        for msg in messages:
            msg_tokens = count_tokens(msg["content"])
            total_content_tokens += msg_tokens

            # Natural mode
            natural_tracker.add_message(msg_tokens, message_idx)
            if natural_tracker.session_tokens >= COMPACTION_TOKEN_LIMIT:
                natural_tracker.add_compaction()

            # Nightly mode
            nightly_tracker.add_message(msg_tokens, message_idx)

            message_idx += 1

    # Final compaction for both
    natural_tracker.add_compaction()
    nightly_tracker.add_compaction()

    n = natural_tracker.summary()
    d = nightly_tracker.summary()

    # --- Print results ---
    print(f"\nTotal messages: {message_idx:,}")
    print(f"Total content tokens: {total_content_tokens:,}")
    print(f"Unique days: {d['compaction_count']}")

    # Token spend comparison
    print(f"\n{'TOKEN SPEND':-^70}")
    print(f"{'Metric':<35} {'Natural':>14} {'Nightly':>14}")
    print(f"{'─' * 63}")
    nt = n["tokens"]
    dt = d["tokens"]
    for key, label in [
        ("replay_input", "Session replay (input)"),
        ("extraction_input", "Extraction (input)"),
        ("extraction_output", "Extraction (output)"),
        ("janitor_input", "Janitor (input)"),
        ("janitor_output", "Janitor (output)"),
    ]:
        print(f"{label:<35} {nt[key]:>14,} {dt[key]:>14,}")

    total_natural_tok = nt["replay_input"] + nt["extraction_input"] + nt["extraction_output"] + nt["janitor_input"] + nt["janitor_output"]
    total_nightly_tok = dt["replay_input"] + dt["extraction_input"] + dt["extraction_output"] + dt["janitor_input"] + dt["janitor_output"]
    print(f"{'─' * 63}")
    print(f"{'TOTAL TOKENS':<35} {total_natural_tok:>14,} {total_nightly_tok:>14,}")
    print(f"{'Compaction count':<35} {n['compaction_count']:>14} {d['compaction_count']:>14}")

    # Per-model cost comparison (production-relevant combos only)
    combos = [
        ("haiku", "opus", "Replay=Haiku, Extract=Opus"),
        ("haiku", "sonnet", "Replay=Haiku, Extract=Sonnet"),
        ("haiku", "haiku", "Replay=Haiku, Extract=Haiku"),
        ("sonnet", "opus", "Replay=Sonnet, Extract=Opus"),
    ]

    print(f"\n{'COST BY MODEL COMBINATION':-^70}")
    print(f"{'Model Config':<35} {'Natural':>14} {'Nightly':>10} {'Savings':>8}")
    print(f"{'─' * 67}")

    for replay_m, extract_m, label in combos:
        nk = f"replay={replay_m},extract={extract_m}"
        dk = f"replay={replay_m},extract={extract_m}"
        nc = n["cost_by_model"][nk]["total"]
        dc = d["cost_by_model"][dk]["total"]
        savings = f"{(1 - dc/nc)*100:.0f}%" if nc > 0 else "N/A"
        print(f"{label:<35} ${nc:>13.4f} ${dc:>9.4f} {savings:>8}")

    # Detailed breakdown for the typical production config
    prod_key = "replay=haiku,extract=opus"
    print(f"\n{'DETAIL: Replay=Haiku, Extract=Opus':-^70}")
    print(f"{'Component':<35} {'Natural':>14} {'Nightly':>14}")
    print(f"{'─' * 63}")
    for component in ["replay", "extraction", "janitor", "total"]:
        nv = n["cost_by_model"][prod_key][component]
        dv = d["cost_by_model"][prod_key][component]
        label = component.capitalize()
        if component == "total":
            print(f"{'─' * 63}")
            label = "TOTAL"
        print(f"{label:<35} ${nv:>13.4f} ${dv:>13.4f}")

    return {
        "total_messages": message_idx,
        "total_content_tokens": total_content_tokens,
        "natural": n,
        "nightly": d,
        "natural_curve": natural_tracker.cost_curve,
        "nightly_curve": nightly_tracker.cost_curve,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AgentLife Pipeline Injector")
    parser.add_argument("--mode", choices=["natural", "nightly", "cost-analysis"],
                        default="cost-analysis",
                        help="Injection mode")
    parser.add_argument("--results-dir", type=str,
                        default=str(_DIR.parent / "data" / "results-pipeline"),
                        help="Output directory")
    parser.add_argument("--assets-dir", type=str,
                        default=str(_DIR.parent.parent.parent / "assets"),
                        help="Arc session review files")
    parser.add_argument("--filler-dir", type=str, default=None,
                        help="Filler sessions directory")
    parser.add_argument("--extract-model", type=str, default="sonnet")
    parser.add_argument("--owner-id", type=str, default="maya")
    parser.add_argument("--no-extract", action="store_true",
                        help="Skip extraction (cost analysis only)")
    parser.add_argument("--no-janitor", action="store_true",
                        help="Skip janitor tasks")
    args = parser.parse_args()

    # Load reviews
    assets_dir = Path(args.assets_dir)
    arc_reviews = load_all_reviews(assets_dir)
    print(f"Loaded {len(arc_reviews)} arc sessions")

    filler_reviews = []
    if args.filler_dir:
        filler_reviews = load_filler_reviews(Path(args.filler_dir))
        print(f"Loaded {len(filler_reviews)} filler sessions")

    if filler_reviews:
        reviews = merge_sessions_chronologically(arc_reviews, filler_reviews)
    else:
        reviews = arc_reviews
    print(f"Total sessions: {len(reviews)}")

    if args.mode == "cost-analysis":
        results = cost_analysis(reviews)
        # Save
        out_path = Path(args.results_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "cost_analysis.json", "w") as f:
            json.dump(results, f, indent=2)
        return

    stats = inject_sessions(
        reviews=reviews,
        results_dir=Path(args.results_dir),
        mode=args.mode,
        extract_model=args.extract_model,
        owner_id=args.owner_id,
        do_extraction_flag=not args.no_extract,
        run_janitor_flag=not args.no_janitor,
    )

    print(f"\n{'=' * 60}")
    print(f"Injection Complete ({args.mode} mode)")
    print(f"{'=' * 60}")
    print(f"  Messages injected: {stats['total_messages']}")
    print(f"  Compaction events: {stats['compaction_count']}")
    print(f"  Facts stored: {stats['total_facts']}")
    print(f"  Edges created: {stats['total_edges']}")
    print(f"  Snippets written: {stats['total_snippets']}")
    print(f"  Journal entries: {stats['total_journal']}")
    print(f"  Cost: ${stats['cost']['total_cost']:.4f}")
    print(f"  Duration: {stats['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
