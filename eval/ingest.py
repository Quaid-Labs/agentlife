#!/usr/bin/env python3
"""AgentLife Benchmark — Ingestion pipeline.

Extracts facts from session transcripts using Sonnet and stores them
in an isolated memory DB. Simulates the projects system for Track 2 sessions.
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from string import Template
from typing import Dict, List, Optional, Tuple

_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _DIR.parent
_WORKSPACE = Path(os.environ.get("CLAWDBOT_WORKSPACE", Path.home() / "clawd"))
_QUAID_DIR = _WORKSPACE / "plugins" / "quaid"

# Add Quaid to path
if str(_QUAID_DIR) not in sys.path:
    sys.path.insert(0, str(_QUAID_DIR))

from dataset import SessionReview, format_transcript_for_extraction, SESSION_DATES, FILLER_DATES

# Lazy imports from Quaid (after path setup)
_quaid_imported = False


def _ensure_quaid():
    global _quaid_imported
    if _quaid_imported:
        return
    # These imports trigger DB initialization
    import memory_graph  # noqa: F401
    _quaid_imported = True


# ---------------------------------------------------------------------------
# Claude backend
# ---------------------------------------------------------------------------

_RUNNER_DIR = _WORKSPACE / "memory-stress-test" / "runner"
if str(_RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(_RUNNER_DIR))

from claude_backend import call_claude, is_available as claude_available


# ---------------------------------------------------------------------------
# Token tracking
# ---------------------------------------------------------------------------

_token_counts = {
    "extraction_calls": 0,
    "extraction_input_tokens_est": 0,
    "extraction_output_tokens_est": 0,
    "janitor_calls": 0,
    "total_duration_s": 0.0,
}


def get_ingest_stats() -> dict:
    return _token_counts.copy()


def reset_ingest_stats():
    global _token_counts
    _token_counts = {k: 0 for k in _token_counts}
    _token_counts["total_duration_s"] = 0.0


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = Template("""\
You are a memory extraction system. Extract personally meaningful facts from this conversation between a user named Maya and her AI assistant.

For each fact, provide:
- text: The fact as a concise statement
- category: One of: fact, preference, belief, experience, event
- extraction_confidence: high, medium, or low
- keywords: Comma-separated searchable keywords (prioritize proper nouns)
- edges: Relationships between entities (subject, relation, object)

For project/code sessions, extract:
- What project is being worked on (name, tech stack)
- Key decisions made (architectural choices, pivots)
- Bugs found/fixed
- Features added
- Who suggested what (speaker attribution)

DO NOT extract:
- Generic assistant knowledge (e.g., "Python is a programming language")
- Temporary conversational mechanics ("let me know if you need help")
- Code syntax/implementation details (extract the WHAT, not the HOW)

Respond with JSON only:
{
  "facts": [
    {
      "text": "Maya lives in Austin, TX",
      "category": "fact",
      "extraction_confidence": "high",
      "keywords": "Maya, Austin, TX, location",
      "edges": [
        {"subject": "Maya", "relation": "lives_in", "object": "Austin, TX"}
      ]
    }
  ],
  "project_state": {
    "project_name": "recipe-app or null",
    "tech_stack": ["express", "sqlite", "..."],
    "features_added": ["dietary filtering", "..."],
    "bugs_fixed": ["SQL injection in search", "..."],
    "decisions": ["switch to GraphQL", "..."]
  }
}
""")


# ---------------------------------------------------------------------------
# DB isolation
# ---------------------------------------------------------------------------

def _switch_to_db(db_path: Path, fresh: bool = False):
    """Switch Quaid's memory_graph to a specific DB file.

    CRITICAL: Must set env var BEFORE importing memory_graph,
    and must pass db_path explicitly to MemoryGraph() constructor.
    The DB_PATH default param is captured at import time.
    """
    if fresh and db_path.exists():
        db_path.unlink()

    os.environ["MEMORY_DB_PATH"] = str(db_path)

    from lib.database import get_connection

    # Initialize schema if fresh
    if fresh:
        with get_connection(db_path=str(db_path)) as conn:
            schema_path = _QUAID_DIR / "schema.sql"
            if schema_path.exists():
                conn.executescript(schema_path.read_text())
                conn.commit()

    # Re-initialize MemoryGraph with explicit db_path
    import memory_graph
    memory_graph._graph = memory_graph.MemoryGraph(db_path=db_path)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

_CONFIDENCE_MAP = {"high": 0.9, "medium": 0.6, "low": 0.3}


def extract_session(
    transcript: str,
    session_num: int,
    extract_model: str = "sonnet",
) -> Tuple[Optional[dict], float]:
    """Extract facts from a session transcript.

    Returns (parsed_json, duration_seconds).
    """
    prompt = EXTRACTION_PROMPT.safe_substitute() + "\n\n--- CONVERSATION ---\n\n" + transcript

    response, duration = call_claude(
        prompt=prompt,
        model=extract_model,
        system_prompt="You are a memory extraction system. Respond with JSON only.",
        timeout=120,
    )

    _token_counts["extraction_calls"] += 1
    _token_counts["extraction_input_tokens_est"] += len(prompt) // 4
    _token_counts["extraction_output_tokens_est"] += len(response or "") // 4
    _token_counts["total_duration_s"] += duration

    if not response:
        print(f"  WARNING: Empty extraction response for session {session_num}")
        return None, duration

    # Parse JSON from response (may have markdown fences)
    text = response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  WARNING: JSON parse error for session {session_num}: {e}")
        # Try to find JSON in the response
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group())
            except json.JSONDecodeError:
                return None, duration
        else:
            return None, duration

    return parsed, duration


def store_session_facts(
    extraction: dict,
    owner_id: str,
    session_date: str,
    session_num: int,
) -> dict:
    """Store extracted facts into the memory DB.

    Returns stats dict with facts_stored, edges_created counts.
    """
    _ensure_quaid()
    from memory_graph import store, create_edge

    facts = extraction.get("facts", [])
    stats = {"facts_stored": 0, "edges_created": 0}

    for fact_json in facts:
        text = fact_json.get("text", "").strip()
        if not text or len(text) < 5:
            continue

        category = fact_json.get("category", "fact")
        confidence_str = fact_json.get("extraction_confidence", "medium")
        confidence = _CONFIDENCE_MAP.get(confidence_str, 0.6)
        keywords = fact_json.get("keywords", "")

        try:
            # store() returns {"id": node_id, "status": "created", ...}
            fact_created_at = fact_json.get("created_at") or session_date
            result = store(
                text=text,
                owner_id=owner_id,
                category=category,
                confidence=confidence,
                status="active",  # Skip pending/review for benchmark
                keywords=keywords,
                source_type="user",
                session_id=f"session-{session_num}",
                created_at=fact_created_at,
            )
            node_id = result.get("id") if isinstance(result, dict) else None
            stats["facts_stored"] += 1

            # Create edges (only if we got a valid node_id)
            if node_id:
                for edge_json in fact_json.get("edges", []):
                    subject = edge_json.get("subject", "")
                    relation = edge_json.get("relation", "")
                    obj = edge_json.get("object", "")
                    if subject and relation and obj:
                        try:
                            create_edge(
                                subject_name=subject,
                                relation=relation,
                                object_name=obj,
                                owner_id=owner_id,
                                source_fact_id=node_id,
                                create_missing_entities=True,
                            )
                            stats["edges_created"] += 1
                        except Exception:
                            pass  # Edge creation failures are non-fatal
        except Exception as e:
            print(f"    WARNING: Failed to store fact: {e}")

    return stats


# ---------------------------------------------------------------------------
# Projects system simulation
# ---------------------------------------------------------------------------

def setup_project_workspace(results_dir: Path, owner_id: str):
    """Create project definitions and docs for the recipe-app project.

    This simulates what Quaid's projects system does in production:
    - Register the project
    - Create PROJECT.md
    - Create TOOLS.md with tech stack info
    """
    project_dir = results_dir / "projects" / "recipe-app"
    project_dir.mkdir(parents=True, exist_ok=True)

    # Initial PROJECT.md
    project_md = project_dir / "PROJECT.md"
    project_md.write_text(
        "# Recipe App\n\n"
        "Maya's recipe organizer app. Motivated by her mom Linda's diabetes diagnosis.\n\n"
        "## Tech Stack\n- Node.js + Express\n- SQLite (better-sqlite3)\n- Jest for testing\n\n"
        "## Status\nIn development.\n"
    )

    # TOOLS.md
    tools_md = project_dir / "TOOLS.md"
    tools_md.write_text(
        "# Recipe App — Tools\n\n"
        "## API Endpoints\n"
        "- GET /api/recipes — List all recipes\n"
        "- POST /api/recipes — Create recipe\n"
        "- GET /api/recipes/search?q= — Search recipes\n"
        "- GET /api/recipes/:id — Get single recipe\n"
        "- PUT /api/recipes/:id — Update recipe\n"
        "- DELETE /api/recipes/:id — Delete recipe\n\n"
        "## Database\n"
        "SQLite with recipes table (title, ingredients, instructions, dietary_tags, "
        "image_url, prep_time, created_at)\n"
    )

    return project_dir


def update_project_state(
    project_dir: Path,
    extraction: dict,
    session_num: int,
):
    """Update PROJECT.md with new project state info from extraction."""
    project_state = extraction.get("project_state", {})
    if not project_state or project_state.get("project_name") is None:
        return

    project_md = project_dir / "PROJECT.md"
    if not project_md.exists():
        return

    features = project_state.get("features_added", [])
    bugs = project_state.get("bugs_fixed", [])
    decisions = project_state.get("decisions", [])
    tech_stack = project_state.get("tech_stack", [])

    if not (features or bugs or decisions):
        return

    # Append session update
    update = f"\n\n## Session {session_num} Updates\n"
    if features:
        update += "### Features Added\n" + "\n".join(f"- {f}" for f in features) + "\n"
    if bugs:
        update += "### Bugs Fixed\n" + "\n".join(f"- {b}" for b in bugs) + "\n"
    if decisions:
        update += "### Decisions\n" + "\n".join(f"- {d}" for d in decisions) + "\n"
    if tech_stack:
        update += f"### Tech Stack\n- {', '.join(tech_stack)}\n"

    with open(project_md, "a") as f:
        f.write(update)


def index_project_docs(results_dir: Path):
    """Index project docs in Quaid's RAG system."""
    _ensure_quaid()
    try:
        from docs_rag import DocsRAG
        project_dir = results_dir / "projects"
        if project_dir.exists():
            rag = DocsRAG()
            stats = rag.reindex_all(str(project_dir))
            print(f"  RAG index updated: {stats}")
    except Exception as e:
        print(f"  WARNING: RAG indexing failed: {e}")


# ---------------------------------------------------------------------------
# Janitor passes
# ---------------------------------------------------------------------------

def run_janitor_pass(tasks: List[str] = None):
    """Run lightweight janitor tasks (embeddings + dedup)."""
    _ensure_quaid()
    try:
        from janitor import run_task_optimized
        tasks = tasks or ["embeddings"]
        for task in tasks:
            run_task_optimized(task, dry_run=False)
    except Exception as e:
        print(f"  WARNING: Janitor {tasks} failed: {e}")


# ---------------------------------------------------------------------------
# Extraction cache
# ---------------------------------------------------------------------------

def _cache_path(cache_dir: Path, session_num: int, model: str) -> Path:
    if session_num < 0:
        # Filler session: F001, F002, etc.
        return cache_dir / f"F{abs(session_num):03d}_{model}.json"
    return cache_dir / f"session-{session_num:02d}_{model}.json"


def _load_cache(cache_dir: Path, session_num: int, model: str) -> Optional[dict]:
    p = _cache_path(cache_dir, session_num, model)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None
    return None


def _save_cache(cache_dir: Path, session_num: int, model: str, data: dict):
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = _cache_path(cache_dir, session_num, model)
    p.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------

def ingest_all(
    reviews: List[SessionReview],
    results_dir: Path,
    extract_model: str = "sonnet",
    owner_id: str = "maya",
    use_cache: bool = True,
    janitor_tasks: Optional[List[str]] = None,
) -> dict:
    """Ingest all sessions sequentially.

    Args:
        reviews: Sorted list of SessionReview objects
        results_dir: Directory for DB, cache, project docs
        extract_model: Model for extraction (sonnet/opus)
        owner_id: Owner ID for all facts
        use_cache: Whether to use extraction cache
        janitor_tasks: Tasks to run after each session (default: embeddings only)

    Returns:
        Aggregated ingestion stats.
    """
    reset_ingest_stats()
    janitor_tasks = janitor_tasks or ["embeddings"]

    db_path = results_dir / "memory.db"
    cache_dir = results_dir / "extraction_cache"

    # Initialize fresh DB
    _switch_to_db(db_path, fresh=True)

    # Setup projects workspace
    project_dir = setup_project_workspace(results_dir, owner_id)

    total_stats = {
        "sessions_processed": 0,
        "facts_stored": 0,
        "edges_created": 0,
        "cache_hits": 0,
        "extraction_errors": 0,
        "elapsed_seconds": 0.0,
    }

    t0 = time.monotonic()

    for review in reviews:
        snum = review.session_num
        if snum < 0:
            # Filler session
            filler_id = f"F{abs(snum):03d}"
            session_date = FILLER_DATES.get(filler_id, "2026-03-15")
            print(f"\n--- {filler_id} (Filler, Track {review.track}) ---")
        else:
            session_date = SESSION_DATES.get(snum, "2026-03-01")
            print(f"\n--- Session {snum} (Track {review.track}) ---")

        # Check cache
        cached = None
        if use_cache:
            cached = _load_cache(cache_dir, snum, extract_model)
            if cached:
                print(f"  Using cached extraction ({len(cached.get('facts', []))} facts)")
                total_stats["cache_hits"] += 1

        # Extract if not cached
        if cached is None:
            transcript = format_transcript_for_extraction(review)
            if not transcript.strip():
                print(f"  WARNING: Empty transcript for session {snum}, skipping")
                continue

            extraction, duration = extract_session(transcript, snum, extract_model)
            if extraction is None:
                total_stats["extraction_errors"] += 1
                continue

            # Cache it
            _save_cache(cache_dir, snum, extract_model, extraction)
            cached = extraction

        # Store facts
        _switch_to_db(db_path, fresh=False)
        stats = store_session_facts(cached, owner_id, session_date, snum)
        total_stats["facts_stored"] += stats["facts_stored"]
        total_stats["edges_created"] += stats["edges_created"]
        total_stats["sessions_processed"] += 1

        print(f"  Stored: {stats['facts_stored']} facts, {stats['edges_created']} edges")

        # Update project docs for Track 2 sessions
        if review.track == 2:
            update_project_state(project_dir, cached, snum)

        # Run janitor pass (embeddings after each session)
        for task in janitor_tasks:
            run_janitor_pass([task])

    # Final RAG indexing
    index_project_docs(results_dir)

    total_stats["elapsed_seconds"] = round(time.monotonic() - t0, 1)
    total_stats["ingest_token_stats"] = get_ingest_stats()

    return total_stats
