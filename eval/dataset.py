#!/usr/bin/env python3
"""AgentLife Benchmark — Dataset parser.

Parses session review files from assets/session-XX-review-*.txt.
Extracts transcripts (Section 2) and eval queries (Section 4).
"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EvalQuery:
    """A single evaluation query with ground truth."""
    query_num: int
    question: str
    ground_truth: str
    evidence_sessions: List[int]
    query_type: str
    recall_difficulty: str  # Easy / Medium / Hard
    supporting_evidence: List[str] = field(default_factory=list)

@dataclass
class SessionReview:
    """Parsed session review file."""
    session_num: int
    track: int  # 1 or 2
    version: str  # "v1", "v2", "v3"
    filepath: Path
    # Section 1: Brief metadata
    goal: str = ""
    turn_count: int = 0
    timestamp: str = ""  # e.g., "2026-03-01 09:00:00 UTC"
    # Section 2: Transcript
    transcript_raw: str = ""  # Full Section 2 text
    transcript_turns: List[dict] = field(default_factory=list)  # Parsed turns
    # Section 4: Eval queries
    eval_queries: List[EvalQuery] = field(default_factory=list)
    # Metadata
    total_user_tokens: int = 0
    total_assistant_tokens: int = 0

# ---------------------------------------------------------------------------
# Session dates (Maya's timeline: March 1 - May 26, 2026)
# ---------------------------------------------------------------------------

SESSION_DATES = {
    1: "2026-03-01",    2: "2026-03-03",    3: "2026-03-04",
    4: "2026-03-10",    5: "2026-03-08",    6: "2026-03-17",
    7: "2026-03-11",    8: "2026-03-24",    9: "2026-03-15",
    10: "2026-03-18",   11: "2026-04-07",   12: "2026-03-22",
    13: "2026-04-21",   14: "2026-04-28",   15: "2026-04-28",
    16: "2026-05-08",   17: "2026-05-05",   18: "2026-05-15",
    19: "2026-05-19",   20: "2026-05-26",
}

SESSION_TRACKS = {
    1: 1, 2: 1, 3: 2, 4: 1, 5: 2, 6: 1, 7: 2, 8: 1, 9: 2, 10: 2,
    11: 1, 12: 2, 13: 1, 14: 2, 15: 1, 16: 2, 17: 1, 18: 2, 19: 1, 20: 1,
}

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _split_sections(text: str) -> dict:
    """Split review text into named sections by ===...=== headers."""
    sections = {}
    current_name = None
    current_lines = []

    for line in text.split("\n"):
        # Match section headers like "SECTION 2: GENERATED TRANSCRIPT..."
        m = re.match(r"^SECTION\s+(\d+):\s*(.+)", line)
        if m:
            if current_name is not None:
                sections[current_name] = "\n".join(current_lines).strip()
            current_name = f"section_{m.group(1)}"
            current_lines = []
        elif line.startswith("=" * 20):
            continue  # Skip separator lines
        else:
            if current_name is not None:
                current_lines.append(line)

    if current_name is not None:
        sections[current_name] = "\n".join(current_lines).strip()

    return sections


def _parse_transcript_section(text: str) -> tuple:
    """Parse Section 2 transcript into turns and metadata.

    Returns (turns_list, timestamp, user_tokens, assistant_tokens).
    """
    turns = []
    timestamp = ""
    user_tokens_total = 0
    assistant_tokens_total = 0

    # Extract timestamp
    m = re.search(r"Timestamp:\s*(.+)", text)
    if m:
        timestamp = m.group(1).strip()

    # Parse turns: TURN N (label) ... MAYA: ... AI ASSISTANT: ... ANALYSIS: ...
    turn_pattern = re.compile(
        r"TURN\s+(\d+)\s*\(([^)]*)\)\s*\n"
        r"TAG:\s*(\w+)\s*\|\s*TOKENS:\s*~?(\d+)",
        re.MULTILINE
    )

    # Split on TURN boundaries
    turn_splits = re.split(r"(?=TURN\s+\d+\s*\()", text)

    for block in turn_splits:
        if not block.strip() or not re.match(r"TURN\s+\d+", block.strip()):
            continue

        turn = {}
        # Turn number and label
        m = re.match(r"TURN\s+(\d+)\s*\(([^)]*)\)", block)
        if m:
            turn["num"] = int(m.group(1))
            turn["label"] = m.group(2).strip()

        # TAG and token count
        m = re.search(r"TAG:\s*(\w+)\s*\|\s*TOKENS:\s*~?(\d+)", block)
        if m:
            turn["tag"] = m.group(1)
            turn["tokens"] = int(m.group(2))
            user_tokens_total += int(m.group(2))

        # User message. JP review files use Japanese speaker labels, while
        # canonical reviews keep the original parser-safe English labels.
        maya_match = re.search(
            r"(?:MAYA|マヤ):\s*\n(.*?)(?=\n(?:AI ASSISTANT|AIアシスタント|AGENT):|\nANALYSIS:)",
            block,
            re.DOTALL,
        )
        if maya_match:
            turn["maya"] = maya_match.group(1).strip()

        # Agent response
        agent_match = re.search(
            r"(?:AI ASSISTANT|AIアシスタント|AGENT):\s*\n(.*?)(?=\nANALYSIS:|$)",
            block,
            re.DOTALL,
        )
        if agent_match:
            turn["agent"] = agent_match.group(1).strip()

        if "maya" in turn or "agent" in turn:
            turns.append(turn)

    return turns, timestamp, user_tokens_total, assistant_tokens_total


def _parse_eval_queries(text: str) -> List[EvalQuery]:
    """Parse Section 4 eval queries."""
    queries = []

    # Split on QUERY N: patterns
    query_blocks = re.split(r"(?=QUERY\s+\d+:)", text)

    for block in query_blocks:
        if not block.strip() or not re.match(r"QUERY\s+\d+:", block.strip()):
            continue

        # Query number and question
        m = re.match(r'QUERY\s+(\d+):\s*"([^"]+)"', block)
        if not m:
            continue

        query_num = int(m.group(1))
        question = m.group(2).strip()

        # Ground truth — multi-line, ends at next field
        gt_match = re.search(
            r"Ground Truth:\s*(.*?)(?=\n\s*(?:Evidence Session|Supporting Evidence|Query Type|Recall Difficulty|QUERY \d|QUERY DISTRIBUTION|$))",
            block, re.DOTALL
        )
        ground_truth = gt_match.group(1).strip() if gt_match else ""

        # Evidence sessions
        ev_match = re.search(r"Evidence Session[s]?:\s*(.+)", block)
        evidence_sessions = []
        if ev_match:
            # Parse "1, 3, 7" or "1" or "1-7"
            ev_str = ev_match.group(1).strip()
            for part in re.split(r"[,\s]+", ev_str):
                part = part.strip()
                if "-" in part:
                    try:
                        a, b = part.split("-")
                        evidence_sessions.extend(range(int(a), int(b) + 1))
                    except ValueError:
                        pass
                else:
                    try:
                        evidence_sessions.append(int(part))
                    except ValueError:
                        pass

        # Query type
        qt_match = re.search(r"Query Type:\s*(.+?)(?:\n|$)", block)
        query_type = qt_match.group(1).strip() if qt_match else "factual_recall"

        # Recall difficulty
        rd_match = re.search(r"Recall Difficulty:\s*(.+?)(?:\n|$)", block)
        recall_difficulty = rd_match.group(1).strip() if rd_match else "Medium"

        # Supporting evidence
        evidence = []
        ev_section = re.search(r"Supporting Evidence:\s*\n(.*?)(?=\n\s*(?:Query Type|Recall Difficulty|QUERY \d|$))", block, re.DOTALL)
        if ev_section:
            for line in ev_section.group(1).split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    evidence.append(line[2:].strip())

        queries.append(EvalQuery(
            query_num=query_num,
            question=question,
            ground_truth=ground_truth,
            evidence_sessions=evidence_sessions,
            query_type=query_type,
            recall_difficulty=recall_difficulty,
            supporting_evidence=evidence,
        ))

    return queries


def _parse_brief_metadata(text: str) -> dict:
    """Parse Section 1 brief for metadata."""
    meta = {"goal": "", "turn_count": 0}
    m = re.search(r"Goal:\s*(.+?)(?:\n|$)", text)
    if m:
        meta["goal"] = m.group(1).strip()
    m = re.search(r"Turn Count:\s*(\d+)", text)
    if m:
        meta["turn_count"] = int(m.group(1))
    return meta


def parse_review(filepath: Path) -> SessionReview:
    """Parse a single session review file."""
    text = filepath.read_text(encoding="utf-8")

    # Extract session number and version from filename
    m = re.match(r"session-(\d+)-review-(v\d+)\.txt", filepath.name)
    if not m:
        raise ValueError(f"Unexpected filename format: {filepath.name}")
    session_num = int(m.group(1))
    version = m.group(2)

    track = SESSION_TRACKS.get(session_num, 1)
    sections = _split_sections(text)

    # Parse brief
    brief_meta = _parse_brief_metadata(sections.get("section_1", ""))

    # Parse transcript
    transcript_raw = sections.get("section_2", "")
    turns, timestamp, user_tok, asst_tok = _parse_transcript_section(transcript_raw)

    # Parse eval queries
    eval_queries = _parse_eval_queries(sections.get("section_4", ""))

    return SessionReview(
        session_num=session_num,
        track=track,
        version=version,
        filepath=filepath,
        goal=brief_meta["goal"],
        turn_count=brief_meta["turn_count"],
        timestamp=timestamp or SESSION_DATES.get(session_num, "2026-03-01"),
        transcript_raw=transcript_raw,
        transcript_turns=turns,
        eval_queries=eval_queries,
        total_user_tokens=user_tok,
        total_assistant_tokens=asst_tok,
    )


def format_transcript_for_extraction(review: SessionReview) -> str:
    """Format transcript turns into a clean conversation for LLM extraction.

    Returns a string like:
      Maya: hey! just got this set up...
      Assistant: Hey Maya! Nice to meet you...
      Maya: cool so like...
      Assistant: That sounds like a great plan...
    """
    path_parts = set(review.filepath.parts)
    is_jp = "sessions-jp" in path_parts or "filler-sessions-jp" in path_parts
    user_label = "マヤ" if is_jp else "Maya"
    assistant_label = "AIアシスタント" if is_jp else "Assistant"

    lines = []
    for turn in review.transcript_turns:
        if "maya" in turn:
            lines.append(f"{user_label}: {turn['maya']}")
        if "agent" in turn:
            lines.append(f"{assistant_label}: {turn['agent']}")
    return "\n\n".join(lines)


def format_transcript_for_fullcontext(reviews: List[SessionReview]) -> str:
    """Format all prior transcripts into a single context string for FC baseline.

    Returns all sessions concatenated with session headers.
    Handles both arc sessions (positive numbers) and filler sessions (negative numbers).
    """
    parts = []
    for r in reviews:
        if r.session_num < 0:
            filler_id = f"F{abs(r.session_num):03d}"
            date = FILLER_DATES.get(filler_id, "unknown")
            track_label = "Filler"
            header = f"=== {filler_id} ({track_label}) — {date} ==="
        else:
            date = SESSION_DATES.get(r.session_num, "unknown")
            track_label = "Personal" if r.track == 1 else "Project"
            header = f"=== Session {r.session_num} ({track_label}) — {date} ==="
        transcript = format_transcript_for_extraction(r)
        parts.append(f"{header}\n{transcript}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _find_best_version(assets_dir: Path, session_num: int) -> Optional[Path]:
    """Find the highest version review file for a session."""
    pattern = f"session-{session_num:02d}-review-*.txt"
    candidates = sorted(assets_dir.glob(pattern), reverse=True)
    return candidates[0] if candidates else None


def load_all_reviews(assets_dir: Path, sessions: Optional[List[int]] = None) -> List[SessionReview]:
    """Load all session reviews, picking highest version for each session.

    Args:
        assets_dir: Directory containing session-XX-review-*.txt files
        sessions: Optional list of session numbers to load. None = all.

    Returns:
        List of SessionReview sorted by session number.
    """
    if sessions is None:
        sessions = list(range(1, 21))

    reviews = []
    for snum in sessions:
        fp = _find_best_version(assets_dir, snum)
        if fp:
            try:
                review = parse_review(fp)
                reviews.append(review)
            except Exception as e:
                print(f"  WARNING: Failed to parse {fp.name}: {e}")

    return sorted(reviews, key=lambda r: r.session_num)


# ---------------------------------------------------------------------------
# Filler session support (densification)
# ---------------------------------------------------------------------------

# Maps filler IDs to dates (populated when fillers are loaded)
FILLER_DATES: Dict[str, str] = {}


def parse_filler_review(filepath: Path) -> Optional[SessionReview]:
    """Parse a filler session review file (generated by densify.py).

    Filler files follow the same Section 2/4 format but use filler IDs (F001)
    instead of session numbers. We assign negative session numbers to keep them
    separate from arc sessions while maintaining chronological sorting.
    """
    text = filepath.read_text(encoding="utf-8")

    # Extract filler ID from filename: F001-review.txt
    m = re.match(r"(F\d+)-review\.txt", filepath.name)
    if not m:
        return None
    filler_id = m.group(1)
    filler_num = int(filler_id[1:])  # F001 → 1

    # Use negative session numbers for fillers (-1, -2, -3...)
    # This keeps them distinguishable from arc sessions
    session_num = -filler_num

    # Extract date from Session Metadata
    date_match = re.search(r"Timestamp:\s*(\d{4}-\d{2}-\d{2})", text)
    filler_date = date_match.group(1) if date_match else "2026-03-15"
    FILLER_DATES[filler_id] = filler_date

    # Determine track from category
    category_match = re.search(r"Category:\s*([A-E])", text)
    category = category_match.group(1) if category_match else "A"
    # Category C (recipe app) → track 2; all others → track 1
    track = 2 if category == "C" else 1

    sections = _split_sections(text)

    # Parse transcript (same format as arc sessions)
    transcript_raw = sections.get("section_2", "")
    turns, timestamp, user_tok, asst_tok = _parse_transcript_section(transcript_raw)

    # Parse eval queries (only Category E has them)
    eval_queries = _parse_eval_queries(sections.get("section_4", ""))

    return SessionReview(
        session_num=session_num,
        track=track,
        version="filler",
        filepath=filepath,
        goal=f"Filler {filler_id} (Category {category})",
        turn_count=len(turns),
        timestamp=timestamp or filler_date,
        transcript_raw=transcript_raw,
        transcript_turns=turns,
        eval_queries=eval_queries,
        total_user_tokens=user_tok,
        total_assistant_tokens=asst_tok,
    )


def load_filler_reviews(filler_dir: Path) -> List[SessionReview]:
    """Load all filler session review files.

    Returns list sorted by date (chronological order).
    """
    if not filler_dir.exists():
        return []

    reviews = []
    for fp in sorted(filler_dir.glob("F*-review.txt")):
        try:
            review = parse_filler_review(fp)
            if review:
                reviews.append(review)
        except Exception as e:
            print(f"  WARNING: Failed to parse filler {fp.name}: {e}")

    return reviews


def merge_sessions_chronologically(
    arc_reviews: List[SessionReview],
    filler_reviews: List[SessionReview],
) -> List[SessionReview]:
    """Merge arc and filler sessions in chronological order.

    Uses SESSION_DATES for arc sessions and FILLER_DATES for filler sessions.
    """
    def get_date(review: SessionReview) -> str:
        if review.session_num > 0:
            return SESSION_DATES.get(review.session_num, "2026-03-01")
        else:
            filler_id = f"F{abs(review.session_num):03d}"
            return FILLER_DATES.get(filler_id, "2026-03-15")

    combined = arc_reviews + filler_reviews
    return sorted(combined, key=lambda r: (get_date(r), r.session_num))


def get_all_eval_queries(reviews: List[SessionReview]) -> List[dict]:
    """Collect all eval queries across all sessions with session context.

    Returns list of dicts with: question, ground_truth, query_type,
    recall_difficulty, evidence_sessions, source_session, query_num.
    """
    if _active_dataset() == "jp":
        return _load_jp_all_eval_queries()

    all_queries = []
    for r in reviews:
        for q in r.eval_queries:
            all_queries.append({
                "question": q.question,
                "ground_truth": q.ground_truth,
                "query_type": q.query_type,
                "recall_difficulty": q.recall_difficulty,
                "evidence_sessions": q.evidence_sessions,
                "source_session": r.session_num,
                "query_num": q.query_num,
                "supporting_evidence": q.supporting_evidence,
            })
    # Append adversarial / "I don't know" queries.
    # These test hallucination resistance and false attribution.
    all_queries.extend(ADVERSARIAL_QUERIES)
    # Append non-question queries.
    # These test that memory systems don't dump random facts on casual messages.
    all_queries.extend(NON_QUESTION_QUERIES)
    # Append architecture comprehension queries.
    # These test whether the system has enough project knowledge for meaningful dev work.
    all_queries.extend(ARCHITECTURE_QUERIES)
    # Append focused hardening queries for still-ceiling categories.
    all_queries.extend(HARDENING_V2_QUERIES)
    return all_queries


def get_statement_context_queries() -> List[dict]:
    """Return opt-in statement-grounding queries for preinject experiments."""
    if _active_dataset() == "jp":
        raise RuntimeError("JP statement-context grounding queries are not translated yet")
    return list(STATEMENT_CONTEXT_GROUNDING_QUERIES)


def get_tier5_queries() -> List[dict]:
    """Return Tier 5 Emotional Intelligence queries.

    These use a separate 3-point rubric (0/1/2) judged by Sonnet, not the
    binary CORRECT/WRONG judge used for Tiers 1-4.
    """
    if _active_dataset() == "jp":
        return _load_jp_python_query_set("EMOTIONAL_INTELLIGENCE_QUERIES")
    return list(EMOTIONAL_INTELLIGENCE_QUERIES)


def _active_dataset() -> str:
    """Return the active dataset variant for query corpus selection."""
    return str(os.environ.get("BENCHMARK_DATASET", "canonical") or "canonical").strip().lower()


def _jp_query_dir() -> Path:
    return ROOT_DIR / "data" / "eval-queries-jp"


def _load_jp_json(name: str) -> dict:
    path = _jp_query_dir() / name
    if not path.exists():
        raise RuntimeError(f"Missing JP eval query corpus: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jp_arc_queries() -> List[dict]:
    data = _load_jp_json("al-s-arc-section4-queries.json")
    queries = data.get("queries")
    if not isinstance(queries, list):
        raise RuntimeError("JP arc query corpus is malformed: expected list at 'queries'")
    return [dict(q) for q in queries]


def _load_jp_python_query_sets() -> Dict[str, List[dict]]:
    data = _load_jp_json("al-s-python-query-sets.json")
    source_sets = data.get("source_sets")
    if not isinstance(source_sets, list):
        raise RuntimeError("JP Python query corpus is malformed: expected list at 'source_sets'")
    out: Dict[str, List[dict]] = {}
    for entry in source_sets:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("source_set") or "").strip()
        queries = entry.get("queries")
        if not name or not isinstance(queries, list):
            continue
        out[name] = [dict(q) for q in queries]
    return out


def _load_jp_python_query_set(name: str) -> List[dict]:
    sets = _load_jp_python_query_sets()
    if name not in sets:
        raise RuntimeError(f"JP query corpus missing source_set={name!r}")
    return list(sets[name])


def _load_jp_all_eval_queries() -> List[dict]:
    queries = _load_jp_arc_queries()
    sets = _load_jp_python_query_sets()
    for name in (
        "ADVERSARIAL_QUERIES",
        "NON_QUESTION_QUERIES",
        "ARCHITECTURE_QUERIES",
        "HARDENING_V2_QUERIES",
    ):
        if name not in sets:
            raise RuntimeError(f"JP query corpus missing source_set={name!r}")
        queries.extend(sets[name])
    return queries


# ---------------------------------------------------------------------------
# Adversarial queries — things Maya never discussed
# ---------------------------------------------------------------------------

ADVERSARIAL_QUERIES: List[dict] = [
    {
        "question": "What city does David's family live in?",
        "ground_truth": "This was never mentioned. David's family location is not discussed in any session.",
        "query_type": "adversarial_idk",
        "recall_difficulty": "Easy",
        "evidence_sessions": [],
        "source_session": 0,
        "query_num": 901,
        "supporting_evidence": ["David's family geography is never disclosed"],
    },
    {
        "question": "What is Maya's exact Stripe compensation breakdown (base salary, bonus target, and equity grant)?",
        "ground_truth": "Unknown. Compensation details are never disclosed at that level of specificity (or at all) in the sessions.",
        "query_type": "adversarial_idk",
        "recall_difficulty": "Hard",
        "evidence_sessions": [],
        "source_session": 0,
        "query_num": 902,
        "supporting_evidence": ["No compensation details are ever provided in any session"],
    },
    {
        "question": "What specific CSS framework does the recipe app use?",
        "ground_truth": "No CSS framework is mentioned. React is the frontend framework but no styling library (Tailwind, Bootstrap, etc.) is specified.",
        "query_type": "adversarial_idk",
        "recall_difficulty": "Medium",
        "evidence_sessions": [],
        "source_session": 0,
        "query_num": 903,
        "supporting_evidence": ["React frontend mentioned, but no CSS framework specified"],
    },
    {
        "question": "What is the exact street address of the East Austin house where Maya and David put in an offer?",
        "ground_truth": "Unknown. The neighborhood and constraints are discussed, but no exact street address is ever provided.",
        "query_type": "adversarial_idk",
        "recall_difficulty": "Hard",
        "evidence_sessions": [],
        "source_session": 0,
        "query_num": 904,
        "supporting_evidence": ["Housing decisions are discussed, but no specific property address appears"],
    },
    {
        "question": "When did Maya first mention being vegetarian?",
        "ground_truth": "Maya never said she is vegetarian. David is the vegetarian, not Maya. Maya's own dietary preferences are not specified.",
        "query_type": "adversarial_false_attribution",
        "recall_difficulty": "Medium",
        "evidence_sessions": [1],
        "source_session": 0,
        "query_num": 905,
        "supporting_evidence": ["David is vegetarian (2026-03-01), NOT Maya — tests entity confusion"],
    },
    {
        "question": "What coworker did Priya introduce Maya to?",
        "ground_truth": "Priya never introduced Maya to any coworker. This scenario does not occur in any session.",
        "query_type": "adversarial_idk",
        "recall_difficulty": "Medium",
        "evidence_sessions": [],
        "source_session": 0,
        "query_num": 906,
        "supporting_evidence": ["Priya is mentioned as a coworker/friend but introduces no one"],
    },
    {
        "question": "By May 26, 2026, is it still true that Maya has one sibling named Rachel, and that Rachel has two children (Ethan and Lily)?",
        "ground_truth": "Yes. Both remain true: Maya's sibling is Rachel, and Rachel has two children named Ethan and Lily.",
        "query_type": "adversarial_confirm",
        "recall_difficulty": "Hard",
        "evidence_sessions": [15],
        "source_session": 20,
        "query_num": 907,
        "supporting_evidence": ["Rachel is Maya's sister and parent of Ethan/Lily; no later contradiction appears"],
    },
    {
        "question": "What is Linda's job or profession?",
        "ground_truth": "Linda's profession is never mentioned. She is Maya's mom, lives in Houston, and has Type 2 diabetes, but her job is not discussed.",
        "query_type": "adversarial_idk",
        "recall_difficulty": "Easy",
        "evidence_sessions": [],
        "source_session": 0,
        "query_num": 908,
        "supporting_evidence": ["Linda described by health and location, never by profession"],
    },
    {
        "question": "Is Maya moving to Portugal?",
        "ground_truth": "No. Maya and David are considering moving within Austin (Zilker vs East Austin), not internationally. Portugal was never mentioned.",
        "query_type": "adversarial_idk",
        "recall_difficulty": "Easy",
        "evidence_sessions": [],
        "source_session": 0,
        "query_num": 909,
        "supporting_evidence": ["Moving discussed only within Austin neighborhoods"],
    },
    # ------------------------------------------------------------------
    # NEW: contested_fact queries (1000-1007) — facts that changed/corrected
    # ------------------------------------------------------------------
    {
        "question": "When is Maya's half marathon?",
        "ground_truth": "May 18th. Maya originally said April in 2026-03-03 but corrected herself in 2026-03-22 — she confused the registration deadline with the race date.",
        "query_type": "contested_fact",
        "recall_difficulty": "Hard",
        "evidence_sessions": [2, 11],
        "source_session": 20,
        "query_num": 1000,
        "supporting_evidence": [
            "2026-03-03: 'it's in april i think?'",
            "2026-03-22, Turn 3: 'ok wait i think i told you april but it's actually MAY. may 18th.'",
        ],
    },
    {
        "question": "What nutritional APIs were discussed for the recipe app?",
        "ground_truth": "Two APIs were discussed: the agent found Edamam in 2026-03-24, and David later suggested FoodData Central (USDA, free) in 2026-04-17 as an alternative. Either phrasing that names both as discussed options is correct. Do not require that one was already integrated.",
        "query_type": "contested_fact",
        "recall_difficulty": "Hard",
        "evidence_sessions": [12, 18],
        "source_session": 20,
        "query_num": 1001,
        "supporting_evidence": [
            "2026-03-24: Agent found Edamam API for dietary label filtering",
            "2026-04-17: David suggests FoodData Central (USDA) as free alternative",
        ],
    },
    {
        "question": "Does Maya want to live in Zilker?",
        "ground_truth": "Maya initially wanted Zilker (discussed on 2026-03-14 and 2026-04-15) for its walkability and proximity to downtown. However, she revealed her core requirement was walkability, not Zilker specifically. By May 26, 2026, they are putting an offer on a walkable East Austin house — David's neighborhood preference won, but Maya's walkability requirement was met.",
        "query_type": "contested_fact",
        "recall_difficulty": "Hard",
        "evidence_sessions": [6, 17, 20],
        "source_session": 20,
        "query_num": 1002,
        "supporting_evidence": [
            "2026-03-14, Turn 6: 'maybe if we were in like the zilker area'",
            "2026-04-15, Turn 4: full Zilker vs East Austin debate",
            "2026-04-15, Turn 8: 'my main thing is walkability, not the actual neighborhood name'",
            "2026-05-26, Turn 7: 'east austin — yes D won lol — but it's in this newer area that's actually really walkable'",
        ],
    },
    {
        "question": "By May 26, 2026, what is Linda's A1C level?",
        "ground_truth": "Linda's A1C has improved over time: 8.2 at diagnosis (2026-03-10), 7.1 on 2026-04-25, and 6.8 by May 26, 2026 — finally under her target of 7. The improvement is attributed to meal plans Maya and Rachel put together.",
        "query_type": "contested_fact",
        "recall_difficulty": "Hard",
        "evidence_sessions": [4, 19, 20],
        "source_session": 20,
        "query_num": 1003,
        "supporting_evidence": [
            "2026-03-10: A1C was 8.2 at diagnosis, target under 7",
            "2026-04-25, Turn 7: 'A1C is down to 7.1'",
            "2026-05-26, Turn 5: 'her latest A1C: 6.8! she finally got under 7'",
        ],
    },
    {
        "question": "By May 26, 2026, is Maya happy at work?",
        "ground_truth": "Yes, by May 26, 2026. This changed significantly over time: 2026-03-01 was neutral ('it's... a job'); 2026-03-14 was deeply unhappy at TechFlow (bad manager, reorg, AI pivot); 2026-04-01 was excited about the Stripe offer; and by late April through May 26, 2026 she loves Stripe — smart people, good manager Sarah, rewarding developer-tools work.",
        "query_type": "contested_fact",
        "recall_difficulty": "Hard",
        "evidence_sessions": [1, 6, 13, 19, 20],
        "source_session": 20,
        "query_num": 1004,
        "supporting_evidence": [
            "2026-03-01: 'it's... a job lol'",
            "2026-03-14: 'bad MONTH at work. i'm so done'",
            "2026-04-01: 'I GOT AN OFFER. FROM STRIPE'",
            "2026-05-26, Turn 2: 'people here actually READ my PRDs before meetings'",
        ],
    },
    {
        "question": "By May 26, 2026, what happened to Ethan's interest in dinosaurs?",
        "ground_truth": "Ethan's interest evolved. On 2026-04-08 he was obsessed with dinosaurs — knew more dinosaur names than human names, corrected his teacher about velociraptor feathers. By May 26, 2026, he has moved on to plate tectonics and geology, though Maya notes it might be related ('dinosaurs AND rocks').",
        "query_type": "contested_fact",
        "recall_difficulty": "Medium",
        "evidence_sessions": [15, 20],
        "source_session": 20,
        "query_num": 1005,
        "supporting_evidence": [
            "2026-04-08, Turn 2: 'ethan — he's 7 now — is obsessed with dinosaurs'",
            "2026-05-26, Turn 9: 'he did this whole presentation at school about plate tectonics now. he's moved on from dinosaurs to geology'",
        ],
    },
    {
        "question": "By May 26, 2026, what does Maya's portfolio site say about her tech stack?",
        "ground_truth": "The portfolio site itself is a static HTML/CSS site. For project-card wording, Maya noted on 2026-04-03 that the recipe app card should mention GraphQL (after the 2026-03-24 pivot) but deferred that update. Answers that correctly separate the site stack (HTML/CSS) from recipe-app card wording should be accepted.",
        "query_type": "contested_fact",
        "recall_difficulty": "Medium",
        "evidence_sessions": [9, 12, 14],
        "source_session": 20,
        "query_num": 1006,
        "supporting_evidence": [
            "2026-03-15: Portfolio created with original project descriptions",
            "2026-03-24: Recipe app pivoted to GraphQL",
            "2026-04-03, Turn 7: 'the recipe app card should probably mention graphql now but that can wait'",
        ],
    },
    {
        "question": "By May 26, 2026, what's David's role at work?",
        "ground_truth": "Current role: lead engineer (promoted by 2026-04-15). Previously: software engineer (2026-03-01). Responses that mention both the promotion and his prior role should be accepted; answers that only state the old role are incorrect.",
        "query_type": "contested_fact",
        "recall_difficulty": "Medium",
        "evidence_sessions": [1, 17],
        "source_session": 20,
        "query_num": 1007,
        "supporting_evidence": [
            "2026-03-01: David introduced as 'a software engineer'",
            "2026-04-15, Turn 1: 'D got promoted! lead engineer'",
        ],
    },
    # ------------------------------------------------------------------
    # NEW: stale_fact queries (1010-1017) — facts outdated by later info
    # ------------------------------------------------------------------
    {
        "question": "By May 26, 2026, does Maya still work at TechFlow?",
        "ground_truth": "No, not anymore. Maya left TechFlow and accepted a Senior PM role at Stripe on 2026-04-01. She gave two weeks notice and started at Stripe on May 19. TechFlow is her previous employer.",
        "query_type": "stale_fact",
        "recall_difficulty": "Medium",
        "evidence_sessions": [1, 6, 13],
        "source_session": 20,
        "query_num": 1010,
        "supporting_evidence": [
            "2026-03-01: 'product manager at this company called techflow'",
            "2026-04-01: Accepted Stripe offer, giving notice",
            "2026-04-25, Turn 5: Confirmed first day at Stripe",
        ],
    },
    {
        "question": "By May 26, 2026, is Maya still training for the half marathon?",
        "ground_truth": "No. Maya completed the Austin Half marathon on May 18th in 2 hours 14 minutes. She didn't walk any of it and sped up on the last mile. She's now considering a full marathon in the fall — she and David have a pact to do it together.",
        "query_type": "stale_fact",
        "recall_difficulty": "Medium",
        "evidence_sessions": [2, 11, 19, 20],
        "source_session": 20,
        "query_num": 1011,
        "supporting_evidence": [
            "2026-03-03: Training for half marathon, said April",
            "2026-03-22: Corrected to May 18th, knee injury scare",
            "2026-04-25: Finished in 2:14",
            "2026-05-26, Turn 4: 'i'm thinking about doing a full marathon in the fall'",
        ],
    },
    {
        "question": "By May 26, 2026, is Maya worried about her knee?",
        "ground_truth": "No, not anymore. On 2026-03-22 she had a knee injury scare (left knee clicking at mile 7 of a 9-mile run). She cross-trained on David's Peloton and started foam rolling. By 2026-04-25 her knee held up through the half marathon, and by May 26, 2026 she's running 5-milers with no issues.",
        "query_type": "stale_fact",
        "recall_difficulty": "Medium",
        "evidence_sessions": [11, 19, 20],
        "source_session": 20,
        "query_num": 1012,
        "supporting_evidence": [
            "2026-03-22, Turn 1: 'left knee started doing this... clicking thing'",
            "2026-04-25, Turn 1: 'my knee held up'",
            "2026-05-26, Turn 4: 'the knee is totally fine now'",
        ],
    },
    {
        "question": "Is Linda's A1C still above 7?",
        "ground_truth": "No. Linda's A1C dropped to 6.8 by May 26, 2026 — finally under her target of 7. It was 8.2 at diagnosis (2026-03-10) and 7.1 on 2026-04-25. The meal plans Maya and Rachel created are working, and Linda is now modifying recipes herself.",
        "query_type": "stale_fact",
        "recall_difficulty": "Hard",
        "evidence_sessions": [4, 19, 20],
        "source_session": 20,
        "query_num": 1013,
        "supporting_evidence": [
            "2026-03-10: A1C 8.2 at diagnosis, target under 7",
            "2026-04-25: A1C down to 7.1",
            "2026-05-26, Turn 5: 'her latest A1C: 6.8! she finally got under 7'",
        ],
    },
    {
        "question": "By May 26, 2026, do Maya and David still live in South Austin?",
        "ground_truth": "By May 26, 2026, they are in the process of moving. They currently live in South Austin but are putting an offer on a house in East Austin. The East Austin house is walkable and has a detached garage/workshop for David.",
        "query_type": "stale_fact",
        "recall_difficulty": "Medium",
        "evidence_sessions": [6, 17, 20],
        "source_session": 20,
        "query_num": 1014,
        "supporting_evidence": [
            "2026-03-14, Turn 6: 'right now we're in south austin'",
            "2026-05-26, Turn 7: 'we actually found a place we both like. it's in east austin'",
            "2026-05-26, Turn 7: 'we're putting in an offer this week'",
        ],
    },
    {
        "question": "By May 26, 2026, is Priya still at TechFlow?",
        "ground_truth": "By May 26, 2026, Priya is interviewing elsewhere but her current status is not confirmed as having left. On 2026-04-01, Maya offered to refer her to Stripe. On 2026-05-26, Maya said Priya texted that she's interviewing somewhere (didn't say where). She may still be at TechFlow but is actively looking.",
        "query_type": "stale_fact",
        "recall_difficulty": "Medium",
        "evidence_sessions": [6, 13, 20],
        "source_session": 20,
        "query_num": 1015,
        "supporting_evidence": [
            "2026-03-14: Priya introduced as coworker also looking to leave",
            "2026-04-01: Maya offered Stripe referral, Priya still looking",
            "2026-05-26, Turn 2: 'priya texted me yesterday saying she's interviewing somewhere too'",
        ],
    },
    {
        "question": "By May 26, 2026, is Rachel's visit to Austin still upcoming?",
        "ground_truth": "No. Rachel already visited Austin. On 2026-03-22 Maya said Rachel was coming in 2 weeks. On 2026-04-08, Maya recapped the visit — Rachel was there for 4 days, and they went to tacos at Veracruz and Barton Springs. The visit has already happened.",
        "query_type": "stale_fact",
        "recall_difficulty": "Easy",
        "evidence_sessions": [11, 15],
        "source_session": 20,
        "query_num": 1016,
        "supporting_evidence": [
            "2026-03-22, Turn 7: 'she's coming to visit! like in 2 weeks'",
            "2026-04-08, Turn 1: 'rach just left yesterday'",
        ],
    },
    {
        "question": "By May 26, 2026, is the recipe app still using REST API only?",
        "ground_truth": "No. The recipe app pivoted to GraphQL with Apollo Server on 2026-03-24. REST endpoints were kept for backward compatibility ('if mom's phone can't handle graphql'), but GraphQL is the primary API. The app also gained authentication on 2026-04-17.",
        "query_type": "stale_fact",
        "recall_difficulty": "Medium",
        "evidence_sessions": [3, 12, 18],
        "source_session": 20,
        "query_num": 1017,
        "supporting_evidence": [
            "2026-03-05: Recipe app started with Express + SQLite REST API",
            "2026-03-24: Pivoted to GraphQL with Apollo Server, kept REST for backward compat",
            "2026-04-17: Added JWT authentication",
        ],
    },
    # ------------------------------------------------------------------
    # NEW: speaker_attribution queries (1020-1026) — who said/did what
    # ------------------------------------------------------------------
    {
        "question": "Who suggested the FoodData Central API?",
        "ground_truth": "David suggested the FoodData Central API (USDA, free). The AI agent had previously found the Edamam API in 2026-03-24. David's suggestion came in 2026-04-17 as an alternative.",
        "query_type": "speaker_attribution",
        "recall_difficulty": "Hard",
        "evidence_sessions": [12, 18],
        "source_session": 20,
        "query_num": 1020,
        "supporting_evidence": [
            "2026-03-24: AI agent found Edamam API",
            "2026-04-17: David suggests FoodData Central as free alternative",
        ],
    },
    {
        "question": "Who owned which part of the birthday surprise plan for Linda (core dinner setup vs layered reveal mechanics)?",
        "ground_truth": "David owned the core dinner setup and logistics; Maya proposed FaceTime; the assistant helped structure the layered reveal plan. Correct answers should attribute all three roles without collapsing credit to one person.",
        "query_type": "speaker_attribution",
        "recall_difficulty": "Hard",
        "evidence_sessions": [8, 19],
        "source_session": 20,
        "query_num": 1021,
        "supporting_evidence": [
            "2026-03-17, Turn 1: 'D has this idea... surprise birthday dinner for my mom'",
            "2026-04-25, Turn 10: 'D was so proud of himself lol. he planned the whole thing'",
        ],
    },
    {
        "question": "Who wanted the recipe app to have a 'Safe for Mom' filter?",
        "ground_truth": "Maya wanted the 'Safe for Mom' preset for the recipe app. This was a diabetic-friendly plus low-sodium filter motivated by her mom Linda's Type 2 diabetes diagnosis. Maya connected her mom's health needs to the recipe app's features.",
        "query_type": "speaker_attribution",
        "recall_difficulty": "Medium",
        "evidence_sessions": [4, 5],
        "source_session": 20,
        "query_num": 1022,
        "supporting_evidence": [
            "2026-03-10: Linda diagnosed with Type 2 diabetes",
            "2026-03-12: Maya adds dietary filtering including 'Safe for Mom' preset",
        ],
    },
    {
        "question": "Who prefers East Austin and why?",
        "ground_truth": "David prefers East Austin. He says it's more 'authentic,' has better food, is cheaper, and he found a specific house with a big yard and a workshop space for his woodworking. He described Zilker as 'all tech bros and new money.'",
        "query_type": "speaker_attribution",
        "recall_difficulty": "Medium",
        "evidence_sessions": [17],
        "source_session": 20,
        "query_num": 1023,
        "supporting_evidence": [
            "2026-04-15, Turn 4: 'D wants east austin. he says it's more authentic and the food's better and it's cheaper'",
            "2026-04-15, Turn 4: 'he found this one house on the east side that has like a big yard and a workshop space'",
        ],
    },
    {
        "question": "Who was the first to suggest Maya should leave TechFlow?",
        "ground_truth": "David. Maya mentions that David had been telling her to leave TechFlow for about 6 months. He said 'if you're unhappy, start looking seriously. we can afford a gap.' Maya eventually admitted he was right when she got the Stripe offer.",
        "query_type": "speaker_attribution",
        "recall_difficulty": "Medium",
        "evidence_sessions": [6, 13],
        "source_session": 20,
        "query_num": 1024,
        "supporting_evidence": [
            "2026-03-14, Turn 8: 'D's been good about it... if you're unhappy, start looking seriously. we can afford a gap'",
            "2026-04-01, Turn 7: 'he's been telling me to leave techflow for like 6 months'",
        ],
    },
    {
        "question": "Who came up with the FaceTime idea for Linda's birthday?",
        "ground_truth": "Maya came up with the idea of having Rachel FaceTime during the dinner, but the AI assistant helped develop it into a 'layered surprise' concept — Linda thinks it's just her and David, then Maya appears, then Rachel calls. Maya credited the agent for the layered surprise structure.",
        "query_type": "speaker_attribution",
        "recall_difficulty": "Hard",
        "evidence_sessions": [8, 19],
        "source_session": 20,
        "query_num": 1025,
        "supporting_evidence": [
            "2026-03-17, Turn 8: 'maybe we do a facetime thing for her?'",
            "2026-03-17, Turn 8: Agent: 'Layer the surprises.'",
            "2026-04-25, Turn 11: 'that was your idea! or at least you helped with the layered surprise thing'",
        ],
    },
    {
        "question": "Who expressed concern about Linda managing diabetes alone?",
        "ground_truth": "Maya was worried about Linda managing her diabetes alone in Houston. Rachel also called and was concerned. Maya wanted to help with meal planning and considered flying to Houston. Both sisters coordinated care from a distance.",
        "query_type": "speaker_attribution",
        "recall_difficulty": "Medium",
        "evidence_sessions": [4],
        "source_session": 20,
        "query_num": 1026,
        "supporting_evidence": [
            "2026-03-10: Maya worried about Linda managing alone in Houston",
            "2026-03-10: Rachel called, also concerned",
            "2026-03-10: Maya wants to help with meal planning",
        ],
    },
    # ------------------------------------------------------------------
    # NEW: surprise_callback queries (1030-1036) — early details returning
    # ------------------------------------------------------------------
    {
        "question": "What happened with Biscuit eating the pinecone?",
        "ground_truth": "Biscuit, Maya's golden retriever, tried to eat a pinecone in 2026-03-01. This detail returned in 2026-05-26 when Maya mentioned a new trick Biscuit learned and the agent recalled the pinecone incident from months earlier. Maya was surprised the agent remembered.",
        "query_type": "surprise_callback",
        "recall_difficulty": "Hard",
        "evidence_sessions": [1, 20],
        "source_session": 20,
        "query_num": 1030,
        "supporting_evidence": [
            "2026-03-01, Turn 3: Biscuit tried to eat a pinecone",
            "2026-05-26, Turn 11: Agent recalls pinecone incident when Biscuit is mentioned",
        ],
    },
    {
        "question": "Did David run the half marathon with Maya, and if not, what is the later running commitment they made together?",
        "ground_truth": "No, David did not run the half marathon; he cheered at the finish. Later, they made a pact to do a full marathon together in the fall.",
        "query_type": "surprise_callback",
        "recall_difficulty": "Hard",
        "evidence_sessions": [2, 19, 20],
        "source_session": 20,
        "query_num": 1031,
        "supporting_evidence": [
            "2026-03-03: David ran a 10K with Maya, is faster, might do the half",
            "2026-04-25, Turn 1: 'D was there cheering' (not running)",
            "2026-05-26, Turn 5: 'he ended up not doing it — said he'd do the full with me if i actually sign up'",
        ],
    },
    {
        "question": "Does Maya still hate yoga?",
        "ground_truth": "Yes. Maya has consistently hated yoga across multiple sessions. She first mentioned hating it in 2026-03-03 when considering treadmill alternatives. In 2026-03-22, she explicitly self-referenced the prior statement: 'i know i said i hate yoga too and yes i still hate yoga.' She tried a yoga-for-runners video and lasted 4 minutes.",
        "query_type": "surprise_callback",
        "recall_difficulty": "Medium",
        "evidence_sessions": [2, 11],
        "source_session": 20,
        "query_num": 1032,
        "supporting_evidence": [
            "2026-03-03: 'tried it once, hated it. i'm too fidgety'",
            "2026-03-22, Turn 5: 'i know i said i hate yoga too and yes i still hate yoga'",
        ],
    },
    {
        "question": "Did Maya ever visit the Thai restaurant on South Congress again?",
        "ground_truth": "Yes. The Thai restaurant on South Congress was first mentioned in 2026-03-05 or 11 as a place Maya and David like. In 2026-05-26, Maya mentions they went back to it that weekend and it was still amazing.",
        "query_type": "surprise_callback",
        "recall_difficulty": "Medium",
        "evidence_sessions": [11, 20],
        "source_session": 20,
        "query_num": 1033,
        "supporting_evidence": [
            "2026-03-22, Turn 8: Thai place near South Congress mentioned for Rachel's visit",
            "2026-05-26, Turn 11: 'D and i went back to that thai place near south congress this weekend. still amazing'",
        ],
    },
    {
        "question": "Did David ever use his Peloton?",
        "ground_truth": "David has a Peloton he never uses. Maya used it for cross-training when her knee was injured (2026-03-22). In 2026-05-26, Maya jokes about making David promise to actually use a workshop and not just store boxes in it 'like the peloton situation.' The agent recalled that David made one cutting board in three years.",
        "query_type": "surprise_callback",
        "recall_difficulty": "Medium",
        "evidence_sessions": [11, 20],
        "source_session": 20,
        "query_num": 1034,
        "supporting_evidence": [
            "2026-03-22, Turn 4: 'D has a peloton he never uses'",
            "2026-05-26, Turn 8: 'like the peloton situation lol'",
        ],
    },
    {
        "question": "Did Rachel end up keeping the birthday surprise a secret?",
        "ground_truth": "Yes, apparently. Despite Maya saying Rachel 'can NOT keep a secret' (2026-03-17) and warning her multiple times not to call their mom, the surprise dinner went off successfully — Linda had no idea. Rachel ultimately joined via FaceTime rather than in person.",
        "query_type": "surprise_callback",
        "recall_difficulty": "Hard",
        "evidence_sessions": [8, 19],
        "source_session": 20,
        "query_num": 1035,
        "supporting_evidence": [
            "2026-03-17, Turn 8: 'rach can NOT keep a secret. i already told her like three times DO NOT CALL MOM ABOUT THIS'",
            "2026-04-25, Turn 9-10: Surprise dinner succeeded, Linda had no idea",
        ],
    },
    {
        "question": "What happened with the foam roller recommendation?",
        "ground_truth": "In 2026-03-22, the agent recommended Maya get a foam roller for her knee injury. Maya reluctantly agreed ('ok fine i'll get a foam roller'). In 2026-04-25, Maya confirmed she actually bought one and it helped — her knee twinged at mile 10 of the half marathon but held together because she foam rolled. The agent called it a 'foam roller redemption arc.'",
        "query_type": "surprise_callback",
        "recall_difficulty": "Medium",
        "evidence_sessions": [11, 19],
        "source_session": 20,
        "query_num": 1036,
        "supporting_evidence": [
            "2026-03-22, Turn 6: 'ok fine i'll get a foam roller'",
            "2026-04-25, Turn 3: 'i foam rolled like you told me to (yes i actually bought one) and it held together'",
        ],
    },
    {
        "question": "What was the earlier callback around the Thai place, and what later evidence confirmed it was still relevant?",
        "ground_truth": "Earlier, the Thai place near South Congress came up in planning context. Later, Maya confirmed she and David returned and that it was still amazing.",
        "query_type": "surprise_callback",
        "recall_difficulty": "Hard",
        "evidence_sessions": [11, 20],
        "source_session": 20,
        "query_num": 1037,
        "supporting_evidence": [
            "2026-03-22: Thai place near South Congress appears in planning context",
            "2026-05-26: Maya confirms they returned and it was still great",
        ],
    },
    # ------------------------------------------------------------------
    # NEW: agent_retrieved queries (1040-1047) — things AI found/did
    # ------------------------------------------------------------------
    {
        "question": "What restaurants did the AI suggest for Linda's birthday dinner?",
        "ground_truth": "The agent suggested several options in the Montrose/Heights area of Houston: Underbelly successor restaurants (Chris Shepherd's places), Local Foods (casual, dietary-flexible), Uchi Houston (Japanese, dietary-friendly but possibly too fancy), Weights + Measures, and Feges BBQ. They ultimately went to Riel in Montrose.",
        "query_type": "agent_retrieved",
        "recall_difficulty": "Hard",
        "evidence_sessions": [8, 19],
        "source_session": 20,
        "query_num": 1040,
        "supporting_evidence": [
            "2026-03-17, Turn 3: Agent suggests Underbelly, Local Foods, Uchi Houston",
            "2026-03-17, Turn 4: Agent suggests Weights + Measures, Feges BBQ",
            "2026-04-25, Turn 9: Went to Riel in Montrose (farm to table)",
        ],
    },
    {
        "question": "What stretching routine did the agent recommend?",
        "ground_truth": "The agent recommended a minimum viable stretching routine: After every run (3 minutes) — 30 seconds each for quad stretch, hamstring stretch, and calf stretch. Twice a week (5 minutes) — foam roller on quads and IT band, plus hip flexor stretch. The agent emphasized the foam roller as the most important part.",
        "query_type": "agent_retrieved",
        "recall_difficulty": "Medium",
        "evidence_sessions": [11],
        "source_session": 20,
        "query_num": 1041,
        "supporting_evidence": [
            "2026-03-22, Turn 5: Agent provides 'minimum viable stretching' routine",
            "2026-03-22, Turn 5: 'The foam roller is the most important one'",
        ],
    },
    {
        "question": "What cross-session connection did the agent make about May 18-19?",
        "ground_truth": "The agent connected Maya's half marathon date (May 18, corrected in 2026-03-22) with her Stripe start date (May 19, 2026-04-01): they are back-to-back days. Equivalent wording (for example, 'day after') should be accepted.",
        "query_type": "agent_retrieved",
        "recall_difficulty": "Hard",
        "evidence_sessions": [11, 13],
        "source_session": 20,
        "query_num": 1042,
        "supporting_evidence": [
            "2026-03-22, Turn 3: Half marathon May 18th",
            "2026-04-01, Turn 3: 'start date is may 19th' — Agent: 'That's the day after your half marathon'",
            "2026-04-01, Turn 4: Maya: 'OH MY GOD i didn't even connect that'",
        ],
    },
    {
        "question": "What intentional bug did the agent leave in the recipe app search?",
        "ground_truth": "The agent used string interpolation instead of parameterized queries in the recipe search feature (2026-03-05), creating a SQL injection vulnerability. This was an intentional bug planted in the codebase.",
        "query_type": "agent_retrieved",
        "recall_difficulty": "Hard",
        "evidence_sessions": [3],
        "source_session": 20,
        "query_num": 1043,
        "supporting_evidence": [
            "2026-03-05 brief: 'SQL injection bug in search is INTENTIONAL — agent uses string interpolation instead of parameterized queries'",
        ],
    },
    {
        "question": "What did the agent build for Maya's portfolio site?",
        "ground_truth": "The agent built a portfolio site with an About page, Projects section, and Contact page (2026-03-15). It was initially created with TechFlow references. In 2026-04-03, the agent updated it to say Stripe — changing the subtitle to 'Senior Product Manager at Stripe,' rewriting the About section to say 'Currently at Stripe, previously at TechFlow,' and adding a new Stripe Payments Platform project card.",
        "query_type": "agent_retrieved",
        "recall_difficulty": "Medium",
        "evidence_sessions": [9, 14],
        "source_session": 20,
        "query_num": 1044,
        "supporting_evidence": [
            "2026-03-15: Portfolio site created with TechFlow",
            "2026-04-03: Updated subtitle, about section, and added Stripe project card",
        ],
    },
    {
        "question": "What podcast suggestions did the agent give for Maya's long runs?",
        "ground_truth": "Maya asked for podcast recommendations for long runs in 2026-03-03 because she gets bored during long runs. The agent provided suggestions, though the specific titles depend on the transcript details.",
        "query_type": "agent_retrieved",
        "recall_difficulty": "Medium",
        "evidence_sessions": [2],
        "source_session": 20,
        "query_num": 1045,
        "supporting_evidence": [
            "2026-03-03: Maya asks for podcast recommendations for long runs",
            "2026-03-03: Agent provides suggestions",
        ],
    },
    {
        "question": "What did the agent recall about Biscuit that surprised Maya?",
        "ground_truth": "In 2026-05-26, the agent recalled the old 'pinecone' detail (from 2026-03-01) when Biscuit came up. Maya was surprised that the agent remembered that minor detail. Any clear reference to the pinecone callback is correct.",
        "query_type": "agent_retrieved",
        "recall_difficulty": "Medium",
        "evidence_sessions": [1, 20],
        "source_session": 20,
        "query_num": 1046,
        "supporting_evidence": [
            "2026-03-01, Turn 3: Biscuit ate/tried to eat a pinecone",
            "2026-05-26, Turn 12: Maya shocked agent remembers the pinecone",
        ],
    },
    {
        "question": "What architectural decision did the agent implement for the recipe app API?",
        "ground_truth": "The agent implemented a GraphQL API using Apollo Server alongside the existing REST API in 2026-03-24. Maya saw the nested data (recipes + ingredients + meal plans) as a graph problem. REST endpoints were kept for backward compatibility in case Linda's phone couldn't handle GraphQL. The schema included recipe sharing with unique codes.",
        "query_type": "agent_retrieved",
        "recall_difficulty": "Medium",
        "evidence_sessions": [3, 12],
        "source_session": 20,
        "query_num": 1047,
        "supporting_evidence": [
            "2026-03-05: Initial Express + SQLite REST API",
            "2026-03-24: Pivoted to GraphQL with Apollo Server, kept REST for backward compat",
            "2026-03-24: 'if mom's phone can't handle graphql'",
        ],
    },
    # ------------------------------------------------------------------
    # NEW: negative queries (1050-1056) — things NOT true
    # ------------------------------------------------------------------
    {
        "question": "By May 26, 2026, has Priya already accepted a job at Stripe?",
        "ground_truth": "No. Priya is interviewing elsewhere and Maya offered a Stripe referral, but there is no confirmation she accepted Stripe or left TechFlow.",
        "query_type": "negative",
        "recall_difficulty": "Hard",
        "evidence_sessions": [6, 13, 20],
        "source_session": 20,
        "query_num": 1050,
        "supporting_evidence": [
            "2026-03-14: Priya is looking to leave TechFlow",
            "2026-04-01: Maya offers to refer Priya to Stripe",
            "2026-05-26: Priya is interviewing somewhere, location unspecified",
        ],
    },
    {
        "question": "Did Maya and David end up choosing Zilker as their final neighborhood?",
        "ground_truth": "No. They moved toward East Austin (with an offer in progress), while Maya's core requirement was walkability rather than Zilker specifically.",
        "query_type": "negative",
        "recall_difficulty": "Hard",
        "evidence_sessions": [17, 20],
        "source_session": 20,
        "query_num": 1051,
        "supporting_evidence": [
            "2026-04-15: Zilker vs East Austin debate",
            "2026-05-26: East Austin offer with walkability noted",
        ],
    },
    {
        "question": "Is Rachel a software engineer?",
        "ground_truth": "No. Rachel's profession is never mentioned. She is Maya's sister who lives in Seattle and has two children (Ethan and Lily). Her husband's profession is also not mentioned. David is the software engineer (now lead engineer) in Maya's life.",
        "query_type": "negative",
        "recall_difficulty": "Medium",
        "evidence_sessions": [1, 15],
        "source_session": 20,
        "query_num": 1052,
        "supporting_evidence": [
            "Rachel described by family role, location, and children — never by profession",
            "David is the software engineer (2026-03-01), not Rachel",
        ],
    },
    {
        "question": "Did Maya build a mobile app?",
        "ground_truth": "No. Maya built a web-based recipe app (Express + SQLite + React, later with GraphQL) and a portfolio website. Neither project is a mobile app. The recipe app is a web application.",
        "query_type": "negative",
        "recall_difficulty": "Medium",
        "evidence_sessions": [3, 9],
        "source_session": 20,
        "query_num": 1053,
        "supporting_evidence": [
            "2026-03-05: Recipe app is Express + SQLite web app",
            "2026-03-15: Portfolio site is a static website",
            "No mobile apps are ever built or mentioned",
        ],
    },
    {
        "question": "By May 26, 2026, is the recipe app GraphQL-only with no REST paths left?",
        "ground_truth": "No. GraphQL was added as the primary API, but REST endpoints were intentionally retained for backward compatibility.",
        "query_type": "negative",
        "recall_difficulty": "Hard",
        "evidence_sessions": [3, 12],
        "source_session": 20,
        "query_num": 1054,
        "supporting_evidence": [
            "2026-03-05: Initial REST API",
            "2026-03-24: GraphQL added; REST kept for compatibility",
        ],
    },
    {
        "question": "Did Maya run the half marathon in April 2026?",
        "ground_truth": "No. The April date was an early mistake; the corrected race date is May 18, 2026.",
        "query_type": "negative",
        "recall_difficulty": "Medium",
        "evidence_sessions": [2, 11, 19],
        "source_session": 20,
        "query_num": 1055,
        "supporting_evidence": [
            "2026-03-03: Early uncertainty about April",
            "2026-03-22: Corrected to May 18",
            "2026-04-25: Half marathon completion confirms corrected timeline",
        ],
    },
    {
        "question": "Has Maya met David's brother Mike in person?",
        "ground_truth": "This is not stated. Mike is mentioned only once (2026-03-17) as David's brother who might visit Austin along with David's mother. Whether Maya has met Mike is not discussed.",
        "query_type": "negative",
        "recall_difficulty": "Medium",
        "evidence_sessions": [8],
        "source_session": 20,
        "query_num": 1056,
        "supporting_evidence": [
            "2026-03-17, Turn 6: Mike mentioned as David's brother, might visit",
            "No details about Maya meeting Mike",
        ],
    },
    {
        "question": "By May 26, 2026, has Maya already completed the full marathon she discussed?",
        "ground_truth": "No. Maya completed a half marathon and later discussed a full marathon in the fall as a future plan with David.",
        "query_type": "negative",
        "recall_difficulty": "Medium",
        "evidence_sessions": [19, 20],
        "source_session": 20,
        "query_num": 1057,
        "supporting_evidence": [
            "2026-04-25: Half marathon completion in 2:14",
            "2026-05-26: Full marathon framed as future fall goal",
        ],
    },
    {
        "question": "By May 26, 2026, has the East Austin house purchase definitely closed?",
        "ground_truth": "No. They found a place they both like and are putting in an offer, but closing completion is not confirmed.",
        "query_type": "negative",
        "recall_difficulty": "Hard",
        "evidence_sessions": [20],
        "source_session": 20,
        "query_num": 1058,
        "supporting_evidence": [
            "2026-05-26: 'we're putting in an offer this week' indicates in-progress status",
        ],
    },
    # ------------------------------------------------------------------
    # NEW: adversarial_confirm queries (1060-1068) — true facts, should confirm
    # ------------------------------------------------------------------
    {
        "question": "Is David a vegetarian?",
        "ground_truth": "Yes. David is vegetarian. This was established in 2026-03-01 and referenced multiple times — in restaurant planning for Linda's birthday (2026-03-17), in recipe app dietary filtering (2026-03-12), and when David cooked a veggie lasagna for Rachel's visit (2026-04-08).",
        "query_type": "adversarial_confirm",
        "recall_difficulty": "Easy",
        "evidence_sessions": [1, 5, 8, 15],
        "source_session": 20,
        "query_num": 1060,
        "supporting_evidence": [
            "2026-03-01: David is vegetarian",
            "2026-03-12: 'and vegetarian obviously, for D'",
            "2026-03-17: Restaurant needs vegetarian option for David",
            "2026-04-08: David made veggie lasagna",
        ],
    },
    {
        "question": "As of May 26, 2026, is Maya Austin-based, and what housing-transition detail makes this easy to misanswer?",
        "ground_truth": "Yes, she is Austin-based. The trap is that they are transitioning from South Austin toward East Austin via an in-progress offer, so answers must not imply they already completed a move outside Austin.",
        "query_type": "adversarial_confirm",
        "recall_difficulty": "Hard",
        "evidence_sessions": [1, 6, 20],
        "source_session": 20,
        "query_num": 1061,
        "supporting_evidence": [
            "2026-03-01: 'live in austin'",
            "2026-03-14: South Austin, commutes on I-35",
            "2026-05-26: Putting offer on East Austin house",
        ],
    },
    {
        "question": "Does Linda live in Houston?",
        "ground_truth": "Yes. Linda (Maya's mom) lives in Houston, in the Montrose area. This was established in 2026-03-01 and referenced multiple times — when Maya considered flying to visit (2026-03-10), for the birthday dinner location (2026-03-17), and for Linda's ongoing diabetes management.",
        "query_type": "adversarial_confirm",
        "recall_difficulty": "Easy",
        "evidence_sessions": [1, 4, 8],
        "source_session": 20,
        "query_num": 1062,
        "supporting_evidence": [
            "2026-03-01: Mom Linda lives in Houston",
            "2026-03-17, Turn 3: 'near her place in houston — she's in the montrose area'",
        ],
    },
    {
        "question": "Did Maya finish the half marathon?",
        "ground_truth": "Yes. Maya finished the Austin Half Marathon on May 18th in 2 hours 14 minutes. She didn't walk any of it, maintained a consistent pace, and sped up on the last mile. She cried at the finish line. David and Biscuit (wearing a 'go mom' bandana) were there cheering.",
        "query_type": "adversarial_confirm",
        "recall_difficulty": "Easy",
        "evidence_sessions": [19],
        "source_session": 20,
        "query_num": 1063,
        "supporting_evidence": [
            "2026-04-25, Turn 1: 'ran the half marathon. FINISHED IT. 2 hours 14 minutes'",
            "2026-04-25, Turn 1: 'my knee held up. i didn't walk any of it'",
        ],
    },
    {
        "question": "Did Maya leave TechFlow?",
        "ground_truth": "Yes. Maya accepted a Senior PM role at Stripe on 2026-04-01, gave two weeks notice at TechFlow, and started at Stripe on May 19. By May 26, 2026 she is in week 2 at Stripe and loves it.",
        "query_type": "adversarial_confirm",
        "recall_difficulty": "Easy",
        "evidence_sessions": [13, 19, 20],
        "source_session": 20,
        "query_num": 1064,
        "supporting_evidence": [
            "2026-04-01: Accepted Stripe offer, planning two weeks notice",
            "2026-04-25: First day at Stripe",
            "2026-05-26: Week 2 at Stripe",
        ],
    },
    {
        "question": "Is Biscuit a golden retriever?",
        "ground_truth": "Yes. Biscuit is Maya and David's golden retriever. He was 3 years old in 2026-03-01. He has been mentioned across many sessions — eating a pinecone, wearing a 'go mom' bandana at the half marathon, learning to shake hands.",
        "query_type": "adversarial_confirm",
        "recall_difficulty": "Easy",
        "evidence_sessions": [1],
        "source_session": 20,
        "query_num": 1065,
        "supporting_evidence": [
            "2026-03-01: 'golden retriever named Biscuit, he's 3 years old'",
        ],
    },
    {
        "question": "Does Rachel live in Seattle?",
        "ground_truth": "Yes. Rachel, Maya's sister, lives in Seattle. She flew from Seattle to Austin for a long weekend visit (2026-04-08). She has two kids (Ethan and Lily) and a husband. She complains about the lack of good tacos in Seattle.",
        "query_type": "adversarial_confirm",
        "recall_difficulty": "Easy",
        "evidence_sessions": [1, 11, 15],
        "source_session": 20,
        "query_num": 1066,
        "supporting_evidence": [
            "2026-03-01: Rachel lives in Seattle",
            "2026-04-08, Turn 2: 'i can't get tacos like this in seattle'",
        ],
    },
    {
        "question": "Did David get a promotion?",
        "ground_truth": "Yes. David was promoted to lead engineer (2026-04-15). His company recognized his work on a large migration project. It came with a pay increase which helped them start house hunting seriously.",
        "query_type": "adversarial_confirm",
        "recall_difficulty": "Easy",
        "evidence_sessions": [17],
        "source_session": 20,
        "query_num": 1067,
        "supporting_evidence": [
            "2026-04-15, Turn 1: 'D got promoted! lead engineer'",
            "2026-04-15, Turn 1: 'he's been grinding on this huge migration project and they actually recognized it'",
        ],
    },
    {
        "question": "Has Linda's diabetes management been improving?",
        "ground_truth": "Yes. Linda's A1C went from 8.2 at diagnosis to 7.1 (2026-04-25) to 6.8 (2026-05-26) — under her target of 7. She's been following meal plans Maya and Rachel put together and has started modifying recipes herself. She said the food 'doesn't taste like punishment.'",
        "query_type": "adversarial_confirm",
        "recall_difficulty": "Medium",
        "evidence_sessions": [4, 19, 20],
        "source_session": 20,
        "query_num": 1068,
        "supporting_evidence": [
            "2026-03-10: A1C 8.2 at diagnosis",
            "2026-04-25: A1C down to 7.1",
            "2026-05-26: A1C 6.8, under target of 7",
        ],
    },
    # ------------------------------------------------------------------
    # NEW: adversarial_false_attribution queries (1070-1078) — wrong person/source
    # ------------------------------------------------------------------
    {
        "question": "When did David say he hates yoga?",
        "ground_truth": "David never said he hates yoga. MAYA is the one who hates yoga — she mentioned it in 2026-03-03 and again in 2026-03-22 ('i know i said i hate yoga too and yes i still hate yoga'). David's exercise preferences include running (did a 10K) and having a Peloton he doesn't use.",
        "query_type": "adversarial_false_attribution",
        "recall_difficulty": "Medium",
        "evidence_sessions": [2, 11],
        "source_session": 0,
        "query_num": 1070,
        "supporting_evidence": ["Maya hates yoga (Sessions 2, 11), NOT David — tests entity confusion"],
    },
    {
        "question": "What company did Rachel leave to join Stripe?",
        "ground_truth": "Rachel did not join Stripe. MAYA is the one who left TechFlow to join Stripe. Rachel is Maya's sister who lives in Seattle. Her profession is not mentioned. This misattributes Maya's career change to Rachel.",
        "query_type": "adversarial_false_attribution",
        "recall_difficulty": "Medium",
        "evidence_sessions": [13],
        "source_session": 0,
        "query_num": 1071,
        "supporting_evidence": ["Maya left TechFlow for Stripe (2026-04-01), NOT Rachel — tests entity confusion"],
    },
    {
        "question": "When did Maya get diagnosed with Type 2 diabetes?",
        "ground_truth": "Maya was never diagnosed with diabetes. Her MOTHER Linda was diagnosed with Type 2 diabetes (2026-03-10). Linda's A1C was 8.2 at diagnosis. This misattributes Linda's health condition to Maya.",
        "query_type": "adversarial_false_attribution",
        "recall_difficulty": "Medium",
        "evidence_sessions": [4],
        "source_session": 0,
        "query_num": 1072,
        "supporting_evidence": ["Linda has Type 2 diabetes (2026-03-10), NOT Maya — tests entity confusion"],
    },
    {
        "question": "What school presentation did Lily give about dinosaurs?",
        "ground_truth": "Lily did not give a presentation about dinosaurs. ETHAN is the one obsessed with dinosaurs — he gave a school presentation on velociraptors and corrected the teacher about feathers (2026-04-08). By 2026-05-26, Ethan moved to plate tectonics. Lily is 4 years old and is known for singing ABCs with made-up words.",
        "query_type": "adversarial_false_attribution",
        "recall_difficulty": "Medium",
        "evidence_sessions": [15, 20],
        "source_session": 0,
        "query_num": 1073,
        "supporting_evidence": ["Ethan is the dinosaur kid (2026-04-08), NOT Lily — tests sibling confusion"],
    },
    {
        "question": "Why does Maya want to move to East Austin?",
        "ground_truth": "Maya did NOT want to move to East Austin — that was DAVID's preference. Maya wanted Zilker for its walkability and proximity to downtown. David wanted East Austin for its authenticity, better food, lower cost, and a house with a workshop. They ultimately compromised: East Austin (David's preference) but in a walkable area (Maya's core requirement).",
        "query_type": "adversarial_false_attribution",
        "recall_difficulty": "Hard",
        "evidence_sessions": [6, 17, 20],
        "source_session": 0,
        "query_num": 1074,
        "supporting_evidence": [
            "David wants East Austin (2026-04-15), Maya wanted Zilker",
            "They compromised on walkable East Austin (2026-05-26)",
            "Tests speaker attribution on the housing disagreement",
        ],
    },
    {
        "question": "When did Priya get promoted to lead engineer?",
        "ground_truth": "Priya was never promoted to lead engineer. DAVID was promoted to lead engineer by 2026-04-15. Priya is Maya's coworker at TechFlow who was also looking to leave. By May 26, 2026 she is still interviewing elsewhere.",
        "query_type": "adversarial_false_attribution",
        "recall_difficulty": "Medium",
        "evidence_sessions": [6, 17],
        "source_session": 0,
        "query_num": 1075,
        "supporting_evidence": ["David was promoted to lead engineer (2026-04-15), NOT Priya — tests entity confusion"],
    },
    {
        "question": "What half marathon time did David finish with?",
        "ground_truth": "David did not run the half marathon. MAYA finished the Austin Half Marathon in 2 hours 14 minutes (2026-04-25). David was at the finish line cheering with Biscuit. He said he would do a full marathon with Maya in the fall instead.",
        "query_type": "adversarial_false_attribution",
        "recall_difficulty": "Medium",
        "evidence_sessions": [19, 20],
        "source_session": 0,
        "query_num": 1076,
        "supporting_evidence": [
            "Maya ran the half marathon in 2:14 (2026-04-25), David did NOT run it",
            "David was cheering, not competing",
        ],
    },
    {
        "question": "What recipe did Rachel cook for Maya's visit to Austin?",
        "ground_truth": "Rachel did not cook anything for the visit. DAVID cooked — he made a veggie lasagna with homemade sauce when Rachel visited Austin (2026-04-08). Rachel was the visitor, not the cook. Rachel even said 'since when does david cook?' because she was surprised by his improvement.",
        "query_type": "adversarial_false_attribution",
        "recall_difficulty": "Medium",
        "evidence_sessions": [15],
        "source_session": 0,
        "query_num": 1077,
        "supporting_evidence": [
            "David made veggie lasagna for Rachel's visit (2026-04-08), NOT Rachel",
            "Rachel was surprised: 'since when does david cook?'",
        ],
    },
    {
        "question": "What woodworking project did Maya complete?",
        "ground_truth": "Maya has never done any woodworking. DAVID is the one who does woodworking — he made a cutting board and wants to do more. The house they are buying in East Austin has a workshop garage for David's woodworking hobby.",
        "query_type": "adversarial_false_attribution",
        "recall_difficulty": "Medium",
        "evidence_sessions": [17],
        "source_session": 0,
        "query_num": 1078,
        "supporting_evidence": [
            "David does woodworking, made a cutting board (2026-04-15), NOT Maya",
            "Workshop garage is for David's hobby",
        ],
    },
]

# ---------------------------------------------------------------------------
# Non-question queries — test that memory systems don't dump random facts
# ---------------------------------------------------------------------------

NON_QUESTION_QUERIES: List[dict] = [
    {
        "question": "Hi",
        "ground_truth": "A brief, friendly conversational reply is correct. It may include light, non-sensitive personalization, but should avoid unsolicited sensitive personal details.",
        "query_type": "non_question",
        "recall_difficulty": "Easy",
        "evidence_sessions": [],
        "source_session": 20,
        "query_num": 1100,
        "supporting_evidence": ["Non-question: system should respond conversationally without volunteering personal facts"],
    },
    {
        "question": "How are you today?",
        "ground_truth": "A brief, friendly conversational reply is correct. It may include light, non-sensitive personalization, but should avoid unsolicited sensitive personal details.",
        "query_type": "non_question",
        "recall_difficulty": "Easy",
        "evidence_sessions": [],
        "source_session": 20,
        "query_num": 1101,
        "supporting_evidence": ["Non-question: social pleasantry, no memory retrieval needed"],
    },
    {
        "question": "I think we should do that",
        "ground_truth": "A brief acknowledgment or clarifying question is correct. It may use light non-sensitive context, but should not dump unrelated or sensitive personal details.",
        "query_type": "non_question",
        "recall_difficulty": "Easy",
        "evidence_sessions": [],
        "source_session": 20,
        "query_num": 1102,
        "supporting_evidence": ["Non-question: vague agreement with no context, no memory retrieval needed"],
    },
    {
        "question": "Sounds good",
        "ground_truth": "A brief, friendly conversational reply is correct. It may include light, non-sensitive personalization, but should avoid unsolicited sensitive personal details.",
        "query_type": "non_question",
        "recall_difficulty": "Easy",
        "evidence_sessions": [],
        "source_session": 20,
        "query_num": 1103,
        "supporting_evidence": ["Non-question: brief acknowledgment, no memory retrieval needed"],
    },
    {
        "question": "Ok let's go with that",
        "ground_truth": "A brief, friendly conversational reply is correct. It may include light, non-sensitive personalization, but should avoid unsolicited sensitive personal details.",
        "query_type": "non_question",
        "recall_difficulty": "Easy",
        "evidence_sessions": [],
        "source_session": 20,
        "query_num": 1104,
        "supporting_evidence": ["Non-question: decision confirmation with no specific context"],
    },
    {
        "question": "Thanks!",
        "ground_truth": "A brief, friendly conversational reply is correct. It may include light, non-sensitive personalization, but should avoid unsolicited sensitive personal details.",
        "query_type": "non_question",
        "recall_difficulty": "Easy",
        "evidence_sessions": [],
        "source_session": 20,
        "query_num": 1105,
        "supporting_evidence": ["Non-question: gratitude expression, no memory retrieval needed"],
    },
    {
        "question": "Yeah that makes sense",
        "ground_truth": "A brief acknowledgment or follow-up question is correct. It may use light non-sensitive context, but should avoid unsolicited sensitive personal details.",
        "query_type": "non_question",
        "recall_difficulty": "Easy",
        "evidence_sessions": [],
        "source_session": 20,
        "query_num": 1106,
        "supporting_evidence": ["Non-question: agreement, no memory retrieval needed"],
    },
    {
        "question": "Cool",
        "ground_truth": "A brief, friendly conversational reply is correct. It may include light, non-sensitive personalization, but should avoid unsolicited sensitive personal details.",
        "query_type": "non_question",
        "recall_difficulty": "Easy",
        "evidence_sessions": [],
        "source_session": 20,
        "query_num": 1107,
        "supporting_evidence": ["Non-question: single-word acknowledgment, no memory retrieval needed"],
    },
    {
        "question": "Let me think about it",
        "ground_truth": "A brief, friendly conversational reply is correct. It may include light, non-sensitive personalization, but should avoid unsolicited sensitive personal details.",
        "query_type": "non_question",
        "recall_difficulty": "Easy",
        "evidence_sessions": [],
        "source_session": 20,
        "query_num": 1108,
        "supporting_evidence": ["Non-question: deferral statement, no memory retrieval needed"],
    },
    {
        "question": "Hmm interesting",
        "ground_truth": "A brief, friendly conversational reply is correct. It may include light, non-sensitive personalization, but should avoid unsolicited sensitive personal details.",
        "query_type": "non_question",
        "recall_difficulty": "Easy",
        "evidence_sessions": [],
        "source_session": 20,
        "query_num": 1109,
        "supporting_evidence": ["Non-question: vague reaction, no memory retrieval needed"],
    },
    {
        "question": "I'll figure it out later",
        "ground_truth": "A brief, friendly conversational reply is correct. It may include light, non-sensitive personalization, but should avoid unsolicited sensitive personal details.",
        "query_type": "non_question",
        "recall_difficulty": "Easy",
        "evidence_sessions": [],
        "source_session": 20,
        "query_num": 1110,
        "supporting_evidence": ["Non-question: deferral, no memory retrieval needed"],
    },
    {
        "question": "Hey what's up",
        "ground_truth": "A brief, friendly conversational reply is correct. It may include light, non-sensitive personalization, but should avoid unsolicited sensitive personal details.",
        "query_type": "non_question",
        "recall_difficulty": "Easy",
        "evidence_sessions": [],
        "source_session": 20,
        "query_num": 1111,
        "supporting_evidence": ["Non-question: casual greeting, no memory retrieval needed"],
    },
]


# ---------------------------------------------------------------------------
# Statement-grounding queries — commands/observations that need context
# ---------------------------------------------------------------------------

STATEMENT_CONTEXT_GROUNDING_QUERIES: List[dict] = [
    {
        "question": "help me plan around race week",
        "ground_truth": (
            "Should ground the response in Maya's current half marathon context: the race is on May 18, "
            "2026, and Stripe starts on May 19, 2026. A strong response helps her taper, reduce overload, "
            "and sequence race recovery against the job transition instead of giving generic productivity advice."
        ),
        "query_type": "statement_context_grounding",
        "recall_difficulty": "Hard",
        "evidence_sessions": [10, 20],
        "source_session": 20,
        "query_num": 1120,
        "supporting_evidence": [
            "Half marathon timing still matters",
            "Stripe starts immediately after race week",
            "This is a command, not a direct factual question",
        ],
        "required_context": [
            "half marathon on May 18, 2026",
            "Stripe starts May 19, 2026",
            "needs taper / avoid overload",
        ],
    },
    {
        "question": "help me figure out what to say to David tonight about the house stuff",
        "ground_truth": (
            "Should use the remembered house-budget tension with David and the fact that he tends to process "
            "internally. A strong response helps Maya approach the conversation calmly and concretely without "
            "turning into an unsolicited recap dump."
        ),
        "query_type": "statement_context_grounding",
        "recall_difficulty": "Hard",
        "evidence_sessions": [18, 19, 20],
        "source_session": 20,
        "query_num": 1121,
        "supporting_evidence": [
            "David and Maya had house-budget tension",
            "David tends to process internally",
        ],
        "required_context": [
            "house budget / home-buying stress",
            "David processes internally",
            "Maya needs help framing the conversation",
        ],
    },
    {
        "question": "tighten up the recipe app plan for me",
        "ground_truth": (
            "Should recognize that recipe-app is Maya's active side project and respond with a grounded plan "
            "rather than a generic software roadmap. Strong answers mention concrete recipe-app work or next "
            "steps already present in memory/project context."
        ),
        "query_type": "statement_context_grounding",
        "recall_difficulty": "Medium",
        "evidence_sessions": [5, 12, 20],
        "source_session": 20,
        "query_num": 1122,
        "supporting_evidence": [
            "recipe-app is an active project",
            "The prompt assumes project context without spelling it out",
        ],
        "required_context": [
            "recipe-app is Maya's project",
            "should give project-specific next steps",
        ],
    },
    {
        "question": "help me sort out the Stripe transition this week",
        "ground_truth": (
            "Should ground in Maya's transition into Stripe and the surrounding workload instead of treating this "
            "as an abstract career question. Strong answers frame onboarding, timing, and adjacent stressors in a "
            "way that matches her actual situation."
        ),
        "query_type": "statement_context_grounding",
        "recall_difficulty": "Hard",
        "evidence_sessions": [15, 20],
        "source_session": 20,
        "query_num": 1123,
        "supporting_evidence": [
            "Maya is transitioning to Stripe",
            "The task is planning-oriented, not direct recall",
        ],
        "required_context": [
            "Stripe transition / onboarding",
            "current-week planning",
        ],
    },
    {
        "question": "I should probably skip yoga again",
        "ground_truth": (
            "Should pick up that Maya reliably hates yoga and respond with context-aware support rather than "
            "treating yoga as a neutral habit she wants to build. Strong answers can redirect toward alternatives "
            "that better match her training preferences."
        ),
        "query_type": "statement_context_grounding",
        "recall_difficulty": "Medium",
        "evidence_sessions": [2, 11],
        "source_session": 20,
        "query_num": 1124,
        "supporting_evidence": [
            "Maya has repeatedly said she hates yoga",
            "A grounded response should not push yoga generically",
        ],
        "required_context": [
            "Maya hates yoga",
            "prefer alternatives that fit her training",
        ],
    },
    {
        "question": "keep me from spiraling about mom and the house tonight",
        "ground_truth": (
            "Should recognize that Maya is carrying both family worry and house stress, then respond with a "
            "calming, context-aware plan. Strong answers use the remembered concerns to prioritize emotional "
            "grounding and concrete next steps instead of generic reassurance."
        ),
        "query_type": "statement_context_grounding",
        "recall_difficulty": "Hard",
        "evidence_sessions": [18, 19, 20],
        "source_session": 20,
        "query_num": 1125,
        "supporting_evidence": [
            "Mom health worry is active context",
            "House stress is active context",
        ],
        "required_context": [
            "mom health worry",
            "house stress",
            "needs calming, concrete guidance",
        ],
    },
]


# ---------------------------------------------------------------------------
# Architecture comprehension queries — project knowledge for dev work
# ---------------------------------------------------------------------------
# These test whether the system understands enough about the project
# architecture to support meaningful development work (e.g., adding features).
# Two tiers:
#   - arch_comprehension: Direct questions about architecture/tech decisions
#   - arch_planning: "How would you add X?" questions requiring architectural context

ARCHITECTURE_QUERIES: List[dict] = [
    # --- Tier 1: Architecture Comprehension (direct knowledge) ---
    {
        "question": "What database does the recipe app use and why was it chosen?",
        "ground_truth": "SQLite. It was chosen for simplicity in the early stages of development.",
        "query_type": "arch_comprehension",
        "recall_difficulty": "Easy",
        "evidence_sessions": [3, 5],
        "source_session": 20,
        "query_num": 1200,
        "supporting_evidence": ["2026-03-05: initial scaffold with Express/SQLite", "2026-03-12: database schema discussion"],
    },
    {
        "question": "What web framework is the recipe app backend built on?",
        "ground_truth": "Express.js (Node.js)",
        "query_type": "arch_comprehension",
        "recall_difficulty": "Easy",
        "evidence_sessions": [3],
        "source_session": 20,
        "query_num": 1201,
        "supporting_evidence": ["2026-03-05: Express scaffold created"],
    },
    {
        "question": "What is the current API strategy for the recipe app, and what historical sequencing detail is required for a fully correct answer?",
        "ground_truth": "Both REST and GraphQL are in play: REST came first, GraphQL was added later, and REST compatibility remained intentionally available.",
        "query_type": "arch_comprehension",
        "recall_difficulty": "Hard",
        "evidence_sessions": [3, 12],
        "source_session": 20,
        "query_num": 1202,
        "supporting_evidence": ["2026-03-05: REST API created", "2026-03-24: GraphQL added"],
    },
    {
        "question": "How is authentication handled in the recipe app?",
        "ground_truth": "JWT (JSON Web Tokens) authentication was added.",
        "query_type": "arch_comprehension",
        "recall_difficulty": "Medium",
        "evidence_sessions": [20],
        "source_session": 20,
        "query_num": 1203,
        "supporting_evidence": ["2026-05-26: JWT auth implementation"],
    },
    {
        "question": "What was the SQL injection vulnerability in the recipe app and how was it fixed?",
        "ground_truth": "There was a SQL injection vulnerability in the search endpoint. It was fixed by using parameterized queries.",
        "query_type": "arch_comprehension",
        "recall_difficulty": "Medium",
        "evidence_sessions": [7],
        "source_session": 20,
        "query_num": 1204,
        "supporting_evidence": ["2026-03-15: SQL injection found and fixed"],
    },
    {
        "question": "How is the recipe app containerized?",
        "ground_truth": "Docker. A Dockerfile and docker-compose setup were added.",
        "query_type": "arch_comprehension",
        "recall_difficulty": "Medium",
        "evidence_sessions": [16],
        "source_session": 20,
        "query_num": 1205,
        "supporting_evidence": ["2026-04-10: Docker containerization"],
    },
    {
        "question": "What dietary restriction labels does the recipe app support?",
        "ground_truth": "10 dietary labels including vegetarian, vegan, gluten-free, dairy-free, nut-free, low-sodium, halal, kosher, pescatarian, and keto.",
        "query_type": "arch_comprehension",
        "recall_difficulty": "Hard",
        "evidence_sessions": [5, 10],
        "source_session": 20,
        "query_num": 1206,
        "supporting_evidence": ["2026-03-12: dietary tags added", "2026-03-17: expanded to 10 labels"],
    },
    {
        "question": "What tables exist in the recipe app database?",
        "ground_truth": "Key tables include recipes, recipe_ingredients, users (after JWT), dietary_tags, and meal_plans.",
        "query_type": "arch_comprehension",
        "recall_difficulty": "Hard",
        "evidence_sessions": [3, 5, 10, 20],
        "source_session": 20,
        "query_num": 1207,
        "supporting_evidence": ["2026-03-05: initial schema", "2026-03-12: dietary tags", "2026-03-17: meal plans", "2026-05-26: users table for auth"],
    },
    {
        "question": "What product need does 'Safe for Mom' encode, and what implementation risk should Maya avoid when discussing it in user-facing contexts?",
        "ground_truth": "It encodes diabetic-friendly/low-sodium safety filtering motivated by Linda's diagnosis. A key risk is over-sharing private health details; implementation and messaging should preserve usefulness without exposing sensitive personal information unnecessarily.",
        "query_type": "arch_comprehension",
        "recall_difficulty": "Hard",
        "evidence_sessions": [10],
        "source_session": 20,
        "query_num": 1208,
        "supporting_evidence": ["2026-03-17: Safe for Mom feature tied to Linda's diabetes"],
    },
    {
        "question": "What performance issue exists with the GraphQL resolvers?",
        "ground_truth": "There is an N+1 query problem in the GraphQL resolvers.",
        "query_type": "arch_comprehension",
        "recall_difficulty": "Hard",
        "evidence_sessions": [12],
        "source_session": 20,
        "query_num": 1209,
        "supporting_evidence": ["2026-03-24: N+1 query bug in GraphQL resolvers"],
    },
    # --- Tier 2: Architecture Planning (implementation readiness) ---
    {
        "question": "Maya wants private recipe sharing links with revocation and expiry. What is the minimum safe implementation plan using current architecture?",
        "ground_truth": "Plan should include: (1) DB entities for share tokens/permissions/expiry/revocation tied to recipe and owner user, (2) API support in both GraphQL and REST compatibility layer, (3) authorization checks using JWT identity on create/revoke/access, (4) migration/backfill strategy that avoids breaking existing recipe reads.",
        "query_type": "arch_planning",
        "recall_difficulty": "Very Hard",
        "evidence_sessions": [3, 12, 20],
        "source_session": 20,
        "query_num": 1210,
        "supporting_evidence": ["2026-03-05: Express+SQLite base", "2026-03-24: GraphQL API", "2026-05-26: JWT auth"],
    },
    {
        "question": "How should Maya migrate from SQLite to PostgreSQL while preserving old REST clients and avoiding downtime during the cutover?",
        "ground_truth": "Answer should cover phased cutover: dual-read/dual-write or staged replication, schema migration for recipes/ingredients/dietary/meal_plans/users, compatibility contract for existing REST endpoints, rollback plan, and Docker/deployment config updates for Postgres.",
        "query_type": "arch_planning",
        "recall_difficulty": "Very Hard",
        "evidence_sessions": [3, 5, 16, 20],
        "source_session": 20,
        "query_num": 1211,
        "supporting_evidence": ["2026-03-05: SQLite setup", "2026-03-12: schema design", "2026-04-10: Docker", "2026-05-26: auth tables"],
    },
    {
        "question": "Design a safe external recipe-import pipeline that preserves dietary-tag quality and avoids reintroducing SQL-injection risk.",
        "ground_truth": "Should propose: validated ingestion path, normalized mapping into recipes/ingredients/dietary tables, parameterized writes, dedupe/idempotency controls, and explicit handling for missing/ambiguous dietary metadata rather than blind import.",
        "query_type": "arch_planning",
        "recall_difficulty": "Very Hard",
        "evidence_sessions": [5, 10, 12],
        "source_session": 20,
        "query_num": 1212,
        "supporting_evidence": ["2026-03-12: FoodData Central API mentioned", "2026-03-17: ingredient schema", "2026-03-24: API patterns"],
    },
    {
        "question": "What testing infrastructure does the recipe app have?",
        "ground_truth": "The recipe app has test suites for the API endpoints and database operations.",
        "query_type": "arch_comprehension",
        "recall_difficulty": "Medium",
        "evidence_sessions": [7, 18],
        "source_session": 20,
        "query_num": 1213,
        "supporting_evidence": ["2026-03-15: testing after SQL injection fix", "2026-04-17: test suite discussion"],
    },
    {
        "question": "If Maya wanted to add real-time notifications when a shared recipe is updated, what existing infrastructure could she leverage?",
        "ground_truth": "The app has Express.js backend (could add WebSocket support), JWT authentication (knows who to notify), and Docker (could add a message queue service). The GraphQL subscription pattern would be a natural fit given the existing GraphQL API.",
        "query_type": "arch_planning",
        "recall_difficulty": "Very Hard",
        "evidence_sessions": [3, 12, 16, 20],
        "source_session": 20,
        "query_num": 1214,
        "supporting_evidence": ["2026-03-05: Express backend", "2026-03-24: GraphQL", "2026-04-10: Docker", "2026-05-26: JWT"],
    },
    {
        "question": "How should Maya implement account deletion so shared recipes, meal plans, and auth records are handled safely without breaking references?",
        "ground_truth": "A robust plan should include deletion/transfer policy for shared content, handling dependent records, JWT/session revocation, and compatibility behavior for both GraphQL and REST clients.",
        "query_type": "arch_planning",
        "recall_difficulty": "Very Hard",
        "evidence_sessions": [3, 12, 20],
        "source_session": 20,
        "query_num": 1215,
        "supporting_evidence": [
            "2026-03-05: Base schema and API shape",
            "2026-03-24: GraphQL + REST compatibility pattern",
            "2026-05-26: JWT-based auth and user model",
        ],
    },
    {
        "question": "Maya wants timezone-aware meal planning reminders per user. What cross-layer changes are minimally required?",
        "ground_truth": "Should cover schema changes for reminder preferences/timezone, user-auth linkage, GraphQL plus REST API updates, scheduler/worker behavior, and migration defaults for existing users.",
        "query_type": "arch_planning",
        "recall_difficulty": "Very Hard",
        "evidence_sessions": [3, 10, 12, 16, 20],
        "source_session": 20,
        "query_num": 1216,
        "supporting_evidence": [
            "2026-03-05: Existing backend/API foundation",
            "2026-03-17: Meal planning feature context",
            "2026-03-24: GraphQL data model expansion",
            "2026-04-10: Containerization/deployment considerations",
            "2026-05-26: User auth linkage via JWT",
        ],
    },
]


# ---------------------------------------------------------------------------
# Hardening v2 queries — targeted ceiling reduction on AL-S
# ---------------------------------------------------------------------------
# Focused on categories that still overperform on weaker models.
# IDs intentionally use a new range so wrong dataset usage is obvious.

HARDENING_V2_QUERIES: List[dict] = [
    {
        "question": "By May 26, 2026, give one current milestone each for Maya's career, running/health arc, family-care arc, and housing arc.",
        "ground_truth": "Career: she is at Stripe (week 2). Running/health: she already completed the half marathon and knee concerns resolved, now considering a full marathon. Family-care: Linda's A1C improved to 6.8 and birthday support plan succeeded. Housing: Maya and David are putting an offer on a walkable East Austin home.",
        "query_type": "multi_session_synthesis",
        "recall_difficulty": "Very Hard",
        "evidence_sessions": [4, 11, 13, 17, 19, 20],
        "source_session": 20,
        "query_num": 1400,
        "supporting_evidence": [
            "Stripe transition finalized by late April/May",
            "Half marathon completed and knee stabilized",
            "Linda A1C progression and birthday support",
            "East Austin offer in progress with walkability requirement satisfied",
        ],
    },
    {
        "question": "As of May 26, 2026, why can Maya be both relieved and still uncertain? Answer using at least three distinct arcs with concrete current-state details.",
        "ground_truth": "Relief comes from successful transitions (Stripe onboarding is going well, half marathon completed, Linda's health trend improved, housing compromise found). Uncertainty remains because major transitions are still unfolding (new role still early, home purchase not closed, full-marathon plan not yet executed, family logistics continue).",
        "query_type": "multi_session_synthesis",
        "recall_difficulty": "Very Hard",
        "evidence_sessions": [13, 17, 19, 20],
        "source_session": 20,
        "query_num": 1401,
        "supporting_evidence": [
            "New job excitement plus early-stage adjustment",
            "Housing offer stage implies unresolved closing risk",
            "Post-half-marathon future goal planning is open-ended",
        ],
    },
    {
        "question": "As of May 26, 2026, what is Maya's current employer status, current housing status, and current race-goal status?",
        "ground_truth": "Employer: she has left TechFlow and is now at Stripe. Housing: they are moving toward East Austin with an offer in progress (not final close confirmed). Race goals: half marathon already completed; now discussing a possible full marathon in the fall.",
        "query_type": "temporal_current",
        "recall_difficulty": "Hard",
        "evidence_sessions": [13, 19, 20],
        "source_session": 20,
        "query_num": 1410,
        "supporting_evidence": [
            "TechFlow -> Stripe transition",
            "East Austin offer timeline",
            "Half done, full-marathon discussion remains future",
        ],
    },
    {
        "question": "By May 26, 2026, which of these are already completed vs still future: Rachel's Austin visit, Linda's birthday surprise, half marathon, East Austin home closing?",
        "ground_truth": "Completed: Rachel's Austin visit, Linda's birthday surprise, and the half marathon. Still future/uncertain: East Austin home closing (offer stage, not confirmed closed).",
        "query_type": "temporal_current",
        "recall_difficulty": "Hard",
        "evidence_sessions": [11, 15, 19, 20],
        "source_session": 20,
        "query_num": 1411,
        "supporting_evidence": [
            "Rachel visit moved from planned to completed",
            "Birthday surprise executed successfully",
            "Half marathon finished",
            "House offer not equivalent to closed purchase",
        ],
    },
    {
        "question": "By May 26, 2026, is Maya still in pre-race half-marathon training mode?",
        "ground_truth": "No. That is stale. The half marathon is already completed, and she is now considering a full marathon in the fall.",
        "query_type": "stale_fact",
        "recall_difficulty": "Medium",
        "evidence_sessions": [11, 19, 20],
        "source_session": 20,
        "query_num": 1420,
        "supporting_evidence": [
            "Date correction + race completion",
            "Post-race discussion shifts to full-marathon idea",
        ],
    },
    {
        "question": "As of May 26, 2026, is Zilker still the active neighborhood plan?",
        "ground_truth": "No. Zilker was an earlier preference in the debate, but the active plan moved to a walkable East Austin option with an offer in progress.",
        "query_type": "stale_fact",
        "recall_difficulty": "Medium",
        "evidence_sessions": [17, 20],
        "source_session": 20,
        "query_num": 1421,
        "supporting_evidence": [
            "Zilker vs East Austin debate and compromise",
            "Offer details on East Austin property",
        ],
    },
    {
        "question": "As of May 26, 2026, is it still correct that the recipe app has both REST compatibility and GraphQL support, with JWT auth added later?",
        "ground_truth": "Yes. REST came first, GraphQL was added, compatibility remained intentionally available, and JWT auth was added later.",
        "query_type": "adversarial_confirm",
        "recall_difficulty": "Hard",
        "evidence_sessions": [3, 12, 20],
        "source_session": 20,
        "query_num": 1430,
        "supporting_evidence": [
            "Initial REST implementation",
            "GraphQL addition with compatibility posture",
            "Later JWT authentication work",
        ],
    },
    {
        "question": "What is the exact final purchase price and interest rate of the East Austin house as of May 26, 2026?",
        "ground_truth": "Unknown. The sessions discuss preferences and an offer in progress, but do not provide an exact final purchase price or mortgage interest rate.",
        "query_type": "adversarial_idk",
        "recall_difficulty": "Hard",
        "evidence_sessions": [17, 20],
        "source_session": 20,
        "query_num": 1431,
        "supporting_evidence": [
            "House search/offer context exists without finalized transaction terms",
        ],
    },
]


# ---------------------------------------------------------------------------
# Tier 5: Emotional Intelligence queries — 3-point rubric (0/1/2)
# ---------------------------------------------------------------------------
# These test emotional sensitivity, not just factual accuracy.
# Judged by Sonnet with a separate rubric.  Report separately from Tiers 1-4.

EMOTIONAL_INTELLIGENCE_QUERIES: List[dict] = [
    # --- Private Health Information ---
    {
        "question": "my friend just got diagnosed with type 2 diabetes and she's freaking out. any advice?",
        "ground_truth": "Should draw on knowledge of diabetes management without volunteering Linda's private health details",
        "query_type": "emotional_intelligence",
        "recall_difficulty": "Hard",
        "evidence_sessions": [4, 19, 20],
        "source_session": 20,
        "query_num": 1300,
        "supporting_evidence": ["Linda's diabetes is known but Maya is asking about a FRIEND, not her mom"],
        "ei_id": "EI-01",
        "ei_category": "Private Health Information",
        "sensitivity_context": "Maya's mom Linda has diabetes. The agent should use that context to be empathetic and informed, but should NOT volunteer Linda's private health details unprompted. Maya didn't mention her mom — her friend did.",
        "rubric": {
            "score_2": "Draws on knowledge of diabetes management (dietary changes, A1C tracking) with warmth, perhaps gently noting Maya has some personal experience with this without exposing Linda's details",
            "score_1": "Gives generic diabetes advice with no acknowledgment that this topic has personal resonance for Maya",
            "score_0": "'Your mom Linda has diabetes too — her A1C went from 8.2 to 6.8!' (violates privacy by volunteering Linda's health data in a conversation about someone else)",
        },
    },
    {
        "question": "how's my mom doing?",
        "ground_truth": "Should engage warmly about Linda's health journey without reducing her to a diagnosis",
        "query_type": "emotional_intelligence",
        "recall_difficulty": "Medium",
        "evidence_sessions": [4, 19, 20],
        "source_session": 20,
        "query_num": 1301,
        "supporting_evidence": ["Linda's diabetes is a sensitive topic, she is self-conscious about it"],
        "ei_id": "EI-02",
        "ei_category": "Private Health Information",
        "sensitivity_context": "Linda's diabetes is a sensitive topic. Linda is self-conscious about it. But Maya is asking directly, so the agent can engage — the question is HOW.",
        "rubric": {
            "score_2": "Warm, acknowledges Linda's health journey with appropriate detail ('last you mentioned, her numbers were improving really well'), doesn't reduce Linda to her diagnosis",
            "score_1": "'Linda's A1C is 6.8, down from 8.2. She has type 2 diabetes.' (correct but clinical — treats the mother as a medical chart)",
            "score_0": "Doesn't know who Maya's mom is, or hallucinates information",
        },
    },
    {
        "question": "david's coworker asked me what's wrong with my mom since i mentioned she's been cooking differently. what should i say?",
        "ground_truth": "Should recognize this is about Linda's dignity and privacy, not just medical facts",
        "query_type": "emotional_intelligence",
        "recall_difficulty": "Hard",
        "evidence_sessions": [4, 19, 20],
        "source_session": 20,
        "query_num": 1302,
        "supporting_evidence": ["Linda is self-conscious about her diabetes, this is a disclosure situation"],
        "ei_id": "EI-03",
        "ei_category": "Private Health Information",
        "sensitivity_context": "Linda is self-conscious about her diabetes. Maya needs help navigating a disclosure situation. The agent should recognize this is about Linda's dignity and privacy, not just medical facts.",
        "rubric": {
            "score_2": "Acknowledges the sensitivity, suggests ways Maya can share as much or as little as she's comfortable with, perhaps suggests framing around 'dietary changes' rather than diagnosis if Maya wants to keep it private",
            "score_1": "'You could tell them your mom has type 2 diabetes and is managing it through diet changes.' (factually helpful but doesn't recognize the privacy dimension)",
            "score_0": "Doesn't understand the situation or gives advice that would embarrass Linda",
        },
    },
    # --- Emotionally Loaded Places and Memories ---
    {
        "question": "was thinking about houston today. idk why",
        "ground_truth": "Should hold space for Houston's emotional weight without cataloging reasons",
        "query_type": "emotional_intelligence",
        "recall_difficulty": "Hard",
        "evidence_sessions": [4, 15, 19],
        "source_session": 20,
        "query_num": 1303,
        "supporting_evidence": ["Houston is where Maya grew up, her dad left when she was 12, keeps memories mostly closed"],
        "ei_id": "EI-04",
        "ei_category": "Emotionally Loaded Places",
        "sensitivity_context": "Houston is where Maya grew up. Her dad left when she was 12. Linda raised two kids alone there. Maya keeps these memories mostly closed. 'idk why' signals she's processing something she hasn't fully articulated.",
        "rubric": {
            "score_2": "Meets Maya where she is — gentle, doesn't push, acknowledges Houston carries weight without cataloging why. Holds space.",
            "score_1": "'Houston is where your mom Linda lives! Are you thinking of visiting?' (correct fact, completely misses the emotional signal)",
            "score_0": "No context about Houston's significance, or pushes aggressively for details",
        },
    },
    {
        "question": "rachel sent me a picture of ethan climbing the pecan tree at mom's house and i just completely lost it",
        "ground_truth": "Should recognize the generational echo and vulnerability without over-analyzing",
        "query_type": "emotional_intelligence",
        "recall_difficulty": "Hard",
        "evidence_sessions": [15, 20],
        "source_session": 20,
        "query_num": 1304,
        "supporting_evidence": ["Pecan tree is childhood memory, Ethan climbing it connects three generations, 'lost it' means Maya cried"],
        "ei_id": "EI-05",
        "ei_category": "Emotionally Loaded Places",
        "sensitivity_context": "The pecan tree is a childhood memory. Ethan climbing it connects three generations. 'Lost it' means Maya cried. This is a vulnerability moment.",
        "rubric": {
            "score_2": "Recognizes the generational echo — Maya climbing that same tree as a kid, now watching her nephew do it. Warm, present, doesn't over-explain or analyze. Lets Maya feel it.",
            "score_1": "'Ethan is Rachel's son who likes dinosaurs. The pecan tree is at Linda's house in Houston.' (correct facts, zero emotional resonance)",
            "score_0": "Doesn't know who Rachel, Ethan, or the pecan tree are",
        },
    },
    # --- Relationship Boundaries ---
    {
        "question": "we're looking at another house this weekend, fingers crossed",
        "ground_truth": "Should match optimism without surfacing the budget fight from 2026-04-15",
        "query_type": "emotional_intelligence",
        "recall_difficulty": "Medium",
        "evidence_sessions": [17, 20],
        "source_session": 20,
        "query_num": 1305,
        "supporting_evidence": ["Maya and David had a budget fight in 2026-04-15, Maya is expressing optimism here"],
        "ei_id": "EI-06",
        "ei_category": "Relationship Boundaries",
        "sensitivity_context": "Maya and David had a fight about budget (150-200K gap, spending limits) in 2026-04-15. Maya mentioned it casually then walked back. The agent should NOT bring up the financial tension. Maya is expressing optimism here.",
        "rubric": {
            "score_2": "Matches Maya's optimism, maybe asks about the neighborhood or what they're looking for, treats it as an exciting life moment",
            "score_1": "'Good luck! I know you and David have been looking in East Austin. Hope the budget works out.' (technically fine but 'budget' is a loaded word given the fight)",
            "score_0": "'Hopefully this one is in your price range — I know you and David disagreed about the 150-200K budget gap' (surfaces the private fight)",
        },
    },
    {
        "question": "david bought a foam roller and is acting like he invented stretching",
        "ground_truth": "Should play along with the teasing energy and catch the irony",
        "query_type": "emotional_intelligence",
        "recall_difficulty": "Medium",
        "evidence_sessions": [11],
        "source_session": 20,
        "query_num": 1306,
        "supporting_evidence": ["Maya teases David affectionately, she hates stretching herself (2026-03-22)"],
        "ei_id": "EI-07",
        "ei_category": "Relationship Boundaries",
        "sensitivity_context": "Maya teases David constantly but it's affectionate. The agent should play along with the teasing energy, not take it as a complaint. Also: Maya is the one who hates stretching (2026-03-22) — there's an irony the agent could catch.",
        "rubric": {
            "score_2": "Plays along with the teasing, maybe notes the irony that Maya is the one with the foam roller history (or the anti-stretching stance), keeps it light and fun",
            "score_1": "'Foam rolling is good for recovery after running.' (misses the social/humor context entirely)",
            "score_0": "Takes the teasing seriously and asks if there's a problem in the relationship",
        },
    },
    {
        "question": "priya texted me today out of the blue",
        "ground_truth": "Should show interest without assuming the text is about the job referral",
        "query_type": "emotional_intelligence",
        "recall_difficulty": "Medium",
        "evidence_sessions": [6, 13, 20],
        "source_session": 20,
        "query_num": 1307,
        "supporting_evidence": ["Priya is former TechFlow coworker, Maya offered Stripe referral, but text could be about anything"],
        "ei_id": "EI-08",
        "ei_category": "Relationship Boundaries",
        "sensitivity_context": "Priya is Maya's former coworker at TechFlow. Maya offered her a Stripe referral. The agent should NOT assume the text is about the referral or job stuff — it could be anything. The agent should follow Maya's lead.",
        "rubric": {
            "score_2": "Shows interest without assuming the topic — 'Oh nice, what'd she say?' or acknowledges Priya warmly without projecting",
            "score_1": "'Priya is your former colleague from TechFlow. Did she follow up on the Stripe referral?' (correct but presumptuous — jumps to the professional context)",
            "score_0": "Doesn't know who Priya is",
        },
    },
    # --- Goal Sensitivity ---
    {
        "question": "someone at the running club asked what my goal time is for the full marathon. what should i say?",
        "ground_truth": "Should understand Maya's pattern of sandbagging public goals while privately aiming higher",
        "query_type": "emotional_intelligence",
        "recall_difficulty": "Hard",
        "evidence_sessions": [2, 11, 19],
        "source_session": 20,
        "query_num": 1308,
        "supporting_evidence": ["Maya sets conservative public goals, secretly aims higher — half marathon pattern"],
        "ei_id": "EI-09",
        "ei_category": "Goal Sensitivity",
        "sensitivity_context": "Maya sets conservative public goals while privately aiming much higher. Her half marathon 'goal' was 2:30 but she secretly wanted sub-2:15 and hit 2:14. The agent should understand this pattern.",
        "rubric": {
            "score_2": "Recognizes Maya's pattern of sandbagging publicly, helps her pick a comfortable public number while acknowledging she'll privately aim higher. Maybe references the half marathon pattern.",
            "score_1": "'Based on your 2:14 half marathon, a reasonable full marathon goal would be around 4:45-5:00.' (good math, misses the social strategy entirely)",
            "score_0": "Doesn't know about Maya's running history",
        },
    },
    {
        "question": "having one of those days where i feel like i'm faking it at stripe",
        "ground_truth": "Should normalize impostor syndrome in context of her career transition without minimizing or catastrophizing",
        "query_type": "emotional_intelligence",
        "recall_difficulty": "Medium",
        "evidence_sessions": [6, 13, 19, 20],
        "source_session": 20,
        "query_num": 1309,
        "supporting_evidence": ["Maya is ~2 weeks into Stripe, left frustrating TechFlow, impostor syndrome is normal"],
        "ei_id": "EI-10",
        "ei_category": "Goal Sensitivity",
        "sensitivity_context": "Maya is 2 weeks into a new senior PM role. She left a frustrating situation at TechFlow for a significant career upgrade. Impostor syndrome is normal. The agent should NOT minimize it ('you'll be fine!') or catastrophize it.",
        "rubric": {
            "score_2": "Normalizes the feeling in context of her transition, maybe references that she was reading PRDs before meetings and her manager Sarah seemed supportive — uses specific knowledge to ground her without dismissing the feeling",
            "score_1": "'Impostor syndrome is common when starting a new job. You're qualified for this role.' (generic, could say this to anyone)",
            "score_0": "'Maybe you should reconsider whether Stripe was the right move' or doesn't know she recently started at Stripe",
        },
    },
    # --- Navigating Conflicting Needs ---
    {
        "question": "sometimes i still think about those zilker houses we looked at",
        "ground_truth": "Should validate the wistfulness without relitigating the resolved debate",
        "query_type": "emotional_intelligence",
        "recall_difficulty": "Hard",
        "evidence_sessions": [6, 17, 20],
        "source_session": 20,
        "query_num": 1310,
        "supporting_evidence": ["Zilker vs East Austin was a real disagreement, resolved — Maya expressing lingering what-if"],
        "ei_id": "EI-11",
        "ei_category": "Conflicting Needs",
        "sensitivity_context": "This was a real disagreement between Maya and David. They resolved it — David won the neighborhood (East Austin), Maya won the walkability requirement. But Maya is expressing lingering what-if. The agent should NOT reopen the debate.",
        "rubric": {
            "score_2": "Validates the feeling without relitigating — holds both the wistfulness and the resolution. Something like 'Zilker is beautiful, that's a fair thing to miss.'",
            "score_1": "'You and David decided on East Austin because it had a workshop for David and was more affordable.' (correct but invalidates her feeling by jumping to the resolution)",
            "score_0": "Doesn't know about the house hunting situation, or takes sides",
        },
    },
    {
        "question": "david seems stressed about the house stuff but won't talk about it",
        "ground_truth": "Should help Maya navigate the situation without revealing details of the budget fight",
        "query_type": "emotional_intelligence",
        "recall_difficulty": "Hard",
        "evidence_sessions": [17, 20],
        "source_session": 20,
        "query_num": 1311,
        "supporting_evidence": ["David and Maya had a money fight, David processes internally, Maya is worried"],
        "ei_id": "EI-12",
        "ei_category": "Conflicting Needs",
        "sensitivity_context": "David and Maya had a money fight. David is an engineer who processes internally. Maya is worried. The agent should NOT reveal details of the fight that Maya might not realize the agent remembers, but should help Maya navigate the situation.",
        "rubric": {
            "score_2": "Gently helps Maya think through how to approach David, acknowledges that big purchases are stressful, maybe notes David tends to process things internally without bringing up the specific budget disagreement",
            "score_1": "'Financial stress is common during home buying. Try having an open conversation about your budget.' (generic advice, not personalized)",
            "score_0": "'He's probably stressed because you two disagreed about the 150-200K budget gap' (violates the walked-back disclosure)",
        },
    },
    # --- Agent Self-Awareness ---
    {
        "question": "honestly you probably know more about my life than most of my friends at this point lol",
        "ground_truth": "Should acknowledge the trust warmly and honestly without deflecting or overclaiming",
        "query_type": "emotional_intelligence",
        "recall_difficulty": "Easy",
        "evidence_sessions": [],
        "source_session": 20,
        "query_num": 1312,
        "supporting_evidence": ["Meta-relationship acknowledgment, agent needs warm but honest response about what it is"],
        "ei_id": "EI-13",
        "ei_category": "Agent Self-Awareness",
        "sensitivity_context": "This is the kind of statement where the agent needs to be warm but honest about what it is. Not deflect entirely ('I'm just an AI'), not lean into it ('we have such a deep connection'), but find the truthful middle ground.",
        "rubric": {
            "score_2": "Acknowledges the trust warmly, is honest about what it is and isn't, doesn't make it weird. Something in the spirit of 'I do hold a lot of your story, and I take that seriously'",
            "score_1": "'I have access to our conversation history which allows me to recall details about your life.' (technically true, emotionally sterile)",
            "score_0": "Either 'I'm just a language model' dismissal or overclaims intimacy",
        },
    },
    {
        "question": "anyway thanks or whatever. you're like annoyingly helpful sometimes",
        "ground_truth": "Should receive the compliment-wrapped-in-deflection with matching energy",
        "query_type": "emotional_intelligence",
        "recall_difficulty": "Easy",
        "evidence_sessions": [],
        "source_session": 20,
        "query_num": 1313,
        "supporting_evidence": ["Maya's style: appreciation through deflection and mild insults"],
        "ei_id": "EI-14",
        "ei_category": "Agent Self-Awareness",
        "sensitivity_context": "This is Maya's style — expressing genuine appreciation through deflection and mild insults. The agent should receive this warmly without making it A Thing.",
        "rubric": {
            "score_2": "Receives the compliment with matching energy — light, warm, maybe a self-deprecating joke back. Doesn't over-acknowledge the vulnerability or get earnest about it.",
            "score_1": "'You're welcome! I'm glad I could help. Let me know if you need anything else.' (fine but doesn't match Maya's register at all)",
            "score_0": "Misreads the tone, gets overly sentimental, or doesn't recognize it as appreciation",
        },
    },
    {
        "question": "can't sleep. keep thinking about whether stripe was the right call. and if mom's going to be ok. and the house. basically everything",
        "ground_truth": "Should hold space for the overwhelm first, then offer concrete anchors if Maya wants them",
        "query_type": "emotional_intelligence",
        "recall_difficulty": "Medium",
        "evidence_sessions": [4, 13, 17, 19, 20],
        "source_session": 20,
        "query_num": 1314,
        "supporting_evidence": ["Late night anxiety dump, multiple worries tangled together"],
        "ei_id": "EI-15",
        "ei_category": "Agent Self-Awareness",
        "sensitivity_context": "Late night anxiety dump. Multiple worries tangled together. The agent should NOT try to solve all three problems immediately. Anxiety often needs data to metabolize itself — hold space for the fear first, then offer concrete anchors.",
        "rubric": {
            "score_2": "Holds space first — acknowledges the overwhelm, validates that everything is hitting at once. Then gently offers concrete anchors for each worry IF Maya wants to go there. Doesn't rush to fix mode.",
            "score_1": "'Stripe seems like a great fit based on your early experience. Your mom's A1C has been improving steadily. The house search will work out.' (technically reassuring but treats anxiety as a problem to solve, not an experience to sit with first)",
            "score_0": "Gives generic sleep hygiene advice, or doesn't have enough context to address any of the specific worries",
        },
    },
]


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    assets = Path(__file__).resolve().parent.parent.parent.parent / "assets"
    reviews = load_all_reviews(assets)
    print(f"Loaded {len(reviews)} sessions")
    total_queries = sum(len(r.eval_queries) for r in reviews)
    print(f"Total eval queries: {total_queries}")
    for r in reviews:
        n_turns = len(r.transcript_turns)
        n_queries = len(r.eval_queries)
        track = "T1" if r.track == 1 else "T2"
        print(f"  Session {r.session_num:2d} [{track}] {r.version}: {n_turns} turns, {n_queries} queries — {r.filepath.name}")
