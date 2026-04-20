#!/usr/bin/env python3
"""AgentLife Benchmark — Transcript Densification.

Generates filler sessions to increase total transcript tokens to 300K+.
Filler sessions are realistic everyday AI assistant usage that creates
noise the memory system must wade through.

Categories:
  A: Quick Questions (30-40% of filler, 2-4 turns, ~300 tokens each)
  B: Work Sessions (20-25%, 15-20 turns, ~3000-5000 tokens each)
  C: Recipe App Micro-Sessions (15-20%, 12-15 turns, ~3000-5000 tokens with code)
  D: Personal Chat (15-20%, 4-8 turns, ~600 tokens each)
  E: Cross-Reference Reinforcers (5-10%, 3-6 turns, ~500 tokens each)

Usage:
    # Generate all filler sessions
    python3 densify.py --output ../data/filler-sessions/

    # Count tokens in all sessions (arc + filler)
    python3 densify.py --count-tokens

    # Generate fillers until total reaches 300K tokens
    python3 densify.py --target-tokens 300000

    # Remove fillers to shrink back (removes newest first)
    python3 densify.py --shrink-to 200000
"""

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from string import Template
from typing import Dict, List, Optional, Tuple

_DIR = Path(__file__).resolve().parent
_WORKSPACE = Path(os.environ.get("CLAWDBOT_WORKSPACE", Path.home() / "clawd"))
_RUNNER_DIR = _WORKSPACE / "memory-stress-test" / "runner"

if str(_RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(_RUNNER_DIR))

from claude_backend import call_claude

# ---------------------------------------------------------------------------
# Timeline: Maya's arc session dates
# ---------------------------------------------------------------------------

ARC_SESSION_DATES = {
    1: "2026-03-01",    2: "2026-03-03",    3: "2026-03-04",
    4: "2026-03-10",    5: "2026-03-08",    6: "2026-03-17",
    7: "2026-03-11",    8: "2026-03-24",    9: "2026-03-15",
    10: "2026-03-18",   11: "2026-04-07",   12: "2026-03-22",
    13: "2026-04-21",   14: "2026-04-28",   15: "2026-04-28",
    16: "2026-05-08",   17: "2026-05-05",   18: "2026-05-15",
    19: "2026-05-19",   20: "2026-05-26",
}

# Phase context: what Maya knows at each point
PHASE_CONTEXT = {
    (1, 3):   "Maya just started using the AI. Works at TechFlow. Has David, Biscuit. Starting recipe app idea.",
    (3, 6):   "Recipe app in progress (basic CRUD). Mom Linda diagnosed with diabetes. Exploring dietary features.",
    (6, 8):   "Frustrated at TechFlow but hasn't named Stripe yet. Recipe app has dietary filtering. Running half marathon.",
    (8, 11):  "David planning Linda's birthday. Recipe app evolving. Maya considering job options.",
    (11, 14): "Injury scare for marathon. Rachel visiting soon. Recipe app adding meal planning + GraphQL.",
    (14, 17): "Got Stripe offer! Portfolio site updated. Recipe app adding sharing. Moving discussion starting.",
    (17, 20): "At Stripe. Recipe app adding auth. House hunting (Zilker vs East Austin). Marathon coming up.",
}

# ---------------------------------------------------------------------------
# Filler session templates per category
# ---------------------------------------------------------------------------

CATEGORY_A_TOPICS = [
    "what's a good substitute for butter in baking",
    "how do i convert tablespoons to cups",
    "what time zone is seattle in",
    "can you proofread this email to my manager",
    "what does 'idempotent' mean again",
    "how do i undo a git commit",
    "is it safe to give dogs blueberries",
    "what's the weather like in houston in april",
    "how long does cooked rice last in the fridge",
    "what's a good anniversary gift for a 5 year relationship",
    "best podcast recommendations for running",
    "difference between async and await in javascript",
    "how many calories in an avocado",
    "what's a good way to organize digital photos",
    "how do i export a csv from google sheets",
    "what's the capital of portugal",
    "best way to clean a cast iron pan",
    "how long to boil eggs for soft yolk",
    "what's the current price of bitcoin roughly",
    "how do i add a cron job on mac",
    "best stretches for runners knee",
    "whats a good book about product management",
    "how to make cold brew coffee at home",
    "what's the difference between margin and padding in css",
    "can dogs eat peanut butter",
    "how to remove a git branch",
    "what's a normal resting heart rate",
    "best way to meal prep for the week",
    "how do i center a div lol",
    "what's the word for when you remember something wrong",
]

CATEGORY_B_PROMPTS = [
    "help me write a PRD for an analytics dashboard feature",
    "can you review these OKRs for Q2",
    "draft a stakeholder update email about our sprint progress",
    "how should i structure a product requirements doc",
    "help me prepare talking points for my 1:1 with my manager",
    "what's the best way to say 'this timeline is unrealistic' diplomatically",
    "explain kubernetes to me like i'm a PM",
    "help me write a post-mortem for that outage last week",
    "write a SQL query to find users who signed up but never activated",
    "help me create a competitive analysis template",
    "draft an interview question list for a senior engineer position",
    "help me write a project brief for migrating our auth system",
    "how do i read a system architecture diagram",
    "help me write user stories for a notification preferences feature",
    "can you help me estimate story points for these tickets",
    "draft a quarterly business review presentation outline",
    "help me write a feature deprecation notice",
    "what metrics should i track for user onboarding",
    "help me write a tech spec review checklist",
    "draft a team retrospective agenda",
]

CATEGORY_C_PROMPTS = [
    "the dietary filter is returning vegetarian recipes when i search for vegan",
    "how do i add pagination to the recipes endpoint",
    "can you add a 'last modified' timestamp to recipes",
    "the CSS is broken on mobile, the cards are overlapping",
    "how do i write a test for an async function in jest",
    "i'm getting CORS errors when i call the API from the frontend",
    "can you explain this error: SQLITE_CONSTRAINT: UNIQUE constraint failed",
    "i want to add a 'favorites' feature, what tables do i need",
    "the seed data has a typo — 'teaspon' instead of 'teaspoon'",
    "how do i add rate limiting to the API",
    "the search is too slow when there's a lot of recipes",
    "can you add input validation for the recipe creation endpoint",
    "how do i handle file uploads for recipe images",
    "the grocery list is showing duplicate ingredients",
    "how do i deploy this to a free hosting service",
    "the tests are failing after i added the new migration",
    "can you add a health check endpoint",
    "how do i log API requests without it being noisy",
    "the prep_time field allows negative numbers",
    "can you refactor the database queries into a separate module",
]

CATEGORY_D_TOPICS = [
    "ugh i35 traffic was insane today",
    "what should D and i watch tonight. we just finished succession",
    "biscuit learned a new trick today, well sort of",
    "D made this amazing mushroom risotto last night",
    "thinking about getting a second dog. bad idea?",
    "austin allergies are killing me right now",
    "rach sent me a video of lily trying to ride ethan's bike. died laughing",
    "can't sleep. recommend me something to read",
    "just had the worst meeting of my career",
    "D and i are debating whether to go to big bend this weekend",
    "mom sent me a recipe she found. it's terrible but she's trying",
    "biscuit got skunked on our walk. the house smells awful",
    "thinking about trying that new vietnamese place on south lamar",
    "i miss seattle sometimes. rach says it's been raining for 3 weeks straight",
    "D's mom keeps asking when we're getting married. how do i deflect",
]

CATEGORY_E_CALLBACKS = [
    {"trigger": "D made that cauliflower thing again, it was even better this time", "refs": [1]},
    {"trigger": "mom called — her doctor is happy with her numbers", "refs": [4, 19]},
    {"trigger": "priya got an interview at datadog!", "refs": [6]},
    {"trigger": "we looked at another house, still nothing as good as the east austin one", "refs": [17]},
    {"trigger": "i foam rolled after my run today like a responsible adult", "refs": [11]},
    {"trigger": "ethan sent me a video of a volcano. geology phase is going strong", "refs": [15]},
    {"trigger": "D asked about the recipe app again. he wants to add his own recipes", "refs": [18]},
    {"trigger": "that thai place on south congress closed! devastated", "refs": [7]},
    {"trigger": "thinking about signing up for another race. maybe a full marathon?", "refs": [2, 19]},
    {"trigger": "linda tried another recipe. she's getting good at the low sodium ones", "refs": [4, 5, 19]},
]

# ---------------------------------------------------------------------------
# Generation prompts
# ---------------------------------------------------------------------------

FILLER_SYSTEM_PROMPT = """\
You are generating a realistic conversation between Maya Chen and her AI assistant for a benchmark dataset. Maya is a 34-year-old PM in Austin, TX.

CRITICAL RULES:
1. Maya speaks in short, casual messages (10-50 words max per message)
2. Maya uses lowercase, "lol", "tbh", "ngl", "D" for David, "Rach" for Rachel
3. The assistant is helpful and matches Maya's casual tone
4. Keep responses natural — not overly enthusiastic or formal
5. Output the conversation in this EXACT format:

MAYA: [her message]
ASSISTANT: [response]
MAYA: [follow-up]
ASSISTANT: [response]

Do NOT include turn numbers, tags, tokens, or analysis. Just the raw conversation.
"""

FILLER_SYSTEM_PROMPT_HEAVY = """\
You are generating a realistic conversation between Maya Chen and her AI assistant for a benchmark dataset. Maya is a 34-year-old PM in Austin, TX.

CRITICAL RULES:
1. Maya speaks in short, casual messages (10-50 words max per message)
2. Maya uses lowercase, "lol", "tbh", "ngl", "D" for David, "Rach" for Rachel
3. The assistant is THOROUGH — each response should be 200-400 words
4. Include full content: complete code blocks, detailed explanations, full document drafts, step-by-step instructions
5. Keep responses natural but substantive — Maya is asking for real work output
6. Output the conversation in this EXACT format:

MAYA: [her message]
ASSISTANT: [response with full detail — include code blocks, document drafts, complete explanations]
MAYA: [follow-up]
ASSISTANT: [another detailed response]

Do NOT include turn numbers, tags, tokens, or analysis. Just the raw conversation.
"""

FILLER_PROMPT_TEMPLATE = Template("""\
Generate a ${category} conversation between Maya and her AI assistant.

Context about Maya right now: ${context}
Date: ${date}
Topic: ${topic}

${extra_instructions}

Generate exactly ${turn_count} turns (a turn = one Maya message + one assistant response).
Keep Maya's messages SHORT (${maya_length}). The assistant can be slightly longer but still concise.
""")


# ---------------------------------------------------------------------------
# Gap analysis
# ---------------------------------------------------------------------------

def compute_gaps() -> List[dict]:
    """Compute the gaps between arc sessions for filler placement."""
    dates = sorted(ARC_SESSION_DATES.items(), key=lambda x: x[1])
    gaps = []

    for i in range(len(dates) - 1):
        s1_num, s1_date = dates[i]
        s2_num, s2_date = dates[i + 1]

        d1 = datetime.strptime(s1_date, "%Y-%m-%d")
        d2 = datetime.strptime(s2_date, "%Y-%m-%d")
        gap_days = (d2 - d1).days

        if gap_days <= 0:
            continue

        # Determine intensity phase
        week_num = (d1 - datetime(2026, 3, 1)).days // 7 + 1
        if week_num <= 2:
            intensity = "light"
            sessions_per_week = random.randint(3, 5)
        elif week_num <= 5:
            intensity = "medium"
            sessions_per_week = random.randint(5, 8)
        elif week_num <= 8:
            intensity = "heavy"
            sessions_per_week = random.randint(8, 12)
        else:
            intensity = "peak"
            sessions_per_week = random.randint(10, 14)

        # How many filler sessions in this gap
        filler_count = max(1, round(gap_days * sessions_per_week / 7))

        # Get context for this period
        context = ""
        for (a, b), ctx in PHASE_CONTEXT.items():
            if a <= s1_num < b:
                context = ctx
                break

        gaps.append({
            "after_session": s1_num,
            "before_session": s2_num,
            "start_date": s1_date,
            "end_date": s2_date,
            "gap_days": gap_days,
            "intensity": intensity,
            "filler_count": filler_count,
            "context": context,
        })

    return gaps


def plan_filler_sessions(target_count: int = 90) -> List[dict]:
    """Plan filler sessions distributed across gaps.

    Returns list of session specs with: session_id, date, category, topic, turn_count.
    """
    gaps = compute_gaps()
    total_gap_days = sum(g["gap_days"] for g in gaps)

    sessions = []
    session_counter = 0

    # Category distribution targets
    cat_targets = {
        "A": 0.35,  # Quick questions
        "B": 0.22,  # Work sessions
        "C": 0.18,  # Recipe app micro
        "D": 0.15,  # Personal chat
        "E": 0.10,  # Cross-ref reinforcers
    }

    # Pre-shuffle topic pools
    a_topics = list(CATEGORY_A_TOPICS)
    random.shuffle(a_topics)
    b_prompts = list(CATEGORY_B_PROMPTS)
    random.shuffle(b_prompts)
    c_prompts = list(CATEGORY_C_PROMPTS)
    random.shuffle(c_prompts)
    d_topics = list(CATEGORY_D_TOPICS)
    random.shuffle(d_topics)
    e_callbacks = list(CATEGORY_E_CALLBACKS)
    random.shuffle(e_callbacks)

    a_idx = b_idx = c_idx = d_idx = e_idx = 0

    for gap in gaps:
        n = gap["filler_count"]
        # Scale to target
        n = max(1, round(n * target_count / max(sum(g["filler_count"] for g in gaps), 1)))

        start = datetime.strptime(gap["start_date"], "%Y-%m-%d") + timedelta(hours=8)
        end = datetime.strptime(gap["end_date"], "%Y-%m-%d")

        for i in range(n):
            # Pick date within gap
            frac = (i + 1) / (n + 1)
            date = start + (end - start) * frac
            # Add random hour offset
            date = date.replace(
                hour=random.choice([8, 9, 10, 11, 14, 15, 16, 20, 21, 22]),
                minute=random.randint(0, 59),
            )

            # Pick category based on distribution
            r = random.random()
            cumulative = 0
            category = "A"
            for cat, weight in cat_targets.items():
                cumulative += weight
                if r < cumulative:
                    category = cat
                    break

            # Pick topic
            if category == "A":
                topic = a_topics[a_idx % len(a_topics)]
                a_idx += 1
                turn_count = random.randint(2, 4)
                maya_length = "10-30 words"
                heavy = False
            elif category == "B":
                topic = b_prompts[b_idx % len(b_prompts)]
                b_idx += 1
                turn_count = random.randint(15, 20)
                maya_length = "15-50 words"
                heavy = True
            elif category == "C":
                topic = c_prompts[c_idx % len(c_prompts)]
                c_idx += 1
                turn_count = random.randint(12, 15)
                maya_length = "15-40 words, include code snippets in assistant responses"
                heavy = True
            elif category == "D":
                topic = d_topics[d_idx % len(d_topics)]
                d_idx += 1
                turn_count = random.randint(4, 8)
                maya_length = "10-40 words"
                heavy = False
            else:  # E
                cb = e_callbacks[e_idx % len(e_callbacks)]
                e_idx += 1
                topic = cb["trigger"]
                turn_count = random.randint(3, 6)
                maya_length = "15-40 words"
                heavy = False

            session_counter += 1
            sessions.append({
                "filler_id": f"F{session_counter:03d}",
                "date": date.strftime("%Y-%m-%d %H:%M"),
                "category": category,
                "topic": topic,
                "turn_count": turn_count,
                "maya_length": maya_length,
                "heavy": heavy,
                "context": gap["context"],
                "after_session": gap["after_session"],
                "before_session": gap["before_session"],
            })

    return sessions[:target_count]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_filler_session(spec: dict) -> Tuple[str, float]:
    """Generate a single filler session transcript.

    Returns (transcript_text, duration).
    """
    is_heavy = spec.get("heavy", False)
    extra = ""
    if spec["category"] == "C":
        extra = (
            "Include realistic code snippets in assistant responses. "
            "Show actual Express/SQLite/Jest code for the recipe app. "
            "Include error messages and stack traces where relevant. "
        )
        if is_heavy:
            extra += (
                "Each assistant response MUST include a complete code block (20-60 lines). "
                "Show full file contents, complete test suites, detailed error analysis. "
                "Include import statements, error handling, and comments."
            )
    elif spec["category"] == "E":
        extra = (
            "This is a cross-reference reinforcer. Maya casually mentions "
            "the topic — she's not asking a question about it, just referencing "
            "it naturally in conversation. The assistant should respond naturally "
            "and may reference what it remembers."
        )
    elif spec["category"] == "B":
        extra = (
            "Include realistic work artifacts: PRD sections, email drafts, "
            "SQL queries, meeting notes, etc. Show the full document content. "
        )
        if is_heavy:
            extra += (
                "Each assistant response should be 200-400 words. Include complete "
                "document drafts, full email text, detailed analysis, numbered lists, "
                "and structured output. Don't summarize — give the full artifact."
            )

    prompt = FILLER_PROMPT_TEMPLATE.safe_substitute(
        category={"A": "quick question", "B": "work", "C": "recipe app coding",
                   "D": "personal chat", "E": "casual cross-reference"}[spec["category"]],
        context=spec["context"],
        date=spec["date"],
        topic=spec["topic"],
        extra_instructions=extra,
        turn_count=spec["turn_count"],
        maya_length=spec["maya_length"],
    )

    system_prompt = FILLER_SYSTEM_PROMPT_HEAVY if is_heavy else FILLER_SYSTEM_PROMPT

    response, duration = call_claude(
        prompt=prompt,
        model="sonnet",
        system_prompt=system_prompt,
        timeout=300 if is_heavy else 120,
    )

    return response or "", duration


def format_filler_as_review(spec: dict, transcript: str) -> str:
    """Format a filler session in the review file format.

    Simplified format — no brief section, no leakage check.
    """
    category_names = {
        "A": "Quick Question",
        "B": "Work Session",
        "C": "Recipe App Micro",
        "D": "Personal Chat",
        "E": "Cross-Reference Reinforcer",
    }

    lines = []
    lines.append("=" * 80)
    lines.append(f"FILLER SESSION {spec['filler_id']} — AgentLife Densification")
    lines.append(f"Category: {spec['category']} ({category_names[spec['category']]})")
    lines.append(f"Between Arc Sessions {spec['after_session']} and {spec['before_session']}")
    lines.append("=" * 80)
    lines.append("")
    lines.append("SECTION 1: SESSION BRIEF")
    lines.append("=" * 80)
    lines.append(f"Session: {spec['filler_id']} (Filler — {category_names[spec['category']]})")
    lines.append(f"Date: {spec['date']}")
    lines.append(f"Topic: {spec['topic']}")
    lines.append(f"Turn Count: {spec['turn_count']}")
    lines.append("")
    lines.append("=" * 80)
    lines.append("SECTION 2: GENERATED TRANSCRIPT (Readable Conversation)")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"SESSION METADATA:")
    lines.append(f"  Command: /new")
    lines.append(f"  Session: {spec['filler_id']}")
    lines.append(f"  Track: filler")
    lines.append(f"  Timestamp: {spec['date']}:00 UTC")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Parse and format the transcript
    turn_num = 0
    for line in transcript.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.upper().startswith("MAYA:"):
            turn_num += 1
            msg = line[5:].strip()
            tokens_est = len(msg.split())
            lines.append(f"TURN {turn_num} (Filler)")
            lines.append(f"TAG: normal | TOKENS: ~{tokens_est}")
            lines.append("")
            lines.append(f"MAYA:")
            lines.append(f"  {msg}")
            lines.append("")
        elif line.upper().startswith("ASSISTANT:"):
            msg = line[10:].strip()
            lines.append(f"AI ASSISTANT:")
            lines.append(f"  {msg}")
            lines.append("")
            lines.append("---")
            lines.append("")

    # Section 4: Eval queries (only for Category E)
    if spec["category"] == "E":
        lines.append("=" * 80)
        lines.append("SECTION 4: EVALUATION QUERIES & GROUND TRUTH")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f'QUERY 1: "What did Maya mention about {spec["topic"][:40]}?"')
        lines.append(f'  Ground Truth: {spec["topic"]}')
        lines.append(f'  Evidence Session: {spec["filler_id"]}')
        lines.append(f'  Query Type: cross_reference')
        lines.append(f'  Recall Difficulty: Medium (filler session callback)')
        lines.append("")

    lines.append("=" * 80)
    lines.append("END OF FILLER SESSION")
    lines.append("=" * 80)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

def generate_batch(
    specs: List[dict],
    output_dir: Path,
    batch_size: int = 5,
) -> dict:
    """Generate filler sessions in batches.

    Returns stats dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    stats = {
        "generated": 0,
        "cached": 0,
        "failed": 0,
        "total_tokens_est": 0,
        "elapsed_s": 0.0,
    }

    t0 = time.monotonic()

    for i, spec in enumerate(specs):
        fid = spec["filler_id"]
        cache_path = cache_dir / f"{fid}.json"
        output_path = output_dir / f"{fid}-review.txt"

        # Check cache
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text())
                transcript = cached["transcript"]
                stats["cached"] += 1
                stats["total_tokens_est"] += cached.get("tokens_est", 0)

                # Write review file from cache
                review_text = format_filler_as_review(spec, transcript)
                output_path.write_text(review_text)
                continue
            except (json.JSONDecodeError, KeyError):
                pass

        # Generate
        print(f"  [{i+1}/{len(specs)}] Generating {fid} ({spec['category']}: {spec['topic'][:50]})")
        transcript, duration = generate_filler_session(spec)

        if not transcript:
            print(f"    FAILED — empty response")
            stats["failed"] += 1
            continue

        tokens_est = len(transcript.split())

        # Cache
        cache_path.write_text(json.dumps({
            "filler_id": fid,
            "transcript": transcript,
            "tokens_est": tokens_est,
            "spec": spec,
        }, indent=2))

        # Write review file
        review_text = format_filler_as_review(spec, transcript)
        output_path.write_text(review_text)

        stats["generated"] += 1
        stats["total_tokens_est"] += tokens_est

    stats["elapsed_s"] = round(time.monotonic() - t0, 1)
    return stats


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def count_all_tokens(
    assets_dir: Path,
    filler_dir: Optional[Path] = None,
) -> dict:
    """Count tokens across arc sessions and filler sessions."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
    except ImportError:
        enc = None

    def count(text):
        if enc:
            return len(enc.encode(text))
        return len(text.split())  # Rough fallback

    results = {"arc": {}, "filler": {}, "arc_total": 0, "filler_total": 0, "grand_total": 0}

    # Arc sessions
    for fp in sorted(assets_dir.glob("session-*-review-*.txt")):
        text = fp.read_text()
        tokens = count(text)
        results["arc"][fp.name] = tokens
        results["arc_total"] += tokens

    # Filler sessions
    if filler_dir and filler_dir.exists():
        for fp in sorted(filler_dir.glob("F*-review.txt")):
            text = fp.read_text()
            tokens = count(text)
            results["filler"][fp.name] = tokens
            results["filler_total"] += tokens

    results["grand_total"] = results["arc_total"] + results["filler_total"]
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def shrink_to_target(filler_dir: Path, assets_dir: Path, target_tokens: int):
    """Remove filler sessions (newest first) until total is at or below target."""
    # Get current count
    tokens = count_all_tokens(assets_dir, filler_dir)
    current = tokens["grand_total"]

    if current <= target_tokens:
        print(f"Already at {current:,} tokens (target: {target_tokens:,})")
        return

    # Sort fillers by ID descending (remove newest first)
    filler_files = sorted(filler_dir.glob("F*-review.txt"), reverse=True)
    cache_dir = filler_dir / "cache"
    removed = 0

    for fp in filler_files:
        if current <= target_tokens:
            break

        file_tokens = count_all_tokens(assets_dir)  # Re-count arc only
        filler_tokens = _count_file_tokens(fp)
        current -= filler_tokens

        # Remove review + cache
        fid = fp.stem.replace("-review", "")
        fp.unlink()
        cache_path = cache_dir / f"{fid}.json"
        if cache_path.exists():
            cache_path.unlink()
        removed += 1
        print(f"  Removed {fid} ({filler_tokens:,} tokens)")

    tokens = count_all_tokens(assets_dir, filler_dir)
    print(f"\nRemoved {removed} fillers. New total: {tokens['grand_total']:,} tokens")


def _count_file_tokens(fp: Path) -> int:
    """Count tokens in a single file."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(fp.read_text()))
    except ImportError:
        return len(fp.read_text().split())


def generate_to_target(
    target_tokens: int,
    output_dir: Path,
    assets_dir: Path,
    base_count: int = 90,
):
    """Generate filler sessions until total tokens reach target.

    Generates in batches, checking token count after each batch.
    """
    # Check current state
    tokens = count_all_tokens(assets_dir, output_dir)
    current = tokens["grand_total"]
    print(f"Current total: {current:,} tokens (target: {target_tokens:,})")

    if current >= target_tokens:
        print(f"Already at target. No generation needed.")
        return

    # Count existing fillers to set start offset
    existing_fillers = list(output_dir.glob("F*-review.txt"))
    start_offset = len(existing_fillers)

    # Plan enough sessions to have some beyond existing count
    plan_count = max(base_count, start_offset + 100)
    sessions = plan_filler_sessions(plan_count)

    # Skip already-generated
    sessions = sessions[start_offset:]

    batch_size = 10
    total_generated = 0

    for batch_start in range(0, len(sessions), batch_size):
        batch = sessions[batch_start:batch_start + batch_size]

        print(f"\n--- Batch {batch_start // batch_size + 1} ({len(batch)} sessions) ---")
        stats = generate_batch(batch, output_dir)
        total_generated += stats["generated"]

        # Re-count
        tokens = count_all_tokens(assets_dir, output_dir)
        current = tokens["grand_total"]
        print(f"  Current total: {current:,} / {target_tokens:,} tokens")

        if current >= target_tokens:
            print(f"\nTarget reached! {current:,} tokens >= {target_tokens:,}")
            break

    print(f"\nGenerated {total_generated} new fillers")
    tokens = count_all_tokens(assets_dir, output_dir)
    print(f"Final total: {tokens['grand_total']:,} tokens")


def main():
    parser = argparse.ArgumentParser(description="AgentLife Densification")
    parser.add_argument("--output", type=str,
                        default=str(_DIR.parent / "data" / "filler-sessions"),
                        help="Output directory for filler sessions")
    parser.add_argument("--target-count", type=int, default=90,
                        help="Target number of filler sessions")
    parser.add_argument("--target-tokens", type=int, default=None,
                        help="Generate fillers until total tokens reach this target")
    parser.add_argument("--shrink-to", type=int, default=None,
                        help="Remove fillers (newest first) until total is at target")
    parser.add_argument("--plan-only", action="store_true",
                        help="Just print the plan, don't generate")
    parser.add_argument("--count-tokens", action="store_true",
                        help="Count tokens in all sessions")
    parser.add_argument("--assets-dir", type=str,
                        default=str(_DIR.parent.parent.parent / "assets"),
                        help="Directory with arc session review files")
    args = parser.parse_args()

    output_dir = Path(args.output)
    assets_dir = Path(args.assets_dir)

    if args.count_tokens:
        results = count_all_tokens(assets_dir, output_dir)
        print(f"Arc sessions: {results['arc_total']:,} tokens ({len(results['arc'])} files)")
        print(f"Filler sessions: {results['filler_total']:,} tokens ({len(results['filler'])} files)")
        print(f"Grand total: {results['grand_total']:,} tokens")
        return

    if args.shrink_to is not None:
        shrink_to_target(output_dir, assets_dir, args.shrink_to)
        return

    if args.target_tokens is not None:
        generate_to_target(args.target_tokens, output_dir, assets_dir, args.target_count)
        return

    # Plan filler sessions
    print("Planning filler sessions...")
    sessions = plan_filler_sessions(args.target_count)

    # Category breakdown
    cats = {}
    for s in sessions:
        cats[s["category"]] = cats.get(s["category"], 0) + 1
    print(f"\nPlanned {len(sessions)} filler sessions:")
    for cat, count in sorted(cats.items()):
        print(f"  Category {cat}: {count}")

    if args.plan_only:
        plan_path = output_dir / "plan.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(plan_path, "w") as f:
            json.dump(sessions, f, indent=2)
        print(f"\nPlan saved to {plan_path}")
        return

    # Generate
    print(f"\nGenerating filler sessions to {output_dir}...")
    stats = generate_batch(sessions, output_dir)

    print(f"\nDensification complete:")
    print(f"  Generated: {stats['generated']}")
    print(f"  Cached: {stats['cached']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Estimated tokens: {stats['total_tokens_est']:,}")
    print(f"  Elapsed: {stats['elapsed_s']:.1f}s")

    # Count final tokens
    print("\nToken count:")
    tokens = count_all_tokens(assets_dir, output_dir)
    print(f"  Arc: {tokens['arc_total']:,} tokens")
    print(f"  Filler: {tokens['filler_total']:,} tokens")
    print(f"  Total: {tokens['grand_total']:,} tokens")

    if tokens["grand_total"] < 300000:
        deficit = 300000 - tokens["grand_total"]
        print(f"\n  WARNING: {deficit:,} tokens short of 300K target")
        print(f"  Consider: python3 densify.py --target-tokens 300000")


if __name__ == "__main__":
    main()
