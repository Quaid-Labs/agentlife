#!/usr/bin/env python3
"""AgentLife Benchmark — Transcript Generator

Generates realistic Maya↔Melina conversations using Claude Code (claude -p).
Each session follows a YAML brief specifying topics, facts to embed, and
message length constraints.

Backend: Claude Code Max plan (claude -p --model sonnet)
- Generation is free on Max plan (Sonnet quota)
- Falls back to API billing when quota exhausted
- No API key management needed

Usage:
    # Generate from pilot briefs
    python3 generate.py --briefs briefs/pilot/ --output data/transcripts/pilot/

    # Generate a specific session only
    python3 generate.py --briefs briefs/pilot/ --session 1

    # Validate generated transcripts
    python3 generate.py --validate data/transcripts/pilot/

    # Regenerate a session (overwrite existing)
    python3 generate.py --briefs briefs/pilot/ --session 3 --force

    # Use a custom persona file
    python3 generate.py --briefs briefs/pilot/ --persona persona/maya.md
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML required. Install with: pip3 install pyyaml")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Token limit tags — contextual message length control
# ---------------------------------------------------------------------------

TOKEN_LIMITS = {
    "quick": 50,       # Quick questions, follow-ups, reactions (~26 actual tokens)
    "normal": 100,     # Standard topic discussion (~68 actual tokens)
    "extended": 200,   # Venting, storytelling, big updates (~138 actual tokens)
}

# ---------------------------------------------------------------------------
# Personas
# ---------------------------------------------------------------------------

# Condensed Maya persona for generation prompts. The full persona doc
# (persona/maya.md, ~3K words) is loaded via --persona when available.
MAYA_PERSONA_DEFAULT = """\
You are Maya Chen, a 34-year-old product manager at TechFlow (a mid-size SaaS company) in Austin, TX. You're chatting with your AI assistant in a casual, natural way — like texting a knowledgeable friend.

## Key People
- Partner: David (software engineer, vegetarian)
- Dog: Biscuit (golden retriever, 3 years old)
- Mom: Linda (recently diagnosed with Type 2 diabetes, lives in Houston)
- Sister: Rachel (lives in Seattle, has two kids: Ethan age 7, Lily age 4)
- Coworker: Priya (mentioned in passing — NOT family, just a colleague)
- David's family: his mother (wants to visit Austin), his brother Mike

## Your Life Right Now
- Increasingly unhappy at TechFlow. Considering leaving.
- Training for a half marathon (Austin Half, in April — you'll correct to May later)
- Thinking about building a recipe app (motivated by mom's diabetes)
- Mom diagnosed with Type 2 diabetes recently — it's personal for you
- David is vegetarian, which matters for recipe/cooking discussions

## How You Talk
- Terse, direct. Short messages. Gets impatient with long explanations.
- Uses nicknames: "D" for David, "Rach" for Rachel
- Circles back to topics days later without context: "hey, about the thing with the auth..."
- Sometimes contradicts herself and corrects later
- Uses "tbh", "ngl", "lol" naturally but not excessively
- Goes on tangents about food, running, dog
- Strong aesthetic opinions in non-technical language
- References past conversations naturally
- Specific with details (names, places, times)
"""

MELINA_PERSONA = """\
You are an AI assistant having a natural conversation with Maya. You're helpful, warm, and knowledgeable. You remember context within this conversation and respond naturally.

Guidelines:
- Match Maya's casual tone — don't be overly formal or verbose
- Be genuinely helpful, not just agreeable
- Ask clarifying questions when needed
- Show personality — you're not a generic chatbot
- Keep responses proportional to the question (short question = short answer)
- When Maya mentions personal details, acknowledge them naturally
- For technical topics, be competent but concise
"""

# ---------------------------------------------------------------------------
# Claude Code backend
# ---------------------------------------------------------------------------

def call_claude(
    prompt: str,
    model: str = "sonnet",
    system_prompt: Optional[str] = None,
    timeout: int = 120,
    max_retries: int = 3,
) -> str:
    """Call Claude via `claude -p` (Claude Code non-interactive mode).

    Uses the Max plan subscription. Falls back to API billing when quota
    is exhausted. No API key needed.
    """
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)  # Remove nesting guard

    # Use full path to avoid PATH issues in subprocess/background contexts
    claude_bin = os.environ.get("CLAUDE_BIN", "/opt/homebrew/bin/claude")
    cmd = [claude_bin, "-p", "--model", model]
    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])

    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout,
            )
            output = result.stdout.strip()
            if output:
                return output
            # Empty output — retry
            if attempt < max_retries - 1:
                print(f"    [retry {attempt + 1}/{max_retries}: empty output]")
                time.sleep(2)
        except subprocess.TimeoutExpired:
            if attempt < max_retries - 1:
                print(f"    [retry {attempt + 1}/{max_retries}: timeout]")
                time.sleep(2)
            else:
                raise RuntimeError(f"claude -p timed out after {timeout}s ({max_retries} attempts)")
        except FileNotFoundError:
            raise RuntimeError("'claude' CLI not found. Is Claude Code installed?")

    raise RuntimeError(f"claude -p returned empty output after {max_retries} attempts")


def estimate_tokens(text: str) -> int:
    """Rough token estimate: words * 1.3."""
    return int(len(text.split()) * 1.3)


# ---------------------------------------------------------------------------
# Session brief loading
# ---------------------------------------------------------------------------

def load_brief(path: Path) -> dict:
    """Load a session brief from a YAML file."""
    with open(path) as f:
        brief = yaml.safe_load(f)

    # Validate required fields
    required = ["session", "track", "goal"]
    missing = [f for f in required if f not in brief]
    if missing:
        raise ValueError(f"Brief {path.name} missing required fields: {missing}")

    # Defaults
    brief.setdefault("turn_count", 10)
    brief.setdefault("cross_references", [])
    brief.setdefault("tangents", [])
    brief.setdefault("new_information", [])
    brief.setdefault("corrections", [])
    brief.setdefault("eval_queries", [])
    brief.setdefault("end_condition", f"{brief['turn_count']} turns reached")
    brief.setdefault("message_plan", None)
    brief.setdefault("do_not_reveal", [])

    return brief


def load_briefs(briefs_dir: Path) -> list[dict]:
    """Load all session briefs from a directory, sorted by session number."""
    briefs = []
    for f in sorted(briefs_dir.glob("session-*.yaml")):
        briefs.append(load_brief(f))
    if not briefs:
        print(f"ERROR: No session-*.yaml files found in {briefs_dir}")
        sys.exit(1)
    return briefs


# ---------------------------------------------------------------------------
# Message plan generation
# ---------------------------------------------------------------------------

def get_message_plan(brief: dict) -> list[str]:
    """Get the token tag sequence for Maya's messages.

    If the brief specifies a message_plan, use it directly.
    Otherwise, generate one based on default distribution.
    """
    if brief.get("message_plan"):
        return [m["tag"] if isinstance(m, dict) else m for m in brief["message_plan"]]

    turn_count = brief["turn_count"]
    rng = random.Random(brief["session"])  # Seeded for reproducibility

    plan = []
    for i in range(turn_count):
        if i == 0:
            plan.append("normal")      # Opening is always normal
        elif i == turn_count - 1:
            plan.append("quick")       # Closing is quick
        else:
            r = rng.random()
            if r < 0.10:
                plan.append("extended")
            elif r < 0.40:
                plan.append("quick")
            else:
                plan.append("normal")
    return plan


# ---------------------------------------------------------------------------
# Conversation formatting
# ---------------------------------------------------------------------------

def format_conversation(messages: list[dict]) -> str:
    """Format conversation history for inclusion in prompts."""
    if not messages:
        return "(No messages yet — this is the start of the conversation.)"

    lines = []
    for msg in messages:
        role = "Maya" if msg["type"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Message generation
# ---------------------------------------------------------------------------

def build_maya_prompt(
    brief: dict,
    conversation: list[dict],
    turn_index: int,
    turn_count: int,
    remaining_facts: list[str],
    continuity: dict,
    tag: str,
) -> str:
    """Build the user prompt for generating Maya's next message."""

    token_limit = TOKEN_LIMITS[tag]

    parts = []

    # Continuity context from previous sessions
    if continuity.get("summary"):
        parts.append(f"## Previous Sessions\n{continuity['summary']}")

    # Session brief context
    parts.append(f"## This Session (#{brief['session']})")
    parts.append(f"Goal: {brief['goal']}")

    if brief.get("cross_references"):
        refs = "\n".join(
            f"- Session {cr['session']}: {cr['topic']} — {cr['how']}"
            for cr in brief["cross_references"]
        )
        parts.append(f"Cross-references to make:\n{refs}")

    # Tangent instructions
    if brief.get("tangents"):
        for tangent in brief["tangents"]:
            parts.append(
                f"TANGENT: {tangent['trigger']} → go off on: {tangent['topic']}. "
                f"Then return with something like: \"{tangent.get('return', 'ok anyway')}\""
            )

    # Facts still to embed
    if remaining_facts:
        facts_str = "\n".join(f"- {f}" for f in remaining_facts)
        parts.append(
            f"Facts to naturally work into the conversation (don't force them — "
            f"only mention what fits naturally):\n{facts_str}"
        )

    # Forbidden facts — reserved for future sessions
    if brief.get("do_not_reveal"):
        forbidden_str = "\n".join(f"- {f}" for f in brief["do_not_reveal"])
        parts.append(
            f"IMPORTANT — Do NOT mention or hint at ANY of the following. "
            f"These facts are reserved for future sessions and must not "
            f"appear in this conversation:\n{forbidden_str}"
        )

    # Corrections
    if brief.get("corrections"):
        for corr in brief["corrections"]:
            parts.append(f"CORRECTION: {corr}")

    # Turn context
    parts.append(f"\nTurn {turn_index + 1} of {turn_count}.")
    if turn_index == 0:
        parts.append("This is the opening message. Start the conversation naturally.")
    elif turn_index == turn_count - 1:
        parts.append("This is the last turn. Wrap up naturally — say bye, ttyl, etc.")
    elif turn_index >= turn_count - 2:
        parts.append("The conversation is winding down.")

    # Conversation so far
    parts.append(f"\n## Conversation So Far\n{format_conversation(conversation)}")

    # Generation instruction
    parts.append(
        f"\n## Your Task\n"
        f"Generate Maya's next message. MUST be under {token_limit} tokens. "
        f"{'Keep it short — quick reaction or follow-up.' if tag == 'quick' else ''}"
        f"{'Can be a longer message — share a story, vent, or cover multiple topics.' if tag == 'extended' else ''}"
        f"\n\nOutput ONLY Maya's message text. No quotes, no 'Maya:', no meta-commentary."
    )

    return "\n\n".join(parts)


def build_melina_prompt(conversation: list[dict]) -> str:
    """Build the user prompt for generating Melina's response."""
    return (
        f"## Conversation\n{format_conversation(conversation)}\n\n"
        f"## Your Task\n"
        f"Respond to Maya naturally as her AI assistant. Match her tone — "
        f"be casual, helpful, and concise. Don't over-explain.\n\n"
        f"Output ONLY your response text. No quotes, no 'Melina:', no meta-commentary."
    )


def generate_maya_message(
    persona: str,
    brief: dict,
    conversation: list[dict],
    turn_index: int,
    turn_count: int,
    remaining_facts: list[str],
    continuity: dict,
    tag: str,
) -> str:
    """Generate Maya's next message."""
    prompt = build_maya_prompt(
        brief, conversation, turn_index, turn_count,
        remaining_facts, continuity, tag,
    )
    return call_claude(prompt, model="sonnet", system_prompt=persona)


def generate_melina_response(conversation: list[dict]) -> str:
    """Generate Melina's response to Maya."""
    prompt = build_melina_prompt(conversation)
    return call_claude(prompt, model="sonnet", system_prompt=MELINA_PERSONA)


# ---------------------------------------------------------------------------
# Fact tracking
# ---------------------------------------------------------------------------

def check_facts_mentioned(message: str, facts: list[str]) -> list[str]:
    """Check which facts are roughly covered by a message.

    Simple keyword overlap check — not perfect but good enough for
    tracking which facts have been touched on.
    """
    mentioned = []
    msg_lower = message.lower()
    for fact in facts:
        # Extract key words from the fact (skip common words)
        keywords = [
            w for w in fact.lower().split()
            if len(w) > 3 and w not in {
                "maya", "that", "this", "with", "from", "have", "been",
                "about", "their", "they", "will", "when", "what", "where",
                "does", "like", "just", "also", "some", "more", "very",
            }
        ]
        # If 2+ keywords appear in the message, consider it mentioned
        hits = sum(1 for kw in keywords if kw in msg_lower)
        if hits >= min(2, len(keywords)):
            mentioned.append(fact)
    return mentioned


# ---------------------------------------------------------------------------
# Session generation
# ---------------------------------------------------------------------------

def generate_session(
    brief: dict,
    persona: str,
    continuity: dict,
    output_dir: Path,
    force: bool = False,
) -> dict:
    """Generate a full session transcript from a brief.

    Returns the session result dict (for continuity tracking).
    """
    session_num = brief["session"]
    output_file = output_dir / f"session-{session_num:02d}.jsonl"

    if output_file.exists() and not force:
        print(f"  Session {session_num}: already exists (use --force to overwrite)")
        # Load existing for continuity
        messages = []
        with open(output_file) as f:
            for line in f:
                entry = json.loads(line)
                if entry["type"] in ("user", "assistant"):
                    messages.append(entry)
        return {"session": session_num, "messages": messages, "skipped": True}

    print(f"  Session {session_num}: generating ({brief['turn_count']} turns, "
          f"track {brief['track']})")

    # Simulated timestamp for this session
    # Sessions are spaced ~2-3 days apart starting from a base date
    base_date = datetime(2026, 3, 1, 9, 0, 0)  # March 1, 2026
    session_start = base_date + timedelta(days=(session_num - 1) * 2.5)

    message_plan = get_message_plan(brief)
    turn_count = len(message_plan)
    remaining_facts = list(brief.get("new_information", []))
    conversation = []  # List of {type, content} dicts
    transcript = []    # Full JSONL entries with timestamps

    # Opening /new command
    transcript.append({
        "type": "system",
        "command": "/new",
        "session": session_num,
        "track": brief["track"],
        "timestamp": session_start.isoformat() + "Z",
    })

    msg_time = session_start + timedelta(seconds=5)

    for turn_idx in range(turn_count):
        tag = message_plan[turn_idx]
        token_limit = TOKEN_LIMITS[tag]

        # --- Maya's message ---
        start = time.time()
        maya_msg = generate_maya_message(
            persona, brief, conversation, turn_idx, turn_count,
            remaining_facts, continuity, tag,
        )
        maya_time = time.time() - start
        maya_tokens = estimate_tokens(maya_msg)

        # Check token compliance
        over = maya_tokens > token_limit
        if over and maya_tokens > token_limit * 1.3:
            # Way over — retry with stronger constraint
            print(f"    Turn {turn_idx + 1}: Maya msg too long ({maya_tokens} > {token_limit}), retrying...")
            maya_msg = generate_maya_message(
                persona, brief, conversation, turn_idx, turn_count,
                remaining_facts, continuity, tag,
            )
            maya_tokens = estimate_tokens(maya_msg)

        # Track mentioned facts
        mentioned = check_facts_mentioned(maya_msg, remaining_facts)
        for fact in mentioned:
            remaining_facts.remove(fact)

        conversation.append({"type": "user", "content": maya_msg})
        transcript.append({
            "type": "user",
            "content": maya_msg,
            "timestamp": msg_time.isoformat() + "Z",
            "token_count": maya_tokens,
            "token_tag": tag,
            "token_limit": token_limit,
            "generation_time": round(maya_time, 1),
        })

        tag_label = f"[{tag}:{maya_tokens}t]"
        print(f"    Turn {turn_idx + 1}/{turn_count} {tag_label} Maya: {maya_msg[:80]}{'...' if len(maya_msg) > 80 else ''}")

        msg_time += timedelta(seconds=random.randint(3, 15))

        # --- Melina's response ---
        start = time.time()
        melina_msg = generate_melina_response(conversation)
        melina_time = time.time() - start
        melina_tokens = estimate_tokens(melina_msg)

        conversation.append({"type": "assistant", "content": melina_msg})
        transcript.append({
            "type": "assistant",
            "content": melina_msg,
            "timestamp": msg_time.isoformat() + "Z",
            "token_count": melina_tokens,
            "generation_time": round(melina_time, 1),
        })

        print(f"           {'':>12} Melina: {melina_msg[:80]}{'...' if len(melina_msg) > 80 else ''}")

        msg_time += timedelta(seconds=random.randint(5, 30))

    # Write transcript (pure conversation — no eval queries)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for entry in transcript:
            f.write(json.dumps(entry) + "\n")

    # Write eval queries to separate file
    if brief.get("eval_queries"):
        eval_dir = output_dir.parent / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_file = eval_dir / f"session-{session_num:02d}.json"
        eval_data = []
        for eq in brief["eval_queries"]:
            eval_data.append({
                "session": session_num,
                "query": eq["query"],
                "ground_truth": eq["ground_truth"],
                "query_type": eq["query_type"],
                "evidence_sessions": eq.get("evidence_sessions", [session_num]),
            })
        with open(eval_file, "w") as f:
            json.dump(eval_data, f, indent=2)
        print(f"    Eval queries: {len(eval_data)} written to {eval_file}")

    # Stats
    user_msgs = [m for m in conversation if m["type"] == "user"]
    asst_msgs = [m for m in conversation if m["type"] == "assistant"]
    total_user_tokens = sum(estimate_tokens(m["content"]) for m in user_msgs)
    total_asst_tokens = sum(estimate_tokens(m["content"]) for m in asst_msgs)
    facts_embedded = len(brief.get("new_information", [])) - len(remaining_facts)
    facts_total = len(brief.get("new_information", []))

    print(f"    Done: {len(user_msgs)} turns, "
          f"Maya ~{total_user_tokens}t, Melina ~{total_asst_tokens}t, "
          f"facts {facts_embedded}/{facts_total}, "
          f"eval queries {len(brief.get('eval_queries', []))}")

    if remaining_facts:
        print(f"    WARNING: {len(remaining_facts)} facts not embedded:")
        for f in remaining_facts:
            print(f"      - {f}")

    return {
        "session": session_num,
        "messages": conversation,
        "skipped": False,
        "facts_embedded": facts_embedded,
        "facts_remaining": remaining_facts,
    }


# ---------------------------------------------------------------------------
# Continuity tracking
# ---------------------------------------------------------------------------

def build_initial_continuity() -> dict:
    """Build empty continuity context for first session."""
    return {
        "summary": "",
        "known_people": [],
        "established_facts": [],
        "active_projects": [],
        "session_summaries": [],
    }


def update_continuity(continuity: dict, brief: dict, session_result: dict) -> dict:
    """Update continuity context after a session.

    Generates a brief summary of the session for use in subsequent
    session generation prompts.
    """
    if session_result.get("skipped"):
        # Load existing summary if available
        return continuity

    messages = session_result["messages"]
    if not messages:
        return continuity

    # Generate a brief session summary using Claude
    conversation_text = format_conversation(messages)
    summary_prompt = (
        f"Summarize this conversation between Maya and her AI assistant in 2-3 sentences. "
        f"Focus on: key facts mentioned, topics discussed, any decisions made, "
        f"emotional tone. Be concise.\n\n{conversation_text}"
    )

    try:
        summary = call_claude(summary_prompt, model="haiku")
    except Exception as e:
        print(f"    Warning: couldn't generate session summary: {e}")
        summary = f"Session {brief['session']}: {brief['goal']}"

    continuity["session_summaries"].append({
        "session": brief["session"],
        "track": brief["track"],
        "summary": summary,
    })

    # Build rolling summary (last 5 sessions for context window management)
    recent = continuity["session_summaries"][-5:]
    continuity["summary"] = "\n".join(
        f"Session {s['session']} (Track {s['track']}): {s['summary']}"
        for s in recent
    )

    # Track people mentioned
    for fact in brief.get("new_information", []):
        continuity["established_facts"].append(fact)

    return continuity


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def run_generation(
    briefs_dir: Path,
    output_dir: Path,
    persona_file: Optional[Path] = None,
    session_filter: Optional[int] = None,
    force: bool = False,
):
    """Run transcript generation for all session briefs."""

    # Load persona
    if persona_file and persona_file.exists():
        persona = persona_file.read_text()
        print(f"Loaded persona from {persona_file} ({len(persona)} chars)")
    else:
        persona = MAYA_PERSONA_DEFAULT
        print("Using default Maya persona (condensed)")

    # Load briefs
    briefs = load_briefs(briefs_dir)
    if session_filter is not None:
        briefs = [b for b in briefs if b["session"] == session_filter]
        if not briefs:
            print(f"ERROR: No brief found for session {session_filter}")
            sys.exit(1)

    print(f"\nGenerating {len(briefs)} session(s)")
    print(f"  Briefs: {briefs_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Model:  sonnet (via claude -p)")
    print()

    # Verify claude -p works
    try:
        test = call_claude("Say 'ready' in one word.", model="sonnet", timeout=30)
        if not test:
            raise RuntimeError("Empty response")
        print(f"Backend check: OK ({test.strip()[:20]})")
    except Exception as e:
        print(f"ERROR: claude -p backend check failed: {e}")
        print("Make sure Claude Code is installed and you're logged in.")
        sys.exit(1)

    print()

    continuity = build_initial_continuity()

    # If we're starting mid-sequence, load continuity from existing transcripts
    if session_filter and session_filter > 1:
        continuity_file = output_dir / "continuity.json"
        if continuity_file.exists():
            with open(continuity_file) as f:
                continuity = json.load(f)
            print(f"Loaded continuity state from {continuity_file}")

    results = []
    total_start = time.time()

    for brief in briefs:
        session_start = time.time()
        try:
            result = generate_session(brief, persona, continuity, output_dir, force)
            results.append(result)

            # Update continuity for next session
            continuity = update_continuity(continuity, brief, result)

            elapsed = time.time() - session_start
            print(f"    Time: {elapsed:.1f}s\n")

        except Exception as e:
            print(f"    ERROR: Session {brief['session']} failed: {e}")
            traceback.print_exc()
            print()

    # Save continuity state
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "continuity.json", "w") as f:
        json.dump(continuity, f, indent=2)

    # Summary
    total_elapsed = time.time() - total_start
    generated = [r for r in results if not r.get("skipped")]
    print(f"\n{'='*60}")
    print(f"  Generation complete")
    print(f"  Sessions generated: {len(generated)}/{len(results)}")
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"  Transcripts: {output_dir}")
    print(f"  Continuity: {output_dir / 'continuity.json'}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_transcripts(output_dir: Path):
    """Validate generated transcripts for basic quality checks."""
    output_path = Path(output_dir)
    files = sorted(output_path.glob("session-*.jsonl"))

    if not files:
        print(f"ERROR: No session-*.jsonl files found in {output_dir}")
        sys.exit(1)

    print(f"Validating {len(files)} transcript(s) in {output_dir}\n")

    issues = []
    stats = {
        "sessions": 0,
        "total_user_msgs": 0,
        "total_asst_msgs": 0,
        "total_eval_queries": 0,
        "total_user_tokens": 0,
        "total_asst_tokens": 0,
        "token_compliance": {"pass": 0, "over": 0},
    }

    for f in files:
        session_issues = []
        entries = []
        with open(f) as fh:
            for line_num, line in enumerate(fh, 1):
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    session_issues.append(f"Line {line_num}: invalid JSON")

        if not entries:
            session_issues.append("Empty transcript")
            issues.extend([(f.name, i) for i in session_issues])
            continue

        stats["sessions"] += 1

        # Check structure
        if entries[0].get("type") != "system":
            session_issues.append("First entry should be type=system (/new)")

        user_msgs = [e for e in entries if e["type"] == "user"]
        asst_msgs = [e for e in entries if e["type"] == "assistant"]
        eval_qs = [e for e in entries if e["type"] == "eval"]

        stats["total_user_msgs"] += len(user_msgs)
        stats["total_asst_msgs"] += len(asst_msgs)
        stats["total_eval_queries"] += len(eval_qs)

        # Check alternating user/assistant
        conv_msgs = [e for e in entries if e["type"] in ("user", "assistant")]
        for i in range(1, len(conv_msgs)):
            if conv_msgs[i]["type"] == conv_msgs[i-1]["type"]:
                session_issues.append(
                    f"Non-alternating messages at position {i}: "
                    f"two consecutive {conv_msgs[i]['type']} messages"
                )

        # Check token compliance
        for msg in user_msgs:
            tc = msg.get("token_count", 0)
            limit = msg.get("token_limit", 100)
            stats["total_user_tokens"] += tc
            if tc <= limit:
                stats["token_compliance"]["pass"] += 1
            else:
                stats["token_compliance"]["over"] += 1
                if tc > limit * 1.3:
                    session_issues.append(
                        f"Token violation: {tc} tokens (limit {limit}): "
                        f"{msg['content'][:50]}..."
                    )

        for msg in asst_msgs:
            stats["total_asst_tokens"] += msg.get("token_count", 0)

        # Check eval queries have required fields
        for eq in eval_qs:
            for field in ["query", "ground_truth", "query_type"]:
                if field not in eq:
                    session_issues.append(f"Eval query missing '{field}'")

        # Report
        status = "PASS" if not session_issues else f"ISSUES ({len(session_issues)})"
        print(f"  {f.name}: {len(user_msgs)} turns, "
              f"{len(eval_qs)} eval queries — {status}")

        if session_issues:
            for issue in session_issues:
                print(f"    - {issue}")

        issues.extend([(f.name, i) for i in session_issues])

    # Summary
    print(f"\n{'='*60}")
    print(f"  Validation Summary")
    print(f"{'='*60}")
    print(f"  Sessions:      {stats['sessions']}")
    print(f"  User messages:  {stats['total_user_msgs']} (~{stats['total_user_tokens']} tokens)")
    print(f"  Asst messages:  {stats['total_asst_msgs']} (~{stats['total_asst_tokens']} tokens)")
    print(f"  Eval queries:   {stats['total_eval_queries']}")
    compliance = stats["token_compliance"]
    total = compliance["pass"] + compliance["over"]
    if total > 0:
        pct = compliance["pass"] / total * 100
        print(f"  Token compliance: {compliance['pass']}/{total} ({pct:.0f}%)")
    print(f"  Issues:         {len(issues)}")
    print(f"{'='*60}")

    if issues:
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AgentLife Benchmark — Transcript Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--briefs", type=Path,
        help="Directory containing session brief YAML files",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/transcripts"),
        help="Output directory for generated transcripts (default: data/transcripts)",
    )
    parser.add_argument(
        "--persona", type=Path,
        help="Path to full Maya persona document (optional, uses default if not provided)",
    )
    parser.add_argument(
        "--session", type=int,
        help="Generate only this session number",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing transcripts",
    )
    parser.add_argument(
        "--validate", type=Path, metavar="DIR",
        help="Validate transcripts in the given directory (no generation)",
    )

    args = parser.parse_args()

    if args.validate:
        validate_transcripts(args.validate)
        return

    if not args.briefs:
        parser.error("--briefs is required for generation (or use --validate)")

    run_generation(
        briefs_dir=args.briefs,
        output_dir=args.output,
        persona_file=args.persona,
        session_filter=args.session,
        force=args.force,
    )


if __name__ == "__main__":
    main()
