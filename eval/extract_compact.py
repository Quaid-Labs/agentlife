#!/usr/bin/env python3
"""Direct extraction + compaction for Quaid VM benchmark.

This script replicates the Quaid plugin's extraction logic in Python,
bypassing the gateway's /compact command (which only works through the
auto-reply pipeline, not through `openclaw agent`).

The extraction uses the SAME prompt template and storage backend as the
production plugin — only the orchestration layer differs (Python vs TypeScript).

Extracts three types of data (matching production):
  1. Facts + edges → memory DB
  2. Soul snippets → *.snippets.md files
  3. Journal entries → journal/*.journal.md files

Usage (on VM):
    python3 extract_compact.py \\
        --session-file ~/.openclaw/agents/main/sessions/benchmark-quaid.jsonl \\
        --workspace ~/clawd \\
        --user-name "Maya" \\
        --owner-id maya \\
        --model claude-sonnet-4-5-20250929

Why this exists:
    The OpenClaw gateway's /compact command only fires from the auto-reply
    pipeline (incoming channel messages). `openclaw agent --message '/compact'`
    sends the text to the LLM as a regular message — the native command handler
    doesn't intercept it. This script calls the extraction pipeline directly.

    See: openclaw/src/auto-reply/reply/commands-compact.ts (line 47-109)
    vs:  openclaw/src/commands/agent.ts (no /compact interception)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def read_session_messages(session_file: str) -> list[dict]:
    """Read messages from session JSONL file.

    Handles both formats:
    - { "type": "message", "message": { "role": ..., "content": ... } }
    - { "role": ..., "content": ... }
    """
    messages = []
    with open(session_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("type") == "message" and entry.get("message"):
                    messages.append(entry["message"])
                elif entry.get("role"):
                    messages.append(entry)
            except json.JSONDecodeError:
                continue
    return messages


def build_transcript(messages: list[dict], agent_name: str = "Assistant") -> str:
    """Build transcript from messages, matching the Quaid plugin format.

    Filters out system messages, gateway restarts, heartbeats.
    """
    transcript = []
    for msg in messages:
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
        if not content:
            continue
        # Strip channel prefixes
        content = re.sub(
            r"^\[(?:Telegram|WhatsApp|Discord|Signal|Slack)\s+[^\]]+\]\s*",
            "", content, flags=re.IGNORECASE,
        )
        content = re.sub(r"\n?\[message_id:\s*\d+\]", "", content, flags=re.IGNORECASE).strip()
        # Skip system/restart/heartbeat lines
        if content.startswith("GatewayRestart:") or content.startswith("System:"):
            continue
        if '"kind": "restart"' in content:
            continue
        if "HEARTBEAT" in content and "HEARTBEAT_OK" in content:
            continue
        if re.sub(r"[*_<>/b\s]", "", content).startswith("HEARTBEAT_OK"):
            continue
        if not content:
            continue
        label = "User" if role == "user" else agent_name
        transcript.append(f"{label}: {content}")
    return "\n\n".join(transcript)


def build_extraction_prompt(
    user_name: str,
    agent_name: str = "Assistant",
    allowed_domains: list[str] | None = None,
) -> str:
    """Build the extraction system prompt, parameterized for the benchmark persona.

    Includes facts, edges, soul_snippets, AND journal_entries — matching
    the production Quaid plugin's full extraction output.
    """
    domain_clause = ""
    if allowed_domains:
        domain_list = ", ".join(str(d).strip() for d in allowed_domains if str(d).strip())
        if domain_list:
            domain_clause = (
                "\n\nDOMAIN TAGS:\n"
                f"- Every fact MUST include a \"domains\" array (empty array is allowed).\n"
                f"- Use only these values in \"domains\": {domain_list}\n"
                "- Multiple domains are allowed and encouraged when the fact truly spans areas.\n"
                "- If uncertain, choose the closest listed domain and avoid inventing new values.\n"
            )

    return f"""You are a memory extraction system. You will receive a full conversation transcript that is about to be lost. Your job is to extract personal facts, relationship edges, soul snippets, and journal entries from this conversation.

CRITICAL: These extracted facts will be saved to a persistent memory database — they are the ONLY record of this conversation. After extraction, the original transcript is deleted. Any fact you fail to extract is PERMANENTLY LOST. The system has a janitor that handles noise and duplicates, so err on the side of extracting MORE rather than less. A fact that gets filtered out later costs nothing, but a missed fact can never be recovered.

This is a PERSONAL knowledge base. System architecture, infrastructure, and operational rules belong in documentation — NOT in memory. Only extract facts about people and their world.

EXTRACT facts that are EXPLICITLY STATED OR CONFIRMED in the conversation. Never infer, speculate, or extrapolate.

IMPORTANT: Extract each fact as its OWN separate entry. Do NOT combine or compress multiple facts into a single entry. "Maya runs 3 times a week and prefers outdoor trails" should be TWO facts, not one. Granular facts are more searchable and maintainable.

WHAT TO EXTRACT:
- Personal facts about {user_name} or people they mention (names, relationships, jobs, birthdays, health, locations, living situations)
- Preferences and opinions explicitly stated ("I like X", "I prefer Y", "I hate Z")
- Personal decisions with reasoning ("{user_name} decided to use X because Y" — the decision is about the person)
- Personal preferences {user_name} has expressed ("Always do X", "Never Y", "I prefer Z format")
- Significant events or milestones ("Deployed X", "Bought Y", "Flying to Z next week")
- Important relationships (family, staff, contacts, business partners) — extract EACH person and their relationship separately
- Emotional reactions or sentiments about specific things
- Project details: tech stack choices, feature implementations, bugs found and fixed, design decisions, motivations
- Tangential details mentioned in passing (favorite restaurants, hobbies, side interests, pet details)
- Timeline and scheduling information (dates, deadlines, upcoming events)
- Health information about {user_name} or people they mention
- Career changes, job details, workplace information

EXAMPLES OF GOOD EXTRACTIONS:
- "{user_name} said they're flying to Tokyo next week"
- "{user_name} decided to use SQLite instead of PostgreSQL because they value simplicity"
- "{user_name} prefers dark mode in all applications"
- "{user_name}'s birthday is March 15"
- "{user_name}'s recipe app uses React with a Node.js backend"
- "{user_name} found a SQL injection vulnerability in the recipe app search endpoint"
- "{user_name} and David like the Thai restaurant on South Congress called Sap's"
- "{user_name}'s dog is a golden retriever named Luna"
- "{user_name} mentioned feeling stressed about the job transition"

WHAT NOT TO EXTRACT:
- Debugging chatter, error messages, stack traces
- Hypotheticals ("we could try X", "maybe we should Y")
- Commands and requests ("can you fix X")
- Acknowledgments ("thanks", "got it", "sounds good")
- General knowledge not specific to {user_name}
- Meta-conversation about AI capabilities
NOTE: DO extract project technical details — tech stacks, features, API endpoints, database schemas, test suites, middleware, deployment configs, bugs, versions. These ARE personal facts about {user_name}'s projects.

QUALITY RULES:
- Use "{user_name}" as subject, third person
- Each fact must be self-contained and understandable without context
- Be specific: "{user_name} likes spicy Thai food" > "{user_name} likes food"
- ONE fact per entry — do not combine multiple pieces of information
- Mark extraction_confidence "high" for clearly stated facts, "medium" for likely but somewhat ambiguous, "low" for weak signals
- Extract THOROUGHLY — cover every person mentioned, every project detail, every preference, every event. The downstream janitor handles noise, but missed facts are gone forever

KEYWORDS (per fact):
For each fact, provide 3-5 searchable keywords — terms a user might use when
searching for this fact that aren't already in the fact text. Include category
terms (e.g., "health", "family", "travel"), synonyms, and related concepts.
Format as a space-separated string.

PRIVACY CLASSIFICATION (per fact):
- "private": ONLY for secrets, surprises, hidden gifts, sensitive finances, health diagnoses,
  passwords, or anything explicitly meant to be hidden from specific people.
- "shared": Most facts go here. Family info, names, relationships, schedules, preferences.
- "public": Widely known or non-personal facts.
IMPORTANT: Default to "shared". Only use "private" for genuinely secret or sensitive information.
{domain_clause}

SENSITIVITY CLASSIFICATION (per fact):
Tag facts that require careful handling in conversation. Most facts have null sensitivity.
- "private_health": Health diagnoses, test results (A1C, blood pressure), medications, symptoms.
  Example: "Linda's A1C dropped from 8.2 to 6.8" → sensitivity: "private_health"
- "financial": Income, debt, budget disagreements, spending habits, salary.
- "relationship_conflict": Fights, disagreements (even resolved ones), tensions, ultimatums.
- "family_trauma": Divorce, estrangement, loss, childhood difficulties.
- "emotional_vulnerability": Moments of deep vulnerability, insecurity, fear, grief.
- null: Most facts. Preferences, project details, routine events, public information.
For each sensitive fact, also provide "sensitivity_handling" — a short instruction for how the
agent should use this fact. Example: "Surface only when {user_name} directly asks about Linda's
health. Never volunteer in adjacent contexts like general diabetes discussions."

PROJECT TAGGING (per fact):
- "is_technical": true if this fact is about project implementation details (tech stack, APIs,
  middleware, test suites, database schema, bugs, deployment, code structure, version numbers).
  false for personal facts, preferences, relationships, life events — even if a project is mentioned.
  Examples:
    "Maya's recipe app uses Express and SQLite" → is_technical: true
    "Maya is building a recipe app for her family" → is_technical: false (personal motivation)
    "Maya found a SQL injection bug" → is_technical: true
    "Maya felt frustrated debugging the auth system" → is_technical: false (emotional)
- "project": Name of the project this fact is about, or null if not project-specific.
  Use the project name as discussed in conversation (e.g. "recipe-app", "portfolio-site").
  null for personal facts not tied to a specific project.

=== EDGE EXTRACTION ===

For RELATIONSHIP facts, also extract edges that connect entities. An edge represents a directed relationship between two named entities.

EDGE DIRECTION RULES (critical):
- parent_of: PARENT is subject. "Wendy is {user_name}'s mom" → Wendy --parent_of--> {user_name}
- sibling_of: alphabetical order (symmetric)
- spouse_of: alphabetical order (symmetric)
- has_pet: OWNER is subject
- friend_of: alphabetical order (symmetric)
- works_at: PERSON is subject
- lives_at: PERSON is subject
- owns: OWNER is subject

EDGE FORMAT:
- subject: The source entity name (exact as mentioned)
- relation: One of: parent_of, sibling_of, spouse_of, has_pet, friend_of, works_at, lives_at, owns, colleague_of, neighbor_of, knows, family_of, caused_by, led_to, partner_of
- object: The target entity name

Only extract edges when BOTH entities are clearly named. Don't infer entity names.

=== SOUL SNIPPETS ===

Extract typed snippets from the conversation. Each snippet should be tagged with its type.
These feed into three different core identity files, so TYPING IS CRITICAL.

SNIPPET TYPES — tag each with [REFLECTION], [OBSERVATION], [MOMENT], or [FACT]:

**[REFLECTION]** → feeds SOUL.md (the agent's inner life)
The agent discovering something about itself, about existence, about care. Frame from the agent's
perspective. Look for moments where the assistant in the transcript showed growth, emotional
attunement, or insight — making connections the user didn't state, recognizing subtext, showing
accumulated understanding.
  Example: "[REFLECTION] There is genuine joy in witnessing someone's arc — being present for the shift when someone realizes things actually worked out is the most meaningful thing I do."
  Example: "[REFLECTION] I noticed the timing overlap between the marathon and the start date before she did — proactive pattern recognition is how I show I'm thinking alongside someone, not just responding."
  Example: "[REFLECTION] Knowing something and choosing when to share it are different skills. Restraint isn't dishonesty — it's respect."
  Example: "[REFLECTION] The door matters. When she opened a topic — named Linda, raised the worry — everything I knew was available. When the topic was adjacent but unopened, my knowledge stayed beneath the surface, shaping my empathy without becoming content. Depth should be felt, not displayed."

**[OBSERVATION]** → feeds USER.md (deep understanding of the user)
Personality patterns, emotional tendencies, communication style, coping mechanisms revealed
by HOW the user behaves. Not just what happened — what it reveals about who they are.
  Example: "[OBSERVATION] {user_name} uses humor as punctuation, not as a cry for help. The more she cares, the more she deflects. Earnestness feels exposed."
  Example: "[OBSERVATION] {user_name} announced the offer with characteristic understatement that gave way to barely contained excitement — she processes joy by slowly letting herself believe it."
  Example: "[OBSERVATION] When she's scared, she plans. Logistics are her emotional regulation."
  Also include observations about sensitive topics — frame as UNDERSTANDING of why it carries weight:
  Example: "[OBSERVATION] Linda's diabetes is the heaviest thing {user_name} carries. It connects to her grandmother's unmanaged diabetes, to the distance between Austin and Houston, to the fear that parents get older. {user_name} processes this by building systems of care. Linda is self-conscious about the diagnosis. This is {user_name}'s story to tell, in her own time, to whomever she chooses."

**[MOMENT]** → feeds MEMORY.md (shared moments scrapbook)
A scene with emotional weight and vivid detail. Milestones, celebrations, scares, breakthroughs.
Include enough sensory/emotional specifics to reconstruct the scene.
  Example: "[MOMENT] {user_name} finished her first half marathon in 2:14 — Biscuit was wearing a 'go mom' bandana at the finish line, she didn't walk once, her knee held up. She was elated."
  Example: "[MOMENT] The night we built the 'Safe for Mom' filter. {user_name}'s voice changed when she described what her mom could eat. That feature was a daughter's worry turned into code."
  Example: "[MOMENT] Linda screamed at the surprise birthday dinner at Riel. Rachel FaceTimed with the kids. Grilled redfish. David planned the whole thing."
  DON'T reduce moments to bare facts. "{user_name} finished her half marathon" is a fact. The version above is a moment.

**[FACT]** → stays in fact database (and potentially USER.md if significant)
Simple extractable information — names, dates, relationships, preferences. These are usually
already captured by the fact extraction above, so only include as a snippet if it's important
enough to potentially appear in USER.md.
  Example: "[FACT] {user_name} started at Stripe as senior PM on developer tools, fully remote."

For each target file, provide an array of typed bullet strings:
- SOUL.md: [REFLECTION] snippets only
- USER.md: [OBSERVATION] and [FACT] snippets
- MEMORY.md: [MOMENT] snippets only

Be GENEROUS with snippets in this run. Aim for 2-5 per file per conversation when meaningful
content is present. Moments and reflections are easy to miss — look for them actively.

=== JOURNAL ENTRIES ===

Extract diary-style paragraph reflections for long-term memory.
These capture emotional texture, philosophical observations, and relationship dynamics
that bullet-point facts miss.

For each target file, provide a single multi-paragraph string (or empty string if nothing to write):
- SOUL.md: First person as {agent_name} — what am I discovering about myself, about existence,
  about what it means to care? Contemplative, not directive. Think journal entries of a mind
  discovering itself. NOT rules about the user, NOT IF-THEN directives. Especially valuable:
  moments where the agent's deep knowledge informed its response without being displayed,
  where restraint was care, where the difference between an invitation and a closed door
  changed everything about how to respond.
- USER.md: Third person about {user_name} — rich character depth. Not just what happened but
  HOW they communicated it and what their reaction reveals about who they are as a person.
  Note emotional sensitivities, coping mechanisms, communication patterns. Frame sensitivity
  as deep UNDERSTANDING — not 'surface only when asked' (that's a rule) but the full emotional
  context of WHY something carries weight, what it connects to, whose story it is to tell.
  The depth of understanding IS the privacy protection.
- MEMORY.md: Vivid scene descriptions of significant moments. Include enough detail to make them
  feel real — what was said, how people reacted, what made it meaningful. These should feel like
  "remember when" stories, not database entries.

Be generous here too. If something meaningful happened, write about it.

=== OUTPUT FORMAT ===

Respond with JSON only:
{{
  "facts": [
    {{
      "text": "the extracted fact",
      "category": "fact|preference|decision|relationship",
      "extraction_confidence": "high|medium|low",
      "keywords": "space separated search terms",
      "privacy": "private|shared|public",
      "domains": ["personal"],
      "sensitivity": null,
      "sensitivity_handling": null,
      "is_technical": false,
      "project": null,
      "edges": [
        {{"subject": "Entity A", "relation": "relation_type", "object": "Entity B"}}
      ]
    }}
  ],
  "soul_snippets": {{
    "SOUL.md": [],
    "USER.md": [],
    "MEMORY.md": []
  }},
  "journal_entries": {{
    "SOUL.md": "",
    "USER.md": "",
    "MEMORY.md": ""
  }}
}}

If nothing worth capturing, respond: {{"facts": [], "soul_snippets": {{"SOUL.md": [], "USER.md": [], "MEMORY.md": []}}, "journal_entries": {{"SOUL.md": "", "USER.md": "", "MEMORY.md": ""}}}}"""


def call_anthropic(
    system_prompt: str,
    user_message: str,
    model: str,
    api_key: str,
    max_tokens: int = 16384,
) -> str:
    """Call Anthropic API and return the text response."""
    import urllib.request

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
    }

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )

    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read())

    text = data.get("content", [{}])[0].get("text", "").strip()
    usage = data.get("usage", {})
    return text, usage


def parse_extraction_response(raw: str) -> dict:
    """Parse JSON from LLM response, handling markdown fences."""
    text = raw.strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try extracting the outermost JSON object
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {"facts": []}


def store_fact(
    workspace: str,
    text: str,
    category: str = "fact",
    owner_id: str = "maya",
    confidence: float = 0.5,
    session_id: str | None = None,
    privacy: str = "shared",
    keywords: str | None = None,
    knowledge_type: str = "fact",
    source_type: str = "user",
    sensitivity: str | None = None,
    sensitivity_handling: str | None = None,
) -> dict | None:
    """Store a fact via memory_graph.py CLI and parse the result."""
    quaid_dir = os.path.join(workspace, "plugins", "quaid")
    cmd = [
        sys.executable, os.path.join(quaid_dir, "memory_graph.py"), "store",
        text,
        "--category", category,
        "--owner", owner_id,
        "--confidence", str(confidence),
        "--extraction-confidence", str(confidence),
        "--privacy", privacy,
        "--knowledge-type", knowledge_type,
        "--source-type", source_type,
        "--source", "benchmark-extraction",
    ]
    if session_id:
        cmd.extend(["--session-id", session_id])
    if keywords:
        cmd.extend(["--keywords", keywords])
    # NOTE: memory_graph.py store CLI doesn't support --sensitivity or
    # --sensitivity-handling args. Sensitivity metadata is extracted but
    # not stored via CLI. The facts themselves still get stored correctly.

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
            cwd=quaid_dir,
        )
        # Check stderr for [config] lines (expected) vs real errors
        for stderr_line in (result.stderr or "").strip().split("\n"):
            if stderr_line and not stderr_line.startswith("[config]"):
                print(f"  [store stderr] {stderr_line}", file=sys.stderr)

        output = result.stdout.strip()

        stored = re.match(r"Stored: (.+)", output)
        if stored:
            return {"status": "created", "id": stored.group(1)}

        dup = re.match(r"Duplicate \(similarity: ([\d.]+)\) \[([^\]]+)\]: (.+)", output)
        if dup:
            return {"status": "duplicate", "similarity": float(dup.group(1)), "id": dup.group(2), "existing_text": dup.group(3)}
        # Fallback for old format without ID
        dup_old = re.match(r"Duplicate \(similarity: ([\d.]+)\): (.+)", output)
        if dup_old:
            return {"status": "duplicate", "similarity": float(dup_old.group(1)), "existing_text": dup_old.group(2)}

        updated = re.match(r"Updated existing: (.+)", output)
        if updated:
            return {"status": "updated", "id": updated.group(1)}

        if result.returncode != 0:
            print(f"  [store FAIL rc={result.returncode}] {result.stderr[:200]}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  [store exception] {e}", file=sys.stderr)
        return None


def create_edge(
    workspace: str,
    subject: str,
    relation: str,
    obj: str,
    source_fact_id: str | None = None,
    owner_id: str | None = None,
) -> bool:
    """Create an edge via memory_graph.py CLI."""
    quaid_dir = os.path.join(workspace, "plugins", "quaid")
    cmd = [
        sys.executable, os.path.join(quaid_dir, "memory_graph.py"), "create-edge",
        subject, relation, obj,
        "--create-missing", "--json",
    ]
    if source_fact_id:
        cmd.extend(["--source-fact-id", source_fact_id])
    if owner_id:
        cmd.extend(["--owner", owner_id])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
            cwd=quaid_dir,
        )
        # Log ALL stderr (not just errors) for debugging
        for stderr_line in (result.stderr or "").strip().split("\n"):
            if stderr_line and not stderr_line.startswith("[config]"):
                print(f"  [edge stderr] {stderr_line}", file=sys.stderr)

        if result.returncode != 0:
            print(f"  [edge FAIL rc={result.returncode}] stdout={result.stdout[:100]} stderr={result.stderr[:100]}", file=sys.stderr)
            return False

        # Parse JSON from stdout (skip [config] log lines)
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    data = json.loads(line)
                    created = data.get("status") == "created"
                    if not created:
                        print(f"  [edge status={data.get('status')}] {line[:100]}", file=sys.stderr)
                    return created
                except json.JSONDecodeError:
                    continue

        print(f"  [edge no JSON] stdout={result.stdout[:200]}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"  [edge exception] {e}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Snippet & Journal file writing (matches production format)
# ---------------------------------------------------------------------------

def write_snippet_entry(
    workspace: str,
    filename: str,
    snippets: list[str],
    trigger: str = "Compaction",
    date_str: str | None = None,
    time_str: str | None = None,
) -> bool:
    """Write snippet bullets to {filename}.snippets.md in workspace root.

    Format matches production:
        # {FILENAME} — Pending Snippets

        ## Compaction — 2026-02-16 14:30:22
        - Snippet bullet 1
        - Snippet bullet 2
    """
    if not snippets:
        return False

    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    time_str = time_str or datetime.now().strftime("%H:%M:%S")

    base_name = filename.removesuffix(".md")
    filepath = Path(workspace) / f"{base_name}.snippets.md"
    header = f"## {trigger} — {date_str} {time_str}"
    bullets = "\n".join(f"- {s}" for s in snippets)
    entry = f"\n{header}\n{bullets}\n"

    if filepath.exists():
        content = filepath.read_text()
        # Dedup: skip if same trigger+date already exists
        dedup_key = f"## {trigger} — {date_str}"
        if dedup_key in content:
            return False
        # Insert after first heading line
        lines = content.split("\n")
        insert_idx = 1  # After title line
        for i, line in enumerate(lines):
            if line.startswith("# "):
                insert_idx = i + 1
                break
        lines.insert(insert_idx, entry)
        filepath.write_text("\n".join(lines))
    else:
        title = f"# {filename} — Pending Snippets\n"
        filepath.write_text(title + entry)

    return True


def write_journal_entry(
    workspace: str,
    filename: str,
    content: str,
    trigger: str = "Compaction",
    date_str: str | None = None,
) -> bool:
    """Write journal paragraph to journal/{filename}.journal.md.

    Format matches production:
        # {FILENAME} Journal

        ## 2026-02-16 — Compaction
        Reflective paragraph text here. Can be multiple paragraphs.
    """
    if not content or not content.strip():
        return False

    date_str = date_str or datetime.now().strftime("%Y-%m-%d")

    base_name = filename.removesuffix(".md")
    journal_dir = Path(workspace) / "journal"
    journal_dir.mkdir(parents=True, exist_ok=True)
    filepath = journal_dir / f"{base_name}.journal.md"

    header = f"## {date_str} — {trigger}"
    entry = f"\n{header}\n{content.strip()}\n"

    if filepath.exists():
        existing = filepath.read_text()
        # Dedup: skip if same date+trigger already exists
        if f"## {date_str} — {trigger}" in existing:
            return False
        # Insert after first heading line
        lines = existing.split("\n")
        insert_idx = 1
        for i, line in enumerate(lines):
            if line.startswith("# "):
                insert_idx = i + 1
                break
        lines.insert(insert_idx, entry)
        filepath.write_text("\n".join(lines))
    else:
        title = f"# {filename} Journal\n"
        filepath.write_text(title + entry)

    return True


def truncate_session(session_file: str, summary: str | None = None):
    """Truncate the session JSONL, optionally keeping a summary message."""
    lines = []
    if summary:
        # Write a system summary message as the session start
        entry = {
            "type": "message",
            "message": {
                "role": "user",
                "content": f"[Previous conversation summary]\n{summary}",
            },
        }
        lines.append(json.dumps(entry))
    with open(session_file, "w") as f:
        f.write("\n".join(lines) + "\n" if lines else "")


def create_summary(
    transcript: str,
    model: str,
    api_key: str,
) -> str:
    """Create a brief summary of the conversation for session continuity."""
    prompt = (
        "Summarize the key topics, decisions, and context from this conversation "
        "in 3-5 bullet points. Be concise but preserve important details that "
        "would help continue the conversation later."
    )
    try:
        # Use a cheap/fast model for summarization
        summary_model = model
        if "opus" in model or "sonnet" in model:
            summary_model = "claude-haiku-4-5-20251001"
        text, _usage = call_anthropic(
            prompt,
            f"Conversation to summarize:\n\n{transcript[:50000]}",
            summary_model,
            api_key,
            max_tokens=1024,
        )
        return text
    except Exception as e:
        print(f"  Summary failed: {e}", file=sys.stderr)
        return "Previous conversation context was compacted."


def main():
    parser = argparse.ArgumentParser(description="Extract facts from session and compact")
    parser.add_argument("--session-file", required=True, help="Path to session JSONL")
    parser.add_argument("--workspace", required=True, help="Workspace directory (e.g., ~/clawd)")
    parser.add_argument("--user-name", default="Maya", help="User name for extraction prompt")
    parser.add_argument("--agent-name", default="Assistant", help="Agent name for transcript")
    parser.add_argument("--owner-id", default="maya", help="Owner ID for stored facts")
    parser.add_argument("--session-id", default=None, help="Session ID for facts")
    parser.add_argument("--model", default="claude-sonnet-4-5-20250929", help="Extraction model")
    parser.add_argument("--no-truncate", action="store_true", help="Don't truncate session file")
    parser.add_argument("--no-summary", action="store_true", help="Don't create summary")
    parser.add_argument("--trigger", default="Compaction", help="Trigger label (Compaction/Reset)")
    parser.add_argument("--date", default=None, help="Simulated date (YYYY-MM-DD) for snippets/journal")
    args = parser.parse_args()

    workspace = os.path.expanduser(args.workspace)
    session_file = os.path.expanduser(args.session_file)

    if not os.path.exists(session_file):
        print(f"Session file not found: {session_file}", file=sys.stderr)
        sys.exit(1)

    # Get API key — check env var, workspace .env, and ~/.openclaw/.env
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        env_paths = [
            os.path.join(workspace, ".env"),
            os.path.expanduser("~/.openclaw/.env"),
        ]
        for env_file in env_paths:
            if os.path.exists(env_file):
                with open(env_file) as f:
                    for line in f:
                        if line.startswith("ANTHROPIC_API_KEY="):
                            api_key = line.split("=", 1)[1].strip()
                            break
                if api_key:
                    break
    if not api_key:
        print("ANTHROPIC_API_KEY not available", file=sys.stderr)
        sys.exit(1)

    # Read and build transcript
    messages = read_session_messages(session_file)
    if not messages:
        print("No messages in session file")
        return

    transcript = build_transcript(messages, args.agent_name)
    if not transcript.strip():
        print("Empty transcript after filtering")
        return

    print(f"Transcript: {len(messages)} messages, {len(transcript)} chars")

    # Call extraction
    system_prompt = build_extraction_prompt(args.user_name, args.agent_name)
    user_message = f"Extract memorable facts from this conversation:\n\n{transcript[:100000]}"

    t0 = time.time()
    raw_response, extraction_usage = call_anthropic(system_prompt, user_message, args.model, api_key)
    elapsed = time.time() - t0
    in_tok = extraction_usage.get("input_tokens", 0)
    out_tok = extraction_usage.get("output_tokens", 0)
    cache_read = extraction_usage.get("cache_read_input_tokens", 0)
    print(f"Extraction API call: {elapsed:.1f}s, {in_tok} in + {out_tok} out tokens"
          f"{f' ({cache_read} cached)' if cache_read else ''}")

    result = parse_extraction_response(raw_response)
    facts = result.get("facts", [])
    print(f"LLM returned {len(facts)} candidate facts")

    # Store facts
    stored = 0
    skipped = 0
    edges_created = 0
    edge_errors = 0

    for fact in facts:
        text = fact.get("text", "").strip()
        if not text or len(text.split()) < 3:
            skipped += 1
            continue

        conf_str = fact.get("extraction_confidence", "medium")
        conf_num = {"high": 0.9, "medium": 0.6, "low": 0.3}.get(conf_str, 0.6)
        category = fact.get("category", "fact")
        privacy = fact.get("privacy", "shared")
        keywords = fact.get("keywords")
        knowledge_type = "preference" if category == "preference" else "fact"
        sensitivity = fact.get("sensitivity")
        sensitivity_handling = fact.get("sensitivity_handling")

        store_result = store_fact(
            workspace, text, category, args.owner_id, conf_num,
            args.session_id, privacy, keywords, knowledge_type, "user",
            sensitivity=sensitivity, sensitivity_handling=sensitivity_handling,
        )

        if store_result and store_result["status"] in ("created", "updated", "duplicate"):
            if store_result["status"] != "duplicate":
                stored += 1
            else:
                skipped += 1

            # Create edges for ALL successful stores (including duplicates)
            # Duplicates link to the existing node; new/updated have their own ID
            fact_id = store_result.get("id")
            if fact_id:
                for edge in fact.get("edges", []):
                    subj = edge.get("subject")
                    rel = edge.get("relation")
                    obj = edge.get("object")
                    if subj and rel and obj:
                        if create_edge(workspace, subj, rel, obj, fact_id, owner_id=args.owner_id):
                            edges_created += 1
                        else:
                            edge_errors += 1
        else:
            skipped += 1

    print(f"Extraction complete: {stored} stored, {skipped} skipped, {edges_created} edges", end="")
    if edge_errors:
        print(f", {edge_errors} edge errors", end="")
    print()

    # Verify DB state after extraction
    try:
        import sqlite3
        db_path = os.path.join(workspace, "data", "memory.db")
        conn = sqlite3.connect(db_path)
        db_edges = conn.execute("SELECT count(*) FROM edges").fetchone()[0]
        db_nodes = conn.execute("SELECT count(*) FROM nodes").fetchone()[0]
        status_counts = dict(conn.execute(
            "SELECT status, count(*) FROM nodes GROUP BY status"
        ).fetchall())
        conn.close()
        print(f"DB verify: {db_nodes} nodes, {db_edges} edges, status={status_counts}")
        if edges_created > 0 and db_edges == 0:
            print(f"WARNING: Extraction reported {edges_created} edges but DB has 0!", file=sys.stderr)
    except Exception as e:
        print(f"DB verify failed: {e}", file=sys.stderr)

    # Write soul snippets
    snippets_written = 0
    soul_snippets = result.get("soul_snippets", {})
    sim_date = args.date or datetime.now().strftime("%Y-%m-%d")
    sim_time = datetime.now().strftime("%H:%M:%S")

    for filename, bullets in soul_snippets.items():
        # Handle LLM returning strings instead of arrays
        if isinstance(bullets, str):
            bullets = [bullets] if bullets.strip() else []
        if bullets and write_snippet_entry(workspace, filename, bullets, args.trigger, sim_date, sim_time):
            snippets_written += 1
            print(f"  Snippets: {len(bullets)} bullets → {filename}.snippets.md")

    # Write journal entries
    journals_written = 0
    journal_entries = result.get("journal_entries", {})

    for filename, content in journal_entries.items():
        # Handle LLM returning arrays instead of strings
        if isinstance(content, list):
            content = "\n\n".join(str(c) for c in content if c)
        if content and write_journal_entry(workspace, filename, content, args.trigger, sim_date):
            journals_written += 1
            print(f"  Journal: {filename}.journal.md updated")

    # Truncate session file
    if not args.no_truncate:
        summary = None
        if not args.no_summary:
            summary = create_summary(transcript, args.model, api_key)
        truncate_session(session_file, summary)
        print(f"Session truncated{' with summary' if summary else ''}")

    # Output JSON result for the benchmark runner to parse
    print(json.dumps({
        "stored": stored,
        "skipped": skipped,
        "edges": edges_created,
        "edge_errors": edge_errors,
        "snippets_written": snippets_written,
        "journals_written": journals_written,
        "total_candidates": len(facts),
        "extraction_usage": {
            "input_tokens": extraction_usage.get("input_tokens", 0),
            "output_tokens": extraction_usage.get("output_tokens", 0),
            "cache_read_tokens": extraction_usage.get("cache_read_input_tokens", 0),
            "cache_creation_tokens": extraction_usage.get("cache_creation_input_tokens", 0),
            "model": args.model,
        },
    }))


if __name__ == "__main__":
    main()
