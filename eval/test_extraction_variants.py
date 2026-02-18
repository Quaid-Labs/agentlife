#!/usr/bin/env python3
"""Compare extraction prompt variants on the same transcript.

Runs 3 extraction variants on the full benchmark transcript:
1. Current prompt + Sonnet (baseline — what we just ran)
2. Loosened prompt + Sonnet (more permissive extraction)
3. Current prompt + Opus (better model, same prompt)

Compares fact count, edge count, and sample facts.
"""

import json
import os
import re
import sys
import time

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    env_path = os.path.expanduser("~/clawd/.env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("ANTHROPIC_API_KEY="):
                    API_KEY = line.strip().split("=", 1)[1]
                    break

SONNET = "claude-sonnet-4-5-20250929"
OPUS = "claude-opus-4-6"

USER_NAME = "Maya"


def build_current_prompt():
    """Current strict extraction prompt."""
    return f"""You are a memory extraction system. You will receive a full conversation transcript that is about to be lost. Your job is to extract personal facts, relationship edges, soul snippets, and journal entries from this conversation.

This is a PERSONAL knowledge base. System architecture, infrastructure, and operational rules belong in documentation — NOT in memory. Only extract facts about people and their world.

EXTRACT facts that are EXPLICITLY STATED OR CONFIRMED in the conversation. Never infer, speculate, or extrapolate.

WHAT TO EXTRACT:
- Personal facts about {USER_NAME} or people they mention (names, relationships, jobs, birthdays, health, locations)
- Preferences and opinions explicitly stated ("I like X", "I prefer Y", "I hate Z")
- Personal decisions with reasoning ("{USER_NAME} decided to use X because Y" — the decision is about the person)
- Personal preferences {USER_NAME} has expressed ("Always do X", "Never Y", "I prefer Z format")
- Significant events or milestones ("Deployed X", "Bought Y", "Flying to Z next week")
- Important relationships (family, staff, contacts, business partners)
- Emotional reactions or sentiments about specific things
- Project details: tech stack choices, feature implementations, bugs found and fixed, design decisions

EXAMPLES OF GOOD EXTRACTIONS:
- "{USER_NAME} said they're flying to Tokyo next week"
- "{USER_NAME} decided to use SQLite instead of PostgreSQL because they value simplicity"
- "{USER_NAME} prefers dark mode in all applications"
- "{USER_NAME}'s birthday is March 15"
- "{USER_NAME}'s recipe app uses React with a Node.js backend"
- "{USER_NAME} found a SQL injection vulnerability in the recipe app search endpoint"

WHAT NOT TO EXTRACT (belongs in docs/RAG, not personal memory):
- System architecture descriptions ("The memory system uses SQLite with WAL mode")
- Infrastructure knowledge ("Ollama runs on port 11434")
- Operational rules for AI agents
- Tool/config descriptions
- Debugging chatter, error messages, stack traces
- Hypotheticals ("we could try X", "maybe we should Y")
- Commands and requests ("can you fix X")
- Acknowledgments ("thanks", "got it", "sounds good")
- General knowledge not specific to {USER_NAME}
- Meta-conversation about AI capabilities

QUALITY RULES:
- Use "{USER_NAME}" as subject, third person
- Each fact must be self-contained and understandable without context
- Be specific: "{USER_NAME} likes spicy Thai food" > "{USER_NAME} likes food"
- Mark extraction_confidence "high" for clearly stated facts, "medium" for likely but somewhat ambiguous, "low" for weak signals
- Extract personal facts AND project/app details comprehensively — the nightly janitor handles noise, but missed facts are gone forever

KEYWORDS (per fact):
For each fact, provide 3-5 searchable keywords — terms a user might use when
searching for this fact that aren't already in the fact text. Include category
terms (e.g., "health", "family", "travel"), synonyms, and related concepts.
Format as a space-separated string.

PRIVACY CLASSIFICATION (per fact):
- "private": ONLY for secrets, surprises, hidden gifts, sensitive finances, health diagnoses, passwords, or anything explicitly meant to be hidden from specific people.
- "shared": Most facts go here. Family info, names, relationships, schedules, preferences.
- "public": Widely known or non-personal facts.
IMPORTANT: Default to "shared". Only use "private" for genuinely secret or sensitive information.

=== EDGE EXTRACTION ===

For RELATIONSHIP facts, also extract edges that connect entities.

EDGE DIRECTION RULES (critical):
- parent_of: PARENT is subject
- sibling_of: alphabetical order (symmetric)
- spouse_of: alphabetical order (symmetric)
- has_pet: OWNER is subject
- friend_of: alphabetical order (symmetric)
- works_at: PERSON is subject
- lives_at: PERSON is subject

EDGE FORMAT:
- subject: The source entity name
- relation: One of: parent_of, sibling_of, spouse_of, has_pet, friend_of, works_at, lives_at, owns, colleague_of, neighbor_of, knows, family_of, partner_of
- object: The target entity name

Only extract edges when BOTH entities are clearly named.

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
      "edges": []
    }}
  ]
}}"""


def build_loose_prompt():
    """Loosened extraction prompt — cast a wider net."""
    return f"""You are a memory extraction system. You will receive a full conversation transcript that is about to be lost forever. Your job is to extract EVERY personal fact worth remembering.

CRITICAL: This transcript covers ~2 months of conversation. Extract COMPREHENSIVELY — err on the side of extracting too much. The downstream janitor will filter noise, but facts you miss here are GONE FOREVER.

WHAT TO EXTRACT (be thorough):
- ALL personal facts about {USER_NAME} and people they mention
- Names, relationships, jobs, ages, birthdays, locations, health details
- ALL preferences, opinions, likes, dislikes, habits, routines
- Hobbies, exercise habits, dietary preferences, favorite places
- Emotional states, frustrations, celebrations, milestones
- Career details: job title, company, coworkers, work dynamics, promotions, job changes
- Relationship dynamics: how people interact, supportiveness, conflicts
- Project details: EVERY feature, bug, tech choice, design decision, version, endpoint
- Casual mentions: restaurants, recipes tried, weekend plans, travel
- Health information: conditions, diagnoses, test results, treatments
- Future plans: things being considered, goals, aspirations
- Anecdotes and stories shared (summarize the key facts from each)
- Things that changed over time (moved, got promoted, updated opinion)

EXTRACTION RULES:
- Use "{USER_NAME}" as subject, third person
- Each fact must be self-contained
- Be specific and detailed
- HIGH confidence for clearly stated facts
- MEDIUM for reasonable inferences from context
- LOW for weak signals or implications
- When in doubt, EXTRACT IT — better to capture too much than too little
- For long conversations, aim for 100+ facts if the content supports it

KEYWORDS: 3-5 searchable terms per fact.
PRIVACY: Default "shared". Only "private" for genuinely secret info.

=== EDGE EXTRACTION ===

Extract relationship edges between named entities.
- parent_of, sibling_of, spouse_of, has_pet, friend_of, works_at, lives_at, owns, colleague_of, partner_of
- Symmetric relations (sibling, spouse, friend): alphabetical order

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
      "edges": []
    }}
  ]
}}"""


def call_anthropic(system_prompt, user_message, model, max_tokens=16384):
    """Call Anthropic API."""
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
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01",
        },
    )

    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())

    text = data.get("content", [{}])[0].get("text", "").strip()
    usage = data.get("usage", {})
    return text, usage


def parse_response(raw):
    """Parse JSON from LLM response."""
    text = raw.strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group(0))
        return None


def load_transcript(path):
    """Load JSONL transcript and format as conversation text."""
    messages = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("type") == "message":
                msg = entry.get("message", {})
                role = msg.get("role", "")
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        b.get("text", "") for b in content if b.get("type") == "text"
                    )
                if role and content:
                    speaker = USER_NAME if role == "user" else "Assistant"
                    messages.append(f"{speaker}: {content}")
    return "\n\n".join(messages)


def run_variant(name, prompt, model, transcript):
    """Run one extraction variant and return results."""
    print(f"\n{'='*60}")
    print(f" {name}")
    print(f" Model: {model}")
    print(f"{'='*60}")

    t0 = time.time()
    raw, usage = call_anthropic(prompt, transcript, model)
    elapsed = time.time() - t0

    in_tok = usage.get("input_tokens", 0)
    out_tok = usage.get("output_tokens", 0)
    cache_read = usage.get("cache_read_input_tokens", 0)
    cost = in_tok * 3 / 1e6 + out_tok * 15 / 1e6  # Sonnet pricing
    if "opus" in model:
        cost = in_tok * 15 / 1e6 + out_tok * 75 / 1e6  # Opus pricing

    print(f"  Time: {elapsed:.1f}s")
    print(f"  Tokens: {in_tok:,} in + {out_tok:,} out"
          f"{f' ({cache_read:,} cached)' if cache_read else ''}")
    print(f"  Cost: ${cost:.4f}")

    parsed = parse_response(raw)
    if not parsed:
        print("  ERROR: Failed to parse response")
        return {"name": name, "error": "parse_failed", "raw": raw[:500]}

    facts = parsed.get("facts", [])
    edges = sum(len(f.get("edges", [])) for f in facts)

    print(f"  Facts: {len(facts)}")
    print(f"  Edges: {edges}")

    # Confidence breakdown
    from collections import Counter
    conf = Counter(f.get("extraction_confidence", "?") for f in facts)
    cat = Counter(f.get("category", "?") for f in facts)
    print(f"  Confidence: {dict(conf)}")
    print(f"  Categories: {dict(cat)}")

    # Sample facts
    print(f"\n  Sample facts (first 10):")
    for f in facts[:10]:
        print(f"    - {f['text'][:100]}")

    print(f"\n  Sample facts (last 10):")
    for f in facts[-10:]:
        print(f"    - {f['text'][:100]}")

    # Save full output
    output_path = f"/tmp/extraction-{name.lower().replace(' ', '-')}.json"
    with open(output_path, "w") as fp:
        json.dump({"facts": facts, "usage": usage, "elapsed": elapsed}, fp, indent=2)
    print(f"  Saved: {output_path}")

    return {
        "name": name,
        "model": model,
        "facts": len(facts),
        "edges": edges,
        "confidence": dict(conf),
        "categories": dict(cat),
        "tokens_in": in_tok,
        "tokens_out": out_tok,
        "cost": round(cost, 4),
        "elapsed": round(elapsed, 1),
    }


def main():
    transcript_path = "/tmp/benchmark-transcript.jsonl"
    if not os.path.exists(transcript_path):
        print(f"ERROR: {transcript_path} not found")
        sys.exit(1)

    transcript = load_transcript(transcript_path)
    print(f"Transcript: {len(transcript):,} chars, ~{len(transcript.split()):,} words")

    results = []

    # Variant 1: Current prompt + Sonnet
    results.append(run_variant(
        "Current Sonnet",
        build_current_prompt(),
        SONNET,
        transcript,
    ))

    # Variant 2: Loose prompt + Sonnet
    results.append(run_variant(
        "Loose Sonnet",
        build_loose_prompt(),
        SONNET,
        transcript,
    ))

    # Variant 3: Current prompt + Opus
    results.append(run_variant(
        "Current Opus",
        build_current_prompt(),
        OPUS,
        transcript,
    ))

    # Variant 4: Loose prompt + Opus
    results.append(run_variant(
        "Loose Opus",
        build_loose_prompt(),
        OPUS,
        transcript,
    ))

    # Summary
    print(f"\n{'='*60}")
    print(f" COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Variant':<20} {'Facts':>6} {'Edges':>6} {'Cost':>8} {'Time':>6}")
    print(f"{'─'*50}")
    for r in results:
        if "error" in r:
            print(f"{r['name']:<20} ERROR")
        else:
            print(f"{r['name']:<20} {r['facts']:>6} {r['edges']:>6} ${r['cost']:>6.4f} {r['elapsed']:>5.1f}s")


if __name__ == "__main__":
    main()
