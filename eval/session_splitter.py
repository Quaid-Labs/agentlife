#!/usr/bin/env python3
"""AgentLife — Session splitting strategies for Quaid extraction.

Determines WHEN to trigger extraction and HOW to split large accumulated
buffers. Works with timestamped messages from annotate_timestamps.py.

Three strategies (composable):

1. **Timeout-based**: 2hr inactivity gap triggers extraction of buffer
2. **Chunk by size**: If buffer > threshold, split into ~20K token chunks
   at message boundaries
3. **Chunk by topic (LLM)**: Use LLM to find conversation switching points,
   fall back to size-based if chunks are too large/small

Production flow:
  Messages accumulate → timeout or size threshold → split → extract each chunk

Usage:
    from session_splitter import SessionSplitter

    splitter = SessionSplitter(
        timeout_minutes=120,
        max_buffer_tokens=30000,
        target_chunk_tokens=20000,
        split_mode="size",  # or "topic"
    )

    # Feed chronological messages with timestamps
    chunks = splitter.split(timestamped_messages)
    for chunk in chunks:
        extract(chunk.messages)  # Send to Quaid extraction
"""

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))
from injector import count_tokens


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class TimestampedMessage:
    """A message with a timestamp."""
    role: str               # "user" or "assistant"
    content: str            # Message text
    timestamp_ms: int       # Milliseconds since epoch
    session_id: str = ""    # Which session this came from
    tokens: int = 0         # Cached token count

    def __post_init__(self):
        if self.tokens == 0:
            self.tokens = count_tokens(self.content)


@dataclass
class ExtractionChunk:
    """A chunk of messages to be extracted together."""
    messages: List[TimestampedMessage]
    trigger: str            # "timeout", "size", "topic", "day", "end"
    chunk_index: int = 0    # Index within a split (0 if no split needed)
    split_total: int = 1    # Total chunks from this split

    @property
    def total_tokens(self) -> int:
        return sum(m.tokens for m in self.messages)

    @property
    def timestamp_range(self) -> tuple[int, int]:
        if not self.messages:
            return (0, 0)
        return (self.messages[0].timestamp_ms, self.messages[-1].timestamp_ms)

    @property
    def session_ids(self) -> list[str]:
        seen = []
        for m in self.messages:
            if m.session_id and (not seen or seen[-1] != m.session_id):
                seen.append(m.session_id)
        return seen


# ---------------------------------------------------------------------------
# Size-based splitting
# ---------------------------------------------------------------------------

def split_by_size(
    messages: List[TimestampedMessage],
    target_tokens: int = 20000,
) -> List[List[TimestampedMessage]]:
    """Split messages into chunks of approximately target_tokens.

    Splits at message pair boundaries (user + assistant) to avoid
    cutting a conversation turn in half.

    Returns list of message lists.
    """
    if not messages:
        return []

    total = sum(m.tokens for m in messages)
    if total <= target_tokens:
        return [messages]

    chunks = []
    current_chunk = []
    current_tokens = 0

    i = 0
    while i < len(messages):
        msg = messages[i]

        # Try to keep user+assistant pairs together
        pair = [msg]
        pair_tokens = msg.tokens
        if (msg.role == "user" and i + 1 < len(messages)
                and messages[i + 1].role == "assistant"):
            pair.append(messages[i + 1])
            pair_tokens += messages[i + 1].tokens

        # Would adding this pair exceed the target?
        if current_tokens > 0 and current_tokens + pair_tokens > target_tokens:
            # Flush current chunk
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0

        current_chunk.extend(pair)
        current_tokens += pair_tokens
        i += len(pair)

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# ---------------------------------------------------------------------------
# Topic-based splitting (LLM)
# ---------------------------------------------------------------------------

def split_by_topic(
    messages: List[TimestampedMessage],
    target_tokens: int = 20000,
    min_chunk_tokens: int = 3000,
    max_chunk_tokens: int = 30000,
) -> List[List[TimestampedMessage]]:
    """Use LLM to find conversation topic switching points.

    Falls back to size-based splitting if:
    - LLM finds chunks that are too large (> max_chunk_tokens)
    - LLM finds chunks that are too small (< min_chunk_tokens)
    - LLM call fails

    Returns list of message lists.
    """
    if not messages:
        return []

    total = sum(m.tokens for m in messages)
    if total <= target_tokens:
        return [messages]

    # Build a condensed view of the conversation for the LLM
    condensed = _build_condensed_view(messages)

    # Ask LLM to find topic switch points
    try:
        split_indices = _llm_find_splits(condensed, messages)
    except Exception as e:
        print(f"  [topic-split] LLM failed ({e}), falling back to size-based")
        return split_by_size(messages, target_tokens)

    if not split_indices:
        return split_by_size(messages, target_tokens)

    # Build chunks from split points
    chunks = []
    prev = 0
    for idx in sorted(split_indices):
        if prev < idx <= len(messages):
            chunks.append(messages[prev:idx])
            prev = idx
    if prev < len(messages):
        chunks.append(messages[prev:])

    # Validate chunk sizes — merge tiny chunks, split huge ones
    validated = _validate_topic_chunks(chunks, min_chunk_tokens, max_chunk_tokens, target_tokens)
    return validated


def _build_condensed_view(messages: List[TimestampedMessage]) -> str:
    """Build a condensed conversation view for topic detection.

    Shows message index, role, and first ~80 chars of content.
    """
    lines = []
    for i, m in enumerate(messages):
        preview = m.content[:80].replace("\n", " ")
        lines.append(f"[{i}] {m.role}: {preview}...")
    return "\n".join(lines)


def _llm_find_splits(condensed: str, messages: List[TimestampedMessage]) -> List[int]:
    """Call LLM to find natural topic switching points.

    Returns list of message indices where splits should occur.
    """
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package not installed")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        # Try .env file
        env_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break
    if not api_key:
        raise RuntimeError("No ANTHROPIC_API_KEY found")

    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""You are analyzing a conversation to find natural topic switching points.
Each line shows [message_index] role: content_preview.

Find the message indices where the conversation shifts to a significantly different topic.
A topic switch is when the user starts talking about something clearly different —
NOT minor sub-topic changes within the same discussion.

Good splits: "recipe app" → "running training", "job situation" → "mom's health"
Bad splits: "recipe app features" → "recipe app bugs" (same topic, different aspect)

Conversation:
{condensed}

Return ONLY a JSON array of message indices where splits should occur.
Example: [12, 28, 45]
If no clear topic switches exist, return: []"""

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    # Extract JSON array from response
    match = re.search(r'\[[\d,\s]*\]', text)
    if match:
        indices = json.loads(match.group())
        # Filter to valid indices
        return [i for i in indices if 0 < i < len(messages)]
    return []


def _validate_topic_chunks(
    chunks: List[List[TimestampedMessage]],
    min_tokens: int,
    max_tokens: int,
    target_tokens: int,
) -> List[List[TimestampedMessage]]:
    """Validate and fix topic-detected chunks.

    - Merge consecutive tiny chunks (< min_tokens)
    - Split huge chunks (> max_tokens) by size
    """
    if not chunks:
        return chunks

    # First pass: merge tiny chunks with their neighbor
    merged = []
    buffer = []
    buffer_tokens = 0

    for chunk in chunks:
        chunk_tokens = sum(m.tokens for m in chunk)

        if buffer_tokens + chunk_tokens < min_tokens:
            # Too small — merge with buffer
            buffer.extend(chunk)
            buffer_tokens += chunk_tokens
        else:
            if buffer:
                # Flush buffer into this chunk
                chunk = buffer + chunk
                chunk_tokens = buffer_tokens + chunk_tokens
                buffer = []
                buffer_tokens = 0
            merged.append(chunk)

    # Don't leave orphan buffer — merge into last chunk
    if buffer:
        if merged:
            merged[-1].extend(buffer)
        else:
            merged.append(buffer)

    # Second pass: split oversized chunks
    result = []
    for chunk in merged:
        chunk_tokens = sum(m.tokens for m in chunk)
        if chunk_tokens > max_tokens:
            sub_chunks = split_by_size(chunk, target_tokens)
            result.extend(sub_chunks)
        else:
            result.append(chunk)

    return result


# ---------------------------------------------------------------------------
# Main splitter
# ---------------------------------------------------------------------------

class SessionSplitter:
    """Splits a chronological message stream into extraction chunks.

    Combines timeout detection with buffer size management.
    """

    def __init__(
        self,
        timeout_minutes: int = 120,
        max_buffer_tokens: int = 30000,
        target_chunk_tokens: int = 20000,
        min_chunk_tokens: int = 3000,
        split_mode: str = "size",  # "size" or "topic"
        janitor_at_day_boundary: bool = True,
    ):
        self.timeout_ms = timeout_minutes * 60 * 1000
        self.max_buffer_tokens = max_buffer_tokens
        self.target_chunk_tokens = target_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.split_mode = split_mode
        self.janitor_at_day_boundary = janitor_at_day_boundary

    def split(
        self,
        messages: List[TimestampedMessage],
    ) -> List[ExtractionChunk]:
        """Process a chronological stream of timestamped messages.

        Returns a list of ExtractionChunks in chronological order.
        Each chunk should be extracted separately. Day boundary markers
        are also emitted for janitor scheduling.
        """
        if not messages:
            return []

        chunks = []
        buffer: List[TimestampedMessage] = []
        buffer_tokens = 0

        for i, msg in enumerate(messages):
            # Check for timeout gap
            if buffer and (msg.timestamp_ms - buffer[-1].timestamp_ms) > self.timeout_ms:
                # Flush buffer — timeout triggered
                new_chunks = self._flush_buffer(buffer, buffer_tokens, trigger="timeout")
                chunks.extend(new_chunks)
                buffer = []
                buffer_tokens = 0

            # Check for day boundary (for janitor scheduling)
            if self.janitor_at_day_boundary and buffer:
                prev_day = self._ms_to_day(buffer[-1].timestamp_ms)
                curr_day = self._ms_to_day(msg.timestamp_ms)
                if prev_day != curr_day:
                    # Day changed — flush and mark as day boundary
                    new_chunks = self._flush_buffer(buffer, buffer_tokens, trigger="day")
                    chunks.extend(new_chunks)
                    buffer = []
                    buffer_tokens = 0

            # Add to buffer
            buffer.append(msg)
            buffer_tokens += msg.tokens

            # Check if buffer exceeds max size — split immediately
            if buffer_tokens > self.max_buffer_tokens:
                new_chunks = self._flush_buffer(buffer, buffer_tokens, trigger="size")
                # Keep the overflow in the buffer if the last chunk was partial
                # Actually, flush everything — split_buffer handles chunking
                chunks.extend(new_chunks)
                buffer = []
                buffer_tokens = 0

        # Flush remaining buffer
        if buffer:
            new_chunks = self._flush_buffer(buffer, buffer_tokens, trigger="end")
            chunks.extend(new_chunks)

        return chunks

    def _flush_buffer(
        self,
        buffer: List[TimestampedMessage],
        buffer_tokens: int,
        trigger: str,
    ) -> List[ExtractionChunk]:
        """Convert a buffer into one or more ExtractionChunks."""
        if not buffer:
            return []

        # If buffer is small enough, return as single chunk
        if buffer_tokens <= self.max_buffer_tokens:
            return [ExtractionChunk(
                messages=list(buffer),
                trigger=trigger,
                chunk_index=0,
                split_total=1,
            )]

        # Buffer is too large — split it
        if self.split_mode == "topic":
            sub_lists = split_by_topic(
                buffer,
                target_tokens=self.target_chunk_tokens,
                min_chunk_tokens=self.min_chunk_tokens,
                max_chunk_tokens=self.max_buffer_tokens,
            )
        else:
            sub_lists = split_by_size(buffer, self.target_chunk_tokens)

        return [
            ExtractionChunk(
                messages=sub,
                trigger=trigger,
                chunk_index=ci,
                split_total=len(sub_lists),
            )
            for ci, sub in enumerate(sub_lists)
        ]

    @staticmethod
    def _ms_to_day(ms: int) -> str:
        from datetime import datetime, timezone
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Build timestamped messages from dataset + annotations
# ---------------------------------------------------------------------------

def build_message_stream(
    timestamps_path: Path,
    scale: str = "L",
) -> List[TimestampedMessage]:
    """Load sessions and merge with timestamp annotations.

    Returns a flat, chronologically sorted list of TimestampedMessages.
    """
    from dataset import (
        SESSION_DATES, FILLER_DATES,
        load_all_reviews, load_filler_reviews,
    )
    from injector import transcript_to_messages

    _eval_dir = Path(__file__).resolve().parent
    _project_dir = _eval_dir.parent
    assets_dir = _project_dir / "data" / "sessions"
    filler_dir = _project_dir / "data" / "filler-sessions"

    arcs = list(load_all_reviews(assets_dir))
    if scale == "L":
        fillers = list(load_filler_reviews(filler_dir))
        all_sessions = arcs + fillers
    else:
        all_sessions = arcs

    # Load timestamp annotations
    ts_data = json.load(open(timestamps_path))

    # Build session_id → review mapping
    review_map = {}
    for r in all_sessions:
        if r.session_num > 0:
            sid = f"S{r.session_num:02d}"
        else:
            sid = f"F{abs(r.session_num):03d}"
        review_map[sid] = r

    # Build timestamped message stream
    all_messages = []
    for sid, ts_msgs in ts_data.items():
        review = review_map.get(sid)
        if not review:
            continue

        messages = transcript_to_messages(review)
        if len(messages) != len(ts_msgs):
            # Mismatch — use timestamps for as many as we have
            pass

        for mi, msg in enumerate(messages):
            if mi < len(ts_msgs):
                ts_ms = ts_msgs[mi]["timestamp_ms"]
            else:
                # Fallback: increment from last known
                ts_ms = all_messages[-1].timestamp_ms + 15000 if all_messages else 0

            all_messages.append(TimestampedMessage(
                role=msg["role"],
                content=msg["content"],
                timestamp_ms=ts_ms,
                session_id=sid,
            ))

    # Sort chronologically
    all_messages.sort(key=lambda m: m.timestamp_ms)
    return all_messages


# ---------------------------------------------------------------------------
# CLI — analyze splitting strategies
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze session splitting strategies")
    parser.add_argument("--scale", choices=["L", "S"], default="L")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout in minutes")
    parser.add_argument("--max-buffer", type=int, default=30000, help="Max buffer tokens before split")
    parser.add_argument("--target-chunk", type=int, default=20000, help="Target chunk size in tokens")
    parser.add_argument("--split-mode", choices=["size", "topic"], default="size")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    ts_path = Path(__file__).parent.parent / "data" / f"timestamps-{args.scale}.json"
    if not ts_path.exists():
        print(f"Timestamps file not found: {ts_path}")
        print(f"Run: python3 annotate_timestamps.py --scale {args.scale}")
        return

    print(f"Loading message stream ({args.scale} scale)...")
    messages = build_message_stream(ts_path, args.scale)
    total_tokens = sum(m.tokens for m in messages)
    print(f"  {len(messages)} messages, {total_tokens:,} tokens")

    splitter = SessionSplitter(
        timeout_minutes=args.timeout,
        max_buffer_tokens=args.max_buffer,
        target_chunk_tokens=args.target_chunk,
        split_mode=args.split_mode,
    )

    print(f"\nSplitting with: timeout={args.timeout}min, max_buffer={args.max_buffer}, "
          f"target_chunk={args.target_chunk}, mode={args.split_mode}")

    chunks = splitter.split(messages)

    # Analyze results
    print(f"\n{'='*60}")
    print(f" Splitting Results")
    print(f"{'='*60}")
    print(f"  Total chunks: {len(chunks)}")

    from collections import Counter
    triggers = Counter(c.trigger for c in chunks)
    print(f"  By trigger: {dict(triggers)}")

    token_sizes = [c.total_tokens for c in chunks]
    if token_sizes:
        import statistics
        print(f"  Chunk tokens: min={min(token_sizes):,}, max={max(token_sizes):,}, "
              f"median={statistics.median(token_sizes):,.0f}, mean={statistics.mean(token_sizes):,.0f}")

    # How many chunks needed splitting?
    multi_split = [c for c in chunks if c.split_total > 1]
    print(f"  Chunks from splits: {len(multi_split)} (buffers that exceeded {args.max_buffer:,} tokens)")

    # Day-level view
    if args.verbose:
        print(f"\n{'='*60}")
        print(f" Day-by-day chunk detail")
        print(f"{'='*60}")
        current_day = None
        for ci, chunk in enumerate(chunks):
            day = splitter._ms_to_day(chunk.messages[0].timestamp_ms)
            if day != current_day:
                print(f"\n  {day}:")
                current_day = day

            from datetime import datetime, timezone
            t0 = datetime.fromtimestamp(chunk.timestamp_range[0] / 1000, tz=timezone.utc)
            t1 = datetime.fromtimestamp(chunk.timestamp_range[1] / 1000, tz=timezone.utc)
            sessions = ", ".join(chunk.session_ids)
            split_label = f" [{chunk.chunk_index+1}/{chunk.split_total}]" if chunk.split_total > 1 else ""
            print(f"    chunk {ci:>3}: {t0.strftime('%H:%M')}-{t1.strftime('%H:%M')} "
                  f"{chunk.total_tokens:>5} tok, {len(chunk.messages):>3} msgs, "
                  f"trigger={chunk.trigger:<8} sessions=[{sessions}]{split_label}")

    # Compare with per-day extraction
    print(f"\n{'='*60}")
    print(f" Comparison with per-day extraction")
    print(f"{'='*60}")
    per_day_chunks = SessionSplitter(
        timeout_minutes=999999,  # Never timeout
        max_buffer_tokens=999999,  # Never split
        janitor_at_day_boundary=True,
    ).split(messages)
    print(f"  Per-day extraction: {len(per_day_chunks)} chunks")
    print(f"  Timeout+size extraction: {len(chunks)} chunks")
    print(f"  Difference: {len(chunks) - len(per_day_chunks):+d} chunks "
          f"({len(chunks)/max(1,len(per_day_chunks)):.1f}x)")


if __name__ == "__main__":
    main()
