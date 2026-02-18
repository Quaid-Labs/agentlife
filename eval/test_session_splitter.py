#!/usr/bin/env python3
"""Tests for session_splitter.py and annotate_timestamps.py."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from session_splitter import (
    TimestampedMessage,
    ExtractionChunk,
    SessionSplitter,
    split_by_size,
    build_message_stream,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ms(dt_str: str) -> int:
    """Convert 'YYYY-MM-DD HH:MM' to epoch ms."""
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _msg(role: str, content: str, ts_str: str, session_id: str = "S01") -> TimestampedMessage:
    return TimestampedMessage(
        role=role,
        content=content,
        timestamp_ms=_ms(ts_str),
        session_id=session_id,
    )


def _quick_stream(specs: list) -> list[TimestampedMessage]:
    """Build a message stream from [(role, content, 'HH:MM', session_id), ...].

    All on 2026-03-01 unless otherwise specified.
    """
    msgs = []
    for spec in specs:
        role, content, time_str, sid = spec[0], spec[1], spec[2], spec[3] if len(spec) > 3 else "S01"
        if len(time_str) <= 5:
            time_str = f"2026-03-01 {time_str}"
        msgs.append(_msg(role, content, time_str, sid))
    return msgs


# ---------------------------------------------------------------------------
# Tests: split_by_size
# ---------------------------------------------------------------------------

def test_split_by_size_small_buffer():
    """Buffer under target → single chunk."""
    msgs = [
        _msg("user", "hello", "2026-03-01 09:00"),
        _msg("assistant", "hi there", "2026-03-01 09:01"),
    ]
    chunks = split_by_size(msgs, target_tokens=20000)
    assert len(chunks) == 1
    assert chunks[0] == msgs


def test_split_by_size_splits_at_pair_boundary():
    """Large buffer splits at user+assistant pair boundaries."""
    # Create messages where each pair is ~50 tokens
    msgs = []
    for i in range(40):
        msgs.append(_msg("user", f"Question {i}: " + "word " * 20, f"2026-03-01 09:{i:02d}"))
        msgs.append(_msg("assistant", f"Answer {i}: " + "word " * 20, f"2026-03-01 09:{i:02d}"))

    # Target 500 tokens per chunk — should split into multiple chunks
    chunks = split_by_size(msgs, target_tokens=500)
    assert len(chunks) > 1

    # Every chunk should start with a user message (pair boundary)
    for chunk in chunks:
        assert chunk[0].role == "user", f"Chunk starts with {chunk[0].role}, expected user"

    # All messages accounted for
    total = sum(len(c) for c in chunks)
    assert total == len(msgs)


def test_split_by_size_empty():
    """Empty input → empty output."""
    assert split_by_size([], target_tokens=20000) == []


# ---------------------------------------------------------------------------
# Tests: SessionSplitter — timeout
# ---------------------------------------------------------------------------

def test_timeout_triggers_on_gap():
    """2hr+ gap between messages triggers extraction."""
    msgs = _quick_stream([
        ("user", "hello", "09:00", "S01"),
        ("assistant", "hi", "09:01", "S01"),
        ("user", "question", "09:05", "S01"),
        ("assistant", "answer", "09:06", "S01"),
        # 3 hour gap
        ("user", "back again", "12:06", "S02"),
        ("assistant", "welcome back", "12:07", "S02"),
    ])

    splitter = SessionSplitter(timeout_minutes=120, janitor_at_day_boundary=False)
    chunks = splitter.split(msgs)

    assert len(chunks) == 2
    assert chunks[0].trigger == "timeout"
    assert len(chunks[0].messages) == 4  # First 4 messages
    assert chunks[1].trigger == "end"
    assert len(chunks[1].messages) == 2  # Last 2 messages


def test_no_timeout_within_window():
    """Messages within 2hr window stay in one chunk."""
    msgs = _quick_stream([
        ("user", "hello", "09:00"),
        ("assistant", "hi", "09:01"),
        ("user", "another", "10:00"),  # 59 min gap
        ("assistant", "reply", "10:01"),
        ("user", "more", "10:45"),     # 44 min gap
        ("assistant", "sure", "10:46"),
    ])

    splitter = SessionSplitter(timeout_minutes=120, janitor_at_day_boundary=False)
    chunks = splitter.split(msgs)

    assert len(chunks) == 1
    assert chunks[0].trigger == "end"
    assert len(chunks[0].messages) == 6


def test_multiple_timeouts():
    """Multiple 2hr+ gaps create multiple chunks."""
    msgs = _quick_stream([
        ("user", "morning", "08:00", "S01"),
        ("assistant", "hi", "08:01", "S01"),
        # 3hr gap
        ("user", "lunch", "11:01", "S02"),
        ("assistant", "hey", "11:02", "S02"),
        # 4hr gap
        ("user", "evening", "15:02", "S03"),
        ("assistant", "hello", "15:03", "S03"),
    ])

    splitter = SessionSplitter(timeout_minutes=120, janitor_at_day_boundary=False)
    chunks = splitter.split(msgs)

    assert len(chunks) == 3
    assert chunks[0].trigger == "timeout"
    assert chunks[1].trigger == "timeout"
    assert chunks[2].trigger == "end"
    assert chunks[0].session_ids == ["S01"]
    assert chunks[1].session_ids == ["S02"]
    assert chunks[2].session_ids == ["S03"]


# ---------------------------------------------------------------------------
# Tests: SessionSplitter — day boundary
# ---------------------------------------------------------------------------

def test_day_boundary_triggers_flush():
    """Messages crossing midnight trigger day-boundary extraction."""
    msgs = [
        _msg("user", "evening chat", "2026-03-01 22:00", "S01"),
        _msg("assistant", "hey", "2026-03-01 22:01", "S01"),
        _msg("user", "morning chat", "2026-03-02 08:00", "S02"),
        _msg("assistant", "good morning", "2026-03-02 08:01", "S02"),
    ]

    splitter = SessionSplitter(timeout_minutes=999999, janitor_at_day_boundary=True)
    chunks = splitter.split(msgs)

    assert len(chunks) == 2
    assert chunks[0].trigger == "day"
    assert chunks[1].trigger == "end"


def test_day_boundary_disabled():
    """With janitor_at_day_boundary=False, day change doesn't flush."""
    msgs = [
        _msg("user", "evening", "2026-03-01 22:00"),
        _msg("assistant", "hi", "2026-03-01 22:01"),
        _msg("user", "morning", "2026-03-02 08:00"),
        _msg("assistant", "hey", "2026-03-02 08:01"),
    ]

    splitter = SessionSplitter(timeout_minutes=999999, janitor_at_day_boundary=False)
    chunks = splitter.split(msgs)

    assert len(chunks) == 1


# ---------------------------------------------------------------------------
# Tests: SessionSplitter — size threshold
# ---------------------------------------------------------------------------

def test_size_threshold_triggers_split():
    """Buffer exceeding max_buffer_tokens gets split."""
    # Create a large buffer (~2000 tokens)
    msgs = []
    for i in range(50):
        msgs.append(_msg("user", "word " * 20, f"2026-03-01 09:{i:02d}", "S01"))
        msgs.append(_msg("assistant", "reply " * 20, f"2026-03-01 09:{i:02d}", "S01"))

    total_tokens = sum(m.tokens for m in msgs)

    splitter = SessionSplitter(
        timeout_minutes=999999,
        max_buffer_tokens=total_tokens // 3,  # Force ~3 chunks
        target_chunk_tokens=total_tokens // 4,
        janitor_at_day_boundary=False,
    )
    chunks = splitter.split(msgs)

    assert len(chunks) >= 2, f"Expected >=2 chunks, got {len(chunks)}"
    # At least one chunk should be from a size-triggered split
    size_triggered = [c for c in chunks if c.trigger == "size"]
    assert len(size_triggered) >= 1

    # All messages accounted for
    total_msgs = sum(len(c.messages) for c in chunks)
    assert total_msgs == len(msgs)


# ---------------------------------------------------------------------------
# Tests: ExtractionChunk properties
# ---------------------------------------------------------------------------

def test_chunk_properties():
    """ExtractionChunk computes tokens, time range, session IDs correctly."""
    msgs = _quick_stream([
        ("user", "hello world", "09:00", "S01"),
        ("assistant", "hi there friend", "09:01", "S01"),
        ("user", "another question here", "09:10", "S02"),
        ("assistant", "another answer here", "09:11", "S02"),
    ])

    chunk = ExtractionChunk(messages=msgs, trigger="timeout")

    assert chunk.total_tokens > 0
    assert chunk.timestamp_range[0] == msgs[0].timestamp_ms
    assert chunk.timestamp_range[1] == msgs[-1].timestamp_ms
    assert chunk.session_ids == ["S01", "S02"]


def test_chunk_session_ids_dedup():
    """Session IDs are deduplicated but preserve order."""
    msgs = _quick_stream([
        ("user", "a", "09:00", "S01"),
        ("assistant", "b", "09:01", "S01"),
        ("user", "c", "09:02", "S02"),
        ("assistant", "d", "09:03", "S02"),
        ("user", "e", "09:04", "S01"),  # Back to S01
        ("assistant", "f", "09:05", "S01"),
    ])

    chunk = ExtractionChunk(messages=msgs, trigger="end")
    assert chunk.session_ids == ["S01", "S02", "S01"]


# ---------------------------------------------------------------------------
# Tests: Timeout + day interaction
# ---------------------------------------------------------------------------

def test_timeout_fires_before_day_boundary():
    """When timeout and day boundary both apply, timeout fires first."""
    msgs = [
        _msg("user", "evening", "2026-03-01 20:00", "S01"),
        _msg("assistant", "hi", "2026-03-01 20:01", "S01"),
        # 12 hour gap (both timeout and day boundary)
        _msg("user", "morning", "2026-03-02 08:01", "S02"),
        _msg("assistant", "hey", "2026-03-02 08:02", "S02"),
    ]

    splitter = SessionSplitter(timeout_minutes=120, janitor_at_day_boundary=True)
    chunks = splitter.split(msgs)

    assert len(chunks) == 2
    # Timeout fires first (checked before day boundary in the loop)
    assert chunks[0].trigger == "timeout"


# ---------------------------------------------------------------------------
# Tests: Empty / edge cases
# ---------------------------------------------------------------------------

def test_empty_stream():
    """Empty message stream returns no chunks."""
    splitter = SessionSplitter()
    assert splitter.split([]) == []


def test_single_message():
    """Single message returns one chunk."""
    msgs = [_msg("user", "hello", "2026-03-01 09:00")]
    splitter = SessionSplitter(janitor_at_day_boundary=False)
    chunks = splitter.split(msgs)
    assert len(chunks) == 1
    assert len(chunks[0].messages) == 1


# ---------------------------------------------------------------------------
# Tests: annotate_timestamps
# ---------------------------------------------------------------------------

def test_timestamps_no_overlaps():
    """Verify no session timestamps overlap within the same day."""
    from annotate_timestamps import annotate

    for scale in ["S", "L"]:
        timestamps, stats = annotate(scale)

        # Group by day
        from collections import defaultdict
        day_sessions = defaultdict(list)
        for sid, msgs in timestamps.items():
            if not msgs:
                continue
            start = msgs[0]["timestamp_ms"]
            end = msgs[-1]["timestamp_ms"]
            dt = datetime.fromtimestamp(start / 1000, tz=timezone.utc)
            day = dt.strftime("%Y-%m-%d")
            day_sessions[day].append((sid, start, end))

        for day, sessions in day_sessions.items():
            sessions.sort(key=lambda x: x[1])
            for i in range(1, len(sessions)):
                prev_sid, _, prev_end = sessions[i-1]
                curr_sid, curr_start, _ = sessions[i]
                assert curr_start >= prev_end, (
                    f"{scale} {day}: {curr_sid} starts at {curr_start} "
                    f"before {prev_sid} ends at {prev_end}"
                )


def test_timestamps_within_waking_hours():
    """All timestamps fall within 7am-11pm UTC."""
    from annotate_timestamps import annotate

    timestamps, _ = annotate("S")
    for sid, msgs in timestamps.items():
        for m in msgs:
            dt = datetime.fromtimestamp(m["timestamp_ms"] / 1000, tz=timezone.utc)
            hour = dt.hour
            assert 7 <= hour <= 23, f"{sid}: message at {dt} outside waking hours"


def test_timestamps_monotonic_within_session():
    """Timestamps are strictly increasing within each session."""
    from annotate_timestamps import annotate

    for scale in ["S", "L"]:
        timestamps, _ = annotate(scale)
        for sid, msgs in timestamps.items():
            for i in range(1, len(msgs)):
                assert msgs[i]["timestamp_ms"] > msgs[i-1]["timestamp_ms"], (
                    f"{scale} {sid}: timestamp not monotonic at message {i}"
                )


def test_timestamps_deterministic():
    """Same seed produces identical timestamps."""
    from annotate_timestamps import annotate

    ts1, _ = annotate("S")
    ts2, _ = annotate("S")
    assert ts1 == ts2


# ---------------------------------------------------------------------------
# Tests: Integration — build_message_stream + splitter
# ---------------------------------------------------------------------------

def test_build_message_stream_loads():
    """build_message_stream loads data and returns sorted messages."""
    ts_path = Path(__file__).parent.parent / "data" / "timestamps-S.json"
    if not ts_path.exists():
        from annotate_timestamps import annotate
        import json
        timestamps, _ = annotate("S")
        ts_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ts_path, "w") as f:
            json.dump(timestamps, f)

    messages = build_message_stream(ts_path, "S")
    assert len(messages) > 0

    # Should be chronologically sorted
    for i in range(1, len(messages)):
        assert messages[i].timestamp_ms >= messages[i-1].timestamp_ms


def test_splitter_on_real_data_s():
    """Run splitter on real S-scale data and verify basic properties."""
    ts_path = Path(__file__).parent.parent / "data" / "timestamps-S.json"
    if not ts_path.exists():
        from annotate_timestamps import annotate
        import json
        timestamps, _ = annotate("S")
        ts_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ts_path, "w") as f:
            json.dump(timestamps, f)

    messages = build_message_stream(ts_path, "S")
    splitter = SessionSplitter(timeout_minutes=120)
    chunks = splitter.split(messages)

    assert len(chunks) > 0

    # All messages accounted for
    total_msgs = sum(len(c.messages) for c in chunks)
    assert total_msgs == len(messages), f"Lost messages: {total_msgs} vs {len(messages)}"

    # Chunks are chronologically ordered
    for i in range(1, len(chunks)):
        assert chunks[i].timestamp_range[0] >= chunks[i-1].timestamp_range[1]

    # Every chunk has at least one message
    for c in chunks:
        assert len(c.messages) > 0


def test_per_day_matches_day_count():
    """Per-day splitting should produce one chunk per day."""
    ts_path = Path(__file__).parent.parent / "data" / "timestamps-S.json"
    if not ts_path.exists():
        from annotate_timestamps import annotate
        import json
        timestamps, _ = annotate("S")
        ts_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ts_path, "w") as f:
            json.dump(timestamps, f)

    messages = build_message_stream(ts_path, "S")

    # Count unique days
    days = set()
    for m in messages:
        dt = datetime.fromtimestamp(m.timestamp_ms / 1000, tz=timezone.utc)
        days.add(dt.strftime("%Y-%m-%d"))

    splitter = SessionSplitter(
        timeout_minutes=999999,
        max_buffer_tokens=999999,
        janitor_at_day_boundary=True,
    )
    chunks = splitter.split(messages)

    assert len(chunks) == len(days), f"Expected {len(days)} day chunks, got {len(chunks)}"


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0

    for test in tests:
        name = test.__name__
        try:
            test()
            print(f"  PASS: {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name}")
            traceback.print_exc()
            failed += 1
            print()

    print(f"\n{'='*40}")
    print(f"  {passed} passed, {failed} failed, {passed + failed} total")
    if failed:
        sys.exit(1)
