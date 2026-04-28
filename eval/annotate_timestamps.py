#!/usr/bin/env python3
"""AgentLife — Realistic per-message timestamp annotation.

Generates timestamps for every message in every session, with realistic
clustering behavior that enables testing session-splitting strategies:

1. **Timeout-based**: 2hr inactivity gap triggers extraction
2. **Chunk-based**: Large accumulated buffers get split

Clustering model:
- Sessions on the same day are grouped into 1-4 clusters
- Intra-cluster gaps: 5-40 minutes (within 2hr timeout window)
- Inter-cluster gaps: 2-5 hours (triggers 2hr timeout)
- Waking hours: 7am-11pm CST (Maya is in Austin)

Per-message timing:
- User typing: ~3-5 tokens/sec + thinking pause
- Agent response: 10-30s (generation time)
- Between turns: user reads response, thinks, types

Output: data/timestamps-{L,S}.json
  {session_id: [{role, timestamp_ms, content_preview}, ...]}

Deterministic (seeded RNG) for reproducibility.
"""

import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dataset import (
    SESSION_DATES, FILLER_DATES,
    load_all_reviews, load_filler_reviews,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
WAKING_START_HOUR = 7   # 7am CST
WAKING_END_HOUR = 23    # 11pm CST
TIMEOUT_THRESHOLD_MIN = 120  # 2hr timeout for extraction trigger

# Intra-cluster gaps (minutes)
INTRA_CLUSTER_GAP_MIN = 5
INTRA_CLUSTER_GAP_MAX = 40

# Inter-cluster gaps (minutes) — must exceed TIMEOUT_THRESHOLD_MIN
INTER_CLUSTER_GAP_MIN = 130   # 2h10m
INTER_CLUSTER_GAP_MAX = 300   # 5h

# Per-message timing
USER_TYPING_SPEED_TPS = 4.0     # tokens per second
USER_THINK_BEFORE_TYPING_S = 8  # seconds of thought before typing
USER_READ_SPEED_TPS = 6.0       # tokens per second reading agent response
AGENT_LATENCY_BASE_S = 12       # base agent response time
AGENT_LATENCY_PER_TOKEN_S = 0.03  # additional per output token


# ---------------------------------------------------------------------------
# Clustering logic
# ---------------------------------------------------------------------------

def cluster_sessions(n_sessions: int, rng: random.Random) -> list[list[int]]:
    """Divide n sessions into clusters of realistic sizes.

    Returns list of clusters, each cluster is a list of session indices.
    Cluster sizes follow realistic patterns:
    - 1-2 sessions: 1 cluster (quick check-in)
    - 3-4 sessions: 1-2 clusters
    - 5-7 sessions: 2-3 clusters
    - 8-11 sessions: 3-4 clusters
    """
    if n_sessions <= 2:
        return [list(range(n_sessions))]

    if n_sessions <= 4:
        n_clusters = rng.choice([1, 2])
    elif n_sessions <= 7:
        n_clusters = rng.choice([2, 2, 3])
    else:
        n_clusters = rng.choice([3, 3, 4])

    # Distribute sessions across clusters
    clusters = [[] for _ in range(n_clusters)]
    indices = list(range(n_sessions))
    rng.shuffle(indices)

    for i, idx in enumerate(indices):
        clusters[i % n_clusters].append(idx)

    # Sort within each cluster
    for c in clusters:
        c.sort()

    return clusters


def _estimate_session_duration_min(review, rng: random.Random) -> float:
    """Estimate how long a session takes in minutes.

    Must be generous enough that assign_session_start_times never places
    the next session before this one's messages finish.  Actual per-turn
    timing (think + type + agent latency + read) averages ~2-4 min/turn
    for long agent responses.
    """
    turns = review.transcript_turns if hasattr(review, 'transcript_turns') else []
    if not turns:
        return 2

    total_s = 0.0
    for turn in turns:
        user_tok = turn.get("tokens", 20)
        agent_text = turn.get("agent", "")
        agent_tok = max(20, len(agent_text) // 4) if agent_text else 20

        # Match the constants used in generate_message_timestamps
        think = USER_THINK_BEFORE_TYPING_S
        type_s = user_tok / USER_TYPING_SPEED_TPS
        agent_s = AGENT_LATENCY_BASE_S + agent_tok * AGENT_LATENCY_PER_TOKEN_S
        read_s = agent_tok / USER_READ_SPEED_TPS
        total_s += think + type_s + agent_s + read_s

    # Add 20% buffer for variance
    return max(1, total_s * 1.2 / 60)


def assign_session_start_times(
    day_str: str,
    sessions: list[tuple[str, object]],
    clusters: list[list[int]],
    rng: random.Random,
) -> list[datetime]:
    """Assign start times for each session on a given day.

    Sessions within a cluster start after the previous one ends (no overlaps).
    Inter-cluster gaps exceed the 2hr timeout threshold.

    Returns list of datetime objects indexed by session position.
    """
    day = datetime.strptime(day_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    waking_start = day + timedelta(hours=WAKING_START_HOUR)
    waking_end = day + timedelta(hours=WAKING_END_HOUR)
    waking_minutes = (WAKING_END_HOUR - WAKING_START_HOUR) * 60
    n = len(sessions)

    start_times = [None] * n

    if len(clusters) == 1:
        cluster = clusters[0]
        max_start_offset = max(30, waking_minutes - len(cluster) * 30)
        cluster_start_offset = rng.randint(30, min(max_start_offset, waking_minutes - 60))
        t = waking_start + timedelta(minutes=cluster_start_offset)

        for idx in cluster:
            start_times[idx] = t
            # Session duration + gap before next session
            duration = _estimate_session_duration_min(sessions[idx][1], rng)
            gap = rng.randint(INTRA_CLUSTER_GAP_MIN, INTRA_CLUSTER_GAP_MAX)
            t = t + timedelta(minutes=duration + gap)
    else:
        # Start first cluster early-ish in the day
        first_start_offset = rng.randint(15, 90)
        t = waking_start + timedelta(minutes=first_start_offset)

        for ci, cluster in enumerate(clusters):
            for si, idx in enumerate(cluster):
                start_times[idx] = t
                duration = _estimate_session_duration_min(sessions[idx][1], rng)
                if si < len(cluster) - 1:
                    gap = rng.randint(INTRA_CLUSTER_GAP_MIN, INTRA_CLUSTER_GAP_MAX)
                    t = t + timedelta(minutes=duration + gap)
                else:
                    t = t + timedelta(minutes=duration)

            # Inter-cluster gap
            if ci < len(clusters) - 1:
                gap = rng.randint(INTER_CLUSTER_GAP_MIN, INTER_CLUSTER_GAP_MAX)
                t = t + timedelta(minutes=gap)

                # Clamp to waking hours
                if t > waking_end:
                    t = waking_end - timedelta(minutes=rng.randint(30, 60))

    return start_times


def generate_message_timestamps(
    session_start: datetime,
    turns: list[dict],
    rng: random.Random,
) -> list[dict]:
    """Generate per-message timestamps for a session.

    Each turn has {maya, agent, tokens, ...}.
    Returns list of {role, timestamp_ms, content_preview}.
    """
    messages = []
    t = session_start

    for turn in turns:
        maya_text = turn.get("maya", "")
        agent_text = turn.get("agent", "")
        user_tokens = turn.get("tokens", 20)
        agent_tokens = max(20, len(agent_text) // 4) if agent_text else 20

        if maya_text:
            # User thinks, then types
            think_time = USER_THINK_BEFORE_TYPING_S * rng.uniform(0.5, 1.5)
            type_time = user_tokens / USER_TYPING_SPEED_TPS * rng.uniform(0.8, 1.3)
            if turn.get("num", 1) == 1:
                # First message — no reading delay, just natural start
                think_time = rng.uniform(0, 3)
            t = t + timedelta(seconds=think_time + type_time)

            messages.append({
                "role": "user",
                "timestamp_ms": int(t.timestamp() * 1000),
                "content_preview": maya_text[:60],
            })

        if agent_text:
            # Agent generates response
            latency = AGENT_LATENCY_BASE_S + agent_tokens * AGENT_LATENCY_PER_TOKEN_S
            latency *= rng.uniform(0.7, 1.4)
            t = t + timedelta(seconds=latency)

            messages.append({
                "role": "assistant",
                "timestamp_ms": int(t.timestamp() * 1000),
                "content_preview": agent_text[:60],
            })

            # User reads agent response before next turn
            if turn != turns[-1]:
                read_time = agent_tokens / USER_READ_SPEED_TPS * rng.uniform(0.6, 1.5)
                t = t + timedelta(seconds=read_time)

    return messages


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def annotate(scale: str = "L") -> dict:
    """Generate timestamps for all sessions at the given scale.

    Args:
        scale: "L" (279 sessions) or "S" (20 arc sessions only)

    Returns:
        Dict of {session_id: [message timestamps]}
    """
    rng = random.Random(SEED)

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
        fillers = []

    # Build day -> session list
    day_map: dict[str, list[tuple[str, object]]] = defaultdict(list)
    for r in all_sessions:
        if r.session_num > 0:
            day = SESSION_DATES.get(r.session_num, "2026-03-01")
            sid = f"S{r.session_num:02d}"
        else:
            fid = f"F{abs(r.session_num):03d}"
            day = FILLER_DATES.get(fid, "2026-03-15").split(" ")[0]
            sid = fid

        day_map[day].append((sid, r))

    # Sort sessions within each day by their original timestamp (fillers have times)
    for day in day_map:
        day_map[day].sort(key=lambda x: _session_sort_key(x[0], x[1]))

    result = {}
    stats = {
        "total_sessions": 0,
        "total_messages": 0,
        "total_clusters": 0,
        "timeout_triggers": 0,  # inter-cluster gaps > 2hrs
        "days": len(day_map),
    }

    for day in sorted(day_map.keys()):
        sessions = day_map[day]
        n = len(sessions)
        clusters = cluster_sessions(n, rng)
        start_times = assign_session_start_times(day, sessions, clusters, rng)

        stats["total_clusters"] += len(clusters)
        stats["timeout_triggers"] += max(0, len(clusters) - 1)

        for i, (sid, review) in enumerate(sessions):
            session_start = start_times[i]
            if session_start is None:
                # Fallback
                session_start = datetime.strptime(day, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc, hour=9
                )

            msgs = generate_message_timestamps(session_start, review.transcript_turns, rng)
            result[sid] = msgs
            stats["total_sessions"] += 1
            stats["total_messages"] += len(msgs)

    return result, stats


def _session_sort_key(sid: str, review) -> str:
    """Sort key: use timestamp if available, otherwise session number."""
    if review.timestamp:
        return review.timestamp
    if sid.startswith("S"):
        num = int(sid[1:])
        return SESSION_DATES.get(num, "2026-03-01") + " 09:00:00"
    return "2026-03-15 12:00:00"


def print_day_summary(timestamps: dict, scale: str):
    """Print a summary showing cluster structure per day."""
    from collections import defaultdict
    day_msgs = defaultdict(list)

    for sid, msgs in timestamps.items():
        if not msgs:
            continue
        first_ts = msgs[0]["timestamp_ms"]
        dt = datetime.fromtimestamp(first_ts / 1000, tz=timezone.utc)
        day = dt.strftime("%Y-%m-%d")
        last_ts = msgs[-1]["timestamp_ms"]
        day_msgs[day].append((sid, first_ts, last_ts, len(msgs)))

    print(f"\n{'='*70}")
    print(f" Timestamp Annotation Summary — AgentLife {scale}")
    print(f"{'='*70}\n")

    total_timeouts = 0
    for day in sorted(day_msgs.keys()):
        sessions = sorted(day_msgs[day], key=lambda x: x[1])
        print(f"  {day} ({len(sessions)} sessions):")

        prev_end = None
        for sid, start, end, n_msgs in sessions:
            start_dt = datetime.fromtimestamp(start / 1000, tz=timezone.utc)
            end_dt = datetime.fromtimestamp(end / 1000, tz=timezone.utc)
            duration = (end - start) / 1000 / 60  # minutes

            gap_str = ""
            if prev_end is not None:
                gap_min = (start - prev_end) / 1000 / 60
                if gap_min >= TIMEOUT_THRESHOLD_MIN:
                    gap_str = f"  [GAP {gap_min:.0f}min — TIMEOUT]"
                    total_timeouts += 1
                else:
                    gap_str = f"  [gap {gap_min:.0f}min]"

            print(f"    {sid:>5}: {start_dt.strftime('%H:%M')}-{end_dt.strftime('%H:%M')} "
                  f"({duration:.0f}min, {n_msgs} msgs){gap_str}")
            prev_end = end
        print()

    print(f"  Total timeout triggers: {total_timeouts}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Annotate AgentLife sessions with timestamps")
    parser.add_argument("--scale", choices=["L", "S", "both"], default="both",
                        help="Which dataset scale to annotate")
    parser.add_argument("--summary", action="store_true",
                        help="Print day-by-day summary")
    parser.add_argument("--stats-only", action="store_true",
                        help="Print stats without writing files")
    args = parser.parse_args()

    scales = ["L", "S"] if args.scale == "both" else [args.scale]

    for scale in scales:
        timestamps, stats = annotate(scale)

        print(f"\n=== AgentLife {scale} Timestamp Stats ===")
        print(f"  Days: {stats['days']}")
        print(f"  Sessions: {stats['total_sessions']}")
        print(f"  Messages: {stats['total_messages']}")
        print(f"  Clusters: {stats['total_clusters']}")
        print(f"  Timeout triggers (>2hr gaps): {stats['timeout_triggers']}")

        if args.summary:
            print_day_summary(timestamps, scale)

        if not args.stats_only:
            out_path = Path(__file__).parent.parent / "data" / f"timestamps-{scale}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(timestamps, f, indent=2)
            print(f"  Written: {out_path}")


if __name__ == "__main__":
    main()
