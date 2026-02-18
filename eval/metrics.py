#!/usr/bin/env python3
"""AgentLife Benchmark — Metrics and scoring.

Computes per-query-type accuracy, cost estimates, and generates reports.
"""

import json
from collections import defaultdict
from typing import Dict, List

# ---------------------------------------------------------------------------
# Claude API pricing (per 1K tokens, as of Feb 2026)
# ---------------------------------------------------------------------------

PRICING = {
    "sonnet": {"input": 0.003, "output": 0.015},
    "haiku": {"input": 0.0008, "output": 0.004},
    "opus": {"input": 0.015, "output": 0.075},
    # GPT-4o-mini for judge (if used)
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_results(results: List[dict]) -> dict:
    """Compute per-query-type and overall accuracy.

    Returns:
        {
            "overall": {"count": N, "accuracy": 0.X, "correct": N, "partial": N, "wrong": N},
            "per_type": {"factual_recall": {"count": N, "accuracy": 0.X, ...}, ...},
            "per_difficulty": {"Easy": {...}, "Medium": {...}, "Hard": {...}},
            "per_session": {1: {...}, 2: {...}, ...},
        }
    """
    # Overall
    overall = _score_group(results)

    # Per query type
    by_type = defaultdict(list)
    for r in results:
        qt = r.get("query_type", "unknown")
        # Normalize: remove parenthetical notes
        qt = qt.split("(")[0].strip()
        by_type[qt].append(r)

    per_type = {qt: _score_group(items) for qt, items in sorted(by_type.items())}

    # Per difficulty
    by_diff = defaultdict(list)
    for r in results:
        rd = r.get("recall_difficulty", "unknown")
        # Extract just the difficulty level
        if rd.startswith("Easy"):
            rd = "Easy"
        elif rd.startswith("Medium"):
            rd = "Medium"
        elif rd.startswith("Hard"):
            rd = "Hard"
        by_diff[rd].append(r)

    per_difficulty = {d: _score_group(items) for d, items in sorted(by_diff.items())}

    # Per source session
    by_session = defaultdict(list)
    for r in results:
        s = r.get("source_session", 0)
        by_session[s].append(r)

    per_session = {s: _score_group(items) for s, items in sorted(by_session.items())}

    # Retrieval-only accuracy (if available)
    ret_results = [r for r in results if r.get("retrieval_label") in ("CORRECT", "PARTIAL", "WRONG")]
    retrieval = _score_group_retrieval(ret_results) if ret_results else None

    return {
        "overall": overall,
        "per_type": per_type,
        "per_difficulty": per_difficulty,
        "per_session": per_session,
        "retrieval": retrieval,
    }


def _score_group_retrieval(results: List[dict]) -> dict:
    """Score retrieval-only results (uses retrieval_label instead of judge_label)."""
    if not results:
        return {"count": 0, "accuracy": 0.0, "correct": 0, "partial": 0, "wrong": 0}

    correct = sum(1 for r in results if r.get("retrieval_label") == "CORRECT")
    partial = sum(1 for r in results if r.get("retrieval_label") == "PARTIAL")
    wrong = sum(1 for r in results if r.get("retrieval_label") == "WRONG")

    scored = correct + partial + wrong
    accuracy = (correct + 0.5 * partial) / scored if scored > 0 else 0.0

    return {
        "count": scored,
        "accuracy": round(accuracy * 100, 2),
        "correct": correct,
        "partial": partial,
        "wrong": wrong,
    }


def _score_group(results: List[dict]) -> dict:
    """Score a group of results."""
    if not results:
        return {"count": 0, "accuracy": 0.0, "correct": 0, "partial": 0, "wrong": 0, "error": 0}

    correct = sum(1 for r in results if r.get("judge_label") == "CORRECT")
    partial = sum(1 for r in results if r.get("judge_label") == "PARTIAL")
    wrong = sum(1 for r in results if r.get("judge_label") == "WRONG")
    error = sum(1 for r in results if r.get("judge_label") == "ERROR")

    scored = correct + partial + wrong
    accuracy = (correct + 0.5 * partial) / scored if scored > 0 else 0.0

    return {
        "count": len(results),
        "scored": scored,
        "accuracy": round(accuracy * 100, 2),
        "correct": correct,
        "partial": partial,
        "wrong": wrong,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

def estimate_cost(ingest_stats: dict, eval_stats: dict, config: dict = None) -> dict:
    """Estimate API cost based on token counts.

    Args:
        ingest_stats: Token stats from ingestion
        eval_stats: Token stats from evaluation
        config: Optional config with model names

    Returns:
        Cost breakdown dict.
    """
    config = config or {}
    extract_model = config.get("extract_model", "sonnet")
    answer_model = config.get("answer_model", "haiku")
    judge_model = config.get("judge_model", "haiku")

    # Extraction cost
    ext_input = ingest_stats.get("extraction_input_tokens_est", 0)
    ext_output = ingest_stats.get("extraction_output_tokens_est", 0)
    ext_price = PRICING.get(extract_model, PRICING["sonnet"])
    extraction_cost = (ext_input / 1000 * ext_price["input"] +
                       ext_output / 1000 * ext_price["output"])

    # Answer cost
    ans_input = eval_stats.get("answer_input_tokens_est", 0)
    ans_output = eval_stats.get("answer_output_tokens_est", 0)
    ans_price = PRICING.get(answer_model, PRICING["haiku"])
    answer_cost = (ans_input / 1000 * ans_price["input"] +
                   ans_output / 1000 * ans_price["output"])

    # Judge cost
    judge_calls = eval_stats.get("judge_calls", 0)
    # Estimate ~500 tokens input, ~5 tokens output per judge call
    judge_price = PRICING.get(judge_model, PRICING["haiku"])
    judge_cost = judge_calls * (0.5 * judge_price["input"] + 0.005 * judge_price["output"])

    # Context token stats
    context_tokens = eval_stats.get("context_tokens_total", 0)

    total_cost = extraction_cost + answer_cost + judge_cost

    return {
        "extraction_cost": round(extraction_cost, 4),
        "answer_cost": round(answer_cost, 4),
        "judge_cost": round(judge_cost, 4),
        "total_cost": round(total_cost, 4),
        "extraction_tokens": {"input": ext_input, "output": ext_output},
        "answer_tokens": {"input": ans_input, "output": ans_output},
        "context_tokens_total": context_tokens,
        "avg_context_tokens_per_query": round(context_tokens / max(eval_stats.get("answer_calls", 1), 1)),
        "models": {
            "extraction": extract_model,
            "answer": answer_model,
            "judge": judge_model,
        },
    }


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def retrieval_metrics(results: List[dict]) -> dict:
    """Compute retrieval efficiency metrics."""
    if not results:
        return {}

    n_memories = [r.get("num_memories", 0) for r in results]
    n_project_docs = [r.get("num_project_docs", 0) for r in results]
    context_tokens = [r.get("context_tokens_est", 0) for r in results]
    recall_latencies = [r.get("recall_latency_ms", 0) for r in results]

    return {
        "avg_memories_per_query": round(sum(n_memories) / len(n_memories), 1),
        "avg_project_docs_per_query": round(sum(n_project_docs) / len(n_project_docs), 1),
        "avg_context_tokens": round(sum(context_tokens) / len(context_tokens)),
        "total_context_tokens": sum(context_tokens),
        "avg_recall_latency_ms": round(sum(recall_latencies) / len(recall_latencies), 1),
        "max_recall_latency_ms": round(max(recall_latencies), 1) if recall_latencies else 0,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def format_report(
    scores: dict,
    cost: dict,
    retrieval: dict,
    mode: str = "quaid",
    ingest_stats: dict = None,
) -> str:
    """Format a human-readable results report."""
    lines = []
    lines.append(f"{'=' * 60}")
    lines.append(f"AgentLife Benchmark Results — {mode.upper()}")
    lines.append(f"{'=' * 60}")

    # Overall
    o = scores["overall"]
    lines.append(f"\nOverall Accuracy: {o['accuracy']:.1f}%")
    lines.append(f"  Questions: {o['count']} ({o['scored']} scored)")
    lines.append(f"  Correct: {o['correct']} | Partial: {o['partial']} | Wrong: {o['wrong']} | Error: {o['error']}")

    # Per type
    lines.append(f"\n{'─' * 60}")
    lines.append(f"{'Query Type':<30} {'Count':>5} {'Accuracy':>8}")
    lines.append(f"{'─' * 60}")
    for qt, s in scores["per_type"].items():
        lines.append(f"{qt:<30} {s['count']:>5} {s['accuracy']:>7.1f}%")

    # Per difficulty
    lines.append(f"\n{'─' * 60}")
    lines.append(f"{'Difficulty':<30} {'Count':>5} {'Accuracy':>8}")
    lines.append(f"{'─' * 60}")
    for d, s in scores["per_difficulty"].items():
        lines.append(f"{d:<30} {s['count']:>5} {s['accuracy']:>7.1f}%")

    # Retrieval stats
    if retrieval:
        lines.append(f"\n{'─' * 60}")
        lines.append("Retrieval Metrics:")
        lines.append(f"  Avg memories/query: {retrieval.get('avg_memories_per_query', 0)}")
        lines.append(f"  Avg project docs/query: {retrieval.get('avg_project_docs_per_query', 0)}")
        lines.append(f"  Avg context tokens: {retrieval.get('avg_context_tokens', 0):,}")
        lines.append(f"  Avg recall latency: {retrieval.get('avg_recall_latency_ms', 0):.0f}ms")

    # Cost
    if cost:
        lines.append(f"\n{'─' * 60}")
        lines.append("Estimated API Cost:")
        lines.append(f"  Extraction ({cost['models']['extraction']}): ${cost['extraction_cost']:.4f}")
        lines.append(f"  Answers ({cost['models']['answer']}): ${cost['answer_cost']:.4f}")
        lines.append(f"  Judge ({cost['models']['judge']}): ${cost['judge_cost']:.4f}")
        lines.append(f"  Total: ${cost['total_cost']:.4f}")
        lines.append(f"  Avg context tokens/query: {cost.get('avg_context_tokens_per_query', 0):,}")

    # Ingestion stats
    if ingest_stats:
        lines.append(f"\n{'─' * 60}")
        lines.append("Ingestion Stats:")
        lines.append(f"  Sessions processed: {ingest_stats.get('sessions_processed', 0)}")
        lines.append(f"  Facts stored: {ingest_stats.get('facts_stored', 0)}")
        lines.append(f"  Edges created: {ingest_stats.get('edges_created', 0)}")
        lines.append(f"  Cache hits: {ingest_stats.get('cache_hits', 0)}")
        lines.append(f"  Elapsed: {ingest_stats.get('elapsed_seconds', 0):.1f}s")

    lines.append(f"\n{'=' * 60}")
    return "\n".join(lines)


def format_comparison_table(
    quaid_scores: dict,
    fc_scores: dict,
    quaid_cost: dict = None,
    fc_cost: dict = None,
) -> str:
    """Format a side-by-side comparison table."""
    lines = []
    lines.append(f"\n{'=' * 70}")
    lines.append("AgentLife Benchmark — Comparison")
    lines.append(f"{'=' * 70}")

    q = quaid_scores["overall"]
    f = fc_scores["overall"]

    lines.append(f"\n{'System':<20} {'Accuracy':>10} {'Count':>8} {'Correct':>8}")
    lines.append(f"{'─' * 50}")
    lines.append(f"{'Quaid (Memory)':<20} {q['accuracy']:>9.1f}% {q['count']:>8} {q['correct']:>8}")
    lines.append(f"{'Full Context':<20} {f['accuracy']:>9.1f}% {f['count']:>8} {f['correct']:>8}")

    # Per-type comparison
    all_types = sorted(set(list(quaid_scores["per_type"].keys()) + list(fc_scores["per_type"].keys())))
    if all_types:
        lines.append(f"\n{'Query Type':<25} {'Quaid':>8} {'FC':>8} {'Delta':>8}")
        lines.append(f"{'─' * 55}")
        for qt in all_types:
            q_acc = quaid_scores["per_type"].get(qt, {}).get("accuracy", 0)
            f_acc = fc_scores["per_type"].get(qt, {}).get("accuracy", 0)
            delta = q_acc - f_acc
            sign = "+" if delta >= 0 else ""
            lines.append(f"{qt:<25} {q_acc:>7.1f}% {f_acc:>7.1f}% {sign}{delta:>6.1f}%")

    # Cost comparison
    if quaid_cost and fc_cost:
        lines.append(f"\n{'Metric':<25} {'Quaid':>12} {'FC':>12}")
        lines.append(f"{'─' * 55}")
        lines.append(f"{'Total API cost':<25} ${quaid_cost['total_cost']:>10.4f} ${fc_cost['total_cost']:>10.4f}")
        lines.append(f"{'Avg ctx tokens/query':<25} {quaid_cost.get('avg_context_tokens_per_query', 0):>12,} {fc_cost.get('avg_context_tokens_per_query', 0):>12,}")

    lines.append(f"\n{'=' * 70}")
    return "\n".join(lines)
