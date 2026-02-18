#!/usr/bin/env python3
"""AgentLife Benchmark — Main runner.

Usage:
    # Full run (ingest + eval + FC baseline)
    python3 run_agentlife.py --mode full

    # Ingest only (extract facts, no evaluation)
    python3 run_agentlife.py --mode ingest

    # Eval only (assumes DB already exists)
    python3 run_agentlife.py --mode eval --results-dir data/results

    # FC baseline only
    python3 run_agentlife.py --mode fc

    # Single session smoke test
    python3 run_agentlife.py --mode full --sessions 1

    # Custom models
    python3 run_agentlife.py --mode full --extract-model sonnet --answer-model haiku
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))

from dataset import load_all_reviews, load_filler_reviews, merge_sessions_chronologically, get_all_eval_queries
from ingest import ingest_all, get_ingest_stats
from evaluate import evaluate_all, evaluate_fullcontext, get_eval_stats, reset_eval_stats
from metrics import score_results, estimate_cost, retrieval_metrics, format_report, format_comparison_table


def parse_sessions(s: str):
    """Parse session specification: '1', '1,3,5', '1-10', 'all'."""
    if s == "all":
        return None  # All sessions
    result = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            result.extend(range(int(a), int(b) + 1))
        else:
            result.append(int(part))
    return result


def main():
    parser = argparse.ArgumentParser(description="AgentLife Benchmark Runner")
    parser.add_argument("--mode", choices=["full", "ingest", "eval", "fc", "compare"],
                        default="full", help="Run mode")
    parser.add_argument("--sessions", type=str, default="all",
                        help="Sessions to process: 'all', '1', '1,3,5', '1-10'")
    parser.add_argument("--assets-dir", type=str,
                        default=str(_DIR.parent.parent.parent / "assets"),
                        help="Directory with session review files")
    parser.add_argument("--results-dir", type=str,
                        default=str(_DIR.parent / "data" / "results"),
                        help="Output directory for results")
    parser.add_argument("--extract-model", type=str, default="sonnet",
                        help="Model for extraction (sonnet/opus)")
    parser.add_argument("--answer-model", type=str, default="haiku",
                        help="Model for answer generation (haiku/sonnet/opus)")
    parser.add_argument("--judge-model", type=str, default="haiku",
                        help="Model for judging (haiku/gpt-4o-mini)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of memories to retrieve per query")
    parser.add_argument("--owner-id", type=str, default="maya",
                        help="Owner ID for facts")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable extraction cache")
    parser.add_argument("--fc-results-dir", type=str, default=None,
                        help="Separate results dir for FC baseline")
    parser.add_argument("--filler-dir", type=str, default=None,
                        help="Directory with filler sessions (densification)")
    args = parser.parse_args()

    # Paths
    assets_dir = Path(args.assets_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    fc_results_dir = Path(args.fc_results_dir) if args.fc_results_dir else results_dir.parent / "results-fc"

    # Parse sessions
    sessions = parse_sessions(args.sessions)

    print(f"AgentLife Benchmark Runner")
    print(f"  Mode: {args.mode}")
    print(f"  Sessions: {sessions or 'all'}")
    print(f"  Assets: {assets_dir}")
    print(f"  Results: {results_dir}")
    print(f"  Extract model: {args.extract_model}")
    print(f"  Answer model: {args.answer_model}")
    print(f"  Judge model: {args.judge_model}")
    print(f"  Top-K: {args.top_k}")
    print()

    # Load reviews
    print("Loading session reviews...")
    arc_reviews = load_all_reviews(assets_dir, sessions)
    if not arc_reviews:
        print("ERROR: No session reviews found!")
        sys.exit(1)
    print(f"  Loaded {len(arc_reviews)} arc sessions")

    # Load filler sessions if provided
    filler_reviews = []
    if args.filler_dir:
        filler_dir = Path(args.filler_dir)
        filler_reviews = load_filler_reviews(filler_dir)
        print(f"  Loaded {len(filler_reviews)} filler sessions")

    # Merge for ingestion (chronological order)
    if filler_reviews:
        reviews = merge_sessions_chronologically(arc_reviews, filler_reviews)
        print(f"  Total sessions (arc + filler): {len(reviews)}")
    else:
        reviews = arc_reviews

    # Collect eval queries (arc sessions only — fillers have no meaningful eval queries)
    all_queries = get_all_eval_queries(arc_reviews)
    print(f"  Total eval queries: {len(all_queries)}")
    print()

    t_start = time.monotonic()
    ingest_stats = None
    quaid_results = None
    quaid_eval_stats_snapshot = None
    fc_results = None

    # --- Ingestion ---
    if args.mode in ("full", "ingest"):
        print("=" * 60)
        print("PHASE 1: INGESTION")
        print("=" * 60)
        ingest_stats = ingest_all(
            reviews=reviews,
            results_dir=results_dir,
            extract_model=args.extract_model,
            owner_id=args.owner_id,
            use_cache=not args.no_cache,
        )
        print(f"\nIngestion complete:")
        print(f"  Sessions: {ingest_stats['sessions_processed']}")
        print(f"  Facts: {ingest_stats['facts_stored']}")
        print(f"  Edges: {ingest_stats['edges_created']}")
        print(f"  Errors: {ingest_stats['extraction_errors']}")
        print(f"  Duration: {ingest_stats['elapsed_seconds']:.1f}s")
        print()

    # --- Evaluation (Quaid) ---
    if args.mode in ("full", "eval"):
        print("=" * 60)
        print("PHASE 2: EVALUATION (QUAID)")
        print("=" * 60)
        db_path = results_dir / "memory.db"
        if not db_path.exists():
            print(f"ERROR: No DB found at {db_path}. Run ingestion first.")
            sys.exit(1)

        quaid_results = evaluate_all(
            queries=all_queries,
            results_dir=results_dir,
            db_path=db_path,
            answer_model=args.answer_model,
            judge_model=args.judge_model,
            owner_id=args.owner_id,
            top_k=args.top_k,
        )

        # Capture Quaid eval stats BEFORE FC runs (FC resets counters)
        quaid_eval_stats_snapshot = get_eval_stats()

        # Save results
        eval_results_path = results_dir / "evaluation_results.json"
        with open(eval_results_path, "w") as f:
            json.dump(quaid_results, f, indent=2)
        print(f"\nSaved {len(quaid_results)} results to {eval_results_path}")

    # --- Full Context Baseline ---
    if args.mode in ("full", "fc"):
        print("\n" + "=" * 60)
        print("PHASE 3: FULL CONTEXT BASELINE")
        print("=" * 60)
        fc_results_dir.mkdir(parents=True, exist_ok=True)

        # FC uses all sessions (arc + filler) as context
        fc_results = evaluate_fullcontext(
            queries=all_queries,
            reviews=reviews,  # Includes filler if --filler-dir provided
            answer_model=args.answer_model,
            judge_model=args.judge_model,
        )

        # Save FC results
        fc_path = fc_results_dir / "fc_results.json"
        with open(fc_path, "w") as f:
            json.dump(fc_results, f, indent=2)
        print(f"\nSaved {len(fc_results)} FC results to {fc_path}")

    # --- Scoring ---
    print("\n" + "=" * 60)
    print("PHASE 4: SCORING")
    print("=" * 60)

    config = {
        "extract_model": args.extract_model,
        "answer_model": args.answer_model,
        "judge_model": args.judge_model,
    }

    # Load results if not already in memory
    if quaid_results is None and args.mode in ("eval", "compare"):
        eval_path = results_dir / "evaluation_results.json"
        if eval_path.exists():
            quaid_results = json.loads(eval_path.read_text())

    if fc_results is None and args.mode in ("fc", "compare"):
        fc_path = fc_results_dir / "fc_results.json"
        if fc_path.exists():
            fc_results = json.loads(fc_path.read_text())

    # Score Quaid results
    if quaid_results:
        quaid_scores = score_results(quaid_results)
        quaid_retrieval = retrieval_metrics(quaid_results)
        # Use snapshot if captured (before FC reset), else current stats
        quaid_eval_stats = quaid_eval_stats_snapshot if quaid_eval_stats_snapshot else get_eval_stats()
        quaid_cost = estimate_cost(
            ingest_stats.get("ingest_token_stats", {}) if ingest_stats else {},
            quaid_eval_stats,
            config,
        )
        report = format_report(quaid_scores, quaid_cost, quaid_retrieval, "quaid", ingest_stats)
        print(report)

        # Save scores
        scores_path = results_dir / "scores.json"
        with open(scores_path, "w") as f:
            json.dump({
                "scores": quaid_scores,
                "cost": quaid_cost,
                "retrieval": quaid_retrieval,
                "ingest_stats": ingest_stats,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "mode": args.mode,
                    "sessions": sessions or "all",
                    "models": config,
                },
            }, f, indent=2)

    # Score FC results
    fc_scores = None
    fc_cost = None
    if fc_results:
        fc_scores = score_results(fc_results)
        fc_eval_stats = get_eval_stats()
        fc_cost = estimate_cost({}, fc_eval_stats, config)
        fc_report = format_report(fc_scores, fc_cost, {}, "full_context")
        print(fc_report)

        fc_scores_path = fc_results_dir / "fc_scores.json"
        with open(fc_scores_path, "w") as f:
            json.dump({"scores": fc_scores, "cost": fc_cost}, f, indent=2)

    # Comparison table
    if quaid_scores and fc_scores:
        comparison = format_comparison_table(quaid_scores, fc_scores, quaid_cost, fc_cost)
        print(comparison)

        comp_path = results_dir / "comparison.txt"
        with open(comp_path, "w") as f:
            f.write(comparison)

    elapsed = round(time.monotonic() - t_start, 1)
    print(f"\nTotal elapsed: {elapsed}s")


if __name__ == "__main__":
    main()
