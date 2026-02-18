#!/usr/bin/env python3
"""AgentLife A/B Test Runner — Compare retrieval configurations.

Tests different retrieval parameters against the same ingested memory DB.
Each variant reuses the same extracted facts — only retrieval changes.

Usage:
    # Run all variants
    python3 run_ab_tests.py --source-dir data/results-20session --tests all

    # Run specific tests
    python3 run_ab_tests.py --source-dir data/results-20session --tests no-project-docs,top-k-5

    # Aggregate existing results only
    python3 run_ab_tests.py --source-dir data/results-20session --aggregate
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

_DIR = Path(__file__).resolve().parent
_WORKSPACE = Path(os.environ.get("CLAWDBOT_WORKSPACE", Path.home() / "clawd"))
_QUAID_DIR = _WORKSPACE / "plugins" / "quaid"

sys.path.insert(0, str(_DIR))
if str(_QUAID_DIR) not in sys.path:
    sys.path.insert(0, str(_QUAID_DIR))

from dataset import load_all_reviews, load_filler_reviews, merge_sessions_chronologically, get_all_eval_queries
from evaluate import evaluate_all, _switch_to_db, reset_eval_stats, get_eval_stats
from metrics import score_results, estimate_cost, retrieval_metrics, format_report

# Config patching for Quaid internals
try:
    from config import get_config, reload_config
    _HAS_CONFIG = True
except ImportError:
    _HAS_CONFIG = False

# ═══════════════════════════════════════════════════════════════════════
# Test Variant Definitions
# ═══════════════════════════════════════════════════════════════════════

TEST_VARIANTS = {
    # --- Top-K variations ---
    "top-k-5": {
        "description": "Retrieve 5 memories instead of 10",
        "top_k": 5,
    },
    "top-k-15": {
        "description": "Retrieve 15 memories instead of 10",
        "top_k": 15,
    },
    "top-k-20": {
        "description": "Retrieve 20 memories instead of 10",
        "top_k": 20,
    },

    # --- Project docs ablation ---
    "no-project-docs": {
        "description": "Disable project doc RAG search",
        "include_project_docs": False,
    },
    "project-docs-only": {
        "description": "Only project docs, no memory DB",
        "include_project_docs": True,
        "top_k": 0,  # No memories
    },

    # --- Quaid config patches (require config module) ---
    "no-hyde": {
        "description": "Disable HyDE query expansion",
        "config_patches": {"retrieval.useHyde": False},
    },
    "no-reranker": {
        "description": "Disable Haiku reranker",
        "config_patches": {"retrieval.reranker.enabled": False},
    },
    "no-multi-pass": {
        "description": "Disable multi-pass retrieval",
        "config_patches": {"retrieval.multiPass": False},
    },
    "no-alias": {
        "description": "Disable entity alias resolution",
        "config_patches": {"retrieval.useAliases": False},
    },

    # --- Combined ablations ---
    "minimal": {
        "description": "Minimal retrieval: no HyDE, no reranker, top-k=5",
        "top_k": 5,
        "config_patches": {
            "retrieval.useHyde": False,
            "retrieval.reranker.enabled": False,
        },
    },
    "maximal": {
        "description": "Maximum retrieval: top-k=20, all features enabled",
        "top_k": 20,
        "config_patches": {
            "retrieval.reranker.poolSize": 60,  # 3x default
        },
    },
}

ALL_TESTS = list(TEST_VARIANTS.keys())


# ═══════════════════════════════════════════════════════════════════════
# Config patching
# ═══════════════════════════════════════════════════════════════════════

def _apply_config_patches(patches: dict):
    """Apply nested config patches. Keys like 'retrieval.useHyde' → set nested value."""
    if not _HAS_CONFIG or not patches:
        return

    import json as json_mod
    config_path = _WORKSPACE / "config" / "memory.json"
    if not config_path.exists():
        return

    config_data = json_mod.loads(config_path.read_text())
    for key, value in patches.items():
        parts = key.split(".")
        obj = config_data
        for part in parts[:-1]:
            obj = obj.setdefault(part, {})
        obj[parts[-1]] = value

    config_path.write_text(json_mod.dumps(config_data, indent=2))
    reload_config()


def _restore_config(original_config: str):
    """Restore original config from backup string."""
    config_path = _WORKSPACE / "config" / "memory.json"
    config_path.write_text(original_config)
    if _HAS_CONFIG:
        reload_config()


# ═══════════════════════════════════════════════════════════════════════
# Test execution
# ═══════════════════════════════════════════════════════════════════════

def run_variant(
    variant_name: str,
    variant_def: dict,
    queries: List[dict],
    source_db: Path,
    results_dir: Path,
    answer_model: str = "haiku",
    judge_model: str = "haiku",
    owner_id: str = "maya",
) -> dict:
    """Run a single A/B test variant.

    Returns results summary dict.
    """
    print(f"\n{'='*60}")
    print(f"A/B Test: {variant_name}")
    print(f"  {variant_def.get('description', '')}")
    print(f"{'='*60}")

    variant_results_dir = results_dir / f"ab-{variant_name}"
    variant_results_dir.mkdir(parents=True, exist_ok=True)

    # Symlink DB from source (use absolute paths for reliable symlinks)
    abs_source_db = source_db.resolve()
    variant_db = variant_results_dir / "memory.db"
    if variant_db.exists() or variant_db.is_symlink():
        variant_db.unlink()
    variant_db.symlink_to(abs_source_db)

    # Also symlink projects directory
    source_projects = source_db.parent.resolve() / "projects"
    variant_projects = variant_results_dir / "projects"
    if variant_projects.exists() or variant_projects.is_symlink():
        if variant_projects.is_symlink():
            variant_projects.unlink()
        else:
            shutil.rmtree(variant_projects)
    if source_projects.exists():
        variant_projects.symlink_to(source_projects)

    # Apply config patches
    original_config = None
    config_patches = variant_def.get("config_patches", {})
    if config_patches and _HAS_CONFIG:
        config_path = _WORKSPACE / "config" / "memory.json"
        if config_path.exists():
            original_config = config_path.read_text()
            _apply_config_patches(config_patches)
            print(f"  Applied {len(config_patches)} config patches")

    try:
        # Setup parameters
        top_k = variant_def.get("top_k", 10)
        include_project_docs = variant_def.get("include_project_docs", True)

        # Run evaluation
        _switch_to_db(variant_db)
        reset_eval_stats()

        results = []
        for i, query in enumerate(queries):
            print(f"  [{i+1}/{len(queries)}] {query['question'][:50]}...")
            try:
                from evaluate import evaluate_single
                result = evaluate_single(
                    query,
                    variant_results_dir,
                    answer_model=answer_model,
                    judge_model=judge_model,
                    owner_id=owner_id,
                    top_k=top_k,
                    include_project_docs=include_project_docs,
                )
            except Exception as e:
                print(f"    ERROR: {e}")
                result = {
                    "question": query["question"],
                    "ground_truth": query["ground_truth"],
                    "prediction": "",
                    "judge_label": "ERROR",
                    "score": 0.0,
                    "query_type": query.get("query_type", "unknown"),
                    "error": str(e),
                }
            results.append(result)

        # Save results
        eval_path = variant_results_dir / "evaluation_results.json"
        with open(eval_path, "w") as f:
            json.dump(results, f, indent=2)

        # Score
        scores = score_results(results)
        retrieval = retrieval_metrics(results)
        eval_stats = get_eval_stats()
        cost = estimate_cost({}, eval_stats, {
            "extract_model": "sonnet",
            "answer_model": answer_model,
            "judge_model": judge_model,
        })

        # Save scores
        scores_path = variant_results_dir / "scores.json"
        with open(scores_path, "w") as f:
            json.dump({
                "variant": variant_name,
                "description": variant_def.get("description", ""),
                "scores": scores,
                "cost": cost,
                "retrieval": retrieval,
                "config_patches": config_patches,
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2)

        overall = scores["overall"]
        print(f"\n  Result: {overall['accuracy']:.1f}% accuracy ({overall['correct']}/{overall['scored']})")

        return {
            "variant": variant_name,
            "accuracy": overall["accuracy"],
            "correct": overall["correct"],
            "partial": overall["partial"],
            "wrong": overall["wrong"],
            "count": overall["count"],
            "cost": cost["total_cost"],
            "avg_context_tokens": cost.get("avg_context_tokens_per_query", 0),
        }

    finally:
        # Restore config
        if original_config:
            _restore_config(original_config)
            print("  Restored original config")


# ═══════════════════════════════════════════════════════════════════════
# Aggregation
# ═══════════════════════════════════════════════════════════════════════

def aggregate_results(results_dir: Path) -> str:
    """Aggregate all A/B test results into a comparison table."""
    variants = []
    for vdir in sorted(results_dir.glob("ab-*")):
        scores_path = vdir / "scores.json"
        if scores_path.exists():
            data = json.loads(scores_path.read_text())
            overall = data["scores"]["overall"]
            variants.append({
                "variant": data["variant"],
                "description": data.get("description", ""),
                "accuracy": overall["accuracy"],
                "correct": overall["correct"],
                "partial": overall["partial"],
                "wrong": overall["wrong"],
                "count": overall["count"],
                "cost": data["cost"]["total_cost"],
                "avg_ctx": data["cost"].get("avg_context_tokens_per_query", 0),
            })

    if not variants:
        return "No A/B test results found."

    # Sort by accuracy descending
    variants.sort(key=lambda v: v["accuracy"], reverse=True)

    lines = []
    lines.append(f"\n{'='*80}")
    lines.append("AgentLife A/B Test Results")
    lines.append(f"{'='*80}")
    lines.append(f"\n{'Variant':<25} {'Accuracy':>8} {'C/P/W':>10} {'Cost':>8} {'Ctx Tok':>8}")
    lines.append(f"{'─'*65}")

    for v in variants:
        cpw = f"{v['correct']}/{v['partial']}/{v['wrong']}"
        lines.append(
            f"{v['variant']:<25} {v['accuracy']:>7.1f}% {cpw:>10} ${v['cost']:>6.4f} {v['avg_ctx']:>7,}"
        )

    # Per-type breakdown
    lines.append(f"\n{'─'*80}")
    lines.append("Per-Type Accuracy (top variants)")
    lines.append(f"{'─'*80}")

    # Load per-type data for top variants
    top_variants = variants[:5]
    all_types = set()
    type_data = {}

    for v in top_variants:
        vdir = results_dir / f"ab-{v['variant']}"
        scores_path = vdir / "scores.json"
        if scores_path.exists():
            data = json.loads(scores_path.read_text())
            per_type = data["scores"]["per_type"]
            type_data[v["variant"]] = per_type
            all_types.update(per_type.keys())

    if type_data:
        header = f"{'Type':<25}"
        for v in top_variants[:5]:
            header += f" {v['variant'][:12]:>12}"
        lines.append(header)
        lines.append(f"{'─'*80}")

        for qt in sorted(all_types):
            row = f"{qt:<25}"
            for v in top_variants[:5]:
                acc = type_data.get(v["variant"], {}).get(qt, {}).get("accuracy", 0)
                row += f" {acc:>11.1f}%"
            lines.append(row)

    lines.append(f"\n{'='*80}")

    report = "\n".join(lines)

    # Save report
    report_path = results_dir / "ab_comparison.txt"
    report_path.write_text(report)
    print(f"\nComparison saved to {report_path}")

    return report


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="AgentLife A/B Test Runner")
    parser.add_argument("--source-dir", type=str, required=True,
                        help="Directory with pre-ingested memory.db")
    parser.add_argument("--tests", type=str, default="all",
                        help="Comma-separated test names, or 'all'")
    parser.add_argument("--assets-dir", type=str,
                        default=str(_DIR.parent.parent.parent / "assets"),
                        help="Directory with session review files")
    parser.add_argument("--filler-dir", type=str, default=None,
                        help="Directory with filler sessions (for densified runs)")
    parser.add_argument("--answer-model", type=str, default="haiku")
    parser.add_argument("--judge-model", type=str, default="haiku")
    parser.add_argument("--owner-id", type=str, default="maya")
    parser.add_argument("--aggregate", action="store_true",
                        help="Only aggregate existing results")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    source_db = source_dir / "memory.db"

    if not source_db.exists():
        print(f"ERROR: No memory.db found at {source_db}")
        sys.exit(1)

    # Load queries
    assets_dir = Path(args.assets_dir)
    arc_reviews = load_all_reviews(assets_dir)
    all_queries = get_all_eval_queries(arc_reviews)
    print(f"Loaded {len(all_queries)} eval queries from {len(arc_reviews)} sessions")

    if args.aggregate:
        report = aggregate_results(source_dir)
        print(report)
        return

    # Determine tests to run
    if args.tests == "all":
        tests = ALL_TESTS
    else:
        tests = [t.strip() for t in args.tests.split(",")]

    print(f"\nRunning {len(tests)} A/B test variants:")
    for t in tests:
        desc = TEST_VARIANTS.get(t, {}).get("description", "unknown")
        print(f"  - {t}: {desc}")

    # Run each variant
    summaries = []
    t_start = time.monotonic()

    for test_name in tests:
        if test_name not in TEST_VARIANTS:
            print(f"\n  WARNING: Unknown test '{test_name}', skipping")
            continue

        summary = run_variant(
            variant_name=test_name,
            variant_def=TEST_VARIANTS[test_name],
            queries=all_queries,
            source_db=source_db,
            results_dir=source_dir,
            answer_model=args.answer_model,
            judge_model=args.judge_model,
            owner_id=args.owner_id,
        )
        summaries.append(summary)

    elapsed = time.monotonic() - t_start
    print(f"\nA/B tests complete in {elapsed:.0f}s")

    # Aggregate
    report = aggregate_results(source_dir)
    print(report)


if __name__ == "__main__":
    main()
