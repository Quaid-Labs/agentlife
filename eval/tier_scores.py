#!/usr/bin/env python3
"""Calculate Tier 1+2, Tier 3, and Tier 4 scores for v10 benchmark systems."""

import json
import sys

# Define the result files and their labels
SYSTEMS = {
    "FC-Sonnet": "~/<username>/clawd/projects/agentlife/data/results-v10-fc/fc_baselines/fc_claude_sonnet_4_5_20250929_results.json",
    "FC-Opus": "~/<username>/clawd/projects/agentlife/data/results-v10-fc/fc_baselines/fc_claude_opus_4_6_results.json",
    "Quaid v8 (per-day)": "~/<username>/clawd/projects/agentlife/data/results-production-v8/evaluation_results.json",
    "Mem0 (per-day)": "~/<username>/clawd/projects/agentlife/data/results-v10-mem0-perday/eval_results.json",
    "Quaid perday-nt": "~/<username>/clawd/projects/agentlife/data/results-production-perday/evaluation_results.json",
}

# Tier definitions based on query_type prefix
TIER3_PREFIXES = ["non_question"]
TIER4_PREFIXES = ["arch_comprehension", "arch_planning"]


def classify_tier(query_type: str) -> str:
    """Classify a query into Tier 3, Tier 4, or Tier 1+2."""
    for prefix in TIER3_PREFIXES:
        if query_type == prefix or query_type.startswith(prefix + " "):
            return "tier3"
    for prefix in TIER4_PREFIXES:
        if query_type == prefix or query_type.startswith(prefix + " "):
            return "tier4"
    return "tier12"


def score_file(path: str) -> dict:
    """Read a results JSON and compute per-tier accuracy."""
    with open(path) as f:
        results = json.load(f)

    counts = {
        "tier12": {"correct": 0, "total": 0},
        "tier3": {"correct": 0, "total": 0},
        "tier4": {"correct": 0, "total": 0},
        "overall": {"correct": 0, "total": 0},
    }

    for r in results:
        qt = r["query_type"]
        tier = classify_tier(qt)
        is_correct = r.get("judge_label", "").upper() == "CORRECT"

        counts[tier]["total"] += 1
        counts["overall"]["total"] += 1
        if is_correct:
            counts[tier]["correct"] += 1
            counts["overall"]["correct"] += 1

    return counts


def fmt_acc(correct: int, total: int) -> str:
    """Format accuracy as percentage with count."""
    if total == 0:
        return "N/A (0q)"
    pct = 100.0 * correct / total
    return f"{pct:.1f}% ({correct}/{total})"


def main():
    # First, verify tier counts from one file
    print("=== Tier Classification Verification ===")
    first_path = list(SYSTEMS.values())[0]
    with open(first_path) as f:
        results = json.load(f)

    from collections import Counter
    tier_counts = Counter()
    tier_types = {"tier12": [], "tier3": [], "tier4": []}
    for r in results:
        qt = r["query_type"]
        tier = classify_tier(qt)
        tier_counts[tier] += 1
        if qt not in tier_types[tier]:
            tier_types[tier].append(qt)

    for tier in ["tier12", "tier3", "tier4"]:
        print(f"\n{tier.upper()} ({tier_counts[tier]}q):")
        for qt in sorted(tier_types[tier]):
            count = sum(1 for r in results if r["query_type"] == qt)
            print(f"  {qt}: {count}")

    total = sum(tier_counts.values())
    print(f"\nTotal: {total} (Tier 1+2: {tier_counts['tier12']}, "
          f"Tier 3: {tier_counts['tier3']}, Tier 4: {tier_counts['tier4']})")

    # Now score all systems
    print("\n\n=== Tier Scores ===\n")

    # Table header
    header = f"{'System':<25} | {'Overall (219q)':>16} | {'Tier 1+2':>16} | {'Tier 3 (non_q)':>16} | {'Tier 4 (arch)':>16}"
    print(header)
    print("-" * len(header))

    for name, path in SYSTEMS.items():
        try:
            counts = score_file(path)
            overall = fmt_acc(counts["overall"]["correct"], counts["overall"]["total"])
            t12 = fmt_acc(counts["tier12"]["correct"], counts["tier12"]["total"])
            t3 = fmt_acc(counts["tier3"]["correct"], counts["tier3"]["total"])
            t4 = fmt_acc(counts["tier4"]["correct"], counts["tier4"]["total"])
            print(f"{name:<25} | {overall:>16} | {t12:>16} | {t3:>16} | {t4:>16}")
        except FileNotFoundError:
            print(f"{name:<25} | {'FILE NOT FOUND':>16} | {'':>16} | {'':>16} | {'':>16}")
        except Exception as e:
            print(f"{name:<25} | {'ERROR: ' + str(e)[:30]:>16}")

    # Also print markdown table
    print("\n\n=== Markdown Table ===\n")
    print("| System | Overall (219q) | Tier 1+2 | Tier 3 (non_question, 12q) | Tier 4 (arch, 15q) |")
    print("|--------|---------------|----------|---------------------------|-------------------|")

    for name, path in SYSTEMS.items():
        try:
            counts = score_file(path)
            o_pct = 100.0 * counts["overall"]["correct"] / counts["overall"]["total"]
            t12_pct = 100.0 * counts["tier12"]["correct"] / counts["tier12"]["total"]
            t3_pct = 100.0 * counts["tier3"]["correct"] / counts["tier3"]["total"] if counts["tier3"]["total"] > 0 else 0
            t4_pct = 100.0 * counts["tier4"]["correct"] / counts["tier4"]["total"] if counts["tier4"]["total"] > 0 else 0
            t12_n = counts["tier12"]["total"]
            t3_n = counts["tier3"]["total"]
            t4_n = counts["tier4"]["total"]
            print(f"| {name} | {o_pct:.1f}% ({counts['overall']['correct']}/{counts['overall']['total']}) | "
                  f"{t12_pct:.1f}% ({counts['tier12']['correct']}/{t12_n}) | "
                  f"{t3_pct:.1f}% ({counts['tier3']['correct']}/{t3_n}) | "
                  f"{t4_pct:.1f}% ({counts['tier4']['correct']}/{t4_n}) |")
        except FileNotFoundError:
            print(f"| {name} | FILE NOT FOUND | | | |")


if __name__ == "__main__":
    main()
