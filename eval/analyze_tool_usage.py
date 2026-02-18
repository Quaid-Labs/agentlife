#!/usr/bin/env python3
"""Analyze tool usage patterns correlated with correctness across AgentLife benchmark runs."""

import json
import sys
from pathlib import Path
from collections import defaultdict


def load_results(results_dir: Path) -> list:
    for name in ["evaluation_results.json", "eval_results.json"]:
        path = results_dir / name
        if path.exists():
            with open(path) as f:
                return json.load(f)
    print(f"  ERROR: no results file in {results_dir}")
    return []


def _acc(b: dict) -> float:
    """Compute accuracy as (C + 0.5*P) / total * 100, matching main scoring."""
    if b["total"] == 0:
        return 0.0
    return (b["C"] + 0.5 * b["P"]) / b["total"] * 100


def _bucket():
    return {"C": 0, "P": 0, "W": 0, "total": 0}


def _tally(bucket: dict, label: str):
    bucket["total"] += 1
    if label == "CORRECT":
        bucket["C"] += 1
    elif label == "PARTIAL":
        bucket["P"] += 1
    else:
        bucket["W"] += 1


def analyze_tool_usage(results: list, label: str):
    """Analyze tool usage patterns and correctness."""
    print(f"\n{'='*70}")
    print(f"  TOOL USAGE ANALYSIS: {label}")
    print(f"{'='*70}")
    print(f"  {len(results)} queries total\n")

    # --- 1. Overall by tool count ---
    by_count = defaultdict(_bucket)
    for r in results:
        n = len(r.get("tool_calls", []))
        _tally(by_count[n], r["judge_label"])

    print("  Tool Calls    Count  Correct  Partial  Wrong    Accuracy")
    print("  " + "─" * 60)
    for n in sorted(by_count.keys()):
        b = by_count[n]
        c_pct = b["C"] / b["total"] * 100 if b["total"] else 0
        p_pct = b["P"] / b["total"] * 100 if b["total"] else 0
        w_pct = b["W"] / b["total"] * 100 if b["total"] else 0
        print(f"  {n:>10}    {b['total']:>5}  {b['C']:>4} ({c_pct:4.0f}%)  {b['P']:>4} ({p_pct:4.0f}%)  {b['W']:>4} ({w_pct:4.0f}%)    {_acc(b):5.1f}%")

    # --- 2. Pre-inject vs follow-up tool calls (v8-style) ---
    has_preinject = any("pre-inject" in tc for r in results for tc in r.get("tool_calls", []))
    if has_preinject:
        print(f"\n  Pre-inject Analysis:")
        print("  " + "─" * 60)
        categories = {
            "pre-inject only": _bucket(),
            "pre-inject + follow-up": _bucket(),
            "no pre-inject": _bucket(),
        }
        for r in results:
            tc = r.get("tool_calls", [])
            has_pi = any("pre-inject" in t for t in tc)
            follow_ups = [t for t in tc if "pre-inject" not in t]
            if has_pi and len(follow_ups) == 0:
                cat = "pre-inject only"
            elif has_pi and len(follow_ups) > 0:
                cat = "pre-inject + follow-up"
            else:
                cat = "no pre-inject"
            _tally(categories[cat], r["judge_label"])

        print(f"  {'Category':<25} Count  Correct  Partial  Wrong    Accuracy")
        print("  " + "─" * 70)
        for cat, b in categories.items():
            if b["total"] == 0:
                continue
            c_pct = b["C"] / b["total"] * 100
            p_pct = b["P"] / b["total"] * 100
            w_pct = b["W"] / b["total"] * 100
            print(f"  {cat:<25} {b['total']:>5}  {b['C']:>4} ({c_pct:4.0f}%)  {b['P']:>4} ({p_pct:4.0f}%)  {b['W']:>4} ({w_pct:4.0f}%)    {_acc(b):5.1f}%")

        # Follow-up tool count breakdown
        print(f"\n  Follow-up Calls (after pre-inject):")
        print("  " + "─" * 60)
        by_followup = defaultdict(_bucket)
        for r in results:
            tc = r.get("tool_calls", [])
            if not any("pre-inject" in t for t in tc):
                continue
            follow_ups = [t for t in tc if "pre-inject" not in t]
            _tally(by_followup[len(follow_ups)], r["judge_label"])

        print(f"  {'Follow-ups':<15} Count  Correct  Partial  Wrong    Accuracy")
        print("  " + "─" * 60)
        for n in sorted(by_followup.keys()):
            b = by_followup[n]
            c_pct = b["C"] / b["total"] * 100
            p_pct = b["P"] / b["total"] * 100
            w_pct = b["W"] / b["total"] * 100
            print(f"  {n:>10}      {b['total']:>5}  {b['C']:>4} ({c_pct:4.0f}%)  {b['P']:>4} ({p_pct:4.0f}%)  {b['W']:>4} ({w_pct:4.0f}%)    {_acc(b):5.1f}%")

    # --- 3. By query type × tool count ---
    print(f"\n  By Query Type:")
    print("  " + "─" * 60)
    type_stats = defaultdict(lambda: defaultdict(_bucket))
    type_totals = defaultdict(_bucket)
    for r in results:
        qt = r.get("query_type", "unknown")
        if "(" in qt:
            qt = qt[:qt.index("(")].strip()
        n = len(r.get("tool_calls", []))
        _tally(type_stats[qt][n], r["judge_label"])
        _tally(type_totals[qt], r["judge_label"])

    for qt in sorted(type_totals.keys(), key=lambda x: type_totals[x]["total"], reverse=True):
        t = type_totals[qt]
        print(f"\n  {qt} ({t['total']}q, {_acc(t):.0f}% overall):")
        for n in sorted(type_stats[qt].keys()):
            b = type_stats[qt][n]
            print(f"    {n} tools: {b['total']:>3}q → {b['C']}C/{b['P']}P/{b['W']}W ({_acc(b):.0f}%)")

    # --- 4. By difficulty × tool count ---
    print(f"\n  By Difficulty:")
    print("  " + "─" * 60)
    diff_stats = defaultdict(lambda: defaultdict(_bucket))
    diff_totals = defaultdict(_bucket)
    for r in results:
        diff = r.get("recall_difficulty", "unknown")
        if "(" in diff:
            diff = diff[:diff.index("(")].strip()
        n = len(r.get("tool_calls", []))
        _tally(diff_stats[diff][n], r["judge_label"])
        _tally(diff_totals[diff], r["judge_label"])

    for diff in ["Easy", "Medium", "Hard", "Very Hard"]:
        if diff not in diff_totals:
            continue
        t = diff_totals[diff]
        print(f"\n  {diff} ({t['total']}q, {_acc(t):.0f}% overall):")
        for n in sorted(diff_stats[diff].keys()):
            b = diff_stats[diff][n]
            print(f"    {n} tools: {b['total']:>3}q → {b['C']}C/{b['P']}P/{b['W']}W ({_acc(b):.0f}%)")

    # --- 5. Tool type effectiveness ---
    print(f"\n  Tool Type Breakdown:")
    print("  " + "─" * 60)
    tool_types = defaultdict(_bucket)
    for r in results:
        tc = r.get("tool_calls", [])
        tools_used = set()
        for t in tc:
            if "pre-inject" in t:
                tools_used.add("pre-inject")
            elif "memory_recall" in t:
                tools_used.add("memory_recall")
            elif "search_project_docs" in t:
                tools_used.add("search_project_docs")
            else:
                tools_used.add(t)
        key = " + ".join(sorted(tools_used)) if tools_used else "none"
        _tally(tool_types[key], r["judge_label"])

    print(f"  {'Tool Combo':<45} Count  Accuracy")
    print("  " + "─" * 60)
    for key in sorted(tool_types.keys(), key=lambda x: tool_types[x]["total"], reverse=True):
        b = tool_types[key]
        print(f"  {key:<45} {b['total']:>5}  {_acc(b):5.1f}% ({b['C']}C/{b['P']}P/{b['W']}W)")

    # --- 6. Summary stats ---
    print(f"\n  Summary Stats:")
    print("  " + "─" * 60)
    total_tools = sum(len(r.get("tool_calls", [])) for r in results)
    avg_tools = total_tools / len(results) if results else 0
    no_tool = sum(1 for r in results if len(r.get("tool_calls", [])) == 0)
    one_tool = sum(1 for r in results if len(r.get("tool_calls", [])) == 1)
    multi_tool = sum(1 for r in results if len(r.get("tool_calls", [])) > 1)

    total_correct = sum(1 for r in results if r["judge_label"] == "CORRECT")
    total_partial = sum(1 for r in results if r["judge_label"] == "PARTIAL")
    total_wrong = sum(1 for r in results if r["judge_label"] == "WRONG")

    overall_acc = (total_correct + 0.5 * total_partial) / len(results) * 100 if results else 0

    print(f"  Total tool calls: {total_tools}")
    print(f"  Avg tools/query: {avg_tools:.1f}")
    print(f"  0 tools: {no_tool} ({no_tool/len(results)*100:.0f}%)" if results else "")
    print(f"  1 tool:  {one_tool} ({one_tool/len(results)*100:.0f}%)" if results else "")
    print(f"  2+ tools: {multi_tool} ({multi_tool/len(results)*100:.0f}%)" if results else "")
    print(f"  Overall: {total_correct}C/{total_partial}P/{total_wrong}W = {overall_acc:.1f}%")

    # --- 7. Duration analysis ---
    durations = [r.get("answer_duration_s", 0) for r in results if r.get("answer_duration_s")]
    if durations:
        print(f"\n  Duration Analysis:")
        print("  " + "─" * 60)
        by_label = defaultdict(list)
        for r in results:
            d = r.get("answer_duration_s", 0)
            if d:
                by_label[r["judge_label"]].append(d)

        for lbl in ["CORRECT", "PARTIAL", "WRONG"]:
            ds = by_label.get(lbl, [])
            if ds:
                avg_d = sum(ds) / len(ds)
                print(f"  {lbl:>8}: avg {avg_d:.1f}s (n={len(ds)})")

        by_tc = defaultdict(list)
        for r in results:
            d = r.get("answer_duration_s", 0)
            if d:
                by_tc[len(r.get("tool_calls", []))].append(d)

        print()
        for n in sorted(by_tc.keys()):
            ds = by_tc[n]
            avg_d = sum(ds) / len(ds)
            print(f"  {n} tools: avg {avg_d:.1f}s (n={len(ds)})")


def compare_runs(runs: dict):
    """Compare tool usage across multiple runs."""
    print(f"\n\n{'='*70}")
    print(f"  CROSS-RUN COMPARISON")
    print(f"{'='*70}\n")

    headers = ["Metric"]
    rows = {
        "Overall Accuracy": [],
        "Avg Tools/Query": [],
        "0-tool Accuracy": [],
        "1-tool Accuracy": [],
        "2+ tool Accuracy": [],
        "Queries (0 tools)": [],
        "Queries (1 tool)": [],
        "Queries (2+ tools)": [],
    }

    for label, results in runs.items():
        headers.append(label)
        total = len(results)
        c = sum(1 for r in results if r["judge_label"] == "CORRECT")
        p = sum(1 for r in results if r["judge_label"] == "PARTIAL")
        rows["Overall Accuracy"].append(f"{(c + 0.5*p)/total*100:.1f}%")
        total_tools = sum(len(r.get("tool_calls", [])) for r in results)
        rows["Avg Tools/Query"].append(f"{total_tools/total:.1f}")

        for bucket_name, bucket_filter in [
            ("0-tool", lambda r: len(r.get("tool_calls", [])) == 0),
            ("1-tool", lambda r: len(r.get("tool_calls", [])) == 1),
            ("2+ tool", lambda r: len(r.get("tool_calls", [])) >= 2),
        ]:
            matching = [r for r in results if bucket_filter(r)]
            n = len(matching)
            bc = sum(1 for r in matching if r["judge_label"] == "CORRECT")
            bp = sum(1 for r in matching if r["judge_label"] == "PARTIAL")
            rows[f"Queries ({bucket_name.replace('-tool', ' tools')})"].append(str(n))
            if n > 0:
                rows[f"{bucket_name} Accuracy"].append(f"{(bc + 0.5*bp)/n*100:.1f}%")
            else:
                rows[f"{bucket_name} Accuracy"].append("n/a")

    col_widths = [max(len(h), 18) for h in headers]
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(f"  {header_line}")
    print("  " + "─" * len(header_line))
    for metric, values in rows.items():
        row = [metric] + values
        print("  " + "  ".join(v.ljust(w) for v, w in zip(row, col_widths)))


def main():
    data_dir = Path(__file__).parent.parent / "data"

    run_configs = {
        "v4 (Sonnet)": "results-production-v4",
        "v7s (Sonnet)": "results-production-v7s",
        "v8 (CI+Sonnet)": "results-production-v8",
        "Mem0 per-day": "results-mem0-perday",
    }

    if len(sys.argv) > 1:
        run_configs = {}
        for arg in sys.argv[1:]:
            name = Path(arg).name
            run_configs[name] = name

    loaded = {}
    for label, dirname in run_configs.items():
        results = load_results(data_dir / dirname)
        if results:
            loaded[label] = results
            print(f"  Loaded {label}: {len(results)} results")

    if not loaded:
        print("No results found!")
        return

    for label, results in loaded.items():
        analyze_tool_usage(results, label)

    if len(loaded) > 1:
        compare_runs(loaded)


if __name__ == "__main__":
    main()
