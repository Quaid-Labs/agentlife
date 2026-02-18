#!/usr/bin/env python3
"""Compile all AgentLife benchmark results into a single report.

Handles three result sources:
1. Simulation results (results-20session, results-densified)
2. A/B test results (ab-* subdirectories)
3. VM benchmark results (results-vm/*)
"""

import json
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_DATA = _DIR.parent / "data"


def _get_overall(data: dict) -> dict:
    """Extract overall scores from either nested or flat structure."""
    if "scores" in data and "overall" in data["scores"]:
        return data["scores"]["overall"]
    if "overall" in data:
        return data["overall"]
    return data


def _get_per_type(data: dict) -> dict:
    """Extract per-type scores."""
    if "scores" in data and "per_type" in data["scores"]:
        return data["scores"]["per_type"]
    return data.get("per_type", data.get("by_type", {}))


def load_scores(results_dir: Path) -> dict:
    """Load scores.json from a results directory."""
    f = results_dir / "scores.json"
    if f.exists():
        return json.loads(f.read_text())
    return None


def load_fc_scores(results_dir: Path) -> dict:
    """Load fc_scores.json from an FC results directory."""
    for name in ["fc_scores.json", "scores.json"]:
        f = results_dir / name
        if f.exists():
            return json.loads(f.read_text())
    return None


def load_ab_results(base_dir: Path) -> dict:
    """Load A/B test results from ab-* subdirectories."""
    results = {}
    if not base_dir.exists():
        return results
    for variant_dir in sorted(base_dir.glob("ab-*")):
        if not variant_dir.is_dir():
            continue
        sf = variant_dir / "scores.json"
        if sf.exists():
            data = json.loads(sf.read_text())
            name = variant_dir.name.replace("ab-", "")
            results[name] = data
    return results


def load_vm_results(vm_dir: Path) -> dict:
    """Load VM benchmark results from results-vm/*."""
    results = {}
    if not vm_dir.exists():
        return results
    for system_dir in sorted(vm_dir.iterdir()):
        if not system_dir.is_dir():
            continue
        sf = system_dir / "scores.json"
        if sf.exists():
            data = json.loads(sf.read_text())
            name = system_dir.name
            results[name] = {
                "scores": data,
            }
            # Load injection stats if present
            inj = system_dir / "injection_stats.json"
            if inj.exists():
                results[name]["injection"] = json.loads(inj.read_text())
            # Load markdown quality if present
            mdq = system_dir / "markdown_quality.json"
            if mdq.exists():
                results[name]["markdown_quality"] = json.loads(mdq.read_text())
    return results


def load_cost_analysis(cost_dir: Path) -> dict:
    """Load cost analysis data."""
    f = cost_dir / "cost_analysis.json"
    if f.exists():
        return json.loads(f.read_text())
    return None


def print_section(title: str):
    print(f"\n{'=' * 70}")
    print(f" {title}")
    print(f"{'=' * 70}")


def compile_report():
    """Generate the full benchmark report."""
    print("AgentLife Benchmark — Complete Results")
    print(f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # --- Original 20-Session Results ---
    print_section("20-Session Original Results (349 facts)")

    quaid_raw = load_scores(_DATA / "results-20session")
    fc_raw = load_fc_scores(_DATA / "results-20session-fc")

    if quaid_raw:
        q = _get_overall(quaid_raw)
        print(f"\nQuaid:        {q['accuracy']:.1f}% ({q['correct']}/{q['count']})")
        print(f"  Correct: {q['correct']}, Partial: {q.get('partial', 0)}, Wrong: {q.get('wrong', 0)}")
    else:
        print("\n  [No Quaid results found]")

    if fc_raw:
        f = _get_overall(fc_raw)
        print(f"Full Context: {f['accuracy']:.1f}% ({f['correct']}/{f['count']})")
        print(f"  Correct: {f['correct']}, Partial: {f.get('partial', 0)}, Wrong: {f.get('wrong', 0)}")

    if quaid_raw and fc_raw:
        qa = _get_overall(quaid_raw)["accuracy"]
        fa = _get_overall(fc_raw)["accuracy"]
        eff = qa / fa * 100
        print(f"\nEfficiency: {eff:.1f}% of FC ceiling")

    # Per-type breakdown
    if quaid_raw and fc_raw:
        q_types = _get_per_type(quaid_raw)
        fc_types = _get_per_type(fc_raw)
        all_types = sorted(set(list(q_types.keys()) + list(fc_types.keys())))
        if all_types:
            print(f"\n{'Query Type':<30} {'Quaid':>8} {'FC':>8} {'Delta':>8}")
            print(f"{'─' * 54}")
            for t in all_types:
                qa = q_types.get(t, {}).get("accuracy", 0)
                fa = fc_types.get(t, {}).get("accuracy", 0)
                delta = qa - fa
                sign = "+" if delta >= 0 else ""
                print(f"{t:<30} {qa:>7.1f}% {fa:>7.1f}% {sign}{delta:>6.1f}pp")

    # --- Densified Results ---
    print_section("Densified Results (838 facts)")

    quaid_d_raw = load_scores(_DATA / "results-densified")
    fc_d_raw = load_fc_scores(_DATA / "results-densified-fc")

    if quaid_d_raw:
        q = _get_overall(quaid_d_raw)
        print(f"\nQuaid:        {q['accuracy']:.1f}% ({q['correct']}/{q['count']})")
        print(f"  Correct: {q['correct']}, Partial: {q.get('partial', 0)}, Wrong: {q.get('wrong', 0)}")

    if fc_d_raw:
        f = _get_overall(fc_d_raw)
        print(f"Full Context: {f['accuracy']:.1f}% ({f['correct']}/{f['count']})")

    if quaid_d_raw and fc_d_raw:
        qa = _get_overall(quaid_d_raw)["accuracy"]
        fa = _get_overall(fc_d_raw)["accuracy"]
        eff = qa / fa * 100
        print(f"\nEfficiency: {eff:.1f}% of FC ceiling")

    # Per-type breakdown for densified
    if quaid_d_raw and fc_d_raw:
        q_types = _get_per_type(quaid_d_raw)
        fc_types = _get_per_type(fc_d_raw)
        all_types = sorted(set(list(q_types.keys()) + list(fc_types.keys())))
        if all_types:
            print(f"\n{'Query Type':<30} {'Quaid':>8} {'FC':>8} {'Delta':>8}")
            print(f"{'─' * 54}")
            for t in all_types:
                qa = q_types.get(t, {}).get("accuracy", 0)
                fa = fc_types.get(t, {}).get("accuracy", 0)
                delta = qa - fa
                sign = "+" if delta >= 0 else ""
                print(f"{t:<30} {qa:>7.1f}% {fa:>7.1f}% {sign}{delta:>6.1f}pp")

    # --- A/B Test Results (both datasets) ---
    for label, base_dir, baseline_raw in [
        ("A/B Tests — Original (349 facts)", _DATA / "results-20session", quaid_raw),
        ("A/B Tests — Densified (838 facts)", _DATA / "results-densified", quaid_d_raw),
    ]:
        ab = load_ab_results(base_dir)
        if not ab:
            continue
        print_section(label)
        baseline_acc = _get_overall(baseline_raw)["accuracy"] if baseline_raw else 75.0

        # Sort by accuracy descending
        sorted_variants = sorted(
            ab.items(),
            key=lambda x: _get_overall(x[1])["accuracy"],
            reverse=True,
        )

        print(f"\n  Baseline: {baseline_acc:.1f}%")
        print(f"\n  {'Variant':<25} {'Accuracy':>10} {'vs Baseline':>12}")
        print(f"  {'─' * 47}")
        for name, data in sorted_variants:
            acc = _get_overall(data)["accuracy"]
            delta = acc - baseline_acc
            sign = "+" if delta >= 0 else ""
            print(f"  {name:<25} {acc:>9.1f}% {sign}{delta:>10.1f}pp")

    # --- Cross-Size Comparison ---
    ab_orig = load_ab_results(_DATA / "results-20session")
    ab_dens = load_ab_results(_DATA / "results-densified")
    shared = set(ab_orig.keys()) & set(ab_dens.keys())
    if shared:
        print_section("Cross-Size Comparison (349 vs 838 facts)")
        print(f"\n  {'Variant':<25} {'349 facts':>10} {'838 facts':>10} {'Change':>10}")
        print(f"  {'─' * 55}")

        baseline_o = _get_overall(quaid_raw)["accuracy"] if quaid_raw else 75.0
        baseline_d = _get_overall(quaid_d_raw)["accuracy"] if quaid_d_raw else 72.9

        # Add baseline row
        print(f"  {'BASELINE':<25} {baseline_o:>9.1f}% {baseline_d:>9.1f}% {baseline_d - baseline_o:>+9.1f}pp")
        for name in sorted(shared):
            o = _get_overall(ab_orig[name])["accuracy"]
            d = _get_overall(ab_dens[name])["accuracy"]
            delta = d - o
            print(f"  {name:<25} {o:>9.1f}% {d:>9.1f}% {delta:>+9.1f}pp")

    # --- VM Benchmark Results ---
    vm_results = load_vm_results(_DATA / "results-vm")
    if vm_results:
        print_section("VM Benchmark — Multi-System Comparison")

        print(f"\n  {'System':<25} {'Accuracy':>10} {'Compactions':>12} {'Cost':>10}")
        print(f"  {'─' * 60}")

        for name, data in sorted(vm_results.items()):
            scores = data.get("scores", {})
            overall = _get_overall(scores)
            acc = overall.get("accuracy", 0)

            inj = data.get("injection", {})
            compactions = inj.get("compaction_count", "?")

            # Get cost from injection stats
            cost_data = inj.get("cost", {})
            cost_by_model = cost_data.get("cost_by_model", {})
            # Use haiku+opus combo as default
            prod_key = "replay=haiku,extract=opus"
            cost = cost_by_model.get(prod_key, {}).get("total", 0)
            cost_str = f"${cost:.2f}" if cost > 0 else "—"

            print(f"  {name:<25} {acc:>9.1f}% {compactions:>12} {cost_str:>10}")

        # Per-type breakdown across systems
        all_types = set()
        for data in vm_results.values():
            pt = _get_per_type(data.get("scores", {}))
            all_types.update(pt.keys())

        if all_types:
            system_names = sorted(vm_results.keys())
            # Header
            header = f"\n  {'Query Type':<25}"
            for sn in system_names:
                header += f" {sn:>10}"
            print(header)
            print(f"  {'─' * (25 + 11 * len(system_names))}")

            for qt in sorted(all_types):
                line = f"  {qt:<25}"
                for sn in system_names:
                    pt = _get_per_type(vm_results[sn].get("scores", {}))
                    acc = pt.get(qt, {}).get("accuracy", 0)
                    line += f" {acc:>9.1f}%"
                print(line)

        # Markdown quality (Quaid systems only)
        md_systems = [
            (name, data) for name, data in vm_results.items()
            if data.get("markdown_quality")
        ]
        if md_systems:
            print_section("VM Benchmark — Markdown Quality (Quaid)")

            print(f"\n  {'File':<15} {'Lines':>6} {'Max':>6} {'Bloat':>7}")
            print(f"  {'─' * 40}")

            for name, data in md_systems:
                mdq = data["markdown_quality"]
                print(f"\n  [{name}]")

                for filename, fdata in mdq.get("core_files", {}).items():
                    if fdata.get("exists"):
                        print(
                            f"  {filename:<15} {fdata['lines']:>6} "
                            f"{fdata['max_lines']:>6} {fdata['bloat_pct']:>6.0f}%"
                        )
                    else:
                        print(f"  {filename:<15} {'MISSING':>6}")

                totals = mdq.get("totals", {})
                print(f"  Snippets: {totals.get('total_snippets', 0)}, "
                      f"Journal: {totals.get('total_journal_entries', 0)}")

                # Score if we have markdown_quality module
                try:
                    from markdown_quality import score_quality
                    scores = score_quality(mdq)
                    print(f"  Quality Score: {scores['overall']}/10")
                except ImportError:
                    pass

    # --- Cost Analysis ---
    cost = load_cost_analysis(_DATA / "cost-analysis")
    if cost:
        print_section("Cost Analysis: Natural vs Nightly Compaction")

        print(f"\nTotal messages: {cost.get('total_messages', 'N/A'):,}")
        print(f"Total content tokens: {cost.get('total_content_tokens', 'N/A'):,}")

        n = cost.get("natural", {})
        d = cost.get("nightly", {})

        nt = n.get("tokens", {})
        dt = d.get("tokens", {})

        print(f"\n{'Token Spend':<35} {'Natural':>14} {'Nightly':>14} {'Savings':>10}")
        print(f"{'─' * 73}")
        for key, lbl in [
            ("replay_input", "Session replay (input)"),
            ("extraction_input", "Extraction (input)"),
            ("extraction_output", "Extraction (output)"),
            ("janitor_input", "Janitor (input)"),
            ("janitor_output", "Janitor (output)"),
        ]:
            nv = nt.get(key, 0)
            dv = dt.get(key, 0)
            sav = f"{(1 - dv/nv)*100:.0f}%" if nv > 0 else "—"
            print(f"{lbl:<35} {nv:>14,} {dv:>14,} {sav:>10}")

        # Per-model cost breakdown
        cbm_n = n.get("cost_by_model", {})
        cbm_d = d.get("cost_by_model", {})
        if cbm_n:
            print(f"\n{'Model Combo':<35} {'Natural':>14} {'Nightly':>14} {'Savings':>10}")
            print(f"{'─' * 73}")
            for combo in sorted(cbm_n.keys()):
                nc = cbm_n.get(combo, {}).get("total", 0)
                dc = cbm_d.get(combo, {}).get("total", 0)
                sav = f"{(1 - dc/nc)*100:.0f}%" if nc > 0 else "—"
                print(f"{combo:<35} ${nc:>13.2f} ${dc:>13.2f} {sav:>10}")

    # --- Summary ---
    print_section("Summary")

    if quaid_raw and fc_raw:
        qa = _get_overall(quaid_raw)["accuracy"]
        fa = _get_overall(fc_raw)["accuracy"]
        print(f"\n20-Session Original:")
        print(f"  Quaid: {qa:.1f}% | FC: {fa:.1f}% | Efficiency: {qa/fa*100:.0f}%")

    if quaid_d_raw and fc_d_raw:
        qa = _get_overall(quaid_d_raw)["accuracy"]
        fa = _get_overall(fc_d_raw)["accuracy"]
        print(f"\nDensified:")
        print(f"  Quaid: {qa:.1f}% | FC: {fa:.1f}% | Efficiency: {qa/fa*100:.0f}%")

    if vm_results:
        print(f"\nVM Benchmark:")
        for name, data in sorted(vm_results.items()):
            acc = _get_overall(data.get("scores", {})).get("accuracy", 0)
            print(f"  {name}: {acc:.1f}%")

    # Key findings
    if ab_orig and ab_dens:
        print(f"\nKey Findings:")
        # Top-k scaling
        for name in ["top-k-5", "top-k-15", "top-k-20"]:
            if name in ab_orig and name in ab_dens:
                o = _get_overall(ab_orig[name])["accuracy"]
                d = _get_overall(ab_dens[name])["accuracy"]
                print(f"  {name}: {o:.1f}% (349) -> {d:.1f}% (838) [{d-o:+.1f}pp]")

    print()


if __name__ == "__main__":
    compile_report()
