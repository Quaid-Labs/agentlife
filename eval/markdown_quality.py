#!/usr/bin/env python3
"""AgentLife Benchmark — Markdown Quality Metrics.

Tracks core markdown file evolution during Quaid benchmark runs.
Measures bloat, snippet/journal activity, and project doc freshness.

Scoring rubric (0-10):
    10: Files at ideal size, active management, fresh project docs
     8: Minor bloat (<110% of target), regular snippet folding
     6: Moderate bloat (<130%), some stale project docs
     4: Significant bloat (>130%), infrequent management
     2: Critical bloat (>150%), no active management
     0: Files missing or corrupted

Usage:
    # Collect metrics from VM
    python3 markdown_quality.py --vm-ip 192.168.64.3

    # Analyze local workspace
    python3 markdown_quality.py --local ~/clawd

    # Score existing metrics file
    python3 markdown_quality.py --score data/results-vm/quaid/markdown_quality.json
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

_DIR = Path(__file__).resolve().parent

# Core markdown files and their bloat limits (from memory.json coreMarkdown.files)
CORE_FILES = {
    "SOUL.md": {"max_lines": 1200, "purpose": "Agent personality and interaction style"},
    "USER.md": {"max_lines": 800, "purpose": "User biographical information"},
    "MEMORY.md": {"max_lines": 400, "purpose": "Critical always-loaded memories"},
}

# Snippet and journal file patterns
SNIPPET_PATTERN = "*.snippets.md"
JOURNAL_PATTERN = "journal/*.journal.md"
JOURNAL_ARCHIVE_PATTERN = "journal/archive/*.md"


# ---------------------------------------------------------------------------
# Metric collection
# ---------------------------------------------------------------------------

def collect_local_metrics(workspace: Path) -> dict:
    """Collect markdown quality metrics from a local workspace.

    Args:
        workspace: Path to the workspace directory (e.g., results-dir/workspace)

    Returns:
        Metrics dict with file sizes, snippet counts, journal stats.
    """
    metrics = {
        "core_files": {},
        "snippets": {},
        "journal": {},
        "projects": {},
        "totals": {},
    }

    # Core markdown files
    for filename, config in CORE_FILES.items():
        filepath = workspace / filename
        if filepath.exists():
            content = filepath.read_text()
            lines = content.count("\n") + 1
            metrics["core_files"][filename] = {
                "exists": True,
                "lines": lines,
                "max_lines": config["max_lines"],
                "bloat_pct": round(lines / config["max_lines"] * 100, 1),
                "chars": len(content),
            }
        else:
            metrics["core_files"][filename] = {
                "exists": False,
                "lines": 0,
                "max_lines": config["max_lines"],
                "bloat_pct": 0,
                "chars": 0,
            }

    # Snippet files
    total_snippets = 0
    for snippet_file in workspace.glob(SNIPPET_PATTERN):
        content = snippet_file.read_text()
        # Count bullet points (lines starting with "- ")
        bullets = sum(1 for line in content.split("\n") if line.strip().startswith("- "))
        # Count sections (## headers = compaction events)
        sections = sum(1 for line in content.split("\n") if line.startswith("## "))
        metrics["snippets"][snippet_file.name] = {
            "bullets": bullets,
            "sections": sections,
            "lines": content.count("\n") + 1,
        }
        total_snippets += bullets

    # Journal files
    journal_dir = workspace / "journal"
    total_journal_entries = 0
    if journal_dir.exists():
        for journal_file in journal_dir.glob("*.journal.md"):
            content = journal_file.read_text()
            # Count entries (## headers)
            entries = sum(1 for line in content.split("\n") if line.startswith("## "))
            metrics["journal"][journal_file.name] = {
                "entries": entries,
                "lines": content.count("\n") + 1,
                "chars": len(content),
            }
            total_journal_entries += entries

        # Journal archives
        archive_dir = journal_dir / "archive"
        if archive_dir.exists():
            archive_files = list(archive_dir.glob("*.md"))
            metrics["journal"]["_archive_count"] = len(archive_files)
            metrics["journal"]["_archive_total_lines"] = sum(
                f.read_text().count("\n") + 1 for f in archive_files
            )

    # Projects
    projects_dir = workspace / "projects"
    if projects_dir.exists():
        for project_dir in projects_dir.iterdir():
            if not project_dir.is_dir():
                continue
            project_md = project_dir / "PROJECT.md"
            if project_md.exists():
                content = project_md.read_text()
                metrics["projects"][project_dir.name] = {
                    "lines": content.count("\n") + 1,
                    "chars": len(content),
                }

    # Totals
    metrics["totals"] = {
        "core_files_total_lines": sum(
            f["lines"] for f in metrics["core_files"].values()
        ),
        "total_snippets": total_snippets,
        "total_journal_entries": total_journal_entries,
        "project_count": len(metrics["projects"]),
    }

    return metrics


def collect_vm_metrics(vm) -> dict:
    """Collect markdown quality metrics from a Tart VM via SSH.

    Args:
        vm: TartVM instance with ssh() method

    Returns:
        Metrics dict.
    """
    metrics = {
        "core_files": {},
        "snippets": {},
        "journal": {},
        "projects": {},
        "totals": {},
    }

    # Core markdown files
    for filename, config in CORE_FILES.items():
        result = vm.ssh(f"wc -l ~/clawd/{filename} 2>/dev/null || echo '0'", timeout=10)
        lines = 0
        if result.returncode == 0:
            m = re.search(r"(\d+)", result.stdout)
            if m:
                lines = int(m.group(1))

        metrics["core_files"][filename] = {
            "exists": lines > 0,
            "lines": lines,
            "max_lines": config["max_lines"],
            "bloat_pct": round(lines / config["max_lines"] * 100, 1) if lines > 0 else 0,
        }

    # Snippet files
    result = vm.ssh(
        "for f in ~/clawd/journal/*.snippets.md ~/clawd/*.snippets.md; do "
        "[ -f \"$f\" ] && echo \"$(basename $f):$(grep -c '^- ' $f 2>/dev/null || echo 0)\"; "
        "done",
        timeout=10,
    )
    total_snippets = 0
    if result.returncode == 0:
        for line in result.stdout.strip().split("\n"):
            if ":" in line:
                name, count_str = line.rsplit(":", 1)
                count = int(count_str) if count_str.isdigit() else 0
                metrics["snippets"][name] = {"bullets": count}
                total_snippets += count

    # Journal files
    result = vm.ssh(
        "for f in ~/clawd/journal/*.journal.md; do "
        "[ -f \"$f\" ] && echo \"$(basename $f):$(grep -c '^## ' $f 2>/dev/null || echo 0)\"; "
        "done",
        timeout=10,
    )
    total_journal = 0
    if result.returncode == 0:
        for line in result.stdout.strip().split("\n"):
            if ":" in line:
                name, count_str = line.rsplit(":", 1)
                count = int(count_str) if count_str.isdigit() else 0
                metrics["journal"][name] = {"entries": count}
                total_journal += count

    # Journal archives
    result = vm.ssh(
        "ls ~/clawd/journal/archive/*.md 2>/dev/null | wc -l",
        timeout=10,
    )
    if result.returncode == 0:
        m = re.search(r"(\d+)", result.stdout)
        if m:
            metrics["journal"]["_archive_count"] = int(m.group(1))

    # Memory DB stats (Quaid only)
    result = vm.ssh(
        "cd ~/clawd/plugins/quaid && python3 memory_graph.py stats 2>/dev/null",
        timeout=15,
    )
    if result.returncode == 0:
        metrics["db_stats"] = result.stdout.strip()

    metrics["totals"] = {
        "core_files_total_lines": sum(
            f["lines"] for f in metrics["core_files"].values()
        ),
        "total_snippets": total_snippets,
        "total_journal_entries": total_journal,
    }

    return metrics


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_quality(metrics: dict) -> dict:
    """Score markdown quality from collected metrics.

    Returns:
        Score dict with overall score (0-10), per-file scores, breakdown.
    """
    scores = {}
    breakdown = []

    # Score core files (0-4 points)
    core_score = 0
    total_bloat = 0
    files_present = 0

    for filename, data in metrics.get("core_files", {}).items():
        if not data.get("exists", False):
            breakdown.append(f"{filename}: MISSING (-1)")
            continue

        files_present += 1
        bloat_pct = data.get("bloat_pct", 0)
        total_bloat += bloat_pct

        if bloat_pct <= 50:
            # Under half capacity — could mean file is barely populated
            file_score = 0.8
            breakdown.append(f"{filename}: {bloat_pct:.0f}% (sparse)")
        elif bloat_pct <= 80:
            file_score = 1.0
            breakdown.append(f"{filename}: {bloat_pct:.0f}% (healthy)")
        elif bloat_pct <= 100:
            file_score = 0.9
            breakdown.append(f"{filename}: {bloat_pct:.0f}% (near limit)")
        elif bloat_pct <= 110:
            file_score = 0.7
            breakdown.append(f"{filename}: {bloat_pct:.0f}% (minor bloat)")
        elif bloat_pct <= 130:
            file_score = 0.5
            breakdown.append(f"{filename}: {bloat_pct:.0f}% (moderate bloat)")
        elif bloat_pct <= 150:
            file_score = 0.3
            breakdown.append(f"{filename}: {bloat_pct:.0f}% (significant bloat)")
        else:
            file_score = 0.1
            breakdown.append(f"{filename}: {bloat_pct:.0f}% (critical bloat)")

        scores[filename] = file_score

    if files_present > 0:
        core_score = sum(scores.values()) / len(CORE_FILES) * 4
    else:
        core_score = 0

    # Score snippet activity (0-2 points)
    total_snippets = metrics.get("totals", {}).get("total_snippets", 0)
    if total_snippets >= 20:
        snippet_score = 2.0
        breakdown.append(f"Snippets: {total_snippets} (excellent)")
    elif total_snippets >= 10:
        snippet_score = 1.5
        breakdown.append(f"Snippets: {total_snippets} (good)")
    elif total_snippets >= 5:
        snippet_score = 1.0
        breakdown.append(f"Snippets: {total_snippets} (moderate)")
    elif total_snippets > 0:
        snippet_score = 0.5
        breakdown.append(f"Snippets: {total_snippets} (low)")
    else:
        snippet_score = 0
        breakdown.append(f"Snippets: 0 (none)")

    # Score journal activity (0-2 points)
    total_journal = metrics.get("totals", {}).get("total_journal_entries", 0)
    if total_journal >= 10:
        journal_score = 2.0
        breakdown.append(f"Journal entries: {total_journal} (excellent)")
    elif total_journal >= 5:
        journal_score = 1.5
        breakdown.append(f"Journal entries: {total_journal} (good)")
    elif total_journal >= 2:
        journal_score = 1.0
        breakdown.append(f"Journal entries: {total_journal} (moderate)")
    elif total_journal > 0:
        journal_score = 0.5
        breakdown.append(f"Journal entries: {total_journal} (low)")
    else:
        journal_score = 0
        breakdown.append(f"Journal entries: 0 (none)")

    # Score project docs (0-2 points)
    project_count = metrics.get("totals", {}).get("project_count", 0)
    if project_count >= 2:
        project_score = 2.0
        breakdown.append(f"Projects: {project_count} (good)")
    elif project_count >= 1:
        project_score = 1.0
        breakdown.append(f"Projects: {project_count} (minimal)")
    else:
        project_score = 0
        breakdown.append(f"Projects: 0 (none)")

    overall = round(core_score + snippet_score + journal_score + project_score, 1)
    overall = min(10.0, max(0.0, overall))

    return {
        "overall": overall,
        "core_files_score": round(core_score, 1),
        "snippet_score": round(snippet_score, 1),
        "journal_score": round(journal_score, 1),
        "project_score": round(project_score, 1),
        "breakdown": breakdown,
        "per_file": scores,
    }


def format_quality_report(metrics: dict, scores: dict) -> str:
    """Format a human-readable markdown quality report."""
    lines = []
    lines.append("Markdown Quality Report")
    lines.append("=" * 50)

    lines.append(f"\nOverall Score: {scores['overall']}/10")
    lines.append(f"  Core files: {scores['core_files_score']}/4")
    lines.append(f"  Snippets:   {scores['snippet_score']}/2")
    lines.append(f"  Journal:    {scores['journal_score']}/2")
    lines.append(f"  Projects:   {scores['project_score']}/2")

    # Core file details
    lines.append(f"\n{'File':<15} {'Lines':>6} {'Max':>6} {'Bloat':>7}")
    lines.append(f"{'─' * 40}")
    for filename, data in metrics.get("core_files", {}).items():
        if data.get("exists"):
            lines.append(
                f"{filename:<15} {data['lines']:>6} {data['max_lines']:>6} "
                f"{data['bloat_pct']:>6.0f}%"
            )
        else:
            lines.append(f"{filename:<15} {'MISSING':>6}")

    # Snippet details
    snippets = metrics.get("snippets", {})
    if snippets:
        lines.append(f"\nSnippet Files:")
        for name, data in snippets.items():
            lines.append(f"  {name}: {data.get('bullets', 0)} bullets")

    # Journal details
    journal = metrics.get("journal", {})
    if journal:
        lines.append(f"\nJournal Files:")
        for name, data in journal.items():
            if name.startswith("_"):
                continue
            lines.append(f"  {name}: {data.get('entries', 0)} entries")
        if "_archive_count" in journal:
            lines.append(f"  Archives: {journal['_archive_count']} files")

    # Breakdown
    lines.append(f"\nScoring Breakdown:")
    for note in scores.get("breakdown", []):
        lines.append(f"  - {note}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Timeline tracking
# ---------------------------------------------------------------------------

class QualityTracker:
    """Track markdown quality over time during a benchmark run.

    Records snapshots at each compaction event to show file evolution.
    """

    def __init__(self):
        self.snapshots: List[dict] = []

    def record_snapshot(
        self,
        label: str,
        metrics: dict,
        session_idx: int = 0,
        compaction_idx: int = 0,
    ):
        """Record a quality snapshot."""
        snapshot = {
            "label": label,
            "session_idx": session_idx,
            "compaction_idx": compaction_idx,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "metrics": metrics,
        }
        self.snapshots.append(snapshot)

    def get_timeline(self) -> List[dict]:
        """Get all snapshots as a timeline."""
        return self.snapshots

    def format_timeline(self) -> str:
        """Format timeline as a readable table."""
        if not self.snapshots:
            return "No snapshots recorded."

        lines = []

        # Get all core files from first snapshot
        first = self.snapshots[0].get("metrics", {}).get("core_files", {})
        file_names = list(first.keys())

        # Header
        header = f"{'#':>3} {'Label':<20} "
        for fn in file_names:
            header += f"{fn:>12} "
        header += f"{'Snippets':>10} {'Journal':>10}"
        lines.append(header)
        lines.append("─" * len(header))

        for i, snap in enumerate(self.snapshots):
            m = snap.get("metrics", {})
            line = f"{i:>3} {snap['label']:<20} "
            for fn in file_names:
                lines_count = m.get("core_files", {}).get(fn, {}).get("lines", 0)
                line += f"{lines_count:>12} "
            totals = m.get("totals", {})
            line += f"{totals.get('total_snippets', 0):>10} "
            line += f"{totals.get('total_journal_entries', 0):>10}"
            lines.append(line)

        return "\n".join(lines)

    def save(self, path: Path):
        """Save timeline to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.snapshots, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Markdown Quality Metrics")
    parser.add_argument("--local", type=str, default=None,
                        help="Analyze local workspace directory")
    parser.add_argument("--vm-ip", type=str, default=None,
                        help="Collect from VM at this IP")
    parser.add_argument("--score", type=str, default=None,
                        help="Score existing metrics JSON file")
    args = parser.parse_args()

    if args.score:
        with open(args.score) as f:
            metrics = json.load(f)
        scores = score_quality(metrics)
        print(format_quality_report(metrics, scores))
        return

    if args.local:
        workspace = Path(args.local)
        metrics = collect_local_metrics(workspace)
    elif args.vm_ip:
        sys.path.insert(0, str(_DIR))
        from vm_benchmark import TartVM
        vm = TartVM(ip=args.vm_ip)
        metrics = collect_vm_metrics(vm)
    else:
        print("Specify --local <path> or --vm-ip <ip> or --score <file>")
        return

    scores = score_quality(metrics)
    print(format_quality_report(metrics, scores))

    # Save
    output = _DIR.parent / "data" / "markdown_quality.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump({"metrics": metrics, "scores": scores}, f, indent=2)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
