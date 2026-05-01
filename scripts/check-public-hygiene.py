#!/usr/bin/env python3
"""Fail release/push if public-tree hygiene hazards are present."""

from __future__ import annotations

import fnmatch
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ROOT_SCRATCH_PATTERNS = [
    "recovered-from-spark-*",
    "tmp",
    "__pycache__",
    "reports",
    "data/timestamps-L.json",
    "data/timestamps-S.json",
]

TRACKED_FORBIDDEN_PATTERNS = [
    "recovered-from-spark-*",
    "tmp/*",
    "__pycache__/*",
    "reports/*",
    "data/timestamps-L.json",
    "data/timestamps-S.json",
    "*.pyc",
]

README_BANNED_PHRASES = [
    "release-gate KPI",
    "Benchmark OAuth Token",
    "claude setup-token",
    "Rolling Replay Utilities",
    "Historical experiment artifacts",
    "local scratch runs may still exist",
]


def _tracked_files() -> list[str]:
    out = subprocess.check_output(
        ["git", "-C", str(ROOT), "ls-files"], text=True
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


def main() -> int:
    failures: list[str] = []

    for pattern in ROOT_SCRATCH_PATTERNS:
        for path in ROOT.glob(pattern):
            if path.exists():
                failures.append(f"root scratch artifact present: {path.relative_to(ROOT)}")

    for tracked in _tracked_files():
        for pattern in TRACKED_FORBIDDEN_PATTERNS:
            if fnmatch.fnmatch(tracked, pattern):
                failures.append(f"forbidden tracked path: {tracked}")
                break

    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    for phrase in README_BANNED_PHRASES:
        if phrase in text:
            failures.append(f"README contains internal/release-only phrase: {phrase!r}")

    if failures:
        print("[public-hygiene] FAIL", file=sys.stderr)
        for item in failures:
            print(f"- {item}", file=sys.stderr)
        return 1

    print("[public-hygiene] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
