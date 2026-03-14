#!/usr/bin/env python3
"""Retired shim for the canonical benchmark monitor."""

from __future__ import annotations

import os
import sys


def main() -> int:
    canonical = os.path.expanduser("~/agentlife-benchmark/scripts/monitor_benchmarks.py")
    sys.stderr.write(
        "ERROR: recovered-from-spark monitor_benchmarks.py is retired.\n\n"
        f"Use the canonical monitor instead:\n  {canonical}\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
