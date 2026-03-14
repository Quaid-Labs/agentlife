#!/usr/bin/env bash
set -euo pipefail

CANONICAL_WATCHDOG="$HOME/quaid/util/scripts/bench-monitor.sh"

cat >&2 <<EOF
ERROR: recovered-from-spark watch_benchmark_run.sh is retired.

Use the canonical per-run watchdog instead:
  $CANONICAL_WATCHDOG
EOF

exit 1
