#!/usr/bin/env bash
set -euo pipefail

CANONICAL_LAUNCHER="$HOME/agentlife-benchmark/scripts/launch-remote-benchmark.sh"

cat >&2 <<EOF
ERROR: launch-benchmark.sh is retired.

There is exactly one canonical benchmark launcher now:
  $CANONICAL_LAUNCHER

Example:
  $CANONICAL_LAUNCHER --remote spark --scale s -- --mode full --backend api

Recovered snapshots are not an allowed launch path.
EOF

exit 1
