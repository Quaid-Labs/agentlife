#!/usr/bin/env bash
set -euo pipefail

# Watchdog for long benchmark runs.
# - Polls every N seconds (default 1200 = 20m)
# - If run process is dead and run is not complete, relaunches with resume flags
# - Exits once completion artifacts exist
#
# Usage:
#   watch_benchmark_run.sh <run_dir_rel> <base_command...>
#
# Example:
#   watch_benchmark_run.sh runs/quaid-s-r2-... \
#     python3 agentlife/eval/run_production_benchmark.py --mode full ...

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <run_dir_rel> <base_command...>" >&2
  exit 2
fi

RUN_DIR_REL="$1"
shift
BASE_CMD=("$@")

ROOT="${BENCHMARK_ROOT:-$HOME/clawd-benchmark}"
RUN_DIR="$ROOT/$RUN_DIR_REL"
POLL_SECONDS="${WATCH_POLL_SECONDS:-1200}"     # 20 minutes
MAX_RESTARTS="${WATCH_MAX_RESTARTS:-20}"
RESTARTS=0

mkdir -p "$RUN_DIR"

WATCH_LOG="$RUN_DIR/watchdog.log"
RESUME_LOG="$RUN_DIR/resume.launch.log"

ts() { date +"%Y-%m-%dT%H:%M:%S%z"; }
log() { echo "[$(ts)] $*" | tee -a "$WATCH_LOG"; }

is_complete() {
  [[ -f "$RUN_DIR/scores.json" && -f "$RUN_DIR/tier5_results.json" ]]
}

run_pattern() {
  # Match by explicit results dir path segment.
  printf "%s" "run_production_benchmark.py.*--results-dir[[:space:]]+$RUN_DIR_REL"
}

is_running() {
  pgrep -f "$(run_pattern)" >/dev/null 2>&1
}

launch_resume() {
  log "relaunching with resume flags"
  (
    cd "$ROOT"
    nohup env PYTHONUNBUFFERED=1 "${BASE_CMD[@]}" \
      --resume-extraction \
      --resume-eval \
      >>"$RESUME_LOG" 2>&1 < /dev/null &
    echo $! > "$RUN_DIR/watchdog_child.pid"
  )
  RESTARTS=$((RESTARTS + 1))
}

log "watchdog started: run_dir=$RUN_DIR_REL poll=${POLL_SECONDS}s max_restarts=$MAX_RESTARTS"
log "base_cmd=${BASE_CMD[*]}"

while true; do
  if is_complete; then
    log "completion detected (scores.json + tier5_results.json). exiting watchdog."
    exit 0
  fi

  if is_running; then
    log "run active"
  else
    if (( RESTARTS >= MAX_RESTARTS )); then
      log "max restarts reached ($MAX_RESTARTS). exiting watchdog."
      exit 1
    fi
    launch_resume
  fi

  sleep "$POLL_SECONDS"
done
