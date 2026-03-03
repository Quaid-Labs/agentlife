#!/usr/bin/env bash
# bench-monitor.sh — Persistent benchmark run monitor (v2)
#
# Lightweight bash loop that watches a Spark benchmark run for:
#   - Hard failures (process death, errors, crashes)
#   - Stalls (alive but 0% CPU, no log growth, no progress)
#   - Progress milestones (chunk/eval changes)
#   - Periodic 30m status dumps sent via tmux for LLM review
#
# IMPORTANT: Run this from the LOCAL control host (Mac mini), NOT on Spark.
# All Spark access is via SSH. Running locally preserves outage visibility
# and avoids self-SSH failures (Spark can't SSH to itself reliably).
#
# Usage:
#   bench-monitor.sh <run-dir> [pid]
#   bench-monitor.sh runs/quaid-s-r133-20260225-054419
#   bench-monitor.sh runs/quaid-s-r133-20260225-054419 2927013
#
# Environment:
#   MONITOR_INTERVAL      - fast-loop seconds (default: 30)
#   MONITOR_REPORT_SEC    - full report interval (default: 1800 = 30m)
#   MONITOR_STALL_SEC     - stall alert threshold (default: 600 = 10m)
#   MONITOR_TARGET        - tmux target for reports (default: codex-bench)
#   MONITOR_ALERT_TARGET  - tmux target for alerts (default: same as MONITOR_TARGET)
#   MONITOR_SENDER        - sender label for tmux-msg.sh prefix (default: bench-monitor)
#   MONITOR_SOURCE        - source pane tag for tmux-msg.sh prefix (default: main:3.0)
#   SPARK_HOST            - override spark-env.sh default

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUAID_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=spark-env.sh
source "$SCRIPT_DIR/spark-env.sh"

# --- Args ---
if [[ $# -lt 1 ]]; then
  echo "Usage: bench-monitor.sh <run-dir> [pid]" >&2
  echo "  run-dir: relative path from benchmark root (e.g. runs/quaid-s-r133-...)" >&2
  echo "  MUST be run from local control host, not on Spark." >&2
  exit 2
fi

RUN_REL="$1"
if [[ ! "$RUN_REL" =~ ^runs/[A-Za-z0-9._-]+$ ]]; then
  echo "ERROR: run-dir must match runs/<name> with [A-Za-z0-9._-] only" >&2
  exit 2
fi

# --- Lockfile: one monitor per run ---
LOCK_DIR="/tmp"
RUN_SLUG="$(echo "$RUN_REL" | tr '/' '-')"
PIDFILE="${LOCK_DIR}/bench-monitor-${RUN_SLUG}.pid"

# Atomic pidfile claim to avoid TOCTOU races when multiple monitors start together.
if ! ( set -o noclobber; echo "$$" > "$PIDFILE" ) 2>/dev/null; then
  OTHER_PID=$(cat "$PIDFILE" 2>/dev/null || echo "")
  if [[ -n "$OTHER_PID" ]] && kill -0 "$OTHER_PID" 2>/dev/null; then
    echo "ERROR: Another bench-monitor is already running for $RUN_REL (PID $OTHER_PID)" >&2
    echo "  pidfile: $PIDFILE" >&2
    echo "  Kill it first or remove the pidfile to proceed." >&2
    exit 1
  fi
  echo "Stale pidfile found (PID ${OTHER_PID:-unknown} dead). Removing." >&2
  rm -f "$PIDFILE"
  if ! ( set -o noclobber; echo "$$" > "$PIDFILE" ) 2>/dev/null; then
    echo "ERROR: Failed to claim pidfile $PIDFILE" >&2
    exit 1
  fi
fi

# --- Trap: cleanup children + pidfile on exit ---
cleanup() {
  rm -f "$PIDFILE"
  # Kill any child processes (sleep, ssh, etc.)
  local children=()
  local pid
  mapfile -t children < <(jobs -p 2>/dev/null || true)
  for pid in "${children[@]}"; do
    [[ -n "$pid" ]] || continue
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${children[@]}"; do
    [[ -n "$pid" ]] || continue
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM HUP
RUN_DIR="${SPARK_BENCHMARK_ROOT}/${RUN_REL}"
GIVEN_PID="${2:-}"

# Log file: check sidecar (RDIR.launch.log), then inside dir (run.log, launch.log)
LAUNCH_LOG=""

# --- Config ---
INTERVAL="${MONITOR_INTERVAL:-30}"
REPORT_SEC="${MONITOR_REPORT_SEC:-1800}"
STALL_SEC="${MONITOR_STALL_SEC:-600}"
TMUX_TARGET="${MONITOR_TARGET:-codex-bench}"
ALERT_TARGET="${MONITOR_ALERT_TARGET:-$TMUX_TARGET}"
TMUX_SENDER="${MONITOR_SENDER:-bench-monitor}"
TMUX_SOURCE="${MONITOR_SOURCE:-main:3.0}"
TMUX_MSG="${TMUX_MSG:-$QUAID_ROOT/scripts/tmux-msg.sh}"

# --- State ---
CHECK=0
LAST_REPORT=0  # force first report immediately
LAST_CHUNK=-1
LAST_EVAL=-1
ALERTED_DEAD=0
ALERTED_PATTERNS=""  # track which error patterns we've already alerted on
PROGRESS_CHANGED=0

# Stall detection state
LAST_LOG_SIZE=-1        # launch log byte count
LAST_LOG_CHANGE_TS=0    # epoch when log last changed
LAST_PROGRESS_CHANGE_TS=0  # epoch when chunk/eval last changed
ZERO_CPU_STREAK=0       # consecutive checks with 0% CPU
ALERTED_STALL_LOG=0     # already alerted on log stall
ALERTED_STALL_PROGRESS=0  # already alerted on progress stall
ALERTED_STALL_CPU=0     # already alerted on CPU stall
START_TS=$(date +%s)

# --- Helpers ---
ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

send_msg() {
  local target="$1"
  shift
  TMUX_MSG_SENDER="$TMUX_SENDER" TMUX_MSG_SOURCE="$TMUX_SOURCE" "$TMUX_MSG" "$target" "$*" 2>/dev/null || true
}

ssh_cmd() {
  local out=""
  local rc=0
  local attempt
  for attempt in 1 2 3; do
    out=$(ssh -o ConnectTimeout=10 -o BatchMode=yes "$SPARK_HOST" "$@" 2>/dev/null) && rc=0 || rc=$?
    if [[ $rc -eq 0 ]]; then
      printf "%s" "$out"
      return 0
    fi
    sleep 1
  done
  # Never fail hard in monitor loop; return empty payload on SSH failure.
  printf ""
  return 0
}

# grep -c returns exit 1 on zero matches — wrap to always succeed
ssh_grep_count() {
  local pattern="$1" file="$2"
  local val
  val=$(ssh_cmd "grep -c '$pattern' '$file' 2>/dev/null || echo 0" | tail -1 | tr -d '[:space:]')
  echo "${val:-0}"
}

ssh_json_field() {
  local file="$1" field="$2"
  ssh_cmd "python3 - '$file' '$field' <<'PY'
import json
import sys
from pathlib import Path
try:
    path = Path(sys.argv[1])
    field = sys.argv[2]
    data = json.loads(path.read_text())
    val = data.get(field, '')
    print('' if val is None else val)
except Exception:
    print('')
PY" | tr -d '[:space:]'
}

# --- Log file resolution ---
# Check multiple possible locations for the run's output log
resolve_log() {
  local candidates=(
    "${RUN_DIR}.launch.log"       # sidecar (traditional)
    "${RUN_DIR}/launch.log"       # inside dir (symlink)
    "${RUN_DIR}/run.log"          # inside dir (new convention)
  )
  for f in "${candidates[@]}"; do
    local size
    size=$(ssh_cmd "wc -c < '$f' 2>/dev/null" | tr -d '[:space:]')
    if [[ "${size:-0}" -gt 0 ]] 2>/dev/null; then
      echo "$f"
      return
    fi
  done
  # Fallback to sidecar path even if empty
  echo "${RUN_DIR}.launch.log"
}

# --- PID resolution ---
resolve_pid() {
  if [[ -n "$GIVEN_PID" ]]; then
    echo "$GIVEN_PID"
    return
  fi
  local pid
  pid=$(ssh_cmd "ps -eo pid,comm,args 2>/dev/null | awk -v runrel='$RUN_REL' '\$2==\"python3\" && index(\$0,\"run_production_benchmark.py\") && index(\$0,runrel) {print \$1; exit}'" | tr -d '[:space:]')
  if [[ -z "$pid" ]]; then
    pid=$(ssh_cmd "ps -eo pid,args 2>/dev/null | awk -v runrel='$RUN_REL' 'index(\$0,\"run_production_benchmark.py\") && index(\$0,runrel) {print \$1; exit}'" | tr -d '[:space:]')
  fi
  echo "${pid:-}"
}

# --- Process alive check ---
# Uses kill -0 instead of ps -p (more portable, works in all SSH contexts)
check_alive() {
  local pid="$1"
  if [[ -z "$pid" ]] || [[ ! "$pid" =~ ^[0-9]+$ ]]; then echo ""; return; fi
  local result
  result=$(ssh_cmd "if kill -0 '$pid' 2>/dev/null; then echo alive; elif ps -p '$pid' >/dev/null 2>&1; then echo alive; else echo dead; fi" | tr -d '[:space:]')
  # If SSH probe returned nothing, treat as unknown and keep monitoring
  # rather than declaring a hard process death.
  if [[ -z "$result" ]]; then
    echo "$pid"
    return
  fi
  if [[ "$result" == "alive" ]]; then
    echo "$pid"
  else
    echo ""
  fi
}

# --- Get process CPU% ---
get_cpu() {
  local pid="$1"
  if [[ -z "$pid" ]] || [[ ! "$pid" =~ ^[0-9]+$ ]]; then echo "0"; return; fi
  local cpu
  cpu=$(ssh_cmd "ps -p '$pid' -o %cpu= 2>/dev/null" | tr -d '[:space:]')
  echo "${cpu:-0}"
}

# --- Phase detection ---
detect_phase() {
  local chunk="$1" total="$2" eval_done="$3" eval_total="$4"

  if [[ -n "$eval_done" && "$eval_done" != "" ]]; then
    if [[ "$eval_total" =~ ^[0-9]+$ ]] && [[ "$eval_total" -gt 0 ]]; then
      echo "Eval($eval_done/$eval_total)"
    else
      echo "Eval($eval_done/?)"
    fi
    return
  fi

  if [[ "$chunk" -ge 0 ]] 2>/dev/null; then
    if [[ "$total" -gt 0 && "$chunk" -ge "$((total - 1))" ]] 2>/dev/null; then
      local store_count
      store_count=$(ssh_grep_count 'Phase 2\|store_and_run_janitor' "$LAUNCH_LOG")
      if [[ "${store_count:-0}" -gt 0 ]] 2>/dev/null; then
        echo "Store+Janitor"
        return
      fi
    fi
    if [[ "$total" -gt 0 ]] 2>/dev/null; then
      echo "Extraction($((chunk+1))/$total)"
    else
      echo "Extraction($((chunk+1))/?)"
    fi
    return
  fi

  echo "Starting"
}

# --- ETA calculation ---
calc_eta() {
  local pid="$1" chunk="$2" total="$3"
  if [[ -z "$pid" ]] || [[ ! "$pid" =~ ^[0-9]+$ ]]; then echo "unknown"; return; fi
  if [[ "$chunk" -le 0 ]] 2>/dev/null; then echo "n/a"; return; fi
  if [[ "$total" -le 0 ]] 2>/dev/null; then echo "unknown"; return; fi
  local uptime
  uptime=$(ssh_cmd "ps -p '$pid' -o etimes= 2>/dev/null" | tr -d '[:space:]')
  if [[ "${uptime:-0}" -gt 0 ]] 2>/dev/null; then
    local remaining=$((total - chunk - 1))
    if [[ "$remaining" -lt 0 ]]; then
      echo "unknown"
      return
    fi
    local spc=$((uptime / (chunk + 1)))
    local eta_s=$((remaining * spc))
    echo "~$((eta_s / 60))m"
  else
    echo "n/a"
  fi
}

# --- Gather full status ---
gather_status() {
  local pid="$1"

  local chunk total
  chunk=$(ssh_json_field "$RUN_DIR/extraction_cache/progress.json" "last_completed_chunk")
  total=$(ssh_json_field "$RUN_DIR/extraction_cache/progress.json" "total_chunks")
  chunk="${chunk:--1}"
  total="${total:-0}"
  # Early extraction often writes cache files before progress.json exists.
  if [[ "$chunk" -lt 0 ]] 2>/dev/null; then
    local cache_count
    cache_count=$(ssh_cmd "ls '$RUN_DIR/extraction_cache'/chunk-*.json 2>/dev/null | wc -l" | tr -d '[:space:]')
    cache_count="${cache_count:-0}"
    if [[ "$cache_count" -gt 0 ]] 2>/dev/null; then
      chunk=$((cache_count - 1))
    fi
  fi
  if [[ "$total" -le 0 ]] 2>/dev/null; then
    local inferred_total
    inferred_total=$(ssh_cmd "grep -oE 'Timeout chunks: [0-9]+' '$LAUNCH_LOG' 2>/dev/null | tail -1 | awk '{print \$3}'" | tr -d '[:space:]')
    if [[ -n "$inferred_total" ]] && [[ "$inferred_total" -gt 0 ]] 2>/dev/null; then
      total="$inferred_total"
    fi
  fi

  local eval_done eval_total eval_done_raw
  # Current schema writes last_completed_query/total_queries.
  # Keep fallback to older completed/total fields for compatibility.
  eval_done_raw=$(ssh_json_field "$RUN_DIR/logs/eval_progress.json" "last_completed_query")
  eval_total=$(ssh_json_field "$RUN_DIR/logs/eval_progress.json" "total_queries")
  if [[ -n "$eval_done_raw" && "$eval_done_raw" =~ ^-?[0-9]+$ ]]; then
    # last_completed_query is 0-based index; display as completed-count for humans.
    eval_done="$((eval_done_raw + 1))"
    if [[ "$eval_done" -lt 0 ]]; then
      eval_done=0
    fi
  else
    eval_done=""
  fi
  if [[ -z "$eval_done" ]]; then
    eval_done=$(ssh_json_field "$RUN_DIR/logs/eval_progress.json" "completed")
  fi
  if [[ -z "$eval_total" ]]; then
    eval_total=$(ssh_json_field "$RUN_DIR/logs/eval_progress.json" "total")
  fi
  if [[ "$eval_done" =~ ^[0-9]+$ ]] && [[ "$eval_total" =~ ^[0-9]+$ ]] && [[ "$eval_total" -gt 0 ]] 2>/dev/null; then
    if [[ "$eval_done" -gt "$eval_total" ]]; then
      eval_done="$eval_total"
    fi
  fi

  local nodes
  nodes=$(ssh_cmd "sqlite3 '$RUN_DIR/data/memory.db' 'SELECT status, COUNT(*) FROM nodes GROUP BY status' 2>/dev/null" | tr '\n' ' ')
  nodes="${nodes:-n/a}"

  local core
  core=$(ssh_cmd "cat '$RUN_DIR/SOUL.md' '$RUN_DIR/USER.md' '$RUN_DIR/MEMORY.md' '$RUN_DIR/TOOLS.md' 2>/dev/null | wc -c" | tr -d '[:space:]')
  core="${core:-0}"

  local threads
  # Prefer configured run parallelism for stable reporting; fallback to live claude -p count.
  threads=$(ssh_json_field "$RUN_DIR/run_metadata.json" "parallel")
  if [[ -z "$threads" ]]; then
    threads=$(ssh_cmd "ps -eo args 2>/dev/null | grep 'claude.*-p' | grep -v grep | wc -l" | tr -d '[:space:]')
  fi
  if ! [[ "$threads" =~ ^[0-9]+$ ]]; then
    threads=$(ssh_cmd "ps -eo args 2>/dev/null | grep 'claude.*-p' | grep -v grep | wc -l" | tr -d '[:space:]')
  fi
  threads="${threads:-0}"

  local rate gate pgrad jtimeout
  rate=$(ssh_cmd "grep -ci '429\|overloaded\|RateLimitError' '$LAUNCH_LOG' 2>/dev/null || echo 0" | tail -1 | tr -d '[:space:]')
  gate=$(ssh_grep_count 'review gate failed' "$LAUNCH_LOG")
  pgrad=$(ssh_grep_count 'invalid state.*pending' "$LAUNCH_LOG")
  jtimeout=$(ssh_grep_count 'janitor task failed\|task.*timed out' "$LAUNCH_LOG")
  rate="${rate:-0}"

  local phase eta
  phase=$(detect_phase "$chunk" "$total" "$eval_done" "$eval_total")
  eta=$(calc_eta "$pid" "$chunk" "$total")

  local janitor
  janitor=$(ssh_cmd "python3 -c \"import json; d=json.load(open('$RUN_DIR/logs/janitor-stats.json')); print(f'task={d.get(\\\"last_task\\\",\\\"?\\\")}, changes={d.get(\\\"applied_changes\\\",\\\"?\\\")}')\" 2>/dev/null" || echo "n/a")

  local cpu
  cpu=$(get_cpu "$pid")

  cat <<EOF
phase=$phase | ETA=$eta | threads=$threads | cpu=${cpu}%
core=${core}b | nodes=[$nodes]
errors: rate=$rate gate=$gate pgrad=$pgrad jtimeout=$jtimeout
janitor: $janitor
EOF
}

# --- Hard failure scan ---
scan_failures() {
  local alerts=""

  local tail50
  tail50=$(ssh_cmd "tail -50 '$LAUNCH_LOG' 2>/dev/null" || echo "")

  if echo "$tail50" | grep -q 'RuntimeError\|ImportError\|ModuleNotFoundError'; then
    if [[ "$ALERTED_PATTERNS" != *"runtime"* ]]; then
      local err_line
      err_line=$(echo "$tail50" | grep -m1 'RuntimeError\|ImportError\|ModuleNotFoundError' | head -c 200)
      alerts="${alerts}RUNTIME_ERROR: $err_line\n"
      ALERTED_PATTERNS="${ALERTED_PATTERNS} runtime"
    fi
  fi

  if echo "$tail50" | grep -q 'Traceback (most recent call last)'; then
    if [[ "$ALERTED_PATTERNS" != *"traceback"* ]]; then
      alerts="${alerts}TRACEBACK detected in log tail\n"
      ALERTED_PATTERNS="${ALERTED_PATTERNS} traceback"
    fi
  fi

  local gate
  gate=$(ssh_grep_count 'review gate failed' "$LAUNCH_LOG")
  if [[ "${gate:-0}" -gt 0 ]] 2>/dev/null; then
    if [[ "$ALERTED_PATTERNS" != *"gate"* ]]; then
      alerts="${alerts}REVIEW_GATE_FAIL($gate)\n"
      ALERTED_PATTERNS="${ALERTED_PATTERNS} gate"
    fi
  fi

  local pgrad
  pgrad=$(ssh_grep_count 'invalid state.*pending' "$LAUNCH_LOG")
  if [[ "${pgrad:-0}" -gt 0 ]] 2>/dev/null; then
    if [[ "$ALERTED_PATTERNS" != *"pgrad"* ]]; then
      alerts="${alerts}PENDING_AFTER_GRADUATE($pgrad)\n"
      ALERTED_PATTERNS="${ALERTED_PATTERNS} pgrad"
    fi
  fi

  local rate
  rate=$(ssh_cmd "grep -ci '429\|overloaded\|RateLimitError' '$LAUNCH_LOG' 2>/dev/null || echo 0" | tail -1 | tr -d '[:space:]')
  if [[ "${rate:-0}" -gt 5 ]] 2>/dev/null; then
    if [[ "$ALERTED_PATTERNS" != *"rate"* ]]; then
      alerts="${alerts}RATE_LIMIT_SPIKE($rate)\n"
      ALERTED_PATTERNS="${ALERTED_PATTERNS} rate"
    fi
  fi

  echo -e "$alerts"
}

# --- Stall detection ---
# Checks three independent stall signals:
#   1. Launch log not growing (no new output)
#   2. Progress artifacts unchanged (chunk/eval stuck)
#   3. Process at 0% CPU for consecutive checks
check_stall() {
  local pid="$1"
  local now="$2"
  local alerts=""

  # --- 1. Log freshness ---
  local log_size
  log_size=$(ssh_cmd "wc -c < '$LAUNCH_LOG' 2>/dev/null" | tr -d '[:space:]')
  log_size="${log_size:-0}"

  if [[ "$log_size" != "$LAST_LOG_SIZE" ]]; then
    LAST_LOG_SIZE="$log_size"
    LAST_LOG_CHANGE_TS=$now
    ALERTED_STALL_LOG=0
  fi

  local log_stale_sec=$((now - LAST_LOG_CHANGE_TS))
  if [[ $LAST_LOG_CHANGE_TS -gt 0 && $log_stale_sec -ge $STALL_SEC && $ALERTED_STALL_LOG -eq 0 ]]; then
    alerts="${alerts}LOG_STALL: launch.log unchanged for ${log_stale_sec}s (size=${log_size}b)\n"
    ALERTED_STALL_LOG=1
  fi

  # --- 2. Progress freshness ---
  local chunk eval_done
  chunk="${CHUNK:--1}"
  eval_done="${EVAL_DONE:-}"

  # Check if either chunk or eval has changed since last progress change
  if [[ "$PROGRESS_CHANGED" -eq 1 ]]; then
    LAST_PROGRESS_CHANGE_TS=$now
    ALERTED_STALL_PROGRESS=0
  fi

  # Only check progress stall after initial grace period (2 min from start)
  # Also check extraction_cache growth as secondary signal (parallel extraction
  # writes chunk files before progress.json updates)
  local uptime=$((now - START_TS))
  local progress_stale_sec=$((now - LAST_PROGRESS_CHANGE_TS))
  if [[ $LAST_PROGRESS_CHANGE_TS -gt 0 && $uptime -gt 120 && $progress_stale_sec -ge $STALL_SEC && $ALERTED_STALL_PROGRESS -eq 0 ]]; then
    local cache_count
    cache_count=$(ssh_cmd "ls '$RUN_DIR/extraction_cache'/chunk-*.json 2>/dev/null | wc -l" | tr -d '[:space:]')
    local cache_newest
    cache_newest=$(ssh_cmd "python3 - <<'PY'
import glob
import os

files = glob.glob(r'''$RUN_DIR/extraction_cache/chunk-*.json''')
if not files:
    print(0)
else:
    newest = max(files, key=os.path.getmtime)
    print(int(os.path.getmtime(newest)))
PY" | tr -d '[:space:]')
    local cache_age=999999
    if [[ -n "$cache_newest" && "$cache_newest" -gt 0 ]] 2>/dev/null; then
      local remote_now
      remote_now=$(ssh_cmd "date +%s" | tr -d '[:space:]')
      cache_age=$((remote_now - cache_newest))
    fi
    # If newest chunk file is recent (< STALL_SEC), extraction is still active
    # Also check if log file is growing — store+janitor phase updates log, not progress.json
    local log_stale=$((now - LAST_LOG_CHANGE_TS))
    if [[ $cache_age -lt $STALL_SEC ]] || [[ $log_stale -lt $STALL_SEC ]]; then
      LAST_PROGRESS_CHANGE_TS=$now  # reset, work is happening
    else
      alerts="${alerts}PROGRESS_STALL: no chunk/eval change for ${progress_stale_sec}s, cache stale ${cache_age}s, log stale ${log_stale}s (chunk=$chunk eval=$eval_done, ${cache_count:-0} chunk files)\n"
      ALERTED_STALL_PROGRESS=1
    fi
  fi

  # Special case: still at "Starting" (no progress.json) after STALL_SEC
  # But check extraction_cache for chunk files — parallel extraction writes chunks
  # before progress.json exists, so chunk files growing = not stalled
  if [[ "$chunk" == "-1" && -z "$eval_done" && $uptime -ge $STALL_SEC && $ALERTED_STALL_PROGRESS -eq 0 ]]; then
    local cache_count
    cache_count=$(ssh_cmd "ls '$RUN_DIR/extraction_cache'/chunk-*.json 2>/dev/null | wc -l" | tr -d '[:space:]')
    cache_count="${cache_count:-0}"
    if [[ "$cache_count" -eq 0 ]] 2>/dev/null; then
      alerts="${alerts}STARTUP_STALL: no progress.json and no chunk files after ${uptime}s\n"
      ALERTED_STALL_PROGRESS=1
    else
      # Chunks exist but no progress.json — extraction in progress, not a stall
      # Reset the progress change timestamp so we don't re-alert
      LAST_PROGRESS_CHANGE_TS=$now
    fi
  fi

  # --- 3. CPU stall (0% for 3+ consecutive checks) ---
  local cpu
  cpu=$(get_cpu "$pid")
  # Compare as float: treat anything <= 0.1 as zero
  if awk "BEGIN {exit !($cpu <= 0.1)}" 2>/dev/null; then
    ZERO_CPU_STREAK=$((ZERO_CPU_STREAK + 1))
  else
    ZERO_CPU_STREAK=0
    ALERTED_STALL_CPU=0
  fi

  if [[ $ZERO_CPU_STREAK -ge 3 && $ALERTED_STALL_CPU -eq 0 ]]; then
    alerts="${alerts}CPU_STALL: 0% CPU for $ZERO_CPU_STREAK consecutive checks (~$((ZERO_CPU_STREAK * INTERVAL))s)\n"
    ALERTED_STALL_CPU=1
  fi

  echo -e "$alerts"
}

# ============================================================
# MAIN LOOP
# ============================================================

RUN_NAME=$(basename "$RUN_REL")
LAUNCH_LOG=$(resolve_log)
log "bench-monitor v2 starting: $RUN_NAME"
log "log_file=$LAUNCH_LOG"
log "run_dir=$RUN_DIR"
log "interval=${INTERVAL}s, report every ${REPORT_SEC}s, stall threshold ${STALL_SEC}s"
log "tmux target=$TMUX_TARGET, alert target=$ALERT_TARGET"

# Resolve initial PID
PID=$(resolve_pid)
if [[ -z "$PID" ]]; then
  log "WARNING: could not find PID for $RUN_REL. Will keep trying."
else
  log "tracking PID=$PID"
fi

# Initialize stall baselines
LAST_LOG_CHANGE_TS=$START_TS
LAST_PROGRESS_CHANGE_TS=$START_TS

while true; do
  CHECK=$((CHECK + 1))
  NOW=$(date +%s)
  TS=$(ts)

  # --- Check if already complete ---
  SCORES=$(ssh_cmd "cat '$RUN_DIR/scores.json' 2>/dev/null" || echo "")
  if [[ -n "$SCORES" ]]; then
    log "COMPLETED — scores.json found"
    log "$SCORES"
    send_msg "$TMUX_TARGET" "bench-monitor: $RUN_NAME COMPLETED. scores=$SCORES"
    break
  fi

  # --- Resolve PID if we don't have one ---
  if [[ -z "$PID" ]]; then
    PID=$(resolve_pid)
    if [[ -n "$PID" ]]; then
      log "found PID=$PID"
    fi
  fi

  # --- Process alive check (kill -0, not ps -p) ---
  ALIVE=$(check_alive "$PID")

  if [[ -z "$ALIVE" && -n "$PID" ]]; then
    if [[ "$ALERTED_DEAD" -eq 0 ]]; then
      ALERTED_DEAD=1
      LOGTAIL=$(ssh_cmd "tail -15 '$LAUNCH_LOG' 2>/dev/null" | head -c 500 || echo "no log")
      log "PROCESS DEAD (PID=$PID)"
      send_msg "$ALERT_TARGET" "bench-monitor: $RUN_NAME PROCESS DEAD (PID=$PID). No scores.json. Log tail:
$LOGTAIL"
      log "Keeping fast poll; attempting PID re-resolve."
      PID=""
      ALERTED_DEAD=0  # reset so we can alert again if relaunched PID also dies
    fi
  fi

  # --- Hard failure scan ---
  FAILURES=$(scan_failures)
  if [[ -n "$FAILURES" ]]; then
    log "ALERT: $FAILURES"
    send_msg "$ALERT_TARGET" "bench-monitor: $RUN_NAME ALERT at $TS:
$FAILURES"
  fi

  # --- Progress tracking (chunk/eval changes) ---
  CHUNK=$(ssh_json_field "$RUN_DIR/extraction_cache/progress.json" "last_completed_chunk" 2>/dev/null)
  CHUNK="${CHUNK:--1}"
  EVAL_DONE=$(ssh_json_field "$RUN_DIR/logs/eval_progress.json" "last_completed_query" 2>/dev/null)
  if [[ -n "$EVAL_DONE" && "$EVAL_DONE" =~ ^-?[0-9]+$ ]]; then
    EVAL_DONE="$((EVAL_DONE + 1))"
    if [[ "$EVAL_DONE" -lt 0 ]]; then
      EVAL_DONE=0
    fi
  fi
  if [[ -z "$EVAL_DONE" ]]; then
    EVAL_DONE=$(ssh_json_field "$RUN_DIR/logs/eval_progress.json" "completed" 2>/dev/null)
  fi
  EVAL_DONE="${EVAL_DONE:-}"

  PROGRESS_CHANGED=0
  if [[ "$CHUNK" != "$LAST_CHUNK" && "$CHUNK" != "-1" ]]; then
    PROGRESS_CHANGED=1
    LAST_CHUNK="$CHUNK"
  fi
  if [[ -n "$EVAL_DONE" && "$EVAL_DONE" != "$LAST_EVAL" ]]; then
    PROGRESS_CHANGED=1
    LAST_EVAL="$EVAL_DONE"
  fi

  # --- Stall detection (only when process is alive) ---
  if [[ -n "$ALIVE" ]]; then
    STALLS=$(check_stall "$PID" "$NOW")
    if [[ -n "$STALLS" ]]; then
      log "STALL: $STALLS"
      send_msg "$ALERT_TARGET" "bench-monitor: $RUN_NAME STALL at $TS (PID=$PID alive):
$STALLS"
    fi
  fi

  # --- Periodic report (every REPORT_SEC or on progress change) ---
  ELAPSED=$((NOW - LAST_REPORT))
  if [[ $ELAPSED -ge $REPORT_SEC ]] || [[ $PROGRESS_CHANGED -eq 1 && $ELAPSED -ge 60 ]]; then
    STATUS=$(gather_status "${PID:-0}")
    log "STATUS: $RUN_NAME | $STATUS"

    if [[ $ELAPSED -ge $REPORT_SEC ]]; then
      send_msg "$TMUX_TARGET" "bench-monitor: $RUN_NAME status at $TS:
$STATUS
(next auto-report in ${REPORT_SEC}s)"
      LAST_REPORT=$NOW
    fi
  fi

  # --- Console heartbeat ---
  local_cpu=$(get_cpu "$PID")
  if [[ -n "$ALIVE" ]]; then
    log "check=$CHECK alive=Y cpu=${local_cpu}% chunk=$CHUNK eval=$EVAL_DONE"
  elif [[ -n "$PID" ]]; then
    log "check=$CHECK alive=N chunk=$CHUNK eval=$EVAL_DONE"
  else
    log "check=$CHECK pid=? chunk=$CHUNK eval=$EVAL_DONE"
  fi

  sleep "$INTERVAL"
done

log "bench-monitor exiting."
