#!/usr/bin/env bash
set -euo pipefail

# deploy-to-spark.sh — Sync harness + plugin checkpoint to Spark.
#
# Usage:
#   ./scripts/deploy-to-spark.sh [--dry-run]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/spark-env.sh"

DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help)
      echo "Usage: $0 [--dry-run]"
      echo ""
      echo "Syncs harness code and plugin checkpoint to Spark."
      echo "  --dry-run   Preview changes without syncing"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

RSYNC_OPTS=(
  -avz
  --itemize-changes
  --exclude='.git'
  --exclude='node_modules'
  --exclude='__pycache__'
  --exclude='*.pyc'
  --exclude='data/'
  --exclude='*.db'
  --exclude='*.db-wal'
  --exclude='*.db-shm'
  --exclude='.env'
  --exclude='runs/'
)

if $DRY_RUN; then
  RSYNC_OPTS+=(--dry-run)
  echo "=== DRY RUN — no files will be transferred ==="
  echo ""
fi

# --- 1. Sync harness code ---
echo "--- Syncing harness: agentlife/eval/ ---"
echo "  Local:  $LOCAL_BENCHMARK_ROOT/agentlife/eval/"
echo "  Remote: $SPARK_HOST:$SPARK_BENCHMARK_ROOT/agentlife/eval/"
echo ""

rsync "${RSYNC_OPTS[@]}" \
  "$LOCAL_BENCHMARK_ROOT/agentlife/eval/" \
  "$SPARK_HOST:$SPARK_BENCHMARK_ROOT/agentlife/eval/"

echo ""

# --- 2. Sync plugin checkpoint ---
if [[ -z "$LOCAL_CHECKPOINT_ROOT" || ! -d "$LOCAL_CHECKPOINT_ROOT" ]]; then
  echo "WARNING: Local checkpoint not found at expected path."
  echo "  Expected: $(dirname "$SCRIPT_DIR")/../../benchmark-checkpoint"
  echo "  Skipping checkpoint sync."
else
  echo "--- Syncing checkpoint: benchmark-checkpoint/ ---"
  echo "  Local:  $LOCAL_CHECKPOINT_ROOT/"
  echo "  Remote: $SPARK_HOST:$SPARK_CHECKPOINT_ROOT/"
  echo ""

  rsync "${RSYNC_OPTS[@]}" \
    "$LOCAL_CHECKPOINT_ROOT/" \
    "$SPARK_HOST:$SPARK_CHECKPOINT_ROOT/"

  echo ""
fi

# --- 3. Post-sync validation ---
echo "--- Post-sync validation ---"

CHECKS_PASSED=0
CHECKS_FAILED=0

validate_remote_file() {
  local path="$1"
  local label="$2"
  if ssh "$SPARK_HOST" "test -f '$path'" 2>/dev/null; then
    echo "  [OK] $label"
    ((CHECKS_PASSED+=1))
  else
    echo "  [MISSING] $label — $path"
    ((CHECKS_FAILED+=1))
  fi
}

validate_remote_file "$SPARK_BENCHMARK_ROOT/agentlife/eval/run_production_benchmark.py" "Harness entrypoint"
validate_remote_file "$SPARK_BENCHMARK_ROOT/agentlife/eval/pathing.py" "Pathing module"
validate_remote_file "$SPARK_CHECKPOINT_ROOT/plugins/quaid/schema.sql" "Plugin schema.sql"
validate_remote_file "$SPARK_CHECKPOINT_ROOT/plugins/quaid/memory_graph.py" "Plugin memory_graph.py"
validate_remote_file "$SPARK_CHECKPOINT_ROOT/plugins/quaid/janitor.py" "Plugin janitor.py"
validate_remote_file "$SPARK_CHECKPOINT_ROOT/config/memory.json" "Config memory.json"

echo ""
if [[ $CHECKS_FAILED -gt 0 ]]; then
  echo "WARN: $CHECKS_FAILED validation check(s) failed. Verify remote paths."
  exit 1
else
  echo "All $CHECKS_PASSED validation checks passed."
fi

if $DRY_RUN; then
  echo ""
  echo "(Dry run complete — no files were transferred.)"
fi
