#!/usr/bin/env bash
set -euo pipefail

# launch-remote-benchmark.sh
# Orchestrates benchmark runs from local machine:
#  1) sync local benchmark repo to remote benchmark root
#  2) sync local checkpoint repo to remote checkpoint root
#  3) launch benchmark remotely from canonical benchmark root
#
# Usage:
#   ./scripts/launch-remote-benchmark.sh --remote spark -- --mode eval --run-id r500

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_BENCH_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCAL_CHECKPOINT_ROOT="${LOCAL_CHECKPOINT_ROOT:-$HOME/quaid/benchmark-checkpoint}"

REMOTE=""
REMOTE_BENCH_ROOT=""
REMOTE_CHECKPOINT_ROOT=""
REMOTE_CHECKPOINT_PLUGIN_ROOT=""
DRY_RUN=false
SKIP_LOCAL_CHECKS=false
PARALLEL="${BENCHMARK_PARALLEL:-6}"

usage() {
  cat <<'USAGE'
Usage:
  launch-remote-benchmark.sh --remote <host> [options] -- [benchmark args...]

Required:
  --remote HOST                    Remote host or SSH alias (example: spark)

Options:
  --remote-bench-root PATH         Remote canonical benchmark root
                                  (default: ~/agentlife-benchmark)
  --remote-checkpoint-root PATH    Remote canonical checkpoint root
                                  (default: ~/quaid/benchmark-checkpoint)
  --remote-plugin-root PATH        Remote plugin checkout used by harness
                                  (default: ~/clawd/plugins/quaid)
  --local-checkpoint-root PATH     Local checkpoint root
                                  (default: ~/quaid/benchmark-checkpoint)
  --dry-run                        Print actions without running rsync/ssh
  --skip-local-checks              Skip local compile/test gate before sync+launch
  --parallel N                     Parallel workers hint (default: 6)
  -h, --help                       Show help

Examples:
  ./scripts/launch-remote-benchmark.sh --remote spark -- --mode full --backend claude-code
  ./scripts/launch-remote-benchmark.sh --remote spark --dry-run -- --mode eval --results-dir runs/quaid-s-r500
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote) REMOTE="$2"; shift 2 ;;
    --remote-bench-root) REMOTE_BENCH_ROOT="$2"; shift 2 ;;
    --remote-checkpoint-root) REMOTE_CHECKPOINT_ROOT="$2"; shift 2 ;;
    --remote-plugin-root) REMOTE_CHECKPOINT_PLUGIN_ROOT="$2"; shift 2 ;;
    --local-checkpoint-root) LOCAL_CHECKPOINT_ROOT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --skip-local-checks) SKIP_LOCAL_CHECKS=true; shift ;;
    --parallel) PARALLEL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$REMOTE" ]]; then
  echo "ERROR: --remote is required" >&2
  usage
  exit 1
fi

if ! [[ "$PARALLEL" =~ ^[0-9]+$ ]] || [[ "$PARALLEL" -lt 1 ]]; then
  echo "ERROR: --parallel must be a positive integer" >&2
  exit 1
fi

if [[ -z "$REMOTE_BENCH_ROOT" ]]; then
  REMOTE_BENCH_ROOT="~/agentlife-benchmark"
fi
if [[ -z "$REMOTE_CHECKPOINT_ROOT" ]]; then
  REMOTE_CHECKPOINT_ROOT="~/quaid/benchmark-checkpoint"
fi
if [[ -z "$REMOTE_CHECKPOINT_PLUGIN_ROOT" ]]; then
  REMOTE_CHECKPOINT_PLUGIN_ROOT="~/clawd/plugins/quaid"
fi

if [[ ! -d "$LOCAL_BENCH_ROOT" ]]; then
  echo "ERROR: local benchmark root missing: $LOCAL_BENCH_ROOT" >&2
  exit 1
fi
if [[ ! -d "$LOCAL_CHECKPOINT_ROOT" ]]; then
  echo "ERROR: local checkpoint root missing: $LOCAL_CHECKPOINT_ROOT" >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  echo "ERROR: benchmark args missing; pass them after --" >&2
  usage
  exit 1
fi

LAUNCH_ARGS=("$@")
LAUNCH_ARGS_ESCAPED=""
printf -v LAUNCH_ARGS_ESCAPED '%q ' "${LAUNCH_ARGS[@]}"
LAUNCH_ARGS_ESCAPED="${LAUNCH_ARGS_ESCAPED% }"
RESULTS_DIR=""
AUTO_RESULTS_DIR=false
for ((i=0; i<${#LAUNCH_ARGS[@]}; i++)); do
  if [[ "${LAUNCH_ARGS[$i]}" == "--results-dir" ]] && (( i + 1 < ${#LAUNCH_ARGS[@]} )); then
    RESULTS_DIR="${LAUNCH_ARGS[$((i+1))]}"
    break
  fi
done
if [[ -z "$RESULTS_DIR" ]]; then
  AUTO_RESULTS_DIR=true
fi

run_cmd() {
  echo "+ $*"
  if ! $DRY_RUN; then
    "$@"
  fi
}

run_local_checks() {
  echo "--- Local harness checks (required) ---"
  run_cmd python3 -m py_compile "$LOCAL_BENCH_ROOT/eval/run_production_benchmark.py"
  run_cmd bash -lc "cd '$LOCAL_BENCH_ROOT' && pytest -q eval/tests/test_benchmark_regressions.py eval/tests/test_store_edge_retry.py"
}

assert_checkpoint_has_no_agentlife_local() {
  local checkpoint_root="$1"
  local hit
  hit="$(find "$checkpoint_root" -type d -name 'agentlife' -print -quit 2>/dev/null || true)"
  if [[ -n "$hit" ]]; then
    echo "ERROR: checkpoint purity gate failed (local)." >&2
    echo "Found forbidden path in checkpoint: $hit" >&2
    echo "Harness code must live only in ~/agentlife-benchmark." >&2
    exit 1
  fi
}

assert_checkpoint_has_no_agentlife_remote() {
  local remote="$1"
  local checkpoint_root="$2"
  local hit
  hit="$(ssh "${SSH_OPTS[@]}" "$remote" "find $checkpoint_root -type d -name agentlife -print -quit 2>/dev/null || true" || true)"
  if [[ -n "${hit//[[:space:]]/}" ]]; then
    echo "ERROR: checkpoint purity gate failed (remote)." >&2
    echo "Found forbidden path in remote checkpoint: $hit" >&2
    echo "Harness code must not exist anywhere under checkpoint on remote." >&2
    exit 1
  fi
}

echo "=== Remote Benchmark Launcher ==="
echo "Remote:                     $REMOTE"
echo "Local benchmark root:       $LOCAL_BENCH_ROOT"
echo "Local checkpoint root:      $LOCAL_CHECKPOINT_ROOT"
echo "Remote benchmark root:      $REMOTE_BENCH_ROOT"
echo "Remote checkpoint root:     $REMOTE_CHECKPOINT_ROOT"
echo "Remote plugin root:         $REMOTE_CHECKPOINT_PLUGIN_ROOT"
echo "Parallel workers:           $PARALLEL"
echo "Benchmark args:             $*"
echo ""

if ! $SKIP_LOCAL_CHECKS; then
  run_local_checks
  echo ""
else
  echo "--- Local harness checks skipped via --skip-local-checks ---"
  echo ""
fi

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10)

if $AUTO_RESULTS_DIR; then
  ts="$(date +%Y%m%d-%H%M%S)"
  if $DRY_RUN; then
    RESULTS_DIR="runs/quaid-s-r000-${ts}"
  else
    next_run="$(
      ssh "${SSH_OPTS[@]}" "$REMOTE" '
        set -euo pipefail
        mkdir -p ~/agentlife-benchmark/runs
        ls -1 ~/agentlife-benchmark/runs 2>/dev/null \
          | sed -n "s/^quaid-s-r\\([0-9]\\+\\)-.*/\\1/p" \
          | sort -n \
          | tail -1
      ' | tr -d '[:space:]'
    )"
    if [[ -z "$next_run" ]]; then
      next_run=1
    else
      next_run=$((10#$next_run + 1))
    fi
    printf -v run_id "r%03d" "$next_run"
    RESULTS_DIR="runs/quaid-s-${run_id}-${ts}"
  fi
fi

# Ensure benchmark writes artifacts into the canonical run directory when
# caller did not explicitly set --results-dir.
if $AUTO_RESULTS_DIR; then
  LAUNCH_ARGS+=("--results-dir" "$RESULTS_DIR")
  printf -v LAUNCH_ARGS_ESCAPED '%q ' "${LAUNCH_ARGS[@]}"
  LAUNCH_ARGS_ESCAPED="${LAUNCH_ARGS_ESCAPED% }"
fi

echo "--- Checkpoint purity gate (no agentlife in checkpoint) ---"
assert_checkpoint_has_no_agentlife_local "$LOCAL_CHECKPOINT_ROOT"
if ! $DRY_RUN; then
  assert_checkpoint_has_no_agentlife_remote "$REMOTE" "$REMOTE_CHECKPOINT_ROOT"
fi
echo "  checkpoint purity OK"
echo ""

echo "--- 1) Ensure remote roots exist ---"
run_cmd ssh "${SSH_OPTS[@]}" "$REMOTE" \
  "mkdir -p $REMOTE_BENCH_ROOT $REMOTE_CHECKPOINT_ROOT $REMOTE_CHECKPOINT_PLUGIN_ROOT"

echo ""
echo "--- 1b) Sync fresh Claude Code credentials when backend is claude-code ---"
BACKEND_ARG=""
for ((i=0; i<${#LAUNCH_ARGS[@]}; i++)); do
  if [[ "${LAUNCH_ARGS[$i]}" == "--backend" ]] && (( i + 1 < ${#LAUNCH_ARGS[@]} )); then
    BACKEND_ARG="${LAUNCH_ARGS[$((i+1))]}"
    break
  fi
done
if [[ "$BACKEND_ARG" == "claude-code" ]]; then
  LOCAL_CLAUDE_CREDS="$HOME/.claude/.credentials.json"
  if [[ -f "$LOCAL_CLAUDE_CREDS" ]]; then
    run_cmd ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p ~/.claude"
    run_cmd scp -p "$LOCAL_CLAUDE_CREDS" "$REMOTE:~/.claude/.credentials.json"
    echo "  synced ~/.claude/.credentials.json to $REMOTE"
  else
    echo "  local ~/.claude/.credentials.json not found; skipping credential sync"
  fi
else
  echo "  backend is not claude-code; skipping credential sync"
fi

echo ""
echo "--- 2) Sync canonical benchmark repo ---"
RSYNC_COMMON=(
  -az
  --delete
  --itemize-changes
  --exclude='.git'
  --exclude='node_modules'
  --exclude='.ruff_cache'
  --exclude='.venv'
  --exclude='.pytest_cache'
  --exclude='__pycache__'
  --exclude='*.pyc'
  --exclude='.DS_Store'
  --exclude='runs/'
  --exclude='*.db'
  --exclude='*.db-shm'
  --exclude='*.db-wal'
)
run_cmd rsync "${RSYNC_COMMON[@]}" "$LOCAL_BENCH_ROOT/" "$REMOTE:$REMOTE_BENCH_ROOT/"

echo ""
echo "--- 3) Sync canonical checkpoint repo ---"
run_cmd rsync "${RSYNC_COMMON[@]}" "$LOCAL_CHECKPOINT_ROOT/" "$REMOTE:$REMOTE_CHECKPOINT_ROOT/"

echo ""
echo "--- 4) Sync checkpoint plugin mirror used by harness ---"
run_cmd rsync "${RSYNC_COMMON[@]}" "$LOCAL_CHECKPOINT_ROOT/plugins/quaid/" "$REMOTE:$REMOTE_CHECKPOINT_PLUGIN_ROOT/"

echo ""
echo "--- 5) Re-check remote checkpoint purity post-sync ---"
if ! $DRY_RUN; then
  assert_checkpoint_has_no_agentlife_remote "$REMOTE" "$REMOTE_CHECKPOINT_ROOT"
  echo "  remote checkpoint purity OK"
else
  echo "  skipped (dry-run)"
fi

echo ""
echo "--- 6) Launch remote benchmark ---"
REMOTE_PY_CMD="
set -euo pipefail
cd $REMOTE_BENCH_ROOT
if [[ -d $(printf %q "$REMOTE_CHECKPOINT_ROOT")/modules/quaid ]]; then
  export BENCHMARK_PLUGIN_DIR=$(printf %q "$REMOTE_CHECKPOINT_ROOT")/modules/quaid
elif [[ -d $(printf %q "$REMOTE_CHECKPOINT_ROOT")/plugins/quaid ]]; then
  export BENCHMARK_PLUGIN_DIR=$(printf %q "$REMOTE_CHECKPOINT_ROOT")/plugins/quaid
elif [[ -d $(printf %q "$REMOTE_CHECKPOINT_PLUGIN_ROOT") ]]; then
  export BENCHMARK_PLUGIN_DIR=$(printf %q "$REMOTE_CHECKPOINT_PLUGIN_ROOT")
fi
export BENCHMARK_PARALLEL=$(printf %q "$PARALLEL")
export BENCHMARK_LIFECYCLE_PREPASS_WORKERS=$(printf %q "$PARALLEL")
export BENCHMARK_JANITOR_LLM_WORKERS=$(printf %q "$PARALLEL")
export BENCHMARK_JANITOR_REVIEW_WORKERS=$(printf %q "$PARALLEL")
export AGENTLIFE_ASSETS_DIR=$(printf %q "$REMOTE_BENCH_ROOT")/data/sessions
if [[ -z \"\${ANTHROPIC_API_KEY:-}\" && -f \"/home/solomon/clawd/.env\" ]]; then
  export ANTHROPIC_API_KEY=\$(python3 - <<'PY'
from pathlib import Path
path = Path('/home/solomon/clawd/.env')
value = ''
try:
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, raw = line.split('=', 1)
        if key.strip() != 'ANTHROPIC_API_KEY':
            continue
        value = raw.strip().strip('\"').strip(\"'\")
        break
except Exception:
    value = ''
print(value)
PY
)
fi
if [[ -z \"\${CLAUDE_CODE_OAUTH_TOKEN:-}\" && -f \"\$HOME/.claude/.credentials.json\" ]]; then
  export CLAUDE_CODE_OAUTH_TOKEN=\$(python3 - <<'PY'
import json
from pathlib import Path
path = Path.home() / '.claude' / '.credentials.json'
try:
    data = json.loads(path.read_text())
except Exception:
    print('')
else:
    print(str((data.get('claudeAiOauth') or {}).get('accessToken') or '').strip())
PY
)
fi
if [[ -n \"\${CLAUDE_CODE_OAUTH_TOKEN:-}\" ]]; then
  echo \"Claude Code token: present\"
else
  echo \"Claude Code token: missing\"
fi
if [[ $(printf %q "$BACKEND_ARG") == "claude-code" ]]; then
  export CLAUDE_CODE_TIMEOUT_MULTIPLIER=2
fi
if [[ -f eval/run_production_benchmark.py ]]; then
  RUNNER=eval/run_production_benchmark.py
elif [[ -f agentlife/eval/run_production_benchmark.py ]]; then
  RUNNER=agentlife/eval/run_production_benchmark.py
else
  echo 'ERROR: run_production_benchmark.py not found under eval/ or agentlife/eval/' >&2
  exit 1
fi
echo \"Using runner: \$RUNNER\"
mkdir -p $(printf %q "$RESULTS_DIR")
LAUNCH_LOG=${RESULTS_DIR}.launch.log
nohup env PYTHONUNBUFFERED=1 python3 \"\$RUNNER\" $LAUNCH_ARGS_ESCAPED > \"\$LAUNCH_LOG\" 2>&1 &
RPID=\$!
sleep 1
if ps -p \"\$RPID\" >/dev/null 2>&1; then
  echo \"Remote launched: pid=\$RPID log=\$LAUNCH_LOG\"
else
  echo \"ERROR: remote launch failed; check \$LAUNCH_LOG\" >&2
  exit 1
fi
"
run_cmd ssh "${SSH_OPTS[@]}" "$REMOTE" "$REMOTE_PY_CMD"

echo ""
echo "Done."
