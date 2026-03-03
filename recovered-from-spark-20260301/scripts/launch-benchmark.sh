#!/usr/bin/env bash
set -euo pipefail

# launch-benchmark.sh — Launch a benchmark run on Spark.
#
# Usage:
#   ./scripts/launch-benchmark.sh --scale s --run-number 8
#   ./scripts/launch-benchmark.sh --scale l --run-number 1 --notify
#   ./scripts/launch-benchmark.sh --scale s --run-number 8 --resume
#   ./scripts/launch-benchmark.sh --scale s --run-number 8 --dry-run

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/spark-env.sh"

# Defaults
SCALE=""
RUN_NUMBER=""
RESUME=false
NOTIFY=false
DRY_RUN=false
BACKEND="claude-code"
MAX_SESSIONS="20"
PARALLEL="6"
MODE="full"
TIER5=true
SKIP_PREFLIGHT=false

usage() {
  cat <<'USAGE'
Usage: launch-benchmark.sh --scale s|l --run-number N [options]

Required:
  --scale s|l          AL-S or AL-L benchmark
  --run-number N       Run number for naming (e.g., 8 → quaid-s-r8-...)

Options:
  --resume             Add --resume-extraction --resume-eval flags
  --notify             Send Telegram launch notification
  --dry-run            Print full command without executing
  --backend TYPE       claude-code (default) or api
  --max-sessions N     Session cap (default: 20)
  --parallel N         Parallel workers for extract/eval (default: 6)
  --mode MODE          full (default), ingest, or eval
  --no-tier5           Skip Tier-5 EI eval (faster smoke runs)
  --skip-preflight     Skip pre-flight checks
  -h, --help           Show this help
USAGE
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scale) SCALE="$2"; shift 2 ;;
    --run-number) RUN_NUMBER="$2"; shift 2 ;;
    --resume) RESUME=true; shift ;;
    --notify) NOTIFY=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --max-sessions) MAX_SESSIONS="$2"; shift 2 ;;
    --parallel) PARALLEL="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --no-tier5) TIER5=false; shift ;;
    --skip-preflight) SKIP_PREFLIGHT=true; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# Validate required args
if [[ -z "$SCALE" ]]; then
  echo "ERROR: --scale is required (s or l)"
  exit 1
fi
if [[ "$SCALE" != "s" && "$SCALE" != "l" ]]; then
  echo "ERROR: --scale must be 's' or 'l'"
  exit 1
fi
if [[ -z "$RUN_NUMBER" ]]; then
  echo "ERROR: --run-number is required"
  exit 1
fi
if ! [[ "$RUN_NUMBER" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --run-number must be a non-negative integer"
  exit 1
fi
if [[ "$MODE" != "full" && "$MODE" != "ingest" && "$MODE" != "eval" ]]; then
  echo "ERROR: --mode must be one of: full, ingest, eval"
  exit 1
fi
if [[ "$BACKEND" != "claude-code" && "$BACKEND" != "api" && "$BACKEND" != "vllm" ]]; then
  echo "ERROR: --backend must be one of: claude-code, api, vllm"
  exit 1
fi
if ! [[ "$MAX_SESSIONS" =~ ^[0-9]+$ ]] || [[ "$MAX_SESSIONS" -lt 1 ]]; then
  echo "ERROR: --max-sessions must be a positive integer"
  exit 1
fi
if ! [[ "$PARALLEL" =~ ^[0-9]+$ ]] || [[ "$PARALLEL" -lt 1 ]]; then
  echo "ERROR: --parallel must be a positive integer"
  exit 1
fi
for token_name in MODEL EVAL_MODEL JUDGE; do
  token_value="${!token_name:-}"
  if [[ ! "$token_value" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: $token_name contains invalid characters: $token_value" >&2
    exit 1
  fi
done

# Generate run name
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RUN_NAME="quaid-${SCALE}-r${RUN_NUMBER}-${TIMESTAMP}"
RUN_DIR="runs/${RUN_NAME}"

echo "=== Launch Benchmark ==="
echo "  Scale:       AL-$(echo "$SCALE" | tr 'a-z' 'A-Z')"
echo "  Run name:    $RUN_NAME"
echo "  Host:        $SPARK_HOST"
echo "  Backend:     $BACKEND"
echo "  Model:       $MODEL"
echo "  Eval model:  $EVAL_MODEL"
echo "  Judge:       $JUDGE"
echo "  Max sessions: $MAX_SESSIONS"
echo "  Parallel:    $PARALLEL"
echo "  Mode:        $MODE"
echo "  Tier 5:      $TIER5"
echo "  Resume:      $RESUME"
echo ""

# --- Pre-flight checks ---
if ! $SKIP_PREFLIGHT; then
  echo "--- Running pre-flight checks ---"
  if ! "$SCRIPT_DIR/preflight-check.sh" --scale "$SCALE" --backend "$BACKEND"; then
    echo ""
    echo "Pre-flight failed. Fix issues or use --skip-preflight to bypass."
    exit 1
  fi
  echo ""
fi

# --- Check for name collision ---
echo "--- Checking for run name collision ---"

# Helper for remote commands
remote() {
  if [[ "$SPARK_HOST" == "localhost" ]]; then
    bash -c "$1"
  else
    ssh -o ConnectTimeout=10 "$SPARK_HOST" "$1"
  fi
}

# Check all possible locations for collisions
COLLISION=false
for check_dir in "runs" "runs/successful-runs" "runs/failed-runs"; do
  if remote "test -d '$SPARK_BENCHMARK_ROOT/$check_dir/$RUN_NAME'" 2>/dev/null; then
    echo "FATAL: Run name collision — $check_dir/$RUN_NAME already exists on $SPARK_HOST"
    COLLISION=true
  fi
done
if $COLLISION; then
  echo "Choose a different --run-number or wait for a new timestamp."
  exit 1
fi
echo "  No collision detected."
echo ""

# --- Build launch command ---
HARNESS_ARGS=(
  python3
  agentlife/eval/run_production_benchmark.py
  --mode "$MODE"
  --backend "$BACKEND"
  --model "$MODEL"
  --eval-model "$EVAL_MODEL"
  --judge "$JUDGE"
)
if $TIER5; then
  HARNESS_ARGS+=(--tier5)
else
  HARNESS_ARGS+=(--no-tier5)
fi
HARNESS_ARGS+=(--max-sessions "$MAX_SESSIONS")

# Backward compatibility for older runner builds that do not expose --parallel.
RUNNER_HELP=$(remote "cd '$SPARK_BENCHMARK_ROOT' && python3 agentlife/eval/run_production_benchmark.py --help 2>/dev/null || true")
if printf "%s" "$RUNNER_HELP" | grep -q -- "--parallel"; then
  HARNESS_ARGS+=(--parallel "$PARALLEL")
fi

HARNESS_ARGS+=(--results-dir "$RUN_DIR")

if [[ "$SCALE" == "l" ]]; then
  HARNESS_ARGS+=(--filler-dir "$SPARK_FILLER_DIR")
fi

if $RESUME; then
  HARNESS_ARGS+=(--resume-extraction --resume-eval)
fi
printf -v HARNESS_CMD '%q ' "${HARNESS_ARGS[@]}"
HARNESS_CMD="${HARNESS_CMD% }"

# Full remote command with env vars
ENV_PREAMBLE="cd $(printf %q "$SPARK_BENCHMARK_ROOT")"
ENV_PREAMBLE+=" && source $(printf %q "$SPARK_ENV_FILE")"
ENV_PREAMBLE+=" && export OPENAI_API_KEY"

FULL_CMD="$ENV_PREAMBLE && nohup env"
FULL_CMD+=" BENCHMARK_PARALLEL=$(printf %q "$PARALLEL")"
FULL_CMD+=" BENCHMARK_LIFECYCLE_PREPASS_WORKERS=$(printf %q "$PARALLEL")"
FULL_CMD+=" BENCHMARK_TIMEOUT_MINUTES=$(printf %q "${BENCHMARK_TIMEOUT_MINUTES:-120}")"
FULL_CMD+=" BENCHMARK_MAX_BUFFER_TOKENS=$(printf %q "${BENCHMARK_MAX_BUFFER_TOKENS:-8000}")"
FULL_CMD+=" BENCHMARK_TARGET_CHUNK_TOKENS=$(printf %q "${BENCHMARK_TARGET_CHUNK_TOKENS:-6000}")"
FULL_CMD+=" BENCHMARK_MAX_CHUNK_CHARS=$(printf %q "${BENCHMARK_MAX_CHUNK_CHARS:-40000}")"
FULL_CMD+=" ANTHROPIC_RETRY_ATTEMPTS=$(printf %q "${ANTHROPIC_RETRY_ATTEMPTS:-12}")"
FULL_CMD+=" ANTHROPIC_RETRY_BACKOFF_S=$(printf %q "${ANTHROPIC_RETRY_BACKOFF_S:-5}")"
FULL_CMD+=" ANTHROPIC_RETRY_BACKOFF_CAP_S=$(printf %q "${ANTHROPIC_RETRY_BACKOFF_CAP_S:-120}")"
FULL_CMD+=" JANITOR_TIMEOUT_S=$(printf %q "$JANITOR_TIMEOUT_S")"
FULL_CMD+=" ANTHROPIC_TIMEOUT_S=$(printf %q "$ANTHROPIC_TIMEOUT_S")"
FULL_CMD+=" CLAUDE_CODE_TIMEOUT_S=$(printf %q "$CLAUDE_CODE_TIMEOUT_S")"
FULL_CMD+=" CLAUDE_CODE_FAST_TIMEOUT_S=$(printf %q "$CLAUDE_CODE_FAST_TIMEOUT_S")"
FULL_CMD+=" CLAUDE_CODE_DEEP_TIMEOUT_S=$(printf %q "$CLAUDE_CODE_DEEP_TIMEOUT_S")"
FULL_CMD+=" CLAUDE_CODE_TIMEOUT_CAP_S=$(printf %q "$CLAUDE_CODE_TIMEOUT_CAP_S")"
FULL_CMD+=" OPENAI_JUDGE_BACKEND=api"
FULL_CMD+=" OPENAI_API_KEY=\$OPENAI_API_KEY"
# Keep Anthropic key available for mixed/HyDE fast-tier paths.
FULL_CMD+=" ANTHROPIC_API_KEY=\${ANTHROPIC_API_KEY:-}"
FULL_CMD+=" ANTHROPIC_AUTH_TOKEN=\${ANTHROPIC_AUTH_TOKEN:-}"
FULL_CMD+=" CLAUDE_CODE_OAUTH_TOKEN=\${CLAUDE_CODE_OAUTH_TOKEN:-}"
FULL_CMD+=" AGENTLIFE_ASSETS_DIR=$(printf %q "$SPARK_ASSETS_DIR")"
FULL_CMD+=" $HARNESS_CMD"
FULL_CMD+=" > $(printf %q "$RUN_DIR.launch.log") 2>&1 &"

echo "--- Launch command ---"
echo ""
DISPLAY_CMD="${FULL_CMD//OPENAI_API_KEY=\\\$OPENAI_API_KEY/OPENAI_API_KEY=<redacted>}"
echo "ssh $SPARK_HOST \"$DISPLAY_CMD\""
echo ""

if $DRY_RUN; then
  echo "(Dry run — command not executed.)"
  exit 0
fi

# --- Execute ---
echo "--- Launching on $SPARK_HOST ---"

# Create run dir first to ensure log file has a place
remote "mkdir -p '$SPARK_BENCHMARK_ROOT/$RUN_DIR'"

# Launch
remote "$FULL_CMD"

echo "Launched. Waiting 10s for process startup..."
sleep 10

# --- Verify ---
echo "--- Post-launch verification ---"

PID=$(remote "ps -eo pid,cmd 2>/dev/null | awk -v runname='$RUN_NAME' 'index(\$0,\"run_production_benchmark.py\") && index(\$0,runname) {print \$1; exit}'" 2>/dev/null || echo "")
PID=$(echo "$PID" | tr -d ' ' | head -1)

if [[ -n "$PID" ]]; then
  echo "  [OK] Process alive: PID $PID"
else
  echo "  [WARN] Could not confirm PID. Check manually:"
  echo "    ssh $SPARK_HOST 'ps -eo pid,etime,cmd | grep run_production_benchmark | grep -v grep'"
fi

RUN_DIR_EXISTS=$(remote "test -d '$SPARK_BENCHMARK_ROOT/$RUN_DIR' && echo yes || echo no" 2>/dev/null)
if [[ "$RUN_DIR_EXISTS" == "yes" ]]; then
  echo "  [OK] Run directory created: $SPARK_BENCHMARK_ROOT/$RUN_DIR"
else
  echo "  [WARN] Run directory not yet created"
fi

LOG_PATH="$SPARK_BENCHMARK_ROOT/$RUN_DIR.launch.log"
echo ""
echo "=== Launch Summary ==="
echo "  Run name: $RUN_NAME"
echo "  PID:      ${PID:-unknown}"
echo "  Log:      $LOG_PATH"
echo "  Monitor:  ssh $SPARK_HOST 'tail -f $LOG_PATH'"
echo "  Progress: ssh $SPARK_HOST 'cat $SPARK_BENCHMARK_ROOT/$RUN_DIR/extraction_cache/progress.json'"

# --- Optional Telegram notification ---
if $NOTIFY; then
  MSG="Benchmark launched: $RUN_NAME (AL-$(echo "$SCALE" | tr 'a-z' 'A-Z'), ${BACKEND}, PID=${PID:-?})"
  if command -v clawdbot >/dev/null 2>&1; then
    clawdbot message send --channel telegram --target <telegram-id> -m "$MSG" 2>/dev/null && \
      echo "  Telegram notification sent." || \
      echo "  WARN: Telegram notification failed."
  else
    echo "  WARN: clawdbot CLI not found — skipping Telegram notification."
  fi
fi
