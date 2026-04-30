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
LOCAL_CHECKPOINT_ROOT="${LOCAL_CHECKPOINT_ROOT:-$HOME/quaidcode/benchmark-checkpoint}"
LOCAL_BENCH_CONFIG="${AGENTLIFE_BENCH_LOCAL_CONFIG:-$LOCAL_BENCH_ROOT/.agentlife-benchmark.local.json}"
LOCAL_SHARED_DEV_CONFIG="${HOME}/quaidcode/dev/.quaid-dev.local.json"

REMOTE=""
REMOTE_BENCH_ROOT=""
REMOTE_CHECKPOINT_ROOT=""
REMOTE_CHECKPOINT_PLUGIN_ROOT=""
REMOTE_WORKSPACE_ROOT=""
DRY_RUN=false
SKIP_LOCAL_CHECKS=false
PARALLEL="${BENCHMARK_PARALLEL:-6}"
SCALE="${BENCHMARK_SCALE:-s}"
DATASET="${BENCHMARK_DATASET:-canonical}"
RUN_NOTE=""
LOOSE_TIMEOUTS=false

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
  --remote-workspace-root PATH     Neutral remote root for fresh Quaid workspaces
                                  (default: ~/quaid-workspaces)
  --local-checkpoint-root PATH     Local checkpoint root
                                  (default: ~/quaidcode/benchmark-checkpoint)
  --dry-run                        Print actions without running rsync/ssh
  --skip-local-checks              Skip local compile/test gate before sync+launch
  --parallel N                     Parallel workers hint (default: 6)
  --scale s|l                      AgentLife scale for naming/env (default: s)
  --dataset canonical|jp           Dataset variant (default: canonical)
  --loose-timeouts                 Apply benchmark loose-timeout profile for local/provider-variance runs
  --note TEXT                      Short dashboard note for this run (optional)
  -h, --help                       Show help

Examples:
  ./scripts/launch-remote-benchmark.sh --remote spark -- --mode full --backend claude-code
  ./scripts/launch-remote-benchmark.sh --remote spark --dry-run -- --mode eval --results-dir runs/quaid-s-r500
  ./scripts/launch-remote-benchmark.sh --remote spark --scale l -- --mode full --backend oauth
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote) REMOTE="$2"; shift 2 ;;
    --remote-bench-root) REMOTE_BENCH_ROOT="$2"; shift 2 ;;
    --remote-checkpoint-root) REMOTE_CHECKPOINT_ROOT="$2"; shift 2 ;;
    --remote-plugin-root) REMOTE_CHECKPOINT_PLUGIN_ROOT="$2"; shift 2 ;;
    --remote-workspace-root) REMOTE_WORKSPACE_ROOT="$2"; shift 2 ;;
    --local-checkpoint-root) LOCAL_CHECKPOINT_ROOT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --skip-local-checks) SKIP_LOCAL_CHECKS=true; shift ;;
    --parallel) PARALLEL="$2"; shift 2 ;;
    --scale) SCALE="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --loose-timeouts) LOOSE_TIMEOUTS=true; shift ;;
    --note) RUN_NOTE="$2"; shift 2 ;;
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
if [[ "$SCALE" != "s" && "$SCALE" != "l" ]]; then
  echo "ERROR: --scale must be one of: s, l" >&2
  exit 1
fi
if [[ "$DATASET" != "canonical" && "$DATASET" != "jp" ]]; then
  echo "ERROR: --dataset must be one of: canonical, jp" >&2
  exit 1
fi

if $LOOSE_TIMEOUTS; then
  # Local/self-hosted lanes can exceed prod-style request budgets.
  # Disable per-request timeout by default (0 => no timeout) and guard the
  # run with a large stall watchdog instead of premature request aborts.
  : "${OPENAI_COMPAT_ANSWER_TIMEOUT_S:=0}"
  : "${OPENAI_COMPAT_RETRY_ATTEMPTS:=3}"
  : "${BENCHMARK_EVAL_STALL_FAIL_S:=2400}"
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
if [[ -z "$REMOTE_WORKSPACE_ROOT" ]]; then
  REMOTE_WORKSPACE_ROOT="~/quaid-workspaces"
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
WORKSPACE_BASENAME=""
AUTO_RESULTS_DIR=false
RUNNER_KIND="production"
LOCAL_RUNNER_PATH=""
for ((i=0; i<${#LAUNCH_ARGS[@]}; i++)); do
  if [[ "${LAUNCH_ARGS[$i]}" == "--results-dir" ]] && (( i + 1 < ${#LAUNCH_ARGS[@]} )); then
    RESULTS_DIR="${LAUNCH_ARGS[$((i+1))]}"
    break
  fi
done
if [[ -z "$RESULTS_DIR" ]]; then
  AUTO_RESULTS_DIR=true
fi

launch_args_include_flag() {
  local flag="$1"
  local arg
  for arg in "${LAUNCH_ARGS[@]}"; do
    if [[ "$arg" == "$flag" ]]; then
      return 0
    fi
  done
  return 1
}

resolve_local_runner() {
  if [[ "$RUNNER_KIND" == "vm" ]]; then
    if [[ -f "$LOCAL_BENCH_ROOT/eval/vm_benchmark.py" ]]; then
      LOCAL_RUNNER_PATH="$LOCAL_BENCH_ROOT/eval/vm_benchmark.py"
    elif [[ -f "$LOCAL_BENCH_ROOT/agentlife/eval/vm_benchmark.py" ]]; then
      LOCAL_RUNNER_PATH="$LOCAL_BENCH_ROOT/agentlife/eval/vm_benchmark.py"
    else
      echo "ERROR: vm_benchmark.py not found under eval/ or agentlife/eval/" >&2
      exit 1
    fi
  else
    if [[ -f "$LOCAL_BENCH_ROOT/eval/run_production_benchmark.py" ]]; then
      LOCAL_RUNNER_PATH="$LOCAL_BENCH_ROOT/eval/run_production_benchmark.py"
    elif [[ -f "$LOCAL_BENCH_ROOT/agentlife/eval/run_production_benchmark.py" ]]; then
      LOCAL_RUNNER_PATH="$LOCAL_BENCH_ROOT/agentlife/eval/run_production_benchmark.py"
    else
      echo "ERROR: run_production_benchmark.py not found under eval/ or agentlife/eval/" >&2
      exit 1
    fi
  fi
}

if launch_args_include_flag "--system" \
  || launch_args_include_flag "--vm-name" \
  || launch_args_include_flag "--vm-ip" \
  || launch_args_include_flag "--tart-host"; then
  RUNNER_KIND="vm"
fi
resolve_local_runner

run_cmd() {
  echo "+ $*"
  if ! $DRY_RUN; then
    "$@"
  fi
}

run_cmd_redacted() {
  local display="$1"
  shift
  echo "+ $display"
  if ! $DRY_RUN; then
    "$@"
  fi
}

resolve_local_config_secret_path() {
  local section="$1"
  local key_name="$2"
  python3 - "$section" "$key_name" "$LOCAL_BENCH_CONFIG" "$LOCAL_SHARED_DEV_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

section = sys.argv[1]
key_name = sys.argv[2]
candidate_paths = [Path(p).expanduser() for p in sys.argv[3:]]
alias_map = {
    "firstKeyPath": ("firstKeyPath", "primaryKeyPath"),
    "secondKeyPath": ("secondKeyPath", "secondaryKeyPath"),
    "thirdKeyPath": ("thirdKeyPath",),
    "solKeyPath": ("solKeyPath",),
    "yuniKeyPath": ("yuniKeyPath",),
}

for cfg_path in candidate_paths:
    if not cfg_path.is_file():
        continue
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        continue

    paths = cfg.get("paths") if isinstance(cfg.get("paths"), dict) else {}
    auth = cfg.get("auth") if isinstance(cfg.get("auth"), dict) else {}
    section_cfg = auth.get(section) if isinstance(auth.get(section), dict) else {}
    raw_path = ""
    for candidate_key in alias_map.get(key_name, (key_name,)):
        raw_path = str(section_cfg.get(candidate_key) or "").strip()
        if raw_path:
            break
    if not raw_path:
        continue

    root_key = "benchRoot" if "benchRoot" in paths else "devRoot"
    base_root = str(paths.get(root_key) or ".").strip() or "."
    base_root_path = Path(base_root)
    if not base_root_path.is_absolute():
        base_root_path = (cfg_path.parent / base_root_path).resolve()
    else:
        base_root_path = base_root_path.expanduser().resolve()

    secret_path = Path(raw_path).expanduser()
    if not secret_path.is_absolute():
        secret_path = (base_root_path / secret_path).resolve()
    else:
        secret_path = secret_path.resolve()

    print(secret_path)
    raise SystemExit(0)
PY
}

read_codex_token_file() {
  local token_path="$1"
  python3 - "$token_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]).expanduser()
raw = path.read_text(encoding="utf-8").strip()
if not raw:
    raise SystemExit(1)
if raw.startswith("{"):
    try:
        data = json.loads(raw)
    except Exception:
        print(raw)
        raise SystemExit(0)
    access = str(data.get("access") or data.get("token") or "").strip()
    if not access:
        tokens = data.get("tokens")
        if isinstance(tokens, dict):
            access = str(tokens.get("access_token") or "").strip()
    if access:
        print(access)
        raise SystemExit(0)
print(raw)
PY
}

write_normalized_codex_profile() {
  local source_path="${1:-}"
  local direct_token="${2:-}"
  python3 - "$source_path" "$direct_token" <<'PY'
import base64
import json
import sys
from pathlib import Path

source_path = sys.argv[1].strip()
direct_token = sys.argv[2].strip()


def decode_claims(token: str) -> dict:
    parts = str(token or "").split(".")
    if len(parts) < 2:
        return {}
    payload = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        raw = base64.urlsafe_b64decode(payload.encode("ascii"))
        data = json.loads(raw.decode("utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def emit_profile(access: str, refresh: str = "", account_id: str = "", expires=None) -> None:
    profile = {
        "type": "oauth",
        "provider": "openai-codex",
        "access": access,
    }
    if refresh:
        profile["refresh"] = refresh
    if account_id:
        profile["accountId"] = account_id
    if isinstance(expires, int) and expires > 0:
        profile["expires"] = expires
    claims = decode_claims(access)
    if "expires" not in profile:
        exp = claims.get("exp")
        if isinstance(exp, int) and exp > 0:
            profile["expires"] = exp * 1000
    if "accountId" not in profile:
        auth = claims.get("https://api.openai.com/auth")
        if isinstance(auth, dict):
            account_id = str(auth.get("chatgpt_account_id") or "").strip()
            if account_id:
                profile["accountId"] = account_id
    print(json.dumps(profile))
    raise SystemExit(0)


if direct_token:
    emit_profile(direct_token)

if source_path:
    raw = Path(source_path).expanduser().read_text(encoding="utf-8").strip()
    if raw:
        if raw.startswith("{"):
            try:
                data = json.loads(raw)
            except Exception:
                emit_profile(raw)
            if isinstance(data, dict):
                access = str(data.get("access") or data.get("token") or "").strip()
                refresh = str(data.get("refresh") or "").strip()
                account_id = str(data.get("accountId") or "").strip()
                expires = data.get("expires")
                if access:
                    emit_profile(access, refresh=refresh, account_id=account_id, expires=expires)
                tokens = data.get("tokens")
                if isinstance(tokens, dict):
                    access = str(tokens.get("access_token") or "").strip()
                    refresh = str(tokens.get("refresh_token") or "").strip()
                    account_id = str(tokens.get("account_id") or "").strip()
                    if access:
                        emit_profile(access, refresh=refresh, account_id=account_id)
        emit_profile(raw)

raise SystemExit(1)
PY
}

run_local_checks() {
  echo "--- Local harness checks (required) ---"
  run_cmd python3 -m py_compile "$LOCAL_RUNNER_PATH"
  run_cmd bash -lc "cd '$LOCAL_BENCH_ROOT' && env \
    -u BENCHMARK_QUERY_NUMS \
    -u BENCHMARK_QUERY_SHA1S \
    -u BENCHMARK_QUERY_PROFILE \
    -u BENCHMARK_QUERY_PROFILE_SIZE \
    -u BENCHMARK_QUERY_PROFILE_MIN_PER_TYPE \
    -u BENCHMARK_REQUIRE_QUERY_COUNT \
    -u BENCHMARK_MAX_QUERIES \
    -u BENCHMARK_EVAL_CONTEXT_PROFILE \
    -u BENCHMARK_SKIP_CONTEXT_PREFLIGHT \
    -u BENCHMARK_DEEP_REASONING_MODEL \
    -u BENCHMARK_FAST_REASONING_MODEL \
    -u BENCHMARK_OPENAI_MODEL \
    -u BENCHMARK_OPENAI_JUDGE_MODEL \
    pytest -q eval/tests/test_benchmark_regressions.py eval/tests/test_store_edge_retry.py"
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
echo "Remote workspace root:      $REMOTE_WORKSPACE_ROOT"
echo "Parallel workers:           $PARALLEL"
echo "Loose timeouts:             $LOOSE_TIMEOUTS"
echo "Run note:                   ${RUN_NOTE:-<empty>}"
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
  WORKSPACE_BASENAME="personal-workspace-${ts}"
  if $DRY_RUN; then
    RESULTS_DIR="runs/quaid-${SCALE}-r000-${ts}"
  else
    next_run="$(
      ssh "${SSH_OPTS[@]}" "$REMOTE" '
        set -euo pipefail
        mkdir -p ~/agentlife-benchmark/runs
        ls -1 ~/agentlife-benchmark/runs 2>/dev/null \
          | sed -n "s/^quaid-[sl]-r\\([0-9]\\+\\)-.*/\\1/p" \
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
    RESULTS_DIR="runs/quaid-${SCALE}-${run_id}-${ts}"
  fi
fi
RESULTS_BASENAME="$(basename "$RESULTS_DIR")"
if [[ -z "$WORKSPACE_BASENAME" ]]; then
  WORKSPACE_BASENAME="$RESULTS_BASENAME"
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
  "mkdir -p $REMOTE_BENCH_ROOT $REMOTE_CHECKPOINT_ROOT $REMOTE_CHECKPOINT_PLUGIN_ROOT $REMOTE_WORKSPACE_ROOT"

echo ""
echo "--- 1b) Sync fresh Claude credentials for OAuth-backed runs ---"
BACKEND_ARG=""
for ((i=0; i<${#LAUNCH_ARGS[@]}; i++)); do
  if [[ "${LAUNCH_ARGS[$i]}" == "--backend" ]] && (( i + 1 < ${#LAUNCH_ARGS[@]} )); then
    BACKEND_ARG="${LAUNCH_ARGS[$((i+1))]}"
    break
  fi
done
OPTIONAL_BENCH_ENV=""
LOCAL_CODEX_TOKEN_PATH=""
LOCAL_CODEX_TEMP_PATH=""
LOCAL_CODEX_TOKEN=""
LOCAL_CODEX_PROFILE_JSON=""
if [[ -n "${BENCHMARK_CODEX_TOKEN_PATH:-}" ]]; then
  LOCAL_CODEX_TOKEN_PATH="$(python3 - <<'PY'
from pathlib import Path
import os
print(Path(os.environ["BENCHMARK_CODEX_TOKEN_PATH"]).expanduser().resolve())
PY
)"
elif [[ -f "$HOME/.codex/auth.json" ]]; then
  LOCAL_CODEX_TOKEN_PATH="$HOME/.codex/auth.json"
elif [[ -n "${BENCHMARK_CODEX_API_KEY:-}" ]]; then
  LOCAL_CODEX_TOKEN="$BENCHMARK_CODEX_API_KEY"
else
  LOCAL_CODEX_TOKEN_PATH="$(resolve_local_config_secret_path codex solKeyPath || true)"
  if [[ -z "$LOCAL_CODEX_TOKEN_PATH" || ! -f "$LOCAL_CODEX_TOKEN_PATH" ]]; then
    LOCAL_CODEX_TOKEN_PATH="$(resolve_local_config_secret_path codex yuniKeyPath || true)"
  fi
fi
if [[ -z "$LOCAL_CODEX_TOKEN" && -n "$LOCAL_CODEX_TOKEN_PATH" && -f "$LOCAL_CODEX_TOKEN_PATH" ]]; then
  LOCAL_CODEX_TOKEN="$(read_codex_token_file "$LOCAL_CODEX_TOKEN_PATH" | tr -d '\r\n')"
fi
if [[ -n "$LOCAL_CODEX_TOKEN_PATH" && -f "$LOCAL_CODEX_TOKEN_PATH" ]]; then
  LOCAL_CODEX_PROFILE_JSON="$(write_normalized_codex_profile "$LOCAL_CODEX_TOKEN_PATH" "" || true)"
elif [[ -n "$LOCAL_CODEX_TOKEN" ]]; then
  LOCAL_CODEX_PROFILE_JSON="$(write_normalized_codex_profile "" "$LOCAL_CODEX_TOKEN" || true)"
fi
if [[ -n "$LOCAL_CODEX_PROFILE_JSON" ]]; then
  LOCAL_CODEX_TOKEN_PATH="$(python3 - <<'PY'
import tempfile
handle = tempfile.NamedTemporaryFile(prefix="benchmark-codex-auth.", suffix=".json", delete=False)
handle.close()
print(handle.name)
PY
)"
  LOCAL_CODEX_TEMP_PATH="$LOCAL_CODEX_TOKEN_PATH"
  printf '%s\n' "$LOCAL_CODEX_PROFILE_JSON" > "$LOCAL_CODEX_TOKEN_PATH"
  chmod 600 "$LOCAL_CODEX_TOKEN_PATH"
fi
if [[ -n "$LOCAL_CODEX_TEMP_PATH" ]]; then
  trap 'rm -f "$LOCAL_CODEX_TEMP_PATH"' EXIT
fi
if [[ "$BACKEND_ARG" == "codex" && -n "$LOCAL_CODEX_TOKEN" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_CODEX_API_KEY=$(printf %q "$LOCAL_CODEX_TOKEN")"$'\n'
fi
LOCAL_CLAUDE_CREDS="$HOME/.claude/.credentials.json"
if [[ -f "$LOCAL_CLAUDE_CREDS" ]]; then
  run_cmd ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p ~/.claude"
  run_cmd scp -p "$LOCAL_CLAUDE_CREDS" "$REMOTE:~/.claude/.credentials.json"
  echo "  synced ~/.claude/.credentials.json to $REMOTE"
  if [[ "$BACKEND_ARG" == "claude-code" ]]; then
    echo "  backend=claude-code will use synced Claude OAuth credentials"
  elif [[ "$BACKEND_ARG" == "vllm" || "$BACKEND_ARG" == "llama-cpp" || "$BACKEND_ARG" == "codex" || "$BACKEND_ARG" == "openai" ]]; then
    echo "  backend=${BACKEND_ARG} does not require synced Claude OAuth credentials"
  else
    echo "  backend=${BACKEND_ARG:-api} can use synced Claude OAuth credentials for direct API runs"
  fi
else
  echo "  local ~/.claude/.credentials.json not found; skipping credential sync"
fi
if [[ -n "$LOCAL_CODEX_TOKEN_PATH" && -f "$LOCAL_CODEX_TOKEN_PATH" ]]; then
  echo ""
  echo "--- 1c) Sync benchmark Codex OAuth profile for remote VM answer lanes ---"
  run_cmd ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p ~/.codex"
  run_cmd scp -p "$LOCAL_CODEX_TOKEN_PATH" "$REMOTE:~/.codex/benchmark-auth.json"
  OPTIONAL_BENCH_ENV+="export BENCHMARK_CODEX_TOKEN_PATH=~/.codex/benchmark-auth.json"$'\n'
  echo "  synced benchmark Codex OAuth profile to $REMOTE:~/.codex/benchmark-auth.json"
fi

echo ""
echo "--- 2) Sync canonical benchmark repo ---"
RSYNC_COMMON=(
  -az
  --delete
  --itemize-changes
  --exclude='.git'
  --exclude='.env'
  --exclude='.agentlife-benchmark.local.json'
  --exclude='node_modules'
  --exclude='.ruff_cache'
  --exclude='.venv'
  --exclude='.pytest_cache'
  --exclude='.pytest-home'
  --exclude='pytest-home'
  --exclude='__pycache__'
  --exclude='*.pyc'
  --exclude='.DS_Store'
  --exclude='runs/'
  --exclude='release/'
  --exclude='recovered-from-spark-*'
  --exclude='*.db'
  --exclude='*.db-shm'
  --exclude='*.db-wal'
)
run_cmd rsync "${RSYNC_COMMON[@]}" "$LOCAL_BENCH_ROOT/" "$REMOTE:$REMOTE_BENCH_ROOT/"

echo ""
echo "--- 2b) Remove local-only benchmark artifacts from remote root ---"
run_cmd ssh "${SSH_OPTS[@]}" "$REMOTE" \
  "rm -f $REMOTE_BENCH_ROOT/.agentlife-benchmark.local.json $REMOTE_BENCH_ROOT/.env && rm -rf $REMOTE_BENCH_ROOT/release $REMOTE_BENCH_ROOT/recipe-app $REMOTE_BENCH_ROOT/portfolio-site"

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
if [[ -n "${BENCHMARK_ANTHROPIC_OAUTH_TOKEN:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_ANTHROPIC_OAUTH_TOKEN=$(printf %q "$BENCHMARK_ANTHROPIC_OAUTH_TOKEN")"$'\n'
else
  LOCAL_BENCHMARK_PRIMARY_KEY_PATH="$(resolve_local_config_secret_path anthropic firstKeyPath || true)"
  if [[ -n "$LOCAL_BENCHMARK_PRIMARY_KEY_PATH" && -f "$LOCAL_BENCHMARK_PRIMARY_KEY_PATH" ]]; then
    LOCAL_BENCHMARK_OAUTH_TOKEN="$(tr -d '[:space:]' < "$LOCAL_BENCHMARK_PRIMARY_KEY_PATH")"
  else
    LOCAL_BENCHMARK_OAUTH_TOKEN=""
  fi
  if [[ -n "$LOCAL_BENCHMARK_OAUTH_TOKEN" ]]; then
    OPTIONAL_BENCH_ENV+="export BENCHMARK_ANTHROPIC_OAUTH_TOKEN=$(printf %q "$LOCAL_BENCHMARK_OAUTH_TOKEN")"$'\n'
  fi
fi
if [[ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export CLAUDE_CODE_OAUTH_TOKEN=$(printf %q "$CLAUDE_CODE_OAUTH_TOKEN")"$'\n'
fi
if [[ -n "${BENCHMARK_MAX_QUERIES:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_MAX_QUERIES=$(printf %q "$BENCHMARK_MAX_QUERIES")"$'\n'
fi
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export OPENAI_API_KEY=$(printf %q "$OPENAI_API_KEY")"$'\n'
else
  LOCAL_OPENAI_KEY_PATH="$(resolve_local_config_secret_path openai judgeKeyPath || true)"
  if [[ -z "$LOCAL_OPENAI_KEY_PATH" ]]; then
    LOCAL_OPENAI_KEY_PATH="$(resolve_local_config_secret_path openai keyPath || true)"
  fi
  if [[ -n "$LOCAL_OPENAI_KEY_PATH" && -f "$LOCAL_OPENAI_KEY_PATH" ]]; then
    LOCAL_OPENAI_KEY="$(tr -d '\r\n' < "$LOCAL_OPENAI_KEY_PATH")"
  else
    LOCAL_OPENAI_KEY=""
  fi
  if [[ -n "$LOCAL_OPENAI_KEY" ]]; then
    OPTIONAL_BENCH_ENV+="export OPENAI_API_KEY=$(printf %q "$LOCAL_OPENAI_KEY")"$'\n'
  fi
fi
if [[ -n "${BENCHMARK_REQUIRE_QUERY_COUNT:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_REQUIRE_QUERY_COUNT=$(printf %q "$BENCHMARK_REQUIRE_QUERY_COUNT")"$'\n'
fi
if [[ -n "${BENCHMARK_ALLOW_NON_HAIKU_ANSWER_MODEL:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_ALLOW_NON_HAIKU_ANSWER_MODEL=$(printf %q "$BENCHMARK_ALLOW_NON_HAIKU_ANSWER_MODEL")"$'\n'
fi
if [[ -n "${BENCHMARK_EVAL_PARALLEL:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_EVAL_PARALLEL=$(printf %q "$BENCHMARK_EVAL_PARALLEL")"$'\n'
fi
if [[ -n "${OPENAI_COMPAT_ANSWER_TIMEOUT_S:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export OPENAI_COMPAT_ANSWER_TIMEOUT_S=$(printf %q "$OPENAI_COMPAT_ANSWER_TIMEOUT_S")"$'\n'
fi
if [[ -n "${OPENAI_COMPAT_RETRY_ATTEMPTS:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export OPENAI_COMPAT_RETRY_ATTEMPTS=$(printf %q "$OPENAI_COMPAT_RETRY_ATTEMPTS")"$'\n'
fi
if [[ -n "${BENCHMARK_EVAL_STALL_FAIL_S:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_EVAL_STALL_FAIL_S=$(printf %q "$BENCHMARK_EVAL_STALL_FAIL_S")"$'\n'
fi
if [[ -n "${OPENAI_COMPAT_TRACE:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export OPENAI_COMPAT_TRACE=$(printf %q "$OPENAI_COMPAT_TRACE")"$'\n'
fi
if [[ -n "${OPENAI_COMPAT_TRACE_INTERVAL_S:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export OPENAI_COMPAT_TRACE_INTERVAL_S=$(printf %q "$OPENAI_COMPAT_TRACE_INTERVAL_S")"$'\n'
fi
if [[ -n "${BENCHMARK_OPENAI_URL:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_OPENAI_URL=$(printf %q "$BENCHMARK_OPENAI_URL")"$'\n'
fi
if [[ -n "${BENCHMARK_OPENAI_MODEL:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_OPENAI_MODEL=$(printf %q "$BENCHMARK_OPENAI_MODEL")"$'\n'
fi
if [[ -n "${BENCHMARK_OPENAI_RUNTIME_URL:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_OPENAI_RUNTIME_URL=$(printf %q "$BENCHMARK_OPENAI_RUNTIME_URL")"$'\n'
fi
if [[ -n "${BENCHMARK_OPENAI_RUNTIME_MODEL:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_OPENAI_RUNTIME_MODEL=$(printf %q "$BENCHMARK_OPENAI_RUNTIME_MODEL")"$'\n'
fi
if [[ -n "${BENCHMARK_OPENAI_JUDGE_URL:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_OPENAI_JUDGE_URL=$(printf %q "$BENCHMARK_OPENAI_JUDGE_URL")"$'\n'
fi
if [[ -n "${BENCHMARK_OPENAI_JUDGE_MODEL:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_OPENAI_JUDGE_MODEL=$(printf %q "$BENCHMARK_OPENAI_JUDGE_MODEL")"$'\n'
fi
if [[ -n "${BENCHMARK_TIER5_JUDGE_THINKING:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_TIER5_JUDGE_THINKING=$(printf %q "$BENCHMARK_TIER5_JUDGE_THINKING")"$'\n'
fi
if [[ -n "${BENCHMARK_FAST_REASONING_MODEL:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_FAST_REASONING_MODEL=$(printf %q "$BENCHMARK_FAST_REASONING_MODEL")"$'\n'
fi
if [[ -n "${BENCHMARK_DEEP_REASONING_MODEL:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_DEEP_REASONING_MODEL=$(printf %q "$BENCHMARK_DEEP_REASONING_MODEL")"$'\n'
fi
if [[ -n "${BENCHMARK_CAPTURE_CHUNK_TOKENS:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_CAPTURE_CHUNK_TOKENS=$(printf %q "$BENCHMARK_CAPTURE_CHUNK_TOKENS")"$'\n'
fi
if [[ -n "${BENCHMARK_CAPTURE_CHUNK_MAX_LINES:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_CAPTURE_CHUNK_MAX_LINES=$(printf %q "$BENCHMARK_CAPTURE_CHUNK_MAX_LINES")"$'\n'
fi
if [[ -n "${BENCHMARK_EXTRACTION_PROMPT_APPENDIX:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_EXTRACTION_PROMPT_APPENDIX=$(printf %q "$BENCHMARK_EXTRACTION_PROMPT_APPENDIX")"$'\n'
fi
if [[ -n "${BENCHMARK_EVAL_CONTEXT_PROFILE:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_EVAL_CONTEXT_PROFILE=$(printf %q "$BENCHMARK_EVAL_CONTEXT_PROFILE")"$'\n'
fi
if [[ -n "${BENCHMARK_SKIP_CONTEXT_PREFLIGHT:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_SKIP_CONTEXT_PREFLIGHT=$(printf %q "$BENCHMARK_SKIP_CONTEXT_PREFLIGHT")"$'\n'
fi
if [[ -n "${BENCHMARK_QUERY_PROFILE:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_QUERY_PROFILE=$(printf %q "$BENCHMARK_QUERY_PROFILE")"$'\n'
fi
if [[ -n "${BENCHMARK_QUERY_PROFILE_SIZE:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_QUERY_PROFILE_SIZE=$(printf %q "$BENCHMARK_QUERY_PROFILE_SIZE")"$'\n'
fi
if [[ -n "${BENCHMARK_QUERY_PROFILE_MIN_PER_TYPE:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_QUERY_PROFILE_MIN_PER_TYPE=$(printf %q "$BENCHMARK_QUERY_PROFILE_MIN_PER_TYPE")"$'\n'
fi
if [[ -n "${BENCHMARK_QUERY_NUMS:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_QUERY_NUMS=$(printf %q "$BENCHMARK_QUERY_NUMS")"$'\n'
fi
if [[ -n "${BENCHMARK_QUERY_SHA1S:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_QUERY_SHA1S=$(printf %q "$BENCHMARK_QUERY_SHA1S")"$'\n'
fi
if [[ -n "${BENCHMARK_DISABLE_PROJECT_DOCS:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_DISABLE_PROJECT_DOCS=$(printf %q "$BENCHMARK_DISABLE_PROJECT_DOCS")"$'\n'
fi
if [[ -n "${QUAID_RECALL_NON_GRAPH_LABELS:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export QUAID_RECALL_NON_GRAPH_LABELS=$(printf %q "$QUAID_RECALL_NON_GRAPH_LABELS")"$'\n'
fi
if [[ -n "${BENCHMARK_RECALL_FAST_TIMEOUT_S:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_RECALL_FAST_TIMEOUT_S=$(printf %q "$BENCHMARK_RECALL_FAST_TIMEOUT_S")"$'\n'
fi
if [[ -n "${BENCHMARK_RECALL_TIMEOUT_S:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_RECALL_TIMEOUT_S=$(printf %q "$BENCHMARK_RECALL_TIMEOUT_S")"$'\n'
fi
if [[ -n "${QUAID_RECALL_FAST_PLANNER_TIMEOUT_S:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export QUAID_RECALL_FAST_PLANNER_TIMEOUT_S=$(printf %q "$QUAID_RECALL_FAST_PLANNER_TIMEOUT_S")"$'\n'
fi
if [[ -n "${QUAID_RECALL_QUERY_PLANNER_TIMEOUT_S:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export QUAID_RECALL_QUERY_PLANNER_TIMEOUT_S=$(printf %q "$QUAID_RECALL_QUERY_PLANNER_TIMEOUT_S")"$'\n'
fi
if [[ -n "${QUAID_RECALL_DRILL_TIMEOUT_S:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export QUAID_RECALL_DRILL_TIMEOUT_S=$(printf %q "$QUAID_RECALL_DRILL_TIMEOUT_S")"$'\n'
fi
if [[ -n "${QUAID_RECALL_SUBQUERY_TIMEOUT_S:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export QUAID_RECALL_SUBQUERY_TIMEOUT_S=$(printf %q "$QUAID_RECALL_SUBQUERY_TIMEOUT_S")"$'\n'
fi
if [[ -n "${QUAID_TOOL_HINT_TIMEOUT_S:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export QUAID_TOOL_HINT_TIMEOUT_S=$(printf %q "$QUAID_TOOL_HINT_TIMEOUT_S")"$'\n'
fi
if [[ -n "${BENCHMARK_OBD_EXTRACT_TIMEOUT:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_OBD_EXTRACT_TIMEOUT=$(printf %q "$BENCHMARK_OBD_EXTRACT_TIMEOUT")"$'\n'
fi
if [[ -n "${BENCHMARK_OBD_DISABLE_CARRY_CONTEXT:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_OBD_DISABLE_CARRY_CONTEXT=$(printf %q "$BENCHMARK_OBD_DISABLE_CARRY_CONTEXT")"$'\n'
fi
if [[ -n "${BENCHMARK_OBD_PARALLEL_ROOT_WORKERS:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_OBD_PARALLEL_ROOT_WORKERS=$(printf %q "$BENCHMARK_OBD_PARALLEL_ROOT_WORKERS")"$'\n'
fi
if [[ -n "${BENCHMARK_OBD_CHUNK_TOKENS:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_OBD_CHUNK_TOKENS=$(printf %q "$BENCHMARK_OBD_CHUNK_TOKENS")"$'\n'
fi
if [[ -n "${QUAID_CAPTURE_CHUNK_MAX_LINES:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export QUAID_CAPTURE_CHUNK_MAX_LINES=$(printf %q "$QUAID_CAPTURE_CHUNK_MAX_LINES")"$'\n'
fi
if [[ -n "${BENCHMARK_OBD_CHUNK_MAX_LINES:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_OBD_CHUNK_MAX_LINES=$(printf %q "$BENCHMARK_OBD_CHUNK_MAX_LINES")"$'\n'
fi
if [[ -n "${BENCHMARK_EMBEDDINGS_PROVIDER:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_EMBEDDINGS_PROVIDER=$(printf %q "$BENCHMARK_EMBEDDINGS_PROVIDER")"$'\n'
fi
if [[ -n "${BENCHMARK_OLLAMA_URL:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_OLLAMA_URL=$(printf %q "$BENCHMARK_OLLAMA_URL")"$'\n'
fi
if [[ -n "${BENCHMARK_EMBEDDING_MODEL:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_EMBEDDING_MODEL=$(printf %q "$BENCHMARK_EMBEDDING_MODEL")"$'\n'
fi
if [[ -n "${BENCHMARK_EMBEDDING_DIM:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_EMBEDDING_DIM=$(printf %q "$BENCHMARK_EMBEDDING_DIM")"$'\n'
fi
if [[ -n "${BENCHMARK_VLLM_URL:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_VLLM_URL=$(printf %q "$BENCHMARK_VLLM_URL")"$'\n'
fi
if [[ -n "${BENCHMARK_VLLM_MODEL:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_VLLM_MODEL=$(printf %q "$BENCHMARK_VLLM_MODEL")"$'\n'
fi
if [[ -n "${BENCHMARK_VLLM_API_KEY:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_VLLM_API_KEY=$(printf %q "$BENCHMARK_VLLM_API_KEY")"$'\n'
fi
if [[ -n "${BENCHMARK_VLLM_RUNTIME_URL:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_VLLM_RUNTIME_URL=$(printf %q "$BENCHMARK_VLLM_RUNTIME_URL")"$'\n'
fi
if [[ -n "${BENCHMARK_VLLM_RUNTIME_MODEL:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_VLLM_RUNTIME_MODEL=$(printf %q "$BENCHMARK_VLLM_RUNTIME_MODEL")"$'\n'
fi
if [[ -n "${BENCHMARK_VLLM_RUNTIME_API_KEY:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_VLLM_RUNTIME_API_KEY=$(printf %q "$BENCHMARK_VLLM_RUNTIME_API_KEY")"$'\n'
fi
if [[ -n "${BENCHMARK_VLLM_JUDGE_URL:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_VLLM_JUDGE_URL=$(printf %q "$BENCHMARK_VLLM_JUDGE_URL")"$'\n'
fi
if [[ -n "${BENCHMARK_VLLM_JUDGE_MODEL:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_VLLM_JUDGE_MODEL=$(printf %q "$BENCHMARK_VLLM_JUDGE_MODEL")"$'\n'
fi
if [[ -n "${BENCHMARK_VLLM_JUDGE_API_KEY:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_VLLM_JUDGE_API_KEY=$(printf %q "$BENCHMARK_VLLM_JUDGE_API_KEY")"$'\n'
fi
if [[ -n "${BENCHMARK_LLAMA_CPP_URL:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_LLAMA_CPP_URL=$(printf %q "$BENCHMARK_LLAMA_CPP_URL")"$'\n'
fi
if [[ -n "${BENCHMARK_LLAMA_CPP_MODEL:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_LLAMA_CPP_MODEL=$(printf %q "$BENCHMARK_LLAMA_CPP_MODEL")"$'\n'
fi
if [[ -n "${BENCHMARK_LLAMA_CPP_API_KEY:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_LLAMA_CPP_API_KEY=$(printf %q "$BENCHMARK_LLAMA_CPP_API_KEY")"$'\n'
fi
if [[ -n "${BENCHMARK_LLAMA_CPP_RUNTIME_URL:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_LLAMA_CPP_RUNTIME_URL=$(printf %q "$BENCHMARK_LLAMA_CPP_RUNTIME_URL")"$'\n'
fi
if [[ -n "${BENCHMARK_LLAMA_CPP_RUNTIME_MODEL:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_LLAMA_CPP_RUNTIME_MODEL=$(printf %q "$BENCHMARK_LLAMA_CPP_RUNTIME_MODEL")"$'\n'
fi
if [[ -n "${BENCHMARK_LLAMA_CPP_RUNTIME_API_KEY:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_LLAMA_CPP_RUNTIME_API_KEY=$(printf %q "$BENCHMARK_LLAMA_CPP_RUNTIME_API_KEY")"$'\n'
fi
if [[ -n "${BENCHMARK_LLAMA_CPP_JUDGE_URL:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_LLAMA_CPP_JUDGE_URL=$(printf %q "$BENCHMARK_LLAMA_CPP_JUDGE_URL")"$'\n'
fi
if [[ -n "${BENCHMARK_LLAMA_CPP_JUDGE_MODEL:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_LLAMA_CPP_JUDGE_MODEL=$(printf %q "$BENCHMARK_LLAMA_CPP_JUDGE_MODEL")"$'\n'
fi
if [[ -n "${BENCHMARK_LLAMA_CPP_JUDGE_API_KEY:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_LLAMA_CPP_JUDGE_API_KEY=$(printf %q "$BENCHMARK_LLAMA_CPP_JUDGE_API_KEY")"$'\n'
fi
if [[ -n "${OPENAI_COMPATIBLE_DEEP_BASE_URL:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export OPENAI_COMPATIBLE_DEEP_BASE_URL=$(printf %q "$OPENAI_COMPATIBLE_DEEP_BASE_URL")"$'\n'
fi
if [[ -n "${OPENAI_COMPATIBLE_FAST_BASE_URL:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export OPENAI_COMPATIBLE_FAST_BASE_URL=$(printf %q "$OPENAI_COMPATIBLE_FAST_BASE_URL")"$'\n'
fi
if [[ -n "$RUN_NOTE" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_RUN_NOTE=$(printf %q "$RUN_NOTE")"$'\n'
fi
OPTIONAL_BENCH_ENV+="export BENCHMARK_SCALE=$(printf %q "$SCALE")"$'\n'
OPTIONAL_BENCH_ENV+="export BENCHMARK_DATASET=$(printf %q "$DATASET")"$'\n'
if [[ -n "${BENCHMARK_ASSETS_DIR:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_ASSETS_DIR_RAW=$(printf %q "$BENCHMARK_ASSETS_DIR")"$'\n'
elif [[ -n "${AGENTLIFE_ASSETS_DIR:-}" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_ASSETS_DIR_RAW=$(printf %q "$AGENTLIFE_ASSETS_DIR")"$'\n'
fi
if [[ "$SCALE" == "l" ]]; then
  OPTIONAL_BENCH_ENV+="export BENCHMARK_INCLUDE_FILLER=1"$'\n'
  if [[ -n "${BENCHMARK_FILLER_DIR:-}" ]]; then
    OPTIONAL_BENCH_ENV+="export BENCHMARK_FILLER_DIR_RAW=$(printf %q "$BENCHMARK_FILLER_DIR")"$'\n'
  fi
else
  OPTIONAL_BENCH_ENV+="export BENCHMARK_INCLUDE_FILLER=0"$'\n'
fi
REMOTE_PY_CMD="
set -euo pipefail
resolve_remote_path() {
  python3 - \"\$1\" <<'PY'
import os, sys
print(os.path.abspath(os.path.expanduser(os.path.expandvars(sys.argv[1]))))
PY
}
REMOTE_BENCH_ROOT_RESOLVED=\$(resolve_remote_path $(printf %q "$REMOTE_BENCH_ROOT"))
REMOTE_CHECKPOINT_ROOT_RESOLVED=\$(resolve_remote_path $(printf %q "$REMOTE_CHECKPOINT_ROOT"))
REMOTE_CHECKPOINT_PLUGIN_ROOT_RESOLVED=\$(resolve_remote_path $(printf %q "$REMOTE_CHECKPOINT_PLUGIN_ROOT"))
REMOTE_WORKSPACE_ROOT_RESOLVED=\$(resolve_remote_path $(printf %q "$REMOTE_WORKSPACE_ROOT"))
cd \"\$REMOTE_BENCH_ROOT_RESOLVED\"
if [[ -d \"\$REMOTE_CHECKPOINT_ROOT_RESOLVED/modules/quaid\" ]]; then
  export BENCHMARK_PLUGIN_DIR=\"\$REMOTE_CHECKPOINT_ROOT_RESOLVED/modules/quaid\"
elif [[ -d \"\$REMOTE_CHECKPOINT_ROOT_RESOLVED/plugins/quaid\" ]]; then
  export BENCHMARK_PLUGIN_DIR=\"\$REMOTE_CHECKPOINT_ROOT_RESOLVED/plugins/quaid\"
elif [[ -d \"\$REMOTE_CHECKPOINT_PLUGIN_ROOT_RESOLVED\" ]]; then
  export BENCHMARK_PLUGIN_DIR=\"\$REMOTE_CHECKPOINT_PLUGIN_ROOT_RESOLVED\"
fi
export BENCHMARK_PARALLEL=$(printf %q "$PARALLEL")
export BENCHMARK_LIFECYCLE_PREPASS_WORKERS=$(printf %q "$PARALLEL")
export BENCHMARK_JANITOR_LLM_WORKERS=$(printf %q "$PARALLEL")
export BENCHMARK_JANITOR_REVIEW_WORKERS=$(printf %q "$PARALLEL")
$OPTIONAL_BENCH_ENV
if [[ -n \"\${BENCHMARK_FILLER_DIR_RAW:-}\" ]]; then
  export BENCHMARK_FILLER_DIR=\$(resolve_remote_path \"\$BENCHMARK_FILLER_DIR_RAW\")
elif [[ \"\${BENCHMARK_INCLUDE_FILLER:-0}\" == \"1\" ]]; then
  if [[ \"\${BENCHMARK_DATASET:-canonical}\" == \"jp\" ]]; then
    export BENCHMARK_FILLER_DIR=\"\$REMOTE_BENCH_ROOT_RESOLVED/data/filler-sessions-jp\"
  else
    export BENCHMARK_FILLER_DIR=\"\$REMOTE_BENCH_ROOT_RESOLVED/data/filler-sessions\"
  fi
fi
if [[ -n \"\${BENCHMARK_ASSETS_DIR_RAW:-}\" ]]; then
  export BENCHMARK_ASSETS_DIR=\$(resolve_remote_path \"\$BENCHMARK_ASSETS_DIR_RAW\")
else
  if [[ \"\${BENCHMARK_DATASET:-canonical}\" == \"jp\" ]]; then
    export BENCHMARK_ASSETS_DIR=\"\$REMOTE_BENCH_ROOT_RESOLVED/data/sessions-jp\"
  else
    export BENCHMARK_ASSETS_DIR=\"\$REMOTE_BENCH_ROOT_RESOLVED/data/sessions\"
  fi
fi
export AGENTLIFE_ASSETS_DIR=\"\$BENCHMARK_ASSETS_DIR\"
echo \"Benchmark dataset: \${BENCHMARK_DATASET:-canonical} assets=\$BENCHMARK_ASSETS_DIR filler=\${BENCHMARK_FILLER_DIR:-none}\"
BENCHMARK_OAUTH_TOKEN="\${BENCHMARK_ANTHROPIC_OAUTH_TOKEN:-}"
if [[ $(printf %q "$BACKEND_ARG") == "vllm" || $(printf %q "$BACKEND_ARG") == "llama-cpp" || $(printf %q "$BACKEND_ARG") == "codex" || $(printf %q "$BACKEND_ARG") == "openai" ]]; then
  echo \"Benchmark Anthropic OAuth: not required for backend=$(printf %q "$BACKEND_ARG")\"
elif [[ -n \"\$BENCHMARK_OAUTH_TOKEN\" ]]; then
  export BENCHMARK_ANTHROPIC_OAUTH_TOKEN=\"\$BENCHMARK_OAUTH_TOKEN\"
  export ANTHROPIC_API_KEY=\"\$BENCHMARK_OAUTH_TOKEN\"
else
  echo \"ERROR: BENCHMARK_ANTHROPIC_OAUTH_TOKEN missing; set it explicitly or configure auth.anthropic.firstKeyPath in .agentlife-benchmark.local.json before launch\" >&2
  exit 1
fi
if [[ $(printf %q "$BACKEND_ARG") == "vllm" || $(printf %q "$BACKEND_ARG") == "llama-cpp" || $(printf %q "$BACKEND_ARG") == "codex" || $(printf %q "$BACKEND_ARG") == "openai" ]]; then
  echo \"Benchmark Anthropic OAuth: skipped\"
elif [[ -n \"\${BENCHMARK_ANTHROPIC_OAUTH_TOKEN:-}\" ]]; then
  echo \"Benchmark Anthropic OAuth: present\"
else
  echo \"Benchmark Anthropic OAuth: missing\"
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
RUNNER_KIND=$(printf %q "$RUNNER_KIND")
if [[ "$RUNNER_KIND" == "vm" ]]; then
  if [[ -f eval/vm_benchmark.py ]]; then
    RUNNER=eval/vm_benchmark.py
  elif [[ -f agentlife/eval/vm_benchmark.py ]]; then
    RUNNER=agentlife/eval/vm_benchmark.py
  else
    echo 'ERROR: vm_benchmark.py not found under eval/ or agentlife/eval/' >&2
    exit 1
  fi
elif [[ -f eval/run_production_benchmark.py ]]; then
  RUNNER=eval/run_production_benchmark.py
elif [[ -f agentlife/eval/run_production_benchmark.py ]]; then
  RUNNER=agentlife/eval/run_production_benchmark.py
else
  echo 'ERROR: run_production_benchmark.py not found under eval/ or agentlife/eval/' >&2
  exit 1
fi
echo \"Using runner: \$RUNNER\"

RESULTS_DIR=$(printf %q "$RESULTS_DIR")
RESULTS_BASENAME=$(printf %q "$RESULTS_BASENAME")
WORKSPACE_BASENAME=$(printf %q "$WORKSPACE_BASENAME")
if [[ "$AUTO_RESULTS_DIR" == "true" && \"\${BENCHMARK_NEUTRAL_WORKSPACE:-1}\" != "0" ]]; then
  NEUTRAL_WORKSPACE=\"\$REMOTE_WORKSPACE_ROOT_RESOLVED/\$WORKSPACE_BASENAME\"
  if [[ -e \"\$RESULTS_DIR\" || -L \"\$RESULTS_DIR\" ]]; then
    echo \"ERROR: results dir already exists: \$RESULTS_DIR\" >&2
    exit 1
  fi
  if [[ -e \"\$NEUTRAL_WORKSPACE\" || -L \"\$NEUTRAL_WORKSPACE\" ]]; then
    echo \"ERROR: neutral workspace already exists: \$NEUTRAL_WORKSPACE\" >&2
    exit 1
  fi
  mkdir -p \"\$(dirname -- \"\$RESULTS_DIR\")\" \"\$REMOTE_WORKSPACE_ROOT_RESOLVED\"
  mkdir -p \"\$NEUTRAL_WORKSPACE\"
  ln -s \"\$NEUTRAL_WORKSPACE\" \"\$RESULTS_DIR\"
  echo \"Neutral Quaid workspace: \$RESULTS_DIR -> \$NEUTRAL_WORKSPACE\"
else
  mkdir -p \"\$RESULTS_DIR\"
fi
if [[ -n \"\${BENCHMARK_RUN_NOTE:-}\" ]]; then
  printf '%s\n' \"\$BENCHMARK_RUN_NOTE\" > \"\$RESULTS_DIR/run_note.txt\"
fi
if [[ $(printf %q "$BACKEND_ARG") == "codex" ]]; then
  CODEX_TOKEN="\${BENCHMARK_CODEX_API_KEY:-\${OPENAI_API_KEY:-}}"
  if [[ -z "\$CODEX_TOKEN" ]]; then
    echo \"ERROR: Codex backend requires a Codex OAuth access token. Set BENCHMARK_CODEX_API_KEY, BENCHMARK_CODEX_TOKEN_PATH, auth.codex.solKeyPath/yuniKeyPath, or make OPENAI_API_KEY available on the remote host.\" >&2
    exit 1
  fi
  mkdir -p \"\$RESULTS_DIR/adaptors/codex\"
  printf '%s\n' "\$CODEX_TOKEN" > \"\$RESULTS_DIR/adaptors/codex/.auth-token\"
  chmod 600 \"\$RESULTS_DIR/adaptors/codex/.auth-token\"
  echo \"Codex auth token: wrote \$RESULTS_DIR/adaptors/codex/.auth-token\"
fi
LAUNCH_LOG=\${RESULTS_DIR}.launch.log
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

OLLAMA_PREFLIGHT_URL="${BENCHMARK_OLLAMA_URL:-http://127.0.0.1:11434}"
EMBED_PREFLIGHT_MODEL="${BENCHMARK_EMBEDDING_MODEL:-nomic-embed-text}"
EMBED_PREFLIGHT_DIM="${BENCHMARK_EMBEDDING_DIM:-768}"
EMBED_PREFLIGHT_PROVIDER="${BENCHMARK_EMBEDDINGS_PROVIDER:-ollama}"
REMOTE_OLLAMA_PREFLIGHT_CMD="
set -euo pipefail
$OPTIONAL_BENCH_ENV
OLLAMA_URL=$(printf %q "$OLLAMA_PREFLIGHT_URL")
EMBED_MODEL=$(printf %q "$EMBED_PREFLIGHT_MODEL")
EMBED_DIM=$(printf %q "$EMBED_PREFLIGHT_DIM")
EMBED_PROVIDER=$(printf %q "$EMBED_PREFLIGHT_PROVIDER")
if [[ \"\$EMBED_PROVIDER\" != \"ollama\" ]]; then
  echo \"ERROR: benchmark embedding policy requires BENCHMARK_EMBEDDINGS_PROVIDER=ollama; got \$EMBED_PROVIDER\" >&2
  exit 1
fi
if [[ \"\$EMBED_MODEL\" != \"nomic-embed-text\" ]]; then
  echo \"ERROR: benchmark embedding policy requires BENCHMARK_EMBEDDING_MODEL=nomic-embed-text; got \$EMBED_MODEL\" >&2
  exit 1
fi
if [[ \"\$EMBED_DIM\" != \"768\" ]]; then
  echo \"ERROR: benchmark embedding policy requires BENCHMARK_EMBEDDING_DIM=768; got \$EMBED_DIM\" >&2
  exit 1
fi
echo \"Preflight: ollama embeddings at \$OLLAMA_URL (provider=\$EMBED_PROVIDER model=\$EMBED_MODEL dim=\$EMBED_DIM)\"
TAGS=\$(curl -sS \"\$OLLAMA_URL/api/tags\" || true)
if [[ -z \"\$TAGS\" ]]; then
  echo \"ERROR: Ollama tags endpoint unavailable at \$OLLAMA_URL/api/tags\" >&2
  exit 1
fi
python3 - \"\$TAGS\" \"\$EMBED_MODEL\" <<'PY'
import json, sys
payload = json.loads(sys.argv[1])
target = sys.argv[2]
names = set()
for row in payload.get('models', []):
    if isinstance(row, dict):
        for key in ('name', 'model'):
            val = str(row.get(key) or '').strip()
            if val:
                names.add(val)
                if val.endswith(':latest'):
                    names.add(val[:-7])
if target not in names:
    preview = ', '.join(sorted(names)[:12]) or '<none>'
    raise SystemExit(f'Missing embedding model {target!r} in ollama tags: {preview}')
PY
EMBED=\$(curl -sS -X POST \"\$OLLAMA_URL/api/embed\" -H 'Content-Type: application/json' -d \"{\\\"model\\\":\\\"\$EMBED_MODEL\\\",\\\"input\\\":\\\"preflight cache check\\\"}\" || true)
python3 - \"\$EMBED\" \"\$EMBED_DIM\" <<'PY'
import json, sys
payload = json.loads(sys.argv[1])
target_dim = int(sys.argv[2])
emb = payload.get('embeddings') or []
if not emb or not isinstance(emb[0], list) or not emb[0]:
    raise SystemExit('Ollama /api/embed did not return embeddings')
actual_dim = len(emb[0])
if actual_dim != target_dim:
    raise SystemExit(f'Ollama /api/embed returned dim {actual_dim}, expected {target_dim}')
PY
echo \"Preflight OK: ollama embeddings\"
"
run_cmd_redacted "ssh ${SSH_OPTS[*]} $REMOTE [remote ollama embedding preflight]" \
  ssh "${SSH_OPTS[@]}" "$REMOTE" "$REMOTE_OLLAMA_PREFLIGHT_CMD"

if [[ "$BACKEND_ARG" == "llama-cpp" ]]; then
  LLAMA_PREFLIGHT_URL="${BENCHMARK_LLAMA_CPP_URL:-http://127.0.0.1:30001}"
  ENFORCE_PARALLEL_MATCH="${BENCHMARK_ENFORCE_LLAMA_PARALLEL_MATCH:-1}"
  ENFORCE_THREADS_MATCH="${BENCHMARK_ENFORCE_LLAMA_THREADS_MATCH:-1}"
  MIN_CTX_PER_SLOT="${BENCHMARK_MIN_CTX_PER_SLOT:-35000}"
  REMOTE_LLAMA_PREFLIGHT_CMD="
set -euo pipefail
export BENCHMARK_PARALLEL=$(printf %q "$PARALLEL")
$OPTIONAL_BENCH_ENV
LLAMA_URL=$(printf %q "$LLAMA_PREFLIGHT_URL")
ENFORCE_MATCH=$(printf %q "$ENFORCE_PARALLEL_MATCH")
ENFORCE_THREADS=$(printf %q "$ENFORCE_THREADS_MATCH")
MIN_CTX_SLOT=$(printf %q "$MIN_CTX_PER_SLOT")

echo \"Preflight: llama.cpp readiness at \$LLAMA_URL/health\"
READY=0
for _ in \$(seq 1 90); do
  BODY=\$(curl -sS \"\$LLAMA_URL/health\" || true)
  if [[ -n \"\$BODY\" ]]; then
    if python3 - \"\$BODY\" <<'PY'
import json, sys
try:
    payload = json.loads(sys.argv[1])
except Exception:
    raise SystemExit(1)
msg = str(payload.get('error', {}).get('message', '')).lower()
code = str(payload.get('error', {}).get('code', '')).lower()
if 'loading model' in msg or 'unavailable' in code:
    raise SystemExit(2)
raise SystemExit(0)
PY
    then
      READY=1
      break
    fi
  fi
  sleep 2
done
if [[ \"\$READY\" != \"1\" ]]; then
  echo \"ERROR: llama.cpp did not become ready at \$LLAMA_URL\" >&2
  exit 1
fi

if [[ \"\$ENFORCE_MATCH\" == \"1\" ]]; then
  LLAMA_PORT=\$(python3 - \"\$LLAMA_URL\" <<'PY'
import sys
from urllib.parse import urlparse
u = urlparse(sys.argv[1].strip())
print(u.port or (443 if u.scheme == 'https' else 80))
PY
)
  read -r SERVER_PARALLEL SERVER_CTX_SIZE SERVER_THREADS SERVER_CTX_SLOT <<EOF
\$(python3 - \"\$LLAMA_PORT\" <<'PY'
import subprocess, sys
port = str(sys.argv[1])
out = subprocess.check_output(['bash', '-lc', f\"pgrep -af 'llama-server.*--port {port}' || true\"], text=True)
def _int_token(parts, flag):
    for i, token in enumerate(parts):
        if token == flag and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except Exception:
                return None
    return None
for row in out.splitlines():
    if '--port' in row and 'llama-server' in row:
        parts = row.split()
        parallel = _int_token(parts, '--parallel')
        ctx_size = _int_token(parts, '--ctx-size')
        threads = _int_token(parts, '--threads')
        ctx_slot = int(ctx_size / parallel) if parallel and ctx_size else 0
        print(f\"{parallel or 0} {ctx_size or 0} {threads or 0} {ctx_slot}\")
        raise SystemExit(0)
print('0 0 0 0')
PY
)
EOF
  if [[ -z \"\$SERVER_PARALLEL\" ]]; then
    echo \"ERROR: could not determine llama-server --parallel\" >&2
    exit 1
  fi
  if [[ \"\$SERVER_PARALLEL\" != \"$(printf %q "$PARALLEL")\" ]]; then
    echo \"ERROR: llama-server/harness parallel mismatch: server=\$SERVER_PARALLEL harness=$(printf %q "$PARALLEL")\" >&2
    exit 1
  fi
  if [[ \"\$ENFORCE_THREADS\" == \"1\" && \"\$SERVER_THREADS\" != \"$(printf %q "$PARALLEL")\" ]]; then
    echo \"ERROR: llama-server --threads must equal harness parallel: threads=\$SERVER_THREADS harness=$(printf %q "$PARALLEL")\" >&2
    exit 1
  fi
  if [[ \"\$SERVER_CTX_SLOT\" -lt \"\$MIN_CTX_SLOT\" ]]; then
    echo \"ERROR: llama-server per-slot context too small: ctx_size=\$SERVER_CTX_SIZE parallel=\$SERVER_PARALLEL per_slot=\$SERVER_CTX_SLOT required_min=\$MIN_CTX_SLOT\" >&2
    exit 1
  fi
fi
echo \"Preflight OK: llama.cpp runtime\"
"
  run_cmd_redacted "ssh ${SSH_OPTS[*]} $REMOTE [remote llama-cpp preflight]" \
    ssh "${SSH_OPTS[@]}" "$REMOTE" "$REMOTE_LLAMA_PREFLIGHT_CMD"
fi

run_cmd_redacted "ssh ${SSH_OPTS[*]} $REMOTE [remote benchmark launch redacted]" \
  ssh "${SSH_OPTS[@]}" "$REMOTE" "$REMOTE_PY_CMD"

echo ""
echo "Done."
