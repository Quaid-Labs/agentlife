#!/usr/bin/env bash
set -euo pipefail

# Kill project-docs worker processes scoped to one benchmark run.
#
# The project docs supervisor starts one project_docs_worker.py process per
# project. Those workers do not include the run directory in argv, so this
# script scopes by /proc metadata instead: worker cwd or environment must
# reference the requested run path before it is eligible for termination.

REMOTE=""
REMOTE_BENCH_ROOT="~/agentlife-benchmark"
RUN_SPEC=""
DRY_RUN=false
SIGNAL="TERM"

usage() {
  cat <<'USAGE'
Usage:
  kill-project-docs-workers.sh --run <run-dir|run-name> [options]

Options:
  --remote HOST                 SSH host/alias (example: spark). If omitted, run locally.
  --remote-bench-root PATH      Remote benchmark root for relative run names
                                (default: ~/agentlife-benchmark)
  --run PATH_OR_NAME            Required. Absolute run path, runs/<name>, or run directory name.
  --dry-run                     Print matching workers without killing.
  --signal SIGNAL              Signal to send (default: TERM).
  -h, --help                    Show help.

Examples:
  ./scripts/kill-project-docs-workers.sh --remote spark --run quaid-s-r1327-...
  ./scripts/kill-project-docs-workers.sh --remote spark --run runs/quaid-s-r1327-...
  ./scripts/kill-project-docs-workers.sh --remote spark --run /home/solomon/agentlife-benchmark/runs/quaid-s-r1327-...
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote) REMOTE="$2"; shift 2 ;;
    --remote-bench-root) REMOTE_BENCH_ROOT="$2"; shift 2 ;;
    --run) RUN_SPEC="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --signal) SIGNAL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$RUN_SPEC" ]]; then
  echo "ERROR: --run is required" >&2
  usage
  exit 1
fi

run_payload() {
  bash -s -- "$RUN_SPEC" "$REMOTE_BENCH_ROOT" "$DRY_RUN" "$SIGNAL" <<'REMOTE_SCRIPT'
set -euo pipefail

run_spec="$1"
bench_root="$2"
dry_run="$3"
signal_name="$4"

bench_root="${bench_root/#\~/$HOME}"

case "$run_spec" in
  /*) run_path="$run_spec" ;;
  runs/*) run_path="$bench_root/$run_spec" ;;
  *) run_path="$bench_root/runs/$run_spec" ;;
esac

if [[ ! -d "$run_path" ]]; then
  echo "ERROR: run path does not exist: $run_path" >&2
  exit 1
fi

if [[ ! -d /proc ]]; then
  echo "ERROR: /proc is required for run-scoped worker matching" >&2
  exit 1
fi

matched=()

for proc in /proc/[0-9]*; do
  [[ -d "$proc" ]] || continue
  pid="${proc##*/}"
  cmd="$(tr '\0' ' ' < "$proc/cmdline" 2>/dev/null || true)"
  [[ "$cmd" == *"project_docs_worker.py run"* ]] || continue

  cwd="$(readlink "$proc/cwd" 2>/dev/null || true)"
  env_text="$(tr '\0' '\n' < "$proc/environ" 2>/dev/null || true)"

  if [[ "$cwd" == "$run_path" || "$cwd" == "$run_path/"* || "$env_text" == *"$run_path"* ]]; then
    matched+=("$pid")
    printf 'MATCH pid=%s cwd=%s cmd=%s\n' "$pid" "${cwd:-?}" "$cmd"
  fi
done

if [[ "${#matched[@]}" -eq 0 ]]; then
  echo "No project-docs workers matched run_path=$run_path"
  exit 0
fi

if [[ "$dry_run" == "true" ]]; then
  echo "DRY_RUN: would send SIG$signal_name to ${matched[*]}"
  exit 0
fi

kill "-$signal_name" "${matched[@]}"
echo "Sent SIG$signal_name to project-docs workers: ${matched[*]}"
REMOTE_SCRIPT
}

if [[ -n "$REMOTE" ]]; then
  printf -v REMOTE_ARGS '%q ' "$RUN_SPEC" "$REMOTE_BENCH_ROOT" "$DRY_RUN" "$SIGNAL"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$REMOTE" "bash -s -- ${REMOTE_ARGS% }" <<'REMOTE_SCRIPT'
set -euo pipefail

run_spec="$1"
bench_root="$2"
dry_run="$3"
signal_name="$4"

bench_root="${bench_root/#\~/$HOME}"

case "$run_spec" in
  /*) run_path="$run_spec" ;;
  runs/*) run_path="$bench_root/$run_spec" ;;
  *) run_path="$bench_root/runs/$run_spec" ;;
esac

if [[ ! -d "$run_path" ]]; then
  echo "ERROR: run path does not exist: $run_path" >&2
  exit 1
fi

if [[ ! -d /proc ]]; then
  echo "ERROR: /proc is required for run-scoped worker matching" >&2
  exit 1
fi

matched=()

for proc in /proc/[0-9]*; do
  [[ -d "$proc" ]] || continue
  pid="${proc##*/}"
  cmd="$(tr '\0' ' ' < "$proc/cmdline" 2>/dev/null || true)"
  [[ "$cmd" == *"project_docs_worker.py run"* ]] || continue

  cwd="$(readlink "$proc/cwd" 2>/dev/null || true)"
  env_text="$(tr '\0' '\n' < "$proc/environ" 2>/dev/null || true)"

  if [[ "$cwd" == "$run_path" || "$cwd" == "$run_path/"* || "$env_text" == *"$run_path"* ]]; then
    matched+=("$pid")
    printf 'MATCH pid=%s cwd=%s cmd=%s\n' "$pid" "${cwd:-?}" "$cmd"
  fi
done

if [[ "${#matched[@]}" -eq 0 ]]; then
  echo "No project-docs workers matched run_path=$run_path"
  exit 0
fi

if [[ "$dry_run" == "true" ]]; then
  echo "DRY_RUN: would send SIG$signal_name to ${matched[*]}"
  exit 0
fi

kill "-$signal_name" "${matched[@]}"
echo "Sent SIG$signal_name to project-docs workers: ${matched[*]}"
REMOTE_SCRIPT
else
  run_payload
fi
