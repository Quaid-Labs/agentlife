#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROMOTE_TAG=""

usage() {
  cat <<'EOF'
Usage:
  ./scripts/release-sync.sh [--promote-tag <tag>]

Behavior:
  - verifies release-candidate publish layout exists
  - runs ./scripts/release-check.sh --lite
  - optional: promotes release-candidate artifacts into published/<tag>/ trees
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --promote-tag)
      [[ $# -ge 2 ]] || { echo "missing value for --promote-tag" >&2; exit 1; }
      PROMOTE_TAG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

cd "$ROOT_DIR"

RUNBOOK_RC_DIR="$ROOT_DIR/published/runbooks/release-candidate"
CHECKPOINT_RC_DIR="$ROOT_DIR/published/checkpoints/release-candidate"

test -d "$RUNBOOK_RC_DIR"
test -d "$CHECKPOINT_RC_DIR"
test -f "$RUNBOOK_RC_DIR/README.md"
test -f "$CHECKPOINT_RC_DIR/README.md"

if ! find "$RUNBOOK_RC_DIR" -maxdepth 1 -type f ! -name 'README.md' | grep -q .; then
  echo "[release-sync] ERROR: no release-candidate runbook files found" >&2
  exit 1
fi

if ! find "$CHECKPOINT_RC_DIR" -maxdepth 1 -type f ! -name 'README.md' | grep -q .; then
  echo "[release-sync] ERROR: no release-candidate checkpoint index files found" >&2
  exit 1
fi

echo "[release-sync] running lite release gate"
"$ROOT_DIR/scripts/release-check.sh" --lite

if [[ -n "$PROMOTE_TAG" ]]; then
  RUNBOOK_OUT_DIR="$ROOT_DIR/published/runbooks/$PROMOTE_TAG"
  CHECKPOINT_OUT_DIR="$ROOT_DIR/published/checkpoints/$PROMOTE_TAG"
  mkdir -p "$RUNBOOK_OUT_DIR" "$CHECKPOINT_OUT_DIR"

  rsync -a --delete --exclude 'README.md' "$RUNBOOK_RC_DIR/" "$RUNBOOK_OUT_DIR/"
  rsync -a --delete --exclude 'README.md' "$CHECKPOINT_RC_DIR/" "$CHECKPOINT_OUT_DIR/"

  echo "[release-sync] promoted release-candidate artifacts to:"
  echo "  - $RUNBOOK_OUT_DIR"
  echo "  - $CHECKPOINT_OUT_DIR"
else
  echo "[release-sync] release-candidate layout + lite gate OK"
fi

