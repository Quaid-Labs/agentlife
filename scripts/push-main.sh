#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${1:-origin}"
TARGET_BRANCH="main"

die() {
  echo "[push-main] ERROR: $*" >&2
  exit 1
}

cd "$ROOT_DIR"

branch="$(git rev-parse --abbrev-ref HEAD)"
[[ "$branch" == "$TARGET_BRANCH" ]] || die "current branch is '$branch'; switch to '$TARGET_BRANCH' first"
[[ -z "$(git status --porcelain)" ]] || die "worktree is dirty; commit or stash before push"

echo "[push-main] release-candidate sync + lite gate"
"$ROOT_DIR/scripts/release-sync.sh"

echo "[push-main] pushing ${REMOTE} ${TARGET_BRANCH}"
git push "$REMOTE" "HEAD:${TARGET_BRANCH}"

echo "[push-main] PASS"
