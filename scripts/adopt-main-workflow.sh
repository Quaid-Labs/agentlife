#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${1:-origin}"
TODAY="$(date +%Y%m%d)"

cd "$ROOT_DIR"

current_branch="$(git rev-parse --abbrev-ref HEAD)"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "[adopt-main] ERROR: worktree is dirty; commit/stash first" >&2
  exit 1
fi

git fetch "$REMOTE" --prune

if ! git rev-parse --verify "$REMOTE/main" >/dev/null 2>&1; then
  echo "[adopt-main] ERROR: missing $REMOTE/main" >&2
  exit 1
fi

if [[ "$current_branch" != "main" ]]; then
  archive_branch="archive/${current_branch}-pre-main-only-${TODAY}"
  if git rev-parse --verify "$archive_branch" >/dev/null 2>&1; then
    echo "[adopt-main] archive branch already exists: $archive_branch"
  else
    git branch "$archive_branch" "$current_branch"
    echo "[adopt-main] created archive branch: $archive_branch -> $current_branch"
  fi
fi

git checkout -B main "$REMOTE/main"
git branch --set-upstream-to="$REMOTE/main" main >/dev/null

if [[ "$current_branch" != "main" ]]; then
  git branch -D "$current_branch"
  echo "[adopt-main] removed active local branch: $current_branch"
fi

echo "[adopt-main] now on main tracking $REMOTE/main"
