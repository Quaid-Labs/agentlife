#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-full}"

if [[ "$MODE" != "full" && "$MODE" != "--full" && "$MODE" != "lite" && "$MODE" != "--lite" ]]; then
  echo "Usage: $0 [--lite|--full]" >&2
  exit 1
fi
if [[ "$MODE" == "--full" ]]; then MODE="full"; fi
if [[ "$MODE" == "--lite" ]]; then MODE="lite"; fi

echo "[release-check] docs/readme references"
python3 - "$ROOT_DIR" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
targets = [
    root / "README.md",
    root / "METHODOLOGY.md",
    root / "docs" / "README.md",
]
pattern = re.compile(r"\[([^\]]+)\]\(([^)#]+)\)")
for target in targets:
    text = target.read_text(encoding="utf-8")
    for _, rel in pattern.findall(text):
        if rel.startswith("http://") or rel.startswith("https://") or rel.startswith("#"):
            continue
        candidate = (target.parent / rel).resolve()
        if not candidate.exists():
            raise SystemExit(f"missing doc target from {target}: {rel}")
print("docs references OK")
PY

echo "[release-check] local config + published layout"
test -f "$ROOT_DIR/.agentlife-benchmark.example.json"
test -f "$ROOT_DIR/docs/LOCAL-DEVELOPMENT.md"
test -f "$ROOT_DIR/docs/RELEASE-CHECKLIST.md"
test -f "$ROOT_DIR/published/README.md"
test -f "$ROOT_DIR/published/runbooks/README.md"
test -f "$ROOT_DIR/published/checkpoints/README.md"
test -f "$ROOT_DIR/published/runbooks/release-candidate/README.md"
test -f "$ROOT_DIR/published/checkpoints/release-candidate/README.md"
git -C "$ROOT_DIR" check-ignore -q .agentlife-benchmark.local.json

echo "[release-check] shell syntax"
bash -n "$ROOT_DIR/scripts/launch-remote-benchmark.sh"
bash -n "$ROOT_DIR/scripts/release-check.sh"
bash -n "$ROOT_DIR/scripts/build-release-tarball.sh"
bash -n "$ROOT_DIR/scripts/release-sync.sh"
bash -n "$ROOT_DIR/scripts/push-main.sh"
bash -n "$ROOT_DIR/scripts/adopt-main-workflow.sh"

echo "[release-check] launcher sync exclusions"
grep -q -- "--exclude='.env'" "$ROOT_DIR/scripts/launch-remote-benchmark.sh"
grep -q -- "--exclude='.agentlife-benchmark.local.json'" "$ROOT_DIR/scripts/launch-remote-benchmark.sh"
grep -q -- "--exclude='release/'" "$ROOT_DIR/scripts/launch-remote-benchmark.sh"
grep -q -- "Remove local-only benchmark artifacts from remote root" "$ROOT_DIR/scripts/launch-remote-benchmark.sh"

if [[ "$MODE" == "full" ]]; then
  echo "[release-check] release tarball contents"
  "$ROOT_DIR/scripts/build-release-tarball.sh" >/dev/null
  python3 - "$ROOT_DIR/release/agentlife-benchmark-release.tar.gz" <<'PY'
import sys
import tarfile

archive = sys.argv[1]
forbidden = [
    ".agentlife-benchmark.local.json",
    "release/",
    "runs/",
    "agentlife/",
]
with tarfile.open(archive, "r:gz") as tf:
    names = tf.getnames()
for needle in forbidden:
    if any(needle in name for name in names):
        raise SystemExit(f"forbidden tarball entry matched: {needle}")
    if any("/._" in name or name.startswith("._") for name in names):
        raise SystemExit("forbidden AppleDouble entry found in release tarball")
print("release tarball OK")
PY
else
  echo "[release-check] lite mode: skipping tarball build/inspection"
fi

echo "[release-check] python compile"
python3 -m py_compile \
  "$ROOT_DIR/eval/run_production_benchmark.py" \
  "$ROOT_DIR/scripts/benchmark_dashboard.py" \
  "$ROOT_DIR/scripts/benchmark_run_state.py" \
  "$ROOT_DIR/scripts/monitor_benchmarks.py"

echo "[release-check] harness tests"
(
  cd "$ROOT_DIR"
  pytest -q \
    eval/tests/test_benchmark_regressions.py \
    eval/tests/test_store_edge_retry.py \
    eval/tests/test_benchmark_run_state.py
)

echo "[release-check] PASS"
