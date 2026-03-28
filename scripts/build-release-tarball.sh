#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT_DIR/release/agentlife-benchmark-release.tar.gz"
TMP_TAR="/tmp/agentlife-benchmark-release.tar.gz"

mkdir -p "$(dirname "$OUT")"

cd "$ROOT_DIR"

export COPYFILE_DISABLE=1
export COPY_EXTENDED_ATTRIBUTES_DISABLE=1

tar \
  --exclude='.git' \
  --exclude='node_modules' \
  --exclude='.pytest_cache' \
  --exclude='.pytest-home' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  --exclude='._*' \
  --exclude='.env' \
  --exclude='.agentlife-benchmark.local.json' \
  --exclude='agentlife' \
  --exclude='runs' \
  --exclude='tmp' \
  --exclude='data/imported-*' \
  --exclude='data/results*' \
  --exclude='recovered-from-spark-*' \
  --exclude='release' \
  -czf "$TMP_TAR" \
  .

cp "$TMP_TAR" "$OUT"

echo "[release-tarball] wrote:"
ls -lh "$OUT"

echo "[release-tarball] sha256:"
shasum -a 256 "$OUT"
