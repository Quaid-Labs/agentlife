#!/usr/bin/env bash
# spark-env.sh — Shared configuration for Spark deploy & launch scripts.
# Sourced by all toolkit scripts. Every value is overridable via env var.

# Target host (change for different servers, or "localhost" for local runs)
# Default to direct host to avoid fragile local SSH alias dependencies.
SPARK_HOST="${SPARK_HOST:-solomon@192.168.0.139}"

# Remote paths
SPARK_BENCHMARK_ROOT="${SPARK_BENCHMARK_ROOT:-/home/solomon/clawd-benchmark}"
SPARK_CHECKPOINT_ROOT="${SPARK_CHECKPOINT_ROOT:-${SPARK_BENCHMARK_ROOT}/benchmark-checkpoint}"
SPARK_ASSETS_DIR="${SPARK_ASSETS_DIR:-${SPARK_BENCHMARK_ROOT}/benchmark-assets}"
SPARK_ENV_FILE="${SPARK_ENV_FILE:-${SPARK_BENCHMARK_ROOT}/.env}"
SPARK_FILLER_DIR="${SPARK_FILLER_DIR:-agentlife/data/filler-sessions-L}"

# Local paths (resolved relative to benchmark/ root)
LOCAL_BENCHMARK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_CHECKPOINT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../benchmark-checkpoint" 2>/dev/null && pwd || echo "")"

# NAS backup
NAS_BACKUP_DIR="${NAS_BACKUP_DIR:-/Volumes/Alfie/benchmark-backups}"

# Models (update when switching model versions)
MODEL="${BENCHMARK_MODEL:-claude-sonnet-4-6}"
EVAL_MODEL="${BENCHMARK_EVAL_MODEL:-claude-sonnet-4-6}"
JUDGE="${BENCHMARK_JUDGE:-gpt-4o-mini}"

# Timeouts
JANITOR_TIMEOUT_S="${JANITOR_TIMEOUT_S:-7200}"
ANTHROPIC_TIMEOUT_S="${ANTHROPIC_TIMEOUT_S:-600}"
CLAUDE_CODE_TIMEOUT_S="${CLAUDE_CODE_TIMEOUT_S:-900}"
CLAUDE_CODE_FAST_TIMEOUT_S="${CLAUDE_CODE_FAST_TIMEOUT_S:-30}"
CLAUDE_CODE_DEEP_TIMEOUT_S="${CLAUDE_CODE_DEEP_TIMEOUT_S:-90}"
CLAUDE_CODE_TIMEOUT_CAP_S="${CLAUDE_CODE_TIMEOUT_CAP_S:-0}"
