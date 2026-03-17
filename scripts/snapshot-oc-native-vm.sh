#!/usr/bin/env bash
set -euo pipefail

VM_NAME="test-openclaw"
SNAPSHOT_NAME="clean-openclaw"
RESTART_VM=true
REPLACE_SNAPSHOT=false

usage() {
  cat <<'USAGE'
Usage: snapshot-oc-native-vm.sh [options]

Create a Tart snapshot for the OpenClaw benchmark VM after bootstrap.

Options:
  --vm-name <name>           Tart VM name (default: test-openclaw)
  --snapshot <name>          Snapshot name (default: clean-openclaw)
  --no-restart               Do not restart the VM after snapshot
  --replace                  Replace an existing snapshot with the same name
  -h, --help                 Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vm-name) VM_NAME="$2"; shift 2 ;;
    --snapshot) SNAPSHOT_NAME="$2"; shift 2 ;;
    --no-restart) RESTART_VM=false; shift ;;
    --replace) REPLACE_SNAPSHOT=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if ! command -v tart >/dev/null 2>&1; then
  echo "Missing required tool: tart" >&2
  exit 1
fi

if ! tart list | awk 'NR>1 {print $2}' | grep -qx "$VM_NAME"; then
  echo "Tart VM not found: $VM_NAME" >&2
  exit 1
fi

if ! tart help 2>/dev/null | grep -q "snapshot"; then
  echo "[oc-native] This Tart build does not support snapshots."
  echo "[oc-native] vm_benchmark.py will fall back to verifying the VM is running."
  exit 0
fi

if [[ "$REPLACE_SNAPSHOT" == "true" ]]; then
  tart snapshot delete "$VM_NAME" "$SNAPSHOT_NAME" >/dev/null 2>&1 || true
fi

echo "[oc-native] Stopping ${VM_NAME} before snapshot..."
tart stop "$VM_NAME" >/dev/null 2>&1 || true
sleep 2

echo "[oc-native] Creating snapshot ${SNAPSHOT_NAME}..."
tart snapshot create "$VM_NAME" "$SNAPSHOT_NAME"

if [[ "$RESTART_VM" == "true" ]]; then
  echo "[oc-native] Restarting ${VM_NAME}..."
  tart run "$VM_NAME" --no-graphics >/dev/null 2>&1 &
fi

echo "[oc-native] Snapshot ready: ${VM_NAME}@${SNAPSHOT_NAME}"
