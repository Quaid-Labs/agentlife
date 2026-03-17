#!/usr/bin/env bash
set -euo pipefail

VM_IP="192.168.64.3"
VM_USER="admin"
VM_PASSWORD="admin"
WORKSPACE="~/clawd"
OPENCLAW_VERSION="latest"
GATEWAY_PORT="18789"
OLLAMA_BASE_URL="http://192.168.64.1:11434/v1"
INSTALL_OPENCLAW=true
START_GATEWAY=true
WIPE_WORKSPACE=true
AUTH_PROFILES_FROM=""

usage() {
  cat <<'USAGE'
Usage: bootstrap-oc-native-vm.sh [options]

Provision a clean OpenClaw target for the AgentLife VM benchmark native-memory
baseline. This prepares OpenClaw only; per-system benchmark setup still happens
inside eval/vm_benchmark.py.

Options:
  --vm-ip <ip>                Target VM IP (default: 192.168.64.3)
  --user <name>               SSH user (default: admin)
  --password <password>       SSH password (default: admin)
  --workspace <path>          Benchmark workspace path on target (default: ~/clawd)
  --openclaw-version <ver>    npm package version or "latest" (default: latest)
  --gateway-port <port>       Gateway port to probe (default: 18789)
  --ollama-base-url <url>     Host embeddings endpoint visible from guest
                              (default: http://192.168.64.1:11434/v1)
  --skip-openclaw-install     Do not install/update OpenClaw on target
  --skip-gateway-start        Do not start/probe the gateway
  --no-wipe-workspace         Keep existing benchmark workspace contents
  --auth-profiles-from <p>    Copy local OpenClaw auth-profiles.json into guest
  -h, --help                  Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vm-ip) VM_IP="$2"; shift 2 ;;
    --user) VM_USER="$2"; shift 2 ;;
    --password) VM_PASSWORD="$2"; shift 2 ;;
    --workspace) WORKSPACE="$2"; shift 2 ;;
    --openclaw-version) OPENCLAW_VERSION="$2"; shift 2 ;;
    --gateway-port) GATEWAY_PORT="$2"; shift 2 ;;
    --ollama-base-url) OLLAMA_BASE_URL="$2"; shift 2 ;;
    --skip-openclaw-install) INSTALL_OPENCLAW=false; shift ;;
    --skip-gateway-start) START_GATEWAY=false; shift ;;
    --no-wipe-workspace) WIPE_WORKSPACE=false; shift ;;
    --auth-profiles-from) AUTH_PROFILES_FROM="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if ! command -v ssh >/dev/null 2>&1; then
  echo "Missing required tool: ssh" >&2
  exit 1
fi
if ! command -v sshpass >/dev/null 2>&1; then
  echo "Missing required tool: sshpass" >&2
  exit 1
fi
if [[ -n "$AUTH_PROFILES_FROM" && ! -f "$AUTH_PROFILES_FROM" ]]; then
  echo "Auth profiles file not found: $AUTH_PROFILES_FROM" >&2
  exit 1
fi

ssh_vm() {
  local cmd="$1"
  sshpass -p "$VM_PASSWORD" ssh \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o ConnectTimeout=10 \
    -o PreferredAuthentications=password \
    -o PubkeyAuthentication=no \
    -o IdentitiesOnly=yes \
    "${VM_USER}@${VM_IP}" "$cmd"
}

copy_file_to_vm() {
  local local_path="$1"
  local remote_path="$2"
  cat "$local_path" | sshpass -p "$VM_PASSWORD" ssh \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o PreferredAuthentications=password \
    -o PubkeyAuthentication=no \
    -o IdentitiesOnly=yes \
    "${VM_USER}@${VM_IP}" "cat > ${remote_path}"
}

echo "[oc-native] Checking SSH access to ${VM_USER}@${VM_IP}..."
ssh_vm "hostname; whoami"

if [[ -n "$AUTH_PROFILES_FROM" ]]; then
  echo "[oc-native] Seeding auth profiles from ${AUTH_PROFILES_FROM}..."
  ssh_vm "mkdir -p ~/.openclaw/agents/main/agent"
  copy_file_to_vm "$AUTH_PROFILES_FROM" "~/.openclaw/agents/main/agent/auth-profiles.json"
fi

remote_script=$(cat <<'EOF'
set -euo pipefail

WORKSPACE="${WORKSPACE}"
OPENCLAW_VERSION="${OPENCLAW_VERSION}"
GATEWAY_PORT="${GATEWAY_PORT}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL}"
INSTALL_OPENCLAW="${INSTALL_OPENCLAW}"
START_GATEWAY="${START_GATEWAY}"
WIPE_WORKSPACE="${WIPE_WORKSPACE}"
export PATH="/opt/homebrew/bin:/usr/local/bin:${PATH}"

need_cmd() {
  local name="$1"
  command -v "$name" >/dev/null 2>&1 || {
    echo "[oc-native] Missing required command on target: $name" >&2
    exit 1
  }
}

need_cmd python3

if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  if ! command -v brew >/dev/null 2>&1; then
    echo "[oc-native] Missing node/npm and Homebrew is unavailable on target" >&2
    exit 1
  fi
  echo "[oc-native] Installing node via Homebrew..."
  HOMEBREW_NO_AUTO_UPDATE=1 brew install node
fi

need_cmd node
need_cmd npm

if [[ "${INSTALL_OPENCLAW}" == "true" ]]; then
  if [[ "${OPENCLAW_VERSION}" == "latest" ]]; then
    if ! command -v openclaw >/dev/null 2>&1; then
      echo "[oc-native] Installing OpenClaw latest..."
      npm install -g openclaw
    fi
  else
    echo "[oc-native] Installing OpenClaw ${OPENCLAW_VERSION}..."
    npm install -g "openclaw@${OPENCLAW_VERSION}"
  fi
fi

need_cmd openclaw
openclaw --version || true

if ! python3 - <<'PY' >/dev/null 2>&1
import sqlite_vec  # noqa: F401
PY
then
  echo "[oc-native] Installing sqlite-vec..."
  python3 -m pip install --upgrade pip sqlite-vec || \
  python3 -m pip install --user --break-system-packages sqlite-vec
fi

mkdir -p "${HOME}/.openclaw"
if [[ ! -f "${HOME}/.openclaw/openclaw.json" ]]; then
  printf '{}\n' > "${HOME}/.openclaw/openclaw.json"
fi

python3 - <<'PY'
import json
import os

cfg_path = os.path.expanduser("~/.openclaw/openclaw.json")
data = json.load(open(cfg_path))
gateway = data.setdefault("gateway", {})
gateway["mode"] = "local"
gateway["port"] = int(os.environ["GATEWAY_PORT"])
json.dump(data, open(cfg_path, "w"), indent=2)
print("[oc-native] Configured gateway.mode=local")
PY

if [[ "${WIPE_WORKSPACE}" == "true" ]]; then
  echo "[oc-native] Clearing benchmark workspace state..."
  rm -rf "${WORKSPACE}"
  rm -f "${HOME}/.openclaw/agents/main/sessions/"*.jsonl 2>/dev/null || true
  rm -f "${HOME}/.openclaw/agents/main/sessions/sessions.json" 2>/dev/null || true
  rm -rf "${HOME}/.openclaw/workspace/memory" 2>/dev/null || true
  rm -f "${HOME}/.openclaw/workspace/MEMORY.md" 2>/dev/null || true
  rm -f "${HOME}/.openclaw/memory/"*.sqlite 2>/dev/null || true
fi

mkdir -p "${WORKSPACE}" "${WORKSPACE}/logs"

if [[ "${START_GATEWAY}" == "true" ]]; then
  echo "[oc-native] Restarting gateway..."
  openclaw gateway stop >/dev/null 2>&1 || true
  openclaw gateway install >/tmp/openclaw-gateway-bootstrap.log 2>&1 || true
  if ! openclaw gateway start --allow-unconfigured --port "${GATEWAY_PORT}" >>/tmp/openclaw-gateway-bootstrap.log 2>&1; then
    echo "[oc-native] gateway start fell back to foreground run"
    nohup openclaw gateway run --allow-unconfigured --force --port "${GATEWAY_PORT}" \
      >/tmp/openclaw-gateway-bootstrap.log 2>&1 &
    sleep 5
  fi
  openclaw gateway probe
fi

python3 - <<'PY'
import json
import os
import urllib.request

cfg_path = os.path.expanduser("~/.openclaw/openclaw.json")
data = json.load(open(cfg_path))
print("[oc-native] OpenClaw config path:", cfg_path)
print("[oc-native] Plugin slots:", json.dumps((data.get("plugins") or {}).get("slots") or {}, sort_keys=True))
PY

python3 - <<'PY'
import json
import os
import sys
import urllib.error
import urllib.request

base = os.environ["OLLAMA_BASE_URL"].rstrip("/")
url = base + "/models"
try:
    with urllib.request.urlopen(url, timeout=10) as resp:
        body = resp.read().decode("utf-8", errors="replace")
except Exception as exc:
    print(f"[oc-native] WARNING: could not probe embeddings endpoint {url}: {exc}", file=sys.stderr)
    raise SystemExit(1)
print("[oc-native] Embeddings endpoint probe OK:", url)
print(body[:200])
PY

echo "[oc-native] Bootstrap complete for ${WORKSPACE}"
EOF
)

ssh_vm "WORKSPACE=$(printf '%q' "$WORKSPACE") OPENCLAW_VERSION=$(printf '%q' "$OPENCLAW_VERSION") GATEWAY_PORT=$(printf '%q' "$GATEWAY_PORT") OLLAMA_BASE_URL=$(printf '%q' "$OLLAMA_BASE_URL") INSTALL_OPENCLAW=$(printf '%q' "$INSTALL_OPENCLAW") START_GATEWAY=$(printf '%q' "$START_GATEWAY") WIPE_WORKSPACE=$(printf '%q' "$WIPE_WORKSPACE") bash -s" <<<"$remote_script"

echo
echo "[oc-native] Next step:"
echo "  python3 eval/vm_benchmark.py --system oc-native --vm-ip ${VM_IP} --answer-model claude-haiku-4-5-20251001 --dry-run"
