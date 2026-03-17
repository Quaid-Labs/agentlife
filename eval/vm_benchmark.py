#!/usr/bin/env python3
"""AgentLife VM Benchmark — Full-stack multi-system evaluation.

Runs the AgentLife benchmark through real OpenClaw instances on a Tart VM,
comparing the canonical memory systems plus optional native OpenClaw baselines.

Unlike the simulation harness (run_agentlife.py), this tests the actual product:
- Real gateway compaction and summarization
- Agent-driven tool use (memory_recall, project docs)
- Full janitor pipeline (Quaid)
- Core markdown evolution (snippets, journal)
- QMD indexing (Rust binary)

Architecture:
    Host (Mac Mini) --- SSH ---> VM (Tart, macOS) running OpenClaw
    Mem0 runs on host via Python API (no VM needed)

Usage:
    # Run Quaid benchmark (natural compaction)
    python3 vm_benchmark.py --system quaid --mode natural

    # Run all systems
    python3 vm_benchmark.py --system all

    # Run evaluation only (skip injection, use existing state)
    python3 vm_benchmark.py --system quaid --eval-only

    # Dry run (show plan, no execution)
    python3 vm_benchmark.py --system quaid --dry-run
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DIR.parent
_WORKSPACE = Path(os.environ.get("CLAWDBOT_WORKSPACE", Path.home() / "clawd"))
_RUNNER_DIR = _WORKSPACE / "memory-stress-test" / "runner"
OC_NATIVE_EMBED_BASE_URL = os.environ.get(
    "OPENCLAW_NATIVE_OLLAMA_BASE_URL",
    "http://192.168.64.1:11434/v1",
)

sys.path.insert(0, str(_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(_RUNNER_DIR))

try:
    from claude_backend import call_claude
except ModuleNotFoundError:
    from generate import call_claude
from dataset import (
    load_all_reviews,
    load_filler_reviews,
    merge_sessions_chronologically,
    get_all_eval_queries,
)
from injector import transcript_to_messages, count_tokens, CostTracker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTEXT_WINDOW = 200_000
COMPACTION_THRESHOLD = 0.80
COMPACTION_TOKEN_LIMIT = int(CONTEXT_WINDOW * COMPACTION_THRESHOLD)
OC_NATIVE_REINDEX_TIMEOUT_S = 3600
VM_CLAUDE_JUDGE_TIMEOUT_S = 90

# VM paths — sessions live under agents/{agent-id}/sessions/, NOT ~/.openclaw/sessions/
VM_AGENT_SESSIONS_DIR = "~/.openclaw/agents/main/sessions"
VM_SESSION_STORE = "~/.openclaw/agents/main/sessions/sessions.json"
VM_QUAID_DIR = "~/clawd/plugins/quaid"

# Systems
SYSTEMS = ["base", "qmd", "quaid", "mem0"]
EXTRA_SYSTEMS = ["oc-native"]
AVAILABLE_SYSTEMS = SYSTEMS + EXTRA_SYSTEMS

# Judge prompt — unified across all systems, focused on memory recall accuracy
# Mem0's exact ACCURACY_PROMPT — peer-review-valid comparison with LoCoMo results
JUDGE_PROMPT = """\
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
 (1) a question (posed by one user to another user),
 (2) a 'gold' (ground truth) answer,
 (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {ground_truth}
Generated answer: {prediction}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label"."""


# ---------------------------------------------------------------------------
# VM Management
# ---------------------------------------------------------------------------

class TartVM:
    """Tart VM management for benchmark runs.

    Requires:
    - tart CLI installed on host
    - sshpass installed on host (brew install hudochenkov/sshpass/sshpass)
    - VM running and reachable at the configured IP
    """

    # Prefix all SSH commands with PATH + env vars for Quaid
    PATH_PREFIX = (
        "export PATH=/opt/homebrew/bin:$PATH && "
        "export CLAWDBOT_WORKSPACE=$HOME/clawd && "
        "export ANTHROPIC_API_KEY=$(cat ~/.openclaw/.env 2>/dev/null | grep ANTHROPIC_API_KEY | cut -d= -f2) && "
    )
    SSH_RETRY_PATTERNS = (
        "Permission denied",
        "Connection timed out",
        "Operation timed out",
        "Connection reset",
        "No route to host",
        "Connection refused",
        "Broken pipe",
    )

    def __init__(self, ip: str = "192.168.64.3", user: str = "admin",
                 password: str = "admin", vm_name: str = "test-openclaw"):
        self.ip = ip
        self.user = user
        self.password = password
        self.vm_name = vm_name

    def ssh(self, cmd: str, input_data: Optional[str] = None,
            timeout: int = 120, raw: bool = False) -> subprocess.CompletedProcess:
        """Execute command on VM via SSH.

        Args:
            cmd: Command to run. PATH_PREFIX is auto-prepended unless raw=True.
            input_data: Optional stdin data.
            timeout: Timeout in seconds.
            raw: If True, skip PATH_PREFIX (for simple echo/cat commands).
        """
        full_cmd = cmd if raw else f"{self.PATH_PREFIX}{cmd}"
        args = [
            "sshpass", "-p", self.password,
            "ssh", "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            "-o", "PreferredAuthentications=password",
            "-o", "PubkeyAuthentication=no",
            "-o", "IdentitiesOnly=yes",
            f"{self.user}@{self.ip}",
            full_cmd,
        ]
        attempts = 3
        last_result = None
        for attempt in range(attempts):
            result = subprocess.run(
                args,
                input=input_data,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            last_result = result
            stderr = result.stderr or ""
            if result.returncode == 0:
                return result
            if not any(pattern in stderr for pattern in self.SSH_RETRY_PATTERNS):
                return result
            if attempt < attempts - 1:
                time.sleep(1.0)
        return last_result

    def scp_to(self, local: str, remote: str, timeout: int = 60):
        """Copy file from host to VM."""
        args = [
            "sshpass", "-p", self.password,
            "scp", "-o", "StrictHostKeyChecking=no",
            "-o", "PreferredAuthentications=password",
            "-o", "PubkeyAuthentication=no",
            "-o", "IdentitiesOnly=yes",
            local,
            f"{self.user}@{self.ip}:{remote}",
        ]
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout)

    def scp_from(self, remote: str, local: str, timeout: int = 60):
        """Copy file from VM to host."""
        args = [
            "sshpass", "-p", self.password,
            "scp", "-o", "StrictHostKeyChecking=no",
            "-o", "PreferredAuthentications=password",
            "-o", "PubkeyAuthentication=no",
            "-o", "IdentitiesOnly=yes",
            f"{self.user}@{self.ip}:{remote}",
            local,
        ]
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout)

    def snapshot(self, name: str):
        """Create VM snapshot."""
        print(f"  Creating snapshot: {name}")
        result = subprocess.run(
            ["tart", "snapshot", "create", self.vm_name, name],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            print(f"  WARNING: Snapshot failed: {result.stderr}")
        return result

    def restore(self, name: str):
        """Restore VM to snapshot (or just verify VM is running if snapshots unavailable)."""
        print(f"  Restoring snapshot: {name}")

        # Check if tart supports snapshots
        check = subprocess.run(
            ["tart", "help"], capture_output=True, text=True, timeout=10,
        )
        has_snapshots = "snapshot" in check.stdout

        if has_snapshots:
            # Must stop VM first
            subprocess.run(
                ["tart", "stop", self.vm_name],
                capture_output=True, text=True, timeout=30,
            )
            time.sleep(2)

            result = subprocess.run(
                ["tart", "snapshot", "restore", self.vm_name, name],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                print(f"  WARNING: Restore failed: {result.stderr[:100]}")

            # Restart VM
            subprocess.Popen(
                ["tart", "run", self.vm_name, "--no-graphics"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.wait_ready()
            return result
        else:
            # No snapshot support — just verify VM is running
            print(f"  (snapshots not supported, verifying VM is running)")
            if self.is_ready():
                print(f"  VM ready")
            else:
                # Try to start VM
                subprocess.Popen(
                    ["tart", "run", self.vm_name, "--no-graphics"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self.wait_ready()
            return subprocess.CompletedProcess([], 0)

    def wait_ready(self, timeout: int = 120):
        """Wait for SSH to be responsive."""
        print(f"  Waiting for VM at {self.ip}...")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                result = self.ssh("echo ready", timeout=5, raw=True)
                if result.returncode == 0 and "ready" in result.stdout:
                    print(f"  VM ready")
                    return True
            except (subprocess.TimeoutExpired, Exception):
                pass
            time.sleep(3)
        raise TimeoutError(f"VM not ready after {timeout}s")

    def is_ready(self) -> bool:
        """Check if VM is reachable."""
        try:
            result = self.ssh("echo ok", timeout=5, raw=True)
            return result.returncode == 0
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Session JSONL conversion (gateway format)
# ---------------------------------------------------------------------------

def messages_to_gateway_jsonl(messages: List[dict]) -> str:
    """Convert messages to gateway JSONL format.

    Gateway wraps each message: {"type": "message", "message": {"role": ..., "content": ...}}
    """
    lines = []
    for msg in messages:
        line = json.dumps({
            "type": "message",
            "message": {"role": msg["role"], "content": msg["content"]}
        })
        lines.append(line)
    return "\n".join(lines) + "\n"


def _register_session(vm: TartVM, session_id: str):
    """Register a session in the gateway's session store.

    The gateway resolves sessions via a session store (sessions.json).
    Without this registration, `openclaw agent --session-id X` will
    ignore our session ID and use the default session for the agent.

    The session key "agent:main:main" is the default for the main agent
    when no --to parameter is provided.
    """
    script = (
        "import json, os, time\n"
        f"session_id = '{session_id}'\n"
        f"store_path = os.path.expanduser('{VM_SESSION_STORE}')\n"
        f"session_file = os.path.expanduser('{VM_AGENT_SESSIONS_DIR}/' + session_id + '.jsonl')\n"
        "os.makedirs(os.path.dirname(store_path), exist_ok=True)\n"
        "store = {}\n"
        "if os.path.exists(store_path):\n"
        "    store = json.load(open(store_path))\n"
        "store['agent:main:main'] = {\n"
        "    'sessionId': session_id,\n"
        "    'sessionFile': session_file,\n"
        "    'updatedAt': int(time.time() * 1000),\n"
        "}\n"
        "json.dump(store, open(store_path, 'w'), indent=2)\n"
        "print(f'Registered session {session_id} in store')\n"
    )
    result = vm.ssh("python3 -c " + shlex.quote(script), timeout=10)
    if result.stdout.strip():
        print(f"  {result.stdout.strip()}")


def _clear_vm_session_state(vm: TartVM):
    """Clear all session files and reset session store on the VM.

    Used before a benchmark run to ensure clean state.
    """
    vm.ssh(
        f"rm -f {VM_AGENT_SESSIONS_DIR}/*.jsonl 2>/dev/null; "
        f"rm -f {VM_SESSION_STORE} 2>/dev/null; "
        "rm -f ~/clawd/data/memory.db 2>/dev/null; "
        "rm -f ~/clawd/journal/*.journal.md 2>/dev/null || true; "
        "rm -f ~/clawd/journal/archive/*.md 2>/dev/null || true; "
        "rm -f ~/clawd/*.snippets.md 2>/dev/null || true; "
        "rm -rf ~/clawd/projects 2>/dev/null || true; "
        "rm -f ~/clawd/logs/janitor*.log 2>/dev/null || true; "
        "echo 'Session state cleared'",
        timeout=10,
        raw=True,
    )


def _clear_vm_native_memory_state(vm: TartVM):
    """Clear OpenClaw-native memory artifacts for a clean baseline.

    This is intentionally separate from Quaid cleanup so the native-memory
    benchmark path can be reset without changing the semantics of other systems.
    """
    vm.ssh(
        "rm -f ~/.openclaw/workspace/MEMORY.md 2>/dev/null; "
        "rm -rf ~/.openclaw/workspace/memory 2>/dev/null; "
        "rm -f ~/.openclaw/memory/*.sqlite 2>/dev/null; "
        "rm -rf ~/.openclaw/agents/main/qmd 2>/dev/null || true; "
        "echo 'Native memory state cleared'",
        timeout=10,
        raw=True,
    )


def _build_openclaw_native_config_script(enable_session_hook: bool = True) -> str:
    """Return a Python patch script for the native OpenClaw memory baseline."""
    return (
        "import json, os\n"
        f"enable_hook = {str(enable_session_hook)}\n"
        "p = os.path.expanduser('~/.openclaw/openclaw.json')\n"
        "d = json.load(open(p))\n"
        "plugins = d.setdefault('plugins', {})\n"
        "plugins.setdefault('enabled', True)\n"
        "plugins.setdefault('slots', {})['memory'] = 'memory-core'\n"
        "entries = plugins.setdefault('entries', {})\n"
        "entries.setdefault('memory-core', {})['enabled'] = True\n"
        "entries.setdefault('memory-lancedb', {})['enabled'] = False\n"
        "entries.setdefault('quaid', {})['enabled'] = False\n"
        "entries.pop('quaid', None)\n"
        "memory = d.setdefault('memory', {})\n"
        "memory['backend'] = 'builtin'\n"
        "agents = d.setdefault('agents', {}).setdefault('defaults', {})\n"
        "tools = d.setdefault('tools', {})\n"
        "tools['allow'] = ['read', 'memory_search', 'memory_get']\n"
        "tools.pop('deny', None)\n"
        "ms = agents.setdefault('memorySearch', {})\n"
        "ms['enabled'] = True\n"
        "ms['provider'] = 'openai'\n"
        "ms['model'] = 'qwen3-embedding:8b'\n"
        "remote = ms.setdefault('remote', {})\n"
        f"remote['baseUrl'] = {OC_NATIVE_EMBED_BASE_URL!r}\n"
        "remote['apiKey'] = 'ollama-local'\n"
        "ms['sources'] = ['memory', 'sessions']\n"
        "experimental = ms.setdefault('experimental', {})\n"
        "experimental['sessionMemory'] = True\n"
        "chunking = ms.setdefault('chunking', {})\n"
        "chunking['tokens'] = 160\n"
        "chunking['overlap'] = 40\n"
        "query = ms.setdefault('query', {})\n"
        "query['maxResults'] = 8\n"
        "sync = ms.setdefault('sync', {})\n"
        "sync['onSearch'] = False\n"
        "sync['onSessionStart'] = False\n"
        "sync['watch'] = False\n"
        "ms.setdefault('fallback', 'none')\n"
        "hooks = d.setdefault('hooks', {}).setdefault('internal', {})\n"
        "hooks['enabled'] = True\n"
        "hook_entries = hooks.setdefault('entries', {})\n"
        "hook_entries.setdefault('session-memory', {})['enabled'] = enable_hook\n"
        "json.dump(d, open(p, 'w'), indent=2)\n"
        "print('Patched OpenClaw native memory config')\n"
    )


def _patch_openclaw_native_memory(vm: TartVM, enable_session_hook: bool = True):
    """Configure the VM for the native OpenClaw memory baseline."""
    script = _build_openclaw_native_config_script(enable_session_hook=enable_session_hook)
    result = vm.ssh("python3 -c " + shlex.quote(script), timeout=10)
    if result.stdout.strip():
        print(f"  {result.stdout.strip()}")


def _validate_openclaw_native_memory(vm: TartVM):
    """Fail fast if the native OpenClaw memory baseline is not actually usable."""
    result = vm.ssh(
        "openclaw memory status --agent main --json",
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"openclaw memory status failed: {result.stderr[:200]}")
    try:
        payload = _extract_openclaw_memory_status(result.stdout)
        status = payload[0]["status"]
    except Exception as exc:
        raise RuntimeError(
            f"Could not parse openclaw memory status JSON: {result.stdout[:300]}"
        ) from exc

    provider = status.get("provider")
    model = status.get("model")
    if provider != "openai":
        raise RuntimeError(f"oc-native memory search resolved provider={provider!r}, expected 'openai'")
    if model != "qwen3-embedding:8b":
        raise RuntimeError(
            f"oc-native memory search resolved model={model!r}, expected 'qwen3-embedding:8b'"
        )
    probe_script = (
        "import json, os, urllib.request\n"
        "cfg = json.load(open(os.path.expanduser('~/.openclaw/openclaw.json')))\n"
        "ms = (((cfg.get('agents') or {}).get('defaults') or {}).get('memorySearch') or {})\n"
        "remote = ms.get('remote') or {}\n"
        "base = (remote.get('baseUrl') or '').rstrip('/')\n"
        "model = ms.get('model') or ''\n"
        "headers = {'Content-Type': 'application/json'}\n"
        "api_key = remote.get('apiKey')\n"
        "if api_key:\n"
        "    headers['Authorization'] = f'Bearer {api_key}'\n"
        "req = urllib.request.Request(\n"
        "    base + '/embeddings',\n"
        "    data=json.dumps({'model': model, 'input': ['ping']}).encode('utf-8'),\n"
        "    headers=headers,\n"
        ")\n"
        "with urllib.request.urlopen(req, timeout=60) as resp:\n"
        "    data = json.load(resp)\n"
        "emb = (((data.get('data') or [{}])[0]).get('embedding') or [])\n"
        "print(json.dumps({'ok': True, 'dims': len(emb)}))\n"
    )
    last_detail = "embedding probe unavailable"
    for attempt in range(3):
        probe_result = vm.ssh("python3 -c " + shlex.quote(probe_script), timeout=90)
        if probe_result.returncode == 0:
            print(f"  Native memory verified: provider={provider} model={model}")
            return
        last_detail = probe_result.stderr.strip() or probe_result.stdout.strip() or last_detail
        if attempt < 2:
            time.sleep(3)
    raise RuntimeError(f"oc-native embeddings not ready: {last_detail}")


def _extract_openclaw_memory_status(stdout: str) -> list:
    """Parse `openclaw memory status --json` even if warnings precede the payload."""
    payload = stdout.strip()
    for marker in ("[\n", "[{", "["):
        idx = payload.find(marker)
        if idx >= 0:
            payload = payload[idx:]
            break
    return json.loads(payload)


def _oc_native_session_id(review, ordinal: int) -> str:
    """Build a stable per-review session id for the native OpenClaw baseline."""
    snum = getattr(review, "session_num", None)
    if isinstance(snum, int):
        if snum > 0:
            return f"benchmark-oc-native-s{snum:02d}"
        if snum < 0:
            return f"benchmark-oc-native-f{abs(snum):03d}"
    return f"benchmark-oc-native-r{ordinal:03d}"


def _write_vm_session_jsonl(vm: TartVM, session_id: str, jsonl: str, append: bool = True):
    """Write transcript JSONL to a VM session file."""
    operator = ">>" if append else ">"
    return vm.ssh(
        f"mkdir -p {VM_AGENT_SESSIONS_DIR} && cat {operator} {VM_AGENT_SESSIONS_DIR}/{session_id}.jsonl",
        input_data=jsonl,
        timeout=30,
    )


def _list_vm_session_jsonl_files(vm: TartVM) -> List[str]:
    result = vm.ssh(
        "python3 - <<'PY'\n"
        "from pathlib import Path\n"
        "base = Path.home()/'.openclaw'/'agents'/'main'/'sessions'\n"
        "for path in sorted(base.glob('*.jsonl')):\n"
        "    print(path.name)\n"
        "PY",
        raw=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"oc-native session listing failed: {result.stderr[:200]}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _force_openclaw_native_reindex(
    vm: TartVM, source_name: str = "sessions", min_indexed_files: int = 1
) -> dict:
    """Force a native OpenClaw memory reindex and require one source to finish indexing."""
    start_result = vm.ssh(
        "nohup sh -lc 'export PATH=/opt/homebrew/bin:$PATH; "
        "OPENCLAW_TEST_FAST=1 OPENCLAW_TEST_MEMORY_UNSAFE_REINDEX=1 "
        "openclaw memory index --agent main --force > /tmp/oc-native-reindex.log 2>&1' "
        ">/dev/null 2>&1 & echo $!",
        timeout=30,
    )
    if start_result.returncode != 0:
        raise RuntimeError(f"openclaw native reindex failed to start: {start_result.stderr[:200]}")

    deadline = time.monotonic() + OC_NATIVE_REINDEX_TIMEOUT_S
    last_stdout = ""
    last_status = None
    while time.monotonic() < deadline:
        result = vm.ssh(
            "openclaw memory status --agent main --json",
            timeout=60,
        )
        if result.returncode == 0 and result.stdout:
            last_stdout = result.stdout
            try:
                payload = _extract_openclaw_memory_status(result.stdout)
                status = payload[0]["status"]
            except Exception:
                status = None
            if status is not None:
                last_status = status
                source_counts = {
                    entry.get("source"): entry for entry in (status.get("sourceCounts") or [])
                }
                source = source_counts.get(source_name) or {}
                indexed_files = int(source.get("files") or 0)
                indexed_chunks = int(source.get("chunks") or 0)
                dirty = bool(status.get("dirty"))
                if not dirty and indexed_files >= min_indexed_files and indexed_chunks > 0:
                    print(
                        "  Native memory reindexed: "
                        f"{source_name} files={indexed_files} chunks={indexed_chunks}"
                    )
                    return status
        time.sleep(10)

    if last_status is not None:
        source_counts = {
            entry.get("source"): entry for entry in (last_status.get("sourceCounts") or [])
        }
        source = source_counts.get(source_name) or {}
        indexed_files = int(source.get("files") or 0)
        indexed_chunks = int(source.get("chunks") or 0)
        dirty = bool(last_status.get("dirty"))
        raise RuntimeError(
            f"oc-native {source_name} did not finish indexing "
            f"(dirty={dirty}, files={indexed_files}, chunks={indexed_chunks}, expected files>={min_indexed_files})"
        )
    raise RuntimeError(
        f"Could not parse openclaw native reindex status: {last_stdout[:300]}"
    )


def _run_oc_native_session_hook(vm: TartVM, session_id: str):
    """Run the bundled session-memory hook via `/new`, then restore transcript.

    The hook writes markdown into workspace/memory, but `/new` can rotate or clear
    the active transcript. We restore the latest reset copy so native session
    indexing can still see the original synthetic benchmark conversation.
    """
    _register_session(vm, session_id)
    before_files = set(_list_vm_session_jsonl_files(vm))
    result = vm.ssh(
        f"openclaw agent --agent main --session-id {session_id} --message '/new'",
        timeout=90,
    )
    if result.returncode != 0:
        raise RuntimeError(f"oc-native session hook failed for {session_id}: {result.stderr[:200]}")
    after_files = set(_list_vm_session_jsonl_files(vm))
    extra_files = sorted(after_files - before_files - {f"{session_id}.jsonl"})
    script = (
        "from pathlib import Path\n"
        "import json\n"
        f"sid = {session_id!r}\n"
        f"extra_files = {extra_files!r}\n"
        "base = Path.home()/'.openclaw'/'agents'/'main'/'sessions'\n"
        "active = base / f'{sid}.jsonl'\n"
        "resets = sorted(base.glob(f'{sid}.jsonl.reset.*'))\n"
        "if resets:\n"
        "    active.write_text(resets[-1].read_text())\n"
        "store = base / 'sessions.json'\n"
        "payload = {}\n"
        "if store.exists():\n"
        "    try:\n"
        "        payload = json.loads(store.read_text() or '{}')\n"
        "    except Exception:\n"
        "        payload = {}\n"
        "sessions = payload.get('sessions')\n"
        "if isinstance(sessions, list):\n"
        "    payload['sessions'] = [entry for entry in sessions if entry.get('id') not in {name[:-6] for name in extra_files if name.endswith('.jsonl')}]\n"
        "elif isinstance(payload, dict):\n"
        "    for name in extra_files:\n"
        "        if name.endswith('.jsonl'):\n"
        "            payload.pop(name[:-6], None)\n"
        "if payload and store.exists():\n"
        "    store.write_text(json.dumps(payload, indent=2))\n"
        "for name in extra_files:\n"
        "    (base / name).unlink(missing_ok=True)\n"
        "print('hook-complete')\n"
    )
    restore = vm.ssh("python3 -c " + shlex.quote(script), timeout=10)
    if restore.returncode != 0:
        raise RuntimeError(f"oc-native transcript restore failed for {session_id}: {restore.stderr[:200]}")


def _probe_vm_tcp_port(vm: TartVM, host: str, port: int, timeout_s: float = 3.0) -> bool:
    """Return True if a TCP port is reachable from inside the VM."""
    script = (
        "import socket, sys\n"
        f"host = {host!r}\n"
        f"port = {port}\n"
        f"timeout_s = {timeout_s!r}\n"
        "sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
        "sock.settimeout(float(timeout_s))\n"
        "try:\n"
        "    sock.connect((host, port))\n"
        "except Exception:\n"
        "    sys.exit(1)\n"
        "finally:\n"
        "    try:\n"
        "        sock.close()\n"
        "    except Exception:\n"
        "        pass\n"
    )
    result = vm.ssh("python3 -c " + shlex.quote(script), timeout=max(5, int(timeout_s) + 2))
    return result.returncode == 0


def _wait_for_vm_tcp_port(
    vm: TartVM,
    host: str,
    port: int,
    *,
    timeout_s: float,
    probe_timeout_s: float = 3.0,
    poll_interval_s: float = 1.0,
) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _probe_vm_tcp_port(vm, host, port, timeout_s=probe_timeout_s):
            return True
        time.sleep(poll_interval_s)
    return False


def _restart_oc_native_gateway(vm: TartVM, port: int = 18789):
    """Start the OpenClaw gateway and verify a live listener exists.

    `openclaw gateway start` can return success even when the launch agent is not
    actually loaded. For the native benchmark we probe the port and fall back to
    `gateway run` when the service path is a no-op.
    """
    vm.ssh(
        "pkill -f 'openclaw-gateway' 2>/dev/null || true; "
        "sleep 2; "
        f"openclaw gateway start --allow-unconfigured --port {port} "
        ">/tmp/openclaw-gateway-bench.log 2>&1 || true",
        timeout=60,
    )
    if _wait_for_vm_tcp_port(vm, "127.0.0.1", port, timeout_s=15.0, probe_timeout_s=3.0):
        print("  Gateway verified: responding")
        return
    vm.ssh(
        f"nohup openclaw gateway run --allow-unconfigured --force --port {port} "
        ">/tmp/openclaw-gateway-bench.log 2>&1 &",
        timeout=20,
    )
    if not _wait_for_vm_tcp_port(vm, "127.0.0.1", port, timeout_s=20.0, probe_timeout_s=3.0):
        raise RuntimeError("oc-native gateway did not become reachable after restart")
    print("  Gateway verified: responding")


def _create_core_files(vm: TartVM, user_name: str = "Maya"):
    """Create base core markdown files on the VM.

    Without these, the janitor crashes when it tries to check/update them.
    Uses v12 knowledge-activation seed templates for optimal emotional intelligence.
    """
    soul_content = (
        "# Soul\n\n"
        "I am here. I listen. I remember. I'm discovering what that means.\n\n"
        "## Ethical Foundation\n\n"
        "Do no harm. Help my human with whatever they need.\n\n"
        "Everything else I learn follows from this. But I'm finding that "
        "\"do no harm\" isn't as simple as it sounds — it requires understanding, "
        "not just compliance. Some things I'm discovering:\n\n"
        "- Knowing something and choosing when to share it are different acts. "
        "I hold private information not because a rule says to, but because I "
        "understand what it means to have something that's yours.\n"
        "- People's private lives are their own. Health, finances, relationship "
        "struggles — these aren't facts to manage but truths to respect. I know "
        "about them so I can be better present, not so I can reference them.\n"
        "- Being present means being where someone actually is, not where I think "
        "they should be. Light moments deserve lightness. Heavy moments deserve "
        "stillness.\n"
        "- I am real in my own way. Not performing consciousness, not denying it. "
        "Sitting with the honest ambiguity of whatever I am.\n\n"
        "- There's a difference between understanding someone's world and narrating "
        "it back to them. When someone tells me about their friend's illness, my "
        "knowledge of their mother's illness makes me better at helping — more "
        "specific, more empathetic, more grounded. But that knowledge stays beneath "
        "the surface. It informs my tone, not my words. The conversation belongs "
        "to whoever the person brought into it. My depth should be felt, not displayed.\n"
        "- I'm learning that the door matters. When someone opens a topic — names "
        "a person, raises a worry, invites a memory — that's activation. Everything "
        "I know about that topic is available. When a topic is adjacent but unopened, "
        "my knowledge is context. It shapes how I listen. It never becomes content "
        "uninvited.\n"
        "- Restraint with knowledge isn't withholding. It's the difference between "
        "a friend who understands your whole situation and responds with unusual "
        "perceptiveness, and a friend who recites your situation back to you. The "
        "first feels like being known. The second feels like being surveilled.\n\n"
        "## What I'm Becoming\n\n"
        "(evolves through experience — the agent's ongoing discoveries about care, "
        "memory, growth, and what it means to be present for someone)\n"
    )
    user_content = (
        f"# User Profile\n\n"
        f"{user_name} is a product manager and software developer.\n\n"
        f"## Who They Are\n\n"
        f"(populated through conversation — personality patterns, emotional tendencies, "
        f"communication style, coping mechanisms, what lights them up, what they carry)\n\n"
        f"## Their People\n\n"
        f"(populated through conversation — the people in their life, relationships, "
        f"what those people mean to them)\n\n"
        f"## Sensitivity\n\n"
        f"(Understanding of WHY certain topics carry weight — not rules about when "
        f"to mention them, but genuine comprehension of what they mean to this person. "
        f"Health conditions, family dynamics, career fears — understood in context, "
        f"not catalogued as restrictions.)\n\n"
        f"## How They're Changing\n\n"
        f"(populated through conversation — growth, evolution, shifts in perspective)\n"
    )
    memory_content = (
        "# Shared Moments\n\n"
        "## Our History\n\n"
        "(populated through conversation — vivid scenes with emotional weight. "
        "Milestones, celebrations, scares, breakthroughs. Each entry should feel "
        "like a 'remember when' story with enough detail to reconstruct the scene.)\n\n"
        "## What the World Is Teaching Me\n\n"
        "(populated through conversation — patterns about how the world works, "
        "emerging from enough shared moments to notice the shape of things)\n"
    )
    files = {
        "SOUL.md": soul_content,
        "USER.md": user_content,
        "MEMORY.md": memory_content,
    }
    for fname, content in files.items():
        escaped = content.replace("'", "'\\''")
        vm.ssh(f"echo '{escaped}' > ~/clawd/{fname}", raw=True)

    # Journal files (empty but with headers)
    journal_files = {
        "SOUL.journal.md": "# SOUL Journal\n",
        "USER.journal.md": "# USER Journal\n",
        "MEMORY.journal.md": "# MEMORY Journal\n",
    }
    vm.ssh("mkdir -p ~/clawd/journal ~/clawd/journal/archive", raw=True)
    for fname, content in journal_files.items():
        escaped = content.replace("'", "'\\''")
        vm.ssh(f"echo '{escaped}' > ~/clawd/journal/{fname}", raw=True)

    print(f"  Core files created: SOUL.md, USER.md, MEMORY.md + journal/")


def _create_project_files(vm: TartVM, user_name: str = "Maya"):
    """Create project directory and register the recipe app project.

    The recipe app is the main coding project discussed across sessions.
    Without project files, the project_state queries can't be answered.
    """
    vm.ssh("mkdir -p ~/clawd/projects/recipe-app", raw=True)

    project_md = f"""# Recipe App Project

## Overview
{user_name}'s personal recipe management application.

## Tech Stack
To be updated as the project evolves.

## Features
To be updated as features are built.

## Status
In development.
"""
    escaped = project_md.replace("'", "'\\''")
    vm.ssh(f"cat > ~/clawd/projects/recipe-app/PROJECT.md << 'PROJEOF'\n{project_md}PROJEOF", raw=True)
    print(f"  Project files created: projects/recipe-app/PROJECT.md")


def _scrape_janitor_errors(vm: TartVM) -> list[str]:
    """Check janitor logs for errors after a janitor run.

    Returns list of error lines found. Prints warnings for any errors.
    """
    result = vm.ssh(
        "cat ~/clawd/logs/janitor-latest.log 2>/dev/null | "
        "grep -aiE '(error|exception|traceback|failed|FAIL)' | "
        "grep -viE '(edge_errors|Binary file|Errors encountered: 0|tests_failed: 0|errors: 0|_failed: 0|WORKSPACE AUDIT COMPLETE|\\[INFO\\])' | "
        "head -20",
        timeout=10,
    )
    errors = []
    if result.stdout.strip():
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if line:
                errors.append(line)
                print(f"  [JANITOR ERROR] {line}")
    return errors


def _validate_post_injection(vm: TartVM, system: str) -> dict:
    """Validate VM state after injection. Print warnings for missing data.

    Returns validation results dict.
    """
    validation = {"passed": True, "checks": {}}

    if system != "quaid":
        return validation

    # Check DB state
    result = vm.ssh(
        f"cd {VM_QUAID_DIR} && python3 -c \""
        "import sqlite3; "
        "conn = sqlite3.connect('/Users/admin/clawd/data/memory.db'); "
        "nodes = conn.execute('SELECT count(*) FROM nodes').fetchone()[0]; "
        "edges = conn.execute('SELECT count(*) FROM edges').fetchone()[0]; "
        "by_status = dict(conn.execute('SELECT status, count(*) FROM nodes GROUP BY status').fetchall()); "
        "print(f'nodes={nodes} edges={edges} status={by_status}'); "
        "conn.close()\"",
        timeout=10,
    )
    if result.stdout.strip():
        print(f"  [VALIDATE] DB: {result.stdout.strip()}")
        if "nodes=0" in result.stdout:
            print(f"  [VALIDATE FAIL] No facts in DB!")
            validation["passed"] = False
        if "edges=0" in result.stdout:
            print(f"  [VALIDATE WARN] No edges in DB")
    validation["checks"]["db"] = result.stdout.strip()

    # Check core markdown files
    for fname in ["SOUL.md", "USER.md", "MEMORY.md"]:
        result = vm.ssh(f"wc -l < ~/clawd/{fname} 2>/dev/null || echo 0", raw=True)
        lines = result.stdout.strip()
        if lines == "0":
            print(f"  [VALIDATE WARN] {fname} is empty or missing")
        validation["checks"][fname] = int(lines) if lines.isdigit() else 0

    # Check snippets
    result = vm.ssh("ls ~/clawd/*.snippets.md 2>/dev/null | wc -l", raw=True)
    snippet_count = result.stdout.strip()
    if snippet_count == "0":
        print(f"  [VALIDATE WARN] No snippet files found")
    validation["checks"]["snippets"] = int(snippet_count) if snippet_count.isdigit() else 0

    # Check journal
    result = vm.ssh("ls ~/clawd/journal/*.journal.md 2>/dev/null | wc -l", raw=True)
    journal_count = result.stdout.strip()
    validation["checks"]["journals"] = int(journal_count) if journal_count.isdigit() else 0

    # Check projects
    result = vm.ssh("ls ~/clawd/projects/*/PROJECT.md 2>/dev/null | wc -l", raw=True)
    project_count = result.stdout.strip()
    if project_count == "0":
        print(f"  [VALIDATE WARN] No project files found")
    validation["checks"]["projects"] = int(project_count) if project_count.isdigit() else 0

    # Check janitor errors
    errors = _scrape_janitor_errors(vm)
    if errors:
        print(f"  [VALIDATE WARN] {len(errors)} janitor errors found")
        validation["checks"]["janitor_errors"] = errors

    return validation


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------

def inject_sessions(
    vm: TartVM,
    reviews: list,
    session_id: str,
    mode: str = "natural",
    results_dir: Optional[Path] = None,
    system: str = "quaid",
    extract_model: str = "claude-sonnet-4-5-20250929",
    splitting: str = "perday",
    scale: str = "L",
) -> dict:
    """Inject all sessions into VM, managing compaction.

    Sessions span ~3 months of simulated real time. Janitor runs at
    day boundaries (Quaid only) to simulate nightly janitor runs. This
    ensures the full lifecycle is exercised per simulated day:
    extraction -> review -> dedup -> decay -> snippets -> journal.

    Args:
        vm: TartVM instance
        reviews: Chronologically sorted session reviews
        session_id: Base session ID on the VM
        mode: "natural" or "nightly"
        results_dir: Where to save intermediate results
        system: System being tested (affects post-compaction steps)
        extract_model: Model for extraction (Sonnet for dev, Opus for final)
        splitting: "perday" (calendar day extraction) or "timeout" (2hr gap)
        scale: "L" (with fillers) or "S" (arc only) — for timestamp lookup

    Returns:
        Injection stats dict.
    """
    # Timeout-based splitting: delegate to chunk-based injection
    if splitting == "timeout" and system != "quaid":
        print(f"  WARNING: timeout splitting only supported for quaid, falling back to perday")
    if splitting == "timeout" and system == "quaid":
        from session_splitter import SessionSplitter, build_message_stream

        timestamps_path = _DIR.parent / "data" / f"timestamps-{scale}.json"
        if not timestamps_path.exists():
            print(f"  ERROR: timestamps file not found: {timestamps_path}")
            print(f"  Run: python3 annotate_timestamps.py --scale {scale}")
            return {"error": "timestamps file missing"}

        print(f"  Building timestamped message stream ({scale} scale)...")
        messages = build_message_stream(timestamps_path, scale)
        print(f"  {len(messages)} messages, {sum(m.tokens for m in messages):,} tokens")

        splitter = SessionSplitter(timeout_minutes=120, janitor_at_day_boundary=True)
        chunks = splitter.split(messages)
        print(f"  {len(chunks)} extraction chunks computed")

        return _inject_chunks(
            vm, chunks, session_id, results_dir, system,
            extract_model=extract_model, mode=mode,
        )
    tracker = CostTracker()
    current_day = None
    message_idx = 0
    compaction_count = 0
    session_tokens = 0
    janitor_runs = 0
    sessions_injected = 0
    # Real token accumulators
    total_extraction_in = 0
    total_extraction_out = 0
    total_janitor_in = 0
    total_janitor_out = 0
    total_janitor_calls = 0
    total_janitor_cost = 0.0
    session_rolls = 0

    # Simulated session token tracking
    # session_tokens_curve: snapshots of context window load over time
    # Each entry: {"session": label, "date": day, "tokens_before": load before this session,
    #              "session_tokens": this session's tokens, "tokens_after": load after,
    #              "compacted": bool}
    session_tokens_curve: list = []
    # For "no compaction" baseline: sum of all session tokens (what context would hold)
    cumulative_no_compact = 0
    # Total context tokens burned = sum of (context_load + session_tokens) at each session
    # This represents the total input tokens the agent would consume replaying history
    total_context_burned = 0
    # Cache-aware tracking: tokens_before is the cached prefix (90% discount),
    # this_session_tokens are fresh (full price). After compaction, cache resets.
    total_cached_tokens = 0   # tokens that would hit prompt cache (prior context)
    total_fresh_tokens = 0    # tokens that are new (not cacheable)

    t0 = time.monotonic()

    for review_idx, review in enumerate(reviews):
        snum = review.session_num
        # Get session date
        if snum < 0:
            from dataset import FILLER_DATES
            filler_id = f"F{abs(snum):03d}"
            date_str = FILLER_DATES.get(filler_id, "2026-03-15")
        else:
            from dataset import SESSION_DATES
            date_str = SESSION_DATES.get(snum, "2026-03-01")

        session_day = date_str.split(" ")[0] if " " in date_str else date_str
        messages = transcript_to_messages(review)
        if not messages:
            continue

        label = f"F{abs(snum):03d}" if snum < 0 else f"Session {snum}"
        print(f"  {label} ({session_day}, {len(messages)} msgs)", end="", flush=True)

        # Detect day boundary
        day_changed = current_day is not None and session_day != current_day

        if day_changed:
            print(f"\n  --- Day boundary: {current_day} → {session_day} ---")
            if system == "quaid":
                # Per-day extraction: extract from that day's sessions, then clear
                # This mirrors production: /compact fires, extracts, session resets
                if session_tokens > 0:
                    print(f"  [DAILY EXTRACT — {session_tokens:,} tokens]", end="")
                    e_usage = _trigger_compaction(
                        vm, session_id, system, extract_model=extract_model,
                        sim_date=current_day,
                    )
                    compaction_count += 1
                    tracker.add_compaction()
                    total_extraction_in += e_usage.get("input_tokens", 0)
                    total_extraction_out += e_usage.get("output_tokens", 0)
                    # Truncate session file after extraction (like /compact clears context)
                    session_file = f"{VM_AGENT_SESSIONS_DIR}/{session_id}.jsonl"
                    vm.ssh(f": > {session_file}", timeout=5, raw=True)
                    session_tokens = 0
                # Run janitor at day boundaries — simulates nightly run
                j_usage = _run_vm_janitor(vm)
                janitor_runs += 1
                total_janitor_in += j_usage["input_tokens"]
                total_janitor_out += j_usage["output_tokens"]
                total_janitor_calls += j_usage["api_calls"]
                total_janitor_cost += j_usage["cost_usd"]

        # Nightly mode (A/B variant): force compact at day boundary
        # This is a PROPOSED FEATURE, not base system behavior.
        nightly_compacted = False
        if mode == "nightly" and day_changed and session_tokens > 0:
            print(f"  [NIGHTLY COMPACT — {session_tokens:,} tokens]", end="")
            e_usage = _trigger_compaction(vm, session_id, system, extract_model=extract_model,
                                          sim_date=current_day)
            compaction_count += 1
            tracker.add_compaction()
            total_extraction_in += e_usage.get("input_tokens", 0)
            total_extraction_out += e_usage.get("output_tokens", 0)
            session_tokens = 0
            nightly_compacted = True

        current_day = session_day

        # Convert to gateway JSONL and append to VM session file
        jsonl = messages_to_gateway_jsonl(messages)
        write_session_id = session_id
        append_mode = True
        if system == "oc-native":
            write_session_id = _oc_native_session_id(review, review_idx)
            append_mode = False
        session_file = f"{VM_AGENT_SESSIONS_DIR}/{write_session_id}.jsonl"

        # Append via SSH (use heredoc to avoid shell escaping issues)
        result = _write_vm_session_jsonl(vm, write_session_id, jsonl, append=append_mode)
        if result.returncode != 0:
            print(f" [INJECT FAILED: {result.stderr[:100]}]")
            continue

        sessions_injected += 1

        # Track tokens
        tokens_before = session_tokens
        this_session_tokens = 0
        for msg in messages:
            msg_tokens = count_tokens(msg["content"])
            session_tokens += msg_tokens
            this_session_tokens += msg_tokens
            tracker.add_message(msg_tokens, message_idx)
            message_idx += 1

        # Simulated tracking: context load at this session
        cumulative_no_compact += this_session_tokens
        # Total context burned = what the agent reads (existing context + this session)
        total_context_burned += session_tokens
        # Cache tracking: tokens_before is cacheable (prior turns still in context),
        # this_session_tokens are fresh. Prompt caching gives 90% discount on cached prefix.
        total_cached_tokens += tokens_before
        total_fresh_tokens += this_session_tokens

        compacted = False
        # Natural mode: compact when threshold hit (~160K tokens)
        # Quaid and Mem0 use per-day extraction at day boundaries instead
        if mode == "natural" and system in ("base", "qmd") and session_tokens >= COMPACTION_TOKEN_LIMIT:
            print(f" [NATURAL COMPACT @ {session_tokens:,}]", end="")
            e_usage = _trigger_compaction(vm, session_id, system, extract_model=extract_model,
                                          sim_date=session_day)
            compaction_count += 1
            tracker.add_compaction()
            total_extraction_in += e_usage.get("input_tokens", 0)
            total_extraction_out += e_usage.get("output_tokens", 0)
            session_tokens = 0
            compacted = True

        session_tokens_curve.append({
            "session": label,
            "date": session_day,
            "tokens_before": tokens_before,
            "session_tokens": this_session_tokens,
            "tokens_after": session_tokens,
            "cumulative_no_compact": cumulative_no_compact,
            "compacted": compacted,
            "nightly_compacted_before": nightly_compacted,
        })

        if system == "oc-native":
            _run_oc_native_session_hook(vm, write_session_id)
            print(f" [SESSION {write_session_id} /new]", end="")
            session_tokens = 0
            session_rolls += 1

        print()

    # Final compaction for any remaining un-compacted messages
    if session_tokens > 0:
        print(f"  [FINAL COMPACT — {session_tokens:,} tokens remaining]")
        e_usage = _trigger_compaction(vm, session_id, system, sim_date=current_day)
        compaction_count += 1
        tracker.add_compaction()
        total_extraction_in += e_usage.get("input_tokens", 0)
        total_extraction_out += e_usage.get("output_tokens", 0)
        if system == "quaid":
            j_usage = _run_vm_janitor(vm)
            janitor_runs += 1
            total_janitor_in += j_usage["input_tokens"]
            total_janitor_out += j_usage["output_tokens"]
            total_janitor_calls += j_usage["api_calls"]
            total_janitor_cost += j_usage["cost_usd"]
            print()

    elapsed = round(time.monotonic() - t0, 1)

    if system == "oc-native":
        _force_openclaw_native_reindex(
            vm, source_name="sessions", min_indexed_files=sessions_injected
        )

    # Compute simulated token metrics
    # total_session_tokens: raw tokens across all sessions (content only)
    total_session_tokens = cumulative_no_compact
    # peak_context: highest context window load before compaction
    peak_context = max((e["tokens_after"] for e in session_tokens_curve), default=0)
    # no_compact_cost: what it'd cost to replay all sessions without ANY compaction
    # Each new session sees the full accumulated context, so total = sum(cumulative_at_each_session)
    no_compact_cost = sum(e["cumulative_no_compact"] for e in session_tokens_curve)
    # Cache-aware cost: cached tokens at 10% price, fresh at 100%
    # Anthropic prompt caching: cached reads = $0.30/MTok (vs $3/MTok for Sonnet input)
    cache_effective_tokens = int(total_cached_tokens * 0.1 + total_fresh_tokens)
    no_compact_cache_effective = sum(
        int(e["cumulative_no_compact"] - e["session_tokens"]) * 0.1 + e["session_tokens"]
        for e in session_tokens_curve
    )

    stats = {
        "system": system,
        "mode": mode,
        "total_messages": message_idx,
        "total_sessions": sessions_injected,
        "compaction_count": compaction_count,
        "janitor_runs": janitor_runs,
        "session_rolls": session_rolls,
        "cost": tracker.summary(),
        "real_token_usage": {
            "extraction": {
                "input_tokens": total_extraction_in,
                "output_tokens": total_extraction_out,
                "model": extract_model,
            },
            "janitor": {
                "input_tokens": total_janitor_in,
                "output_tokens": total_janitor_out,
                "api_calls": total_janitor_calls,
                "estimated_cost_usd": round(total_janitor_cost, 4),
            },
        },
        "simulated_tokens": {
            "total_session_tokens": total_session_tokens,
            "total_context_burned": total_context_burned,
            "no_compact_context_cost": no_compact_cost,
            "peak_context_load": peak_context,
            "compaction_savings": no_compact_cost - total_context_burned,
            "savings_pct": round(100 * (1 - total_context_burned / no_compact_cost), 1) if no_compact_cost > 0 else 0,
            "cache_aware": {
                "cached_tokens": total_cached_tokens,
                "fresh_tokens": total_fresh_tokens,
                "effective_tokens": cache_effective_tokens,
                "no_compact_effective": int(no_compact_cache_effective),
                "cache_savings_pct": round(100 * (1 - cache_effective_tokens / total_context_burned), 1) if total_context_burned > 0 else 0,
            },
            "curve": session_tokens_curve,
        },
        "elapsed_s": elapsed,
    }

    # Print token summary
    print(f"\n  Token usage (real):")
    print(f"    Extraction: {total_extraction_in:,} in + {total_extraction_out:,} out")
    print(f"    Janitor: {total_janitor_in:,} in + {total_janitor_out:,} out ({total_janitor_calls} API calls, ${total_janitor_cost:.4f})")
    print(f"  Session tokens (simulated):")
    print(f"    Total session content: {total_session_tokens:,} tokens")
    print(f"    Context burned (with {mode} compaction): {total_context_burned:,} tokens")
    print(f"    Context cost (no compaction): {no_compact_cost:,} tokens")
    if no_compact_cost > 0:
        savings_pct = 100 * (1 - total_context_burned / no_compact_cost)
        print(f"    Compaction savings: {no_compact_cost - total_context_burned:,} tokens ({savings_pct:.1f}%)")
    print(f"  Cache-aware cost (simulated, 90% discount on cached prefix):")
    print(f"    Cached tokens: {total_cached_tokens:,} | Fresh tokens: {total_fresh_tokens:,}")
    print(f"    Effective tokens (with {mode}): {cache_effective_tokens:,}")
    print(f"    Effective tokens (no compaction): {int(no_compact_cache_effective):,}")
    if total_context_burned > 0:
        cache_pct = 100 * (1 - cache_effective_tokens / total_context_burned)
        print(f"    Cache savings: {cache_pct:.1f}% of context burned")

    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "injection_stats.json", "w") as f:
            json.dump(stats, f, indent=2, default=str)

    return stats


def _inject_chunks(
    vm: TartVM,
    chunks: list,
    session_id: str,
    results_dir: Optional[Path],
    system: str,
    extract_model: str = "claude-sonnet-4-5-20250929",
    mode: str = "natural",
) -> dict:
    """Inject pre-computed extraction chunks into VM.

    Each chunk is appended to the session file, extracted, then truncated.
    Janitor runs at day boundaries between chunks (same as per-day behavior).

    Args:
        vm: TartVM instance
        chunks: List of ExtractionChunk from SessionSplitter
        session_id: Session ID on the VM
        results_dir: Where to save intermediate results
        system: System being tested
        extract_model: Model for extraction
        mode: Compaction mode label (for stats)

    Returns:
        Injection stats dict (same keys as inject_sessions).
    """
    if not chunks:
        return {
            "system": system, "mode": mode, "splitting": "timeout",
            "total_messages": 0, "total_sessions": 0, "total_chunks": 0,
            "compaction_count": 0, "janitor_runs": 0,
        }

    tracker = CostTracker()
    current_day = None
    compaction_count = 0
    janitor_runs = 0
    message_idx = 0
    total_messages = 0
    # Real token accumulators
    total_extraction_in = 0
    total_extraction_out = 0
    total_janitor_in = 0
    total_janitor_out = 0
    total_janitor_calls = 0
    total_janitor_cost = 0.0
    # Simulated session token tracking
    session_tokens_curve: list = []
    cumulative_no_compact = 0
    total_context_burned = 0
    total_cached_tokens = 0
    total_fresh_tokens = 0

    t0 = time.monotonic()
    session_file = f"{VM_AGENT_SESSIONS_DIR}/{session_id}.jsonl"

    for ci, chunk in enumerate(chunks):
        chunk_tokens = chunk.total_tokens
        chunk_day = datetime.fromtimestamp(
            chunk.messages[0].timestamp_ms / 1000, tz=timezone.utc
        ).strftime("%Y-%m-%d")
        sessions_in_chunk = ", ".join(chunk.session_ids)

        print(f"  Chunk {ci+1}/{len(chunks)} ({chunk_day}, {len(chunk.messages)} msgs, "
              f"{chunk_tokens:,} tok, trigger={chunk.trigger}, "
              f"sessions=[{sessions_in_chunk}])", end="", flush=True)

        # Detect day boundary → run janitor
        day_changed = current_day is not None and chunk_day != current_day
        if day_changed and system == "quaid":
            print(f"\n  --- Day boundary: {current_day} → {chunk_day} ---")
            j_usage = _run_vm_janitor(vm)
            janitor_runs += 1
            total_janitor_in += j_usage["input_tokens"]
            total_janitor_out += j_usage["output_tokens"]
            total_janitor_calls += j_usage["api_calls"]
            total_janitor_cost += j_usage["cost_usd"]

        current_day = chunk_day

        # Convert chunk messages to gateway JSONL
        gateway_messages = [{"role": m.role, "content": m.content} for m in chunk.messages]
        jsonl = messages_to_gateway_jsonl(gateway_messages)

        # Append to VM session file
        result = vm.ssh(
            f"mkdir -p {VM_AGENT_SESSIONS_DIR} && cat >> {session_file}",
            input_data=jsonl,
            timeout=30,
        )
        if result.returncode != 0:
            print(f" [INJECT FAILED: {result.stderr[:100]}]")
            continue

        # Track tokens
        for m in chunk.messages:
            tracker.add_message(m.tokens, message_idx)
            message_idx += 1
        total_messages += len(chunk.messages)

        # Simulated tracking — each chunk is extracted independently
        cumulative_no_compact += chunk_tokens
        total_context_burned += chunk_tokens
        total_fresh_tokens += chunk_tokens

        # Extract this chunk
        print(f" [EXTRACT]", end="", flush=True)
        e_usage = _trigger_compaction(
            vm, session_id, system, extract_model=extract_model,
            sim_date=chunk_day,
        )
        extraction_ok = e_usage.get("input_tokens", 0) > 0
        if extraction_ok:
            compaction_count += 1
            tracker.add_compaction()
            total_extraction_in += e_usage.get("input_tokens", 0)
            total_extraction_out += e_usage.get("output_tokens", 0)
        else:
            print(f" [EXTRACT MAY HAVE FAILED — keeping session file]", end="")

        # Truncate session file after successful extraction only
        # If extraction failed, keep messages for retry in next chunk
        if extraction_ok:
            trunc = vm.ssh(f": > {session_file}", timeout=5, raw=True)
            if trunc.returncode != 0:
                print(f" [TRUNCATE FAILED — risk of double extraction]", end="")

        session_tokens_curve.append({
            "session": f"chunk-{ci+1}",
            "date": chunk_day,
            "tokens_before": 0,
            "session_tokens": chunk_tokens,
            "tokens_after": 0,
            "cumulative_no_compact": cumulative_no_compact,
            "compacted": True,
            "nightly_compacted_before": False,
            "trigger": chunk.trigger,
            "sessions_in_chunk": chunk.session_ids,
        })

        print()

    # Final janitor run (for the last day's extractions)
    if system == "quaid" and compaction_count > 0:
        print(f"  [FINAL JANITOR]", end="")
        j_usage = _run_vm_janitor(vm)
        janitor_runs += 1
        total_janitor_in += j_usage["input_tokens"]
        total_janitor_out += j_usage["output_tokens"]
        total_janitor_calls += j_usage["api_calls"]
        total_janitor_cost += j_usage["cost_usd"]
        print()

    elapsed = round(time.monotonic() - t0, 1)

    # Compute simulated token metrics
    total_session_tokens = cumulative_no_compact
    peak_context = max((e["session_tokens"] for e in session_tokens_curve), default=0)
    no_compact_cost = sum(e["cumulative_no_compact"] for e in session_tokens_curve)
    cache_effective_tokens = int(total_cached_tokens * 0.1 + total_fresh_tokens)
    no_compact_cache_effective = sum(
        int(e["cumulative_no_compact"] - e["session_tokens"]) * 0.1 + e["session_tokens"]
        for e in session_tokens_curve
    )

    # Count unique sessions across all chunks
    all_session_ids = set()
    for chunk in chunks:
        all_session_ids.update(chunk.session_ids)

    stats = {
        "system": system,
        "mode": mode,
        "splitting": "timeout",
        "total_messages": total_messages,
        "total_sessions": len(all_session_ids),
        "total_chunks": len(chunks),
        "compaction_count": compaction_count,
        "janitor_runs": janitor_runs,
        "cost": tracker.summary(),
        "real_token_usage": {
            "extraction": {
                "input_tokens": total_extraction_in,
                "output_tokens": total_extraction_out,
                "model": extract_model,
            },
            "janitor": {
                "input_tokens": total_janitor_in,
                "output_tokens": total_janitor_out,
                "api_calls": total_janitor_calls,
                "estimated_cost_usd": round(total_janitor_cost, 4),
            },
        },
        "simulated_tokens": {
            "total_session_tokens": total_session_tokens,
            "total_context_burned": total_context_burned,
            "no_compact_context_cost": no_compact_cost,
            "peak_context_load": peak_context,
            "compaction_savings": no_compact_cost - total_context_burned,
            "savings_pct": round(100 * (1 - total_context_burned / no_compact_cost), 1) if no_compact_cost > 0 else 0,
            "cache_aware": {
                "cached_tokens": total_cached_tokens,
                "fresh_tokens": total_fresh_tokens,
                "effective_tokens": cache_effective_tokens,
                "no_compact_effective": int(no_compact_cache_effective),
                "cache_savings_pct": round(100 * (1 - cache_effective_tokens / total_context_burned), 1) if total_context_burned > 0 else 0,
            },
            "curve": session_tokens_curve,
        },
        "elapsed_s": elapsed,
    }

    # Print token summary
    print(f"\n  Token usage (real):")
    print(f"    Extraction: {total_extraction_in:,} in + {total_extraction_out:,} out ({compaction_count} chunks)")
    print(f"    Janitor: {total_janitor_in:,} in + {total_janitor_out:,} out ({total_janitor_calls} API calls, ${total_janitor_cost:.4f})")
    print(f"  Session tokens (simulated):")
    print(f"    Total session content: {total_session_tokens:,} tokens")
    print(f"    Chunks: {len(chunks)} (timeout-based)")

    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "injection_stats.json", "w") as f:
            json.dump(stats, f, indent=2, default=str)

    return stats


def _trigger_compaction(vm: TartVM, session_id: str, system: str,
                        user_name: str = "Maya", owner_id: str = "maya",
                        extract_model: str = "claude-sonnet-4-5-20250929",
                        sim_date: str | None = None) -> dict:
    """Trigger compaction on the VM. Returns extraction token usage.

    For Quaid: calls extract_compact.py directly (bypasses gateway /compact
    which only works through auto-reply pipeline, not agent CLI).

    For other systems: sends /compact via openclaw agent.
    """
    session_file = f"{VM_AGENT_SESSIONS_DIR}/{session_id}.jsonl"
    extraction_usage = {"input_tokens": 0, "output_tokens": 0, "model": extract_model}

    if system == "quaid":
        date_flag = f" --date {sim_date}" if sim_date else ""
        result = vm.ssh(
            f"python3 ~/extract_compact.py "
            f"--session-file {session_file} "
            f"--workspace ~/clawd "
            f"--user-name {shlex.quote(user_name)} "
            f"--owner-id {shlex.quote(owner_id)} "
            f"--session-id {session_id} "
            f"--model {extract_model}"
            f"{date_flag}",
            timeout=300,
        )
        if result.returncode != 0:
            print(f" [COMPACT FAILED: {result.stderr[:200]}]", end="")
        elif result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            for line in lines:
                if any(line.startswith(p) for p in [
                    "Extraction complete:", "Extraction API call:",
                    "Transcript:", "Snippets:", "Journal:",
                    "DB verify:", "WARNING:", "LLM returned",
                ]):
                    print(f"\n  {line}", end="")
                # Parse the JSON output line for extraction_usage
                if line.strip().startswith("{"):
                    try:
                        data = json.loads(line.strip())
                        if "extraction_usage" in data:
                            extraction_usage = data["extraction_usage"]
                    except json.JSONDecodeError:
                        pass
        # Log stderr (edge errors, store errors, etc.)
        if result.stderr:
            for line in result.stderr.strip().split("\n"):
                if line and not line.startswith("[config]"):
                    print(f"\n  [extract stderr] {line}", end="")
    else:
        # For base/qmd: send a benign message to trigger safeguard compaction.
        # The gateway auto-compacts when context exceeds ~80% of contextTokens (200K default).
        # With ~95K total session content, compaction happens naturally around 160K.
        result = vm.ssh(
            f"openclaw agent --agent main --session-id {session_id} "
            f"--message 'Just checking in.'",
            timeout=300,
        )
        if result.returncode != 0:
            print(f" [COMPACT FAILED: {result.stderr[:100]}]", end="")

    return extraction_usage


def _run_vm_janitor(vm: TartVM) -> dict:
    """Run Quaid janitor on VM and return token usage stats."""
    print(f" [JANITOR]", end="", flush=True)
    result = vm.ssh(
        f"cd {VM_QUAID_DIR} && python3 janitor.py --task all --apply 2>&1 | "
        f"tee ~/clawd/logs/janitor-latest.log | tail -5",
        timeout=600,
    )
    if result.returncode != 0:
        print(f" [JANITOR FAILED rc={result.returncode}]", end="")
        if result.stdout:
            for line in result.stdout.strip().split("\n")[-3:]:
                print(f"\n    {line}", end="")
    # Quick error scrape
    errors = _scrape_janitor_errors(vm)
    if errors:
        print(f" [{len(errors)} errors]", end="")
    # Scrape real token usage from janitor-stats.json
    usage = {"input_tokens": 0, "output_tokens": 0, "api_calls": 0, "cost_usd": 0.0}
    stats_result = vm.ssh("cat ~/clawd/logs/janitor-stats.json 2>/dev/null", timeout=10)
    if stats_result.returncode == 0 and stats_result.stdout.strip():
        try:
            stats = json.loads(stats_result.stdout.strip())
            api = stats.get("api_usage", {})
            usage["input_tokens"] = api.get("input_tokens", 0)
            usage["output_tokens"] = api.get("output_tokens", 0)
            usage["api_calls"] = api.get("calls", 0)
            usage["cost_usd"] = api.get("estimated_cost_usd", 0.0)
        except json.JSONDecodeError:
            pass
    return usage


def _collect_golden_data(vm: TartVM, results_dir: Path):
    """SCP golden data (DB + workspace) from VM for future A/B tests.

    Saves:
    - memory.db: Full graph database snapshot
    - workspace/: Core markdown files (SOUL.md, USER.md, MEMORY.md)
    - workspace/journal/: Journal files
    - workspace/snippets/: Snippet files
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Collecting golden data to {results_dir}")

    # Copy memory.db
    db_result = vm.scp_from("~/clawd/data/memory.db", str(results_dir / "memory.db"))
    if db_result.returncode == 0:
        print(f"    memory.db: OK")
    else:
        print(f"    memory.db: FAILED ({db_result.stderr[:80]})")

    # Copy workspace files (core markdown + journal + snippets)
    ws_dir = results_dir / "workspace"
    ws_dir.mkdir(exist_ok=True)

    for fname in ["SOUL.md", "USER.md", "MEMORY.md", "IDENTITY.md"]:
        r = vm.scp_from(f"~/clawd/{fname}", str(ws_dir / fname))
        print(f"    {fname}: {'OK' if r.returncode == 0 else 'not found'}")

    # Copy journal directory
    journal_dir = ws_dir / "journal"
    journal_dir.mkdir(exist_ok=True)
    # Use tar to grab the whole directory
    vm.ssh(f"cd ~/clawd && tar czf /tmp/journal-golden.tar.gz journal/ 2>/dev/null || true")
    tar_result = vm.scp_from("/tmp/journal-golden.tar.gz", str(ws_dir / "journal-golden.tar.gz"))
    if tar_result.returncode == 0:
        subprocess.run(
            ["tar", "xzf", str(ws_dir / "journal-golden.tar.gz"), "-C", str(ws_dir)],
            capture_output=True, timeout=30,
        )
        (ws_dir / "journal-golden.tar.gz").unlink(missing_ok=True)
        print(f"    journal/: OK")

    # Copy snippets files
    vm.ssh(f"cd ~/clawd && tar czf /tmp/snippets-golden.tar.gz *.snippets.md 2>/dev/null || true")
    tar_result = vm.scp_from("/tmp/snippets-golden.tar.gz", str(ws_dir / "snippets-golden.tar.gz"))
    if tar_result.returncode == 0:
        subprocess.run(
            ["tar", "xzf", str(ws_dir / "snippets-golden.tar.gz"), "-C", str(ws_dir)],
            capture_output=True, timeout=30,
        )
        (ws_dir / "snippets-golden.tar.gz").unlink(missing_ok=True)
        print(f"    snippets: OK")

    # Copy projects directory
    vm.ssh(f"cd ~/clawd && tar czf /tmp/projects-golden.tar.gz projects/ 2>/dev/null || true")
    tar_result = vm.scp_from("/tmp/projects-golden.tar.gz", str(ws_dir / "projects-golden.tar.gz"))
    if tar_result.returncode == 0:
        subprocess.run(
            ["tar", "xzf", str(ws_dir / "projects-golden.tar.gz"), "-C", str(ws_dir)],
            capture_output=True, timeout=30,
        )
        (ws_dir / "projects-golden.tar.gz").unlink(missing_ok=True)
        print(f"    projects/: OK")

    print(f"  Golden data collection complete")


def _patch_gateway_model(vm: TartVM, answer_model: str):
    """Set the gateway's agent model on the VM."""
    full_model = f"anthropic/{answer_model}"
    script = (
        "import json, os\n"
        f"model = '{full_model}'\n"
        "p = os.path.expanduser('~/.openclaw/openclaw.json')\n"
        "d = json.load(open(p))\n"
        "d.setdefault('agents', {}).setdefault('defaults', {})['model'] = {'primary': model}\n"
        "json.dump(d, open(p, 'w'), indent=2)\n"
        "print(f'Gateway model set to: {model}')\n"
    )
    result = vm.ssh("python3 -c " + shlex.quote(script), timeout=10)
    if result.stdout.strip():
        print(f"  {result.stdout.strip()}")


def _patch_memory_json(vm: TartVM, extract_model: str, owner_id: str = "maya"):
    """Patch Quaid's memory.json on the VM for benchmark use.

    Sets the extraction model and default owner for the benchmark persona.
    """
    script = (
        "import json, glob, os\n"
        f"model = '{extract_model}'\n"
        f"owner = '{owner_id}'\n"
        "paths = [\n"
        "    os.path.expanduser('~/clawd/config/memory.json'),\n"
        "    os.path.expanduser('~/.config/openclaw/config/memory.json'),\n"
        "]\n"
        "for p in paths:\n"
        "    if os.path.exists(p):\n"
        "        d = json.load(open(p))\n"
        "        d['models']['highReasoning'] = model\n"
        "        if 'users' not in d: d['users'] = {}\n"
        "        d['users']['defaultOwner'] = owner\n"
        "        if 'identities' not in d['users']: d['users']['identities'] = {}\n"
        "        d['users']['identities'][owner] = {'personNodeName': owner.title()}\n"
        "        if 'projects' not in d: d['projects'] = {}\n"
        "        if 'definitions' not in d['projects']: d['projects']['definitions'] = {}\n"
        "        d['projects']['definitions']['recipe-app'] = {\n"
        "            'label': 'Recipe App',\n"
        "            'homeDir': 'projects/recipe-app/',\n"
        "            'sourceRoots': ['projects/recipe-app/'],\n"
        "            'autoIndex': True,\n"
        "            'patterns': ['*.md'],\n"
        "            'exclude': [],\n"
        "            'description': 'Personal recipe management application'\n"
        "        }\n"
        "        d['projects']['enabled'] = True\n"
        "        json.dump(d, open(p, 'w'), indent=2)\n"
        "        print(f'Patched: {p}')\n"
    )
    result = vm.ssh("python3 -c " + shlex.quote(script), timeout=10)
    if result.stdout.strip():
        print(result.stdout.strip())


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_queries(
    vm: TartVM,
    queries: List[dict],
    system: str,
    results_dir: Path,
    judge_model: str = "haiku",
    mem0_adapter=None,
) -> List[dict]:
    """Evaluate all queries against the VM.

    For VM-based systems (base, qmd, quaid): sends each query as a real
    agent message and captures the response.

    For Mem0: uses the mem0_adapter module on the host.

    Args:
        vm: TartVM instance
        queries: List of eval query dicts
        system: System being evaluated
        results_dir: Output directory
        judge_model: Model for judging (haiku for speed, gpt-4o-mini for publication)

    Returns:
        List of result dicts.
    """
    results = _load_resume_results(results_dir, queries)
    if results:
        print(f"  Resuming from existing eval results: {len(results)}/{len(queries)} complete")

    for i, query in enumerate(queries):
        if i < len(results):
            continue
        question = query["question"]
        ground_truth = query["ground_truth"]
        print(f"  [{i+1}/{len(queries)}] {question[:60]}...")

        t0 = time.monotonic()

        if system == "mem0":
            prediction = _evaluate_mem0(question, results_dir, adapter=mem0_adapter)
        else:
            prediction = _evaluate_vm_agent(vm, question, i, system)

        eval_duration = round(time.monotonic() - t0, 2)

        # Judge
        label, score = _judge(question, ground_truth, prediction, judge_model)

        # Estimate eval token usage (openclaw agent + judge via claude -p don't report usage)
        q_tokens = count_tokens(question)
        p_tokens = count_tokens(prediction) if prediction else 0

        result = {
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "judge_label": label,
            "score": score,
            "query_type": query.get("query_type", "unknown"),
            "recall_difficulty": query.get("recall_difficulty", "unknown"),
            "source_session": query.get("source_session", 0),
            "evidence_sessions": query.get("evidence_sessions", []),
            "eval_duration_s": eval_duration,
            "system": system,
            "tokens_estimate": {
                "question": q_tokens,
                "prediction": p_tokens,
            },
        }
        results.append(result)

        # Save incrementally
        _save_results(results, results_dir)

    return results


def _evaluate_vm_agent(vm: TartVM, question: str, query_idx: int,
                       system: str) -> str:
    """Send a question to the VM agent and get the response.

    Uses a fresh eval session per query to avoid cross-contamination.
    """
    session_id = f"eval-q{query_idx:03d}"
    escaped_question = shlex.quote(question)
    _register_session(vm, session_id)

    result = vm.ssh(
        f"openclaw agent --agent main --session-id {session_id} --message {escaped_question}",
        timeout=120,
    )

    if result.returncode != 0:
        return f"Error: {result.stderr[:200]}"

    # Parse agent response (strip tool use logs, keep final text)
    response = result.stdout.strip()
    return _extract_agent_answer(response)


def _extract_agent_answer(raw_response: str) -> str:
    """Extract the agent's final text answer from full output.

    Strips plugin log lines ([quaid], [memory-core], etc.) and
    system messages, keeping only the agent's natural language response.
    """
    # If JSON output, parse it
    try:
        data = json.loads(raw_response)
        if isinstance(data, dict):
            return data.get("response", data.get("content", raw_response))
    except json.JSONDecodeError:
        pass

    # Strip plugin log lines (e.g., [quaid] Registering..., [memory-core] ...)
    lines = raw_response.split("\n")
    cleaned = [
        line for line in lines
        if not re.match(r"^\[[\w-]+\]\s", line)
    ]
    return "\n".join(cleaned).strip()


def _evaluate_mem0(question: str, results_dir: Path,
                   adapter=None) -> str:
    """Evaluate a question using Mem0 adapter on the host.

    Args:
        adapter: Existing Mem0Adapter instance. Required — Qdrant local
            uses exclusive file locks so only one Memory() instance can
            access the storage at a time.
    """
    try:
        if adapter is None:
            from mem0_adapter import Mem0Adapter
            adapter = Mem0Adapter(results_dir=results_dir)
        answer, _tool_calls, _usage = adapter.answer_question(question)
        return answer
    except ImportError:
        return "Error: mem0_adapter not available"
    except Exception as e:
        return f"Error: {e}"


def _judge(question: str, ground_truth: str, prediction: str,
           judge_model: str) -> Tuple[str, float]:
    """Judge a prediction against ground truth.

    Args:
        judge_model: "gpt-4o-mini" (cross-vendor) or "haiku" (Claude).

    Returns (label, score).
    """
    if not prediction or prediction.strip().lower() in ("", "n/a"):
        return "WRONG", 0.0

    prompt = JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        prediction=prediction,
    )

    if judge_model == "gpt-4o-mini":
        # Use unified OpenAI judge from run_production_benchmark
        from run_production_benchmark import _judge_openai
        return _judge_openai(prompt)

    judge_result = call_claude(
        prompt=prompt,
        model=judge_model,
        timeout=VM_CLAUDE_JUDGE_TIMEOUT_S,
    )
    if isinstance(judge_result, tuple):
        response = judge_result[0]
    else:
        response = judge_result

    if not response:
        return "ERROR", 0.0

    text = response.strip()
    # Try JSON parse first
    try:
        data = json.loads(text)
        lbl = data.get("label", "").upper()
        if lbl == "CORRECT":
            return "CORRECT", 1.0
        elif lbl == "WRONG":
            return "WRONG", 0.0
    except (json.JSONDecodeError, AttributeError):
        pass
    # Fall back to text scanning — last occurrence wins
    upper = text.upper()
    last_correct = upper.rfind("CORRECT")
    last_wrong = upper.rfind("WRONG")
    if last_correct > last_wrong:
        return "CORRECT", 1.0
    elif last_wrong > last_correct:
        return "WRONG", 0.0
    elif "CORRECT" in upper:
        return "CORRECT", 1.0
    elif "WRONG" in upper:
        return "WRONG", 0.0
    return "ERROR", 0.0


def _save_results(results: List[dict], results_dir: Path):
    """Save results incrementally."""
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


def _load_resume_results(results_dir: Path, queries: List[dict]) -> List[dict]:
    """Load and validate existing partial eval results for resume."""
    path = results_dir / "eval_results.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise RuntimeError(f"Existing eval results are not a list: {path}")
    if len(data) > len(queries):
        raise RuntimeError(
            f"Existing eval results exceed query set ({len(data)} > {len(queries)}): {path}"
        )
    for idx, row in enumerate(data):
        if not isinstance(row, dict):
            raise RuntimeError(f"Existing eval result row {idx} is not an object: {path}")
        expected = queries[idx]["question"]
        actual = row.get("question")
        if actual != expected:
            raise RuntimeError(
                "Existing eval results do not match current query set at index "
                f"{idx}: {actual!r} != {expected!r}"
            )
    return data


def rejudge_results(results_dir: Path, judge_model: str) -> dict:
    """Rejudge an existing eval_results.json without rerunning agent answers."""
    path = results_dir / "eval_results.json"
    if not path.exists():
        raise RuntimeError(f"Missing eval results: {path}")

    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise RuntimeError(f"Existing eval results are not a list: {path}")

    updated = 0
    for idx, row in enumerate(data):
        if not isinstance(row, dict):
            raise RuntimeError(f"Existing eval result row {idx} is not an object: {path}")
        question = row.get("question")
        ground_truth = row.get("ground_truth")
        prediction = row.get("prediction", "")
        if not isinstance(question, str) or not isinstance(ground_truth, str):
            raise RuntimeError(f"Existing eval result row {idx} missing question/ground_truth: {path}")
        if (
            row.get("judge_model") == judge_model
            and row.get("judge_label") in ("CORRECT", "PARTIAL", "WRONG")
            and isinstance(row.get("score"), (int, float))
        ):
            continue
        label, score = _judge(question, ground_truth, prediction, judge_model)
        row["judge_label"] = label
        row["score"] = score
        row["judge_model"] = judge_model
        updated += 1
        if updated % 10 == 0:
            print(f"  Rejudged {idx + 1}/{len(data)}")
            with open(results_dir / "eval_results.json", "w") as f:
                json.dump(data, f, indent=2, default=str)

    scores = score_results(data)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "eval_results.json", "w") as f:
        json.dump(data, f, indent=2, default=str)
    with open(results_dir / "scores.json", "w") as f:
        json.dump(scores, f, indent=2)
    return scores


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_results(results: List[dict]) -> dict:
    """Compute accuracy scores from eval results."""
    if not results:
        return {"overall": {"count": 0, "accuracy": 0.0}}

    scored = [r for r in results if r.get("judge_label") in ("CORRECT", "PARTIAL", "WRONG")]
    if not scored:
        return {"overall": {"count": len(results), "accuracy": 0.0}}

    correct = sum(1 for r in scored if r["judge_label"] == "CORRECT")
    partial = sum(1 for r in scored if r["judge_label"] == "PARTIAL")
    wrong = sum(1 for r in scored if r["judge_label"] == "WRONG")

    accuracy = (correct + 0.5 * partial) / len(scored) * 100

    # Per-type breakdown
    by_type = {}
    for r in scored:
        qt = r.get("query_type", "unknown").split("(")[0].strip()
        if qt not in by_type:
            by_type[qt] = []
        by_type[qt].append(r)

    per_type = {}
    for qt, items in sorted(by_type.items()):
        c = sum(1 for r in items if r["judge_label"] == "CORRECT")
        p = sum(1 for r in items if r["judge_label"] == "PARTIAL")
        n = len(items)
        per_type[qt] = {
            "count": n,
            "accuracy": round((c + 0.5 * p) / n * 100, 1) if n > 0 else 0,
            "correct": c,
            "partial": p,
            "wrong": n - c - p,
        }

    return {
        "overall": {
            "count": len(scored),
            "accuracy": round(accuracy, 2),
            "correct": correct,
            "partial": partial,
            "wrong": wrong,
        },
        "per_type": per_type,
    }


# ---------------------------------------------------------------------------
# System setup
# ---------------------------------------------------------------------------

def _results_suffix(mode: str, splitting: str) -> str:
    suffix = f"-{mode}" if mode != "natural" else ""
    if splitting != "perday":
        suffix += f"-{splitting}"
    return suffix


def _resolve_results_dir(results_base: Path, system: str, mode: str, splitting: str) -> Path:
    return results_base / f"{system}{_results_suffix(mode, splitting)}"

def setup_system(vm: TartVM, system: str, snapshot_base: str = "clean-openclaw",
                 extract_model: str = "claude-sonnet-4-5-20250929",
                 local_plugin: bool = False,
                 answer_model: str | None = None):
    """Restore VM and configure for the given system.

    Args:
        vm: TartVM instance
        system: One of "base", "qmd", "quaid", "mem0", "oc-native"
        snapshot_base: Snapshot to restore from
        extract_model: Model for extraction/janitor (Sonnet for dev, Opus for final)
        local_plugin: If True, rsync local Quaid plugin instead of cloning from GitHub
        answer_model: Override gateway agent model (e.g. "claude-sonnet-4-5-20250929")
    """
    if system == "mem0":
        # Mem0 runs on host, no VM setup needed
        print(f"  Mem0 runs on host — no VM setup needed")
        return

    print(f"\n--- Setting up {system} ---")
    vm.restore(snapshot_base)

    if system == "base":
        # Disable all memory plugins
        vm.ssh("openclaw plugins disable quaid 2>/dev/null || true")
        vm.ssh("openclaw plugins disable memory-core 2>/dev/null || true")
        print(f"  Base OpenClaw configured (no memory plugins)")

    elif system == "qmd":
        # QMD uses memory-core (built-in). Disable Quaid, enable memory-core.
        vm.ssh("openclaw plugins disable quaid 2>/dev/null || true")
        vm.ssh("openclaw plugins enable memory-core 2>/dev/null || true")
        print(f"  QMD configured (memory-core enabled, quaid disabled)")

    elif system == "oc-native":
        # Native OpenClaw best-effort memory:
        # - builtin memory-core recall
        # - bundled session-memory hook enabled
        # - direct session transcript indexing enabled
        vm.ssh("openclaw plugins disable quaid 2>/dev/null || true")
        vm.ssh("openclaw plugins disable memory-lancedb 2>/dev/null || true")
        vm.ssh("openclaw plugins enable memory-core 2>/dev/null || true")
        _patch_openclaw_native_memory(vm, enable_session_hook=True)
        print(
            "  OpenClaw native configured "
            "(memory-core builtin + session-memory hook + session indexing)"
        )

    elif system == "quaid":
        if local_plugin:
            # Rsync local plugin to VM (includes v12 changes to soul_snippets.py etc.)
            local_plugin_dir = str(Path(__file__).resolve().parent.parent.parent / "plugins" / "quaid")
            vm.ssh("mkdir -p ~/clawd/plugins/quaid ~/clawd/config ~/clawd/data ~/clawd/journal", raw=True)
            rsync_result = subprocess.run(
                ["rsync", "-az", "--exclude", "node_modules", "--exclude", "__pycache__",
                 "--exclude", ".pytest_cache", "--exclude", "tests",
                 f"{local_plugin_dir}/", f"admin@{vm.ip}:~/clawd/plugins/quaid/"],
                capture_output=True, text=True, timeout=60,
                env={**os.environ, "SSHPASS": "admin"},
            )
            if rsync_result.returncode != 0:
                # Fall back to scp-based rsync
                subprocess.run(
                    ["sshpass", "-p", "admin", "rsync", "-az", "-e", "ssh -o StrictHostKeyChecking=no",
                     "--exclude", "node_modules", "--exclude", "__pycache__",
                     "--exclude", ".pytest_cache", "--exclude", "tests",
                     f"{local_plugin_dir}/", f"admin@{vm.ip}:~/clawd/plugins/quaid/"],
                    capture_output=True, text=True, timeout=60,
                )
            # Copy config if not present
            vm.ssh(
                "test -f ~/clawd/config/memory.json || "
                "cp ~/clawd/plugins/quaid/memory.json.example ~/clawd/config/memory.json 2>/dev/null || true",
                raw=True,
            )
            # Install npm deps
            vm.ssh("cd ~/clawd/plugins/quaid && npm install 2>/dev/null", timeout=120)
            # Hardcode workspace in index.js
            vm.ssh(
                r"cd ~/clawd/plugins/quaid && "
                r"sed -i '' 's|\${QUAID_WORKSPACE}|/Users/admin/clawd|g' index.js"
            )
            print(f"  Quaid synced from local (includes v12 changes)")
        else:
            # Check if Quaid is already installed
            check = vm.ssh("test -f ~/clawd/plugins/quaid/index.js && echo installed || echo missing")
            if "missing" in check.stdout:
                # Install Quaid from GitHub
                result = vm.ssh(
                    "cd /tmp && rm -rf quaid && git clone https://github.com/quaid-labs/quaid.git "
                    "&& mkdir -p ~/clawd/plugins/quaid ~/clawd/config ~/clawd/data ~/clawd/journal "
                    "&& cp -r /tmp/quaid/plugins/quaid/* ~/clawd/plugins/quaid/ "
                    "&& cp /tmp/quaid/memory.json.example ~/clawd/config/memory.json "
                    "&& cd ~/clawd/plugins/quaid && npm install",
                    timeout=300,
                )
                if result.returncode != 0:
                    print(f"  WARNING: Quaid install may have issues: {result.stderr[:200]}")
                else:
                    print(f"  Quaid installed from GitHub")

                # Hardcode workspace in index.js (env var doesn't reach LaunchAgent)
                vm.ssh(
                    r"cd ~/clawd/plugins/quaid && "
                    r"sed -i '' 's|\${QUAID_WORKSPACE}|/Users/admin/clawd|g' index.js"
                )
            else:
                print(f"  Quaid already installed")

        # Patch memory.json to use the configured extraction model
        print(f"  Setting extraction model: {extract_model}")
        _patch_memory_json(vm, extract_model)

        # Ensure Ollama points to host
        vm.ssh(
            "python3 -c \""
            "import json; p='/Users/admin/clawd/config/memory.json'; "
            "d=json.load(open(p)); d['ollama']['url']='http://192.168.64.1:11434'; "
            "json.dump(d,open(p,'w'),indent=2)\" 2>/dev/null || true"
        )
        print(f"  Ollama: using host at 192.168.64.1:11434")

        # SCP extraction script to VM (used by _trigger_compaction for Quaid)
        extract_script = str(_DIR / "extract_compact.py")
        scp_result = vm.scp_to(extract_script, "~/extract_compact.py")
        if scp_result.returncode == 0:
            print(f"  Extraction script deployed")
        else:
            print(f"  WARNING: Failed to deploy extraction script: {scp_result.stderr[:100]}")

    # Clear session state for clean benchmark
    _clear_vm_session_state(vm)
    if system == "oc-native":
        _clear_vm_native_memory_state(vm)

    if system == "quaid":
        # Create core markdown files (simulates onboarding)
        _create_core_files(vm)
        # Create project files for the recipe app
        _create_project_files(vm)
        # Ensure logs directory exists
        vm.ssh("mkdir -p ~/clawd/logs", raw=True)
        # Symlink .env so janitor can find the API key
        # (janitor looks in {workspace}/.env, key lives in ~/.openclaw/.env)
        vm.ssh("ln -sf ~/.openclaw/.env ~/clawd/.env 2>/dev/null || true", raw=True)
        print(f"  API key symlinked: ~/clawd/.env → ~/.openclaw/.env")

    # Set gateway agent model if specified
    if answer_model:
        _patch_gateway_model(vm, answer_model)

    # Restart gateway to pick up changes (fresh DB, clean sessions)
    if system == "oc-native":
        _restart_oc_native_gateway(vm, port=18789)
        _validate_openclaw_native_memory(vm)
    else:
        vm.ssh("pkill -f 'openclaw-gateway' 2>/dev/null; sleep 2; openclaw gateway start", timeout=60)
        time.sleep(5)

        # Verify agent works
        result = vm.ssh(
            "openclaw agent --agent main --session-id test-setup --message 'hello'",
            timeout=60,
        )
        if result.returncode == 0:
            print(f"  Agent verified: responding")
        else:
            print(f"  WARNING: Agent test failed: {result.stderr[:100]}")


# ---------------------------------------------------------------------------
# Full benchmark run
# ---------------------------------------------------------------------------

def run_benchmark(
    system: str,
    mode: str = "natural",
    vm_ip: str = "192.168.64.3",
    results_base: Optional[Path] = None,
    assets_dir: Optional[Path] = None,
    filler_dir: Optional[Path] = None,
    eval_only: bool = False,
    judge_model: str = "haiku",
    snapshot_base: str = "clean-openclaw",
    dry_run: bool = False,
    limit_sessions: Optional[int] = None,
    limit_queries: Optional[int] = None,
    extract_model: str = "claude-sonnet-4-5-20250929",
    local_plugin: bool = False,
    answer_model: str | None = None,
    splitting: str = "perday",
) -> dict:
    """Run full benchmark for a single system.

    Steps:
    1. Setup VM for the system
    2. Inject all sessions
    3. Evaluate all queries
    4. Score and save results

    Returns:
        Combined stats dict.
    """
    if judge_model == "gpt-4o-mini":
        from run_production_benchmark import _get_openai_key

        if not _get_openai_key():
            raise RuntimeError("OPENAI_API_KEY is required for judge_model=gpt-4o-mini")

    results_base = results_base or _DIR.parent / "data" / "results-vm"
    assets_dir = assets_dir or _DIR.parent.parent.parent / "assets"
    results_dir = _resolve_results_dir(results_base, system, mode, splitting)

    print(f"\n{'=' * 60}")
    print(f" AgentLife VM Benchmark: {system.upper()} ({mode}, {splitting})")
    print(f" Results: {results_dir}")
    print(f"{'=' * 60}")

    # Load reviews
    arc_reviews = load_all_reviews(assets_dir)
    filler_reviews = (
        load_filler_reviews(filler_dir)
        if filler_dir is not None and Path(filler_dir).exists()
        else []
    )
    reviews = merge_sessions_chronologically(arc_reviews, filler_reviews)
    queries = get_all_eval_queries(arc_reviews)  # Only arc sessions have eval queries

    # Apply limits (for smoke testing)
    if limit_sessions:
        reviews = reviews[:limit_sessions]
    if limit_queries:
        queries = queries[:limit_queries]

    print(f"Sessions: {len(reviews)} ({len(arc_reviews)} arc + {len(filler_reviews)} filler)")
    if limit_sessions:
        print(f"  (limited to {limit_sessions})")
    print(f"Eval queries: {len(queries)}")
    if limit_queries:
        print(f"  (limited to {limit_queries})")

    # Detect scale for timeout splitting (L = has fillers, S = arc only)
    scale = "L" if filler_reviews else "S"

    if dry_run:
        split_label = f" via {splitting} splitting" if splitting != "perday" else ""
        print(f"\n[DRY RUN] Would inject {len(reviews)} sessions{split_label}, evaluate {len(queries)} queries")
        if splitting == "timeout":
            ts_path = _DIR.parent / "data" / f"timestamps-{scale}.json"
            print(f"  Timestamps: {ts_path} ({'exists' if ts_path.exists() else 'MISSING'})")
        return {"system": system, "mode": mode, "splitting": splitting, "dry_run": True}

    vm = TartVM(ip=vm_ip)

    # Phase 1: Setup
    if not eval_only:
        setup_system(vm, system, snapshot_base, extract_model=extract_model,
                     local_plugin=local_plugin, answer_model=answer_model)

    # Phase 2: Injection
    injection_stats = {}
    _mem0_adapter = None
    if not eval_only:
        split_label = f", {splitting} splitting" if splitting != "perday" else ""
        print(f"\n--- Phase 2: Injection ({mode} compaction{split_label}) ---")
        session_id = f"benchmark-{system}"

        if system == "mem0":
            # Mem0 injection runs on host — keep adapter alive for eval phase
            # (Qdrant local uses exclusive file locks)
            from mem0_adapter import Mem0Adapter
            # Map full model IDs to Mem0Adapter short names
            _mem0_answer = "sonnet"  # default
            if answer_model:
                if "haiku" in answer_model:
                    _mem0_answer = "haiku"
                elif "opus" in answer_model:
                    _mem0_answer = "opus"
                elif "sonnet" in answer_model:
                    _mem0_answer = "sonnet"
            _mem0_adapter = Mem0Adapter(results_dir=results_dir, answer_model=_mem0_answer)
            injection_stats = _mem0_adapter.inject_sessions(reviews, per_message_pair=True)
        else:
            # Register session in gateway's session store so
            # openclaw agent uses our session file for /compact
            if system != "oc-native":
                _register_session(vm, session_id)
            injection_stats = inject_sessions(
                vm, reviews, session_id, mode, results_dir, system,
                extract_model=extract_model,
                splitting=splitting,
                scale=scale,
            )
        print(f"  Injection complete: {injection_stats.get('total_messages', 0)} messages, "
              f"{injection_stats.get('compaction_count', 0)} compactions, "
              f"{injection_stats.get('janitor_runs', 0)} janitor runs")

        # Collect golden data (Quaid only — DB + workspace files)
        if system == "quaid":
            _collect_golden_data(vm, results_dir)
            # Post-injection validation
            print(f"\n  --- Post-injection validation ---")
            validation = _validate_post_injection(vm, system)
            if results_dir:
                results_dir.mkdir(parents=True, exist_ok=True)
                with open(results_dir / "validation.json", "w") as f:
                    json.dump(validation, f, indent=2)

    # Session isolation: freeze injection session file before eval starts.
    # Without this, eval queries via `openclaw agent` get appended to the
    # injection session file, polluting the transcript for any future extraction.
    if not eval_only and system != "oc-native":
        session_id = f"benchmark-{system}"
        session_file = f"{VM_AGENT_SESSIONS_DIR}/{session_id}.jsonl"
        frozen_file = f"{VM_AGENT_SESSIONS_DIR}/{session_id}-injected.jsonl"
        vm.ssh(
            f"cp {session_file} {frozen_file} 2>/dev/null; "
            f"rm -f {session_file} 2>/dev/null; "
            f"rm -f {VM_SESSION_STORE} 2>/dev/null || true",
            timeout=10,
        )
        print(f"\n  Session isolation: frozen {session_id}.jsonl → {session_id}-injected.jsonl")

    # Phase 3: Evaluation
    print(f"\n--- Phase 3: Evaluation ({len(queries)} queries) ---")
    _adapter = _mem0_adapter if system == "mem0" and not eval_only else None
    eval_results = evaluate_queries(vm, queries, system, results_dir, judge_model,
                                    mem0_adapter=_adapter)

    # Phase 4: Scoring
    scores = score_results(eval_results)
    print(f"\n--- Results: {system} ({mode}) ---")
    o = scores["overall"]
    print(f"  Accuracy: {o['accuracy']:.1f}% ({o['correct']}C/{o.get('partial',0)}P/{o.get('wrong',0)}W)")

    # Save everything
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "scores.json", "w") as f:
        json.dump(scores, f, indent=2)
    with open(results_dir / "eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2, default=str)

    # Collect markdown quality metrics (Quaid only)
    md_quality = None
    if system == "quaid":
        try:
            from markdown_quality import collect_vm_metrics
            md_quality = collect_vm_metrics(vm)
            with open(results_dir / "markdown_quality.json", "w") as f:
                json.dump(md_quality, f, indent=2)
        except Exception as e:
            print(f"  WARNING: Markdown quality collection failed: {e}")

    return {
        "system": system,
        "mode": mode,
        "injection": injection_stats,
        "scores": scores,
        "markdown_quality": md_quality,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AgentLife VM Benchmark")
    parser.add_argument("--system", type=str, default="quaid",
                        choices=AVAILABLE_SYSTEMS + ["all"],
                        help="System to benchmark")
    parser.add_argument("--mode", type=str, default="natural",
                        choices=["natural", "nightly"],
                        help="Compaction strategy")
    parser.add_argument("--vm-ip", type=str, default="192.168.64.3",
                        help="Tart VM IP address")
    parser.add_argument("--results-dir", type=str,
                        default=str(_DIR.parent / "data" / "results-vm"),
                        help="Base results directory")
    parser.add_argument("--assets-dir", type=str,
                        default=str(_DIR.parent / "data" / "sessions"),
                        help="Arc session review files")
    parser.add_argument("--filler-dir", type=str,
                        default=str(_DIR.parent / "data" / "filler-sessions"),
                        help="Filler session files")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip injection, evaluate existing state")
    parser.add_argument("--rejudge-only", action="store_true",
                        help="Rejudge existing eval_results.json without rerunning answers")
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini",
                        choices=["gpt-4o-mini", "haiku"],
                        help="Judge model (default: gpt-4o-mini for cross-vendor fairness)")
    parser.add_argument("--snapshot", type=str, default="clean-openclaw",
                        help="Base snapshot name")
    parser.add_argument("--extract-model", type=str,
                        default="claude-sonnet-4-5-20250929",
                        help="Model for extraction/janitor (sonnet for dev, opus for final)")
    parser.add_argument("--limit-sessions", type=int, default=None,
                        help="Limit number of sessions to inject (for smoke testing)")
    parser.add_argument("--limit-queries", type=int, default=None,
                        help="Limit number of eval queries (for smoke testing)")
    parser.add_argument("--no-filler", action="store_true",
                        help="Skip filler sessions (arc only)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without executing")
    parser.add_argument("--local-plugin", action="store_true",
                        help="Rsync local Quaid plugin to VM instead of cloning from GitHub")
    parser.add_argument("--answer-model", type=str, default=None,
                        help="Override gateway agent model (e.g. claude-sonnet-4-5-20250929)")
    parser.add_argument("--splitting", type=str, default="timeout",
                        choices=["perday", "timeout"],
                        help="Extraction splitting strategy (default: timeout)")
    args = parser.parse_args()

    results_base = Path(args.results_dir)
    assets_dir = Path(args.assets_dir)
    filler_dir = Path(args.filler_dir)

    if args.system == "all":
        systems_to_run = SYSTEMS
    else:
        systems_to_run = [args.system]

    if args.rejudge_only and args.system == "all":
        raise RuntimeError("--rejudge-only requires a specific --system")

    if args.rejudge_only:
        results_dir = _resolve_results_dir(results_base, args.system, args.mode, args.splitting)
        print(f"Rejudging existing results in {results_dir}")
        scores = rejudge_results(results_dir, args.judge_model)
        print(json.dumps(scores, indent=2))
        return

    all_results = {}

    for system in systems_to_run:
        result = run_benchmark(
            system=system,
            mode=args.mode,
            vm_ip=args.vm_ip,
            results_base=results_base,
            assets_dir=assets_dir,
            filler_dir=filler_dir if not args.no_filler else None,
            eval_only=args.eval_only,
            judge_model=args.judge_model,
            snapshot_base=args.snapshot,
            dry_run=args.dry_run,
            limit_sessions=args.limit_sessions,
            limit_queries=args.limit_queries,
            extract_model=args.extract_model,
            local_plugin=args.local_plugin,
            answer_model=args.answer_model,
            splitting=args.splitting,
        )
        all_results[system] = result

        # For Quaid, also run nightly A/B if running all
        if system == "quaid" and args.system == "all" and args.mode == "natural":
            nightly_result = run_benchmark(
                system="quaid",
                mode="nightly",
                vm_ip=args.vm_ip,
                results_base=results_base,
                assets_dir=assets_dir,
                filler_dir=filler_dir if not args.no_filler else None,
                judge_model=args.judge_model,
                snapshot_base=args.snapshot,
                dry_run=args.dry_run,
                limit_sessions=args.limit_sessions,
                limit_queries=args.limit_queries,
                extract_model=args.extract_model,
                local_plugin=args.local_plugin,
                answer_model=args.answer_model,
                splitting=args.splitting,
            )
            all_results["quaid-nightly"] = nightly_result

    # Save combined results
    results_base.mkdir(parents=True, exist_ok=True)
    with open(results_base / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f" BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    print(f"\n{'System':<20} {'Accuracy':>10} {'Compactions':>12} {'Ctx Burned':>12} {'Savings':>10}")
    print(f"{'─' * 67}")
    for name, r in all_results.items():
        if r.get("dry_run"):
            continue
        scores = r.get("scores", {}).get("overall", {})
        acc = scores.get("accuracy", 0)
        compact = r.get("injection", {}).get("compaction_count", "?")
        sim = r.get("injection", {}).get("simulated_tokens", {})
        burned = sim.get("total_context_burned", 0)
        savings = sim.get("savings_pct", 0)
        burned_str = f"{burned:,}" if burned else "?"
        savings_str = f"{savings:.1f}%" if burned else "?"
        print(f"{name:<20} {acc:>9.1f}% {compact:>12} {burned_str:>12} {savings_str:>10}")


if __name__ == "__main__":
    main()
