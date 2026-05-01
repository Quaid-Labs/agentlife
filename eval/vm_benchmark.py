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
import base64
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DIR.parent
_WORKSPACE = Path(os.environ.get("CLAWDBOT_WORKSPACE", Path.home() / "clawd"))
_RUNNER_DIR = _WORKSPACE / "memory-stress-test" / "runner"
OC_NATIVE_EMBED_PROXY_SCRIPT = _REPO_ROOT / "scripts" / "ollama-openai-embed-proxy.py"
OC_NATIVE_EMBED_PROXY_REMOTE_SCRIPT = "/tmp/oc-native-embed-proxy.py"
OC_NATIVE_EMBED_PROXY_PIDFILE = Path("/tmp/oc-native-embed-proxy.pid")
OC_NATIVE_EMBED_PROXY_LOG = Path("/tmp/oc-native-embed-proxy.log")
OC_NATIVE_EMBED_TUNNEL_PIDFILE = Path("/tmp/oc-native-embed-tunnel.pid")
OC_NATIVE_EMBED_TUNNEL_LOG = Path("/tmp/oc-native-embed-tunnel.log")
OC_NATIVE_EMBED_UPSTREAM = os.environ.get(
    "OPENCLAW_NATIVE_OLLAMA_UPSTREAM_URL",
    "http://127.0.0.1:11434",
)
OC_NATIVE_EMBED_BASE_URL = os.environ.get(
    "OPENCLAW_NATIVE_OLLAMA_BASE_URL",
    "http://192.168.64.1:11435/v1",
)
OC_NATIVE_EMBED_MODEL = "nomic-embed-text"
OC_NATIVE_EMBED_DIMS = 768
_OC_NATIVE_GATEWAY_ANSWER_MODEL: Optional[str] = None
_OC_NATIVE_GATEWAY_OPENAI_AUTH_MODE = "api"
OC_NATIVE_GATEWAY_CALL_RESTART_LIMIT = 3
OC_NATIVE_LOCAL_VM_NAMESPACE_PREFIXES = ("benchmark-", "oc-bench-", "oc-native-bench-")

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
OC_NATIVE_GATEWAY_START_WAIT_S = 120.0
OC_NATIVE_GATEWAY_RUN_WAIT_S = 180.0
OC_NATIVE_GATEWAY_LOG_TAIL_LINES = 80
OC_NATIVE_EMBED_UPSTREAM_WARMUP_TIMEOUT_S = 180
OC_NATIVE_EMBED_UPSTREAM_READY_WAIT_S = 360
OC_NATIVE_EMBED_UPSTREAM_READY_POLL_S = 3
# Keep the host-side warmup client longer than the proxy's own upstream relay
# budget. If the client disconnects first, the proxy keeps waiting on Ollama,
# then trips BrokenPipe and leaves overlapping upstream work in flight.
OC_NATIVE_EMBED_PROXY_WARMUP_TIMEOUT_S = 120
OC_NATIVE_EMBED_PROXY_READY_WAIT_S = 240
OC_NATIVE_EMBED_PROXY_READY_POLL_S = 3
OC_NATIVE_EMBED_VALIDATION_TIMEOUT_S = 180
OC_NATIVE_EMBED_VALIDATION_PROBE_TIMEOUT_S = 30
OC_NATIVE_REINDEX_STATUS_WAIT_S = 90.0
OC_NATIVE_REINDEX_STATUS_POLL_S = 3.0
OC_NATIVE_SESSION_QUIET_TIMEOUT_S = 15.0
OC_NATIVE_SESSION_QUIET_WINDOW_S = 1.0
OC_NATIVE_SESSION_RESTORE_ATTEMPTS = 3
VM_CLAUDE_JUDGE_TIMEOUT_S = 90
VM_AGENT_EVAL_TIMEOUT_S = 240
VM_AGENT_EVAL_MAX_TIMEOUT_RETRIES = 2
OC_NATIVE_MEMORY_TOOLS = [
    "read",
    "memory_search",
    "memory_get",
    "wiki_status",
    "wiki_search",
    "wiki_get",
]

# VM paths — sessions live under agents/{agent-id}/sessions/, NOT ~/.openclaw/sessions/
VM_AGENT_SESSIONS_DIR = "~/.openclaw/agents/main/sessions"
VM_SESSION_STORE = "~/.openclaw/agents/main/sessions/sessions.json"
VM_OC_EVAL_AGENT_ID = "benchmark-eval"
VM_OC_EVAL_SESSIONS_DIR = f"~/.openclaw/agents/{VM_OC_EVAL_AGENT_ID}/sessions"
VM_QUAID_DIR = "~/clawd/plugins/quaid"
VM_QUAID_HOME = "/Users/admin/clawd"
VM_QUAID_INSTANCE = "openclaw-main"
VM_QUAID_INSTANCE_ROOT_DIR = f"{VM_QUAID_HOME}/instances/{VM_QUAID_INSTANCE}"
# Benchmark-triggered direct compaction needs an adapter-owned transcript path.
# Keep benchmark files under the OC sessions tree, but isolated in a subdir so
# cleanup remains scoped. The harness stops the background daemon during
# injection to avoid the old double-processing race on these files.
VM_QUAID_BENCH_SESSIONS_DIR = f"{VM_AGENT_SESSIONS_DIR}/benchmark"
VM_QUAID_OLLAMA_URL = "http://192.168.64.1:11435"
VM_QUAID_INSTANCE_DB_PATH = f"{VM_QUAID_HOME}/instances/{VM_QUAID_INSTANCE}/data/memory.db"
VM_QUAID_INSTANCE_ARCHIVE_DB_PATH = f"{VM_QUAID_HOME}/instances/{VM_QUAID_INSTANCE}/data/memory_archive.db"
VM_QUAID_INSTANCE_LOGS_DIR = f"{VM_QUAID_HOME}/instances/{VM_QUAID_INSTANCE}/logs"
VM_QUAID_INSTANCE_JOURNAL_DIR = f"{VM_QUAID_INSTANCE_ROOT_DIR}/journal"
VM_QUAID_LLM_USAGE_LOG_PATH = f"{VM_QUAID_INSTANCE_LOGS_DIR}/llm-usage-events.jsonl"
VM_QUAID_DAEMON_ROLLING_METRICS_PATH = f"{VM_QUAID_INSTANCE_LOGS_DIR}/daemon/rolling-extraction.jsonl"
VM_QUAID_JANITOR_LATEST_LOG = f"{VM_QUAID_INSTANCE_LOGS_DIR}/janitor-latest.log"
VM_QUAID_JANITOR_STATS_PATH = f"{VM_QUAID_INSTANCE_LOGS_DIR}/janitor-stats.json"
VM_LIFECYCLE_RESUME_ROOT = "vm_lifecycle_resume"
VM_LIFECYCLE_RESUME_STATE = "resume_state.json"
VM_LIFECYCLE_RESUME_ARCHIVE = "guest-state.tar.gz"
VM_BENCHMARK_PROJECTS = {
    "recipe-app": {
        "label": "Recipe App",
        "description": "Recipe app project workspace",
        "home_dir": f"{VM_QUAID_HOME}/projects/recipe-app",
        "source_root": f"{VM_QUAID_HOME}/projects/recipe-app",
        "patterns": ["*.md", "*.js", "*.json", "*.html", "*.css"],
        "exclude": ["node_modules/", "*.db", ".git/", "package-lock.json"],
        "project_md": """# Recipe App Project

## Overview
Recipe app project workspace. Current details should be learned from source artifacts and conversations.

## Tech Stack
To be updated as the project evolves.

## Features
To be updated as features are built.

## Status
In development.
""",
    },
    "portfolio-site": {
        "label": "Portfolio Site",
        "description": "Portfolio site project workspace",
        "home_dir": f"{VM_QUAID_HOME}/projects/portfolio-site",
        "source_root": f"{VM_QUAID_HOME}/projects/portfolio-site",
        "patterns": ["*.md", "*.html", "*.css"],
        "exclude": [".git/"],
        "project_md": """# Portfolio Site Project

## Overview
Portfolio site project workspace. Current details should be learned from source artifacts and conversations.

## Tech Stack
To be updated as the project evolves.

## Features
To be updated as features are built.

## Status
In development.
""",
    },
}

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

    def __init__(
        self,
        ip: str = "192.168.64.3",
        user: str = "admin",
        password: str = "admin",
        vm_name: str = "test-openclaw",
        tart_host: Optional[str] = None,
    ):
        self.ip = ip
        self.user = user
        self.password = password
        self.vm_name = vm_name
        self.tart_host = str(tart_host or "").strip() or None

    def _list_local_tart_vms(self) -> list[tuple[str, str]]:
        """Return local Tart VMs as (name, state)."""
        result = self._tart_cmd("list", timeout=10)
        if result.returncode != 0:
            return []
        rows: list[tuple[str, str]] = []
        for raw in result.stdout.splitlines():
            line = raw.strip()
            if not line or line.lower().startswith("source "):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            rows.append((parts[1], parts[-1]))
        return rows

    def _current_tart_vm_state(self) -> Optional[str]:
        """Return the current Tart state for the configured VM when known."""
        for name, state in self._list_local_tart_vms():
            if name == self.vm_name:
                return state
        return None

    def _resolve_local_vm_name(self) -> None:
        """Resolve the configured local Tart VM name safely.

        Benchmarks must not silently hijack an unrelated running VM, especially
        a coordinator/live-test VM. Allow the old single-running-VM fallback
        only under an explicit override.
        """
        if self.tart_host:
            return
        vms = self._list_local_tart_vms()
        if any(name == self.vm_name for name, _state in vms):
            return
        running = [name for name, state in vms if state == "running"]
        allow_fallback = str(os.environ.get("VM_BENCHMARK_ALLOW_RUNNING_VM_FALLBACK", "")).strip() == "1"
        if len(running) == 1 and allow_fallback:
            print(
                f"  VM name fallback: configured {self.vm_name!r} not found; "
                f"using running VM {running[0]!r}"
            )
            self.vm_name = running[0]
            return
        available = ", ".join(f"{name}({state})" for name, state in vms) or "none"
        if len(running) == 1:
            raise RuntimeError(
                f"Configured Tart VM {self.vm_name!r} not found. "
                f"Refusing to reuse running VM {running[0]!r}; "
                "create or pass a dedicated benchmark VM explicitly, or set "
                "VM_BENCHMARK_ALLOW_RUNNING_VM_FALLBACK=1 to override. "
                f"Available VMs: {available}"
            )
        raise RuntimeError(
            f"Configured Tart VM {self.vm_name!r} not found. "
            f"Available VMs: {available}"
        )

    def _refresh_ip_from_tart(self) -> None:
        """Update VM IP from Tart when available."""
        if self.tart_host:
            return
        result = self._tart_cmd("ip", self.vm_name, timeout=10)
        if result.returncode != 0:
            return
        resolved = str(result.stdout or "").strip()
        if not resolved:
            return
        if resolved != self.ip:
            print(f"  VM IP refresh: {self.ip} -> {resolved}")
            self.ip = resolved

    def _readiness_probe_timeout(self) -> int:
        """SSH probe timeout for readiness checks."""
        # Guest SSH probes routed through a tart host add one extra SSH hop.
        return 15 if self.tart_host else 5

    def _tart_cmd(self, *parts: str, timeout: int = 120) -> subprocess.CompletedProcess:
        cmd = ["tart", *parts]
        if self.tart_host:
            remote_cmd = " ".join(shlex.quote(part) for part in cmd)
            return subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", self.tart_host, remote_cmd],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def _tart_popen(self, *parts: str):
        cmd = ["tart", *parts]
        if self.tart_host:
            remote_cmd = " ".join(shlex.quote(part) for part in cmd)
            detached = f"nohup {remote_cmd} >/tmp/{self.vm_name}-tart.log 2>&1 </dev/null &"
            return subprocess.Popen(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", self.tart_host, detached],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    @staticmethod
    def _is_auth_failure(stderr: str) -> bool:
        detail = str(stderr or "")
        return "Permission denied" in detail or "publickey,password,keyboard-interactive" in detail

    def _guest_ssh_shell_cmd(self, remote_cmd: str, prefer_password: bool = False) -> str:
        parts: list[str] = []
        if prefer_password and self.password:
            parts.extend([
                "sshpass", "-p", shlex.quote(self.password),
                "ssh", "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=10",
                "-o", "PreferredAuthentications=password",
                "-o", "PubkeyAuthentication=no",
                "-o", "IdentitiesOnly=yes",
            ])
        else:
            parts.extend([
                "ssh", "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=10",
                "-o", "PreferredAuthentications=publickey",
                "-o", "IdentitiesOnly=yes",
            ])
        parts.extend([
            shlex.quote(f"{self.user}@{self.ip}"),
            shlex.quote(remote_cmd),
        ])
        return " ".join(parts)

    def _guest_scp_shell_cmd(self, source: str, dest: str, prefer_password: bool = False) -> str:
        parts: list[str] = []
        if prefer_password and self.password:
            parts.extend([
                "sshpass", "-p", shlex.quote(self.password),
                "scp", "-o", "StrictHostKeyChecking=no",
                "-o", "PreferredAuthentications=password",
                "-o", "PubkeyAuthentication=no",
                "-o", "IdentitiesOnly=yes",
            ])
        else:
            parts.extend([
                "scp", "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=10",
                "-o", "PreferredAuthentications=publickey",
                "-o", "IdentitiesOnly=yes",
            ])
        parts.extend([shlex.quote(source), shlex.quote(dest)])
        return " ".join(parts)

    def _run_guest_host_command(
        self,
        key_cmd: str,
        password_cmd: Optional[str] = None,
        *,
        input_data: Optional[str] = None,
        timeout: int = 120,
    ) -> subprocess.CompletedProcess:
        args = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", self.tart_host, key_cmd]
        result = subprocess.run(
            args,
            input=input_data,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0 or not password_cmd or not self._is_auth_failure(result.stderr):
            return result
        return subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", self.tart_host, password_cmd],
            input=input_data,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

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
        attempts = 12 if self.tart_host else 3
        last_result = None
        for attempt in range(attempts):
            self._refresh_ip_from_tart()
            attempt_ip = self.ip
            if self.tart_host:
                key_guest_cmd = self._guest_ssh_shell_cmd(full_cmd, prefer_password=False)
                password_guest_cmd = self._guest_ssh_shell_cmd(full_cmd, prefer_password=True)
            else:
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
            try:
                if self.tart_host:
                    result = self._run_guest_host_command(
                        key_guest_cmd,
                        password_guest_cmd,
                        input_data=input_data,
                        timeout=timeout,
                    )
                else:
                    result = subprocess.run(
                        args,
                        input=input_data,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
            except subprocess.TimeoutExpired:
                self._refresh_ip_from_tart()
                if self.ip != attempt_ip and attempt < attempts - 1:
                    time.sleep(min(12.0, 2.0 + (attempt * 1.25)))
                    continue
                raise
            last_result = result
            stderr = result.stderr or ""
            if result.returncode == 0:
                return result
            if not any(pattern in stderr for pattern in self.SSH_RETRY_PATTERNS):
                return result
            if attempt < attempts - 1:
                # Alfie hop mode can intermittently reject auth under load;
                # use longer backoff to ride out transient SSH auth windows.
                time.sleep(min(12.0, 2.0 + (attempt * 1.25)))
        return last_result

    def scp_to(self, local: str, remote: str, timeout: int = 60):
        """Copy file from host to VM."""
        self._refresh_ip_from_tart()
        if self.tart_host:
            key_remote_cmd = " ".join([
                "cat", shlex.quote("/tmp/vm-benchmark-upload"), "|",
                self._guest_ssh_shell_cmd(f"cat > {remote}", prefer_password=False),
            ])
            password_remote_cmd = " ".join([
                "cat", shlex.quote("/tmp/vm-benchmark-upload"), "|",
                self._guest_ssh_shell_cmd(f"cat > {remote}", prefer_password=True),
            ])
            prep = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", self.tart_host, "rm -f /tmp/vm-benchmark-upload"],
                capture_output=True, text=True, timeout=timeout,
            )
            if prep.returncode != 0:
                return prep
            copy_to_host = subprocess.run(
                ["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", local, f"{self.tart_host}:/tmp/vm-benchmark-upload"],
                capture_output=True, text=True, timeout=timeout,
            )
            if copy_to_host.returncode != 0:
                return copy_to_host
            return self._run_guest_host_command(
                key_remote_cmd,
                password_remote_cmd,
                timeout=timeout,
            )
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
        self._refresh_ip_from_tart()
        if self.tart_host:
            pull = self._run_guest_host_command(
                self._guest_scp_shell_cmd(
                    f"{self.user}@{self.ip}:{remote}",
                    "/tmp/vm-benchmark-download",
                    prefer_password=False,
                ),
                self._guest_scp_shell_cmd(
                    f"{self.user}@{self.ip}:{remote}",
                    "/tmp/vm-benchmark-download",
                    prefer_password=True,
                ),
                timeout=timeout,
            )
            if pull.returncode != 0:
                return pull
            return subprocess.run(
                ["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", f"{self.tart_host}:/tmp/vm-benchmark-download", local],
                capture_output=True, text=True, timeout=timeout,
            )
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
        result = self._tart_cmd("snapshot", "create", self.vm_name, name, timeout=120)
        if result.returncode != 0:
            print(f"  WARNING: Snapshot failed: {result.stderr}")
        return result

    def restore(self, name: str):
        """Restore VM to snapshot (or just verify VM is running if snapshots unavailable)."""
        self._resolve_local_vm_name()
        self._refresh_ip_from_tart()
        print(f"  Restoring snapshot: {name}")

        # Check if tart supports snapshots
        check = self._tart_cmd("help", timeout=10)
        has_snapshots = "snapshot" in check.stdout

        if has_snapshots:
            # Must stop VM first
            self._tart_cmd("stop", self.vm_name, timeout=30)
            time.sleep(2)

            result = self._tart_cmd("snapshot", "restore", self.vm_name, name, timeout=120)
            if result.returncode != 0:
                print(f"  WARNING: Restore failed: {result.stderr[:100]}")

            # Restart VM
            self._tart_popen("run", self.vm_name, "--no-graphics")
            self.wait_ready()
            return result
        else:
            # No snapshot support — just verify VM is running
            print(f"  (snapshots not supported, verifying VM is running)")
            vm_state = self._current_tart_vm_state()
            if vm_state and vm_state != "running":
                self._tart_popen("run", self.vm_name, "--no-graphics")
                self.wait_ready()
            elif self.is_ready():
                print(f"  VM ready")
            else:
                # Try to start VM
                self._tart_popen("run", self.vm_name, "--no-graphics")
                self.wait_ready()
            return subprocess.CompletedProcess([], 0)

    def wait_ready(self, timeout: int = 120):
        """Wait for SSH to be responsive."""
        self._resolve_local_vm_name()
        self._refresh_ip_from_tart()
        print(f"  Waiting for VM at {self.ip}...")
        probe_timeout = self._readiness_probe_timeout()
        deadline = time.monotonic() + timeout
        attempts = 0
        last_error = ""
        while time.monotonic() < deadline:
            attempts += 1
            if not self.tart_host and attempts % 3 == 1:
                self._refresh_ip_from_tart()
            try:
                result = self.ssh("echo ready", timeout=probe_timeout, raw=True)
                if result.returncode == 0 and "ready" in result.stdout:
                    print(f"  VM ready")
                    return True
                detail = (result.stderr or result.stdout or "").strip()
                if detail:
                    last_error = f"rc={result.returncode}: {detail[:200]}"
                else:
                    last_error = f"rc={result.returncode}"
            except subprocess.TimeoutExpired:
                last_error = f"ssh probe timed out after {probe_timeout}s"
            except Exception as exc:
                last_error = str(exc)[:200]
            if attempts % 5 == 0 and last_error:
                print(f"  Still waiting ({attempts} probes): {last_error}")
            time.sleep(3)
        suffix = f" (last error: {last_error})" if last_error else ""
        raise TimeoutError(f"VM not ready after {timeout}s{suffix}")

    def is_ready(self) -> bool:
        """Check if VM is reachable."""
        try:
            result = self.ssh("echo ok", timeout=self._readiness_probe_timeout(), raw=True)
            return result.returncode == 0
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Session JSONL conversion
# ---------------------------------------------------------------------------

def messages_to_gateway_jsonl(messages: List[dict]) -> str:
    """Convert messages to the lightweight gateway JSONL format."""
    lines = []
    for msg in messages:
        line = json.dumps(
            {
                "type": "message",
                "message": {"role": msg["role"], "content": msg["content"]},
            }
        )
        lines.append(line)
    return "\n".join(lines) + "\n"


def _oc_native_event_timestamp(timestamp_ms: int) -> str:
    """Return an ISO-8601 timestamp in the shape OpenClaw session files emit."""
    return (
        datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _oc_native_message_content(content) -> List[dict]:
    """Normalize benchmark transcript content into OpenClaw's typed content array."""
    if isinstance(content, list):
        normalized = []
        for item in content:
            if isinstance(item, dict) and item.get("type"):
                normalized.append(item)
            elif isinstance(item, str):
                normalized.append({"type": "text", "text": item})
        if normalized:
            return normalized
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [{"type": "text", "text": json.dumps(content, ensure_ascii=False)}]


def _message_field(message, key: str, default=None):
    if isinstance(message, dict):
        return message.get(key, default)
    return getattr(message, key, default)


def messages_to_oc_native_jsonl(
    messages: List[dict],
    session_id: str,
    *,
    started_at_ms: Optional[int] = None,
    model_id: str = "gpt-5.4",
    provider: str = "openai",
) -> str:
    """Convert messages to OpenClaw's native session JSONL format.

    OC-native hooks now expect a real OpenClaw session file, not the older
    lightweight gateway dump. Seed synthetic benchmark transcripts with a valid
    session header and structured message records so the runtime can append new
    turns without zeroing or rejecting the file.
    """
    if started_at_ms is None and messages:
        first_ts = _message_field(messages[0], "timestamp_ms")
        if isinstance(first_ts, (int, float)) and first_ts > 0:
            started_at_ms = int(first_ts)
    started_at_ms = started_at_ms or int(time.time() * 1000)
    timestamp_ms = started_at_ms
    cwd = str(Path.home() / ".openclaw" / "workspace")
    parent_id = None
    lines = [
        json.dumps(
            {
                "type": "session",
                "version": 3,
                "id": session_id,
                "timestamp": _oc_native_event_timestamp(timestamp_ms),
                "cwd": cwd,
            },
            ensure_ascii=False,
        )
    ]
    model_event_id = uuid.uuid4().hex[:8]
    lines.append(
        json.dumps(
            {
                "type": "model_change",
                "id": model_event_id,
                "parentId": parent_id,
                "timestamp": _oc_native_event_timestamp(timestamp_ms),
                "provider": provider,
                "modelId": model_id,
            },
            ensure_ascii=False,
        )
    )
    parent_id = model_event_id
    timestamp_ms += 1
    for msg in messages:
        msg_ts = _message_field(msg, "timestamp_ms")
        if isinstance(msg_ts, (int, float)) and msg_ts > 0:
            timestamp_ms = int(msg_ts)
        event_id = uuid.uuid4().hex[:8]
        lines.append(
            json.dumps(
                {
                    "type": "message",
                    "id": event_id,
                    "parentId": parent_id,
                    "timestamp": _oc_native_event_timestamp(timestamp_ms),
                    "message": {
                        "role": _message_field(msg, "role"),
                        "content": _oc_native_message_content(_message_field(msg, "content")),
                        "timestamp": timestamp_ms,
                    },
                },
                ensure_ascii=False,
            )
        )
        parent_id = event_id
        timestamp_ms += 1
    return "\n".join(lines) + "\n"


def _quaid_chunk_session_id(base_session_id: str, chunk, chunk_index: int) -> str:
    source_ids = [str(s).strip().lower() for s in getattr(chunk, "session_ids", []) if str(s).strip()]
    if source_ids:
        return f"{base_session_id}-{'-'.join(source_ids)}"
    return f"{base_session_id}-chunk{chunk_index + 1:02d}"


def _safe_artifact_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "").strip())
    return cleaned.strip("-._") or "artifact"


def _write_local_extraction_artifact(results_dir: Optional[Path], session_id: str, artifact: dict) -> None:
    if results_dir is None or not isinstance(artifact, dict):
        return
    out_dir = Path(results_dir) / "extract-artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_safe_artifact_name(session_id)}.json"
    out_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _register_session(
    vm: TartVM,
    session_id: str,
    *,
    session_key: str = "agent:main:main",
    session_file: Optional[str] = None,
):
    """Register a session in the gateway's session store.

    The gateway resolves sessions via a session store (sessions.json).
    Without this registration, `openclaw agent --session-id X` will
    ignore our session ID and use the default session for the agent.

    The canonical main session key ("agent:main:main") may already be bound to
    an internal UUID session in current OpenClaw builds. OC-native benchmark
    paths should therefore use a unique agent session key per synthetic
    benchmark session so the seeded transcript file actually remains attached to
    the gateway turn we trigger.
    """
    if not session_file:
        session_file = f"{VM_AGENT_SESSIONS_DIR}/{session_id}.jsonl"
    script = (
        "import json, os, time\n"
        f"session_id = '{session_id}'\n"
        f"session_key = {session_key!r}\n"
        f"store_path = os.path.expanduser('{VM_SESSION_STORE}')\n"
        f"session_file = os.path.expanduser({session_file!r})\n"
        "os.makedirs(os.path.dirname(store_path), exist_ok=True)\n"
        "os.makedirs(os.path.dirname(session_file), exist_ok=True)\n"
        "store = {}\n"
        "if os.path.exists(store_path):\n"
        "    store = json.load(open(store_path))\n"
        "store[session_key] = {\n"
        "    'sessionId': session_id,\n"
        "    'sessionFile': session_file,\n"
        "    'updatedAt': int(time.time() * 1000),\n"
        "}\n"
        "json.dump(store, open(store_path, 'w'), indent=2)\n"
        "print(f'Registered session {session_id} in store key={session_key}')\n"
    )
    result = vm.ssh("python3 -c " + shlex.quote(script), timeout=10)
    if result.stdout.strip():
        print(f"  {result.stdout.strip()}")


def _cleanup_oc_native_eval_session(
    vm: TartVM,
    session_id: str,
    *,
    session_key: Optional[str] = None,
):
    """Remove an OC-native eval session so later eval queries cannot retrieve it."""
    if not session_key:
        session_key = _oc_native_session_key(session_id)
    script = (
        "import json, os\n"
        f"session_id = {session_id!r}\n"
        f"session_key = {session_key!r}\n"
        "sessions_dir = os.path.realpath(os.path.expanduser('~/.openclaw/agents/main/sessions'))\n"
        "eval_sessions_dir = os.path.realpath(os.path.expanduser('~/.openclaw/agents/benchmark-eval/sessions'))\n"
        "store_path = os.path.join(sessions_dir, 'sessions.json')\n"
        "paths = set()\n"
        "store_updated = False\n"
        "fallbacks = {\n"
        "    os.path.realpath(os.path.join(sessions_dir, session_id + '.jsonl')),\n"
        "    os.path.realpath(os.path.join(eval_sessions_dir, session_id + '.jsonl')),\n"
        "}\n"
        "for fallback in sorted(fallbacks):\n"
        "    if os.path.commonpath([fallback, sessions_dir]) == sessions_dir or os.path.commonpath([fallback, eval_sessions_dir]) == eval_sessions_dir:\n"
        "        paths.add(fallback)\n"
        "store = {}\n"
        "if os.path.exists(store_path):\n"
        "    try:\n"
        "        store = json.load(open(store_path))\n"
        "    except Exception:\n"
        "        store = {}\n"
        "    entry = store.pop(session_key, None)\n"
        "    if entry is not None:\n"
        "        store_updated = True\n"
        "        if isinstance(entry, dict):\n"
        "            session_file = entry.get('sessionFile')\n"
        "            if isinstance(session_file, str) and session_file.strip():\n"
        "                candidate = os.path.realpath(os.path.expanduser(session_file))\n"
        "                if os.path.commonpath([candidate, sessions_dir]) == sessions_dir or os.path.commonpath([candidate, eval_sessions_dir]) == eval_sessions_dir:\n"
        "                    paths.add(candidate)\n"
        "    if store_updated:\n"
        "        if store:\n"
        "            json.dump(store, open(store_path, 'w'), indent=2)\n"
        "        else:\n"
        "            try:\n"
        "                os.remove(store_path)\n"
        "            except FileNotFoundError:\n"
        "                pass\n"
        "removed = []\n"
        "for path in sorted(paths):\n"
        "    try:\n"
        "        os.remove(path)\n"
        "    except FileNotFoundError:\n"
        "        continue\n"
        "    except IsADirectoryError:\n"
        "        continue\n"
        "    else:\n"
        "        removed.append(path)\n"
        "print(json.dumps({'sessionId': session_id, 'sessionKey': session_key, 'removed': removed, 'storeUpdated': store_updated}))\n"
    )
    result = vm.ssh("python3 -c " + shlex.quote(script), timeout=20, raw=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"oc-native eval session cleanup failed for {session_id}: {(result.stderr or result.stdout)[:200]}"
        )


def _clear_vm_session_state(vm: TartVM):
    """Clear all session files and reset session store on the VM.

    Used before a benchmark run to ensure clean state.
    """
    script = (
        "import os, shutil, subprocess, time\n"
        "from pathlib import Path\n"
        "subprocess.run(['pkill', '-f', 'openclaw-gateway'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "subprocess.run(['pkill', '-f', 'project_docs_supervisor.py'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "subprocess.run(['pkill', '-f', 'core/lifecycle/janitor.py'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "subprocess.run(['pkill', '-f', '/Users/admin/extract_compact.py'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "time.sleep(1)\n"
        f"sessions_dir = Path(os.path.expanduser({VM_AGENT_SESSIONS_DIR!r})).resolve()\n"
        "sessions_dir.mkdir(parents=True, exist_ok=True)\n"
        f"eval_sessions_dir = Path(os.path.expanduser({VM_OC_EVAL_SESSIONS_DIR!r})).resolve()\n"
        f"store_path = Path(os.path.expanduser({VM_SESSION_STORE!r})).resolve()\n"
        "removed = []\n"
        "for pattern in ('*.jsonl', '*.jsonl.reset.*'):\n"
        "    for path in sessions_dir.glob(pattern):\n"
        "        try:\n"
        "            path.unlink()\n"
        "            removed.append(str(path))\n"
        "        except FileNotFoundError:\n"
        "            pass\n"
        "shutil.rmtree(eval_sessions_dir, ignore_errors=True)\n"
        "try:\n"
        "    store_path.unlink()\n"
        "except FileNotFoundError:\n"
        "    pass\n"
        "direct_files = [\n"
        "    Path('~/clawd/data/memory.db').expanduser(),\n"
        f"    Path({VM_QUAID_INSTANCE_DB_PATH!r}).expanduser(),\n"
        f"    Path({VM_QUAID_INSTANCE_ARCHIVE_DB_PATH!r}).expanduser(),\n"
        f"    Path({VM_QUAID_DAEMON_ROLLING_METRICS_PATH!r}).expanduser(),\n"
        f"    Path({VM_QUAID_LLM_USAGE_LOG_PATH!r}).expanduser(),\n"
        "    Path('~/clawd/USER.md').expanduser(),\n"
        "    Path('~/clawd/SOUL.md').expanduser(),\n"
        "    Path('~/clawd/MEMORY.md').expanduser(),\n"
        "    Path('~/clawd/ENVIRONMENT.md').expanduser(),\n"
        f"    Path({(VM_QUAID_INSTANCE_ROOT_DIR + '/USER.md')!r}).expanduser(),\n"
        f"    Path({(VM_QUAID_INSTANCE_ROOT_DIR + '/SOUL.md')!r}).expanduser(),\n"
        f"    Path({(VM_QUAID_INSTANCE_ROOT_DIR + '/MEMORY.md')!r}).expanduser(),\n"
        f"    Path({(VM_QUAID_INSTANCE_ROOT_DIR + '/ENVIRONMENT.md')!r}).expanduser(),\n"
        "]\n"
        "for path in direct_files:\n"
        "    try:\n"
        "        path.unlink()\n"
        "    except FileNotFoundError:\n"
        "        pass\n"
        "state_dirs = [\n"
        f"    Path({(VM_QUAID_HOME + '/instances/' + VM_QUAID_INSTANCE + '/data/session-cursors')!r}).expanduser(),\n"
        f"    Path({(VM_QUAID_HOME + '/instances/' + VM_QUAID_INSTANCE + '/data/extraction-signals')!r}).expanduser(),\n"
        f"    Path({(VM_QUAID_HOME + '/instances/' + VM_QUAID_INSTANCE + '/data/rolling-extraction')!r}).expanduser(),\n"
        f"    Path({(VM_QUAID_INSTANCE_ROOT_DIR + '/.runtime/locks')!r}).expanduser(),\n"
        f"    Path({(VM_QUAID_HOME + '/data/project-docs/locks')!r}).expanduser(),\n"
        f"    Path({VM_QUAID_BENCH_SESSIONS_DIR!r}).expanduser(),\n"
        "]\n"
        "for path in state_dirs:\n"
        "    shutil.rmtree(path, ignore_errors=True)\n"
        "glob_patterns = [\n"
        "    ('~/clawd/journal', '*.journal.md'),\n"
        "    ('~/clawd/journal/archive', '*.md'),\n"
        "    ('~/clawd', '*.snippets.md'),\n"
        f"    ({(VM_QUAID_INSTANCE_ROOT_DIR + '/journal')!r}, '*.journal.md'),\n"
        f"    ({VM_QUAID_INSTANCE_ROOT_DIR!r}, '*.snippets.md'),\n"
        "    ('~/clawd/logs', 'janitor*.log'),\n"
        f"    ({VM_QUAID_INSTANCE_LOGS_DIR!r}, 'janitor*.log'),\n"
        "]\n"
        "for root, pattern in glob_patterns:\n"
        "    root_path = Path(root).expanduser()\n"
        "    if not root_path.exists():\n"
        "        continue\n"
        "    for path in root_path.glob(pattern):\n"
        "        try:\n"
        "            path.unlink()\n"
        "        except FileNotFoundError:\n"
        "            pass\n"
        "for extra in [\n"
        f"    Path({(VM_QUAID_INSTANCE_ROOT_DIR + '/.runtime/events/history.jsonl.lock')!r}).expanduser(),\n"
        f"    Path({(VM_QUAID_INSTANCE_ROOT_DIR + '/.runtime/events/queue.json.lock')!r}).expanduser(),\n"
        f"    Path({(VM_QUAID_INSTANCE_ROOT_DIR + '/.runtime/notes/delayed-llm-requests.json.lock')!r}).expanduser(),\n"
        f"    Path({(VM_QUAID_INSTANCE_LOGS_DIR + '/janitor/pending-approval-requests.json')!r}).expanduser(),\n"
        f"    Path({(VM_QUAID_INSTANCE_LOGS_DIR + '/janitor/pending-approval-requests.md')!r}).expanduser(),\n"
        "]:\n"
        "    try:\n"
        "        extra.unlink()\n"
        "    except FileNotFoundError:\n"
        "        pass\n"
        "shutil.rmtree(Path('~/clawd/projects').expanduser(), ignore_errors=True)\n"
        "print('Session state cleared')\n"
    )
    vm.ssh("python3 -c " + shlex.quote(script), timeout=20, raw=True)


def _clear_vm_native_memory_state(vm: TartVM):
    """Clear OpenClaw-native memory artifacts for a clean baseline.

    This is intentionally separate from Quaid cleanup so the native-memory
    benchmark path can be reset without changing the semantics of other systems.
    """
    vm.ssh(
        "rm -f ~/.openclaw/workspace/MEMORY.md 2>/dev/null; "
        "rm -rf ~/.openclaw/workspace/memory 2>/dev/null; "
        "rm -rf ~/.openclaw/workspace/wiki 2>/dev/null; "
        "rm -rf ~/.openclaw/plugins/active-memory 2>/dev/null || true; "
        "rm -rf ~/.openclaw/plugins/memory-wiki 2>/dev/null || true; "
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
        "d.pop('slots', None)\n"
        "plugins = d.setdefault('plugins', {})\n"
        "plugins.setdefault('enabled', True)\n"
        "plugins.pop('disable', None)\n"
        "plugins.pop('disabled', None)\n"
        "plugins.pop('slots', None)\n"
        "plugins['allow'] = ['memory-core', 'active-memory', 'memory-wiki']\n"
        "entries = plugins.setdefault('entries', {})\n"
        "for entry in entries.values():\n"
        "    if isinstance(entry, dict):\n"
        "        entry.pop('disable', None)\n"
        "        entry.pop('disabled', None)\n"
        "entries.setdefault('matrix', {})['enabled'] = False\n"
        "entries.setdefault('memory-core', {})['enabled'] = True\n"
        "entries.setdefault('active-memory', {})['enabled'] = True\n"
        "active_memory = entries.setdefault('active-memory', {}).setdefault('config', {})\n"
        "active_memory['enabled'] = True\n"
        "active_memory['agents'] = ['main']\n"
        "active_memory['allowedChatTypes'] = ['direct']\n"
        "active_memory['modelFallbackPolicy'] = 'default-remote'\n"
        "active_memory['queryMode'] = 'recent'\n"
        "active_memory['promptStyle'] = 'balanced'\n"
        "active_memory['timeoutMs'] = 15000\n"
        "active_memory['maxSummaryChars'] = 220\n"
        "active_memory['persistTranscripts'] = False\n"
        "active_memory['logging'] = True\n"
        "entries.setdefault('memory-wiki', {})['enabled'] = True\n"
        "memory_wiki = entries.setdefault('memory-wiki', {}).setdefault('config', {})\n"
        "memory_wiki['vaultMode'] = 'bridge'\n"
        "memory_wiki['vault'] = {'path': '~/.openclaw/workspace/wiki', 'renderMode': 'native'}\n"
        "memory_wiki['bridge'] = {\n"
        "    'enabled': True,\n"
        "    'readMemoryArtifacts': True,\n"
        "    'indexDreamReports': True,\n"
        "    'indexDailyNotes': True,\n"
        "    'indexMemoryRoot': True,\n"
        "    'followMemoryEvents': True,\n"
        "}\n"
        "memory_wiki['ingest'] = {'autoCompile': True, 'maxConcurrentJobs': 1, 'allowUrlIngest': True}\n"
        "memory_wiki['search'] = {'backend': 'shared', 'corpus': 'all'}\n"
        "memory_wiki['context'] = {'includeCompiledDigestPrompt': True}\n"
        "memory_wiki['render'] = {\n"
        "    'preserveHumanBlocks': True,\n"
        "    'createBacklinks': True,\n"
        "    'createDashboards': True,\n"
        "}\n"
        "entries.setdefault('memory-lancedb', {})['enabled'] = False\n"
        "entries.setdefault('quaid', {})['enabled'] = False\n"
        "entries.pop('quaid', None)\n"
        "memory = d.setdefault('memory', {})\n"
        "memory['backend'] = 'builtin'\n"
        "channels = d.setdefault('channels', {})\n"
        "for entry in channels.values():\n"
        "    if isinstance(entry, dict):\n"
        "        entry.pop('disable', None)\n"
        "        entry.pop('disabled', None)\n"
        "channels.setdefault('matrix', {})['enabled'] = False\n"
        "agents = d.setdefault('agents', {}).setdefault('defaults', {})\n"
        "tools = d.setdefault('tools', {})\n"
        f"tools['allow'] = {OC_NATIVE_MEMORY_TOOLS!r}\n"
        "tools.pop('deny', None)\n"
        "ms = agents.setdefault('memorySearch', {})\n"
        "ms['enabled'] = True\n"
        "ms['provider'] = 'openai'\n"
        f"ms['model'] = {OC_NATIVE_EMBED_MODEL!r}\n"
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
        "for entry in hook_entries.values():\n"
        "    if isinstance(entry, dict):\n"
        "        entry.pop('disable', None)\n"
        "        entry.pop('disabled', None)\n"
        "hook_entries.setdefault('session-memory', {})['enabled'] = enable_hook\n"
        "json.dump(d, open(p, 'w'), indent=2)\n"
        "print('Patched OpenClaw native memory config')\n"
    )


def _disable_openclaw_quaid_config_guard(vm: TartVM):
    """Unload the Quaid config guard and kill any live setup wrappers mutating OC config."""
    kill_script = (
        "import os, signal, subprocess, time\n"
        "rows = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,command='], text=True).splitlines()\n"
        "procs = {}\n"
        "for row in rows:\n"
        "    row = row.strip()\n"
        "    if not row:\n"
        "        continue\n"
        "    parts = row.split(None, 2)\n"
        "    if len(parts) < 3:\n"
        "        continue\n"
        "    pid = int(parts[0])\n"
        "    ppid = int(parts[1])\n"
        "    procs[pid] = {'ppid': ppid, 'cmd': parts[2]}\n"
        "targets = set()\n"
        "for pid, meta in procs.items():\n"
        "    cmd = meta['cmd']\n"
        "    if 'openclaw-config-guard.mjs' not in cmd and 'setup-quaid.mjs --agent' not in cmd:\n"
        "        continue\n"
        "    targets.add(pid)\n"
        "    parent = meta['ppid']\n"
        "    while parent in procs:\n"
        "        parent_cmd = procs[parent]['cmd']\n"
        "        if (\n"
        "            'openclaw-config-guard.mjs' in parent_cmd\n"
        "            or 'setup-quaid.mjs --agent' in parent_cmd\n"
        "        ):\n"
        "            targets.add(parent)\n"
        "            parent = procs[parent]['ppid']\n"
        "            continue\n"
        "        break\n"
        "changed = True\n"
        "while changed:\n"
        "    changed = False\n"
        "    for pid, meta in procs.items():\n"
        "        if meta['ppid'] in targets and pid not in targets:\n"
        "            targets.add(pid)\n"
        "            changed = True\n"
        "killed = set()\n"
        "for sig in (signal.SIGTERM, signal.SIGKILL):\n"
        "    for pid in sorted(targets, reverse=True):\n"
        "        try:\n"
        "            os.kill(pid, sig)\n"
        "        except (ProcessLookupError, PermissionError):\n"
        "            continue\n"
        "        else:\n"
        "            killed.add(pid)\n"
        "    if sig == signal.SIGTERM and targets:\n"
        "        time.sleep(0.5)\n"
        "msg = 'Quaid config guard disabled'\n"
        "if killed:\n"
        "    msg += '; killed=' + ','.join(str(pid) for pid in sorted(killed))\n"
        "print(msg)\n"
    )
    command = (
        "uid=$(id -u); "
        "plist=$HOME/Library/LaunchAgents/ai.openclaw.quaid-config-guard.plist; "
        "launchctl bootout gui/$uid \"$plist\" 2>/dev/null || "
        "launchctl unload \"$plist\" 2>/dev/null || true; "
        "launchctl disable gui/$uid/ai.openclaw.quaid-config-guard 2>/dev/null || true; "
        "launchctl remove ai.openclaw.quaid-config-guard 2>/dev/null || true; "
        "python3 - <<'PY'\n"
        f"{kill_script}"
        "PY\n"
    )
    result = vm.ssh(
        "sh -lc " + shlex.quote(command),
        timeout=20,
        raw=True,
    )
    if result.stdout.strip():
        print(f"  {result.stdout.strip()}")


def _patch_openclaw_native_memory(vm: TartVM, enable_session_hook: bool = True):
    """Configure the VM for the native OpenClaw memory baseline."""
    _disable_openclaw_quaid_config_guard(vm)
    script = _build_openclaw_native_config_script(enable_session_hook=enable_session_hook)
    result = vm.ssh("python3 -c " + shlex.quote(script), timeout=10)
    if result.stdout.strip():
        print(f"  {result.stdout.strip()}")


def _read_text_tail(path: Path, limit: int = 800) -> str:
    try:
        text = path.read_text(errors="replace")
    except Exception:
        return ""
    return text[-limit:]


def _run_host_command(
    args: list[str],
    *,
    tart_host: Optional[str] = None,
    timeout: int = 120,
) -> subprocess.CompletedProcess:
    if tart_host:
        remote_cmd = " ".join(shlex.quote(str(part)) for part in args)
        return subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", tart_host, remote_cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _run_host_shell(
    command: str,
    *,
    tart_host: Optional[str] = None,
    timeout: int = 120,
) -> subprocess.CompletedProcess:
    if tart_host:
        return subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", tart_host, command],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    return subprocess.run(
        ["sh", "-lc", command],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _read_host_text_tail(path: Path, *, limit: int = 800, tart_host: Optional[str] = None) -> str:
    if not tart_host:
        return _read_text_tail(path, limit=limit)
    script = (
        "from pathlib import Path\n"
        "import sys\n"
        "path = Path(sys.argv[1])\n"
        "limit = int(sys.argv[2])\n"
        "try:\n"
        "    text = path.read_text(errors='replace')\n"
        "except Exception:\n"
        "    text = ''\n"
        "print(text[-limit:], end='')\n"
    )
    result = _run_host_command(
        ["python3", "-c", script, str(path), str(limit)],
        tart_host=tart_host,
        timeout=30,
    )
    if result.returncode != 0:
        return ""
    return (result.stdout or "")[-limit:]


def _probe_json_url(url: str, timeout: int = 5, *, tart_host: Optional[str] = None) -> tuple[bool, str]:
    if tart_host:
        script = (
            "import json, sys\n"
            "from urllib.request import Request, urlopen\n"
            "url = sys.argv[1]\n"
            "timeout = float(sys.argv[2])\n"
            "try:\n"
            "    req = Request(url, headers={'Accept': 'application/json'})\n"
            "    with urlopen(req, timeout=timeout) as resp:\n"
            "        payload = json.load(resp)\n"
            "    print(json.dumps(payload)[:200])\n"
            "except Exception as exc:\n"
            "    print(str(exc))\n"
            "    raise SystemExit(1)\n"
        )
        result = _run_host_command(
            ["python3", "-c", script, url, str(timeout)],
            tart_host=tart_host,
            timeout=max(15, int(timeout) + 10),
        )
        detail = (result.stdout or result.stderr or "").strip()
        return result.returncode == 0, detail[:200]
    try:
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=timeout) as resp:
            payload = json.load(resp)
        return True, json.dumps(payload)[:200]
    except Exception as exc:
        return False, str(exc)


def _warm_oc_native_embed_endpoint(
    base_url: str,
    *,
    timeout_s: int,
    tart_host: Optional[str] = None,
) -> int:
    if tart_host:
        script = (
            "import json, sys\n"
            "from urllib.request import Request, urlopen\n"
            "base = sys.argv[1].rstrip('/')\n"
            "timeout = float(sys.argv[2])\n"
            f"model = {OC_NATIVE_EMBED_MODEL!r}\n"
            "payload = json.dumps({'model': model, 'input': ['ping']}).encode('utf-8')\n"
            "req = Request(\n"
            "    base + '/v1/embeddings',\n"
            "    data=payload,\n"
            "    headers={\n"
            "        'Content-Type': 'application/json',\n"
            "        'Authorization': 'Bearer ollama-local',\n"
            "    },\n"
            ")\n"
            "with urlopen(req, timeout=timeout) as resp:\n"
            "    data = json.load(resp)\n"
            "emb = (((data.get('data') or [{}])[0]).get('embedding') or [])\n"
            "if not emb:\n"
            "    raise RuntimeError('oc-native embed warmup returned empty embedding')\n"
            "print(len(emb))\n"
        )
        result = _run_host_command(
            ["python3", "-c", script, base_url, str(timeout_s)],
            tart_host=tart_host,
            timeout=max(30, int(timeout_s) + 15),
        )
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout or "").strip() or "remote embed warmup failed")
        return int((result.stdout or "").strip())
    base = base_url.rstrip("/")
    payload = json.dumps({"model": OC_NATIVE_EMBED_MODEL, "input": ["ping"]}).encode("utf-8")
    req = Request(
        f"{base}/v1/embeddings",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer ollama-local",
        },
    )
    with urlopen(req, timeout=timeout_s) as resp:
        data = json.load(resp)
    emb = (((data.get("data") or [{}])[0]).get("embedding") or [])
    if not emb:
        raise RuntimeError("oc-native embed warmup returned empty embedding")
    return len(emb)


def _warm_oc_native_embed_upstream(*, tart_host: Optional[str] = None) -> None:
    dims = _warm_oc_native_embed_endpoint(
        OC_NATIVE_EMBED_UPSTREAM,
        timeout_s=OC_NATIVE_EMBED_UPSTREAM_WARMUP_TIMEOUT_S,
        tart_host=tart_host,
    )
    print(f"  OC native embed upstream warmed ({dims} dims)")


def _list_loaded_ollama_models(*, tart_host: Optional[str] = None) -> list[str]:
    """Return loaded host-local Ollama models."""
    if tart_host:
        script = (
            "import json\n"
            "from urllib.request import Request, urlopen\n"
            f"url = {OC_NATIVE_EMBED_UPSTREAM.rstrip('/') + '/api/ps'!r}\n"
            "req = Request(url, headers={'Accept': 'application/json'})\n"
            "try:\n"
            "    with urlopen(req, timeout=10) as resp:\n"
            "        payload = json.load(resp)\n"
            "except Exception:\n"
            "    print('[]')\n"
            "    raise SystemExit(0)\n"
            "models = []\n"
            "for item in payload.get('models') or []:\n"
            "    model = item.get('model') or item.get('name')\n"
            "    if isinstance(model, str) and model not in models:\n"
            "        models.append(model)\n"
            "print(json.dumps(models))\n"
        )
        result = _run_host_command(
            ["python3", "-c", script],
            tart_host=tart_host,
            timeout=20,
        )
        if result.returncode != 0:
            return []
        try:
            return json.loads((result.stdout or "[]").strip() or "[]")
        except Exception:
            return []
    req = Request(
        f"{OC_NATIVE_EMBED_UPSTREAM.rstrip('/')}/api/ps",
        headers={"Accept": "application/json"},
    )
    try:
        with urlopen(req, timeout=10) as resp:
            payload = json.load(resp)
    except Exception:
        return []
    models = []
    for item in payload.get("models") or []:
        model = item.get("model") or item.get("name")
        if isinstance(model, str) and model not in models:
            models.append(model)
    return models


def _stop_ollama_model(model: str, *, tart_host: Optional[str] = None) -> None:
    try:
        _run_host_command(
            ["ollama", "stop", model],
            tart_host=tart_host,
            timeout=30,
        )
    except Exception:
        return


def _prepare_oc_native_embed_upstream(*, tart_host: Optional[str] = None) -> None:
    """Unload resident Ollama models so the OC-native embed model can cold-start cleanly."""
    loaded = _list_loaded_ollama_models(tart_host=tart_host)
    if not loaded:
        return
    print(f"  Reclaiming Ollama model memory: {', '.join(loaded)}")
    for model in loaded:
        _stop_ollama_model(model, tart_host=tart_host)
    time.sleep(2)


def _wait_for_oc_native_embed_upstream_ready(*, tart_host: Optional[str] = None) -> None:
    """Wait until the host-local Ollama embeddings endpoint is actually serving."""
    deadline = time.time() + OC_NATIVE_EMBED_UPSTREAM_READY_WAIT_S
    last_exc: Exception | None = None
    while time.time() < deadline:
        try:
            _warm_oc_native_embed_upstream(tart_host=tart_host)
            return
        except Exception as exc:
            last_exc = exc
            time.sleep(OC_NATIVE_EMBED_UPSTREAM_READY_POLL_S)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("oc-native embed upstream readiness wait elapsed without a probe result")


def _warm_oc_native_embed_proxy(port: int, *, tart_host: Optional[str] = None) -> None:
    dims = _warm_oc_native_embed_endpoint(
        f"http://127.0.0.1:{port}",
        timeout_s=OC_NATIVE_EMBED_PROXY_WARMUP_TIMEOUT_S,
        tart_host=tart_host,
    )
    print(f"  OC native embed proxy warmed ({dims} dims)")


def _wait_for_oc_native_embed_proxy_ready(port: int, *, tart_host: Optional[str] = None) -> None:
    """Wait until the host-side proxy can actually serve embeddings."""
    deadline = time.time() + OC_NATIVE_EMBED_PROXY_READY_WAIT_S
    last_exc: Exception | None = None
    while time.time() < deadline:
        try:
            _warm_oc_native_embed_proxy(port, tart_host=tart_host)
            return
        except Exception as exc:
            last_exc = exc
            time.sleep(OC_NATIVE_EMBED_PROXY_READY_POLL_S)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("oc-native embed proxy readiness wait elapsed without a probe result")


def _sync_oc_native_embed_proxy_script(tart_host: str) -> None:
    result = subprocess.run(
        [
            "scp",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            str(OC_NATIVE_EMBED_PROXY_SCRIPT),
            f"{tart_host}:{OC_NATIVE_EMBED_PROXY_REMOTE_SCRIPT}",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "failed to sync oc-native embed proxy script to tart host: "
            f"{(result.stderr or result.stdout or '').strip()}"
        )


def _stop_oc_native_embed_proxy(port: int, *, tart_host: Optional[str] = None) -> None:
    if tart_host:
        script = (
            f"PIDFILE={shlex.quote(str(OC_NATIVE_EMBED_PROXY_PIDFILE))}\n"
            f"PROXY={shlex.quote(OC_NATIVE_EMBED_PROXY_REMOTE_SCRIPT)}\n"
            "pids=''\n"
            "if [ -f \"$PIDFILE\" ]; then pids=\"$pids $(cat \"$PIDFILE\" 2>/dev/null || true)\"; fi\n"
            f"pids=\"$pids $(lsof -tiTCP:{port} -sTCP:LISTEN 2>/dev/null || true)\"\n"
            "pids=\"$pids $(pgrep -f \"$PROXY\" 2>/dev/null || true)\"\n"
            "for pid in $pids; do kill \"$pid\" 2>/dev/null || true; done\n"
            "sleep 1\n"
            "for pid in $pids; do kill -9 \"$pid\" 2>/dev/null || true; done\n"
            "rm -f \"$PIDFILE\"\n"
        )
        _run_host_shell(script, tart_host=tart_host, timeout=30)
        return
    pids: set[int] = set()
    if OC_NATIVE_EMBED_PROXY_PIDFILE.exists():
        try:
            pids.add(int(OC_NATIVE_EMBED_PROXY_PIDFILE.read_text().strip()))
        except ValueError:
            pass
    try:
        listeners = (
            subprocess.check_output(
                ["lsof", f"-tiTCP:{port}", "-sTCP:LISTEN"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
            .splitlines()
        )
    except subprocess.CalledProcessError:
        listeners = []
    for raw_pid in listeners:
        try:
            pids.add(int(raw_pid))
        except ValueError:
            pass
    try:
        proxy_pids = (
            subprocess.check_output(
                ["pgrep", "-f", str(OC_NATIVE_EMBED_PROXY_SCRIPT)],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
            .splitlines()
        )
    except subprocess.CalledProcessError:
        proxy_pids = []
    for raw_pid in proxy_pids:
        try:
            pids.add(int(raw_pid))
        except ValueError:
            pass
    for pid in sorted(pids):
        try:
            os.kill(pid, 15)
        except OSError:
            pass
    deadline = time.time() + 3
    while time.time() < deadline:
        remaining = []
        for pid in sorted(pids):
            try:
                os.kill(pid, 0)
            except OSError:
                continue
            remaining.append(pid)
        if not remaining:
            break
        time.sleep(0.2)
    for pid in sorted(pids):
        try:
            os.kill(pid, 0)
        except OSError:
            continue
        try:
            os.kill(pid, 9)
        except OSError:
            pass
    OC_NATIVE_EMBED_PROXY_PIDFILE.unlink(missing_ok=True)


def _start_oc_native_embed_proxy(upstream: str, port: int, *, tart_host: Optional[str] = None) -> None:
    if tart_host:
        _stop_oc_native_embed_proxy(port, tart_host=tart_host)
    else:
        _stop_oc_native_embed_proxy(port)
    if tart_host:
        _sync_oc_native_embed_proxy_script(tart_host)
        command = (
            f"nohup python3 {shlex.quote(OC_NATIVE_EMBED_PROXY_REMOTE_SCRIPT)} "
            f"--host 0.0.0.0 --port {port} --upstream {shlex.quote(upstream)} "
            f">{shlex.quote(str(OC_NATIVE_EMBED_PROXY_LOG))} 2>&1 </dev/null & "
            f"echo $! > {shlex.quote(str(OC_NATIVE_EMBED_PROXY_PIDFILE))}"
        )
        result = _run_host_shell(command, tart_host=tart_host, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(
                "failed to start remote oc-native embed proxy: "
                f"{(result.stderr or result.stdout or '').strip()}"
            )
        return
    with OC_NATIVE_EMBED_PROXY_LOG.open("ab") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                str(OC_NATIVE_EMBED_PROXY_SCRIPT),
                "--host",
                "0.0.0.0",
                "--port",
                str(port),
                "--upstream",
                upstream,
            ],
            stdout=log,
            stderr=log,
            stdin=subprocess.DEVNULL,
        )
    OC_NATIVE_EMBED_PROXY_PIDFILE.write_text(f"{proc.pid}\n")


def _ensure_oc_native_embed_proxy(tart_host: Optional[str] = None) -> None:
    """Ensure a fresh host-local OpenAI-compatible proxy exists before tunneling it into the VM."""
    upstream = OC_NATIVE_EMBED_UPSTREAM.rstrip("/")
    maybe_remote = {"tart_host": tart_host} if tart_host else {}
    healthy = False
    detail = "unprobed"
    for attempt in range(3):
        healthy, detail = _probe_json_url(f"{upstream}/v1/models", timeout=15, **maybe_remote)
        if healthy:
            break
        if attempt < 2:
            time.sleep(2)
    if not healthy:
        raise RuntimeError(f"oc-native host embed upstream not ready: {detail}")
    _prepare_oc_native_embed_upstream(**maybe_remote)
    try:
        _wait_for_oc_native_embed_upstream_ready(**maybe_remote)
    except Exception as exc:
        raise RuntimeError(f"oc-native host embed upstream embeddings not ready: {exc}") from exc

    parsed = urlparse(OC_NATIVE_EMBED_BASE_URL)
    port = parsed.port or 80
    _start_oc_native_embed_proxy(upstream, port, **maybe_remote)

    last_detail = "proxy startup probe unavailable"
    for _ in range(20):
        healthy, last_detail = _probe_json_url(f"http://127.0.0.1:{port}/v1/models", **maybe_remote)
        if healthy:
            try:
                _wait_for_oc_native_embed_proxy_ready(port, **maybe_remote)
            except Exception as exc:
                tail = _read_host_text_tail(OC_NATIVE_EMBED_PROXY_LOG, **maybe_remote)
                raise RuntimeError(
                    "oc-native embed proxy became reachable but embeddings never became ready: "
                    f"{exc} | log_tail={tail}"
                ) from exc
            proxy_host = tart_host or "localhost"
            print(f"  OC native embed proxy ready on host {proxy_host}:{port}")
            return
        time.sleep(0.5)

    tail = _read_host_text_tail(OC_NATIVE_EMBED_PROXY_LOG, **maybe_remote)
    raise RuntimeError(
        "oc-native embed proxy did not become ready: "
        f"{last_detail} | log_tail={tail}"
    )


def _validate_openclaw_native_memory(vm: TartVM):
    """Fail fast if the native OpenClaw memory baseline is not actually usable."""
    config_script = (
        "import json, os\n"
        "cfg = json.load(open(os.path.expanduser('~/.openclaw/openclaw.json')))\n"
        "ms = (((cfg.get('agents') or {}).get('defaults') or {}).get('memorySearch') or {})\n"
        "print(json.dumps({'provider': ms.get('provider'), 'model': ms.get('model'), 'baseUrl': (ms.get('remote') or {}).get('baseUrl')}, ensure_ascii=False))\n"
    )
    config_result = vm.ssh("python3 -c " + shlex.quote(config_script), timeout=30)
    if config_result.returncode != 0:
        raise RuntimeError(f"oc-native config read failed: {config_result.stderr[:200]}")
    try:
        status = json.loads(config_result.stdout.strip())
    except Exception as exc:
        raise RuntimeError(
            f"Could not parse oc-native memory config JSON: {config_result.stdout[:300]}"
        ) from exc

    provider = status.get("provider")
    model = status.get("model")
    if provider != "openai":
        raise RuntimeError(f"oc-native memory search resolved provider={provider!r}, expected 'openai'")
    if model != OC_NATIVE_EMBED_MODEL:
        raise RuntimeError(
            f"oc-native memory search resolved model={model!r}, expected {OC_NATIVE_EMBED_MODEL!r}"
        )
    probe_timeout = max(5, min(OC_NATIVE_EMBED_VALIDATION_PROBE_TIMEOUT_S, int(OC_NATIVE_EMBED_VALIDATION_TIMEOUT_S)))
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
        f"with urllib.request.urlopen(req, timeout={probe_timeout}) as resp:\n"
        "    data = json.load(resp)\n"
        "emb = (((data.get('data') or [{}])[0]).get('embedding') or [])\n"
        "print(json.dumps({'ok': True, 'dims': len(emb)}))\n"
    )
    last_detail = "embedding probe unavailable"
    deadline = time.time() + OC_NATIVE_EMBED_VALIDATION_TIMEOUT_S
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        refresh_ip = getattr(vm, "_refresh_ip_from_tart", None)
        if callable(refresh_ip):
            refresh_ip()
        probe_result = vm.ssh(
            "python3 -c " + shlex.quote(probe_script),
            timeout=probe_timeout + 15,
        )
        if probe_result.returncode == 0:
            try:
                probe_status = json.loads(probe_result.stdout.strip())
            except Exception as exc:
                raise RuntimeError(
                    f"oc-native embeddings probe returned invalid JSON: {probe_result.stdout[:300]}"
                ) from exc
            dims = probe_status.get("dims")
            if dims != OC_NATIVE_EMBED_DIMS:
                raise RuntimeError(
                    f"oc-native embeddings probe returned dims={dims!r}, expected {OC_NATIVE_EMBED_DIMS}"
                )
            print(f"  Native memory verified: provider={provider} model={model}")
            return
        last_detail = probe_result.stderr.strip() or probe_result.stdout.strip() or last_detail
        if time.time() + 3 >= deadline:
            break
        time.sleep(3)
    raise RuntimeError(f"oc-native embeddings not ready: {last_detail}")


def _validate_quaid_vm_embeddings(
    vm: TartVM,
    *,
    instance_id: str = VM_QUAID_INSTANCE,
    ollama_url: str = VM_QUAID_OLLAMA_URL,
) -> None:
    """Fail fast if the Quaid VM cannot reach the host embeddings proxy."""
    probe_timeout = max(
        5,
        min(
            OC_NATIVE_EMBED_VALIDATION_PROBE_TIMEOUT_S,
            int(OC_NATIVE_EMBED_VALIDATION_TIMEOUT_S),
        ),
    )
    probe_script = (
        "import json, urllib.request\n"
        f"base = {ollama_url!r}.rstrip('/')\n"
        f"model = {OC_NATIVE_EMBED_MODEL!r}\n"
        "headers = {'Content-Type': 'application/json'}\n"
        "req = urllib.request.Request(\n"
        "    base + '/api/embed',\n"
        "    data=json.dumps({'model': model, 'input': ['ping'], 'keep_alive': -1}).encode('utf-8'),\n"
        "    headers=headers,\n"
        ")\n"
        f"with urllib.request.urlopen(req, timeout={probe_timeout}) as resp:\n"
        "    data = json.load(resp)\n"
        "embeddings = data.get('embeddings') or []\n"
        "first = embeddings[0] if embeddings else []\n"
        "print(json.dumps({'ok': True, 'dims': len(first)}))\n"
    )
    last_detail = "embedding probe unavailable"
    deadline = time.time() + OC_NATIVE_EMBED_VALIDATION_TIMEOUT_S
    while time.time() < deadline:
        refresh_ip = getattr(vm, "_refresh_ip_from_tart", None)
        if callable(refresh_ip):
            refresh_ip()
        probe_result = vm.ssh(
            "python3 -c " + shlex.quote(probe_script),
            timeout=probe_timeout + 15,
        )
        if probe_result.returncode == 0:
            try:
                probe_status = json.loads(probe_result.stdout.strip())
            except Exception as exc:
                raise RuntimeError(
                    f"quaid vm embeddings probe returned invalid JSON: {probe_result.stdout[:300]}"
                ) from exc
            dims = probe_status.get("dims")
            if dims != OC_NATIVE_EMBED_DIMS:
                raise RuntimeError(
                    f"quaid vm embeddings probe returned dims={dims!r}, expected {OC_NATIVE_EMBED_DIMS}"
                )
            print(
                "  Quaid embeddings verified: "
                f"instance={instance_id} model={OC_NATIVE_EMBED_MODEL} url={ollama_url}"
            )
            return
        last_detail = probe_result.stderr.strip() or probe_result.stdout.strip() or last_detail
        if time.time() + 3 >= deadline:
            break
        time.sleep(3)
    raise RuntimeError(f"quaid vm embeddings not ready for instance={instance_id}: {last_detail}")


def _extract_openclaw_memory_status(stdout: str) -> list:
    """Parse `openclaw memory status --json` even if warnings precede the payload."""
    payload = stdout.strip()
    for marker in ("[\n", "[{", "["):
        idx = payload.find(marker)
        if idx >= 0:
            payload = payload[idx:]
            break
    return json.loads(payload)


def _is_openclaw_memory_status_unsupported(detail: str) -> bool:
    lowered = str(detail or "").lower()
    return "unknown command 'memory'" in lowered


def _oc_native_session_id(review, ordinal: int) -> str:
    """Build a stable per-review session id for the native OpenClaw baseline."""
    snum = getattr(review, "session_num", None)
    if isinstance(snum, int):
        if snum > 0:
            return f"benchmark-oc-native-s{snum:02d}"
        if snum < 0:
            return f"benchmark-oc-native-f{abs(snum):03d}"
    return f"benchmark-oc-native-r{ordinal:03d}"


def _oc_native_session_key(session_id: str) -> str:
    """Return a stable agent-scoped session key for OC-native gateway turns."""
    return f"agent:main:{session_id}"


def _oc_native_eval_session_key(session_id: str) -> str:
    """Return a hook-scoped key for OC-native eval turns.

    Hook-prefixed keys are excluded from the adapter's user-session fallback
    scans, which prevents eval turns from being rediscovered as candidate user
    transcripts while keeping prompt-time recall injection enabled.
    """
    return f"agent:main:hook:{session_id}"


def _oc_native_eval_session_file(session_id: str) -> str:
    """Store OC eval transcripts under a sibling agent's session tree.

    OpenClaw normalizes session transcripts into an ``agents/<id>/sessions``
    layout. Keeping eval turns under a sibling agent preserves the live
    transcript path but keeps them outside the main agent's native-memory
    session index.
    """
    return f"{VM_OC_EVAL_SESSIONS_DIR}/{session_id}.jsonl"


def _uses_oc_gateway_eval_isolation(system: str) -> bool:
    """Return True when eval turns must stay on hook-scoped OC sessions.

    Both the native OpenClaw baseline and Quaid-on-OC-VM answer through the
    OpenClaw agent stack. Their eval turns must therefore avoid the default
    ``agent:main:main`` store key and ``main/sessions`` transcript path, or the
    live eval session can be recalled during scoring.
    """
    return system in {"oc-native", "quaid"}


def _require_quaid_extraction_usage(extraction_usage: dict, *, context: str) -> dict:
    """Fail hard when Quaid extraction did not report real token usage.

    A zero-usage return means the extraction path failed, short-circuited, or
    stopped emitting the structured usage payload the benchmark relies on. Any
    of those states invalidate the run and must abort immediately.
    """
    if extraction_usage.get("input_tokens", 0) <= 0:
        raise RuntimeError(f"Quaid extraction failed to report usage ({context})")
    return extraction_usage


def _quaid_benchmark_session_file(session_id: str) -> str:
    """Return the adapter-owned transcript path used for direct Quaid extraction."""
    return f"{VM_QUAID_BENCH_SESSIONS_DIR}/{session_id}.jsonl"


def _write_vm_session_jsonl(
    vm: TartVM,
    session_id: str,
    jsonl: str,
    append: bool = True,
    *,
    sessions_dir: str = VM_AGENT_SESSIONS_DIR,
):
    """Write transcript JSONL to a VM session file."""
    operator = ">>" if append else ">"
    return vm.ssh(
        f"mkdir -p {sessions_dir} && cat {operator} {sessions_dir}/{session_id}.jsonl",
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


def _read_vm_session_jsonl(vm: TartVM, session_id: str) -> str:
    """Read a VM session JSONL file, returning an empty string if missing."""
    result = vm.ssh(
        f"cat {VM_AGENT_SESSIONS_DIR}/{session_id}.jsonl 2>/dev/null || true",
        timeout=15,
        raw=True,
    )
    return result.stdout or ""


def _wait_for_vm_session_jsonl_quiet(
    vm: TartVM,
    session_id: str,
    *,
    timeout_s: float = OC_NATIVE_SESSION_QUIET_TIMEOUT_S,
    quiet_s: float = OC_NATIVE_SESSION_QUIET_WINDOW_S,
) -> None:
    """Wait until a session file stops changing before restoring benchmark content."""
    script = (
        "from pathlib import Path\n"
        "import sys, time\n"
        f"session_id = {session_id!r}\n"
        f"timeout_s = float({timeout_s!r})\n"
        f"quiet_s = float({quiet_s!r})\n"
        "path = Path.home()/'.openclaw'/'agents'/'main'/'sessions'/f'{session_id}.jsonl'\n"
        "deadline = time.time() + timeout_s\n"
        "last_sig = None\n"
        "stable_since = None\n"
        "while time.time() < deadline:\n"
        "    if path.exists():\n"
        "        stat = path.stat()\n"
        "        sig = (stat.st_size, stat.st_mtime_ns)\n"
        "        now = time.time()\n"
        "        if sig == last_sig:\n"
        "            if stable_since is None:\n"
        "                stable_since = now\n"
        "            if now - stable_since >= quiet_s:\n"
        "                print(f'{sig[0]}:{sig[1]}')\n"
        "                sys.exit(0)\n"
        "        else:\n"
        "            last_sig = sig\n"
        "            stable_since = now\n"
        "    time.sleep(0.25)\n"
        "print('timeout')\n"
        "sys.exit(1)\n"
    )
    result = vm.ssh(
        "python3 -c " + shlex.quote(script),
        timeout=max(int(timeout_s) + 5, 20),
        raw=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"oc-native session file did not go quiet for {session_id}: {(result.stdout or result.stderr)[:200]}"
        )


def _extract_oc_native_gateway_run_id(payload: dict) -> str:
    run_id = payload.get("runId") if isinstance(payload, dict) else None
    if (not isinstance(run_id, str) or not run_id.strip()) and isinstance(payload.get("result"), dict):
        run_id = payload["result"].get("runId")
    return run_id.strip() if isinstance(run_id, str) else ""


def _extract_json_object_from_mixed_stdout(text: str) -> dict:
    text = text or ""
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    return json.loads(text[start:])


def _is_oc_native_invalid_config_error(detail: str) -> bool:
    lower_detail = str(detail or "").lower()
    return (
        "invalid config at" in lower_detail
        or "config invalid" in lower_detail
        or ("unrecognized key" in lower_detail and "plugins.entries." in lower_detail)
    )


def _repair_oc_native_loopback_pairing(vm: TartVM) -> bool:
    """Approve the latest local loopback device repair request if present."""
    repair_script = (
        "import json, time\n"
        "from pathlib import Path\n"
        "# OC_NATIVE_PAIR_REPAIR\n"
        "base = Path.home() / '.openclaw' / 'devices'\n"
        "pending_path = base / 'pending.json'\n"
        "paired_path = base / 'paired.json'\n"
        "pending = json.loads(pending_path.read_text()) if pending_path.exists() else {}\n"
        "paired = json.loads(paired_path.read_text()) if paired_path.exists() else {}\n"
        "if not pending:\n"
        "    print(json.dumps({'approved': False, 'reason': 'no-pending'}))\n"
        "    raise SystemExit(0)\n"
        "request_id, req = sorted(\n"
        "    pending.items(),\n"
        "    key=lambda kv: (kv[1] or {}).get('ts', 0),\n"
        "    reverse=True,\n"
        ")[0]\n"
        "device_id = str((req or {}).get('deviceId') or '').strip()\n"
        "if not device_id:\n"
        "    print(json.dumps({'approved': False, 'reason': 'missing-device-id'}))\n"
        "    raise SystemExit(0)\n"
        "existing = dict(paired.get(device_id) or {})\n"
        "merged_scopes = []\n"
        "for scope in list(existing.get('approvedScopes') or existing.get('scopes') or []) + list(req.get('scopes') or []):\n"
        "    if isinstance(scope, str) and scope and scope not in merged_scopes:\n"
        "        merged_scopes.append(scope)\n"
        "roles = []\n"
        "for role in list(existing.get('roles') or []) + list(req.get('roles') or []):\n"
        "    if isinstance(role, str) and role and role not in roles:\n"
        "        roles.append(role)\n"
        "role = req.get('role') or existing.get('role')\n"
        "now = int(time.time() * 1000)\n"
        "tokens = dict(existing.get('tokens') or {})\n"
        "if isinstance(role, str) and role:\n"
        "    token_entry = dict(tokens.get(role) or {})\n"
        "    if token_entry:\n"
        "        token_entry['scopes'] = merged_scopes\n"
        "        token_entry.setdefault('createdAtMs', now)\n"
        "        token_entry['rotatedAtMs'] = now\n"
        "        tokens[role] = token_entry\n"
        "paired[device_id] = {\n"
        "    **existing,\n"
        "    'deviceId': device_id,\n"
        "    'publicKey': req.get('publicKey') or existing.get('publicKey'),\n"
        "    'displayName': req.get('displayName') or existing.get('displayName'),\n"
        "    'platform': req.get('platform') or existing.get('platform'),\n"
        "    'deviceFamily': req.get('deviceFamily') or existing.get('deviceFamily'),\n"
        "    'clientId': req.get('clientId') or existing.get('clientId'),\n"
        "    'clientMode': req.get('clientMode') or existing.get('clientMode'),\n"
        "    'role': role,\n"
        "    'roles': roles or existing.get('roles') or ([role] if role else []),\n"
        "    'scopes': merged_scopes,\n"
        "    'approvedScopes': merged_scopes,\n"
        "    'remoteIp': req.get('remoteIp') or existing.get('remoteIp'),\n"
        "    'tokens': tokens,\n"
        "    'createdAtMs': existing.get('createdAtMs') or now,\n"
        "    'approvedAtMs': now,\n"
        "}\n"
        "del pending[request_id]\n"
        "base.mkdir(parents=True, exist_ok=True)\n"
        "pending_path.write_text(json.dumps(pending, indent=2) + '\\n')\n"
        "paired_path.write_text(json.dumps(paired, indent=2) + '\\n')\n"
        "print(json.dumps({'approved': True, 'requestId': request_id, 'deviceId': device_id}))\n"
    )
    repair_result = vm.ssh(
        "python3 - <<'PY'\n" + repair_script + "PY",
        timeout=45,
    )
    if repair_result.returncode != 0:
        return False
    try:
        payload = _extract_json_object_from_mixed_stdout(
            (repair_result.stdout or "") + "\n" + (repair_result.stderr or "")
        )
    except Exception:
        return False
    return bool(payload.get("approved"))


def _oc_native_gateway_call(
    vm: TartVM,
    method: str,
    params: dict,
    *,
    timeout_s: int,
    ssh_timeout_s: Optional[int] = None,
) -> dict:
    """Call the OC-native gateway on the VM and return parsed JSON."""
    command = (
        "OPENCLAW_CONFIG_PATH=$HOME/.openclaw/openclaw.json "
        f"openclaw gateway call {method} --json "
        f"--params {shlex.quote(json.dumps(params, ensure_ascii=False))} "
        f"--timeout {int(timeout_s * 1000)}"
    )
    ssh_timeout = ssh_timeout_s or max(timeout_s + 15, 30)
    repaired_pairing = False
    gateway_restart_attempts = 0
    invalid_config_restart_attempts = 0
    result = None
    detail = ""
    while True:
        result = vm.ssh(command, timeout=ssh_timeout)
        detail = ((result.stderr or "") + "\n" + (result.stdout or "")).strip()
        if result.returncode == 0:
            break
        lower_detail = detail.lower()
        if not repaired_pairing and "pairing required" in lower_detail:
            repaired_pairing = True
            if _repair_oc_native_loopback_pairing(vm):
                continue
        if "gateway closed" in lower_detail and gateway_restart_attempts < OC_NATIVE_GATEWAY_CALL_RESTART_LIMIT:
            gateway_restart_attempts += 1
            _restart_oc_native_gateway(vm)
            continue
        if (
            _is_oc_native_invalid_config_error(detail)
            and invalid_config_restart_attempts < OC_NATIVE_GATEWAY_CALL_RESTART_LIMIT
        ):
            invalid_config_restart_attempts += 1
            _restart_oc_native_gateway(vm)
            continue
        raise RuntimeError(
            f"oc-native gateway call failed method={method}: {detail[:300]}"
        )
    try:
        return json.loads(result.stdout or "{}")
    except Exception as exc:
        raise RuntimeError(
            f"oc-native gateway call returned invalid JSON for {method}: {(result.stdout or '')[:200]}"
        ) from exc


def _read_oc_native_session_tail_state(
    vm: TartVM,
    session_id: str,
    *,
    session_key: Optional[str] = None,
) -> dict:
    """Return the latest assistant event details from an OC-native session file.

    OC gateway may remap a session key onto a generated session id/file even when
    the caller supplied a stable benchmark session id. Follow the session store
    first, then fall back to the legacy <session_id>.jsonl path.
    """
    script = (
        "import json\n"
        "from pathlib import Path\n"
        f"session_id = {session_id!r}\n"
        f"session_key = {session_key!r}\n"
        "sessions_dir = Path.home()/'.openclaw'/'agents'/'main'/'sessions'\n"
        "path = None\n"
        "store = sessions_dir/'sessions.json'\n"
        "if session_key and store.exists():\n"
        "    try:\n"
        "        data = json.loads(store.read_text(errors='replace'))\n"
        "    except Exception:\n"
        "        data = {}\n"
        "    entry = data.get(session_key)\n"
        "    if isinstance(entry, dict):\n"
        "        session_file = entry.get('sessionFile')\n"
        "        if isinstance(session_file, str) and session_file.strip():\n"
        "            candidate = Path(session_file)\n"
        "            if candidate.exists():\n"
        "                path = candidate\n"
        "if path is None:\n"
        "    fallback = sessions_dir/f'{session_id}.jsonl'\n"
        "    if fallback.exists():\n"
        "        path = fallback\n"
        "state = {\n"
        "    'path': str(path) if path else '',\n"
        "    'line_count': 0,\n"
        "    'assistant_text': '',\n"
        "    'assistant_event_id': '',\n"
        "    'assistant_timestamp': '',\n"
        "}\n"
        "if path and path.exists():\n"
        "    lines = path.read_text(errors='replace').splitlines()\n"
        "    state['line_count'] = len(lines)\n"
        "    for raw in reversed(lines):\n"
        "        if not raw.strip():\n"
        "            continue\n"
        "        try:\n"
        "            event = json.loads(raw)\n"
        "        except Exception:\n"
        "            continue\n"
        "        if event.get('type') != 'message':\n"
        "            continue\n"
        "        message = event.get('message') or {}\n"
        "        if message.get('role') != 'assistant':\n"
        "            continue\n"
        "        parts = []\n"
        "        for item in message.get('content') or []:\n"
        "            if isinstance(item, dict) and item.get('type') == 'text':\n"
        "                text = item.get('text')\n"
        "                if isinstance(text, str) and text.strip():\n"
        "                    parts.append(text)\n"
        "        state['assistant_text'] = '\\n'.join(parts).strip()\n"
        "        state['assistant_event_id'] = str(event.get('id') or '')\n"
        "        state['assistant_timestamp'] = str(event.get('timestamp') or '')\n"
        "        break\n"
        "print(json.dumps(state))\n"
    )
    result = vm.ssh(
        "python3 -c " + shlex.quote(script),
        timeout=20,
        raw=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"oc-native session parse failed for {session_id}: {(result.stderr or result.stdout)[:200]}"
        )
    try:
        state = json.loads((result.stdout or "").strip() or "{}")
    except Exception as exc:
        raise RuntimeError(
            f"oc-native session state parse failed for {session_id}: {(result.stdout or '')[:200]}"
        ) from exc
    if not isinstance(state, dict):
        raise RuntimeError(f"oc-native session state was not an object for {session_id}")
    return state


def _read_oc_native_last_assistant_message(
    vm: TartVM,
    session_id: str,
    *,
    session_key: Optional[str] = None,
) -> str:
    state = _read_oc_native_session_tail_state(vm, session_id, session_key=session_key)
    return str(state.get("assistant_text") or "").strip()


def _wait_for_oc_native_session_completion(
    vm: TartVM,
    session_id: str,
    *,
    session_key: Optional[str],
    previous_state: dict,
    timeout_s: int = 20,
) -> str:
    """Accept late OC completion only when the session transcript proves progress."""
    deadline = time.time() + max(timeout_s, 1)
    previous_line_count = int(previous_state.get("line_count") or 0)
    previous_event_id = str(previous_state.get("assistant_event_id") or "")
    previous_timestamp = str(previous_state.get("assistant_timestamp") or "")
    while time.time() < deadline:
        state = _read_oc_native_session_tail_state(vm, session_id, session_key=session_key)
        line_count = int(state.get("line_count") or 0)
        assistant_text = str(state.get("assistant_text") or "").strip()
        assistant_event_id = str(state.get("assistant_event_id") or "")
        assistant_timestamp = str(state.get("assistant_timestamp") or "")
        progressed = line_count > previous_line_count
        assistant_changed = (
            assistant_event_id != previous_event_id
            or assistant_timestamp != previous_timestamp
        )
        if progressed and assistant_changed and assistant_text:
            print(
                f"  WARN: oc-native agent.wait timed out for {session_id}; "
                "using transcript-confirmed completion"
            )
            return assistant_text
        time.sleep(2)
    return ""


def _count_vm_session_jsonl_files(vm: TartVM) -> Optional[int]:
    """Count guest session transcript files when native status underreports them."""
    script = (
        "from pathlib import Path\n"
        "path = Path.home()/'.openclaw'/'agents'/'main'/'sessions'\n"
        "count = sum(1 for child in path.glob('*.jsonl') if child.is_file())\n"
        "print(count)\n"
    )
    result = vm.ssh(
        "python3 -c " + shlex.quote(script),
        timeout=20,
        raw=True,
    )
    if result.returncode != 0:
        return None
    try:
        return int((result.stdout or "").strip())
    except Exception:
        return None


def _run_oc_native_gateway_turn(
    vm: TartVM,
    session_id: str,
    message: str,
    *,
    timeout_s: int,
    session_key: Optional[str] = None,
) -> str:
    """Run an OC-native agent turn via gateway RPC and return the assistant text."""
    if not session_key:
        session_key = _oc_native_session_key(session_id)
    previous_state = _read_oc_native_session_tail_state(
        vm,
        session_id,
        session_key=session_key,
    )
    payload = _oc_native_gateway_call(
        vm,
        "agent",
        {
            "agentId": "main",
            "sessionKey": session_key,
            "sessionId": session_id,
            "message": message,
            "idempotencyKey": f"bench-oc-{session_id}-{uuid.uuid4().hex[:12]}",
        },
        timeout_s=max(timeout_s, 45),
        # OC-native gateway acceptance can lag well past the logical RPC
        # timeout under heavy startup/indexing load. Do not let the outer SSH
        # wrapper kill a still-live gateway call prematurely.
        ssh_timeout_s=max(timeout_s + 150, 240),
    )
    run_id = _extract_oc_native_gateway_run_id(payload)
    if run_id:
        wait_timeout_s = min(timeout_s + 60, 240)
        try:
            wait_payload = _oc_native_gateway_call(
                vm,
                "agent.wait",
                {
                    "runId": run_id,
                    "timeoutMs": wait_timeout_s * 1000,
                },
                # Keep the gateway-call transport timeout at or above the logical
                # wait window. Otherwise the CLI can self-time out before the
                # requested agent.wait deadline is reached.
                timeout_s=max(wait_timeout_s + 15, 120),
                ssh_timeout_s=max(wait_timeout_s + 180, 300),
            )
        except RuntimeError as exc:
            transcript_answer = _wait_for_oc_native_session_completion(
                vm,
                session_id,
                session_key=session_key,
                previous_state=previous_state,
            )
            if transcript_answer:
                print(
                    f"  WARN: oc-native agent.wait transport failed for {session_id}; "
                    "using transcript-confirmed completion"
                )
                return transcript_answer
            raise exc
        status = str(
            wait_payload.get("status")
            or wait_payload.get("state")
            or (
                wait_payload.get("result", {}).get("status")
                if isinstance(wait_payload.get("result"), dict)
                else ""
            )
            or ""
        ).strip().lower()
        if status not in {"ok", "succeeded", "success", "done", "completed", "complete"}:
            transcript_answer = _wait_for_oc_native_session_completion(
                vm,
                session_id,
                session_key=session_key,
                previous_state=previous_state,
            )
            if transcript_answer:
                return transcript_answer
            raise RuntimeError(
                f"oc-native agent.wait did not complete cleanly for {session_id}: {wait_payload}"
            )
    answer = _read_oc_native_last_assistant_message(vm, session_id, session_key=session_key)
    if not answer:
        raise RuntimeError(f"oc-native gateway turn produced no assistant text for {session_id}")
    return answer


def _sync_openclaw_native_memory(
    vm: TartVM,
    source_name: str = "sessions",
    min_indexed_files: int = 1,
    *,
    force: bool = False,
) -> dict:
    """Run OC native memory indexing directly, then require clean indexed status."""
    # The OC guest can respawn setup-quaid and drift the config mid-run.
    # Reapply the benchmark-owned config immediately before every native sync.
    _patch_openclaw_native_memory(vm, enable_session_hook=True)
    force_flag = " --force" if force else ""
    action = "reindexed" if force else "synced"
    index_result = vm.ssh(
        "sh -lc 'export PATH=/opt/homebrew/bin:$PATH; "
        f"openclaw memory index --agent main{force_flag} > /tmp/oc-native-reindex.log 2>&1'",
        timeout=OC_NATIVE_REINDEX_TIMEOUT_S + 120,
    )
    if index_result.returncode != 0:
        log = vm.ssh("tail -80 /tmp/oc-native-reindex.log 2>/dev/null", timeout=10, raw=True)
        log_detail = log.stdout.strip() if log.returncode == 0 else ""
        raise RuntimeError(
            f"oc-native memory index failed: {log_detail[:500] or (index_result.stderr or index_result.stdout)[:200]}"
        )
    index_log = vm.ssh("tail -80 /tmp/oc-native-reindex.log 2>/dev/null", timeout=10, raw=True)
    index_log_detail = index_log.stdout.strip() if index_log.returncode == 0 else ""

    deadline = time.time() + OC_NATIVE_REINDEX_STATUS_WAIT_S
    last_detail = ""
    last_status = None
    while True:
        result = vm.ssh(
            "openclaw memory status --agent main --json",
            timeout=90,
        )
        if result.returncode == 0 and result.stdout:
            try:
                payload = _extract_openclaw_memory_status(result.stdout)
                status = payload[0]["status"]
            except Exception:
                last_detail = f"Could not parse openclaw native reindex status: {result.stdout[:300]}"
            else:
                source_counts = {entry.get("source"): entry for entry in (status.get("sourceCounts") or [])}
                source = source_counts.get(source_name) or {}
                indexed_files = int(source.get("files") or 0)
                indexed_chunks = int(source.get("chunks") or 0)
                dirty = bool(status.get("dirty"))
                last_status = {
                    "dirty": dirty,
                    "files": indexed_files,
                    "chunks": indexed_chunks,
                }
                if not dirty and indexed_files >= min_indexed_files and indexed_chunks > 0:
                    print(f"  Native memory {action}: {source_name} files={indexed_files} chunks={indexed_chunks}")
                    return status
                if (
                    source_name == "sessions"
                    and not dirty
                    and indexed_chunks > 0
                    and indexed_files < min_indexed_files
                    and "Memory index updated (" in index_log_detail
                ):
                    disk_session_files = _count_vm_session_jsonl_files(vm)
                    if disk_session_files is not None and disk_session_files >= min_indexed_files:
                        source["files"] = disk_session_files
                        if not source_counts.get(source_name):
                            status.setdefault("sourceCounts", []).append(source)
                        print(
                            "  Native memory "
                            f"{action}: {source_name} files>={disk_session_files} chunks={indexed_chunks} "
                            "(compat fallback: status underreported sourceCounts.files)"
                        )
                        return status
                last_detail = (
                    f"oc-native {source_name} did not finish indexing "
                    f"(dirty={dirty}, files={indexed_files}, chunks={indexed_chunks}, expected files>={min_indexed_files})"
                )
        else:
            detail = (result.stderr or result.stdout or "").strip()
            if _is_openclaw_memory_status_unsupported(detail):
                if "Memory index updated (" in index_log_detail:
                    print(
                        "  Native memory "
                        f"{action}: {source_name} files>={min_indexed_files} "
                        "(compat fallback: memory status unavailable)"
                    )
                    return {
                        "dirty": False,
                        "sourceCounts": [
                            {
                                "source": source_name,
                                "files": min_indexed_files,
                                "chunks": 1,
                            }
                        ],
                    }
            last_detail = f"oc-native memory status failed after index: {detail[:200]}".strip()

        if time.time() >= deadline:
            log = vm.ssh("tail -80 /tmp/oc-native-reindex.log 2>/dev/null", timeout=10, raw=True)
            log_detail = log.stdout.strip() if log.returncode == 0 else ""
            if last_status:
                raise RuntimeError(
                    f"oc-native {source_name} did not finish indexing "
                    f"(dirty={last_status['dirty']}, files={last_status['files']}, chunks={last_status['chunks']}, "
                    f"expected files>={min_indexed_files}) {log_detail[:500]}".strip()
                )
            raise RuntimeError(
                f"{last_detail} {log_detail[:500]}".strip()
            )
        time.sleep(OC_NATIVE_REINDEX_STATUS_POLL_S)


def _force_openclaw_native_reindex(
    vm: TartVM, source_name: str = "sessions", min_indexed_files: int = 1
) -> dict:
    """Force a native OpenClaw memory reindex and require one source to finish indexing."""
    return _sync_openclaw_native_memory(
        vm,
        source_name=source_name,
        min_indexed_files=min_indexed_files,
        force=True,
    )


def _sync_openclaw_native_wiki(vm: TartVM):
    """Sync memory-core public artifacts into memory-wiki after injection."""
    result = vm.ssh(
        "sh -lc '"
        "openclaw wiki init >/tmp/oc-native-wiki-sync.log 2>&1 && "
        "openclaw wiki bridge import >>/tmp/oc-native-wiki-sync.log 2>&1 && "
        "openclaw wiki compile >>/tmp/oc-native-wiki-sync.log 2>&1 && "
        "openclaw wiki status --json"
        "'",
        timeout=300,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        log = vm.ssh("tail -80 /tmp/oc-native-wiki-sync.log 2>/dev/null", timeout=10, raw=True)
        log_detail = log.stdout.strip() if log.returncode == 0 else ""
        raise RuntimeError(
            "oc-native memory-wiki sync failed: "
            f"{detail[:200]} {log_detail[:500]}".strip()
        )
    print("  Native memory wiki synced")


def _run_oc_native_session_hook(vm: TartVM, session_id: str):
    """Run the bundled session-memory hook via a benign agent turn, then restore transcript.

    A full `/new` round-trip is flaky on the current OC VM. A short normal agent
    turn through the gateway still exercises the same startup/session hook path.
    We restore the synthetic transcript afterward so native session indexing sees
    only the benchmark conversation, not the benign hook turn.
    """
    session_key = _oc_native_session_key(session_id)
    _register_session(vm, session_id, session_key=session_key)
    original = _read_vm_session_jsonl(vm, session_id)
    _run_oc_native_gateway_turn(vm, session_id, "hello", timeout_s=45, session_key=session_key)
    last_restore_detail = ""
    for attempt in range(1, OC_NATIVE_SESSION_RESTORE_ATTEMPTS + 1):
        _wait_for_vm_session_jsonl_quiet(vm, session_id)
        restore = _write_vm_session_jsonl(vm, session_id, original, append=False)
        if restore.returncode != 0:
            raise RuntimeError(
                f"oc-native transcript restore failed for {session_id}: {(restore.stderr or restore.stdout)[:200]}"
            )
        restored = _read_vm_session_jsonl(vm, session_id)
        if restored == original:
            return
        last_restore_detail = (
            f"attempt={attempt} expected_bytes={len(original)} restored_bytes={len(restored)}"
        )
        time.sleep(0.5 * attempt)
    raise RuntimeError(
        f"oc-native transcript restore did not stick for {session_id}: {last_restore_detail}"
    )


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
    try:
        result = vm.ssh(
            "python3 -c " + shlex.quote(script),
            timeout=max(10, int(timeout_s) + 5),
        )
    except subprocess.TimeoutExpired:
        return False
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
    script = (
        "import socket, sys, time\n"
        f"host = {host!r}\n"
        f"port = {port}\n"
        f"deadline = time.time() + float({timeout_s!r})\n"
        f"probe_timeout_s = float({probe_timeout_s!r})\n"
        f"poll_interval_s = float({poll_interval_s!r})\n"
        "while time.time() < deadline:\n"
        "    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
        "    sock.settimeout(probe_timeout_s)\n"
        "    try:\n"
        "        sock.connect((host, port))\n"
        "    except Exception:\n"
        "        time.sleep(poll_interval_s)\n"
        "    else:\n"
        "        print('ready')\n"
        "        sys.exit(0)\n"
        "    finally:\n"
        "        try:\n"
        "            sock.close()\n"
        "        except Exception:\n"
        "            pass\n"
        "print('timeout')\n"
        "sys.exit(1)\n"
    )
    try:
        result = vm.ssh(
            "python3 -c " + shlex.quote(script),
            timeout=max(int(timeout_s + probe_timeout_s + 15), 20),
        )
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0


def _tail_vm_file(vm: TartVM, path: str, *, lines: int = OC_NATIVE_GATEWAY_LOG_TAIL_LINES) -> str:
    try:
        result = vm.ssh(
            f"tail -n {int(lines)} {shlex.quote(path)} 2>/dev/null || true",
            timeout=20,
        )
    except Exception:
        return ""
    return str(result.stdout or "").strip()


def _restart_oc_native_gateway(vm: TartVM, port: int = 18789):
    """Start the OpenClaw gateway and verify a live listener exists.

    `openclaw gateway start` can return success even when the launch agent is not
    actually loaded. For the native benchmark we probe the port and fall back to
    `gateway run` when the service path is a no-op.
    """
    _disable_openclaw_quaid_config_guard(vm)
    _patch_openclaw_native_memory(vm, enable_session_hook=True)
    _reapply_oc_native_gateway_runtime(vm)
    vm.ssh(
        "rm -f /tmp/openclaw-gateway-bench.log 2>/dev/null || true; "
        "pkill -f 'openclaw-gateway' 2>/dev/null || true; "
        "sleep 2; "
        "nohup env PATH=/opt/homebrew/bin:$PATH openclaw gateway start "
        "</dev/null >/tmp/openclaw-gateway-bench.log 2>&1 &",
        timeout=20,
    )
    if _wait_for_vm_tcp_port(
        vm,
        "127.0.0.1",
        port,
        timeout_s=OC_NATIVE_GATEWAY_START_WAIT_S,
        probe_timeout_s=3.0,
    ):
        print("  Gateway verified: responding")
        return
    vm.ssh(
        f"nohup env PATH=/opt/homebrew/bin:$PATH openclaw gateway run --force --port {port} "
        "</dev/null >/tmp/openclaw-gateway-bench.log 2>&1 &",
        timeout=20,
    )
    if not _wait_for_vm_tcp_port(
        vm,
        "127.0.0.1",
        port,
        timeout_s=OC_NATIVE_GATEWAY_RUN_WAIT_S,
        probe_timeout_s=3.0,
    ):
        gateway_tail = _tail_vm_file(vm, "/tmp/openclaw-gateway-bench.log")
        detail = f"; gateway log tail:\n{gateway_tail}" if gateway_tail else ""
        raise RuntimeError(f"oc-native gateway did not become reachable after restart{detail}")
    print("  Gateway verified: responding")


def _configure_openclaw_quaid_plugin(vm: TartVM) -> None:
    """Wire the benchmark VM's OpenClaw config to the synced Quaid extension."""
    _disable_openclaw_quaid_config_guard(vm)
    stage = vm.ssh(
        "test -d ~/clawd/plugins/quaid && "
        "mkdir -p ~/.openclaw/extensions && "
        "rm -rf ~/.openclaw/extensions/quaid && "
        "cp -R ~/clawd/plugins/quaid ~/.openclaw/extensions/quaid",
        raw=True,
        timeout=60,
    )
    if stage.returncode != 0:
        raise RuntimeError(
            "Failed to stage Quaid OpenClaw extension on benchmark VM: "
            f"{(stage.stderr or stage.stdout or '').strip()[:300]}"
        )
    script = (
        "import json, pathlib\n"
        f"workspace = pathlib.Path({VM_QUAID_HOME!r}).expanduser()\n"
        f"instance_id = {VM_QUAID_INSTANCE!r}\n"
        "cfg_path = pathlib.Path.home() / '.openclaw' / 'openclaw.json'\n"
        "parsed = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}\n"
        "plugins = parsed.setdefault('plugins', {})\n"
        "plugins['enabled'] = True\n"
        "allow = [str(entry or '').strip() for entry in plugins.get('allow', []) if str(entry or '').strip()]\n"
        "allow = [entry for entry in allow if entry not in ('active-memory', 'memory-core', 'memory-wiki')]\n"
        "for entry in ('quaid', 'telegram', 'openai', 'anthropic'):\n"
        "    if entry not in allow:\n"
        "        allow.append(entry)\n"
        "plugins['allow'] = allow\n"
        "entries = plugins.setdefault('entries', {})\n"
        "for entry in ('active-memory', 'memory-core', 'memory-wiki'):\n"
        "    entries.pop(entry, None)\n"
        "quaid = entries.get('quaid') or {}\n"
        "quaid['enabled'] = True\n"
        "quaid.pop('workspace', None)\n"
        "quaid.pop('hooks', None)\n"
        "entries['quaid'] = quaid\n"
        "plugins.setdefault('slots', {})['memory'] = 'quaid'\n"
        "env_block = parsed.setdefault('env', {})\n"
        "vars_block = env_block.setdefault('vars', {})\n"
        "for target in (env_block, vars_block):\n"
        "    target['QUAID_HOME'] = str(workspace)\n"
        "    target['QUAID_INSTANCE'] = instance_id\n"
        "    target['OPENCLAW_WORKSPACE'] = str(workspace)\n"
        "cfg_path.write_text(json.dumps(parsed, indent=2) + '\\n')\n"
        "print('Configured OpenClaw to load Quaid memory plugin')\n"
    )
    result = vm.ssh("python3 -c " + shlex.quote(script), timeout=20)
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to configure OpenClaw for Quaid benchmark lane: "
            f"{(result.stderr or result.stdout or '').strip()[:300]}"
        )
    if result.stdout.strip():
        print(f"  {result.stdout.strip()}")


def _restart_quaid_gateway(vm: TartVM, port: int = 18789) -> None:
    """Start the benchmark-owned OpenClaw gateway under the Quaid VM environment."""
    vm.ssh(
        f"rm -f /tmp/openclaw-gateway-bench.log 2>/dev/null || true; "
        f"pkill -f 'openclaw-gateway' 2>/dev/null || true; "
        f"sleep 2; "
        f"nohup env PATH=/opt/homebrew/bin:$PATH "
        f"QUAID_HOME={shlex.quote(VM_QUAID_HOME)} "
        f"QUAID_INSTANCE={shlex.quote(VM_QUAID_INSTANCE)} "
        f"openclaw gateway run --force --port {port} "
        "</dev/null >/tmp/openclaw-gateway-bench.log 2>&1 &",
        timeout=20,
    )
    if not _wait_for_vm_tcp_port(
        vm,
        "127.0.0.1",
        port,
        timeout_s=OC_NATIVE_GATEWAY_RUN_WAIT_S,
        probe_timeout_s=3.0,
    ):
        gateway_tail = _tail_vm_file(vm, "/tmp/openclaw-gateway-bench.log")
        detail = f"; gateway log tail:\n{gateway_tail}" if gateway_tail else ""
        raise RuntimeError(f"quaid gateway did not become reachable after restart{detail}")
    print("  Gateway verified: responding")


def _stop_quaid_instance_daemon(vm: TartVM) -> None:
    """Stop the background Quaid daemon for benchmark-driven extraction."""
    result = vm.ssh(
        f"QUAID_HOME={VM_QUAID_HOME} "
        f"QUAID_INSTANCE={VM_QUAID_INSTANCE} "
        "~/clawd/plugins/quaid/quaid daemon stop >/tmp/quaid-daemon-stop-bench.log 2>&1 "
        "|| true; "
        "for pid in $(pgrep -f extraction_daemon.py 2>/dev/null); do "
        f"if ps eww $pid 2>/dev/null | grep -q 'QUAID_INSTANCE={VM_QUAID_INSTANCE}'; then "
        "kill -9 $pid 2>/dev/null || true; fi; "
        "done",
        raw=True,
        timeout=20,
    )
    if result.returncode == 0:
        print("  Quaid daemon stopped for benchmark-driven extraction")


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
        vm.ssh(
            f"mkdir -p {shlex.quote(VM_QUAID_INSTANCE_ROOT_DIR)} && "
            f"echo '{escaped}' > {shlex.quote(VM_QUAID_INSTANCE_ROOT_DIR)}/{fname}",
            raw=True,
        )

    # Journal files (empty but with headers)
    journal_files = {
        "SOUL.journal.md": "# SOUL Journal\n",
        "USER.journal.md": "# USER Journal\n",
        "MEMORY.journal.md": "# MEMORY Journal\n",
    }
    vm.ssh(
        f"mkdir -p ~/clawd/journal ~/clawd/journal/archive "
        f"{shlex.quote(VM_QUAID_INSTANCE_JOURNAL_DIR)}",
        raw=True,
    )
    for fname, content in journal_files.items():
        escaped = content.replace("'", "'\\''")
        vm.ssh(f"echo '{escaped}' > ~/clawd/journal/{fname}", raw=True)
        vm.ssh(
            f"echo '{escaped}' > {shlex.quote(VM_QUAID_INSTANCE_JOURNAL_DIR)}/{fname}",
            raw=True,
        )

    print(f"  Core files created: SOUL.md, USER.md, MEMORY.md + journal/")


def _create_project_files(vm: TartVM, user_name: str = "Maya"):
    """Create benchmark project home dirs and seed minimal PROJECT.md files."""
    _ = user_name
    for name, spec in VM_BENCHMARK_PROJECTS.items():
        vm.ssh(f"mkdir -p {shlex.quote(spec['home_dir'])}", raw=True)
        vm.ssh(
            f"cat > {shlex.quote(spec['home_dir'])}/PROJECT.md << 'PROJEOF'\n"
            f"{spec['project_md']}PROJEOF",
            raw=True,
        )
    print("  Project files created: projects/recipe-app/PROJECT.md, projects/portfolio-site/PROJECT.md")


def _register_vm_benchmark_projects(vm: TartVM) -> None:
    """Register and link benchmark projects through Quaid's product project CLI."""
    projects_payload = [
        {
            "name": name,
            "description": spec["description"],
            "source_root": spec["source_root"],
        }
        for name, spec in VM_BENCHMARK_PROJECTS.items()
    ]
    command = (
        f"cd {VM_QUAID_DIR} && "
        f"QUAID_HOME={shlex.quote(VM_QUAID_HOME)} "
        f"QUAID_INSTANCE={shlex.quote(VM_QUAID_INSTANCE)} "
        "python3 - <<'PY'\n"
        "import subprocess, sys\n"
        f"projects = {json.dumps(projects_payload)}\n"
        "for project in projects:\n"
        "    name = str(project['name'])\n"
        "    description = str(project['description'])\n"
        "    source_root = str(project['source_root'])\n"
        "    create = subprocess.run(\n"
        "        ['./quaid', 'project', 'create', name, '--description', description, '--source-root', source_root],\n"
        "        capture_output=True,\n"
        "        text=True,\n"
        "    )\n"
        "    if create.returncode != 0:\n"
        "        detail = (create.stderr or create.stdout or '').strip()\n"
        "        if 'Project already exists' not in detail:\n"
        "            raise SystemExit(f'project create failed for {name}: {detail[:800]}')\n"
        "    for cmd in (\n"
        "        ['./quaid', 'project', 'link', name],\n"
        "        ['./quaid', 'project', 'update', name, '--description', description, '--source-root', source_root],\n"
        "    ):\n"
        "        result = subprocess.run(cmd, capture_output=True, text=True)\n"
        "        if result.returncode != 0:\n"
        "            detail = (result.stderr or result.stdout or '').strip()\n"
        "            raise SystemExit(f'project command failed for {name}: {cmd[2]} {detail[:800]}')\n"
        "print('registered benchmark projects via product CLI')\n"
        "PY"
    )
    result = vm.ssh(command, timeout=120, raw=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"VM benchmark project registration failed: {detail[:1200]}")
    if result.stdout.strip():
        print(f"  {result.stdout.strip()}")


def _scrape_janitor_errors(vm: TartVM) -> list[str]:
    """Check janitor logs for errors after a janitor run.

    Returns list of error lines found. Prints warnings for any errors.
    """
    result = vm.ssh(
        f"tr -d '\\000' < {shlex.quote(VM_QUAID_JANITOR_LATEST_LOG)} 2>/dev/null | "
        "grep -aiE '(error|exception|traceback|failed|FAIL)' | "
        "grep -viE '(edge_errors|Binary file|Errors encountered: 0|tests_failed: 0|errors: 0|_failed: 0|WORKSPACE AUDIT COMPLETE|\\[INFO\\]|^Decayed \\()' | "
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
        f"conn = sqlite3.connect({VM_QUAID_INSTANCE_DB_PATH!r}); "
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

    # Check core markdown files (instance silo first, then root fallback)
    for fname in ["SOUL.md", "USER.md", "MEMORY.md"]:
        result = vm.ssh(
            f"if [ -f {shlex.quote(VM_QUAID_INSTANCE_ROOT_DIR)}/{fname} ]; then "
            f"wc -l < {shlex.quote(VM_QUAID_INSTANCE_ROOT_DIR)}/{fname}; "
            f"else wc -l < ~/clawd/{fname} 2>/dev/null || echo 0; fi",
            raw=True,
        )
        lines = result.stdout.strip()
        if lines == "0":
            print(f"  [VALIDATE WARN] {fname} is empty or missing")
        validation["checks"][fname] = int(lines) if lines.isdigit() else 0

    # Check snippets (instance silo first, then root fallback)
    result = vm.ssh(
        f"count=$(ls {shlex.quote(VM_QUAID_INSTANCE_ROOT_DIR)}/*.snippets.md 2>/dev/null | wc -l); "
        "if [ \"$count\" = \"0\" ]; then count=$(ls ~/clawd/*.snippets.md 2>/dev/null | wc -l); fi; "
        "echo $count",
        raw=True,
    )
    snippet_count = result.stdout.strip()
    if snippet_count == "0":
        print(f"  [VALIDATE WARN] No snippet files found")
    validation["checks"]["snippets"] = int(snippet_count) if snippet_count.isdigit() else 0

    # Check journal (instance silo first, then root fallback)
    result = vm.ssh(
        f"count=$(ls {shlex.quote(VM_QUAID_INSTANCE_JOURNAL_DIR)}/*.journal.md 2>/dev/null | wc -l); "
        "if [ \"$count\" = \"0\" ]; then count=$(ls ~/clawd/journal/*.journal.md 2>/dev/null | wc -l); fi; "
        "echo $count",
        raw=True,
    )
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
    resume_state: Optional[dict] = None,
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
            resume_state=resume_state,
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
                        vm,
                        session_id,
                        system,
                        extract_model=extract_model,
                        session_file=_quaid_benchmark_session_file(session_id) if system == "quaid" else None,
                        sim_date=current_day,
                    )
                    e_usage = _require_quaid_extraction_usage(
                        e_usage,
                        context=f"per-day extract session={session_id} date={current_day}",
                    )
                    compaction_count += 1
                    tracker.add_compaction()
                    total_extraction_in += e_usage.get("input_tokens", 0)
                    total_extraction_out += e_usage.get("output_tokens", 0)
                    # Truncate session file after extraction (like /compact clears context)
                    session_file = (
                        _quaid_benchmark_session_file(session_id)
                        if system == "quaid"
                        else f"{VM_AGENT_SESSIONS_DIR}/{session_id}.jsonl"
                    )
                    vm.ssh(f": > {session_file}", timeout=5, raw=True)
                    session_tokens = 0
                # Run janitor at day boundaries — simulates nightly run
                j_usage = _run_vm_janitor(vm)
                janitor_runs += 1
                total_janitor_in += j_usage["input_tokens"]
                total_janitor_out += j_usage["output_tokens"]
                total_janitor_calls += j_usage["api_calls"]
                total_janitor_cost += j_usage["cost_usd"]
            if system == "oc-native" and sessions_injected > 0:
                _sync_openclaw_native_memory(
                    vm,
                    source_name="sessions",
                    min_indexed_files=sessions_injected,
                )

        # Nightly mode (A/B variant): force compact at day boundary
        # This is a PROPOSED FEATURE, not base system behavior.
        nightly_compacted = False
        if mode == "nightly" and day_changed and session_tokens > 0:
            print(f"  [NIGHTLY COMPACT — {session_tokens:,} tokens]", end="")
            e_usage = _trigger_compaction(
                vm,
                session_id,
                system,
                extract_model=extract_model,
                session_file=_quaid_benchmark_session_file(session_id) if system == "quaid" else None,
                sim_date=current_day,
            )
            if system == "quaid":
                e_usage = _require_quaid_extraction_usage(
                    e_usage,
                    context=f"nightly extract session={session_id} date={current_day}",
                )
            compaction_count += 1
            tracker.add_compaction()
            total_extraction_in += e_usage.get("input_tokens", 0)
            total_extraction_out += e_usage.get("output_tokens", 0)
            session_tokens = 0
            nightly_compacted = True

        current_day = session_day

        write_session_id = session_id
        append_mode = True
        if system == "oc-native":
            write_session_id = _oc_native_session_id(review, review_idx)
            append_mode = False
            jsonl = messages_to_oc_native_jsonl(messages, write_session_id)
        else:
            jsonl = messages_to_gateway_jsonl(messages)
        session_file = (
            _quaid_benchmark_session_file(write_session_id)
            if system == "quaid"
            else f"{VM_AGENT_SESSIONS_DIR}/{write_session_id}.jsonl"
        )

        # Append via SSH (use heredoc to avoid shell escaping issues)
        result = _write_vm_session_jsonl(
            vm,
            write_session_id,
            jsonl,
            append=append_mode,
            sessions_dir=VM_QUAID_BENCH_SESSIONS_DIR if system == "quaid" else VM_AGENT_SESSIONS_DIR,
        )
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
        e_usage = _trigger_compaction(
            vm,
            session_id,
            system,
            session_file=_quaid_benchmark_session_file(session_id) if system == "quaid" else None,
            sim_date=current_day,
        )
        if system == "quaid":
            e_usage = _require_quaid_extraction_usage(
                e_usage,
                context=f"final extract session={session_id} date={current_day}",
            )
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
        _sync_openclaw_native_memory(
            vm, source_name="sessions", min_indexed_files=sessions_injected
        )
        _sync_openclaw_native_wiki(vm)

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
    resume_state: Optional[dict] = None,
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
    resume_counters = dict((resume_state or {}).get("counters") or {})
    completed_chunks = max(0, int((resume_state or {}).get("completed_chunks", 0) or 0))
    current_day_raw = str((resume_state or {}).get("resume_current_day", "") or "").strip()
    current_day = current_day_raw or None
    compaction_count = int(resume_counters.get("compaction_count", 0) or 0)
    janitor_runs = int(resume_counters.get("janitor_runs", 0) or 0)
    message_idx = int(resume_counters.get("message_idx", 0) or 0)
    total_messages = int(resume_counters.get("total_messages", 0) or 0)
    # Real token accumulators
    total_extraction_in = int(resume_counters.get("total_extraction_in", 0) or 0)
    total_extraction_out = int(resume_counters.get("total_extraction_out", 0) or 0)
    total_janitor_in = int(resume_counters.get("total_janitor_in", 0) or 0)
    total_janitor_out = int(resume_counters.get("total_janitor_out", 0) or 0)
    total_janitor_calls = int(resume_counters.get("total_janitor_calls", 0) or 0)
    total_janitor_cost = float(resume_counters.get("total_janitor_cost", 0.0) or 0.0)
    # Simulated session token tracking
    session_tokens_curve: list = list(resume_counters.get("session_tokens_curve") or [])
    cumulative_no_compact = int(resume_counters.get("cumulative_no_compact", 0) or 0)
    total_context_burned = int(resume_counters.get("total_context_burned", 0) or 0)
    total_cached_tokens = int(resume_counters.get("total_cached_tokens", 0) or 0)
    total_fresh_tokens = int(resume_counters.get("total_fresh_tokens", 0) or 0)

    t0 = time.monotonic()
    session_file = f"{VM_AGENT_SESSIONS_DIR}/{session_id}.jsonl"
    if completed_chunks:
        print(
            "  Resuming timeout lifecycle checkpoint: "
            f"completed_chunks={completed_chunks}/{len(chunks)} "
            f"current_day={current_day or 'unknown'}"
        )
    if completed_chunks >= len(chunks):
        print("  Injection already completed at latest timeout lifecycle checkpoint")

    for ci, chunk in enumerate(chunks):
        if ci < completed_chunks:
            continue
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
            if results_dir is not None:
                checkpoint_counters = {
                    "compaction_count": compaction_count,
                    "janitor_runs": janitor_runs,
                    "message_idx": message_idx,
                    "total_messages": total_messages,
                    "total_extraction_in": total_extraction_in,
                    "total_extraction_out": total_extraction_out,
                    "total_janitor_in": total_janitor_in,
                    "total_janitor_out": total_janitor_out,
                    "total_janitor_calls": total_janitor_calls,
                    "total_janitor_cost": total_janitor_cost,
                    "session_tokens_curve": session_tokens_curve,
                    "cumulative_no_compact": cumulative_no_compact,
                    "total_context_burned": total_context_burned,
                    "total_cached_tokens": total_cached_tokens,
                    "total_fresh_tokens": total_fresh_tokens,
                }
                try:
                    _save_vm_quaid_timeout_resume_checkpoint(
                        vm,
                        results_dir,
                        completed_chunks=ci,
                        resume_current_day=chunk_day,
                        last_completed_day=current_day,
                        counters=checkpoint_counters,
                    )
                except Exception as exc:
                    print(f"  [RESUME CHECKPOINT WARN] save failed: {exc}")

        current_day = chunk_day

        write_session_id = session_id
        append_mode = True
        if system == "quaid":
            write_session_id = _quaid_chunk_session_id(session_id, chunk, ci)
            append_mode = False
            jsonl = messages_to_oc_native_jsonl(
                chunk.messages,
                write_session_id,
                started_at_ms=chunk.messages[0].timestamp_ms if chunk.messages else None,
                model_id=extract_model,
                provider="anthropic",
            )
        else:
            # Convert chunk messages to gateway JSONL
            gateway_messages = [{"role": m.role, "content": m.content} for m in chunk.messages]
            jsonl = messages_to_gateway_jsonl(gateway_messages)
        session_file = (
            _quaid_benchmark_session_file(write_session_id)
            if system == "quaid"
            else f"{VM_AGENT_SESSIONS_DIR}/{write_session_id}.jsonl"
        )

        # Write chunk session file
        result = _write_vm_session_jsonl(
            vm,
            write_session_id,
            jsonl,
            append=append_mode,
            sessions_dir=VM_QUAID_BENCH_SESSIONS_DIR if system == "quaid" else VM_AGENT_SESSIONS_DIR,
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
            vm,
            write_session_id,
            system,
            extract_model=extract_model,
            session_file=session_file if system == "quaid" else None,
            sim_date=chunk_day,
        )
        e_usage = _require_quaid_extraction_usage(
            e_usage,
            context=f"timeout chunk={ci+1}/{len(chunks)} session={write_session_id} date={chunk_day}",
        )
        extraction_ok = e_usage.get("input_tokens", 0) > 0
        if extraction_ok:
            compaction_count += 1
            tracker.add_compaction()
            total_extraction_in += e_usage.get("input_tokens", 0)
            total_extraction_out += e_usage.get("output_tokens", 0)
            _write_local_extraction_artifact(results_dir, write_session_id, e_usage.get("artifact"))

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
    if system == "quaid" and compaction_count > 0 and completed_chunks < len(chunks):
        print(f"  [FINAL JANITOR]", end="")
        j_usage = _run_vm_janitor(vm)
        janitor_runs += 1
        total_janitor_in += j_usage["input_tokens"]
        total_janitor_out += j_usage["output_tokens"]
        total_janitor_calls += j_usage["api_calls"]
        total_janitor_cost += j_usage["cost_usd"]
        if results_dir is not None:
            checkpoint_counters = {
                "compaction_count": compaction_count,
                "janitor_runs": janitor_runs,
                "message_idx": message_idx,
                "total_messages": total_messages,
                "total_extraction_in": total_extraction_in,
                "total_extraction_out": total_extraction_out,
                "total_janitor_in": total_janitor_in,
                "total_janitor_out": total_janitor_out,
                "total_janitor_calls": total_janitor_calls,
                "total_janitor_cost": total_janitor_cost,
                "session_tokens_curve": session_tokens_curve,
                "cumulative_no_compact": cumulative_no_compact,
                "total_context_burned": total_context_burned,
                "total_cached_tokens": total_cached_tokens,
                "total_fresh_tokens": total_fresh_tokens,
            }
            try:
                _save_vm_quaid_timeout_resume_checkpoint(
                    vm,
                    results_dir,
                    completed_chunks=len(chunks),
                    resume_current_day="",
                    last_completed_day=current_day,
                    counters=checkpoint_counters,
                )
            except Exception as exc:
                print(f"  [RESUME CHECKPOINT WARN] final save failed: {exc}")
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
                        session_file: str | None = None,
                        sim_date: str | None = None) -> dict:
    """Trigger compaction on the VM. Returns extraction token usage.

    For Quaid: calls extract_compact.py directly (bypasses gateway /compact
    which only works through auto-reply pipeline, not agent CLI).

    For other systems: sends /compact via openclaw agent.
    """
    session_file = session_file or f"{VM_AGENT_SESSIONS_DIR}/{session_id}.jsonl"
    extraction_usage = {"input_tokens": 0, "output_tokens": 0, "model": extract_model}

    if system == "quaid":
        date_flag = f" --date {sim_date}" if sim_date else ""
        result = vm.ssh(
            f"QUAID_HOME={VM_QUAID_HOME} "
            f"QUAID_INSTANCE={VM_QUAID_INSTANCE} "
            f"QUAID_LLM_USAGE_LOG_PATH={shlex.quote(VM_QUAID_LLM_USAGE_LOG_PATH)} "
            f"QUAID_LLM_USAGE_PHASE=ingest "
            f"QUAID_LLM_USAGE_SOURCE=benchmark_extract "
            f"BENCHMARK_EXTRACTION_PROMPT_FILE=~/clawd/plugins/quaid/prompts/extraction.txt "
            f"BENCHMARK_EXTRACTION_INCLUDE_ARTIFACT=1 "
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
        compact_error = None
        if result.returncode != 0:
            print(f" [COMPACT FAILED: {result.stderr[:200]}]", end="")
            detail = (result.stderr or result.stdout).strip()
            compact_error = detail or "no stderr/stdout from extract_compact.py"
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
                        if "artifact" in data:
                            extraction_usage["artifact"] = data["artifact"]
                    except json.JSONDecodeError:
                        pass
        # Log stderr (edge errors, store errors, etc.)
        if result.stderr:
            for line in result.stderr.strip().split("\n"):
                if line and not line.startswith("[config]"):
                    print(f"\n  [extract stderr] {line}", end="")
        if compact_error is not None:
            raise RuntimeError(
                f"Quaid compaction failed for session={session_id} date={sim_date or 'unknown'}: "
                f"{compact_error[:300]}"
            )
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
        f"cd {VM_QUAID_DIR} && "
        f"mkdir -p {shlex.quote(VM_QUAID_INSTANCE_LOGS_DIR)} && "
        f"QUAID_HOME={shlex.quote(VM_QUAID_HOME)} "
        f"QUAID_INSTANCE={shlex.quote(VM_QUAID_INSTANCE)} "
        f"QUAID_LLM_USAGE_LOG_PATH={shlex.quote(VM_QUAID_LLM_USAGE_LOG_PATH)} "
        f"QUAID_LLM_USAGE_PHASE=janitor "
        f"QUAID_LLM_USAGE_SOURCE=benchmark_janitor "
        f"QUAID_BENCHMARK_MODE=1 "
        f"./quaid janitor --task all --apply --approve --no-resume-checkpoint "
        f"> {shlex.quote(VM_QUAID_JANITOR_LATEST_LOG)} 2>&1; "
        f"janitor_rc=$?; tail -5 {shlex.quote(VM_QUAID_JANITOR_LATEST_LOG)} 2>/dev/null || true; exit $janitor_rc",
        timeout=600,
    )
    if result.returncode != 0:
        print(f" [JANITOR FAILED rc={result.returncode}]", end="")
        if result.stdout:
            for line in result.stdout.strip().split("\n")[-3:]:
                print(f"\n    {line}", end="")
        raise RuntimeError(
            f"Quaid janitor failed rc={result.returncode}: "
            f"{(result.stdout or result.stderr or '').strip()[:300]}"
        )
    # Quick error scrape
    errors = _scrape_janitor_errors(vm)
    if errors:
        print(f" [{len(errors)} errors]", end="")
    status_result = vm.ssh(
        "python3 -c "
        + shlex.quote(
            "import json, sqlite3; "
            f"conn=sqlite3.connect({VM_QUAID_INSTANCE_DB_PATH!r}); "
            "rows=conn.execute(\"SELECT status, COUNT(*) FROM nodes GROUP BY status\").fetchall(); "
            "print(json.dumps({str(k): int(v) for k, v in rows}))"
        ),
        timeout=10,
    )
    if status_result.returncode == 0 and status_result.stdout.strip():
        try:
            status_counts = json.loads(status_result.stdout.strip())
        except json.JSONDecodeError:
            status_counts = None
        if isinstance(status_counts, dict):
            pending = int(status_counts.get("pending", 0) or 0)
            approved = int(status_counts.get("approved", 0) or 0)
            if pending > 0 or approved > 0:
                raise RuntimeError(
                    "Quaid janitor left unresolved node statuses: "
                    f"pending={pending}, approved={approved}, counts={status_counts}"
                )
    # Scrape real token usage from janitor-stats.json
    usage = {"input_tokens": 0, "output_tokens": 0, "api_calls": 0, "cost_usd": 0.0}
    stats_result = vm.ssh(f"cat {shlex.quote(VM_QUAID_JANITOR_STATS_PATH)} 2>/dev/null", timeout=10)
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
    db_result = vm.scp_from(VM_QUAID_INSTANCE_DB_PATH, str(results_dir / "memory.db"))
    if db_result.returncode == 0:
        print(f"    memory.db: OK")
    else:
        print(f"    memory.db: FAILED ({db_result.stderr[:80]})")

    # Copy workspace files (core markdown + journal + snippets)
    ws_dir = results_dir / "workspace"
    ws_dir.mkdir(exist_ok=True)

    for fname in ["SOUL.md", "USER.md", "MEMORY.md", "IDENTITY.md"]:
        r = vm.scp_from(f"{VM_QUAID_INSTANCE_ROOT_DIR}/{fname}", str(ws_dir / fname))
        if r.returncode != 0:
            r = vm.scp_from(f"~/clawd/{fname}", str(ws_dir / fname))
        print(f"    {fname}: {'OK' if r.returncode == 0 else 'not found'}")

    # Copy journal directory
    journal_dir = ws_dir / "journal"
    journal_dir.mkdir(exist_ok=True)
    # Use tar to grab the whole directory
    vm.ssh(
        f"if [ -d {shlex.quote(VM_QUAID_INSTANCE_JOURNAL_DIR)} ]; then "
        f"cd {shlex.quote(VM_QUAID_INSTANCE_ROOT_DIR)} && tar czf /tmp/journal-golden.tar.gz journal/ 2>/dev/null || true; "
        "else cd ~/clawd && tar czf /tmp/journal-golden.tar.gz journal/ 2>/dev/null || true; fi"
    )
    tar_result = vm.scp_from("/tmp/journal-golden.tar.gz", str(ws_dir / "journal-golden.tar.gz"))
    if tar_result.returncode == 0:
        subprocess.run(
            ["tar", "xzf", str(ws_dir / "journal-golden.tar.gz"), "-C", str(ws_dir)],
            capture_output=True, timeout=30,
        )
        (ws_dir / "journal-golden.tar.gz").unlink(missing_ok=True)
        print(f"    journal/: OK")

    # Copy snippets files
    vm.ssh(
        f"cd {shlex.quote(VM_QUAID_INSTANCE_ROOT_DIR)} && "
        "tar czf /tmp/snippets-golden.tar.gz *.snippets.md 2>/dev/null || "
        "cd ~/clawd && tar czf /tmp/snippets-golden.tar.gz *.snippets.md 2>/dev/null || true"
    )
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


def _vm_lifecycle_resume_root(results_dir: Path) -> Path:
    return results_dir / VM_LIFECYCLE_RESUME_ROOT


def _vm_lifecycle_resume_state_path(results_dir: Path) -> Path:
    return _vm_lifecycle_resume_root(results_dir) / VM_LIFECYCLE_RESUME_STATE


def _vm_lifecycle_resume_archive_path(results_dir: Path) -> Path:
    return _vm_lifecycle_resume_root(results_dir) / VM_LIFECYCLE_RESUME_ARCHIVE


def _load_vm_lifecycle_resume_state(results_dir: Optional[Path]) -> Optional[dict]:
    if results_dir is None:
        return None
    path = _vm_lifecycle_resume_state_path(results_dir)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    archive = _vm_lifecycle_resume_archive_path(results_dir)
    if not archive.exists():
        return None
    payload["archive_path"] = str(archive)
    return payload


def _save_vm_quaid_timeout_resume_checkpoint(
    vm: TartVM,
    results_dir: Path,
    *,
    completed_chunks: int,
    resume_current_day: Optional[str],
    last_completed_day: Optional[str],
    counters: dict,
) -> dict:
    """Persist a latest-only OC Quaid timeout resume checkpoint.

    This is intentionally narrow: it captures the guest state only after a
    successful Quaid day-boundary janitor or the final janitor. That keeps the
    resume boundary semantically clean and avoids resuming from partially
    extracted/pending guest state.
    """
    root = _vm_lifecycle_resume_root(results_dir)
    root.mkdir(parents=True, exist_ok=True)
    archive_path = _vm_lifecycle_resume_archive_path(results_dir)
    state_path = _vm_lifecycle_resume_state_path(results_dir)
    remote_archive = "/tmp/vm-benchmark-lifecycle-resume.tar.gz"
    build_script = (
        "python3 - <<'PY'\n"
        "import tarfile\n"
        "from pathlib import Path\n"
        f"root = Path({str(Path('/Users/admin'))!r})\n"
        f"archive = Path({remote_archive!r})\n"
        "archive.unlink(missing_ok=True)\n"
        "targets = [\n"
        f"    root / {str(Path('clawd/instances/openclaw-main'))!r},\n"
        f"    root / {str(Path('clawd/projects'))!r},\n"
        f"    root / {str(Path('.openclaw/agents/main/sessions/benchmark'))!r},\n"
        "]\n"
        "with tarfile.open(archive, 'w:gz') as bundle:\n"
        "    for target in targets:\n"
        "        if target.exists():\n"
        "            bundle.add(target, arcname=str(target.relative_to(root)))\n"
        "    clawd_root = root / 'clawd'\n"
        "    if clawd_root.exists():\n"
        "        patterns = ('*.md', '*.snippets.md', '*.journal.md')\n"
        "        seen = set()\n"
        "        for pattern in patterns:\n"
        "            for target in sorted(clawd_root.glob(pattern)):\n"
        "                if not target.is_file():\n"
        "                    continue\n"
        "                rel = target.relative_to(root)\n"
        "                if str(rel) in seen:\n"
        "                    continue\n"
        "                seen.add(str(rel))\n"
        "                bundle.add(target, arcname=str(rel))\n"
        "print(archive)\n"
        "PY"
    )
    result = vm.ssh(build_script, timeout=120, raw=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to build VM lifecycle checkpoint: {result.stderr[:200]}")
    fetched = vm.scp_from(remote_archive, str(archive_path), timeout=120)
    if fetched.returncode != 0:
        raise RuntimeError(f"Failed to fetch VM lifecycle checkpoint: {fetched.stderr[:200]}")
    payload = {
        "mode": "quaid-timeout",
        "completed_chunks": int(completed_chunks),
        "resume_current_day": str(resume_current_day or ""),
        "last_completed_day": str(last_completed_day or ""),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "archive_path": str(archive_path),
        "counters": dict(counters or {}),
    }
    state_path.write_text(json.dumps(payload, indent=2))
    return payload


def _restore_vm_quaid_timeout_resume_checkpoint(vm: TartVM, results_dir: Path) -> Optional[dict]:
    payload = _load_vm_lifecycle_resume_state(results_dir)
    if not payload:
        return None
    archive_path = Path(str(payload.get("archive_path", "") or ""))
    if not archive_path.exists():
        return None
    remote_archive = "/tmp/vm-benchmark-lifecycle-resume.tar.gz"
    copied = vm.scp_to(str(archive_path), remote_archive, timeout=120)
    if copied.returncode != 0:
        raise RuntimeError(f"Failed to upload VM lifecycle checkpoint: {copied.stderr[:200]}")
    restore_script = (
        "python3 - <<'PY'\n"
        "import tarfile, shutil\n"
        "from pathlib import Path\n"
        f"root = Path({str(Path('/Users/admin'))!r})\n"
        f"archive = Path({remote_archive!r})\n"
        "targets = [\n"
        f"    root / {str(Path('clawd/instances/openclaw-main'))!r},\n"
        f"    root / {str(Path('clawd/projects'))!r},\n"
        f"    root / {str(Path('.openclaw/agents/main/sessions/benchmark'))!r},\n"
        "]\n"
        "for target in targets:\n"
        "    if target.exists():\n"
        "        if target.is_dir():\n"
        "            shutil.rmtree(target)\n"
        "        else:\n"
        "            target.unlink()\n"
        "clawd_root = root / 'clawd'\n"
        "if clawd_root.exists():\n"
        "    patterns = ('*.md', '*.snippets.md', '*.journal.md')\n"
        "    for pattern in patterns:\n"
        "        for target in clawd_root.glob(pattern):\n"
        "            if target.is_file():\n"
        "                target.unlink()\n"
        "with tarfile.open(archive, 'r:gz') as bundle:\n"
        "    bundle.extractall(root)\n"
        "print('restored')\n"
        "PY"
    )
    restored = vm.ssh(restore_script, timeout=180, raw=True)
    if restored.returncode != 0:
        raise RuntimeError(f"Failed to restore VM lifecycle checkpoint: {restored.stderr[:200]}")
    return payload


def _patch_gateway_model(vm: TartVM, answer_model: str):
    """Set the gateway's agent model on the VM."""
    full_model = answer_model if "/" in answer_model else f"anthropic/{answer_model}"
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


def _resolve_gateway_answer_model(
    answer_model: str,
    *,
    system: str,
    openai_auth_mode: str,
) -> str:
    """Normalize gateway model id for provider-specific auth transports."""
    model = str(answer_model or "").strip()
    if not model:
        return model
    if openai_auth_mode == "codex-oauth" and model.startswith("openai/"):
        return "openai-codex/" + model.split("/", 1)[1]
    return model


def _set_oc_native_gateway_runtime_context(answer_model: Optional[str], openai_auth_mode: str = "api"):
    """Record the benchmark-owned OC gateway runtime state for later restarts."""
    global _OC_NATIVE_GATEWAY_ANSWER_MODEL, _OC_NATIVE_GATEWAY_OPENAI_AUTH_MODE
    model = str(answer_model or "").strip()
    _OC_NATIVE_GATEWAY_ANSWER_MODEL = model or None
    _OC_NATIVE_GATEWAY_OPENAI_AUTH_MODE = str(openai_auth_mode or "api").strip() or "api"


def _resolve_openai_api_key_for_vm() -> str:
    """Resolve the OpenAI API key used by direct OpenClaw OpenAI VM runs."""
    direct = os.environ.get("OPENAI_API_KEY", "").strip()
    if direct:
        return direct
    try:
        from run_production_benchmark import _get_openai_key
    except Exception:
        return ""
    try:
        return (_get_openai_key() or "").strip()
    except Exception:
        return ""


def _resolve_anthropic_credential_for_vm() -> str:
    """Resolve the primary Anthropic benchmark credential for guest extraction."""
    for env_name in ("BENCHMARK_ANTHROPIC_OAUTH_TOKEN", "ANTHROPIC_API_KEY"):
        direct = os.environ.get(env_name, "").strip()
        if direct:
            return direct

    dev_cfg = Path.home() / "quaidcode" / "dev" / ".quaid-dev.local.json"
    if not dev_cfg.exists():
        return ""
    try:
        cfg = json.loads(dev_cfg.read_text(encoding="utf-8"))
    except Exception:
        return ""

    anth_cfg = (((cfg or {}).get("auth") or {}).get("anthropic") or {})
    rel = str(anth_cfg.get("firstKeyPath") or "").strip()
    if not rel:
        return ""
    rel_path = Path(rel).expanduser()
    base_root = Path.home() / "quaidcode"
    if rel_path.is_absolute():
        candidates = [rel_path]
    else:
        candidates = [(dev_cfg.parent / rel_path), (base_root / rel_path)]
    token_path = next((candidate.resolve() for candidate in candidates if candidate.exists()), None)
    if token_path is None or not token_path.exists():
        return ""
    try:
        return token_path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _validate_anthropic_credential_for_vm(extract_model: str) -> None:
    """Fail fast if the configured Anthropic guest credential cannot access the deep model."""
    credential = _resolve_anthropic_credential_for_vm()
    if not credential:
        raise RuntimeError(
            "Anthropic benchmark credential is required to validate Quaid VM extraction"
        )

    model = _normalize_extract_model(extract_model)
    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    betas = ["prompt-caching-2024-07-31"]
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": 8,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Reply with OK."}]}],
    }
    if credential.startswith("sk-ant-oat"):
        headers["Authorization"] = f"Bearer {credential}"
        headers["Accept"] = "application/json"
        headers["user-agent"] = "claude-cli/2.1.2 (external, cli)"
        headers["x-app"] = "cli"
        betas.extend(["claude-code-20250219", "oauth-2025-04-20"])
        payload["system"] = [
            {
                "type": "text",
                "text": "You are Claude Code, Anthropic's official CLI for Claude.",
                "cache_control": {"type": "ephemeral"},
            }
        ]
    else:
        headers["x-api-key"] = credential
    headers["anthropic-beta"] = ",".join(betas)

    req = Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(req, timeout=45) as resp:
            resp.read()
    except HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", "replace").strip()
        except Exception:
            body = ""
        detail = body or str(exc)
        raise RuntimeError(
            f"Anthropic benchmark credential cannot access model {model}: HTTP {exc.code} {detail}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"Anthropic benchmark credential probe failed for model {model}: {exc}"
        ) from exc


def _provision_openclaw_anthropic_key(vm: TartVM):
    """Install the primary Anthropic benchmark credential into guest runtime auth stores."""
    credential = _resolve_anthropic_credential_for_vm()
    if not credential:
        raise RuntimeError(
            "Anthropic benchmark credential is required to run Quaid extraction on the VM"
        )
    script = (
        "import json, os, pathlib, sys\n"
        "token = sys.stdin.read().strip()\n"
        "if not token:\n"
        "    raise SystemExit('missing Anthropic credential')\n"
        "env_path = pathlib.Path.home() / '.openclaw' / '.env'\n"
        "env_path.parent.mkdir(parents=True, exist_ok=True)\n"
        "env_lines = []\n"
        "if env_path.exists():\n"
        "    env_lines = [line for line in env_path.read_text().splitlines() if not line.startswith('ANTHROPIC_API_KEY=')]\n"
        "env_lines.append(f'ANTHROPIC_API_KEY={token}')\n"
        "tmp_env = env_path.with_suffix('.env.tmp')\n"
        "tmp_env.write_text('\\n'.join(env_lines) + '\\n')\n"
        "os.chmod(tmp_env, 0o600)\n"
        "tmp_env.replace(env_path)\n"
        f"runtime_quaid_home = pathlib.Path({VM_QUAID_HOME!r}).expanduser()\n"
        "paths = [\n"
        "    pathlib.Path.home() / '.openclaw' / 'shared' / 'auth' / 'credentials.json',\n"
        "    runtime_quaid_home / 'shared' / 'auth' / 'credentials.json',\n"
        "    pathlib.Path.home() / '.quaid' / 'shared' / 'auth' / 'credentials.json',\n"
        "]\n"
        "for p in paths:\n"
        "    p.parent.mkdir(parents=True, exist_ok=True)\n"
        "    try:\n"
        "        data = json.loads(p.read_text() or '{}') if p.exists() else {}\n"
        "    except Exception:\n"
        "        data = {}\n"
        "    creds = data.setdefault('credentials', {})\n"
        "    creds['anthropic_oauth'] = {'token': token}\n"
        "    tmp = p.with_suffix(p.suffix + '.tmp')\n"
        "    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\\n')\n"
        "    os.chmod(tmp, 0o600)\n"
        "    tmp.replace(p)\n"
        "print('OpenClaw Anthropic runtime credential installed (.env + shared auth)')\n"
    )
    result = vm.ssh("python3 -c " + shlex.quote(script), input_data=credential, timeout=10)
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to provision Anthropic credential on benchmark VM: "
            f"{(result.stderr or result.stdout or '').strip()}"
        )
    if result.stdout.strip():
        print(f"  {result.stdout.strip()}")


def _resolve_codex_oauth_profile_for_vm() -> dict:
    """Resolve a Codex OAuth profile payload used by OpenClaw on the benchmark VM."""
    direct_path = os.environ.get("BENCHMARK_CODEX_TOKEN_PATH", "").strip()
    if direct_path:
        try:
            raw = Path(direct_path).expanduser().resolve().read_text(encoding="utf-8").strip()
        except Exception:
            raw = ""
        if raw:
            try:
                data = json.loads(raw)
            except Exception:
                return {
                    "type": "oauth",
                    "provider": "openai-codex",
                    "access": raw,
                }
            if isinstance(data, dict):
                access = str(data.get("access") or data.get("token") or "").strip()
                if not access:
                    tokens = data.get("tokens")
                    if isinstance(tokens, dict):
                        access = str(tokens.get("access_token") or "").strip()
                        if access:
                            profile = {
                                "type": "oauth",
                                "provider": "openai-codex",
                                "access": access,
                            }
                            refresh = str(tokens.get("refresh_token") or "").strip()
                            if refresh:
                                profile["refresh"] = refresh
                            account_id = str(tokens.get("account_id") or "").strip()
                            if account_id:
                                profile["accountId"] = account_id
                            return profile
                if access:
                    profile = dict(data)
                    profile.setdefault("type", "oauth")
                    profile.setdefault("provider", "openai-codex")
                    return profile

    direct = os.environ.get("BENCHMARK_CODEX_API_KEY", "").strip()
    if direct:
        return {
            "type": "oauth",
            "provider": "openai-codex",
            "access": direct,
        }
    codex_cli_auth = Path.home() / ".codex" / "auth.json"
    if codex_cli_auth.exists():
        try:
            data = json.loads(codex_cli_auth.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        tokens = (data or {}).get("tokens")
        if isinstance(tokens, dict):
            access = str(tokens.get("access_token") or "").strip()
            if access:
                profile = {
                    "type": "oauth",
                    "provider": "openai-codex",
                    "access": access,
                }
                refresh = str(tokens.get("refresh_token") or "").strip()
                if refresh:
                    profile["refresh"] = refresh
                account_id = str(tokens.get("account_id") or "").strip()
                if account_id:
                    profile["accountId"] = account_id
                return profile
    try:
        from run_production_benchmark import _read_codex_auth_token
    except Exception:
        token = ""
    else:
        try:
            token = (_read_codex_auth_token() or "").strip()
        except Exception:
            token = ""
    if token:
        return {
            "type": "oauth",
            "provider": "openai-codex",
            "access": token,
        }

    dev_cfg = Path.home() / "quaidcode" / "dev" / ".quaid-dev.local.json"
    if not dev_cfg.exists():
        return {}
    try:
        cfg = json.loads(dev_cfg.read_text(encoding="utf-8"))
    except Exception:
        return {}
    codex_cfg = (((cfg or {}).get("auth") or {}).get("codex") or {})
    base_root = Path.home() / "quaidcode"
    for key_name in ("solKeyPath", "yuniKeyPath"):
        rel = str(codex_cfg.get(key_name) or "").strip()
        if not rel:
            continue
        rel_path = Path(rel).expanduser()
        candidates: List[Path]
        if rel_path.is_absolute():
            candidates = [rel_path]
        else:
            # Prefer paths relative to the config file itself; keep legacy
            # resolution under ~/quaidcode for compatibility.
            candidates = [(dev_cfg.parent / rel_path), (base_root / rel_path)]
        token_path = next((candidate.resolve() for candidate in candidates if candidate.exists()), None)
        if token_path is None or not token_path.exists():
            continue
        try:
            raw = token_path.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            return {
                "type": "oauth",
                "provider": "openai-codex",
                "access": raw,
            }
        if not isinstance(data, dict):
            continue
        access = str(data.get("access") or data.get("token") or "").strip()
        if not access:
            continue
        profile = dict(data)
        profile.setdefault("type", "oauth")
        profile.setdefault("provider", "openai-codex")
        return profile
    return {}


def _decode_jwt_claims_unverified(token: str) -> dict:
    """Best-effort JWT payload decode without signature verification."""
    parts = str(token).split(".")
    if len(parts) < 2:
        return {}
    payload = parts[1]
    payload += "=" * (-len(payload) % 4)
    try:
        raw = base64.urlsafe_b64decode(payload.encode("ascii"))
        data = json.loads(raw.decode("utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _normalize_codex_oauth_profile_for_openclaw(profile: dict) -> dict:
    """Shape a Codex OAuth payload to match OpenClaw's auth-profiles schema."""
    access = str((profile or {}).get("access") or (profile or {}).get("token") or "").strip()
    if not access:
        return {}

    normalized = {
        "type": "oauth",
        "provider": "openai-codex",
        "access": access,
    }
    refresh = str((profile or {}).get("refresh") or "").strip()
    if refresh:
        normalized["refresh"] = refresh
    account_id = str((profile or {}).get("accountId") or "").strip()
    expires = (profile or {}).get("expires")

    claims = _decode_jwt_claims_unverified(access)
    if not account_id:
        auth_claims = claims.get("https://api.openai.com/auth")
        if isinstance(auth_claims, dict):
            account_id = str(auth_claims.get("chatgpt_account_id") or "").strip()
    if account_id:
        normalized["accountId"] = account_id

    if not isinstance(expires, int):
        exp = claims.get("exp")
        if isinstance(exp, int):
            expires = exp * 1000
    if isinstance(expires, int) and expires > 0:
        normalized["expires"] = expires

    return normalized


def _provision_openclaw_openai_key(vm: TartVM):
    """Install direct OpenAI API auth for OpenClaw on the benchmark VM."""
    key = _resolve_openai_api_key_for_vm()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is required to run oc-native with direct OpenAI models"
        )
    script = (
        "import json, os, pathlib, sys\n"
        "token = sys.stdin.read().strip()\n"
        "if not token:\n"
        "    raise SystemExit('missing OpenAI token')\n"
        "openclaw_root = pathlib.Path.home() / '.openclaw'\n"
        "p = openclaw_root / 'agents' / 'main' / 'agent' / 'auth-profiles.json'\n"
        "p.parent.mkdir(parents=True, exist_ok=True)\n"
        "try:\n"
        "    data = json.loads(p.read_text() or '{}') if p.exists() else {}\n"
        "except Exception:\n"
        "    data = {}\n"
        "data.setdefault('version', 1)\n"
        "profiles = data.setdefault('profiles', {})\n"
        "profile = profiles.get('openai:default') if isinstance(profiles.get('openai:default'), dict) else {}\n"
        "profile['type'] = 'api_key'\n"
        "profile['provider'] = 'openai'\n"
        "profile['key'] = token\n"
        "profile.pop('token', None)\n"
        "profile.pop('access', None)\n"
        "profiles['openai:default'] = profile\n"
        "last_good = data.setdefault('lastGood', {})\n"
        "last_good['openai'] = 'openai:default'\n"
        "tmp = p.with_suffix(p.suffix + '.tmp')\n"
        "tmp.write_text(json.dumps(data, indent=2) + '\\n')\n"
        "os.chmod(tmp, 0o600)\n"
        "tmp.replace(p)\n"
        "env_path = openclaw_root / '.env'\n"
        "env_lines = []\n"
        "if env_path.exists():\n"
        "    env_lines = [line for line in env_path.read_text().splitlines() if not line.startswith('OPENAI_API_KEY=')]\n"
        "env_lines.append(f'OPENAI_API_KEY={token}')\n"
        "tmp_env = env_path.with_suffix('.env.tmp')\n"
        "tmp_env.write_text('\\n'.join(env_lines) + '\\n')\n"
        "os.chmod(tmp_env, 0o600)\n"
        "tmp_env.replace(env_path)\n"
        "print('OpenClaw direct OpenAI auth installed (.env + auth profile)')\n"
    )
    result = vm.ssh("python3 -c " + shlex.quote(script), input_data=key, timeout=10)
    if result.returncode != 0:
        raise RuntimeError(f"OpenClaw OpenAI auth provisioning failed: {result.stderr[:200]}")
    if result.stdout.strip():
        print(f"  {result.stdout.strip()}")


def _provision_openclaw_codex_oauth(vm: TartVM):
    """Install Codex OAuth shared credentials for OpenClaw on the benchmark VM."""
    raw_profile = _resolve_codex_oauth_profile_for_vm()
    profile = _normalize_codex_oauth_profile_for_openclaw(raw_profile)
    access = str(profile.get("access") or "").strip()
    if not access:
        raise RuntimeError(
            "Codex OAuth profile is required to run oc-native with Codex OAuth transport"
        )
    script = (
        "import json, os, pathlib, sys\n"
        "payload = json.loads(sys.stdin.read())\n"
        "token = str(payload.get('access') or '').strip()\n"
        "if not token:\n"
        "    raise SystemExit('missing Codex OAuth token')\n"
        "auth_profiles = pathlib.Path.home() / '.openclaw' / 'agents' / 'main' / 'agent' / 'auth-profiles.json'\n"
        "auth_profiles.parent.mkdir(parents=True, exist_ok=True)\n"
        "try:\n"
        "    auth_data = json.loads(auth_profiles.read_text() or '{}') if auth_profiles.exists() else {}\n"
        "except Exception:\n"
        "    auth_data = {}\n"
        "auth_data.setdefault('version', 1)\n"
        "profiles = auth_data.setdefault('profiles', {})\n"
        "for profile_key, existing_profile in list(profiles.items()):\n"
        "    if profile_key == 'openai-codex:default':\n"
        "        continue\n"
        "    provider = ''\n"
        "    if isinstance(existing_profile, dict):\n"
        "        provider = str(existing_profile.get('provider') or '').strip()\n"
        "    if (\n"
        "        profile_key == 'openai-codex'\n"
        "        or str(profile_key).startswith('openai-codex:')\n"
        "        or str(profile_key).startswith('openai:')\n"
        "        or provider == 'openai'\n"
        "    ):\n"
        "        profiles.pop(profile_key, None)\n"
        "profile = profiles.get('openai-codex:default') if isinstance(profiles.get('openai-codex:default'), dict) else {}\n"
        "profile['type'] = 'oauth'\n"
        "profile['provider'] = 'openai-codex'\n"
        "profile['access'] = token\n"
        "refresh = str(payload.get('refresh') or '').strip()\n"
        "if refresh:\n"
        "    profile['refresh'] = refresh\n"
        "else:\n"
        "    profile.pop('refresh', None)\n"
        "expires = payload.get('expires')\n"
        "if isinstance(expires, int) and expires > 0:\n"
        "    profile['expires'] = expires\n"
        "else:\n"
        "    profile.pop('expires', None)\n"
        "account_id = str(payload.get('accountId') or '').strip()\n"
        "if account_id:\n"
        "    profile['accountId'] = account_id\n"
        "else:\n"
        "    profile.pop('accountId', None)\n"
        "for legacy_key in ('mode', 'token', 'key', 'access_token', 'accessToken'):\n"
        "    profile.pop(legacy_key, None)\n"
        "profiles['openai-codex:default'] = profile\n"
        "last_good = auth_data.setdefault('lastGood', {})\n"
        "last_good.pop('openai', None)\n"
        "last_good['openai-codex'] = 'openai-codex:default'\n"
        "order = auth_data.setdefault('order', {})\n"
        "order.pop('openai', None)\n"
        "order['openai-codex'] = ['openai-codex:default']\n"
        "tmp_auth = auth_profiles.with_suffix(auth_profiles.suffix + '.tmp')\n"
        "tmp_auth.write_text(json.dumps(auth_data, indent=2) + '\\n')\n"
        "os.chmod(tmp_auth, 0o600)\n"
        "tmp_auth.replace(auth_profiles)\n"
        "env_path = pathlib.Path.home() / '.openclaw' / '.env'\n"
        "env_lines = []\n"
        "if env_path.exists():\n"
        "    env_lines = [\n"
        "        line for line in env_path.read_text().splitlines()\n"
        "        if not line.startswith('OPENAI_OAUTH_TOKEN=')\n"
        "        and not line.startswith('OPENAI_API_KEY=')\n"
        "    ]\n"
        "env_lines.append(f'OPENAI_OAUTH_TOKEN={token}')\n"
        "tmp_env = env_path.with_suffix('.env.tmp')\n"
        "tmp_env.write_text('\\n'.join(env_lines) + '\\n')\n"
        "os.chmod(tmp_env, 0o600)\n"
        "tmp_env.replace(env_path)\n"
        f"runtime_quaid_home = pathlib.Path({VM_QUAID_HOME!r}).expanduser()\n"
        "adapter_token = runtime_quaid_home / 'adaptors' / 'openclaw' / '.auth-token'\n"
        "adapter_token.parent.mkdir(parents=True, exist_ok=True)\n"
        "tmp_token = adapter_token.with_suffix('.tmp')\n"
        "tmp_token.write_text(token + '\\n')\n"
        "os.chmod(tmp_token, 0o600)\n"
        "tmp_token.replace(adapter_token)\n"
        "paths = [\n"
        "    pathlib.Path.home() / '.openclaw' / 'shared' / 'auth' / 'credentials.json',\n"
        "    runtime_quaid_home / 'shared' / 'auth' / 'credentials.json',\n"
        "    pathlib.Path.home() / '.quaid' / 'shared' / 'auth' / 'credentials.json',\n"
        "]\n"
        "for p in paths:\n"
        "    p.parent.mkdir(parents=True, exist_ok=True)\n"
        "    try:\n"
        "        data = json.loads(p.read_text() or '{}') if p.exists() else {}\n"
        "    except Exception:\n"
        "        data = {}\n"
        "    creds = data.setdefault('credentials', {})\n"
        "    creds['codex_oauth'] = {'token': token}\n"
        "    tmp = p.with_suffix(p.suffix + '.tmp')\n"
        "    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\\n')\n"
        "    os.chmod(tmp, 0o600)\n"
        "    tmp.replace(p)\n"
        "print('OpenClaw Codex OAuth shared credential installed')\n"
    )
    result = vm.ssh(
        "python3 -c " + shlex.quote(script),
        input_data=json.dumps(profile),
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(f"OpenClaw Codex OAuth provisioning failed: {result.stderr[:200]}")
    if result.stdout.strip():
        print(f"  {result.stdout.strip()}")
    _assert_openclaw_codex_oauth_ready(vm)


def _read_openclaw_codex_oauth_status(vm: TartVM) -> dict:
    """Inspect benchmark-owned OpenClaw Codex OAuth profile state on the guest."""
    script = (
        "import json, pathlib, time\n"
        "auth_profiles = pathlib.Path.home() / '.openclaw' / 'agents' / 'main' / 'agent' / 'auth-profiles.json'\n"
        "if not auth_profiles.exists():\n"
        "    raise SystemExit('missing auth-profiles.json')\n"
        "data = json.loads(auth_profiles.read_text() or '{}')\n"
        "profiles = data.get('profiles', {}) if isinstance(data.get('profiles', {}), dict) else {}\n"
        "default_profile = profiles.get('openai-codex:default') if isinstance(profiles.get('openai-codex:default'), dict) else {}\n"
        "expires = default_profile.get('expires')\n"
        "now_ms = int(time.time() * 1000)\n"
        "order = data.get('order', {}) if isinstance(data.get('order', {}), dict) else {}\n"
        "codex_order = order.get('openai-codex') if isinstance(order.get('openai-codex'), list) else []\n"
        "competing = sorted(\n"
        "    key for key in profiles.keys()\n"
        "    if key != 'openai-codex:default' and (key == 'openai-codex' or str(key).startswith('openai-codex:'))\n"
        ")\n"
        "print(json.dumps({\n"
        "    'last_good': (data.get('lastGood', {}) or {}).get('openai-codex'),\n"
        "    'default_present': bool(default_profile),\n"
        "    'default_has_access': bool(str(default_profile.get('access') or '').strip()),\n"
        "    'default_has_refresh': bool(str(default_profile.get('refresh') or '').strip()),\n"
        "    'default_account_id': str(default_profile.get('accountId') or '').strip(),\n"
        "    'default_expires': expires,\n"
        "    'default_expired': bool(isinstance(expires, int) and expires <= now_ms),\n"
        "    'competing_profiles': competing,\n"
        "    'codex_order': codex_order,\n"
        "}))\n"
    )
    result = vm.ssh("python3 -c " + shlex.quote(script), timeout=10)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"OpenClaw Codex OAuth status probe failed: {detail[:200]}")
    try:
        status = json.loads(result.stdout or "{}")
    except Exception as exc:
        raise RuntimeError(
            f"OpenClaw Codex OAuth status probe returned invalid JSON: {(result.stdout or '')[:200]}"
        ) from exc
    if not isinstance(status, dict):
        raise RuntimeError("OpenClaw Codex OAuth status probe returned non-object payload")
    return status


def _assert_openclaw_codex_oauth_ready(vm: TartVM) -> dict:
    """Require a single fresh benchmark-owned Codex OAuth profile on the guest."""
    status = _read_openclaw_codex_oauth_status(vm)
    problems = []
    if not status.get("default_present"):
        problems.append("missing openai-codex:default profile")
    if not status.get("default_has_access"):
        problems.append("default profile missing access token")
    if not status.get("default_has_refresh"):
        problems.append("default profile missing refresh token")
    if status.get("default_expired"):
        problems.append("default profile already expired")
    if status.get("last_good") != "openai-codex:default":
        problems.append(f"lastGood.openai-codex={status.get('last_good')!r}")
    if status.get("competing_profiles"):
        problems.append(f"competing profiles present={status.get('competing_profiles')!r}")
    if status.get("codex_order") != ["openai-codex:default"]:
        problems.append(f"codex order={status.get('codex_order')!r}")
    if problems:
        raise RuntimeError(
            "OpenClaw Codex OAuth guest state invalid: " + "; ".join(problems)
        )
    return status


def _provision_openclaw_gateway_openai_auth(
    vm: TartVM,
    answer_model: Optional[str],
    openai_auth_mode: str,
) -> None:
    """Provision guest OpenAI auth when the gateway answer model uses OpenAI."""
    model = str(answer_model or "").strip()
    if not model.startswith("openai/"):
        return
    if str(openai_auth_mode or "api").strip() == "codex-oauth":
        _provision_openclaw_codex_oauth(vm)
    else:
        _provision_openclaw_openai_key(vm)


def _reapply_oc_native_gateway_runtime(vm: TartVM):
    """Reassert the benchmark-owned auth/model state after an OC gateway restart."""
    model = _OC_NATIVE_GATEWAY_ANSWER_MODEL
    if not model:
        return
    auth_mode = _OC_NATIVE_GATEWAY_OPENAI_AUTH_MODE
    _provision_openclaw_gateway_openai_auth(vm, model, auth_mode)
    gateway_model = _resolve_gateway_answer_model(
        model,
        system="oc-native",
        openai_auth_mode=auth_mode,
    )
    _patch_gateway_model(vm, gateway_model)


def _normalize_extract_model(model: str) -> str:
    normalized = str(model or "").strip()
    if not normalized:
        return "claude-sonnet-4-5-20250929"
    alias = normalized.lower()
    if alias == "sonnet":
        return "claude-sonnet-4-5-20250929"
    if alias == "haiku":
        return "claude-haiku-4-5-20251001"
    if alias == "opus":
        return "claude-opus-4-6"
    return normalized


def _patch_memory_json(
    vm: TartVM,
    extract_model: str,
    owner_id: str = "maya",
    user_name: str = "Maya",
):
    """Patch Quaid's memory.json on the VM for benchmark use.

    Sets the extraction model and default owner for the benchmark persona.
    """
    extract_model = _normalize_extract_model(extract_model)
    project_definitions = json.dumps({
        name: {
            "label": spec["label"],
            "homeDir": f"projects/{name}/",
            "sourceRoots": [spec["source_root"]],
            "autoIndex": True,
            "patterns": spec["patterns"],
            "exclude": spec["exclude"],
            "description": spec["description"],
        }
        for name, spec in VM_BENCHMARK_PROJECTS.items()
    })
    script = (
        "import json, glob, os\n"
        f"model = '{extract_model}'\n"
        f"owner = '{owner_id}'\n"
        f"display_name = {user_name!r}\n"
        f"project_definitions = json.loads({json.dumps(project_definitions)!r})\n"
        "paths = [\n"
        "    os.path.expanduser('~/clawd/config/memory.json'),\n"
        f"    os.path.expanduser('{VM_QUAID_INSTANCE_ROOT_DIR}/config.json'),\n"
        "    os.path.expanduser('~/.config/openclaw/config/memory.json'),\n"
        "]\n"
        "for p in paths:\n"
        "    if os.path.exists(p):\n"
        "        d = json.load(open(p))\n"
        "        d['models']['highReasoning'] = model\n"
        "        if 'users' not in d: d['users'] = {}\n"
        "        d['users']['default_owner'] = owner\n"
        "        d['users']['defaultOwner'] = owner\n"
        "        if 'identities' not in d['users']: d['users']['identities'] = {}\n"
        "        identity = d['users']['identities'].get(owner) or {}\n"
        "        identity['person_node_name'] = display_name\n"
        "        identity['personNodeName'] = display_name\n"
        "        d['users']['identities'][owner] = identity\n"
        "        if 'projects' not in d: d['projects'] = {}\n"
        "        if 'definitions' not in d['projects']: d['projects']['definitions'] = {}\n"
        "        d['projects']['definitions'].update(project_definitions)\n"
        "        d['projects']['enabled'] = True\n"
        "        json.dump(d, open(p, 'w'), indent=2)\n"
        "        print(f'Patched: {p}')\n"
    )
    result = vm.ssh("python3 -c " + shlex.quote(script), timeout=10)
    if result.stdout.strip():
        print(result.stdout.strip())


def _derive_quaid_runtime_llm_config(
    *,
    extract_model: str,
    answer_model: Optional[str] = None,
) -> dict[str, str]:
    """Resolve the Quaid runtime LLM config for VM lanes.

    Quaid's direct extraction script chooses its own extraction model, but
    store-time helper calls (dedup, review, janitor) still resolve through the
    runtime instance config under QUAID_HOME. Without concrete tier models the
    adapter passes the literal string "default" down to providers.

    For Quaid-on-OC-VM apples-to-apples runs, Quaid itself should stay on its
    native Anthropic Sonnet/Haiku split even when the outer OC answer lane is
    patched to an OpenAI OAuth gateway model. The gateway answer lane is
    configured separately; this runtime config is only for Quaid-owned helper
    calls.
    """
    deep_model = _normalize_extract_model(extract_model)
    fast_model = "claude-haiku-4-5-20251001"
    return {
        "llmProvider": "anthropic",
        "fastReasoningProvider": "anthropic",
        "deepReasoningProvider": "anthropic",
        "fastReasoning": fast_model,
        "deepReasoning": deep_model,
        "fastReasoningEffort": "none",
        "deepReasoningEffort": "high",
        "llm_provider": "anthropic",
        "fast_reasoning_provider": "anthropic",
        "deep_reasoning_provider": "anthropic",
        "fast_reasoning": fast_model,
        "deep_reasoning": deep_model,
        "fast_reasoning_effort": "none",
        "deep_reasoning_effort": "high",
    }


def _patch_quaid_runtime_instance_config(
    vm: TartVM,
    instance_id: str = VM_QUAID_INSTANCE,
    ollama_url: str = VM_QUAID_OLLAMA_URL,
    *,
    extract_model: str = "claude-sonnet-4-5-20250929",
    answer_model: Optional[str] = None,
    owner_id: str = "maya",
    user_name: str = "Maya",
):
    """Patch the active Quaid instance config used by gateway/store subprocesses."""
    llm_config = _derive_quaid_runtime_llm_config(
        extract_model=extract_model,
        answer_model=answer_model,
    )
    script = (
        "import json, os\n"
        f"instance_id = {instance_id!r}\n"
        f"ollama_url = {ollama_url!r}\n"
        f"owner = {owner_id!r}\n"
        f"display_name = {user_name!r}\n"
        f"llm_config = {json.dumps(llm_config, sort_keys=True)!r}\n"
        "path = os.path.expanduser(f'~/clawd/instances/{instance_id}/config.json')\n"
        "os.makedirs(os.path.dirname(path), exist_ok=True)\n"
        "try:\n"
        "    d = json.load(open(path))\n"
        "    if not isinstance(d, dict):\n"
        "        d = {}\n"
        "except Exception:\n"
        "    d = {}\n"
        "d.setdefault('instance', {})\n"
        "d['instance'].setdefault('id', instance_id)\n"
        "d.setdefault('adapter', {})\n"
        "d['adapter'].setdefault('type', 'openclaw')\n"
        "d.setdefault('users', {})\n"
        "d['users']['default_owner'] = owner\n"
        "d['users']['defaultOwner'] = owner\n"
        "identities = d['users'].setdefault('identities', {})\n"
        "identity = identities.get(owner) or {}\n"
        "identity['person_node_name'] = display_name\n"
        "identity['personNodeName'] = display_name\n"
        "identities[owner] = identity\n"
        "d.setdefault('models', {})\n"
        "for key, value in json.loads(llm_config).items():\n"
        "    d['models'][key] = value\n"
        "d.setdefault('retrieval', {})\n"
        "d['retrieval']['defaultLimit'] = 5\n"
        "d['retrieval']['maxLimit'] = 8\n"
        "d['retrieval']['failHard'] = True\n"
        "d['retrieval']['fail_hard'] = True\n"
        "d['models']['embeddings_provider'] = 'ollama'\n"
        "d['models']['embeddingsProvider'] = 'ollama'\n"
        "d.setdefault('ollama', {})\n"
        "d['ollama']['url'] = ollama_url\n"
        "d['ollama']['embeddingModel'] = 'nomic-embed-text'\n"
        "d['ollama']['embeddingDim'] = 768\n"
        "d['ollama']['embedding_model'] = 'nomic-embed-text'\n"
        "d['ollama']['embedding_dim'] = 768\n"
        "json.dump(d, open(path, 'w'), indent=2)\n"
        "print(f'Patched runtime config: {path}')\n"
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
    answer_model: Optional[str] = None,
    openai_auth_mode: str = "api",
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
    if (
        system != "mem0"
        and str(openai_auth_mode or "api").strip() == "codex-oauth"
        and str(answer_model or "").strip().startswith("openai/")
    ):
        _provision_openclaw_codex_oauth(vm)
        status = _assert_openclaw_codex_oauth_ready(vm)
        print(
            "  OpenClaw Codex OAuth ready: "
            f"expires={status.get('default_expires')} "
            f"account={status.get('default_account_id') or 'unknown'}"
        )

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
        judge_prompt_tokens = count_tokens(
            JUDGE_PROMPT.format(
                question=question,
                ground_truth=ground_truth,
                prediction=prediction,
            )
        )

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
                "agent_visible_total": q_tokens + p_tokens,
                "judge_prompt": judge_prompt_tokens,
            },
        }
        results.append(result)

        # Save incrementally
        _save_results(results, results_dir)

    return results


def _evaluate_vm_agent(vm: TartVM, question: str, query_idx: int,
                       system: str) -> str:
    """Send a question to the VM agent and get the response.

    Uses a fresh eval session per query to avoid cross-contamination. For
    oc-native, the guest eval session is deleted immediately after capture so
    later eval queries cannot retrieve prior eval transcripts.
    """
    session_id = f"eval-q{query_idx:03d}"
    escaped_question = shlex.quote(question)
    session_key = None
    use_oc_gateway_eval = _uses_oc_gateway_eval_isolation(system)
    if use_oc_gateway_eval:
        session_key = _oc_native_eval_session_key(session_id)
        _register_session(
            vm,
            session_id,
            session_key=session_key,
            session_file=_oc_native_eval_session_file(session_id),
        )
    else:
        _register_session(vm, session_id)

    try:
        for attempt in range(1, VM_AGENT_EVAL_MAX_TIMEOUT_RETRIES + 2):
            try:
                if use_oc_gateway_eval:
                    response = _run_oc_native_gateway_turn(
                        vm,
                        session_id,
                        question,
                        timeout_s=VM_AGENT_EVAL_TIMEOUT_S,
                        session_key=session_key,
                    )
                else:
                    command = (
                        f"openclaw agent --agent main --session-id {session_id} "
                        f"--message {escaped_question}"
                    )
                    result = vm.ssh(
                        command,
                        timeout=VM_AGENT_EVAL_TIMEOUT_S,
                    )
                break
            except subprocess.TimeoutExpired as exc:
                if attempt <= VM_AGENT_EVAL_MAX_TIMEOUT_RETRIES:
                    print(
                        f"  WARN: eval query timeout ({attempt}/{VM_AGENT_EVAL_MAX_TIMEOUT_RETRIES + 1}) "
                        f"session={session_id}; retrying..."
                    )
                    continue
                raise RuntimeError(
                    f"Eval query timed out after {VM_AGENT_EVAL_TIMEOUT_S}s "
                    f"(session={session_id}, query_idx={query_idx})"
                ) from exc

        if use_oc_gateway_eval:
            return _extract_agent_answer(response)

        if result.returncode != 0:
            return f"Error: {result.stderr[:200]}"

        # Parse agent response (strip tool use logs, keep final text)
        response = result.stdout.strip()
        return _extract_agent_answer(response)
    finally:
        if use_oc_gateway_eval:
            _cleanup_oc_native_eval_session(vm, session_id, session_key=session_key)


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


def _apply_vm_eval_query_profile(queries: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Apply the shared eval-query selectors used by the non-VM harness."""
    try:
        from run_production_benchmark import _apply_eval_query_profile
    except ImportError:
        return list(queries), {"profile": "full", "requested": len(queries), "selected": len(queries)}

    return _apply_eval_query_profile(list(queries))


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

    eval_token_estimate = _summarize_eval_token_estimates(scored)

    return {
        "overall": {
            "count": len(scored),
            "accuracy": round(accuracy, 2),
            "correct": correct,
            "partial": partial,
            "wrong": wrong,
        },
        "per_type": per_type,
        "eval_token_estimate": eval_token_estimate,
    }


def _summarize_eval_token_estimates(results: List[dict]) -> dict:
    """Aggregate visible eval and judge-token estimates from result rows."""
    totals = {
        "question_tokens": 0,
        "prediction_tokens": 0,
        "agent_visible_total": 0,
        "judge_prompt_tokens": 0,
    }
    for row in results:
        estimate = row.get("tokens_estimate") or {}
        totals["question_tokens"] += int(estimate.get("question") or 0)
        totals["prediction_tokens"] += int(estimate.get("prediction") or 0)
        totals["agent_visible_total"] += int(
            estimate.get("agent_visible_total")
            or ((estimate.get("question") or 0) + (estimate.get("prediction") or 0))
        )
        totals["judge_prompt_tokens"] += int(estimate.get("judge_prompt") or 0)
    count = len(results)
    totals["count"] = count
    totals["total_lower_bound"] = totals["agent_visible_total"] + totals["judge_prompt_tokens"]
    totals["notes"] = (
        "Lower-bound estimate only: VM agent provider usage may include hidden prompt, "
        "tool, active-memory, and gateway context not reported by openclaw agent stdout."
    )
    if count:
        totals["per_query_avg"] = {
            "agent_visible_total": round(totals["agent_visible_total"] / count, 1),
            "judge_prompt_tokens": round(totals["judge_prompt_tokens"] / count, 1),
            "total_lower_bound": round(totals["total_lower_bound"] / count, 1),
        }
    else:
        totals["per_query_avg"] = {
            "agent_visible_total": 0.0,
            "judge_prompt_tokens": 0.0,
            "total_lower_bound": 0.0,
        }
    return totals


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


def _resolve_local_quaid_plugin_dir() -> Path:
    """Resolve the local Quaid plugin/module tree for VM sync."""
    explicit = os.environ.get("BENCHMARK_PLUGIN_DIR", "").strip()
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if p.exists():
            return p

    candidates = [
        Path.home() / "quaidcode" / "benchmark-checkpoint" / "modules" / "quaid",
        Path.home() / "quaidcode" / "benchmark-checkpoint" / "plugins" / "quaid",
        Path(__file__).resolve().parent.parent / "benchmark-checkpoint" / "modules" / "quaid",
        Path(__file__).resolve().parent.parent / "benchmark-checkpoint" / "plugins" / "quaid",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    try:
        from run_production_benchmark import _resolve_quaid_dir  # type: ignore
    except Exception:
        resolved = None
    else:
        try:
            resolved = Path(_resolve_quaid_dir()).expanduser().resolve()
        except Exception:
            resolved = None
    if resolved and resolved.exists():
        return resolved
    raise FileNotFoundError("Could not resolve local Quaid plugin/module directory for VM sync")


def _resolve_local_quaid_memory_example(plugin_dir: Path) -> Optional[Path]:
    """Resolve memory.json.example adjacent to the local Quaid tree."""
    candidates = [
        plugin_dir / "memory.json.example",
        plugin_dir.parent.parent / "memory.json.example",
        plugin_dir.parent / "memory.json.example",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _build_local_quaid_plugin_tarball(plugin_dir: Path) -> Path:
    """Bundle the local Quaid tree for guest upload, excluding bulky dev-only dirs."""
    excludes = {"node_modules", "__pycache__", ".pytest_cache", "tests"}
    handle = tempfile.NamedTemporaryFile(prefix="quaid-plugin-", suffix=".tar.gz", delete=False)
    archive_path = Path(handle.name)
    handle.close()

    def _filter(info: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
        rel = Path(info.name)
        if any(part in excludes for part in rel.parts):
            return None
        return info

    with tarfile.open(archive_path, "w:gz") as bundle:
        bundle.add(plugin_dir, arcname=".", filter=_filter)
    return archive_path


def _upload_local_file_via_vm_ssh(vm: "TartVM", local_path: Path, remote_path: str, *, timeout: int = 300):
    """Upload a local file to the guest over the already-working SSH command path."""
    payload = base64.b64encode(local_path.read_bytes()).decode("ascii")
    script = (
        "import base64, pathlib, sys\n"
        f"target = pathlib.Path({remote_path!r}).expanduser()\n"
        "target.parent.mkdir(parents=True, exist_ok=True)\n"
        "target.write_bytes(base64.b64decode(sys.stdin.read()))\n"
    )
    return vm.ssh(
        "python3 -c " + shlex.quote(script),
        input_data=payload,
        timeout=timeout,
        raw=True,
    )


def setup_system(vm: TartVM, system: str, snapshot_base: str = "clean-openclaw",
                 extract_model: str = "claude-sonnet-4-5-20250929",
                 local_plugin: bool = False,
                 answer_model: str | None = None,
                 openai_auth_mode: str = "api"):
    """Restore VM and configure for the given system.

    Args:
        vm: TartVM instance
        system: One of "base", "qmd", "quaid", "mem0", "oc-native"
        snapshot_base: Snapshot to restore from
        extract_model: Model for extraction/janitor (Sonnet for dev, Opus for final)
        local_plugin: If True, rsync local Quaid plugin instead of cloning from GitHub
        answer_model: Override gateway agent model (e.g. "openai/gpt-5.4")
        openai_auth_mode: "api" for direct OpenAI API auth, "codex-oauth" for Codex OAuth shared auth
    """
    # Quaid VM benchmarks should exercise the local checkpoint under test, not
    # a stale guest install or a fresh GitHub clone. There is no CLI false
    # override today, so treat the local checkpoint plugin as the benchmark
    # default for Quaid lanes.
    if system == "quaid":
        local_plugin = True

    if system == "mem0":
        # Mem0 runs on host, no VM setup needed
        print(f"  Mem0 runs on host — no VM setup needed")
        return

    print(f"\n--- Setting up {system} ---")
    vm.restore(snapshot_base)
    _set_oc_native_gateway_runtime_context(answer_model if system == "oc-native" else None, openai_auth_mode)

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
        # - active-memory blocking recall sub-agent enabled
        # - memory-wiki bridge enabled and synced after injection
        # - direct session transcript indexing enabled
        #
        # Keep this config-driven. On current OC VMs, `openclaw plugins
        # enable/disable ...` can hang over SSH even though directly patching
        # ~/.openclaw/openclaw.json is sufficient and more stable.
        _ensure_oc_native_embed_proxy(vm.tart_host)
        _patch_openclaw_native_memory(vm, enable_session_hook=True)
        print(
            "  OpenClaw native configured "
            "(memory-core + session-memory + active-memory + memory-wiki + session indexing)"
        )

    elif system == "quaid":
        _ensure_oc_native_embed_proxy(vm.tart_host)
        if local_plugin:
            # Upload local Quaid tree to the guest through the existing VM copy path.
            local_plugin_dir = _resolve_local_quaid_plugin_dir()
            memory_example = _resolve_local_quaid_memory_example(local_plugin_dir)
            plugin_archive = _build_local_quaid_plugin_tarball(local_plugin_dir)
            try:
                vm.ssh(
                    "rm -rf ~/clawd/plugins/quaid "
                    "&& mkdir -p ~/clawd/plugins/quaid ~/clawd/config ~/clawd/data ~/clawd/journal",
                    raw=True,
                )
                upload = _upload_local_file_via_vm_ssh(vm, plugin_archive, "/tmp/quaid-plugin.tgz", timeout=300)
                if upload.returncode != 0:
                    raise RuntimeError(
                        "Failed to upload local Quaid plugin archive to VM: "
                        f"{(upload.stderr or upload.stdout or '').strip()}"
                    )
                unpack = vm.ssh(
                    "tar -xzf /tmp/quaid-plugin.tgz -C ~/clawd/plugins/quaid "
                    "&& rm -f /tmp/quaid-plugin.tgz",
                    raw=True,
                    timeout=300,
                )
                if unpack.returncode != 0:
                    raise RuntimeError(
                        "Failed to unpack local Quaid plugin archive on VM: "
                        f"{(unpack.stderr or unpack.stdout or '').strip()}"
                    )
            finally:
                try:
                    plugin_archive.unlink()
                except FileNotFoundError:
                    pass
            if memory_example is not None:
                copied = _upload_local_file_via_vm_ssh(
                    vm,
                    memory_example,
                    "~/clawd/plugins/quaid/memory.json.example",
                    timeout=120,
                )
                if copied.returncode != 0:
                    raise RuntimeError(
                        "Failed to upload Quaid memory example to VM: "
                        f"{(copied.stderr or copied.stdout or '').strip()}"
                    )
            # Copy config if not present
            vm.ssh(
                "test -f ~/clawd/config/memory.json || "
                "cp ~/clawd/plugins/quaid/memory.json.example ~/clawd/config/memory.json 2>/dev/null || true",
                raw=True,
            )
            # Install runtime deps only; the OpenClaw peer is provided by the host.
            vm.ssh("cd ~/clawd/plugins/quaid && npm install --omit=dev --legacy-peer-deps", timeout=300)
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
                    "&& cd ~/clawd/plugins/quaid && npm install --omit=dev --legacy-peer-deps",
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
        _patch_memory_json(vm, extract_model, owner_id="maya", user_name="Maya")
        _patch_quaid_runtime_instance_config(
            vm,
            extract_model=extract_model,
            answer_model=answer_model,
            owner_id="maya",
            user_name="Maya",
        )

        # Ensure Quaid embeddings use the host Ollama service, not guest localhost.
        vm.ssh(
            "python3 -c \""
            f"import json; p='/Users/admin/clawd/config/memory.json'; "
            f"d=json.load(open(p)); d.setdefault('ollama', {{}})['url']={VM_QUAID_OLLAMA_URL!r}; "
            "json.dump(d,open(p,'w'),indent=2)\" 2>/dev/null || true"
        )
        print(f"  Ollama: using host at {VM_QUAID_OLLAMA_URL}")

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
        _validate_anthropic_credential_for_vm(extract_model)
        # Create core markdown files (simulates onboarding)
        _create_core_files(vm)
        # Create benchmark project homes and register them through the product CLI.
        _create_project_files(vm)
        _register_vm_benchmark_projects(vm)
        # Ensure logs directory exists
        vm.ssh("mkdir -p ~/clawd/logs", raw=True)
        # Symlink .env so janitor can find the API key
        # (janitor looks in {workspace}/.env, key lives in ~/.openclaw/.env)
        vm.ssh("ln -sf ~/.openclaw/.env ~/clawd/.env 2>/dev/null || true", raw=True)
        print(f"  API key symlinked: ~/clawd/.env → ~/.openclaw/.env")
        _provision_openclaw_anthropic_key(vm)
        _configure_openclaw_quaid_plugin(vm)

    # Set gateway agent model if specified
    if system == "oc-native":
        _reapply_oc_native_gateway_runtime(vm)
    elif answer_model:
        _provision_openclaw_gateway_openai_auth(vm, answer_model, openai_auth_mode)
        gateway_answer_model = _resolve_gateway_answer_model(
            answer_model,
            system=system,
            openai_auth_mode=openai_auth_mode,
        )
        _patch_gateway_model(vm, gateway_answer_model)

    # Restart gateway to pick up changes (fresh DB, clean sessions)
    if system == "oc-native":
        _restart_oc_native_gateway(vm, port=18789)
        _validate_openclaw_native_memory(vm)
    else:
        _restart_quaid_gateway(vm, port=18789)
        if system == "quaid":
            _validate_quaid_vm_embeddings(vm)
            _stop_quaid_instance_daemon(vm)

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
    vm_name: str = "test-openclaw",
    tart_host: Optional[str] = None,
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
    openai_auth_mode: str = "api",
    splitting: str = "perday",
    resume_day_lifecycle: bool = False,
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

    if system == "oc-native":
        host_name = socket.gethostname().strip().lower()
        safe_tart_host = str(tart_host or "").strip()
        safe_vm_name = str(vm_name or "").strip()
        if not safe_tart_host:
            if safe_vm_name.startswith("quaid-livetest-"):
                raise RuntimeError(
                    "oc-native benchmark refused to use shared local VM "
                    f"{safe_vm_name!r}; pass --tart-host alfie.local or use a "
                    "dedicated local benchmark VM name prefixed with "
                    f"{OC_NATIVE_LOCAL_VM_NAMESPACE_PREFIXES[0]!r}"
                )
            if host_name in {"testbench", "testbench.local"} and not safe_vm_name.startswith(
                OC_NATIVE_LOCAL_VM_NAMESPACE_PREFIXES
            ):
                raise RuntimeError(
                    "oc-native benchmark on testbench.local requires --tart-host "
                    "alfie.local or a dedicated local benchmark VM name prefixed "
                    f"with {OC_NATIVE_LOCAL_VM_NAMESPACE_PREFIXES[0]!r}"
                )

    extract_model = _normalize_extract_model(extract_model)
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
    queries, query_profile_meta = _apply_vm_eval_query_profile(queries)

    # Apply limits (for smoke testing)
    if limit_sessions:
        reviews = reviews[:limit_sessions]
    if limit_queries:
        queries = queries[:limit_queries]

    print(f"Sessions: {len(reviews)} ({len(arc_reviews)} arc + {len(filler_reviews)} filler)")
    if limit_sessions:
        print(f"  (limited to {limit_sessions})")
    print(f"Eval queries: {len(queries)}")
    if query_profile_meta.get("profile") not in {"full", "canonical"}:
        print(
            "  Query profile: "
            f"{query_profile_meta.get('profile')} "
            f"({query_profile_meta.get('selected', len(queries))}/"
            f"{query_profile_meta.get('requested', len(queries))})"
        )
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

    vm = TartVM(ip=vm_ip, vm_name=vm_name, tart_host=tart_host)

    # Phase 1: Setup
    if not eval_only:
        setup_system(vm, system, snapshot_base, extract_model=extract_model,
                     local_plugin=local_plugin, answer_model=answer_model,
                     openai_auth_mode=openai_auth_mode)
        if resume_day_lifecycle:
            if system != "quaid" or splitting != "timeout":
                raise RuntimeError(
                    "--resume-day-lifecycle in vm_benchmark currently supports only "
                    "system=quaid with --splitting timeout"
                )
            restored = _restore_vm_quaid_timeout_resume_checkpoint(vm, results_dir)
            if restored:
                print(
                    "  Restored timeout lifecycle checkpoint: "
                    f"completed_chunks={restored.get('completed_chunks', 0)} "
                    f"current_day={restored.get('resume_current_day') or 'unknown'}"
                )
            else:
                raise RuntimeError(
                    "Resume requested but no VM lifecycle checkpoint was found in "
                    f"{_vm_lifecycle_resume_root(results_dir)}"
                )

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
                resume_state=_load_vm_lifecycle_resume_state(results_dir) if resume_day_lifecycle else None,
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
    eval_results = evaluate_queries(
        vm,
        queries,
        system,
        results_dir,
        judge_model,
        mem0_adapter=_adapter,
        answer_model=answer_model,
        openai_auth_mode=openai_auth_mode,
    )

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
    parser.add_argument("--vm-name", type=str, default="test-openclaw",
                        help="Tart VM name (used for local restore/ip refresh)")
    parser.add_argument("--tart-host", type=str, default="",
                        help="Optional SSH host that runs Tart (e.g. alfie.local)")
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
    parser.add_argument("--resume-day-lifecycle", action="store_true",
                        help="Resume Quaid OC timeout injection from the latest saved day/janitor checkpoint in results-dir")
    parser.add_argument("--resume-extraction", dest="resume_day_lifecycle", action="store_true",
                        help=argparse.SUPPRESS)
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
                        help="Override gateway agent model (e.g. openai/gpt-5.4)")
    parser.add_argument("--openai-auth-mode", type=str, default="api",
                        choices=["api", "codex-oauth"],
                        help="For openai/* answer models on oc-native: use direct OpenAI API auth or Codex OAuth shared auth")
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
            vm_name=args.vm_name,
            tart_host=args.tart_host or None,
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
            openai_auth_mode=args.openai_auth_mode,
            splitting=args.splitting,
            resume_day_lifecycle=args.resume_day_lifecycle,
        )
        all_results[system] = result

        # For Quaid, also run nightly A/B if running all
        if system == "quaid" and args.system == "all" and args.mode == "natural":
            nightly_result = run_benchmark(
                system="quaid",
                mode="nightly",
                vm_ip=args.vm_ip,
                tart_host=args.tart_host or None,
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
                openai_auth_mode=args.openai_auth_mode,
                splitting=args.splitting,
                resume_day_lifecycle=args.resume_day_lifecycle,
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
