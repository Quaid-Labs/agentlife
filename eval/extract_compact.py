#!/usr/bin/env python3
"""Trigger native Quaid compaction for the VM benchmark.

The benchmark writes native session JSONL files into the OpenClaw session
store, then needs a deterministic way to force Quaid extraction without
going through the gateway `/compact` text-command path.

This wrapper is benchmark-only orchestration. It does not implement its own
extraction/store logic anymore; it imports the installed Quaid plugin,
invokes the real daemon `process_signal()` path against the session file,
then reads back daemon metrics for benchmark telemetry.

Usage (on VM):
    python3 extract_compact.py \\
        --session-file ~/.openclaw/agents/main/sessions/benchmark-quaid.jsonl \\
        --workspace ~/clawd \\
        --user-name "Maya" \\
        --owner-id maya \\
        --model claude-sonnet-4-5-20250929

Why this exists:
    The OpenClaw gateway's /compact command only fires from the auto-reply
    pipeline (incoming channel messages). `openclaw agent --message '/compact'`
    sends the text to the LLM as a regular message — the native command handler
    doesn't intercept it. This wrapper calls Quaid's real extraction daemon
    signal path directly.

    See: openclaw/src/auto-reply/reply/commands-compact.ts (line 47-109)
    vs:  openclaw/src/commands/agent.ts (no /compact interception)
"""

import argparse
import importlib
import importlib.util
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

_DEFAULT_OWNER_ID = os.environ.get("BENCH_OWNER_ID", "maya").strip() or "maya"
_PROJECT_UPDATER_ENV_LOCK = threading.Lock()
_ANTHROPIC_OAUTH_IDENTITY_TEXT = (
    "You are Claude Code, Anthropic's official CLI for Claude."
)
_ANTHROPIC_OAUTH_USER_AGENT = "claude-cli/2.1.2 (external, cli)"
_ANTHROPIC_OAUTH_CLAUDE_CODE_BETA = "claude-code-20250219"
_WEAKLY_INTERPRETED_FACT_RE = re.compile(
    r"\b(mentioned in passing|ongoing context|showing that|showing she|showing he|showing they|indicating that|someone referred to as|presumably|likely|probably|which suggests|which indicates|reflecting)\b",
    re.IGNORECASE,
)


def _is_storeable_extracted_fact_text(text: str) -> bool:
    """Return True when extracted fact text has enough substance to store.

    Keep the historical three-token guard for whitespace-separated languages,
    but allow unsegmented scripts where valid sentences have no spaces.
    """
    value = (text or "").strip()
    if not value:
        return False
    tokens = value.split()
    if len(tokens) >= 3:
        return True
    if len(tokens) > 1:
        return False
    return sum(1 for ch in value if ch.isalnum()) >= 6


def read_session_messages(session_file: str) -> list[dict]:
    """Read messages from session JSONL file.

    Handles both formats:
    - { "type": "message", "message": { "role": ..., "content": ... } }
    - { "role": ..., "content": ... }
    """
    messages = []
    with open(session_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("type") == "message" and entry.get("message"):
                    messages.append(entry["message"])
                elif entry.get("role"):
                    messages.append(entry)
            except json.JSONDecodeError:
                continue
    return messages


def _read_env_key(env_file: str, key: str) -> str | None:
    """Read a key from a .env-style file with basic shell parsing."""
    if not os.path.exists(env_file):
        return None
    with open(env_file, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("export "):
                stripped = stripped[len("export "):].strip()
            if "=" not in stripped:
                continue
            name, raw_val = stripped.split("=", 1)
            if name.strip() != key:
                continue
            try:
                parts = shlex.split(raw_val, comments=True, posix=True)
            except ValueError:
                parts = [raw_val]
            return parts[0] if parts else ""
    return None


def _adjust_extraction_confidence(text: str, confidence: float) -> float:
    if confidence <= 0.3:
        return confidence
    if not _WEAKLY_INTERPRETED_FACT_RE.search(str(text or "")):
        return confidence
    if confidence >= 0.85:
        return 0.6
    return 0.3


def _date_to_created_at(date_str: str | None) -> str | None:
    raw = str(date_str or "").strip()
    if not raw:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        return f"{raw}T23:59:59"
    if re.search(r"\b\d{4}-\d{2}-\d{2}\b", raw):
        return raw
    return None


def _resolve_runtime_plugin_root(workspace: str) -> Path:
    workspace_root = Path(os.path.expanduser(workspace))
    candidates = [
        workspace_root / "plugins" / "quaid",
        Path.home() / "quaidcode" / "benchmark-checkpoint" / "modules" / "quaid",
        Path.home() / "quaidcode" / "dev" / "modules" / "quaid",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise RuntimeError(f"Unable to locate Quaid runtime plugin root; searched: {searched}")


def _quaid_signal_type(trigger: str) -> str:
    normalized = str(trigger or "").strip().lower()
    if "reset" in normalized:
        return "reset"
    if "session_end" in normalized or "session end" in normalized:
        return "session_end"
    if "timeout" in normalized:
        return "timeout"
    return "compaction"


def _latest_session_metric(metric_path: Path, session_id: str, *, start_offset: int = 0) -> dict:
    if not metric_path.exists():
        return {}
    latest = {}
    with metric_path.open("r", encoding="utf-8") as fh:
        if start_offset > 0:
            fh.seek(start_offset)
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(row.get("session_id") or "") != str(session_id or ""):
                continue
            if str(row.get("event") or "") not in {"rolling_flush", "rolling_flush_error"}:
                continue
            latest = row
    return latest


def _metric_usage(metric: dict, usage_fallback: dict) -> dict:
    metric = dict(metric or {})
    usage = dict(usage_fallback or {})
    extract_input = int(metric.get("extract_input_tokens", 0) or 0)
    extract_output = int(metric.get("extract_output_tokens", 0) or 0)
    publish_input = int(metric.get("publish_input_tokens", 0) or 0)
    publish_output = int(metric.get("publish_output_tokens", 0) or 0)
    total_input = extract_input + publish_input
    total_output = extract_output + publish_output
    if total_input > 0 or total_output > 0:
        return {
            "calls": int(metric.get("extract_llm_calls", 0) or 0)
            + int(metric.get("publish_llm_calls", 0) or 0),
            "input_tokens": total_input,
            "output_tokens": total_output,
            "fast_calls": int(metric.get("extract_fast_calls", 0) or 0)
            + int(metric.get("publish_fast_calls", 0) or 0),
            "deep_calls": int(metric.get("extract_deep_calls", 0) or 0)
            + int(metric.get("publish_deep_calls", 0) or 0),
        }
    return usage


def _run_native_daemon_compaction(
    *,
    session_file: str,
    workspace: str,
    session_id: str,
    trigger: str,
    sim_date: str | None,
) -> dict:
    plugin_root = _resolve_runtime_plugin_root(workspace)
    if str(plugin_root) not in sys.path:
        sys.path.insert(0, str(plugin_root))

    daemon = importlib.import_module("core.extraction_daemon")
    adapter_mod = importlib.import_module("lib.adapter")

    metric_path = daemon._rolling_metrics_path()
    metric_offset = metric_path.stat().st_size if metric_path.exists() else 0
    usage_before = daemon._read_usage_totals()
    adapter = adapter_mod.get_adapter()

    signal = {
        "type": _quaid_signal_type(trigger),
        "session_id": session_id,
        "transcript_path": session_file,
        "adapter": "openclaw",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "meta": {"benchmark_trigger": True},
    }

    previous_quaid_now = os.environ.get("QUAID_NOW")
    if sim_date:
        os.environ["QUAID_NOW"] = str(sim_date)
    try:
        daemon.process_signal(signal)
    finally:
        if sim_date:
            if previous_quaid_now is None:
                os.environ.pop("QUAID_NOW", None)
            else:
                os.environ["QUAID_NOW"] = previous_quaid_now

    usage_after = daemon._read_usage_totals()
    metric = _latest_session_metric(metric_path, session_id, start_offset=metric_offset)
    usage = _metric_usage(metric, daemon._usage_delta(usage_before, usage_after))
    transcript = adapter.parse_session_jsonl(Path(session_file))
    return {
        "usage": usage,
        "metric": metric,
        "transcript": transcript,
    }


def _is_anthropic_oauth_token(token: str) -> bool:
    return str(token or "").strip().startswith("sk-ant-oat")


def _anthropic_headers(credential: str) -> dict:
    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    if _is_anthropic_oauth_token(credential):
        headers["Authorization"] = f"Bearer {credential}"
        headers["Accept"] = "application/json"
        headers["user-agent"] = _ANTHROPIC_OAUTH_USER_AGENT
        headers["x-app"] = "cli"
        headers["anthropic-beta"] = (
            f"{_ANTHROPIC_OAUTH_CLAUDE_CODE_BETA},oauth-2025-04-20,prompt-caching-2024-07-31"
        )
    else:
        headers["x-api-key"] = credential
        headers["anthropic-beta"] = "prompt-caching-2024-07-31"
    return headers


def _anthropic_system_blocks(system_prompt: str, credential: str) -> list[dict]:
    blocks: list[dict] = []
    if _is_anthropic_oauth_token(credential):
        blocks.append(
            {
                "type": "text",
                "text": _ANTHROPIC_OAUTH_IDENTITY_TEXT,
                "cache_control": {"type": "ephemeral"},
            }
        )
    if system_prompt:
        blocks.append(
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        )
    return blocks


def build_transcript(messages: list[dict], agent_name: str = "Assistant") -> str:
    """Build transcript from messages, matching the Quaid plugin format.

    Filters out system messages, gateway restarts, heartbeats.
    """
    transcript = []
    for msg in messages:
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
        if not content:
            continue
        # Strip channel prefixes
        content = re.sub(
            r"^\[(?:Telegram|WhatsApp|Discord|Signal|Slack)\s+[^\]]+\]\s*",
            "", content, flags=re.IGNORECASE,
        )
        content = re.sub(r"\n?\[message_id:\s*\d+\]", "", content, flags=re.IGNORECASE).strip()
        # Skip system/restart/heartbeat lines
        if content.startswith("GatewayRestart:") or content.startswith("System:"):
            continue
        if '"kind": "restart"' in content:
            continue
        if "HEARTBEAT" in content and "HEARTBEAT_OK" in content:
            continue
        if re.sub(r"[*_<>/b\s]", "", content).startswith("HEARTBEAT_OK"):
            continue
        if not content:
            continue
        label = "User" if role == "user" else agent_name
        transcript.append(f"{label}: {content}")
    return "\n\n".join(transcript)


def _runtime_prompt_candidates() -> list[Path]:
    explicit = os.environ.get("BENCHMARK_EXTRACTION_PROMPT_FILE", "").strip()
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    here = Path(__file__).resolve()
    repo_root = here.parents[1]
    cwd = Path.cwd()
    candidates.extend([
        cwd / "prompts" / "extraction.txt",
        repo_root / "benchmark-checkpoint" / "modules" / "quaid" / "prompts" / "extraction.txt",
        repo_root.parent / "benchmark-checkpoint" / "modules" / "quaid" / "prompts" / "extraction.txt",
        Path.home() / "quaidcode" / "benchmark-checkpoint" / "modules" / "quaid" / "prompts" / "extraction.txt",
        Path.home() / "quaid" / "benchmark-checkpoint" / "modules" / "quaid" / "prompts" / "extraction.txt",
        Path.home() / "quaidcode" / "dev" / "modules" / "quaid" / "prompts" / "extraction.txt",
    ])
    return candidates


def _load_runtime_extraction_prompt() -> str:
    """Load the same extraction prompt file used by Quaid runtime."""
    for candidate in _runtime_prompt_candidates():
        try:
            if candidate.exists():
                text = candidate.read_text(encoding="utf-8").strip()
                if text:
                    return text
        except OSError:
            continue
    searched = ", ".join(str(p) for p in _runtime_prompt_candidates())
    raise RuntimeError(f"Unable to locate Quaid runtime extraction prompt; searched: {searched}")


def _benchmark_emit_artifact() -> bool:
    return str(os.environ.get("BENCHMARK_EXTRACTION_INCLUDE_ARTIFACT", "")).strip().lower() in {
        "1", "true", "yes", "on"
    }


def _normalize_domain_defs(allowed_domains: object) -> dict[str, str]:
    if not allowed_domains:
        return {}
    if isinstance(allowed_domains, dict):
        return {
            str(k).strip().lower(): str(v or "").strip()
            for k, v in allowed_domains.items()
            if str(k).strip()
        }
    domain_defs: dict[str, str] = {}
    for item in allowed_domains if isinstance(allowed_domains, list) else list(allowed_domains):
        if isinstance(item, (list, tuple)) and item:
            key = str(item[0]).strip().lower()
            desc = str(item[1] if len(item) > 1 else "").strip()
        else:
            key = str(item).strip().lower()
            desc = ""
        if key:
            domain_defs[key] = desc
    return domain_defs


def _normalize_project_defs(known_projects: object) -> dict[str, str]:
    if not known_projects:
        return {}
    if isinstance(known_projects, dict):
        return {
            str(k).strip(): str(v or "").strip()
            for k, v in known_projects.items()
            if str(k).strip()
        }
    project_defs: dict[str, str] = {}
    for item in known_projects if isinstance(known_projects, list) else list(known_projects):
        if isinstance(item, (list, tuple)) and item:
            key = str(item[0]).strip()
            desc = str(item[1] if len(item) > 1 else "").strip()
        else:
            key = str(item).strip()
            desc = ""
        if key:
            project_defs[key] = desc
    return project_defs


def build_extraction_prompt(
    user_name: str,
    agent_name: str = "Assistant",
    focus: str = "all",
    allowed_domains: object = None,
    known_projects: object = None,
) -> str:
    """Build the extraction system prompt from the runtime Quaid prompt.

    Benchmark extraction must not carry a forked prompt or dataset-specific
    examples. It uses the same prompt file as product runtime, plus the same
    dynamic owner/domain/project blocks that runtime injects.
    """
    _ = (agent_name, focus)
    prompt = _load_runtime_extraction_prompt()

    owner = str(user_name or "").strip()
    if owner:
        prompt = (
            f"The user who owns this knowledge base is: {owner}\n"
            f"When the transcript uses first-person pronouns (I, my, me, mine), "
            f"the subject is {owner}. Use this name when writing facts and edges "
            f"about the user themselves.\n\n"
        ) + prompt

    domain_defs = _normalize_domain_defs(allowed_domains)
    if domain_defs:
        lines = ["", "AVAILABLE DOMAINS (use exact ids in facts[].domains):"]
        for domain_id, desc in sorted(domain_defs.items()):
            lines.append(f"- {domain_id}: {desc}" if desc else f"- {domain_id}")
        lines.extend([
            "",
            "DOMAIN OUTPUT CONTRACT (MANDATORY):",
            '- Every fact MUST include "domains": ["..."] with at least one allowed domain id.',
        ])
        prompt += "\n".join(lines) + "\n"

    project_defs = _normalize_project_defs(known_projects)
    if project_defs:
        lines = [
            "",
            "REGISTERED PROJECTS (use exact names as keys in project_logs — no other names are valid):",
        ]
        for project_name, desc in sorted(project_defs.items()):
            lines.append(f"- {project_name}: {desc}" if desc else f"- {project_name}")
        lines.extend([
            "",
            "PROJECT LOG CONTRACT (MANDATORY):",
            "- Only emit project_logs entries for projects listed above.",
            "- Use the exact project name as the key (case-sensitive).",
            "- If nothing noteworthy happened for a project, omit it from project_logs.",
        ])
        prompt += "\n".join(lines) + "\n"

    extra_appendix = str(os.environ.get("BENCHMARK_EXTRACTION_PROMPT_APPENDIX", "") or "").strip()
    if extra_appendix:
        prompt += "\n\n=== BENCHMARK EXTRACTION APPENDIX ===\n" + extra_appendix + "\n"
    return prompt

def call_anthropic(
    system_prompt: str,
    user_message: str,
    model: str,
    api_key: str,
    max_tokens: int = 16384,
) -> tuple[str, dict]:
    """Call Anthropic API and return the text response."""
    import urllib.request

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": _anthropic_system_blocks(system_prompt, api_key),
        "messages": [{"role": "user", "content": user_message}],
    }

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode(),
        headers=_anthropic_headers(api_key),
    )

    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read())

    text = data.get("content", [{}])[0].get("text", "").strip()
    usage = data.get("usage", {})
    return text, usage


def parse_extraction_response(raw: str) -> dict:
    """Parse JSON from LLM response, handling markdown fences."""
    text = raw.strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Repair the common truncation case where the model emits valid JSON
        # but drops one or more trailing closing braces/brackets at EOF.
        if text:
            repaired = text + ("]" * max(0, text.count("[") - text.count("]"))) + ("}" * max(0, text.count("{") - text.count("}")))
            if repaired != text:
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    pass
        # Try extracting the outermost JSON object
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {"facts": [], "soul_snippets": {}, "journal_entries": {}, "project_logs": {}}


def _resolve_quaid_dir(workspace: str) -> str:
    base = Path(workspace)
    candidates = [
        base / "modules" / "quaid",
        base / "plugins" / "quaid",
        base / "plugins" / "quaid" / "modules" / "quaid",
    ]
    for c in candidates:
        if (c / "quaid").exists() or (c / "memory_graph.py").exists() or (c / "datastore" / "memorydb" / "memory_graph.py").exists():
            return str(c)
    return str(candidates[0])


def _resolve_effective_memory_db_path(workspace: str) -> Path:
    """Resolve the active Quaid memory DB path for the current instance.

    Under instance isolation, the runtime DB lives under:
    `QUAID_HOME/instances/<instance>/data/memory.db`, not necessarily under the
    workspace root. Ask Quaid's own config helpers so the benchmark follows the
    product path instead of hardcoding the legacy workspace DB location.
    """
    quaid_dir = Path(_resolve_quaid_dir(workspace))
    config_path = quaid_dir / "lib" / "config.py"
    if config_path.is_file():
        spec = importlib.util.spec_from_file_location("_benchmark_quaid_config", config_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return Path(module.get_db_path())

    quaid_pkg_root = str(quaid_dir)
    if quaid_pkg_root not in sys.path:
        sys.path.insert(0, quaid_pkg_root)
    importlib.invalidate_caches()
    sys.modules.pop("lib.config", None)
    from lib.config import get_db_path  # type: ignore

    return Path(get_db_path())


def _load_runtime_allowed_domains(workspace: str) -> dict[str, str]:
    """Load active domain ids/descriptions from the live runtime state."""
    try:
        import sqlite3

        db_path = _resolve_effective_memory_db_path(workspace)
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT domain, description FROM domain_registry WHERE active = 1 ORDER BY domain"
            ).fetchall()
        domains = {
            str(domain).strip().lower(): str(description or "").strip()
            for domain, description in rows
            if str(domain).strip()
        }
        if domains:
            return domains
    except Exception:
        pass

    try:
        quaid_dir = _resolve_quaid_dir(workspace)
        quaid_pkg_root = str(Path(quaid_dir))
        if quaid_pkg_root not in sys.path:
            sys.path.insert(0, quaid_pkg_root)
        from config import get_config  # type: ignore

        raw = getattr(get_config().retrieval, "domains", {}) or {}
        if isinstance(raw, dict):
            return {
                str(domain).strip().lower(): str(description or "").strip()
                for domain, description in raw.items()
                if str(domain).strip()
            }
    except Exception:
        pass
    return {}


def _load_runtime_known_projects(workspace: str) -> dict[str, str]:
    """Load registered project names/descriptions from the runtime config."""
    try:
        quaid_dir = _resolve_quaid_dir(workspace)
        quaid_pkg_root = str(Path(quaid_dir))
        if quaid_pkg_root not in sys.path:
            sys.path.insert(0, quaid_pkg_root)
        from config import get_config  # type: ignore

        defs = getattr(get_config().projects, "definitions", {}) or {}
        if not isinstance(defs, dict):
            return {}
        out: dict[str, str] = {}
        for name, definition in defs.items():
            key = str(name).strip()
            if not key:
                continue
            out[key] = str(getattr(definition, "description", "") or "").strip()
        return out
    except Exception:
        return {}


def _memory_cmd(quaid_dir: str, *args: str) -> list[str]:
    qd = Path(quaid_dir)
    cli = qd / "quaid"
    if cli.exists():
        return ["/bin/bash", str(cli), *args]
    mg = qd / "memory_graph.py"
    if not mg.exists():
        mg = qd / "datastore" / "memorydb" / "memory_graph.py"
    return [sys.executable, str(mg), *args]


def store_fact(
    workspace: str,
    text: str,
    category: str = "fact",
    owner_id: str = "maya",
    confidence: float = 0.5,
    session_id: str | None = None,
    privacy: str = "shared",
    keywords: str | None = None,
    knowledge_type: str = "fact",
    source_type: str = "user",
    sensitivity: str | None = None,
    sensitivity_handling: str | None = None,
    domains: list[str] | None = None,
    project: str | None = None,
    created_at: str | None = None,
) -> dict | None:
    """Store a fact via memory_graph.py CLI and parse the result."""
    quaid_dir = _resolve_quaid_dir(workspace)
    cmd = _memory_cmd(
        quaid_dir,
        "store",
        text,
        "--category", category,
        "--owner", owner_id,
        "--confidence", str(confidence),
        "--extraction-confidence", str(confidence),
        "--privacy", privacy,
        "--knowledge-type", knowledge_type,
        "--source-type", source_type,
        "--source", "benchmark-extraction",
    )
    if session_id:
        cmd.extend(["--session-id", session_id])
    if keywords:
        cmd.extend(["--keywords", keywords])
    if domains:
        clean_domains = [str(d).strip().lower() for d in domains if str(d).strip()]
        if clean_domains:
            cmd.extend(["--domains", ",".join(dict.fromkeys(clean_domains))])
    if project:
        cmd.extend(["--project", project])
    if created_at:
        cmd.extend(["--created-at", created_at])
    if sensitivity:
        cmd.extend(["--sensitivity", sensitivity])
    if sensitivity_handling:
        cmd.extend(["--sensitivity-handling", sensitivity_handling])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
            cwd=quaid_dir,
        )
        # Check stderr for [config] lines (expected) vs real errors
        for stderr_line in (result.stderr or "").strip().split("\n"):
            if stderr_line and not stderr_line.startswith("[config]"):
                print(f"  [store stderr] {stderr_line}", file=sys.stderr)

        output = result.stdout.strip()

        stored = re.match(r"Stored: (.+)", output)
        if stored:
            return {"status": "created", "id": stored.group(1)}

        dup = re.match(r"Duplicate \(similarity: ([\d.]+)\) \[([^\]]+)\]: (.+)", output)
        if dup:
            return {"status": "duplicate", "similarity": float(dup.group(1)), "id": dup.group(2), "existing_text": dup.group(3)}
        # Fallback for old format without ID
        dup_old = re.match(r"Duplicate \(similarity: ([\d.]+)\): (.+)", output)
        if dup_old:
            return {"status": "duplicate", "similarity": float(dup_old.group(1)), "existing_text": dup_old.group(2)}

        updated = re.match(r"Updated existing: (.+)", output)
        if updated:
            return {"status": "updated", "id": updated.group(1)}

        if result.returncode != 0:
            print(f"  [store FAIL rc={result.returncode}] {result.stderr[:200]}", file=sys.stderr)
            detail = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"store_fact failed rc={result.returncode}: {detail[:400]}")
        return None
    except RuntimeError:
        raise
    except Exception as e:
        print(f"  [store exception] {e}", file=sys.stderr)
        return None


def create_edge(
    workspace: str,
    subject: str,
    relation: str,
    obj: str,
    source_fact_id: str | None = None,
    owner_id: str | None = None,
) -> bool:
    """Create an edge via memory_graph.py CLI."""
    quaid_dir = _resolve_quaid_dir(workspace)
    cmd = _memory_cmd(quaid_dir, "create-edge",
        subject, relation, obj,
        "--create-missing", "--json",
    )
    if source_fact_id:
        cmd.extend(["--source-fact-id", source_fact_id])
    if owner_id:
        cmd.extend(["--owner", owner_id])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
            cwd=quaid_dir,
        )
        # Log ALL stderr (not just errors) for debugging
        for stderr_line in (result.stderr or "").strip().split("\n"):
            if stderr_line and not stderr_line.startswith("[config]"):
                print(f"  [edge stderr] {stderr_line}", file=sys.stderr)

        if result.returncode != 0:
            print(f"  [edge FAIL rc={result.returncode}] stdout={result.stdout[:100]} stderr={result.stderr[:100]}", file=sys.stderr)
            return False

        # Parse JSON from stdout (skip [config] log lines)
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    data = json.loads(line)
                    created = data.get("status") == "created"
                    if not created:
                        print(f"  [edge status={data.get('status')}] {line[:100]}", file=sys.stderr)
                    return created
                except json.JSONDecodeError:
                    continue

        print(f"  [edge no JSON] stdout={result.stdout[:200]}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"  [edge exception] {e}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Snippet & Journal file writing (matches production format)
# ---------------------------------------------------------------------------

def write_snippet_entry(
    workspace: str,
    filename: str,
    snippets: list[str],
    trigger: str = "Compaction",
    date_str: str | None = None,
    time_str: str | None = None,
) -> bool:
    """Write snippet bullets to {filename}.snippets.md in workspace root.

    Format matches production:
        # {FILENAME} — Pending Snippets

        ## Compaction — 2026-02-16 14:30:22
        - Snippet bullet 1
        - Snippet bullet 2
    """
    if not snippets:
        return False

    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    time_str = time_str or datetime.now().strftime("%H:%M:%S")

    base_name = filename.removesuffix(".md")
    filepath = Path(workspace) / f"{base_name}.snippets.md"
    header = f"## {trigger} — {date_str} {time_str}"
    bullets = "\n".join(f"- {s}" for s in snippets)
    entry = f"\n{header}\n{bullets}\n"

    if filepath.exists():
        content = filepath.read_text()
        # Dedup: skip if same trigger+date already exists
        dedup_key = f"## {trigger} — {date_str}"
        if dedup_key in content:
            return False
        # Insert after first heading line
        lines = content.split("\n")
        insert_idx = 1  # After title line
        for i, line in enumerate(lines):
            if line.startswith("# "):
                insert_idx = i + 1
                break
        lines.insert(insert_idx, entry)
        filepath.write_text("\n".join(lines))
    else:
        title = f"# {filename} — Pending Snippets\n"
        filepath.write_text(title + entry)

    return True


def write_journal_entry(
    workspace: str,
    filename: str,
    content: str,
    trigger: str = "Compaction",
    date_str: str | None = None,
) -> bool:
    """Write journal paragraph to journal/{filename}.journal.md.

    Format matches production:
        # {FILENAME} Journal

        ## 2026-02-16 — Compaction
        Reflective paragraph text here. Can be multiple paragraphs.
    """
    if not content or not content.strip():
        return False

    date_str = date_str or datetime.now().strftime("%Y-%m-%d")

    base_name = filename.removesuffix(".md")
    journal_dir = Path(workspace) / "journal"
    journal_dir.mkdir(parents=True, exist_ok=True)
    filepath = journal_dir / f"{base_name}.journal.md"

    header = f"## {date_str} — {trigger}"
    entry = f"\n{header}\n{content.strip()}\n"

    if filepath.exists():
        existing = filepath.read_text()
        # Dedup: skip if same date+trigger already exists
        if f"## {date_str} — {trigger}" in existing:
            return False
        # Insert after first heading line
        lines = existing.split("\n")
        insert_idx = 1
        for i, line in enumerate(lines):
            if line.startswith("# "):
                insert_idx = i + 1
                break
        lines.insert(insert_idx, entry)
        filepath.write_text("\n".join(lines))
    else:
        title = f"# {filename} Journal\n"
        filepath.write_text(title + entry)

    return True


def write_project_logs(
    workspace: str,
    project_logs: dict,
    trigger: str = "Compaction",
    date_str: str | None = None,
    quaid_instance: str | None = None,
) -> dict:
    """Append project logs to PROJECT.md via core project_updater."""
    if not isinstance(project_logs, dict) or not project_logs:
        return {}

    def _normalize_entry(entry):
        if isinstance(entry, dict):
            text = str(
                entry.get("text", entry.get("entry", entry.get("note", "")))
            ).strip()
            if not text:
                return None
            normalized = {"text": text}
            created_at = str(
                entry.get("created_at", entry.get("timestamp", entry.get("date", "")))
            ).strip()
            if created_at:
                normalized["created_at"] = created_at
            return normalized
        text = str(entry).strip()
        if not text:
            return None
        return text

    normalized = {}
    for raw_name, raw_entries in project_logs.items():
        name = str(raw_name).strip()
        if not name:
            continue
        entries = raw_entries if isinstance(raw_entries, list) else [raw_entries]
        cleaned = []
        seen = set()
        for entry in entries:
            normalized_entry = _normalize_entry(entry)
            if not normalized_entry:
                continue
            if isinstance(normalized_entry, dict):
                dedupe_key = (
                    normalized_entry["text"],
                    normalized_entry.get("created_at", ""),
                )
            else:
                dedupe_key = normalized_entry
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            cleaned.append(normalized_entry)
        if cleaned:
            normalized[name] = cleaned
    if not normalized:
        return {}

    quaid_dir = ""
    explicit = os.environ.get("BENCHMARK_PLUGIN_DIR", "").strip()
    if explicit and Path(explicit).exists():
        quaid_dir = explicit
    if not quaid_dir:
        clawd_ws = os.environ.get("CLAWDBOT_WORKSPACE", "").strip()
        candidates = [
            Path(workspace) / "modules" / "quaid",
            Path(workspace) / "plugins" / "quaid",
            Path(workspace) / "benchmark-checkpoint" / "modules" / "quaid",
            Path(workspace) / "benchmark-checkpoint" / "plugins" / "quaid",
            Path(clawd_ws) / "modules" / "quaid" if clawd_ws else None,
            Path(clawd_ws) / "plugins" / "quaid" if clawd_ws else None,
            Path(clawd_ws) / "benchmark-checkpoint" / "modules" / "quaid" if clawd_ws else None,
            Path(clawd_ws) / "benchmark-checkpoint" / "plugins" / "quaid" if clawd_ws else None,
        ]
        script_parents = list(Path(__file__).resolve().parents)
        for base in script_parents[2:4]:
            candidates.extend([
                base / "modules" / "quaid",
                base / "plugins" / "quaid",
                base / "benchmark-checkpoint" / "modules" / "quaid",
                base / "benchmark-checkpoint" / "plugins" / "quaid",
            ])
        for candidate in candidates:
            if candidate and candidate.exists():
                quaid_dir = str(candidate)
                break
    if not quaid_dir:
        raise RuntimeError("Project log append failed: unable to resolve quaid module path")

    quaid_pkg_root = str(Path(quaid_dir))
    if quaid_pkg_root not in sys.path:
        sys.path.insert(0, quaid_pkg_root)

    with _PROJECT_UPDATER_ENV_LOCK:
        prev_workspace = os.environ.get("CLAWDBOT_WORKSPACE")
        prev_quaid_home = os.environ.get("QUAID_HOME")
        prev_memory_db = os.environ.get("MEMORY_DB_PATH")
        prev_instance = os.environ.get("QUAID_INSTANCE")
        try:
            os.environ["CLAWDBOT_WORKSPACE"] = workspace
            os.environ["QUAID_HOME"] = workspace
            os.environ["MEMORY_DB_PATH"] = str(_resolve_effective_memory_db_path(workspace))
            resolved_instance = str(quaid_instance or os.environ.get("QUAID_INSTANCE", "")).strip()
            if not resolved_instance:
                raise RuntimeError(
                    "Project log append failed: QUAID_INSTANCE environment variable is not set"
                )
            os.environ["QUAID_INSTANCE"] = resolved_instance
            import datastore.docsdb.project_updater as _project_updater  # type: ignore

            append_fn = getattr(_project_updater, "append_project_logs", None)
            if callable(append_fn):
                metrics = append_fn(
                    normalized,
                    trigger=trigger,
                    date_str=date_str,
                    dry_run=False,
                )
                return metrics if isinstance(metrics, dict) else {}

            legacy_append_fn = getattr(_project_updater, "append_project_log_entries", None)
            if callable(legacy_append_fn):
                # Legacy checkpoint signature: append_project_log_entries(project_name, entries)
                metrics = {
                    "projects_seen": len(normalized),
                    "projects_updated": 0,
                    "entries_seen": sum(len(v) for v in normalized.values()),
                    "entries_written": 0,
                    "projects_unknown": 0,
                    "projects_missing_file": 0,
                }
                for project_name, entries in normalized.items():
                    legacy_entries = [
                        entry["text"] if isinstance(entry, dict) else entry
                        for entry in entries
                    ]
                    try:
                        written = int(legacy_append_fn(project_name, legacy_entries) or 0)
                        metrics["entries_written"] += max(0, written)
                        if written > 0:
                            metrics["projects_updated"] += 1
                    except Exception as exc:
                        raise RuntimeError(
                            f"Legacy project log append failed for {project_name}: {exc}"
                        ) from exc
                return metrics

            raise RuntimeError(
                "Project log append failed: no append_project_logs/append_project_log_entries symbol"
            )
        finally:
            if prev_workspace is None:
                os.environ.pop("CLAWDBOT_WORKSPACE", None)
            else:
                os.environ["CLAWDBOT_WORKSPACE"] = prev_workspace
            if prev_quaid_home is None:
                os.environ.pop("QUAID_HOME", None)
            else:
                os.environ["QUAID_HOME"] = prev_quaid_home
            if prev_memory_db is None:
                os.environ.pop("MEMORY_DB_PATH", None)
            else:
                os.environ["MEMORY_DB_PATH"] = prev_memory_db
            if prev_instance is None:
                os.environ.pop("QUAID_INSTANCE", None)
            else:
                os.environ["QUAID_INSTANCE"] = prev_instance


def truncate_session(session_file: str, summary: str | None = None):
    """Truncate the session JSONL, optionally keeping a summary message."""
    lines = []
    if summary:
        # Write a system summary message as the session start
        entry = {
            "type": "message",
            "message": {
                "role": "user",
                "content": f"[Previous conversation summary]\n{summary}",
            },
        }
        lines.append(json.dumps(entry))
    with open(session_file, "w") as f:
        f.write("\n".join(lines) + "\n" if lines else "")


def create_summary(
    transcript: str,
    model: str,
    api_key: str,
) -> str:
    """Create a brief summary of the conversation for session continuity."""
    prompt = (
        "Summarize the key topics, decisions, and context from this conversation "
        "in 3-5 bullet points. Be concise but preserve important details that "
        "would help continue the conversation later."
    )
    try:
        # Use a cheap/fast model for summarization
        summary_model = model
        if "opus" in model or "sonnet" in model:
            summary_model = "claude-haiku-4-5-20251001"
        text, _usage = call_anthropic(
            prompt,
            f"Conversation to summarize:\n\n{transcript[:50000]}",
            summary_model,
            api_key,
            max_tokens=1024,
        )
        return text
    except Exception as e:
        print(f"  Summary failed: {e}", file=sys.stderr)
        return "Previous conversation context was compacted."


def main():
    parser = argparse.ArgumentParser(description="Extract facts from session and compact")
    parser.add_argument("--session-file", required=True, help="Path to session JSONL")
    parser.add_argument("--workspace", required=True, help="Workspace directory (e.g., ~/clawd)")
    parser.add_argument("--user-name", default="Maya", help="User name for extraction prompt")
    parser.add_argument("--agent-name", default="Assistant", help="Agent name for transcript")
    parser.add_argument("--owner-id", default=_DEFAULT_OWNER_ID, help="Owner ID for stored facts")
    parser.add_argument("--session-id", default=None, help="Session ID for facts")
    parser.add_argument("--model", default="claude-sonnet-4-5-20250929", help="Extraction model")
    parser.add_argument("--no-truncate", action="store_true", help="Don't truncate session file")
    parser.add_argument("--no-summary", action="store_true", help="Don't create summary")
    parser.add_argument("--trigger", default="Compaction", help="Trigger label (Compaction/Reset)")
    parser.add_argument("--date", default=None, help="Simulated date (YYYY-MM-DD) for snippets/journal")
    args = parser.parse_args()

    workspace = os.path.expanduser(args.workspace)
    session_file = os.path.expanduser(args.session_file)

    if not os.path.exists(session_file):
        print(f"Session file not found: {session_file}", file=sys.stderr)
        sys.exit(1)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    messages = read_session_messages(session_file)
    if not messages:
        print("No messages in session file")
        return

    t0 = time.time()
    native = _run_native_daemon_compaction(
        session_file=session_file,
        workspace=workspace,
        session_id=args.session_id or Path(session_file).stem,
        trigger=args.trigger,
        sim_date=args.date,
    )
    elapsed = time.time() - t0
    transcript = str(native.get("transcript") or "")
    if not transcript.strip():
        print("Empty transcript after filtering")
        return

    print(f"Transcript: {len(messages)} messages, {len(transcript)} chars")
    usage = dict(native.get("usage") or {})
    metric = dict(native.get("metric") or {})
    in_tok = int(usage.get("input_tokens", 0) or 0)
    out_tok = int(usage.get("output_tokens", 0) or 0)
    print(
        f"Extraction API call: {elapsed:.1f}s, {in_tok} in + {out_tok} out tokens"
        f" ({int(usage.get('calls', 0) or 0)} calls)"
    )

    stored = int(metric.get("final_facts_stored", 0) or 0)
    skipped = int(metric.get("final_facts_skipped", 0) or 0)
    edges_created = int(metric.get("final_edges_created", 0) or 0)
    edge_errors = 0

    print(f"Extraction complete: {stored} stored, {skipped} skipped, {edges_created} edges", end="")
    if edge_errors:
        print(f", {edge_errors} edge errors", end="")
    print()

    # Verify DB state after extraction
    try:
        import sqlite3
        db_path = _resolve_effective_memory_db_path(workspace)
        with sqlite3.connect(db_path) as conn:
            db_edges = conn.execute("SELECT count(*) FROM edges").fetchone()[0]
            db_nodes = conn.execute("SELECT count(*) FROM nodes").fetchone()[0]
            status_counts = dict(conn.execute(
                "SELECT status, count(*) FROM nodes GROUP BY status"
            ).fetchall())
        print(f"DB verify: {db_nodes} nodes, {db_edges} edges, status={status_counts}")
        if edges_created > 0 and db_edges == 0:
            print(f"WARNING: Extraction reported {edges_created} edges but DB has 0!", file=sys.stderr)
    except Exception as e:
        print(f"DB verify failed: {e}", file=sys.stderr)

    snippets_written = int(metric.get("snippets_count", 0) or 0)
    journals_written = int(metric.get("journals_count", 0) or 0)
    if snippets_written:
        print(f"  Snippets: {snippets_written} files updated")
    if journals_written:
        print(f"  Journal: {journals_written} files updated")

    project_log_metrics = {
        "entries_seen": int(metric.get("project_logs_seen", 0) or 0),
        "entries_written": int(metric.get("project_logs_written", 0) or 0),
        "projects_updated": int(metric.get("project_logs_projects_updated", 0) or 0),
    }
    if project_log_metrics:
        print(
            "  Project logs: "
            f"seen={project_log_metrics.get('entries_seen', 0)} "
            f"written={project_log_metrics.get('entries_written', 0)} "
            f"projects_updated={project_log_metrics.get('projects_updated', 0)}"
        )

    # Output JSON result for the benchmark runner to parse
    print(json.dumps({
        "stored": stored,
        "skipped": skipped,
        "edges": edges_created,
        "edge_errors": edge_errors,
        "snippets_written": snippets_written,
        "journals_written": journals_written,
        "project_log_metrics": project_log_metrics,
        "total_candidates": None,
        "extraction_usage": {
            "input_tokens": int(usage.get("input_tokens", 0) or 0),
            "output_tokens": int(usage.get("output_tokens", 0) or 0),
            "calls": int(usage.get("calls", 0) or 0),
            "fast_calls": int(usage.get("fast_calls", 0) or 0),
            "deep_calls": int(usage.get("deep_calls", 0) or 0),
            "model": args.model,
        },
        **(
            {
                "artifact": {
                    "session_file": session_file,
                    "session_id": args.session_id,
                    "date": args.date,
                    "trigger": args.trigger,
                    "model": args.model,
                    "messages_count": len(messages),
                    "transcript": transcript,
                    "rolling_metric": metric,
                }
            }
            if _benchmark_emit_artifact()
            else {}
        ),
    }))


if __name__ == "__main__":
    main()
