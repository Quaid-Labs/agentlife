import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_production_benchmark as rpb  # noqa: E402


class _ProcResult:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_store_facts_retries_edge_timeout_and_succeeds(monkeypatch, tmp_path):
    monkeypatch.setattr(rpb, "_MEMORY_GRAPH_SCRIPT", tmp_path / "dummy.py")
    monkeypatch.setattr(rpb, "_load_active_domain_ids", lambda _ws: ["personal", "project", "work", "technical"])

    calls = {"n": 0}

    def _run(cmd, **kwargs):
        calls["n"] += 1
        # First command is store; second is edge timeout; third is edge success.
        if calls["n"] == 1:
            return _ProcResult(returncode=0, stdout="Stored: fact-1\n")
        if calls["n"] == 2:
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=30)
        return _ProcResult(returncode=0, stdout='{"status":"created"}\n')

    monkeypatch.setattr(rpb.subprocess, "run", _run)
    monkeypatch.setenv("BENCHMARK_EDGE_RETRIES", "1")
    monkeypatch.setenv("BENCHMARK_EDGE_RETRY_BACKOFF_SECONDS", "0")
    monkeypatch.setenv("BENCHMARK_EDGE_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("BENCHMARK_FAIL_ON_STORE_FAILURE", "1")

    facts = [{
        "text": "Maya works at TechFlow as a product manager",
        "category": "fact",
        "privacy": "shared",
        "domains": ["work"],
        "edges": [{"subject": "Maya", "relation": "works_at", "object": "TechFlow"}],
    }]

    stored, edges = rpb._store_facts(tmp_path, facts, os.environ.copy(), 1, "2026-03-01")
    assert stored == 1
    assert edges == 1
    assert calls["n"] == 3


def test_store_facts_counts_edge_nonzero_as_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(rpb, "_MEMORY_GRAPH_SCRIPT", tmp_path / "dummy.py")
    monkeypatch.setattr(rpb, "_load_active_domain_ids", lambda _ws: ["personal", "project", "work", "technical"])

    calls = {"n": 0}

    def _run(cmd, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _ProcResult(returncode=0, stdout="Stored: fact-1\n")
        return _ProcResult(returncode=2, stdout="", stderr="edge failed")

    monkeypatch.setattr(rpb.subprocess, "run", _run)
    monkeypatch.setenv("BENCHMARK_EDGE_RETRIES", "0")
    monkeypatch.setenv("BENCHMARK_FAIL_ON_STORE_FAILURE", "1")

    facts = [{
        "text": "Maya works at TechFlow as a product manager",
        "category": "fact",
        "privacy": "shared",
        "domains": ["work"],
        "edges": [{"subject": "Maya", "relation": "works_at", "object": "TechFlow"}],
    }]

    with pytest.raises(RuntimeError, match="Store phase encountered failures"):
        rpb._store_facts(tmp_path, facts, os.environ.copy(), 1, "2026-03-01")


def test_store_facts_retries_store_timeout_and_succeeds(monkeypatch, tmp_path):
    monkeypatch.setattr(rpb, "_MEMORY_GRAPH_SCRIPT", tmp_path / "dummy.py")
    monkeypatch.setattr(rpb, "_load_active_domain_ids", lambda _ws: ["personal", "project", "work", "technical"])

    calls = {"n": 0}

    def _run(cmd, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=90)
        return _ProcResult(returncode=0, stdout="Stored: fact-1\n")

    monkeypatch.setattr(rpb.subprocess, "run", _run)
    monkeypatch.setenv("BENCHMARK_STORE_RETRIES", "1")
    monkeypatch.setenv("BENCHMARK_STORE_RETRY_BACKOFF_SECONDS", "0")
    monkeypatch.setenv("BENCHMARK_STORE_TIMEOUT_SECONDS", "90")
    monkeypatch.setenv("BENCHMARK_FAIL_ON_STORE_FAILURE", "1")

    facts = [{
        "text": "Maya has a dog named Biscuit",
        "category": "fact",
        "privacy": "shared",
        "domains": ["personal"],
        "edges": [],
    }]

    stored, edges = rpb._store_facts(tmp_path, facts, os.environ.copy(), 6, "2026-03-14")
    assert stored == 1
    assert edges == 0
    assert calls["n"] == 2
