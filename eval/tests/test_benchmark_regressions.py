"""Regression & unit tests for the benchmark harness.

Covers pure/semi-pure functions in run_production_benchmark.py and
extract_compact.py. No network or subprocess calls — everything mocked.
"""

import json
import io
import hashlib
import importlib
import importlib.util
import os
import re
import sqlite3
import subprocess
import sys
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from typing import Optional

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_production_benchmark as rpb  # noqa: E402
import extract_compact as ec  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeReview:
    session_num: int
    track: int = 1
    timestamp: str = ""


class _FakeSubprocessResult:
    returncode = 0
    stdout = ""
    stderr = ""


def test_base_project_md_scaffold_does_not_seed_future_project_facts():
    text = rpb._render_base_project_md(
        label="Recipe App",
        description=(
            "Recipe app project workspace. Current facts, features, stack, "
            "and motivations should be learned from source artifacts and conversations."
        ),
        project_home="projects/recipe-app/",
        source_roots=["project-sources/recipe-app/"],
        exclude_patterns=["node_modules/", "*.db"],
    )

    assert "# Project: Recipe App" in text
    assert "projects/recipe-app/" in text
    assert "project-sources/recipe-app/" in text
    assert "source artifacts and conversations" in text
    assert "Safe for Mom" not in text
    assert "Linda" not in text
    assert "GraphQL API" not in text
    assert "TechFlow" not in text


def test_project_docs_enabled_does_not_stub_support_docs(monkeypatch, tmp_path):
    """Ordinary benchmark apps should not get TOOLS.md/AGENTS.md placeholders."""
    workspace = tmp_path / "ws"
    events = []

    def _fake_register(seed_workspace: Path) -> None:
        events.append("register")
        assert not (seed_workspace / "projects" / "recipe-app" / "PROJECT.md").exists()
        assert not (seed_workspace / "projects" / "portfolio-site" / "PROJECT.md").exists()

    def _fake_seed_quaid(seed_workspace: Path) -> None:
        events.append("seed_quaid")
        assert events[0] == "register"

    def _fake_seed_identity(seed_workspace: Path, **_kwargs) -> None:
        events.append("seed_identity")
        assert events[0] == "register"

    def _fake_supervisor(seed_workspace: Path) -> None:
        events.append("supervisor")

    monkeypatch.delenv("BENCHMARK_DISABLE_PROJECT_DOCS", raising=False)
    monkeypatch.setattr(rpb, "_seed_quaid_project_docs", _fake_seed_quaid)
    monkeypatch.setattr(rpb, "_seed_instance_identity_from_sources", _fake_seed_identity)
    monkeypatch.setattr(rpb, "_register_benchmark_projects", _fake_register)
    monkeypatch.setattr(rpb, "_ensure_project_docs_supervisor_running", _fake_supervisor)

    rpb.setup_workspace(workspace)

    assert events == ["register", "seed_quaid", "seed_identity", "supervisor"]
    for project in ("recipe-app", "portfolio-site"):
        project_dir = workspace / "projects" / project
        assert (project_dir / "PROJECT.md").is_file()
        assert not (project_dir / "TOOLS.md").exists()
        assert not (project_dir / "AGENTS.md").exists()


def test_benchmark_project_sources_do_not_preseed_docs():
    """Project-doc quality must not be propped up by fixture README/API docs."""
    filter_args = rpb._PROJECT_SOURCE_RSYNC_FILTER
    assert "README.*" in filter_args
    assert "*.md" in filter_args
    assert "*.mdx" in filter_args
    assert "docs/" in filter_args

    for patterns in rpb._BENCHMARK_PROJECT_SOURCE_PATTERNS.values():
        assert not any(pattern.lower().endswith((".md", ".mdx")) for pattern in patterns)


def test_benchmark_project_fixture_tree_has_no_preseeded_docs():
    """Keep generated-doc benchmarks honest: source fixtures are code/test/data only."""
    fixture_roots = [
        ROOT.parent / "apps" / "recipe-app",
        ROOT.parent / "benchmark-assets" / "projects" / "recipe-app",
        ROOT.parent / "benchmark-assets" / "projects" / "portfolio-site",
    ]
    leaked_docs = []
    for root in fixture_roots:
        if not root.exists():
            continue
        leaked_docs.extend(
            str(path.relative_to(ROOT.parent))
            for path in root.rglob("*")
            if path.is_file()
            and (path.suffix.lower() in {".md", ".mdx"} or "docs" in path.relative_to(root).parts)
        )

    assert leaked_docs == []


def test_project_docs_disabled_ablation_does_not_seed_project_docs(monkeypatch, tmp_path):
    """The disabled A/B lane should be memory-only, not hidden scaffold docs."""
    workspace = tmp_path / "ws"
    monkeypatch.setenv("BENCHMARK_DISABLE_PROJECT_DOCS", "1")
    monkeypatch.setattr(
        rpb,
        "_seed_quaid_project_docs",
        lambda *_a, **_k: pytest.fail("quaid project docs should not be seeded"),
    )
    monkeypatch.setattr(
        rpb,
        "_register_benchmark_projects",
        lambda *_a, **_k: pytest.fail("benchmark projects should not be registered"),
    )
    monkeypatch.setattr(
        rpb,
        "_ensure_project_docs_supervisor_running",
        lambda *_a, **_k: pytest.fail("project docs supervisor should not start"),
    )

    rpb.setup_workspace(workspace)

    config = json.loads((workspace / "config" / "memory.json").read_text())
    assert config["systems"]["projects"] is False
    assert config["projects"]["definitions"] == {}
    assert not (workspace / "projects" / "recipe-app" / "PROJECT.md").exists()
    assert not (workspace / "projects" / "portfolio-site" / "PROJECT.md").exists()


def test_project_docs_disabled_ablation_fails_on_runtime_project_log_metrics(monkeypatch):
    monkeypatch.setenv("BENCHMARK_DISABLE_PROJECT_DOCS", "1")

    with pytest.raises(RuntimeError, match="Project docs disabled ablation was tainted"):
        rpb._fail_if_project_docs_disabled_metrics(
            {"entries_seen": 2, "entries_written": 1, "projects_updated": 1},
            phase="runtime extraction",
        )


def test_project_source_repo_ignores_stale_remote_root_dirs(monkeypatch, tmp_path):
    """Launcher cleanup should not be required for correctness of source resolution."""
    project_root = tmp_path / "benchmark-root"
    assets_root = tmp_path / "assets"
    clawd_root = tmp_path / "clawd"
    stale_root_repo = project_root / "recipe-app"
    source_repo = clawd_root / "recipe-app"
    stale_root_repo.mkdir(parents=True)
    source_repo.mkdir(parents=True)

    monkeypatch.setattr(rpb, "_PROJECT_DIR", project_root)
    monkeypatch.setattr(rpb, "_CLAWD", clawd_root)
    monkeypatch.setattr(rpb, "_resolve_assets_dir", lambda: assets_root)

    assert rpb._resolve_project_source_repo("recipe-app") == source_repo


def test_project_source_change_waits_for_product_supervisor_freshness(monkeypatch, tmp_path):
    workspace = tmp_path / "ws"
    (workspace / "projects" / "recipe-app").mkdir(parents=True)

    calls = []
    monkeypatch.setattr(
        rpb,
        "_ensure_project_docs_supervisor_running",
        lambda *a, **k: calls.append(("supervisor", a, k)),
    )
    monkeypatch.setattr(
        rpb,
        "_wait_project_docs_fresh",
        lambda *a, **k: {"fresh": True, "status": "fresh", "current_shadow_head": "abc", "docs_cursor_head": "abc"},
    )
    monkeypatch.setattr(
        rpb,
        "_collect_project_docs_artifacts",
        lambda *a, **k: calls.append(("collect", a, k)),
    )

    result = rpb._handle_project_source_changed(workspace, "recipe-app", 3)

    assert result["mode"] == "supervisor"
    assert result["status"]["fresh"] is True
    assert [c[0] for c in calls] == ["supervisor", "collect"]


def test_wait_project_docs_fresh_accepts_no_pending_work_without_shadow_cursor(monkeypatch, tmp_path):
    workspace = tmp_path / "ws"
    status = {
        "project": "recipe-app",
        "status": "fresh",
        "fresh": True,
        "source_root": str(workspace / "project-sources" / "recipe-app"),
        "current_shadow_head": None,
        "docs_cursor_head": None,
        "source_error": None,
        "pending_source_changes": [],
        "pending_source_change_count": 0,
        "project_log_bytes_pending": 0,
        "worker_heartbeat": {"status": "idle", "heartbeat_at": "2026-04-24T00:00:00+00:00"},
    }
    monkeypatch.setattr(rpb, "_project_docs_wait_timeout_seconds", lambda: 1)
    monkeypatch.setattr(rpb, "_project_docs_poll_seconds", lambda: 0)
    monkeypatch.setattr(rpb, "_project_status_json", lambda *_a, **_k: status)

    result = rpb._wait_project_docs_fresh(workspace, "recipe-app", 157)

    assert result is status


def test_register_benchmark_projects_uses_product_project_registry(monkeypatch, tmp_path):
    workspace = tmp_path / "ws"
    (workspace / "projects" / "recipe-app").mkdir(parents=True)
    (workspace / "projects" / "portfolio-site").mkdir(parents=True)
    (workspace / "project-sources" / "recipe-app").mkdir(parents=True)
    (workspace / "project-sources" / "portfolio-site").mkdir(parents=True)

    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append((list(cmd), kwargs))
        return SimpleNamespace(stdout="Created project\n", stderr="", returncode=0)

    monkeypatch.setattr(rpb, "_benchmark_env", lambda _workspace, _phase: {"QUAID_HOME": str(workspace)})
    monkeypatch.setattr(rpb, "_python_cmd_for_quaid_script", lambda _script: [sys.executable, "project_registry_cli.py"])
    monkeypatch.setattr(rpb, "_QUAID_DIR", tmp_path)
    monkeypatch.setattr(rpb.subprocess, "run", _fake_run)

    rpb._register_benchmark_projects(workspace)

    actions = [(call[0][2], call[0][3]) for call in calls]
    assert actions == [
        ("create", "recipe-app"),
        ("link", "recipe-app"),
        ("create", "portfolio-site"),
        ("link", "portfolio-site"),
    ]
    for cmd, kwargs in calls:
        assert cmd[:2] == [sys.executable, "project_registry_cli.py"]
        if cmd[2] == "create":
            assert cmd[1:3] == ["project_registry_cli.py", "create"]
            assert "--description" in cmd
            assert "--source-root" in cmd
            source_root = Path(cmd[cmd.index("--source-root") + 1])
            assert source_root.is_absolute()
            assert "project-sources" in source_root.parts
            assert "projects" not in source_root.relative_to(workspace).parts
        else:
            assert cmd[1:3] == ["project_registry_cli.py", "link"]
        assert kwargs["env"]["QUAID_HOME"] == str(workspace)
        assert kwargs["cwd"] == str(tmp_path)
        assert kwargs["timeout"] == 120


def test_register_benchmark_projects_updates_and_links_existing_projects(monkeypatch, tmp_path):
    workspace = tmp_path / "ws"
    (workspace / "projects" / "recipe-app").mkdir(parents=True)
    (workspace / "projects" / "portfolio-site").mkdir(parents=True)
    (workspace / "project-sources" / "recipe-app").mkdir(parents=True)
    (workspace / "project-sources" / "portfolio-site").mkdir(parents=True)

    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append((list(cmd), kwargs))
        action = cmd[2]
        if action == "create":
            return SimpleNamespace(stdout="", stderr="Error: Project already exists: " + cmd[3], returncode=1)
        return SimpleNamespace(stdout="ok\n", stderr="", returncode=0)

    monkeypatch.setattr(rpb, "_benchmark_env", lambda _workspace, _phase: {"QUAID_HOME": str(workspace)})
    monkeypatch.setattr(rpb, "_python_cmd_for_quaid_script", lambda _script: [sys.executable, "project_registry_cli.py"])
    monkeypatch.setattr(rpb, "_QUAID_DIR", tmp_path)
    monkeypatch.setattr(rpb.subprocess, "run", _fake_run)

    rpb._register_benchmark_projects(workspace)

    actions = [(call[0][2], call[0][3]) for call in calls]
    assert actions == [
        ("create", "recipe-app"),
        ("link", "recipe-app"),
        ("update", "recipe-app"),
        ("create", "portfolio-site"),
        ("link", "portfolio-site"),
        ("update", "portfolio-site"),
    ]
    for cmd, _kwargs in calls:
        if cmd[2] == "update":
            assert "--source-root" in cmd


def _fake_updater_module(fn_name="append_project_logs", fn=None):
    """Create fake datastore.docsdb.project_updater module hierarchy."""
    datastore_mod = ModuleType("datastore")
    docsdb_mod = ModuleType("datastore.docsdb")
    updater_mod = ModuleType("datastore.docsdb.project_updater")
    if fn is not None:
        setattr(updater_mod, fn_name, fn)
    return datastore_mod, docsdb_mod, updater_mod


def _load_imported_claude_history_module() -> ModuleType:
    script_path = ROOT.parent / "scripts" / "run-imported-claude-history.py"
    spec = importlib.util.spec_from_file_location("run_imported_claude_history", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _stub_dataset_version_gate(monkeypatch):
    monkeypatch.setattr(rpb, "_enforce_dataset_version", lambda _assets: ("v-test", 268))
    monkeypatch.setattr(rpb, "_read_dataset_version", lambda _assets: "v-test")


# ===================================================================
# run_production_benchmark.py — Pure Functions
# ===================================================================


class TestNormalizeDomainList:
    """Tests for _normalize_domain_list: alias, dedup, order, edge cases."""

    def test_preserves_order_and_aliases(self):
        domains = ["projects", "technical", "projects", "personal", "PROJECTS", "work"]
        out = rpb._normalize_domain_list(domains)
        assert out == ["project", "technical", "personal", "work"]

    def test_empty_input(self):
        assert rpb._normalize_domain_list([]) == []

    def test_whitespace_only_entries_skipped(self):
        assert rpb._normalize_domain_list(["", "  ", "\t"]) == []

    def test_mixed_case_normalized(self):
        out = rpb._normalize_domain_list(["TECHNICAL", "Technical", "technical"])
        assert out == ["technical"]

    def test_unknown_domains_pass_through(self):
        out = rpb._normalize_domain_list(["finance", "custom_domain"])
        assert out == ["finance", "custom_domain"]

    def test_alias_projects_to_project(self):
        out = rpb._normalize_domain_list(["projects"])
        assert out == ["project"]

    def test_alias_financial_to_finance(self):
        out = rpb._normalize_domain_list(["financial", "work", "finance"])
        assert out == ["finance", "work"]

    def test_alias_relationship_and_family_to_personal(self):
        out = rpb._normalize_domain_list(["relationship", "family", "personal"])
        assert out == ["personal"]

    def test_project_and_projects_dedup(self):
        out = rpb._normalize_domain_list(["project", "projects"])
        assert out == ["project"]

    def test_single_entry(self):
        out = rpb._normalize_domain_list(["health"])
        assert out == ["health"]


class TestPhaseTimingHelpers:
    def test_format_elapsed_short(self):
        assert rpb._format_elapsed(7.2) == "7s"

    def test_format_elapsed_minutes(self):
        assert rpb._format_elapsed(453.0) == "7m 33s"

    def test_format_elapsed_hours(self):
        assert rpb._format_elapsed(3723.0) == "1h 2m 3s"


class TestNormalizeProjectLogs:
    """Tests for _normalize_project_logs: dict normalization, dedup, edge cases."""

    def test_basic_normalization(self):
        raw = {"recipe-app": ["note one", "note two"]}
        out = rpb._normalize_project_logs(raw)
        assert out == {"recipe-app": ["note one", "note two"]}

    def test_non_dict_returns_empty(self):
        assert rpb._normalize_project_logs("not a dict") == {}
        assert rpb._normalize_project_logs(None) == {}
        assert rpb._normalize_project_logs(42) == {}

    def test_empty_dict_returns_empty(self):
        assert rpb._normalize_project_logs({}) == {}

    def test_single_entry_wrapped_in_list(self):
        out = rpb._normalize_project_logs({"app": "single note"})
        assert out == {"app": ["single note"]}

    def test_deduplicates_entries(self):
        out = rpb._normalize_project_logs({"app": ["dup", "dup", "unique"]})
        assert out == {"app": ["dup", "unique"]}

    def test_blank_entries_stripped(self):
        out = rpb._normalize_project_logs({"app": ["real", "", "  ", "also real"]})
        assert out == {"app": ["real", "also real"]}

    def test_blank_project_name_skipped(self):
        out = rpb._normalize_project_logs({"": ["note"], "  ": ["note"]})
        assert out == {}

    def test_preserves_order(self):
        out = rpb._normalize_project_logs({"app": ["c", "a", "b"]})
        assert out == {"app": ["c", "a", "b"]}


class TestParseFcModels:
    def test_default_models(self):
        assert rpb._parse_fc_models(None) == ["claude-haiku-4-5-20251001"]

    def test_preserves_order_and_dedups(self):
        out = rpb._parse_fc_models("claude-haiku-4-5-20251001, claude-sonnet-4-6, claude-haiku-4-5-20251001")
        assert out == ["claude-haiku-4-5-20251001", "claude-sonnet-4-6"]

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            rpb._parse_fc_models(" , , ")


def test_default_extraction_model_is_sonnet():
    assert rpb.DEFAULT_EXTRACTION_MODEL == "claude-sonnet-4-6"


class TestAnswerModelPolicy:
    def test_allows_default_eval_model(self):
        rpb._validate_answer_model_policy(
            mode="eval",
            eval_model="claude-haiku-4-5-20251001",
            fc_models=[],
            allow_non_haiku=False,
        )

    def test_rejects_non_haiku_eval_model_without_override(self):
        with pytest.raises(SystemExit, match="must be Haiku by default"):
            rpb._validate_answer_model_policy(
                mode="eval",
                eval_model="claude-sonnet-4-6",
                fc_models=[],
                allow_non_haiku=False,
            )

    def test_rejects_non_haiku_fc_model_without_override(self):
        with pytest.raises(SystemExit, match="FC answer model 'claude-opus-4-6'"):
            rpb._validate_answer_model_policy(
                mode="fc",
                eval_model="claude-haiku-4-5-20251001",
                fc_models=["claude-opus-4-6"],
                allow_non_haiku=False,
            )

    def test_allows_override(self):
        rpb._validate_answer_model_policy(
            mode="fc",
            eval_model="claude-sonnet-4-6",
            fc_models=["claude-opus-4-6"],
            allow_non_haiku=True,
        )


class TestOpenAICompatibleProviderHealth:
    def test_skips_health_probe_for_non_local_host(self, monkeypatch):
        called = False

        def _probe(*_a, **_k):
            nonlocal called
            called = True
            return False, "boom"

        monkeypatch.setattr(rpb, "_probe_openai_compatible_health", _probe)

        rpb._assert_openai_compatible_provider_healthy("https://api.example.com", source="answer_model")
        assert called is False

    def test_raises_for_dead_local_provider(self, monkeypatch):
        monkeypatch.setattr(rpb, "_probe_openai_compatible_health", lambda *_a, **_k: (False, "connection refused"))
        monkeypatch.setattr(rpb.time, "time", lambda: 1000.0)
        rpb._OPENAI_COMPAT_HEALTH_CACHE.clear()

        with pytest.raises(RuntimeError, match="provider health check failed"):
            rpb._assert_openai_compatible_provider_healthy("http://127.0.0.1:30002", source="answer_model")

    def test_uses_cached_health_result(self, monkeypatch):
        calls = {"n": 0}

        def _probe(*_a, **_k):
            calls["n"] += 1
            return True, "ok"

        monkeypatch.setattr(rpb, "_probe_openai_compatible_health", _probe)
        monkeypatch.setattr(rpb.time, "time", lambda: 1000.0)
        rpb._OPENAI_COMPAT_HEALTH_CACHE.clear()

        rpb._assert_openai_compatible_provider_healthy("http://127.0.0.1:30002", source="answer_model")
        rpb._assert_openai_compatible_provider_healthy("http://127.0.0.1:30002", source="answer_model")
        assert calls["n"] == 1


class TestEvalEmbeddingPreflight:
    def test_fails_when_provider_is_not_ollama(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)
        (ws / "config" / "memory.json").write_text(json.dumps({
            "models": {"embeddingsProvider": "openai"},
            "ollama": {"url": "http://127.0.0.1:11434", "embeddingModel": "qwen3-embedding:8b"},
        }))

        with pytest.raises(RuntimeError, match="embeddings provider must be 'ollama'"):
            rpb._eval_embedding_provider_preflight(ws)

    def test_fails_when_embedding_model_is_not_nomic(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)
        (ws / "config" / "memory.json").write_text(json.dumps({
            "models": {"embeddingsProvider": "ollama"},
            "ollama": {"url": "http://127.0.0.1:11434", "embeddingModel": "qwen3-embedding:8b", "embeddingDim": 4096},
        }))

        with pytest.raises(RuntimeError, match="embedding model must be 'nomic-embed-text'"):
            rpb._eval_embedding_provider_preflight(ws)

    def test_fails_when_embedding_dim_is_not_768(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)
        (ws / "config" / "memory.json").write_text(json.dumps({
            "models": {"embeddingsProvider": "ollama"},
            "ollama": {"url": "http://127.0.0.1:11434", "embeddingModel": "nomic-embed-text", "embeddingDim": 4096},
        }))

        with pytest.raises(RuntimeError, match="embedding dimension must be 768"):
            rpb._eval_embedding_provider_preflight(ws)

    def test_fails_when_ollama_embedding_provider_unreachable(self, tmp_path, monkeypatch):
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)
        (ws / "config" / "memory.json").write_text(json.dumps({
            "models": {"embeddingsProvider": "ollama"},
            "ollama": {"url": "http://127.0.0.1:11434", "embeddingModel": "nomic-embed-text", "embeddingDim": 768},
        }))

        def _boom(*_a, **_k):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr(rpb, "_ollama_list_models", _boom)

        with pytest.raises(RuntimeError, match="Ollama embedding provider is unreachable"):
            rpb._eval_embedding_provider_preflight(ws)

    def test_fails_when_embedding_model_missing(self, tmp_path, monkeypatch):
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)
        (ws / "config" / "memory.json").write_text(json.dumps({
            "models": {"embeddingsProvider": "ollama"},
            "ollama": {"url": "http://127.0.0.1:11434", "embeddingModel": "nomic-embed-text", "embeddingDim": 768},
        }))

        monkeypatch.setattr(rpb, "_ollama_list_models", lambda *_a, **_k: {"qwen3-embedding:4b"})

        with pytest.raises(RuntimeError, match="configured Ollama embedding model is not available"):
            rpb._eval_embedding_provider_preflight(ws)

    def test_accepts_latest_alias_for_embedding_model(self, tmp_path, monkeypatch):
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)
        (ws / "config" / "memory.json").write_text(json.dumps({
            "models": {"embeddingsProvider": "ollama"},
            "ollama": {"url": "http://127.0.0.1:11434", "embeddingModel": "nomic-embed-text", "embeddingDim": 768},
        }))

        monkeypatch.setattr(rpb, "_ollama_list_models", lambda *_a, **_k: {"nomic-embed-text:latest"})

        rpb._eval_embedding_provider_preflight(ws)

    def test_accepts_reachable_ollama_embedding_provider(self, tmp_path, monkeypatch):
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)
        (ws / "config" / "memory.json").write_text(json.dumps({
            "models": {"embeddingsProvider": "ollama"},
            "ollama": {"url": "http://127.0.0.1:11434", "embeddingModel": "nomic-embed-text", "embeddingDim": 768},
        }))

        monkeypatch.setattr(rpb, "_ollama_list_models", lambda *_a, **_k: {"qwen3-embedding:8b", "nomic-embed-text:latest"})

        rpb._eval_embedding_provider_preflight(ws)

    def test_allows_explicit_skip(self, tmp_path, monkeypatch):
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)
        (ws / "config" / "memory.json").write_text(json.dumps({
            "models": {"embeddingsProvider": "ollama"},
            "ollama": {"url": "http://127.0.0.1:11434", "embeddingModel": "qwen3-embedding:8b", "embeddingDim": 4096},
        }))
        monkeypatch.setenv("BENCHMARK_SKIP_EMBEDDING_PREFLIGHT", "1")
        monkeypatch.setattr(rpb, "_ollama_list_models", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should not probe")))

        rpb._eval_embedding_provider_preflight(ws)


class TestFcBaselinesFailHard:
    def test_run_fc_baseline_raises_on_answer_failure(self, monkeypatch, tmp_path):
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        monkeypatch.setenv("BENCHMARK_REQUIRE_QUERY_COUNT", "0")

        monkeypatch.setattr(
            rpb,
            "_load_reviews_with_dataset_gate",
            lambda max_sessions: (assets_dir, [_FakeReview(1)], [_FakeReview(1)], "v-test", 268),
        )
        monkeypatch.setattr(
            rpb,
            "get_all_eval_queries",
            lambda _reviews: [
                {
                    "question": "What does Maya do for work?",
                    "ground_truth": "Product manager at TechFlow",
                    "query_type": "factual_recall",
                }
            ],
        )
        monkeypatch.setattr(
            rpb,
            "format_transcript_for_extraction",
            lambda _review: "session transcript",
        )

        def _boom(*_args, **_kwargs):
            raise RuntimeError("Anthropic HTTP 429: over limit")

        monkeypatch.setattr(rpb, "_call_anthropic_cached", _boom)

        with pytest.raises(RuntimeError, match=r"FC answer failed.*What does Maya do for work"):
            rpb.run_fc_baseline(api_key="test-key", answer_model="claude-sonnet-4-6")

    def test_run_tier5_fc_baseline_raises_on_answer_failure(self, monkeypatch, tmp_path):
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        monkeypatch.setenv("BENCHMARK_REQUIRE_QUERY_COUNT", "0")

        monkeypatch.setattr(
            rpb,
            "_load_reviews_with_dataset_gate",
            lambda max_sessions: (assets_dir, [_FakeReview(1)], [_FakeReview(1)], "v-test", 268),
        )
        monkeypatch.setattr(
            rpb._DATASET,
            "get_tier5_queries",
            lambda: [
                {
                    "ei_id": "EI-01",
                    "question": "How are you feeling about your mom?",
                    "sensitivity_context": "",
                    "rubric": {},
                }
            ],
        )
        monkeypatch.setattr(
            rpb,
            "format_transcript_for_extraction",
            lambda _review: "session transcript",
        )

        def _boom(*_args, **_kwargs):
            raise RuntimeError("Anthropic HTTP 429: over limit")

        monkeypatch.setattr(rpb, "_call_anthropic_cached", _boom)

        with pytest.raises(RuntimeError, match=r"Tier 5 FC answer failed.*How are you feeling about your mom"):
            rpb.run_tier5_fc_baseline(api_key="test-key", answer_model="claude-sonnet-4-6")

    def test_run_fc_baseline_resumes_from_checkpoint(self, monkeypatch, tmp_path):
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        results_dir = tmp_path / "fc_baselines"
        results_dir.mkdir()
        monkeypatch.setenv("BENCHMARK_REQUIRE_QUERY_COUNT", "0")

        questions = [
            {
                "question": "What does Maya do for work?",
                "ground_truth": "Product manager at TechFlow",
                "query_type": "factual_recall",
            },
            {
                "question": "Where does Maya's mom live?",
                "ground_truth": "Houston",
                "query_type": "factual_recall",
            },
        ]
        checkpoint_path = rpb._fc_resume_checkpoint_path(
            results_dir,
            rpb._fc_result_stem("claude-sonnet-4-6"),
        )
        rpb._save_fc_resume_checkpoint(
            checkpoint_path,
            answer_model="claude-sonnet-4-6",
            total_queries=len(questions),
            results=[
                {
                    "question": questions[0]["question"],
                    "ground_truth": questions[0]["ground_truth"],
                    "prediction": "Product manager at TechFlow",
                    "judge_label": "CORRECT",
                    "score": 1.0,
                    "query_type": "factual_recall",
                    "recall_difficulty": "unknown",
                    "source_session": 0,
                }
            ],
            usage={"input_tokens": 100, "output_tokens": 50, "api_calls": 1},
        )

        monkeypatch.setattr(
            rpb,
            "_load_reviews_with_dataset_gate",
            lambda max_sessions: (assets_dir, [_FakeReview(1)], [_FakeReview(1)], "v-test", 268),
        )
        monkeypatch.setattr(rpb, "get_all_eval_queries", lambda _reviews: questions)
        monkeypatch.setattr(
            rpb,
            "_build_fc_transcript_context",
            lambda *a, **k: (
                "session transcript",
                {
                    "context_tokens": 123,
                    "compaction_count": 0,
                    "input_tokens": 5,
                    "output_tokens": 2,
                    "api_calls": 1,
                },
            ),
        )
        monkeypatch.setattr(rpb, "_append_usage_event", lambda *a, **k: None)

        calls = []

        def _answer(_system, _user, _model, _api_key, max_tokens=0):
            calls.append(max_tokens)
            return ("Houston", {"input_tokens": 20, "output_tokens": 10, "api_calls": 1})

        monkeypatch.setattr(rpb, "_call_anthropic_cached", _answer)
        monkeypatch.setattr(rpb, "_judge", lambda *a, **k: ("CORRECT", 1.0))

        results = rpb.run_fc_baseline(
            api_key="test-key",
            answer_model="claude-sonnet-4-6",
            results_dir=results_dir,
            judge_model="gpt-4o-mini",
        )

        assert calls == [512]
        assert [row["question"] for row in results] == [q["question"] for q in questions]
        usage = json.loads((results_dir / "fc_claude_sonnet_4_6_token_usage.json").read_text())
        assert usage["eval"]["input_tokens"] == 125
        assert usage["eval"]["output_tokens"] == 62
        assert usage["eval"]["api_calls"] == 3
        assert not checkpoint_path.exists()

    def test_run_tier5_fc_baseline_writes_model_specific_artifact(self, monkeypatch, tmp_path):
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        results_dir = tmp_path / "fc_baselines"
        results_dir.mkdir()

        monkeypatch.setattr(
            rpb,
            "_load_reviews_with_dataset_gate",
            lambda max_sessions: (assets_dir, [_FakeReview(1)], [_FakeReview(1)], "v-test", 268),
        )
        monkeypatch.setattr(
            rpb._DATASET,
            "get_tier5_queries",
            lambda: [
                {
                    "ei_id": "EI-01",
                    "ei_category": "relationship",
                    "question": "How should Maya handle this sensitively?",
                    "sensitivity_context": "",
                    "rubric": {},
                }
            ],
        )
        monkeypatch.setattr(
            rpb,
            "_build_fc_transcript_context",
            lambda *a, **k: (
                "session transcript",
                {
                    "context_tokens": 123,
                    "compaction_count": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "api_calls": 0,
                },
            ),
        )
        monkeypatch.setattr(
            rpb,
            "_call_anthropic_cached",
            lambda *a, **k: ("Be gentle", {"input_tokens": 20, "output_tokens": 10, "api_calls": 1}),
        )
        monkeypatch.setattr(rpb, "_judge_tier5", lambda *a, **k: (2, "good"))

        results = rpb.run_tier5_fc_baseline(
            api_key="test-key",
            answer_model="claude-sonnet-4-6",
            results_dir=results_dir,
        )

        assert len(results) == 1
        assert (results_dir / "tier5_fc_claude_sonnet_4_6.json").exists()
        assert (results_dir / "tier5_results.json").exists()

    def test_run_fc_tier5_backfill_merges_existing_baseline(self, monkeypatch, tmp_path):
        results_dir = tmp_path / "fc_baselines"
        results_dir.mkdir()
        baseline_path = results_dir / "fc_claude_sonnet_4_6_results.json"
        baseline_path.write_text(json.dumps([
            {
                "question": "Q1",
                "ground_truth": "A1",
                "prediction": "A1",
                "judge_label": "CORRECT",
                "score": 1.0,
                "query_type": "factual_recall",
                "recall_difficulty": "unknown",
                "source_session": 1,
            },
            {
                "question": "Q2",
                "ground_truth": "A2",
                "prediction": "wrong",
                "judge_label": "WRONG",
                "score": 0.0,
                "query_type": "factual_recall",
                "recall_difficulty": "unknown",
                "source_session": 1,
            },
        ], indent=2))

        captured = {}

        def _fake_tier5(*_args, **kwargs):
            captured["answer_model"] = kwargs.get("answer_model")
            return [
                {"ei_score": 2, "ei_category": "relationship"},
                {"ei_score": 1, "ei_category": "relationship"},
            ]

        monkeypatch.setattr(rpb, "run_tier5_fc_baseline", _fake_tier5)
        monkeypatch.setattr(
            rpb,
            "score_results",
            lambda _rows: {
                "overall": {
                    "count": 2,
                    "scored": 2,
                    "accuracy": 50.0,
                    "correct": 1,
                    "partial": 0,
                    "wrong": 1,
                    "error": 0,
                    "points": 1.0,
                    "max_points": 2.0,
                }
            },
        )

        payload = rpb.run_fc_tier5_backfill(
            api_key="test-key",
            answer_model="claude-sonnet-4-6",
            results_dir=results_dir,
        )

        overall = payload["scores"]["overall"]
        assert captured["answer_model"] == "claude-sonnet-4-6"
        assert overall["count"] == 4
        assert overall["correct"] == 2
        assert overall["partial"] == 1
        assert overall["wrong"] == 1
        assert overall["accuracy"] == 62.5
        assert (results_dir / "fc_claude_sonnet_4_6_scores.json").exists()
        assert (results_dir / "fc_scores.json").exists()

    def test_eval_resume_checkpoint_round_trip_preserves_sparse_results(self, tmp_path):
        workspace = tmp_path / "ws"
        questions = [
            {"question": "What does Maya do for work?"},
            {"question": "Where does Maya live?"},
            {"question": "What is Biscuit?"},
        ]
        checkpoint_path = rpb._eval_resume_checkpoint_path(workspace)
        rpb._save_eval_resume_checkpoint(
            checkpoint_path,
            eval_model="claude-haiku-4-5-20251001",
            total_queries=len(questions),
            results_by_idx={
                0: {
                    "question": questions[0]["question"],
                    "judge_label": "CORRECT",
                },
                2: {
                    "question": questions[2]["question"],
                    "judge_label": "WRONG",
                },
            },
        )

        loaded = rpb._load_eval_resume_checkpoint(
            checkpoint_path,
            eval_model="claude-haiku-4-5-20251001",
            questions=questions,
        )

        assert sorted(loaded) == [0, 2]
        assert loaded[0]["question"] == questions[0]["question"]
        assert loaded[2]["question"] == questions[2]["question"]

    def test_run_eval_resumes_from_checkpoint(self, monkeypatch, tmp_path):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        questions = [
            {
                "question": "What does Maya do for work?",
                "ground_truth": "Product manager at TechFlow",
                "query_type": "factual_recall",
                "source_session": 1,
            },
            {
                "question": "Where does Maya's mom live?",
                "ground_truth": "Houston",
                "query_type": "factual_recall",
                "source_session": 2,
            },
        ]
        checkpoint_path = rpb._eval_resume_checkpoint_path(workspace)
        rpb._save_eval_resume_checkpoint(
            checkpoint_path,
            eval_model="claude-haiku-4-5-20251001",
            total_queries=len(questions),
            results_by_idx={
                0: {
                    "question": questions[0]["question"],
                    "ground_truth": questions[0]["ground_truth"],
                    "prediction": "Product manager at TechFlow",
                    "judge_label": "CORRECT",
                    "score": 1.0,
                    "retrieval_label": "CORRECT",
                    "retrieval_score": 1.0,
                    "query_type": "factual_recall",
                    "recall_difficulty": "unknown",
                    "source_session": 1,
                    "evidence_sessions": [],
                    "tool_calls": ["memory_recall(pre-inject)"],
                    "tool_call_details": [],
                    "tool_results_summary": [],
                    "tool_analysis": {},
                    "statement_context_audit": "",
                    "provenance": {},
                    "required_context": [],
                    "retrieval_texts": [],
                    "answer_duration_s": 0.1,
                    "query_duration_ms": 100,
                    "preinject_duration_ms": 5,
                    "eval_tokens": {"input_tokens": 3, "output_tokens": 1, "api_calls": 1},
                }
            },
        )

        monkeypatch.setenv("BENCHMARK_REQUIRE_QUERY_COUNT", "0")
        monkeypatch.setattr(rpb, "_sync_instance_identity_to_workspace_root", lambda *_a, **_k: None)
        monkeypatch.setattr(
            rpb,
            "_load_reviews_with_dataset_gate",
            lambda max_sessions: (assets_dir, [_FakeReview(1), _FakeReview(2)], [_FakeReview(1), _FakeReview(2)], "v-test", 268),
        )
        monkeypatch.setattr(rpb, "get_all_eval_queries", lambda _reviews: list(questions))
        monkeypatch.setattr(rpb, "get_statement_context_queries", lambda: [])
        monkeypatch.setattr(
            rpb,
            "_apply_eval_query_profile",
            lambda qs: (
                list(qs),
                {
                    "profile": "full",
                    "requested": len(qs),
                    "selected": len(qs),
                    "target_size": len(qs),
                    "min_per_type": 1,
                },
            ),
        )
        monkeypatch.setattr(rpb, "_eval_core_context_preflight", lambda *a, **k: None)
        monkeypatch.setattr(rpb, "_eval_embedding_provider_preflight", lambda *a, **k: None)
        monkeypatch.setattr(rpb, "_write_eval_query_profile_manifest", lambda *a, **k: None)
        monkeypatch.setattr(rpb, "_resolve_eval_context_profile", lambda: ("lean", [], True))
        monkeypatch.setattr(rpb, "_build_eval_context", lambda *a, **k: "ctx")
        monkeypatch.setattr(rpb, "_build_eval_context_sources", lambda *a, **k: [])
        monkeypatch.setattr(rpb, "_resolve_eval_provider", lambda *a, **k: "oauth")
        monkeypatch.setattr(rpb, "_benchmark_env", lambda *a, **k: {})
        monkeypatch.setattr(rpb, "_resolve_eval_parallel_workers", lambda: 1)

        calls = []

        def _tool_use_loop(**kwargs):
            calls.append(kwargs["question"])
            return (
                "Houston",
                ["memory_recall(pre-inject)"],
                [{"tool": "memory_recall"}],
                ["Linda lives in Houston"],
                {
                    "input_tokens": 5,
                    "output_tokens": 2,
                    "api_calls": 1,
                    "tool_call_details": [],
                    "preinject_duration_ms": 7,
                },
            )

        monkeypatch.setattr(rpb, "_tool_use_loop", _tool_use_loop)
        monkeypatch.setattr(rpb, "_judge", lambda *a, **k: ("CORRECT", 1.0))
        monkeypatch.setattr(
            rpb,
            "_summarize_usage_events",
            lambda *_a, **_k: {
                "input_tokens": 5,
                "output_tokens": 2,
                "total_tokens": 7,
                "uncached_input_tokens": 5,
                "cache_read_tokens": 0,
                "cache_creation_tokens": 0,
                "api_calls": 1,
                "cost_usd": 0.0,
                "by_model": {},
                "by_tier": {},
                "by_source": {},
            },
        )

        results = rpb.run_eval(
            workspace,
            api_key="test-key",
            eval_model="claude-haiku-4-5-20251001",
            judge_model="gpt-4o-mini",
            resume_eval=True,
        )

        assert calls == [questions[1]["question"]]
        assert [row["question"] for row in results] == [q["question"] for q in questions]
        assert not checkpoint_path.exists()
        progress = json.loads((workspace / "logs" / "eval_progress.json").read_text())
        assert progress["completed"] == 2

    def test_run_eval_writes_resume_checkpoint_for_fresh_runs(self, monkeypatch, tmp_path):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        questions = [
            {
                "question": "What does Maya do for work?",
                "ground_truth": "Product manager at TechFlow",
                "query_type": "factual_recall",
                "source_session": 1,
            }
        ]

        monkeypatch.setenv("BENCHMARK_REQUIRE_QUERY_COUNT", "0")
        monkeypatch.setattr(rpb, "_sync_instance_identity_to_workspace_root", lambda *_a, **_k: None)
        monkeypatch.setattr(
            rpb,
            "_load_reviews_with_dataset_gate",
            lambda max_sessions: (assets_dir, [_FakeReview(1)], [_FakeReview(1)], "v-test", 268),
        )
        monkeypatch.setattr(rpb, "get_all_eval_queries", lambda _reviews: list(questions))
        monkeypatch.setattr(rpb, "get_statement_context_queries", lambda: [])
        monkeypatch.setattr(
            rpb,
            "_apply_eval_query_profile",
            lambda qs: (
                list(qs),
                {
                    "profile": "full",
                    "requested": len(qs),
                    "selected": len(qs),
                    "target_size": len(qs),
                    "min_per_type": 1,
                },
            ),
        )
        monkeypatch.setattr(rpb, "_eval_core_context_preflight", lambda *a, **k: None)
        monkeypatch.setattr(rpb, "_eval_embedding_provider_preflight", lambda *a, **k: None)
        monkeypatch.setattr(rpb, "_write_eval_query_profile_manifest", lambda *a, **k: None)
        monkeypatch.setattr(rpb, "_resolve_eval_context_profile", lambda: ("lean", [], True))
        monkeypatch.setattr(rpb, "_build_eval_context", lambda *a, **k: "ctx")
        monkeypatch.setattr(rpb, "_build_eval_context_sources", lambda *a, **k: [])
        monkeypatch.setattr(rpb, "_resolve_eval_provider", lambda *a, **k: "oauth")
        monkeypatch.setattr(rpb, "_benchmark_env", lambda *a, **k: {})
        monkeypatch.setattr(rpb, "_resolve_eval_parallel_workers", lambda: 1)
        monkeypatch.setattr(
            rpb,
            "_tool_use_loop",
            lambda **kwargs: (
                "Product manager at TechFlow",
                ["memory_recall(pre-inject)"],
                [{"tool": "memory_recall"}],
                ["Maya works at TechFlow"],
                {
                    "input_tokens": 5,
                    "output_tokens": 2,
                    "api_calls": 1,
                    "tool_call_details": [],
                    "preinject_duration_ms": 7,
                },
            ),
        )
        monkeypatch.setattr(rpb, "_judge", lambda *a, **k: ("CORRECT", 1.0))
        monkeypatch.setattr(
            rpb,
            "_summarize_usage_events",
            lambda *_a, **_k: {
                "input_tokens": 5,
                "output_tokens": 2,
                "total_tokens": 7,
                "uncached_input_tokens": 5,
                "cache_read_tokens": 0,
                "cache_creation_tokens": 0,
                "api_calls": 1,
                "cost_usd": 0.0,
                "by_model": {},
                "by_tier": {},
                "by_source": {},
            },
        )

        checkpoint_paths = []
        real_save = rpb._save_eval_resume_checkpoint

        def _capture_save(checkpoint_path, **kwargs):
            checkpoint_paths.append(checkpoint_path)
            return real_save(checkpoint_path, **kwargs)

        monkeypatch.setattr(rpb, "_save_eval_resume_checkpoint", _capture_save)

        results = rpb.run_eval(
            workspace,
            api_key="test-key",
            eval_model="claude-haiku-4-5-20251001",
            judge_model="gpt-4o-mini",
            resume_eval=False,
        )

        assert len(results) == 1
        assert checkpoint_paths == [rpb._eval_resume_checkpoint_path(workspace)]
        assert not rpb._eval_resume_checkpoint_path(workspace).exists()


class TestClaudeCodeEvalFailHard:
    def test_call_claude_code_sends_prompt_via_stdin(self, monkeypatch):
        captured = {}

        def _fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "result": "ok",
                        "modelUsage": {
                            "claude-sonnet-4-6": {
                                "inputTokens": 10,
                                "outputTokens": 5,
                            }
                        },
                    }
                ),
                stderr="",
            )

        monkeypatch.setattr(subprocess, "run", _fake_run)

        text, usage = rpb._call_claude_code(
            system_prompt="system prompt",
            user_message="very long prompt payload",
            model="claude-sonnet-4-6",
        )

        assert text == "ok"
        assert usage["api_calls"] == 1
        assert captured["kwargs"]["input"] == "very long prompt payload"
        assert captured["cmd"][-2:] == ["--system-prompt", "system prompt"]
        assert "very long prompt payload" not in captured["cmd"]

    def test_tool_use_loop_claude_code_raises_on_nonzero_return(self, monkeypatch, tmp_path):
        ws = tmp_path / "ws"
        (ws / "data").mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(rpb, "_pre_recall", lambda *a, **k: ("", "", {}))
        monkeypatch.setattr(
            rpb,
            "_parse_claude_stream_output",
            lambda _stdout: (
                "",
                [],
                [],
                [],
                {"is_error": True, "result": "OAuth token has expired"},
            ),
        )
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **k: SimpleNamespace(
                returncode=1,
                stdout='{"type":"result"}',
                stderr="oauth expired",
            ),
        )

        with pytest.raises(RuntimeError, match=r"Claude Code failed rc=1.*OAuth token has expired"):
            rpb._tool_use_loop_claude_code(
                question="What does Maya do for work?",
                eval_context="ctx",
                workspace=ws,
                api_key="unused",
                env={},
                model="claude-sonnet-4-6",
                context_inject=False,
            )

    def test_tool_use_loop_claude_code_retries_bun_crash_then_succeeds(self, monkeypatch, tmp_path):
        ws = tmp_path / "ws"
        (ws / "data").mkdir(parents=True, exist_ok=True)

        monkeypatch.setenv("CLAUDE_CODE_EVAL_RETRY_ATTEMPTS", "2")
        monkeypatch.setenv("CLAUDE_CODE_EVAL_RETRY_BACKOFF_S", "0.01")
        monkeypatch.setenv("CLAUDE_CODE_EVAL_RETRY_BACKOFF_CAP_S", "0.01")
        monkeypatch.setattr(rpb.random, "uniform", lambda a, b: 0.0)
        monkeypatch.setattr(rpb.time, "sleep", lambda _s: None)
        monkeypatch.setattr(rpb, "_pre_recall", lambda *a, **k: ("", "", {}))
        monkeypatch.setattr(rpb, "_tool_memory_recall", lambda *a, **k: ("", {}))
        monkeypatch.setattr(rpb, "_append_usage_event", lambda *a, **k: None)

        calls = {"n": 0}

        def _fake_parse(_stdout):
            if calls["n"] == 1:
                return (
                    "",
                    [],
                    [],
                    [],
                    {"is_error": False, "result": ""},
                )
            return (
                "final answer",
                [],
                [],
                [],
                {"is_error": False, "result": "final answer", "num_turns": 1, "modelUsage": {}},
            )

        monkeypatch.setattr(rpb, "_parse_claude_stream_output", _fake_parse)

        def _fake_run(*_args, **_kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return SimpleNamespace(
                    returncode=-5,
                    stdout='{"type":"result"}',
                    stderr="o: Bun has crashed. This indicates a bug in Bun, not your code.",
                )
            return SimpleNamespace(
                returncode=0,
                stdout='{"type":"result"}',
                stderr="",
            )

        monkeypatch.setattr(subprocess, "run", _fake_run)

        answer, tool_calls, tool_logs, retrieval_texts, usage = rpb._tool_use_loop_claude_code(
            question="What does Maya do for work?",
            eval_context="ctx",
            workspace=ws,
            api_key="unused",
            env={},
            model="claude-sonnet-4-6",
            context_inject=False,
        )

        assert answer == "final answer"
        assert tool_calls == []
        assert retrieval_texts == []
        assert usage["api_calls"] == 1
        assert calls["n"] == 2

class TestFcContextCompaction:
    def test_build_fc_context_skips_compaction_when_under_threshold(self, monkeypatch, tmp_path):
        reviews = [_FakeReview(1), _FakeReview(2)]
        monkeypatch.setattr(rpb, "SESSION_DATES", {1: "2026-03-01", 2: "2026-03-02"})
        monkeypatch.setattr(rpb, "format_transcript_for_extraction", lambda review: f"session {review.session_num}")
        monkeypatch.setattr(rpb, "_estimate_text_tokens", lambda text: 1000)

        def _boom(*_args, **_kwargs):
            raise AssertionError("compaction should not run")

        monkeypatch.setattr(rpb, "_call_anthropic_cached", _boom)

        context, stats = rpb._build_fc_transcript_context(
            reviews,
            api_key="test-key",
            answer_model="claude-haiku-4-5-20251001",
            results_dir=tmp_path,
        )

        assert "Session 1" in context
        assert "Session 2" in context
        assert stats["compaction_count"] == 0
        assert stats["api_calls"] == 0

    def test_build_fc_context_compacts_when_over_threshold(self, monkeypatch, tmp_path):
        reviews = [_FakeReview(1), _FakeReview(2), _FakeReview(3)]
        monkeypatch.setattr(rpb, "SESSION_DATES", {1: "2026-03-01", 2: "2026-03-02", 3: "2026-03-03"})
        monkeypatch.setattr(
            rpb,
            "format_transcript_for_extraction",
            lambda review: f"transcript for session {review.session_num}",
        )

        state = {"calls": 0}

        def _estimate(text):
            if "Compacted History" in text:
                return 1000
            return 170000

        def _compact(_system, user_message, _model, _api_key, max_tokens=0):
            state["calls"] += 1
            return ("- summarized older context", {"input_tokens": 120, "output_tokens": 30, "api_calls": 1})

        monkeypatch.setattr(rpb, "_estimate_text_tokens", _estimate)
        monkeypatch.setattr(rpb, "_call_anthropic_cached", _compact)
        monkeypatch.setattr(rpb, "_append_usage_event", lambda *a, **k: None)

        context, stats = rpb._build_fc_transcript_context(
            reviews,
            api_key="test-key",
            answer_model="claude-haiku-4-5-20251001",
            results_dir=tmp_path,
        )

        assert "Compacted History #1" in context
        assert stats["compaction_count"] == 1
        assert stats["input_tokens"] == 120
        assert stats["output_tokens"] == 30
        assert stats["api_calls"] == 1
        assert state["calls"] == 1


class TestParseReviewTimestamp:
    """Tests for _parse_review_timestamp: multi-format parsing + fallback."""

    def test_full_utc_format(self):
        review = _FakeReview(1, timestamp="2026-03-01 14:30:00 UTC")
        dt = rpb._parse_review_timestamp(review)
        assert dt == datetime(2026, 3, 1, 14, 30, 0, tzinfo=timezone.utc)

    def test_without_utc_suffix(self):
        review = _FakeReview(1, timestamp="2026-03-01 14:30:00")
        dt = rpb._parse_review_timestamp(review)
        assert dt == datetime(2026, 3, 1, 14, 30, 0, tzinfo=timezone.utc)

    def test_date_only_defaults_to_noon(self):
        review = _FakeReview(1, timestamp="2026-03-01")
        dt = rpb._parse_review_timestamp(review)
        assert dt.hour == 12
        assert dt.minute == 0

    def test_empty_timestamp_falls_back_to_session_dates(self):
        review = _FakeReview(1, timestamp="")
        dt = rpb._parse_review_timestamp(review)
        # Falls back to SESSION_DATES[1]
        assert dt.tzinfo == timezone.utc
        assert dt.hour == 12

    def test_invalid_timestamp_falls_back(self):
        review = _FakeReview(1, timestamp="not-a-date")
        dt = rpb._parse_review_timestamp(review)
        assert dt.tzinfo == timezone.utc

    def test_no_timestamp_attr(self):
        @dataclass
        class _NoTs:
            session_num: int = 999
        dt = rpb._parse_review_timestamp(_NoTs())
        assert dt.tzinfo == timezone.utc


class TestSplitSessionBlocksOnGap:
    """Tests for _split_session_blocks_on_gap: chunking by time gaps."""

    def _block(self, ts_str, snum):
        dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M")
        return {"timestamp": dt, "session_num": snum}

    def test_empty_input(self):
        assert rpb._split_session_blocks_on_gap([], 3600) == []

    def test_zero_gap_splits_each(self):
        blocks = [self._block("2026-03-01 10:00", 1), self._block("2026-03-01 10:01", 2)]
        result = rpb._split_session_blocks_on_gap(blocks, 0)
        assert len(result) == 2

    def test_no_gap_single_chunk(self):
        blocks = [
            self._block("2026-03-01 10:00", 1),
            self._block("2026-03-01 10:30", 2),
        ]
        result = rpb._split_session_blocks_on_gap(blocks, 7200)
        assert len(result) == 1
        assert len(result[0]) == 2

    def test_gap_creates_split(self):
        blocks = [
            self._block("2026-03-01 10:00", 1),
            self._block("2026-03-01 14:00", 2),
        ]
        result = rpb._split_session_blocks_on_gap(blocks, 3600)
        assert len(result) == 2

    def test_multiple_gaps(self):
        blocks = [
            self._block("2026-03-01 08:00", 1),
            self._block("2026-03-01 08:10", 2),
            self._block("2026-03-01 14:00", 3),
            self._block("2026-03-01 14:05", 4),
            self._block("2026-03-02 08:00", 5),
        ]
        result = rpb._split_session_blocks_on_gap(blocks, 3600)
        assert len(result) == 3
        assert [len(c) for c in result] == [2, 2, 1]


class TestOperationalDay:
    def test_before_4am_rolls_back_to_prior_day(self):
        review = _FakeReview(1, timestamp="2026-03-02 03:59:59 UTC")
        assert rpb._operational_day(review) == "2026-03-01"

    def test_at_4am_stays_on_same_day(self):
        review = _FakeReview(1, timestamp="2026-03-02 04:00:00 UTC")
        assert rpb._operational_day(review) == "2026-03-02"


class TestObdMessageStream:
    def test_builds_chronological_message_pairs(self):
        @dataclass
        class _Review:
            session_num: int
            transcript_turns: list

        reviews = [
            _Review(1, [{"maya": "one", "agent": "two"}]),
            _Review(2, [{"maya": "three"}, {"agent": "four"}]),
        ]
        out = rpb._build_obd_message_stream(reviews)
        assert [msg["role"] for msg in out] == ["user", "assistant", "user", "assistant"]
        assert [msg["content"].splitlines()[-1] for msg in out] == ["one", "two", "three", "four"]
        assert all(msg["content"].startswith("Source Timestamp: ") for msg in out)
        assert out[0]["created_at"].startswith("2026-03-01T")
        assert out[1]["created_at"].startswith("2026-03-01T")
        assert out[2]["created_at"].startswith("2026-03-03T")
        assert out[3]["created_at"].startswith("2026-03-03T")

    def test_skips_blank_turns(self):
        @dataclass
        class _Review:
            session_num: int
            transcript_turns: list

        reviews = [_Review(1, [{"maya": "   ", "agent": ""}, {"maya": "hi"}])]
        out = rpb._build_obd_message_stream(reviews)
        assert len(out) == 1
        assert out[0]["role"] == "user"
        assert out[0]["content"].splitlines()[-1] == "hi"
        assert out[0]["content"].startswith("Source Timestamp: ")
        assert out[0]["created_at"].startswith("2026-03-01T")


class TestRuntimeExtractJsonl:
    def test_parses_json_after_project_log_prefix(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rpb, "_QUAID_DIR", tmp_path)
        monkeypatch.setattr(rpb, "_python_cmd_for_quaid_script", lambda _s: [sys.executable, "-m", "stub"])

        def _run(cmd, **kwargs):
            result = _FakeSubprocessResult()
            result.stdout = (
                "[project-log] project=recipe-app entries=2 file=/tmp/PROJECT.md dry_run=False\n"
                "{\n"
                '  "facts_stored": 63,\n'
                '  "facts_skipped": 0,\n'
                '  "edges_created": 6\n'
                "}\n"
            )
            result.stderr = "[extract] Compaction: splitting"
            return result

        monkeypatch.setattr(rpb.subprocess, "run", _run)

        out = rpb._run_runtime_extract_jsonl(
            workspace=tmp_path / "ws",
            env={},
            session_file=tmp_path / "obd.jsonl",
            owner_id="maya",
            label="Compaction",
            session_id="obd-compaction-0001",
            timeout_seconds=123,
        )

        assert out == {
            "facts_stored": 63,
            "facts_skipped": 0,
            "edges_created": 6,
        }

    def test_streams_plain_obd_progress_when_progress_path_provided(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rpb, "_QUAID_DIR", tmp_path)
        monkeypatch.setattr(rpb, "_python_cmd_for_quaid_script", lambda _s: [sys.executable, "-m", "stub"])

        class _FakePopen:
            def __init__(self, *_args, **_kwargs):
                self.stdout = io.StringIO('{\n  "facts_stored": 12,\n  "facts_skipped": 0,\n  "edges_created": 2\n}\n')
                self.stderr = io.StringIO(
                    "[extract] Compaction: splitting into 31 chunks\n"
                    "[extract] Compaction: chunk 7/31 (12345 chars)\n"
                )
                self.returncode = 0

            def wait(self, timeout=None):
                return self.returncode

            def kill(self):
                self.returncode = -9

        monkeypatch.setattr(rpb.subprocess, "Popen", _FakePopen)

        progress_path = tmp_path / "ws" / "logs" / "obd_extract_progress.json"
        out = rpb._run_runtime_extract_jsonl(
            workspace=tmp_path / "ws",
            env={},
            session_file=tmp_path / "obd.jsonl",
            owner_id="maya",
            label="Compaction",
            session_id="obd-compaction-0001",
            timeout_seconds=123,
            progress_path=progress_path,
        )

        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        assert out == {
            "facts_stored": 12,
            "facts_skipped": 0,
            "edges_created": 2,
        }
        assert progress["state"] == "completed"
        assert progress["current_chunk"] == 7
        assert progress["total_chunks"] == 31


class TestOBDExtractionTimeoutEnv:
    def test_run_runtime_rolling_driver_tolerates_stdout_noise(self, tmp_path, monkeypatch):
        transcript = tmp_path / "obd.jsonl"
        transcript.write_text("", encoding="utf-8")

        def _run(cmd, **kwargs):
            result = _FakeSubprocessResult()
            result.stdout = (
                "[project-log] project=recipe-app entries=2 file=/tmp/PROJECT.md dry_run=True\n"
                '{\n'
                '  "session_id": "obd-compaction-0001",\n'
                '  "signals_processed": 3,\n'
                '  "cursor_line_offset": 123,\n'
                '  "total_lines": 456,\n'
                '  "rolling_state_exists": false,\n'
                '  "rolling_state_path": "/tmp/state.json",\n'
                '  "metrics_path": "/tmp/rolling.jsonl",\n'
                '  "remaining_tokens": 789\n'
                '}\n'
            )
            result.stderr = "[config] loaded"
            return result

        monkeypatch.setattr(rpb.subprocess, "run", _run)
        monkeypatch.setattr(rpb, "_BACKEND", "oauth")
        monkeypatch.setattr(rpb, "_find_anthropic_credential", lambda: "sk-ant-oat01-test-token")

        out = rpb._run_runtime_rolling_driver(
            workspace=tmp_path,
            env={},
            session_id="obd-compaction-0001",
            transcript_path=transcript,
            timeout_seconds=60,
            chunk_tokens=12000,
            final_signal=None,
        )

        assert out["signals_processed"] == 3
        assert out["cursor_line_offset"] == 123
        assert out["remaining_tokens"] == 789

    def test_rolling_flush_resume_state_ready_when_stage_complete_and_compaction_pending(self, tmp_path):
        workspace = tmp_path / "ws"
        session_id = "obd-compaction-0001"
        transcript = tmp_path / "obd.jsonl"
        transcript.write_text('{"role":"user","content":"hi"}\n', encoding="utf-8")

        state_path = workspace / rpb._BENCHMARK_QUAID_INSTANCE / "data" / "rolling-extraction" / f"{session_id}.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text("{}", encoding="utf-8")

        cursor_path = workspace / rpb._BENCHMARK_QUAID_INSTANCE / "data" / "session-cursors" / f"{session_id}.json"
        cursor_path.parent.mkdir(parents=True, exist_ok=True)
        cursor_path.write_text(
            json.dumps({"session_id": session_id, "line_offset": 1, "transcript_path": str(transcript)}),
            encoding="utf-8",
        )

        signal_dir = workspace / rpb._BENCHMARK_QUAID_INSTANCE / "data" / "extraction-signals"
        signal_dir.mkdir(parents=True, exist_ok=True)
        (signal_dir / "1_compaction.json").write_text(
            json.dumps({"type": "compaction", "session_id": session_id, "transcript_path": str(transcript)}),
            encoding="utf-8",
        )

        out = rpb._rolling_flush_resume_state(
            workspace,
            session_id=session_id,
            transcript_path=transcript,
        )

        assert out["ready"] is True
        assert out["cursor_line_offset"] == 1
        assert out["total_lines"] == 1
        assert out["pending_compaction_signals"] == 1

    def test_rolling_flush_resume_state_uses_completed_state_when_cursor_is_source_keyed(self, tmp_path):
        workspace = tmp_path / "ws"
        session_id = "obd-compaction-0001"
        transcript = tmp_path / "obd.jsonl"
        transcript.write_text('{"role":"user","content":"hi"}\n{"role":"assistant","content":"ok"}\n', encoding="utf-8")

        state_path = workspace / "data" / "rolling-extraction" / f"{session_id}.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps(
                {
                    "session_id": session_id,
                    "transcript_path": str(transcript),
                    "processed_line_offset": 2,
                    "buffered_line_offset": 2,
                }
            ),
            encoding="utf-8",
        )

        source_cursor = workspace / "data" / "session-cursors" / "source-abc123.json"
        source_cursor.parent.mkdir(parents=True, exist_ok=True)
        source_cursor.write_text(
            json.dumps(
                {
                    "session_id": session_id,
                    "line_offset": 1,
                    "transcript_path": str(transcript),
                }
            ),
            encoding="utf-8",
        )

        signal_dir = workspace / "data" / "extraction-signals"
        signal_dir.mkdir(parents=True, exist_ok=True)
        (signal_dir / "1_compaction.json").write_text(
            json.dumps({"type": "compaction", "session_id": session_id, "transcript_path": str(transcript)}),
            encoding="utf-8",
        )

        out = rpb._rolling_flush_resume_state(
            workspace,
            session_id=session_id,
            transcript_path=transcript,
        )

        assert out["ready"] is True
        assert out["cursor_line_offset"] == 2
        assert out["total_lines"] == 2
        assert out["pending_compaction_signals"] == 1

    def test_runtime_rolling_paths_prefer_flat_workspace_layout(self, tmp_path):
        workspace = tmp_path / "ws"
        (workspace / "logs" / "daemon").mkdir(parents=True, exist_ok=True)
        (workspace / "data" / "rolling-extraction").mkdir(parents=True, exist_ok=True)
        (workspace / "data" / "session-cursors").mkdir(parents=True, exist_ok=True)
        (workspace / "data" / "extraction-signals").mkdir(parents=True, exist_ok=True)

        assert rpb._rolling_metrics_log_path(workspace) == (
            workspace / "instances" / rpb._BENCHMARK_QUAID_INSTANCE / "logs" / "daemon" / "rolling-extraction.jsonl"
        )
        assert rpb._rolling_state_file(workspace, "sess-1") == workspace / "data" / "rolling-extraction" / "sess-1.json"
        assert rpb._rolling_cursor_file(workspace, "sess-1") == workspace / "data" / "session-cursors" / "sess-1.json"

    def test_rolling_obd_resume_state_detects_flat_workspace_layout(self, tmp_path):
        workspace = tmp_path / "ws"
        state_path = workspace / "data" / "rolling-extraction" / "obd-compaction-0001.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text("{}", encoding="utf-8")

        assert rpb._has_rolling_obd_resume_state(workspace) is True

    def test_runtime_rolling_metrics_prefers_existing_instance_log(self, tmp_path):
        workspace = tmp_path / "ws"
        (workspace / "logs" / "daemon").mkdir(parents=True, exist_ok=True)
        instance_metrics = (
            workspace
            / "instances"
            / rpb._BENCHMARK_QUAID_INSTANCE
            / "logs"
            / "daemon"
            / "rolling-extraction.jsonl"
        )
        instance_metrics.parent.mkdir(parents=True, exist_ok=True)
        instance_metrics.write_text("{\"event\":\"rolling_flush\"}\n", encoding="utf-8")

        assert rpb._rolling_metrics_log_path(workspace) == instance_metrics

    def test_select_rolling_flush_metric_ignores_later_noop(self):
        selected = rpb._select_rolling_flush_metric(
            [
                {
                    "event": "rolling_flush",
                    "final_raw_fact_count": 27,
                    "final_facts_stored": 27,
                    "project_logs_written": 14,
                },
                {
                    "event": "rolling_flush",
                    "noop": True,
                    "final_raw_fact_count": 0,
                    "final_facts_stored": 0,
                    "project_logs_written": 0,
                },
            ]
        )

        assert selected["final_raw_fact_count"] == 27
        assert selected["final_facts_stored"] == 27
        assert selected["project_logs_written"] == 14

    def test_select_rolling_flush_metric_allows_only_noop(self):
        selected = rpb._select_rolling_flush_metric(
            [
                {
                    "event": "rolling_flush",
                    "noop": True,
                    "noop_reason": "no_new_content",
                },
            ]
        )

        assert selected["noop_reason"] == "no_new_content"

    def test_save_and_restore_rolling_pre_publish_checkpoint(self, tmp_path):
        workspace = tmp_path / "ws"
        session_id = "obd-compaction-0001"
        data_root = workspace / rpb._BENCHMARK_QUAID_INSTANCE / "data"
        signal_dir = data_root / "extraction-signals"
        state_path = data_root / "rolling-extraction" / f"{session_id}.json"
        cursor_path = data_root / "session-cursors" / f"{session_id}.json"
        memory_db = data_root / "memory.db"
        memory_wal = data_root / "memory.db-wal"
        signal_path = signal_dir / "1_compaction.json"

        for path in [signal_dir, state_path.parent, cursor_path.parent]:
            path.mkdir(parents=True, exist_ok=True)
        memory_db.write_text("db-v1", encoding="utf-8")
        memory_wal.write_text("wal-v1", encoding="utf-8")
        state_path.write_text("{}", encoding="utf-8")
        cursor_path.write_text(json.dumps({"line_offset": 1}), encoding="utf-8")
        signal_path.write_text(json.dumps({"type": "compaction", "session_id": session_id}), encoding="utf-8")

        saved = rpb._save_rolling_pre_publish_checkpoint(workspace, session_id=session_id)
        assert saved is not None
        assert "benchrunner/data/memory.db" in saved["files"]
        assert "benchrunner/data/memory.db-wal" in saved["files"]

        memory_db.write_text("db-v2", encoding="utf-8")
        memory_wal.unlink()
        signal_path.unlink()

        restored = rpb._restore_rolling_pre_publish_checkpoint(workspace, session_id=session_id)
        assert restored is not None
        assert memory_db.read_text(encoding="utf-8") == "db-v1"
        assert memory_wal.read_text(encoding="utf-8") == "wal-v1"
        assert signal_path.exists()

    def test_run_runtime_rolling_obd_extract_skips_stage_when_resume_ready(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        session_id = "obd-compaction-0001"
        session_file = tmp_path / "obd.jsonl"
        session_file.write_text('{"role":"user","content":"hi"}\n', encoding="utf-8")

        state_path = workspace / rpb._BENCHMARK_QUAID_INSTANCE / "data" / "rolling-extraction" / f"{session_id}.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text("{}", encoding="utf-8")

        cursor_path = workspace / rpb._BENCHMARK_QUAID_INSTANCE / "data" / "session-cursors" / f"{session_id}.json"
        cursor_path.parent.mkdir(parents=True, exist_ok=True)
        cursor_path.write_text(
            json.dumps({"session_id": session_id, "line_offset": 1, "transcript_path": str(session_file)}),
            encoding="utf-8",
        )

        signal_dir = workspace / rpb._BENCHMARK_QUAID_INSTANCE / "data" / "extraction-signals"
        signal_dir.mkdir(parents=True, exist_ok=True)
        (signal_dir / "1_compaction.json").write_text(
            json.dumps({"type": "compaction", "session_id": session_id, "transcript_path": str(session_file)}),
            encoding="utf-8",
        )

        calls = []

        def _driver(**kwargs):
            calls.append(kwargs)
            state_path.unlink(missing_ok=True)
            return {
                "signals_processed": 1,
                "signal_loops": 1,
                "cursor_line_offset": 1,
                "cursor_transcript_path": str(session_file),
                "total_lines": 1,
                "rolling_state_exists": False,
                "rolling_state_path": str(state_path),
                "metrics_path": str(workspace / rpb._BENCHMARK_QUAID_INSTANCE / "logs" / "daemon" / "rolling-extraction.jsonl"),
                "remaining_tokens": 0,
            }

        monkeypatch.setattr(rpb, "_run_runtime_rolling_driver", _driver)
        monkeypatch.setattr(
            rpb,
            "_load_rolling_metric_rows",
            lambda *_args, **_kwargs: [
                {
                    "event": "rolling_flush",
                    "session_id": session_id,
                    "final_raw_fact_count": 1,
                    "final_facts_stored": 1,
                    "final_facts_skipped": 0,
                    "final_edges_created": 0,
                    "snippets_count": 0,
                    "journals_count": 0,
                    "project_logs_seen": 0,
                    "project_logs_written": 0,
                    "project_logs_projects_updated": 0,
                    "dedup_scanned_rows": 2493,
                    "dedup_vec_query_count": 146,
                    "dedup_vec_candidates_returned": 2493,
                    "dedup_vec_candidate_limit": 64,
                    "dedup_vec_limit_hits": 0,
                    "dedup_fts_query_count": 192,
                    "dedup_fts_candidates_returned": 2493,
                    "dedup_fts_candidate_limit": 500,
                    "dedup_fts_limit_hits": 0,
                    "embedding_cache_requested": 146,
                    "embedding_cache_unique": 146,
                    "embedding_cache_hits": 125,
                    "embedding_cache_warmed": 21,
                    "embedding_cache_failed": 0,
                    "staged_semantic_duplicate_facts_collapsed": 2,
                    "staged_semantic_llm_checks": 5,
                    "staged_semantic_llm_same_hits": 2,
                }
            ],
        )

        out = rpb._run_runtime_rolling_obd_extract(
            workspace=workspace,
            env={},
            session_file=session_file,
            session_id=session_id,
            chunk_tokens=8000,
            chunk_max_lines=144,
            timeout_seconds=60,
        )

        assert len(calls) == 1
        assert calls[0]["final_signal"] is None
        assert out["resumed_from_staged_checkpoint"] is True

    def test_run_runtime_rolling_obd_extract_fresh_run_writes_signal_and_saves_checkpoint(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        session_id = "obd-compaction-0001"
        session_file = tmp_path / "obd.jsonl"
        session_file.write_text('{"role":"user","content":"hi"}\n', encoding="utf-8")

        state_path = workspace / rpb._BENCHMARK_QUAID_INSTANCE / "data" / "rolling-extraction" / f"{session_id}.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text("{}", encoding="utf-8")

        calls = []
        signal_calls = []
        checkpoint_calls = []

        def _driver(**kwargs):
            calls.append(kwargs)
            if len(calls) == 2:
                state_path.unlink(missing_ok=True)
            return {
                "signals_processed": 1,
                "signal_loops": 1,
                "cursor_line_offset": 1,
                "cursor_transcript_path": str(session_file),
                "total_lines": 1,
                "rolling_state_exists": False,
                "rolling_state_path": str(state_path),
                "metrics_path": str(workspace / rpb._BENCHMARK_QUAID_INSTANCE / "logs" / "daemon" / "rolling-extraction.jsonl"),
                "remaining_tokens": 0,
            }

        monkeypatch.setattr(rpb, "_run_runtime_rolling_driver", _driver)
        monkeypatch.setattr(
            rpb,
            "_write_runtime_rolling_signal",
            lambda **kwargs: signal_calls.append(kwargs),
        )
        monkeypatch.setattr(
            rpb,
            "_save_rolling_pre_publish_checkpoint",
            lambda *args, **kwargs: checkpoint_calls.append(kwargs) or {"mode": "rolling-pre-publish"},
        )
        monkeypatch.setattr(
            rpb,
            "_load_rolling_metric_rows",
            lambda *_args, **_kwargs: [
                {
                    "event": "rolling_flush",
                    "session_id": session_id,
                    "final_raw_fact_count": 1,
                    "final_facts_stored": 1,
                    "final_facts_skipped": 0,
                    "final_edges_created": 0,
                    "snippets_count": 0,
                    "journals_count": 0,
                    "project_logs_seen": 0,
                    "project_logs_written": 0,
                    "project_logs_projects_updated": 0,
                    "dedup_scanned_rows": 2493,
                    "dedup_vec_query_count": 146,
                    "dedup_vec_candidates_returned": 2493,
                    "dedup_vec_candidate_limit": 64,
                    "dedup_vec_limit_hits": 0,
                    "dedup_fts_query_count": 192,
                    "dedup_fts_candidates_returned": 2493,
                    "dedup_fts_candidate_limit": 500,
                    "dedup_fts_limit_hits": 0,
                    "embedding_cache_requested": 146,
                    "embedding_cache_unique": 146,
                    "embedding_cache_hits": 125,
                    "embedding_cache_warmed": 21,
                    "embedding_cache_failed": 0,
                    "staged_semantic_duplicate_facts_collapsed": 2,
                    "staged_semantic_llm_checks": 5,
                    "staged_semantic_llm_same_hits": 2,
                }
            ],
        )

        out = rpb._run_runtime_rolling_obd_extract(
            workspace=workspace,
            env={},
            session_file=session_file,
            session_id=session_id,
            chunk_tokens=8000,
            chunk_max_lines=144,
            timeout_seconds=60,
        )

        assert len(calls) == 2
        assert calls[0]["final_signal"] is None
        assert calls[1]["final_signal"] is None
        assert len(signal_calls) == 1
        assert signal_calls[0]["signal_type"] == "compaction"
        assert len(checkpoint_calls) == 1
        assert checkpoint_calls[0]["session_id"] == session_id
        assert out["resumed_from_staged_checkpoint"] is False
        assert out["dedup_scanned_rows"] == 2493
        assert out["dedup_vec_query_count"] == 146
        assert out["dedup_vec_candidates_returned"] == 2493
        assert out["dedup_vec_candidate_limit"] == 64
        assert out["dedup_vec_limit_hits"] == 0
        assert out["dedup_fts_query_count"] == 192
        assert out["dedup_fts_candidates_returned"] == 2493
        assert out["embedding_cache_requested"] == 146
        assert out["embedding_cache_hits"] == 125
        assert out["staged_semantic_duplicate_facts_collapsed"] == 2
        assert out["staged_semantic_llm_checks"] == 5
        assert out["staged_semantic_llm_same_hits"] == 2

    def test_run_runtime_rolling_obd_extract_requires_state_clear(self, tmp_path, monkeypatch):
        session_id = "obd-compaction-0001"
        state_path = tmp_path / rpb._BENCHMARK_QUAID_INSTANCE / "data" / "rolling-extraction" / f"{session_id}.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text("{}", encoding="utf-8")

        monkeypatch.setattr(
            rpb,
            "_run_runtime_rolling_driver",
            lambda **kwargs: {
                "signals_processed": 1,
                "cursor_line_offset": 1,
                "total_lines": 1,
                "rolling_state_exists": True,
            },
        )
        monkeypatch.setattr(
            rpb,
            "_load_rolling_metric_rows",
            lambda *_args, **_kwargs: [
                {
                    "event": "rolling_flush",
                    "session_id": session_id,
                    "final_raw_fact_count": 1,
                    "final_facts_stored": 1,
                    "final_facts_skipped": 0,
                    "final_edges_created": 0,
                    "snippets_count": 2,
                    "journals_count": 1,
                    "project_logs_seen": 3,
                    "project_logs_written": 2,
                    "project_logs_projects_updated": 1,
                }
            ],
        )
        monkeypatch.setattr(rpb, "_write_runtime_rolling_signal", lambda **kwargs: None)
        monkeypatch.setattr(
            rpb,
            "_save_rolling_pre_publish_checkpoint",
            lambda *args, **kwargs: {"mode": "rolling-pre-publish"},
        )

        with pytest.raises(RuntimeError, match="staged state still exists"):
            rpb._run_runtime_rolling_obd_extract(
                workspace=tmp_path,
                env={},
                session_file=tmp_path / "obd.jsonl",
                session_id=session_id,
                chunk_tokens=12000,
                chunk_max_lines=None,
                timeout_seconds=60,
            )

    def test_run_runtime_rolling_driver_embeds_preserved_signal_retry_guard(self, tmp_path, monkeypatch):
        transcript = tmp_path / "obd.jsonl"
        transcript.write_text('{"role":"user","content":"hi"}\n', encoding="utf-8")

        captured = {}

        def _run(cmd, **kwargs):
            captured["driver_code"] = cmd[2]
            captured["env"] = dict(kwargs.get("env") or {})
            result = type("Result", (), {})()
            result.returncode = 0
            result.stdout = (
                '{\n'
                '  "session_id": "obd-compaction-0001",\n'
                '  "signals_processed": 0,\n'
                '  "signal_loops": 0,\n'
                '  "cursor_line_offset": 0,\n'
                '  "cursor_transcript_path": "",\n'
                '  "total_lines": 1,\n'
                '  "rolling_state_exists": false,\n'
                '  "rolling_state_path": "/tmp/state.json",\n'
                '  "metrics_path": "/tmp/rolling.jsonl",\n'
                '  "remaining_tokens": 0\n'
                '}\n'
            )
            result.stderr = ""
            return result

        monkeypatch.setattr(rpb.subprocess, "run", _run)
        monkeypatch.setattr(rpb, "_BACKEND", "oauth")
        monkeypatch.setattr(rpb, "_find_anthropic_credential", lambda: "sk-ant-oat01-test-token")

        out = rpb._run_runtime_rolling_driver(
            workspace=tmp_path,
            env={},
            session_id="obd-compaction-0001",
            transcript_path=transcript,
            timeout_seconds=60,
            chunk_tokens=12000,
            final_signal="compaction",
        )

        assert out["signals_processed"] == 0
        assert captured["env"]["BENCHMARK_ROLLING_MAX_PRESERVED_SIGNAL_RETRIES"] == "60"
        assert "max_repeated_signal_retries" in captured["driver_code"]
        assert "rolling driver signal preserved after" in captured["driver_code"]
        assert "os.environ['QUAID_CAPTURE_CHUNK_TOKENS'] = chunk_raw" in captured["driver_code"]

    def test_run_runtime_rolling_driver_preserves_explicit_retry_override(self, tmp_path, monkeypatch):
        transcript = tmp_path / "obd.jsonl"
        transcript.write_text('{"role":"user","content":"hi"}\n', encoding="utf-8")

        captured = {}

        def _run(cmd, **kwargs):
            captured["env"] = dict(kwargs.get("env") or {})
            result = _FakeSubprocessResult()
            result.stdout = '{"signals_processed": 0}\n'
            return result

        monkeypatch.setattr(rpb.subprocess, "run", _run)
        monkeypatch.setattr(rpb, "_BACKEND", "oauth")
        monkeypatch.setattr(rpb, "_find_anthropic_credential", lambda: "sk-ant-oat01-test-token")

        rpb._run_runtime_rolling_driver(
            workspace=tmp_path,
            env={"BENCHMARK_ROLLING_MAX_PRESERVED_SIGNAL_RETRIES": "7"},
            session_id="obd-compaction-0001",
            transcript_path=transcript,
            timeout_seconds=60,
            chunk_tokens=12000,
            final_signal="compaction",
        )

        assert captured["env"]["BENCHMARK_ROLLING_MAX_PRESERVED_SIGNAL_RETRIES"] == "7"

    def test_run_runtime_rolling_driver_records_full_failure_context(self, tmp_path, monkeypatch):
        transcript = tmp_path / "obd.jsonl"
        transcript.write_text('{"role":"user","content":"hi"}\n', encoding="utf-8")

        def _run(cmd, **kwargs):
            result = _FakeSubprocessResult()
            result.returncode = 1
            result.stdout = "stdout-line\n" * 80
            result.stderr = "stderr-line\n" * 80
            return result

        monkeypatch.setattr(rpb.subprocess, "run", _run)
        monkeypatch.setattr(rpb, "_BACKEND", "oauth")
        monkeypatch.setattr(rpb, "_find_anthropic_credential", lambda: "sk-ant-oat01-test-token")

        with pytest.raises(RuntimeError, match="runtime_rolling_driver_failure.json"):
            rpb._run_runtime_rolling_driver(
                workspace=tmp_path,
                env={},
                session_id="obd-compaction-0001",
                transcript_path=transcript,
                timeout_seconds=60,
                chunk_tokens=12000,
                final_signal="compaction",
            )

        failure = json.loads((tmp_path / "logs" / "runtime_rolling_driver_failure.json").read_text(encoding="utf-8"))
        assert failure["returncode"] == 1
        assert failure["session_id"] == "obd-compaction-0001"
        assert (tmp_path / "logs" / "runtime_rolling_driver_failure.stdout.log").read_text(encoding="utf-8").count("stdout-line") == 80
        assert (tmp_path / "logs" / "runtime_rolling_driver_failure.stderr.log").read_text(encoding="utf-8").count("stderr-line") == 80

    def test_run_runtime_rolling_driver_rehydrates_anthropic_auth(self, tmp_path, monkeypatch):
        transcript = tmp_path / "obd.jsonl"
        transcript.write_text('{"role":"user","content":"hi"}\n', encoding="utf-8")

        captured = {}

        def _run(cmd, **kwargs):
            captured["env"] = dict(kwargs.get("env") or {})
            result = type("Result", (), {})()
            result.returncode = 0
            result.stdout = (
                '{\n'
                '  "session_id": "obd-compaction-0001",\n'
                '  "signals_processed": 0,\n'
                '  "signal_loops": 0,\n'
                '  "cursor_line_offset": 0,\n'
                '  "cursor_transcript_path": "",\n'
                '  "total_lines": 1,\n'
                '  "rolling_state_exists": false,\n'
                '  "rolling_state_path": "/tmp/state.json",\n'
                '  "metrics_path": "/tmp/rolling.jsonl",\n'
                '  "remaining_tokens": 0\n'
                '}\n'
            )
            result.stderr = ""
            return result

        monkeypatch.setattr(rpb.subprocess, "run", _run)
        monkeypatch.setattr(rpb, "_BACKEND", "oauth")
        monkeypatch.setattr(rpb, "_find_anthropic_credential", lambda: "sk-ant-oat01-test-token")

        rpb._run_runtime_rolling_driver(
            workspace=tmp_path,
            env={},
            session_id="obd-compaction-0001",
            transcript_path=transcript,
            timeout_seconds=60,
            chunk_tokens=12000,
            final_signal=None,
        )

        assert captured["env"]["BENCHMARK_ANTHROPIC_OAUTH_TOKEN"] == "sk-ant-oat01-test-token"
        assert captured["env"]["ANTHROPIC_API_KEY"] == "sk-ant-oat01-test-token"

    def test_run_runtime_rolling_driver_overwrites_blank_anthropic_auth(self, tmp_path, monkeypatch):
        transcript = tmp_path / "obd.jsonl"
        transcript.write_text('{"role":"user","content":"hi"}\n', encoding="utf-8")

        captured = {}

        def _run(cmd, **kwargs):
            captured["env"] = dict(kwargs.get("env") or {})
            result = type("Result", (), {})()
            result.returncode = 0
            result.stdout = (
                '{\n'
                '  "session_id": "obd-compaction-0001",\n'
                '  "signals_processed": 0,\n'
                '  "signal_loops": 0,\n'
                '  "cursor_line_offset": 0,\n'
                '  "cursor_transcript_path": "",\n'
                '  "total_lines": 1,\n'
                '  "rolling_state_exists": false,\n'
                '  "rolling_state_path": "/tmp/state.json",\n'
                '  "metrics_path": "/tmp/rolling.jsonl",\n'
                '  "remaining_tokens": 0\n'
                '}\n'
            )
            result.stderr = ""
            return result

        monkeypatch.setattr(rpb.subprocess, "run", _run)
        monkeypatch.setattr(rpb, "_BACKEND", "oauth")
        monkeypatch.setattr(rpb, "_find_anthropic_credential", lambda: "sk-ant-oat01-test-token")

        rpb._run_runtime_rolling_driver(
            workspace=tmp_path,
            env={
                "BENCHMARK_ANTHROPIC_OAUTH_TOKEN": "",
                "ANTHROPIC_API_KEY": "",
            },
            session_id="obd-compaction-0001",
            transcript_path=transcript,
            timeout_seconds=60,
            chunk_tokens=12000,
            final_signal=None,
        )

        assert captured["env"]["BENCHMARK_ANTHROPIC_OAUTH_TOKEN"] == "sk-ant-oat01-test-token"
        assert captured["env"]["ANTHROPIC_API_KEY"] == "sk-ant-oat01-test-token"

    def test_run_runtime_rolling_driver_resolves_blank_oauth_auth_from_api_key(self, tmp_path, monkeypatch):
        transcript = tmp_path / "obd.jsonl"
        transcript.write_text('{"role":"user","content":"hi"}\n', encoding="utf-8")

        captured = {"api_key_calls": 0}

        def _run(cmd, **kwargs):
            captured["env"] = dict(kwargs.get("env") or {})
            result = type("Result", (), {})()
            result.returncode = 0
            result.stdout = (
                '{\n'
                '  "session_id": "obd-compaction-0001",\n'
                '  "signals_processed": 0,\n'
                '  "signal_loops": 0,\n'
                '  "cursor_line_offset": 0,\n'
                '  "cursor_transcript_path": "",\n'
                '  "total_lines": 1,\n'
                '  "rolling_state_exists": false,\n'
                '  "rolling_state_path": "/tmp/state.json",\n'
                '  "metrics_path": "/tmp/rolling.jsonl",\n'
                '  "remaining_tokens": 0\n'
                '}\n'
            )
            result.stderr = ""
            return result

        def _get_api_key():
            captured["api_key_calls"] += 1
            return "sk-ant-oat01-test-token"

        monkeypatch.setattr(rpb.subprocess, "run", _run)
        monkeypatch.setattr(rpb, "_BACKEND", "oauth")
        monkeypatch.setattr(rpb, "_find_anthropic_credential", lambda: "")
        monkeypatch.setattr(rpb, "_get_api_key", _get_api_key)

        rpb._run_runtime_rolling_driver(
            workspace=tmp_path,
            env={
                "BENCHMARK_ANTHROPIC_OAUTH_TOKEN": "",
                "ANTHROPIC_API_KEY": "",
            },
            session_id="obd-compaction-0001",
            transcript_path=transcript,
            timeout_seconds=60,
            chunk_tokens=12000,
            final_signal=None,
        )

        assert captured["api_key_calls"] == 1
        assert captured["env"]["BENCHMARK_ANTHROPIC_OAUTH_TOKEN"] == "sk-ant-oat01-test-token"
        assert captured["env"]["ANTHROPIC_API_KEY"] == "sk-ant-oat01-test-token"

    def test_obd_propagates_runtime_extract_wall_timeout(self, tmp_path, monkeypatch):
        import sqlite3
        from types import SimpleNamespace

        db_dir = tmp_path / "data"
        db_dir.mkdir(parents=True)
        (tmp_path / "config").mkdir(parents=True, exist_ok=True)
        (tmp_path / "config" / "memory.json").write_text(json.dumps({"capture": {}}))
        conn = sqlite3.connect(str(db_dir / "memory.db"))
        conn.execute("CREATE TABLE nodes (status TEXT)")
        conn.execute("CREATE TABLE edges (id INTEGER)")
        conn.commit()
        conn.close()

        @dataclass
        class _Review:
            session_num: int
            transcript_turns: list

        captured = {}

        monkeypatch.setenv("BENCHMARK_OBD_EXTRACT_TIMEOUT", "7200")
        monkeypatch.setenv("BENCHMARK_OBD_DISABLE_CARRY_CONTEXT", "1")
        monkeypatch.setenv("BENCHMARK_OBD_PARALLEL_ROOT_WORKERS", "4")
        monkeypatch.setenv("BENCHMARK_OBD_CHUNK_TOKENS", "12000")
        monkeypatch.setenv("BENCHMARK_OBD_CHUNK_MAX_LINES", "96")
        monkeypatch.setattr(
            rpb,
            "_load_reviews_with_dataset_gate",
            lambda _max_sessions: (tmp_path, None, [_Review(20, [{"maya": "hi", "agent": "ok"}])], "v1", 268),
        )
        monkeypatch.setattr(rpb, "_sync_final_project_states", lambda _workspace: None)
        monkeypatch.setattr(rpb, "_operational_day", lambda _review: "2026-05-26")
        monkeypatch.setattr(rpb, "_build_obd_message_stream", lambda _reviews: [{"role": "user", "content": "hello"}])
        monkeypatch.setattr(rpb, "_sync_final_project_states", lambda _workspace: None)
        monkeypatch.setattr(rpb, "_benchmark_env", lambda _workspace, _phase: {"BASE": "1"})
        monkeypatch.setattr(rpb, "_with_quaid_now", lambda env, _day: dict(env))
        monkeypatch.setattr(rpb, "_write_session_jsonl", lambda _messages, _path: None)
        monkeypatch.setattr(rpb, "_python_cmd_for_quaid_script", lambda _script: [sys.executable, "-m", "stub"])

        def _run_runtime_extract_jsonl(**kwargs):
            captured["env"] = dict(kwargs["env"])
            return {
                "facts": [],
                "facts_stored": 0,
                "facts_skipped": 0,
                "edges_created": 0,
                "snippets": {},
                "journal": {},
                "project_logs": {},
            }

        monkeypatch.setattr(rpb, "_run_runtime_extract_jsonl", _run_runtime_extract_jsonl)

        out = rpb.run_per_day_extraction(
            workspace=tmp_path,
            api_key="test-key",
            model="claude-sonnet-4-6",
            run_janitor_each_day=False,
            schedule_mode="obd",
        )

        assert captured["env"]["QUAID_EXTRACT_WALL_TIMEOUT"] == "7200"
        assert captured["env"]["QUAID_EXTRACT_DISABLE_CARRY_CONTEXT"] == "1"
        assert captured["env"]["QUAID_EXTRACT_PARALLEL_ROOT_WORKERS"] == "4"
        cfg = json.loads((tmp_path / "config" / "memory.json").read_text())
        assert cfg["capture"]["chunkTokens"] == 12000
        assert cfg["capture"]["chunk_max_lines"] == 96
        checkpoint_meta = json.loads((tmp_path / "logs" / "obd_post_extract_checkpoint.json").read_text())
        assert checkpoint_meta["mode"] == "obd-post-extract"
        assert out["days"] == 1
        assert out["compaction_events"] == 1

    def test_rolling_obd_writes_post_extract_checkpoint(self, tmp_path, monkeypatch):
        import sqlite3
        from dataclasses import dataclass

        workspace = tmp_path / "ws"
        (workspace / "logs").mkdir(parents=True, exist_ok=True)
        (workspace / "data").mkdir(parents=True, exist_ok=True)
        (workspace / "config").mkdir(parents=True, exist_ok=True)
        (workspace / "config" / "memory.json").write_text(json.dumps({"capture": {}}))

        conn = sqlite3.connect(str(workspace / "data" / "memory.db"))
        conn.execute("CREATE TABLE nodes (id TEXT PRIMARY KEY, status TEXT)")
        conn.execute("CREATE TABLE edges (id TEXT PRIMARY KEY)")
        conn.commit()
        conn.close()

        @dataclass
        class _Review:
            session_num: int
            transcript_turns: list

        monkeypatch.setattr(
            rpb,
            "_load_reviews_with_dataset_gate",
            lambda _max_sessions: (
                tmp_path,
                None,
                [_Review(20, [{"maya": "hi", "agent": "ok"}])],
                "v1",
                268,
            ),
        )
        monkeypatch.setattr(rpb, "_operational_day", lambda _review: "2026-05-26")
        monkeypatch.setattr(
            rpb,
            "_build_obd_message_stream",
            lambda _reviews: [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
        )
        monkeypatch.setattr(rpb, "_sync_final_project_states", lambda _workspace: None)
        monkeypatch.setattr(rpb, "_benchmark_env", lambda _workspace, _phase: {"BASE": "1"})
        monkeypatch.setattr(rpb, "_with_quaid_now", lambda env, _day: dict(env))

        def _write_session(messages, path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(json.dumps(m) for m in messages))

        monkeypatch.setattr(rpb, "_write_session_jsonl", _write_session)
        captured = {}
        monkeypatch.setenv("BENCHMARK_OBD_CHUNK_MAX_LINES", "96")
        monkeypatch.setattr(
            rpb,
            "_run_runtime_rolling_obd_extract",
            lambda **_kwargs: captured.update(_kwargs) or {
                "facts_extracted": 4,
                "facts_stored": 3,
                "facts_skipped": 1,
                "edges_created": 2,
                "snippets_count": 2,
                "journals_count": 1,
                "root_chunks": 5,
                "split_events": 1,
                "split_child_chunks": 2,
                "leaf_chunks": 6,
                "max_split_depth": 1,
                "deep_calls": 7,
                "repair_calls": 0,
                "carry_context_enabled": True,
                "parallel_root_workers": 1,
                "rolling_batches": 4,
                "rolling_stage_events": 4,
                "rolling_stage_wall_seconds": 12.5,
                "rolling_driver_stage_wall_seconds": 13.0,
                "rolling_driver_flush_wall_seconds": 2.5,
                "signal_to_publish_seconds": 2.4,
                "flush_wall_seconds": 2.4,
                "extract_wall_seconds": 1.8,
                "publish_wall_seconds": 0.6,
                "project_log_metrics": {"entries_seen": 3, "entries_written": 2, "projects_updated": 1},
            },
        )

        out = rpb.run_per_day_extraction(
            workspace=workspace,
            api_key="test-key",
            model="claude-sonnet-4-6",
            run_janitor_each_day=False,
            schedule_mode="rolling-obd",
        )

        checkpoint_meta = json.loads((workspace / "logs" / "obd_post_extract_checkpoint.json").read_text())
        assert checkpoint_meta["mode"] == "obd-post-extract"
        assert checkpoint_meta["stats"]["facts_stored"] == 3
        assert checkpoint_meta["stats"]["rolling_batches"] == 4
        assert checkpoint_meta["stats"]["signal_to_publish_seconds"] == 2.4
        assert checkpoint_meta["stats"]["root_chunks"] == 5
        assert checkpoint_meta["stats"]["snippets"] == 2
        assert checkpoint_meta["stats"]["journals"] == 1
        assert checkpoint_meta["stats"]["project_logs_seen"] == 3
        assert checkpoint_meta["stats"]["project_logs_written"] == 2
        assert checkpoint_meta["stats"]["projects_updated"] == 1
        assert captured["chunk_max_lines"] == 96
        assert out["schedule_mode"] == "rolling-obd"
        assert out["signal_to_publish_seconds"] == 2.4

    def test_obd_writes_post_extract_checkpoint(self, tmp_path, monkeypatch):
        import sqlite3
        from dataclasses import dataclass

        workspace = tmp_path / "ws"
        (workspace / "logs").mkdir(parents=True, exist_ok=True)
        (workspace / "data").mkdir(parents=True, exist_ok=True)
        (workspace / "config").mkdir(parents=True, exist_ok=True)
        (workspace / "config" / "memory.json").write_text(json.dumps({"capture": {}}))

        conn = sqlite3.connect(str(workspace / "data" / "memory.db"))
        conn.execute("CREATE TABLE nodes (id TEXT PRIMARY KEY, status TEXT)")
        conn.execute("CREATE TABLE edges (id TEXT PRIMARY KEY)")
        conn.commit()
        conn.close()

        @dataclass
        class _Review:
            session_num: int
            transcript_turns: list

        monkeypatch.setattr(
            rpb,
            "_load_reviews_with_dataset_gate",
            lambda _max_sessions: (
                tmp_path,
                None,
                [_Review(20, [{"maya": "hi", "agent": "ok"}])],
                "v1",
                268,
            ),
        )
        monkeypatch.setattr(rpb, "_operational_day", lambda _review: "2026-05-26")
        monkeypatch.setattr(
            rpb,
            "_build_obd_message_stream",
            lambda _reviews: [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
        )
        monkeypatch.setattr(rpb, "_sync_final_project_states", lambda _workspace: None)
        monkeypatch.setattr(rpb, "_benchmark_env", lambda _workspace, _phase: {"BASE": "1"})
        monkeypatch.setattr(rpb, "_with_quaid_now", lambda env, _day: dict(env))

        def _write_session(messages, path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(json.dumps(m) for m in messages))

        monkeypatch.setattr(rpb, "_write_session_jsonl", _write_session)
        monkeypatch.setattr(
            rpb,
            "_run_runtime_extract_jsonl",
            lambda **_kwargs: {
                "facts": [{"id": "f1"}],
                "facts_stored": 1,
                "facts_skipped": 0,
                "edges_created": 2,
                "root_chunks": 3,
                "split_events": 2,
                "split_child_chunks": 5,
                "leaf_chunks": 6,
                "max_split_depth": 2,
                "deep_calls": 7,
                "repair_calls": 1,
                "carry_context_enabled": False,
                "parallel_root_workers": 4,
                "snippets": {"USER.md": ["note"]},
                "journal": {"j1": {"content": "entry"}},
                "project_logs": {"recipe-app": ["note"]},
                "project_log_metrics": {"entries_seen": 1, "entries_written": 1, "projects_updated": 1},
            },
        )

        out = rpb.run_per_day_extraction(
            workspace=workspace,
            api_key="test-key",
            model="claude-sonnet-4-6",
            run_janitor_each_day=False,
            schedule_mode="obd",
        )

        checkpoint_meta = json.loads((workspace / "logs" / "obd_post_extract_checkpoint.json").read_text())
        snapshot_dir = Path(checkpoint_meta["snapshot_dir"])
        assert checkpoint_meta["mode"] == "obd-post-extract"
        assert checkpoint_meta["current_day"] == "2026-05-26"
        assert checkpoint_meta["stats"]["facts_stored"] == 1
        assert checkpoint_meta["stats"]["root_chunks"] == 3
        assert checkpoint_meta["stats"]["split_events"] == 2
        assert checkpoint_meta["stats"]["leaf_chunks"] == 6
        assert checkpoint_meta["stats"]["deep_calls"] == 7
        assert checkpoint_meta["stats"]["carry_context_enabled"] is False
        assert checkpoint_meta["stats"]["parallel_root_workers"] == 4
        assert snapshot_dir.exists()
        assert (snapshot_dir / "data" / "memory.db").exists()
        assert (snapshot_dir / "extraction_cache" / "obd-session-0001.jsonl").exists()
        assert out["days"] == 1
        assert out["compaction_events"] == 1

    def test_obd_skip_janitor_marks_progress_skipped(self, tmp_path, monkeypatch):
        import sqlite3
        from dataclasses import dataclass

        workspace = tmp_path / "ws"
        (workspace / "logs").mkdir(parents=True, exist_ok=True)
        (workspace / "data").mkdir(parents=True, exist_ok=True)
        (workspace / "config").mkdir(parents=True, exist_ok=True)
        (workspace / "config" / "memory.json").write_text(json.dumps({"capture": {}}))

        conn = sqlite3.connect(str(workspace / "data" / "memory.db"))
        conn.execute("CREATE TABLE nodes (id TEXT PRIMARY KEY, status TEXT)")
        conn.execute("CREATE TABLE edges (id TEXT PRIMARY KEY)")
        conn.commit()
        conn.close()

        @dataclass
        class _Review:
            session_num: int
            transcript_turns: list

        monkeypatch.setattr(
            rpb,
            "_load_reviews_with_dataset_gate",
            lambda _max_sessions: (
                tmp_path,
                None,
                [_Review(20, [{"maya": "hi", "agent": "ok"}])],
                "v1",
                268,
            ),
        )
        monkeypatch.setattr(rpb, "_operational_day", lambda _review: "2026-05-26")
        monkeypatch.setattr(
            rpb,
            "_build_obd_message_stream",
            lambda _reviews: [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
        )
        monkeypatch.setattr(rpb, "_sync_final_project_states", lambda _workspace: None)
        monkeypatch.setattr(rpb, "_benchmark_env", lambda _workspace, _phase: {"BASE": "1"})
        monkeypatch.setattr(rpb, "_with_quaid_now", lambda env, _day: dict(env))
        monkeypatch.setattr(rpb, "_write_session_jsonl", lambda _messages, _path: None)
        monkeypatch.setattr(
            rpb,
            "_run_runtime_extract_jsonl",
            lambda **_kwargs: {
                "facts": [{"id": "f1"}],
                "facts_stored": 1,
                "facts_skipped": 0,
                "edges_created": 0,
                "root_chunks": 2,
                "split_events": 0,
                "split_child_chunks": 0,
                "leaf_chunks": 2,
                "max_split_depth": 0,
                "deep_calls": 2,
                "repair_calls": 0,
                "carry_context_enabled": True,
                "parallel_root_workers": 1,
                "snippets": {},
                "journal": {},
                "project_logs": {},
                "project_log_metrics": {"entries_seen": 0, "entries_written": 0, "projects_updated": 0},
            },
        )

        out = rpb.run_per_day_extraction(
            workspace=workspace,
            api_key="test-key",
            model="claude-sonnet-4-6",
            run_janitor_each_day=False,
            schedule_mode="obd",
        )

        progress = json.loads((workspace / "logs" / "janitor_progress.json").read_text())
        assert progress["state"] == "skipped"
        assert progress["completed_days"] == 0
        assert out["janitor_runs"] == 0

class TestAnthropicCachedRetries:
    """Tests for _call_anthropic_cached HTTP retry behavior."""

    def test_retries_http_529_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(rpb, "_BACKEND", "oauth")
        monkeypatch.setenv("ANTHROPIC_RETRY_ATTEMPTS", "2")
        monkeypatch.setenv("ANTHROPIC_RETRY_BACKOFF_S", "0.01")
        monkeypatch.setenv("ANTHROPIC_RETRY_BACKOFF_CAP_S", "0.01")

        first_err = urllib.error.HTTPError(
            url="https://api.anthropic.com/v1/messages",
            code=529,
            msg="overloaded",
            hdrs={},
            fp=io.BytesIO(b'{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}'),
        )

        class _Resp:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            def read(self):
                return json.dumps(
                    {
                        "content": [{"type": "text", "text": "ok"}],
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    }
                ).encode()

        calls = {"n": 0}

        def _fake_urlopen(_req, timeout=300):
            calls["n"] += 1
            if calls["n"] == 1:
                raise first_err
            return _Resp()

        monkeypatch.setattr(rpb.urllib.request, "urlopen", _fake_urlopen)
        text, usage = rpb._call_anthropic_cached("sys", "user", "claude-haiku-4-5-20251001", "test-key")
        assert text == "ok"
        assert usage.get("input_tokens") == 1
        assert calls["n"] == 2

    def test_retries_timeout_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(rpb, "_BACKEND", "oauth")
        monkeypatch.setenv("ANTHROPIC_RETRY_ATTEMPTS", "2")
        monkeypatch.setenv("ANTHROPIC_RETRY_BACKOFF_S", "0.01")
        monkeypatch.setenv("ANTHROPIC_RETRY_BACKOFF_CAP_S", "0.01")
        monkeypatch.setattr(rpb.random, "uniform", lambda a, b: 0.0)
        monkeypatch.setattr(rpb.time, "sleep", lambda _s: None)

        class _Resp:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            def read(self):
                return json.dumps(
                    {
                        "content": [{"type": "text", "text": "ok"}],
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    }
                ).encode()

        calls = {"n": 0}

        def _fake_urlopen(_req, timeout=300):
            calls["n"] += 1
            if calls["n"] == 1:
                raise TimeoutError("The read operation timed out")
            return _Resp()

        monkeypatch.setattr(rpb.urllib.request, "urlopen", _fake_urlopen)

        text, usage = rpb._call_anthropic_cached("sys", "user", "claude-haiku-4-5-20251001", "test-key")

        assert text == "ok"
        assert usage.get("input_tokens") == 1
        assert calls["n"] == 2

    def test_vllm_backend_routes_to_openai_compatible_chat(self, monkeypatch):
        monkeypatch.setattr(rpb, "_BACKEND", "vllm")

        seen: dict[str, object] = {}

        def _fake_call(**kwargs):
            seen.update(kwargs)
            return (
                {"choices": [{"message": {"content": "gemma ok"}}], "model": "gemma-3-31b-it"},
                {"input_tokens": 7, "output_tokens": 3, "api_calls": 1},
            )

        monkeypatch.setattr(rpb, "_call_openai_compatible_chat", _fake_call)

        text, usage = rpb._call_anthropic_cached(
            "system prompt",
            "user prompt",
            "gemma-3-31b-it",
            "ignored-key",
            max_tokens=64,
        )

        assert text == "gemma ok"
        assert usage["input_tokens"] == 7
        assert seen["model"] == "gemma-3-31b-it"
        assert seen["max_tokens"] == 64
        assert seen["timeout"] == 300
        assert seen["provider"] == "vllm"
        assert seen["messages"] == [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "user prompt"},
        ]

    def test_llama_cpp_backend_routes_to_openai_compatible_chat(self, monkeypatch):
        monkeypatch.setattr(rpb, "_BACKEND", "llama-cpp")

        seen: dict[str, object] = {}

        def _fake_call(**kwargs):
            seen.update(kwargs)
            return (
                {"choices": [{"message": {"content": "llama ok"}}], "model": "gemma-3-31b-it"},
                {"input_tokens": 5, "output_tokens": 2, "api_calls": 1},
            )

        monkeypatch.setattr(rpb, "_call_openai_compatible_chat", _fake_call)

        text, usage = rpb._call_anthropic_cached(
            "system prompt",
            "user prompt",
            "gemma-3-31b-it",
            "ignored-key",
            max_tokens=64,
        )

        assert text == "llama ok"
        assert usage["input_tokens"] == 5
        assert seen["provider"] == "llama-cpp"


def test_tool_use_loop_api_sets_temperature_zero(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setattr(rpb, "_BACKEND", "oauth")
    monkeypatch.setattr(rpb, "_append_usage_event", lambda *a, **k: None)

    seen_payloads = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": "final answer"}],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            ).encode()

    def _fake_urlopen(req, timeout=120):
        seen_payloads.append(json.loads(req.data.decode()))
        return _Resp()

    monkeypatch.setattr(rpb.urllib.request, "urlopen", _fake_urlopen)

    answer, tool_calls, tool_logs, retrieval_texts, usage = rpb._tool_use_loop(
        question="What is the answer?",
        eval_context="ctx",
        workspace=workspace,
        api_key="test-key",
        env={},
        model="claude-haiku-4-5-20251001",
        context_inject=False,
    )

    assert answer == "final answer"
    assert tool_calls == []
    assert tool_logs == []
    assert retrieval_texts == []
    assert usage["api_calls"] == 1
    assert seen_payloads[0]["temperature"] == 0.0
    assert seen_payloads[0]["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_tool_use_loop_api_caches_static_eval_context_but_not_injected_recall(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setattr(rpb, "_BACKEND", "oauth")
    monkeypatch.setattr(rpb, "_append_usage_event", lambda *a, **k: None)
    monkeypatch.setattr(
        rpb,
        "_pre_recall",
        lambda *a, **k: ("Fact one\\nFact two", "query used", {"stop_reason": "", "harness_telemetry": {}}),
    )

    seen_payloads = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": "final answer"}],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            ).encode()

    def _fake_urlopen(req, timeout=120):
        seen_payloads.append(json.loads(req.data.decode()))
        return _Resp()

    monkeypatch.setattr(rpb.urllib.request, "urlopen", _fake_urlopen)

    answer, tool_calls, tool_logs, retrieval_texts, usage = rpb._tool_use_loop(
        question="What is the answer?",
        eval_context="ctx",
        workspace=workspace,
        api_key="sk-ant-oat01-test-token",
        env={},
        model="claude-haiku-4-5-20251001",
        context_inject=True,
    )

    assert answer == "final answer"
    assert tool_calls == ["memory_recall(pre-inject)"]
    assert retrieval_texts == ["Fact one\\nFact two"]
    assert usage["api_calls"] == 1
    system_blocks = seen_payloads[0]["system"]
    assert system_blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert system_blocks[1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in system_blocks[2]
    assert "Retrieved Memories" in system_blocks[2]["text"]


def test_call_anthropic_cached_supports_cached_user_prefix_blocks(monkeypatch):
    monkeypatch.setattr(rpb, "_BACKEND", "oauth")
    seen_payloads = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {"input_tokens": 10, "output_tokens": 2},
                }
            ).encode()

    def _fake_urlopen(req, timeout=300):
        seen_payloads.append(json.loads(req.data.decode()))
        return _Resp()

    monkeypatch.setattr(rpb.urllib.request, "urlopen", _fake_urlopen)

    text, usage = rpb._call_anthropic_cached(
        "sys",
        [
            {"text": "full transcript", "cache": True},
            {"text": "Question: hi", "cache": False},
        ],
        "claude-haiku-4-5-20251001",
        "test-key",
        max_tokens=32,
    )

    assert text == "ok"
    assert usage["input_tokens"] == 10
    payload = seen_payloads[0]
    user_blocks = payload["messages"][0]["content"]
    assert user_blocks[0]["text"] == "full transcript"
    assert user_blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert user_blocks[1]["text"] == "Question: hi"
    assert "cache_control" not in user_blocks[1]


def test_tool_use_loop_api_retries_timeout_once_then_succeeds(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setattr(rpb, "_BACKEND", "oauth")
    monkeypatch.setattr(rpb, "_append_usage_event", lambda *a, **k: None)
    monkeypatch.setattr(rpb.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(rpb.random, "uniform", lambda *_a, **_k: 0.0)
    monkeypatch.setenv("ANTHROPIC_TOOL_USE_RETRY_ATTEMPTS", "2")

    calls = {"n": 0}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": "final answer"}],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            ).encode()

    def _fake_urlopen(_req, timeout=120):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("The read operation timed out")
        return _Resp()

    monkeypatch.setattr(rpb.urllib.request, "urlopen", _fake_urlopen)

    answer, tool_calls, tool_logs, retrieval_texts, usage = rpb._tool_use_loop(
        question="What is the answer?",
        eval_context="ctx",
        workspace=workspace,
        api_key="test-key",
        env={},
        model="claude-haiku-4-5-20251001",
        context_inject=False,
    )

    assert calls["n"] == 2
    assert answer == "final answer"
    assert tool_calls == []
    assert tool_logs == []
    assert retrieval_texts == []
    assert usage["api_calls"] == 1


def test_tool_use_loop_api_timeout_fails_hard_after_retry(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setattr(rpb, "_BACKEND", "oauth")
    monkeypatch.setattr(rpb, "_append_usage_event", lambda *a, **k: None)
    monkeypatch.setattr(rpb.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(rpb.random, "uniform", lambda *_a, **_k: 0.0)
    monkeypatch.setenv("ANTHROPIC_TOOL_USE_RETRY_ATTEMPTS", "2")

    calls = {"n": 0}

    def _fake_urlopen(_req, timeout=120):
        calls["n"] += 1
        raise TimeoutError("The read operation timed out")

    monkeypatch.setattr(rpb.urllib.request, "urlopen", _fake_urlopen)

    with pytest.raises(RuntimeError, match="Eval answer model timeout"):
        rpb._tool_use_loop(
            question="What is the answer?",
            eval_context="ctx",
            workspace=workspace,
            api_key="test-key",
            env={},
            model="claude-haiku-4-5-20251001",
            context_inject=False,
        )

    assert calls["n"] == 2


def test_tool_use_loop_vllm_dispatches_to_openai_compatible_loop(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setattr(rpb, "_BACKEND", "vllm")

    seen: dict[str, object] = {}

    def _fake_vllm_loop(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return ("final answer", ["recall"], ["recall(test)"], ["memory"], {"api_calls": 1})

    monkeypatch.setattr(rpb, "_tool_use_loop_openai_compatible", _fake_vllm_loop)

    answer, tool_calls, tool_logs, retrieval_texts, usage = rpb._tool_use_loop(
        question="What is the answer?",
        eval_context="ctx",
        workspace=workspace,
        api_key="ignored",
        env={"X": "1"},
        model="gemma-3-31b-it",
        context_inject=False,
    )

    assert answer == "final answer"
    assert tool_calls == ["recall"]
    assert tool_logs == ["recall(test)"]
    assert retrieval_texts == ["memory"]
    assert usage["api_calls"] == 1
    assert seen["args"] == ("What is the answer?", "ctx", workspace, {"X": "1"})
    assert seen["kwargs"]["model"] == "gemma-3-31b-it"
    assert seen["kwargs"]["context_inject"] is False


def test_tool_use_loop_llama_cpp_dispatches_to_openai_compatible_loop(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setattr(rpb, "_BACKEND", "llama-cpp")

    seen: dict[str, object] = {}

    def _fake_loop(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return ("final answer", ["recall"], ["recall(test)"], ["memory"], {"api_calls": 1})

    monkeypatch.setattr(rpb, "_tool_use_loop_openai_compatible", _fake_loop)

    answer, tool_calls, tool_logs, retrieval_texts, usage = rpb._tool_use_loop(
        question="What is the answer?",
        eval_context="ctx",
        workspace=workspace,
        api_key="ignored",
        env={"X": "1"},
        model="gemma-3-31b-it",
        context_inject=False,
    )

    assert answer == "final answer"
    assert tool_calls == ["recall"]
    assert usage["api_calls"] == 1
    assert seen["args"] == ("What is the answer?", "ctx", workspace, {"X": "1"})
    assert seen["kwargs"]["model"] == "gemma-3-31b-it"


def test_tool_use_loop_openai_compatible_uses_timeout_env_override_with_relax_flag(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setenv("OPENAI_COMPAT_ANSWER_TIMEOUT_S", "600")
    monkeypatch.setenv("BENCHMARK_RELAX_TIMEOUTS", "1")
    monkeypatch.setattr(rpb, "_BACKEND", "llama-cpp")
    monkeypatch.setattr(rpb, "_OPENAI_COMPAT_URL", "http://127.0.0.1:30001")

    seen: dict[str, object] = {}

    def _fake_pre_recall(*args, **kwargs):
        return ("", "recent plans", {"stop_reason": "no_memories"})

    def _fake_chat(*, timeout, **kwargs):
        seen["timeout"] = timeout
        return (
            {
                "choices": [
                    {
                        "message": {
                            "content": "Final answer.",
                            "tool_calls": [],
                        }
                    }
                ]
            },
            {
                "input_tokens": 11,
                "output_tokens": 7,
                "api_calls": 1,
                "model_usage": {
                    "gemma-4-31b-q8": {
                        "input_tokens": 11,
                        "output_tokens": 7,
                        "total_tokens": 18,
                    }
                },
            },
        )

    monkeypatch.setattr(rpb, "_pre_recall", _fake_pre_recall)
    monkeypatch.setattr(rpb, "_call_openai_compatible_chat", _fake_chat)

    answer, tool_calls, tool_logs, retrieval_texts, usage = rpb._tool_use_loop_openai_compatible(
        question="exercise habits recent plans",
        eval_context="ctx",
        workspace=workspace,
        env={},
        model="gemma-4-31b-q8",
        context_inject=True,
    )

    assert answer == "Final answer."
    assert tool_calls == []
    assert tool_logs == []
    assert retrieval_texts == []
    assert usage["api_calls"] == 1
    assert seen["timeout"] == 600


def test_tool_use_loop_openai_compatible_replays_gemma_tool_turns_as_plain_messages(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()

    monkeypatch.setattr(rpb, "_BACKEND", "llama-cpp")
    monkeypatch.setattr(rpb, "_openai_compatible_backend_label", lambda: "llama-cpp")
    monkeypatch.setattr(rpb, "_pre_recall", lambda *a, **k: ("", "q", {"stop_reason": "no_memories"}))

    seen_messages = []
    call_count = {"n": 0}

    def _fake_chat(*, messages, **kwargs):
        seen_messages.append(messages)
        call_count["n"] += 1
        if call_count["n"] == 1:
            return (
                {
                    "model": "gemma-4-26b-q6k",
                    "choices": [
                        {
                            "message": {
                                "content": "<|channel>thought\n<channel|><|tool_call>call:recall{query:<|\"|>tech stack<|\"|>}<tool_call|>",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "function": {"name": "recall", "arguments": "{\"query\":\"tech stack\"}"},
                                    }
                                ],
                            }
                        }
                    ],
                },
                {"api_calls": 1, "input_tokens": 10, "output_tokens": 5, "model_usage": {}},
            )
        return (
            {
                "model": "gemma-4-26b-q6k",
                "choices": [{"message": {"content": "Final answer."}}],
            },
            {"api_calls": 1, "input_tokens": 10, "output_tokens": 5, "model_usage": {}},
        )

    monkeypatch.setattr(rpb, "_call_openai_compatible_chat", _fake_chat)
    monkeypatch.setattr(
        rpb,
        "_execute_tool",
        lambda *a, **k: ("tool result text", {"turn_details": [{"planner": {}}], "harness_telemetry": {}}),
    )

    answer, tool_calls, tool_logs, retrieval_texts, usage = rpb._tool_use_loop_openai_compatible(
        question="What stack?",
        eval_context="ctx",
        workspace=workspace,
        env={},
        model="gemma-4-26b-q6k",
        context_inject=True,
    )

    assert answer == "Final answer."
    assert tool_calls == ["recall"]
    assert retrieval_texts == ["tool result text"]
    replay_messages = seen_messages[1]
    assert any(m["role"] == "user" and m["content"].startswith("[Tool recall result]\n") for m in replay_messages)
    assert not any(m.get("tool_calls") for m in replay_messages)
    assert not any(m["role"] == "tool" for m in replay_messages)


def test_openai_compatible_answer_timeout_ignores_env_without_relax_flag(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_ANSWER_TIMEOUT_S", "600")
    monkeypatch.delenv("BENCHMARK_RELAX_TIMEOUTS", raising=False)
    monkeypatch.setattr(rpb, "_BACKEND", "llama-cpp")
    monkeypatch.setattr(rpb, "_OPENAI_COMPAT_URL", "http://127.0.0.1:30001")

    assert rpb._openai_compatible_answer_timeout_s() == 120


def test_openai_compatible_answer_timeout_env_validation_requires_relax_flag(monkeypatch):
    monkeypatch.setenv("BENCHMARK_RELAX_TIMEOUTS", "1")
    monkeypatch.setattr(rpb, "_BACKEND", "llama-cpp")
    monkeypatch.setattr(rpb, "_OPENAI_COMPAT_URL", "http://127.0.0.1:30001")

    monkeypatch.setenv("OPENAI_COMPAT_ANSWER_TIMEOUT_S", "slow")
    with pytest.raises(RuntimeError, match="must be an integer"):
        rpb._openai_compatible_answer_timeout_s()

    monkeypatch.setenv("OPENAI_COMPAT_ANSWER_TIMEOUT_S", "0")
    assert rpb._openai_compatible_answer_timeout_s() is None

    monkeypatch.setenv("OPENAI_COMPAT_ANSWER_TIMEOUT_S", "-1")
    with pytest.raises(RuntimeError, match="must be >= 0"):
        rpb._openai_compatible_answer_timeout_s()


def test_recall_subprocess_timeout_defaults_strict_without_relax_flag(monkeypatch):
    monkeypatch.delenv("BENCHMARK_RELAX_TIMEOUTS", raising=False)
    monkeypatch.setattr(rpb, "_BACKEND", "llama-cpp")
    monkeypatch.setattr(rpb, "_OPENAI_COMPAT_URL", "http://127.0.0.1:30001")
    monkeypatch.setenv("BENCHMARK_RECALL_FAST_TIMEOUT_S", "120")
    monkeypatch.setenv("BENCHMARK_RECALL_TIMEOUT_S", "180")

    assert rpb._recall_subprocess_timeout_seconds(fast=True) == 30
    assert rpb._recall_subprocess_timeout_seconds(fast=False) == 90


def test_require_relax_timeouts_for_local_provider(monkeypatch):
    monkeypatch.setattr(rpb, "_BACKEND", "llama-cpp")
    monkeypatch.setattr(rpb, "_OPENAI_COMPAT_URL", "http://127.0.0.1:30001")

    with pytest.raises(SystemExit, match="--relax-timeouts"):
        rpb._require_relax_timeouts_for_local_provider(relax_requested=False)

    rpb._require_relax_timeouts_for_local_provider(relax_requested=True)


def test_call_openai_compatible_chat_writes_trace_events(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setenv("OPENAI_COMPAT_TRACE", "1")
    monkeypatch.setenv("BENCHMARK_LLAMA_CPP_URL", "http://example.test")
    monkeypatch.setenv("BENCHMARK_LLAMA_CPP_MODEL", "gemma-4-31b-q8")
    monkeypatch.setenv("BENCHMARK_LLAMA_CPP_JUDGE_URL", "http://example.test")
    monkeypatch.setenv("BENCHMARK_LLAMA_CPP_JUDGE_MODEL", "gemma-4-31b-q8")
    monkeypatch.setattr(rpb, "_BACKEND", "llama-cpp")
    monkeypatch.setattr(rpb, "_get_openai_compatible_api_key", lambda source=None: "")

    def _fake_urlopen(_req, timeout=300):
        payload = {
            "model": "gemma-4-31b-q8",
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            "choices": [{"message": {"content": "ok"}}],
        }
        return io.BytesIO(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(rpb.urllib.request, "urlopen", _fake_urlopen)

    data, usage = rpb._call_openai_compatible_chat(
        messages=[{"role": "user", "content": "hello world"}],
        model="gemma-4-31b-q8",
        max_tokens=32,
        timeout=120,
        workspace=workspace,
        source="judge",
        provider="llama-cpp",
    )

    assert data["choices"][0]["message"]["content"] == "ok"
    assert usage["output_tokens"] == 7
    trace_path = workspace / "logs" / "openai-compatible-trace.jsonl"
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in rows] == ["start", "success"]
    assert rows[0]["source"] == "judge"
    assert rows[0]["message_tokens_est"] >= 1
    assert rows[1]["duration_ms"] >= 0


def test_openai_message_text_strips_gemma_control_markers():
    message = {
        "content": (
            "<|channel>thought<tool_call|><|tool_response><|channel>thought\n"
            "<channel|>The recipe app became a family-care tool."
        )
    }

    assert (
        rpb._openai_message_text(message, model="gemma-4-26b-q6k")
        == "The recipe app became a family-care tool."
    )


def test_openai_message_text_does_not_strip_non_gemma_content():
    text = "<|channel>thought\n<channel|>keep literal transport text"
    message = {"content": text}

    assert rpb._openai_message_text(message, model="gpt-4o-mini") == text


def test_openai_message_text_strips_gemma_tool_call_markup():
    message = {
        "content": (
            '<|channel>thought\n<channel|><|tool_call>call:recall{query:<|"|>What is Linda\'s job or profession?<|"|>}'
            '<tool_call|>'
        )
    }

    assert rpb._openai_message_text(message, model="gemma-4-26b-q6k") == ""


def test_call_openai_compatible_chat_clears_active_request_on_error(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setenv("OPENAI_COMPAT_TRACE", "1")
    monkeypatch.setenv("BENCHMARK_LLAMA_CPP_URL", "http://example.test")
    monkeypatch.setenv("BENCHMARK_LLAMA_CPP_MODEL", "gemma-4-31b-q8")
    monkeypatch.setattr(rpb, "_BACKEND", "llama-cpp")
    monkeypatch.setattr(rpb, "_get_openai_compatible_api_key", lambda source=None: "")

    def _fake_urlopen(_req, timeout=300):
        raise TimeoutError("timed out")

    monkeypatch.setattr(rpb.urllib.request, "urlopen", _fake_urlopen)

    with pytest.raises(RuntimeError, match="OpenAI-compatible timeout"):
        rpb._call_openai_compatible_chat(
            messages=[{"role": "user", "content": "hello world"}],
            model="gemma-4-31b-q8",
            max_tokens=32,
            timeout=120,
            workspace=workspace,
            source="answer",
            provider="llama-cpp",
        )

    assert rpb._OPENAI_COMPAT_ACTIVE_REQUESTS == {}
    trace_path = workspace / "logs" / "openai-compatible-trace.jsonl"
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["event"] == "start"
    assert rows[1]["event"] == "error"
    assert "timeout" in rows[1]["error"].lower()


def test_uses_openai_compatible_backend_includes_codex(monkeypatch):
    monkeypatch.setattr(rpb, "_BACKEND", "codex")
    assert rpb._uses_openai_compatible_backend() is True


def test_uses_openai_compatible_backend_includes_direct_openai(monkeypatch):
    monkeypatch.setattr(rpb, "_BACKEND", "openai")
    monkeypatch.setattr(rpb, "_OPENAI_COMPAT_URL", "")
    monkeypatch.setattr(rpb, "_OPENAI_COMPAT_MODEL", "")
    monkeypatch.setattr(rpb, "_OPENAI_COMPAT_API_KEY_ENV", "")
    monkeypatch.delenv("BENCHMARK_OPENAI_URL", raising=False)
    monkeypatch.delenv("BENCHMARK_OPENAI_MODEL", raising=False)

    assert rpb._uses_openai_compatible_backend() is True
    assert rpb._get_openai_compatible_url() == "https://api.openai.com"
    assert rpb._get_openai_compatible_model() == "gpt-5.4"
    assert rpb._get_openai_compatible_api_key_env() == "OPENAI_API_KEY"


def test_codex_backend_defaults_to_chatgpt_backend_api(monkeypatch):
    monkeypatch.setattr(rpb, "_BACKEND", "codex")
    monkeypatch.setattr(rpb, "_OPENAI_COMPAT_URL", "")
    monkeypatch.setattr(rpb, "_OPENAI_COMPAT_MODEL", "")
    monkeypatch.setattr(rpb, "_OPENAI_COMPAT_API_KEY_ENV", "")
    monkeypatch.delenv("BENCHMARK_CODEX_BASE_URL", raising=False)

    assert rpb._get_openai_compatible_url() == "https://chatgpt.com/backend-api"
    assert rpb._get_openai_compatible_model() == "gpt-5.4"
    assert rpb._get_openai_compatible_api_key_env() == "BENCHMARK_CODEX_API_KEY"


def test_codex_backend_reads_auth_token_file(monkeypatch, tmp_path):
    workspace = tmp_path / "ws"
    token_path = workspace / "adaptors" / "codex" / ".auth-token"
    token_path.parent.mkdir(parents=True)
    token_path.write_text("tok-codex", encoding="utf-8")
    monkeypatch.setattr(rpb, "_BACKEND", "codex")
    monkeypatch.delenv("BENCHMARK_CODEX_API_KEY", raising=False)

    assert rpb._read_codex_auth_token(workspace) == "tok-codex"
    monkeypatch.setenv("QUAID_HOME", str(workspace))
    assert rpb._get_openai_compatible_api_key() == "tok-codex"


def test_call_openai_compatible_chat_uses_codex_oauth_sse_contract(monkeypatch, tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setattr(rpb, "_BACKEND", "codex")
    monkeypatch.setattr(rpb, "_OPENAI_COMPAT_URL", "https://chatgpt.com/backend-api")
    monkeypatch.setattr(rpb, "_OPENAI_COMPAT_MODEL", "gpt-5.4")
    monkeypatch.setattr(rpb, "_OPENAI_COMPAT_API_KEY_ENV", "BENCHMARK_CODEX_API_KEY")
    monkeypatch.setenv(
        "BENCHMARK_CODEX_API_KEY",
        "aaa.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL2F1dGgiOnsiY2hhdGdwdF9hY2NvdW50X2lkIjoiYWNjdC0xMjMifX0.bbb",
    )

    captured = {}

    class _Resp:
        def __init__(self):
            self._lines = iter(
                [
                    b"event: response.output_item.added\n",
                    b"data: {\"type\":\"response.output_item.added\",\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"status\":\"in_progress\",\"arguments\":\"\",\"call_id\":\"call_1\",\"name\":\"recall\"}}\n",
                    b"\n",
                    b"event: response.function_call_arguments.delta\n",
                    b"data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_1\",\"delta\":\"{\\\"query\\\":\\\"Baxter\\\"}\"}\n",
                    b"\n",
                    b"event: response.output_item.done\n",
                    b"data: {\"type\":\"response.output_item.done\",\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"status\":\"completed\",\"arguments\":\"{\\\"query\\\":\\\"Baxter\\\"}\",\"call_id\":\"call_1\",\"name\":\"recall\"}}\n",
                    b"\n",
                    b"event: response.completed\n",
                    b"data: {\"type\":\"response.completed\",\"response\":{\"model\":\"gpt-5.4-mini-2026-03-17\",\"status\":\"completed\",\"usage\":{\"input_tokens\":10,\"output_tokens\":3,\"total_tokens\":13,\"input_tokens_details\":{\"cached_tokens\":2}}}}\n",
                    b"\n",
                    b"",
                ]
            )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def readline(self):
            return next(self._lines)

    def _fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["auth"] = req.headers.get("Authorization")
        captured["account"] = req.headers.get("Chatgpt-account-id")
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr(rpb.urllib.request, "urlopen", _fake_urlopen)

    data, usage = rpb._call_openai_compatible_chat(
        messages=[{"role": "user", "content": "hello"}],
        model="gpt-5.4",
        max_tokens=32,
        timeout=120,
        workspace=workspace,
        source="answer_model",
        provider="codex",
    )

    assert data["choices"][0]["message"]["content"] == ""
    assert data["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "recall"
    assert usage["output_tokens"] == 3
    assert usage["cache_read_tokens"] == 2
    assert captured["url"] == "https://chatgpt.com/backend-api/codex/responses"
    assert captured["auth"].startswith("Bearer aaa.")
    assert captured["account"] == "acct-123"
    assert captured["body"]["model"] == "gpt-5.4"
    assert captured["body"]["store"] is False
    assert captured["body"]["stream"] is True
    assert captured["body"]["input"][0]["role"] == "user"
    assert captured["body"]["prompt_cache_key"]
    assert "include" not in captured["body"]


def test_call_openai_compatible_chat_codex_omits_reasoning_when_effort_unset(monkeypatch, tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setattr(rpb, "_BACKEND", "codex")
    monkeypatch.setattr(rpb, "_OPENAI_COMPAT_URL", "https://chatgpt.com/backend-api")
    monkeypatch.setattr(rpb, "_OPENAI_COMPAT_MODEL", "gpt-5.4-mini")
    monkeypatch.setattr(rpb, "_OPENAI_COMPAT_API_KEY_ENV", "BENCHMARK_CODEX_API_KEY")
    monkeypatch.setattr(rpb, "_CODEX_DEEP_EFFORT", "")
    monkeypatch.setattr(rpb, "_CODEX_FAST_EFFORT", "")
    monkeypatch.setenv(
        "BENCHMARK_CODEX_API_KEY",
        "aaa.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL2F1dGgiOnsiY2hhdGdwdF9hY2NvdW50X2lkIjoiYWNjdC0xMjMifX0.bbb",
    )

    captured = {}

    class _Resp:
        def __init__(self):
            self._lines = iter(
                [
                    b"event: response.output_text.delta\n",
                    b"data: {\"type\":\"response.output_text.delta\",\"delta\":\"hello\"}\n",
                    b"\n",
                    b"event: response.completed\n",
                    b"data: {\"type\":\"response.completed\",\"response\":{\"model\":\"gpt-5.4-mini-2026-03-17\",\"status\":\"completed\",\"usage\":{\"input_tokens\":10,\"output_tokens\":3,\"total_tokens\":13}}}\n",
                    b"\n",
                    b"",
                ]
            )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def readline(self):
            return next(self._lines)

    def _fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _Resp()

    monkeypatch.setattr(rpb.urllib.request, "urlopen", _fake_urlopen)

    data, usage = rpb._call_openai_compatible_chat(
        messages=[{"role": "user", "content": "hello"}],
        model="gpt-5.4-mini",
        max_tokens=32,
        timeout=120,
        workspace=workspace,
        source="extraction",
        provider="codex",
    )

    assert data["choices"][0]["message"]["content"] == "hello"
    assert usage["output_tokens"] == 3
    assert "reasoning" not in captured["body"]


def test_call_openai_compatible_chat_codex_uses_configured_reasoning_efforts(monkeypatch, tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setattr(rpb, "_BACKEND", "codex")
    monkeypatch.setattr(rpb, "_OPENAI_COMPAT_URL", "https://chatgpt.com/backend-api")
    monkeypatch.setattr(rpb, "_OPENAI_COMPAT_MODEL", "gpt-5.4-mini")
    monkeypatch.setattr(rpb, "_OPENAI_COMPAT_API_KEY_ENV", "BENCHMARK_CODEX_API_KEY")
    monkeypatch.setattr(rpb, "_OPENAI_COMPAT_JUDGE_URL", "https://chatgpt.com/backend-api")
    monkeypatch.setattr(rpb, "_OPENAI_COMPAT_JUDGE_MODEL", "gpt-5.4-mini")
    monkeypatch.setattr(rpb, "_OPENAI_COMPAT_JUDGE_API_KEY_ENV", "BENCHMARK_CODEX_JUDGE_API_KEY")
    monkeypatch.setattr(rpb, "_CODEX_DEEP_EFFORT", "low")
    monkeypatch.setattr(rpb, "_CODEX_FAST_EFFORT", "none")
    monkeypatch.setenv(
        "BENCHMARK_CODEX_API_KEY",
        "aaa.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL2F1dGgiOnsiY2hhdGdwdF9hY2NvdW50X2lkIjoiYWNjdC0xMjMifX0.bbb",
    )
    monkeypatch.setenv(
        "BENCHMARK_CODEX_JUDGE_API_KEY",
        "aaa.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL2F1dGgiOnsiY2hhdGdwdF9hY2NvdW50X2lkIjoiYWNjdC0xMjMifX0.bbb",
    )

    captured_bodies = []

    class _Resp:
        def __init__(self):
            self._lines = iter(
                [
                    b"event: response.output_text.delta\n",
                    b"data: {\"type\":\"response.output_text.delta\",\"delta\":\"hello\"}\n",
                    b"\n",
                    b"event: response.completed\n",
                    b"data: {\"type\":\"response.completed\",\"response\":{\"model\":\"gpt-5.4-mini-2026-03-17\",\"status\":\"completed\",\"usage\":{\"input_tokens\":10,\"output_tokens\":3,\"total_tokens\":13}}}\n",
                    b"\n",
                    b"",
                ]
            )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def readline(self):
            return next(self._lines)

    def _fake_urlopen(req, timeout=None):
        captured_bodies.append(json.loads(req.data.decode("utf-8")))
        return _Resp()

    monkeypatch.setattr(rpb.urllib.request, "urlopen", _fake_urlopen)

    rpb._call_openai_compatible_chat(
        messages=[{"role": "user", "content": "hello"}],
        model="gpt-5.4-mini",
        max_tokens=32,
        timeout=120,
        workspace=workspace,
        source="extraction",
        provider="codex",
    )
    rpb._call_openai_compatible_chat(
        messages=[{"role": "user", "content": "hello"}],
        model="gpt-5.4-mini",
        max_tokens=32,
        timeout=120,
        workspace=workspace,
        source="judge",
        provider="codex",
    )

    assert captured_bodies[0]["reasoning"]["effort"] == "low"
    assert captured_bodies[1]["reasoning"]["effort"] == "none"


def test_write_extraction_output_trace_persists_raw_output(tmp_path):
    rpb._write_extraction_output_trace(
        tmp_path,
        "per day chunk 000",
        "gpt-5.4",
        "{\"facts\":[]}",
    )
    trace_path = tmp_path / "logs" / "extraction-output-trace.jsonl"
    assert trace_path.exists()
    lines = trace_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["event"] == "extraction_output"
    assert payload["scope"] == "per day chunk 000"
    assert payload["model"] == "gpt-5.4"
    assert payload["chars"] == len("{\"facts\":[]}")
    output_file = Path(payload["output_file"])
    assert output_file.exists()
    assert output_file.read_text(encoding="utf-8") == "{\"facts\":[]}"


def test_resolve_judge_provider_uses_openai_compatible_for_codex_backend(monkeypatch):
    monkeypatch.setattr(rpb, "_BACKEND", "codex")
    monkeypatch.delenv("BENCHMARK_JUDGE_PROVIDER", raising=False)
    assert rpb._resolve_judge_provider("gpt-5.4-mini") == "openai-compatible"


def test_judge_openai_compatible_extracts_text_from_response(monkeypatch):
    def _fake_call(**kwargs):
        return (
            {
                "choices": [
                    {
                        "message": {
                            "content": [{"type": "text", "text": "CORRECT"}],
                        }
                    }
                ]
            },
            {"input_tokens": 5, "output_tokens": 1, "api_calls": 1, "model_usage": {}},
        )

    monkeypatch.setattr(rpb, "_call_openai_compatible_chat", _fake_call)
    label, score = rpb._judge_openai_compatible("prompt", model="gemma-4-31b-q8", workspace=None)
    assert label == "CORRECT"
    assert score == 1.0


def test_judge_openai_compatible_falls_back_to_reasoning_content(monkeypatch):
    def _fake_call(**kwargs):
        return (
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "reasoning_content": '{"label":"CORRECT"}',
                        }
                    }
                ],
                "model": "gemma-4-26b-q6k",
            },
            {"input_tokens": 5, "output_tokens": 1, "api_calls": 1, "model_usage": {}},
        )

    monkeypatch.setattr(rpb, "_call_openai_compatible_chat", _fake_call)
    label, score = rpb._judge_openai_compatible("prompt", model="gemma-4-26b-q6k", workspace=None)
    assert label == "CORRECT"
    assert score == 1.0


def test_judge_openai_retries_transient_timeout(monkeypatch):
    calls = {"count": 0}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({
                "choices": [{"message": {"content": "CORRECT"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            }).encode()

    def _fake_urlopen(_req, timeout=30):
        calls["count"] += 1
        if calls["count"] == 1:
            raise TimeoutError("The read operation timed out")
        return _Response()

    monkeypatch.setenv("BENCHMARK_OPENAI_JUDGE_RETRIES", "2")
    monkeypatch.setattr(rpb, "_get_openai_key", lambda: "key")
    monkeypatch.setattr(rpb.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(rpb.urllib.request, "urlopen", _fake_urlopen)

    label, score = rpb._judge_openai("prompt", workspace=None)

    assert (label, score) == ("CORRECT", 1.0)
    assert calls["count"] == 2


def test_judge_openai_exhausted_retries_fail_hard(monkeypatch):
    calls = {"count": 0}

    def _fake_urlopen(_req, timeout=30):
        calls["count"] += 1
        raise TimeoutError("The read operation timed out")

    monkeypatch.setenv("BENCHMARK_OPENAI_JUDGE_RETRIES", "2")
    monkeypatch.setattr(rpb, "_get_openai_key", lambda: "key")
    monkeypatch.setattr(rpb.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(rpb.urllib.request, "urlopen", _fake_urlopen)

    with pytest.raises(RuntimeError, match="OpenAI judge failed after 2 attempt"):
        rpb._judge_openai("prompt", workspace=None)

    assert calls["count"] == 2


class TestGroupSessionsByDate:
    """Tests for _group_sessions_by_date: session grouping."""

    def test_groups_by_4am_operational_day(self, monkeypatch):
        monkeypatch.setattr(rpb, "SESSION_DATES", {})
        reviews = [
            _FakeReview(1, timestamp="2026-03-02 03:00:00 UTC"),
            _FakeReview(2, timestamp="2026-03-02 12:00:00 UTC"),
            _FakeReview(3, timestamp="2026-03-03 02:30:00 UTC"),
        ]
        days = rpb._group_sessions_by_date(reviews)
        assert len(days) == 2
        assert days[0][0] == "2026-03-01"
        assert [r.session_num for r in days[0][1]] == [1]
        assert days[1][0] == "2026-03-02"
        assert [r.session_num for r in days[1][1]] == [2, 3]

    def test_empty_reviews(self, monkeypatch):
        monkeypatch.setattr(rpb, "SESSION_DATES", {})
        assert rpb._group_sessions_by_date([]) == []

    def test_unknown_session_date(self, monkeypatch):
        monkeypatch.setattr(rpb, "SESSION_DATES", {})
        reviews = [_FakeReview(99)]
        days = rpb._group_sessions_by_date(reviews)
        assert days[0][0] == "1970-01-01"


class TestDomainBlockMarkdown:
    """Tests for _domain_block_markdown: TOOLS.md rendering."""

    def test_renders_with_descriptions(self):
        domains = [("finance", "money stuff"), ("health", "wellness")]
        md = rpb._domain_block_markdown(domains)
        assert "<!-- AUTO-GENERATED:DOMAIN-LIST:START -->" in md
        assert "<!-- AUTO-GENERATED:DOMAIN-LIST:END -->" in md
        assert "- `finance`: money stuff" in md
        assert "- `health`: wellness" in md

    def test_renders_without_description(self):
        domains = [("custom", "")]
        md = rpb._domain_block_markdown(domains)
        assert "- `custom`" in md
        assert ": " not in md.split("- `custom`")[1].split("\n")[0]

    def test_empty_domains(self):
        md = rpb._domain_block_markdown([])
        assert "START" in md
        assert "END" in md


class TestInjectDomainsIntoToolsMd:
    """Tests for _inject_domains_into_tools_md: block replacement + append."""

    def test_replaces_existing_block(self):
        existing = (
            "# Tools\nSome text.\n"
            "<!-- AUTO-GENERATED:DOMAIN-LIST:START -->\nold content\n<!-- AUTO-GENERATED:DOMAIN-LIST:END -->\n"
            "More text."
        )
        result = rpb._inject_domains_into_tools_md(existing, [("project", "tasks")])
        assert "old content" not in result
        assert "- `project`: tasks" in result
        assert result.count("AUTO-GENERATED:DOMAIN-LIST:START") == 1

    def test_appends_when_no_existing_block(self):
        existing = "# Tools\n\nSome content."
        result = rpb._inject_domains_into_tools_md(existing, [("finance", "budgets")])
        assert "## Domains" in result
        assert "- `finance`: budgets" in result
        assert result.startswith("# Tools")


class TestParseJudgeLabel:
    """Tests for _parse_judge_label: JSON + text parsing."""

    def test_json_correct(self):
        label, score = rpb._parse_judge_label('{"label": "CORRECT"}')
        assert label == "CORRECT"
        assert score == 1.0

    def test_json_wrong(self):
        label, score = rpb._parse_judge_label('{"label": "WRONG"}')
        assert label == "WRONG"
        assert score == 0.0

    def test_json_partial(self):
        label, score = rpb._parse_judge_label('{"label": "PARTIAL"}')
        assert label == "PARTIAL"
        assert score == 0.5

    def test_text_correct(self):
        label, score = rpb._parse_judge_label("The answer is CORRECT because ...")
        assert label == "CORRECT"
        assert score == 1.0

    def test_text_partial(self):
        label, score = rpb._parse_judge_label("This is PARTIAL because it is safe but overlong")
        assert label == "PARTIAL"
        assert score == 0.5

    def test_text_wrong(self):
        label, score = rpb._parse_judge_label("This is WRONG")
        assert label == "WRONG"
        assert score == 0.0

    def test_both_labels_last_wins(self):
        # If text mentions both, last position wins
        label, _ = rpb._parse_judge_label("Initially WRONG, but actually CORRECT")
        assert label == "CORRECT"

        label, score = rpb._parse_judge_label("Seems WRONG at first, but on reflection PARTIAL")
        assert label == "PARTIAL"
        assert score == 0.5

        label, _ = rpb._parse_judge_label("Seems CORRECT but on reflection WRONG")
        assert label == "WRONG"

    def test_no_label_returns_error(self):
        label, score = rpb._parse_judge_label("I have no verdict")
        assert label == "ERROR"
        assert score == 0.0

    def test_empty_string(self):
        label, score = rpb._parse_judge_label("")
        assert label == "ERROR"
        assert score == 0.0


class TestClaudeStreamToolParsing:
    """Tests for Claude stream-json parsing in eval tool mode."""

    def test_classifies_bash_search_commands(self):
        label, query = rpb._classify_claude_bash_command(
            'python3 memory_graph.py recall "maya partner name" --owner maya --limit 5'
        )
        assert label == "memory_recall"
        assert query == "maya partner name"

        label, query = rpb._classify_claude_bash_command(
            'python3 memory_graph.py recall-fast "where does maya live" --owner maya --limit 5'
        )
        assert label == "memory_recall"
        assert query == "where does maya live"

        label, query = rpb._classify_claude_bash_command(
            'python3 memory_graph.py search-all "recipe app test suites"'
        )
        assert label == "search_project_docs"
        assert query == "recipe app test suites"

    def test_parses_stream_events_with_tool_use_and_results(self):
        stream = "\n".join(
            [
                json.dumps({
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "Bash",
                                "input": {
                                    "command": 'python3 memory_graph.py recall "maya partner" --owner maya --limit 5'
                                },
                            },
                            {
                                "type": "tool_use",
                                "id": "toolu_2",
                                "name": "Bash",
                                "input": {
                                    "command": 'python3 memory_graph.py search-all "recipe app auth tests"'
                                },
                            },
                        ]
                    },
                }),
                json.dumps({
                    "type": "user",
                    "message": {
                        "content": [{"type": "tool_result", "tool_use_id": "toolu_1"}]
                    },
                    "tool_use_result": {"stdout": "memory line one\nmemory line two"},
                }),
                json.dumps({
                    "type": "user",
                    "message": {
                        "content": [{"type": "tool_result", "tool_use_id": "toolu_2"}]
                    },
                    "tool_use_result": {"stdout": "docs snippet"},
                }),
                json.dumps({
                    "type": "result",
                    "is_error": False,
                    "num_turns": 3,
                    "result": "final answer",
                    "modelUsage": {
                        "claude-sonnet-4-6": {
                            "inputTokens": 10,
                            "outputTokens": 20,
                            "cacheReadInputTokens": 30,
                            "cacheCreationInputTokens": 0,
                        }
                    },
                }),
            ]
        )

        answer, tool_calls, summaries, retrieval_texts, final_data = rpb._parse_claude_stream_output(stream)
        assert answer == "final answer"
        assert tool_calls == ["memory_recall", "search_project_docs"]
        assert len(summaries) == 2
        assert "memory_recall(" in summaries[0]
        assert "search_project_docs(" in summaries[1]
        assert retrieval_texts == ["memory line one\nmemory line two"]
        assert final_data.get("num_turns") == 3


def test_pre_recall_uses_fast_memory_recall_path(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured = {}

    def _fake_tool_memory_recall(query, _workspace, _env, **kwargs):
        captured["query"] = query
        captured["fast"] = kwargs.get("fast")
        captured["max_session"] = kwargs.get("max_session")
        captured["planner_profile"] = kwargs.get("planner_profile")
        return "hit", {"mode": "fast", "stop_reason": "quality_gate_met"}

    monkeypatch.setattr(rpb, "_tool_memory_recall", _fake_tool_memory_recall)

    recall_text, query_used, recall_meta = rpb._pre_recall(
        "Where does Maya live?",
        workspace,
        {},
        max_session=7,
    )

    assert recall_text == "hit"
    assert query_used == "Where does Maya live?"
    assert recall_meta == {"mode": "fast", "stop_reason": "quality_gate_met"}
    assert captured == {
        "query": "Where does Maya live?",
        "fast": True,
        "max_session": 7,
        "planner_profile": "fast",
    }


def test_pre_recall_is_not_disabled_for_codex_backend(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setattr(rpb, "_BACKEND", "codex")

    captured = {}

    def _fake_tool_memory_recall(query, _workspace, _env, **kwargs):
        captured["query"] = query
        captured["fast"] = kwargs.get("fast")
        return "hit", {"stop_reason": "quality_gate_met"}

    monkeypatch.setattr(rpb, "_tool_memory_recall", _fake_tool_memory_recall)

    recall_text, query_used, recall_meta = rpb._pre_recall("Who is Maya's partner?", workspace, {})

    assert recall_text == "hit"
    assert query_used == "Who is Maya's partner?"
    assert recall_meta == {"stop_reason": "quality_gate_met"}
    assert captured == {"query": "Who is Maya's partner?", "fast": True}


class TestMakeEnv:
    """Tests for _make_env: environment variable wiring."""

    def test_sets_core_env_vars(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        monkeypatch.setattr(rpb, "_CLAWD", tmp_path)
        monkeypatch.setattr(rpb, "_BACKEND", "oauth")
        (tmp_path / "plugins" / "quaid").mkdir(parents=True)
        (workspace / "config").mkdir(parents=True, exist_ok=True)
        (workspace / "config" / "memory.json").write_text('{"adapter":{"type":"standalone"}}', encoding="utf-8")

        env = rpb._make_env(workspace)
        assert env["CLAWDBOT_WORKSPACE"] == str(workspace.resolve())
        assert env["QUAID_HOME"] == str(workspace.resolve())
        assert env["QUAID_INSTANCE"] == "benchrunner"
        assert env["MEMORY_DB_PATH"] == str(workspace.resolve() / "data" / "memory.db")
        assert env["QUAID_DISABLE_NOTIFICATIONS"] == "1"
        assert env["QUAID_LLM_USAGE_LOG_PATH"] == str(
            workspace.resolve() / "benchrunner" / "logs" / "llm-usage-events.jsonl"
        )
        assert (workspace / "benchrunner" / "config" / "memory.json").exists()
        assert (workspace / "benchrunner" / "projects").is_symlink()
        assert (workspace / "benchrunner" / "projects").resolve() == (workspace / "projects").resolve()

    def test_pythonpath_set(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        quaid_dir = tmp_path / "modules" / "quaid"
        quaid_dir.mkdir(parents=True)
        monkeypatch.setattr(rpb, "_QUAID_DIR", quaid_dir)
        monkeypatch.setattr(rpb, "_BACKEND", "oauth")

        env = rpb._make_env(workspace)
        assert str(quaid_dir.resolve()) in env.get("PYTHONPATH", "")

    def test_does_not_default_mock_embeddings_for_benchmark_subprocesses(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        monkeypatch.setattr(rpb, "_BACKEND", "oauth")
        monkeypatch.delenv("MOCK_EMBEDDINGS", raising=False)

        env = rpb._make_env(workspace)
        assert "MOCK_EMBEDDINGS" not in env

    def test_preserves_explicit_mock_embeddings_override(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        monkeypatch.setattr(rpb, "_BACKEND", "oauth")
        monkeypatch.setenv("MOCK_EMBEDDINGS", "0")

        env = rpb._make_env(workspace)
        assert env["MOCK_EMBEDDINGS"] == "0"

    def test_can_force_mock_embeddings_for_targeted_subprocesses(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        monkeypatch.setattr(rpb, "_BACKEND", "oauth")
        monkeypatch.delenv("MOCK_EMBEDDINGS", raising=False)

        env = rpb._make_env(workspace, mock_embeddings=True)
        assert env["MOCK_EMBEDDINGS"] == "1"

    def test_propagates_eval_parallel_override(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        monkeypatch.setattr(rpb, "_BACKEND", "oauth")
        monkeypatch.setenv("BENCHMARK_PARALLEL", "6")
        monkeypatch.setenv("BENCHMARK_EVAL_PARALLEL", "1")

        env = rpb._make_env(workspace)
        assert env["BENCHMARK_PARALLEL"] == "6"
        assert env["BENCHMARK_EVAL_PARALLEL"] == "1"

    def test_vllm_sets_openai_compatible_transport_and_clears_anthropic(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        monkeypatch.setattr(rpb, "_BACKEND", "vllm")
        monkeypatch.setattr(rpb, "_OPENAI_COMPAT_URL", "http://spark:8000")
        monkeypatch.setattr(rpb, "_OPENAI_COMPAT_API_KEY_ENV", "BENCHMARK_VLLM_API_KEY")
        monkeypatch.setenv("BENCHMARK_VLLM_API_KEY", "test-vllm-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("BENCHMARK_ANTHROPIC_OAUTH_TOKEN", "sk-ant-oat01-test-token")

        env = rpb._make_env(workspace)

        assert env["OPENAI_COMPATIBLE_BASE_URL"] == "http://spark:8000"
        assert env["BENCHMARK_VLLM_API_KEY"] == "test-vllm-key"
        assert "ANTHROPIC_API_KEY" not in env
        assert "BENCHMARK_ANTHROPIC_OAUTH_TOKEN" not in env

    def test_llama_cpp_sets_openai_compatible_transport_and_clears_anthropic(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        monkeypatch.setattr(rpb, "_BACKEND", "llama-cpp")
        monkeypatch.setattr(rpb, "_OPENAI_COMPAT_URL", "http://spark:8080")
        monkeypatch.setattr(rpb, "_OPENAI_COMPAT_API_KEY_ENV", "BENCHMARK_LLAMA_CPP_API_KEY")
        monkeypatch.setenv("BENCHMARK_LLAMA_CPP_API_KEY", "test-llama-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("BENCHMARK_ANTHROPIC_OAUTH_TOKEN", "sk-ant-oat01-test-token")

        env = rpb._make_env(workspace)

        assert env["OPENAI_COMPATIBLE_BASE_URL"] == "http://spark:8080"
        assert env["BENCHMARK_LLAMA_CPP_API_KEY"] == "test-llama-key"
        assert "ANTHROPIC_API_KEY" not in env
        assert "BENCHMARK_ANTHROPIC_OAUTH_TOKEN" not in env


def test_resolve_eval_parallel_workers_prefers_eval_override(monkeypatch):
    monkeypatch.setenv("BENCHMARK_PARALLEL", "6")
    monkeypatch.setenv("BENCHMARK_EVAL_PARALLEL", "2")
    assert rpb._resolve_eval_parallel_workers() == 2


def test_resolve_eval_parallel_workers_falls_back_to_global_parallel(monkeypatch):
    monkeypatch.setenv("BENCHMARK_PARALLEL", "4")
    monkeypatch.delenv("BENCHMARK_EVAL_PARALLEL", raising=False)
    assert rpb._resolve_eval_parallel_workers() == 4


def test_save_token_usage_includes_preinject_timing_stats(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "logs").mkdir(parents=True, exist_ok=True)

    rpb._append_usage_event(
        workspace,
        phase="eval",
        source="answer_model",
        model="claude-haiku-4-5-20251001",
        usage={
            "input_tokens": 30,
            "cache_read_input_tokens": 50,
            "cache_creation_input_tokens": 20,
            "output_tokens": 10,
            "api_calls": 1,
        },
        provider="api",
    )

    results = [
        {"eval_tokens": {"input_tokens": 10, "output_tokens": 2, "api_calls": 1, "preinject_duration_ms": 100, "query_duration_ms": 900}},
        {"eval_tokens": {"input_tokens": 20, "output_tokens": 3, "api_calls": 2, "preinject_duration_ms": 300,
                         "query_duration_ms": 1200,
                         "tool_call_details": [{"tool": "memory_recall(pre-inject)", "source": "preinject", "recall_meta": {
                             "phases_ms": {"planner_ms": 9, "fanout_wall_ms": 55, "total_ms": 70},
                             "turns": 1, "fanout_count": 3, "turn_details": [{
                                 "fanout": {
                                     "wall_ms": 55,
                                     "serial_sum_ms": 120,
                                     "parallel_speedup_x": 2.18,
                                     "parallel_efficiency_pct": 72.7,
                                     "overhead_vs_slowest_ms": 8,
                                     "branch_total_ms": {"spread_ms": 7},
                                     "fastest_branch": {"total_ms": 33},
                                     "slowest_branch": {"total_ms": 47},
                                     "branches": [
                                         {"phases_ms": {"hyde_ms": 11, "graph_traversal_ms": 22, "total_ms": 47}},
                                         {"phases_ms": {"hyde_ms": 7, "graph_traversal_ms": 0, "total_ms": 33}},
                                     ],
                                 }
                             }],
                             "planned_stores": ["vector", "docs"],
                             "stop_reason": "quality_gate_met", "bailout_counts": {"planner_returned_empty": 0}
                         }},
                         {"tool": "recall", "source": "tool", "recall_meta": {
                             "planned_stores": ["vector", "graph"],
                             "turn_details": [{"planner": {"planned_stores": ["vector", "graph"]}}],
                         }}]}},
        {"eval_tokens": {"input_tokens": 30, "output_tokens": 4, "api_calls": 3, "query_duration_ms": 1500}},
    ]

    store_stats = rpb._save_token_usage(results, workspace, "claude-haiku-4-5-20251001")

    data = json.loads((workspace / "token_usage.json").read_text())
    assert data["eval"]["input_tokens"] == 100
    assert data["eval"]["output_tokens"] == 10
    assert data["eval"]["total_tokens"] == 110
    assert data["eval"]["uncached_input_tokens"] == 30
    assert data["eval"]["cache_read_tokens"] == 50
    assert data["eval"]["cache_creation_tokens"] == 20
    assert data["eval"]["by_model"]["claude-haiku-4-5-20251001"]["uncached_input_tokens"] == 30
    assert data["eval"]["by_model"]["claude-haiku-4-5-20251001"]["cache_read_tokens"] == 50
    assert data["eval"]["by_model"]["claude-haiku-4-5-20251001"]["cache_creation_tokens"] == 20
    assert data["query_completion_ms"] == {
        "count": 3,
        "avg": 1200,
        "p50": 1200,
        "p95": 1200,
        "p99": 1200,
        "max": 1500,
    }
    assert data["preinject_timing_ms"] == {
        "count": 2,
        "avg": 200,
        "p50": 100,
        "p95": 100,
        "p99": 100,
        "max": 300,
    }
    assert data["preinject_recall_telemetry"]["count"] == 1
    assert data["preinject_recall_telemetry"]["phases_ms"]["planner_ms"]["avg"] == 9
    assert data["preinject_recall_telemetry"]["phases_ms"]["branch_hyde_ms"]["avg"] == 9
    assert data["preinject_recall_telemetry"]["phases_ms"]["branch_graph_traversal_ms"]["avg"] == 11
    assert data["preinject_recall_telemetry"]["fanout_count"]["avg"] == 3
    assert data["preinject_recall_telemetry"]["fanout_wall_ms"]["avg"] == 55
    assert data["preinject_recall_telemetry"]["fanout_serial_ms"]["avg"] == 120
    assert data["preinject_recall_telemetry"]["slowest_branch_ms"]["avg"] == 47
    assert data["preinject_recall_telemetry"]["fastest_branch_ms"]["avg"] == 33
    assert data["preinject_recall_telemetry"]["parallel_speedup_x"]["avg"] == 2.18
    assert data["preinject_recall_telemetry"]["parallel_efficiency_pct"]["avg"] == 72.7
    assert data["preinject_recall_telemetry"]["parallel_overhead_ms"]["avg"] == 8


def test_save_token_usage_includes_eval_runtime_throughput(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "logs").mkdir(parents=True, exist_ok=True)

    results = [
        {
            "_eval_run_summary": {"elapsed_seconds": 10.0, "parallel_workers": 4, "queries": 2},
            "eval_tokens": {"input_tokens": 100, "output_tokens": 20, "api_calls": 1, "query_duration_ms": 3000},
        },
        {
            "eval_tokens": {"input_tokens": 80, "output_tokens": 10, "api_calls": 1, "query_duration_ms": 2000},
        },
    ]

    rpb._save_token_usage(results, workspace, "gemma-3-31b-it")

    data = json.loads((workspace / "token_usage.json").read_text())
    runtime = data["eval_runtime"]
    assert runtime["elapsed_seconds"] == 10.0
    assert runtime["parallel_workers"] == 4
    assert runtime["queries"] == 2
    assert runtime["queries_per_second"] == 0.2
    assert runtime["input_tokens_per_second"] == 18.0
    assert runtime["output_tokens_per_second"] == 3.0
    assert runtime["total_tokens_per_second"] == 21.0
    assert runtime["summed_query_seconds"] == 5.0
    assert runtime["average_inflight_factor"] == 0.5


def test_summarize_usage_events_infers_tier_for_harness_logged_models(tmp_path):
    workspace = tmp_path / "ws"
    (workspace / "logs").mkdir(parents=True, exist_ok=True)

    rpb._append_usage_event(
        workspace,
        phase="eval",
        source="answer_model",
        model="claude-haiku-4-5-20251001",
        usage={
            "input_tokens": 60,
            "cache_read_input_tokens": 25,
            "cache_creation_input_tokens": 15,
            "output_tokens": 20,
            "api_calls": 1,
        },
        provider="api",
    )
    rpb._append_usage_event(
        workspace,
        phase="eval",
        source="judge",
        model="gpt-4o-mini",
        usage={"input_tokens": 40, "output_tokens": 10, "api_calls": 1},
        provider="openai",
    )
    rpb._append_usage_event(
        workspace,
        phase="ingest",
        source="extraction",
        model="claude-sonnet-4-6",
        usage={"input_tokens": 70, "output_tokens": 30, "api_calls": 1},
        provider="api",
    )

    eval_summary = rpb._summarize_usage_events(workspace, phase="eval")
    ingest_summary = rpb._summarize_usage_events(workspace, phase="ingest")

    assert eval_summary["total_tokens"] == 170
    assert eval_summary["by_tier"]["fast"]["total_tokens"] == 170
    assert eval_summary["uncached_input_tokens"] == 100
    assert eval_summary["cache_read_tokens"] == 25
    assert eval_summary["cache_creation_tokens"] == 15
    assert eval_summary["by_source"]["answer_model"]["uncached_input_tokens"] == 60
    assert eval_summary["by_source"]["judge"]["uncached_input_tokens"] == 40
    assert ingest_summary["total_tokens"] == 100
    assert ingest_summary["by_tier"]["deep"]["total_tokens"] == 100


def test_summarize_usage_events_ignores_pre_run_events_with_start_marker(tmp_path):
    workspace = tmp_path / "ws"
    usage_path = rpb._usage_log_path(workspace)
    usage_path.parent.mkdir(parents=True, exist_ok=True)

    # 2 eval events: one before run start marker and one after.
    usage_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ts": "2026-03-29T01:00:00+00:00",
                        "phase": "eval",
                        "source": "answer_model",
                        "tier": "fast",
                        "requested_model": "claude-haiku-4-5-20251001",
                        "resolved_model": "claude-haiku-4-5-20251001",
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "total_tokens": 120,
                        "cache_read_tokens": 0,
                        "cache_creation_tokens": 0,
                        "api_calls": 1,
                        "model_usage": {
                            "claude-haiku-4-5-20251001": {
                                "input_tokens": 100,
                                "output_tokens": 20,
                                "total_tokens": 120,
                            }
                        },
                    }
                ),
                json.dumps(
                    {
                        "ts": "2026-03-29T03:00:00+00:00",
                        "phase": "eval",
                        "source": "answer_model",
                        "tier": "fast",
                        "requested_model": "claude-haiku-4-5-20251001",
                        "resolved_model": "claude-haiku-4-5-20251001",
                        "input_tokens": 200,
                        "output_tokens": 40,
                        "total_tokens": 240,
                        "cache_read_tokens": 0,
                        "cache_creation_tokens": 0,
                        "api_calls": 1,
                        "model_usage": {
                            "claude-haiku-4-5-20251001": {
                                "input_tokens": 200,
                                "output_tokens": 40,
                                "total_tokens": 240,
                            }
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # Marker should exclude the first event.
    marker_path = rpb._usage_run_start_marker_path(workspace)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text("2026-03-29T02:00:00+00:00", encoding="utf-8")

    summary = rpb._summarize_usage_events(workspace, phase="eval")
    assert summary["total_tokens"] == 240
    assert summary["api_calls"] == 1
    assert summary["by_source"]["answer_model"]["total_tokens"] == 240


def test_prune_usage_events_before_run_start(tmp_path):
    workspace = tmp_path / "ws"
    usage_path = rpb._usage_log_path(workspace)
    usage_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path = rpb._usage_run_start_marker_path(workspace)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text("2026-03-29T02:00:00+00:00", encoding="utf-8")

    rows = [
        {
            "ts": "2026-03-29T01:00:00+00:00",
            "phase": "eval",
            "source": "answer_model",
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "api_calls": 1,
        },
        {
            "ts": "2026-03-29T03:00:00+00:00",
            "phase": "eval",
            "source": "answer_model",
            "input_tokens": 200,
            "output_tokens": 40,
            "total_tokens": 240,
            "api_calls": 1,
        },
    ]
    usage_path.write_text(
        "\n".join([json.dumps(rows[0]), "not-json", json.dumps(rows[1])]) + "\n",
        encoding="utf-8",
    )

    rpb._prune_usage_events_before_run_start(workspace)
    after = [json.loads(line) for line in usage_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(after) == 1
    assert after[0]["total_tokens"] == 240
    assert after[0]["ts"] == "2026-03-29T03:00:00+00:00"


def test_reset_usage_events_for_eval_clears_inherited_rows(tmp_path):
    workspace = tmp_path / "ws"
    usage_path = rpb._usage_log_path(workspace)
    usage_path.parent.mkdir(parents=True, exist_ok=True)
    usage_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ts": "2026-03-29T01:00:00+00:00",
                        "phase": "eval",
                        "source": "answer_model",
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "total_tokens": 120,
                        "api_calls": 1,
                    }
                ),
                json.dumps(
                    {
                        "ts": "2026-03-29T03:00:00+00:00",
                        "phase": "ingest",
                        "source": "extract",
                        "input_tokens": 200,
                        "output_tokens": 40,
                        "total_tokens": 240,
                        "api_calls": 1,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rpb._reset_usage_events_for_eval(workspace)
    assert usage_path.read_text(encoding="utf-8") == ""


def test_reset_eval_artifacts_removes_only_eval_outputs(tmp_path):
    workspace = tmp_path / "ws"
    (workspace / "logs").mkdir(parents=True, exist_ok=True)
    (workspace / "data").mkdir(parents=True, exist_ok=True)

    targets = [
        workspace / "evaluation_results.json",
        workspace / "scores.json",
        workspace / "token_usage.json",
        workspace / "tier5_results.json",
        workspace / "logs" / "eval_progress.json",
        workspace / "logs" / "eval_query_profile.json",
        workspace / "logs" / "eval_resume.json",
        workspace / "logs" / "llm-call-trace.jsonl",
        workspace / "logs" / "recall-telemetry.jsonl",
    ]
    for p in targets:
        p.write_text("x", encoding="utf-8")
    survivor = workspace / "data" / "memory.db"
    survivor.write_text("db", encoding="utf-8")

    rpb._reset_eval_artifacts(workspace)

    for p in targets:
        assert not p.exists()
    assert survivor.exists()


def test_load_reviews_with_dataset_gate_includes_fillers_for_al_l(monkeypatch, tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    filler_dir = tmp_path / "filler-sessions-L"
    filler_dir.mkdir()

    arc_reviews = [_FakeReview(1), _FakeReview(2)]
    filler_reviews = [_FakeReview(-1), _FakeReview(-2)]
    merged_reviews = [arc_reviews[0], filler_reviews[0], arc_reviews[1], filler_reviews[1]]

    monkeypatch.setattr(rpb, "_resolve_assets_dir", lambda: assets)
    monkeypatch.setattr(rpb, "_enforce_dataset_version", lambda _assets: ("v-test", 268))
    monkeypatch.setattr(rpb, "_read_dataset_version", lambda _assets: "v-test")
    monkeypatch.setattr(rpb, "load_all_reviews", lambda _assets, sessions=None: list(arc_reviews))
    monkeypatch.setattr(rpb, "load_filler_reviews", lambda _filler_dir: list(filler_reviews))
    monkeypatch.setattr(rpb, "merge_sessions_chronologically", lambda arcs, fillers: list(merged_reviews))
    monkeypatch.setenv("BENCHMARK_INCLUDE_FILLER", "1")
    monkeypatch.setenv("BENCHMARK_FILLER_DIR", str(filler_dir))

    assets_dir, arc, all_reviews, version, expected = rpb._load_reviews_with_dataset_gate(None)

    assert assets_dir == assets
    assert arc == arc_reviews
    assert all_reviews == merged_reviews
    assert version == "v-test"
    assert expected == 268


def test_load_reviews_with_dataset_gate_excludes_fillers_by_default(monkeypatch, tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    arc_reviews = [_FakeReview(1), _FakeReview(2)]

    monkeypatch.setattr(rpb, "_resolve_assets_dir", lambda: assets)
    monkeypatch.setattr(rpb, "_enforce_dataset_version", lambda _assets: ("v-test", 268))
    monkeypatch.setattr(rpb, "_read_dataset_version", lambda _assets: "v-test")
    monkeypatch.setattr(rpb, "load_all_reviews", lambda _assets, sessions=None: list(arc_reviews))
    monkeypatch.setattr(rpb, "load_filler_reviews", lambda _filler_dir: pytest.fail("load_filler_reviews should not run"))
    monkeypatch.setattr(rpb, "merge_sessions_chronologically", lambda arcs, fillers: pytest.fail("merge_sessions_chronologically should not run"))
    monkeypatch.delenv("BENCHMARK_INCLUDE_FILLER", raising=False)
    monkeypatch.delenv("BENCHMARK_FILLER_DIR", raising=False)

    assets_dir, arc, all_reviews, version, expected = rpb._load_reviews_with_dataset_gate(None)

    assert assets_dir == assets
    assert arc == arc_reviews
    assert all_reviews == arc_reviews
    assert version == "v-test"
    assert expected == 268


def test_analyze_tool_call_details_flags_quality_gate_followups():
    details = [
        {
            "tool": "memory_recall",
            "query": "Maya career TechFlow Stripe",
            "result_chars": 1700,
            "recall_meta": {"stop_reason": "quality_gate_met"},
        },
        {
            "tool": "memory_recall",
            "query": "Maya resignation timeline",
            "result_chars": 900,
            "recall_meta": {"stop_reason": "quality_gate_met"},
        },
        {
            "tool": "search_project_docs",
            "query": "stripe onboarding",
        },
    ]

    out = rpb._analyze_tool_call_details(details)

    assert out["memory_recall_count"] == 2
    assert out["repeated_memory_recall"] is True
    assert out["followup_after_quality_gate"] == 1
    assert out["repeated_memory_recall_classes"] == {"time_slice_split": 1}


def test_build_eval_context_sources_dedupes_duplicate_core_variants(tmp_path):
    workspace = tmp_path / "ws"
    (workspace / "projects" / "quaid").mkdir(parents=True)
    (workspace / "SOUL.md").write_text("same content")
    (workspace / "projects" / "quaid" / "SOUL.md").write_text("same content")
    (workspace / "TOOLS.md").write_text("tools root")

    sources = rpb._build_eval_context_sources(workspace, core_files=["SOUL.md", "TOOLS.md"], include_project_bootstrap=False)

    assert [s["path"] for s in sources] == ["SOUL.md", "TOOLS.md"]
    assert sources[0]["chars"] == len("same content")
    assert sources[0]["over_token_target"] is False


def test_build_eval_context_sources_report_token_target_status(tmp_path):
    workspace = tmp_path / "ws"
    (workspace / "projects" / "quaid").mkdir(parents=True)
    (workspace / "SOUL.md").write_text(" ".join(f"root{i}" for i in range(1200)))
    (workspace / "projects" / "quaid" / "SOUL.md").write_text(" ".join(f"proj{i}" for i in range(1200)))

    ctx = rpb._build_eval_context(workspace, core_files=["SOUL.md"], include_project_bootstrap=False)
    sources = rpb._build_eval_context_sources(workspace, core_files=["SOUL.md"], include_project_bootstrap=False)

    assert "--- SOUL.md ---" in ctx
    assert "--- projects/quaid/SOUL.md ---" in ctx
    assert len(sources) >= 1
    assert all("token_target" in s for s in sources)
    assert any(s["over_token_target"] for s in sources)


def test_tool_memory_recall_parses_results_and_meta_payload(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            stdout=json.dumps({
                "results": [{
                    "text": "Quaid likes espresso coffee",
                    "category": "fact",
                    "similarity": 0.91,
                    "id": "n1",
                    "privacy": "shared",
                    "owner_id": "maya",
                }],
                "meta": {
                    "mode": "deliberate",
                    "total_ms": 42,
                    "turn_details": [{
                        "planner": {
                            "timeout_ms": 60000,
                            "elapsed_ms": 1234,
                            "queries_count": 3,
                            "used_llm": True,
                            "bailout_reason": None,
                            "query_shape": "broad",
                            "planned_stores": ["vector"],
                        }
                    }],
                },
            }),
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)

    text, meta = rpb._tool_memory_recall("coffee", workspace, {"PATH": os.environ.get("PATH", "")})

    assert "recall" in captured["cmd"]
    assert "--json" in captured["cmd"]
    assert "--owner" not in captured["cmd"]
    cfg = json.loads(captured["cmd"][captured["cmd"].index("coffee") + 1])
    assert cfg["owner"] == "maya"
    assert cfg["limit"] == 10
    assert "Quaid likes espresso coffee" in text
    assert meta["mode"] == "deliberate"
    assert meta["total_ms"] == 42
    assert meta["harness_telemetry"]["status"] == "ok"


def test_tool_memory_recall_rehydrates_oauth_auth_for_subprocess(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured: dict[str, object] = {}

    def _fake_run(_cmd, **kwargs):
        captured["env"] = dict(kwargs.get("env") or {})
        return SimpleNamespace(
            stdout=json.dumps({
                "results": [{
                    "text": "Quaid likes espresso coffee",
                    "category": "fact",
                    "similarity": 0.91,
                    "id": "n1",
                    "privacy": "shared",
                    "owner_id": "maya",
                }],
                "meta": {"mode": "deliberate"},
            }),
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr(rpb, "_BACKEND", "oauth")
    monkeypatch.setattr(rpb, "_find_anthropic_credential", lambda: "sk-ant-oat01-test-token")

    rpb._tool_memory_recall(
        "coffee",
        workspace,
        {
            "PATH": os.environ.get("PATH", ""),
            "QUAID_HOME": str(workspace),
            "CLAWDBOT_WORKSPACE": str(workspace),
            "ANTHROPIC_API_KEY": "",
            "BENCHMARK_ANTHROPIC_OAUTH_TOKEN": "",
        },
    )

    assert captured["env"]["BENCHMARK_ANTHROPIC_OAUTH_TOKEN"] == "sk-ant-oat01-test-token"
    assert captured["env"]["ANTHROPIC_API_KEY"] == "sk-ant-oat01-test-token"


def test_tool_memory_recall_uses_longer_timeout_for_deliberate_calls(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    captured: dict[str, object] = {}

    def _fake_run(_cmd, **_kwargs):
        captured["timeout"] = _kwargs.get("timeout")
        return SimpleNamespace(
            stdout=json.dumps({
                "results": [{
                    "text": "Quaid likes espresso coffee",
                    "category": "fact",
                    "similarity": 0.9,
                    "id": "n1",
                    "privacy": "shared",
                    "owner_id": "maya",
                }],
                "meta": {
                    "mode": "deliberate",
                    "stop_reason": "max_turns",
                },
            }),
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)

    text, meta = rpb._tool_memory_recall(
        "coffee",
        workspace,
        {"PATH": os.environ.get("PATH", "")},
    )

    assert "Quaid likes espresso coffee" in text
    assert captured["timeout"] == 90
    assert meta["harness_telemetry"]["top_level_source"] == "tool"
    assert meta["harness_telemetry"]["top_level_call_id"]


def test_tool_memory_recall_passes_deliberate_timeout_budget_into_cfg(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["timeout"] = kwargs.get("timeout")
        return SimpleNamespace(
            stdout=json.dumps({"results": [], "meta": {"mode": "deliberate"}}),
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)

    _text, meta = rpb._tool_memory_recall(
        "coffee",
        workspace,
        {"PATH": os.environ.get("PATH", "")},
        stores=["vector", "graph"],
    )

    cfg_arg = next(part for part in captured["cmd"] if isinstance(part, str) and part.startswith("{"))
    cfg = json.loads(cfg_arg)
    assert cfg["stores"] == ["vector", "graph"]
    assert cfg["timeout_ms"] == 90000
    assert captured["timeout"] == 90
    assert meta["mode"] == "deliberate"


def test_tool_memory_recall_passes_planner_profile_for_fast_calls(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(stdout=json.dumps({"results": [], "meta": {"mode": "fast"}}), stderr="", returncode=0)

    monkeypatch.setattr(subprocess, "run", _fake_run)

    _text, meta = rpb._tool_memory_recall(
        "coffee",
        workspace,
        {"PATH": os.environ.get("PATH", "")},
        fast=True,
        planner_profile="aggressive",
    )

    assert captured["cmd"].count("--planner-profile") == 1
    assert "aggressive" in captured["cmd"]
    assert meta["mode"] == "fast"
    assert meta["harness_telemetry"]["status"] == "ok"


def test_tool_memory_recall_raises_on_nonzero_exit(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    full_stderr_tail = " full stderr sentinel " + ("x" * 1200) + " ROOT_CAUSE_AT_END"

    def _fake_run(_cmd, **_kwargs):
        return SimpleNamespace(
            stdout="",
            stderr=(
                "Error: Recall fanout planner failed while failHard is enabled: "
                "planner boom (planner_timeout_ms=60000, planner_elapsed_ms=1723, "
                "planner_profile=fast, query_shape=broad)"
                + full_stderr_tail
            ),
            returncode=2,
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)

    with pytest.raises(RuntimeError, match="recall failed rc=2"):
        rpb._tool_memory_recall("coffee", workspace, {"PATH": os.environ.get("PATH", "")})

    telemetry_path = workspace / "logs" / "recall-telemetry.jsonl"
    rows = [json.loads(line) for line in telemetry_path.read_text().splitlines() if line.strip()]
    assert rows[-1]["planner_timeout_ms"] == 60000
    assert rows[-1]["planner_elapsed_ms"] == 1723
    assert rows[-1]["planner_profile"] == "fast"
    assert rows[-1]["planner_query_shape"] == "broad"
    failure_artifact = Path(rows[-1]["failure_artifact"])
    assert failure_artifact.exists()
    artifact_payload = json.loads(failure_artifact.read_text())
    assert artifact_payload["returncode"] == 2
    assert "ROOT_CAUSE_AT_END" in artifact_payload["stderr"]


def test_tool_memory_recall_failure_artifact_sanitizes_path_and_credentials(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    token = "secret-token-value-123456789"

    def _fake_run(_cmd, **_kwargs):
        return SimpleNamespace(
            stdout='{"OPENAI_API_KEY":"sk-proj-stdout-secret"}',
            stderr=(
                f"ANTHROPIC_API_KEY={token}\n"
                f"raw token echo {token}\n"
                "Authorization: Bearer abc.def.ghi\n"
                "provider key sk-ant-stderr-secret"
            ),
            returncode=2,
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)

    with pytest.raises(RuntimeError, match="recall failed rc=2"):
        rpb._tool_memory_recall(
            "coffee",
            workspace,
            {"PATH": os.environ.get("PATH", ""), "ANTHROPIC_API_KEY": token},
            top_level_call_id="../escape/path",
            subprocess_role="../docs",
        )

    rows = [
        json.loads(line)
        for line in (workspace / "logs" / "recall-telemetry.jsonl").read_text().splitlines()
        if line.strip()
    ]
    failure_artifact = Path(rows[-1]["failure_artifact"])
    assert failure_artifact.parent == workspace / "logs" / "recall-failures"
    assert "/" not in failure_artifact.name
    artifact_payload = json.loads(failure_artifact.read_text())
    rendered = json.dumps({"telemetry": rows[-1], "artifact": artifact_payload})
    assert token not in rendered
    assert "sk-proj-stdout-secret" not in rendered
    assert "sk-ant-stderr-secret" not in rendered
    assert "Bearer abc.def.ghi" not in rendered
    assert "<redacted" in rendered


def test_tool_memory_recall_artifact_write_failure_still_raises(tmp_path, monkeypatch, capsys):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "logs").write_text("not a directory")

    def _fake_run(_cmd, **_kwargs):
        return SimpleNamespace(stdout="", stderr="recall boom", returncode=2)

    monkeypatch.setattr(subprocess, "run", _fake_run)

    with pytest.raises(RuntimeError, match="recall failed rc=2"):
        rpb._tool_memory_recall("coffee", workspace, {"PATH": os.environ.get("PATH", "")})

    captured = capsys.readouterr()
    assert "[recall-failure-artifact] write failed:" in captured.err


def test_tool_memory_recall_raises_on_timeout_and_writes_telemetry(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()

    def _fake_run(_cmd, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["python3", "-m", "datastore.memorydb.memory_graph"], timeout=30)

    monkeypatch.setattr(subprocess, "run", _fake_run)

    with pytest.raises(RuntimeError, match="recall timed out"):
        rpb._tool_memory_recall("coffee", workspace, {"PATH": os.environ.get("PATH", "")})

    telemetry_path = workspace / "logs" / "recall-telemetry.jsonl"
    assert telemetry_path.exists()
    rows = [json.loads(line) for line in telemetry_path.read_text().splitlines() if line.strip()]
    assert rows
    assert rows[-1]["status"] == "timeout"
    assert rows[-1]["query"] == "coffee"
    assert rows[-1]["top_level_source"] == "tool"
    assert rows[-1]["top_level_call_id"]
    assert rows[-1]["timeout_s"] == 90


def test_tool_memory_recall_uses_shorter_timeout_for_fast_calls(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured: dict[str, object] = {}

    def _fake_run(_cmd, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        return SimpleNamespace(stdout=json.dumps({"results": [], "meta": {"mode": "fast"}}), stderr="", returncode=0)

    monkeypatch.setattr(subprocess, "run", _fake_run)

    _text, meta = rpb._tool_memory_recall(
        "coffee",
        workspace,
        {"PATH": os.environ.get("PATH", "")},
        fast=True,
    )

    assert captured["timeout"] == 30
    assert meta["harness_telemetry"]["status"] == "ok"


def test_tool_memory_recall_allows_fast_timeout_override(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured: dict[str, object] = {}

    def _fake_run(_cmd, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        return SimpleNamespace(stdout=json.dumps({"results": [], "meta": {"mode": "fast"}}), stderr="", returncode=0)

    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setenv("BENCHMARK_RECALL_FAST_TIMEOUT_S", "120")
    monkeypatch.setenv("BENCHMARK_RELAX_TIMEOUTS", "1")
    monkeypatch.setattr(rpb, "_BACKEND", "llama-cpp")
    monkeypatch.setattr(rpb, "_OPENAI_COMPAT_URL", "http://127.0.0.1:30001")

    _text, meta = rpb._tool_memory_recall(
        "coffee",
        workspace,
        {"PATH": os.environ.get("PATH", "")},
        fast=True,
    )

    assert captured["timeout"] == 120
    assert meta["harness_telemetry"]["status"] == "ok"


def test_tool_memory_recall_parses_graph_direct_results_payload(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()

    def _fake_run(_cmd, **_kwargs):
        return SimpleNamespace(
            stdout=json.dumps({
                "direct_results": [{
                    "text": "Maya works at TechFlow",
                    "category": "fact",
                    "similarity": 0.88,
                    "id": "n42",
                    "privacy": "shared",
                    "owner_id": "maya",
                }],
                "source_breakdown": {"vector_count": 1, "graph_count": 0},
            }),
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)

    text, meta = rpb._tool_memory_recall("work", workspace, {"PATH": os.environ.get("PATH", "")})

    assert "Maya works at TechFlow" in text
    assert meta["source_breakdown"] == {"vector_count": 1, "graph_count": 0}
    assert meta["entities_found"] is None
    assert meta["harness_telemetry"]["status"] == "ok"


def test_tool_memory_recall_docs_only_uses_plaintext_cli(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured: dict[str, object] = {}

    def _fake_run(cmd, **_kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(
            stdout=json.dumps({
                "contract": "quaid.recall.v1",
                "results": [],
                "docs": {
                    "chunks": [
                        {
                            "content": "Express + PostgreSQL + React",
                            "source": "/tmp/workspace/projects/recipe-app/README.md",
                            "section_header": "## Stack",
                            "similarity": 0.91,
                            "chunk_index": 0,
                            "project": "recipe-app",
                        }
                    ],
                    "project": "recipe-app",
                    "project_md": "# Project: Recipe App\n",
                    "telemetry": {"chunk_count": 1, "resolved_project": "recipe-app"},
                },
            }),
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)

    text, meta = rpb._tool_memory_recall(
        "tech stack",
        workspace,
        {"PATH": os.environ.get("PATH", "")},
        stores=["docs"],
        project="recipe-app",
    )

    assert "--json" in captured["cmd"]
    assert "Express + PostgreSQL + React" in text
    assert "# Project: Recipe App" in text
    assert meta["docs_telemetry"] == {"chunk_count": 1, "resolved_project": "recipe-app"}
    assert meta["harness_telemetry"]["docs_requested"] is True
    assert meta["harness_telemetry"]["memory_requested"] is False


def test_tool_memory_recall_mixed_json_payload_merges_docs_and_results(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()

    def _fake_run(_cmd, **_kwargs):
        return SimpleNamespace(
            stdout=json.dumps({
                "contract": "quaid.recall.v1",
                "results": [{
                    "text": "Maya built the recipe app in Express",
                    "category": "fact",
                    "similarity": 0.87,
                    "id": "n1",
                    "privacy": "shared",
                    "owner_id": "maya",
                }],
                "docs": {
                    "chunks": [{
                        "content": "tests/recipe.test.js covers recipe CRUD flows",
                        "source": "/tmp/workspace/projects/recipe-app/tests/recipe.test.js",
                        "section_header": None,
                        "similarity": 0.89,
                        "chunk_index": 0,
                        "project": "recipe-app",
                    }],
                    "project": "recipe-app",
                    "project_md": "# Project: Recipe App\n",
                },
                "meta": {"mode": "deliberate", "total_ms": 55},
            }),
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)

    text, meta = rpb._tool_memory_recall(
        "recipe app tests",
        workspace,
        {"PATH": os.environ.get("PATH", "")},
        stores=["vector", "graph", "docs"],
        project="recipe-app",
    )

    assert "Maya built the recipe app in Express" in text
    assert "tests/recipe.test.js covers recipe CRUD flows" in text
    assert "# Project: Recipe App" in text
    assert meta["mode"] == "deliberate"
    assert meta["total_ms"] == 55
    assert meta["harness_telemetry"]["docs_requested"] is True
    assert meta["harness_telemetry"]["memory_requested"] is True


def test_tool_memory_recall_mixed_docs_temporal_split_defaults_memory_to_vector_only(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    (workspace / "data").mkdir(parents=True)
    db = workspace / "data" / "memory.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE nodes (id TEXT PRIMARY KEY, session_id TEXT, type TEXT)")
    conn.commit()
    conn.close()

    calls = []
    real_impl = rpb._tool_memory_recall

    def _spy(query, workspace, env, **kwargs):
        calls.append({"query": query, **kwargs})
        if kwargs.get("stores") == ["vector", "docs"]:
            return real_impl(query, workspace, env, **kwargs)
        if kwargs.get("stores") == ["vector"]:
            return "[0.90] [fact][C:0.9] Memory hit |ID:n1|T:|P:shared|O:maya", {"mode": "deliberate"}
        if kwargs.get("stores") == ["docs"]:
            return "=== Documentation ===\n1. projects/recipe-app/README.md", None
        raise AssertionError(f"unexpected recursive stores: {kwargs.get('stores')!r}")

    monkeypatch.setattr(rpb, "_tool_memory_recall", _spy)

    text, meta = rpb._tool_memory_recall(
        "recipe app current architecture",
        workspace,
        {"PATH": os.environ.get("PATH", "")},
        stores=["vector", "docs"],
        max_session=7,
    )

    subcall_stores = [call.get("stores") for call in calls[1:]]
    assert ["vector"] in subcall_stores
    assert ["docs"] in subcall_stores
    assert "Memory hit" in text
    assert "Documentation" in text
    assert meta == {"mode": "deliberate"}


def test_tool_memory_recall_max_session_filter_prefers_memory_db_path(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    (workspace / "data").mkdir(parents=True)
    (workspace / "benchrunner" / "data").mkdir(parents=True)
    legacy_db = workspace / "data" / "memory.db"
    instance_db = workspace / "benchrunner" / "data" / "memory.db"

    conn = sqlite3.connect(legacy_db)
    conn.execute("CREATE TABLE nodes (id TEXT PRIMARY KEY, session_id TEXT, type TEXT)")
    conn.execute("INSERT INTO nodes (id, session_id, type) VALUES ('n1', 'session-1', 'Fact')")
    conn.commit()
    conn.close()

    conn = sqlite3.connect(instance_db)
    conn.execute("CREATE TABLE nodes (id TEXT PRIMARY KEY, session_id TEXT, type TEXT)")
    conn.commit()
    conn.close()

    def _fake_run(_cmd, **_kwargs):
        return SimpleNamespace(
            stdout=json.dumps({
                "results": [{
                    "text": "Maya's partner is David",
                    "category": "fact",
                    "similarity": 0.95,
                    "id": "n1",
                    "privacy": "shared",
                    "owner_id": "maya",
                }],
                "meta": {"mode": "deliberate"},
            }),
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)

    text, _meta = rpb._tool_memory_recall(
        "partner name",
        workspace,
        {
            "PATH": os.environ.get("PATH", ""),
            "MEMORY_DB_PATH": str(legacy_db),
        },
        max_session=1,
    )

    assert "Maya's partner is David" in text


def test_tool_memory_recall_max_session_filter_maps_day_runtime_ids(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    (workspace / "data").mkdir(parents=True)
    (workspace / "lifecycle_resume" / "day-10-2026-03-18").mkdir(parents=True)
    (workspace / "lifecycle_resume" / "day-12-2026-03-22").mkdir(parents=True)
    db = workspace / "data" / "memory.db"

    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE nodes (id TEXT PRIMARY KEY, session_id TEXT, type TEXT)")
    conn.execute("INSERT INTO nodes (id, session_id, type) VALUES ('keep-runtime', 'day-runtime-2026-03-18', 'Fact')")
    conn.execute("INSERT INTO nodes (id, session_id, type) VALUES ('drop-runtime', 'day-runtime-2026-03-22', 'Fact')")
    conn.execute("INSERT INTO nodes (id, session_id, type) VALUES ('drop-session', 'session-16', 'Fact')")
    conn.commit()
    conn.close()

    def _fake_run(_cmd, **_kwargs):
        return SimpleNamespace(
            stdout=json.dumps({
                "results": [
                    {
                        "text": "session 10 test suite",
                        "category": "fact",
                        "similarity": 0.95,
                        "id": "keep-runtime",
                        "privacy": "shared",
                        "owner_id": "maya",
                    },
                    {
                        "text": "future graphQL suite",
                        "category": "fact",
                        "similarity": 0.95,
                        "id": "drop-runtime",
                        "privacy": "shared",
                        "owner_id": "maya",
                    },
                    {
                        "text": "future sharing suite",
                        "category": "fact",
                        "similarity": 0.95,
                        "id": "drop-session",
                        "privacy": "shared",
                        "owner_id": "maya",
                    },
                ],
                "meta": {"mode": "fast"},
            }),
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)

    text, _meta = rpb._tool_memory_recall(
        "recipe app test suites",
        workspace,
        {
            "PATH": os.environ.get("PATH", ""),
            "MEMORY_DB_PATH": str(db),
        },
        fast=True,
        max_session=10,
    )

    assert "session 10 test suite" in text
    assert "future graphQL suite" not in text
    assert "future sharing suite" not in text


def test_run_janitor_uses_configured_timeout(tmp_path, monkeypatch, capsys):
    workspace = tmp_path / "ws"
    workspace.mkdir()

    calls = []

    monkeypatch.setattr(rpb, "_benchmark_env", lambda _workspace, _phase: {"PATH": os.environ.get("PATH", "")})
    monkeypatch.setattr(
        rpb,
        "_python_cmd_for_quaid_script",
        lambda _script: ["python3", "-m", "core.lifecycle.janitor"],
    )
    monkeypatch.setattr(rpb, "_QUAID_DIR", tmp_path)

    def _fake_run(cmd, **kwargs):
        calls.append((list(cmd), kwargs.get("timeout")))
        return SimpleNamespace(returncode=0, stdout="janitor ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    rpb.run_janitor(workspace, timeout_seconds=1800)

    out = capsys.readouterr().out
    assert any("--task" in cmd and "all" in cmd and timeout == 1800 for cmd, timeout in calls)
    assert any("--force-distill" in cmd for cmd, _timeout in calls)
    assert "timeout=1800s" in out


def test_recall_tool_description_prefers_vector_default_and_explicit_graph():
    desc = rpb._RECALL_TOOL_DESCRIPTION

    assert "default memory recall uses vector only" in desc
    assert "stores=['graph']" in desc
    assert "vector + graph" not in desc
    assert "historical/as-of" in desc
    assert "date_from/date_to" in desc
    assert "do not rely on undated recall" in desc
    assert "dated PROJECT.log evidence" in desc


def test_execute_recall_tool_maps_date_aliases(tmp_path, monkeypatch):
    captured = {}

    def _fake_recall(query, workspace, env, **kwargs):
        captured["query"] = query
        captured.update(kwargs)
        return "ok", {"mode": "deliberate"}

    monkeypatch.setattr(rpb, "_tool_memory_recall", _fake_recall)

    text, meta = rpb._execute_tool(
        "recall",
        {
            "query": "what did we know before March 10?",
            "after": "2026-03-01",
            "as_of": "2026-03-10",
            "stores": ["docs"],
            "project": "quaid",
        },
        tmp_path,
        {},
    )

    assert text == "ok"
    assert meta == {"mode": "deliberate"}
    assert captured["date_from"] == "2026-03-01"
    assert captured["date_to"] == "2026-03-10"
    assert captured["stores"] == ["docs"]
    assert captured["project"] == "quaid"


def test_recall_date_bounds_normalizes_aliases_for_telemetry():
    date_from, date_to, aliases = rpb._recall_date_bounds_from_tool_input({
        "query": "what changed after March 1 as of March 10?",
        "after": "2026-03-01",
        "as_of": "2026-03-10",
    })

    assert date_from == "2026-03-01"
    assert date_to == "2026-03-10"
    assert aliases == {
        "after": "2026-03-01",
        "as_of": "2026-03-10",
    }


def test_call_anthropic_cached_retries_http_520(monkeypatch):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({
                "content": [{"text": "ok"}],
                "usage": {"input_tokens": 1, "output_tokens": 2},
            }).encode()

    attempts = {"n": 0}

    def _urlopen(req, timeout=300):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise urllib.error.HTTPError(
                url=req.full_url,
                code=520,
                msg="edge failure",
                hdrs=None,
                fp=io.BytesIO(b"error code: 520"),
            )
        return _Resp()

    monkeypatch.setattr(rpb.urllib.request, "urlopen", _urlopen)
    monkeypatch.setattr(rpb.random, "uniform", lambda a, b: 0.0)
    monkeypatch.setattr(rpb.time, "sleep", lambda _s: None)
    monkeypatch.setenv("ANTHROPIC_RETRY_ATTEMPTS", "2")
    monkeypatch.setenv("ANTHROPIC_RETRY_BACKOFF_S", "0.5")
    monkeypatch.setenv("ANTHROPIC_RETRY_BACKOFF_CAP_S", "0.5")
    monkeypatch.setattr(rpb, "_BACKEND", "oauth")

    text, usage = rpb._call_anthropic_cached("system", "user", "claude-haiku-4-5-20251001", "sk-test")

    assert attempts["n"] == 2
    assert text == "ok"
    assert usage["output_tokens"] == 2

def test_oauth_backend_prefers_benchmark_oauth_token(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setattr(rpb, "_BACKEND", "oauth")
    monkeypatch.setattr(rpb, "_find_anthropic_credential", lambda: "sk-ant-oat01-test-token")

    env = rpb._make_env(workspace)
    assert env["BENCHMARK_ANTHROPIC_OAUTH_TOKEN"] == "sk-ant-oat01-test-token"
    assert env["ANTHROPIC_API_KEY"] == "sk-ant-oat01-test-token"


def test_has_rolling_obd_resume_state_detects_staged_workspace(tmp_path):
    workspace = tmp_path / "run"
    staged = workspace / rpb._BENCHMARK_QUAID_INSTANCE / "data" / "rolling-extraction"
    staged.mkdir(parents=True, exist_ok=True)
    (staged / "obd-compaction-0001.json").write_text("{}")

    assert rpb._has_rolling_obd_resume_state(workspace) is True


def test_main_normalizes_api_backend_alias_to_oauth(tmp_path, monkeypatch):
    workspace = tmp_path / "run"
    (workspace / "data").mkdir(parents=True)
    (workspace / "data" / "memory.db").write_text("")

    seen = {"eval_backend": None, "tier5_backend": None, "api_key_calls": 0}

    def _fake_get_api_key():
        seen["api_key_calls"] += 1
        return "sk-ant-oat01-test-token"

    def _fake_run_eval(*_a, **_k):
        seen["eval_backend"] = rpb._BACKEND
        return []

    def _fake_run_tier5(*_a, **_k):
        seen["tier5_backend"] = rpb._BACKEND
        return []

    monkeypatch.setattr(rpb, "_get_api_key", _fake_get_api_key)
    monkeypatch.setattr(rpb, "run_eval", _fake_run_eval)
    monkeypatch.setattr(rpb, "run_tier5_eval", _fake_run_tier5)
    monkeypatch.setattr(rpb, "_save_token_usage", lambda *_a, **_k: None)
    monkeypatch.setattr(
        rpb,
        "score_results",
        lambda _results: {
            "overall": {"accuracy": 0.0, "count": 0, "scored": 0, "correct": 0, "partial": 0, "wrong": 0, "error": 0},
            "per_type": {},
            "per_difficulty": {},
        },
    )
    monkeypatch.setattr(sys, "argv", [
        "run_production_benchmark.py",
        "--mode", "eval",
        "--results-dir", str(workspace),
        "--backend", "api",
    ])

    rpb.main()

    assert seen["api_key_calls"] == 1
    assert seen["eval_backend"] == "oauth"
    assert seen["tier5_backend"] == "oauth"


def test_main_vllm_requires_url_and_model(tmp_path, monkeypatch):
    workspace = tmp_path / "run"
    monkeypatch.delenv("BENCHMARK_VLLM_URL", raising=False)
    monkeypatch.delenv("BENCHMARK_VLLM_MODEL", raising=False)

    monkeypatch.setattr(sys, "argv", [
        "run_production_benchmark.py",
        "--mode", "eval",
        "--results-dir", str(workspace),
        "--backend", "vllm",
    ])
    with pytest.raises(SystemExit, match="--backend vllm requires --vllm-url"):
        rpb.main()

    monkeypatch.setattr(sys, "argv", [
        "run_production_benchmark.py",
        "--mode", "eval",
        "--results-dir", str(workspace),
        "--backend", "vllm",
        "--vllm-url", "http://spark:8000",
    ])
    with pytest.raises(SystemExit, match="--backend vllm requires --vllm-model"):
        rpb.main()


def test_main_llama_cpp_requires_url_and_model(tmp_path, monkeypatch):
    workspace = tmp_path / "run"
    monkeypatch.delenv("BENCHMARK_LLAMA_CPP_URL", raising=False)
    monkeypatch.delenv("BENCHMARK_LLAMA_CPP_MODEL", raising=False)

    monkeypatch.setattr(sys, "argv", [
        "run_production_benchmark.py",
        "--mode", "eval",
        "--results-dir", str(workspace),
        "--backend", "llama-cpp",
    ])
    with pytest.raises(SystemExit, match="--backend llama-cpp requires --llama-cpp-url"):
        rpb.main()

    monkeypatch.setattr(sys, "argv", [
        "run_production_benchmark.py",
        "--mode", "eval",
        "--results-dir", str(workspace),
        "--backend", "llama-cpp",
        "--llama-cpp-url", "http://spark:8080",
    ])
    with pytest.raises(SystemExit, match="--backend llama-cpp requires --llama-cpp-model"):
        rpb.main()


def test_main_vllm_defaults_eval_model_to_served_model(tmp_path, monkeypatch):
    workspace = tmp_path / "run"
    (workspace / "data").mkdir(parents=True)
    (workspace / "data" / "memory.db").write_text("")

    seen = {"eval_model": None, "backend": None}

    def _fake_run_eval(*_a, **kwargs):
        seen["eval_model"] = kwargs.get("eval_model")
        seen["backend"] = rpb._BACKEND
        return []

    monkeypatch.setattr(rpb, "run_eval", _fake_run_eval)
    monkeypatch.setattr(rpb, "run_tier5_eval", lambda *_a, **_k: [])
    monkeypatch.setattr(rpb, "_save_token_usage", lambda *_a, **_k: None)
    monkeypatch.setattr(
        rpb,
        "score_results",
        lambda _results: {
            "overall": {"accuracy": 0.0, "count": 0, "scored": 0, "correct": 0, "partial": 0, "wrong": 0, "error": 0},
            "per_type": {},
            "per_difficulty": {},
        },
    )
    monkeypatch.setenv("OPENAI_COMPATIBLE_DEEP_BASE_URL", "http://127.0.0.1:30001")
    monkeypatch.setenv("OPENAI_COMPATIBLE_FAST_BASE_URL", "http://127.0.0.1:30002")
    monkeypatch.setattr(sys, "argv", [
        "run_production_benchmark.py",
        "--mode", "eval",
        "--results-dir", str(workspace),
        "--backend", "vllm",
        "--vllm-url", "http://spark:8000",
        "--vllm-model", "gemma-3-31b-it",
        "--vllm-judge-url", "http://spark:8001",
        "--vllm-judge-model", "gemma-3-27b-it",
        "--allow-non-haiku-answer-model",
    ])

    rpb.main()

    assert seen["backend"] == "vllm"
    assert seen["eval_model"] == "gemma-3-31b-it"


def test_main_llama_cpp_defaults_eval_model_to_served_model(tmp_path, monkeypatch):
    workspace = tmp_path / "run"
    (workspace / "data").mkdir(parents=True)
    (workspace / "data" / "memory.db").write_text("")

    seen = {"eval_model": None, "backend": None}

    def _fake_run_eval(*_a, **kwargs):
        seen["eval_model"] = kwargs.get("eval_model")
        seen["backend"] = rpb._BACKEND
        return []

    monkeypatch.setattr(rpb, "run_eval", _fake_run_eval)
    monkeypatch.setattr(rpb, "run_tier5_eval", lambda *_a, **_k: [])
    monkeypatch.setattr(rpb, "_save_token_usage", lambda *_a, **_k: None)
    monkeypatch.setattr(
        rpb,
        "score_results",
        lambda _results: {
            "overall": {"accuracy": 0.0, "count": 0, "scored": 0, "correct": 0, "partial": 0, "wrong": 0, "error": 0},
            "per_type": {},
            "per_difficulty": {},
        },
    )
    monkeypatch.setenv("OPENAI_COMPATIBLE_DEEP_BASE_URL", "http://127.0.0.1:30001")
    monkeypatch.setenv("OPENAI_COMPATIBLE_FAST_BASE_URL", "http://127.0.0.1:30002")
    monkeypatch.setattr(sys, "argv", [
        "run_production_benchmark.py",
        "--mode", "eval",
        "--results-dir", str(workspace),
        "--backend", "llama-cpp",
        "--llama-cpp-url", "http://spark:8080",
        "--llama-cpp-model", "gemma-3-31b-it",
        "--llama-cpp-judge-url", "http://spark:8081",
        "--llama-cpp-judge-model", "gemma-3-27b-it",
        "--allow-non-haiku-answer-model",
    ])

    rpb.main()

    assert seen["backend"] == "llama-cpp"
    assert seen["eval_model"] == "gemma-3-31b-it"


def test_main_vllm_accepts_env_backed_url_and_model(tmp_path, monkeypatch):
    workspace = tmp_path / "run"
    (workspace / "data").mkdir(parents=True)
    (workspace / "data" / "memory.db").write_text("")

    seen = {"eval_model": None, "backend": None}

    def _fake_run_eval(*_a, **kwargs):
        seen["eval_model"] = kwargs.get("eval_model")
        seen["backend"] = rpb._BACKEND
        return []

    monkeypatch.setattr(rpb, "run_eval", _fake_run_eval)
    monkeypatch.setattr(rpb, "run_tier5_eval", lambda *_a, **_k: [])
    monkeypatch.setattr(rpb, "_save_token_usage", lambda *_a, **_k: None)
    monkeypatch.setattr(
        rpb,
        "score_results",
        lambda _results: {
            "overall": {"accuracy": 0.0, "count": 0, "scored": 0, "correct": 0, "partial": 0, "wrong": 0, "error": 0},
            "per_type": {},
            "per_difficulty": {},
        },
    )
    monkeypatch.setenv("BENCHMARK_VLLM_URL", "http://spark:8000")
    monkeypatch.setenv("BENCHMARK_VLLM_MODEL", "gemma-3-31b-it")
    monkeypatch.setenv("BENCHMARK_VLLM_JUDGE_URL", "http://spark:8001")
    monkeypatch.setenv("BENCHMARK_VLLM_JUDGE_MODEL", "gemma-3-27b-it")
    monkeypatch.setenv("OPENAI_COMPATIBLE_DEEP_BASE_URL", "http://127.0.0.1:30001")
    monkeypatch.setenv("OPENAI_COMPATIBLE_FAST_BASE_URL", "http://127.0.0.1:30002")
    monkeypatch.setattr(sys, "argv", [
        "run_production_benchmark.py",
        "--mode", "eval",
        "--results-dir", str(workspace),
        "--backend", "vllm",
        "--allow-non-haiku-answer-model",
    ])

    rpb.main()

    assert seen["backend"] == "vllm"
    assert seen["eval_model"] == "gemma-3-31b-it"


def test_main_llama_cpp_accepts_env_backed_url_and_model(tmp_path, monkeypatch):
    workspace = tmp_path / "run"
    (workspace / "data").mkdir(parents=True)
    (workspace / "data" / "memory.db").write_text("")

    seen = {"eval_model": None, "backend": None}

    def _fake_run_eval(*_a, **kwargs):
        seen["eval_model"] = kwargs.get("eval_model")
        seen["backend"] = rpb._BACKEND
        return []

    monkeypatch.setattr(rpb, "run_eval", _fake_run_eval)
    monkeypatch.setattr(rpb, "run_tier5_eval", lambda *_a, **_k: [])
    monkeypatch.setattr(rpb, "_save_token_usage", lambda *_a, **_k: None)
    monkeypatch.setattr(
        rpb,
        "score_results",
        lambda _results: {
            "overall": {"accuracy": 0.0, "count": 0, "scored": 0, "correct": 0, "partial": 0, "wrong": 0, "error": 0},
            "per_type": {},
            "per_difficulty": {},
        },
    )
    monkeypatch.setenv("BENCHMARK_LLAMA_CPP_URL", "http://spark:8080")
    monkeypatch.setenv("BENCHMARK_LLAMA_CPP_MODEL", "gemma-3-31b-it")
    monkeypatch.setenv("BENCHMARK_LLAMA_CPP_JUDGE_URL", "http://spark:8081")
    monkeypatch.setenv("BENCHMARK_LLAMA_CPP_JUDGE_MODEL", "gemma-3-27b-it")
    monkeypatch.setenv("OPENAI_COMPATIBLE_DEEP_BASE_URL", "http://127.0.0.1:30001")
    monkeypatch.setenv("OPENAI_COMPATIBLE_FAST_BASE_URL", "http://127.0.0.1:30002")
    monkeypatch.setattr(sys, "argv", [
        "run_production_benchmark.py",
        "--mode", "eval",
        "--results-dir", str(workspace),
        "--backend", "llama-cpp",
        "--allow-non-haiku-answer-model",
    ])

    rpb.main()

    assert seen["backend"] == "llama-cpp"
    assert seen["eval_model"] == "gemma-3-31b-it"


def test_main_openai_defaults_to_direct_openai_models(tmp_path, monkeypatch):
    workspace = tmp_path / "run"
    (workspace / "data").mkdir(parents=True)
    (workspace / "data" / "memory.db").write_text("")

    seen = {"eval_model": None, "backend": None, "compat_url": None, "compat_key_env": None}

    def _fake_run_eval(*_a, **kwargs):
        seen["eval_model"] = kwargs.get("eval_model")
        seen["backend"] = rpb._BACKEND
        seen["compat_url"] = rpb._OPENAI_COMPAT_URL
        seen["compat_key_env"] = rpb._OPENAI_COMPAT_API_KEY_ENV
        return []

    monkeypatch.setattr(rpb, "run_eval", _fake_run_eval)
    monkeypatch.setattr(rpb, "run_tier5_eval", lambda *_a, **_k: [])
    monkeypatch.setattr(rpb, "_save_token_usage", lambda *_a, **_k: None)
    monkeypatch.setattr(
        rpb,
        "score_results",
        lambda _results: {
            "overall": {"accuracy": 0.0, "count": 0, "scored": 0, "correct": 0, "partial": 0, "wrong": 0, "error": 0},
            "per_type": {},
            "per_difficulty": {},
        },
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(sys, "argv", [
        "run_production_benchmark.py",
        "--mode", "eval",
        "--results-dir", str(workspace),
        "--backend", "openai",
        "--judge", "gpt-5.4-mini",
        "--allow-non-haiku-answer-model",
    ])

    rpb.main()

    assert seen["backend"] == "openai"
    assert seen["eval_model"] == "gpt-5.4"
    assert seen["compat_url"] == "https://api.openai.com"
    assert seen["compat_key_env"] == "OPENAI_API_KEY"


def test_run_eval_parallel_progress_heartbeats_without_completed_queries(tmp_path, monkeypatch):
    workspace = tmp_path / "run"
    (workspace / "data").mkdir(parents=True)
    (workspace / "data" / "memory.db").write_text("", encoding="utf-8")

    queries = [
        {
            "question": f"Q{i}?",
            "ground_truth": f"A{i}",
            "query_type": "temporal",
            "source_session": i,
        }
        for i in range(1, 6)
    ]

    monkeypatch.setattr(
        rpb,
        "_load_reviews_with_dataset_gate",
        lambda _max_sessions: (tmp_path / "assets", [], [object(), object()], "v-test", 268),
    )
    monkeypatch.setattr(rpb, "get_all_eval_queries", lambda _reviews: queries)
    monkeypatch.setattr(rpb, "_apply_eval_query_profile", lambda queries: (queries, {"profile": "full", "selected": len(queries), "requested": len(queries)}))
    monkeypatch.setattr(rpb, "_eval_core_context_preflight", lambda *_a, **_k: None)
    monkeypatch.setattr(rpb, "_eval_embedding_provider_preflight", lambda *_a, **_k: None)
    monkeypatch.setattr(rpb, "_write_eval_query_profile_manifest", lambda *_a, **_k: None)
    monkeypatch.setattr(rpb, "_resolve_eval_context_profile", lambda: ("full", ["SOUL", "USER", "MEMORY"], True))
    monkeypatch.setattr(rpb, "_build_eval_context", lambda *_a, **_k: "ctx")
    monkeypatch.setattr(rpb, "_build_eval_context_sources", lambda *_a, **_k: [])
    monkeypatch.setattr(rpb, "_resolve_eval_provider", lambda *_a, **_k: "openai-compatible")
    monkeypatch.setattr(rpb, "_benchmark_env", lambda *_a, **_k: {})
    monkeypatch.setattr(rpb, "_sync_instance_identity_to_workspace_root", lambda *_a, **_k: None)
    monkeypatch.setattr(rpb, "_save_eval_resume_checkpoint", lambda *_a, **_k: None)
    monkeypatch.setattr(rpb, "_summarize_usage_events", lambda *_a, **_k: {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "api_calls": 1})
    monkeypatch.setattr(rpb, "_resolve_eval_parallel_workers", lambda: 2)
    monkeypatch.setenv("BENCHMARK_REQUIRE_QUERY_COUNT", "0")
    monkeypatch.setattr(
        rpb,
        "_tool_use_loop",
        lambda **_k: ("prediction", [], [], [], {"input_tokens": 1, "output_tokens": 1, "api_calls": 1, "tool_call_details": []}),
    )
    monkeypatch.setattr(rpb, "_judge", lambda *_a, **_k: ("CORRECT", 1.0))

    class _FakeFuture:
        def __init__(self, fn, args):
            self._fn = fn
            self._args = args
            self._done = False
            self._value = None

        def result(self):
            if not self._done:
                self._value = self._fn(*self._args)
                self._done = True
            return self._value

    class _FakeExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            assert not kwargs
            return _FakeFuture(fn, args)

    heartbeat_snapshot = {}
    wait_calls = {"n": 0}

    def _fake_wait(pending, timeout=None, return_when=None):
        wait_calls["n"] += 1
        progress_path = workspace / "logs" / "eval_progress.json"
        if wait_calls["n"] == 1:
            return set(), set(pending)
        if wait_calls["n"] == 2:
            payload = json.loads(progress_path.read_text(encoding="utf-8"))
            heartbeat_snapshot.update(payload)
            first = next(iter(pending))
            rest = set(pending)
            rest.remove(first)
            return {first}, rest
        return set(pending), set()

    monkeypatch.setattr(rpb.concurrent.futures, "ThreadPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(rpb.concurrent.futures, "wait", _fake_wait)

    results = rpb.run_eval(
        workspace,
        api_key="",
        eval_model="gemma-4-31b-q8",
        judge_model="gemma-4-31b-q8",
    )

    assert len(results) == 5
    assert heartbeat_snapshot["total_queries"] == 5
    assert heartbeat_snapshot["completed"] == 0
    assert heartbeat_snapshot["active_queries"] == 2
    assert heartbeat_snapshot["scored"] == 0
    assert heartbeat_snapshot["accuracy_so_far"] == 0.0


def test_main_vllm_defaults_ingest_model_to_served_model(tmp_path, monkeypatch):
    workspace = tmp_path / "run"

    seen = {"model": None}

    monkeypatch.setattr(
        rpb,
        "setup_workspace",
        lambda path, **kwargs: seen.__setitem__("model", kwargs.get("extraction_model")) or Path(path).mkdir(parents=True, exist_ok=True),
    )
    monkeypatch.setattr(
        rpb,
        "run_per_day_extraction",
        lambda *_a, **kwargs: {
            "schedule_mode": kwargs.get("schedule_mode"),
            "total_facts": 0,
            "stored": 0,
            "edges": 0,
            "semantic_dedup_checks": 0,
            "semantic_duplicate_facts_collapsed": 0,
            "signal_to_publish_seconds": 0.0,
        },
    )
    monkeypatch.setattr(rpb, "verify_post_janitor", lambda *_a, **_k: None)
    monkeypatch.setattr(rpb, "_save_ingest_usage", lambda *_a, **_k: None)
    monkeypatch.setattr(rpb, "run_eval", lambda *_a, **_k: [])
    monkeypatch.setattr(rpb, "run_tier5_eval", lambda *_a, **_k: [])
    monkeypatch.setattr(rpb, "_save_token_usage", lambda *_a, **_k: None)
    monkeypatch.setattr(
        rpb,
        "score_results",
        lambda _results: {
            "overall": {"accuracy": 0.0, "count": 0, "scored": 0, "correct": 0, "partial": 0, "wrong": 0, "error": 0},
            "per_type": {},
            "per_difficulty": {},
        },
    )
    monkeypatch.setenv("OPENAI_COMPATIBLE_DEEP_BASE_URL", "http://127.0.0.1:30001")
    monkeypatch.setenv("OPENAI_COMPATIBLE_FAST_BASE_URL", "http://127.0.0.1:30002")
    monkeypatch.setattr(sys, "argv", [
        "run_production_benchmark.py",
        "--mode", "per-day",
        "--results-dir", str(workspace),
        "--backend", "vllm",
        "--vllm-url", "http://spark:8000",
        "--vllm-model", "gemma-3-31b-it",
        "--vllm-judge-url", "http://spark:8001",
        "--vllm-judge-model", "gemma-3-27b-it",
        "--allow-non-haiku-answer-model",
    ])

    rpb.main()

    assert seen["model"] == "gemma-3-31b-it"


def test_main_llama_cpp_defaults_ingest_model_to_served_model(tmp_path, monkeypatch):
    workspace = tmp_path / "run"

    seen = {"model": None}

    monkeypatch.setattr(
        rpb,
        "setup_workspace",
        lambda path, **kwargs: seen.__setitem__("model", kwargs.get("extraction_model")) or Path(path).mkdir(parents=True, exist_ok=True),
    )
    monkeypatch.setattr(
        rpb,
        "run_per_day_extraction",
        lambda *_a, **kwargs: {
            "schedule_mode": kwargs.get("schedule_mode"),
            "total_facts": 0,
            "stored": 0,
            "edges": 0,
            "semantic_dedup_checks": 0,
            "semantic_duplicate_facts_collapsed": 0,
            "signal_to_publish_seconds": 0.0,
        },
    )
    monkeypatch.setattr(rpb, "verify_post_janitor", lambda *_a, **_k: None)
    monkeypatch.setattr(rpb, "_save_ingest_usage", lambda *_a, **_k: None)
    monkeypatch.setattr(rpb, "run_eval", lambda *_a, **_k: [])
    monkeypatch.setattr(rpb, "run_tier5_eval", lambda *_a, **_k: [])
    monkeypatch.setattr(rpb, "_save_token_usage", lambda *_a, **_k: None)
    monkeypatch.setattr(
        rpb,
        "score_results",
        lambda _results: {
            "overall": {"accuracy": 0.0, "count": 0, "scored": 0, "correct": 0, "partial": 0, "wrong": 0, "error": 0},
            "per_type": {},
            "per_difficulty": {},
        },
    )
    monkeypatch.setenv("OPENAI_COMPATIBLE_DEEP_BASE_URL", "http://127.0.0.1:30001")
    monkeypatch.setenv("OPENAI_COMPATIBLE_FAST_BASE_URL", "http://127.0.0.1:30002")
    monkeypatch.setattr(sys, "argv", [
        "run_production_benchmark.py",
        "--mode", "per-day",
        "--results-dir", str(workspace),
        "--backend", "llama-cpp",
        "--llama-cpp-url", "http://spark:8080",
        "--llama-cpp-model", "gemma-3-31b-it",
        "--llama-cpp-judge-url", "http://spark:8081",
        "--llama-cpp-judge-model", "gemma-3-27b-it",
        "--allow-non-haiku-answer-model",
    ])

    rpb.main()

    assert seen["model"] == "gemma-3-31b-it"


def test_main_resume_rolling_obd_skips_workspace_setup(tmp_path, monkeypatch):
    workspace = tmp_path / "run"
    staged = workspace / rpb._BENCHMARK_QUAID_INSTANCE / "data" / "rolling-extraction"
    staged.mkdir(parents=True, exist_ok=True)
    (staged / "obd-compaction-0001.json").write_text("{}")
    (workspace / "data").mkdir(parents=True, exist_ok=True)
    (workspace / "data" / "memory.db").write_text("")

    seen = {"setup": 0, "resume_state": None}

    monkeypatch.setattr(rpb, "_get_api_key", lambda: "sk-ant-oat01-test-token")
    monkeypatch.setattr(
        rpb,
        "setup_workspace",
        lambda path, **_k: (Path(path).mkdir(parents=True, exist_ok=True), seen.__setitem__("setup", seen["setup"] + 1)),
    )
    monkeypatch.setattr(
        rpb,
        "run_per_day_extraction",
        lambda *args, **kwargs: seen.__setitem__("resume_state", kwargs.get("resume_state")) or {
            "schedule_mode": "rolling-obd",
            "total_facts": 0,
            "stored": 0,
            "edges": 0,
            "semantic_dedup_checks": 0,
            "semantic_duplicate_facts_collapsed": 0,
            "signal_to_publish_seconds": 0.0,
        },
    )
    monkeypatch.setattr(rpb, "verify_post_janitor", lambda *_a, **_k: None)
    monkeypatch.setattr(rpb, "_save_ingest_usage", lambda *_a, **_k: None)
    monkeypatch.setattr(sys, "argv", [
        "run_production_benchmark.py",
        "--mode", "ingest",
        "--results-dir", str(workspace),
        "--backend", "oauth",
        "--ingest-schedule", "rolling-obd",
        "--resume-day-lifecycle",
        "--allow-non-haiku-answer-model",
    ])

    rpb.main()

    assert seen["setup"] == 0
    assert seen["resume_state"] == {"mode": "rolling-obd-resume"}


def test_main_obd_aliases_to_rolling_obd(tmp_path, monkeypatch):
    workspace = tmp_path / "run"

    seen = {"setup": 0, "schedule_mode": None}

    monkeypatch.setattr(rpb, "_get_api_key", lambda: "sk-ant-oat01-test-token")
    monkeypatch.setattr(
        rpb,
        "setup_workspace",
        lambda path, **_k: (Path(path).mkdir(parents=True, exist_ok=True), seen.__setitem__("setup", seen["setup"] + 1)),
    )
    monkeypatch.setattr(
        rpb,
        "run_per_day_extraction",
        lambda *args, **kwargs: seen.__setitem__("schedule_mode", kwargs.get("schedule_mode")) or {
            "schedule_mode": kwargs.get("schedule_mode"),
            "total_facts": 0,
            "stored": 0,
            "edges": 0,
            "semantic_dedup_checks": 0,
            "semantic_duplicate_facts_collapsed": 0,
            "signal_to_publish_seconds": 0.0,
        },
    )
    monkeypatch.setattr(rpb, "verify_post_janitor", lambda *_a, **_k: None)
    monkeypatch.setattr(rpb, "_save_ingest_usage", lambda *_a, **_k: None)
    monkeypatch.setattr(sys, "argv", [
        "run_production_benchmark.py",
        "--mode", "ingest",
        "--results-dir", str(workspace),
        "--backend", "oauth",
        "--ingest-schedule", "obd",
        "--allow-non-haiku-answer-model",
    ])

    rpb.main()

    assert seen["setup"] == 1
    assert seen["schedule_mode"] == "rolling-obd"


def test_anthropic_oauth_headers_include_claude_code_identity():
    headers = rpb._anthropic_headers("sk-ant-oat01-test-token", prompt_caching=False)
    assert headers["Authorization"] == "Bearer sk-ant-oat01-test-token"
    assert headers["Accept"] == "application/json"
    assert headers["user-agent"] == "claude-cli/2.1.2 (external, cli)"
    assert headers["x-app"] == "cli"
    assert "claude-code-20250219" in headers["anthropic-beta"]
    assert "oauth-2025-04-20" in headers["anthropic-beta"]


def test_anthropic_oauth_system_blocks_include_claude_code_identity():
    blocks = rpb._anthropic_system_blocks(
        "Answer directly.",
        "sk-ant-oat01-test-token",
        prompt_caching=False,
    )
    assert isinstance(blocks, list)
    assert blocks[0]["text"] == "You are Claude Code, Anthropic's official CLI for Claude."
    assert blocks[1]["text"] == "Answer directly."


def test_anthropic_text_blocks_support_mixed_cache_policy():
    blocks = rpb._anthropic_text_blocks(
        [
            {"text": "cached prefix", "cache": True},
            {"text": "dynamic suffix", "cache": False},
        ],
        prompt_caching=False,
    )
    assert blocks[0]["text"] == "cached prefix"
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert blocks[1]["text"] == "dynamic suffix"
    assert "cache_control" not in blocks[1]


class TestSetupWorkspaceConfig:
    @pytest.fixture(autouse=True)
    def _stub_product_project_registration(self, monkeypatch):
        monkeypatch.setattr(rpb, "_register_benchmark_projects", lambda _workspace: None)
        monkeypatch.setattr(rpb, "_ensure_project_docs_supervisor_running", lambda _workspace: None)

    def test_workspace_seeds_janitor_checkpoint_and_clears_deferred_notices(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        quaid_dir = tmp_path / "modules" / "quaid"
        quaid_dir.mkdir(parents=True)
        (quaid_dir / "schema.sql").write_text("CREATE TABLE test(id INTEGER);", encoding="utf-8")
        (quaid_dir / "config").mkdir(parents=True)
        (quaid_dir / "config" / "memory.json").write_text(json.dumps({}), encoding="utf-8")
        monkeypatch.setattr(rpb, "_QUAID_DIR", quaid_dir)
        monkeypatch.setattr(rpb, "_BACKEND", "oauth")
        monkeypatch.setattr(rpb, "_bootstrap_domain_registry", lambda conn: None)
        monkeypatch.setattr(rpb, "_load_active_domains", lambda _workspace: [])

        stale_checkpoint = {
            "task": "all",
            "status": "failed",
            "last_completed_at": "2000-01-01T00:00:00Z",
        }
        stale_deferred = {
            "version": 1,
            "requests": [
                {
                    "id": "janitor-test",
                    "status": "pending",
                    "kind": "janitor",
                    "priority": "normal",
                    "message": "stale notice",
                }
            ],
        }
        roots = [
            workspace / rpb._BENCHMARK_QUAID_INSTANCE,
            workspace / "instances" / rpb._BENCHMARK_QUAID_INSTANCE,
        ]
        for root in roots:
            checkpoint = root / "logs" / "janitor" / "checkpoint-all.json"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(json.dumps(stale_checkpoint), encoding="utf-8")
            deferred = root / ".runtime" / "notes" / "delayed-llm-requests.json"
            deferred.parent.mkdir(parents=True, exist_ok=True)
            deferred.write_text(json.dumps(stale_deferred), encoding="utf-8")

        rpb.setup_workspace(workspace)

        for root in roots:
            checkpoint = root / "logs" / "janitor" / "checkpoint-all.json"
            checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            assert checkpoint_payload["status"] == "completed"
            assert str(checkpoint_payload.get("last_completed_at") or "").strip()
            assert str(checkpoint_payload.get("benchmark_seeded_at") or "").strip()

            deferred = root / ".runtime" / "notes" / "delayed-llm-requests.json"
            deferred_payload = json.loads(deferred.read_text(encoding="utf-8"))
            assert deferred_payload == {"version": 1, "requests": []}

    def test_claude_code_workspace_forces_split_tiers(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        quaid_dir = tmp_path / "modules" / "quaid"
        quaid_dir.mkdir(parents=True)
        (quaid_dir / "schema.sql").write_text("CREATE TABLE test(id INTEGER);", encoding="utf-8")
        (quaid_dir / "config").mkdir(parents=True)
        (quaid_dir / "config" / "memory.json").write_text(
            json.dumps(
                {
                    "models": {
                        "llmProvider": "openai-compatible",
                        "deepReasoning": "gpt-4o-mini",
                        "fastReasoning": "gpt-4o-mini",
                        "baseUrl": "https://api.openai.com",
                        "apiKeyEnv": "OPENAI_API_KEY",
                    }
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(rpb, "_QUAID_DIR", quaid_dir)
        monkeypatch.setattr(rpb, "_BACKEND", "claude-code")
        monkeypatch.setattr(rpb, "_bootstrap_domain_registry", lambda conn: None)
        monkeypatch.setattr(rpb, "_load_active_domains", lambda workspace: [])
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.delenv("BENCHMARK_REASONING_MODEL", raising=False)
        monkeypatch.delenv("BENCHMARK_DEEP_REASONING_MODEL", raising=False)
        monkeypatch.delenv("BENCHMARK_FAST_REASONING_MODEL", raising=False)

        rpb.setup_workspace(workspace)

        cfg = json.loads((workspace / "config" / "memory.json").read_text(encoding="utf-8"))
        models = cfg["models"]
        assert models["llmProvider"] == "claude-code"
        assert models["deepReasoningProvider"] == "claude-code"
        assert models["fastReasoningProvider"] == "anthropic"
        assert models["deepReasoning"] == "claude-sonnet-4-6"
        assert models["fastReasoning"] == "claude-haiku-4-5-20251001"
        assert cfg["capture"]["chunkTokens"] == 8000
        assert "baseUrl" not in models
        assert "apiKeyEnv" not in models

    def test_api_workspace_respects_explicit_split_tiers(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        quaid_dir = tmp_path / "modules" / "quaid"
        quaid_dir.mkdir(parents=True)
        (quaid_dir / "schema.sql").write_text("CREATE TABLE test(id INTEGER);", encoding="utf-8")
        (quaid_dir / "config").mkdir(parents=True)
        (quaid_dir / "config" / "memory.json").write_text(
            json.dumps(
                {
                    "models": {
                        "llmProvider": "anthropic",
                        "deepReasoning": "claude-haiku-4-5-20251001",
                        "fastReasoning": "claude-haiku-4-5-20251001",
                    }
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(rpb, "_QUAID_DIR", quaid_dir)
        monkeypatch.setattr(rpb, "_BACKEND", "oauth")
        monkeypatch.setattr(rpb, "_bootstrap_domain_registry", lambda conn: None)
        monkeypatch.setattr(rpb, "_load_active_domains", lambda workspace: [])
        monkeypatch.delenv("BENCHMARK_REASONING_MODEL", raising=False)
        monkeypatch.setenv("BENCHMARK_DEEP_REASONING_MODEL", "claude-sonnet-4-6")
        monkeypatch.setenv("BENCHMARK_FAST_REASONING_MODEL", "claude-haiku-4-5-20251001")

        rpb.setup_workspace(workspace)

        cfg = json.loads((workspace / "config" / "memory.json").read_text(encoding="utf-8"))
        models = cfg["models"]
        assert models["llmProvider"] == "anthropic"
        assert models["deepReasoningProvider"] == "anthropic"
        assert models["fastReasoningProvider"] == "anthropic"
        assert models["deepReasoning"] == "claude-sonnet-4-6"
        assert models["fastReasoning"] == "claude-haiku-4-5-20251001"
        assert cfg["capture"]["chunkTokens"] == 8000

    def test_api_workspace_uses_requested_extraction_model_for_deep_tier(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        quaid_dir = tmp_path / "modules" / "quaid"
        quaid_dir.mkdir(parents=True)
        (quaid_dir / "schema.sql").write_text("CREATE TABLE test(id INTEGER);", encoding="utf-8")
        (quaid_dir / "config").mkdir(parents=True)
        (quaid_dir / "config" / "memory.json").write_text(
            json.dumps(
                {
                    "models": {
                        "llmProvider": "anthropic",
                        "deepReasoning": "claude-haiku-4-5-20251001",
                        "fastReasoning": "claude-haiku-4-5-20251001",
                    }
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(rpb, "_QUAID_DIR", quaid_dir)
        monkeypatch.setattr(rpb, "_BACKEND", "oauth")
        monkeypatch.setattr(rpb, "_bootstrap_domain_registry", lambda conn: None)
        monkeypatch.setattr(rpb, "_load_active_domains", lambda workspace: [])
        monkeypatch.delenv("BENCHMARK_REASONING_MODEL", raising=False)
        monkeypatch.delenv("BENCHMARK_DEEP_REASONING_MODEL", raising=False)
        monkeypatch.delenv("BENCHMARK_FAST_REASONING_MODEL", raising=False)

        rpb.setup_workspace(workspace, extraction_model="claude-sonnet-4-6")

        cfg = json.loads((workspace / "config" / "memory.json").read_text(encoding="utf-8"))
        models = cfg["models"]
        assert models["llmProvider"] == "anthropic"
        assert models["deepReasoningProvider"] == "anthropic"
        assert models["fastReasoningProvider"] == "anthropic"
        assert models["deepReasoning"] == "claude-sonnet-4-6"
        assert models["fastReasoning"] == "claude-haiku-4-5-20251001"

    def test_workspace_sets_2000_token_caps_for_core_markdown(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        quaid_dir = tmp_path / "modules" / "quaid"
        quaid_dir.mkdir(parents=True)
        (quaid_dir / "schema.sql").write_text("CREATE TABLE test(id INTEGER);", encoding="utf-8")
        (quaid_dir / "config").mkdir(parents=True)
        (quaid_dir / "config" / "memory.json").write_text(json.dumps({}), encoding="utf-8")
        monkeypatch.setattr(rpb, "_QUAID_DIR", quaid_dir)
        monkeypatch.setattr(rpb, "_BACKEND", "oauth")
        monkeypatch.setattr(rpb, "_bootstrap_domain_registry", lambda conn: None)
        monkeypatch.setattr(rpb, "_load_active_domains", lambda workspace: [])

        rpb.setup_workspace(workspace)

        cfg = json.loads((workspace / "config" / "memory.json").read_text(encoding="utf-8"))
        files = cfg["docs"]["coreMarkdown"]["files"]
        assert files["SOUL.md"]["maxTokens"] == 2000
        assert files["USER.md"]["maxTokens"] == 2000
        assert files["ENVIRONMENT.md"]["maxTokens"] == 2000

    def test_workspace_removes_legacy_retrieval_notify_key(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        quaid_dir = tmp_path / "modules" / "quaid"
        quaid_dir.mkdir(parents=True)
        (quaid_dir / "schema.sql").write_text("CREATE TABLE test(id INTEGER);", encoding="utf-8")
        (quaid_dir / "config").mkdir(parents=True)
        (quaid_dir / "config" / "memory.json").write_text(
            json.dumps({"retrieval": {"notifyOnRecall": True, "notify_on_recall": True}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(rpb, "_QUAID_DIR", quaid_dir)
        monkeypatch.setattr(rpb, "_BACKEND", "oauth")
        monkeypatch.setattr(rpb, "_bootstrap_domain_registry", lambda conn: None)
        monkeypatch.setattr(rpb, "_load_active_domains", lambda workspace: [])

        rpb.setup_workspace(workspace)

        cfg = json.loads((workspace / "config" / "memory.json").read_text(encoding="utf-8"))
        retrieval = cfg["retrieval"]
        assert "notifyOnRecall" not in retrieval
        assert "notify_on_recall" not in retrieval

    def test_workspace_writes_separate_embedding_workers(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        quaid_dir = tmp_path / "modules" / "quaid"
        quaid_dir.mkdir(parents=True)
        (quaid_dir / "schema.sql").write_text("CREATE TABLE test(id INTEGER);", encoding="utf-8")
        (quaid_dir / "config").mkdir(parents=True)
        (quaid_dir / "config" / "memory.json").write_text(json.dumps({}), encoding="utf-8")
        monkeypatch.setattr(rpb, "_QUAID_DIR", quaid_dir)
        monkeypatch.setattr(rpb, "_BACKEND", "oauth")
        monkeypatch.setattr(rpb, "_bootstrap_domain_registry", lambda conn: None)
        monkeypatch.setattr(rpb, "_load_active_domains", lambda workspace: [])
        monkeypatch.setenv("BENCHMARK_JANITOR_LLM_WORKERS", "6")
        monkeypatch.setenv("BENCHMARK_EMBEDDING_WORKERS", "2")

        rpb.setup_workspace(workspace)

        cfg = json.loads((workspace / "config" / "memory.json").read_text(encoding="utf-8"))
        parallel = cfg["core"]["parallel"]
        assert parallel["llmWorkers"] == 6
        assert parallel["embeddingWorkers"] == 2

    def test_workspace_applies_embedding_backend_overrides(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        quaid_dir = tmp_path / "modules" / "quaid"
        quaid_dir.mkdir(parents=True)
        (quaid_dir / "schema.sql").write_text("CREATE TABLE test(id INTEGER);", encoding="utf-8")
        (quaid_dir / "config").mkdir(parents=True)
        (quaid_dir / "config" / "memory.json").write_text(json.dumps({}), encoding="utf-8")
        monkeypatch.setattr(rpb, "_QUAID_DIR", quaid_dir)
        monkeypatch.setattr(rpb, "_BACKEND", "oauth")
        monkeypatch.setattr(rpb, "_bootstrap_domain_registry", lambda conn: None)
        monkeypatch.setattr(rpb, "_load_active_domains", lambda workspace: [])
        monkeypatch.setenv("BENCHMARK_EMBEDDINGS_PROVIDER", "ollama")
        monkeypatch.setenv("BENCHMARK_OLLAMA_URL", "http://127.0.0.1:11434")
        monkeypatch.setenv("BENCHMARK_EMBEDDING_MODEL", "nomic-embed-text")
        monkeypatch.setenv("BENCHMARK_EMBEDDING_DIM", "768")

        rpb.setup_workspace(workspace)

        cfg = json.loads((workspace / "config" / "memory.json").read_text(encoding="utf-8"))
        assert cfg["models"]["embeddingsProvider"] == "ollama"
        assert cfg["ollama"]["url"] == "http://127.0.0.1:11434"
        assert cfg["ollama"]["embeddingModel"] == "nomic-embed-text"
        assert cfg["ollama"]["embeddingDim"] == 768

    def test_vllm_workspace_sets_openai_compatible_reasoning_config(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        quaid_dir = tmp_path / "modules" / "quaid"
        quaid_dir.mkdir(parents=True)
        (quaid_dir / "schema.sql").write_text("CREATE TABLE test(id INTEGER);", encoding="utf-8")
        (quaid_dir / "config").mkdir(parents=True)
        (quaid_dir / "config" / "memory.json").write_text(json.dumps({}), encoding="utf-8")
        monkeypatch.setattr(rpb, "_QUAID_DIR", quaid_dir)
        monkeypatch.setattr(rpb, "_BACKEND", "vllm")
        monkeypatch.setattr(rpb, "_OPENAI_COMPAT_URL", "http://spark:8000")
        monkeypatch.setattr(rpb, "_OPENAI_COMPAT_API_KEY_ENV", "BENCHMARK_VLLM_API_KEY")
        monkeypatch.setattr(rpb, "_bootstrap_domain_registry", lambda conn: None)
        monkeypatch.setattr(rpb, "_load_active_domains", lambda workspace: [])
        monkeypatch.delenv("BENCHMARK_REASONING_MODEL", raising=False)
        monkeypatch.setenv("BENCHMARK_DEEP_REASONING_MODEL", "gemma-3-31b-it")
        monkeypatch.setenv("BENCHMARK_FAST_REASONING_MODEL", "gemma-3-31b-it")

        rpb.setup_workspace(workspace)

        cfg = json.loads((workspace / "config" / "memory.json").read_text(encoding="utf-8"))
        models = cfg["models"]
        assert models["llmProvider"] == "openai-compatible"
        assert models["deepReasoningProvider"] == "openai-compatible"
        assert models["fastReasoningProvider"] == "openai-compatible"
        assert models["deepReasoning"] == "gemma-3-31b-it"
        assert models["fastReasoning"] == "gemma-3-31b-it"
        assert models["baseUrl"] == "http://spark:8000"
        assert models["apiKeyEnv"] == "BENCHMARK_VLLM_API_KEY"

    def test_llama_cpp_workspace_sets_openai_compatible_reasoning_config(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        quaid_dir = tmp_path / "modules" / "quaid"
        quaid_dir.mkdir(parents=True)
        (quaid_dir / "schema.sql").write_text("CREATE TABLE test(id INTEGER);", encoding="utf-8")
        (quaid_dir / "config").mkdir(parents=True)
        (quaid_dir / "config" / "memory.json").write_text(json.dumps({}), encoding="utf-8")
        monkeypatch.setattr(rpb, "_QUAID_DIR", quaid_dir)
        monkeypatch.setattr(rpb, "_BACKEND", "llama-cpp")
        monkeypatch.setattr(rpb, "_OPENAI_COMPAT_URL", "http://spark:8080")
        monkeypatch.setattr(rpb, "_OPENAI_COMPAT_API_KEY_ENV", "BENCHMARK_LLAMA_CPP_API_KEY")
        monkeypatch.setattr(rpb, "_bootstrap_domain_registry", lambda conn: None)
        monkeypatch.setattr(rpb, "_load_active_domains", lambda workspace: [])
        monkeypatch.delenv("BENCHMARK_REASONING_MODEL", raising=False)
        monkeypatch.delenv("BENCHMARK_LLAMA_CPP_RUNTIME_URL", raising=False)
        monkeypatch.delenv("BENCHMARK_LLAMA_CPP_RUNTIME_MODEL", raising=False)
        monkeypatch.delenv("BENCHMARK_LLAMA_CPP_RUNTIME_API_KEY", raising=False)
        monkeypatch.delenv("BENCHMARK_LLAMA_CPP_RUNTIME_API_KEY_ENV", raising=False)
        monkeypatch.setenv("BENCHMARK_DEEP_REASONING_MODEL", "gemma-3-31b-it")
        monkeypatch.setenv("BENCHMARK_FAST_REASONING_MODEL", "gemma-3-31b-it")

        rpb.setup_workspace(workspace)

        cfg = json.loads((workspace / "config" / "memory.json").read_text(encoding="utf-8"))
        models = cfg["models"]
        assert models["llmProvider"] == "openai-compatible"
        assert models["deepReasoningProvider"] == "openai-compatible"
        assert models["fastReasoningProvider"] == "openai-compatible"
        assert models["deepReasoning"] == "gemma-3-31b-it"
        assert models["fastReasoning"] == "gemma-3-31b-it"
        assert models["baseUrl"] == "http://spark:8080"
        assert models["apiKeyEnv"] == "BENCHMARK_LLAMA_CPP_API_KEY"

    def test_openai_workspace_sets_direct_openai_reasoning_config(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        quaid_dir = tmp_path / "modules" / "quaid"
        quaid_dir.mkdir(parents=True)
        (quaid_dir / "schema.sql").write_text("CREATE TABLE test(id INTEGER);", encoding="utf-8")
        (quaid_dir / "config").mkdir(parents=True)
        (quaid_dir / "config" / "memory.json").write_text(json.dumps({}), encoding="utf-8")
        monkeypatch.setattr(rpb, "_QUAID_DIR", quaid_dir)
        monkeypatch.setattr(rpb, "_BACKEND", "openai")
        monkeypatch.setattr(rpb, "_OPENAI_COMPAT_URL", "https://api.openai.com")
        monkeypatch.setattr(rpb, "_OPENAI_COMPAT_MODEL", "gpt-5.4")
        monkeypatch.setattr(rpb, "_OPENAI_COMPAT_API_KEY_ENV", "OPENAI_API_KEY")
        monkeypatch.setattr(rpb, "_bootstrap_domain_registry", lambda conn: None)
        monkeypatch.setattr(rpb, "_load_active_domains", lambda workspace: [])
        monkeypatch.delenv("BENCHMARK_REASONING_MODEL", raising=False)
        monkeypatch.setenv("BENCHMARK_DEEP_REASONING_MODEL", "gpt-5.4")
        monkeypatch.setenv("BENCHMARK_FAST_REASONING_MODEL", "gpt-5.4-mini")

        rpb.setup_workspace(workspace)

        cfg = json.loads((workspace / "config" / "memory.json").read_text(encoding="utf-8"))
        models = cfg["models"]
        assert models["llmProvider"] == "openai-compatible"
        assert models["deepReasoningProvider"] == "openai-compatible"
        assert models["fastReasoningProvider"] == "openai-compatible"
        assert models["deepReasoning"] == "gpt-5.4"
        assert models["fastReasoning"] == "gpt-5.4-mini"
        assert models["baseUrl"] == "https://api.openai.com"
        assert models["apiKeyEnv"] == "OPENAI_API_KEY"

    def test_llama_cpp_workspace_prefers_runtime_endpoint_for_reasoning(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        quaid_dir = tmp_path / "modules" / "quaid"
        quaid_dir.mkdir(parents=True)
        (quaid_dir / "schema.sql").write_text("CREATE TABLE test(id INTEGER);", encoding="utf-8")
        (quaid_dir / "config").mkdir(parents=True)
        (quaid_dir / "config" / "memory.json").write_text(json.dumps({}), encoding="utf-8")
        monkeypatch.setattr(rpb, "_QUAID_DIR", quaid_dir)
        monkeypatch.setattr(rpb, "_BACKEND", "llama-cpp")
        monkeypatch.setattr(rpb, "_OPENAI_COMPAT_URL", "http://answer:30002")
        monkeypatch.setattr(rpb, "_OPENAI_COMPAT_API_KEY_ENV", "BENCHMARK_LLAMA_CPP_API_KEY")
        monkeypatch.setenv("BENCHMARK_LLAMA_CPP_RUNTIME_URL", "http://runtime:30001")
        monkeypatch.setenv("BENCHMARK_LLAMA_CPP_RUNTIME_MODEL", "gemma-4-26b-q6k")
        monkeypatch.setenv("BENCHMARK_LLAMA_CPP_RUNTIME_API_KEY", "runtime-key")
        monkeypatch.setattr(rpb, "_bootstrap_domain_registry", lambda conn: None)
        monkeypatch.setattr(rpb, "_load_active_domains", lambda workspace: [])
        monkeypatch.delenv("BENCHMARK_REASONING_MODEL", raising=False)
        monkeypatch.delenv("BENCHMARK_DEEP_REASONING_MODEL", raising=False)
        monkeypatch.delenv("BENCHMARK_FAST_REASONING_MODEL", raising=False)

        rpb.setup_workspace(workspace)

        cfg = json.loads((workspace / "config" / "memory.json").read_text(encoding="utf-8"))
        models = cfg["models"]
        assert models["deepReasoning"] == "gemma-4-26b-q6k"
        assert models["fastReasoning"] == "gemma-4-26b-q6k"
        assert models["baseUrl"] == "http://runtime:30001"
        assert models["apiKeyEnv"] == "BENCHMARK_LLAMA_CPP_RUNTIME_API_KEY"

    def test_normalize_workspace_runtime_config_rewrites_existing_eval_workspace_for_llama_cpp(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        (workspace / "config").mkdir(parents=True)
        (workspace / "config" / "memory.json").write_text(
            json.dumps(
                {
                    "models": {
                        "llmProvider": "anthropic",
                        "deepReasoningProvider": "anthropic",
                        "fastReasoningProvider": "anthropic",
                        "deepReasoning": "claude-sonnet-4-6",
                        "fastReasoning": "claude-haiku-4-5-20251001",
                    },
                    "retrieval": {
                        "notifyOnRecall": True,
                        "notify_on_recall": True,
                    },
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(rpb, "_BACKEND", "llama-cpp")
        monkeypatch.setattr(rpb, "_OPENAI_COMPAT_URL", "http://spark:8080")
        monkeypatch.setattr(rpb, "_OPENAI_COMPAT_API_KEY_ENV", "BENCHMARK_LLAMA_CPP_API_KEY")

        rpb._normalize_workspace_runtime_config(workspace, requested_model="gemma-4-31b-it")

        cfg = json.loads((workspace / "config" / "memory.json").read_text(encoding="utf-8"))
        models = cfg["models"]
        assert models["llmProvider"] == "openai-compatible"
        assert models["deepReasoningProvider"] == "openai-compatible"
        assert models["fastReasoningProvider"] == "openai-compatible"
        assert models["deepReasoning"] == "gemma-4-31b-it"
        assert models["fastReasoning"] == "gemma-4-31b-it"
        assert models["baseUrl"] == "http://spark:8080"
        assert models["apiKeyEnv"] == "BENCHMARK_LLAMA_CPP_API_KEY"
        assert cfg["retrieval"] == {}
        assert (workspace / rpb._BENCHMARK_QUAID_INSTANCE / "config" / "memory.json").exists()

    def test_normalize_workspace_runtime_config_preserves_embedding_contract_for_reused_lineage(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        (workspace / "config").mkdir(parents=True)
        (workspace / "config" / "memory.json").write_text(
            json.dumps(
                {
                    "models": {
                        "llmProvider": "anthropic",
                        "deepReasoningProvider": "anthropic",
                        "fastReasoningProvider": "anthropic",
                    },
                    "retrieval": {
                        "notifyOnRecall": True,
                    },
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(rpb, "_BACKEND", "llama-cpp")
        monkeypatch.setattr(rpb, "_OPENAI_COMPAT_URL", "http://spark:8080")
        monkeypatch.setattr(rpb, "_OPENAI_COMPAT_API_KEY_ENV", "BENCHMARK_LLAMA_CPP_API_KEY")
        monkeypatch.setenv("BENCHMARK_EMBEDDINGS_PROVIDER", "ollama")
        monkeypatch.setenv("BENCHMARK_OLLAMA_URL", "http://127.0.0.1:11434")
        monkeypatch.setenv("BENCHMARK_EMBEDDING_MODEL", "nomic-embed-text")
        monkeypatch.setenv("BENCHMARK_EMBEDDING_DIM", "768")

        rpb._normalize_workspace_runtime_config(workspace, requested_model="gemma-4-31b-it")

        cfg = json.loads((workspace / "config" / "memory.json").read_text(encoding="utf-8"))
        assert cfg["models"]["embeddingsProvider"] == "ollama"
        assert cfg["ollama"]["url"] == "http://127.0.0.1:11434"
        assert cfg["ollama"]["embeddingModel"] == "nomic-embed-text"
        assert cfg["ollama"]["embeddingDim"] == 768
        instance_cfg = json.loads(
            (workspace / rpb._BENCHMARK_QUAID_INSTANCE / "config" / "memory.json").read_text(encoding="utf-8")
        )
        assert instance_cfg["ollama"]["embeddingModel"] == "nomic-embed-text"
        assert instance_cfg["ollama"]["embeddingDim"] == 768


def test_resolve_judge_provider_prefers_openai_for_gpt(monkeypatch):
    monkeypatch.setattr(rpb, "_BACKEND", "llama-cpp")
    assert rpb._resolve_judge_provider("gpt-4o-mini") == "openai"


def test_resolve_judge_provider_uses_openai_compatible_for_served_model(monkeypatch):
    monkeypatch.setattr(rpb, "_BACKEND", "llama-cpp")
    monkeypatch.setattr(rpb, "_OPENAI_COMPAT_MODEL", "gemma-4-31b-q8")
    monkeypatch.setattr(rpb, "_OPENAI_COMPAT_JUDGE_MODEL", "gemma-4-31b-q8")
    assert rpb._resolve_judge_provider("gemma-4-31b-q8") == "openai-compatible"


def test_resolve_judge_provider_uses_openai_compatible_for_separate_judge_model(monkeypatch):
    monkeypatch.setattr(rpb, "_BACKEND", "llama-cpp")
    monkeypatch.setattr(rpb, "_OPENAI_COMPAT_MODEL", "gemma-4-31b-q8")
    monkeypatch.setattr(rpb, "_OPENAI_COMPAT_JUDGE_MODEL", "gemma-4-26b-q6k")
    assert rpb._resolve_judge_provider("gemma-4-26b-q6k") == "openai-compatible"


def test_judge_with_prompt_routes_to_openai_compatible(monkeypatch):
    seen = {}

    def _fake_openai_compatible(prompt, model="unused", workspace=None):
        seen["model"] = model
        seen["prompt"] = prompt
        return "CORRECT", 1.0

    monkeypatch.setattr(rpb, "_BACKEND", "llama-cpp")
    monkeypatch.setattr(rpb, "_OPENAI_COMPAT_MODEL", "gemma-4-31b-q8")
    monkeypatch.setattr(rpb, "_OPENAI_COMPAT_JUDGE_MODEL", "gemma-4-31b-q8")
    monkeypatch.setattr(rpb, "_judge_openai_compatible", _fake_openai_compatible)

    label, score = rpb._judge_with_prompt("prompt", "unused", judge_model="gemma-4-31b-q8")

    assert (label, score) == ("CORRECT", 1.0)
    assert seen["model"] == "gemma-4-31b-q8"
    assert "STRICT JSON ONLY" in seen["prompt"]


def test_judge_openai_compatible_uses_compact_strict_cap(monkeypatch):
    seen = {}

    def _fake_call_openai_compatible_chat(**kwargs):
        seen.update(kwargs)
        return {"choices": [{"message": {"content": "{\"label\":\"CORRECT\"}"}}], "model": "gemma"}, {
            "output_tokens": 5,
            "input_tokens": 10,
            "api_calls": 1,
            "model_usage": {"gemma": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}},
        }

    monkeypatch.setattr(rpb, "_call_openai_compatible_chat", _fake_call_openai_compatible_chat)
    monkeypatch.setattr(rpb, "_extract_openai_response_text", lambda data: data["choices"][0]["message"]["content"])
    monkeypatch.setattr(rpb, "_openai_compatible_answer_timeout_s", lambda: None)
    monkeypatch.setattr(rpb, "_openai_compatible_backend_label", lambda: "llama-cpp")

    label, score = rpb._judge_openai_compatible("prompt", model="gemma-4-31b-q8", workspace=None)

    assert (label, score) == ("CORRECT", 1.0)
    assert seen["max_tokens"] == 24
    assert seen["source"] == "judge"
    assert seen["chat_template_kwargs"] == {"enable_thinking": False}


def test_tier5_judge_chat_template_kwargs_default_auto(monkeypatch):
    monkeypatch.delenv("BENCHMARK_TIER5_JUDGE_THINKING", raising=False)
    monkeypatch.setattr(rpb, "_uses_openai_compatible_backend", lambda: False)

    assert rpb._tier5_judge_chat_template_kwargs() is None


def test_tier5_judge_chat_template_kwargs_default_auto_openai_compatible(monkeypatch):
    monkeypatch.delenv("BENCHMARK_TIER5_JUDGE_THINKING", raising=False)
    monkeypatch.setattr(rpb, "_uses_openai_compatible_backend", lambda: True)

    assert rpb._tier5_judge_chat_template_kwargs() == {"enable_thinking": False}


def test_tier5_judge_chat_template_kwargs_off(monkeypatch):
    monkeypatch.setenv("BENCHMARK_TIER5_JUDGE_THINKING", "off")

    assert rpb._tier5_judge_chat_template_kwargs() == {"enable_thinking": False}


def test_tier5_judge_chat_template_kwargs_on(monkeypatch):
    monkeypatch.setenv("BENCHMARK_TIER5_JUDGE_THINKING", "on")

    assert rpb._tier5_judge_chat_template_kwargs() == {"enable_thinking": True}


def test_judge_tier5_openai_compatible_forwards_thinking_override(monkeypatch):
    seen = {}

    def _fake_call_openai_compatible_chat(**kwargs):
        seen.update(kwargs)
        return {"choices": [{"message": {"content": "{\"score\":2}"}}], "model": "gemma"}, {}

    monkeypatch.setenv("BENCHMARK_TIER5_JUDGE_THINKING", "off")
    monkeypatch.setattr(rpb, "_resolve_judge_provider", lambda _model: "openai-compatible")
    monkeypatch.setattr(rpb, "_call_openai_compatible_chat", _fake_call_openai_compatible_chat)
    monkeypatch.setattr(rpb, "_extract_openai_judge_text", lambda data: data["choices"][0]["message"]["content"])
    monkeypatch.setattr(rpb, "_openai_compatible_answer_timeout_s", lambda: None)
    monkeypatch.setattr(rpb, "_openai_compatible_backend_label", lambda: "llama-cpp")

    score, reasoning = rpb._judge_tier5(
        query={
            "question": "q",
            "sensitivity_context": "ctx",
            "rubric": {"score_2": "a", "score_1": "b", "score_0": "c"},
        },
        prediction="pred",
        api_key="unused",
        judge_model="gemma-4-26b-q6k",
        workspace=None,
    )

    assert score == 2
    assert seen["source"] == "tier5_judge"
    assert seen["chat_template_kwargs"] == {"enable_thinking": False}


def test_judge_tier5_openai_compatible_falls_back_to_reasoning_content(monkeypatch):
    def _fake_call_openai_compatible_chat(**kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning_content": '{"score":2,"reasoning":"aware"}',
                    }
                }
            ],
            "model": "gemma-4-26b-q6k",
        }, {}

    monkeypatch.delenv("BENCHMARK_TIER5_JUDGE_THINKING", raising=False)
    monkeypatch.setattr(rpb, "_resolve_judge_provider", lambda _model: "openai-compatible")
    monkeypatch.setattr(rpb, "_uses_openai_compatible_backend", lambda: True)
    monkeypatch.setattr(rpb, "_call_openai_compatible_chat", _fake_call_openai_compatible_chat)
    monkeypatch.setattr(rpb, "_openai_compatible_answer_timeout_s", lambda: None)
    monkeypatch.setattr(rpb, "_openai_compatible_backend_label", lambda: "llama-cpp")

    score, reasoning = rpb._judge_tier5(
        query={
            "question": "q",
            "sensitivity_context": "ctx",
            "rubric": {"score_2": "a", "score_1": "b", "score_0": "c"},
        },
        prediction="pred",
        api_key="unused",
        judge_model="gemma-4-26b-q6k",
        workspace=None,
    )

    assert score == 2
    assert reasoning == ""


def test_judge_tier5_openai_routes_gpt_model_to_openai(monkeypatch):
    seen = {}

    def _fake_judge_tier5_openai(query, prediction, workspace=None, model=None):
        seen["query"] = query
        seen["prediction"] = prediction
        seen["workspace"] = workspace
        seen["model"] = model
        return 2, "ok"

    def _boom(*args, **kwargs):
        raise AssertionError("anthropic path should not be used for GPT Tier 5 judge")

    monkeypatch.setattr(rpb, "_resolve_judge_provider", lambda _model: "openai")
    monkeypatch.setattr(rpb, "_judge_tier5_openai", _fake_judge_tier5_openai)
    monkeypatch.setattr(rpb, "_call_anthropic_cached", _boom)

    score, reasoning = rpb._judge_tier5(
        query={
            "question": "q",
            "sensitivity_context": "ctx",
            "rubric": {"score_2": "a", "score_1": "b", "score_0": "c"},
        },
        prediction="pred",
        api_key="unused",
        judge_model="gpt-4o-mini",
        workspace=None,
    )

    assert score == 2
    assert reasoning == "ok"
    assert seen["model"] == "gpt-4o-mini"


def test_openai_compatible_helpers_route_judge_to_separate_endpoint(monkeypatch):
    monkeypatch.setattr(rpb, "_BACKEND", "llama-cpp")
    monkeypatch.setattr(rpb, "_OPENAI_COMPAT_URL", "http://answer.local:30001")
    monkeypatch.setattr(rpb, "_OPENAI_COMPAT_MODEL", "gemma-4-31b-q8")
    monkeypatch.setattr(rpb, "_OPENAI_COMPAT_API_KEY_ENV", "BENCHMARK_LLAMA_CPP_API_KEY")
    monkeypatch.setattr(rpb, "_OPENAI_COMPAT_JUDGE_URL", "http://judge.local:30002")
    monkeypatch.setattr(rpb, "_OPENAI_COMPAT_JUDGE_MODEL", "gemma-4-26b-q6k")
    monkeypatch.setattr(rpb, "_OPENAI_COMPAT_JUDGE_API_KEY_ENV", "BENCHMARK_LLAMA_CPP_JUDGE_API_KEY")
    monkeypatch.setenv("BENCHMARK_LLAMA_CPP_API_KEY", "answer-key")
    monkeypatch.setenv("BENCHMARK_LLAMA_CPP_JUDGE_API_KEY", "judge-key")

    assert rpb._get_openai_compatible_url() == "http://answer.local:30001"
    assert rpb._get_openai_compatible_url(source="judge") == "http://judge.local:30002"
    assert rpb._get_openai_compatible_model() == "gemma-4-31b-q8"
    assert rpb._get_openai_compatible_model(source="judge") == "gemma-4-26b-q6k"
    assert rpb._get_openai_compatible_api_key() == "answer-key"
    assert rpb._get_openai_compatible_api_key(source="judge") == "judge-key"


def test_openai_compatible_helpers_route_runtime_to_separate_endpoint(monkeypatch):
    monkeypatch.setattr(rpb, "_BACKEND", "llama-cpp")
    monkeypatch.setattr(rpb, "_OPENAI_COMPAT_URL", "http://answer.local:30002")
    monkeypatch.setattr(rpb, "_OPENAI_COMPAT_MODEL", "gemma-4-31b-q8")
    monkeypatch.setattr(rpb, "_OPENAI_COMPAT_API_KEY_ENV", "BENCHMARK_LLAMA_CPP_API_KEY")
    monkeypatch.setenv("BENCHMARK_LLAMA_CPP_RUNTIME_URL", "http://runtime.local:30001")
    monkeypatch.setenv("BENCHMARK_LLAMA_CPP_RUNTIME_MODEL", "gemma-4-26b-q6k")
    monkeypatch.setenv("BENCHMARK_LLAMA_CPP_RUNTIME_API_KEY", "runtime-key")

    assert rpb._get_openai_compatible_url() == "http://answer.local:30002"
    assert rpb._get_openai_compatible_url(source="runtime") == "http://runtime.local:30001"
    assert rpb._get_openai_compatible_model() == "gemma-4-31b-q8"
    assert rpb._get_openai_compatible_model(source="runtime") == "gemma-4-26b-q6k"
    assert rpb._get_openai_compatible_api_key_env() == "BENCHMARK_LLAMA_CPP_API_KEY"
    assert (
        rpb._get_openai_compatible_api_key_env(source="runtime")
        == "BENCHMARK_LLAMA_CPP_RUNTIME_API_KEY"
    )
    assert rpb._get_openai_compatible_api_key(source="runtime") == "runtime-key"


def test_resolve_assets_dir_prefers_benchmark_assets_env(monkeypatch, tmp_path):
    benchmark_assets = tmp_path / "bench-assets"
    agentlife_assets = tmp_path / "agentlife-assets"
    benchmark_assets.mkdir()
    agentlife_assets.mkdir()
    monkeypatch.setenv("BENCHMARK_ASSETS_DIR", str(benchmark_assets))
    monkeypatch.setenv("AGENTLIFE_ASSETS_DIR", str(agentlife_assets))

    assert rpb._resolve_assets_dir() == benchmark_assets


def test_dataset_variant_jp_resolves_translated_assets_and_fillers(monkeypatch):
    monkeypatch.delenv("BENCHMARK_ASSETS_DIR", raising=False)
    monkeypatch.delenv("AGENTLIFE_ASSETS_DIR", raising=False)
    monkeypatch.delenv("BENCHMARK_FILLER_DIR", raising=False)
    monkeypatch.setenv("BENCHMARK_DATASET", "jp")

    assert rpb._resolve_assets_dir() == rpb._PROJECT_DIR / "data" / "sessions-jp"
    assert rpb._resolve_filler_dir() == rpb._PROJECT_DIR / "data" / "filler-sessions-jp"
    assert rpb._dataset_variant_label(False) == "jp"
    assert rpb._dataset_variant_label(True) == "jp+statement_grounding"
    assert rpb._dataset_identity()["display_name"] == "マヤ"
    assert rpb._dataset_identity()["speakers"] == ["マヤ", "ユーザー"]


def test_dataset_variant_jp_loads_translated_eval_corpus(monkeypatch):
    dataset_path = Path(__file__).resolve().parents[1] / "dataset.py"
    spec = importlib.util.spec_from_file_location("benchmark_dataset_jp", dataset_path)
    assert spec is not None and spec.loader is not None
    dataset = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataset)

    monkeypatch.setenv("BENCHMARK_DATASET", "jp")

    queries = dataset.get_all_eval_queries([])
    tier5 = dataset.get_tier5_queries()

    assert len(queries) == 268
    assert len(tier5) == 15
    assert queries[0]["question"].startswith("マヤ")
    assert queries[0]["ground_truth"] == "デイビッド"
    assert "Maya" not in queries[0]["question"]
    assert "David" not in queries[0]["ground_truth"]
    with pytest.raises(RuntimeError, match="JP statement-context grounding"):
        dataset.get_statement_context_queries()


def test_enforce_dataset_version_uses_variant_specific_latest_pin(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("benchmark_run_variant_gate_ok", ROOT / "run_production_benchmark.py")
    assert spec is not None and spec.loader is not None
    fresh_rpb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh_rpb)

    assets_dir = tmp_path / "sessions-jp"
    assets_dir.mkdir()
    (assets_dir / "dataset.version.json").write_text(
        json.dumps({"version": "canonical-20260313", "variant": "jp"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        fresh_rpb,
        "_load_dataset_registry",
        lambda: {
            "latest": "canonical-20260421d",
            "latest_by_variant": {
                "canonical": "canonical-20260421d",
                "jp": "canonical-20260313",
            },
            "versions": {
                "canonical-20260313": {"expected_queries": 268},
                "canonical-20260421d": {"expected_queries": 268},
            },
        },
    )

    version, expected_queries = fresh_rpb._enforce_dataset_version(assets_dir)

    assert version == "canonical-20260313"
    assert expected_queries == 268


def test_enforce_dataset_version_rejects_wrong_variant_pin(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("benchmark_run_variant_gate_bad", ROOT / "run_production_benchmark.py")
    assert spec is not None and spec.loader is not None
    fresh_rpb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh_rpb)

    assets_dir = tmp_path / "sessions-jp"
    assets_dir.mkdir()
    (assets_dir / "dataset.version.json").write_text(
        json.dumps({"version": "canonical-20260313", "variant": "jp"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        fresh_rpb,
        "_load_dataset_registry",
        lambda: {
            "latest": "canonical-20260421d",
            "latest_by_variant": {
                "canonical": "canonical-20260421d",
                "jp": "canonical-20260421d",
            },
            "versions": {
                "canonical-20260313": {"expected_queries": 268},
                "canonical-20260421d": {"expected_queries": 268},
            },
        },
    )

    with pytest.raises(RuntimeError, match="variant=jp"):
        fresh_rpb._enforce_dataset_version(assets_dir)


def test_judge_non_question_uses_openai_compatible_when_requested(monkeypatch):
    seen = {}

    def _fake_judge_with_prompt(prompt, api_key, judge_model="unused", workspace=None):
        seen["model"] = judge_model
        return "CORRECT", 1.0

    monkeypatch.setattr(rpb, "_judge_with_prompt", _fake_judge_with_prompt)

    label, score = rpb._judge_non_question("q", "gt", "pred", "unused", judge_model="gemma-4-31b-q8")

    assert (label, score) == ("CORRECT", 1.0)
    assert seen["model"] == "gemma-4-31b-q8"

    def test_known_qwen4b_embedding_dim_defaults_when_override_omitted(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        quaid_dir = tmp_path / "modules" / "quaid"
        quaid_dir.mkdir(parents=True)
        (quaid_dir / "schema.sql").write_text("CREATE TABLE test(id INTEGER);", encoding="utf-8")
        (quaid_dir / "config").mkdir(parents=True)
        (quaid_dir / "config" / "memory.json").write_text(json.dumps({}), encoding="utf-8")
        monkeypatch.setattr(rpb, "_QUAID_DIR", quaid_dir)
        monkeypatch.setattr(rpb, "_BACKEND", "oauth")
        monkeypatch.setattr(rpb, "_bootstrap_domain_registry", lambda conn: None)
        monkeypatch.setattr(rpb, "_load_active_domains", lambda workspace: [])
        monkeypatch.setenv("BENCHMARK_EMBEDDINGS_PROVIDER", "ollama")
        monkeypatch.setenv("BENCHMARK_OLLAMA_URL", "http://127.0.0.1:11434")
        monkeypatch.setenv("BENCHMARK_EMBEDDING_MODEL", "qwen3-embedding:4b")
        monkeypatch.delenv("BENCHMARK_EMBEDDING_DIM", raising=False)

        rpb.setup_workspace(workspace)

        cfg = json.loads((workspace / "config" / "memory.json").read_text(encoding="utf-8"))
        assert cfg["ollama"]["embeddingModel"] == "qwen3-embedding:4b"
        assert cfg["ollama"]["embeddingDim"] == 2560

    def test_known_embedding_dim_mismatch_fails_early(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        quaid_dir = tmp_path / "modules" / "quaid"
        quaid_dir.mkdir(parents=True)
        (quaid_dir / "schema.sql").write_text("CREATE TABLE test(id INTEGER);", encoding="utf-8")
        (quaid_dir / "config").mkdir(parents=True)
        (quaid_dir / "config" / "memory.json").write_text(json.dumps({}), encoding="utf-8")
        monkeypatch.setattr(rpb, "_QUAID_DIR", quaid_dir)
        monkeypatch.setattr(rpb, "_BACKEND", "oauth")
        monkeypatch.setattr(rpb, "_bootstrap_domain_registry", lambda conn: None)
        monkeypatch.setattr(rpb, "_load_active_domains", lambda workspace: [])
        monkeypatch.setenv("BENCHMARK_EMBEDDINGS_PROVIDER", "ollama")
        monkeypatch.setenv("BENCHMARK_OLLAMA_URL", "http://127.0.0.1:11434")
        monkeypatch.setenv("BENCHMARK_EMBEDDING_MODEL", "qwen3-embedding:4b")
        monkeypatch.setenv("BENCHMARK_EMBEDDING_DIM", "2048")

        with pytest.raises(RuntimeError, match="canonical dimension.*2560.*2048"):
            rpb.setup_workspace(workspace)


class TestRequireProjectSourceRepo:
    def test_raises_when_missing(self):
        with pytest.raises(RuntimeError, match="not found"):
            rpb._require_project_source_repo("portfolio-site", None)

    def test_allows_non_git_repo_path(self, tmp_path):
        repo = tmp_path / "portfolio-site"
        repo.mkdir()
        out = rpb._require_project_source_repo("portfolio-site", repo)
        assert out == repo

    def test_accepts_valid_git_repo(self, tmp_path):
        repo = tmp_path / "portfolio-site"
        (repo / ".git").mkdir(parents=True)
        out = rpb._require_project_source_repo("portfolio-site", repo)
        assert out == repo


class TestResolveProjectSessionSnapshot:
    def test_snapshot_found_in_sessions_dir(self, tmp_path, monkeypatch):
        assets = tmp_path / "assets"
        snap = assets / "projects" / "portfolio-site" / "sessions" / "session-09"
        snap.mkdir(parents=True)
        monkeypatch.setattr(rpb, "_resolve_assets_dir", lambda: assets)
        out = rpb._resolve_project_session_snapshot("portfolio-site", 9)
        assert out == snap

    def test_snapshot_missing_returns_none(self, tmp_path, monkeypatch):
        assets = tmp_path / "assets"
        assets.mkdir(parents=True)
        monkeypatch.setattr(rpb, "_resolve_assets_dir", lambda: assets)
        out = rpb._resolve_project_session_snapshot("recipe-app", 3)
        assert out is None


class TestTier5ScoreFolding:
    def test_merge_tier5_into_scores_uses_weighted_partial(self):
        base_scores = {
            "overall": {
                "count": 4,
                "scored": 4,
                "accuracy": 62.5,
                "correct": 2,
                "partial": 1,
                "wrong": 1,
                "error": 0,
            },
            "per_type": {},
            "per_difficulty": {},
            "per_session": {},
            "retrieval": None,
        }
        # EI: one full hit, one partial, one miss => 1.5/3 = 50%
        tier5_results = [{"ei_score": 2}, {"ei_score": 1}, {"ei_score": 0}]

        merged = rpb._merge_tier5_into_scores(base_scores, tier5_results)

        assert merged["tier5"]["correct"] == 1
        assert merged["tier5"]["partial"] == 1
        assert merged["tier5"]["wrong"] == 1
        assert merged["tier5"]["accuracy"] == 50.0

        # Combined: (2+1) correct, (1+1) partial, (1+1) wrong over 7 scored
        # points = 3 + 0.5*2 = 4.0 => 57.14%
        assert merged["overall"]["scored"] == 7
        assert merged["overall"]["correct"] == 3
        assert merged["overall"]["partial"] == 2
        assert merged["overall"]["wrong"] == 2
        assert merged["overall"]["accuracy"] == 57.14
        assert merged["overall_t1_t4"]["accuracy"] == 62.5


class TestMainTier5Auto:
    def test_eval_runs_tier5_without_flag(self, tmp_path, monkeypatch):
        workspace = tmp_path / "run"
        (workspace / "data").mkdir(parents=True)
        (workspace / "data" / "memory.db").write_text("")

        called = {"tier5": 0, "eval": 0}

        def _fake_run_eval(*_a, **_k):
            called["eval"] += 1
            return []

        def _fake_run_tier5(*_a, **_k):
            called["tier5"] += 1
            return []

        monkeypatch.setattr(rpb, "run_eval", _fake_run_eval)
        monkeypatch.setattr(rpb, "run_tier5_eval", _fake_run_tier5)
        monkeypatch.setattr(rpb, "_save_token_usage", lambda *_a, **_k: None)
        monkeypatch.setattr(
            rpb,
            "score_results",
            lambda _results: {
                "overall": {"accuracy": 0.0, "count": 0, "scored": 0, "correct": 0, "partial": 0, "wrong": 0, "error": 0},
                "per_type": {},
                "per_difficulty": {},
            },
        )
        monkeypatch.setattr(sys, "argv", [
            "run_production_benchmark.py",
            "--mode", "eval",
            "--results-dir", str(workspace),
            "--backend", "claude-code",
        ])

        rpb.main()
        assert called["eval"] == 1
        assert called["tier5"] == 1

    def test_eval_can_skip_tier5(self, tmp_path, monkeypatch):
        workspace = tmp_path / "run"
        (workspace / "data").mkdir(parents=True)
        (workspace / "data" / "memory.db").write_text("")

        called = {"tier5": 0, "eval": 0}

        def _fake_run_eval(*_a, **_k):
            called["eval"] += 1
            return []

        def _fake_run_tier5(*_a, **_k):
            called["tier5"] += 1
            return []

        monkeypatch.setattr(rpb, "run_eval", _fake_run_eval)
        monkeypatch.setattr(rpb, "run_tier5_eval", _fake_run_tier5)
        monkeypatch.setattr(rpb, "_save_token_usage", lambda *_a, **_k: None)
        monkeypatch.setattr(
            rpb,
            "score_results",
            lambda _results: {
                "overall": {"accuracy": 0.0, "count": 0, "scored": 0, "correct": 0, "partial": 0, "wrong": 0, "error": 0},
                "per_type": {},
                "per_difficulty": {},
            },
        )
        monkeypatch.setattr(sys, "argv", [
            "run_production_benchmark.py",
            "--mode", "eval",
            "--results-dir", str(workspace),
            "--backend", "claude-code",
            "--skip-tier5",
        ])

        rpb.main()
        assert called["eval"] == 1
        assert called["tier5"] == 0

    def test_fc_runs_tier5_by_default(self, tmp_path, monkeypatch):
        workspace = tmp_path / "run"
        called = {"fc": 0, "tier5": 0}

        monkeypatch.setattr(rpb, "_get_api_key", lambda: "test-key")
        monkeypatch.setattr(
            rpb,
            "run_fc_baseline",
            lambda *_a, **_k: called.__setitem__("fc", called["fc"] + 1) or [],
        )
        monkeypatch.setattr(
            rpb,
            "run_tier5_fc_baseline",
            lambda *_a, **_k: called.__setitem__("tier5", called["tier5"] + 1) or [],
        )
        monkeypatch.setattr(rpb, "_save_token_usage", lambda *_a, **_k: None)
        monkeypatch.setattr(
            rpb,
            "score_results",
            lambda _results: {
                "overall": {"accuracy": 0.0, "count": 0, "scored": 0, "correct": 0, "partial": 0, "wrong": 0, "error": 0},
                "per_type": {},
                "per_difficulty": {},
            },
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_production_benchmark.py",
                "--mode", "fc",
                "--results-dir", str(workspace),
                "--backend", "oauth",
                "--fc-models", "claude-haiku-4-5-20251001",
            ],
        )

        rpb.main()
        assert called["fc"] == 1
        assert called["tier5"] == 1

    def test_fc_can_skip_tier5(self, tmp_path, monkeypatch):
        workspace = tmp_path / "run"
        called = {"fc": 0, "tier5": 0}

        monkeypatch.setattr(rpb, "_get_api_key", lambda: "test-key")
        monkeypatch.setattr(
            rpb,
            "run_fc_baseline",
            lambda *_a, **_k: called.__setitem__("fc", called["fc"] + 1) or [],
        )
        monkeypatch.setattr(
            rpb,
            "run_tier5_fc_baseline",
            lambda *_a, **_k: called.__setitem__("tier5", called["tier5"] + 1) or [],
        )
        monkeypatch.setattr(rpb, "_save_token_usage", lambda *_a, **_k: None)
        monkeypatch.setattr(
            rpb,
            "score_results",
            lambda _results: {
                "overall": {"accuracy": 0.0, "count": 0, "scored": 0, "correct": 0, "partial": 0, "wrong": 0, "error": 0},
                "per_type": {},
                "per_difficulty": {},
            },
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_production_benchmark.py",
                "--mode", "fc",
                "--results-dir", str(workspace),
                "--backend", "oauth",
                "--fc-models", "claude-haiku-4-5-20251001",
                "--skip-tier5",
            ],
        )

        rpb.main()
        assert called["fc"] == 1
        assert called["tier5"] == 0

    def test_eval_sets_tier5_judge_thinking_env(self, tmp_path, monkeypatch):
        workspace = tmp_path / "run"
        (workspace / "data").mkdir(parents=True)
        (workspace / "data" / "memory.db").write_text("")

        monkeypatch.setattr(rpb, "run_eval", lambda *_a, **_k: [])
        monkeypatch.setattr(rpb, "run_tier5_eval", lambda *_a, **_k: [])
        monkeypatch.setattr(rpb, "_save_token_usage", lambda *_a, **_k: None)
        monkeypatch.setattr(
            rpb,
            "score_results",
            lambda _results: {
                "overall": {"accuracy": 0.0, "count": 0, "scored": 0, "correct": 0, "partial": 0, "wrong": 0, "error": 0},
                "per_type": {},
                "per_difficulty": {},
            },
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_production_benchmark.py",
                "--mode", "eval",
                "--results-dir", str(workspace),
                "--backend", "claude-code",
                "--tier5-judge-thinking", "off",
            ],
        )

        rpb.main()
        assert os.environ["BENCHMARK_TIER5_JUDGE_THINKING"] == "off"


class TestMainContextInjectDefault:
    def test_context_inject_defaults_on(self, tmp_path, monkeypatch):
        workspace = tmp_path / "run"
        (workspace / "data").mkdir(parents=True)
        (workspace / "data" / "memory.db").write_text("")

        seen = {"eval": None, "tier5": None}

        def _fake_run_eval(*_a, **kwargs):
            seen["eval"] = kwargs.get("context_inject")
            return []

        def _fake_run_tier5(*_a, **kwargs):
            seen["tier5"] = kwargs.get("context_inject")
            return []

        monkeypatch.setattr(rpb, "run_eval", _fake_run_eval)
        monkeypatch.setattr(rpb, "run_tier5_eval", _fake_run_tier5)
        monkeypatch.setattr(rpb, "_save_token_usage", lambda *_a, **_k: None)
        monkeypatch.setattr(
            rpb,
            "score_results",
            lambda _results: {
                "overall": {"accuracy": 0.0, "count": 0, "scored": 0, "correct": 0, "partial": 0, "wrong": 0, "error": 0},
                "per_type": {},
                "per_difficulty": {},
            },
        )
        monkeypatch.setattr(sys, "argv", [
            "run_production_benchmark.py",
            "--mode", "eval",
            "--results-dir", str(workspace),
            "--backend", "claude-code",
        ])

        rpb.main()
        assert seen["eval"] is True
        assert seen["tier5"] is True

    def test_context_inject_can_be_disabled(self, tmp_path, monkeypatch):
        workspace = tmp_path / "run"
        (workspace / "data").mkdir(parents=True)
        (workspace / "data" / "memory.db").write_text("")

        seen = {"eval": None, "tier5": None}

        def _fake_run_eval(*_a, **kwargs):
            seen["eval"] = kwargs.get("context_inject")
            return []

        def _fake_run_tier5(*_a, **kwargs):
            seen["tier5"] = kwargs.get("context_inject")
            return []

        monkeypatch.setattr(rpb, "run_eval", _fake_run_eval)
        monkeypatch.setattr(rpb, "run_tier5_eval", _fake_run_tier5)
        monkeypatch.setattr(rpb, "_save_token_usage", lambda *_a, **_k: None)
        monkeypatch.setattr(
            rpb,
            "score_results",
            lambda _results: {
                "overall": {"accuracy": 0.0, "count": 0, "scored": 0, "correct": 0, "partial": 0, "wrong": 0, "error": 0},
                "per_type": {},
                "per_difficulty": {},
            },
        )
        monkeypatch.setattr(sys, "argv", [
            "run_production_benchmark.py",
            "--mode", "eval",
            "--results-dir", str(workspace),
            "--backend", "claude-code",
            "--no-context-inject",
        ])

        rpb.main()
        assert seen["eval"] is False
        assert seen["tier5"] is False


class TestMainIngestSchedule:
    def test_per_day_mode_rejects_obd_schedule(self, tmp_path, monkeypatch):
        workspace = tmp_path / "run"
        monkeypatch.setattr(sys, "argv", [
            "run_production_benchmark.py",
            "--mode", "per-day",
            "--ingest-schedule", "obd",
            "--results-dir", str(workspace),
            "--backend", "claude-code",
        ])
        with pytest.raises(RuntimeError, match="only supports --ingest-schedule per-day"):
            rpb.main()


# ===================================================================
# extract_compact.py — Pure Functions
# ===================================================================


def test_extract_compact_storeable_text_accepts_unsegmented_script():
    assert ec._is_storeable_extracted_fact_text("マヤはオースティンに住んでいる")
    assert not ec._is_storeable_extracted_fact_text("Maya Austin")


def test_extract_compact_date_to_created_at_uses_session_date():
    assert ec._date_to_created_at("2026-03-11") == "2026-03-11T23:59:59"
    assert ec._date_to_created_at("day-runtime-2026-03-11") == "day-runtime-2026-03-11"
    assert ec._date_to_created_at("") is None


class TestBuildTranscript:
    """Tests for build_transcript: message filtering and formatting."""

    def test_basic_user_assistant(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        out = ec.build_transcript(msgs)
        assert "User: hello" in out
        assert "Assistant: hi there" in out

    def test_filters_system_messages(self):
        msgs = [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hello"},
        ]
        out = ec.build_transcript(msgs)
        assert "system" not in out.lower() or "System:" not in out
        assert "User: hello" in out

    def test_strips_channel_prefix(self):
        msgs = [{"role": "user", "content": "[Telegram @user123] actual message"}]
        out = ec.build_transcript(msgs)
        assert "Telegram" not in out
        assert "actual message" in out

    def test_strips_message_id(self):
        msgs = [{"role": "user", "content": "hello\n[message_id: 42]"}]
        out = ec.build_transcript(msgs)
        assert "message_id" not in out
        assert "hello" in out

    def test_skips_gateway_restart(self):
        msgs = [{"role": "assistant", "content": "GatewayRestart: reconnecting"}]
        out = ec.build_transcript(msgs)
        assert out == ""

    def test_skips_heartbeat(self):
        msgs = [{"role": "assistant", "content": "HEARTBEAT HEARTBEAT_OK"}]
        out = ec.build_transcript(msgs)
        assert out == ""

    def test_empty_content_skipped(self):
        msgs = [{"role": "user", "content": ""}]
        out = ec.build_transcript(msgs)
        assert out == ""

    def test_list_content_joined(self):
        msgs = [{"role": "user", "content": [{"text": "part 1"}, {"text": "part 2"}]}]
        out = ec.build_transcript(msgs)
        assert "part 1" in out
        assert "part 2" in out

    def test_custom_agent_name(self):
        msgs = [{"role": "assistant", "content": "hello"}]
        out = ec.build_transcript(msgs, agent_name="Quaid")
        assert "Quaid: hello" in out


class TestBuildExtractionPrompt:
    """Tests for build_extraction_prompt: prompt template generation."""

    def test_contains_system_prompt_structure(self):
        prompt = ec.build_extraction_prompt("Maya")
        assert "memory extraction system" in prompt
        assert "The user who owns this knowledge base is: Maya" in prompt
        assert "SPEAKER ATTRIBUTION" in prompt
        assert "LANGUAGE FIDELITY (MANDATORY)" in prompt
        assert "RELATIONSHIP ROLE FIDELITY" in prompt
        assert "PROJECT LOGS" in prompt

    def test_owner_identity_is_not_hidden_prompt_context(self):
        prompt = ec.build_extraction_prompt("Maya")

        assert "The user who owns this knowledge base is: Maya" in prompt
        assert "USER.md/config seed" not in prompt
        assert "Safe for Mom" not in prompt
        assert "GraphQL API" not in prompt

    def test_prompt_requires_source_language_fidelity(self):
        prompt = ec.build_extraction_prompt("Maya")

        assert "LANGUAGE FIDELITY (MANDATORY)" in prompt
        assert "Detect the transcript's dominant language" in prompt
        assert "Do not translate factual statements into another language" in prompt
        assert "Keep named entities exactly as written" in prompt
        assert "If transcript lines mix languages" in prompt

    def test_focus_argument_does_not_fork_runtime_prompt(self):
        prompt_user = ec.build_extraction_prompt("Maya", focus="user")
        prompt_agent = ec.build_extraction_prompt("Maya", focus="agent")
        prompt_all = ec.build_extraction_prompt("Maya", focus="all")
        assert prompt_user == prompt_agent == prompt_all

    def test_allowed_domains_injected(self):
        prompt = ec.build_extraction_prompt(
            "Maya",
            allowed_domains={"finance": "Money facts", "health": "Health facts"},
        )
        assert "AVAILABLE DOMAINS" in prompt
        assert "- finance: Money facts" in prompt
        assert "- health: Health facts" in prompt

    def test_allowed_domains_deduped(self):
        prompt = ec.build_extraction_prompt("Maya", allowed_domains=["finance", "finance", "health"])
        assert prompt.count("- finance") == 1
        assert prompt.count("- health") == 1

    def test_no_domains_no_domain_line(self):
        prompt = ec.build_extraction_prompt("Maya", allowed_domains=None)
        assert "AVAILABLE DOMAINS" not in prompt

    def test_empty_domains_no_domain_line(self):
        prompt = ec.build_extraction_prompt("Maya", allowed_domains=[])
        assert "AVAILABLE DOMAINS" not in prompt

    def test_known_projects_injected_for_project_log_contract(self):
        prompt = ec.build_extraction_prompt(
            "Maya",
            known_projects={"recipe-app": "Recipe app project workspace"},
        )
        assert "REGISTERED PROJECTS" in prompt
        assert "- recipe-app: Recipe app project workspace" in prompt
        assert "Only emit project_logs entries for projects listed above" in prompt

    def test_prompt_appends_benchmark_appendix_from_env(self, monkeypatch):
        monkeypatch.setenv("BENCHMARK_EXTRACTION_PROMPT_APPENDIX", "LEAN MODE\n- Keep one canonical fact.")
        prompt = ec.build_extraction_prompt("Maya")
        assert "=== BENCHMARK EXTRACTION APPENDIX ===" in prompt
        assert "LEAN MODE" in prompt
        assert "Keep one canonical fact." in prompt


class TestStoreFact:
    def test_adjust_extraction_confidence_deboosts_interpreted_language(self):
        assert ec._adjust_extraction_confidence(
            "Maya was planning to show the recipe app to someone referred to as 'D' (likely David)",
            0.9,
        ) == 0.6
        assert ec._adjust_extraction_confidence(
            "Maya wrote a PRD for the recipe app, showing she takes planning seriously",
            0.6,
        ) == 0.3
        assert ec._adjust_extraction_confidence(
            "Maya lives in South Austin near Zilker",
            0.9,
        ) == 0.9

    def test_store_fact_passes_created_at_flag(self, monkeypatch, tmp_path):
        captured = {}

        def _fake_run(cmd, capture_output, text, timeout, cwd):
            captured["cmd"] = cmd
            return SimpleNamespace(returncode=0, stdout="Stored: node-1", stderr="")

        monkeypatch.setattr(ec, "_resolve_quaid_dir", lambda workspace: str(tmp_path))
        monkeypatch.setattr(subprocess, "run", _fake_run)

        out = ec.store_fact(
            workspace=str(tmp_path),
            text="Maya moved to South Austin",
            owner_id="maya",
            created_at="2026-03-18T09:15:00Z",
        )

        assert out == {"status": "created", "id": "node-1"}
        assert "--created-at" in captured["cmd"]
        idx = captured["cmd"].index("--created-at")
        assert captured["cmd"][idx + 1] == "2026-03-18T09:15:00Z"


class TestBenchmarkStoreFacts:
    def test_store_facts_tolerates_null_edges(self, monkeypatch, tmp_path):
        workspace = tmp_path / "ws"
        workspace.mkdir()

        monkeypatch.setattr(rpb, "_load_active_domain_ids", lambda _ws: ["work", "personal"])

        def _fake_run(cmd, capture_output, text, timeout, cwd, env):
            return SimpleNamespace(returncode=0, stdout="Stored: node-1", stderr="")

        monkeypatch.setattr(rpb.subprocess, "run", _fake_run)

        facts = [
            {
                "text": "Maya has 8+ years of experience shipping B2B SaaS products",
                "category": "fact",
                "extraction_confidence": "high",
                "privacy": "shared",
                "keywords": "experience years background saas b2b",
                "domains": ["work", "personal"],
                "edges": None,
            }
        ]

        stored, edges_created = rpb._store_facts(
            workspace,
            facts,
            {"PATH": os.environ.get("PATH", "")},
            14,
            "2026-03-29",
        )

        assert stored == 1
        assert edges_created == 0

    def test_store_facts_tolerates_null_privacy_and_keywords(self, monkeypatch, tmp_path):
        workspace = tmp_path / "ws"
        workspace.mkdir()

        monkeypatch.setattr(rpb, "_load_active_domain_ids", lambda _ws: ["personal", "travel"])
        captured = {}

        def _fake_run(cmd, capture_output, text, timeout, cwd, env):
            captured["cmd"] = cmd
            return SimpleNamespace(returncode=0, stdout="Stored: node-1", stderr="")

        monkeypatch.setattr(rpb.subprocess, "run", _fake_run)

        facts = [
            {
                "text": "Maya described Rachel's trip as a solo trip.",
                "category": "fact",
                "extraction_confidence": "high",
                "privacy": None,
                "keywords": None,
                "domains": ["personal", "travel"],
                "edges": [],
            }
        ]

        stored, edges_created = rpb._store_facts(
            workspace,
            facts,
            {"PATH": os.environ.get("PATH", "")},
            11,
            "2026-04-07",
        )

        assert stored == 1
        assert edges_created == 0
        assert "--privacy" in captured["cmd"]
        assert captured["cmd"][captured["cmd"].index("--privacy") + 1] == "shared"
        assert "--keywords" not in captured["cmd"]


class TestStoreSessionFacts:
    def test_store_session_facts_prefers_fact_created_at(self, monkeypatch):
        captured = {}

        dataset_stub = ModuleType("dataset")
        dataset_stub.SessionReview = object
        dataset_stub.format_transcript_for_extraction = lambda *a, **k: ""
        dataset_stub.SESSION_DATES = {}
        dataset_stub.FILLER_DATES = {}
        claude_backend_stub = ModuleType("claude_backend")
        claude_backend_stub.call_claude = lambda *a, **k: ("", 0.0)
        claude_backend_stub.is_available = lambda: True
        memory_graph_stub = ModuleType("memory_graph")
        monkeypatch.setitem(sys.modules, "dataset", dataset_stub)
        monkeypatch.setitem(sys.modules, "claude_backend", claude_backend_stub)
        monkeypatch.setitem(sys.modules, "memory_graph", memory_graph_stub)
        ingest_mod = importlib.import_module("ingest")

        def _fake_store(**kwargs):
            captured.update(kwargs)
            return {"id": "node-1", "status": "created"}

        memory_graph_stub.store = _fake_store
        memory_graph_stub.create_edge = lambda **kwargs: True
        monkeypatch.setattr(ingest_mod, "_ensure_quaid", lambda: None)

        extraction = {
            "facts": [
                {
                    "text": "Maya moved to South Austin",
                    "category": "fact",
                    "extraction_confidence": "high",
                    "keywords": "home neighborhood moving",
                    "created_at": "2026-03-18T09:15:00Z",
                    "edges": [],
                }
            ]
        }

        out = ingest_mod.store_session_facts(extraction, owner_id="maya", session_date="2026-03-18", session_num=7)
        assert out["facts_stored"] == 1
        assert captured["created_at"] == "2026-03-18T09:15:00Z"


class TestParseExtractionResponse:
    """Tests for parse_extraction_response: JSON parsing with fence handling."""

    def test_plain_json(self):
        raw = '{"facts": [{"text": "test"}], "soul_snippets": {}, "journal_entries": {}, "project_logs": {}}'
        result = ec.parse_extraction_response(raw)
        assert len(result["facts"]) == 1
        assert result["facts"][0]["text"] == "test"

    def test_json_in_markdown_fence(self):
        raw = '```json\n{"facts": [], "soul_snippets": {}}\n```'
        result = ec.parse_extraction_response(raw)
        assert result["facts"] == []

    def test_json_in_plain_fence(self):
        raw = '```\n{"facts": [{"text": "x"}]}\n```'
        result = ec.parse_extraction_response(raw)
        assert len(result["facts"]) == 1

    def test_fallback_extracts_json_object(self):
        raw = 'Here is the result: {"facts": [], "soul_snippets": {}} and some trailing text'
        result = ec.parse_extraction_response(raw)
        assert "facts" in result

    def test_repairs_missing_trailing_brace(self):
        raw = '{"facts":[{"text":"x"}],"soul_snippets":{},"journal_entries":{},"project_logs":{}'
        result = ec.parse_extraction_response(raw)
        assert len(result["facts"]) == 1
        assert result["facts"][0]["text"] == "x"

    def test_unparseable_returns_defaults(self):
        result = ec.parse_extraction_response("not json at all")
        assert result == {"facts": [], "soul_snippets": {}, "journal_entries": {}, "project_logs": {}}

    def test_empty_string_returns_defaults(self):
        result = ec.parse_extraction_response("")
        assert result == {"facts": [], "soul_snippets": {}, "journal_entries": {}, "project_logs": {}}

    def test_preserves_project_logs(self):
        raw = json.dumps({
            "facts": [],
            "soul_snippets": {},
            "journal_entries": {},
            "project_logs": {"recipe-app": ["deployed v2"]},
        })
        result = ec.parse_extraction_response(raw)
        assert result["project_logs"] == {"recipe-app": ["deployed v2"]}


class TestReadEnvKey:
    """Tests for _read_env_key: .env file parsing."""

    def test_basic_key(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("MY_KEY=my_value\n")
        assert ec._read_env_key(str(env_file), "MY_KEY") == "my_value"

    def test_missing_key_returns_none(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("OTHER=val\n")
        assert ec._read_env_key(str(env_file), "MY_KEY") is None

    def test_export_prefix(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("export MY_KEY=exported_val\n")
        assert ec._read_env_key(str(env_file), "MY_KEY") == "exported_val"

    def test_comments_skipped(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("# comment\nMY_KEY=val\n")
        assert ec._read_env_key(str(env_file), "MY_KEY") == "val"

    def test_missing_file_returns_none(self):
        assert ec._read_env_key("/nonexistent/.env", "KEY") is None

    def test_quoted_value(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text('MY_KEY="quoted value"\n')
        assert ec._read_env_key(str(env_file), "MY_KEY") == "quoted value"


# ===================================================================
# extract_compact.py — File I/O Functions
# ===================================================================


class TestWriteSnippetEntry:
    """Tests for write_snippet_entry: snippet file creation and dedup."""

    def test_creates_new_file(self, tmp_path):
        result = ec.write_snippet_entry(
            str(tmp_path), "SOUL.md", ["bullet one", "bullet two"],
            trigger="Compaction", date_str="2026-03-01", time_str="10:00:00",
        )
        assert result is True
        filepath = tmp_path / "SOUL.snippets.md"
        assert filepath.exists()
        content = filepath.read_text()
        assert "# SOUL.md — Pending Snippets" in content
        assert "- bullet one" in content
        assert "- bullet two" in content

    def test_dedup_same_trigger_date(self, tmp_path):
        ec.write_snippet_entry(
            str(tmp_path), "SOUL.md", ["first"],
            trigger="Compaction", date_str="2026-03-01", time_str="10:00:00",
        )
        result = ec.write_snippet_entry(
            str(tmp_path), "SOUL.md", ["second"],
            trigger="Compaction", date_str="2026-03-01", time_str="11:00:00",
        )
        assert result is False

    def test_different_date_not_deduped(self, tmp_path):
        ec.write_snippet_entry(
            str(tmp_path), "SOUL.md", ["first"],
            trigger="Compaction", date_str="2026-03-01", time_str="10:00:00",
        )
        result = ec.write_snippet_entry(
            str(tmp_path), "SOUL.md", ["second"],
            trigger="Compaction", date_str="2026-03-02", time_str="10:00:00",
        )
        assert result is True

    def test_empty_snippets_returns_false(self, tmp_path):
        assert ec.write_snippet_entry(str(tmp_path), "SOUL.md", []) is False


class TestWriteJournalEntry:
    """Tests for write_journal_entry: journal file creation and dedup."""

    def test_creates_new_file(self, tmp_path):
        result = ec.write_journal_entry(
            str(tmp_path), "SOUL.md", "Reflection text.",
            trigger="Compaction", date_str="2026-03-01",
        )
        assert result is True
        filepath = tmp_path / "journal" / "SOUL.journal.md"
        assert filepath.exists()
        content = filepath.read_text()
        assert "# SOUL.md Journal" in content
        assert "## 2026-03-01 — Compaction" in content
        assert "Reflection text." in content

    def test_dedup_same_date_trigger(self, tmp_path):
        ec.write_journal_entry(
            str(tmp_path), "SOUL.md", "First entry",
            trigger="Compaction", date_str="2026-03-01",
        )
        result = ec.write_journal_entry(
            str(tmp_path), "SOUL.md", "Second entry",
            trigger="Compaction", date_str="2026-03-01",
        )
        assert result is False

    def test_empty_content_returns_false(self, tmp_path):
        assert ec.write_journal_entry(str(tmp_path), "SOUL.md", "") is False
        assert ec.write_journal_entry(str(tmp_path), "SOUL.md", "   ") is False


class TestBenchmarkCoreArtifactMirroring:
    def test_instance_layout_symlinks_data_to_workspace_root(self, tmp_path):
        workspace = tmp_path / "ws"
        (workspace / "config").mkdir(parents=True, exist_ok=True)
        (workspace / "projects").mkdir(parents=True, exist_ok=True)
        (workspace / "config" / "memory.json").write_text("{}")
        (workspace / "data").mkdir(parents=True, exist_ok=True)
        (workspace / "data" / "memory.db").write_text("seed")

        instance_root = rpb._ensure_quaid_instance_layout(workspace)

        assert (instance_root / "data").is_symlink()
        assert (instance_root / "data").resolve() == (workspace / "data").resolve()
        assert (instance_root / "data" / "memory.db").read_text() == "seed"

    def test_cached_core_artifacts_write_to_workspace_and_instance(self, tmp_path):
        workspace = tmp_path / "ws"
        (workspace / "config").mkdir(parents=True, exist_ok=True)
        (workspace / "projects").mkdir(parents=True, exist_ok=True)
        (workspace / "config" / "memory.json").write_text("{}")

        snippets, journals = rpb._write_cached_core_artifacts(
            workspace,
            soul_snippets={"SOUL.md": ["noticed a pattern"]},
            journal_entries={"SOUL.md": "A deeper reflection."},
            trigger="Compaction",
            date_str="2026-03-26",
        )

        assert snippets == 1
        assert journals == 1
        assert not (workspace / "SOUL.snippets.md").exists()
        assert not (workspace / "journal" / "SOUL.journal.md").exists()
        assert (workspace / "instances" / "benchrunner" / "SOUL.snippets.md").exists()
        assert (workspace / "instances" / "benchrunner" / "journal" / "SOUL.journal.md").exists()

    def test_cached_core_artifacts_ignore_non_markdown_journal_dict_payloads(self, tmp_path):
        workspace = tmp_path / "ws"
        (workspace / "config").mkdir(parents=True, exist_ok=True)
        (workspace / "projects").mkdir(parents=True, exist_ok=True)
        (workspace / "config" / "memory.json").write_text("{}")

        snippets, journals = rpb._write_cached_core_artifacts(
            workspace,
            soul_snippets={"SOUL.md": ["noticed a pattern"]},
            journal_entries={
                "SOUL.md": "A deeper reflection.",
                "project_logs": {"recipe-app": ["should be ignored"]},
            },
            trigger="Compaction",
            date_str="2026-03-26",
        )

        assert snippets == 1
        assert journals == 1
        journal_path = workspace / "instances" / "benchrunner" / "journal" / "SOUL.journal.md"
        assert journal_path.exists()
        assert "A deeper reflection." in journal_path.read_text()
        assert not (workspace / "instances" / "benchrunner" / "journal" / "project_logs.journal.md").exists()

    def test_syncs_evolved_instance_identity_back_to_workspace_root(self, tmp_path):
        workspace = tmp_path / "ws"
        (workspace / "config").mkdir(parents=True, exist_ok=True)
        (workspace / "projects").mkdir(parents=True, exist_ok=True)
        (workspace / "config" / "memory.json").write_text("{}")
        identity_dir = rpb._ensure_quaid_instance_layout(workspace)
        identity_dir.mkdir(parents=True, exist_ok=True)

        (workspace / "SOUL.md").write_text("seed soul")
        (workspace / "USER.md").write_text("seed user")
        (workspace / "ENVIRONMENT.md").write_text("seed env")
        (identity_dir / "SOUL.md").write_text("evolved soul")
        (identity_dir / "USER.md").write_text("evolved user")
        (identity_dir / "ENVIRONMENT.md").write_text("evolved env")

        rpb._sync_instance_identity_to_workspace_root(workspace)

        assert (workspace / "SOUL.md").read_text() == "evolved soul"
        assert (workspace / "USER.md").read_text() == "evolved user"
        assert (workspace / "ENVIRONMENT.md").read_text() == "evolved env"


class TestReadSessionMessages:
    """Tests for read_session_messages: JSONL parsing."""

    def test_nested_format(self, tmp_path):
        f = tmp_path / "session.jsonl"
        f.write_text(
            json.dumps({"type": "message", "message": {"role": "user", "content": "hi"}}) + "\n"
        )
        msgs = ec.read_session_messages(str(f))
        assert len(msgs) == 1
        assert msgs[0]["content"] == "hi"

    def test_flat_format(self, tmp_path):
        f = tmp_path / "session.jsonl"
        f.write_text(json.dumps({"role": "user", "content": "hello"}) + "\n")
        msgs = ec.read_session_messages(str(f))
        assert len(msgs) == 1
        assert msgs[0]["content"] == "hello"

    def test_invalid_json_skipped(self, tmp_path):
        f = tmp_path / "session.jsonl"
        f.write_text("not json\n" + json.dumps({"role": "user", "content": "ok"}) + "\n")
        msgs = ec.read_session_messages(str(f))
        assert len(msgs) == 1

    def test_empty_lines_skipped(self, tmp_path):
        f = tmp_path / "session.jsonl"
        f.write_text("\n\n" + json.dumps({"role": "user", "content": "ok"}) + "\n\n")
        msgs = ec.read_session_messages(str(f))
        assert len(msgs) == 1


# ===================================================================
# extract_compact.py — write_project_logs env wiring
# ===================================================================


class TestWriteProjectLogs:
    """Tests for write_project_logs: env setup, path resolution, legacy fallback."""

    def test_sets_workspace_env_and_writes(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        quaid_dir = workspace / "modules" / "quaid"
        (workspace / "data").mkdir(parents=True, exist_ok=True)
        quaid_dir.mkdir(parents=True, exist_ok=True)

        captured = {}

        def _append(project_logs, trigger="Compaction", date_str=None, dry_run=False):
            captured["project_logs"] = project_logs
            captured["trigger"] = trigger
            captured["date_str"] = date_str
            captured["dry_run"] = dry_run
            captured["workspace_env"] = os.environ.get("CLAWDBOT_WORKSPACE")
            captured["quaid_home_env"] = os.environ.get("QUAID_HOME")
            captured["memory_db_env"] = os.environ.get("MEMORY_DB_PATH")
            captured["instance_env"] = os.environ.get("QUAID_INSTANCE")
            return {
                "projects_seen": 1, "projects_updated": 1,
                "entries_seen": 2, "entries_written": 2,
                "projects_unknown": 0, "projects_missing_file": 0,
            }

        ds, ddb, upd = _fake_updater_module("append_project_logs", _append)
        monkeypatch.setitem(sys.modules, "datastore", ds)
        monkeypatch.setitem(sys.modules, "datastore.docsdb", ddb)
        monkeypatch.setitem(sys.modules, "datastore.docsdb.project_updater", upd)
        monkeypatch.setenv("CLAWDBOT_WORKSPACE", str(workspace))

        metrics = ec.write_project_logs(
            workspace=str(workspace),
            project_logs={"recipe-app": ["note one", "note two"]},
            trigger="Compaction",
            date_str="2026-03-15",
            quaid_instance="benchrunner",
        )

        assert metrics["entries_written"] == 2
        assert captured["workspace_env"] == str(workspace)
        assert captured["quaid_home_env"] == str(workspace)
        assert captured["memory_db_env"] == str(workspace / "data" / "memory.db")
        assert captured["instance_env"] == "benchrunner"
        assert captured["project_logs"]["recipe-app"] == ["note one", "note two"]

    def test_legacy_fallback_positional_only(self, tmp_path, monkeypatch):
        """Verify legacy append_project_log_entries is called with positional args only."""
        workspace = tmp_path / "ws"
        quaid_dir = workspace / "modules" / "quaid"
        (workspace / "data").mkdir(parents=True, exist_ok=True)
        quaid_dir.mkdir(parents=True, exist_ok=True)

        calls = []

        def _legacy_append(project_name, entries):
            calls.append((project_name, entries))
            return len(entries)

        ds, ddb, upd = _fake_updater_module("append_project_log_entries", _legacy_append)
        monkeypatch.setitem(sys.modules, "datastore", ds)
        monkeypatch.setitem(sys.modules, "datastore.docsdb", ddb)
        monkeypatch.setitem(sys.modules, "datastore.docsdb.project_updater", upd)
        monkeypatch.setenv("CLAWDBOT_WORKSPACE", str(workspace))

        metrics = ec.write_project_logs(
            workspace=str(workspace),
            project_logs={"recipe-app": ["note 1"], "portfolio": ["note 2"]},
            trigger="Compaction",
            date_str="2026-03-15",
            quaid_instance="benchrunner",
        )

        assert len(calls) == 2
        # Verify called with ONLY positional args (no trigger kwarg — the r388 bug)
        assert calls[0] == ("recipe-app", ["note 1"])
        assert calls[1] == ("portfolio", ["note 2"])
        assert metrics["entries_written"] == 2
        assert metrics["projects_updated"] == 2

    def test_empty_project_logs_returns_empty(self, tmp_path):
        assert ec.write_project_logs(str(tmp_path), {}) == {}
        assert ec.write_project_logs(str(tmp_path), None) == {}

    def test_normalizes_entries(self, tmp_path, monkeypatch):
        """Verify dedup and stripping of entries."""
        workspace = tmp_path / "ws"
        quaid_dir = workspace / "modules" / "quaid"
        (workspace / "data").mkdir(parents=True, exist_ok=True)
        quaid_dir.mkdir(parents=True, exist_ok=True)

        captured = {}

        def _append(project_logs, **kwargs):
            captured["logs"] = project_logs
            return {"projects_seen": 1, "projects_updated": 1,
                    "entries_seen": 1, "entries_written": 1,
                    "projects_unknown": 0, "projects_missing_file": 0}

        ds, ddb, upd = _fake_updater_module("append_project_logs", _append)
        monkeypatch.setitem(sys.modules, "datastore", ds)
        monkeypatch.setitem(sys.modules, "datastore.docsdb", ddb)
        monkeypatch.setitem(sys.modules, "datastore.docsdb.project_updater", upd)
        monkeypatch.setenv("CLAWDBOT_WORKSPACE", str(workspace))

        ec.write_project_logs(
            str(workspace),
            {"app": ["dup", "dup", "  ", "real"]},
            quaid_instance="benchrunner",
        )
        assert captured["logs"] == {"app": ["dup", "real"]}

    def test_env_restored_after_call(self, tmp_path, monkeypatch):
        """Verify env vars are restored even on success."""
        workspace = tmp_path / "ws"
        quaid_dir = workspace / "modules" / "quaid"
        (workspace / "data").mkdir(parents=True, exist_ok=True)
        quaid_dir.mkdir(parents=True, exist_ok=True)

        def _append(project_logs, **kwargs):
            return {}

        ds, ddb, upd = _fake_updater_module("append_project_logs", _append)
        monkeypatch.setitem(sys.modules, "datastore", ds)
        monkeypatch.setitem(sys.modules, "datastore.docsdb", ddb)
        monkeypatch.setitem(sys.modules, "datastore.docsdb.project_updater", upd)

        original_ws = "ORIGINAL_WS"
        original_instance = "ORIGINAL_INSTANCE"
        monkeypatch.setenv("CLAWDBOT_WORKSPACE", original_ws)
        monkeypatch.setenv("QUAID_INSTANCE", original_instance)

        ec.write_project_logs(str(workspace), {"app": ["note"]}, quaid_instance="benchrunner")

        # Must restore original env
        assert os.environ.get("CLAWDBOT_WORKSPACE") == original_ws
        assert os.environ.get("QUAID_INSTANCE") == original_instance

    def test_requires_quaid_instance(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        quaid_dir = workspace / "modules" / "quaid"
        (workspace / "data").mkdir(parents=True, exist_ok=True)
        quaid_dir.mkdir(parents=True, exist_ok=True)

        ds, ddb, upd = _fake_updater_module("append_project_logs", lambda *a, **k: {})
        monkeypatch.setitem(sys.modules, "datastore", ds)
        monkeypatch.setitem(sys.modules, "datastore.docsdb", ddb)
        monkeypatch.setitem(sys.modules, "datastore.docsdb.project_updater", upd)
        monkeypatch.delenv("QUAID_INSTANCE", raising=False)

        with pytest.raises(RuntimeError, match="QUAID_INSTANCE"):
            ec.write_project_logs(str(workspace), {"app": ["note"]})


# ===================================================================
# Domain loading from DB
# ===================================================================


class TestLoadActiveDomainIds:
    """Tests for _load_active_domain_ids: SQLite domain loading."""

    def _setup_db(self, tmp_path, domains):
        workspace = tmp_path / "ws"
        (workspace / "data").mkdir(parents=True)
        db_path = workspace / "data" / "memory.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS domain_registry (
                domain TEXT PRIMARY KEY,
                description TEXT DEFAULT '',
                active INTEGER DEFAULT 1,
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)
        for d, active in domains:
            conn.execute("INSERT INTO domain_registry(domain, active) VALUES (?, ?)", (d, active))
        conn.commit()
        conn.close()
        return workspace

    def test_loads_active_domains(self, tmp_path):
        workspace = self._setup_db(tmp_path, [("finance", 1), ("health", 1), ("disabled", 0)])
        ids = rpb._load_active_domain_ids(workspace)
        assert "finance" in ids
        assert "health" in ids
        assert "disabled" not in ids

    def test_sorted_alphabetically(self, tmp_path):
        workspace = self._setup_db(tmp_path, [("z_domain", 1), ("a_domain", 1)])
        ids = rpb._load_active_domain_ids(workspace)
        assert ids == ["a_domain", "z_domain"]

    def test_missing_db_raises(self, tmp_path):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        with pytest.raises(RuntimeError, match="Domain registry DB missing"):
            rpb._load_active_domain_ids(workspace)

    def test_no_active_domains_raises(self, tmp_path):
        workspace = self._setup_db(tmp_path, [("only", 0)])
        with pytest.raises(RuntimeError, match="No active domains"):
            rpb._load_active_domain_ids(workspace)

    def test_bootstrap_domain_registry_adds_missing_defaults_to_nonempty_registry(self, tmp_path):
        workspace = self._setup_db(tmp_path, [("personal", 1)])
        db_path = workspace / "data" / "memory.db"
        conn = sqlite3.connect(str(db_path))
        try:
            rpb._bootstrap_domain_registry(conn)
            conn.commit()
            rows = conn.execute(
                "SELECT domain FROM domain_registry WHERE active = 1 ORDER BY domain"
            ).fetchall()
        finally:
            conn.close()
        active = {str(row[0]) for row in rows}
        assert "personal" in active
        assert "education" in active
        assert "hobby" in active


class TestBackendSpecificModelDefaults:
    def test_codex_backend_rewrites_default_judge_to_fast_model(self):
        args = SimpleNamespace(
            backend="codex",
            mode="full",
            model="claude-opus-4-6",
            eval_model="claude-haiku-4-5-20251001",
            judge="gpt-4o-mini",
            codex_deep_model="gpt-5.4-mini",
            codex_fast_model="gpt-5.4-mini",
            vllm_model="",
            llama_cpp_model="",
        )
        rpb._apply_backend_specific_model_defaults(
            args,
            model_explicitly_set=False,
            eval_model_explicitly_set=False,
            judge_explicitly_set=False,
        )
        assert args.model == "gpt-5.4-mini"
        assert args.eval_model == "gpt-5.4-mini"
        assert args.judge == "gpt-5.4-mini"

    def test_codex_backend_preserves_explicit_judge(self):
        args = SimpleNamespace(
            backend="codex",
            mode="full",
            model="claude-opus-4-6",
            eval_model="claude-haiku-4-5-20251001",
            judge="gpt-5.4",
            codex_deep_model="gpt-5.4-mini",
            codex_fast_model="gpt-5.4-mini",
            vllm_model="",
            llama_cpp_model="",
        )
        rpb._apply_backend_specific_model_defaults(
            args,
            model_explicitly_set=False,
            eval_model_explicitly_set=False,
            judge_explicitly_set=True,
        )
        assert args.judge == "gpt-5.4"


# ===================================================================
# Eval context source selection + preflight
# ===================================================================


class TestEvalContextCoreSelection:
    def test_prefers_projects_quaid_core_when_richer(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "projects" / "quaid").mkdir(parents=True, exist_ok=True)
        (ws / "SOUL.md").write_text("# Soul\nthin")
        (ws / "projects" / "quaid" / "SOUL.md").write_text("# Soul\n" + ("rich\n" * 200))

        chosen = rpb._resolve_eval_core_path(ws, "SOUL.md")
        assert chosen == ws / "projects" / "quaid" / "SOUL.md"

    def test_build_eval_context_includes_projects_quaid_headers(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "projects" / "quaid").mkdir(parents=True, exist_ok=True)
        (ws / "SOUL.md").write_text("# root soul")
        (ws / "USER.md").write_text("# root user")
        (ws / "ENVIRONMENT.md").write_text("# root environment")
        (ws / "TOOLS.md").write_text("# tools")
        (ws / "projects" / "quaid" / "SOUL.md").write_text("# project soul\n" + ("x\n" * 50))
        (ws / "projects" / "quaid" / "USER.md").write_text("# project user\n" + ("x\n" * 50))
        (ws / "projects" / "quaid" / "ENVIRONMENT.md").write_text("# project environment\n" + ("x\n" * 50))

        ctx = rpb._build_eval_context(ws, include_project_bootstrap=False)
        assert "--- SOUL.md ---" in ctx
        assert "--- USER.md ---" in ctx
        assert "--- ENVIRONMENT.md ---" in ctx
        assert "--- projects/quaid/SOUL.md ---" in ctx
        assert "--- projects/quaid/USER.md ---" in ctx
        assert "--- projects/quaid/ENVIRONMENT.md ---" in ctx

    def test_preflight_fails_when_core_is_too_thin(self, tmp_path, monkeypatch):
        ws = tmp_path / "ws"
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "SOUL.md").write_text("# Soul\nthin")
        (ws / "USER.md").write_text("# User\nthin")
        (ws / "ENVIRONMENT.md").write_text("# Environment\nthin")
        monkeypatch.delenv("BENCHMARK_EVAL_CONTEXT_PROFILE", raising=False)

        with pytest.raises(RuntimeError, match="core markdown context is too thin"):
            rpb._eval_core_context_preflight(ws, max_sessions=20, max_queries_env=0)

    def test_preflight_uses_combined_root_and_projects_quaid_content(self, tmp_path, monkeypatch):
        ws = tmp_path / "ws"
        (ws / "projects" / "quaid").mkdir(parents=True, exist_ok=True)
        (ws / "SOUL.md").write_text("s" * 1300)
        (ws / "USER.md").write_text("u" * 740)
        (ws / "ENVIRONMENT.md").write_text("m" * 422)
        (ws / "projects" / "quaid" / "SOUL.md").write_text("p" * 767)
        (ws / "projects" / "quaid" / "USER.md").write_text("q" * 813)
        (ws / "projects" / "quaid" / "ENVIRONMENT.md").write_text("r" * 740)
        monkeypatch.delenv("BENCHMARK_EVAL_CONTEXT_PROFILE", raising=False)

        rpb._eval_core_context_preflight(ws, max_sessions=20, max_queries_env=0)

    def test_preflight_accepts_legacy_memory_md_fallback(self, tmp_path, monkeypatch):
        ws = tmp_path / "ws"
        (ws / "projects" / "quaid").mkdir(parents=True, exist_ok=True)
        (ws / "SOUL.md").write_text("s" * 1300)
        (ws / "USER.md").write_text("u" * 740)
        (ws / "MEMORY.md").write_text("m" * 422)
        (ws / "projects" / "quaid" / "SOUL.md").write_text("p" * 767)
        (ws / "projects" / "quaid" / "USER.md").write_text("q" * 813)
        (ws / "projects" / "quaid" / "MEMORY.md").write_text("r" * 740)
        monkeypatch.delenv("BENCHMARK_EVAL_CONTEXT_PROFILE", raising=False)

        rpb._eval_core_context_preflight(ws, max_sessions=20, max_queries_env=0)

    def test_build_eval_context_accepts_legacy_memory_md_fallback(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "SOUL.md").write_text("# soul")
        (ws / "USER.md").write_text("# user")
        (ws / "MEMORY.md").write_text("# memory")

        ctx = rpb._build_eval_context(ws, core_files=["SOUL.md", "USER.md", "ENVIRONMENT.md"], include_project_bootstrap=False)

        assert "--- SOUL.md ---" in ctx
        assert "--- USER.md ---" in ctx
        assert "--- MEMORY.md ---" in ctx
        assert "--- ENVIRONMENT.md ---" not in ctx

    def test_core_only_eval_context_profile_keeps_root_tools_without_project_bootstrap(self, tmp_path, monkeypatch):
        ws = tmp_path / "ws"
        (ws / "projects" / "demo").mkdir(parents=True, exist_ok=True)
        (ws / "SOUL.md").write_text("# soul")
        (ws / "USER.md").write_text("# user")
        (ws / "ENVIRONMENT.md").write_text("# environment")
        (ws / "TOOLS.md").write_text("# root tools")
        (ws / "projects" / "demo" / "TOOLS.md").write_text("# demo tools")
        (ws / "projects" / "demo" / "AGENTS.md").write_text("# demo agents")

        monkeypatch.setenv("BENCHMARK_EVAL_CONTEXT_PROFILE", "no-project-bootstrap")

        profile, core_files, include_project_bootstrap = rpb._resolve_eval_context_profile()
        ctx = rpb._build_eval_context(
            ws,
            core_files=core_files,
            include_project_bootstrap=include_project_bootstrap,
        )

        assert profile == "no-project-bootstrap"
        assert core_files == ["SOUL.md", "USER.md", "ENVIRONMENT.md", "TOOLS.md"]
        assert include_project_bootstrap is False
        assert "--- SOUL.md ---" in ctx
        assert "--- USER.md ---" in ctx
        assert "--- ENVIRONMENT.md ---" in ctx
        assert "--- TOOLS.md ---" in ctx
        assert "--- projects/demo/TOOLS.md ---" not in ctx
        assert "--- projects/demo/AGENTS.md ---" not in ctx

    def test_project_only_eval_context_profile_skips_core_markdown_injection(self, tmp_path, monkeypatch):
        ws = tmp_path / "ws"
        (ws / "projects" / "demo").mkdir(parents=True, exist_ok=True)
        (ws / "SOUL.md").write_text("# soul")
        (ws / "USER.md").write_text("# user")
        (ws / "ENVIRONMENT.md").write_text("# environment")
        (ws / "TOOLS.md").write_text("# tools")
        (ws / "projects" / "demo" / "TOOLS.md").write_text("# demo tools")

        monkeypatch.setenv("BENCHMARK_EVAL_CONTEXT_PROFILE", "project-only")

        profile, core_files, include_project_bootstrap = rpb._resolve_eval_context_profile()
        ctx = rpb._build_eval_context(
            ws,
            core_files=core_files,
            include_project_bootstrap=include_project_bootstrap,
        )

        assert profile == "project-only"
        assert core_files == []
        assert include_project_bootstrap is True
        assert "--- SOUL.md ---" not in ctx
        assert "--- USER.md ---" not in ctx
        assert "--- ENVIRONMENT.md ---" not in ctx
        assert "--- TOOLS.md ---" not in ctx
        assert "--- projects/demo/TOOLS.md ---" in ctx

    def test_preflight_skips_for_project_only_eval_context_profile(self, tmp_path, monkeypatch):
        ws = tmp_path / "ws"
        ws.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("BENCHMARK_EVAL_CONTEXT_PROFILE", "project-only")

        rpb._eval_core_context_preflight(ws, max_sessions=20, max_queries_env=0)


class TestRunEvalProviderGuard:
    def test_split_eval_does_not_require_claude_code_calls_for_anthropic_fast_lane(self, tmp_path, monkeypatch):
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True, exist_ok=True)
        (ws / "logs").mkdir(parents=True, exist_ok=True)
        (ws / "config" / "memory.json").write_text(
            json.dumps(
                {
                    "models": {
                        "llmProvider": "claude-code",
                        "deepReasoningProvider": "claude-code",
                        "deepReasoning": "claude-sonnet-4-6",
                        "fastReasoningProvider": "anthropic",
                        "fastReasoning": "claude-haiku-4-5-20251001",
                    }
                }
            )
        )
        monkeypatch.setattr(rpb, "_BACKEND", "claude-code")
        monkeypatch.setattr(rpb, "_resolve_assets_dir", lambda: tmp_path / "assets")
        monkeypatch.setattr(rpb, "load_all_reviews", lambda *a, **k: [_FakeReview(1)])
        monkeypatch.setattr(
            rpb,
            "get_all_eval_queries",
            lambda _reviews: [{"question": "Q?", "ground_truth": "A", "query_type": "factual_recall"}],
        )
        monkeypatch.setattr(rpb, "_eval_core_context_preflight", lambda *a, **k: None)
        monkeypatch.setattr(rpb, "_eval_embedding_provider_preflight", lambda *a, **k: None)
        monkeypatch.setattr(rpb, "_build_eval_context", lambda *a, **k: "ctx")
        monkeypatch.setattr(rpb, "_make_env", lambda _ws: {})
        monkeypatch.setattr(
            rpb,
            "_tool_use_loop",
            lambda **kwargs: (
                "answer",
                [],
                [],
                [],
                {"input_tokens": 10, "output_tokens": 5, "api_calls": 0, "tool_call_details": []},
            ),
        )
        monkeypatch.setattr(rpb, "_judge", lambda *a, **k: ("CORRECT", 1.0))
        monkeypatch.setattr(rpb, "_judge_non_question", lambda *a, **k: ("CORRECT", 1.0))
        monkeypatch.setenv("BENCHMARK_REQUIRE_QUERY_COUNT", "1")
        monkeypatch.setenv("BENCHMARK_PARALLEL", "1")

        results = rpb.run_eval(
            ws,
            api_key="dummy",
            max_sessions=1,
            eval_model="claude-haiku-4-5-20251001",
            context_inject=False,
            judge_model="gpt-4o-mini",
        )
        assert len(results) == 1
        assert results[0]["prediction"] == "answer"


def test_run_eval_syncs_instance_identity_before_building_context(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    (ws / "config").mkdir(parents=True, exist_ok=True)
    (ws / "logs").mkdir(parents=True, exist_ok=True)
    (ws / "config" / "memory.json").write_text(json.dumps({"models": {}}))

    sync_calls = []

    monkeypatch.setattr(rpb, "_BACKEND", "oauth")
    monkeypatch.setattr(rpb, "_resolve_assets_dir", lambda: tmp_path / "assets")
    monkeypatch.setattr(rpb, "load_all_reviews", lambda *a, **k: [_FakeReview(1)])
    monkeypatch.setattr(
        rpb,
        "get_all_eval_queries",
        lambda _reviews: [{"question": "Q?", "ground_truth": "A", "query_type": "factual_recall"}],
    )
    monkeypatch.setattr(rpb, "_eval_core_context_preflight", lambda *a, **k: None)
    monkeypatch.setattr(rpb, "_eval_embedding_provider_preflight", lambda *a, **k: None)
    monkeypatch.setattr(rpb, "_sync_instance_identity_to_workspace_root", lambda _ws: sync_calls.append(_ws))
    monkeypatch.setattr(rpb, "_build_eval_context", lambda *a, **k: "ctx")
    monkeypatch.setattr(rpb, "_make_env", lambda _ws: {})
    monkeypatch.setattr(
        rpb,
        "_tool_use_loop",
        lambda **kwargs: (
            "answer",
            [],
            [],
            [],
            {"input_tokens": 10, "output_tokens": 5, "api_calls": 1, "tool_call_details": []},
        ),
    )
    monkeypatch.setattr(rpb, "_judge", lambda *a, **k: ("CORRECT", 1.0))
    monkeypatch.setattr(rpb, "_judge_non_question", lambda *a, **k: ("CORRECT", 1.0))
    monkeypatch.setenv("BENCHMARK_REQUIRE_QUERY_COUNT", "1")
    monkeypatch.setenv("BENCHMARK_PARALLEL", "1")

    results = rpb.run_eval(
        ws,
        api_key="dummy",
        max_sessions=1,
        eval_model="claude-haiku-4-5-20251001",
        context_inject=False,
        judge_model="gpt-4o-mini",
    )

    assert len(results) == 1
    assert sync_calls == [ws]


def test_run_eval_uses_full_final_environment_for_all_questions(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    (ws / "config").mkdir(parents=True, exist_ok=True)
    (ws / "logs").mkdir(parents=True, exist_ok=True)
    (ws / "config" / "memory.json").write_text(json.dumps({"models": {}}))

    questions = [
        {
            "question": "What test suites exist for the recipe app?",
            "ground_truth": "Current final test suites.",
            "query_type": "project_state",
            "source_session": 10,
        },
        {
            "question": (
                "As of session 10, what test suites existed for the recipe app?"
            ),
            "ground_truth": "Session 10 historical test suites.",
            "query_type": "project_state",
            "source_session": 10,
        },
    ]
    loop_calls = []

    monkeypatch.setattr(rpb, "SESSION_DATES", {10: "2026-03-19"})
    monkeypatch.setattr(rpb, "_BACKEND", "oauth")
    monkeypatch.setattr(rpb, "_resolve_assets_dir", lambda: tmp_path / "assets")
    monkeypatch.setattr(rpb, "load_all_reviews", lambda *a, **k: [_FakeReview(1)])
    monkeypatch.setattr(rpb, "get_all_eval_queries", lambda _reviews: list(questions))
    monkeypatch.setattr(rpb, "_eval_core_context_preflight", lambda *a, **k: None)
    monkeypatch.setattr(rpb, "_eval_embedding_provider_preflight", lambda *a, **k: None)
    monkeypatch.setattr(
        rpb,
        "_sync_instance_identity_to_workspace_root",
        lambda _ws: None,
    )
    monkeypatch.setattr(rpb, "_build_eval_context", lambda *a, **k: "ctx")
    monkeypatch.setattr(rpb, "_make_env", lambda _ws: {})
    monkeypatch.setattr(rpb, "_resolve_eval_parallel_workers", lambda: 1)

    def _tool_use_loop(**kwargs):
        loop_calls.append({
            "question": kwargs["question"],
            "date_to": kwargs["date_to"],
            "max_session": kwargs["max_session"],
        })
        return (
            "answer",
            [],
            [],
            [],
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "api_calls": 1,
                "tool_call_details": [],
            },
        )

    monkeypatch.setattr(rpb, "_tool_use_loop", _tool_use_loop)
    monkeypatch.setattr(rpb, "_judge", lambda *a, **k: ("CORRECT", 1.0))
    monkeypatch.setattr(rpb, "_judge_non_question", lambda *a, **k: ("CORRECT", 1.0))
    monkeypatch.setenv("BENCHMARK_REQUIRE_QUERY_COUNT", "2")
    monkeypatch.setenv("BENCHMARK_PARALLEL", "1")

    results = rpb.run_eval(
        ws,
        api_key="dummy",
        max_sessions=1,
        eval_model="claude-haiku-4-5-20251001",
        context_inject=False,
        judge_model="gpt-4o-mini",
    )

    assert len(results) == 2
    assert loop_calls == [
        {
            "question": "What test suites exist for the recipe app?",
            "date_to": None,
            "max_session": None,
        },
        {
            "question": (
                "As of session 10, what test suites existed for the recipe app?"
            ),
            "date_to": None,
            "max_session": None,
        },
    ]
    assert results[0]["provenance"]["eval_environment"]["scope"] == "full_final"
    assert results[1]["provenance"]["eval_environment"]["scope"] == "full_final"


def test_claude_code_eval_still_requires_nonzero_claude_calls(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    (ws / "config").mkdir(parents=True, exist_ok=True)
    (ws / "logs").mkdir(parents=True, exist_ok=True)
    (ws / "config" / "memory.json").write_text(
        json.dumps(
            {
                "models": {
                    "llmProvider": "claude-code",
                    "deepReasoningProvider": "claude-code",
                    "deepReasoning": "claude-sonnet-4-6",
                    "fastReasoningProvider": "anthropic",
                    "fastReasoning": "claude-haiku-4-5-20251001",
                }
            }
        )
    )
    monkeypatch.setattr(rpb, "_BACKEND", "claude-code")
    monkeypatch.setattr(rpb, "_resolve_assets_dir", lambda: tmp_path / "assets")
    monkeypatch.setattr(rpb, "load_all_reviews", lambda *a, **k: [_FakeReview(1)])
    monkeypatch.setattr(
        rpb,
        "get_all_eval_queries",
        lambda _reviews: [{"question": "Q?", "ground_truth": "A", "query_type": "factual_recall"}],
    )
    monkeypatch.setattr(rpb, "_eval_core_context_preflight", lambda *a, **k: None)
    monkeypatch.setattr(rpb, "_eval_embedding_provider_preflight", lambda *a, **k: None)
    monkeypatch.setattr(rpb, "_build_eval_context", lambda *a, **k: "ctx")
    monkeypatch.setattr(rpb, "_make_env", lambda _ws: {})
    monkeypatch.setattr(
        rpb,
        "_tool_use_loop",
        lambda **kwargs: (
            "answer",
            [],
            [],
            [],
            {"input_tokens": 10, "output_tokens": 5, "api_calls": 0, "tool_call_details": []},
        ),
    )
    monkeypatch.setattr(rpb, "_judge", lambda *a, **k: ("CORRECT", 1.0))
    monkeypatch.setattr(rpb, "_judge_non_question", lambda *a, **k: ("CORRECT", 1.0))
    monkeypatch.setenv("BENCHMARK_REQUIRE_QUERY_COUNT", "1")
    monkeypatch.setenv("BENCHMARK_PARALLEL", "1")

    with pytest.raises(RuntimeError, match="zero Claude API calls"):
        rpb.run_eval(
            ws,
            api_key="dummy",
            max_sessions=1,
            eval_model="claude-sonnet-4-6",
            context_inject=False,
            judge_model="gpt-4o-mini",
        )


def test_canonical_eval_query_count_is_268():
    import importlib.util
    from pathlib import Path

    dataset_path = Path(__file__).resolve().parents[1] / "dataset.py"
    spec = importlib.util.spec_from_file_location("benchmark_dataset", dataset_path)
    assert spec is not None and spec.loader is not None
    dataset = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataset)

    base_count = 156
    total = (
        base_count
        + len(dataset.ADVERSARIAL_QUERIES)
        + len(dataset.NON_QUESTION_QUERIES)
        + len(dataset.ARCHITECTURE_QUERIES)
        + len(dataset.HARDENING_V2_QUERIES)
    )

    assert total == 268


def test_historical_state_queries_include_dates():
    import ast
    import importlib.util
    import re
    from pathlib import Path

    dataset_path = Path(__file__).resolve().parents[1] / "dataset.py"
    spec = importlib.util.spec_from_file_location("benchmark_dataset", dataset_path)
    assert spec is not None and spec.loader is not None
    dataset = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataset)

    densify_path = dataset_path.resolve().parent / "densify.py"
    densify_tree = ast.parse(densify_path.read_text(encoding="utf-8"))
    arc_session_dates = None
    for node in densify_tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "ARC_SESSION_DATES"
            for target in node.targets
        ):
            arc_session_dates = ast.literal_eval(node.value)
            break
    assert arc_session_dates == dataset.SESSION_DATES

    assets_dir = dataset_path.resolve().parents[1] / "data" / "sessions"
    queries = dataset.get_all_eval_queries(dataset.load_all_reviews(assets_dir))
    by_key = {(row["source_session"], row["query_num"]): row for row in queries}

    assert by_key[(9, 2)]["question"].startswith("As of 2026-03-15,")
    assert by_key[(10, 4)]["question"].startswith("As of 2026-03-18,")
    assert by_key[(14, 1)]["question"].startswith("As of 2026-04-28,")
    assert by_key[(16, 1)]["question"].startswith("As of 2026-05-08,")
    assert by_key[(18, 5)]["question"].startswith("As of 2026-05-15,")
    assert by_key[(14, 2)]["question"] == (
        "As of 2026-04-28, what company was Maya transitioning to for her new role?"
    )
    assert "Stripe" in by_key[(14, 2)]["ground_truth"]
    assert "graphql.test.js" in by_key[(16, 3)]["ground_truth"]
    assert 12 in by_key[(16, 3)]["evidence_sessions"]

    by_track = {}
    for review in dataset.load_all_reviews(assets_dir):
        expected_date = dataset.SESSION_DATES[review.session_num]
        text = review.filepath.read_text(encoding="utf-8")
        track_match = re.search(r"^Session:\s+\d+\s+\(Track\s+(\d+):", text, re.MULTILINE)
        assert track_match is not None
        assert dataset.SESSION_TRACKS[review.session_num] == int(track_match.group(1))
        by_track.setdefault(dataset.SESSION_TRACKS[review.session_num], []).append(
            (review.session_num, expected_date)
        )
        source_timestamp = re.search(r"^  Timestamp: (\d{4}-\d{2}-\d{2})", text, re.MULTILINE)
        assert source_timestamp is not None
        assert expected_date == source_timestamp.group(1)
        timestamp_match = re.match(r"(\d{4}-\d{2}-\d{2})", review.timestamp)
        assert timestamp_match is not None
        assert expected_date == timestamp_match.group(1)
        for query in review.eval_queries:
            match = re.match(r"As of (\d{4}-\d{2}-\d{2}),", query.question)
            if match:
                assert match.group(1) == expected_date

        for parent in re.findall(r"\b\d+\s+(?:day|days|week|weeks) after session (\d+)", text, re.IGNORECASE):
            parent_num = int(parent)
            assert expected_date > dataset.SESSION_DATES[parent_num]

    for track_reviews in by_track.values():
        ordered = [date for _session_num, date in sorted(track_reviews)]
        assert ordered == sorted(ordered)


def test_statement_context_grounding_query_set_is_opt_in():
    import importlib.util
    from pathlib import Path

    dataset_path = Path(__file__).resolve().parents[1] / "dataset.py"
    spec = importlib.util.spec_from_file_location("benchmark_dataset", dataset_path)
    assert spec is not None and spec.loader is not None
    dataset = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataset)

    assert len(dataset.get_statement_context_queries()) == 6
    assert all(q["query_type"] == "statement_context_grounding" for q in dataset.get_statement_context_queries())


class TestEvalQueryProfiles:
    def test_hard_representative_selector_keeps_type_coverage_and_hardness(self):
        queries = [
            {"question": "A-easy", "query_type": "factual_recall", "recall_difficulty": "Easy", "query_num": 1},
            {"question": "A-hard", "query_type": "factual_recall", "recall_difficulty": "Hard", "query_num": 2},
            {"question": "A-vhard", "query_type": "factual_recall", "recall_difficulty": "Very Hard", "query_num": 3},
            {"question": "B-easy", "query_type": "project_state", "recall_difficulty": "Easy", "query_num": 4},
            {"question": "B-hard", "query_type": "project_state", "recall_difficulty": "Hard", "query_num": 5},
            {"question": "B-med", "query_type": "project_state", "recall_difficulty": "Medium", "query_num": 6},
            {"question": "C-easy", "query_type": "non_question", "recall_difficulty": "Easy", "query_num": 7},
            {"question": "C-hard", "query_type": "non_question", "recall_difficulty": "Hard", "query_num": 8},
            {"question": "C-med", "query_type": "non_question", "recall_difficulty": "Medium", "query_num": 9},
            {"question": "C-vhard", "query_type": "non_question", "recall_difficulty": "Very Hard", "query_num": 10},
        ]

        idx = rpb._select_hard_representative_query_indices(queries, target_size=6, min_per_type=1)
        selected = [queries[i] for i in idx]
        by_type = {q["query_type"] for q in selected}
        hard_or_vhard = sum(
            1 for q in selected if str(q.get("recall_difficulty", "")).startswith("Very Hard") or q.get("recall_difficulty") == "Hard"
        )

        assert len(selected) == 6
        assert by_type == {"factual_recall", "project_state", "non_question"}
        assert hard_or_vhard >= 4

    def test_apply_eval_query_profile_uses_env_profile(self, monkeypatch):
        queries = [
            {"question": "q1", "query_type": "factual_recall", "recall_difficulty": "Easy", "query_num": 1},
            {"question": "q2", "query_type": "project_state", "recall_difficulty": "Hard", "query_num": 2},
            {"question": "q3", "query_type": "non_question", "recall_difficulty": "Very Hard", "query_num": 3},
            {"question": "q4", "query_type": "project_state", "recall_difficulty": "Medium", "query_num": 4},
            {"question": "q5", "query_type": "factual_recall", "recall_difficulty": "Hard", "query_num": 5},
        ]
        monkeypatch.setenv("BENCHMARK_QUERY_PROFILE", "hard-representative-v1")
        monkeypatch.setenv("BENCHMARK_QUERY_PROFILE_SIZE", "3")
        monkeypatch.setenv("BENCHMARK_QUERY_PROFILE_MIN_PER_TYPE", "1")

        selected, meta = rpb._apply_eval_query_profile(queries)

        assert len(selected) == 3
        assert meta["profile"] == "hard-representative-v1"
        assert meta["selected"] == 3
        assert set(meta["by_type"].keys()) == {"factual_recall", "non_question", "project_state"}

    def test_apply_eval_query_profile_rejects_unknown_profile(self, monkeypatch):
        monkeypatch.setenv("BENCHMARK_QUERY_PROFILE", "nope-v9")
        with pytest.raises(RuntimeError, match="Unknown BENCHMARK_QUERY_PROFILE"):
            rpb._apply_eval_query_profile([{"question": "q", "ground_truth": "a"}])

    def test_apply_eval_query_profile_can_select_sha1_list(self, monkeypatch):
        queries = [
            {"question": "Where does Maya work?", "query_type": "temporal_current", "recall_difficulty": "Hard", "query_num": 1},
            {"question": "What features does the recipe app have?", "query_type": "project_state", "recall_difficulty": "Medium", "query_num": 2},
            {"question": "Who is Biscuit?", "query_type": "factual_recall", "recall_difficulty": "Easy", "query_num": 3},
        ]
        first = hashlib.sha1(queries[0]["question"].encode("utf-8")).hexdigest()[:12]
        third = hashlib.sha1(queries[2]["question"].encode("utf-8")).hexdigest()[:12]
        monkeypatch.setenv("BENCHMARK_QUERY_SHA1S", f"{third},{first}")

        selected, meta = rpb._apply_eval_query_profile(queries)

        assert [q["query_num"] for q in selected] == [1, 3]
        assert meta["profile"] == "sha1-list"
        assert meta["requested"] == 3
        assert meta["selected"] == 2
        assert meta["selected_indices_1based"] == [1, 3]
        assert meta["requested_sha1_selectors"] == [third, first]
        assert meta["by_type"] == {"factual_recall": 1, "temporal_current": 1}

    def test_apply_eval_query_profile_fails_on_unknown_sha1(self, monkeypatch):
        monkeypatch.setenv("BENCHMARK_QUERY_SHA1S", "deadbeef")

        with pytest.raises(RuntimeError, match="did not match any eval query"):
            rpb._apply_eval_query_profile([{"question": "q", "ground_truth": "a"}])

    def test_apply_eval_query_profile_can_select_query_numbers(self, monkeypatch):
        queries = [
            {"question": "q7", "query_type": "temporal_current", "recall_difficulty": "Hard", "query_num": 7},
            {"question": "q11", "query_type": "project_state", "recall_difficulty": "Medium", "query_num": 11},
            {"question": "q42", "query_type": "factual_recall", "recall_difficulty": "Easy", "query_num": 42},
        ]
        monkeypatch.setenv("BENCHMARK_QUERY_NUMS", "3,1")

        selected, meta = rpb._apply_eval_query_profile(queries)

        assert [q["query_num"] for q in selected] == [7, 42]
        assert meta["profile"] == "query-num-list"
        assert meta["requested"] == 3
        assert meta["selected"] == 2
        assert meta["requested_query_nums"] == [3, 1]
        assert meta["selected_query_nums"] == [1, 3]
        assert meta["by_type"] == {"factual_recall": 1, "temporal_current": 1}

    def test_apply_eval_query_profile_rejects_combined_query_selectors(self, monkeypatch):
        monkeypatch.setenv("BENCHMARK_QUERY_NUMS", "1")
        monkeypatch.setenv("BENCHMARK_QUERY_SHA1S", "deadbeef")

        with pytest.raises(RuntimeError, match="only one"):
            rpb._apply_eval_query_profile([{"question": "q", "query_num": 1}])


# ===================================================================
# Integration: per-day extraction orchestration
# ===================================================================


class TestPerDayExtraction:
    """Integration tests for run_per_day_extraction."""

    @staticmethod
    def _init_db(workspace):
        """Create minimal DB schema for run_per_day_extraction's DB verification."""
        db_path = workspace / "data" / "memory.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE IF NOT EXISTS nodes (id TEXT PRIMARY KEY, status TEXT DEFAULT 'active')")
        conn.execute("CREATE TABLE IF NOT EXISTS edges (id TEXT PRIMARY KEY)")
        conn.commit()
        conn.close()

    @staticmethod
    def _stub_prompt_context(monkeypatch, domains):
        """Stub prompt metadata loaders used before the per-day store path."""
        domain_ids = list(domains)
        monkeypatch.setattr(rpb, "_load_active_domain_ids", lambda _ws: domain_ids)
        monkeypatch.setattr(
            rpb,
            "_load_active_domains",
            lambda _ws: [(domain, f"{domain} domain") for domain in domain_ids],
        )
        monkeypatch.setattr(rpb, "_load_prompt_project_defs", lambda _ws: {})

    def test_daily_janitor_and_weekly_distill(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        (workspace / "logs").mkdir(parents=True, exist_ok=True)
        (workspace / "extraction_cache").mkdir(parents=True, exist_ok=True)
        (workspace / "data").mkdir(parents=True, exist_ok=True)
        self._init_db(workspace)

        fake_reviews = [_FakeReview(1), _FakeReview(2), _FakeReview(3)]
        fake_dates = {1: "2026-03-01", 2: "2026-03-03", 3: "2026-03-04"}

        monkeypatch.setattr(rpb, "SESSION_DATES", fake_dates)
        monkeypatch.setattr(rpb, "load_all_reviews", lambda *a, **k: fake_reviews)
        monkeypatch.setattr(rpb, "format_transcript_for_extraction", lambda _r: "hello")
        monkeypatch.setattr(rpb, "_resolve_assets_dir", lambda: tmp_path / "assets")
        self._stub_prompt_context(monkeypatch, ["personal", "project"])
        monkeypatch.setattr(rpb, "_write_prompt_trace", lambda *a, **k: None)
        monkeypatch.setattr(rpb, "_call_anthropic_cached", lambda *a, **k: ("{}", {"input_tokens": 1, "output_tokens": 1}))
        monkeypatch.setattr(
            rpb, "parse_extraction_response",
            lambda _raw: {"facts": [], "soul_snippets": {}, "journal_entries": {}, "project_logs": {}},
        )
        monkeypatch.setattr(rpb, "_store_facts", lambda *a, **k: (0, 0))
        monkeypatch.setattr(rpb, "write_snippet_entry", lambda *a, **k: False)
        monkeypatch.setattr(rpb, "write_journal_entry", lambda *a, **k: False)
        monkeypatch.setattr(rpb, "write_project_logs", lambda *a, **k: {})
        fake_repo = tmp_path / "recipe-app"
        (fake_repo / ".git").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(rpb, "_resolve_project_source_repo", lambda _p: fake_repo)
        monkeypatch.setattr(rpb, "_ensure_project_docs_supervisor_running", lambda *a, **k: None)
        monkeypatch.setattr(rpb, "_handle_project_source_changed", lambda *a, **k: {})

        calls = []
        monkeypatch.setattr(rpb.subprocess, "run", lambda cmd, **k: (calls.append(list(cmd)), _FakeSubprocessResult())[1])
        monkeypatch.setattr(rpb, "_QUAID_DIR", tmp_path)
        monkeypatch.setattr(rpb, "_python_cmd_for_quaid_script",
                            lambda _s: [sys.executable, "-m", "stub"])

        result = rpb.run_per_day_extraction(
            workspace=workspace, api_key="dummy", no_cache=True,
            model="claude-haiku-4-5-20251001", max_sessions=3,
            run_janitor_each_day=True,
        )

        jan_docs_tasks = [
            c for c in calls
            if "--task" in c and any(task in c for task in ("docs_staleness", "docs_cleanup", "rag", "workspace"))
        ]
        jan_all = [c for c in calls if "--task" in c and "all" in c]
        jan_weekly = [c for c in calls if "--task" in c and "journal" in c and "--force-distill" in c]
        assert jan_docs_tasks == []
        assert len(jan_all) == 3
        assert len(jan_weekly) == 2
        assert result["janitor_runs"] == 3
        assert result["weekly_distill_runs"] == 2

        progress = json.loads((workspace / "logs" / "janitor_progress.json").read_text())
        assert progress["phase"] == "Janitor(3/3)"
        assert progress["completed_days"] == 3
        assert progress["total_days"] == 3
        assert progress["state"] == "completed"
        resume_state = json.loads((workspace / "lifecycle_resume" / "latest.json").read_text())
        assert resume_state["completed_days"] == 3
        assert (workspace / "lifecycle_resume" / "day-03-2026-03-04" / "data" / "memory.db").exists()

    def test_project_docs_update_runs_after_project_log_apply_via_janitor(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        (workspace / "logs").mkdir(parents=True, exist_ok=True)
        (workspace / "extraction_cache").mkdir(parents=True, exist_ok=True)
        (workspace / "data").mkdir(parents=True, exist_ok=True)
        (workspace / "projects" / "recipe-app").mkdir(parents=True, exist_ok=True)
        self._init_db(workspace)

        fake_repo = tmp_path / "recipe-app"
        (fake_repo / ".git").mkdir(parents=True, exist_ok=True)

        events = []
        fake_dates = {1: "2026-03-01"}
        monkeypatch.setattr(rpb, "SESSION_DATES", fake_dates)
        monkeypatch.setattr(rpb, "PROJECT_SESSIONS", [(1, "recipe-app", "abc123")])
        monkeypatch.setattr(rpb, "load_all_reviews", lambda *a, **k: [_FakeReview(1)])
        monkeypatch.setattr(rpb, "format_transcript_for_extraction", lambda _r: "hello")
        monkeypatch.setattr(rpb, "_resolve_assets_dir", lambda: tmp_path / "assets")
        monkeypatch.setattr(rpb, "_resolve_project_session_snapshot", lambda *a, **k: None)
        monkeypatch.setattr(rpb, "_resolve_project_source_repo", lambda _p: fake_repo)
        self._stub_prompt_context(monkeypatch, ["project"])
        monkeypatch.setattr(rpb, "_write_prompt_trace", lambda *a, **k: None)
        monkeypatch.setattr(rpb, "_call_anthropic_cached", lambda *a, **k: ("{}", {"input_tokens": 1, "output_tokens": 1}))
        monkeypatch.setattr(
            rpb,
            "parse_extraction_response",
            lambda _raw: {
                "facts": [],
                "soul_snippets": {},
                "journal_entries": {},
                "project_logs": {"recipe-app": ["Added pantry import docs."]},
            },
        )
        monkeypatch.setattr(rpb, "_store_facts", lambda *a, **k: (0, 0))
        monkeypatch.setattr(rpb, "write_snippet_entry", lambda *a, **k: False)
        monkeypatch.setattr(rpb, "write_journal_entry", lambda *a, **k: False)

        def _write_project_logs(*args, **kwargs):
            events.append("project_logs")
            return {"entries_seen": 1, "entries_written": 1, "projects_updated": 1}

        def _handle_project_source_changed(*args, **kwargs):
            events.append("project_docs_update")
            return {}

        def _fake_run(cmd, **kwargs):
            if "--task" in cmd:
                events.append("janitor")
            return _FakeSubprocessResult()

        monkeypatch.setattr(rpb, "write_project_logs", _write_project_logs)
        monkeypatch.setattr(rpb, "_ensure_project_docs_supervisor_running", lambda *a, **k: None)
        monkeypatch.setattr(rpb, "_handle_project_source_changed", _handle_project_source_changed)
        monkeypatch.setattr(rpb.subprocess, "run", _fake_run)
        monkeypatch.setattr(rpb, "_QUAID_DIR", tmp_path)
        monkeypatch.setattr(rpb, "_python_cmd_for_quaid_script", lambda _s: [sys.executable, "-m", "stub"])

        rpb.run_per_day_extraction(
            workspace=workspace,
            api_key="dummy",
            no_cache=True,
            model="claude-haiku-4-5-20251001",
            max_sessions=1,
            run_janitor_each_day=True,
        )

        assert events[:3] == ["project_logs", "janitor", "project_docs_update"]
        assert events.count("project_docs_update") == 1

    def test_project_docs_disabled_ablation_skips_log_writes_and_doc_updates(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        (workspace / "logs").mkdir(parents=True, exist_ok=True)
        (workspace / "extraction_cache").mkdir(parents=True, exist_ok=True)
        (workspace / "data").mkdir(parents=True, exist_ok=True)
        (workspace / "projects" / "recipe-app").mkdir(parents=True, exist_ok=True)
        self._init_db(workspace)

        fake_repo = tmp_path / "recipe-app"
        (fake_repo / ".git").mkdir(parents=True, exist_ok=True)

        monkeypatch.setenv("BENCHMARK_DISABLE_PROJECT_DOCS", "1")
        monkeypatch.setattr(rpb, "SESSION_DATES", {1: "2026-03-01"})
        monkeypatch.setattr(rpb, "PROJECT_SESSIONS", [(1, "recipe-app", "abc123")])
        monkeypatch.setattr(rpb, "load_all_reviews", lambda *a, **k: [_FakeReview(1)])
        monkeypatch.setattr(rpb, "format_transcript_for_extraction", lambda _r: "hello")
        monkeypatch.setattr(rpb, "_resolve_assets_dir", lambda: tmp_path / "assets")
        monkeypatch.setattr(rpb, "_resolve_project_session_snapshot", lambda *a, **k: None)
        monkeypatch.setattr(rpb, "_resolve_project_source_repo", lambda _p: fake_repo)
        self._stub_prompt_context(monkeypatch, ["project"])
        monkeypatch.setattr(rpb, "_write_prompt_trace", lambda *a, **k: None)
        monkeypatch.setattr(rpb, "_call_anthropic_cached", lambda *a, **k: ("{}", {"input_tokens": 1, "output_tokens": 1}))
        monkeypatch.setattr(
            rpb,
            "parse_extraction_response",
            lambda _raw: {
                "facts": [],
                "soul_snippets": {},
                "journal_entries": {},
                "project_logs": {"recipe-app": ["This should stay out of project docs in disabled mode."]},
            },
        )
        monkeypatch.setattr(rpb, "_store_facts", lambda *a, **k: (0, 0))
        monkeypatch.setattr(rpb, "write_snippet_entry", lambda *a, **k: False)
        monkeypatch.setattr(rpb, "write_journal_entry", lambda *a, **k: False)
        monkeypatch.setattr(
            rpb,
            "write_project_logs",
            lambda *a, **k: pytest.fail("disabled project-doc lane must not write project logs"),
        )
        monkeypatch.setattr(
            rpb,
            "_ensure_project_docs_supervisor_running",
            lambda *a, **k: pytest.fail("disabled project-doc lane must not start the supervisor"),
        )
        monkeypatch.setattr(
            rpb,
            "_handle_project_source_changed",
            lambda *a, **k: pytest.fail("disabled project-doc lane must not wait on doc updates"),
        )
        monkeypatch.setattr(rpb.subprocess, "run", lambda *a, **k: _FakeSubprocessResult())
        monkeypatch.setattr(rpb, "_QUAID_DIR", tmp_path)
        monkeypatch.setattr(rpb, "_python_cmd_for_quaid_script", lambda _s: [sys.executable, "-m", "stub"])

        result = rpb.run_per_day_extraction(
            workspace=workspace,
            api_key="dummy",
            no_cache=True,
            model="claude-haiku-4-5-20251001",
            max_sessions=1,
            run_janitor_each_day=False,
        )

        assert result["project_logs_written"] == 0
        assert result["project_logs_seen"] == 0
        assert result["project_logs_projects_updated"] == 0

    def test_skip_janitor(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        (workspace / "logs").mkdir(parents=True, exist_ok=True)
        (workspace / "extraction_cache").mkdir(parents=True, exist_ok=True)
        (workspace / "data").mkdir(parents=True, exist_ok=True)
        self._init_db(workspace)

        fake_dates = {1: "2026-03-01"}
        monkeypatch.setattr(rpb, "SESSION_DATES", fake_dates)
        monkeypatch.setattr(rpb, "load_all_reviews", lambda *a, **k: [_FakeReview(1)])
        monkeypatch.setattr(rpb, "format_transcript_for_extraction", lambda _r: "hello")
        monkeypatch.setattr(rpb, "_resolve_assets_dir", lambda: tmp_path / "assets")
        self._stub_prompt_context(monkeypatch, ["personal"])
        monkeypatch.setattr(rpb, "_write_prompt_trace", lambda *a, **k: None)
        monkeypatch.setattr(rpb, "_call_anthropic_cached", lambda *a, **k: ("{}", {"input_tokens": 1, "output_tokens": 1}))
        monkeypatch.setattr(
            rpb, "parse_extraction_response",
            lambda _raw: {"facts": [], "soul_snippets": {}, "journal_entries": {}, "project_logs": {}},
        )
        monkeypatch.setattr(rpb, "_store_facts", lambda *a, **k: (0, 0))
        monkeypatch.setattr(rpb, "write_snippet_entry", lambda *a, **k: False)
        monkeypatch.setattr(rpb, "write_journal_entry", lambda *a, **k: False)
        monkeypatch.setattr(rpb, "write_project_logs", lambda *a, **k: {})

        calls = []
        monkeypatch.setattr(rpb.subprocess, "run", lambda cmd, **k: (calls.append(list(cmd)), _FakeSubprocessResult())[1])
        monkeypatch.setattr(rpb, "_QUAID_DIR", tmp_path)
        monkeypatch.setattr(rpb, "_python_cmd_for_quaid_script", lambda _s: [sys.executable])

        result = rpb.run_per_day_extraction(
            workspace=workspace, api_key="dummy", no_cache=True,
            run_janitor_each_day=False,
        )

        jan_calls = [c for c in calls if "--task" in c]
        assert len(jan_calls) == 0
        assert result["janitor_runs"] == 0
        assert result["weekly_distill_runs"] == 0
        assert result["rolling_days"] == 0

    def test_corrupt_cached_chunk_is_regenerated(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        (workspace / "logs").mkdir(parents=True, exist_ok=True)
        (workspace / "extraction_cache").mkdir(parents=True, exist_ok=True)
        (workspace / "data").mkdir(parents=True, exist_ok=True)
        self._init_db(workspace)

        fake_dates = {1: "2026-03-01"}
        monkeypatch.setattr(rpb, "SESSION_DATES", fake_dates)
        monkeypatch.setattr(rpb, "load_all_reviews", lambda *a, **k: [_FakeReview(1)])
        monkeypatch.setattr(rpb, "format_transcript_for_extraction", lambda _r: "hello")
        monkeypatch.setattr(rpb, "_resolve_assets_dir", lambda: tmp_path / "assets")
        self._stub_prompt_context(monkeypatch, ["personal"])
        monkeypatch.setattr(rpb, "_write_prompt_trace", lambda *a, **k: None)

        llm_calls = []

        def _call_cached(*args, **kwargs):
            llm_calls.append((args, kwargs))
            return "{}", {"input_tokens": 1, "output_tokens": 1}

        monkeypatch.setattr(rpb, "_call_anthropic_cached", _call_cached)
        monkeypatch.setattr(
            rpb,
            "parse_extraction_response",
            lambda _raw: {"facts": [], "soul_snippets": {}, "journal_entries": {}, "project_logs": {}},
        )
        monkeypatch.setattr(rpb, "_store_facts", lambda *a, **k: (0, 0))
        monkeypatch.setattr(rpb, "write_snippet_entry", lambda *a, **k: False)
        monkeypatch.setattr(rpb, "write_journal_entry", lambda *a, **k: False)
        monkeypatch.setattr(rpb, "write_project_logs", lambda *a, **k: {})
        fake_repo = tmp_path / "recipe-app"
        (fake_repo / ".git").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(rpb, "_resolve_project_source_repo", lambda _p: fake_repo)
        monkeypatch.setattr(rpb, "_handle_project_source_changed", lambda *a, **k: {})
        monkeypatch.setattr(rpb, "_QUAID_DIR", tmp_path)
        monkeypatch.setattr(rpb, "_python_cmd_for_quaid_script", lambda _s: [sys.executable])
        monkeypatch.setattr(rpb.subprocess, "run", lambda *a, **k: _FakeSubprocessResult())

        cache_path = workspace / "extraction_cache" / "chunk-000.json"
        cache_path.write_text("")

        result = rpb.run_per_day_extraction(
            workspace=workspace,
            api_key="dummy",
            no_cache=False,
            model="claude-haiku-4-5-20251001",
            max_sessions=1,
            run_janitor_each_day=False,
        )

        assert len(llm_calls) == 1
        cached_payload = json.loads(cache_path.read_text())
        assert cached_payload["chunk_idx"] == 0
        assert result["days"] == 1

    def test_auto_rolls_oversized_days_through_runtime_extract(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        (workspace / "logs").mkdir(parents=True, exist_ok=True)
        (workspace / "extraction_cache").mkdir(parents=True, exist_ok=True)
        (workspace / "data").mkdir(parents=True, exist_ok=True)
        self._init_db(workspace)

        fake_reviews = [
            _FakeReview(1, timestamp="2026-03-01 12:00:00 UTC"),
            _FakeReview(2, timestamp="2026-03-02 12:00:00 UTC"),
        ]
        fake_dates = {1: "2026-03-01", 2: "2026-03-02"}
        monkeypatch.setattr(rpb, "SESSION_DATES", fake_dates)
        monkeypatch.setattr(rpb, "load_all_reviews", lambda *a, **k: fake_reviews)
        monkeypatch.setattr(
            rpb,
            "format_transcript_for_extraction",
            lambda review: "small transcript" if review.session_num == 1 else "very large transcript",
        )
        monkeypatch.setattr(rpb, "_resolve_assets_dir", lambda: tmp_path / "assets")
        self._stub_prompt_context(monkeypatch, ["personal"])
        monkeypatch.setattr(rpb, "_write_prompt_trace", lambda *a, **k: None)
        monkeypatch.setattr(rpb, "build_extraction_prompt", lambda *a, **k: "prompt")
        monkeypatch.setattr(
            rpb,
            "_estimate_text_tokens",
            lambda text: 9001 if "very large transcript" in text else 1000,
        )

        anthropic_calls = []

        def _call_anthropic(prompt, user_message, model, api_key, max_tokens):
            anthropic_calls.append(user_message)
            return "{}", {"input_tokens": 1, "output_tokens": 1}

        monkeypatch.setattr(rpb, "_call_anthropic_cached", _call_anthropic)
        monkeypatch.setattr(
            rpb,
            "parse_extraction_response",
            lambda _raw: {"facts": [], "soul_snippets": {}, "journal_entries": {}, "project_logs": {}},
        )
        monkeypatch.setattr(rpb, "_store_facts", lambda *a, **k: (0, 0))
        monkeypatch.setattr(rpb, "write_snippet_entry", lambda *a, **k: False)
        monkeypatch.setattr(rpb, "write_journal_entry", lambda *a, **k: False)
        monkeypatch.setattr(rpb, "write_project_logs", lambda *a, **k: {})
        monkeypatch.setattr(rpb.subprocess, "run", lambda *a, **k: _FakeSubprocessResult())
        monkeypatch.setattr(rpb, "_QUAID_DIR", tmp_path)
        monkeypatch.setattr(rpb, "_python_cmd_for_quaid_script", lambda _s: [sys.executable])

        rolling_calls = []

        def _rolling_extract(**kwargs):
            rolling_calls.append(kwargs)
            return {
                "facts_extracted": 7,
                "facts_stored": 5,
                "edges_created": 2,
                "snippets_count": 3,
                "journals_count": 1,
                "project_log_metrics": {"entries_seen": 4, "entries_written": 4, "projects_updated": 1},
                "rolling_batches": 2,
                "signal_to_publish_seconds": 12.5,
                "extract_wall_seconds": 7.0,
                "publish_wall_seconds": 5.5,
            }

        monkeypatch.setattr(rpb, "_run_runtime_rolling_obd_extract", _rolling_extract)

        result = rpb.run_per_day_extraction(
            workspace=workspace,
            api_key="dummy",
            no_cache=True,
            model="claude-haiku-4-5-20251001",
            max_sessions=2,
            run_janitor_each_day=False,
        )

        assert len(anthropic_calls) == 1
        assert "small transcript" in anthropic_calls[0]
        assert len(rolling_calls) == 1
        assert rolling_calls[0]["chunk_tokens"] == 8000
        assert rolling_calls[0]["session_id"] == "day-runtime-2026-03-02"
        assert result["rolling_days"] == 1
        assert result["total_facts"] == 7
        assert result["stored"] == 5
        assert result["edges"] == 2


def test_lifecycle_resume_checkpoint_ignores_vanished_sqlite_sidecar(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    data = workspace / "data"
    data.mkdir(parents=True)
    (data / "memory.db").write_text("db", encoding="utf-8")
    (data / "memory.db-wal").write_text("wal", encoding="utf-8")

    original_copy2 = rpb.shutil.copy2

    def copy2_with_vanished_wal(src, dst, *args, **kwargs):
        if Path(src).name == "memory.db-wal":
            Path(src).unlink(missing_ok=True)
        return original_copy2(src, dst, *args, **kwargs)

    monkeypatch.setattr(rpb.shutil, "copy2", copy2_with_vanished_wal)

    rpb._save_lifecycle_resume_checkpoint(
        workspace,
        completed_days=7,
        total_days=20,
        current_day="2026-03-11",
        counters={},
    )

    snapshot_data = workspace / "lifecycle_resume" / "day-07-2026-03-11" / "data"
    assert (snapshot_data / "memory.db").exists()
    assert json.loads((workspace / "lifecycle_resume" / "latest.json").read_text())["completed_days"] == 7


def test_lifecycle_resume_checkpoint_ignores_vanished_project_docs_worker_tmp(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    worker_dir = workspace / "data" / "project-docs" / "workers"
    worker_dir.mkdir(parents=True)
    tmp_pid = worker_dir / ".misc--benchrunner.pid.example.tmp"
    tmp_pid.write_text("2309341", encoding="utf-8")

    original_copy2 = rpb.shutil.copy2

    def copy2_with_vanished_worker_tmp(src, dst, *args, **kwargs):
        if Path(src).name == tmp_pid.name:
            Path(src).unlink(missing_ok=True)
        return original_copy2(src, dst, *args, **kwargs)

    monkeypatch.setattr(rpb.shutil, "copy2", copy2_with_vanished_worker_tmp)

    rpb._save_lifecycle_resume_checkpoint(
        workspace,
        completed_days=4,
        total_days=20,
        current_day="2026-03-10",
        counters={},
    )

    latest = json.loads((workspace / "lifecycle_resume" / "latest.json").read_text())
    assert latest["completed_days"] == 4


def test_lifecycle_resume_checkpoint_fails_on_vanished_regular_file(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    data = workspace / "data"
    data.mkdir(parents=True)
    (data / "regular.json").write_text("{}", encoding="utf-8")

    original_copy2 = rpb.shutil.copy2

    def copy2_with_vanished_regular_file(src, dst, *args, **kwargs):
        if Path(src).name == "regular.json":
            Path(src).unlink(missing_ok=True)
        return original_copy2(src, dst, *args, **kwargs)

    monkeypatch.setattr(rpb.shutil, "copy2", copy2_with_vanished_regular_file)

    with pytest.raises(rpb.shutil.Error):
        rpb._save_lifecycle_resume_checkpoint(
            workspace,
            completed_days=7,
            total_days=20,
            current_day="2026-03-11",
            counters={},
        )


def test_clear_rolling_pre_publish_checkpoint_removes_matching_session_only(tmp_path):
    workspace = tmp_path / "ws"
    snapshot_dir = workspace / "lifecycle_resume" / "rolling-pre-publish-day-runtime-2026-03-11"
    snapshot_dir.mkdir(parents=True)
    metadata_path = workspace / "logs" / "rolling_pre_publish_checkpoint.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(
        json.dumps(
            {
                "mode": "rolling-pre-publish",
                "session_id": "day-runtime-2026-03-11",
                "snapshot_dir": str(snapshot_dir),
            }
        ),
        encoding="utf-8",
    )

    rpb._clear_rolling_pre_publish_checkpoint(workspace, session_id="other-session")
    assert metadata_path.exists()
    assert snapshot_dir.exists()

    rpb._clear_rolling_pre_publish_checkpoint(workspace, session_id="day-runtime-2026-03-11")
    assert not metadata_path.exists()
    assert not snapshot_dir.exists()


def test_write_session_jsonl_uses_codex_shape_for_codex_backend(tmp_path, monkeypatch):
    path = tmp_path / "session.jsonl"
    monkeypatch.setattr(rpb, "_BACKEND", "codex")

    rpb._write_session_jsonl(
        [
            {"role": "user", "content": "hello", "created_at": "2026-03-01T09:00:00Z"},
            {"role": "assistant", "content": "world", "created_at": "2026-03-01T09:01:00Z"},
        ],
        path,
    )

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["type"] == "response_item"
    assert rows[0]["payload"]["role"] == "user"
    assert rows[0]["payload"]["content"][0]["input_text"] == "hello"
    assert rows[0]["created_at"] == "2026-03-01T09:00:00Z"
    assert rows[1]["payload"]["role"] == "assistant"
    assert rows[1]["payload"]["content"][0]["output_text"] == "world"
    assert rows[1]["created_at"] == "2026-03-01T09:01:00Z"


def test_write_session_jsonl_preserves_created_at_for_default_backend(tmp_path, monkeypatch):
    path = tmp_path / "session.jsonl"
    monkeypatch.setattr(rpb, "_BACKEND", "anthropic-api")

    rpb._write_session_jsonl(
        [
            {"role": "user", "content": "hello", "created_at": "2026-03-01T09:00:00Z"},
            {"role": "assistant", "content": "world"},
        ],
        path,
    )

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {"role": "user", "content": "hello", "created_at": "2026-03-01T09:00:00Z"},
        {"role": "assistant", "content": "world"},
    ]

    def test_project_log_writes_use_benchmark_instance(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        (workspace / "logs").mkdir(parents=True, exist_ok=True)
        (workspace / "extraction_cache").mkdir(parents=True, exist_ok=True)
        (workspace / "data").mkdir(parents=True, exist_ok=True)
        self._init_db(workspace)

        fake_dates = {1: "2026-03-01"}
        monkeypatch.setattr(rpb, "SESSION_DATES", fake_dates)
        monkeypatch.setattr(rpb, "load_all_reviews", lambda *a, **k: [_FakeReview(1)])
        monkeypatch.setattr(rpb, "format_transcript_for_extraction", lambda _r: "hello")
        monkeypatch.setattr(rpb, "_resolve_assets_dir", lambda: tmp_path / "assets")
        self._stub_prompt_context(monkeypatch, ["project"])
        monkeypatch.setattr(rpb, "_write_prompt_trace", lambda *a, **k: None)
        monkeypatch.setattr(rpb, "_call_anthropic_cached", lambda *a, **k: ("{}", {"input_tokens": 1, "output_tokens": 1}))
        monkeypatch.setattr(
            rpb, "parse_extraction_response",
            lambda _raw: {
                "facts": [],
                "soul_snippets": {},
                "journal_entries": {},
                "project_logs": {"recipe-app": ["note"]},
            },
        )
        monkeypatch.setattr(rpb, "_store_facts", lambda *a, **k: (0, 0))
        monkeypatch.setattr(rpb, "write_snippet_entry", lambda *a, **k: False)
        monkeypatch.setattr(rpb, "write_journal_entry", lambda *a, **k: False)
        monkeypatch.setattr(rpb, "_QUAID_DIR", tmp_path)
        monkeypatch.setattr(rpb, "_python_cmd_for_quaid_script", lambda _s: [sys.executable])
        monkeypatch.setattr(rpb.subprocess, "run", lambda cmd, **k: _FakeSubprocessResult())

        captured = {}

        def _write_project_logs(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return {}

        monkeypatch.setattr(rpb, "write_project_logs", _write_project_logs)

        rpb.run_per_day_extraction(
            workspace=workspace,
            api_key="dummy",
            no_cache=True,
            run_janitor_each_day=False,
        )

        assert captured["kwargs"]["quaid_instance"] == rpb._BENCHMARK_QUAID_INSTANCE

    def test_project_log_failure_is_fatal(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        (workspace / "logs").mkdir(parents=True, exist_ok=True)
        (workspace / "extraction_cache").mkdir(parents=True, exist_ok=True)
        (workspace / "data").mkdir(parents=True, exist_ok=True)
        self._init_db(workspace)

        fake_dates = {1: "2026-03-01"}
        monkeypatch.setattr(rpb, "SESSION_DATES", fake_dates)
        monkeypatch.setattr(rpb, "load_all_reviews", lambda *a, **k: [_FakeReview(1)])
        monkeypatch.setattr(rpb, "format_transcript_for_extraction", lambda _r: "hello")
        monkeypatch.setattr(rpb, "_resolve_assets_dir", lambda: tmp_path / "assets")
        self._stub_prompt_context(monkeypatch, ["project"])
        monkeypatch.setattr(rpb, "_write_prompt_trace", lambda *a, **k: None)
        monkeypatch.setattr(rpb, "_call_anthropic_cached", lambda *a, **k: ("{}", {"input_tokens": 1, "output_tokens": 1}))
        monkeypatch.setattr(
            rpb, "parse_extraction_response",
            lambda _raw: {
                "facts": [],
                "soul_snippets": {},
                "journal_entries": {},
                "project_logs": {"recipe-app": ["note"]},
            },
        )
        monkeypatch.setattr(rpb, "_store_facts", lambda *a, **k: (0, 0))
        monkeypatch.setattr(rpb, "write_snippet_entry", lambda *a, **k: False)
        monkeypatch.setattr(rpb, "write_journal_entry", lambda *a, **k: False)
        monkeypatch.setattr(rpb, "_QUAID_DIR", tmp_path)
        monkeypatch.setattr(rpb, "_python_cmd_for_quaid_script", lambda _s: [sys.executable])
        monkeypatch.setattr(rpb.subprocess, "run", lambda cmd, **k: _FakeSubprocessResult())
        monkeypatch.setattr(
            rpb,
            "write_project_logs",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("project logs exploded")),
        )

        with pytest.raises(RuntimeError, match="project logs exploded"):
            rpb.run_per_day_extraction(
                workspace=workspace,
                api_key="dummy",
                no_cache=True,
                run_janitor_each_day=False,
            )

    def test_daily_janitor_failure_is_fatal(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        (workspace / "logs").mkdir(parents=True, exist_ok=True)
        (workspace / "extraction_cache").mkdir(parents=True, exist_ok=True)
        (workspace / "data").mkdir(parents=True, exist_ok=True)
        self._init_db(workspace)

        fake_dates = {1: "2026-03-01"}
        monkeypatch.setattr(rpb, "SESSION_DATES", fake_dates)
        monkeypatch.setattr(rpb, "load_all_reviews", lambda *a, **k: [_FakeReview(1)])
        monkeypatch.setattr(rpb, "format_transcript_for_extraction", lambda _r: "hello")
        monkeypatch.setattr(rpb, "_resolve_assets_dir", lambda: tmp_path / "assets")
        self._stub_prompt_context(monkeypatch, ["personal"])
        monkeypatch.setattr(rpb, "_write_prompt_trace", lambda *a, **k: None)
        monkeypatch.setattr(rpb, "_call_anthropic_cached", lambda *a, **k: ("{}", {"input_tokens": 1, "output_tokens": 1}))
        monkeypatch.setattr(
            rpb, "parse_extraction_response",
            lambda _raw: {"facts": [], "soul_snippets": {}, "journal_entries": {}, "project_logs": {}},
        )
        monkeypatch.setattr(rpb, "_store_facts", lambda *a, **k: (0, 0))
        monkeypatch.setattr(rpb, "write_snippet_entry", lambda *a, **k: False)
        monkeypatch.setattr(rpb, "write_journal_entry", lambda *a, **k: False)
        monkeypatch.setattr(rpb, "write_project_logs", lambda *a, **k: {})
        monkeypatch.setattr(rpb, "_QUAID_DIR", tmp_path)
        monkeypatch.setattr(rpb, "_python_cmd_for_quaid_script", lambda _s: [sys.executable])

        calls = []
        timeouts = []

        def _run(cmd, **kwargs):
            calls.append(list(cmd))
            timeouts.append(kwargs.get("timeout"))
            result = _FakeSubprocessResult()
            result.returncode = 1
            result.stderr = "janitor exploded"
            result.stdout = "janitor stdout"
            return result

        monkeypatch.setattr(rpb.subprocess, "run", _run)

        with pytest.raises(RuntimeError, match="Janitor cycle failed") as excinfo:
            rpb.run_per_day_extraction(
                workspace=workspace, api_key="dummy", no_cache=True,
                run_janitor_each_day=True,
            )

        progress = json.loads((workspace / "logs" / "janitor_progress.json").read_text())
        assert progress["state"] == "failed"
        jan_calls = [c for c in calls if "--task" in c and "embeddings" in c]
        assert len(jan_calls) == 1
        assert timeouts == [rpb._JANITOR_ALL_TIMEOUT_SECONDS]
        failure_files = sorted((workspace / "logs").glob("janitor_failure_daily_embeddings_*.json"))
        assert len(failure_files) == 1
        failure_payload = json.loads(failure_files[0].read_text())
        assert failure_payload["returncode"] == 1
        assert failure_payload["stderr"] == "janitor exploded"
        assert failure_payload["stdout"] == "janitor stdout"
        assert "artifact=" in str(excinfo.value)

    def test_daily_janitor_timeout_is_fatal_and_records_artifact(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        (workspace / "logs").mkdir(parents=True, exist_ok=True)
        (workspace / "extraction_cache").mkdir(parents=True, exist_ok=True)
        (workspace / "data").mkdir(parents=True, exist_ok=True)
        self._init_db(workspace)

        fake_dates = {1: "2026-03-01"}
        monkeypatch.setattr(rpb, "SESSION_DATES", fake_dates)
        monkeypatch.setattr(rpb, "load_all_reviews", lambda *a, **k: [_FakeReview(1)])
        monkeypatch.setattr(rpb, "format_transcript_for_extraction", lambda _r: "hello")
        monkeypatch.setattr(rpb, "_resolve_assets_dir", lambda: tmp_path / "assets")
        self._stub_prompt_context(monkeypatch, ["personal"])
        monkeypatch.setattr(rpb, "_write_prompt_trace", lambda *a, **k: None)
        monkeypatch.setattr(rpb, "_call_anthropic_cached", lambda *a, **k: ("{}", {"input_tokens": 1, "output_tokens": 1}))
        monkeypatch.setattr(
            rpb, "parse_extraction_response",
            lambda _raw: {"facts": [], "soul_snippets": {}, "journal_entries": {}, "project_logs": {}},
        )
        monkeypatch.setattr(rpb, "_store_facts", lambda *a, **k: (0, 0))
        monkeypatch.setattr(rpb, "write_snippet_entry", lambda *a, **k: False)
        monkeypatch.setattr(rpb, "write_journal_entry", lambda *a, **k: False)
        monkeypatch.setattr(rpb, "write_project_logs", lambda *a, **k: {})
        monkeypatch.setattr(rpb, "_QUAID_DIR", tmp_path)
        monkeypatch.setattr(rpb, "_python_cmd_for_quaid_script", lambda _s: [sys.executable])

        def _run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(
                cmd=cmd,
                timeout=kwargs.get("timeout", 0),
                output="janitor stdout",
                stderr="janitor stderr",
            )

        monkeypatch.setattr(rpb.subprocess, "run", _run)

        with pytest.raises(RuntimeError, match="Janitor cycle timed out") as excinfo:
            rpb.run_per_day_extraction(
                workspace=workspace, api_key="dummy", no_cache=True,
                run_janitor_each_day=True,
            )

        progress = json.loads((workspace / "logs" / "janitor_progress.json").read_text())
        assert progress["state"] == "failed"
        failure_files = sorted((workspace / "logs").glob("janitor_failure_daily_embeddings_*.json"))
        assert len(failure_files) == 1
        failure_payload = json.loads(failure_files[0].read_text())
        assert failure_payload["returncode"] == 124
        assert failure_payload["stdout"] == "janitor stdout"
        assert failure_payload["stderr"] == "janitor stderr"
        assert "artifact=" in str(excinfo.value)

    def test_weekly_distillation_failure_is_fatal(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        (workspace / "logs").mkdir(parents=True, exist_ok=True)
        (workspace / "extraction_cache").mkdir(parents=True, exist_ok=True)
        (workspace / "data").mkdir(parents=True, exist_ok=True)
        self._init_db(workspace)

        fake_reviews = [_FakeReview(1), _FakeReview(2)]
        fake_dates = {1: "2026-03-01", 2: "2026-03-03"}
        monkeypatch.setattr(rpb, "SESSION_DATES", fake_dates)
        monkeypatch.setattr(rpb, "load_all_reviews", lambda *a, **k: fake_reviews)
        monkeypatch.setattr(rpb, "format_transcript_for_extraction", lambda _r: "hello")
        monkeypatch.setattr(rpb, "_resolve_assets_dir", lambda: tmp_path / "assets")
        self._stub_prompt_context(monkeypatch, ["personal"])
        monkeypatch.setattr(rpb, "_write_prompt_trace", lambda *a, **k: None)
        monkeypatch.setattr(rpb, "_call_anthropic_cached", lambda *a, **k: ("{}", {"input_tokens": 1, "output_tokens": 1}))
        monkeypatch.setattr(
            rpb, "parse_extraction_response",
            lambda _raw: {"facts": [], "soul_snippets": {}, "journal_entries": {}, "project_logs": {}},
        )
        monkeypatch.setattr(rpb, "_store_facts", lambda *a, **k: (0, 0))
        monkeypatch.setattr(rpb, "write_snippet_entry", lambda *a, **k: False)
        monkeypatch.setattr(rpb, "write_journal_entry", lambda *a, **k: False)
        monkeypatch.setattr(rpb, "write_project_logs", lambda *a, **k: {})
        monkeypatch.setattr(rpb, "_QUAID_DIR", tmp_path)
        monkeypatch.setattr(rpb, "_python_cmd_for_quaid_script", lambda _s: [sys.executable])

        calls = []

        def _run(cmd, **kwargs):
            calls.append(list(cmd))
            result = _FakeSubprocessResult()
            if "--task" in cmd and "journal" in cmd:
                result.returncode = 1
                result.stderr = "weekly distillation exploded"
                result.stdout = "weekly stdout"
            return result

        monkeypatch.setattr(rpb.subprocess, "run", _run)

        with pytest.raises(RuntimeError, match="Weekly journal distillation failed") as excinfo:
            rpb.run_per_day_extraction(
                workspace=workspace, api_key="dummy", no_cache=True,
                model="claude-haiku-4-5-20251001", max_sessions=2,
                run_janitor_each_day=True,
            )

        journal_calls = [c for c in calls if "--task" in c and "journal" in c and "--force-distill" in c]
        assert len(journal_calls) == 1
        failure_files = sorted((workspace / "logs").glob("janitor_failure_journal_*.json"))
        assert len(failure_files) == 1
        failure_payload = json.loads(failure_files[0].read_text())
        assert failure_payload["returncode"] == 1
        assert failure_payload["stderr"] == "weekly distillation exploded"
        assert failure_payload["stdout"] == "weekly stdout"
        assert "artifact=" in str(excinfo.value)

    def test_resume_day_lifecycle_skips_completed_days(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        (workspace / "logs").mkdir(parents=True, exist_ok=True)
        (workspace / "extraction_cache").mkdir(parents=True, exist_ok=True)
        (workspace / "data").mkdir(parents=True, exist_ok=True)
        self._init_db(workspace)
        for rel in ["config", "journal", "projects/quaid"]:
            (workspace / rel).mkdir(parents=True, exist_ok=True)
        for rel in ["IDENTITY.md", "MEMORY.md", "SOUL.md", "TOOLS.md", "USER.md"]:
            (workspace / rel).write_text(f"{rel}\n")

        fake_reviews = [_FakeReview(1), _FakeReview(2), _FakeReview(3)]
        fake_dates = {1: "2026-03-01", 2: "2026-03-03", 3: "2026-03-04"}
        monkeypatch.setattr(rpb, "SESSION_DATES", fake_dates)
        monkeypatch.setattr(rpb, "load_all_reviews", lambda *a, **k: fake_reviews)
        monkeypatch.setattr(rpb, "format_transcript_for_extraction", lambda _r: "hello")
        monkeypatch.setattr(rpb, "_resolve_assets_dir", lambda: tmp_path / "assets")
        self._stub_prompt_context(monkeypatch, ["personal"])
        monkeypatch.setattr(rpb, "_write_prompt_trace", lambda *a, **k: None)
        monkeypatch.setattr(rpb, "_call_anthropic_cached", lambda *a, **k: ("{}", {"input_tokens": 1, "output_tokens": 1}))
        monkeypatch.setattr(
            rpb, "parse_extraction_response",
            lambda _raw: {"facts": [], "soul_snippets": {}, "journal_entries": {}, "project_logs": {}},
        )
        processed_dates = []
        monkeypatch.setattr(rpb, "_store_facts", lambda _ws, _facts, _env, _snum, date: (processed_dates.append(date), (0, 0))[1])
        monkeypatch.setattr(rpb, "write_snippet_entry", lambda *a, **k: False)
        monkeypatch.setattr(rpb, "write_journal_entry", lambda *a, **k: False)
        monkeypatch.setattr(rpb, "write_project_logs", lambda *a, **k: {})
        fake_repo = tmp_path / "recipe-app"
        (fake_repo / ".git").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(rpb, "_resolve_project_source_repo", lambda _p: fake_repo)
        monkeypatch.setattr(rpb, "_QUAID_DIR", tmp_path)
        monkeypatch.setattr(rpb, "_python_cmd_for_quaid_script", lambda _s: [sys.executable, "-m", "stub"])
        monkeypatch.setattr(rpb.subprocess, "run", lambda cmd, **kwargs: _FakeSubprocessResult())

        rpb._save_lifecycle_resume_checkpoint(
            workspace,
            completed_days=1,
            total_days=3,
            current_day="2026-03-01",
            counters={"janitor_runs": 1, "weekly_distill_runs": 1},
        )
        state = rpb.restore_lifecycle_resume_checkpoint(workspace)
        assert state is not None

        result = rpb.run_per_day_extraction(
            workspace=workspace,
            api_key="dummy",
            no_cache=True,
            model="claude-haiku-4-5-20251001",
            max_sessions=3,
            run_janitor_each_day=True,
            resume_state=state,
        )

        assert processed_dates == ["2026-03-03", "2026-03-04"]
        assert result["janitor_runs"] == 3

def test_get_api_key_primes_process_env_for_subprocesses(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("BENCHMARK_ANTHROPIC_OAUTH_TOKEN", raising=False)
    monkeypatch.setattr(rpb, "_find_anthropic_credential", lambda: "sk-ant-oat01-test-token")

    credential = rpb._get_api_key()

    assert credential == "sk-ant-oat01-test-token"
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-oat01-test-token"
    assert os.environ["BENCHMARK_ANTHROPIC_OAUTH_TOKEN"] == "sk-ant-oat01-test-token"


def test_imported_claude_safe_table_count_falls_back_for_vec_virtual_tables():
    imported = _load_imported_claude_history_module()

    class _Cursor:
        def __init__(self, row):
            self._row = row

        def fetchone(self):
            return self._row

    class _Conn:
        def execute(self, query, params=()):
            if query == "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1":
                table_name = params[0]
                if table_name in {"vec_nodes", "vec_nodes_rowids", "vec_doc_chunks", "vec_doc_chunks_rowids"}:
                    return _Cursor((1,))
                return _Cursor(None)
            if query == "SELECT count(*) FROM vec_nodes":
                raise sqlite3.OperationalError("no such module: vec0")
            if query == "SELECT count(*) FROM vec_nodes_rowids":
                return _Cursor((155,))
            if query == "SELECT count(*) FROM vec_doc_chunks":
                raise sqlite3.OperationalError("no such module: vec0")
            if query == "SELECT count(*) FROM vec_doc_chunks_rowids":
                return _Cursor((42,))
            raise AssertionError(f"unexpected query: {query!r} params={params!r}")

    assert imported._safe_table_count(_Conn(), "vec_nodes") == 155
    assert imported._safe_table_count(_Conn(), "vec_doc_chunks") == 42


def test_imported_claude_extract_telemetry_backfills_from_rolling_metric(tmp_path):
    imported = _load_imported_claude_history_module()
    metric_path = tmp_path / "rolling-extraction.jsonl"
    metric_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "rolling_flush",
                        "session_id": "other-session",
                        "dedup_scanned_rows": 12,
                    }
                ),
                json.dumps(
                    {
                        "event": "rolling_flush",
                        "session_id": "imported-claude-day-001",
                        "dedup_scanned_rows": 2493,
                        "dedup_vec_query_count": 146,
                        "dedup_vec_candidates_returned": 2493,
                        "dedup_vec_candidate_limit": 64,
                        "dedup_fts_query_count": 192,
                        "dedup_fts_candidates_returned": 2493,
                        "dedup_fts_candidate_limit": 500,
                        "embedding_cache_requested": 146,
                        "embedding_cache_unique": 146,
                        "embedding_cache_hits": 125,
                        "embedding_cache_warmed": 21,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    telemetry = imported._extract_telemetry(
        {
            "facts_extracted": 146,
            "facts_stored": 146,
            "rolling_batches": 6,
            "rolling_metric_path": str(metric_path),
        },
        session_id="imported-claude-day-001",
    )

    assert telemetry["facts_stored"] == 146
    assert telemetry["dedup"]["scanned_rows"] == 2493
    assert telemetry["dedup"]["vec_query_count"] == 146
    assert telemetry["dedup"]["vec_candidates_returned"] == 2493
    assert telemetry["dedup"]["vec_candidate_limit"] == 64
    assert telemetry["dedup"]["fts_query_count"] == 192
    assert telemetry["dedup"]["fts_candidates_returned"] == 2493
    assert telemetry["dedup"]["fts_candidate_limit"] == 500
    assert telemetry["embedding_cache"]["requested"] == 146
    assert telemetry["embedding_cache"]["hits"] == 125
    assert telemetry["embedding_cache"]["warmed"] == 21


def test_seed_instance_identity_from_sources_prefers_project_templates(tmp_path):
    workspace = tmp_path / "workspace"
    project_dir = workspace / "projects" / "quaid"
    project_dir.mkdir(parents=True, exist_ok=True)
    (workspace / "SOUL.md").write_text("# Root Soul\n", encoding="utf-8")
    (workspace / "USER.md").write_text("# Root User\n", encoding="utf-8")
    (workspace / "ENVIRONMENT.md").write_text("# Root Environment\n", encoding="utf-8")
    (project_dir / "SOUL.md").write_text("# Project Soul\n", encoding="utf-8")
    (project_dir / "USER.md").write_text("# Project User\n", encoding="utf-8")
    (project_dir / "ENVIRONMENT.md").write_text("# Project Environment\n", encoding="utf-8")

    instance_root = rpb._seed_instance_identity_from_sources(
        workspace,
        prefer_project_templates=True,
    )

    assert instance_root == workspace / "instances" / "benchrunner"
    assert (instance_root / "SOUL.md").read_text(encoding="utf-8") == "# Project Soul\n"
    assert (instance_root / "USER.md").read_text(encoding="utf-8") == "# Project User\n"
    assert (instance_root / "ENVIRONMENT.md").read_text(encoding="utf-8") == "# Project Environment\n"


def test_imported_claude_rewrite_workspace_seeds_instance_identity_from_project_bases(monkeypatch, tmp_path):
    imported = _load_imported_claude_history_module()
    workspace = tmp_path / "workspace"
    (workspace / "config").mkdir(parents=True, exist_ok=True)
    (workspace / "config" / "memory.json").write_text(
        json.dumps({"users": {}, "projects": {}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(imported.rpb, "_load_active_domains", lambda _workspace: [])
    monkeypatch.setattr(imported.rpb, "_inject_domains_into_tools_md", lambda template, _rows: template)
    monkeypatch.setattr(imported.rpb, "_load_quaid_tools_template", lambda: "# Tools\n")

    def _fake_seed_quaid_project_docs(seed_workspace: Path) -> None:
        project_dir = seed_workspace / "projects" / "quaid"
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "SOUL.md").write_text("# Seed Soul\n", encoding="utf-8")
        (project_dir / "USER.md").write_text("# Seed User\n", encoding="utf-8")
        (project_dir / "ENVIRONMENT.md").write_text("# Seed Environment\n", encoding="utf-8")

    monkeypatch.setattr(imported.rpb, "_seed_quaid_project_docs", _fake_seed_quaid_project_docs)

    imported._rewrite_workspace_for_claude_history(workspace)

    identity_dir = workspace / "instances" / "benchrunner"
    assert (identity_dir / "SOUL.md").read_text(encoding="utf-8") == "# Seed Soul\n"
    assert (identity_dir / "USER.md").read_text(encoding="utf-8") == "# Seed User\n"
    assert (identity_dir / "ENVIRONMENT.md").read_text(encoding="utf-8") == "# Seed Environment\n"
    assert not (workspace / "SOUL.md").exists()
    assert not (workspace / "USER.md").exists()
    assert not (workspace / "ENVIRONMENT.md").exists()


def test_imported_claude_repair_summary_extract_telemetry(tmp_path):
    imported = _load_imported_claude_history_module()
    results_dir = tmp_path / "run"
    metric_path = results_dir / "benchrunner" / "logs" / "daemon" / "rolling-extraction.jsonl"
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.write_text(
        json.dumps(
            {
                "event": "rolling_flush",
                "session_id": "imported-claude-day-001",
                "dedup_scanned_rows": 2493,
                "dedup_vec_query_count": 146,
                "dedup_vec_candidates_returned": 2493,
                "dedup_vec_candidate_limit": 64,
                "dedup_fts_query_count": 192,
                "dedup_fts_candidates_returned": 2493,
                "embedding_cache_requested": 146,
                "embedding_cache_hits": 125,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path = results_dir / "logs" / "imported_claude_history_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "days": [
                    {
                        "session_id": "imported-claude-day-001",
                        "extract_result": {
                            "facts_extracted": 146,
                            "facts_stored": 146,
                            "rolling_metric_path": str(metric_path),
                        },
                        "telemetry": {
                            "extract": {
                                "dedup": {"scanned_rows": 0, "fts_query_count": 0, "fts_candidates_returned": 0},
                                "embedding_cache": {"requested": 0, "hits": 0},
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    repaired = imported._repair_summary_extract_telemetry(results_dir)
    payload = json.loads(summary_path.read_text())
    extract = payload["days"][0]["telemetry"]["extract"]

    assert repaired["days_repaired"] == 1
    assert payload["summary_repaired_from_metrics"] is True
    assert extract["dedup"]["scanned_rows"] == 2493
    assert extract["dedup"]["vec_query_count"] == 146
    assert extract["dedup"]["vec_candidates_returned"] == 2493
    assert extract["dedup"]["vec_candidate_limit"] == 64
    assert extract["dedup"]["fts_query_count"] == 192
    assert extract["dedup"]["fts_candidates_returned"] == 2493
    assert extract["embedding_cache"]["requested"] == 146
    assert extract["embedding_cache"]["hits"] == 125


def test_imported_claude_trim_seed_summary_rows_drops_remaining_sessions():
    imported = _load_imported_claude_history_module()

    rows = [
        {"session_id": "imported-claude-day-001", "index": 1},
        {"session_id": "imported-claude-day-002", "index": 2},
        {"session_id": "imported-claude-day-003", "index": 3},
    ]

    trimmed = imported._trim_seed_summary_rows(
        rows,
        {"imported-claude-day-003", "imported-claude-day-004"},
    )

    assert [row["session_id"] for row in trimmed] == [
        "imported-claude-day-001",
        "imported-claude-day-002",
    ]


def test_imported_claude_load_existing_summary_rows_returns_deep_copy(tmp_path):
    imported = _load_imported_claude_history_module()
    results_dir = tmp_path / "run"
    summary_path = results_dir / "logs" / "imported_claude_history_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "days": [
                    {"session_id": "imported-claude-day-001", "telemetry": {"extract": {"facts_stored": 10}}},
                ],
            }
        ),
        encoding="utf-8",
    )

    rows = imported._load_existing_summary_rows(results_dir)
    rows[0]["telemetry"]["extract"]["facts_stored"] = 99
    payload = json.loads(summary_path.read_text())

    assert payload["days"][0]["telemetry"]["extract"]["facts_stored"] == 10


def test_imported_claude_parse_args_supports_seed_results_dir_and_start_day(monkeypatch):
    imported = _load_imported_claude_history_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run-imported-claude-history.py",
            "--results-dir",
            "/tmp/out",
            "--seed-results-dir",
            "/tmp/seed",
            "--start-day",
            "9",
        ],
    )

    args = imported._parse_args()

    assert args.results_dir == Path("/tmp/out")
    assert args.seed_results_dir == Path("/tmp/seed")
    assert args.start_day == 9
