"""Regression & unit tests for the benchmark harness.

Covers pure/semi-pure functions in run_production_benchmark.py and
extract_compact.py. No network or subprocess calls — everything mocked.
"""

import json
import io
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


def _fake_updater_module(fn_name="append_project_logs", fn=None):
    """Create fake datastore.docsdb.project_updater module hierarchy."""
    datastore_mod = ModuleType("datastore")
    docsdb_mod = ModuleType("datastore.docsdb")
    updater_mod = ModuleType("datastore.docsdb.project_updater")
    if fn is not None:
        setattr(updater_mod, fn_name, fn)
    return datastore_mod, docsdb_mod, updater_mod


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

    def test_project_and_projects_dedup(self):
        out = rpb._normalize_domain_list(["project", "projects"])
        assert out == ["project"]

    def test_single_entry(self):
        out = rpb._normalize_domain_list(["health"])
        assert out == ["health"]


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


class TestAnthropicCachedRetries:
    """Tests for _call_anthropic_cached HTTP retry behavior."""

    def test_retries_http_529_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(rpb, "_BACKEND", "api")
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
    }


class TestMakeEnv:
    """Tests for _make_env: environment variable wiring."""

    def test_sets_core_env_vars(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        monkeypatch.setattr(rpb, "_CLAWD", tmp_path)
        monkeypatch.setattr(rpb, "_BACKEND", "api")
        (tmp_path / "plugins" / "quaid").mkdir(parents=True)

        env = rpb._make_env(workspace)
        assert env["CLAWDBOT_WORKSPACE"] == str(workspace.resolve())
        assert env["QUAID_HOME"] == str(workspace.resolve())
        assert env["MEMORY_DB_PATH"] == str(workspace.resolve() / "data" / "memory.db")
        assert env["QUAID_DISABLE_NOTIFICATIONS"] == "1"

    def test_pythonpath_set(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        quaid_dir = tmp_path / "modules" / "quaid"
        quaid_dir.mkdir(parents=True)
        monkeypatch.setattr(rpb, "_QUAID_DIR", quaid_dir)
        monkeypatch.setattr(rpb, "_BACKEND", "api")

        env = rpb._make_env(workspace)
        assert str(quaid_dir.resolve()) in env.get("PYTHONPATH", "")

    def test_does_not_default_mock_embeddings_for_benchmark_subprocesses(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        monkeypatch.setattr(rpb, "_BACKEND", "api")
        monkeypatch.delenv("MOCK_EMBEDDINGS", raising=False)

        env = rpb._make_env(workspace)
        assert "MOCK_EMBEDDINGS" not in env

    def test_preserves_explicit_mock_embeddings_override(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        monkeypatch.setattr(rpb, "_BACKEND", "api")
        monkeypatch.setenv("MOCK_EMBEDDINGS", "0")

        env = rpb._make_env(workspace)
        assert env["MOCK_EMBEDDINGS"] == "0"

    def test_can_force_mock_embeddings_for_targeted_subprocesses(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        monkeypatch.setattr(rpb, "_BACKEND", "api")
        monkeypatch.delenv("MOCK_EMBEDDINGS", raising=False)

        env = rpb._make_env(workspace, mock_embeddings=True)
        assert env["MOCK_EMBEDDINGS"] == "1"


def test_save_token_usage_includes_preinject_timing_stats(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()

    results = [
        {"eval_tokens": {"input_tokens": 10, "output_tokens": 2, "api_calls": 1, "preinject_duration_ms": 100}},
        {"eval_tokens": {"input_tokens": 20, "output_tokens": 3, "api_calls": 2, "preinject_duration_ms": 300,
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
                             "stop_reason": "quality_gate_met", "bailout_counts": {"planner_returned_empty": 0}
                         }}]}},
        {"eval_tokens": {"input_tokens": 30, "output_tokens": 4, "api_calls": 3}},
    ]

    rpb._save_token_usage(results, workspace, "claude-haiku-4-5-20251001")

    data = json.loads((workspace / "token_usage.json").read_text())
    assert data["eval"]["total_tokens"] == 69
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
                "meta": {"mode": "deliberate", "total_ms": 42},
            }),
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)

    text, meta = rpb._tool_memory_recall("coffee", workspace, {"PATH": os.environ.get("PATH", "")})

    assert "recall" in captured["cmd"]
    assert "--json" in captured["cmd"]
    assert "Quaid likes espresso coffee" in text
    assert meta == {"mode": "deliberate", "total_ms": 42}


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
    monkeypatch.setattr(rpb, "_BACKEND", "api")

    text, usage = rpb._call_anthropic_cached("system", "user", "claude-haiku-4-5-20251001", "sk-test")

    assert attempts["n"] == 2
    assert text == "ok"
    assert usage["output_tokens"] == 2

    def test_api_backend_prefers_benchmark_oauth_token(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        monkeypatch.setattr(rpb, "_BACKEND", "api")
        monkeypatch.setattr(rpb, "_find_anthropic_credential", lambda: "sk-ant-oat01-test-token")

        env = rpb._make_env(workspace)
        assert env["BENCHMARK_ANTHROPIC_OAUTH_TOKEN"] == "sk-ant-oat01-test-token"
        assert env["ANTHROPIC_API_KEY"] == "sk-ant-oat01-test-token"


class TestSetupWorkspaceConfig:
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
        assert "baseUrl" not in models
        assert "apiKeyEnv" not in models


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


# ===================================================================
# extract_compact.py — Pure Functions
# ===================================================================


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
        assert "Maya" in prompt

    def test_focus_user_mode(self):
        prompt = ec.build_extraction_prompt("Maya", focus="user")
        assert "USER-FACTS ONLY" in prompt

    def test_focus_agent_mode(self):
        prompt = ec.build_extraction_prompt("Maya", focus="agent")
        assert "AGENT-FACTS ONLY" in prompt

    def test_focus_balanced_default(self):
        prompt = ec.build_extraction_prompt("Maya", focus="all")
        assert "BALANCED" in prompt

    def test_allowed_domains_injected(self):
        prompt = ec.build_extraction_prompt("Maya", allowed_domains=["finance", "health"])
        assert "finance" in prompt
        assert "health" in prompt
        assert "Allowed domain ids" in prompt

    def test_allowed_domains_deduped(self):
        prompt = ec.build_extraction_prompt("Maya", allowed_domains=["finance", "finance", "health"])
        # Should only appear once in the domain line
        domain_match = re.search(r"Allowed domain ids.*?:\s*(.*)", prompt)
        assert domain_match
        domain_str = domain_match.group(1)
        assert domain_str.count("finance") == 1

    def test_no_domains_no_domain_line(self):
        prompt = ec.build_extraction_prompt("Maya", allowed_domains=None)
        assert "Allowed domain ids" not in prompt

    def test_empty_domains_no_domain_line(self):
        prompt = ec.build_extraction_prompt("Maya", allowed_domains=[])
        assert "Allowed domain ids" not in prompt


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
        )

        assert metrics["entries_written"] == 2
        assert captured["workspace_env"] == str(workspace)
        assert captured["quaid_home_env"] == str(workspace)
        assert captured["memory_db_env"] == str(workspace / "data" / "memory.db")
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
        monkeypatch.setenv("CLAWDBOT_WORKSPACE", original_ws)

        ec.write_project_logs(str(workspace), {"app": ["note"]})

        # Must restore original env
        assert os.environ.get("CLAWDBOT_WORKSPACE") == original_ws


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
        (ws / "MEMORY.md").write_text("# root memory")
        (ws / "TOOLS.md").write_text("# tools")
        (ws / "projects" / "quaid" / "SOUL.md").write_text("# project soul\n" + ("x\n" * 50))
        (ws / "projects" / "quaid" / "USER.md").write_text("# project user\n" + ("x\n" * 50))
        (ws / "projects" / "quaid" / "MEMORY.md").write_text("# project memory\n" + ("x\n" * 50))

        ctx = rpb._build_eval_context(ws, include_project_bootstrap=False)
        assert "--- SOUL.md ---" in ctx
        assert "--- USER.md ---" in ctx
        assert "--- MEMORY.md ---" in ctx
        assert "--- projects/quaid/SOUL.md ---" in ctx
        assert "--- projects/quaid/USER.md ---" in ctx
        assert "--- projects/quaid/MEMORY.md ---" in ctx

    def test_preflight_fails_when_core_is_too_thin(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "SOUL.md").write_text("# Soul\nthin")
        (ws / "USER.md").write_text("# User\nthin")
        (ws / "MEMORY.md").write_text("# Memory\nthin")

        with pytest.raises(RuntimeError, match="core markdown context is too thin"):
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

    dataset_path = Path("~/<username>/agentlife-benchmark/eval/dataset.py")
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
        monkeypatch.setattr(rpb, "_load_active_domain_ids", lambda _ws: ["personal", "project"])
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

        jan_all = [c for c in calls if "--task" in c and "all" in c]
        jan_weekly = [c for c in calls if "--task" in c and "journal" in c and "--force-distill" in c]
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
        monkeypatch.setattr(rpb, "_load_active_domain_ids", lambda _ws: ["personal"])
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
        monkeypatch.setattr(rpb, "_load_active_domain_ids", lambda _ws: ["personal"])
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
        jan_calls = [c for c in calls if "--task" in c and "all" in c]
        assert len(jan_calls) == 1
        failure_files = sorted((workspace / "logs").glob("janitor_failure_all_*.json"))
        assert len(failure_files) == 1
        failure_payload = json.loads(failure_files[0].read_text())
        assert failure_payload["returncode"] == 1
        assert failure_payload["stderr"] == "janitor exploded"
        assert failure_payload["stdout"] == "janitor stdout"
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
        monkeypatch.setattr(rpb, "_load_active_domain_ids", lambda _ws: ["personal"])
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
        monkeypatch.setattr(rpb, "_load_active_domain_ids", lambda _ws: ["personal"])
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
