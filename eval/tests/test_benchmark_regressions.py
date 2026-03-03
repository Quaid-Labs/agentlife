"""Regression & unit tests for the benchmark harness.

Covers pure/semi-pure functions in run_production_benchmark.py and
extract_compact.py. No network or subprocess calls — everything mocked.
"""

import json
import io
import os
import re
import sqlite3
import sys
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
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

    def test_groups_by_date(self, monkeypatch):
        monkeypatch.setattr(rpb, "SESSION_DATES", {1: "2026-03-01", 2: "2026-03-01", 3: "2026-03-03"})
        reviews = [_FakeReview(1), _FakeReview(2), _FakeReview(3)]
        days = rpb._group_sessions_by_date(reviews)
        assert len(days) == 2
        assert days[0][0] == "2026-03-01"
        assert len(days[0][1]) == 2
        assert days[1][0] == "2026-03-03"
        assert len(days[1][1]) == 1

    def test_empty_reviews(self, monkeypatch):
        monkeypatch.setattr(rpb, "SESSION_DATES", {})
        assert rpb._group_sessions_by_date([]) == []

    def test_unknown_session_date(self, monkeypatch):
        monkeypatch.setattr(rpb, "SESSION_DATES", {})
        reviews = [_FakeReview(99)]
        days = rpb._group_sessions_by_date(reviews)
        assert days[0][0] == "unknown"


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

    def test_text_correct(self):
        label, score = rpb._parse_judge_label("The answer is CORRECT because ...")
        assert label == "CORRECT"
        assert score == 1.0

    def test_text_wrong(self):
        label, score = rpb._parse_judge_label("This is WRONG")
        assert label == "WRONG"
        assert score == 0.0

    def test_both_labels_last_wins(self):
        # If text mentions both, last position wins
        label, _ = rpb._parse_judge_label("Initially WRONG, but actually CORRECT")
        assert label == "CORRECT"

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
