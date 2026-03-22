# Rolling Replay

This document covers the benchmark-side rolling extraction utilities used for
large transcript stress tests and migration-style replays.

These paths are not scored leaderboard lanes. They exist to answer questions
like:

- can Quaid ingest a very large real transcript end-to-end?
- where do DB-facing stages start scaling badly?
- can a historical transcript be migrated into current runtime semantics?

---

## Utilities

### `scripts/export-imported-claude-history.py`

Exports a raw Claude Code transcript JSONL into day-sliced JSONL files plus a
manifest.

Important behaviors:

- keeps only human-readable `user` and `assistant` text
- preserves user text-block turns such as `[Request interrupted by user for tool use]`
- buckets by operational day using a configurable UTC cutoff hour
- writes a migration-friendly manifest with per-day counts and timestamps

Example:

```bash
python scripts/export-imported-claude-history.py \
  --source ~/.claude/projects/<session-id>.jsonl \
  --output-dir data/imported-claude-dev-first7-v2 \
  --cutoff-hour 4 \
  --max-days 7
```

### `scripts/run-imported-claude-history.py`

Replays an exported manifest through a fresh Quaid workspace.

Important behaviors:

- uses the real benchmark workspace setup path
- keeps canonical Quaid project docs/rules in the replay workspace
- can run normal or rolling extraction
- runs janitor between imported days unless `--skip-janitor` is set
- supports a replay-specific `--janitor-timeout` because imported transcript
  stress lanes often need a longer janitor window than scored benchmark runs
- writes a per-day replay summary with DB deltas and normalized telemetry
- benchmark workspace config can tune LLM and embedding concurrency
  independently via `BENCHMARK_JANITOR_LLM_WORKERS` and
  `BENCHMARK_EMBEDDING_WORKERS`

Example:

```bash
python scripts/run-imported-claude-history.py \
  --manifest data/imported-claude-dev-first7-v2/manifest.json \
  --results-dir runs/imported-claude-dev-first7-v2-rerun \
  --backend oauth \
  --model claude-sonnet-4-6 \
  --janitor-timeout 3600 \
  --rolling
```

---

## Manifest Schema

Current exporter output is `schema_version: 2`.

Top-level fields:

- `schema_version`
- `source_format`
- `export_format`
- `message_fields`
- `source_path`
- `output_dir`
- `cutoff_hour`
- `days_exported`
- `messages_exported`
- `content_chars_exported`
- `days`

Per-day fields:

- `session_id`
- `operational_day`
- `message_count`
- `role_counts`
- `content_chars`
- `first_timestamp`
- `last_timestamp`
- `path`

This format is intended to be reusable by future migration tooling. The replay
runner only requires the day list plus `path`, but the extra metadata makes it
possible to reason about day density and export fidelity without reopening the
raw transcript.

---

## Replay Summary Schema

Current replay output is `summary_type: imported_claude_history_replay` with
`schema_version: 2`.

Top-level fields:

- manifest metadata
- backend/model/rolling config
- `days`

Per-day fields:

- `index`
- `session_id`
- `operational_day`
- `message_count`
- `manifest_day`
- `extract_result`
- `janitor_stats`
- `db_stats_before`
- `db_stats_after`
- `db_delta`
- `db_stats` (alias of `db_stats_after`)
- `telemetry.extract`
- `telemetry.janitor`

`db_stats_*` includes the high-signal table counts needed for scaling analysis,
including `nodes`, `edges`, `doc_chunks`, `vec_nodes`, and `vec_doc_chunks`.

`telemetry.extract` normalizes the high-signal fields from the raw extract
result, including:

- rolling batch/chunk counts
- stage/flush/publish timing
- dedup metrics
- embedding cache metrics
- project-log metrics

`telemetry.janitor` normalizes:

- success/last run
- total duration
- per-task durations
- applied change counters

---

## Rolling Extraction Surfaces

When `--rolling` is enabled, the runner drives the real daemon signal path from
`eval/run_production_benchmark.py`:

- staged state:
  - `benchrunner/data/rolling-extraction/<session_id>.json`
- rolling daemon metrics:
  - `benchrunner/logs/daemon/rolling-extraction.jsonl`
- cursor state:
  - `benchrunner/data/session-cursors/<session_id>.json`

The harness summarizes that daemon output into:

- `rolling_batches`
- `rolling_stage_wall_seconds`
- `rolling_driver_stage_wall_seconds`
- `rolling_driver_flush_wall_seconds`
- `flush_wall_seconds`
- `signal_to_publish_seconds`

It also preserves dedup scaling counters coming from runtime publish:

- hash exact hits
- scanned rows
- gray-zone rows
- LLM checks / same / different
- vec query count
- vec candidates returned
- vec candidate limit
- vec limit hits
- FTS query count
- FTS candidates returned
- FTS candidate limit
- FTS limit hits
- fallback scan count / candidates returned
- token-prefilter term / skip counts

---

## Operational Notes

- This path is for stress/migration analysis, not benchmark scoring.
- Keep using the normal benchmark launcher for actual AgentLife scored runs.
- If the replay stalls during a rolling stage, inspect:
  - session cursor
  - pending extraction signals
  - rolling state file
  - `benchrunner/logs/daemon/rolling-extraction.jsonl`
- If the replay clears extraction but fails in janitor, the summary file is
  still useful because DB deltas and extract telemetry are written per completed
  day.
