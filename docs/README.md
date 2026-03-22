# AgentLife Benchmark Docs

This folder is the operational and communication layer for benchmark work.

## Files

- `positioning-agentlife-vs-traditional-memory-benches.md`
  - 1-page external positioning brief for why AgentLife is the primary benchmark.
- `public-results-template.md`
  - Reusable format for publishing benchmark results consistently.
- `runbook.md`
  - Canonical run flow and preflight checks.
- `telemetry-guide.md`
  - How to read run telemetry and diagnose failures/regressions.
- `metric-interpretation.md`
  - How to interpret category-level and retrieval metrics correctly.
- `scope-and-policy.md`
  - Benchmark purity and scope boundaries for harness vs runtime logic.
- `rolling-replay.md`
  - Rolling OBD/imported-Claude replay utilities, manifest schema, and telemetry outputs.

## Usage

1. Start with `runbook.md` before launching runs.
2. Use `telemetry-guide.md`, `metric-interpretation.md`, and `rolling-replay.md`
   when analyzing runs or setting up transcript stress lanes.
3. Use `positioning-agentlife-vs-traditional-memory-benches.md` and
   `public-results-template.md` for external communication.
