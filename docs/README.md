# AgentLife Benchmark Docs

This folder holds the tracked operational docs for the benchmark harness.

## Files

- `LOCAL-DEVELOPMENT.md`
  - benchmark-local config, secret-path setup, and local machine conventions
- `RELEASE-CHECKLIST.md`
  - main-only publish flow, release-candidate sync gate, and push path
- `positioning-agentlife-vs-traditional-memory-benches.md`
  - external positioning brief for why AgentLife is the primary benchmark
- `AGENTLIFE_PUBLIC.md`
  - stable launch-facing public benchmark summary
- `AGENTLIFE_PUBLIC_DRAFT_20260329.md`
  - launch-facing public summary draft (why AgentLife first, results second)
- `rolling-replay.md`
  - imported-Claude / rolling replay utilities and telemetry surfaces
- `oc-native-vm-bootstrap.md`
  - native OpenClaw VM bootstrap and benchmark notes

## Related Tracked Locations

- root methodology: [`../METHODOLOGY.md`](../METHODOLOGY.md)
- published release artifacts: [`../published/`](../published/)

## Usage

1. Start with `LOCAL-DEVELOPMENT.md` for setup.
2. Use `RELEASE-CHECKLIST.md` before shipping the harness package.
3. Use `./scripts/push-main.sh origin` for canonical pushes to GitHub `main`.
4. Use `rolling-replay.md` and `oc-native-vm-bootstrap.md` for specialized
   benchmark lanes and VM-native baselines.
