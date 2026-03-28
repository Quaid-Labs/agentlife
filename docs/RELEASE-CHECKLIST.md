# Benchmark Release Checklist

This repo ships the benchmark harness, not the Quaid runtime itself.

Canonical split:

- harness/orchestration: `eval/`, `scripts/`, `docs/`, `published/`
- runtime-under-test: sibling Quaid checkpoint checkout

## Before Release

1. Verify docs are current.
   - `README.md`
   - `METHODOLOGY.md`
   - `docs/README.md`
   - `docs/LOCAL-DEVELOPMENT.md`
2. Make sure local-only files are ignored.
   - `.agentlife-benchmark.local.json`
   - `runs/`
   - `tmp/`
   - `data/imported-*`
   - `recovered-from-spark-*`
   - built tarballs and other `release/` outputs
3. Confirm public artifacts have a tracked home.
   - released markdown summaries go in `published/runbooks/`
   - frozen public result artifacts go in `published/checkpoints/`
   - while review is still in progress, stage tomorrow's package in the tracked
     `published/runbooks/release-candidate/` and
     `published/checkpoints/release-candidate/` placeholders
4. Keep scratch data out of public release artifacts.
   - remote launch sync must not copy `.agentlife-benchmark.local.json`, `.env`,
     or `release/` to the benchmark host
5. Confirm the canonical harness path is `eval/`.
   - `agentlife/eval/` is a legacy mirror and should not be treated as the
     primary entrypoint in release docs
6. Run the release gate:

```bash
./scripts/release-check.sh
```

7. Build a release tarball if needed:

```bash
./scripts/build-release-tarball.sh
```

## Public Artifact Policy

Do not publish working matrices from internal runbooks by accident.

- Internal working notes and rolling reruns can stay outside this repo
- Public benchmark numbers should be copied into `published/runbooks/`
- Frozen public-supporting artifacts should be copied into
  `published/checkpoints/<release-tag>/`
- Before the tag is final, stage them in the matching `release-candidate/`
  directories instead of leaving them in scratch locations

## Notes

- This repo is already connected to the `quaid-labs/agentlife` remote
- do not push from here until the release review is complete
