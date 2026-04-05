# Release Candidate Checkpoint Index

This index enumerates the frozen public-supporting checkpoint artifacts currently
staged under `published/checkpoints/release-candidate/`.

Included artifact groups:

- `scores/`
  - selected benchmark `scores.json` snapshots backing published accuracy claims
- `token-usage/`
  - selected ingest and eval token-usage snapshots backing published cost/token claims
- `selected-eval-results/`
  - selected supporting eval result extracts for release review

This file exists so the release gate can verify that checkpoint artifacts are
present at the release-candidate root in addition to the subdirectories above.
