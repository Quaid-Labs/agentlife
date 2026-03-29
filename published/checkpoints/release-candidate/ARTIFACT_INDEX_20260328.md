# Release Candidate Artifact Index (Draft 2026-03-28)

This file tracks which benchmark artifacts belong in the release-candidate
checkpoint tree and what is already staged locally.

Use this as a packaging checklist, not as a working scratchpad.

## Staged Current Surface Runs

| Lane | Run | Staged Files |
| --- | --- | --- |
| `AL-S` `Haiku/Haiku` | `r864` | `scores.json`, `ingest_usage.json`, `token_usage.json` |
| `AL-S` `Sonnet/Haiku` | `r880 (r847)` | `scores.json`, source `ingest_usage.json`, eval `token_usage.json` |
| `AL-L` `Haiku/Haiku` | `r881 (r849)` | `scores.json`, source `ingest_usage.json`, eval `token_usage.json` |
| `AL-L` `Sonnet/Haiku` | `r895 (r863)` | `scores.json`, source `ingest_usage.json`, eval `token_usage.json` |
| `AL-L OBD` `Haiku/Haiku` | `r890` | `scores.json`, `ingest_usage.json`, `token_usage.json` |

## Pending Corrected Rolling OBD Sonnet

| Lane | Run | Status |
| --- | --- | --- |
| `AL-L OBD` `Sonnet/Haiku` | `r891` | rolling OBD ingest completed cleanly; eval was intentionally stopped before completion, so no keeper `scores.json` is staged yet |

## Newly Landed Overnight Additions

| Lane | Run | Staged Files |
| --- | --- | --- |
| `AL-S` `Opus/Haiku` | `r884` | `scores.json`, `ingest_usage.json`, `token_usage.json` |
| `AL-L` `Opus/Haiku` | `r885` | `scores.json`, `ingest_usage.json`, `token_usage.json` |

## Still Pending Manual Curation

- selected `evaluation_results.json` excerpts for public-facing examples

## Superseded Local Draft Artifacts

These older draft placeholders are still present locally and should be dropped
before a final public package is committed:

- `r865 (r863)` draft score/token payloads now superseded by `r895 (r863)`
- `r882 (r860)` and `r883 (r859)` draft OBD payloads were based on the wrong
  plain-OBD methodology and should be replaced by corrected rolling-OBD rows

## Destination Layout

Stage approved artifacts into:

- `scores/`
- `token-usage/`
- `selected-eval-results/`

Keep the public package narrow:

- include only artifacts that directly support published claims
- exclude scratch reruns, recovered snapshots, and debugging logs
- prefer representative eval excerpts over full working trees when a smaller
  public artifact set can support the same claim
