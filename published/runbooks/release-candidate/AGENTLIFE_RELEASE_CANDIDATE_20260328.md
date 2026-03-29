# AgentLife Release Candidate Runbook (Draft 2026-03-28)

This draft captures the latest reviewed current-surface Quaid benchmark matrix
that is ready to package for release review.

Status:

- refreshed eval surface landed for the core non-Opus Quaid lanes
- `AL-L` `Sonnet/Haiku` keeper is now `r895 (r863)` after the recall hardening
  and graph-anchor relation-surface fixes
- corrected rolling `AL-L OBD` `Haiku/Haiku` replacement landed as `r890`
- corrected rolling `AL-L OBD` `Sonnet/Haiku` ingest lineage exists as `r891`,
  but eval was intentionally stopped before completion, so that public row is
  still pending
- `AL-S Opus/Haiku` is now landed as `r884`
- `AL-L Opus/Haiku` is now landed as `r885`
- only copy numbers from this draft into final public release locations after
  review

## Scope

- benchmark family: AgentLife
- dataset: canonical AgentLife query set
- judge: `gpt-4o-mini`
- current release surface includes:
  - `AL-S`
  - `AL-L`
  - corrected `AL-L OBD` `Haiku/Haiku`
- this draft excludes:
  - FC lanes
  - OpenClaw native baselines
  - historical anchors
  - pending corrected `AL-L OBD` `Sonnet/Haiku`

## Current Surface Matrix

| Lane | Run | Models | Accuracy | Retrieval | Ingest Spend | Eval Spend | Total Spend | Ingest Tokens | Eval Tokens | Total Tokens |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `AL-S` `Haiku/Haiku` | `r864` | ingest `Haiku`, eval `Haiku` | `85.26%` | `64.93%` | `$1.8031` | `$6.6994` | `$8.5025` | `1,157,312` | `7,882,200` | `9,039,512` |
| `AL-S` `Sonnet/Haiku` | `r880 (r847)` | ingest `Sonnet`, eval `Haiku` | `87.69%` | `69.40%` | `$5.5589` | `$12.1063` | `$17.6652` | `967,141` | `14,234,246` | `15,201,387` |
| `AL-L` `Haiku/Haiku` | `r881 (r849)` | ingest `Haiku`, eval `Haiku` | `83.40%` | `67.91%` | `$4.4218` | `$14.9424` | `$19.3642` | `2,373,377` | `17,629,080` | `20,002,457` |
| `AL-L` `Sonnet/Haiku` | `r895 (r863)` | ingest `Sonnet`, eval `Haiku` | `85.82%` | `65.30%` | `$22.2103` | `$12.6924` | `$34.9027` | `3,794,335` | `14,907,315` | `18,701,650` |
| `AL-L OBD` `Haiku/Haiku` | `r890` | ingest `Haiku`, eval `Haiku` | `82.46%` | `66.98%` | `$2.4268` | `$5.9646` | `$8.3914` | `1,406,571` | `6,929,651` | `8,336,222` |

## Pending Corrected OBD Sonnet Row

The earlier draft `AL-L OBD` `Sonnet/Haiku` row was based on the wrong
plain-OBD methodology and should not be treated as the keeper for release.

Current corrected state:

- corrected rolling ingest lineage: `r891`
- models: ingest `Sonnet`, eval `Haiku`
- ingest completed cleanly with rolling OBD
- eval was intentionally stopped before completion so the corrected public row
  is still pending

## Read

- `AL-S`: `Sonnet/Haiku` leads `Haiku/Haiku` on both accuracy and retrieval on
  the current refreshed eval surface.
- `AL-L`: refreshed `Sonnet/Haiku` now leads refreshed `Haiku/Haiku` on
  accuracy, but not on retrieval, so the large Sonnet lane is healthier than
  before without becoming the clean retrieval winner.
- `AL-L OBD` `Haiku/Haiku`: corrected rolling `r890` is in-family with the
  earlier OBD Haiku numbers and is the current reviewed OBD row in this draft.
- `AL-L OBD` `Sonnet/Haiku`: keep pending until the corrected rolling eval is
  rerun; do not use the earlier plain-OBD draft row in public material.

## Pending Before Final Publish

- rerun the corrected rolling `AL-L OBD` `Sonnet/Haiku` eval and replace the
  stale plain-OBD placeholder completely
- decide whether Tier 5 should appear in the first public package or remain a
  supporting internal lane
- curate any public-facing `evaluation_results.json` excerpts for
  `published/checkpoints/release-candidate/selected-eval-results/`

## Opus Overnight Status

| Lane | Run | Status | Accuracy | Retrieval | Ingest Spend | Eval Spend | Total Spend |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `AL-S` `Opus/Haiku` | `r884` | complete | `89.37%` | `68.28%` | `$27.9538` | `$5.9943` | `$33.9481` |
| `AL-L` `Opus/Haiku` | `r885` | complete | `84.14%` | `63.25%` | `$89.2716` | `$6.0235` | `$95.2951` |

Supporting public checkpoint artifacts are already staged locally under:

- `published/checkpoints/release-candidate/scores/`
- `published/checkpoints/release-candidate/token-usage/`

Read:

- `AL-S Opus/Haiku` is the strongest small-lane topline result in this draft, but it gets there with far higher ingest spend than the Haiku or Sonnet small lanes.
- `AL-L Opus/Haiku` does not become the large-lane winner on the refreshed surface; it lands above refreshed `AL-L Haiku/Haiku` on topline, but below refreshed `AL-L Sonnet/Haiku` on both accuracy and retrieval while costing far more.

## Source of Truth

Working-source counterparts for this draft currently live in the internal runner
docs under `~/quaid/util/agents/codex-benchmark/`.
