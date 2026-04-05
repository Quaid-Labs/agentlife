# AgentLife Benchmark — Technical Report (2026-04-05)

## Scope

This document is the second public technical report in the AgentLife benchmark series.

It extends the benchmark record established in:
- [AGENTLIFE_TECHNICAL_REPORT_20260329.md](./AGENTLIFE_TECHNICAL_REPORT_20260329.md)

This report focuses on two questions:
1. How do the April 4-5 embedding candidates compare across the three AgentLife benchmark variants?
2. Which Sonnet-eval runs are strong enough to serve as public headline numbers?

## Embedding Backends

| Embedding backend | Memory usage | Dim size |
|---|---:|---:|
| `nomic-embed-text` | `~270 MB` | `768` |
| `qwen3-embedding:4b` | `~3 GB` | `2560` |
| `qwen3-embedding:8b` | `~6 GB` | `4096` |

## Methodology

- Benchmark family: AgentLife
- Benchmark variants in this report:
  - `AL-S`: short, cleaner memory surface
  - `AL-L`: long, noisier distractor-heavy surface
  - `AL-L OBD`: long one-big-day ingestion surface with large dirty chunks
- Accuracy column: total `T1-T5` benchmark accuracy
- Token columns are split into `Ingest Tokens` and `Eval Tokens`.
- DB size: final `memory.db` size after run completion
- `Sonnet/Haiku` means Sonnet ingest with Haiku eval
- `Sonnet/Sonnet` means Sonnet eval on a fixed ingest lineage
- `FC` means full-context baseline with no memory system
- For eval-only lineage reruns, `Ingest Tokens` refers to the historical source-ingest usage from the reused lineage.
- For eval-only lineage reruns, `Eval Tokens` refers only to the current eval pass.
- Eval token counts in this report are current-run-only, not cumulative across the entire lineage history.

## Main Benchmark Matrix

| Run ID | Run Type | Embedding | Accuracy (T1-5) | Ingest Tokens | Eval Tokens | DB Size | Elapsed |
|---|---|---|---:|---:|---:|---:|---:|
| `r606` | `AL-S FC Sonnet` | `none` | `92.90%` | `0` | `29,828,646` | `—` | `—` |
| `r857` | `AL-L FC Sonnet` | `none` | `87.70%` | `0` | `34,596,206` | `—` | `—` |
| `r1074 (r1066)` | `AL-S Sonnet/Haiku` | `nomic-embed-text` | `88.34%` | `1,019,493` | `9,669,825` | `21.98 MiB` | `13m 20s` |
| `r1069` | `AL-S Sonnet/Haiku` | `qwen3-embedding:4b` | `88.16%` | `1,038,978` | `9,575,266` | `55.06 MiB` | `62m 37s` |
| `r1070` | `AL-S Sonnet/Haiku` | `qwen3-embedding:8b` | `87.81%` | `1,034,507` | `9,239,099` | `79.54 MiB` | `62m 44s` |
| `r1078 (r1074)` | `AL-S Sonnet/Sonnet` | `nomic-embed-text` | `92.23%` | `1,019,493` | `9,524,945` | `24.05 MiB` | `17m 06s` |
| `r1071` | `AL-L OBD Sonnet/Haiku` | `nomic-embed-text` | `85.51%` | `759,461` | `8,545,427` | `24.37 MiB` | `71m 13s` |
| `r1072` | `AL-L OBD Sonnet/Haiku` | `qwen3-embedding:4b` | `82.69%` | `651,262` | `8,221,961` | `64.72 MiB` | `65m 26s` |
| `r1073` | `AL-L OBD Sonnet/Haiku` | `qwen3-embedding:8b` | `86.22%` | `744,607` | `8,022,169` | `92.23 MiB` | `73m 23s` |
| `r1075 (r1071)` | `AL-L OBD Sonnet/Sonnet` | `nomic-embed-text` | `89.58%` | `759,461` | `7,935,069` | `26.85 MiB` | `17m 12s` |
| `r1079 (r1076)` | `AL-L Sonnet/Haiku` | `nomic-embed-text` | `80.57%` | `3,978,805` | `10,539,406` | `45.73 MiB` | `13m 04s` |
| `r1081 (r1076)` | `AL-L Sonnet/Sonnet` | `nomic-embed-text` | `87.81%` | `3,978,805` | `9,973,601` | `45.21 MiB` | `17m 55s` |
| `r1082` | `AL-L Sonnet/Haiku` | `qwen3-embedding:4b` | `84.10%` | `4,079,179` | `11,677,615` | `127.74 MiB` | `184m 00s` |
| `r1083 (r1082)` | `AL-L Sonnet/Sonnet` | `qwen3-embedding:4b` | `87.81%` | `4,079,179` | `10,738,576` | `133.89 MiB` | `18m 10s` |

## Sonnet Eval Headline Matrix

The following Sonnet-eval runs are the strongest headline candidates from this study block.

| Run ID | Run Type | Embedding | Accuracy (T1-5) | Ingest Tokens | Eval Tokens | DB Size | Elapsed |
|---|---|---|---:|---:|---:|---:|---:|
| `r606` | `AL-S FC Sonnet` | `none` | `92.90%` | `0` | `29,828,646` | `—` | `—` |
| `r857` | `AL-L FC Sonnet` | `none` | `87.70%` | `0` | `34,596,206` | `—` | `—` |
| `r946` | `AL-L OBD Sonnet/Sonnet` | `qwen3-embedding:8b` | `86.04%` | `0` | `7,449,129` | `98.91 MB` | `18m 13s` |
| `r1078 (r1074)` | `AL-S Sonnet/Sonnet` | `nomic-embed-text` | `92.23%` | `1,019,493` | `9,524,945` | `24.05 MiB` | `17m 06s` |
| `r1075 (r1071)` | `AL-L OBD Sonnet/Sonnet` | `nomic-embed-text` | `89.58%` | `759,461` | `7,935,069` | `26.85 MiB` | `17m 12s` |
| `r1081 (r1076)` | `AL-L Sonnet/Sonnet` | `nomic-embed-text` | `87.81%` | `3,978,805` | `9,973,601` | `45.21 MiB` | `17m 55s` |
| `r1083 (r1082)` | `AL-L Sonnet/Sonnet` | `qwen3-embedding:4b` | `87.81%` | `4,079,179` | `10,738,576` | `133.89 MiB` | `18m 10s` |

## Findings

### 1. `nomic-embed-text` is the strongest default choice

Across the three benchmark variants in this study, `nomic-embed-text` offers the best overall balance of quality, storage footprint, and model RAM.

On `AL-S`:
- `r1074 (r1066)` slightly outperforms the corresponding `qwen3-embedding:4b` run
- `r1078 (r1074)` reaches `92.23%`, effectively matching the historical `AL-S FC Sonnet` ceiling of `92.90%`

On `AL-L OBD`:
- `r1075 (r1071)` establishes a strong Sonnet-eval headline at `89.58%`
- the final DB size remains compact at `26.85 MiB`

On plain `AL-L`:
- `r1079 (r1076)` is the one clear underperforming nomic row at `80.57%` on `Sonnet/Haiku`
- `r1081 (r1076)` recovers to `87.81%` under `Sonnet/Sonnet`, matching the corresponding `qwen3-embedding:4b` headline exactly
- this suggests the plain `AL-L` gap is driven by answer-model sensitivity to retrieval noise, not by a durable embedding-quality advantage for qwen4b

### 2. `qwen3-embedding:8b` is the highest-scoring OBD option, but at a large storage cost

On `AL-L OBD`:
- `r1073` is the best scorer in the main matrix at `86.22%`
- `r1071` with nomic is close behind at `85.51%`

The quality gain is modest:
- `+0.71pp` over nomic on the same surface

The storage tradeoff is large:
- `24.37 MiB` for `nomic-embed-text`
- `92.23 MiB` for `qwen3-embedding:8b`

### 3. `qwen3-embedding:4b` is not the best choice on these release surfaces

`qwen3-embedding:4b` does not win any major surface in this report.

- On `AL-S`, it trails or effectively ties nomic while using much more storage.
- On `AL-L OBD`, it is clearly behind both nomic and qwen8b.
- On plain `AL-L`, it reaches the same Sonnet-eval headline score as nomic, but at nearly triple the DB size.

The plain `AL-L` Sonnet-eval follow-up on `qwen3-embedding:4b` was included specifically to test whether the only notable gap against nomic would close under Sonnet evaluation. It did: the resulting headline number matched the corresponding nomic Sonnet-eval line exactly at `87.81%`, indicating that the plain `AL-L` difference did not persist once both were evaluated with Sonnet.

This report therefore supersedes the earlier recommendation in `AGENTLIFE_TECHNICAL_REPORT_20260329.md` that positioned `qwen3-embedding:4b` as the preferred alternative. On the April 4-5 study block, `qwen3-embedding:4b` does not establish a durable advantage over nomic on any headline surface.

### 4. `AL-L OBD` remains the strongest long-form measurement variant

The `AL-L OBD` Sonnet-eval result in this report is one of the clearest headline numbers in the current benchmark set:
- `r1075 (r1071)` = `89.58%`

This exceeds the earlier OBD Sonnet/Sonnet reference:
- `r946` = `86.04%`

## Recommended Public Headline Positioning

If one embedding backend is to be presented as the default recommendation:
- `nomic-embed-text`

If one short-surface headline number is needed:
- `r1078 (r1074)`
- `AL-S Sonnet/Sonnet`
- `92.23%`

If one long-surface headline number is needed:
- `r1075 (r1071)`
- `AL-L OBD Sonnet/Sonnet`
- `89.58%`

If the report also needs to mention the highest raw OBD score regardless of storage efficiency:
- `r1073`
- `AL-L OBD Sonnet/Haiku`
- `qwen3-embedding:8b`
- `86.22%`

## Notes

- Final DB size is reported only from completed runs.
- For eval-only lineage rows, `Ingest Tokens` refers to the historical ingest cost of the reused source lineage and `Eval Tokens` refers only to the current eval pass.
- FC rows are included as ceiling references for public comparison.

