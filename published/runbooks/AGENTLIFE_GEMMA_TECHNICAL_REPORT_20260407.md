# AGENTLIFE_GEMMA_TECHNICAL_REPORT_20260407

## Scope

This report compares local Gemma 4 eval runs against the Anthropic Haiku/Sonnet eval pairs on the same nomic lineages:

- `AL-S` lineage: `r1074`
- `AL-L OBD` lineage: `r1071`
- `AL-L` lineage: `r1076`

All rows use `nomic-embed-text` and include historical ingest tokens + current eval tokens for eval-only lineage runs.

## Full Matrix

| Surface | Run ID | Eval Family | Variant | T1-T5 Accuracy | T1-T4 Accuracy | Ingest Tokens | Eval Tokens | Total Tokens | DB Size | Elapsed | Status |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `AL-S` | `r1074` | `Anthropic Haiku` | `Haiku eval-only` | `88.34%` | `88.99%` | `1,019,493` | `9,669,825` | `10,689,318` | `21.98 MiB` | `13.3m` | `done` |
| `AL-S` | `r1078 (r1074)` | `Anthropic Sonnet` | `Sonnet eval-only` | `92.23%` | `92.91%` | `1,019,493` | `9,524,945` | `10,544,438` | `24.05 MiB` | `17.1m` | `done` |
| `AL-S` | `r1128 (r1074)` | `Gemma local` | `Gemma26 single-lane` | `83.04%` | `82.09%` | `1,019,493` | `6,540,290` | `7,559,783` | `22.94 MiB` | `46.5m` | `done` |
| `AL-S` | `r1124 (r1074)` | `Gemma local` | `Gemma26 split d9/f4` | `83.75%` | `83.58%` | `1,019,493` | `6,686,695` | `7,706,188` | `22.72 MiB` | `53.2m` | `done` |
| `AL-S` | `r1139 (r1138)` | `Gemma local` | `Gemma31/26 split d5/f4` | `89.74%` | `—` | `1,019,493` | `9,256,741` | `10,276,234` | `22.83 MiB` | `—` | `done` |
| `AL-L OBD` | `r1071` | `Anthropic Haiku` | `Haiku full` | `85.51%` | `86.19%` | `759,461` | `8,545,427` | `9,304,888` | `24.37 MiB` | `71.2m` | `done` |
| `AL-L OBD` | `r1075 (r1071)` | `Anthropic Sonnet` | `Sonnet eval-only` | `89.58%` | `89.74%` | `759,461` | `7,935,069` | `8,694,530` | `26.85 MiB` | `17.2m` | `done` |
| `AL-L OBD` | `r1165 (r1071)` | `Gemma local` | `Gemma26 single-lane` | `84.63%` | `85.07%` | `759,461` | `5,066,485` | `5,825,946` | `24.77 MiB` | `48.4m` | `done` |
| `AL-L OBD` | `r1166 (r1071)` | `Gemma local` | `Gemma26 split d9/f4` | `83.92%` | `84.33%` | `759,461` | `4,999,273` | `5,758,734` | `24.71 MiB` | `45.1m` | `done` |
| `AL-L` | `r1079 (r1076)` | `Anthropic Haiku` | `Haiku eval-only` | `80.57%` | `80.41%` | `3,978,805` | `10,539,406` | `14,518,211` | `45.73 MiB` | `13.1m` | `done` |
| `AL-L` | `r1081 (r1076)` | `Anthropic Sonnet` | `Sonnet eval-only` | `87.81%` | `87.87%` | `3,978,805` | `9,973,601` | `13,952,406` | `45.21 MiB` | `17.9m` | `done` |
| `AL-L` | `r1162 (r1076)` | `Gemma local` | `Gemma26 single-lane` | `79.86%` | `79.10%` | `3,978,805` | `7,161,726` | `11,140,531` | `44.08 MiB` | `59.2m` | `done` |
| `AL-L` | `r1163 (r1076)` | `Gemma local` | `Gemma26 split d9/f4` | `79.51%` | `78.92%` | `3,978,805` | `7,102,714` | `11,081,519` | `44.71 MiB` | `51.6m` | `done` |
| `AL-L` | `r1172 (r1076)` | `Gemma local` | `Gemma31/26 split d4/f4` | `83.75%` | `83.21%` | `3,978,805` | `6,404,341` | `10,383,146` | `44.00 MiB` | `260.4m` | `done` |

## Pair Comparison (Haiku/Sonnet vs Gemma)

| Surface | Haiku Pair (T1-T5) | Sonnet Pair (T1-T5) | Best Completed Gemma (T1-T5) | Gemma vs Haiku | Gemma vs Sonnet |
|---|---:|---:|---:|---:|---:|
| `AL-S` | `88.34%` (`r1074`) | `92.23%` (`r1078`) | `89.74%` (`r1139`) | `+1.40pp` | `-2.49pp` |
| `AL-L OBD` | `85.51%` (`r1071`) | `89.58%` (`r1075`) | `84.63%` (`r1165`) | `-0.88pp` | `-4.95pp` |
| `AL-L` | `80.57%` (`r1079`) | `87.81%` (`r1081`) | `83.75%` (`r1172`) | `+3.18pp` | `-4.06pp` |

## Key Findings

1. Gemma local is currently strongest on `AL-S`, where mixed `31/26` reached `89.74%` and beat the Haiku pair, but still trails Sonnet pair by `2.49pp`.
2. On `AL-L`, mixed `31/26` materially improved over both completed Gemma 26 variants and finished above the Haiku eval pair, but still remained `4.06pp` below the Sonnet eval pair.
3. On `AL-L OBD`, the completed Gemma 26 variants remain below both Anthropic pair anchors.
4. Token spend for Gemma eval is materially lower than Anthropic eval pairs on the same lineages, but wall-clock runtime is much longer, especially for the mixed `31/26` AL-L run.

## Notes

- Accuracy source: `scores.json` (`scores.overall.accuracy` for T1-T5, `scores.overall_t1_t4.accuracy` for T1-T4).
- Elapsed source: run launch log (`Total elapsed: ...`).
- Eval-only rows intentionally carry historical ingest tokens from the reused lineage and current-run eval tokens.
- Where `T1-T4` or `Elapsed` is unavailable in artifacts, the cell is left as `—`.
