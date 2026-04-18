# Published Runbooks

Put release-reviewed benchmark summaries here.

During prep, stage the pending public markdown in `release-candidate/`. Once
the release tag is final, rename or copy that material into a tag-shaped
directory or dated top-level file.

Suggested naming:

- `2026-03-28-current-surface-matrix.md`
- `2026-03-28-release-runbook.md`

Current public technical reports:

- `AGENTLIFE_TECHNICAL_REPORT_20260329.md`
- `AGENTLIFE_TECHNICAL_REPORT_20260405.md`
- `AGENTLIFE_TECHNICAL_REPORT_20260418.md`

Operator guide:

- `AGENTLIFE_BENCHMARK_RUN_GUIDE.md`

Current public headline table:

|        | Quaid Sonnet/Sonnet | FC Sonnet | OpenClaw Native |
|--------|---------------------|-----------|-----------------|
| AL-S   | `92.23%`            | `92.90%`  | `69.40%`        |
| AL-L   | `87.81%`            | `87.70%`  | `63.06%`        |
| AL-L OBD | `89.58%`          | `87.70%`  | `unknown`       |

Token accounting standard for public rows (effective April 18, 2026):

- Use `eval_tokens_ex_judge` from `token_usage.json`:
  - `eval_tokens_ex_judge = eval.total_tokens - sum(by_source[*judge*].total_tokens)`
- This includes answer + preinject + tool-recall token spend, excluding judge.
- Legacy `evaluation_results.json` per-question `eval_tokens` sums are answer-only
  and should be labeled as legacy if referenced.

These files should be the public-facing counterparts to the internal benchmark
runner notes, not raw scratch logs.
