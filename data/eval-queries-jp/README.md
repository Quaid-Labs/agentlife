# AgentLife AL-S Japanese Eval Query Corpus

This directory is a standalone Japanese eval-query corpus for later W3 wiring.
It is data only: no harness, parser, or dataset-loading logic has been changed.

## Files

- `al-s-arc-section4-queries.json`
  - Parsed from current `data/sessions-jp/session-XX-review-v*.txt` Section 4 eval queries.
  - Preserves parser-derived metadata: `query_num`, `source_session`, `session_date`,
    `review_version`, `review_file`, `query_type`, `recall_difficulty`, and
    `evidence_sessions`.
  - Count: 156 queries.

- `al-s-python-query-sets.json`
  - Translated corpus for the Python-defined query sets currently appended by
    `eval.dataset`.
  - Source sets:
    - `ADVERSARIAL_QUERIES`: 75
    - `NON_QUESTION_QUERIES`: 12
    - `ARCHITECTURE_QUERIES`: 17
    - `HARDENING_V2_QUERIES`: 8
    - `EMOTIONAL_INTELLIGENCE_QUERIES`: 15
  - Total: 127 queries.

## Translation Policy

User-facing text fields are localized to Japanese:

- `question`
- `ground_truth`
- `supporting_evidence`
- `sensitivity_context`
- `rubric` score text

Machine-readable fields are preserved:

- IDs and query numbers
- source set names
- query types
- recall difficulties and tiers/categories
- source and evidence session numbers
- dates and numeric values

Code literals, API names, library names, framework names, company names, and
product names are preserved naturally, for example `GraphQL`, `REST`, `SQLite`,
`PostgreSQL`, `JWT`, `Docker`, `TechFlow`, and `Stripe`.

Canonical Japanese names used here:

- マヤ
- マヤ・チェン
- デイビッド
- レイチェル
- レイチ
- リンダ
- イーサン
- リリー
- マイク
- プリヤ
- ビスケット

## Notes For W3

The current production harness still reads arc queries from review Section 4 and
Python-defined query sets from `eval.dataset`. These JSON files are intentionally
not wired into the harness yet. They are structured so later wiring can choose
between arc-only, Python-set-only, or combined Japanese eval profiles.
