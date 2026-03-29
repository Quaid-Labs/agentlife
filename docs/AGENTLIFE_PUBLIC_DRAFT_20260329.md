# AgentLife Benchmark (Public Draft)

This is a launch-facing draft for README/docs refresh. It is intentionally short
and points to the full technical report for reproducibility details.

## What AgentLife Is (and Why It Exists)

AgentLife is a full-lifecycle benchmark for agent memory systems. It is designed
to evaluate the operating conditions real agents face in production:

- long-running multi-session interaction
- context resets between sessions
- retrieval and synthesis under context pressure
- conflicting or stale facts over time
- project-state and longitudinal memory continuity

Traditional memory benchmarks such as LoCoMo and LongMemEval are useful, but
they primarily score retrospective QA-style recall and do not fully measure
agentic lifecycle behavior. AgentLife was built to close that gap and act as the
primary release-gate KPI for memory quality in real agent workflows.

## Quaid Results (Headline)

Headline comparison rows (recommended Quaid lane vs strongest FC Sonnet baseline
and OpenClaw native):

| Lane | Quaid Sonnet/Haiku | FC Sonnet | OpenClaw Native |
| --- | ---: | ---: | ---: |
| AL-S | 87.69% (`r880`) | 92.90% (`r606`) | 69.40% (`oc-native-als-20260315d`) |
| AL-L | 85.82% (`r895`) | 87.70% (`r857`) | 63.06% (`oc-native-all-20260315d`) |

Interpretation:

- Quaid remains close to FC Sonnet while using substantially fewer eval tokens.
- Quaid strongly outperforms OpenClaw native memory on both AL-S and AL-L.
- On AL-L Sonnet-eval study, Quaid reaches 88.69% (`r944`), above FC Sonnet's
  87.70% baseline on the same corpus.

## Canonical Public Links

Use these as canonical launch links:

- Full technical report (current source in repo):
  `published/runbooks/release-candidate/AGENTLIFE_RELEASE_CANDIDATE_20260328.md`
- Public draft summary (this document):
  `docs/AGENTLIFE_PUBLIC_DRAFT_20260329.md`

Proposed hosted URLs on Quaid:

- `https://quaid.ai/benchmarks/agentlife`
- `https://quaid.ai/benchmarks/agentlife/technical-report`

## Source of Truth

All full tables, methodology details, embedding studies, and reproducible run IDs
are in:

- `published/runbooks/release-candidate/AGENTLIFE_RELEASE_CANDIDATE_20260328.md`
