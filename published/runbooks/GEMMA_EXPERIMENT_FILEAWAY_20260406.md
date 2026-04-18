# Gemma Experiment File-Away (2026-04-06)

## Scope
This note captures current Gemma local-eval findings so work can resume later without re-deriving setup constraints.

## ALS Nomic Comparator Snapshot
| Run | Configuration | Accuracy (T1-T5) | Eval Tokens | Time |
|---|---|---:|---:|---:|
| r1074 (r1066) | Anthropic Haiku eval | 88.34% | 9,669,825 | 13m 20s |
| r1078 (r1074) | Anthropic Sonnet eval | 92.23% | 9,524,945 | 17m 06s |
| r1124 (r1074) | Local Gemma 26B split lane | 83.75% | 6,686,695 | 53.2m |
| r1128 (r1074) | Local Gemma 26B single-lane p16 | 83.04% | 6,540,290 | 46.5m |

## What We Learned
- Single-lane 26B p16 was faster than split-lane 26B on this lineage (~46.5m vs ~53.2m) but similar accuracy.
- Mixed Q6 (31B deep + 26B fast/judge) is sensitive to fast-lane per-slot context budget; too low causes judge overflow.
- Failures were primarily operational (auth/env mismatch, timeout tails, queue pressure), not a single deterministic model defect.
- Bounded in-flight eval queue and explicit stall watchdog improved observability and prevented silent pseudo-progress.

## Deferred Gemma Work
- Continue mixed-Q6 sweep only after OpenAI release block is complete.
- Keep priority on stable, repeatable runs over broad thread sweeps.

## Immediate Priority Shift
Next benchmark block after active mixed run completion is OpenAI route:
- ALS/ALL/ALLOBD with deep=gpt-5.4, fast=gpt-5.3-codex-spark, judge=gpt-5.4-mini
- fast-alt A/B with fast=gpt-5.4-mini
