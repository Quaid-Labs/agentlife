# Positioning: AgentLife vs Traditional Memory Benchmarks

## Core Thesis

Traditional memory benchmarks mostly score retrospective QA recall.  
Agent memory in production must support longitudinal work, project continuity,
tool grounding, and temporal correctness under cost constraints.

AgentLife measures that real operating surface.

## What We Optimize For

- Durable memory quality across long histories.
- Correctness under evolving state (stale/contested fact handling).
- Project-grounded answers backed by docs and memory.
- Stable behavior under realistic tool use and token budgets.
- End-to-end reliability (ingest, janitor, retrieval, answer generation).

## Why This Is Different

AgentLife is designed to measure memory as it is used by real agents over time:

- longitudinal work across many sessions
- project-state continuity and tool-grounded answers
- operational behavior under realistic token and maintenance constraints
- product-general performance instead of narrow benchmark-specific tuning

## External Reporting Strategy

- Report external benchmarks for transparency.
- Treat AgentLife as the primary KPI and release gate.
- Publish category-level outcomes, retrieval behavior, and token economics.
- Prefer changes that improve product-general performance, not benchmark-specific hacks.

## Suggested Public Language

"We optimize for memory that helps agents do work over time, not just answer
retrospective quiz questions. AgentLife measures that longitudinal behavior
directly."
