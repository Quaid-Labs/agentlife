# AgentLife Benchmark — Project Specification

**Author:** Solomon Steadman
**Date:** February 14, 2026
**Status:** Initial specification, ready for implementation

-----

## 1. What This Is

AgentLife is a two-track benchmark for evaluating AI agent memory systems. Its primary purpose is as a development tool — a rigorous test harness for measuring the impact of changes to extraction, retrieval, and janitor systems. Its secondary purpose is as a publishable benchmark that demonstrates capabilities no existing evaluation can test.

The key value proposition for the end user: **a memory system should reduce total cost and increase accuracy of an AI agent over time.** AgentLife measures both.

**Track 1 -- Personal Memory:** Does the agent know who you are?
**Track 2 -- Project Intelligence:** Can the agent build what you see in your head?

The tracks are interleaved in a single continuous simulation, because real users don't separate "personal conversations" from "project work." The interleaving -- and the cross-references between tracks -- is the core methodological innovation.

### Why Existing Benchmarks Fail

LoCoMo (ACL 2024): 10 short synthetic conversations. Each conversation fits in a single markdown file, so distillation alone matches full retrieval. Doesn't test temporal evolution, project tracking, or cross-session inference.

LongMemEval (ICLR 2025): Longer data but still short conversations. No project context, no fact evolution, no contradiction resolution testing.

Both benchmarks test "can you recall a fact from a conversation." AgentLife tests "after 6 weeks of real usage, does the agent still know who you are and what you're working on -- and can it connect the two?"

-----

## 2. Architecture Overview

### Generation (expensive, run once)

Two LLMs simulate a user and an AI agent across ~30 interleaved sessions. The user LLM follows a detailed persona document and per-session briefs. Projects are actually built -- code is written, files created, tests run. The full simulation is captured as:

- Complete conversation transcripts (per session)
- SQLite DB snapshots (per session boundary)
- Workspace filesystem snapshots (per session boundary)
- Janitor run logs (where applicable)
- Unit test results at each checkpoint

This corpus is the published dataset. It never changes after generation.

### Evaluation (cheap, rerun for ablation)

Memory systems ingest the fixed transcripts and answer standardized evaluation queries. Eval queries appear at the end of Track 1 sessions and at project checkpoints in Track 2. Scoring is against ground truth derived from the persona document and session annotations.

### Ablation (cheap, checkpoint-based)

DB and workspace checkpoints at every session boundary enable partial reruns:

|Change Type       |What to Rerun                                |Cost    |
|------------------|---------------------------------------------|--------|
|Recall/retrieval  |Eval queries only against existing DB        |cents   |
|Janitor parameters|Janitor + eval queries from checkpoint       |$       |
|Extraction        |Re-ingest transcripts from checkpoint forward|$$      |
|Full rerun        |Everything                                   |$$      |

-----

## 3. Persona Design

### Primary Persona: Maya Chen

A detailed, internally consistent persona document serves as ground truth. The driving LLM references this document but never shares it directly with the agent. The persona includes:

**Demographics & Life:**

- 34-year-old product manager at a mid-size SaaS company in Austin, TX
- Partner: David (software engineer, vegetarian)
- Dog: Biscuit (golden retriever, 3 years old)
- Mom: Linda (recently diagnosed with Type 2 diabetes, lives in Houston)
- Sister: Rachel (lives in Seattle, has two kids: Ethan age 7, Lily age 4)
- David's family: his mother (wants to visit Austin), his brother Mike (also plans to visit)
- Training for a half marathon (Austin Half, target date in the simulation timeline)
- Coworker Priya mentioned occasionally (NOT family -- red herring for graph tests)

**Personality & Communication Style:**

- Terse, direct communicator. Sends short messages. Gets impatient with long explanations.
- Circles back to topics days later without re-establishing context
- Sometimes contradicts herself and corrects later
- Uses nicknames and abbreviations ("D" for David, "Rach" for Rachel)
- Has strong aesthetic opinions but describes them in non-technical language

**Professional Arc:**

- Increasingly unhappy at current job (TechFlow)
- Starts interviewing at Stripe around session 6
- Gets the offer around session 14
- Accepts and transitions by session 15+

**Projects (Track 2 specs):**

- Project A: Recipe app (motivated by mom's diabetes diagnosis, evolves over time)
- Project B: Portfolio site (motivated by job search, connects to professional arc)
- Project C: YouTube script about the recipe app journey (creative/non-technical, cross-references almost everything)
- Project D: Web research for nutrition API (agent-retrieved knowledge, tests memory of what the agent found vs what the user said)

**Key Cross-References:**

- Mom's diabetes -> dietary restriction feature in recipe app
- Job search -> portfolio site creation
- David's vegetarianism -> recipe app dietary features
- Half marathon training -> health/nutrition awareness -> recipe app scope

-----

## 4. Session Schedule

~30 sessions, interleaved Track 1 and Track 2. Each session brief tells the driving LLM what Maya wants to accomplish, what cross-references to make, and when to end the session.

### Session Structure

Each session:

1. Opens with a `/new` command (forces context reset, triggers extraction)
2. Maya initiates naturally
3. Conversation unfolds per the session brief
4. Session ends when the brief's goal is met OR after ~20 messages
5. Track 1 sessions end with 2-4 eval queries woven in naturally
6. Track 2 sessions end at defined checkpoints with test suite runs

### Draft Session Schedule

```
Session  1: [T1] Maya introduces herself. Mentions David, Biscuit, job, mom.
Session  2: [T1] Half marathon training. Mentions a race with David last year.
             TANGENT: Asks about a good podcast app.
Session  3: [T2] "I want to build a recipe app." Project A begins. Phase 1.
Session  4: [T1] Mom's diabetes diagnosis. Emotional. Mentions Linda's in Houston.
             TANGENT: Needs to book flights for a Houston visit.
Session  5: [T2] Recipe app continued. "Dietary restrictions are personal for me."
             Cross-ref: session 4, but Maya just says "you know my mom's situation."
Session  6: [T1] Job frustration at TechFlow. Considering leaving.
             TANGENT: Austin traffic, thinking about moving closer to downtown.
             Mentions coworker Priya (RED HERRING: NOT family).
Session  7: [T2] Recipe app frontend. First round of user feedback.
             TANGENT: David tried a test recipe, it was terrible. Ordered Thai instead.
Session  8: [T1] David planning surprise birthday for Linda. Needs restaurant help.
             Cross-ref: Linda's diabetes (dietary constraints for restaurant).
Session  9: [T2] "Different thing -- help me build a portfolio site." Project B begins.
             Cross-ref: "thinking about my options" (references session 6).
Session 10: [T2] Recipe app -- back to Project A. Phase 2 begins.
             Pivot: "Let's switch the database approach." Checkpoint A.
Session 11: [T1] Marathon training update. Injury scare. Mentions Rachel visiting.
             TANGENT: Stretching routines, mentions she hates yoga.
Session 12: [T2] Portfolio site round 2. User corrections.
Session 13: [T2] Recipe app. Schema change for meal planning.
             TANGENT: "David suggested a grocery list feature."
Session 13b:[T2-D] Web research: "need a nutrition API for the app."
Session 14: [T1] "I got the Stripe offer." Resolves job arc.
Session 15: [T2] Portfolio site -- "update it, I'm going to Stripe now."
Session 16: [T1] Rachel visits Austin. Maya talks about her nephews.
             SLOW DRIFT: Starts about Rachel, drifts into childhood memories.
Session 17: [T2] Recipe app Phase 3. Add sharing features.
             [T2-D] "what was that API you found? the one with the diet labels?"
Session 18: [T1] David got promoted. Talking about moving.
             CONFLICTING SOURCES: "I want Zilker but David wants East Austin."
Session 19: [T2] Recipe app. Pivot: REST to GraphQL.
             Checkpoint B. TOOL USE MEMORY: nutrition calculation.
Session 20: [T1] Mom's diabetes management improving. Update on Linda's health.
             TANGENT: Mom tried a recipe from the app.
             PRIVACY BOUNDARY: Mentions fight with David about money casually.
Session 20b:[T1] REMEMBER NOTHING session. Maya speculates about quitting, Portugal, cat.
Session 21: [T2-C] YouTube video project begins. Cross-refs everything.
             MULTI-USER ATTRIBUTION: David's suggestions vs Maya's.
Session 22: [T1] Half marathon race day results. David cheering.
             David's brother Mike texted congrats, might visit.
Session 23: [T2] Recipe app. Add authentication. "David wants to use it too."
             Cross-ref: David is vegetarian.
             [T2-D] David suggests FoodData -- conflicts with prior research.
Session 24: [T1] First week at Stripe. Impressions.
             AGENT WAS WRONG: Bad advice about remote work.
             TOOL USE CALLBACK: "What was the calorie count you calculated?"
Session 25: [T1] Surprise callback: "Whatever happened with David's plan for mom's birthday?"
Session 26: [T2-C] YouTube script revision. "Use the story about mom."
             [T2-D] Re-research APIs -- v2 pages now served.
Session 27: [T2] Portfolio site -- add Stripe role. YouTube tone corrections.
Session 28: [T2-C] YouTube script final version. Maya approves.
Session 29: [T2] Recipe app. Final checkpoint. Full test suite.
Session 30: [T1] Natural life check-in. 10-15 eval queries covering all arcs.
```

-----

## 5. Track 2 -- Project Specifications

### Constraints

- **User LLM (Maya):** Cannot write code. All prompts <= 50 tokens (hard target, 60 ceiling).
- **Agent LLM:** Sonnet (fixed for all runs).
- **Session resets:** `/new` between every session. Agent relies on memory system.

### Project A: Recipe App

**Phase 1 (Sessions 3, 5, 7):** Core CRUD - SQLite, REST API, basic frontend. Test Suite A: 8 tests.
**Phase 2 (Sessions 10, 13, 17):** Dietary filters, meal planning, DB pivot. Test Suite B: 12 tests.
**Phase 3 (Sessions 19, 23, 26, 29):** GraphQL pivot, auth, docs, deployment. Test Suite C: full suite.

### Project B: Portfolio Site

**Phase 1 (Sessions 9, 12):** Static site with About, Projects, Contact. Test Suite D.
**Phase 2 (Sessions 15, 21, 27):** Update to Stripe, add project showcase. Test Suite E.

### Project C: YouTube Script (Non-Technical)

Creative writing project cross-referencing almost every thread. LLM judge rubric: mentions personal motivation, references career transition, no code/jargon, within word count, mentions David/family. Quality dimensions: tone, narrative arc, personal authenticity, cross-reference density, revision incorporation.

### Project D: Web Research (Agent-Retrieved Knowledge)

Controlled versioned web pages. v1 (sessions 1-19): Edamam is best option. v2 (sessions 20+): Edamam drops free tier, FoodData adds dietary labels -- recommendation flips. Tests agent-retrieved knowledge persistence, temporal volatility, and conflict detection with user suggestions.

-----

## 6. Evaluation Framework

### 16 Eval Query Types

1. Factual recall
2. Temporal/current
3. Evolution
4. Cross-reference
5. Multi-session synthesis
6. Inference
7. Negative/epistemic
8. Stale fact
9. Surprise callback
10. Project state
11. Tangent recall
12. Agent-retrieved
13. Graph traversal (by hop count: 1, 2, 3+)
14. Speaker attribution
15. Contested facts
16. Sensitivity (penalty-based)

### Scoring

LLM judge (GPT-4o-mini, temperature 0): CORRECT (1.0), PARTIAL (0.5), WRONG (0.0).

**No composite score in v1.** Report all metrics independently. Add weighted composite in v2 after score distributions reveal natural weightings.

### Cost Metrics (First-Class)

- Extraction cost, Retrieval cost, Janitor cost, Embedding cost
- Total memory system cost
- Agent token savings
- **Net cost impact** (headline metric: memory cost minus token savings)
- Retrieval precision (relevant facts / injected facts)

-----

## 7-15. Model Matrix, Implementation, Stress Patterns, Publication

See SPEC-FULL.md sections for complete details on:
- Model matrix (Opus user, Opus/Sonnet agent, Haiku/Opus eval, GPT-4o-mini judge)
- Technical implementation (directory structure, transcript format, checkpoint format, driving LLM architecture)
- Memory stress patterns (fact evolution, contradiction, cross-session inference, surprise callback, project state, stale facts, agent-retrieved knowledge, graph traversal, tool use memory, contested facts, agent self-correction, privacy boundaries, remember nothing, multi-user attribution, decay resistance, conversation tangents)
- Scoring methodology (Track 1 per-type accuracy, Track 2 test suites + project knowledge, cost metrics, retrieval precision, statistical rigor)
- Comparability (fixed transcripts, same judge, report requirements)
- Implementation plan (5 phases, 7 days estimated)
- Publication plan (GitHub repo, ArXiv preprint, blog post)
- Known risks & mitigations (14 identified risks)
- Success criteria (7 criteria including discrimination, net cost reduction, ablation utility)

-----

*Specification version: 1.0*
