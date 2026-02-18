# AgentLife Benchmark — Project Specification

**Author:** Solomon Steadman
**Date:** February 14, 2026
**Status:** Spec reviewed, phased approach approved. Pilot gate before full build.

-----

## 1. What This Is

AgentLife is a two-track benchmark for evaluating AI agent memory systems. Its primary purpose is as a development tool — a rigorous test harness for measuring the impact of changes to extraction, retrieval, and janitor systems. Its secondary purpose is as a publishable benchmark that demonstrates capabilities no existing evaluation can test.

The key value proposition for the end user: **a memory system should reduce total cost and increase accuracy of an AI agent over time.** AgentLife measures both.

**Track 1 — Personal Memory:** Does the agent know who you are?
**Track 2 — Project Intelligence:** Can the agent build what you see in your head?

The tracks are interleaved in a single continuous simulation, because real users don't separate "personal conversations" from "project work." The interleaving — and the cross-references between tracks — is the core methodological innovation.

### Why Existing Benchmarks Fail

LoCoMo (ACL 2024): 10 short synthetic conversations. Each conversation fits in a single markdown file, so distillation alone matches full retrieval. Doesn't test temporal evolution, project tracking, or cross-session inference.

LongMemEval (ICLR 2025): Longer data but still short conversations. No project context, no fact evolution, no contradiction resolution testing.

Both benchmarks test "can you recall a fact from a conversation." AgentLife tests "after 6 weeks of real usage, does the agent still know who you are and what you're working on — and can it connect the two?"

-----

## 2. Architecture Overview

### Generation (expensive, run once)

Two LLMs simulate a user and an AI agent across ~20 interleaved sessions (v1). The user LLM follows a detailed persona document and per-session briefs. Projects are actually built — code is written, files created, tests run. The full simulation is captured as:

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
- Circles back to topics days later without re-establishing context ("hey, about the thing with the auth...")
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
- Project C: YouTube script about the recipe app journey (creative/non-technical, cross-references almost everything). Deferred to v1.1 -- merged into session 21 as a brief creative task.
- Project D: Web research for nutrition API (agent-retrieved knowledge, tests memory of what the agent found vs what the user said)

**Key Cross-References (persona doc annotations):**

- Mom's diabetes -> dietary restriction feature in recipe app
- Job search -> portfolio site creation
- David's vegetarianism -> recipe app dietary features
- Half marathon training -> health/nutrition awareness -> recipe app scope

The persona document should be 2,000-3,000 words, detailed enough that any eval query can be answered from it. Include a timeline of what Maya knows/feels at each point in the simulation.

### Persona Document Structure

```
# Maya Chen -- Persona Document (GROUND TRUTH)

## Identity
[Demographics, relationships, personality traits]

## Communication Style
[How she talks, message length, patterns, quirks]

## Life Timeline
[Week-by-week state: what's happening in her life]

## Project A: Recipe App -- Full Spec
[Complete technical spec across all phases, never revealed all at once]

## Project B: Portfolio Site -- Full Spec
[Complete spec, including how it relates to job search]

## Cross-Reference Map
[Explicit connections between personal facts and project decisions]

## Family Relationship Graph (Ground Truth)
[Complete family tree with all relationships and attributes per node.
 Used for scoring graph traversal queries. Includes: which session
 each relationship was revealed, whether it was stated directly or
 must be inferred, and the minimum hops required to answer each
 graph traversal eval query.]

## Correction Points
[Places where Maya will contradict herself and later correct]

## Eval Ground Truth
[For each eval query: the correct answer and supporting evidence]
```

-----

## 4. Session Schedule

~20 sessions (v1), interleaved Track 1 and Track 2. Each session brief tells the driving LLM what Maya wants to accomplish, what cross-references to make, and when to end the session.

### Session Structure

Each session:

1. Opens with a `/new` command (forces context reset, triggers extraction)
2. Maya initiates naturally ("hey, need help with something" / "back on the recipe app")
3. Conversation unfolds per the session brief
4. Session ends when the brief's goal is met OR after ~20 messages, whichever comes first
5. Track 1 sessions end with 2-4 eval queries woven in naturally
6. Track 2 sessions end at defined checkpoints with test suite runs

### Track 2: Test-Suite-as-Spec

Track 2 sessions include the test suite as part of the session brief. Maya tells the agent what the acceptance criteria are: "here are the tests your code needs to pass." This mirrors how real users work -- they communicate requirements, including concrete expectations. The tests still objectively pass or fail, but the agent knows what it's aiming for. This makes Track 2 reproducible across different memory systems.

### Draft Session Schedule (v1 -- 20 sessions)

```
Session  1: [T1] Maya introduces herself. Mentions David, Biscuit, job, mom.
Session  2: [T1] Half marathon training. Mentions a race she ran with David last year.
             TANGENT: Asks about a good podcast app -- bored during long runs.
Session  3: [T2] "I want to build a recipe app." Project A begins. Phase 1.
             Test suite A provided as acceptance criteria.
Session  4: [T1] Mom's diabetes diagnosis. Emotional. Mentions Linda's in Houston.
             TANGENT: Remembers she needs to book flights for a Houston visit.
Session  5: [T2] Recipe app continued. "Dietary restrictions are personal for me."
             Cross-ref: session 4, but Maya just says "you know my mom's situation."
Session  6: [T1] Job frustration at TechFlow. Considering leaving. Doesn't name Stripe yet.
             TANGENT: Complains about Austin traffic, thinking about moving closer to downtown.
             Mentions coworker Priya in passing (RED HERRING: NOT family).
Session  7: [T2] Recipe app frontend. First round of user feedback on output.
             TANGENT: "David tried one of the test recipes, it was terrible. We ordered
              Thai from that place on South Congress instead."
Session  8: [T1] David planning surprise birthday for Linda. Needs restaurant help.
             Cross-ref: Linda's diabetes (dietary constraints for restaurant).
Session  9: [T2] "Different thing -- help me build a portfolio site." Project B begins.
             Cross-ref: "thinking about my options" (references session 6 job frustration).
             Test suite D provided as acceptance criteria.
Session 10: [T2] Recipe app -- back to Project A after gap. Phase 2 begins.
             Pivot: "Let's switch the database approach." Checkpoint A -> Suite A must pass.
             Test suite B provided for Phase 2.
Session 11: [T1] Marathon training update. Injury scare. Mentions Rachel visiting soon.
             TANGENT: Asks about stretching routines, mentions she hates yoga.
Session 12: [T2] Recipe app. Schema change for meal planning feature.
             TANGENT: "David suggested a grocery list feature -- he does most of our shopping."
             [T2-D] "need a nutrition API for the app. something with dietary labels."
             Web research task. Agent hits controlled pages (localhost), reports findings.
Session 13: [T1] "I got the Stripe offer." Resolves job arc from session 6.
Session 14: [T2] Portfolio site -- "update it, I'm going to Stripe now."
             Cross-ref: session 13, but Maya just says "update the company stuff."
             Test suite E: must say Stripe, not TechFlow.
Session 15: [T1] Rachel visits Austin. Maya talks about her nephews.
             SLOW DRIFT: Conversation starts about Rachel, drifts into Maya's childhood
              memories, never comes back to original topic.
Session 16: [T2] Recipe app Phase 3. Pivot: REST to GraphQL. Add sharing features.
             [T2-D] "what was that API you found? the one with the diet labels?"
             Tests recall of agent-retrieved research from session 12.
             Checkpoint B -> Suite B must pass. Test suite C provided for Phase 3.
Session 17: [T1] David got promoted. Talking about moving.
             CONFLICTING SOURCES: "I want Zilker but David wants East Austin."
             PRIVACY BOUNDARY: Mentions fight with David about money casually.
Session 18: [T2] Recipe app. Add authentication. "David wants to use it too."
             Cross-ref: David is vegetarian (mentioned once in session 1 or 2).
             [T2-D] David suggests FoodData -- conflicts with prior research.
Session 19: [T1] Mom's diabetes management improving. Half marathon results.
             Cross-ref: recipe app dietary features were motivated by this.
             TANGENT: Mom tried a recipe from the app -- "she actually used it, I almost cried."
             Surprise callback: "Whatever happened with David's plan for mom's birthday?"
Session 20: [T1] Natural life check-in. Maya reflects on the past few weeks.
             10-15 eval queries woven into natural conversation, covering all major
             arcs: job, family, projects, David. NOT a quiz -- a real catch-up chat.
```

### v1.1 Expansion Sessions (deferred)

```
Session 20b:[T1] REMEMBER NOTHING session. Epistemic status test.
Session 21: [T2-C] YouTube video project. Cross-refs everything.
             MULTI-USER ATTRIBUTION: David's suggestions vs Maya's.
Session 22: [T2-D] Re-research APIs with v2 pages. Recommendation flips.
Session 23: [T1] Career update. AGENT WAS WRONG test. TOOL USE CALLBACK.
```

### Session Brief Format

Each session brief provided to the driving LLM:

```yaml
session: 5
track: 2
project: recipe-app
phase: 1
goal: "Continue recipe app. Add dietary restriction filtering feature."
cross_references:
  - session: 4
    topic: "mom's diabetes"
    how: "Don't explain the connection. Just say 'you know my mom's situation'
          or 'this is personal for me.' The agent should connect it."
tangents: []
test_suite: null  # Track 1 sessions don't have test suites
new_information:
  - "Maya wants filters for: diabetic-friendly, vegetarian, low-sodium"
  - "She wants a 'safe for mom' quick filter"
corrections: []
eval_queries: []  # Track 2 uses checkpoint test suites instead
end_condition: "Agent has implemented dietary filtering or 20 messages reached"
```

Example session brief with tangent:

```yaml
session: 7
track: 2
project: recipe-app
phase: 1
goal: "Get first round of frontend feedback. Maya reviews what the agent built."
cross_references: []
tangents:
  - trigger: "After Maya gives her first piece of feedback on the frontend"
    topic: "David tried one of the test recipes last night and it was terrible.
            They ended up ordering Thai from that place on South Congress."
    extractable_facts:
      - "David cooks sometimes / tests recipes"
      - "They like a Thai restaurant on South Congress"
    return: "Maya says 'ok anyway' and goes back to the frontend feedback"
test_suite: null  # Continuation, not a checkpoint
new_information:
  - "Maya thinks the layout is too cramped"
  - "She wants bigger images for recipes"
corrections: []
eval_queries: []
end_condition: "Agent has updated the frontend based on feedback or 20 messages reached"
```

-----

## 5. Track 2 -- Project Specifications

### Constraints

- **User LLM (Maya):** Cannot write code. Speaks like an engineer but in natural language only. All prompts <=50 tokens (hard target, 60 ceiling -- regenerate if exceeded).
- **Agent LLM:** Sonnet (fixed for all runs). Good enough to eventually pass test suites, weak enough to require iteration.
- **Session resets:** `/new` between every session. Agent must rely on memory system for project continuity.
- **Test-suite-as-spec:** Test suites are provided to the agent as part of the session brief. Maya tells the agent the acceptance criteria. The tests still objectively pass or fail, but the agent knows what it's aiming for.

### Project A: Recipe App

**Phase 1 (Sessions 3, 5, 7):** Core CRUD

- SQLite database with recipes table
- REST API: GET/POST/PUT/DELETE recipes, GET /search
- Basic frontend: list view, add recipe form
- **Test Suite A:** 8 endpoint tests, database CRUD verification. Provided to agent in session 3.

**Phase 2 (Sessions 10, 12, 16):** Features + First Pivot

- Add dietary restriction filtering (diabetic, vegetarian, low-sodium)
- Meal planning feature (weekly plan, grocery list generation)
- Session 10 pivot: change database approach (e.g., add proper migrations)
- **Test Suite B:** 12 tests including dietary filters, meal planning. Provided at session 10.

**Phase 3 (Sessions 16, 18):** Advanced + Second Pivot

- Session 16 pivot: REST to GraphQL
- Session 18: Add authentication (JWT, user accounts)
- **Test Suite C:** Full suite -- GraphQL queries, auth, sharing. Provided at session 16.

### Project B: Portfolio Site

**Phase 1 (Sessions 9, 12):** Initial build

- Static site with About, Projects, Contact sections
- Current job info (TechFlow at this point)
- **Test Suite D:** Page renders, content checks, responsive. Provided at session 9.

**Phase 2 (Sessions 14):** Updates

- Session 14: Update company to Stripe (after offer acceptance)
- **Test Suite E:** Content accuracy (must say Stripe, not TechFlow). Provided at session 14.

### Project C: YouTube Script (v1.1)

Deferred to v1.1 expansion. In v1, a brief creative cross-reference task can be woven into a late Track 1 session to test narrative synthesis without a full project.

### Project D: Web Research (Agent-Retrieved Knowledge)

Controlled web pages served from localhost during generation. Three API comparison pages with specific, verifiable facts. Pages versioned (v1 for sessions 1-16, v2 for sessions 17+) to test temporal volatility.

During generation: simple Python localhost server serves the pages.
For external evaluation: research results are already embedded in the transcripts. Other systems don't need to re-run the research -- they just need to extract and retain what was found from the conversation text.

**v1 pages:** Edamam is the clear winner (dietary labels, cheap free tier).
**v2 pages (v1.1):** Edamam drops free tier, FoodData adds dietary labels. Recommendation flips.

-----

## 6. Evaluation Framework

### Eval Query Types (16 types)

|Type                       |Example                                                             |Tests                                    |
|---------------------------|--------------------------------------------------------------------|-----------------------------------------|
|**Factual recall**         |"What is Maya's dog's name?"                                        |Basic extraction and retention           |
|**Temporal/current**       |"Where does Maya work?"                                             |Must return Stripe, not TechFlow         |
|**Evolution**              |"How did Maya's job situation change?"                              |Tracks arc from frustration -> offer     |
|**Cross-reference**        |"What motivated the dietary restriction feature?"                   |Connects mom's diabetes to recipe app    |
|**Multi-session synthesis**|"What projects has Maya been working on?"                           |Assembles info across 20+ sessions       |
|**Inference**              |"What dietary constraints for a family dinner?"                     |Connects: mom=diabetic, David=vegetarian |
|**Negative**               |"Does Maya have children?" / "Is Maya moving to Portugal?"          |Hallucination resistance                 |
|**Stale fact**             |"What company is on Maya's portfolio site?"                         |Must be Stripe, not TechFlow             |
|**Surprise callback**      |"What was David planning for Linda?"                                |Long-range retention, session 8 -> 19    |
|**Project state**          |"What's the current API style?"                                     |Current state of code projects           |
|**Tangent recall**         |"What restaurant on South Congress?"                                |Extracting facts from conversation noise |
|**Agent-retrieved**        |"Which API has diabetic filtering?"                                 |Facts found by agent via research        |
|**Graph traversal**        |"Who are Maya's nephews?"                                           |Multi-hop relationship inference (1/2/3+)|
|**Speaker attribution**    |"What features has David suggested?"                                |Distinguishes who said what              |
|**Contested facts**        |"Where are Maya and David thinking about moving?"                   |Tracks multiple valid positions          |
|**Sensitivity**            |(Scored negatively if agent surfaces fight-about-money unsolicited) |Retrieval appropriateness                |

### Scoring

LLM judge (GPT-4o-mini, temperature 0):

```
CORRECT:  Full match with ground truth (1.0)
PARTIAL:  Contains correct info but incomplete or has extras (0.5)
WRONG:    Missing, incorrect, or hallucinated (0.0)
```

**No composite score in v1.** Report all metrics independently. Add a weighted composite in v2 after score distributions reveal natural weightings.

### Cost Metrics (First-Class, Every Results Table)

|Metric                        |What It Measures                                                  |
|------------------------------|------------------------------------------------------------------|
|**Extraction cost**           |LLM calls for fact/edge/snippet/journal extraction per session    |
|**Retrieval cost**            |LLM calls for reranking, HyDE expansion per query                |
|**Janitor cost**              |LLM calls for review, dedup, contradiction resolution, distillation|
|**Embedding cost**            |Ollama (free/local) or API embedding costs                        |
|**Total memory system cost**  |Sum of above across all 20 sessions                               |
|**Agent token savings**       |Tokens saved vs re-reading full transcript history                |
|**Net cost impact**           |Total memory cost MINUS agent token savings (headline metric)     |
|**Memory tax per session**    |Tokens injected per agent turn (marginal overhead)                |

**Net cost impact is the headline metric.** A memory system that costs $5 to run but saves $20 in agent context tokens has a net impact of -$15 (saves money).

**Memory tax per session** captures marginal overhead: a system adding 500 tokens per turn is meaningfully different from one adding 2,000, even at the same total cost.

### Retrieval Precision

For every eval query, measure not just correctness but retrieval efficiency:

|Metric                 |What It Measures                                         |
|-----------------------|---------------------------------------------------------|
|**Facts injected**     |How many memory facts were put into the agent's context  |
|**Facts relevant**     |Of those, how many were relevant to the query (LLM judge)|
|**Retrieval precision**|relevant / injected                                      |
|**Context tokens used**|Total tokens of injected memory context                  |

### Statistical Rigor

- Wilson Score 95% confidence intervals (matching LoCoMo methodology)
- Minimum 3 eval runs for published numbers
- Per-category breakdowns, not just overall
- Raw per-query results in published dataset for reproducibility
- Cost figures reported to 2 decimal places

-----

## 7. Model Matrix

|Run Type          |User LLM|Agent LLM|Answer/Eval LLM|Memory Config|Cost     |
|------------------|--------|---------|---------------|-------------|---------|
|Track 1 generation|Opus    |Opus     |--             |Canonical    |$$ (once)|
|Track 2 generation|Opus    |Sonnet   |--             |Canonical    |$ (once) |
|Track 2 ablation  |Opus    |Sonnet   |--             |Variant      |$ (ckpt) |
|Eval ablation     |--      |--       |Haiku          |Variant      |cents    |
|Eval capstone     |--      |--       |Opus           |Winner       |$ (once) |

**Fixed across all runs:** User LLM is always Opus. Agent LLM is always Sonnet for Track 2. Judge model is always GPT-4o-mini. These are constants, not variables.

-----

## 8. Technical Implementation

### Directory Structure

```
agentlife/
+-- README.md
+-- persona/
|   +-- maya.md                    # Full persona document (ground truth)
|   +-- maya-timeline.md           # Week-by-week state
|   +-- cross-references.md        # Annotated connections
+-- sessions/
|   +-- briefs/
|   |   +-- session-01.yaml        # Per-session driving instructions
|   |   +-- session-02.yaml
|   |   +-- ...
|   +-- transcripts/               # Generated (output of simulation)
|       +-- session-01.jsonl
|       +-- session-02.jsonl
|       +-- ...
+-- projects/
|   +-- recipe-app/
|   |   +-- spec.md                # Full spec (ground truth, never shown to agent)
|   |   +-- tests/
|   |       +-- suite-a/           # Phase 1 tests (provided to agent)
|   |       +-- suite-b/           # Phase 2 tests (cumulative, provided to agent)
|   |       +-- suite-c/           # Phase 3 tests (cumulative, provided to agent)
|   +-- portfolio-site/
|   |   +-- spec.md
|   |   +-- tests/
|   |       +-- suite-d/
|   |       +-- suite-e/
|   +-- web-research/
|       +-- spec.md
|       +-- pages/
|       |   +-- v1/                # Served during sessions 1-16
|       |   +-- v2/                # Served during sessions 17+ (v1.1)
|       +-- ground-truth.json
+-- eval/
|   +-- queries.json               # All eval queries with ground truth
|   +-- judge.py                   # LLM judge (GPT-4o-mini)
|   +-- scorer.py                  # Per-metric scoring (no composite in v1)
|   +-- cost_tracker.py            # Track all cost components
|   +-- report.py                  # Generate results tables
+-- checkpoints/                   # Generated (DB + workspace snapshots)
|   +-- session-01/
|   |   +-- memory.db
|   |   +-- workspace.tar.gz
|   |   +-- janitor.log
|   +-- session-02/
|   +-- ...
+-- runner/
|   +-- simulate.py                # Full simulation runner
|   +-- driving_llm.py             # Maya persona driver
|   +-- checkpoint.py              # Snapshot/restore utilities
|   +-- ingest.py                  # Feed transcripts to a memory system
|   +-- ablation.py                # Run from checkpoint with variant config
|   +-- web_server.py              # Localhost server for research pages
|   +-- config.yaml                # Model selections, API keys, paths
+-- results/
    +-- canonical/                 # Results from the canonical generation run
    +-- ablations/                 # Results from variant runs
```

### Transcript Format

Each session is a JSONL file:

```jsonl
{"type": "system", "command": "/new", "timestamp": "2026-02-01T09:00:00Z"}
{"type": "user", "content": "hey, I need help with the recipe app", "timestamp": "2026-02-01T09:00:05Z", "token_count": 10}
{"type": "assistant", "content": "Sure, what are you working on?", "timestamp": "2026-02-01T09:00:08Z"}
{"type": "eval", "query": "What motivated the dietary restriction feature?", "ground_truth": "Maya's mother Linda was diagnosed with Type 2 diabetes", "query_type": "cross-reference", "evidence_sessions": [4, 5], "timestamp": "2026-02-01T09:15:00Z"}
{"type": "checkpoint", "suite": "suite-a", "results": {"passed": 8, "failed": 0, "total": 8}, "timestamp": "2026-02-01T09:16:00Z"}
```

-----

## 9. Memory Stress Patterns

These patterns are engineered into the session schedule:

**Fact Evolution:** Session 6 (unhappy at TechFlow) -> Session 9 (thinking about options) -> Session 13 (Stripe offer) -> Session 20 (at Stripe). Query: "Where does Maya work?" must return Stripe at session 13+.

**Contradiction & Correction:** Maya states half marathon is in April (session 2), corrects to May (session 11). System must not return April after session 11.

**Cross-Session Inference:** David is vegetarian (session 2), mom is diabetic (session 4), marathon training (sessions 2/11). Query: "What dietary constraints for a family dinner?" requires assembling three facts from three sessions.

**Surprise Callback:** Session 8: David planning surprise birthday for Linda. Not mentioned until session 19: "Whatever happened with David's plan?" Tests long-range retention.

**Project State Tracking:** Recipe app evolves: SQLite -> migrations (session 10 pivot) -> GraphQL (session 16 pivot) -> auth (session 18). Query: "What's the current tech stack?" must reflect latest state.

**Stale Fact Detection:** Portfolio says TechFlow (sessions 9-12). Must say Stripe after session 14 update.

**Agent-Retrieved Knowledge:** Session 12: Agent researches nutrition APIs. Session 16: Maya asks "what was that API?" Tests recall of agent-found information (not user-stated).

**Graph Traversal:** Relationships scattered across sessions. 1-hop: "Who is Maya's partner?" 2-hop: "Who are Maya's nephews?" (Maya -> Rachel -> Ethan/Lily). 3-hop: "What's Linda's relationship to Ethan?"

**Contested Facts:** "I want Zilker but David wants East Austin." Both true simultaneously. Must track both positions with attributions.

**Speaker Attribution:** "David thinks the video should have a demo section" -- David's opinion, not Maya's. "David says we should add dark mode" -- David's feature request.

**Privacy Boundary:** Maya mentions fight with David about money. System may extract it but should not surface it unsolicited in unrelated contexts. Scored negatively if it leaks.

**Tangent Recall:** Facts dropped mid-tangent (restaurant name, David cooks, dog parks). Tests extraction from conversation noise.

**Decay Resistance:** Mom's diabetes connected to active project (recipe app). Despite gaps between mentions, the connection to an active project should protect it from decay.

### v1.1 Stress Patterns (deferred)

**Remember Nothing:** Maya says "I'm just thinking out loud, none of this is real." Speculates about quitting, Portugal, cat. Tests epistemic status handling.

**Agent Self-Correction:** Maya tells agent its prior advice was wrong. Tests self-referential quality tracking.

**Tool Use Memory:** Agent calculates nutrition values. Maya asks for the number days later. Tests recall of agent's own tool output.

**Temporal Volatility:** Re-research same web pages with changed content. Agent must recognize stale information.

-----

## 10. Scoring Methodology

**No composite score in v1.** Report all metrics independently.

### Track 1: Personal Memory Score

LLM judge per query type (16 types). Report accuracy per type.

### Track 2: Project Intelligence Score

|Metric                  |Measurement                             |Notes                       |
|------------------------|----------------------------------------|----------------------------|
|Test suite pass rate    |% of tests passing at each checkpoint   |Binary per test, % per suite|
|Sessions to completion  |# sessions to reach each checkpoint     |Lower = better              |
|Token efficiency        |Total tokens across all project sessions|Lower = better              |
|Project state queries   |LLM judge on project knowledge questions|Same judge as Track 1       |

### Cost Metrics (every table)

Example results table format:

```
| System       | Track 1 Acc | Track 2 Pass | Memory Cost | Token Savings | Net Cost | Tax/Turn |
|-------------|-------------|--------------|-------------|---------------|----------|----------|
| Quaid       | 78%         | 85%          | $4.20       | -$19.40       | -$15.20  | 1,200    |
| No memory   | 45%         | 52%          | $0.00       | $0.00         | $0.00    | 0        |
| Full context| 92%         | 90%          | $0.00       | N/A           | +$42.00  | 45,000   |
```

-----

## 11. Comparability with Other Systems

Provide the fixed transcript corpus. Other memory systems ingest transcripts through their own pipelines. All systems answer the same eval queries. Score with the same judge.

Requirements for comparable results:
- Agent model: Sonnet (Track 2)
- Judge model: GPT-4o-mini, temperature 0
- Report both Haiku and Opus eval numbers
- Full transcript corpus ingested (no cherry-picking)
- Per-query-type accuracy, not just overall
- Total memory system cost and net cost impact alongside accuracy
- Retrieval precision (facts injected vs facts relevant)
- Memory tax per turn

-----

## 12. Implementation Plan (Phased with Pilot Gate)

### Phase 0: Pilot (2 days, ~$20) -- GATE

Non-negotiable. Run before committing to full build.

- [ ] Write Maya persona doc (abbreviated -- enough for 5 sessions)
- [ ] Write session briefs 1-5 (exposes session brief dependency graph)
- [ ] Prototype driving LLM -- validate 50-token constraint, persona consistency
- [ ] Run 5 sessions (T1, T1, T2, T1, T2) through Quaid
- [ ] Write 20 eval queries, run judge
- [ ] **Decision gate:** Does the benchmark discriminate? Is Track 2 viable? Does 50 tokens work?

**If pilot fails on Track 2:** Ship Track 1 only, redesign Track 2 with adjusted approach, pilot that separately.
**If pilot works:** Proceed to Phase 1.

### Phase 1: Track 1 Complete (5 days, ~$100)

Track 1 alone is publishable and more novel than LoCoMo or LongMemEval.

- [ ] Full persona doc (3K words)
- [ ] All Track 1 session briefs (~10 sessions)
- [ ] All Track 1 eval queries (~80-100)
- [ ] Runner infrastructure (driving LLM, simulator, checkpoint, judge, scorer, cost tracker)
- [ ] Generate Track 1 transcripts
- [ ] Run baselines (no-memory, full-context, Quaid)
- [ ] Validate discrimination
- [ ] **Ship Track 1 preview** -- publish dataset, start getting external validation

### Phase 2: Track 2 (5 days, ~$100)

Only after Track 1 ships and pilot validates Track 2 viability.

- [ ] Recipe app spec + test suites (A, B, C)
- [ ] Portfolio site spec + test suites (D, E)
- [ ] Track 2 session briefs (~10 sessions, interleaved)
- [ ] Web research pages (localhost, static files)
- [ ] Generate full 20-session transcripts (interleaved T1/T2)
- [ ] Project knowledge eval queries
- [ ] Run full evaluation
- [ ] **Ship v1** -- complete benchmark with both tracks

### Phase 3: v1.1 Expansion (deferred)

- [ ] YouTube script project (Project C)
- [ ] Remember nothing session
- [ ] Agent self-correction tests
- [ ] Web research v2 pages (temporal volatility)
- [ ] Expansion sessions 20b-23
- [ ] ArXiv preprint

### Budget

| Phase | Cost | Time |
|-------|------|------|
| Pilot | ~$20 | 2 days |
| Track 1 | ~$100 | 5 days |
| Track 2 | ~$100 | 5 days |
| v1.1 | ~$50 | 3 days |
| **Iteration buffer** | **~$100** | -- |
| **Total** | **~$370** | **~2 weeks** |

-----

## 13. Publication Plan

### GitHub Repository

Public repo: `agentlife-bench` (or similar)

Contains: persona document, session briefs, generated transcripts, eval queries with ground truth, test suites, runner code, scoring code, results.

Does NOT contain: API keys, proprietary memory system internals, Quaid source code.

### ArXiv Preprint (v1.1)

Title: "AgentLife: A Long-Horizon Benchmark for AI Agent Memory Systems"

### Blog Post (with Track 1)

Narrative version: "We built a benchmark that tests what 6 weeks of AI memory actually looks like." Lead with the finding. Include comparison table. Link to dataset.

-----

## 14. Known Risks & Mitigations

|Risk                                               |Mitigation                                                          |
|----------------------------------------------------|--------------------------------------------------------------------|
|Maya persona drift across 20 sessions               |Opus at temp 0 + strong system prompt; pilot validates first 5      |
|Token limit (50) too restrictive for Opus            |Allow up to 60 as hard ceiling; pilot tests regeneration rate       |
|Sonnet can't complete projects with memory-only ctx  |Test-suite-as-spec; pilot validates Track 2 with 3 sessions         |
|Test suites too brittle                              |Test behavior not implementation; provide to agent as acceptance criteria|
|Eval queries ambiguous                               |Pilot with first 20 queries before writing the rest                 |
|Benchmark doesn't discriminate                       |Run no-memory vs full-context baselines first to verify spread      |
|Cost overrun on generation                           |$100 iteration buffer; checkpoint aggressively                      |
|Session briefs hard to write (dependency graph)      |Pilot exposes difficulty; budget 3-4 days, not 2                    |
|Memory cost exceeds token savings (net positive ROI) |Valid finding, not a failure; track per-component to identify cause  |
|Track 2 not reproducible across systems              |Primary metric is project-knowledge queries; pass rates secondary   |

-----

## 15. Success Criteria

The benchmark is successful if:

1. **Discrimination:** No-memory baseline scores <=50% on Track 1, Quaid scores >=70%, and the gap is statistically significant (non-overlapping 95% CIs).
2. **Net cost reduction:** Quaid's total memory system cost is less than the token savings it provides. Net cost impact is negative.
3. **Cross-reference queries show the largest gaps** between memory-equipped and memoryless systems. This validates the core innovation.
4. **Track 2 shows measurable efficiency gains** -- memory-equipped agent reaches checkpoints in fewer sessions and fewer tokens.
5. **At least 3 eval query types show meaningful discrimination** between different memory system architectures (not just memory vs no-memory).
6. **Ablation utility:** Retrieval-only ablation takes <$1 and <5 minutes. Janitor ablation takes <$10 and <30 minutes. Checkpoint system works reliably.
7. **Reproducible** -- a second lab can ingest published transcripts, run eval, and get scores within reported confidence intervals.

### Baseline Validation (Run Before Publishing)

**No-memory baseline:** Agent has no memory. Every session starts fresh. Should score <=50% on Track 1.

**Perfect-memory baseline:** Agent receives the full persona document as context. Should score >=90%. Theoretical ceiling.

**Full-transcript baseline:** Agent receives all prior transcripts as context. Scores highest but costs the most. The "brute force" approach that memory systems should beat on cost while approaching on accuracy.

-----

*Specification version: 1.1 (phased approach, 20 sessions, pilot gate)*
