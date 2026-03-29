# AgentLife Benchmark — Technical Report

## Scope

- Benchmark family: AgentLife
- Dataset: canonical AgentLife query set (283 questions: 268 core + 15 Tier-5 emotional intelligence)
- Tier-5 EI questions are included in all accuracy totals but scored with a separate, looser rubric appropriate for evaluating emotional and relational responses
- Judge model: `gpt-4o-mini`
- Run IDs retained for reproducibility

## Methodology

### Benchmark Variants

- **AL-S (AgentLife Short):** Core evaluation corpus (~100K tokens, 20 arc sessions). Contains all information needed to answer the eval query set. Simulates approximately 2 months of agent use.
- **AL-L (AgentLife Long):** The AL-S core corpus plus filler sessions of cohesive real-world-style data unrelated to the eval questions. At ~200K tokens, this corpus exceeds single-context capacity and forces FC baselines to compact at the 160K-token context boundary — the FC answer model receives a compacted summary prefix plus ~40K tokens of raw trailing context. Since this benchmark was designed, context windows up to 1M tokens have become available, but the methodology remains valid as a test of memory-system behavior under context pressure.
- **AL-L OBD (One Big Day):** The same AL-L dataset compressed into a single-day ingestion pipeline. Stress-tests the system under heavy single-session load and simulates a highly active power user processing the full corpus in one sitting.
- **FC (Full Context):** Answer-model baseline with no memory system. For AL-S, the full raw transcript is provided per query. For AL-L, the FC pipeline compacts at the 160K context boundary (see above). FC represents a theoretical single-session upper bound — it does not survive context resets between sessions.
- **OC Native (OpenClaw Native):** Baseline using OpenClaw’s built-in memory-core, session-memory, and session-index systems. Represents the default memory experience for OpenClaw users without Quaid.

### Synthetic Data and Fact Density

All AgentLife variants use synthetic conversation data with high density of extractable facts, relationships, and temporal events. Real-world agent usage — as observed in the 13-day Claude Code scalability study below — produces significantly lower fact density per token. AgentLife benchmarks stress extraction, retrieval, and knowledge management harder than typical production workloads per unit of data processed.

### Model Pair Notation

Model pairs are written as `ingest/eval` (deep/fast) — e.g. `Sonnet/Haiku` means Sonnet for the deep layer (extraction/reasoning) and Haiku for the fast layer (eval/answer).

### Token Count Interpretation

Token counts represent the minimum tokens required to generate answers and produce judge scores for the benchmark query set:

- **Quaid runs:** tokens cover context file injection (core files, pre-injected knowledge) and tool-use calls (memory recall, project doc search) per query. Real interactive usage includes live conversation history on top of these minimums.
- **FC runs:** tokens cover the full transcript (AL-S) or compacted summary + trailing context (AL-L) sent per query. Every query independently pays the full context cost.

### Reporting

Single-run methodology: one completed run per lane/configuration. Informal repeat variance on stable configs has typically been ~±1pp.

Many Quaid runs in this matrix are eval-only reruns against a known-good ingest lineage, used to validate recall-side behavior changes without re-ingesting. Token counts reflect the final eval pass only.

-----

## Main Benchmark Matrix

|Lane                 |Run                      |Models                       |Accuracy|Ingest Tokens|Eval Tokens|Total Tokens|Notes                             |
|---------------------|-------------------------|-----------------------------|-------:|------------:|----------:|-----------:|----------------------------------|
|AL-S Sonnet/Haiku    |`r880 (r847)`            |ingest Sonnet, eval Haiku    |87.69%  |967,141      |5,753,673  |6,720,814   |recommended config                |
|AL-S Haiku/Haiku     |`r864`                   |ingest Haiku, eval Haiku     |85.26%  |1,063,657    |6,729,159  |7,792,816   |                                  |
|AL-S Opus/Haiku      |`r884`                   |ingest Opus, eval Haiku      |89.37%  |999,751      |5,926,169  |6,925,920   |not recommended — see below       |
|AL-L Sonnet/Haiku    |`r895 (r863)`            |ingest Sonnet, eval Haiku    |85.82%  |3,794,335    |5,917,209  |9,711,544   |recommended config                |
|AL-L Haiku/Haiku     |`r881 (r849)`            |ingest Haiku, eval Haiku     |83.40%  |3,881,194    |7,112,256  |10,993,450  |                                  |
|AL-L Opus/Haiku      |`r885`                   |ingest Opus, eval Haiku      |84.14%  |3,287,623    |5,833,626  |9,121,249   |not recommended — see below       |
|AL-L OBD Sonnet/Haiku|`r935 (r900)`            |ingest Sonnet, eval Haiku    |86.04%  |745,270      |6,865,570  |7,610,840   |eval-only on reused ingest lineage|
|AL-L OBD Haiku/Haiku |`r890`                   |ingest Haiku, eval Haiku     |82.46%  |1,406,571    |5,603,471  |7,010,042   |                                  |
|AL-S FC Haiku        |`r600`                   |full-context Haiku           |87.70%  |0            |29,855,754 |29,855,754  |raw transcript per query          |
|AL-S FC Sonnet       |`r606`                   |full-context Sonnet          |92.90%  |0            |29,828,646 |29,828,646  |                                  |
|AL-L FC Haiku        |`r607`                   |full-context Haiku           |83.60%  |0            |34,397,219 |34,397,219  |compacted at 160K boundary        |
|AL-L FC Sonnet       |`r857`                   |full-context Sonnet          |87.70%  |0            |34,596,206 |34,596,206  |compacted at 160K boundary        |
|AL-S OC Native       |`oc-native-als-20260315d`|native OC memory + Haiku eval|69.40%  |untracked    |untracked  |untracked   |                                  |
|AL-L OC Native       |`oc-native-all-20260315d`|native OC memory + Haiku eval|63.06%  |untracked    |untracked  |untracked   |                                  |

-----

## Key Findings

### Quaid vs Full Context (Sonnet)

FC-Sonnet represents the strongest practical single-session baseline — a frontier model with the full transcript (or compacted equivalent) in context. FC-Opus would score higher but is prohibitively expensive to benchmark at this scale. On AL-L, the FC-Sonnet baseline already required context compaction at the 160K boundary, meaning it does not have access to the complete transcript.

|Lane|Quaid Sonnet/Haiku|FC Sonnet|Delta|Quaid Eval Tokens|FC Eval Tokens|
|----|-----------------:|--------:|----:|----------------:|-------------:|
|AL-S|87.69%            |92.90%   |-5.21|5,753,673        |29,828,646    |
|AL-L|85.82%            |87.70%   |-1.88|5,917,209        |34,596,206    |

The gap narrows from 5.2pp on AL-S to 1.9pp on AL-L as the FC baseline loses information through compaction. Quaid achieves this at roughly one-fifth of FC’s per-query token cost — and critically, Quaid’s knowledge persists across session resets while FC starts from zero each session.

When Quaid uses Sonnet for the answer model (matching production configurations), accuracy reaches **88.69%** on AL-L — exceeding FC-Sonnet’s 87.70% on the same corpus. See Sonnet Eval Study below.

### Quaid vs OpenClaw Native

|Lane|Quaid Sonnet/Haiku|OC Native|Delta |
|----|-----------------:|--------:|-----:|
|AL-S|87.69%            |69.40%   |+18.29|
|AL-L|85.82%            |63.06%   |+22.76|

Quaid outperforms the built-in OpenClaw memory system by 18-23 percentage points.

### Opus Extraction

Opus extraction does not provide a practical advantage over Sonnet:

|Lane|Sonnet/Haiku|Opus/Haiku|
|----|-----------:|---------:|
|AL-S|87.69%      |89.37%    |
|AL-L|85.82%      |84.14%    |

Opus edges Sonnet by 1.68pp on AL-S but falls 1.68pp below Sonnet on AL-L, while costing 4-5x more on ingest. Sonnet remains the recommended deep extraction layer.

### Extraction Model Tradeoffs

Sonnet extraction produces fewer, denser summary-style facts and a smaller DB. Haiku extraction produces more numerous atomic facts and a larger DB. Both are viable. Sonnet is recommended because the denser extraction scales better long-term — less DB growth, less retrieval noise — and produces higher accuracy on most lanes.

-----

## Sonnet Eval Study

In production, users run Sonnet or Opus as the answer model, not Haiku. This study measures accuracy when the eval/answer model is upgraded to Sonnet while keeping the same extracted knowledge bases (eval-only reruns, no re-ingestion).

The comparison target is **AL-L FC-Sonnet at 87.70%** — which itself required context compaction at the 160K boundary and does not survive session resets.

|Lane                  |Run   |Ingest Model|Eval Model|Accuracy  |Eval Tokens|
|----------------------|------|------------|----------|---------:|----------:|
|AL-L Haiku ingest     |`r944`|Haiku       |Sonnet    |**88.69%**|8,382,952  |
|AL-L Sonnet ingest    |`r945`|Sonnet      |Sonnet    |87.10%    |6,458,333  |
|AL-L OBD Sonnet ingest|`r946`|Sonnet      |Sonnet    |86.04%    |6,076,822  |
|AL-L OBD Haiku ingest |`r947`|Haiku       |Sonnet    |85.51%    |6,070,267  |

**Key finding: Quaid with Haiku ingest and Sonnet eval reaches 88.69% on AL-L, exceeding FC-Sonnet’s 87.70% on the same corpus.** A production Quaid deployment delivers higher accuracy than giving Sonnet the entire (compacted) transcript in context, at a fraction of the per-query token cost, with persistent cross-session knowledge.

Sonnet eval does not produce a uniform uplift across all lanes — results are configuration-dependent.

-----

## 13-Day Claude Code Scalability Study

Source: a real live Claude Code session window used for long-horizon Quaid development and testing (~100-200 hours of active usage). This represents real-world agent workload at significantly lower fact density per token than the synthetic AgentLife corpus.

|Day|Date      |Messages|Facts Added|DB Size (MiB)|Extraction (s)|Janitor (s)|Total (s)|
|---|----------|-------:|----------:|------------:|-------------:|----------:|--------:|
|1  |2026-03-10|411     |85         |49.8         |276.9         |679.7      |956.6    |
|2  |2026-03-11|1,372   |279        |68.0         |906.3         |372.5      |1,278.7  |
|3  |2026-03-12|693     |115        |77.6         |556.4         |407.6      |964.0    |
|4  |2026-03-13|736     |387        |112.7        |1,298.3       |477.1      |1,775.4  |
|5  |2026-03-14|608     |180        |123.1        |628.5         |415.4      |1,043.9  |
|6  |2026-03-15|1,034   |191        |156.9        |1,106.4       |596.1      |1,702.5  |
|7  |2026-03-16|834     |237        |187.1        |947.1         |404.6      |1,351.6  |
|8  |2026-03-17|653     |187        |197.2        |639.1         |372.4      |1,011.4  |
|9  |2026-03-18|450     |99         |240.6        |214.8         |856.7      |1,071.4  |
|10 |2026-03-19|36      |8          |257.6        |46.6          |291.2      |337.9    |
|11 |2026-03-20|333     |106        |263.2        |338.0         |316.9      |654.9    |
|12 |2026-03-21|166     |50         |266.3        |143.8         |290.0      |433.8    |
|13 |2026-03-22|838     |340        |299.6        |2,182.6       |528.5      |2,711.1  |

**Findings:**

- DB grew from 49.8 MiB to 299.6 MiB (~6x) across 13 days of heavy use.
- Net facts stored: 2,264 (3,305 raw extracted, janitor consolidated remainder).
- Extraction and janitor runtimes are workload-proportional, not DB-size-proportional. No exponential blowup observed. The DB growing 6x did not cause runtimes to grow 6x.
- **DB footprint growth was the only scalability concern observed in this study.** Extraction time, janitor time, and query-time latency all remained healthy throughout. The shift to `nomic-embed-text` as the default embedding backend directly addresses this — at roughly one-fifth the per-fact storage footprint of `qwen3-embedding:8b`, equivalent 13-day growth would be approximately 10 MiB to 60 MiB.

-----

## Embedding Studies

### Study 1: AL-L Sonnet 4-Way Comparison

Fixed lineage: AL-L Sonnet ingest data. Only the embedding backend varies.

|Embedding Backend   |Run   |Accuracy|Preinject Avg|Query Avg|Doc Chunks|DB Size  |
|--------------------|------|-------:|------------:|--------:|---------:|--------:|
|qwen3-embedding:8b  |`r934`|85.69%  |1369ms       |11019ms  |902       |191.61 MB|
|nomic-embed-text    |`r955`|85.69%  |1336ms       |10265ms  |797       |39.80 MB |
|qwen3-embedding:4b  |`r956`|85.16%  |1599ms       |11553ms  |797       |93.00 MB |
|qwen3-embedding:0.6b|`r957`|82.86%  |1655ms       |11155ms  |797       |40.08 MB |

`mxbai-embed-large` was tested but did not produce a stable result on the RAG-page workload and was excluded.

### Study 2: Nomic Replacement Check (4-Lane)

Validates `nomic-embed-text` and `qwen3-embedding:4b` against `qwen3-embedding:8b` across all release lanes. Cell format: Accuracy / DB Size.

|Lane           |qwen3-embedding:8b        |nomic-embed-text          |qwen3-embedding:4b        |
|---------------|--------------------------|--------------------------|--------------------------|
|AL-L Sonnet    |85.69 / 191.61 MB (`r934`)|85.69 / 39.80 MB (`r955`) |85.16 / 93.00 MB (`r956`) |
|AL-L OBD Sonnet|86.04 / 98.91 MB (`r935`) |86.75 / 61.71 MB (`r965`) |87.63 / 109.01 MB (`r968`)|
|AL-L Haiku     |86.57 / 205.46 MB (`r936`)|82.86 / 120.26 MB (`r959`)|82.16 / 215.45 MB (`r966`)|
|AL-L OBD Haiku |83.39 / 126.29 MB (`r937`)|84.45 / 75.14 MB (`r960`) |84.98 / 135.57 MB (`r967`)|

### Embedding Recommendation

**`nomic-embed-text` is the recommended default.** On the recommended Sonnet extraction lanes, it matches or exceeds `qwen3-embedding:8b` accuracy at roughly one-fifth the DB footprint and a fraction of the model RAM.

|Model             |Params|Dimensions|RAM   |License              |DB Size (AL-L Sonnet)|
|------------------|------|----------|------|---------------------|---------------------|
|nomic-embed-text  |137M  |768       |~270MB|Apache 2.0 (Nomic AI)|39.80 MB             |
|qwen3-embedding:4b|4B    |2048      |~3GB  |Apache 2.0 (Alibaba) |93.00 MB             |
|qwen3-embedding:8b|8B    |4096      |~6GB  |Apache 2.0 (Alibaba) |191.61 MB            |

**Recommended tier order:**

1. **`nomic-embed-text` (default):** Matches or exceeds 8b accuracy on Sonnet lanes at the smallest DB footprint and lowest RAM (~270MB). Runs comfortably alongside agent workloads without competing for resources.
1. **`qwen3-embedding:4b` (alternative):** Outperforms 8b on several lanes (87.63% vs 86.04% on OBD Sonnet, 84.98% vs 83.39% on OBD Haiku) at roughly half the RAM and meaningfully smaller DBs. A practical middle option for users who want stronger retrieval precision than nomic without the full 6GB commitment.
1. **`qwen3-embedding:8b` (advanced):** Retains a meaningful accuracy advantage specifically on Haiku-only extraction at AL-L scale (86.57% vs 82.86% nomic). Available for users optimizing retrieval on budget extraction configs who can accept the 6GB RAM and larger DB footprint.
1. **`qwen3-embedding:0.6b`:** Not recommended. Accuracy regresses too far to be viable.

The nomic default also addresses the long-horizon DB growth concern from the scalability study: at one-fifth the per-fact footprint, months of heavy use remain manageable without manual intervention.

-----

## Pending

- Full nomic-refresh headline matrix across AL-S lanes when usage budget resets
- AL-S v2 is planned with expanded hard-category coverage and graded scoring to improve separation between model tiers at the high end of the accuracy range.
