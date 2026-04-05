# AgentLife Benchmark — Run Guide

## Scope

- Benchmark family: AgentLife
- Audience: benchmark operators running AgentLife on Spark or a comparable remote host
- Canonical benchmark repo: `~/agentlife-benchmark`
- Canonical checkpoint repo: `~/quaidcode/benchmark-checkpoint`
- Canonical launcher: `scripts/launch-remote-benchmark.sh`

This guide is the operator companion to [AGENTLIFE_TECHNICAL_REPORT_20260329.md](./AGENTLIFE_TECHNICAL_REPORT_20260329.md). Use the technical report for headline results and ceilings. Use this guide to run, rerun, monitor, and debug benchmarks safely.

## Core Rules

- Launch every benchmark through `scripts/launch-remote-benchmark.sh`.
- Never reuse an existing run directory for a new execution.
- Use fresh incremented run IDs for every new launch, including eval-only reruns.
- Eval-only reruns on prior ingest data must be labeled as lineage:
  - `rNEW (rSOURCE)`
- Treat benchmark failures as fail-hard:
  - fix the real cause
  - rerun with a fresh run ID
- Keep benchmark behavior in the harness repo only:
  - `~/agentlife-benchmark`

## Benchmark Variants

- `AL-S`
  - short AgentLife corpus
  - fastest full benchmark lane
- `AL-L`
  - long AgentLife corpus with filler sessions
  - stresses retrieval under distractor load
- `AL-L OBD`
  - the long corpus compressed into one-big-day ingest
  - strongest stress test for heavy-session ingestion and retrieval
- `FC`
  - full-context baseline with no memory system

## Supported Run Profiles

This section is the operator map for the run families we actively use.

|Profile|Scale|Ingest / Eval|Backend|Status|Primary Use|
|---|---|---|---|---|---|
|Sonnet / Haiku|`AL-S`, `AL-L`, `AL-L OBD`|Sonnet deep, Haiku fast|`oauth`|current stable|main release lanes|
|Haiku / Haiku|`AL-S`, `AL-L`, `AL-L OBD`|Haiku deep, Haiku fast|`oauth` or `api`|current stable|budget and speed comparison|
|Sonnet eval-only|`AL-S`, `AL-L`, `AL-L OBD`|reuse ingest, Sonnet eval|`oauth`|current stable|headline ceiling on a fixed ingest base|
|Single-model local Gemma|usually `AL-S` or eval-only lineage|Gemma answer + Gemma judge|`llama-cpp`|current stable after April 5 reasoning fix|local baseline and smoke tests|
|31B / 26B local hybrid|usually eval-only lineage first|31B answer, 26B runtime/judge|`llama-cpp` split endpoints|experimental|mixed local frontier mapping|

Notes:

- `AL-S`, `AL-L`, and `AL-L OBD` should all have explicit launch recipes.
- Sonnet / Haiku is the main production comparison lane.
- Haiku / Haiku is the speed and cost floor.
- Single-model local Gemma currently means the proven Spark lane:
  - `gemma-4-26b-q6k`
- Hybrid 31B / 26B is intentionally documented now because the harness already supports separate answer/runtime/judge endpoints, even though the run shape is still being tuned.

## Canonical Paths

- Local benchmark repo:
  - `/Users/clawdbot/agentlife-benchmark`
- Local checkpoint repo:
  - `/Users/clawdbot/quaidcode/benchmark-checkpoint`
- Remote benchmark repo:
  - `~/agentlife-benchmark`
- Remote checkpoint repo:
  - `~/quaid/benchmark-checkpoint`
- Remote plugin mirror:
  - `~/clawd/plugins/quaid`
- Spark dashboard:
  - `http://192.168.0.139:8765/`

## Standard Launch Pattern

All benchmark flags go after `--`.

```bash
cd ~/agentlife-benchmark
./scripts/launch-remote-benchmark.sh --remote spark -- \
  --mode full \
  --backend oauth
```

## Common Calls

### 1. Fresh Full Run

Example: `AL-S` Sonnet ingest, Haiku eval, fresh full run.

```bash
cd ~/agentlife-benchmark
BENCHMARK_EMBEDDINGS_PROVIDER=ollama \
BENCHMARK_OLLAMA_URL=http://127.0.0.1:11434 \
BENCHMARK_EMBEDDING_MODEL=nomic-embed-text \
BENCHMARK_EMBEDDING_DIM=768 \
./scripts/launch-remote-benchmark.sh --remote spark --scale s -- \
  --mode full \
  --backend oauth \
  --results-dir runs/quaid-s-rXXXX-YYYYMMDD-als-sonnet-nomic-full
```

### 1a. Fresh Full `AL-L` Sonnet / Haiku

```bash
cd ~/agentlife-benchmark
BENCHMARK_EMBEDDINGS_PROVIDER=ollama \
BENCHMARK_OLLAMA_URL=http://127.0.0.1:11434 \
BENCHMARK_EMBEDDING_MODEL=nomic-embed-text \
BENCHMARK_EMBEDDING_DIM=768 \
./scripts/launch-remote-benchmark.sh --remote spark --scale l -- \
  --mode full \
  --backend oauth \
  --results-dir runs/quaid-l-rXXXX-YYYYMMDD-all-sonnet-nomic-full
```

### 1b. Fresh Full `AL-L OBD` Sonnet / Haiku

```bash
cd ~/agentlife-benchmark
BENCHMARK_EMBEDDINGS_PROVIDER=ollama \
BENCHMARK_OLLAMA_URL=http://127.0.0.1:11434 \
BENCHMARK_EMBEDDING_MODEL=nomic-embed-text \
BENCHMARK_EMBEDDING_DIM=768 \
./scripts/launch-remote-benchmark.sh --remote spark --scale l -- \
  --mode full \
  --backend oauth \
  --ingest-schedule obd \
  --results-dir runs/quaid-l-rXXXX-YYYYMMDD-all-obd-sonnet-nomic-full
```

### 1c. Fresh Full Haiku / Haiku

Use this for speed-floor and budget comparison.

```bash
cd ~/agentlife-benchmark
BENCHMARK_EMBEDDINGS_PROVIDER=ollama \
BENCHMARK_OLLAMA_URL=http://127.0.0.1:11434 \
BENCHMARK_EMBEDDING_MODEL=nomic-embed-text \
BENCHMARK_EMBEDDING_DIM=768 \
./scripts/launch-remote-benchmark.sh --remote spark --scale s -- \
  --mode full \
  --backend api \
  --model claude-haiku-4-5-20251001 \
  --eval-model claude-haiku-4-5-20251001 \
  --judge gpt-4o-mini \
  --results-dir runs/quaid-s-rXXXX-YYYYMMDD-als-haiku-nomic-full
```

### 2. Eval-Only Rerun on Existing Ingest

Use this when ingest is good and only eval needs to be rerun.

Steps:
1. copy the source run directory to a fresh target run directory on Spark
2. remove eval artifacts from the target
3. launch `--mode eval` against the new target

Remote copy example:

```bash
ssh spark 'python3 - <<'"'"'PY'"'"'
from pathlib import Path
import shutil
src = Path("~/agentlife-benchmark/runs/quaid-l-r1071-20260404-all-obd-sonnet-nomic-full").expanduser()
dst = Path("~/agentlife-benchmark/runs/quaid-l-r1075-20260405-r1071-all-obd-nomic-sonnet-eval").expanduser()
shutil.copytree(src, dst, symlinks=True)
for name in [
    "scores.json", "evaluation_results.json", "final_scores.json",
    "tier5_results.json", "token_usage.json",
    "benchmark.status.json", "benchmark_summary.json",
]:
    p = dst / name
    if p.exists() or p.is_symlink():
        p.unlink()
PY'
```

Launch example:

```bash
cd ~/agentlife-benchmark
./scripts/launch-remote-benchmark.sh --remote spark -- \
  --mode eval \
  --backend oauth \
  --results-dir runs/quaid-l-r1075-20260405-r1071-all-obd-nomic-sonnet-eval
```

### 2a. Sonnet Eval-Only Headline on `AL-S`

```bash
cd ~/agentlife-benchmark
./scripts/launch-remote-benchmark.sh --remote spark --scale s -- \
  --mode eval \
  --backend oauth \
  --results-dir runs/quaid-s-rXXXX-YYYYMMDD-rSOURCE-als-nomic-sonnet-eval
```

### 2b. Sonnet Eval-Only Headline on `AL-L`

```bash
cd ~/agentlife-benchmark
./scripts/launch-remote-benchmark.sh --remote spark --scale l -- \
  --mode eval \
  --backend oauth \
  --results-dir runs/quaid-l-rXXXX-YYYYMMDD-rSOURCE-all-nomic-sonnet-eval
```

### 2c. Sonnet Eval-Only Headline on `AL-L OBD`

```bash
cd ~/agentlife-benchmark
./scripts/launch-remote-benchmark.sh --remote spark --scale l -- \
  --mode eval \
  --backend oauth \
  --results-dir runs/quaid-l-rXXXX-YYYYMMDD-rSOURCE-all-obd-nomic-sonnet-eval
```

### 3. Local llama.cpp Eval Run

Use for Gemma or other local OpenAI-compatible lanes.

```bash
cd ~/agentlife-benchmark
BENCHMARK_LLAMA_CPP_URL=http://127.0.0.1:30001 \
BENCHMARK_LLAMA_CPP_MODEL=gemma-4-26b-q6k \
BENCHMARK_LLAMA_CPP_API_KEY=localtest \
BENCHMARK_LLAMA_CPP_JUDGE_URL=http://127.0.0.1:30001 \
BENCHMARK_LLAMA_CPP_JUDGE_MODEL=gemma-4-26b-q6k \
BENCHMARK_LLAMA_CPP_JUDGE_API_KEY=localtest \
OPENAI_COMPAT_ANSWER_TIMEOUT_S=600 \
BENCHMARK_TIER5_JUDGE_THINKING=off \
./scripts/launch-remote-benchmark.sh --remote spark --skip-local-checks -- \
  --mode eval \
  --backend llama-cpp \
  --relax-timeouts \
  --allow-non-haiku-answer-model \
  --judge gemma-4-26b-q6k \
  --results-dir runs/quaid-l-rXXXX-YYYYMMDD-rSOURCE-local-eval
```

### 3a. Single-Model Local Gemma Full or Eval-Only

This is the current single-model local lane. The proven Spark recipe today uses:

- answer model: `gemma-4-26b-q6k`
- judge model: `gemma-4-26b-q6k`
- backend: `llama-cpp`

If a smaller single-model Gemma lane is added later, keep the same operator shape and swap:

- model path
- `BENCHMARK_LLAMA_CPP_MODEL`
- server port if needed

### 3b. Upcoming 31B / 26B Local Hybrid

The harness already supports separate answer, runtime, and judge endpoints.

Operator shape:

```bash
cd ~/agentlife-benchmark
BENCHMARK_LLAMA_CPP_API_KEY=localtest \
BENCHMARK_LLAMA_CPP_URL=http://127.0.0.1:30002 \
BENCHMARK_LLAMA_CPP_MODEL=gemma-4-31b-q8 \
BENCHMARK_LLAMA_CPP_RUNTIME_URL=http://127.0.0.1:30001 \
BENCHMARK_LLAMA_CPP_RUNTIME_MODEL=gemma-4-26b-q6k \
BENCHMARK_LLAMA_CPP_RUNTIME_API_KEY=localtest \
BENCHMARK_LLAMA_CPP_JUDGE_URL=http://127.0.0.1:30001 \
BENCHMARK_LLAMA_CPP_JUDGE_MODEL=gemma-4-26b-q6k \
BENCHMARK_LLAMA_CPP_JUDGE_API_KEY=localtest \
OPENAI_COMPAT_ANSWER_TIMEOUT_S=600 \
BENCHMARK_TIER5_JUDGE_THINKING=off \
./scripts/launch-remote-benchmark.sh --remote spark --skip-local-checks -- \
  --mode eval \
  --backend llama-cpp \
  --relax-timeouts \
  --allow-non-haiku-answer-model \
  --judge gemma-4-26b-q6k \
  --results-dir runs/quaid-l-rXXXX-YYYYMMDD-rSOURCE-hybrid-local-eval
```

Current status:

- supported by harness
- recommended to map with small samples first
- do not treat as production-stable until a full eval clears cleanly

## Local Gemma Callbook

### Correct Gemma llama.cpp Server Flags

For Gemma 4 local tool-use evals, use:

```bash
/home/solomon/llama.cpp/build/bin/llama-server \
  -m /home/solomon/models/hf/unsloth-gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q6_K.gguf \
  --host 0.0.0.0 --port 30001 \
  --parallel 16 --ctx-size 962560 \
  --threads 16 --threads-batch 16 \
  --n-gpu-layers 999 --batch-size 2048 --ubatch-size 1024 \
  --jinja --reasoning on --api-key localtest
```

Important:

- Do not run Gemma tool-use evals with:
  - `--reasoning-format none`
- That setting caused `peg-gemma4` parse failures on tool-capable requests during April 5 local mixed-eval testing.

### Direct Smoke Calls

Health:

```bash
curl -fsS http://127.0.0.1:30001/health
```

Forced tool-call smoke:

```bash
curl -sS \
  -H 'Authorization: Bearer localtest' \
  -H 'Content-Type: application/json' \
  --data '{
    "model": "gemma-4-26b-q6k",
    "messages": [
      {"role": "system", "content": "You must call the recall tool before answering."},
      {"role": "user", "content": "What does Maya do for work?"}
    ],
    "tool_choice": "required",
    "max_tokens": 180,
    "tools": [{
      "type": "function",
      "function": {
        "name": "recall",
        "description": "Recall memory",
        "parameters": {
          "type": "object",
          "properties": {"query": {"type": "string"}},
          "required": ["query"]
        }
      }
    }]
  }' \
  http://127.0.0.1:30001/v1/chat/completions
```

Expected success shape:

- HTTP `200`
- `finish_reason: "tool_calls"`
- `message.tool_calls` populated
- no `Failed to parse input at pos 13`

### Verified Benchmark Smoke

The corrected Gemma local server passed:

- `r1091 (r1071)`
- base: `AL-L OBD` nomic
- lane: local `26B p16`
- sample: `5`
- result:
  - `5/5 correct`
  - `100.0%`
  - no retry storm
  - no Gemma parser `500`s

## Monitoring Calls

### Dashboard

- Spark dashboard:
  - `http://192.168.0.139:8765/`

### Launch Log Tail

```bash
ssh spark 'tail -n 120 ~/agentlife-benchmark/runs/<run>.launch.log'
```

### Per-Run Monitor

```bash
~/quaidcode/util/scripts/bench-monitor.sh spark \
  ~/agentlife-benchmark/runs/<run>
```

### Quick Health Probes

```bash
ssh spark 'curl -fsS http://127.0.0.1:30001/health'
ssh spark 'curl -fsS http://127.0.0.1:11434/api/tags >/dev/null'
ssh spark 'nvidia-smi'
```

## Token and Artifact Outputs

Every finished eval run should preserve:

- `scores.json`
- `evaluation_results.json`
- `token_usage.json`
- `tier5_results.json` when Tier 5 runs
- `final_scores.json` when full closeout completes

Headline reporting should capture:

- run ID and lineage
- lane
- model setup
- T1-T4 accuracy
- Tier 5 score
- combined score
- wall time
- eval tokens
- total tokens when available
- DB size when doing embedding studies

## Failure Signatures

### Local Gemma Parser Failure

Signature:

- `llama-server` returns `500`
- log contains:
  - `Failed to parse input at pos 13: <|channel>thought`

Meaning:

- this is a Gemma/llama.cpp parser configuration problem
- not enough evidence for a memory-capacity claim by itself

First response:

1. confirm GPU and host memory
2. inspect server log
3. verify Gemma server is not using `--reasoning-format none`
4. run direct smoke against the local port before rerunning the benchmark

### Embedding Preflight Failure

Signature:

- eval preflight aborts before scoring
- missing or unreachable Ollama embedding contract

First response:

1. verify `BENCHMARK_EMBEDDINGS_PROVIDER`
2. verify Ollama health
3. verify embedding model presence
4. salvage with eval-only rerun if ingest already completed cleanly

### OAuth Usage Cap

If the operator has explicitly authorized rotation:

- rotate to the secondary key only for usage-cap exhaustion
- do not auto-rotate for invalid-auth or other refusal cases

## Operator Checklist

Before launch:

1. verify fresh run ID
2. verify target run dir does not already contain eval artifacts
3. verify backend-specific env vars
4. verify embedding provider reachability
5. verify local model server health when using `llama-cpp`

After launch:

1. confirm the runner entered the intended phase
2. confirm answer requests are landing
3. confirm scored rows appear
4. confirm `token_usage.json` is being written
5. append successful runs to the agent run ledger

## Current Practical Defaults

- default embedding backend:
  - `nomic-embed-text`
- default fastest validation lane:
  - `AL-S`
- preferred difficult retrieval lane:
  - `AL-L OBD`
- default main release lane:
  - Sonnet / Haiku
- default speed-floor lane:
  - Haiku / Haiku
- default local smoke lane:
  - single-model Gemma on a reused ingest lineage
- canonical launch path:
  - `scripts/launch-remote-benchmark.sh`
