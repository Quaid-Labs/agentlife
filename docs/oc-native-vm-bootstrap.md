# OC Native VM Bootstrap

This is the benchmark-local bootstrap path for the OpenClaw native memory
baseline used by `eval/vm_benchmark.py --system oc-native`.

It is intentionally separate from:
- production benchmark launch (`scripts/launch-remote-benchmark.sh`)
- Quaid runtime bootstrap in `~/quaid/bootstrap`
- Quaid/OpenClaw live E2E flows

The goal is a clean, reproducible OpenClaw-only target that the VM benchmark can
reconfigure per system without cross-repo runtime dependencies.

## Baseline

The native baseline currently means:
- OpenClaw installed and reachable over SSH
- gateway healthy
- no Quaid plugin required in the base image
- host-visible embeddings endpoint for `qwen3-embedding:8b`

Per-system details still happen inside `eval/vm_benchmark.py`.

For `oc-native`, the benchmark itself enables:
- `memory-core`
- builtin memory backend
- bundled `session-memory` hook
- `active-memory` blocking recall sub-agent
- `memory-wiki` bridge/import/compile flow
- embeddings pinned to `http://192.168.64.1:11434/v1`
- model `qwen3-embedding:8b`

## Bootstrap

Provision a clean target:

```bash
cd ~/agentlife-benchmark
scripts/bootstrap-oc-native-vm.sh \
  --vm-ip 192.168.64.3 \
  --user admin \
  --password admin
```

The script:
- verifies SSH access
- ensures `node`, `npm`, `python3`
- installs OpenClaw if needed
- installs `sqlite-vec` if missing
- creates `~/.openclaw/openclaw.json` if absent
- clears benchmark workspace state
- starts and probes the gateway
- probes the guest-visible embeddings endpoint

For direct OpenAI answer models, the host must expose `OPENAI_API_KEY` or the
benchmark-local config must resolve one through `run_production_benchmark.py`.
During `oc-native` setup, the VM harness writes that key to:

```text
~/.openclaw/agents/main/agent/auth-profiles.json
```

The profile written is `profiles["openai:default"].token`, and
`lastGood["openai"]` is set to `openai:default`.

## Snapshot

Once the target is clean and healthy:

```bash
cd ~/agentlife-benchmark
scripts/snapshot-oc-native-vm.sh \
  --vm-name test-openclaw \
  --snapshot clean-openclaw \
  --replace
```

## Smoke

Dry-run:

```bash
cd ~/agentlife-benchmark
python3 eval/vm_benchmark.py \
  --system oc-native \
  --vm-ip 192.168.64.3 \
  --answer-model openai/gpt-5.4 \
  --dry-run
```

Small smoke:

```bash
cd ~/agentlife-benchmark
python3 eval/vm_benchmark.py \
  --system oc-native \
  --vm-ip 192.168.64.3 \
  --answer-model openai/gpt-5.4 \
  --limit-sessions 2 \
  --limit-queries 4 \
  --no-filler
```

## Scored Runs

Run `AL-S` as the arc-only small lane:

```bash
cd ~/agentlife-benchmark
python3 eval/vm_benchmark.py \
  --system oc-native \
  --vm-ip 192.168.64.3 \
  --snapshot clean-openclaw \
  --results-dir data/results-vm-oc-native-current-als \
  --answer-model openai/gpt-5.4 \
  --judge-model gpt-4o-mini \
  --splitting timeout \
  --no-filler
```

Run `AL-L` as the arc-plus-filler long lane:

```bash
cd ~/agentlife-benchmark
python3 eval/vm_benchmark.py \
  --system oc-native \
  --vm-ip 192.168.64.3 \
  --snapshot clean-openclaw \
  --results-dir data/results-vm-oc-native-current-all \
  --answer-model openai/gpt-5.4 \
  --judge-model gpt-4o-mini \
  --splitting timeout
```

Use a fresh restored snapshot and a distinct `--results-dir` base for each
scored lane. The VM harness result suffix does not encode `--no-filler`, so
`AL-S` and `AL-L` can collide if they share the same results base.

## Methodology

The `oc-native` VM lane measures current OpenClaw native memory behavior, not
Quaid. The harness writes benchmark transcripts into real OpenClaw session
files, runs the bundled `/new` `session-memory` hook for each synthetic session,
forces `openclaw memory index --agent main --force`, then imports/compiles the
`memory-wiki` bridge before evaluation.

The evaluated stack is:
- `memory-core` with builtin memory backend
- direct session transcript indexing in `memory_search`
- bundled `session-memory` hook creating workspace memory files
- `active-memory` on `before_prompt_build` for direct `main` sessions
- `memory-wiki` in bridge mode over memory-core public artifacts

### Dream-Cycle Limitation (Important)

Current `oc-native` harness wiring does **not** expose a deterministic
"trigger dream now" control for per-day lifecycle simulation. In this setup,
scored runs do **not** explicitly drive dream/finalize cycles by simulated day.

What is exercised:
- per-session transcript ingestion
- bundled `/new` `session-memory` hook execution
- forced `openclaw memory index --agent main --force`
- memory-wiki bridge import/compile before eval

What is not explicitly exercised:
- deterministic day-boundary dream-cycle triggering

Interpretation guidance:
- Keep OC-native scores as valid baseline numbers for this setup.
- Treat cross-lane comparisons that depend on explicit dream-cycle cadence as
  limited until a deterministic dream trigger is available in harness.

Token accounting is split by reliability:
- `injection_stats.json` records real extraction/janitor usage when the lane has
  instrumented token usage. For `oc-native`, memory-core/active-memory provider
  usage is not currently reported by `openclaw agent`, so those fields are
  lower-bound or zero unless OpenClaw exposes usage in stdout/logs.
- `eval_results.json` records per-query visible token estimates: question,
  prediction, visible agent total, and judge prompt.
- `scores.json` aggregates those visible eval estimates under
  `eval_token_estimate`. Treat this as a lower bound because hidden prompt
  context, tool-call payloads, active-memory sub-agent prompts, and gateway
  provider overhead are not included.

OpenClaw direct OpenAI auth must be provisioned on the VM before scored runs.
Do not use Anthropic OAuth for the `oc-native` answer model lane.

## Notes

- This does not replace the canonical production benchmark launcher.
- These helpers are only for the VM benchmark path.
- If you want a non-Tart fallback later, add it explicitly rather than reusing
  this path implicitly.
