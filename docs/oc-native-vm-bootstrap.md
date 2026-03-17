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
  --answer-model claude-haiku-4-5-20251001 \
  --dry-run
```

Small smoke:

```bash
cd ~/agentlife-benchmark
python3 eval/vm_benchmark.py \
  --system oc-native \
  --vm-ip 192.168.64.3 \
  --answer-model claude-haiku-4-5-20251001 \
  --limit-sessions 2 \
  --limit-queries 4 \
  --no-filler
```

## Notes

- This does not replace the canonical production benchmark launcher.
- These helpers are only for the VM benchmark path.
- If you want a non-Tart fallback later, add it explicitly rather than reusing
  this path implicitly.
