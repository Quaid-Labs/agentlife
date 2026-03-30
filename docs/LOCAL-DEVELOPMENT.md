# Local Development Config

The benchmark harness keeps machine-specific setup in one ignored repo-local
config file:

- tracked example: `.agentlife-benchmark.example.json`
- ignored real file: `.agentlife-benchmark.local.json`

Copy the example and edit the local file:

```bash
cp .agentlife-benchmark.example.json .agentlife-benchmark.local.json
```

## Path Rules

All relative paths inside `.agentlife-benchmark.local.json` resolve from
`paths.benchRoot`.

- `benchRoot` should point at the benchmark repo root
- use relative paths for local secrets and sibling checkouts
- avoid hardcoding user-specific absolute paths in tracked docs or scripts

Example:

```json
{
  "paths": {
    "benchRoot": ".",
    "checkpointRoot": "../quaid/benchmark-checkpoint"
  },
  "auth": {
    "anthropic": {
      "primaryKeyPath": "../secrets/anthropic-primary.txt",
      "secondaryKeyPath": "../secrets/anthropic-secondary.txt"
    },
    "openai": {
      "judgeKeyPath": "../secrets/openai.txt"
    }
  }
}
```

## Supported Fields

`paths`

- `benchRoot`: benchmark repo root used for resolving relative secret paths
- `checkpointRoot`: optional local Quaid checkpoint checkout path for humans;
  current launcher still accepts `--local-checkpoint-root` explicitly

`auth.anthropic`

- `primaryKeyPath`: path to the primary benchmark Anthropic OAuth token file
- `secondaryKeyPath`: optional operator-switched backup token file
- tokens should live outside the repo or in other ignored locations

`auth.openai`

- `judgeKeyPath`: path to the OpenAI key used by the benchmark judge

## Consumers

These benchmark tools read `.agentlife-benchmark.local.json` today:

- `scripts/launch-remote-benchmark.sh`

Current behavior:

- prefers `.agentlife-benchmark.local.json`
- no legacy config fallback path is supported
- never stores secret values in tracked JSON
- benchmark launch sync excludes `.agentlife-benchmark.local.json`, `.env`, and
  `release/`, so local-only secrets and built artifacts are not copied to the
  remote benchmark host

## `.env` vs Local Config

Use `.agentlife-benchmark.local.json` for long-lived local machine setup and
secret file paths.

Use `.env` only for quick local-only scripts that still expect environment
variables directly.

Release-ready benchmark launches should not depend on ad-hoc shell exports when
the local config can provide the same secret paths deterministically.
