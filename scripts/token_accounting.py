#!/usr/bin/env python3
"""
Canonical token accounting helper for benchmark reporting.

Public benchmark rows should report:
  eval_tokens_ex_judge = token_usage.eval.total_tokens
                         - sum(token_usage.eval.by_source[*judge*].total_tokens)

This includes answer + preinject + tool-recall spend while excluding judge spend.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except Exception:
        return 0


def _eval_answer_only_tokens(eval_results_path: Path) -> dict[str, int]:
    if not eval_results_path.exists():
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    rows = _load_json(eval_results_path)
    if not isinstance(rows, list):
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        usage = row.get("eval_tokens")
        if isinstance(usage, dict):
            in_t = _as_int(usage.get("input_tokens"))
            out_t = _as_int(usage.get("output_tokens"))
            input_tokens += in_t
            output_tokens += out_t
            total_tokens += in_t + out_t
        elif isinstance(usage, (int, float)):
            total_tokens += int(usage)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _eval_token_usage(eval_block: dict[str, Any]) -> dict[str, Any]:
    total_tokens = _as_int(eval_block.get("total_tokens"))
    input_tokens = _as_int(eval_block.get("input_tokens"))
    output_tokens = _as_int(eval_block.get("output_tokens"))
    api_calls = _as_int(eval_block.get("api_calls"))

    by_source = eval_block.get("by_source")
    judge_breakdown: dict[str, int] = {}
    judge_total_tokens = 0
    if isinstance(by_source, dict):
        for source_name, payload in by_source.items():
            if "judge" not in str(source_name).lower():
                continue
            source_tokens = _as_int((payload or {}).get("total_tokens"))
            judge_breakdown[str(source_name)] = source_tokens
            judge_total_tokens += source_tokens

    eval_tokens_ex_judge = max(0, total_tokens - judge_total_tokens)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "api_calls": api_calls,
        "judge_total_tokens": judge_total_tokens,
        "judge_breakdown": judge_breakdown,
        "eval_tokens_ex_judge": eval_tokens_ex_judge,
    }


def _ingest_usage(ingest_path: Path) -> dict[str, int]:
    if not ingest_path.exists():
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "api_calls": 0}
    ingest_obj = _load_json(ingest_path)
    ingest = ingest_obj.get("ingest", {}) if isinstance(ingest_obj, dict) else {}
    return {
        "input_tokens": _as_int(ingest.get("input_tokens")),
        "output_tokens": _as_int(ingest.get("output_tokens")),
        "total_tokens": _as_int(ingest.get("total_tokens")),
        "api_calls": _as_int(ingest.get("api_calls")),
    }


def summarize_run(run_dir: Path) -> dict[str, Any]:
    token_usage_path = run_dir / "token_usage.json"
    eval_results_path = run_dir / "evaluation_results.json"
    ingest_usage_path = run_dir / "ingest_usage.json"

    if not token_usage_path.exists():
        raise FileNotFoundError(f"missing required file: {token_usage_path}")

    token_usage_obj = _load_json(token_usage_path)
    eval_block = token_usage_obj.get("eval", {}) if isinstance(token_usage_obj, dict) else {}

    out = {
        "run_dir": str(run_dir),
        "canonical_public_metric": "eval_tokens_ex_judge",
        "token_usage_eval": _eval_token_usage(eval_block if isinstance(eval_block, dict) else {}),
        "evaluation_results_answer_only": _eval_answer_only_tokens(eval_results_path),
        "ingest": _ingest_usage(ingest_usage_path),
    }
    out["totals"] = {
        "combined_ex_judge_plus_ingest": (
            _as_int(out["token_usage_eval"]["eval_tokens_ex_judge"])
            + _as_int(out["ingest"]["total_tokens"])
        ),
        "combined_eval_total_plus_ingest": (
            _as_int(out["token_usage_eval"]["total_tokens"])
            + _as_int(out["ingest"]["total_tokens"])
        ),
    }
    return out


def _print_text(summary: dict[str, Any]) -> None:
    run_dir = summary["run_dir"]
    eval_usage = summary["token_usage_eval"]
    answer_only = summary["evaluation_results_answer_only"]
    ingest = summary["ingest"]
    totals = summary["totals"]

    print(f"run_dir: {run_dir}")
    print("canonical_public_metric: eval_tokens_ex_judge")
    print("")
    print("eval (token_usage.json):")
    print(f"  total_tokens: {eval_usage['total_tokens']}")
    print(f"  judge_total_tokens: {eval_usage['judge_total_tokens']}")
    print(f"  eval_tokens_ex_judge: {eval_usage['eval_tokens_ex_judge']}")
    if eval_usage.get("judge_breakdown"):
        print(f"  judge_breakdown: {eval_usage['judge_breakdown']}")
    print("")
    print("legacy answer-only (evaluation_results.json eval_tokens sum):")
    print(f"  total_tokens: {answer_only['total_tokens']}")
    print("")
    print("ingest (ingest_usage.json):")
    print(f"  total_tokens: {ingest['total_tokens']}")
    print("")
    print("combined totals:")
    print(f"  eval_ex_judge_plus_ingest: {totals['combined_ex_judge_plus_ingest']}")
    print(f"  eval_total_plus_ingest: {totals['combined_eval_total_plus_ingest']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Canonical benchmark token accounting helper.")
    parser.add_argument("--run", required=True, help="Path to benchmark run directory.")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args()

    run_dir = Path(args.run).expanduser().resolve()
    summary = summarize_run(run_dir)
    if args.format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_text(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
