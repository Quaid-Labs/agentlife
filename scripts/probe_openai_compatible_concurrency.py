#!/usr/bin/env python3
"""Probe warm-start latency and throughput for an OpenAI-compatible endpoint.

This is intentionally transport-level only. It helps choose a reasonable
benchmark eval parallelism setting for self-hosted backends such as llama.cpp
without changing benchmark behavior.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _headers(api_key: str) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, int((len(vals) - 1) * q)))
    return float(vals[idx])


def _message_text(message: Dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text = str(item.get("text") or item.get("content") or "").strip()
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(content or "")


def _single_call(
    *,
    url: str,
    model: str,
    api_key: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=_headers(api_key),
    )
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = json.loads(resp.read())
    latency_ms = (time.monotonic() - started) * 1000.0
    choice = ((raw.get("choices") or [{}])[0] or {})
    message = choice.get("message") or {}
    usage = raw.get("usage") or {}
    return {
        "latency_ms": round(latency_ms, 1),
        "response_chars": len(_message_text(message)),
        "model": str(raw.get("model") or model),
        "usage": {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        },
    }


def _run_level(
    *,
    url: str,
    model: str,
    api_key: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout: int,
    concurrency: int,
    requests: int,
) -> Dict[str, Any]:
    started = time.monotonic()
    rows: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [
            ex.submit(
                _single_call,
                url=url,
                model=model,
                api_key=api_key,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )
            for _ in range(requests)
        ]
        for fut in concurrent.futures.as_completed(futs):
            rows.append(fut.result())
    wall_s = max(0.0001, time.monotonic() - started)
    latencies = [float(r["latency_ms"]) for r in rows]
    prompt_tokens = sum(int((r.get("usage") or {}).get("prompt_tokens", 0) or 0) for r in rows)
    completion_tokens = sum(int((r.get("usage") or {}).get("completion_tokens", 0) or 0) for r in rows)
    total_tokens = sum(int((r.get("usage") or {}).get("total_tokens", 0) or 0) for r in rows)
    return {
        "concurrency": concurrency,
        "requests": requests,
        "wall_seconds": round(wall_s, 3),
        "requests_per_second": round(requests / wall_s, 3),
        "latency_ms": {
            "avg": round(statistics.fmean(latencies), 1) if latencies else 0.0,
            "p50": round(_percentile(latencies, 0.50), 1),
            "p95": round(_percentile(latencies, 0.95), 1),
            "max": round(max(latencies), 1) if latencies else 0.0,
        },
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "prompt_tokens_per_second": round(prompt_tokens / wall_s, 2),
            "completion_tokens_per_second": round(completion_tokens / wall_s, 2),
            "total_tokens_per_second": round(total_tokens / wall_s, 2),
        },
        "response_chars_total": sum(int(r.get("response_chars", 0) or 0) for r in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="OpenAI-compatible base URL")
    parser.add_argument("--model", required=True, help="Served model name")
    parser.add_argument("--api-key", default="", help="Bearer token if required")
    parser.add_argument("--concurrency", default="1,2,4,6", help="Comma-separated concurrency levels")
    parser.add_argument("--requests-per-level", type=int, default=12, help="Timed requests per concurrency level")
    parser.add_argument("--warmup-requests", type=int, default=2, help="Warm-start requests excluded from timings")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--system-prompt", default="Answer concisely and directly.")
    parser.add_argument(
        "--prompt",
        default="Summarize the likely tradeoffs of using a local model for memory-question answering in three sentences.",
    )
    parser.add_argument("--prompt-file", default="", help="Optional file overriding --prompt")
    parser.add_argument("--output-json", default="", help="Optional file to write structured results")
    args = parser.parse_args()

    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": prompt},
    ]
    concurrencies = [int(part) for part in str(args.concurrency).split(",") if str(part).strip()]
    concurrencies = [value for value in concurrencies if value > 0]
    if not concurrencies:
        raise SystemExit("No valid concurrency levels provided")

    print(f"Warmup: {args.warmup_requests} request(s) not counted")
    for idx in range(args.warmup_requests):
        row = _single_call(
            url=args.url,
            model=args.model,
            api_key=args.api_key,
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
        )
        print(
            f"  warmup {idx+1}/{args.warmup_requests}: {row['latency_ms']:.1f} ms "
            f"model={row['model']}"
        )

    results: List[Dict[str, Any]] = []
    for level in concurrencies:
        print(f"\nConcurrency {level} | requests={args.requests_per_level}")
        result = _run_level(
            url=args.url,
            model=args.model,
            api_key=args.api_key,
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            concurrency=level,
            requests=args.requests_per_level,
        )
        results.append(result)
        print(
            "  latency_ms:"
            f" avg={result['latency_ms']['avg']:.1f}"
            f" p50={result['latency_ms']['p50']:.1f}"
            f" p95={result['latency_ms']['p95']:.1f}"
            f" max={result['latency_ms']['max']:.1f}"
        )
        print(
            "  throughput:"
            f" req/s={result['requests_per_second']:.3f}"
            f" tok/s={result['usage']['total_tokens_per_second']:.2f}"
            f" out_tok/s={result['usage']['completion_tokens_per_second']:.2f}"
        )

    payload = {
        "url": args.url,
        "model": args.model,
        "warmup_requests": args.warmup_requests,
        "requests_per_level": args.requests_per_level,
        "results": results,
    }
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved {output_path}")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = (exc.read() or b"").decode("utf-8", errors="ignore")
        except Exception:
            body = ""
        raise SystemExit(f"HTTP {exc.code}: {body[:400]}")
    except urllib.error.URLError as exc:
        raise SystemExit(f"URL error: {exc}")
