#!/usr/bin/env python3
"""Expose guest-visible Ollama-backed embeddings surfaces for benchmark VMs.

The local OC-native VM can reach the host on 192.168.64.1 when the host listens
on 0.0.0.0, but local Ollama only listens on 127.0.0.1:11434. This helper
binds a guest-visible host port and relays:
- OpenAI-compatible `/v1/models` and `/v1/embeddings` for OC-native
- raw Ollama `/api/embed` and `/api/tags` for Quaid-on-OC-VM
"""

from __future__ import annotations

import argparse
import json
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from http.client import RemoteDisconnected
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


_UPSTREAM_OPENER = build_opener(ProxyHandler({}))
MODELS_RELAY_TIMEOUT_S = 30
EMBEDDINGS_RELAY_TIMEOUT_S = 90


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Connection", "close")
    try:
        handler.end_headers()
        handler.wfile.write(body)
    except (BrokenPipeError, ConnectionResetError):
        # The benchmark client can time out and disconnect while the proxy is
        # still unwinding a slow upstream call. That should not crash or spam
        # the proxy process; the real failure is surfaced to the caller.
        return


def _read_json(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length) if length > 0 else b"{}"
    return json.loads(raw.decode("utf-8"))


def _relay(req: Request, timeout: int) -> tuple[int, dict]:
    try:
        with _UPSTREAM_OPENER.open(req, timeout=timeout) as resp:
            return getattr(resp, "status", 200), json.load(resp)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return exc.code, {"error": detail or str(exc)}
    except (TimeoutError, socket.timeout) as exc:
        return 504, {"error": f"upstream timeout: {exc}"}
    except URLError as exc:
        return 502, {"error": str(exc)}
    except (RemoteDisconnected, OSError) as exc:
        return 502, {"error": str(exc)}


def _batched_embedding_inputs(value: Any) -> list[Any] | None:
    """Return top-level embedding items when the payload is a true multi-input batch."""
    if not isinstance(value, list) or len(value) <= 1:
        return None
    first = value[0]
    if isinstance(first, int):
        # Single token-array input, not a batch of inputs.
        return None
    return value


def _relay_embeddings(upstream: str, payload: dict, headers: dict[str, str], timeout: int) -> tuple[int, dict]:
    """Relay embeddings, fanning out multi-input batches into single-input upstream calls."""
    batched_inputs = _batched_embedding_inputs(payload.get("input"))
    if not batched_inputs:
        return _relay(
            Request(
                f"{upstream}/v1/embeddings",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            ),
            timeout=timeout,
        )

    merged: list[dict[str, Any]] = []
    usage_totals: dict[str, int] = {}
    model = payload.get("model")
    for index, item in enumerate(batched_inputs):
        one_payload = dict(payload)
        one_payload["input"] = [item]
        status, upstream_payload = _relay(
            Request(
                f"{upstream}/v1/embeddings",
                data=json.dumps(one_payload).encode("utf-8"),
                headers=headers,
                method="POST",
            ),
            timeout=timeout,
        )
        if status >= 400:
            return status, upstream_payload
        batch_data = upstream_payload.get("data") or []
        if not batch_data:
            return 502, {"error": "upstream embeddings response missing data"}
        embedding = (batch_data[0] or {}).get("embedding")
        if not isinstance(embedding, list) or not embedding:
            return 502, {"error": "upstream embeddings response missing embedding"}
        merged.append({"object": "embedding", "embedding": embedding, "index": index})
        model = upstream_payload.get("model") or model
        usage = upstream_payload.get("usage") or {}
        for key in ("prompt_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int):
                usage_totals[key] = usage_totals.get(key, 0) + value

    out: dict[str, Any] = {"object": "list", "data": merged}
    if model:
        out["model"] = model
    if usage_totals:
        out["usage"] = usage_totals
    return 200, out


def _relay_api_embed(upstream: str, payload: dict, headers: dict[str, str], timeout: int) -> tuple[int, dict]:
    """Relay raw Ollama /api/embed for Quaid's standalone embeddings path."""
    return _relay(
        Request(
            f"{upstream}/api/embed",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        ),
        timeout=timeout,
    )


def _relay_api_tags(upstream: str, timeout: int) -> tuple[int, dict]:
    """Relay raw Ollama /api/tags for Quaid's recall-time health check."""
    return _relay(
        Request(
            f"{upstream}/api/tags",
            headers={"Accept": "application/json"},
            method="GET",
        ),
        timeout=timeout,
    )


def build_handler(upstream: str):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):  # noqa: N802
            if self.path == "/v1/models":
                status, payload = _relay(
                    Request(f"{upstream}/v1/models", headers={"Accept": "application/json"}),
                    timeout=MODELS_RELAY_TIMEOUT_S,
                )
                _json_response(self, status, payload)
                return
            if self.path == "/api/tags":
                status, payload = _relay_api_tags(upstream, timeout=MODELS_RELAY_TIMEOUT_S)
                _json_response(self, status, payload)
                return
            else:
                _json_response(self, 404, {"error": "not found"})
                return

        def do_POST(self):  # noqa: N802
            if self.path not in ("/v1/embeddings", "/api/embed"):
                _json_response(self, 404, {"error": "not found"})
                return
            payload = _read_json(self)
            headers = {"Content-Type": "application/json"}
            auth = self.headers.get("Authorization")
            if auth:
                headers["Authorization"] = auth
            if self.path == "/api/embed":
                status, upstream_payload = _relay_api_embed(
                    upstream,
                    payload,
                    headers,
                    timeout=EMBEDDINGS_RELAY_TIMEOUT_S,
                )
            else:
                status, upstream_payload = _relay_embeddings(
                    upstream,
                    payload,
                    headers,
                    timeout=EMBEDDINGS_RELAY_TIMEOUT_S,
                )
            _json_response(self, status, upstream_payload)

        def log_message(self, _format: str, *_args) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=11435)
    parser.add_argument("--upstream", default="http://127.0.0.1:11434")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), build_handler(args.upstream.rstrip("/")))
    print(f"ollama-openai-embed-proxy listening on http://{args.host}:{args.port} -> {args.upstream}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
