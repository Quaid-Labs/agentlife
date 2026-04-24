#!/usr/bin/env python3
"""Expose a guest-visible OpenAI-compatible embeddings surface backed by Ollama.

The local OC-native VM can reach the host on 192.168.64.1 when the host listens
on 0.0.0.0, but local Ollama only listens on 127.0.0.1:11434. This helper
binds a guest-visible host port and relays only the required OpenAI-compatible
`/v1/models` and `/v1/embeddings` routes to the host-local Ollama server.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


_UPSTREAM_OPENER = build_opener(ProxyHandler({}))


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.wfile.write(body)


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
    except URLError as exc:
        return 502, {"error": str(exc)}


def build_handler(upstream: str):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):  # noqa: N802
            if self.path != "/v1/models":
                _json_response(self, 404, {"error": "not found"})
                return
            status, payload = _relay(
                Request(f"{upstream}/v1/models", headers={"Accept": "application/json"}),
                timeout=30,
            )
            _json_response(self, status, payload)

        def do_POST(self):  # noqa: N802
            if self.path != "/v1/embeddings":
                _json_response(self, 404, {"error": "not found"})
                return
            payload = _read_json(self)
            headers = {"Content-Type": "application/json"}
            auth = self.headers.get("Authorization")
            if auth:
                headers["Authorization"] = auth
            status, upstream_payload = _relay(
                Request(
                    f"{upstream}/v1/embeddings",
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST",
                ),
                timeout=300,
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
