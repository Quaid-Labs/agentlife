#!/usr/bin/env python3
"""Expose an OpenAI-compatible embeddings surface backed by Ollama /api/embed.

This is only for the local VM benchmark path where OpenClaw expects an OpenAI
embeddings API but the local Ollama lane responds reliably via `/api/embed`.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length) if length > 0 else b"{}"
    return json.loads(raw.decode("utf-8"))


def _coerce_input(payload: dict):
    value = payload.get("input", "")
    if isinstance(value, list):
        return value
    return [value]


def build_handler(upstream: str):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path != "/v1/models":
                _json_response(self, 404, {"error": "not found"})
                return
            req = Request(f"{upstream}/api/tags", headers={"Accept": "application/json"})
            try:
                with urlopen(req, timeout=30) as resp:
                    payload = json.load(resp)
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                _json_response(self, exc.code, {"error": detail or str(exc)})
                return
            models = [
                {
                    "id": entry.get("model") or entry.get("name"),
                    "object": "model",
                    "created": 0,
                    "owned_by": "ollama",
                }
                for entry in payload.get("models", [])
            ]
            _json_response(self, 200, {"object": "list", "data": models})

        def do_POST(self):  # noqa: N802
            if self.path != "/v1/embeddings":
                _json_response(self, 404, {"error": "not found"})
                return
            payload = _read_json(self)
            body = {
                "model": payload.get("model"),
                "input": _coerce_input(payload),
            }
            req = Request(
                f"{upstream}/api/embed",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(req, timeout=300) as resp:
                    upstream_payload = json.load(resp)
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                _json_response(self, exc.code, {"error": detail or str(exc)})
                return

            embeddings = upstream_payload.get("embeddings") or []
            data = [
                {
                    "object": "embedding",
                    "embedding": vector,
                    "index": idx,
                }
                for idx, vector in enumerate(embeddings)
            ]
            _json_response(
                self,
                200,
                {
                    "object": "list",
                    "data": data,
                    "model": body.get("model"),
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                },
            )

        def log_message(self, _format: str, *_args) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11436)
    parser.add_argument("--upstream", default="http://127.0.0.1:11434")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), build_handler(args.upstream.rstrip("/")))
    print(f"ollama-openai-embed-proxy listening on http://{args.host}:{args.port} -> {args.upstream}")
    server.serve_forever()


if __name__ == "__main__":
    main()
