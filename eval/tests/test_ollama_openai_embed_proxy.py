import importlib.util
import json
import socket
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROXY_PATH = ROOT.parent / "scripts" / "ollama-openai-embed-proxy.py"
MODEL = "nomic-embed-text"


def _load_proxy_module():
    spec = importlib.util.spec_from_file_location("ollama_openai_embed_proxy", PROXY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_batched_embedding_inputs_distinguishes_true_batch():
    proxy = _load_proxy_module()

    assert proxy._batched_embedding_inputs("hello") is None
    assert proxy._batched_embedding_inputs(["hello"]) is None
    assert proxy._batched_embedding_inputs([1, 2, 3]) is None
    assert proxy._batched_embedding_inputs(["one", "two"]) == ["one", "two"]
    assert proxy._batched_embedding_inputs([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]


def test_relay_embeddings_passes_single_input_through(monkeypatch):
    proxy = _load_proxy_module()
    calls = []

    def fake_relay(req, timeout):
        calls.append((req.full_url, json.loads(req.data.decode("utf-8")), timeout))
        return 200, {
            "object": "list",
            "data": [{"embedding": [0.1, 0.2], "index": 0}],
            "model": MODEL,
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }

    monkeypatch.setattr(proxy, "_relay", fake_relay)

    status, payload = proxy._relay_embeddings(
        "http://127.0.0.1:11434",
        {"input": "hello", "model": MODEL},
        {"Content-Type": "application/json"},
        timeout=300,
    )

    assert status == 200
    assert payload["model"] == MODEL
    assert calls == [
        (
            "http://127.0.0.1:11434/v1/embeddings",
            {"input": "hello", "model": MODEL},
            300,
        )
    ]


def test_relay_embeddings_fans_out_multi_input_and_merges_usage(monkeypatch):
    proxy = _load_proxy_module()
    seen_inputs = []

    def fake_relay(req, timeout):
        payload = json.loads(req.data.decode("utf-8"))
        seen_inputs.append((payload["input"], timeout))
        item = payload["input"][0]
        return 200, {
            "object": "list",
            "data": [{"embedding": [float(len(str(item)))], "index": 0}],
            "model": MODEL,
            "usage": {"prompt_tokens": len(str(item)), "total_tokens": len(str(item)) + 1},
        }

    monkeypatch.setattr(proxy, "_relay", fake_relay)

    status, payload = proxy._relay_embeddings(
        "http://127.0.0.1:11434",
        {"input": ["alpha", "beta", "gamma"], "model": MODEL},
        {"Content-Type": "application/json"},
        timeout=300,
    )

    assert status == 200
    assert seen_inputs == [(["alpha"], 300), (["beta"], 300), (["gamma"], 300)]
    assert payload == {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": [5.0], "index": 0},
            {"object": "embedding", "embedding": [4.0], "index": 1},
            {"object": "embedding", "embedding": [5.0], "index": 2},
        ],
        "model": MODEL,
        "usage": {"prompt_tokens": 14, "total_tokens": 17},
    }


def test_relay_embeddings_propagates_upstream_error(monkeypatch):
    proxy = _load_proxy_module()

    def fake_relay(req, timeout):
        payload = json.loads(req.data.decode("utf-8"))
        if payload["input"] == ["bad"]:
            return 502, {"error": "upstream down"}
        return 200, {
            "object": "list",
            "data": [{"embedding": [1.0], "index": 0}],
            "model": MODEL,
        }

    monkeypatch.setattr(proxy, "_relay", fake_relay)

    status, payload = proxy._relay_embeddings(
        "http://127.0.0.1:11434",
        {"input": ["ok", "bad", "later"], "model": MODEL},
        {"Content-Type": "application/json"},
        timeout=300,
    )

    assert status == 502
    assert payload == {"error": "upstream down"}


def test_relay_api_embed_passes_raw_ollama_shape_through(monkeypatch):
    proxy = _load_proxy_module()
    calls = []

    def fake_relay(req, timeout):
        calls.append((req.full_url, json.loads(req.data.decode("utf-8")), timeout))
        return 200, {
            "model": MODEL,
            "embeddings": [[0.1, 0.2, 0.3]],
            "total_duration": 12,
        }

    monkeypatch.setattr(proxy, "_relay", fake_relay)

    status, payload = proxy._relay_api_embed(
        "http://127.0.0.1:11434",
        {"input": ["hello"], "model": MODEL, "keep_alive": -1},
        {"Content-Type": "application/json"},
        timeout=300,
    )

    assert status == 200
    assert payload["embeddings"] == [[0.1, 0.2, 0.3]]
    assert calls == [
        (
            "http://127.0.0.1:11434/api/embed",
            {"input": ["hello"], "model": MODEL, "keep_alive": -1},
            300,
        )
    ]


def test_relay_api_tags_passes_raw_ollama_tags_through(monkeypatch):
    proxy = _load_proxy_module()
    calls = []

    def fake_relay(req, timeout):
        calls.append((req.full_url, req.get_method(), timeout))
        return 200, {"models": [{"name": MODEL}]}

    monkeypatch.setattr(proxy, "_relay", fake_relay)

    status, payload = proxy._relay_api_tags("http://127.0.0.1:11434", timeout=30)

    assert status == 200
    assert payload == {"models": [{"name": MODEL}]}
    assert calls == [("http://127.0.0.1:11434/api/tags", "GET", 30)]


def test_embeddings_relay_timeout_stays_below_guest_validation_budget():
    proxy = _load_proxy_module()
    assert proxy.MODELS_RELAY_TIMEOUT_S == 30
    assert proxy.EMBEDDINGS_RELAY_TIMEOUT_S == 90


def test_relay_maps_socket_timeout_to_504(monkeypatch):
    proxy = _load_proxy_module()

    def _boom(req, timeout):
        raise socket.timeout("slow upstream")

    monkeypatch.setattr(proxy._UPSTREAM_OPENER, "open", _boom)

    status, payload = proxy._relay(
        proxy.Request("http://127.0.0.1:11434/v1/embeddings"),
        timeout=90,
    )

    assert status == 504
    assert payload == {"error": "upstream timeout: slow upstream"}
