import json
import sys
import types

_dataset = sys.modules.get("dataset")
if _dataset is None:
    _dataset = types.ModuleType("dataset")
    sys.modules["dataset"] = _dataset
setattr(_dataset, "SESSION_DATES", getattr(_dataset, "SESSION_DATES", {i: "2026-03-01" for i in range(1, 21)}))
setattr(_dataset, "load_filler_reviews", getattr(_dataset, "load_filler_reviews", lambda *a, **k: []))
setattr(_dataset, "merge_sessions_chronologically", getattr(_dataset, "merge_sessions_chronologically", lambda *a, **k: []))

_cb = types.ModuleType("claude_backend")
_cb.call_claude = lambda *a, **k: ("", 0.0)
sys.modules.setdefault("claude_backend", _cb)

_rpb = types.ModuleType("run_production_benchmark")
_rpb._judge = lambda *a, **k: ("WRONG", 0.0)
_rpb._judge_tier5 = lambda *a, **k: (0, "")
_rpb._get_api_key = lambda: "test-anthropic-key"
_rpb._get_openai_key = lambda: "test-openai-key"
sys.modules.setdefault("run_production_benchmark", _rpb)

from eval.mem0_adapter import Mem0Adapter, _normalize_tz_for_mem0


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_answer_question_gpt_uses_openai_endpoint(monkeypatch, tmp_path):
    called_urls = []

    def fake_urlopen(req, timeout=120):
        called_urls.append(req.full_url)
        return _FakeResponse(
            {
                "choices": [{"message": {"content": "Test answer"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            }
        )

    monkeypatch.setattr("eval.mem0_adapter._get_api_key", lambda: "test-anthropic-key")
    monkeypatch.setattr("eval.mem0_adapter._get_openai_key", lambda: "test-openai-key")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    adapter = Mem0Adapter(results_dir=tmp_path, answer_model="gpt-4o-mini")
    monkeypatch.setattr(adapter, "search", lambda q, limit=10: [{"text": "Maya likes tea", "score": 0.9}])

    answer, tools, usage = adapter.answer_question("What does Maya like?")

    assert answer == "Test answer"
    assert tools == ["mem0_search"]
    assert usage["api_calls"] == 1
    assert called_urls == ["https://api.openai.com/v1/chat/completions"]


def test_answer_question_haiku_uses_anthropic_endpoint(monkeypatch, tmp_path):
    called_urls = []

    def fake_urlopen(req, timeout=120):
        called_urls.append(req.full_url)
        return _FakeResponse(
            {
                "content": [{"type": "text", "text": "Claude answer"}],
                "usage": {"input_tokens": 5, "output_tokens": 3},
            }
        )

    monkeypatch.setattr("eval.mem0_adapter._get_api_key", lambda: "test-anthropic-key")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    adapter = Mem0Adapter(results_dir=tmp_path, answer_model="haiku")
    answer, tools, usage = adapter.answer_question("Test?")

    assert answer == "Claude answer"
    assert tools == []
    assert usage["api_calls"] == 1
    assert called_urls == ["https://api.anthropic.com/v1/messages"]


def test_timezone_alias_is_normalized(monkeypatch):
    monkeypatch.setenv("TZ", "US/Pacific")
    _normalize_tz_for_mem0()
    import os
    assert "Los_Angeles" in (os.environ.get("TZ") or "")
