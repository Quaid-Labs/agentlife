import json
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "claude_backend" not in sys.modules:
    fake = ModuleType("claude_backend")
    fake.call_claude = lambda *args, **kwargs: None
    sys.modules["claude_backend"] = fake

fake_dataset = ModuleType("dataset")
fake_dataset.load_all_reviews = lambda *args, **kwargs: []
fake_dataset.load_filler_reviews = lambda *args, **kwargs: []
fake_dataset.merge_sessions_chronologically = lambda *args, **kwargs: []
fake_dataset.get_all_eval_queries = lambda *args, **kwargs: []
sys.modules["dataset"] = fake_dataset

fake_injector = ModuleType("injector")
fake_injector.transcript_to_messages = lambda *args, **kwargs: []
fake_injector.count_tokens = lambda *args, **kwargs: 0

class _FakeCostTracker:
    def summary(self):
        return {}

fake_injector.CostTracker = _FakeCostTracker
sys.modules["injector"] = fake_injector

import vm_benchmark as vmb  # noqa: E402


class TestOpenClawNativeConfig:
    def test_resolve_local_quaid_plugin_dir_uses_canonical_resolver(self, monkeypatch, tmp_path):
        plugin_dir = tmp_path / "benchmark-checkpoint" / "modules" / "quaid"
        plugin_dir.mkdir(parents=True)

        fake_rpb = ModuleType("run_production_benchmark")
        fake_rpb._resolve_quaid_dir = lambda: plugin_dir
        monkeypatch.setitem(sys.modules, "run_production_benchmark", fake_rpb)
        monkeypatch.setattr(vmb.Path, "home", lambda: tmp_path / "no-home-candidates")
        monkeypatch.delenv("BENCHMARK_PLUGIN_DIR", raising=False)

        assert vmb._resolve_local_quaid_plugin_dir() == plugin_dir.resolve()

    def test_resolve_local_quaid_plugin_dir_prefers_checkpoint_over_dev(self, monkeypatch, tmp_path):
        checkpoint_dir = tmp_path / "quaidcode" / "benchmark-checkpoint" / "modules" / "quaid"
        checkpoint_dir.mkdir(parents=True)
        dev_dir = tmp_path / "quaidcode" / "dev" / "modules" / "quaid"
        dev_dir.mkdir(parents=True)

        fake_rpb = ModuleType("run_production_benchmark")
        fake_rpb._resolve_quaid_dir = lambda: dev_dir
        monkeypatch.setitem(sys.modules, "run_production_benchmark", fake_rpb)
        monkeypatch.setattr(vmb.Path, "home", lambda: tmp_path)
        monkeypatch.delenv("BENCHMARK_PLUGIN_DIR", raising=False)

        assert vmb._resolve_local_quaid_plugin_dir() == checkpoint_dir.resolve()

    def test_resolve_local_quaid_memory_example_finds_repo_root_copy(self, tmp_path):
        plugin_dir = tmp_path / "benchmark-checkpoint" / "modules" / "quaid"
        plugin_dir.mkdir(parents=True)
        memory_example = tmp_path / "benchmark-checkpoint" / "memory.json.example"
        memory_example.write_text("{}\n", encoding="utf-8")

        assert vmb._resolve_local_quaid_memory_example(plugin_dir) == memory_example.resolve()

    def test_build_local_quaid_plugin_tarball_excludes_dev_only_dirs(self, tmp_path):
        plugin_dir = tmp_path / "modules" / "quaid"
        (plugin_dir / "adaptors" / "openclaw").mkdir(parents=True)
        (plugin_dir / "node_modules" / "pkg").mkdir(parents=True)
        (plugin_dir / "tests").mkdir(parents=True)
        (plugin_dir / "__pycache__").mkdir(parents=True)
        (plugin_dir / "package.json").write_text("{}\n", encoding="utf-8")
        (plugin_dir / "adaptors" / "openclaw" / "index.js").write_text("export default {};\n", encoding="utf-8")
        (plugin_dir / "node_modules" / "pkg" / "index.js").write_text("skip\n", encoding="utf-8")
        (plugin_dir / "tests" / "test_x.py").write_text("skip\n", encoding="utf-8")
        (plugin_dir / "__pycache__" / "x.pyc").write_text("skip\n", encoding="utf-8")

        archive = vmb._build_local_quaid_plugin_tarball(plugin_dir)
        try:
            with tarfile.open(archive, "r:gz") as bundle:
                names = set(bundle.getnames())
            assert "./package.json" in names
            assert "./adaptors/openclaw/index.js" in names
            assert "./node_modules/pkg/index.js" not in names
            assert "./tests/test_x.py" not in names
            assert "./__pycache__/x.pyc" not in names
        finally:
            archive.unlink(missing_ok=True)

    def test_upload_local_file_via_vm_ssh_base64_encodes_payload(self, tmp_path):
        source = tmp_path / "sample.txt"
        source.write_text("hello\n", encoding="utf-8")
        captured = {}

        class _Vm:
            def ssh(self, command, **kwargs):
                captured["command"] = command
                captured["kwargs"] = kwargs

                class _Result:
                    returncode = 0
                    stdout = ""
                    stderr = ""

                return _Result()

        result = vmb._upload_local_file_via_vm_ssh(_Vm(), source, "~/remote/sample.txt", timeout=77)
        assert result.returncode == 0
        assert "python3 -c" in captured["command"]
        assert "base64.b64decode" in captured["command"]
        assert captured["kwargs"]["input_data"] == "aGVsbG8K"
        assert captured["kwargs"]["timeout"] == 77
        assert captured["kwargs"]["raw"] is True

    def test_native_config_uses_local_ollama_nomic_embeddings(self):
        script = vmb._build_openclaw_native_config_script(enable_session_hook=True)
        assert "d.pop('slots', None)" in script
        assert "plugins.pop('disable', None)" in script
        assert "plugins.pop('disabled', None)" in script
        assert "plugins.pop('slots', None)" in script
        assert "plugins['allow'] = ['memory-core', 'active-memory', 'memory-wiki']" in script
        assert "for entry in entries.values():" in script
        assert "entry.pop('disabled', None)" in script
        assert "entries.setdefault('matrix', {})['enabled'] = False" in script
        assert "memory['backend'] = 'builtin'" in script
        assert "for entry in channels.values():" in script
        assert "channels.setdefault('matrix', {})['enabled'] = False" in script
        assert f"tools['allow'] = {vmb.OC_NATIVE_MEMORY_TOOLS!r}" in script
        assert "tools.pop('deny', None)" in script
        assert "entries.setdefault('active-memory', {})['enabled'] = True" in script
        assert "active_memory['agents'] = ['main']" in script
        assert "active_memory['queryMode'] = 'recent'" in script
        assert "entries.setdefault('memory-wiki', {})['enabled'] = True" in script
        assert "memory_wiki['vaultMode'] = 'bridge'" in script
        assert "memory_wiki['search'] = {'backend': 'shared', 'corpus': 'all'}" in script
        assert "memory_wiki['context'] = {'includeCompiledDigestPrompt': True}" in script
        assert "ms['provider'] = 'openai'" in script
        assert f"ms['model'] = {vmb.OC_NATIVE_EMBED_MODEL!r}" in script
        assert f"remote['baseUrl'] = {vmb.OC_NATIVE_EMBED_BASE_URL!r}" in script
        assert "remote['apiKey'] = 'ollama-local'" in script
        assert "ms['sources'] = ['memory', 'sessions']" in script
        assert "experimental['sessionMemory'] = True" in script
        assert "chunking['tokens'] = 160" in script
        assert "chunking['overlap'] = 40" in script
        assert "query['maxResults'] = 8" in script
        assert "sync['onSearch'] = False" in script
        assert "sync['onSessionStart'] = False" in script
        assert "sync['watch'] = False" in script
        assert "entries.pop('quaid', None)" in script
        assert "for entry in hook_entries.values():" in script
        assert "hook_entries.setdefault('session-memory', {})['enabled'] = enable_hook" in script

    def test_cleanup_oc_native_eval_session_removes_store_entry_and_transcript(self):
        calls = []

        class _Vm:
            def ssh(self, command, **_kwargs):
                calls.append(command)

                class _Result:
                    returncode = 0
                    stdout = '{"sessionId":"eval-q012","removed":["/Users/admin/.openclaw/agents/main/sessions/eval-q012.jsonl"],"storeUpdated":true}'
                    stderr = ""

                return _Result()

        vmb._cleanup_oc_native_eval_session(
            _Vm(),
            "eval-q012",
            session_key="agent:main:hook:eval-q012",
        )
        command = calls[0]
        assert "sessions.json" in command
        assert "agents/benchmark-eval/sessions" in command
        assert "store.pop(session_key, None)" in command
        assert "sessionFile" in command
        assert "agent:main:hook:eval-q012" in command

    def test_clear_vm_session_state_stops_gateway_and_removes_eval_artifacts(self):
        calls = []

        class _Vm:
            def ssh(self, command, **_kwargs):
                calls.append(command)

                class _Result:
                    returncode = 0
                    stdout = "Session state cleared\n"
                    stderr = ""

                return _Result()

        vmb._clear_vm_session_state(_Vm())
        command = calls[0]
        assert "pkill" in command
        assert "openclaw-gateway" in command
        assert "project_docs_supervisor.py" in command
        assert "core/lifecycle/janitor.py" in command
        assert "/Users/admin/extract_compact.py" in command
        assert "benchmark-eval/sessions" in command
        assert "*.jsonl.reset.*" in command
        assert "memory.db" in command
        assert vmb.VM_QUAID_INSTANCE_DB_PATH in command
        assert vmb.VM_QUAID_INSTANCE_ARCHIVE_DB_PATH in command
        assert "session-cursors" in command
        assert "extraction-signals" in command
        assert "rolling-extraction" in command
        assert "rolling-extraction.jsonl" in command
        assert "llm-usage-events.jsonl" in command
        assert f"{vmb.VM_QUAID_INSTANCE_ROOT_DIR}/.runtime/locks" in command
        assert f"{vmb.VM_QUAID_INSTANCE_ROOT_DIR}/journal" in command
        assert vmb.VM_QUAID_INSTANCE_ROOT_DIR in command
        assert "*.snippets.md" in command
        assert "data/project-docs/locks" in command
        assert "pending-approval-requests.json" in command

    def test_run_vm_janitor_uses_quaid_cli_and_instance_logs(self):
        calls = []

        class _Vm:
            def ssh(self, command, **_kwargs):
                calls.append(command)

                class _Result:
                    def __init__(self, returncode=0, stdout="", stderr=""):
                        self.returncode = returncode
                        self.stdout = stdout
                        self.stderr = stderr

                if len(calls) == 1:
                    return _Result(stdout="janitor ok\n")
                if len(calls) == 2:
                    return _Result(stdout="")
                if len(calls) == 3:
                    return _Result(stdout='{"active": 7}\n')
                return _Result(stdout='{"api_usage":{"calls":1,"input_tokens":2,"output_tokens":3,"estimated_cost_usd":0.4}}\n')

        vmb._run_vm_janitor(_Vm())

        assert "./quaid janitor --task all --apply --approve --no-resume-checkpoint" in calls[0]
        assert f"QUAID_HOME={vmb.VM_QUAID_HOME}" in calls[0]
        assert f"QUAID_INSTANCE={vmb.VM_QUAID_INSTANCE}" in calls[0]
        assert f"QUAID_LLM_USAGE_LOG_PATH={vmb.VM_QUAID_LLM_USAGE_LOG_PATH}" in calls[0]
        assert "QUAID_BENCHMARK_MODE=1" in calls[0]
        assert "QUAID_LLM_USAGE_PHASE=janitor" in calls[0]
        assert "QUAID_LLM_USAGE_SOURCE=benchmark_janitor" in calls[0]
        assert vmb.VM_QUAID_JANITOR_LATEST_LOG in calls[0]
        assert "janitor_rc=$?" in calls[0]
        assert vmb.VM_QUAID_JANITOR_LATEST_LOG in calls[1]
        assert vmb.VM_QUAID_INSTANCE_DB_PATH in calls[2]
        assert vmb.VM_QUAID_JANITOR_STATS_PATH in calls[3]

    def test_patch_memory_json_seeds_both_benchmark_projects(self):
        calls = []

        class _Vm:
            def ssh(self, command, **_kwargs):
                calls.append(command)

                class _Result:
                    returncode = 0
                    stdout = "Patched: /Users/admin/clawd/config/memory.json\n"
                    stderr = ""

                return _Result()

        vmb._patch_memory_json(_Vm(), "claude-sonnet-4-5-20250929")
        command = calls[0]
        assert "recipe-app" in command
        assert "portfolio-site" in command
        assert "/Users/admin/clawd/projects/recipe-app" in command
        assert "/Users/admin/clawd/projects/portfolio-site" in command

    def test_create_project_files_seeds_recipe_and_portfolio_project_md(self):
        calls = []

        class _Vm:
            def ssh(self, command, **_kwargs):
                calls.append(command)

                class _Result:
                    returncode = 0
                    stdout = ""
                    stderr = ""

                return _Result()

        vmb._create_project_files(_Vm())
        assert any("projects/recipe-app" in call for call in calls)
        assert any("projects/portfolio-site" in call for call in calls)
        assert any("Recipe App Project" in call for call in calls)
        assert any("Portfolio Site Project" in call for call in calls)

    def test_register_vm_benchmark_projects_uses_product_cli(self):
        calls = []

        class _Vm:
            def ssh(self, command, **_kwargs):
                calls.append(command)

                class _Result:
                    returncode = 0
                    stdout = "registered benchmark projects via product CLI\n"
                    stderr = ""

                return _Result()

        vmb._register_vm_benchmark_projects(_Vm())
        command = calls[0]
        assert f"cd {vmb.VM_QUAID_DIR}" in command
        assert f"QUAID_HOME={vmb.VM_QUAID_HOME}" in command
        assert f"QUAID_INSTANCE={vmb.VM_QUAID_INSTANCE}" in command
        assert "./quaid', 'project', 'create'" in command
        assert "./quaid', 'project', 'link', name" in command
        assert "./quaid', 'project', 'update', name" in command
        assert "recipe-app" in command
        assert "portfolio-site" in command

    def test_scrape_janitor_errors_strips_nuls_before_grep(self):
        calls = []

        class _Vm:
            def ssh(self, command, **_kwargs):
                calls.append(command)

                class _Result:
                    returncode = 0
                    stdout = ""
                    stderr = ""

                return _Result()

        assert vmb._scrape_janitor_errors(_Vm()) == []
        assert "tr -d '\\000'" in calls[0]
        assert "^Decayed \\(" in calls[0]

    def test_run_vm_janitor_fail_hard_on_residual_pending_or_approved(self):
        class _Vm:
            def __init__(self):
                self.calls = 0

            def ssh(self, command, **_kwargs):
                self.calls += 1

                class _Result:
                    def __init__(self, returncode=0, stdout="", stderr=""):
                        self.returncode = returncode
                        self.stdout = stdout
                        self.stderr = stderr

                if self.calls == 1:
                    return _Result(stdout="janitor ok\n")
                if self.calls == 2:
                    return _Result(stdout="")
                if self.calls == 3:
                    return _Result(stdout='{"active": 7, "pending": 2}\n')
                return _Result(stdout="")

        with pytest.raises(RuntimeError, match="pending=2"):
            vmb._run_vm_janitor(_Vm())

    def test_write_local_extraction_artifact_uses_results_dir_directly(self, tmp_path):
        vmb._write_local_extraction_artifact(tmp_path, "benchmark-quaid-s01", {"facts": []})

        artifact_path = tmp_path / "extract-artifacts" / "benchmark-quaid-s01.json"
        assert artifact_path.exists()
        assert not (tmp_path / "quaid-timeout").exists()

    def test_register_session_accepts_explicit_session_file(self):
        calls = []

        class _Vm:
            def ssh(self, command, **_kwargs):
                calls.append(command)

                class _Result:
                    returncode = 0
                    stdout = ""
                    stderr = ""

                return _Result()

        vmb._register_session(
            _Vm(),
            "eval-q007",
            session_key="agent:main:hook:eval-q007",
            session_file="~/.openclaw/agents/benchmark-eval/sessions/eval-q007.jsonl",
        )
        command = calls[0]
        assert "agents/benchmark-eval/sessions/eval-q007.jsonl" in command
        assert "os.makedirs(os.path.dirname(session_file), exist_ok=True)" in command

    def test_native_config_respects_embed_base_url_override(self, monkeypatch):
        monkeypatch.setattr(vmb, "OC_NATIVE_EMBED_BASE_URL", "http://192.168.64.1:11435/v1")
        script = vmb._build_openclaw_native_config_script(enable_session_hook=True)
        assert "remote['baseUrl'] = 'http://192.168.64.1:11435/v1'" in script

    def test_extract_openclaw_memory_status_tolerates_warning_prefix(self):
        payload = f'[{{"status":{{"provider":"openai","model":"{vmb.OC_NATIVE_EMBED_MODEL}"}}}}]'
        stdout = (
            "Config warnings:\\n"
            "- plugins.entries.quaid: stale config ignored\\n"
            f"{payload}"
        )
        parsed = vmb._extract_openclaw_memory_status(stdout)
        assert parsed[0]["status"]["provider"] == "openai"

    def test_validate_native_memory_retries_embedding_probe(self, monkeypatch):
        calls = []

        class _Vm:
            def ssh(self, command, **_kwargs):
                calls.append(command)

                class _Result:
                    def __init__(self, returncode=0, stdout="", stderr=""):
                        self.returncode = returncode
                        self.stdout = stdout
                        self.stderr = stderr

                if len(calls) == 1:
                    return _Result(
                        0,
                        f'{{"provider":"openai","model":"{vmb.OC_NATIVE_EMBED_MODEL}","baseUrl":"http://127.0.0.1:11435/v1"}}',
                        "",
                    )
                if len(calls) == 2:
                    return _Result(1, "", "socket.timeout: timed out")
                return _Result(0, f'{{"ok": true, "dims": {vmb.OC_NATIVE_EMBED_DIMS}}}', "")

        sleeps = []
        monkeypatch.setattr(vmb.time, "sleep", lambda seconds: sleeps.append(seconds))
        vmb._validate_openclaw_native_memory(_Vm())
        assert sleeps == [3]

    def test_ensure_oc_native_embed_proxy_restarts_proxy_each_run(self, monkeypatch, tmp_path):
        monkeypatch.setattr(vmb, "OC_NATIVE_EMBED_BASE_URL", "http://127.0.0.1:11435/v1")
        monkeypatch.setattr(vmb, "OC_NATIVE_EMBED_PROXY_SCRIPT", tmp_path / "proxy.py")
        monkeypatch.setattr(vmb, "OC_NATIVE_EMBED_PROXY_PIDFILE", tmp_path / "proxy.pid")
        monkeypatch.setattr(vmb, "OC_NATIVE_EMBED_PROXY_LOG", tmp_path / "proxy.log")
        probe_calls = []
        warm_calls = []

        def _probe(url, timeout=5):
            probe_calls.append(url)
            return True, '{"ok":true}'

        monkeypatch.setattr(vmb, "OC_NATIVE_EMBED_UPSTREAM", "http://127.0.0.1:11434")
        monkeypatch.setattr(vmb, "_probe_json_url", _probe)
        monkeypatch.setattr(vmb, "_prepare_oc_native_embed_upstream", lambda: warm_calls.append("prepare"))
        monkeypatch.setattr(vmb, "_wait_for_oc_native_embed_upstream_ready", lambda: warm_calls.append("upstream"))
        monkeypatch.setattr(vmb, "_wait_for_oc_native_embed_proxy_ready", lambda port: warm_calls.append(f"proxy:{port}"))
        popen_calls = []
        stop_calls = []

        class _Proc:
            pid = 43210

        monkeypatch.setattr(vmb, "_stop_oc_native_embed_proxy", lambda port: stop_calls.append(port))
        monkeypatch.setattr(vmb.subprocess, "Popen", lambda *args, **kwargs: popen_calls.append((args, kwargs)) or _Proc())
        vmb._ensure_oc_native_embed_proxy()
        assert stop_calls == [11435]
        assert len(popen_calls) == 1
        assert warm_calls == ["prepare", "upstream", "proxy:11435"]
        assert probe_calls == ["http://127.0.0.1:11434/v1/models", "http://127.0.0.1:11435/v1/models"]

    def test_ensure_oc_native_embed_proxy_raises_when_upstream_embeddings_never_become_ready(self, monkeypatch, tmp_path):
        monkeypatch.setattr(vmb, "OC_NATIVE_EMBED_BASE_URL", "http://127.0.0.1:11435/v1")
        monkeypatch.setattr(vmb, "OC_NATIVE_EMBED_UPSTREAM", "http://127.0.0.1:11434")
        monkeypatch.setattr(vmb, "OC_NATIVE_EMBED_PROXY_SCRIPT", tmp_path / "proxy.py")
        monkeypatch.setattr(vmb, "OC_NATIVE_EMBED_PROXY_PIDFILE", tmp_path / "proxy.pid")
        monkeypatch.setattr(vmb, "OC_NATIVE_EMBED_PROXY_LOG", tmp_path / "proxy.log")
        probe_calls = []

        def _probe(url, timeout=5):
            probe_calls.append(url)
            return True, '{"ok":true}'

        monkeypatch.setattr(vmb, "_probe_json_url", _probe)
        monkeypatch.setattr(vmb, "_prepare_oc_native_embed_upstream", lambda: None)
        monkeypatch.setattr(
            vmb,
            "_wait_for_oc_native_embed_upstream_ready",
            lambda: (_ for _ in ()).throw(RuntimeError("cold start hung")),
        )
        popen_calls = []

        class _Proc:
            pid = 24680

        stop_calls = []
        monkeypatch.setattr(vmb.subprocess, "Popen", lambda *args, **kwargs: popen_calls.append((args, kwargs)) or _Proc())
        monkeypatch.setattr(vmb, "_stop_oc_native_embed_proxy", lambda port: stop_calls.append(port))
        with pytest.raises(RuntimeError, match="host embed upstream embeddings not ready"):
            vmb._ensure_oc_native_embed_proxy()
        assert stop_calls == []
        assert popen_calls == []
        assert probe_calls == ["http://127.0.0.1:11434/v1/models"]

    def test_ensure_oc_native_embed_proxy_starts_fresh_proxy(self, monkeypatch, tmp_path):
        monkeypatch.setattr(vmb, "OC_NATIVE_EMBED_BASE_URL", "http://127.0.0.1:11435/v1")
        monkeypatch.setattr(vmb, "OC_NATIVE_EMBED_UPSTREAM", "http://127.0.0.1:11434")
        monkeypatch.setattr(vmb, "OC_NATIVE_EMBED_PROXY_SCRIPT", tmp_path / "proxy.py")
        monkeypatch.setattr(vmb, "OC_NATIVE_EMBED_PROXY_PIDFILE", tmp_path / "proxy.pid")
        monkeypatch.setattr(vmb, "OC_NATIVE_EMBED_PROXY_LOG", tmp_path / "proxy.log")
        monkeypatch.setattr(vmb.time, "sleep", lambda _seconds: None)
        monkeypatch.setattr(vmb, "_prepare_oc_native_embed_upstream", lambda: None)
        monkeypatch.setattr(vmb, "_wait_for_oc_native_embed_upstream_ready", lambda: None)
        proxy_warm_calls = []
        monkeypatch.setattr(vmb, "_wait_for_oc_native_embed_proxy_ready", lambda port: proxy_warm_calls.append(port))
        stop_calls = []
        monkeypatch.setattr(vmb, "_stop_oc_native_embed_proxy", lambda port: stop_calls.append(port))
        probe_results = iter([
            (True, '{"ok":true}'),
            (True, '{"ok":true}'),
        ])
        monkeypatch.setattr(vmb, "_probe_json_url", lambda url, timeout=5: next(probe_results))
        popen_calls = []

        class _Proc:
            pid = 43210

        monkeypatch.setattr(vmb.subprocess, "Popen", lambda *args, **kwargs: popen_calls.append((args, kwargs)) or _Proc())
        vmb._ensure_oc_native_embed_proxy()
        args, kwargs = popen_calls[0]
        assert args[0][0] == vmb.sys.executable
        assert "--host" in args[0]
        assert "0.0.0.0" in args[0]
        assert "--port" in args[0]
        assert "11435" in args[0]
        assert kwargs["stdout"] is not None
        assert kwargs["stderr"] is not None
        assert kwargs["stdin"] == vmb.subprocess.DEVNULL
        assert (tmp_path / "proxy.pid").read_text().strip() == "43210"
        assert proxy_warm_calls == [11435]
        assert stop_calls == [11435]

    def test_ensure_oc_native_embed_proxy_routes_through_tart_host(self, monkeypatch):
        monkeypatch.setattr(vmb, "OC_NATIVE_EMBED_BASE_URL", "http://127.0.0.1:11435/v1")
        monkeypatch.setattr(vmb, "OC_NATIVE_EMBED_UPSTREAM", "http://127.0.0.1:11434")
        monkeypatch.setattr(vmb.time, "sleep", lambda _seconds: None)
        probe_calls = []
        prepare_calls = []
        upstream_wait_calls = []
        proxy_wait_calls = []
        start_calls = []

        def _probe(url, timeout=5, tart_host=None):
            probe_calls.append((url, timeout, tart_host))
            return True, '{"ok":true}'

        monkeypatch.setattr(vmb, "_probe_json_url", _probe)
        monkeypatch.setattr(
            vmb,
            "_prepare_oc_native_embed_upstream",
            lambda tart_host=None: prepare_calls.append(tart_host),
        )
        monkeypatch.setattr(
            vmb,
            "_wait_for_oc_native_embed_upstream_ready",
            lambda tart_host=None: upstream_wait_calls.append(tart_host),
        )
        monkeypatch.setattr(
            vmb,
            "_start_oc_native_embed_proxy",
            lambda upstream, port, tart_host=None: start_calls.append((upstream, port, tart_host)),
        )
        monkeypatch.setattr(
            vmb,
            "_wait_for_oc_native_embed_proxy_ready",
            lambda port, tart_host=None: proxy_wait_calls.append((port, tart_host)),
        )

        vmb._ensure_oc_native_embed_proxy("alfie.local")

        assert probe_calls == [
            ("http://127.0.0.1:11434/v1/models", 15, "alfie.local"),
            ("http://127.0.0.1:11435/v1/models", 5, "alfie.local"),
        ]
        assert prepare_calls == ["alfie.local"]
        assert upstream_wait_calls == ["alfie.local"]
        assert start_calls == [("http://127.0.0.1:11434", 11435, "alfie.local")]
        assert proxy_wait_calls == [(11435, "alfie.local")]

    def test_ensure_oc_native_embed_proxy_raises_when_proxy_embeddings_never_become_ready(self, monkeypatch, tmp_path):
        monkeypatch.setattr(vmb, "OC_NATIVE_EMBED_BASE_URL", "http://127.0.0.1:11435/v1")
        monkeypatch.setattr(vmb, "OC_NATIVE_EMBED_UPSTREAM", "http://127.0.0.1:11434")
        monkeypatch.setattr(vmb, "OC_NATIVE_EMBED_PROXY_SCRIPT", tmp_path / "proxy.py")
        monkeypatch.setattr(vmb, "OC_NATIVE_EMBED_PROXY_PIDFILE", tmp_path / "proxy.pid")
        monkeypatch.setattr(vmb, "OC_NATIVE_EMBED_PROXY_LOG", tmp_path / "proxy.log")
        monkeypatch.setattr(vmb.time, "sleep", lambda _seconds: None)
        monkeypatch.setattr(vmb, "_prepare_oc_native_embed_upstream", lambda: None)
        monkeypatch.setattr(vmb, "_wait_for_oc_native_embed_upstream_ready", lambda: None)
        probe_results = iter([
            (True, '{"ok":true}'),
            (True, '{"ok":true}'),
        ])
        monkeypatch.setattr(vmb, "_probe_json_url", lambda url, timeout=5: next(probe_results))
        monkeypatch.setattr(
            vmb,
            "_wait_for_oc_native_embed_proxy_ready",
            lambda port: (_ for _ in ()).throw(RuntimeError("slow proxy")),
        )
        stop_calls = []
        monkeypatch.setattr(vmb, "_stop_oc_native_embed_proxy", lambda port: stop_calls.append(port))
        popen_calls = []

        class _Proc:
            pid = 54321

        monkeypatch.setattr(vmb.subprocess, "Popen", lambda *args, **kwargs: popen_calls.append((args, kwargs)) or _Proc())
        with pytest.raises(RuntimeError, match="embeddings never became ready"):
            vmb._ensure_oc_native_embed_proxy()
        assert stop_calls == [11435]
        assert len(popen_calls) == 1

    def test_wait_for_oc_native_embed_proxy_ready_retries_until_success(self, monkeypatch):
        calls = []
        attempts = iter([RuntimeError("cold"), RuntimeError("still cold"), None])
        sleeps = []

        def _warm(port):
            calls.append(port)
            result = next(attempts)
            if result is not None:
                raise result

        now = {"value": 1000.0}

        monkeypatch.setattr(vmb, "_warm_oc_native_embed_proxy", _warm)
        monkeypatch.setattr(vmb, "OC_NATIVE_EMBED_PROXY_READY_WAIT_S", 30)
        monkeypatch.setattr(vmb, "OC_NATIVE_EMBED_PROXY_READY_POLL_S", 3)
        monkeypatch.setattr(vmb.time, "time", lambda: now["value"])

        def _sleep(seconds):
            sleeps.append(seconds)
            now["value"] += seconds

        monkeypatch.setattr(vmb.time, "sleep", _sleep)
        vmb._wait_for_oc_native_embed_proxy_ready(11435)
        assert calls == [11435, 11435, 11435]
        assert sleeps == [3, 3]

    def test_ensure_oc_native_embed_proxy_raises_when_host_upstream_unhealthy(self, monkeypatch):
        monkeypatch.setattr(vmb, "OC_NATIVE_EMBED_UPSTREAM", "http://127.0.0.1:11434")
        monkeypatch.setattr(vmb, "_probe_json_url", lambda url, timeout=5: (False, "refused"))
        with pytest.raises(RuntimeError, match="host embed upstream not ready"):
            vmb._ensure_oc_native_embed_proxy()

    def test_prepare_oc_native_embed_upstream_stops_loaded_models(self, monkeypatch):
        stop_calls = []
        sleeps = []
        monkeypatch.setattr(
            vmb,
            "_list_loaded_ollama_models",
            lambda: ["nomic-embed-text:latest", "qwen3-embedding:8b"],
        )
        monkeypatch.setattr(vmb, "_stop_ollama_model", lambda model: stop_calls.append(model))
        monkeypatch.setattr(vmb.time, "sleep", lambda seconds: sleeps.append(seconds))
        vmb._prepare_oc_native_embed_upstream()
        assert stop_calls == ["nomic-embed-text:latest", "qwen3-embedding:8b"]
        assert sleeps == [2]

    def test_patch_openclaw_native_memory_disables_quaid_config_guard_first(self):
        calls = []

        class _Vm:
            def ssh(self, command, **kwargs):
                calls.append((command, kwargs))

                class _Result:
                    def __init__(self, stdout="ok\n"):
                        self.stdout = stdout
                        self.returncode = 0
                        self.stderr = ""

                return _Result()

        vmb._patch_openclaw_native_memory(_Vm(), enable_session_hook=True)
        assert "ai.openclaw.quaid-config-guard" in calls[0][0]
        assert "setup-quaid.mjs --agent" in calls[0][0]
        assert calls[0][1]["raw"] is True
        assert calls[1][0].startswith("python3 -c ")

    def test_sync_openclaw_native_memory_repairs_config_before_index(self, monkeypatch):
        calls = []
        patch_calls = []

        class _Vm:
            def ssh(self, command, **_kwargs):
                calls.append(command)

                class _Result:
                    def __init__(self, stdout, returncode=0, stderr=""):
                        self.returncode = returncode
                        self.stderr = stderr
                        self.stdout = stdout

                if "openclaw memory index --agent main" in command:
                    return _Result("")
                if command == "openclaw memory status --agent main --json":
                    return _Result(
                        '[{"status":{"dirty":false,"sourceCounts":['
                        '{"source":"sessions","files":2,"chunks":14}'
                        ']}}]'
                    )
                return _Result("Memory index updated (main).")

        monkeypatch.setattr(
            vmb,
            "_patch_openclaw_native_memory",
            lambda vm, enable_session_hook=True: patch_calls.append(enable_session_hook),
        )
        status = vmb._sync_openclaw_native_memory(_Vm(), source_name="sessions", min_indexed_files=2)
        assert status["dirty"] is False
        assert patch_calls == [True]
        assert calls[0].startswith(
            "sh -lc 'export PATH=/opt/homebrew/bin:$PATH; "
            "openclaw memory index --agent main > /tmp/oc-native-reindex.log 2>&1'"
        )

    def test_warm_oc_native_embed_upstream_rejects_empty_embedding(self, monkeypatch):
        monkeypatch.setattr(vmb, "OC_NATIVE_EMBED_UPSTREAM", "http://127.0.0.1:11434")
        seen = {}

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        def _fake_urlopen(req, timeout=120):
            seen["url"] = req.full_url
            seen["timeout"] = timeout
            return _Resp()

        monkeypatch.setattr(vmb, "urlopen", _fake_urlopen)
        monkeypatch.setattr(vmb.json, "load", lambda resp: {"data": [{"embedding": []}]})
        with pytest.raises(RuntimeError, match="empty embedding"):
            vmb._warm_oc_native_embed_upstream()
        assert seen["url"] == "http://127.0.0.1:11434/v1/embeddings"
        assert seen["timeout"] == vmb.OC_NATIVE_EMBED_UPSTREAM_WARMUP_TIMEOUT_S

    def test_warm_oc_native_embed_proxy_uses_local_proxy_endpoint(self, monkeypatch):
        seen = {}

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        def _fake_urlopen(req, timeout=120):
            seen["url"] = req.full_url
            seen["timeout"] = timeout
            return _Resp()

        monkeypatch.setattr(vmb, "urlopen", _fake_urlopen)
        monkeypatch.setattr(vmb.json, "load", lambda resp: {"data": [{"embedding": [0.1, 0.2]}]})
        vmb._warm_oc_native_embed_proxy(11435)
        assert seen["url"] == "http://127.0.0.1:11435/v1/embeddings"
        assert seen["timeout"] == vmb.OC_NATIVE_EMBED_PROXY_WARMUP_TIMEOUT_S

    def test_validate_native_memory_reads_config_and_embedding_probe(self):
        calls = []
        timeouts = []

        class _Vm:
            def ssh(self, command, **kwargs):
                calls.append(command)
                timeouts.append(kwargs.get("timeout"))

                class _Result:
                    def __init__(self, returncode=0, stdout="", stderr=""):
                        self.returncode = returncode
                        self.stdout = stdout
                        self.stderr = stderr

                if len(calls) == 1:
                    return _Result(
                        0,
                        f'{{"provider":"openai","model":"{vmb.OC_NATIVE_EMBED_MODEL}","baseUrl":"http://127.0.0.1:11435/v1"}}',
                        "",
                    )
                return _Result(0, f'{{"ok": true, "dims": {vmb.OC_NATIVE_EMBED_DIMS}}}', "")

        vmb._validate_openclaw_native_memory(_Vm())
        assert "openclaw memory status" not in "\n".join(calls)
        assert f"urlopen(req, timeout={vmb.OC_NATIVE_EMBED_VALIDATION_PROBE_TIMEOUT_S})" in calls[1]
        assert timeouts[1] == vmb.OC_NATIVE_EMBED_VALIDATION_PROBE_TIMEOUT_S + 15

    def test_validate_native_memory_refreshes_tart_ip_before_probe_attempts(self):
        refreshes = []

        class _Vm:
            def __init__(self):
                self.calls = 0

            def _refresh_ip_from_tart(self):
                refreshes.append("refresh")

            def ssh(self, command, **_kwargs):
                self.calls += 1

                class _Result:
                    def __init__(self, returncode=0, stdout="", stderr=""):
                        self.returncode = returncode
                        self.stdout = stdout
                        self.stderr = stderr

                if self.calls == 1:
                    return _Result(
                        0,
                        f'{{"provider":"openai","model":"{vmb.OC_NATIVE_EMBED_MODEL}","baseUrl":"http://127.0.0.1:11435/v1"}}',
                        "",
                    )
                if self.calls == 2:
                    return _Result(1, "", "Host is down")
                return _Result(0, f'{{"ok": true, "dims": {vmb.OC_NATIVE_EMBED_DIMS}}}', "")

        vmb._validate_openclaw_native_memory(_Vm())
        assert refreshes == ["refresh", "refresh"]

    def test_validate_quaid_vm_embeddings_retries_until_success(self, monkeypatch):
        calls = []
        timeouts = []
        sleeps = []

        class _Vm:
            def __init__(self):
                self.calls = 0

            def _refresh_ip_from_tart(self):
                calls.append("refresh")

            def ssh(self, command, **kwargs):
                self.calls += 1
                calls.append(command)
                timeouts.append(kwargs.get("timeout"))

                class _Result:
                    def __init__(self, returncode=0, stdout="", stderr=""):
                        self.returncode = returncode
                        self.stdout = stdout
                        self.stderr = stderr

                if self.calls == 1:
                    return _Result(1, "", "Connection refused")
                return _Result(0, f'{{"ok": true, "dims": {vmb.OC_NATIVE_EMBED_DIMS}}}', "")

        monkeypatch.setattr(vmb.time, "sleep", lambda seconds: sleeps.append(seconds))
        vmb._validate_quaid_vm_embeddings(_Vm())
        assert sleeps == [3]
        assert calls[0] == "refresh"
        assert "/api/embed" in calls[1]
        assert f"urlopen(req, timeout={vmb.OC_NATIVE_EMBED_VALIDATION_PROBE_TIMEOUT_S})" in calls[1]
        assert timeouts[0] == vmb.OC_NATIVE_EMBED_VALIDATION_PROBE_TIMEOUT_S + 15

    def test_oc_native_session_ids_are_stable_per_review(self):
        arc = type("Review", (), {"session_num": 3})()
        filler = type("Review", (), {"session_num": -18})()
        weird = type("Review", (), {"session_num": None})()
        assert vmb._oc_native_session_id(arc, 0) == "benchmark-oc-native-s03"
        assert vmb._oc_native_session_id(filler, 1) == "benchmark-oc-native-f018"
        assert vmb._oc_native_session_id(weird, 2) == "benchmark-oc-native-r002"

    def test_oc_native_session_key_is_agent_scoped_and_unique(self):
        assert vmb._oc_native_session_key("benchmark-oc-native-s03") == "agent:main:benchmark-oc-native-s03"
        assert vmb._oc_native_session_key("eval-q007") == "agent:main:eval-q007"

    def test_quaid_chunk_session_ids_are_derived_from_source_sessions(self):
        chunk = type("Chunk", (), {"session_ids": ["S14", "S15"]})()
        assert vmb._quaid_chunk_session_id("benchmark-quaid", chunk, 0) == "benchmark-quaid-s14-s15"
        empty = type("Chunk", (), {"session_ids": []})()
        assert vmb._quaid_chunk_session_id("benchmark-quaid", empty, 2) == "benchmark-quaid-chunk03"

    def test_messages_to_oc_native_jsonl_seeds_native_session_shape(self):
        jsonl = vmb.messages_to_oc_native_jsonl(
            [
                {"role": "user", "content": "my dog is biscuit"},
                {"role": "assistant", "content": "Got it, Biscuit."},
            ],
            "benchmark-oc-native-s02",
            started_at_ms=1_777_094_924_000,
        )
        lines = [json.loads(line) for line in jsonl.splitlines()]
        assert lines[0] == {
            "type": "session",
            "version": 3,
            "id": "benchmark-oc-native-s02",
            "timestamp": "2026-04-25T05:28:44.000Z",
            "cwd": str(Path.home() / ".openclaw" / "workspace"),
        }
        assert lines[1]["type"] == "model_change"
        assert lines[1]["parentId"] is None
        assert lines[1]["provider"] == "openai"
        assert lines[1]["modelId"] == "gpt-5.4"
        assert lines[2]["type"] == "message"
        assert lines[2]["parentId"] == lines[1]["id"]
        assert lines[2]["message"] == {
            "role": "user",
            "content": [{"type": "text", "text": "my dog is biscuit"}],
            "timestamp": 1_777_094_924_001,
        }
        assert lines[3]["type"] == "message"
        assert lines[3]["parentId"] == lines[2]["id"]
        assert lines[3]["message"] == {
            "role": "assistant",
            "content": [{"type": "text", "text": "Got it, Biscuit."}],
            "timestamp": 1_777_094_924_002,
        }

    def test_messages_to_oc_native_jsonl_preserves_timestamped_message_objects(self):
        msg1 = type("Msg", (), {"role": "user", "content": "first", "timestamp_ms": 1_700_000_000_123})()
        msg2 = type("Msg", (), {"role": "assistant", "content": "second", "timestamp_ms": 1_700_000_000_456})()
        jsonl = vmb.messages_to_oc_native_jsonl(
            [msg1, msg2],
            "benchmark-quaid-s10",
            model_id="claude-sonnet-4-5-20250929",
            provider="anthropic",
        )
        lines = [json.loads(line) for line in jsonl.splitlines()]
        assert lines[0]["timestamp"] == "2023-11-14T22:13:20.123Z"
        assert lines[1]["provider"] == "anthropic"
        assert lines[1]["modelId"] == "claude-sonnet-4-5-20250929"
        assert lines[2]["message"]["timestamp"] == 1_700_000_000_123
        assert lines[3]["message"]["timestamp"] == 1_700_000_000_456

    def test_sync_openclaw_native_wiki_runs_bridge_compile(self):
        calls = []

        class _Vm:
            def ssh(self, command, **_kwargs):
                calls.append(command)

                class _Result:
                    returncode = 0
                    stdout = '{"ok":true}'
                    stderr = ""

                return _Result()

        vmb._sync_openclaw_native_wiki(_Vm())
        assert "openclaw wiki init" in calls[0]
        assert "openclaw wiki bridge import" in calls[0]
        assert "openclaw wiki compile" in calls[0]
        assert "openclaw wiki status --json" in calls[0]

    def test_patch_gateway_model_preserves_provider_prefixed_openai_model(self):
        calls = []

        class _Vm:
            def ssh(self, command, **_kwargs):
                calls.append(command)

                class _Result:
                    stdout = "Gateway model set to: openai/gpt-5.4\n"

                return _Result()

        vmb._patch_gateway_model(_Vm(), "openai/gpt-5.4")
        assert "openai/gpt-5.4" in calls[0]

    def test_patch_gateway_model_keeps_legacy_bare_model_as_anthropic(self):
        calls = []

        class _Vm:
            def ssh(self, command, **_kwargs):
                calls.append(command)

                class _Result:
                    stdout = "Gateway model set to: anthropic/claude-haiku-4-5-20251001\n"

                return _Result()

        vmb._patch_gateway_model(_Vm(), "claude-haiku-4-5-20251001")
        assert "anthropic/claude-haiku-4-5-20251001" in calls[0]

    def test_resolve_gateway_answer_model_maps_openai_to_openai_codex_for_codex_oauth(self):
        resolved = vmb._resolve_gateway_answer_model(
            "openai/gpt-5.4",
            system="quaid",
            openai_auth_mode="codex-oauth",
        )
        assert resolved == "openai-codex/gpt-5.4"

    def test_resolve_gateway_answer_model_keeps_openai_for_api_mode(self):
        resolved = vmb._resolve_gateway_answer_model(
            "openai/gpt-5.4",
            system="oc-native",
            openai_auth_mode="api",
        )
        assert resolved == "openai/gpt-5.4"

    def test_reapply_oc_native_gateway_runtime_reprovisions_openai_api_auth(self, monkeypatch):
        auth_calls = []
        model_calls = []
        monkeypatch.setattr(vmb, "_OC_NATIVE_GATEWAY_ANSWER_MODEL", "openai/gpt-5.4")
        monkeypatch.setattr(vmb, "_OC_NATIVE_GATEWAY_OPENAI_AUTH_MODE", "api")
        monkeypatch.setattr(vmb, "_provision_openclaw_openai_key", lambda vm: auth_calls.append("api"))
        monkeypatch.setattr(vmb, "_provision_openclaw_codex_oauth", lambda vm: auth_calls.append("oauth"))
        monkeypatch.setattr(vmb, "_patch_gateway_model", lambda vm, model: model_calls.append(model))
        vmb._reapply_oc_native_gateway_runtime(object())
        assert auth_calls == ["api"]
        assert model_calls == ["openai/gpt-5.4"]

    def test_reapply_oc_native_gateway_runtime_maps_openai_for_codex_oauth(self, monkeypatch):
        auth_calls = []
        model_calls = []
        monkeypatch.setattr(vmb, "_OC_NATIVE_GATEWAY_ANSWER_MODEL", "openai/gpt-5.4")
        monkeypatch.setattr(vmb, "_OC_NATIVE_GATEWAY_OPENAI_AUTH_MODE", "codex-oauth")
        monkeypatch.setattr(vmb, "_provision_openclaw_openai_key", lambda vm: auth_calls.append("api"))
        monkeypatch.setattr(vmb, "_provision_openclaw_codex_oauth", lambda vm: auth_calls.append("oauth"))
        monkeypatch.setattr(vmb, "_patch_gateway_model", lambda vm, model: model_calls.append(model))
        vmb._reapply_oc_native_gateway_runtime(object())
        assert auth_calls == ["oauth"]
        assert model_calls == ["openai-codex/gpt-5.4"]

    def test_provision_openclaw_openai_key_writes_env_and_auth_profile(self, monkeypatch):
        calls = []

        class _Vm:
            def ssh(self, command, input_data=None, **_kwargs):
                calls.append((command, input_data))

                class _Result:
                    returncode = 0
                    stdout = "OpenClaw direct OpenAI auth installed (.env + auth profile)\n"
                    stderr = ""

                return _Result()

        monkeypatch.setattr(vmb, "_resolve_openai_api_key_for_vm", lambda: "sk-test")
        vmb._provision_openclaw_openai_key(_Vm())
        command = calls[0][0]
        assert calls[0][1] == "sk-test"
        assert "'openai:default'" in command
        assert "version" in command
        assert "api_key" in command
        assert "provider" in command
        assert "openai" in command
        assert "key" in command
        assert "token" in command
        assert "access" in command
        assert "last_good" in command
        assert "OPENAI_API_KEY=" in command
        assert "env_path" in command

    def test_provision_openclaw_openai_key_requires_key(self, monkeypatch):
        monkeypatch.setattr(vmb, "_resolve_openai_api_key_for_vm", lambda: "")

        class _Vm:
            pass

        with pytest.raises(RuntimeError, match="OPENAI_API_KEY is required"):
            vmb._provision_openclaw_openai_key(_Vm())

    def test_resolve_anthropic_credential_for_vm_reads_first_key_path_relative_to_dev_config(self, monkeypatch, tmp_path):
        home = tmp_path / "home"
        dev_cfg = home / "quaidcode" / "dev" / ".quaid-dev.local.json"
        token_file = home / "quaidcode" / "anthtoken-yuni.md"
        dev_cfg.parent.mkdir(parents=True)
        token_file.parent.mkdir(parents=True, exist_ok=True)
        dev_cfg.write_text(
            json.dumps(
                {
                    "auth": {
                        "anthropic": {
                            "firstKeyPath": "../anthtoken-yuni.md",
                            "secondKeyPath": "../anthtoken-mom.md",
                        }
                    }
                }
            )
        )
        token_file.write_text("sk-ant-oat01-primary-token\n")
        monkeypatch.delenv("BENCHMARK_ANTHROPIC_OAUTH_TOKEN", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr(vmb.Path, "home", lambda: home)

        assert vmb._resolve_anthropic_credential_for_vm() == "sk-ant-oat01-primary-token"

    def test_provision_openclaw_anthropic_key_writes_guest_env(self, monkeypatch):
        calls = []

        class _Vm:
            def ssh(self, command, input_data=None, **_kwargs):
                calls.append((command, input_data))

                class _Result:
                    returncode = 0
                    stdout = "OpenClaw Anthropic runtime credential installed (.env + shared auth)\n"
                    stderr = ""

                return _Result()

        monkeypatch.setattr(vmb, "_resolve_anthropic_credential_for_vm", lambda: "sk-ant-oat01-primary-token")
        vmb._provision_openclaw_anthropic_key(_Vm())
        assert calls[0][1] == "sk-ant-oat01-primary-token"
        assert "ANTHROPIC_API_KEY=" in calls[0][0]
        assert ".openclaw" in calls[0][0]
        assert "anthropic_oauth" in calls[0][0]
        assert "runtime_quaid_home" in calls[0][0]
        assert "credentials.json" in calls[0][0]

    def test_provision_openclaw_anthropic_key_requires_credential(self, monkeypatch):
        monkeypatch.setattr(vmb, "_resolve_anthropic_credential_for_vm", lambda: "")

        class _Vm:
            pass

        with pytest.raises(RuntimeError, match="Anthropic benchmark credential is required"):
            vmb._provision_openclaw_anthropic_key(_Vm())

    def test_derive_quaid_runtime_llm_config_keeps_anthropic_runtime_for_openai_answer_lane(self):
        cfg = vmb._derive_quaid_runtime_llm_config(
            extract_model="claude-sonnet-4-5-20250929",
            answer_model="openai/gpt-5.4",
        )
        assert cfg["llmProvider"] == "anthropic"
        assert cfg["fastReasoningProvider"] == "anthropic"
        assert cfg["deepReasoningProvider"] == "anthropic"
        assert cfg["fastReasoning"] == "claude-haiku-4-5-20251001"
        assert cfg["deepReasoning"] == "claude-sonnet-4-5-20250929"
        assert cfg["fastReasoningEffort"] == "none"
        assert cfg["deepReasoningEffort"] == "high"
        assert cfg["llm_provider"] == "anthropic"
        assert cfg["fast_reasoning_provider"] == "anthropic"
        assert cfg["deep_reasoning_provider"] == "anthropic"
        assert cfg["fast_reasoning"] == "claude-haiku-4-5-20251001"
        assert cfg["deep_reasoning"] == "claude-sonnet-4-5-20250929"

    def test_derive_quaid_runtime_llm_config_falls_back_to_anthropic_extract_lane(self):
        cfg = vmb._derive_quaid_runtime_llm_config(
            extract_model="claude-sonnet-4-5-20250929",
            answer_model=None,
        )
        assert cfg["llmProvider"] == "anthropic"
        assert cfg["fastReasoningProvider"] == "anthropic"
        assert cfg["deepReasoningProvider"] == "anthropic"
        assert cfg["fastReasoning"] == "claude-haiku-4-5-20251001"
        assert cfg["deepReasoning"] == "claude-sonnet-4-5-20250929"
        assert cfg["llm_provider"] == "anthropic"
        assert cfg["fast_reasoning_provider"] == "anthropic"
        assert cfg["deep_reasoning_provider"] == "anthropic"

    def test_patch_quaid_runtime_instance_config_targets_instance_override(self):
        calls = []

        class _Vm:
            def ssh(self, command, **_kwargs):
                calls.append(command)

                class _Result:
                    returncode = 0
                    stdout = "Patched runtime config: /Users/admin/clawd/instances/openclaw-main/config.json\n"
                    stderr = ""

                return _Result()

        vmb._patch_quaid_runtime_instance_config(
            _Vm(),
            extract_model="claude-sonnet-4-5-20250929",
            answer_model="openai/gpt-5.4",
            owner_id="maya",
            user_name="Maya",
        )
        command = calls[0]
        assert "instances/{instance_id}/config.json" in command
        assert "openclaw-main" in command
        assert "ollama_url =" in command
        assert vmb.VM_QUAID_OLLAMA_URL in command
        assert "default_owner" in command
        assert "defaultOwner" in command
        assert "person_node_name" in command
        assert "personNodeName" in command
        assert "\"llmProvider\": \"anthropic\"" in command
        assert "\"fastReasoning\": \"claude-haiku-4-5-20251001\"" in command
        assert "\"deepReasoning\": \"claude-sonnet-4-5-20250929\"" in command
        assert "\"llm_provider\": \"anthropic\"" in command
        assert "\"fast_reasoning\": \"claude-haiku-4-5-20251001\"" in command
        assert "\"deep_reasoning\": \"claude-sonnet-4-5-20250929\"" in command
        assert "defaultLimit" in command
        assert "maxLimit" in command
        assert "failHard" in command
        assert "embedding_dim" in command
        assert "embeddingDim" in command
        assert "embeddings_provider" in command
        assert "embeddingsProvider" in command

    def test_patch_memory_json_sets_snake_case_owner_identity_keys(self):
        calls = []

        class _Vm:
            def ssh(self, command, **_kwargs):
                calls.append(command)

                class _Result:
                    returncode = 0
                    stdout = "Patched: /Users/admin/clawd/config/memory.json\n"
                    stderr = ""

                return _Result()

        vmb._patch_memory_json(
            _Vm(),
            "claude-sonnet-4-5-20250929",
            owner_id="maya",
            user_name="Maya",
        )
        command = calls[0]
        assert "default_owner" in command
        assert "defaultOwner" in command
        assert "person_node_name" in command
        assert "personNodeName" in command

    def test_provision_openclaw_gateway_openai_auth_uses_codex_oauth(self, monkeypatch):
        calls = []
        monkeypatch.setattr(vmb, "_provision_openclaw_openai_key", lambda vm: calls.append("api"))
        monkeypatch.setattr(vmb, "_provision_openclaw_codex_oauth", lambda vm: calls.append("oauth"))

        vmb._provision_openclaw_gateway_openai_auth(object(), "openai/gpt-5.4", "codex-oauth")

        assert calls == ["oauth"]

    def test_provision_openclaw_codex_oauth_writes_shared_credentials(self, monkeypatch):
        calls = []

        class _Vm:
            def ssh(self, command, input_data=None, **_kwargs):
                calls.append((command, input_data))

                class _Result:
                    returncode = 0
                    stdout = "OpenClaw Codex OAuth shared credential installed\n"
                    stderr = ""

                return _Result()

        monkeypatch.setattr(
            vmb,
            "_resolve_codex_oauth_profile_for_vm",
            lambda: {
                "type": "oauth",
                "provider": "openai-codex",
                "access": (
                    "eyJhbGciOiJub25lIn0."
                    "eyJleHAiOjEyMzQ1LCJodHRwczovL2FwaS5vcGVuYWkuY29tL2F1dGgiOnsiY2hhdGdwdF9hY2NvdW50X2lkIjoiYWNjdC0xMjMifX0."
                    "sig"
                ),
            },
        )
        vmb._provision_openclaw_codex_oauth(_Vm())
        assert json.loads(calls[0][1]) == {
            "type": "oauth",
            "provider": "openai-codex",
            "access": (
                "eyJhbGciOiJub25lIn0."
                "eyJleHAiOjEyMzQ1LCJodHRwczovL2FwaS5vcGVuYWkuY29tL2F1dGgiOnsiY2hhdGdwdF9hY2NvdW50X2lkIjoiYWNjdC0xMjMifX0."
                "sig"
            ),
            "accountId": "acct-123",
            "expires": 12345000,
        }
        assert "credentials.json" in calls[0][0]
        assert "openai-codex:default" in calls[0][0]
        assert "profile[" in calls[0][0]
        assert "access" in calls[0][0]
        assert "accountId" in calls[0][0]
        assert "expires" in calls[0][0]
        assert "legacy_key" in calls[0][0]
        assert "OPENAI_OAUTH_TOKEN" in calls[0][0]
        assert vmb.VM_QUAID_HOME in calls[0][0]
        assert "runtime_quaid_home" in calls[0][0]
        assert "shared" in calls[0][0]
        assert "credentials.json" in calls[0][0]
        assert "openclaw" in calls[0][0]
        assert ".auth-token" in calls[0][0]

    def test_normalize_codex_oauth_profile_for_openclaw_decodes_jwt_metadata(self):
        profile = vmb._normalize_codex_oauth_profile_for_openclaw(
            {
                "type": "oauth",
                "provider": "openai-codex",
                "access": (
                    "eyJhbGciOiJub25lIn0."
                    "eyJleHAiOjEyMzQ1LCJodHRwczovL2FwaS5vcGVuYWkuY29tL2F1dGgiOnsiY2hhdGdwdF9hY2NvdW50X2lkIjoiYWNjdC0xMjMifX0."
                    "sig"
                ),
            }
        )
        assert profile == {
            "type": "oauth",
            "provider": "openai-codex",
            "access": (
                "eyJhbGciOiJub25lIn0."
                "eyJleHAiOjEyMzQ1LCJodHRwczovL2FwaS5vcGVuYWkuY29tL2F1dGgiOnsiY2hhdGdwdF9hY2NvdW50X2lkIjoiYWNjdC0xMjMifX0."
                "sig"
            ),
            "accountId": "acct-123",
            "expires": 12345000,
        }

    def test_provision_openclaw_codex_oauth_requires_token(self, monkeypatch):
        monkeypatch.setattr(vmb, "_resolve_codex_oauth_profile_for_vm", lambda: {})

        class _Vm:
            pass

        with pytest.raises(RuntimeError, match="Codex OAuth profile is required"):
            vmb._provision_openclaw_codex_oauth(_Vm())

    def test_resolve_codex_oauth_profile_reads_paths_relative_to_dev_config(self, monkeypatch, tmp_path):
        home = tmp_path / "home"
        dev_cfg = home / "quaidcode" / "dev" / ".quaid-dev.local.json"
        token_file = home / "quaidcode" / "codex-oauth-sol.json"
        dev_cfg.parent.mkdir(parents=True)
        token_file.parent.mkdir(parents=True, exist_ok=True)
        dev_cfg.write_text(
            json.dumps(
                {
                    "auth": {
                        "codex": {
                            "solKeyPath": "../codex-oauth-sol.json",
                            "yuniKeyPath": "",
                        }
                    }
                }
            )
        )
        token_file.write_text(json.dumps({"access": "codex.jwt.token"}))
        monkeypatch.delenv("BENCHMARK_CODEX_API_KEY", raising=False)
        monkeypatch.setattr(vmb.Path, "home", lambda: home)

        profile = vmb._resolve_codex_oauth_profile_for_vm()
        assert profile["access"] == "codex.jwt.token"

    def test_resolve_codex_oauth_profile_prefers_local_codex_auth_json(self, monkeypatch, tmp_path):
        home = tmp_path / "home"
        codex_auth = home / ".codex" / "auth.json"
        codex_auth.parent.mkdir(parents=True, exist_ok=True)
        codex_auth.write_text(
            json.dumps(
                {
                    "tokens": {
                        "access_token": "fresh.codex.jwt",
                        "refresh_token": "refresh-token",
                        "account_id": "acct-xyz",
                    }
                }
            )
        )
        monkeypatch.delenv("BENCHMARK_CODEX_API_KEY", raising=False)
        monkeypatch.setattr(vmb.Path, "home", lambda: home)

        profile = vmb._resolve_codex_oauth_profile_for_vm()
        assert profile == {
            "type": "oauth",
            "provider": "openai-codex",
            "access": "fresh.codex.jwt",
            "refresh": "refresh-token",
            "accountId": "acct-xyz",
        }

    def test_resolve_codex_oauth_profile_prefers_token_path_and_preserves_refresh(
        self, monkeypatch, tmp_path
    ):
        token_file = tmp_path / "codex-oauth-yuni.json"
        token_file.write_text(
            json.dumps(
                {
                    "access": "fresh.yuni.jwt",
                    "refresh": "refresh-yuni-token",
                    "accountId": "acct-yuni",
                    "expires": 1776419275000,
                }
            )
        )
        monkeypatch.setenv("BENCHMARK_CODEX_TOKEN_PATH", str(token_file))
        monkeypatch.setenv("BENCHMARK_CODEX_API_KEY", "stale-access-only-token")

        profile = vmb._resolve_codex_oauth_profile_for_vm()
        assert profile == {
            "access": "fresh.yuni.jwt",
            "refresh": "refresh-yuni-token",
            "accountId": "acct-yuni",
            "expires": 1776419275000,
            "type": "oauth",
            "provider": "openai-codex",
        }


class TestTartVmSsh:
    def test_ssh_uses_password_only_auth_flags(self, monkeypatch):
        captured = {}

        def _fake_run(args, **kwargs):
            captured["args"] = args

            class _Result:
                returncode = 0
                stdout = ""
                stderr = ""

            return _Result()

        monkeypatch.setattr(vmb.subprocess, "run", _fake_run)
        vm = vmb.TartVM(ip="192.168.64.6", user="admin", password="admin")
        vm.ssh("echo ok")
        assert "-o" in captured["args"]
        joined = " ".join(captured["args"])
        assert "PreferredAuthentications=password" in joined
        assert "PubkeyAuthentication=no" in joined
        assert "IdentitiesOnly=yes" in joined

    def test_ssh_retries_transient_255_transport_errors(self, monkeypatch):
        calls = {"count": 0}

        def _fake_run(args, **kwargs):
            calls["count"] += 1

            class _Result:
                def __init__(self, returncode, stderr="", stdout=""):
                    self.returncode = returncode
                    self.stderr = stderr
                    self.stdout = stdout

            if calls["count"] == 1:
                return _Result(255, "Permission denied (publickey,password,keyboard-interactive).")
            return _Result(0, "", "ok")

        monkeypatch.setattr(vmb.subprocess, "run", _fake_run)
        monkeypatch.setattr(vmb.time, "sleep", lambda _seconds: None)
        vm = vmb.TartVM(ip="192.168.64.6", user="admin", password="admin")
        result = vm.ssh("echo ok")
        assert calls["count"] == 2
        assert result.returncode == 0
        assert result.stdout == "ok"

    def test_ssh_refreshes_tart_ip_before_each_attempt(self, monkeypatch):
        calls = []
        refreshed_ips = iter(["192.168.64.107", "192.168.64.108"])

        def _fake_run(args, **_kwargs):
            calls.append(args)

            class _Result:
                def __init__(self, returncode, stderr="", stdout=""):
                    self.returncode = returncode
                    self.stderr = stderr
                    self.stdout = stdout

            if len(calls) == 1:
                return _Result(255, "Connection timed out")
            return _Result(0, "", "ok")

        monkeypatch.setattr(vmb.subprocess, "run", _fake_run)
        monkeypatch.setattr(vmb.time, "sleep", lambda _seconds: None)
        vm = vmb.TartVM(ip="192.168.64.6", user="admin", password="admin")
        monkeypatch.setattr(
            vm,
            "_refresh_ip_from_tart",
            lambda: setattr(vm, "ip", next(refreshed_ips)),
        )

        result = vm.ssh("echo ok")

        assert result.returncode == 0
        assert "admin@192.168.64.107" in calls[0]
        assert "admin@192.168.64.108" in calls[1]

    def test_ssh_retries_timeout_when_tart_ip_changes(self, monkeypatch):
        calls = []
        refreshed_ips = iter(["192.168.64.107", "192.168.64.108", "192.168.64.108"])

        def _fake_run(args, **kwargs):
            calls.append(args)
            if len(calls) == 1:
                raise vmb.subprocess.TimeoutExpired(args, kwargs.get("timeout"))

            class _Result:
                def __init__(self, returncode, stderr="", stdout=""):
                    self.returncode = returncode
                    self.stderr = stderr
                    self.stdout = stdout

            return _Result(0, "", "ok")

        monkeypatch.setattr(vmb.subprocess, "run", _fake_run)
        monkeypatch.setattr(vmb.time, "sleep", lambda _seconds: None)
        vm = vmb.TartVM(ip="192.168.64.6", user="admin", password="admin")
        monkeypatch.setattr(
            vm,
            "_refresh_ip_from_tart",
            lambda: setattr(vm, "ip", next(refreshed_ips)),
        )

        result = vm.ssh("echo ok")

        assert result.returncode == 0
        assert len(calls) == 2
        assert "admin@192.168.64.107" in calls[0]
        assert "admin@192.168.64.108" in calls[1]

    def test_tart_cmd_uses_remote_tart_host_when_configured(self, monkeypatch):
        captured = {}

        def _fake_run(args, **kwargs):
            captured["args"] = args

            class _Result:
                returncode = 0
                stdout = "snapshot"
                stderr = ""

            return _Result()

        monkeypatch.setattr(vmb.subprocess, "run", _fake_run)
        vm = vmb.TartVM(ip="192.168.64.6", user="admin", password="admin", tart_host="alfie.local")
        vm._tart_cmd("help", timeout=10)
        assert captured["args"][:4] == ["ssh", "-o", "BatchMode=yes", "-o"]
        assert "alfie.local" in captured["args"]
        assert captured["args"][-1] == "tart help"

    def test_ssh_routes_guest_connection_through_tart_host(self, monkeypatch):
        captured = {}

        def _fake_run(args, **kwargs):
            captured["args"] = args

            class _Result:
                returncode = 0
                stdout = "ok"
                stderr = ""

            return _Result()

        monkeypatch.setattr(vmb.subprocess, "run", _fake_run)
        vm = vmb.TartVM(ip="192.168.64.3", user="admin", password="admin", tart_host="alfie.local")
        vm.ssh("echo ok", raw=True)
        assert captured["args"][0] == "ssh"
        assert "alfie.local" in captured["args"]
        assert "sshpass -p admin ssh" in captured["args"][-1]
        assert "admin@192.168.64.3" in captured["args"][-1]

    def test_scp_to_routes_through_tart_host(self, monkeypatch, tmp_path):
        calls = []

        def _fake_run(args, **kwargs):
            calls.append(args)

            class _Result:
                returncode = 0
                stdout = ""
                stderr = ""

            return _Result()

        monkeypatch.setattr(vmb.subprocess, "run", _fake_run)
        local = tmp_path / "x.txt"
        local.write_text("x")
        vm = vmb.TartVM(ip="192.168.64.3", user="admin", password="admin", tart_host="alfie.local")
        vm.scp_to(str(local), "~/x.txt")
        assert calls[0][0] == "ssh"
        assert calls[1][0] == "scp"
        assert calls[2][0] == "ssh"
        assert "/tmp/vm-benchmark-upload" in calls[2][-1]
        assert "sshpass -p admin ssh" in calls[2][-1]
        assert "admin@192.168.64.3" in calls[2][-1]
        assert "cat > ~/x.txt" in calls[2][-1]

    def test_ssh_retries_permission_denied_even_without_255(self, monkeypatch):
        calls = {"count": 0}

        def _fake_run(args, **kwargs):
            calls["count"] += 1

            class _Result:
                def __init__(self, returncode, stderr="", stdout=""):
                    self.returncode = returncode
                    self.stderr = stderr
                    self.stdout = stdout

            if calls["count"] == 1:
                return _Result(5, "Permission denied (publickey,password,keyboard-interactive).")
            return _Result(0, "", "ok")

        monkeypatch.setattr(vmb.subprocess, "run", _fake_run)
        monkeypatch.setattr(vmb.time, "sleep", lambda _seconds: None)
        vm = vmb.TartVM(ip="192.168.64.6", user="admin", password="admin")
        result = vm.ssh("echo ok")
        assert calls["count"] == 2
        assert result.returncode == 0
        assert result.stdout == "ok"

    def test_wait_ready_uses_longer_probe_timeout_when_routed_via_tart_host(self, monkeypatch):
        captured = {}

        def _fake_ssh(_cmd, timeout=0, raw=False, **_kwargs):
            captured["timeout"] = timeout
            captured["raw"] = raw

            class _Result:
                returncode = 0
                stdout = "ready\n"
                stderr = ""

            return _Result()

        vm = vmb.TartVM(ip="192.168.64.3", user="admin", password="admin", tart_host="alfie.local")
        monkeypatch.setattr(vm, "ssh", _fake_ssh)
        assert vm.wait_ready(timeout=5) is True
        assert captured["timeout"] == 15
        assert captured["raw"] is True

    def test_restore_refuses_single_running_local_vm_without_override(self, monkeypatch):
        calls = []

        def _fake_tart_cmd(*parts, **_kwargs):
            calls.append(parts)

            class _Result:
                def __init__(self, returncode=0, stdout="", stderr=""):
                    self.returncode = returncode
                    self.stdout = stdout
                    self.stderr = stderr

            if parts == ("list",):
                return _Result(stdout=(
                    "Source Name State\n"
                    "local quaid-livetest-run running\n"
                ))
            if parts == ("ip", "quaid-livetest-run"):
                return _Result(stdout="192.168.64.107\n")
            if parts == ("help",):
                return _Result(stdout="")
            return _Result()

        monkeypatch.setattr(vmb.TartVM, "_tart_cmd", lambda self, *parts, **kwargs: _fake_tart_cmd(*parts, **kwargs))
        monkeypatch.setattr(vmb.TartVM, "is_ready", lambda self: True)

        vm = vmb.TartVM(ip="192.168.64.3", vm_name="test-openclaw")
        with pytest.raises(RuntimeError, match="Refusing to reuse running VM"):
            vm.restore("clean-openclaw")

        assert vm.vm_name == "test-openclaw"

    def test_restore_falls_back_to_single_running_local_vm_with_override(self, monkeypatch):
        def _fake_tart_cmd(*parts, **_kwargs):
            class _Result:
                def __init__(self, returncode=0, stdout="", stderr=""):
                    self.returncode = returncode
                    self.stdout = stdout
                    self.stderr = stderr

            if parts == ("list",):
                return _Result(stdout=(
                    "Source Name State\n"
                    "local quaid-livetest-run running\n"
                ))
            if parts == ("ip", "quaid-livetest-run"):
                return _Result(stdout="192.168.64.107\n")
            if parts == ("help",):
                return _Result(stdout="")
            return _Result()

        monkeypatch.setenv("VM_BENCHMARK_ALLOW_RUNNING_VM_FALLBACK", "1")
        monkeypatch.setattr(vmb.TartVM, "_tart_cmd", lambda self, *parts, **kwargs: _fake_tart_cmd(*parts, **kwargs))
        monkeypatch.setattr(vmb.TartVM, "is_ready", lambda self: True)

        vm = vmb.TartVM(ip="192.168.64.3", vm_name="test-openclaw")
        vm.restore("clean-openclaw")

        assert vm.vm_name == "quaid-livetest-run"
        assert vm.ip == "192.168.64.107"

    def test_wait_ready_refreshes_ip_from_tart(self, monkeypatch):
        state = {"ssh_calls": 0}

        def _fake_tart_cmd(self, *parts, **_kwargs):
            class _Result:
                def __init__(self, returncode=0, stdout="", stderr=""):
                    self.returncode = returncode
                    self.stdout = stdout
                    self.stderr = stderr

            if parts == ("list",):
                return _Result(stdout="Source Name State\nlocal quaid-livetest-run running\n")
            if parts == ("ip", "quaid-livetest-run"):
                return _Result(stdout="192.168.64.107\n")
            return _Result()

        def _fake_ssh(_cmd, timeout=0, raw=False, **_kwargs):
            state["ssh_calls"] += 1

            class _Result:
                returncode = 0
                stdout = "ready\n"
                stderr = ""

            return _Result()

        vm = vmb.TartVM(ip="192.168.64.3", vm_name="test-openclaw")
        monkeypatch.setenv("VM_BENCHMARK_ALLOW_RUNNING_VM_FALLBACK", "1")
        monkeypatch.setattr(vmb.TartVM, "_tart_cmd", _fake_tart_cmd)
        monkeypatch.setattr(vm, "ssh", _fake_ssh)

        assert vm.wait_ready(timeout=5) is True
        assert vm.vm_name == "quaid-livetest-run"
        assert vm.ip == "192.168.64.107"
        assert state["ssh_calls"] >= 1


class TestOpenClawNativeReindex:
    def test_sync_openclaw_native_memory_polls_until_source_is_indexed(self, monkeypatch):
        calls = []
        now = [1000.0]
        monkeypatch.setattr(vmb, "_patch_openclaw_native_memory", lambda *args, **kwargs: None)

        class _Vm:
            def __init__(self):
                self.status_calls = 0

            def ssh(self, command, **_kwargs):
                calls.append(command)

                class _Result:
                    def __init__(self, stdout, returncode=0, stderr=""):
                        self.returncode = returncode
                        self.stderr = stderr
                        self.stdout = stdout

                if "openclaw memory index --agent main" in command:
                    return _Result("")
                if command == "openclaw memory status --agent main --json":
                    self.status_calls += 1
                    if self.status_calls == 1:
                        return _Result(
                            '[{"status":{"dirty":false,"sourceCounts":['
                            '{"source":"sessions","files":1,"chunks":14}'
                            ']}}]'
                        )
                    return _Result(
                        '[{"status":{"dirty":false,"sourceCounts":['
                        '{"source":"sessions","files":2,"chunks":28}'
                        ']}}]'
                    )
                return _Result("")

        sleeps = []
        monkeypatch.setattr(vmb.time, "time", lambda: now[0])
        monkeypatch.setattr(
            vmb.time,
            "sleep",
            lambda seconds: (sleeps.append(seconds), now.__setitem__(0, now[0] + seconds)),
        )
        status = vmb._sync_openclaw_native_memory(_Vm(), source_name="sessions", min_indexed_files=2)
        assert status["dirty"] is False
        assert calls.count("openclaw memory status --agent main --json") == 2
        assert sleeps == [vmb.OC_NATIVE_REINDEX_STATUS_POLL_S]

    def test_force_reindex_requires_selected_source_chunks(self):
        calls = []
        now = [1000.0]

        class _Vm:
            def ssh(self, command, **_kwargs):
                calls.append(command)

                class _Result:
                    def __init__(self, stdout, returncode=0, stderr=""):
                        self.returncode = 0
                        self.stderr = stderr
                        self.stdout = stdout

                if "openclaw memory index --agent main --force" in command:
                    return _Result("")
                return _Result(
                    '[{"status":{"dirty":false,"sourceCounts":['
                    '{"source":"memory","files":1,"chunks":1},'
                    '{"source":"sessions","files":0,"chunks":0}'
                    ']}}]'
                )

        sleeps = []
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(vmb, "_patch_openclaw_native_memory", lambda *args, **kwargs: None)
        monkeypatch.setattr(vmb.time, "time", lambda: now[0])
        monkeypatch.setattr(
            vmb.time,
            "sleep",
            lambda seconds: (sleeps.append(seconds), now.__setitem__(0, now[0] + seconds)),
        )
        with pytest.raises(RuntimeError, match="oc-native sessions did not finish indexing"):
            vmb._force_openclaw_native_reindex(_Vm(), source_name="sessions", min_indexed_files=1)
        monkeypatch.undo()
        assert calls[0].startswith(
            "sh -lc 'export PATH=/opt/homebrew/bin:$PATH; "
            "openclaw memory index --agent main --force"
        )
        assert calls[1] == "tail -80 /tmp/oc-native-reindex.log 2>/dev/null"
        assert calls[2] == "openclaw memory status --agent main --json"
        assert sleeps

    def test_sync_openclaw_native_memory_uses_non_force_index(self, monkeypatch):
        calls = []
        monkeypatch.setattr(vmb, "_patch_openclaw_native_memory", lambda *args, **kwargs: None)

        class _Vm:
            def ssh(self, command, **_kwargs):
                calls.append(command)

                class _Result:
                    def __init__(self, stdout, returncode=0, stderr=""):
                        self.returncode = returncode
                        self.stderr = stderr
                        self.stdout = stdout

                if "openclaw memory index --agent main" in command:
                    assert "--force" not in command
                    return _Result("")
                return _Result(
                    '[{"status":{"dirty":false,"sourceCounts":['
                    '{"source":"memory","files":3,"chunks":12}'
                    ']}}]'
                )

        status = vmb._sync_openclaw_native_memory(_Vm(), source_name="memory", min_indexed_files=3)
        assert status["dirty"] is False
        assert calls[0].startswith(
            "sh -lc 'export PATH=/opt/homebrew/bin:$PATH; "
            "openclaw memory index --agent main > /tmp/oc-native-reindex.log 2>&1'"
        )
        assert calls[1] == "tail -80 /tmp/oc-native-reindex.log 2>/dev/null"
        assert calls[2] == "openclaw memory status --agent main --json"

    def test_sync_openclaw_native_memory_falls_back_when_status_command_is_unavailable(self, monkeypatch):
        calls = []
        monkeypatch.setattr(vmb, "_patch_openclaw_native_memory", lambda *args, **kwargs: None)

        class _Vm:
            def ssh(self, command, **_kwargs):
                calls.append(command)

                class _Result:
                    def __init__(self, stdout, returncode=0, stderr=""):
                        self.returncode = returncode
                        self.stderr = stderr
                        self.stdout = stdout

                if "openclaw memory index --agent main" in command:
                    return _Result("")
                if command == "tail -80 /tmp/oc-native-reindex.log 2>/dev/null":
                    return _Result("Memory index updated (main).")
                if command == "openclaw memory status --agent main --json":
                    return _Result("", returncode=1, stderr="error: unknown command 'memory'")
                return _Result("")

        status = vmb._sync_openclaw_native_memory(_Vm(), source_name="sessions", min_indexed_files=2)
        assert status["dirty"] is False
        assert status["sourceCounts"] == [{"source": "sessions", "files": 2, "chunks": 1}]
        assert calls[1] == "tail -80 /tmp/oc-native-reindex.log 2>/dev/null"
        assert calls[2] == "openclaw memory status --agent main --json"

    def test_sync_openclaw_native_memory_falls_back_when_status_underreports_session_files(self, monkeypatch):
        calls = []
        monkeypatch.setattr(vmb, "_patch_openclaw_native_memory", lambda *args, **kwargs: None)

        class _Vm:
            def ssh(self, command, **_kwargs):
                calls.append(command)

                class _Result:
                    def __init__(self, stdout, returncode=0, stderr=""):
                        self.returncode = returncode
                        self.stderr = stderr
                        self.stdout = stdout

                if "openclaw memory index --agent main" in command:
                    return _Result("")
                if command == "tail -80 /tmp/oc-native-reindex.log 2>/dev/null":
                    return _Result("Memory index updated (main).")
                if command == "openclaw memory status --agent main --json":
                    return _Result(
                        '[{"status":{"dirty":false,"sourceCounts":['
                        '{"source":"sessions","files":1,"chunks":14}'
                        ']}}]'
                    )
                return _Result("")

        monkeypatch.setattr(vmb, "_count_vm_session_jsonl_files", lambda _vm: 2)
        status = vmb._sync_openclaw_native_memory(_Vm(), source_name="sessions", min_indexed_files=2)
        assert status["dirty"] is False
        assert status["sourceCounts"] == [{"source": "sessions", "files": 2, "chunks": 14}]
        assert calls[1] == "tail -80 /tmp/oc-native-reindex.log 2>/dev/null"
        assert calls[2] == "openclaw memory status --agent main --json"

    def test_force_reindex_uses_extended_timeout_budget(self, monkeypatch):
        calls = []
        monkeypatch.setattr(vmb, "_patch_openclaw_native_memory", lambda *args, **kwargs: None)

        class _Vm:
            def ssh(self, command, **kwargs):
                calls.append((command, kwargs))

                class _Result:
                    def __init__(self, stdout, returncode=0, stderr=""):
                        self.returncode = returncode
                        self.stderr = stderr
                        self.stdout = stdout

                if "openclaw memory index --agent main --force" in command:
                    return _Result("")
                return _Result(
                    '[{"status":{"dirty":false,"sourceCounts":['
                    '{"source":"sessions","files":277,"chunks":1200}'
                    ']}}]'
                )

        status = vmb._force_openclaw_native_reindex(
            _Vm(), source_name="sessions", min_indexed_files=277
        )
        assert status["dirty"] is False
        assert calls[0][1]["timeout"] == vmb.OC_NATIVE_REINDEX_TIMEOUT_S + 120
        assert calls[1][1]["timeout"] == 10
        assert calls[2][1]["timeout"] == 90

    def test_inject_sessions_syncs_oc_native_at_day_boundaries_and_finish(self, monkeypatch, tmp_path):
        dataset_mod = sys.modules["dataset"]
        dataset_mod.SESSION_DATES = {
            1: "2026-03-01",
            2: "2026-03-01",
            3: "2026-03-02",
        }
        monkeypatch.setattr(
            vmb,
            "transcript_to_messages",
            lambda review: [{"role": "user", "content": f"session-{review.session_num}"}],
        )
        class _Tracker:
            def add_message(self, *_args, **_kwargs):
                return None

            def add_compaction(self, *_args, **_kwargs):
                return None

            def summary(self):
                return {}

        monkeypatch.setattr(vmb, "CostTracker", _Tracker)
        monkeypatch.setattr(vmb, "count_tokens", lambda _text: 1)
        writes = []
        monkeypatch.setattr(
            vmb,
            "_write_vm_session_jsonl",
            lambda vm, session_id, jsonl, append=True, sessions_dir=vmb.VM_AGENT_SESSIONS_DIR: writes.append(
                {
                    "session_id": session_id,
                    "jsonl": jsonl,
                    "append": append,
                    "sessions_dir": sessions_dir,
                }
            )
            or type("_Result", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
        )
        monkeypatch.setattr(vmb, "_run_oc_native_session_hook", lambda vm, session_id: None)
        sync_calls = []
        monkeypatch.setattr(
            vmb,
            "_sync_openclaw_native_memory",
            lambda vm, source_name="sessions", min_indexed_files=1, force=False: sync_calls.append(
                {
                    "source_name": source_name,
                    "min_indexed_files": min_indexed_files,
                    "force": force,
                }
            )
            or {"dirty": False},
        )
        monkeypatch.setattr(vmb, "_sync_openclaw_native_wiki", lambda vm: None)

        class _Review:
            def __init__(self, session_num):
                self.session_num = session_num

        class _Vm:
            pass

        stats = vmb.inject_sessions(
            _Vm(),
            [_Review(1), _Review(2), _Review(3)],
            "bench",
            results_dir=tmp_path,
            system="oc-native",
        )

        assert stats["total_sessions"] == 3
        assert [entry["session_id"] for entry in writes] == [
            "benchmark-oc-native-s01",
            "benchmark-oc-native-s02",
            "benchmark-oc-native-s03",
        ]
        assert all(entry["append"] is False for entry in writes)
        seeded = [json.loads(line) for line in writes[0]["jsonl"].splitlines()]
        assert seeded[0]["type"] == "session"
        assert seeded[0]["id"] == "benchmark-oc-native-s01"
        assert seeded[1]["type"] == "model_change"
        assert seeded[2]["message"]["content"] == [{"type": "text", "text": "session-1"}]
        assert sync_calls == [
            {"source_name": "sessions", "min_indexed_files": 2, "force": False},
            {"source_name": "sessions", "min_indexed_files": 3, "force": False},
        ]


class TestVmEvalIsolation:
    def test_oc_native_gateway_call_repairs_pairing_and_retries(self):
        calls = []

        class _Vm:
            def __init__(self):
                self.gateway_calls = 0

            def ssh(self, command, **kwargs):
                calls.append((command, kwargs))

                class _Result:
                    def __init__(self, returncode=0, stdout="", stderr=""):
                        self.returncode = returncode
                        self.stdout = stdout
                        self.stderr = stderr

                if "openclaw gateway call agent --json" in command:
                    self.gateway_calls += 1
                    if self.gateway_calls == 1:
                        return _Result(
                            1,
                            "",
                            "gateway connect failed: GatewayClientRequestError: pairing required",
                        )
                    return _Result(0, '{"runId":"run-123","status":"accepted"}', "")
                if "OC_NATIVE_PAIR_REPAIR" in command:
                    return _Result(0, '{"approved": true, "requestId":"req-123"}', "")
                return _Result()

        payload = vmb._oc_native_gateway_call(
            _Vm(),
            "agent",
            {"message": "hello"},
            timeout_s=30,
        )
        assert payload["runId"] == "run-123"
        repair_calls = [kwargs for command, kwargs in calls if "OC_NATIVE_PAIR_REPAIR" in command]
        assert repair_calls
        assert all(not kwargs.get("raw", False) for kwargs in repair_calls)

    def test_oc_native_gateway_call_restarts_gateway_on_normal_closure(self, monkeypatch):
        calls = []
        restarts = []

        class _Vm:
            def __init__(self):
                self.gateway_calls = 0

            def ssh(self, command, **kwargs):
                calls.append((command, kwargs))

                class _Result:
                    def __init__(self, returncode=0, stdout="", stderr=""):
                        self.returncode = returncode
                        self.stdout = stdout
                        self.stderr = stderr

                if "openclaw gateway call agent --json" in command:
                    self.gateway_calls += 1
                    if self.gateway_calls == 1:
                        return _Result(
                            1,
                            "",
                            "gateway connect failed: Error: gateway closed (1000 normal closure): no close reason",
                        )
                    return _Result(0, '{"runId":"run-456","status":"accepted"}', "")
                return _Result()

        monkeypatch.setattr(vmb, "_restart_oc_native_gateway", lambda vm, port=18789: restarts.append(port))

        payload = vmb._oc_native_gateway_call(
            _Vm(),
            "agent",
            {"message": "hello"},
            timeout_s=30,
        )

        assert payload["runId"] == "run-456"
        assert restarts == [18789]
        gateway_calls = [command for command, _kwargs in calls if "openclaw gateway call agent --json" in command]
        assert len(gateway_calls) == 2

    def test_oc_native_gateway_call_allows_multiple_gateway_restarts(self, monkeypatch):
        calls = []
        restarts = []

        class _Vm:
            def __init__(self):
                self.gateway_calls = 0

            def ssh(self, command, **kwargs):
                calls.append((command, kwargs))

                class _Result:
                    def __init__(self, returncode=0, stdout="", stderr=""):
                        self.returncode = returncode
                        self.stdout = stdout
                        self.stderr = stderr

                if "openclaw gateway call agent --json" in command:
                    self.gateway_calls += 1
                    if self.gateway_calls in {1, 2}:
                        return _Result(
                            1,
                            "",
                            "Gateway call failed: Error: gateway closed (1006 abnormal closure (no close frame)): no close reason",
                        )
                    return _Result(0, '{"runId":"run-456b","status":"accepted"}', "")
                return _Result()

        monkeypatch.setattr(vmb, "_restart_oc_native_gateway", lambda vm, port=18789: restarts.append(port))

        payload = vmb._oc_native_gateway_call(
            _Vm(),
            "agent",
            {"message": "hello"},
            timeout_s=30,
        )

        assert payload["runId"] == "run-456b"
        assert restarts == [18789, 18789]
        gateway_calls = [command for command, _kwargs in calls if "openclaw gateway call agent --json" in command]
        assert len(gateway_calls) == 3

    def test_oc_native_gateway_call_restarts_gateway_on_invalid_config(self, monkeypatch):
        calls = []
        restarts = []

        class _Vm:
            def __init__(self):
                self.gateway_calls = 0

            def ssh(self, command, **kwargs):
                calls.append((command, kwargs))

                class _Result:
                    def __init__(self, returncode=0, stdout="", stderr=""):
                        self.returncode = returncode
                        self.stdout = stdout
                        self.stderr = stderr

                if "openclaw gateway call agent --json" in command:
                    self.gateway_calls += 1
                    if self.gateway_calls == 1:
                        return _Result(
                            1,
                            "",
                            "Invalid config at /Users/admin/.openclaw/openclaw.json:\n"
                            "- plugins.entries.memory-core: Unrecognized key: \"disabled\"",
                        )
                    return _Result(0, '{"runId":"run-789","status":"accepted"}', "")
                return _Result()

        monkeypatch.setattr(vmb, "_restart_oc_native_gateway", lambda vm, port=18789: restarts.append(port))

        payload = vmb._oc_native_gateway_call(
            _Vm(),
            "agent",
            {"message": "hello"},
            timeout_s=30,
        )

        assert payload["runId"] == "run-789"
        assert restarts == [18789]
        gateway_calls = [command for command, _kwargs in calls if "openclaw gateway call agent --json" in command]
        assert len(gateway_calls) == 2

    def test_oc_native_gateway_call_restarts_gateway_on_config_invalid_variant(self, monkeypatch):
        calls = []
        restarts = []

        class _Vm:
            def __init__(self):
                self.gateway_calls = 0

            def ssh(self, command, **kwargs):
                calls.append((command, kwargs))

                class _Result:
                    def __init__(self, returncode=0, stdout="", stderr=""):
                        self.returncode = returncode
                        self.stdout = stdout
                        self.stderr = stderr

                if "openclaw gateway call agent --json" in command:
                    self.gateway_calls += 1
                    if self.gateway_calls == 1:
                        return _Result(
                            1,
                            "",
                            "Config invalid\n"
                            "File: ~/.openclaw/openclaw.json\n"
                            "Problem:\n"
                            '  - plugins.entries.memory-core: Unrecognized key: "disabled"',
                        )
                    return _Result(0, '{"runId":"run-790","status":"accepted"}', "")
                return _Result()

        monkeypatch.setattr(vmb, "_restart_oc_native_gateway", lambda vm, port=18789: restarts.append(port))

        payload = vmb._oc_native_gateway_call(
            _Vm(),
            "agent",
            {"message": "hello"},
            timeout_s=30,
        )

        assert payload["runId"] == "run-790"
        assert restarts == [18789]
        gateway_calls = [command for command, _kwargs in calls if "openclaw gateway call agent --json" in command]
        assert len(gateway_calls) == 2

    def test_run_oc_native_gateway_turn_uses_gateway_rpc_and_reads_last_assistant_text(self, monkeypatch):
        gateway_calls = []

        def _fake_gateway_call(vm, method, params, *, timeout_s, ssh_timeout_s=None):
            gateway_calls.append(
                {
                    "method": method,
                    "params": params,
                    "timeout_s": timeout_s,
                    "ssh_timeout_s": ssh_timeout_s,
                }
            )
            if method == "agent":
                return {"runId": "run-123", "status": "accepted"}
            if method == "agent.wait":
                return {"runId": "run-123", "status": "ok"}
            raise AssertionError(method)

        monkeypatch.setattr(vmb, "_oc_native_gateway_call", _fake_gateway_call)
        monkeypatch.setattr(
            vmb,
            "_read_oc_native_session_tail_state",
            lambda _vm, _sid, **_kwargs: {
                "line_count": 1,
                "assistant_text": "assistant-answer",
                "assistant_event_id": "evt-123",
                "assistant_timestamp": "ts-123",
            },
        )

        answer = vmb._run_oc_native_gateway_turn(object(), "eval-q007", "Who is Maya?", timeout_s=30)
        assert answer == "assistant-answer"
        assert gateway_calls[0]["method"] == "agent"
        assert gateway_calls[0]["params"]["sessionKey"] == "agent:main:eval-q007"
        assert gateway_calls[0]["params"]["sessionId"] == "eval-q007"
        assert gateway_calls[0]["params"]["message"] == "Who is Maya?"
        assert gateway_calls[0]["ssh_timeout_s"] == 240
        assert gateway_calls[1]["method"] == "agent.wait"
        assert gateway_calls[1]["params"]["runId"] == "run-123"
        assert gateway_calls[1]["params"]["timeoutMs"] == 90000
        assert gateway_calls[1]["timeout_s"] == 120
        assert gateway_calls[1]["ssh_timeout_s"] == 300

    def test_read_oc_native_last_assistant_message_checks_session_store_first(self):
        calls = []

        class _Vm:
            def ssh(self, command, **_kwargs):
                calls.append(command)

                class _Result:
                    returncode = 0
                    stdout = json.dumps(
                        {
                            "path": "/Users/admin/.openclaw/agents/main/sessions/eval-q012.jsonl",
                            "line_count": 31,
                            "assistant_text": "assistant-answer",
                            "assistant_event_id": "evt-123",
                            "assistant_timestamp": "2026-04-25T09:22:39.776Z",
                        }
                    )
                    stderr = ""

                return _Result()

        answer = vmb._read_oc_native_last_assistant_message(
            _Vm(),
            "eval-q012",
            session_key="agent:main:eval-q012",
        )
        assert answer == "assistant-answer"
        command = calls[0]
        assert "sessions.json" in command
        assert "sessionFile" in command
        assert "agent:main:eval-q012" in command
        assert "session_id =" in command

    def test_read_oc_native_session_tail_state_returns_metadata(self):
        class _Vm:
            def ssh(self, command, **_kwargs):
                assert "assistant_event_id" in command
                assert "line_count" in command

                class _Result:
                    returncode = 0
                    stdout = json.dumps(
                        {
                            "path": "/Users/admin/.openclaw/agents/main/sessions/eval-q012.jsonl",
                            "line_count": 31,
                            "assistant_text": "assistant-answer",
                            "assistant_event_id": "evt-123",
                            "assistant_timestamp": "2026-04-25T09:22:39.776Z",
                        }
                    )
                    stderr = ""

                return _Result()

        state = vmb._read_oc_native_session_tail_state(
            _Vm(),
            "eval-q012",
            session_key="agent:main:eval-q012",
        )
        assert state["line_count"] == 31
        assert state["assistant_text"] == "assistant-answer"
        assert state["assistant_event_id"] == "evt-123"

    def test_run_oc_native_gateway_turn_accepts_transcript_confirmed_completion_after_wait_timeout(self, monkeypatch):
        gateway_calls = []

        def _fake_gateway_call(vm, method, params, *, timeout_s, ssh_timeout_s=None):
            gateway_calls.append(method)
            if method == "agent":
                return {"runId": "run-123", "status": "accepted"}
            if method == "agent.wait":
                return {"runId": "run-123", "status": "timeout"}
            raise AssertionError(method)

        states = iter(
            [
                {
                    "line_count": 20,
                    "assistant_text": "old-answer",
                    "assistant_event_id": "evt-old",
                    "assistant_timestamp": "t-old",
                },
                {
                    "line_count": 24,
                    "assistant_text": "new-answer",
                    "assistant_event_id": "evt-new",
                    "assistant_timestamp": "t-new",
                },
            ]
        )

        monkeypatch.setattr(vmb, "_oc_native_gateway_call", _fake_gateway_call)
        monkeypatch.setattr(vmb, "_read_oc_native_session_tail_state", lambda *_args, **_kwargs: next(states))
        monkeypatch.setattr(vmb.time, "sleep", lambda _seconds: None)

        answer = vmb._run_oc_native_gateway_turn(
            object(),
            "benchmark-oc-native-s03",
            "hello",
            timeout_s=45,
            session_key="agent:main:benchmark-oc-native-s03",
        )
        assert answer == "new-answer"
        assert gateway_calls == ["agent", "agent.wait"]

    def test_run_oc_native_gateway_turn_accepts_transcript_confirmed_completion_after_wait_transport_failure(self, monkeypatch):
        gateway_calls = []

        def _fake_gateway_call(vm, method, params, *, timeout_s, ssh_timeout_s=None):
            gateway_calls.append(method)
            if method == "agent":
                return {"runId": "run-123", "status": "accepted"}
            if method == "agent.wait":
                raise RuntimeError("oc-native gateway call failed method=agent.wait: Permission denied")
            raise AssertionError(method)

        states = iter(
            [
                {
                    "line_count": 20,
                    "assistant_text": "old-answer",
                    "assistant_event_id": "evt-old",
                    "assistant_timestamp": "t-old",
                },
                {
                    "line_count": 24,
                    "assistant_text": "new-answer",
                    "assistant_event_id": "evt-new",
                    "assistant_timestamp": "t-new",
                },
            ]
        )

        monkeypatch.setattr(vmb, "_oc_native_gateway_call", _fake_gateway_call)
        monkeypatch.setattr(vmb, "_read_oc_native_session_tail_state", lambda *_args, **_kwargs: next(states))
        monkeypatch.setattr(vmb.time, "sleep", lambda _seconds: None)

        answer = vmb._run_oc_native_gateway_turn(
            object(),
            "benchmark-oc-native-s18",
            "hello",
            timeout_s=45,
            session_key="agent:main:benchmark-oc-native-s18",
        )
        assert answer == "new-answer"
        assert gateway_calls == ["agent", "agent.wait"]

    def test_register_session_persists_session_file_pointer(self, monkeypatch):
        calls = []

        class _Vm:
            def ssh(self, command, **_kwargs):
                calls.append(command)

                class _Result:
                    returncode = 0
                    stdout = "Registered\n"
                    stderr = ""

                return _Result()

        vmb._register_session(
            _Vm(),
            "benchmark-oc-native-s01",
            session_key="agent:main:benchmark-oc-native-s01",
        )
        assert "sessionFile" in calls[0]
        assert "agent:main:benchmark-oc-native-s01" in calls[0]
        assert ".openclaw/agents/main/sessions/" in calls[0]

    def test_evaluate_vm_agent_registers_fresh_session(self, monkeypatch):
        for system in ("oc-native", "quaid"):
            calls = []

            monkeypatch.setattr(
                vmb,
                "_register_session",
                lambda vm, session_id, *, session_key="agent:main:main", session_file=None: calls.append(
                    f"register:{session_id}:{session_key}:{session_file}"
                ),
            )
            monkeypatch.setattr(
                vmb,
                "_run_oc_native_gateway_turn",
                lambda vm, session_id, question, timeout_s, session_key=None: calls.append(
                    f"gateway:{session_id}:{session_key}:{question}:{timeout_s}"
                ) or "ok",
            )
            monkeypatch.setattr(
                vmb,
                "_cleanup_oc_native_eval_session",
                lambda vm, session_id, session_key=None: calls.append(
                    f"cleanup:{session_id}:{session_key}"
                ),
            )
            monkeypatch.setattr(vmb, "_extract_agent_answer", lambda raw: raw)
            answer = vmb._evaluate_vm_agent(object(), "Who is Maya?", 7, system)
            assert answer == "ok"
            assert calls[0] == (
                "register:eval-q007:agent:main:hook:eval-q007:"
                "~/.openclaw/agents/benchmark-eval/sessions/eval-q007.jsonl"
            )
            assert calls[1] == (
                f"gateway:eval-q007:agent:main:hook:eval-q007:"
                f"Who is Maya?:{vmb.VM_AGENT_EVAL_TIMEOUT_S}"
            )
            assert calls[2] == "cleanup:eval-q007:agent:main:hook:eval-q007"

    def test_evaluate_vm_agent_retries_timeout_then_succeeds(self, monkeypatch):
        for system in ("oc-native", "quaid"):
            calls = []
            attempts = {"n": 0}

            monkeypatch.setattr(
                vmb,
                "_register_session",
                lambda vm, session_id, *, session_key="agent:main:main", session_file=None: calls.append(
                    f"register:{session_id}:{session_key}:{session_file}"
                ),
            )

            def _turn(_vm, session_id, question, timeout_s, session_key=None):
                calls.append(f"gateway:{session_id}:{session_key}:{question}:{timeout_s}")
                attempts["n"] += 1
                if attempts["n"] == 1:
                    raise subprocess.TimeoutExpired("gateway", timeout=1)
                return "ok-after-retry"

            monkeypatch.setattr(vmb, "_run_oc_native_gateway_turn", _turn)
            monkeypatch.setattr(
                vmb,
                "_cleanup_oc_native_eval_session",
                lambda vm, session_id, session_key=None: calls.append(
                    f"cleanup:{session_id}:{session_key}"
                ),
            )
            monkeypatch.setattr(vmb, "_extract_agent_answer", lambda raw: raw)
            answer = vmb._evaluate_vm_agent(object(), "Who is Maya?", 8, system)
            assert answer == "ok-after-retry"
            assert attempts["n"] == 2
            assert calls[0] == (
                "register:eval-q008:agent:main:hook:eval-q008:"
                "~/.openclaw/agents/benchmark-eval/sessions/eval-q008.jsonl"
            )
            assert calls[1] == (
                f"gateway:eval-q008:agent:main:hook:eval-q008:"
                f"Who is Maya?:{vmb.VM_AGENT_EVAL_TIMEOUT_S}"
            )
            assert calls[2] == (
                f"gateway:eval-q008:agent:main:hook:eval-q008:"
                f"Who is Maya?:{vmb.VM_AGENT_EVAL_TIMEOUT_S}"
            )
            assert calls[3] == "cleanup:eval-q008:agent:main:hook:eval-q008"

    def test_run_oc_native_session_hook_uses_gateway_and_restores_transcript(self, monkeypatch):
        calls = []

        monkeypatch.setattr(vmb, "_register_session", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(vmb, "_read_vm_session_jsonl", lambda *_args, **_kwargs: "original-jsonl\n")
        monkeypatch.setattr(vmb, "_wait_for_vm_session_jsonl_quiet", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(
            vmb,
            "_run_oc_native_gateway_turn",
            lambda _vm, session_id, message, timeout_s, session_key=None: calls.append(
                f"gateway:{session_id}:{session_key}:{message}:{timeout_s}"
            ) or "hook-ok",
        )

        class _Result:
            returncode = 0
            stdout = ""
            stderr = ""

        monkeypatch.setattr(
            vmb,
            "_write_vm_session_jsonl",
            lambda _vm, session_id, jsonl, append=False: calls.append(
                f"restore:{session_id}:{append}:{jsonl.strip()}"
            ) or _Result(),
        )
        vmb._run_oc_native_session_hook(object(), "hook-test")
        assert calls[0] == "gateway:hook-test:agent:main:hook-test:hello:45"
        assert calls[1] == "restore:hook-test:False:original-jsonl"

    def test_run_oc_native_session_hook_retries_until_restore_sticks(self, monkeypatch):
        calls = []
        reads = iter([
            "original-jsonl\n",
            "hook-transcript\n",
            "original-jsonl\n",
        ])

        monkeypatch.setattr(vmb, "_register_session", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(vmb, "_wait_for_vm_session_jsonl_quiet", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(vmb, "_read_vm_session_jsonl", lambda *_args, **_kwargs: next(reads))
        monkeypatch.setattr(
            vmb,
            "_run_oc_native_gateway_turn",
            lambda *_args, **_kwargs: "hook-ok",
        )
        monkeypatch.setattr(vmb.time, "sleep", lambda *_args, **_kwargs: None)

        class _Result:
            returncode = 0
            stdout = ""
            stderr = ""

        monkeypatch.setattr(
            vmb,
            "_write_vm_session_jsonl",
            lambda _vm, session_id, jsonl, append=False: calls.append(
                f"restore:{session_id}:{append}:{jsonl.strip()}"
            ) or _Result(),
        )

        vmb._run_oc_native_session_hook(object(), "hook-test")
        assert calls == [
            "restore:hook-test:False:original-jsonl",
            "restore:hook-test:False:original-jsonl",
        ]

    def test_evaluate_vm_agent_timeout_retries_exhaust_fail_hard(self, monkeypatch):
        for system in ("oc-native", "quaid"):
            calls = []
            monkeypatch.setattr(vmb, "_register_session", lambda *_args, **_kwargs: None)
            monkeypatch.setattr(
                vmb,
                "_run_oc_native_gateway_turn",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired("gateway", timeout=1)),
            )
            monkeypatch.setattr(
                vmb,
                "_cleanup_oc_native_eval_session",
                lambda vm, session_id, session_key=None: calls.append(
                    f"cleanup:{session_id}:{session_key}"
                ),
            )
            with pytest.raises(RuntimeError, match="Eval query timed out"):
                vmb._evaluate_vm_agent(object(), "Who is Maya?", 9, system)
            assert calls == ["cleanup:eval-q009:agent:main:hook:eval-q009"]


class TestOcNativeGatewayStartup:
    def test_probe_vm_tcp_port_treats_ssh_timeout_as_not_ready(self):
        class _Vm:
            def ssh(self, *_args, **_kwargs):
                raise subprocess.TimeoutExpired("ssh", timeout=5)

        assert vmb._probe_vm_tcp_port(_Vm(), "127.0.0.1", 18789, timeout_s=3.0) is False

    def test_wait_for_vm_tcp_port_uses_single_guest_polling_session(self):
        calls = []

        class _Vm:
            def ssh(self, command, **kwargs):
                calls.append((command, kwargs))

                class _Result:
                    returncode = 0
                    stdout = "ready\n"
                    stderr = ""

                return _Result()

        assert (
            vmb._wait_for_vm_tcp_port(
                _Vm(),
                "127.0.0.1",
                18789,
                timeout_s=120.0,
                probe_timeout_s=3.0,
                poll_interval_s=1.0,
            )
            is True
        )
        assert len(calls) == 1
        assert "while time.time() < deadline" in calls[0][0]
        assert "sock.connect((host, port))" in calls[0][0]
        assert calls[0][1]["timeout"] >= 138

    def test_restart_oc_native_gateway_falls_back_to_gateway_run(self, monkeypatch):
        calls = []
        disable_calls = []
        patch_calls = []
        runtime_calls = []
        wait_kwargs = []
        waits = iter([False, True])

        class _Vm:
            def ssh(self, command, **_kwargs):
                calls.append(command)

                class _Result:
                    returncode = 0
                    stdout = ""
                    stderr = ""

                return _Result()

        def _fake_wait(*_args, **kwargs):
            wait_kwargs.append(kwargs)
            return next(waits)

        monkeypatch.setattr(vmb, "_disable_openclaw_quaid_config_guard", lambda vm: disable_calls.append(vm))
        monkeypatch.setattr(
            vmb,
            "_patch_openclaw_native_memory",
            lambda vm, enable_session_hook=True: patch_calls.append(enable_session_hook),
        )
        monkeypatch.setattr(vmb, "_reapply_oc_native_gateway_runtime", lambda vm: runtime_calls.append(vm))
        monkeypatch.setattr(vmb, "_wait_for_vm_tcp_port", _fake_wait)
        vmb._restart_oc_native_gateway(_Vm(), port=18789)
        assert len(disable_calls) == 1
        assert patch_calls == [True]
        assert len(runtime_calls) == 1
        assert "rm -f /tmp/openclaw-gateway-bench.log" in calls[0]
        assert "nohup env PATH=/opt/homebrew/bin:$PATH openclaw gateway start </dev/null >/tmp/openclaw-gateway-bench.log 2>&1 &" in calls[0]
        assert "nohup env PATH=/opt/homebrew/bin:$PATH openclaw gateway run --force --port 18789 </dev/null >/tmp/openclaw-gateway-bench.log 2>&1 &" in calls[1]
        assert wait_kwargs[0]["timeout_s"] == vmb.OC_NATIVE_GATEWAY_START_WAIT_S
        assert wait_kwargs[1]["timeout_s"] == vmb.OC_NATIVE_GATEWAY_RUN_WAIT_S

    def test_restart_oc_native_gateway_surfaces_gateway_log_tail_on_timeout(self, monkeypatch):
        class _Vm:
            def ssh(self, command, **_kwargs):
                class _Result:
                    returncode = 0
                    stdout = ""
                    stderr = ""

                return _Result()

        disable_calls = []
        patch_calls = []
        monkeypatch.setattr(vmb, "_disable_openclaw_quaid_config_guard", lambda vm: disable_calls.append(vm))
        monkeypatch.setattr(
            vmb,
            "_patch_openclaw_native_memory",
            lambda vm, enable_session_hook=True: patch_calls.append(enable_session_hook),
        )
        runtime_calls = []
        monkeypatch.setattr(vmb, "_reapply_oc_native_gateway_runtime", lambda vm: runtime_calls.append(vm))
        monkeypatch.setattr(vmb, "_wait_for_vm_tcp_port", lambda *_args, **_kwargs: False)
        monkeypatch.setattr(vmb, "_tail_vm_file", lambda *_args, **_kwargs: "gateway still booting")

        with pytest.raises(RuntimeError, match="gateway still booting"):
            vmb._restart_oc_native_gateway(_Vm(), port=18789)
        assert len(disable_calls) == 1
        assert patch_calls == [True]
        assert len(runtime_calls) == 1

    def test_configure_openclaw_quaid_plugin_rebinds_memory_slot_and_env(self, monkeypatch):
        calls = []
        guard_calls = []

        class _Vm:
            def ssh(self, command, **_kwargs):
                calls.append(command)

                class _Result:
                    returncode = 0
                    stdout = "Configured OpenClaw to load Quaid memory plugin\n"
                    stderr = ""

                return _Result()

        monkeypatch.setattr(vmb, "_disable_openclaw_quaid_config_guard", lambda vm: guard_calls.append(vm))
        vmb._configure_openclaw_quaid_plugin(_Vm())
        assert len(guard_calls) == 1
        assert "cp -R ~/clawd/plugins/quaid ~/.openclaw/extensions/quaid" in calls[0]
        command = calls[1]
        assert "active-memory" in command
        assert "memory-core" in command
        assert "memory-wiki" in command
        assert "slots" in command
        assert "memory" in command
        assert "QUAID_HOME" in command
        assert "QUAID_INSTANCE" in command
        assert "OPENCLAW_WORKSPACE" in command

    def test_restart_quaid_gateway_runs_under_quaid_env(self, monkeypatch):
        calls = []
        wait_kwargs = []

        class _Vm:
            def ssh(self, command, **_kwargs):
                calls.append(command)

                class _Result:
                    returncode = 0
                    stdout = ""
                    stderr = ""

                return _Result()

        monkeypatch.setattr(
            vmb,
            "_wait_for_vm_tcp_port",
            lambda *_args, **kwargs: wait_kwargs.append(kwargs) or True,
        )

        vmb._restart_quaid_gateway(_Vm(), port=18789)
        assert "nohup env PATH=/opt/homebrew/bin:$PATH" in calls[0]
        assert f"QUAID_HOME={vmb.VM_QUAID_HOME}" in calls[0]
        assert f"QUAID_INSTANCE={vmb.VM_QUAID_INSTANCE}" in calls[0]
        assert "openclaw gateway run --force --port 18789" in calls[0]
        assert wait_kwargs[0]["timeout_s"] == vmb.OC_NATIVE_GATEWAY_RUN_WAIT_S


class TestJudgeCompatibility:
    def test_judge_accepts_string_return_from_call_claude(self, monkeypatch):
        monkeypatch.setattr(vmb, "call_claude", lambda **_kwargs: '{"label":"CORRECT"}')
        label, score = vmb._judge("q", "gt", "pred", "haiku")
        assert label == "CORRECT"
        assert score == 1.0

    def test_judge_uses_extended_claude_timeout(self, monkeypatch):
        captured = {}

        def _fake_call_claude(**kwargs):
            captured.update(kwargs)
            return '{"label":"WRONG"}'

        monkeypatch.setattr(vmb, "call_claude", _fake_call_claude)
        label, score = vmb._judge("q", "gt", "pred", "haiku")
        assert label == "WRONG"
        assert score == 0.0
        assert captured["timeout"] == vmb.VM_CLAUDE_JUDGE_TIMEOUT_S


class TestEvalTokenEstimates:
    def test_score_results_aggregates_eval_token_estimates(self):
        rows = [
            {
                "judge_label": "CORRECT",
                "query_type": "factual_recall",
                "tokens_estimate": {
                    "question": 10,
                    "prediction": 20,
                    "agent_visible_total": 30,
                    "judge_prompt": 50,
                },
            },
            {
                "judge_label": "WRONG",
                "query_type": "factual_recall",
                "tokens_estimate": {
                    "question": 7,
                    "prediction": 13,
                    "agent_visible_total": 20,
                    "judge_prompt": 40,
                },
            },
        ]

        scores = vmb.score_results(rows)

        estimate = scores["eval_token_estimate"]
        assert estimate["question_tokens"] == 17
        assert estimate["prediction_tokens"] == 33
        assert estimate["agent_visible_total"] == 50
        assert estimate["judge_prompt_tokens"] == 90
        assert estimate["total_lower_bound"] == 140
        assert estimate["per_query_avg"]["total_lower_bound"] == 70.0


class TestRejudgeResults:
    def test_rejudge_results_updates_rows_and_scores(self, monkeypatch, tmp_path):
        results_dir = tmp_path / "oc-native"
        results_dir.mkdir()
        rows = [
            {
                "question": "q1",
                "ground_truth": "gt1",
                "prediction": "pred1",
                "judge_label": "WRONG",
                "score": 0.0,
                "query_type": "factual_recall",
            },
            {
                "question": "q2",
                "ground_truth": "gt2",
                "prediction": "pred2",
                "judge_label": "WRONG",
                "score": 0.0,
                "query_type": "factual_recall",
            },
        ]
        (results_dir / "eval_results.json").write_text(json.dumps(rows))
        labels = iter([("CORRECT", 1.0), ("WRONG", 0.0)])
        monkeypatch.setattr(vmb, "_judge", lambda *_args, **_kwargs: next(labels))

        scores = vmb.rejudge_results(results_dir, "gpt-4o-mini")

        saved = json.loads((results_dir / "eval_results.json").read_text())
        assert saved[0]["judge_label"] == "CORRECT"
        assert saved[0]["judge_model"] == "gpt-4o-mini"
        assert saved[1]["judge_label"] == "WRONG"
        assert scores["overall"]["accuracy"] == 50.0
        assert json.loads((results_dir / "scores.json").read_text())["overall"]["accuracy"] == 50.0

    def test_rejudge_results_skips_rows_already_judged_by_target_model(self, monkeypatch, tmp_path):
        results_dir = tmp_path / "oc-native"
        results_dir.mkdir()
        rows = [
            {
                "question": "q1",
                "ground_truth": "gt1",
                "prediction": "pred1",
                "judge_label": "CORRECT",
                "score": 1.0,
                "judge_model": "gpt-4o-mini",
                "query_type": "factual_recall",
            },
            {
                "question": "q2",
                "ground_truth": "gt2",
                "prediction": "pred2",
                "judge_label": "WRONG",
                "score": 0.0,
                "query_type": "factual_recall",
            },
        ]
        (results_dir / "eval_results.json").write_text(json.dumps(rows))
        calls = []

        def _fake_judge(*_args, **_kwargs):
            calls.append("judge")
            return ("CORRECT", 1.0)

        monkeypatch.setattr(vmb, "_judge", _fake_judge)

        scores = vmb.rejudge_results(results_dir, "gpt-4o-mini")

        saved = json.loads((results_dir / "eval_results.json").read_text())
        assert calls == ["judge"]
        assert saved[0]["judge_label"] == "CORRECT"
        assert saved[0]["judge_model"] == "gpt-4o-mini"
        assert saved[1]["judge_label"] == "CORRECT"
        assert saved[1]["judge_model"] == "gpt-4o-mini"
        assert scores["overall"]["accuracy"] == 100.0


class TestQuaidFailHard:
    def test_trigger_compaction_quaid_raises_on_nonzero_exit(self):
        class _Vm:
            def ssh(self, *_args, **_kwargs):
                class _Result:
                    returncode = 1
                    stdout = ""
                    stderr = "boom"

                return _Result()

        with pytest.raises(RuntimeError, match="Quaid compaction failed"):
            vmb._trigger_compaction(_Vm(), "benchmark-quaid", "quaid", sim_date="2026-04-07")

    def test_inject_chunks_fail_hard_on_zero_extraction_usage(self, monkeypatch):
        class _Tracker:
            def add_message(self, *_args, **_kwargs):
                return None

            def add_compaction(self, *_args, **_kwargs):
                return None

            def summary(self):
                return {}

        class _Msg:
            role = "user"
            content = "hello"
            timestamp_ms = 1_772_323_200_000
            tokens = 10

        class _Chunk:
            total_tokens = 10
            messages = [_Msg()]
            session_ids = ["S01"]
            trigger = "timeout"

        class _Vm:
            def ssh(self, *_args, **_kwargs):
                class _Result:
                    returncode = 0
                    stdout = ""
                    stderr = ""

                return _Result()

        monkeypatch.setattr(vmb, "CostTracker", _Tracker)
        monkeypatch.setattr(vmb, "messages_to_oc_native_jsonl", lambda *_args, **_kwargs: '{"ok":true}\n')
        monkeypatch.setattr(vmb, "_trigger_compaction", lambda *_args, **_kwargs: {"input_tokens": 0, "output_tokens": 0})

        with pytest.raises(RuntimeError, match="Quaid extraction failed to report usage"):
            vmb._inject_chunks(
                _Vm(),
                [_Chunk()],
                "benchmark-quaid",
                results_dir=None,
                system="quaid",
                extract_model="claude-sonnet-4-5-20250929",
                mode="natural",
            )

    def test_inject_chunks_quaid_uses_unwatched_benchmark_session_dir(self, monkeypatch, tmp_path):
        class _Tracker:
            def add_message(self, *_args, **_kwargs):
                return None

            def add_compaction(self, *_args, **_kwargs):
                return None

            def summary(self):
                return {}

        class _Msg:
            role = "user"
            content = "portfolio-site update"
            tokens = 4
            timestamp_ms = int(datetime(2026, 3, 11, tzinfo=timezone.utc).timestamp() * 1000)

        class _Chunk:
            total_tokens = 4
            trigger = "end"
            session_ids = ["S07"]
            messages = [_Msg()]

        class _Vm:
            def ssh(self, *_args, **_kwargs):
                class _Result:
                    returncode = 0
                    stdout = ""
                    stderr = ""

                return _Result()

        writes = []
        monkeypatch.setattr(vmb, "CostTracker", _Tracker)
        monkeypatch.setattr(vmb, "messages_to_oc_native_jsonl", lambda *_args, **_kwargs: '{"ok":true}\n')
        monkeypatch.setattr(
            vmb,
            "_write_vm_session_jsonl",
            lambda _vm, session_id, jsonl, append=False, sessions_dir=vmb.VM_AGENT_SESSIONS_DIR: writes.append(
                {
                    "session_id": session_id,
                    "append": append,
                    "sessions_dir": sessions_dir,
                }
            )
            or type("_Result", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
        )
        monkeypatch.setattr(
            vmb,
            "_trigger_compaction",
            lambda *_args, **_kwargs: {"input_tokens": 1, "output_tokens": 2, "artifact": {"facts": []}},
        )
        monkeypatch.setattr(
            vmb,
            "_run_vm_janitor",
            lambda *_args, **_kwargs: {"input_tokens": 0, "output_tokens": 0, "api_calls": 0, "cost_usd": 0.0},
        )

        vmb._inject_chunks(
            _Vm(),
            [_Chunk()],
            "benchmark-quaid",
            results_dir=tmp_path,
            system="quaid",
            extract_model="claude-sonnet-4-5-20250929",
            mode="natural",
        )

        assert writes[0]["sessions_dir"] == vmb.VM_QUAID_BENCH_SESSIONS_DIR


class TestVmBenchmarkCli:
    def test_cli_accepts_oc_native_without_affecting_all(self, monkeypatch, tmp_path):
        calls = []

        def _fake_run_benchmark(**kwargs):
            calls.append(kwargs["system"])
            return {"system": kwargs["system"], "dry_run": kwargs["dry_run"]}

        monkeypatch.setattr(vmb, "run_benchmark", _fake_run_benchmark)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "vm_benchmark.py",
                "--system",
                "oc-native",
                "--dry-run",
                "--results-dir",
                str(tmp_path / "results"),
            ],
        )

        vmb.main()
        assert calls == ["oc-native"]

    def test_cli_all_keeps_optional_systems_out_of_default_sweep(self, monkeypatch, tmp_path):
        calls = []

        def _fake_run_benchmark(**kwargs):
            calls.append(kwargs["system"])
            return {"system": kwargs["system"], "dry_run": kwargs["dry_run"]}

        monkeypatch.setattr(vmb, "run_benchmark", _fake_run_benchmark)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "vm_benchmark.py",
                "--system",
                "all",
                "--dry-run",
                "--results-dir",
                str(tmp_path / "results"),
            ],
        )

        vmb.main()
        assert calls == ["base", "qmd", "quaid", "quaid", "mem0"]
        assert "oc-native" not in calls

    def test_cli_defaults_arc_assets_to_data_sessions(self, monkeypatch, tmp_path):
        captured = {}

        def _fake_run_benchmark(**kwargs):
            captured["assets_dir"] = kwargs["assets_dir"]
            return {"system": kwargs["system"], "dry_run": kwargs["dry_run"]}

        monkeypatch.setattr(vmb, "run_benchmark", _fake_run_benchmark)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "vm_benchmark.py",
                "--system",
                "oc-native",
                "--dry-run",
                "--results-dir",
                str(tmp_path / "results"),
            ],
        )

        vmb.main()
        assert captured["assets_dir"] == vmb._DIR.parent / "data" / "sessions"

    def test_cli_passes_tart_host_to_run_benchmark(self, monkeypatch, tmp_path):
        captured = {}

        def _fake_run_benchmark(**kwargs):
            captured["tart_host"] = kwargs.get("tart_host")
            return {"system": kwargs["system"], "dry_run": kwargs["dry_run"]}

        monkeypatch.setattr(vmb, "run_benchmark", _fake_run_benchmark)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "vm_benchmark.py",
                "--system",
                "oc-native",
                "--dry-run",
                "--tart-host",
                "alfie.local",
                "--results-dir",
                str(tmp_path / "results"),
            ],
        )

        vmb.main()
        assert captured["tart_host"] == "alfie.local"

    def test_run_benchmark_requires_openai_key_for_gpt_judge(self, monkeypatch):
        fake_rpb = ModuleType("run_production_benchmark")
        fake_rpb._get_openai_key = lambda: ""
        monkeypatch.setitem(sys.modules, "run_production_benchmark", fake_rpb)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY is required"):
            vmb.run_benchmark(system="oc-native", dry_run=True, judge_model="gpt-4o-mini")

    def test_run_benchmark_blocks_shared_local_oc_vm_on_testbench(self, monkeypatch, tmp_path):
        fake_rpb = ModuleType("run_production_benchmark")
        fake_rpb._get_openai_key = lambda: "ok"
        monkeypatch.setitem(sys.modules, "run_production_benchmark", fake_rpb)
        monkeypatch.setattr(vmb.socket, "gethostname", lambda: "testbench.local")
        with pytest.raises(RuntimeError, match="requires --tart-host alfie.local"):
            vmb.run_benchmark(
                system="oc-native",
                dry_run=True,
                judge_model="gpt-4o-mini",
                vm_name="test-openclaw",
                results_base=tmp_path / "results",
            )

    def test_run_benchmark_blocks_shared_quaid_livetest_vm_without_tart_host(self, monkeypatch, tmp_path):
        fake_rpb = ModuleType("run_production_benchmark")
        fake_rpb._get_openai_key = lambda: "ok"
        monkeypatch.setitem(sys.modules, "run_production_benchmark", fake_rpb)
        monkeypatch.setattr(vmb.socket, "gethostname", lambda: "benchbox.local")
        with pytest.raises(RuntimeError, match="refused to use shared local VM"):
            vmb.run_benchmark(
                system="oc-native",
                dry_run=True,
                judge_model="gpt-4o-mini",
                vm_name="quaid-livetest-run",
                results_base=tmp_path / "results",
            )

    def test_run_benchmark_allows_namespaced_local_oc_vm_on_testbench(self, monkeypatch, tmp_path):
        fake_rpb = ModuleType("run_production_benchmark")
        fake_rpb._get_openai_key = lambda: "ok"
        monkeypatch.setitem(sys.modules, "run_production_benchmark", fake_rpb)
        monkeypatch.setattr(vmb.socket, "gethostname", lambda: "testbench.local")
        result = vmb.run_benchmark(
            system="oc-native",
            dry_run=True,
            judge_model="gpt-4o-mini",
            vm_name="benchmark-oc-native-run",
            results_base=tmp_path / "results",
        )
        assert result["dry_run"] is True

    def test_run_benchmark_allows_shared_vm_when_routed_to_alfie(self, monkeypatch, tmp_path):
        fake_rpb = ModuleType("run_production_benchmark")
        fake_rpb._get_openai_key = lambda: "ok"
        monkeypatch.setitem(sys.modules, "run_production_benchmark", fake_rpb)
        monkeypatch.setattr(vmb.socket, "gethostname", lambda: "testbench.local")
        result = vmb.run_benchmark(
            system="oc-native",
            dry_run=True,
            judge_model="gpt-4o-mini",
            vm_name="quaid-livetest-run",
            tart_host="alfie.local",
            results_base=tmp_path / "results",
        )
        assert result["dry_run"] is True

    def test_run_benchmark_applies_query_profile_before_limit(self, monkeypatch, tmp_path):
        captured = {}
        queries = [
            {"question": "q1", "ground_truth": "a1"},
            {"question": "q2", "ground_truth": "a2"},
            {"question": "q3", "ground_truth": "a3"},
        ]

        fake_rpb = ModuleType("run_production_benchmark")
        fake_rpb._apply_eval_query_profile = lambda items: (
            [items[2], items[0]],
            {"profile": "query-num-list", "requested": len(items), "selected": 2},
        )
        monkeypatch.setitem(sys.modules, "run_production_benchmark", fake_rpb)
        monkeypatch.setattr(vmb, "load_all_reviews", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(vmb, "load_filler_reviews", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(vmb, "merge_sessions_chronologically", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(vmb, "get_all_eval_queries", lambda *_args, **_kwargs: list(queries))
        monkeypatch.setattr(vmb, "TartVM", lambda **_kwargs: object())

        def _fake_evaluate_queries(_vm, selected, *_args, **_kwargs):
            captured["questions"] = [row["question"] for row in selected]
            return [{"judge_label": "CORRECT", "query_type": "test"}]

        monkeypatch.setattr(vmb, "evaluate_queries", _fake_evaluate_queries)
        monkeypatch.setattr(
            vmb,
            "score_results",
            lambda _rows: {"overall": {"accuracy": 100.0, "correct": 1, "partial": 0, "wrong": 0}},
        )

        vmb.run_benchmark(
            system="oc-native",
            eval_only=True,
            limit_queries=1,
            tart_host="alfie.local",
            results_base=tmp_path / "results",
        )

        assert captured["questions"] == ["q3"]

    def test_cli_rejudge_only_invokes_rejudge_results(self, monkeypatch, tmp_path):
        captured = {}

        def _fake_rejudge(results_dir, judge_model):
            captured["results_dir"] = results_dir
            captured["judge_model"] = judge_model
            return {"overall": {"accuracy": 12.34}}

        monkeypatch.setattr(vmb, "rejudge_results", _fake_rejudge)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "vm_benchmark.py",
                "--system",
                "oc-native",
                "--results-dir",
                str(tmp_path / "results"),
                "--rejudge-only",
                "--judge-model",
                "gpt-4o-mini",
            ],
        )

        vmb.main()
        assert captured["results_dir"] == tmp_path / "results" / "oc-native-timeout"


class TestSetupSystem:
    def test_oc_native_setup_is_config_driven(self, monkeypatch):
        calls = []

        class _Vm:
            tart_host = None

            def restore(self, name):
                calls.append(("restore", name))

            def ssh(self, command, **_kwargs):
                calls.append(("ssh", command))

                class _Result:
                    returncode = 0
                    stdout = ""
                    stderr = ""

                return _Result()

        monkeypatch.setattr(vmb, "_ensure_oc_native_embed_proxy", lambda _host=None: calls.append(("ensure_proxy", None)))
        monkeypatch.setattr(vmb, "_patch_openclaw_native_memory", lambda vm, enable_session_hook=True: calls.append(("patch_native", enable_session_hook)))
        monkeypatch.setattr(vmb, "_clear_vm_session_state", lambda vm: calls.append(("clear_sessions", None)))
        monkeypatch.setattr(vmb, "_clear_vm_native_memory_state", lambda vm: calls.append(("clear_native", None)))
        monkeypatch.setattr(vmb, "_restart_oc_native_gateway", lambda vm, port=18789: calls.append(("restart_gateway", port)))
        monkeypatch.setattr(vmb, "_validate_openclaw_native_memory", lambda vm: calls.append(("validate_native", None)))

        vmb.setup_system(_Vm(), "oc-native", snapshot_base="clean-openclaw")

        ssh_commands = [payload for kind, payload in calls if kind == "ssh"]
        assert not any("openclaw plugins " in cmd for cmd in ssh_commands)
        assert ("ensure_proxy", None) in calls
        assert ("patch_native", True) in calls

    def test_quaid_local_plugin_uses_vm_archive_upload(self, monkeypatch, tmp_path):
        calls = []
        plugin_dir = tmp_path / "modules" / "quaid"
        plugin_dir.mkdir(parents=True)
        memory_example = tmp_path / "memory.json.example"
        memory_example.write_text("{}\n", encoding="utf-8")
        archive = tmp_path / "quaid-plugin.tgz"
        archive.write_bytes(b"fake")

        class _Vm:
            tart_host = None

            def restore(self, name):
                calls.append(("restore", name))

            def ssh(self, command, **kwargs):
                calls.append(("ssh", command, kwargs))

                class _Result:
                    returncode = 0
                    stdout = ""
                    stderr = ""

                return _Result()

            def scp_to(self, local, remote, timeout=60):
                calls.append(("scp_to", local, remote, timeout))

                class _Result:
                    returncode = 0
                    stdout = ""
                    stderr = ""

                return _Result()

        monkeypatch.setattr(vmb, "_resolve_local_quaid_plugin_dir", lambda: plugin_dir)
        monkeypatch.setattr(vmb, "_resolve_local_quaid_memory_example", lambda _plugin_dir: memory_example)
        monkeypatch.setattr(vmb, "_build_local_quaid_plugin_tarball", lambda _plugin_dir: archive)
        monkeypatch.setattr(vmb, "_ensure_oc_native_embed_proxy", lambda _host=None: calls.append(("ensure_proxy", None)))
        monkeypatch.setattr(vmb, "_provision_openclaw_anthropic_key", lambda _vm: calls.append(("anthropic_key", None)))
        monkeypatch.setattr(vmb, "_configure_openclaw_quaid_plugin", lambda _vm: calls.append(("configure_quaid_plugin", None)))
        monkeypatch.setattr(vmb, "_restart_quaid_gateway", lambda _vm, port=18789: calls.append(("restart_quaid_gateway", port)))
        monkeypatch.setattr(vmb, "_validate_quaid_vm_embeddings", lambda _vm: calls.append(("validate_quaid_embeddings", None)))
        monkeypatch.setattr(
            vmb,
            "_upload_local_file_via_vm_ssh",
            lambda _vm, local, remote, timeout=300: calls.append(("upload", str(local), remote, timeout)) or type(
                "_Result",
                (),
                {"returncode": 0, "stdout": "", "stderr": ""},
            )(),
        )

        vmb.setup_system(_Vm(), "quaid", snapshot_base="clean-openclaw", local_plugin=True)

        assert ("ensure_proxy", None) in calls
        assert ("configure_quaid_plugin", None) in calls
        assert ("restart_quaid_gateway", 18789) in calls
        assert ("upload", str(archive), "/tmp/quaid-plugin.tgz", 300) in calls
        assert ("upload", str(memory_example), "~/clawd/plugins/quaid/memory.json.example", 120) in calls
        ssh_commands = [entry[1] for entry in calls if entry[0] == "ssh"]
        assert any("tar -xzf /tmp/quaid-plugin.tgz -C ~/clawd/plugins/quaid" in cmd for cmd in ssh_commands)
        assert any("cd ~/clawd/plugins/quaid && npm install --omit=dev --legacy-peer-deps" in cmd for cmd in ssh_commands)
        assert any(vmb.VM_QUAID_OLLAMA_URL in cmd for cmd in ssh_commands)
        assert not any("sed -i ''" in cmd for cmd in ssh_commands)

    def test_quaid_setup_defaults_to_local_checkpoint_plugin(self, monkeypatch, tmp_path):
        calls = []
        plugin_dir = tmp_path / "modules" / "quaid"
        plugin_dir.mkdir(parents=True)
        memory_example = tmp_path / "memory.json.example"
        memory_example.write_text("{}\n", encoding="utf-8")
        archive = tmp_path / "quaid-plugin.tgz"
        archive.write_bytes(b"fake")

        class _Vm:
            tart_host = None

            def restore(self, name):
                calls.append(("restore", name))

            def ssh(self, command, **kwargs):
                calls.append(("ssh", command, kwargs))

                class _Result:
                    returncode = 0
                    stdout = ""
                    stderr = ""

                return _Result()

            def scp_to(self, local, remote, timeout=60):
                calls.append(("scp_to", local, remote, timeout))

                class _Result:
                    returncode = 0
                    stdout = ""
                    stderr = ""

                return _Result()

        monkeypatch.setattr(vmb, "_resolve_local_quaid_plugin_dir", lambda: plugin_dir)
        monkeypatch.setattr(vmb, "_resolve_local_quaid_memory_example", lambda _plugin_dir: memory_example)
        monkeypatch.setattr(vmb, "_build_local_quaid_plugin_tarball", lambda _plugin_dir: archive)
        monkeypatch.setattr(vmb, "_ensure_oc_native_embed_proxy", lambda _host=None: calls.append(("ensure_proxy", None)))
        monkeypatch.setattr(vmb, "_provision_openclaw_anthropic_key", lambda _vm: calls.append(("anthropic_key", None)))
        monkeypatch.setattr(vmb, "_configure_openclaw_quaid_plugin", lambda _vm: calls.append(("configure_quaid_plugin", None)))
        monkeypatch.setattr(vmb, "_restart_quaid_gateway", lambda _vm, port=18789: calls.append(("restart_quaid_gateway", port)))
        monkeypatch.setattr(vmb, "_validate_quaid_vm_embeddings", lambda _vm: calls.append(("validate_quaid_embeddings", None)))
        monkeypatch.setattr(
            vmb,
            "_upload_local_file_via_vm_ssh",
            lambda _vm, local, remote, timeout=300: calls.append(("upload", str(local), remote, timeout)) or type(
                "_Result",
                (),
                {"returncode": 0, "stdout": "", "stderr": ""},
            )(),
        )

        vmb.setup_system(_Vm(), "quaid", snapshot_base="clean-openclaw")

        assert ("upload", str(archive), "/tmp/quaid-plugin.tgz", 300) in calls
        ssh_commands = [entry[1] for entry in calls if entry[0] == "ssh"]
        assert any("tar -xzf /tmp/quaid-plugin.tgz -C ~/clawd/plugins/quaid" in cmd for cmd in ssh_commands)

    def test_quaid_setup_provisions_codex_oauth_for_openai_answer_model(self, monkeypatch, tmp_path):
        calls = []
        plugin_dir = tmp_path / "modules" / "quaid"
        plugin_dir.mkdir(parents=True)
        memory_example = tmp_path / "memory.json.example"
        memory_example.write_text("{}\n", encoding="utf-8")
        archive = tmp_path / "quaid-plugin.tgz"
        archive.write_bytes(b"fake")

        class _Vm:
            tart_host = None

            def restore(self, name):
                calls.append(("restore", name))

            def ssh(self, command, **kwargs):
                calls.append(("ssh", command, kwargs))

                class _Result:
                    returncode = 0
                    stdout = ""
                    stderr = ""

                return _Result()

            def scp_to(self, local, remote, timeout=60):
                calls.append(("scp_to", local, remote, timeout))

                class _Result:
                    returncode = 0
                    stdout = ""
                    stderr = ""

                return _Result()

        monkeypatch.setattr(vmb, "_resolve_local_quaid_plugin_dir", lambda: plugin_dir)
        monkeypatch.setattr(vmb, "_resolve_local_quaid_memory_example", lambda _plugin_dir: memory_example)
        monkeypatch.setattr(vmb, "_build_local_quaid_plugin_tarball", lambda _plugin_dir: archive)
        monkeypatch.setattr(vmb, "_ensure_oc_native_embed_proxy", lambda _host=None: calls.append(("ensure_proxy", None)))
        monkeypatch.setattr(vmb, "_provision_openclaw_anthropic_key", lambda _vm: calls.append(("anthropic_key", None)))
        monkeypatch.setattr(vmb, "_provision_openclaw_codex_oauth", lambda _vm: calls.append(("codex_oauth", None)))
        monkeypatch.setattr(vmb, "_configure_openclaw_quaid_plugin", lambda _vm: calls.append(("configure_quaid_plugin", None)))
        monkeypatch.setattr(vmb, "_restart_quaid_gateway", lambda _vm, port=18789: calls.append(("restart_quaid_gateway", port)))
        monkeypatch.setattr(
            vmb,
            "_patch_quaid_runtime_instance_config",
            lambda _vm, **kwargs: calls.append(("patch_runtime_config", kwargs)),
        )
        monkeypatch.setattr(vmb, "_patch_gateway_model", lambda _vm, model: calls.append(("patch_gateway_model", model)))
        monkeypatch.setattr(vmb, "_validate_quaid_vm_embeddings", lambda _vm: calls.append(("validate_quaid_embeddings", None)))
        monkeypatch.setattr(
            vmb,
            "_upload_local_file_via_vm_ssh",
            lambda _vm, local, remote, timeout=300: calls.append(("upload", str(local), remote, timeout)) or type(
                "_Result",
                (),
                {"returncode": 0, "stdout": "", "stderr": ""},
            )(),
        )

        vmb.setup_system(
            _Vm(),
            "quaid",
            snapshot_base="clean-openclaw",
            local_plugin=True,
            answer_model="openai/gpt-5.4",
            openai_auth_mode="codex-oauth",
        )

        assert ("ensure_proxy", None) in calls
        assert ("codex_oauth", None) in calls
        assert ("configure_quaid_plugin", None) in calls
        assert ("restart_quaid_gateway", 18789) in calls
        assert ("validate_quaid_embeddings", None) in calls
        assert ("patch_gateway_model", "openai-codex/gpt-5.4") in calls
        assert (
            "patch_runtime_config",
            {
                "extract_model": "claude-sonnet-4-5-20250929",
                "answer_model": "openai/gpt-5.4",
                "owner_id": "maya",
                "user_name": "Maya",
            },
        ) in calls


class TestQuaidCompaction:
    def test_quaid_benchmark_session_file_uses_runtime_dir(self):
        path = vmb._quaid_benchmark_session_file("benchmark-quaid-s07")
        assert path == f"{vmb.VM_QUAID_BENCH_SESSIONS_DIR}/benchmark-quaid-s07.jsonl"
        assert ".openclaw/agents/main/sessions" not in path

    def test_trigger_compaction_sets_runtime_prompt_file(self):
        calls = []

        class _Vm:
            def ssh(self, command, **_kwargs):
                calls.append(command)

                class _Result:
                    returncode = 0
                    stdout = '{"extraction_usage":{"input_tokens":1,"output_tokens":2,"model":"claude-sonnet-4-5-20250929"}}\n'
                    stderr = ""

                return _Result()

        session_file = vmb._quaid_benchmark_session_file("benchmark-quaid")
        usage = vmb._trigger_compaction(
            _Vm(),
            "benchmark-quaid",
            "quaid",
            session_file=session_file,
        )
        assert f"QUAID_HOME={vmb.VM_QUAID_HOME}" in calls[0]
        assert f"QUAID_INSTANCE={vmb.VM_QUAID_INSTANCE}" in calls[0]
        assert f"QUAID_LLM_USAGE_LOG_PATH={vmb.VM_QUAID_LLM_USAGE_LOG_PATH}" in calls[0]
        assert "QUAID_LLM_USAGE_PHASE=ingest" in calls[0]
        assert "QUAID_LLM_USAGE_SOURCE=benchmark_extract" in calls[0]
        assert "BENCHMARK_EXTRACTION_PROMPT_FILE=~/clawd/plugins/quaid/prompts/extraction.txt" in calls[0]
        assert f"--session-file {session_file}" in calls[0]
        assert usage["input_tokens"] == 1
        assert usage["output_tokens"] == 2
