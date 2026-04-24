import json
import subprocess
import sys
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
    def test_native_config_uses_local_ollama_qwen_embeddings(self):
        script = vmb._build_openclaw_native_config_script(enable_session_hook=True)
        assert "plugins.setdefault('slots', {})['memory'] = 'memory-core'" in script
        assert "memory['backend'] = 'builtin'" in script
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
        assert "ms['model'] = 'qwen3-embedding:8b'" in script
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
        assert "hook_entries.setdefault('session-memory', {})['enabled'] = enable_hook" in script

    def test_native_config_respects_embed_base_url_override(self, monkeypatch):
        monkeypatch.setattr(vmb, "OC_NATIVE_EMBED_BASE_URL", "http://192.168.64.1:11435/v1")
        script = vmb._build_openclaw_native_config_script(enable_session_hook=True)
        assert "remote['baseUrl'] = 'http://192.168.64.1:11435/v1'" in script

    def test_extract_openclaw_memory_status_tolerates_warning_prefix(self):
        payload = '[{"status":{"provider":"openai","model":"qwen3-embedding:8b"}}]'
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

                if "openclaw memory status" in command:
                    return _Result(
                        0,
                        '[{"status":{"provider":"openai","model":"qwen3-embedding:8b"}}]',
                        "",
                    )
                if len([c for c in calls if "python3 -c" in c]) == 1:
                    return _Result(1, "", "socket.timeout: timed out")
                return _Result(0, '{"ok": true, "dims": 4096}', "")

        sleeps = []
        monkeypatch.setattr(vmb.time, "sleep", lambda seconds: sleeps.append(seconds))
        vmb._validate_openclaw_native_memory(_Vm())
        assert sleeps == [3]

    def test_oc_native_session_ids_are_stable_per_review(self):
        arc = type("Review", (), {"session_num": 3})()
        filler = type("Review", (), {"session_num": -18})()
        weird = type("Review", (), {"session_num": None})()
        assert vmb._oc_native_session_id(arc, 0) == "benchmark-oc-native-s03"
        assert vmb._oc_native_session_id(filler, 1) == "benchmark-oc-native-f018"
        assert vmb._oc_native_session_id(weird, 2) == "benchmark-oc-native-r002"

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
            system="oc-native",
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
        assert calls[0][1] == "sk-test"
        assert "'openai:default'" in calls[0][0]
        assert "last_good" in calls[0][0]
        assert "OPENAI_API_KEY=" in calls[0][0]
        assert "env_path" in calls[0][0]

    def test_provision_openclaw_openai_key_requires_key(self, monkeypatch):
        monkeypatch.setattr(vmb, "_resolve_openai_api_key_for_vm", lambda: "")

        class _Vm:
            pass

        with pytest.raises(RuntimeError, match="OPENAI_API_KEY is required"):
            vmb._provision_openclaw_openai_key(_Vm())

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
            lambda: {"type": "oauth", "provider": "openai-codex", "access": "codex.jwt.token"},
        )
        vmb._provision_openclaw_codex_oauth(_Vm())
        assert json.loads(calls[0][1]) == {"token": "codex.jwt.token"}
        assert "credentials.json" in calls[0][0]
        assert "openai-codex:default" in calls[0][0]

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
        assert "admin@192.168.64.3:~/x.txt" in calls[2][-1]

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


class TestOpenClawNativeReindex:
    def test_force_reindex_requires_selected_source_chunks(self):
        calls = []
        ticks = iter([0, 0, 2000, 2000, 4000, 4000])

        class _Vm:
            def ssh(self, command, **_kwargs):
                calls.append(command)

                class _Result:
                    def __init__(self, stdout, returncode=0, stderr=""):
                        self.returncode = 0
                        self.stderr = stderr
                        self.stdout = stdout

                if "nohup sh -lc" in command:
                    return _Result("12345\n")
                return _Result(
                    '[{"status":{"dirty":false,"sourceCounts":['
                    '{"source":"memory","files":1,"chunks":1},'
                    '{"source":"sessions","files":0,"chunks":0}'
                    ']}}]'
                )

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(vmb.time, "monotonic", lambda: next(ticks))
        monkeypatch.setattr(vmb.time, "sleep", lambda _seconds: None)
        try:
            with pytest.raises(RuntimeError, match="oc-native sessions did not finish indexing"):
                vmb._force_openclaw_native_reindex(
                    _Vm(), source_name="sessions", min_indexed_files=1
                )
        finally:
            monkeypatch.undo()
        assert calls[0].startswith(
            "nohup sh -lc 'export PATH=/opt/homebrew/bin:$PATH; "
            "openclaw memory index --agent main --force"
        )
        assert calls[1] == "openclaw memory status --agent main --json"

    def test_force_reindex_polls_until_clean_for_memory_source(self, monkeypatch):
        calls = []
        sleeps = []

        class _Vm:
            def __init__(self):
                self.polls = 0

            def ssh(self, command, **_kwargs):
                calls.append(command)

                class _Result:
                    def __init__(self, stdout, returncode=0, stderr=""):
                        self.returncode = returncode
                        self.stderr = stderr
                        self.stdout = stdout

                if "nohup sh -lc" in command:
                    return _Result("12345\n")
                self.polls += 1
                if self.polls == 1:
                    return _Result(
                        '[{"status":{"dirty":true,"sourceCounts":['
                        '{"source":"memory","files":2,"chunks":10}'
                        ']}}]'
                    )
                return _Result(
                    '[{"status":{"dirty":false,"sourceCounts":['
                    '{"source":"memory","files":3,"chunks":12}'
                    ']}}]'
                )

        monkeypatch.setattr(vmb.time, "sleep", lambda seconds: sleeps.append(seconds))
        vm = _Vm()
        status = vmb._force_openclaw_native_reindex(
            vm, source_name="memory", min_indexed_files=3
        )
        assert status["dirty"] is False
        assert sleeps == [10]

    def test_force_reindex_uses_extended_timeout_budget(self, monkeypatch):
        ticks = iter([0, 0, 1810, 1810, 1820, 1820])

        class _Vm:
            def __init__(self):
                self.polls = 0

            def ssh(self, command, **_kwargs):
                class _Result:
                    def __init__(self, stdout, returncode=0, stderr=""):
                        self.returncode = returncode
                        self.stderr = stderr
                        self.stdout = stdout

                if "nohup sh -lc" in command:
                    return _Result("12345\n")
                self.polls += 1
                if self.polls == 1:
                    return _Result(
                        '[{"status":{"dirty":true,"sourceCounts":['
                        '{"source":"sessions","files":140,"chunks":682}'
                        ']}}]'
                    )
                return _Result(
                    '[{"status":{"dirty":false,"sourceCounts":['
                    '{"source":"sessions","files":277,"chunks":1200}'
                    ']}}]'
                )

        monkeypatch.setattr(vmb.time, "monotonic", lambda: next(ticks))
        monkeypatch.setattr(vmb.time, "sleep", lambda _seconds: None)
        status = vmb._force_openclaw_native_reindex(
            _Vm(), source_name="sessions", min_indexed_files=277
        )
        assert status["dirty"] is False


class TestVmEvalIsolation:
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

        vmb._register_session(_Vm(), "benchmark-oc-native-s01")
        assert "sessionFile" in calls[0]
        assert ".openclaw/agents/main/sessions/" in calls[0]

    def test_evaluate_vm_agent_registers_fresh_session(self, monkeypatch):
        calls = []

        class _Vm:
            def ssh(self, command, **_kwargs):
                calls.append(command)

                class _Result:
                    returncode = 0
                    stdout = "ok"
                    stderr = ""

                return _Result()

        monkeypatch.setattr(vmb, "_register_session", lambda vm, session_id: calls.append(f"register:{session_id}"))
        monkeypatch.setattr(vmb, "_extract_agent_answer", lambda raw: raw)
        answer = vmb._evaluate_vm_agent(_Vm(), "Who is Maya?", 7, "oc-native")
        assert answer == "ok"
        assert calls[0] == "register:eval-q007"
        assert "--session-id eval-q007" in calls[1]

    def test_evaluate_vm_agent_retries_timeout_then_succeeds(self, monkeypatch):
        calls = []
        attempts = {"n": 0}

        class _Vm:
            def ssh(self, command, **_kwargs):
                calls.append(command)
                attempts["n"] += 1
                if attempts["n"] == 1:
                    raise subprocess.TimeoutExpired(command, timeout=1)

                class _Result:
                    returncode = 0
                    stdout = "ok-after-retry"
                    stderr = ""

                return _Result()

        monkeypatch.setattr(vmb, "_register_session", lambda vm, session_id: calls.append(f"register:{session_id}"))
        monkeypatch.setattr(vmb, "_extract_agent_answer", lambda raw: raw)
        answer = vmb._evaluate_vm_agent(_Vm(), "Who is Maya?", 8, "oc-native")
        assert answer == "ok-after-retry"
        assert attempts["n"] == 2
        assert calls[0] == "register:eval-q008"
        assert "--session-id eval-q008" in calls[1]

    def test_evaluate_vm_agent_timeout_retries_exhaust_fail_hard(self, monkeypatch):
        class _Vm:
            def ssh(self, command, **_kwargs):
                raise subprocess.TimeoutExpired(command, timeout=1)

        monkeypatch.setattr(vmb, "_register_session", lambda *_args, **_kwargs: None)
        with pytest.raises(RuntimeError, match="Eval query timed out"):
            vmb._evaluate_vm_agent(_Vm(), "Who is Maya?", 9, "oc-native")


class TestOcNativeGatewayStartup:
    def test_restart_oc_native_gateway_falls_back_to_gateway_run(self, monkeypatch):
        calls = []
        waits = iter([False, True])

        class _Vm:
            def ssh(self, command, **_kwargs):
                calls.append(command)

                class _Result:
                    returncode = 0
                    stdout = ""
                    stderr = ""

                return _Result()

        monkeypatch.setattr(vmb, "_wait_for_vm_tcp_port", lambda *_args, **_kwargs: next(waits))
        vmb._restart_oc_native_gateway(_Vm(), port=18789)
        assert "openclaw gateway start --allow-unconfigured --port 18789" in calls[0]
        assert "nohup openclaw gateway run --allow-unconfigured --force --port 18789" in calls[1]


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
        assert captured["judge_model"] == "gpt-4o-mini"
