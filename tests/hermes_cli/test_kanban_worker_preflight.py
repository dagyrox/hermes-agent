"""Feature 1 startup admission: real DB outcomes and dispatcher boundaries."""

from __future__ import annotations

import json

import pytest


def test_preflight_failure_is_run_bound_and_terminal(tmp_path):
    from hermes_cli import kanban_db as kb

    with kb.connect_closing(tmp_path / "board.db") as conn:
        task_id = kb.create_task(conn, title="preflight fixture", assignee="fixture")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        assert claimed.current_run_id is not None
        payload = {
            "stage": "probe",
            "code": "credential_rejected",
            "outcome": "failed",
            "profile": "fixture",
            "provider": "deepseek",
            "model": "deepseek-flash",
            "exception": "SECRET_RESPONSE_BODY",
            "api_key": "SECRET_KEY",
        }
        assert not kb.record_preflight_outcome(
            conn,
            task_id,
            run_id=claimed.current_run_id + 1,
            passed=False,
            payload=payload,
            kind="capability",
        )
        assert kb.get_task(conn, task_id).status == "running"
        assert kb.record_preflight_outcome(
            conn,
            task_id,
            run_id=claimed.current_run_id,
            passed=False,
            payload=payload,
            kind="capability",
        )
        assert kb.get_task(conn, task_id).status == "blocked"
        assert kb.get_task(conn, task_id).consecutive_failures == 0
        events = kb.list_events(conn, task_id)
        evidence = [event for event in events if event.kind == "preflight_failed"]
        assert len(evidence) == 1
        assert evidence[0].run_id == claimed.current_run_id
        assert "SECRET" not in json.dumps(evidence[0].payload)
        assert not kb.record_preflight_outcome(
            conn,
            task_id,
            run_id=claimed.current_run_id,
            passed=True,
            payload=payload,
        )


def test_dispatch_admission_and_invalid_workspace(monkeypatch, tmp_path):
    from hermes_cli.worker_preflight import (
        WorkerPreflightBlocked,
        dispatch_admission,
        dispatch_checks,
    )

    profile = tmp_path / "profile"
    profile.mkdir()
    profile.joinpath("config.yaml").write_text(
        "kanban:\n  worker_preflight:\n    enabled: true\n    timeout_seconds: 99\n",
    )
    settings = dispatch_admission(str(profile))
    assert settings["timeout_seconds"] == 15
    with pytest.raises(WorkerPreflightBlocked) as caught:
        dispatch_checks(str(tmp_path / "missing"), branch=None, worktree=False)
    assert caught.value.payload["code"] == "workspace_invalid"


@pytest.mark.parametrize(
    "value",
    ["true", 1, {"enabled": "true"}, {"enabled": True, "required_tools": "terminal"}],
)
def test_malformed_opt_in_fails_closed(value, tmp_path):
    import yaml
    from hermes_cli.worker_preflight import WorkerPreflightBlocked, dispatch_admission

    tmp_path.joinpath("config.yaml").write_text(
        yaml.safe_dump({"kanban": {"worker_preflight": value}})
    )
    with pytest.raises(WorkerPreflightBlocked) as caught:
        dispatch_admission(str(tmp_path))
    assert caught.value.payload["code"] == "config_invalid"


def test_hung_probe_is_killed_and_reaped(tmp_path):
    import sys
    import time
    from hermes_cli.worker_preflight import bounded_probe

    started = time.monotonic()
    result = bounded_probe(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        cwd=str(tmp_path),
        timeout=1,
    )
    assert result["code"] == "probe_timeout"
    assert time.monotonic() - started < 4


@pytest.mark.parametrize("trusted", [True, False])
def test_probe_preserves_authorized_provider_access_snapshot(monkeypatch, trusted):
    from types import SimpleNamespace
    from hermes_cli import provider_access, runtime_provider
    from hermes_cli.worker_preflight import _probe

    snapshot = {"deepseek": "api"}
    seen = {}
    monkeypatch.setattr(provider_access, "trusted_profile", lambda: trusted)
    monkeypatch.setattr(provider_access, "load_provider_access", lambda: snapshot)

    def resolve(**kwargs):
        seen.update(kwargs)
        return {
            "provider": "deepseek",
            "api_mode": "chat_completions",
            "base_url": "http://127.0.0.1",
            "api_key": None,
        }

    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", resolve)
    result = _probe(
        SimpleNamespace(
            provider="deepseek", model="fixture-model", timeout=15, max_tokens=16
        )
    )
    assert seen["provider_access_snapshot"] == (snapshot if trusted else {})
    assert result["code"] == "credential_missing"


def test_invalid_diagnostic_values_do_not_escape():
    from hermes_cli.worker_preflight import sanitize_outcome

    result = sanitize_outcome({
        "stage": "secret response",
        "code": "401 secret",
        "outcome": "failed",
        "profile": "bad\ncredential",
        "provider": "https://user:secret@example.test",
        "model": "bad model secret",
        "base_url_host": "user:secret@example.test/path",
        "missing": ["bad name secret"],
        "elapsed_ms": True,
        "http_status": "401",
        "exception": "secret",
        "api_key": "secret",
    })
    assert result == {"stage": "probe", "code": "probe_failed", "outcome": "failed"}


@pytest.mark.parametrize("provider", ["openrouter", "nous", "kilocode"])
def test_aggregator_probe_sends_the_workers_unchanged_model(monkeypatch, provider):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import openai
    from hermes_cli import runtime_provider
    from hermes_cli.worker_preflight import _probe

    model = "claude-sonnet-4.6"
    seen = {}
    monkeypatch.setattr(
        runtime_provider,
        "resolve_runtime_provider",
        lambda **kwargs: {
            "provider": provider,
            "api_mode": "chat_completions",
            "base_url": "http://127.0.0.1",
            "api_key": "sk-fixture-not-a-real-key",
        },
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: seen.update(kwargs))
        )
    )
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: nullcontext(client))
    result = _probe(
        SimpleNamespace(provider=provider, model=model, timeout=15, max_tokens=16)
    )
    assert result["outcome"] == "passed"
    assert seen["model"] == model


def test_operation_readiness_requires_callable_handler(monkeypatch):
    from types import SimpleNamespace
    from hermes_cli.worker_preflight import WorkerPreflightBlocked, _check_tools
    import model_tools
    from tools.registry import registry

    names = ("terminal", "kanban_complete", "kanban_block", "kanban_heartbeat")
    monkeypatch.setattr(
        model_tools,
        "get_tool_definitions",
        lambda **kwargs: [{"function": {"name": name}} for name in names],
    )
    entry = registry.get_entry("kanban_complete")
    assert entry is not None
    monkeypatch.setattr(entry, "handler", None)
    with pytest.raises(WorkerPreflightBlocked) as caught:
        _check_tools(
            SimpleNamespace(toolsets="terminal"), {"required_tools": ["terminal"]}
        )
    assert caught.value.payload["code"] == "operation_unavailable"
    assert caught.value.payload["missing"] == ["kanban_complete"]


def test_native_budget_is_clamped_separately_from_http(tmp_path):
    from hermes_cli.worker_preflight import dispatch_admission

    tmp_path.joinpath("config.yaml").write_text(
        "kanban:\n  worker_preflight:\n    enabled: true\n    native_timeout_seconds: 999\n"
    )
    settings = dispatch_admission(str(tmp_path))
    assert settings["native_timeout_seconds"] == 45
    assert settings["timeout_seconds"] == 15
    assert settings["max_output_tokens"] == 16


def test_outer_probe_rejects_output_flood_without_retaining_it(tmp_path):
    import sys
    from hermes_cli.worker_preflight import bounded_probe

    result = bounded_probe(
        [
            sys.executable,
            "-c",
            "import sys,time; sys.stdout.write('SECRET' * 10000); sys.stdout.flush(); time.sleep(60)",
        ],
        cwd=str(tmp_path),
        timeout=3,
    )
    assert result == {"stage": "probe", "code": "probe_failed", "outcome": "failed"}


def test_supervisor_children_never_inherit_admission_flag(tmp_path):
    from types import SimpleNamespace
    from hermes_cli.backend_failover import BackendLeg
    from hermes_cli.backend_failover_worker import _leg_command

    args = SimpleNamespace(profile="talos", task="t_fixture", required_toolset=[])
    _, env = _leg_command(
        BackendLeg("api", "gpt-6.1-sol", "openai-codex"),
        args,
        {"HERMES_KANBAN_PREFLIGHT": "1", "HERMES_KANBAN_TASK": "t_fixture"},
        None,
        None,
    )
    assert "HERMES_KANBAN_PREFLIGHT" not in env
    assert env["HERMES_KANBAN_TASK"] == "t_fixture"


def test_native_version_and_auth_checks_use_bounded_capture(monkeypatch):
    from hermes_cli import worker_preflight as preflight
    from hermes_cli.backend_failover_worker import _claude_cli_compatible

    calls = []

    def capture(command, **kwargs):
        calls.append((command, kwargs))
        return 0, b"2.1.999", None

    monkeypatch.setattr(preflight, "_capture_bounded", capture)
    assert _claude_cli_compatible("claude-opus-5-5", {}, runner=preflight._native_check)
    preflight._native_check(["claude", "auth", "status", "--json"], env={}, timeout=5)
    assert [command for command, _ in calls] == [
        ["claude", "--version"],
        ["claude", "auth", "status", "--json"],
    ]
    assert all(
        options["limit"] == 4096 and options["timeout"] <= 5 for _, options in calls
    )


@pytest.mark.skipif(
    __import__("os").name != "posix", reason="POSIX process-group cancellation"
)
def test_outer_cancellation_kills_resistant_native_grandchild(tmp_path):
    import os
    from pathlib import Path
    import sys
    import time
    from hermes_cli.worker_preflight import bounded_probe

    marker = tmp_path / "native.pid"
    child_code = (
        "import os,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    parent_code = (
        "from hermes_cli.worker_preflight import _capture_bounded; "
        f"_capture_bounded([{sys.executable!r}, '-c', {child_code!r}], cwd={str(tmp_path)!r}, timeout=60)"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    result = bounded_probe(
        [sys.executable, "-c", parent_code], cwd=str(tmp_path), timeout=3, env=env
    )
    assert result["code"] == "probe_timeout"
    assert marker.exists()
    pid = int(marker.read_text())
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.02)
    else:
        pytest.fail("native grandchild survived outer cancellation")
