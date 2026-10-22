from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


def test_read_only_smoke_environment_drops_parent_lifecycle_context():
    from hermes_cli.backend_smoke import sanitized_smoke_environment

    parent = {
        "HOME": "/root",
        "USER": "root",
        "PATH": "/bin",
        "LANG": "C.UTF-8",
        "HERMES_KANBAN_TASK": "t_parent",
        "HERMES_KANBAN_RUN_ID": "494",
        "HERMES_SESSION_ID": "session-parent",
        "HERMES_UI_SESSION_ID": "ui-parent",
        "HERMES_DELEGATION_ID": "delegation-parent",
        "DELEGATION_CHILD_TIMEOUT_SECONDS": "600",
        "OPENAI_API_KEY": "must-not-pass",
        "ANTHROPIC_API_KEY": "must-not-pass",
    }

    child = sanitized_smoke_environment(parent)

    assert child == {
        "HOME": "/root",
        "USER": "root",
        "PATH": "/bin",
        "LANG": "C.UTF-8",
    }


def test_read_only_smoke_commands_expose_no_write_or_lifecycle_tools(monkeypatch):
    from hermes_cli import backend_smoke

    monkeypatch.setattr(backend_smoke, "_resolve_hermes_argv", lambda: ["/opt/hermes"])
    api = backend_smoke.build_smoke_command(
        profile="talos", provider="openai-codex", model="gpt-6.1-sol"
    )
    claude = backend_smoke.build_smoke_command(
        profile="athena", provider="claude-code-cli", model="claude-opus-5-5"
    )

    assert api[api.index("--toolsets") + 1] == "__backend_smoke_no_tools__"
    assert "--ignore-rules" in api
    assert api.index("--max-turns") > api.index("chat")
    assert not any("kanban" in part.lower() for part in api)
    assert claude[claude.index("--tools") + 1] == ""
    assert "--safe-mode" in claude
    assert "--strict-mcp-config" in claude
    assert not any("mcp__hermes" in part.lower() for part in claude)


def test_read_only_smoke_subprocess_cannot_inherit_parent_task(tmp_path):
    from hermes_cli.backend_smoke import run_smoke_command

    marker = tmp_path / "parent-was-visible"
    command = [
        sys.executable,
        "-c",
        (
            "import os, pathlib; "
            f"p=pathlib.Path({str(marker)!r}); "
            "leaked=any(k.startswith(('HERMES_KANBAN_', 'HERMES_SESSION_', "
            "'HERMES_DELEGATION_', 'DELEGATION_')) for k in os.environ); "
            "p.write_text('leaked') if leaked else print('SMOKE_OK')"
        ),
    ]
    parent = dict(os.environ)
    parent.update({
        "HERMES_KANBAN_TASK": "t_parent",
        "HERMES_KANBAN_RUN_ID": "494",
        "HERMES_SESSION_ID": "session-parent",
        "HERMES_DELEGATION_ID": "delegation-parent",
    })

    completed = run_smoke_command(command, base_env=parent, timeout=10)

    assert completed.returncode == 0
    assert completed.stdout.strip() == "SMOKE_OK"
    assert not marker.exists()


def _leg(
    kind: str = "api",
    provider: str | None = "openai-codex",
    model: str = "gpt-6.1-sol",
    **extra,
):
    from hermes_cli.backend_failover import BackendLeg

    return BackendLeg(kind=kind, provider=provider, model=model, **extra)


def test_system_prompt_loads_identity_from_assignee_profile(tmp_path):
    from hermes_cli.backend_failover_worker import _system_prompt

    (tmp_path / "SOUL.md").write_text("Assignee identity", encoding="utf-8")

    prompt_path = _system_prompt(tmp_path, [])

    assert "Assignee identity" in prompt_path.read_text(encoding="utf-8")


def test_api_leg_command_pins_provider_model_and_disables_nested_chain(monkeypatch):
    from hermes_cli import backend_failover_worker as worker

    monkeypatch.setattr(worker, "_resolve_hermes_argv", lambda: ["/opt/hermes"])
    args = argparse.Namespace(
        profile="talos",
        task="t_probe",
        timeout_seconds=600,
        required_toolset=["terminal", "file"],
    )
    cmd, env = worker._leg_command(_leg(), args, {}, None, None)

    assert cmd[:5] == ["/opt/hermes", "-p", "talos", "--cli", "--accept-hooks"]
    assert cmd[cmd.index("-m") + 1] == "gpt-6.1-sol"
    assert cmd[cmd.index("--provider") + 1] == "openai-codex"
    assert "--accept-hooks" in cmd
    assert cmd[cmd.index("--toolsets") + 1] == "terminal,file"
    assert cmd[-4:] == ["chat", "-q", "work kanban task t_probe", "-Q"]
    assert env["HERMES_BACKEND_FAILOVER_CHILD"] == "1"


def test_dispatch_context_requires_existing_matching_profile_without_creating_identity(
    tmp_path, monkeypatch
):
    from hermes_cli import backend_failover_worker as worker

    missing_home = tmp_path / "profiles" / "talos"
    monkeypatch.setenv("HERMES_HOME", str(missing_home))
    monkeypatch.setenv("HERMES_PROFILE", "talos")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_probe")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "42")

    with pytest.raises(RuntimeError, match="does not exist"):
        worker._validate_dispatch_context(
            argparse.Namespace(profile="talos", task="t_probe"),
            profile_resolver=lambda _profile: str(missing_home),
        )

    assert not missing_home.exists()


def test_dispatch_context_rejects_mismatched_home_before_artifacts(tmp_path, monkeypatch):
    from hermes_cli import backend_failover_worker as worker

    expected = tmp_path / "profiles" / "talos"
    wrong = tmp_path / "profiles" / "dispatcher"
    expected.mkdir(parents=True)
    wrong.mkdir(parents=True)
    (expected / "SOUL.md").write_text("Talos", encoding="utf-8")
    (wrong / "SOUL.md").write_text("Dispatcher", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(wrong))
    monkeypatch.setenv("HERMES_PROFILE", "talos")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_probe")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "42")

    with pytest.raises(RuntimeError, match="does not match"):
        worker._validate_dispatch_context(
            argparse.Namespace(profile="talos", task="t_probe"),
            profile_resolver=lambda _profile: str(expected),
        )

    assert not (wrong / "cache").exists()


def _claimed_dispatch_context(tmp_path, monkeypatch, *, assignee="talos"):
    from hermes_cli import kanban_db as kb

    profile = tmp_path / "profiles" / assignee
    profile.mkdir(parents=True)
    (profile / "SOUL.md").write_text("Assignee identity", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    db_path = tmp_path / "kanban.db"
    kb.init_db(db_path)
    with kb.connect(db_path) as conn:
        task_id = kb.create_task(
            conn,
            title="probe",
            assignee=assignee,
            workspace_kind="dir",
            workspace_path=str(workspace),
        )
        task = kb.claim_task(conn, task_id, claimer="probe-lock")
    assert task is not None and task.current_run_id is not None
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setenv("HERMES_PROFILE", assignee)
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "probe-lock")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    return profile, workspace, db_path, task


def test_dispatch_context_requires_live_owned_task_run_and_workspace(tmp_path, monkeypatch):
    from hermes_cli import backend_failover_worker as worker

    profile, _workspace, _db_path, task = _claimed_dispatch_context(tmp_path, monkeypatch)

    assert worker._validate_dispatch_context(
        argparse.Namespace(profile="talos", task=task.id),
        profile_resolver=lambda _profile: str(profile),
    ) == profile.resolve()


@pytest.mark.parametrize(
    ("env_name", "value", "error"),
    [
        ("HERMES_KANBAN_RUN_ID", "999999", "run"),
        ("HERMES_KANBAN_CLAIM_LOCK", "other-lock", "claim"),
        ("HERMES_KANBAN_WORKSPACE", "/tmp/other-workspace", "workspace"),
    ],
)
def test_dispatch_context_rejects_unowned_database_identity_before_artifacts(
    tmp_path, monkeypatch, env_name, value, error
):
    from hermes_cli import backend_failover_worker as worker

    profile, _workspace, _db_path, task = _claimed_dispatch_context(tmp_path, monkeypatch)
    monkeypatch.setenv(env_name, value)

    with pytest.raises(RuntimeError, match=error):
        worker._validate_dispatch_context(
            argparse.Namespace(profile="talos", task=task.id),
            profile_resolver=lambda _profile: str(profile),
        )

    assert not (profile / "cache").exists()


def test_dispatch_context_does_not_create_missing_board(tmp_path, monkeypatch):
    from hermes_cli import backend_failover_worker as worker

    profile = tmp_path / "profiles" / "talos"
    profile.mkdir(parents=True)
    (profile / "SOUL.md").write_text("Talos", encoding="utf-8")
    missing_db = tmp_path / "missing.db"
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setenv("HERMES_PROFILE", "talos")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_missing")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "999999")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(missing_db))

    with pytest.raises(RuntimeError, match="board database"):
        worker._validate_dispatch_context(
            argparse.Namespace(profile="talos", task="t_missing"),
            profile_resolver=lambda _profile: str(profile),
        )

    assert not missing_db.exists()


def test_task_specific_capability_gate_rejects_unsupported_claude_surface(monkeypatch):
    from hermes_cli import backend_failover_worker as worker

    monkeypatch.setattr(
        worker,
        "_resolved_required_tool_schemas",
        lambda _toolsets: ({"read_file", "browser_navigate", "image_gen"}, 100),
    )
    args = argparse.Namespace(
        required_toolset=["file", "browser", "image_gen"],
        required_context_chars=1000,
    )
    capability_mismatches = getattr(worker, "_capability_mismatches")
    missing = capability_mismatches(_leg(kind="claude_code", provider=None), args)
    assert missing == ("tool:browser_navigate", "tool:image_gen")
    assert capability_mismatches(_leg(), args) == ()


def test_task_specific_capability_gate_rejects_oversized_context():
    from hermes_cli import backend_failover_worker as worker

    args = argparse.Namespace(required_toolset=["file"], required_context_chars=600_001)
    assert getattr(worker, "_capability_mismatches")(
        _leg(kind="claude_code", provider=None), args
    ) == ("context_window",)


def test_api_capability_gate_uses_model_evidence_for_context_and_images(monkeypatch):
    from hermes_cli import backend_failover_worker as worker

    capabilities = SimpleNamespace(
        supports_tools=True,
        supports_vision=False,
        supports_reasoning=True,
        context_window=100,
        max_output_tokens=20,
        context_window_observed=True,
        max_output_tokens_observed=True,
    )
    monkeypatch.setattr(
        "agent.models_dev.get_model_capabilities",
        lambda _provider, _model: capabilities,
    )
    args = argparse.Namespace(
        required_toolset=["file"],
        required_context_chars=401,
        requires_native_image=True,
        native_image_path=["/tmp/fixture.png"],
    )

    assert worker._capability_mismatches(_leg(), args) == (
        "context_window",
        "native_image_input",
    )


def test_api_capability_gate_fails_closed_without_model_evidence(monkeypatch):
    from hermes_cli import backend_failover_worker as worker

    monkeypatch.setattr(
        "agent.models_dev.get_model_capabilities",
        lambda _provider, _model: None,
    )
    args = argparse.Namespace(
        required_toolset=["file"],
        required_context_chars=1,
        requires_native_image=False,
    )

    assert worker._capability_mismatches(_leg(), args) == ("capability_evidence",)


def test_claude_leg_command_uses_subscription_cli_and_scrubs_api_credentials(tmp_path):
    from hermes_cli import backend_failover_worker as worker

    args = argparse.Namespace(profile="athena", task="t_probe", timeout_seconds=600)
    env = {
        "HOME": "/root",
        "USER": "root",
        "PATH": "/bin",
        "ANTHROPIC_API_KEY": "must-not-pass",
        "ANTHROPIC_AUTH_TOKEN": "must-not-pass",
        "OPENAI_API_KEY": "must-not-pass",
        "HERMES_KANBAN_TASK": "t_probe",
    }
    cmd, child_env = worker._leg_command(
        _leg(kind="claude_code", provider=None, model="opus"),
        args,
        env,
        tmp_path / "mcp.json",
        tmp_path / "system.md",
    )

    assert cmd[0] == "claude"
    assert cmd[cmd.index("--model") + 1] == "opus"
    assert "--restricted" in cmd
    assert "--mcp-config" in cmd
    assert "anthropic" not in " ".join(cmd).lower()
    assert "ANTHROPIC_API_KEY" not in child_env
    assert "ANTHROPIC_AUTH_TOKEN" not in child_env
    assert "OPENAI_API_KEY" not in child_env
    assert child_env["HOME"] == "/root"


def test_backend_child_environments_do_not_inherit_unrelated_secrets():
    from hermes_cli.backend_failover_worker import _api_environment, _claude_environment

    base = {
        "HOME": "/tmp/home",
        "PATH": "/bin",
        "HERMES_HOME": "/tmp/profile",
        "HERMES_KANBAN_TASK": "t_1",
        "HERMES_KANBAN_RUN_ID": "9",
        "HERMES_KANBAN_SECRET_TOKEN": "do-not-copy",
        "TERMINAL_CWD": "/tmp/workspace",
        "AWS_ACCESS_KEY_ID": "do-not-copy",
        "DATABASE_URL": "do-not-copy",
        "GITHUB_PAT": "do-not-copy",
        "OPENAI_API_KEY": "do-not-copy",
    }
    claude = _claude_environment(base)
    api = _api_environment(base)
    assert claude == {"HOME": str(Path.home()), "PATH": "/bin"}
    assert api["HERMES_KANBAN_TASK"] == "t_1"
    assert api["HERMES_KANBAN_RUN_ID"] == "9"
    assert "HERMES_KANBAN_SECRET_TOKEN" not in api
    assert api["TERMINAL_CWD"] == "/tmp/workspace"
    assert not ({"AWS_ACCESS_KEY_ID", "DATABASE_URL", "GITHUB_PAT", "OPENAI_API_KEY"} & api.keys())


def test_api_availability_uses_profile_secret_not_supervisor_environment(tmp_path, monkeypatch):
    from hermes_cli import backend_failover_worker as worker

    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / ".env").write_text("DEEPSEEK_API_KEY=profile-only\n", encoding="utf-8")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    assert worker._leg_available(
        _leg(provider="deepseek", model="deepseek-flash"), profile
    )


def test_api_availability_rejects_supervisor_only_secret(tmp_path, monkeypatch):
    from hermes_cli import backend_failover_worker as worker

    profile = tmp_path / "profile"
    profile.mkdir()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "supervisor-only")

    assert not worker._leg_available(
        _leg(provider="deepseek", model="deepseek-flash"), profile
    )


def test_claude_review_leg_cannot_mutate_workspace(tmp_path):
    from hermes_cli import backend_failover_worker as worker
    from hermes_cli.backend_failover import BackendLeg

    args = argparse.Namespace(profile="athena", task="t_probe", timeout_seconds=600)
    cmd, _env = worker._leg_command(
        BackendLeg("claude_code", "opus", capability_mode="read_only"),
        args,
        {},
        tmp_path / "mcp.json",
        tmp_path / "system.md",
    )
    allowed = cmd[cmd.index("--allowedTools") + 1 : cmd.index("--")]
    assert {"Read", "Glob", "Grep"}.issubset(allowed)
    assert not {"Edit", "Write", "Bash"}.intersection(allowed)


def test_claude_full_leg_has_restricted_workspace_execution_surface(tmp_path):
    from hermes_cli import backend_failover_worker as worker
    from hermes_cli.backend_failover import BackendLeg

    args = argparse.Namespace(profile="talos", task="t_probe", timeout_seconds=600)
    cmd, _env = worker._leg_command(
        BackendLeg("claude_code", "claude-sonnet-5-5", capability_mode="full"),
        args,
        {"HOME": "/root", "PATH": "/bin"},
        tmp_path / "mcp.json",
        tmp_path / "system.md",
    )
    tools = cmd[cmd.index("--tools") + 1]
    assert tools == "Read,Glob,Grep,Edit,Write,Bash"
    allowed = cmd[cmd.index("--allowedTools") + 1 : cmd.index("--")]
    assert "Bash" in allowed
    assert "--restricted" in cmd
    assert cmd[cmd.index("--permission-mode") + 1] == "auto"
    assert cmd[cmd.index("--permission-prompts") + 1] == "none"


def test_task_specific_capability_gate_accepts_restricted_claude_execution():
    from hermes_cli import backend_failover_worker as worker

    args = argparse.Namespace(
        required_toolset=["file", "terminal", "code_execution"],
        required_context_chars=1000,
    )
    missing = worker._capability_mismatches(
        _leg(kind="claude_code", provider=None, capability_mode="full"), args
    )
    assert missing == ()


def test_claude_orchestrator_rejects_unavailable_delegate_task():
    from hermes_cli import backend_failover_worker as worker

    args = argparse.Namespace(required_toolset=["delegation"], required_context_chars=1)
    assert worker._capability_mismatches(
        _leg(kind="claude_code", provider=None, capability_mode="orchestrator"),
        args,
    ) == ("tool:delegate_task",)


def test_claude_native_image_read_does_not_claim_vision_tool_parity(monkeypatch):
    from hermes_cli import backend_failover_worker as worker

    monkeypatch.setattr(worker, "_resolved_required_tool_schemas", lambda _toolsets: ({"vision_analyze"}, 100))
    args = argparse.Namespace(
        required_toolset=["vision"],
        required_context_chars=1,
        requires_native_image=False,
    )

    assert worker._capability_mismatches(
        _leg(kind="claude_code", provider=None, capability_mode="full"), args
    ) == ("tool:vision_analyze",)


def test_claude_leg_rejects_untransported_native_image_payload(monkeypatch):
    from hermes_cli import backend_failover_worker as worker

    monkeypatch.setattr(worker, "_resolved_required_tool_schemas", lambda _toolsets: (set(), 0))
    args = argparse.Namespace(
        required_toolset=[],
        required_context_chars=1,
        requires_native_image=True,
    )

    assert worker._capability_mismatches(
        _leg(kind="claude_code", provider=None, capability_mode="full"), args
    ) == ("native_image_input",)


def test_api_gate_fails_closed_when_profile_tool_schemas_cannot_be_resolved(monkeypatch):
    from hermes_cli import backend_failover_worker as worker

    monkeypatch.setattr(worker, "_resolved_required_tool_schemas", lambda _toolsets: (None, 0))
    monkeypatch.setattr(
        "agent.models_dev.get_model_capabilities",
        lambda _provider, _model: SimpleNamespace(
            supports_tools=True,
            supports_vision=True,
            supports_reasoning=True,
            context_window=1000,
            max_output_tokens=10,
            context_window_observed=True,
            max_output_tokens_observed=True,
        ),
    )
    args = argparse.Namespace(
        required_toolset=["file"],
        required_context_chars=1,
        requires_native_image=False,
    )

    assert worker._capability_mismatches(_leg(), args) == ("tool_schema_evidence",)


@pytest.mark.parametrize(
    ("limit", "observed_field"),
    [
        ({"output": 4096}, "context_window_observed"),
        ({"context": 128_000}, "max_output_tokens_observed"),
        ({"context": 0, "output": 4096}, "context_window_observed"),
        ({"context": 128_000, "output": -1}, "max_output_tokens_observed"),
        ({"context": "128000", "output": 4096}, "context_window_observed"),
        ({"context": 128_000, "output": "4096"}, "max_output_tokens_observed"),
        ({"context": 128_000.0, "output": 4096}, "context_window_observed"),
        ({"context": 128_000, "output": 4096.0}, "max_output_tokens_observed"),
        ({"context": True, "output": 4096}, "context_window_observed"),
        ({"context": 128_000, "output": True}, "max_output_tokens_observed"),
    ],
)
def test_api_gate_rejects_missing_or_invalid_catalog_limits(
    monkeypatch, limit, observed_field
):
    from agent import models_dev
    from hermes_cli import backend_failover_worker as worker

    monkeypatch.setattr(
        models_dev,
        "_get_provider_models",
        lambda _provider: {
            "gpt-6.1-sol": {
                "tool_call": True,
                "modalities": {"input": ["text"]},
                "limit": limit,
            }
        },
    )
    capabilities = models_dev.get_model_capabilities("openai-codex", "gpt-6.1-sol")
    assert capabilities is not None
    assert getattr(capabilities, observed_field) is False
    monkeypatch.setattr(models_dev, "get_model_capabilities", lambda *_args: capabilities)
    monkeypatch.setattr(worker, "_resolved_required_tool_schemas", lambda _toolsets: (set(), 2))
    args = argparse.Namespace(
        required_toolset=[],
        required_context_chars=1,
        requires_native_image=False,
        native_image_path=[],
    )

    assert worker._capability_mismatches(_leg(), args) == (
        "context_limit_evidence",
    )


def test_multi_image_parity_denial_happens_before_availability_or_child_spawn(monkeypatch):
    from hermes_cli import backend_failover_worker as worker

    monkeypatch.setattr(
        "agent.models_dev.get_model_capabilities",
        lambda *_args: SimpleNamespace(
            supports_tools=True,
            supports_vision=True,
            supports_reasoning=True,
            context_window=128_000,
            max_output_tokens=4096,
            context_window_observed=True,
            max_output_tokens_observed=True,
        ),
    )
    monkeypatch.setattr(worker, "_resolved_required_tool_schemas", lambda _toolsets: (set(), 2))
    monkeypatch.setattr(worker, "_comment", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        worker,
        "_leg_available",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("availability reached after multi-image parity denial")
        ),
    )
    monkeypatch.setattr(
        worker,
        "_run_child",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("child spawned after multi-image parity denial")
        ),
    )
    args = argparse.Namespace(
        task="t_probe",
        required_toolset=[],
        required_context_chars=1,
        requires_native_image=True,
        native_image_path=["/tmp/a.png", "/tmp/b.png"],
        timeout_seconds=1,
    )

    assert worker._run_leg(_leg(), 0, args, None, None) == worker.EXIT_ELIGIBLE


def test_effective_context_size_counts_native_media_and_envelope(tmp_path, monkeypatch):
    from hermes_cli import backend_failover_worker as worker

    system_prompt = tmp_path / "system.md"
    system_prompt.write_text("system", encoding="utf-8")

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr("hermes_cli.kanban_db.connect_closing", lambda **_kwargs: _Conn())
    monkeypatch.setattr("hermes_cli.kanban_db.build_worker_context", lambda *_args: "task")
    monkeypatch.setattr(worker, "_resolved_required_tool_schemas", lambda _toolsets: (set(), 100))

    without_image = worker._effective_context_chars(
        "t_probe",
        declared_chars=0,
        system_prompt_path=system_prompt,
        required_toolsets=[],
    )
    with_image = worker._effective_context_chars(
        "t_probe",
        declared_chars=0,
        system_prompt_path=system_prompt,
        required_toolsets=[],
        native_image_bytes=300,
    )
    assert with_image - without_image == 400 + worker._NATIVE_IMAGE_ENVELOPE_CHARS


def test_effective_context_size_includes_durable_handoff_and_system_prompt(
    tmp_path, monkeypatch
):
    from hermes_cli import backend_failover_worker as worker

    system_prompt = tmp_path / "system.md"
    system_prompt.write_text("system-prompt", encoding="utf-8")

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(
        "hermes_cli.kanban_db.connect_closing",
        lambda **_kwargs: _Conn(),
    )
    monkeypatch.setattr(
        "hermes_cli.kanban_db.build_worker_context",
        lambda _conn, _task: "parent-handoff-and-comments",
    )

    monkeypatch.setattr(
        worker,
        "_resolved_required_tool_schemas",
        lambda _toolsets: ({"read_file"}, 1234),
    )
    measured = worker._effective_context_chars(
        "t_probe",
        declared_chars=1,
        system_prompt_path=system_prompt,
        required_toolsets=["file"],
    )
    assert measured >= len("parent-handoff-and-comments") + len("system-prompt") + 1234


def test_workspace_execution_path_runs_real_edit_and_test(tmp_path):
    from hermes_cli import backend_failover_worker as worker

    result = worker._run_child(
        [
            "bash",
            "-lc",
            "printf 'ok\\n' > artifact.txt && test \"$(cat artifact.txt)\" = ok",
        ],
        cwd=str(tmp_path),
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert (tmp_path / "artifact.txt").read_text(encoding="utf-8") == "ok\n"


@pytest.mark.parametrize(
    ("profile", "provider", "model"),
    [
        ("talos", "anthropic", "claude-sonnet-5-5"),
        ("talos", "openai", "gpt-6.1-sol"),
        ("talos", "openai-codex", "gpt-6-luna"),
        ("sentinel", "deepseek", "deepseek-flash"),
    ],
)
def test_read_only_smoke_rejects_noncanonical_route(profile, provider, model):
    from hermes_cli.backend_smoke import build_smoke_command

    with pytest.raises(ValueError, match="canonical backend route"):
        build_smoke_command(profile=profile, provider=provider, model=model)


def test_read_only_smoke_requires_exact_structured_claude_result():
    from hermes_cli.backend_smoke import smoke_succeeded

    exact = json.dumps({"type": "result", "result": "BACKEND_SMOKE_OK"})
    echoed = json.dumps({
        "type": "result",
        "result": "I was asked to say BACKEND_SMOKE_OK",
    })
    tool_use = "\n".join((
        json.dumps({"type": "assistant", "message": {"content": [{"type": "tool_use"}]}}),
        exact,
    ))

    assert smoke_succeeded("claude-code-cli", exact, "")
    assert not smoke_succeeded("claude-code-cli", echoed, "")
    assert not smoke_succeeded("claude-code-cli", tool_use, "")
    assert not smoke_succeeded("claude-code-cli", exact, "warning")


def test_read_only_smoke_requires_exact_api_stdout_and_empty_stderr():
    from hermes_cli.backend_smoke import smoke_succeeded

    successful_cli_output = "\n".join((
        "Query: Reply with exactly BACKEND_SMOKE_OK and no other text.",
        "╭─ Hermes ─╮",
        "    BACKEND_SMOKE_OK",
        "╰──────────╯",
        "Messages: 2 (1 user, 0 tool calls)",
    ))
    assert smoke_succeeded("deepseek", successful_cli_output, "")
    assert not smoke_succeeded(
        "deepseek",
        "Query: Reply with exactly BACKEND_SMOKE_OK and no other text.\n"
        "Messages: 2 (1 user, 0 tool calls)\n",
        "",
    )
    assert not smoke_succeeded(
        "deepseek",
        successful_cli_output.replace("    BACKEND_SMOKE_OK", "prefix BACKEND_SMOKE_OK suffix"),
        "",
    )
    assert not smoke_succeeded("deepseek", successful_cli_output, "error")
    assert not smoke_succeeded(
        "deepseek",
        successful_cli_output.replace("0 tool calls", "1 tool call"),
        "",
    )


def test_claude_balanced_reasoning_maps_to_supported_medium_effort(tmp_path):
    from hermes_cli import backend_failover_worker as worker
    from hermes_cli.backend_failover import BackendLeg

    args = argparse.Namespace(profile="apollo", task="t_probe", timeout_seconds=600)
    cmd, _env = worker._leg_command(
        BackendLeg(
            "claude_code",
            "claude-opus-5-5",
            reasoning_effort="balanced",
            capability_mode="read_only",
        ),
        args,
        {},
        tmp_path / "mcp.json",
        tmp_path / "system.md",
    )
    assert cmd[cmd.index("--effort") + 1] == "medium"


def test_claude_availability_checks_binary_and_subscription_auth(monkeypatch, tmp_path):
    from hermes_cli import backend_failover_worker as worker
    from hermes_constants import get_default_hermes_root

    token = get_default_hermes_root() / "claude" / "subscription.token"
    token.parent.mkdir(parents=True, exist_ok=True)
    token.write_text("sk-ant-oat01-test-only")
    token.chmod(0o600)

    monkeypatch.setattr(worker.shutil, "which", lambda _name: "/bin/claude")
    def fake_run(command, *_args, **_kwargs):
        if command[-1] == "--version":
            return SimpleNamespace(returncode=0, stdout="2.1.304 (Claude Code)", stderr="")
        return SimpleNamespace(returncode=0, stdout='{"loggedIn": true}', stderr="")

    monkeypatch.setattr(worker.subprocess, "run", fake_run)
    assert worker._leg_available(
        _leg(kind="claude_code", provider=None, model="claude-opus-5-5")
    )


def test_claude_alias_is_ineligible_on_older_cli(monkeypatch):
    from hermes_cli import backend_failover_worker as worker

    monkeypatch.setattr(worker.shutil, "which", lambda _name: "/bin/claude")
    monkeypatch.setattr(
        worker.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="2.1.226 (Claude Code)", stderr=""
        ),
    )
    assert not worker._leg_available(
        _leg(kind="claude_code", provider=None, model="claude-sonnet-5-5")
    )


def test_unavailable_leg_advances_without_starting_child(monkeypatch):
    from hermes_cli import backend_failover_worker as worker

    monkeypatch.setattr(worker, "_leg_available", lambda _leg: False)
    monkeypatch.setattr(
        worker.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not spawn")),
    )
    args = argparse.Namespace(
        task="t1", profile="talos", timeout_seconds=30, skill=[]
    )
    assert worker._run_leg(_leg(), 0, args, None, None) == 75


def test_child_timeout_terminates_process_tree():
    from hermes_cli import backend_failover_worker as worker

    with pytest.raises(subprocess.TimeoutExpired):
        worker._run_child(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=None,
            env={},
            capture_output=True,
            timeout=0.01,
        )


def test_child_stops_when_dispatcher_ownership_is_lost():
    from hermes_cli import backend_failover_worker as worker

    with pytest.raises(worker._OwnershipLost):
        worker._run_child(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=None,
            env={},
            capture_output=True,
            timeout=10,
            ownership_check=lambda: False,
        )


def test_attempt_ledger_is_exact_run_bound_and_atomic(tmp_path, monkeypatch):
    from hermes_cli import backend_failover_worker as worker

    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "42")
    path = worker._ledger_path("t_probe")
    assert path is not None
    payload = {
        "task_id": "t_probe",
        "run_id": "42",
        "attempts": [{"outcome": "started"}],
    }
    worker._write_ledger(path, payload)
    assert worker._load_ledger(path, "t_probe") == payload
    assert path.stat().st_mode & 0o777 == 0o600


def test_mcp_config_uses_exact_kanban_environment_allowlist(tmp_path, monkeypatch):
    from hermes_cli import backend_failover_worker as worker

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_probe")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "42")
    monkeypatch.setenv("HERMES_KANBAN_SECRET_TOKEN", "must-not-leak")
    path = worker._mcp_config(tmp_path, "t_probe", "talos")
    try:
        env = json.loads(path.read_text(encoding="utf-8"))["mcpServers"]["hermes-tools"]["env"]
        assert env["HERMES_KANBAN_TASK"] == "t_probe"
        assert env["HERMES_KANBAN_RUN_ID"] == "42"
        assert "HERMES_KANBAN_SECRET_TOKEN" not in env
    finally:
        path.unlink(missing_ok=True)


def test_claude_transient_after_tool_use_requires_reconciliation():
    from hermes_cli import backend_failover_worker as worker

    transcript = '{"type":"tool_use","name":"Bash"}\nrate limit exceeded'
    assert worker._classify_claude_failure(1, transcript, "") == 78


def test_unstructured_claude_failure_never_replays():
    from hermes_cli import backend_failover_worker as worker
    from hermes_cli.backend_failover import EXIT_AMBIGUOUS

    assert worker._classify_claude_failure(1, "", "429 rate limit") == EXIT_AMBIGUOUS
    assert worker._classify_claude_failure(-9, "", "") == EXIT_AMBIGUOUS


def test_structured_claude_availability_failure_is_eligible_before_tool_use():
    from hermes_cli import backend_failover_worker as worker
    from hermes_cli.backend_failover import EXIT_ELIGIBLE

    transcript = json.dumps({
        "type": "result",
        "is_error": True,
        "terminal_reason": "api_error",
        "api_error_status": 429,
        "safety_stops": 0,
    })
    assert worker._classify_claude_failure(1, transcript, "") == EXIT_ELIGIBLE


def test_claude_partial_output_and_refusal_dominate_trailing_availability_error():
    from hermes_cli import backend_failover_worker as worker
    from hermes_cli.backend_failover import EXIT_AMBIGUOUS, EXIT_REFUSAL

    partial_then_429 = "\n".join((
        json.dumps({
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "partial answer"}]},
        }),
        json.dumps({"type": "result", "is_error": True, "error": "rate_limit_error"}),
    ))
    assert worker._classify_claude_failure(1, partial_then_429, "") == EXIT_AMBIGUOUS

    trailing_429 = json.dumps({
        "type": "result",
        "is_error": True,
        "error": "rate_limit_error",
    })
    assert worker._classify_claude_failure(
        1,
        trailing_429,
        "Safety refusal: cannot assist with that request",
    ) == EXIT_REFUSAL


def test_claude_malformed_or_unknown_stream_fails_closed_before_trailing_error():
    from hermes_cli import backend_failover_worker as worker
    from hermes_cli.backend_failover import EXIT_AMBIGUOUS

    trailing_429 = json.dumps({
        "type": "result",
        "is_error": True,
        "error": "rate_limit_error",
    })
    assert worker._classify_claude_failure(
        1,
        f"not-json\n{trailing_429}",
        "",
    ) == EXIT_AMBIGUOUS
    assert worker._classify_claude_failure(
        1,
        f'{json.dumps({"type": "future_event", "payload": "unknown"})}\n{trailing_429}',
        "",
    ) == EXIT_AMBIGUOUS


def test_worker_stops_without_replay_when_child_reports_ambiguous(monkeypatch):
    from hermes_cli import backend_failover_worker as worker
    from hermes_cli.backend_failover import EXIT_AMBIGUOUS

    legs = (_leg(model="one"), _leg(model="two"), _leg(model="three", provider="deepseek"))
    attempted = []
    blocked = []
    monkeypatch.setattr(worker, "_load_chain", lambda _profile: legs)
    monkeypatch.setattr(worker, "_run_leg", lambda leg, *_args, **_kwargs: attempted.append(leg.model) or EXIT_AMBIGUOUS)
    monkeypatch.setattr(worker, "_task_status", lambda _task: "running")
    monkeypatch.setattr(worker, "_block_task", lambda task, reason: blocked.append((task, reason)))
    monkeypatch.setattr(worker, "_validate_dispatch_context", lambda _args: Path("/tmp"))

    rc = worker.run(["--profile", "talos", "--task", "t_probe", "--timeout-seconds", "60"])
    assert rc == EXIT_AMBIGUOUS
    assert attempted == ["one"]
    assert blocked and "reconciliation" in blocked[0][1].lower()


def test_worker_exhaustion_blocks_with_all_three_attempts(monkeypatch):
    from hermes_cli import backend_failover_worker as worker
    from hermes_cli.backend_failover import EXIT_ELIGIBLE

    legs = (_leg(model="one"), _leg(model="two"), _leg(model="three", provider="deepseek"))
    attempted = []
    blocked = []
    monkeypatch.setattr(worker, "_load_chain", lambda _profile: legs)
    monkeypatch.setattr(worker, "_run_leg", lambda leg, *_args, **_kwargs: attempted.append(leg.model) or EXIT_ELIGIBLE)
    monkeypatch.setattr(worker, "_task_status", lambda _task: "running")
    monkeypatch.setattr(worker, "_comment", lambda *_args: True)
    monkeypatch.setattr(worker, "_block_task", lambda task, reason: blocked.append((task, reason)))
    monkeypatch.setattr(worker, "_validate_dispatch_context", lambda _args: Path("/tmp"))

    rc = worker.run(["--profile", "talos", "--task", "t_probe", "--timeout-seconds", "60"])
    assert rc == EXIT_ELIGIBLE
    assert attempted == ["one", "two", "three"]
    assert blocked and "backend exhaustion" in blocked[0][1].lower()


@pytest.mark.parametrize(
    ("models", "results", "expected"),
    [
        (("claude", "codex", "flash"), (75, 0), ["claude", "codex"]),
        (("codex", "claude", "flash"), (75, 75, 0), ["codex", "claude", "flash"]),
        (("codex", "claude", "flash"), (75, 75, 75), ["codex", "claude", "flash"]),
    ],
)
def test_ordered_eligible_failover_scenarios(models, results, expected):
    from hermes_cli.backend_failover import BackendLeg, supervise_chain

    legs = (
        BackendLeg("claude_code", models[0])
        if models[0] == "claude"
        else BackendLeg("api", models[0], provider="openai-codex"),
        BackendLeg("api", models[1], provider="openai-codex")
        if models[1] == "codex"
        else BackendLeg("claude_code", models[1]),
        BackendLeg("api", models[2], provider="deepseek"),
    )
    attempted = []
    remaining = iter(results)

    outcome = supervise_chain(
        legs,
        run_leg=lambda leg, _index: attempted.append(leg.model) or next(remaining),
    )
    assert attempted == expected
    assert outcome.exhausted is (results == (75, 75, 75))


def test_worker_refusal_blocks_without_restart_or_fallback(monkeypatch):
    from hermes_cli import backend_failover_worker as worker
    from hermes_cli.backend_failover import EXIT_REFUSAL

    legs = (
        _leg(model="one"),
        _leg(model="two"),
        _leg(model="three", provider="deepseek"),
    )
    attempted = []
    blocked = []
    monkeypatch.setattr(worker, "_load_chain", lambda _profile: legs)
    monkeypatch.setattr(
        worker,
        "_run_leg",
        lambda leg, *_args, **_kwargs: attempted.append(leg.model) or EXIT_REFUSAL,
    )
    monkeypatch.setattr(worker, "_task_status", lambda _task: "running")
    monkeypatch.setattr(
        worker, "_block_task", lambda task, reason: blocked.append((task, reason))
    )
    monkeypatch.setattr(worker, "_validate_dispatch_context", lambda _args: Path("/tmp"))

    rc = worker.run(
        ["--profile", "talos", "--task", "t_probe", "--timeout-seconds", "60"]
    )
    assert rc == EXIT_REFUSAL
    assert attempted == ["one"]
    assert blocked and "refusal" in blocked[0][1].lower()


def test_dispatcher_selects_supervisor_only_for_enabled_role(monkeypatch):
    from hermes_cli import kanban_db as kb

    task = SimpleNamespace(id="t_probe", max_runtime_seconds=60)
    monkeypatch.setattr(
        "hermes_cli.backend_failover.load_profile_chain",
        lambda profile, home: (_leg(), _leg(kind="claude_code", provider=None, model="sonnet"), _leg(provider="deepseek", model="deepseek-flash"))
        if profile == "talos"
        else (),
    )
    monkeypatch.setattr(kb, "_resolve_worker_cli_toolsets", lambda _home: [])
    monkeypatch.setattr(
        kb,
        "_native_image_attachment_evidence",
        lambda *_args, **_kwargs: ([], 0),
    )
    assert kb._backend_failover_worker_argv(
        task, "talos", "/dispatcher", "/profiles/talos"
    ) is not None
    assert kb._backend_failover_worker_argv(
        task, "athena", "/dispatcher", "/profiles/athena"
    ) is None
