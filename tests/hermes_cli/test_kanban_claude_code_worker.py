from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from hermes_cli import claude_code_worker as ccw
from hermes_cli import kanban_db_dispatch as dispatch


def test_lane_options_select_only_explicit_profile(monkeypatch):
    from hermes_cli import config_effective

    monkeypatch.setattr(
        config_effective,
        "load_user_config_effective",
        lambda *_args, **_kwargs: {
            "kanban": {
                "claude_code_worker": {"model": "opus", "read_only": True},
            }
        },
    )

    assert dispatch._claude_code_lane_options("athena", "/profiles/athena") == {
        "model": "opus",
        "read_only": True,
    }
    assert dispatch._claude_code_lane_options("talos", None) is None


def test_worker_argv_uses_real_cli_adapter_without_provider_fallback():
    task = SimpleNamespace(id="t_probe")
    argv = dispatch._claude_code_worker_argv(
        cast(Any, task),
        "athena",
        {"binary": "/opt/claude", "model": "opus", "max_turns": 12, "read_only": True},
    )

    assert argv[:3] == [dispatch.sys.executable, "-m", "hermes_cli.claude_code_worker"]
    assert argv[argv.index("--binary") + 1] == "/opt/claude"
    assert argv[argv.index("--model") + 1] == "opus"
    assert argv[argv.index("--timeout-seconds") + 1] == "3600"
    assert "--read-only" in argv
    assert "anthropic" not in argv
    assert "openai" not in argv


def test_worker_argv_rejects_mutating_lane():
    task = SimpleNamespace(id="t_probe", max_runtime_seconds=60)
    try:
        dispatch._claude_code_worker_argv(cast(Any, task), "athena", {"read_only": False})
    except RuntimeError as exc:
        assert "review-only" in str(exc)
    else:
        raise AssertionError("mutating Claude Code lane must be rejected")


def test_mcp_config_grants_only_callback_scope(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_probe")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "42")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    monkeypatch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", "/fenced")

    path = ccw._mcp_config(tmp_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        server = payload["mcpServers"]["hermes-tools"]
        assert server["command"] == ccw.sys.executable
        assert server["env"]["HERMES_KANBAN_TASK"] == "t_probe"
        assert server["env"]["HERMES_KANBAN_RUN_ID"] == "42"
        assert server["env"]["HERMES_DELEGATED_CHILD_CONTEXT"] == ""
        assert server["env"]["HERMES_MCP_EXPOSED_TOOLS"] == ",".join(ccw.LIFECYCLE_TOOLS)
        assert server["env"]["HERMES_MCP_ASSIGNED_TASK"] == "t_probe"
        assert Path(server["env"]["PYTHONPATH"].split(ccw.os.pathsep)[0]).resolve() == Path(
            ccw.__file__
        ).resolve().parents[1]
        assert path.stat().st_mode & 0o777 == 0o600
    finally:
        path.unlink(missing_ok=True)


def test_read_only_command_denies_mutation_and_keeps_lifecycle_tools(tmp_path):
    args = argparse.Namespace(
        binary="claude",
        model="opus",
        max_turns=20,
        read_only=True,
    )
    cmd = ccw._claude_command(
        args, tmp_path / "mcp.json", tmp_path / "system.md", "review")

    assert cmd[0] == "claude"
    assert "--mcp-config" in cmd
    assert "--strict-mcp-config" in cmd
    assert cmd[cmd.index("--permission-mode") + 1] == "dontAsk"
    assert cmd[cmd.index("--setting-sources") + 1] == ""
    assert cmd[cmd.index("--append-system-prompt-file") + 1].endswith("system.md")
    assert "mcp__hermes-tools__kanban_show" in cmd
    assert "mcp__hermes-tools__kanban_complete" in cmd
    assert "mcp__hermes-tools__kanban_request_changes" in cmd
    allowed = cmd[cmd.index("--allowedTools") + 1 : cmd.index("--disallowedTools")]
    assert "Bash(git *)" not in allowed
    assert "Bash(gh *)" not in allowed
    assert "WebFetch" not in allowed
    assert "Edit" in cmd[cmd.index("--disallowedTools") + 1 :]
    assert "Write" in cmd[cmd.index("--disallowedTools") + 1 :]
    assert "Bash" in cmd[cmd.index("--disallowedTools") + 1 :]


def _prepare_run(monkeypatch, tmp_path, *, returncode: int, status: str, stdout: str = ""):
    profile = tmp_path / "profile"
    workspace = tmp_path / "workspace"
    profile.mkdir()
    workspace.mkdir()
    (profile / "SOUL.md").write_text("You are Athena.", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_probe")
    monkeypatch.setenv("HERMES_PROFILE", "athena")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace))
    monkeypatch.setattr(ccw, "_call_kanban_tool", lambda *_args, **_kwargs: '{"ok": true}')
    monkeypatch.setattr(ccw, "_task_status", lambda _task: status)
    monkeypatch.setattr(ccw, "hermes_subprocess_env", lambda **_kwargs: {})

    class Proc:
        def __init__(self, *_args, **_kwargs):
            self.returncode = returncode
            self.terminated = False
            self.pid = 4242
            self.env = _kwargs.get("env", {})

        def communicate(self, timeout=None):
            return stdout, ""

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(ccw.subprocess, "Popen", Proc)


def test_config_read_failure_does_not_fallback_to_provider(monkeypatch):
    from hermes_cli import config_effective

    def fail(**_kwargs):
        raise OSError("config unreadable")

    monkeypatch.setattr(config_effective, "load_user_config_effective", fail)
    try:
        dispatch._claude_code_lane_options("athena", "/profiles/athena")
    except RuntimeError as exc:
        assert "cannot resolve Claude Code worker lane for athena" in str(exc)
    else:
        raise AssertionError("config failure must fail closed")


def test_process_group_cancellation_escalates(monkeypatch):
    calls = []
    proc = SimpleNamespace(pid=1234, poll=lambda: None)
    monkeypatch.setattr(ccw.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    ccw._terminate_process_group(cast(Any, proc))
    ccw._terminate_process_group(cast(Any, proc), force=True)
    assert calls == [(1234, ccw.signal.SIGTERM), (1234, ccw.signal.SIGKILL)]


def test_tool_failure_parser_fails_closed():
    assert ccw._tool_failed('{"ok": true}') is False
    assert ccw._tool_failed('{"ok": false}') is True
    assert ccw._tool_failed('{"error": "claim lost"}') is True
    assert ccw._tool_failed("unexpected prose") is True


def test_rejected_initial_claim_never_spawns_duplicate_worker(monkeypatch, tmp_path):
    profile = tmp_path / "profile"
    profile.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_probe")
    monkeypatch.setenv("HERMES_PROFILE", "athena")
    monkeypatch.setattr(ccw, "_call_kanban_tool", lambda *_args, **_kwargs: '{"ok": false}')
    monkeypatch.setattr(
        ccw.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not spawn")),
    )
    try:
        ccw.run(["--profile", "athena", "--task", "t_probe", "--read-only"])
    except RuntimeError as exc:
        assert "does not own the claimed Kanban run" in str(exc)
    else:
        raise AssertionError("lost claim must fail before spawning Claude")


def test_wall_clock_timeout_terminates_process_group(monkeypatch, tmp_path):
    profile = tmp_path / "profile"
    workspace = tmp_path / "workspace"
    profile.mkdir()
    workspace.mkdir()
    (profile / "SOUL.md").write_text("You are Athena.", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_probe")
    monkeypatch.setenv("HERMES_PROFILE", "athena")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace))
    calls = []
    monkeypatch.setattr(ccw, "_call_kanban_tool", lambda name, args: calls.append((name, args)) or '{"ok": true}')
    monkeypatch.setattr(ccw, "_task_status", lambda _task: "running")
    monkeypatch.setattr(ccw, "hermes_subprocess_env", lambda **_kwargs: {})
    monkeypatch.setattr(ccw, "_terminate_with_grace", lambda proc: calls.append(("terminate", proc.pid)))

    class TimedOutProc:
        pid = 9001
        returncode = -9
        attempts = 0

        def communicate(self, timeout=None):
            self.attempts += 1
            if self.attempts == 1:
                raise ccw.subprocess.TimeoutExpired("claude", float(timeout or 1))
            return "", ""

        def poll(self):
            return None

    monkeypatch.setattr(ccw.subprocess, "Popen", lambda *_args, **_kwargs: TimedOutProc())
    rc = ccw.run([
        "--profile", "athena", "--task", "t_probe", "--read-only", "--timeout-seconds", "1",
    ])
    assert rc == ccw.TIMEOUT_EXIT_CODE
    assert ("terminate", 9001) in calls
    assert any(name == "kanban_comment" and "wall-clock" in args["body"] for name, args in calls if name != "terminate")


def test_cli_failure_stays_failure_without_fallback(monkeypatch, tmp_path):
    _prepare_run(monkeypatch, tmp_path, returncode=29, status="running")
    assert ccw.run(["--profile", "athena", "--task", "t_probe", "--read-only"]) == 69


def test_success_without_lifecycle_is_protocol_violation(monkeypatch, tmp_path):
    _prepare_run(monkeypatch, tmp_path, returncode=0, status="running")
    assert ccw.run(["--profile", "athena", "--task", "t_probe", "--read-only"]) == ccw.PROTOCOL_VIOLATION_EXIT_CODE


def test_terminal_handoff_wins_and_output_is_redacted(monkeypatch, tmp_path, capsys):
    _prepare_run(
        monkeypatch,
        tmp_path,
        returncode=0,
        status="done",
        stdout=json.dumps({
            "subtype": "success",
            "terminal_reason": "completed",
            "result": "token=super-secret-value",
            "modelUsage": {"claude-opus-5": {}},
        }),
    )
    assert ccw.run(["--profile", "athena", "--task", "t_probe", "--read-only"]) == 0
    captured = capsys.readouterr()
    assert "super-secret-value" not in captured.out
    assert "claude-opus-5" in captured.out
    assert '"result"' not in captured.out
