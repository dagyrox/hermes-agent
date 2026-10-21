"""Exercise supervisor argv through the native CLI, without provider inference."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("effort", ["high", "HIGH", "none", None])
def test_api_worker_native_entry_preserves_reasoning(tmp_path, monkeypatch, effort):
    from hermes_cli import backend_failover_worker as worker
    from hermes_cli.backend_failover import BackendLeg

    # Use the test interpreter, never a machine-specific /opt/hermes fixture.
    # Stop at cmd_chat: the real entry point still performs profile selection,
    # argv preprocessing, parser construction, parsing and command dispatch.
    entry = tmp_path / "native_entry.py"
    entry.write_text(
        "import json\n"
        "import hermes_cli.env_loader as loader\n"
        "loader.load_hermes_dotenv = lambda **kwargs: None\n"
        "import hermes_cli.main as native\n"
        "native._recover_from_interrupted_install = lambda: None\n"
        "native._prepare_agent_startup = lambda args: None\n"
        "native.cmd_chat = lambda args: print(json.dumps(vars(args), default=str))\n"
        "native.main()\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(worker, "_resolve_hermes_argv", lambda: [sys.executable, str(entry)])
    home = tmp_path / ".hermes"
    profile_home = home / "profiles" / "talos"
    profile_home.mkdir(parents=True)
    config = profile_home / "config.yaml"
    config.write_text("agent:\n  reasoning_effort: low\n", encoding="utf-8")
    args = argparse.Namespace(profile="talos", task="t_parse_only", required_toolset=["file"])
    command, _ = worker._leg_command(
        BackendLeg(kind="api", provider="openai-codex", model="gpt-6.1-sol", reasoning_effort=effort),
        args, {}, None, None,
    )
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path),
        "USERPROFILE": str(tmp_path),
        "HERMES_HOME": str(home),
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
    }
    # Windows needs its platform runtime paths even in a scrubbed environment.
    for key in ("SYSTEMROOT", "TEMP", "TMP"):
        if key in os.environ:
            env[key] = os.environ[key]
    result = subprocess.run(command, env=env, cwd=tmp_path, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    parsed = json.loads(result.stdout.strip().splitlines()[-1])
    assert parsed.get("reasoning_effort") == (effort.lower() if effort else None)
    assert parsed["model"] == "gpt-6.1-sol"
    assert parsed["provider"] == "openai-codex"
    assert parsed["query"] == "work kanban task t_parse_only"
    assert parsed["quiet"] is True
    assert parsed["cli"] is True
    assert parsed["toolsets"] == "file"
    assert config.read_text(encoding="utf-8") == "agent:\n  reasoning_effort: low\n"


def test_chat_rejects_invalid_reasoning_effort():
    from hermes_cli._parser import build_top_level_parser

    parser, _, _ = build_top_level_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["chat", "--reasoning-effort", "turbo"])
    assert exc.value.code == 2


@pytest.mark.parametrize(
    ("effort", "expected"),
    [("high", {"enabled": True, "effort": "high"}),
     ("none", {"enabled": False}),
     (None, {"enabled": True, "effort": "medium"})],
)
def test_cmd_chat_reaches_real_cli_with_ephemeral_reasoning(monkeypatch, effort, expected):
    import cli
    import hermes_cli.main as native
    from hermes_cli._parser import build_top_level_parser

    config = deepcopy(cli.CLI_CONFIG)
    config["model"] = {"default": "gpt-6.1-sol", "provider": "openai-codex"}
    config["agent"]["reasoning_effort"] = "low"
    config["agent"]["reasoning_overrides"] = {"gpt-6.1-sol": "medium"}
    before = deepcopy(config)
    monkeypatch.setattr(cli, "CLI_CONFIG", config)
    monkeypatch.setattr(cli, "get_tool_definitions", lambda *args, **kwargs: [])
    monkeypatch.setattr(native, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(native, "_sync_bundled_skills_for_startup", lambda: None)
    monkeypatch.setattr(native, "_termux_should_prefetch_update_check", lambda: False)
    monkeypatch.setattr(native, "_pin_kanban_board_env", lambda: None)
    captured = []
    monkeypatch.setattr(cli.HermesCLI, "run", lambda self: captured.append(self.reasoning_config))
    parser, _, _ = build_top_level_parser()
    argv = ["--cli", "chat"]
    if effort is not None:
        argv += ["--reasoning-effort", effort]
    native.cmd_chat(parser.parse_args(argv))
    assert captured == [expected]
    assert config == before


def test_tui_does_not_silently_drop_reasoning_override(monkeypatch):
    import hermes_cli.main as native
    from hermes_cli._parser import build_top_level_parser

    parser, _, _ = build_top_level_parser()
    monkeypatch.setattr(native, "_resolve_use_tui", lambda args: True)
    with pytest.raises(ValueError, match="requires --cli"):
        native.cmd_chat(parser.parse_args(["chat", "--reasoning-effort", "high"]))
