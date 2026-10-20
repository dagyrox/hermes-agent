from __future__ import annotations

import subprocess

import pytest


def _make_task(kb, *, assignee: str):
    return kb.Task(
        id="t_spawn_tools",
        title="spawn tools",
        body=None,
        assignee=assignee,
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind="dir",
        workspace_path=None,
        claim_lock="lock",
        claim_expires=None,
        tenant=None,
        current_run_id=7,
    )


def test_default_spawn_pins_assignee_profile_cli_toolsets(monkeypatch, tmp_path):
    """Manual profile assignment should keep that profile's CLI tools.

    Regression guard for dispatcher-spawned workers that boot with
    HERMES_KANBAN_TASK: the worker must not collapse to only kanban lifecycle
    tools when the assigned profile's top-level ``toolsets`` is the default
    composite. The spawned CLI gets an explicit --toolsets pin resolved from
    platform_toolsets.cli; model_tools appends task-scoped kanban tools later.
    """
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "elias"
    profile.mkdir(parents=True)
    profile.joinpath("config.yaml").write_text(
        """
platform_toolsets:
  cli:
    - clarify
    - code_execution
    - delegation
    - file
    - memory
    - session_search
    - skills
    - terminal
    - web
toolsets:
  - hermes-cli
agent:
  disabled_toolsets: []
""".lstrip(),
        encoding="utf-8",
    )
    root.joinpath("config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb

    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])

    captured = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env") or {})
        captured["cwd"] = kwargs.get("cwd")
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    pid = kb._default_spawn(_make_task(kb, assignee="elias"), str(workspace))

    assert pid == 4242
    assert captured["env"]["HERMES_HOME"] == str(profile)
    assert captured["env"]["HERMES_KANBAN_TASK"] == "t_spawn_tools"
    assert "--toolsets" in captured["cmd"]
    pinned = captured["cmd"][captured["cmd"].index("--toolsets") + 1].split(",")
    for required in ("terminal", "web", "file", "skills", "code_execution", "delegation"):
        assert required in pinned


def test_default_spawn_never_boots_the_tui(monkeypatch, tmp_path):
    """Workers are headless: an inherited HERMES_TUI=1 (or a TUI-default
    config) must not send the quiet chat run into the Ink TUI, whose no-TTY
    bail-out exits 0 without doing the task — every attempt then ends in
    "protocol violation". The spawn pins --cli (highest-precedence interface
    flag) and strips HERMES_TUI from the child env."""
    root = tmp_path / ".hermes"
    (root / "profiles" / "elias").mkdir(parents=True)
    root.joinpath("config.yaml").write_text("display:\n  interface: tui\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_TUI", "1")

    from hermes_cli import kanban_db as kb

    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])

    captured = {}

    class FakeProc:
        pid = 4243

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env") or {})
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    kb._default_spawn(_make_task(kb, assignee="elias"), str(workspace))

    assert "--cli" in captured["cmd"]
    assert "HERMES_TUI" not in captured["env"]


def test_default_spawn_model_override_survives_real_cli_parse(monkeypatch, tmp_path):
    """The dispatcher's pre-``chat`` model flag must reach ``args.model``.

    This is an integration contract between Kanban's worker argv builder and
    the real CLI parser. A parser default once erased the explicit override,
    silently sending the worker to its profile default or fallback instead.
    """
    root = tmp_path / ".hermes"
    (root / "profiles" / "elias").mkdir(parents=True)
    root.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb
    from hermes_cli._parser import build_top_level_parser

    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    captured = {}

    class FakeProc:
        pid = 4244

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task = _make_task(kb, assignee="elias")
    task.model_override = "gpt-5.6-sol"
    kb._default_spawn(task, str(workspace))

    parser, _subparsers, _chat_parser = build_top_level_parser()
    # Profile selection is attached by the outer CLI bootstrap rather than
    # build_top_level_parser(); remove that already-validated prefix and parse
    # the worker flags/subcommand through the real shared parser.
    assert captured["cmd"][1:3] == ["-p", "elias"]
    args = parser.parse_args(captured["cmd"][3:])

    assert args.command == "chat"
    assert args.model == "gpt-5.6-sol"
    assert args.query == "work kanban task t_spawn_tools"


def test_resolve_worker_cli_toolsets_uses_profile_home_not_parent_config(monkeypatch, tmp_path):
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "elias"
    profile.mkdir(parents=True)
    root.joinpath("config.yaml").write_text("platform_toolsets:\n  cli:\n    - kanban\n", encoding="utf-8")
    profile.joinpath("config.yaml").write_text(
        """
platform_toolsets:
  cli:
    - terminal
    - web
toolsets:
  - hermes-cli
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb

    resolved = kb._resolve_worker_cli_toolsets(str(profile))

    assert resolved is not None
    assert "terminal" in resolved
    assert "web" in resolved
    assert "kanban" in resolved  # recovered worker lifecycle surface
    assert resolved != ["kanban"]


def test_resolve_worker_cli_toolsets_distinguishes_valid_empty_from_error(
    monkeypatch, tmp_path
):
    profile = tmp_path / "profile"
    profile.mkdir()
    profile.joinpath("config.yaml").write_text(
        "platform_toolsets:\n  cli: []\n",
        encoding="utf-8",
    )

    from hermes_cli import kanban_db as kb

    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_args: set())
    assert kb._resolve_worker_cli_toolsets(str(profile)) == []
    assert kb._resolve_worker_cli_toolsets(None) is None


def test_supervised_spawn_uses_dispatcher_routes_and_assignee_tools(monkeypatch, tmp_path):
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "elias"
    profile.mkdir(parents=True)
    (profile / "SOUL.md").write_text("Elias", encoding="utf-8")
    root.joinpath("config.yaml").write_text("agent_backend_failover: {}\n", encoding="utf-8")
    profile.joinpath("config.yaml").write_text(
        "platform_toolsets:\n  cli:\n    - terminal\n    - file\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb
    from hermes_cli.backend_failover import BackendLeg

    seen = {}
    chain = (
        BackendLeg("api", "gpt-6.1-sol", "openai-codex"),
        BackendLeg("claude_code", "claude-sonnet-5-5"),
        BackendLeg("api", "deepseek-flash", "deepseek"),
    )

    def fake_load(profile_name, home):
        seen["route_home"] = home
        return chain

    monkeypatch.setattr("hermes_cli.backend_failover.load_profile_chain", fake_load)
    monkeypatch.setattr(kb, "_native_image_attachment_evidence", lambda *_args, **_kwargs: ([], 0))
    argv = kb._backend_failover_worker_argv(
        _make_task(kb, assignee="elias"),
        "elias",
        str(root),
        str(profile),
    )

    assert seen["route_home"] == str(root)
    assert argv is not None
    values = [argv[i + 1] for i, value in enumerate(argv[:-1]) if value == "--required-toolset"]
    assert "terminal" in values
    assert "file" in values


def test_supervised_spawn_preserves_empty_valid_toolset_evidence(monkeypatch, tmp_path):
    from hermes_cli import backend_failover_worker as worker
    from hermes_cli import kanban_db as kb
    from hermes_cli.backend_failover import BackendLeg

    profile = tmp_path / "profile"
    profile.mkdir()
    chain = (
        BackendLeg("api", "gpt-6.1-sol", "openai-codex"),
        BackendLeg("claude_code", "claude-sonnet-5-5"),
        BackendLeg("api", "deepseek-flash", "deepseek"),
    )
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_spawn_tools")
    monkeypatch.setattr("hermes_cli.backend_failover.load_profile_chain", lambda *_args: chain)
    monkeypatch.setattr(kb, "_resolve_worker_cli_toolsets", lambda _home: [])
    monkeypatch.setattr(kb, "_native_image_attachment_evidence", lambda *_args, **_kwargs: ([], 0))

    argv = kb._backend_failover_worker_argv(
        _make_task(kb, assignee="elias"), "elias", str(tmp_path), str(profile)
    )
    assert argv is not None
    args = worker._parse_args(argv[3:])
    assert args.required_toolset == []
    names, schema_chars = worker._resolved_required_tool_schemas(args.required_toolset)
    assert names is not None and "kanban_show" in names
    assert schema_chars > 0
    api_cmd, _env = worker._leg_command(chain[0], args, {}, None, None)
    assert api_cmd[api_cmd.index("--toolsets") + 1] == worker._EMPTY_TOOLSET_PIN


def test_supervised_spawn_fails_closed_when_toolset_resolution_fails(monkeypatch, tmp_path):
    from hermes_cli import kanban_db as kb
    from hermes_cli.backend_failover import BackendLeg

    profile = tmp_path / "profile"
    profile.mkdir()
    chain = (
        BackendLeg("api", "gpt-6.1-sol", "openai-codex"),
        BackendLeg("claude_code", "claude-sonnet-5-5"),
        BackendLeg("api", "deepseek-flash", "deepseek"),
    )
    monkeypatch.setattr("hermes_cli.backend_failover.load_profile_chain", lambda *_args: chain)
    monkeypatch.setattr(kb, "_resolve_worker_cli_toolsets", lambda _home: None)

    with pytest.raises(RuntimeError, match="could not resolve CLI toolsets"):
        kb._backend_failover_worker_argv(
            _make_task(kb, assignee="elias"), "elias", str(tmp_path), str(profile)
        )


def test_supervised_spawn_derives_native_image_and_reaches_parity_gate(
    monkeypatch, tmp_path
):
    from hermes_cli import backend_failover_worker as worker
    from hermes_cli import kanban_db as kb
    from hermes_cli.backend_failover import BackendLeg

    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db(db_path)
    image = tmp_path / "fixture.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    with kb.connect(db_path) as conn:
        task_id = kb.create_task(conn, title="image task", assignee="elias")
        kb.add_attachment(
            conn,
            task_id,
            filename=image.name,
            stored_path=str(image),
            content_type="image/png",
            size=image.stat().st_size,
        )
        task = kb.get_task(conn, task_id)
    assert task is not None

    profile = tmp_path / "profile"
    profile.mkdir()
    chain = (
        BackendLeg("api", "gpt-6.1-sol", "openai-codex"),
        BackendLeg("claude_code", "claude-sonnet-5-5"),
        BackendLeg("api", "deepseek-flash", "deepseek"),
    )
    monkeypatch.setattr("hermes_cli.backend_failover.load_profile_chain", lambda *_args: chain)
    monkeypatch.setattr(kb, "_resolve_worker_cli_toolsets", lambda _home: [])
    argv = kb._backend_failover_worker_argv(
        task, "elias", str(tmp_path), str(profile)
    )
    assert argv is not None
    args = worker._parse_args(argv[3:])
    assert args.requires_native_image is True
    assert args.native_image_bytes == image.stat().st_size
    assert args.native_image_path == [str(image.resolve())]
    assert worker._capability_mismatches(chain[1], args) == ("native_image_input",)

    monkeypatch.setattr(
        "agent.models_dev.get_model_capabilities",
        lambda *_args: __import__("types").SimpleNamespace(
            supports_tools=True,
            supports_vision=True,
            supports_reasoning=True,
            context_window=100_000,
            max_output_tokens=1_000,
            context_window_observed=True,
            max_output_tokens_observed=True,
        ),
    )
    assert worker._capability_mismatches(chain[0], args) == ()
    api_cmd, _env = worker._leg_command(chain[0], args, {}, None, None)
    assert api_cmd[api_cmd.index("--image") + 1] == str(image.resolve())
