"""Worker-only failover acceptance with isolated durable state.

The provider programs below are deterministic synthetic test fixtures. They are
not recordings of, and do not claim execution against, any live provider.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _claimed_board(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from hermes_cli import kanban_db as kb

    root = tmp_path / "hermes-root"
    profile = root / "profiles" / "talos"
    profile.mkdir(parents=True)
    (profile / "SOUL.md").write_text("Talos test identity", encoding="utf-8")
    (profile / "config.yaml").write_text(
        "platform_toolsets:\n  cli: []\n",
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    db_path = tmp_path / "kanban.db"
    kb.init_db(db_path)
    with kb.connect(db_path) as conn:
        task_id = kb.create_task(
            conn,
            title="synthetic worker-only failover",
            assignee="talos",
            workspace_kind="dir",
            workspace_path=str(workspace),
        )
        task = kb.claim_task(conn, task_id, claimer="synthetic-claim")
    assert task is not None and task.current_run_id is not None

    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setenv("HERMES_PROFILE", "talos")
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "synthetic-claim")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
    monkeypatch.setenv("PYTHONPATH", str(REPO_ROOT))
    monkeypatch.setattr(
        "hermes_cli.profiles.resolve_profile_env",
        lambda _profile: str(profile),
    )
    return kb, profile, workspace, db_path, task


def _install_synthetic_transports(
    tmp_path: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Install local subprocess fixtures at the external transport boundary."""

    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    hermes_fixture = fakebin / "synthetic-hermes"
    claude_fixture = fakebin / "claude"

    _write_executable(
        hermes_fixture,
        f"""#!/usr/local/lib/hermes-agent/venv/bin/python
import pathlib, sys
workspace = pathlib.Path.cwd()
provider = sys.argv[sys.argv.index('--provider') + 1]
with (workspace / 'attempts.log').open('a', encoding='utf-8') as handle:
    handle.write(provider + '\\n')
raise SystemExit(75)
""",
    )
    _write_executable(
        claude_fixture,
        f"""#!/usr/local/lib/hermes-agent/venv/bin/python
import json, os, pathlib, sys
if '--version' in sys.argv:
    print('2.1.999')
    raise SystemExit(0)
if len(sys.argv) > 3 and sys.argv[1:4] == ['auth', 'status', '--json']:
    print('{{"loggedIn": true}}')
    raise SystemExit(0)
workspace = pathlib.Path.cwd()
with (workspace / 'attempts.log').open('a', encoding='utf-8') as handle:
    handle.write('claude-code-subscription\\n')
if not (workspace / 'complete-on-claude').exists():
    print(json.dumps({{'type': 'result', 'is_error': True, 'error': 'rate_limit_error'}}))
    raise SystemExit(1)
config_path = pathlib.Path(sys.argv[sys.argv.index('--mcp-config') + 1])
server = json.loads(config_path.read_text(encoding='utf-8'))['mcpServers']['hermes-tools']
os.environ.update(server['env'])
sys.path.insert(0, {str(REPO_ROOT)!r})
artifact = workspace / 'synthetic-artifact.txt'
artifact.write_text('completed exactly once', encoding='utf-8')
from model_tools import handle_function_call
result = handle_function_call('kanban_complete', {{
    'summary': 'synthetic provider fixture completed worker task',
    'metadata': {{'fixture': 'synthetic-local-transport'}},
}})
parsed = json.loads(result)
if parsed.get('error'):
    print(result, file=sys.stderr)
    raise SystemExit(2)
print(json.dumps({{'type': 'result', 'result': 'completed'}}))
""",
    )

    monkeypatch.setenv("PATH", f"{fakebin}:{os.environ.get('PATH', '')}")
    return hermes_fixture


def _talos_chain_json() -> str:
    return json.dumps([
        {
            "kind": "api",
            "provider": "openai-codex",
            "model": "gpt-6.1-sol",
            "reasoning_effort": "high",
        },
        {
            "kind": "claude_code",
            "model": "claude-sonnet-5-5",
            "reasoning_effort": "high",
            "capability_mode": "full",
        },
        {
            "kind": "api",
            "provider": "deepseek",
            "model": "deepseek-flash",
        },
    ])


def _run_synthetic_supervisor(tmp_path, monkeypatch, *, complete_on_claude: bool):
    from hermes_cli import backend_failover_worker as worker
    from hermes_cli.backend_failover import resolve_role_chain

    kb, profile, workspace, db_path, task = _claimed_board(tmp_path, monkeypatch)
    fixture = _install_synthetic_transports(tmp_path, workspace, monkeypatch)
    if complete_on_claude:
        (workspace / "complete-on-claude").touch()

    # These are the documented dependency-injection seams: provider capability,
    # credential/binary discovery, and the external Hermes executable. The
    # dispatcher selection, supervisor, _run_leg, child process boundary,
    # ledger, and board mutations remain production code.
    chain = resolve_role_chain(
        {"enabled": True, "roles": {"talos": json.loads(_talos_chain_json())}},
        "talos",
    )
    monkeypatch.setattr(
        "hermes_cli.backend_failover.load_profile_chain",
        lambda *_args: chain,
    )
    monkeypatch.setattr(kb, "_resolve_worker_cli_toolsets", lambda _home: [])
    monkeypatch.setattr(kb, "_native_image_attachment_evidence", lambda *_args, **_kwargs: ([], 0))
    supervisor_argv = kb._backend_failover_worker_argv(
        task,
        "talos",
        str(tmp_path / "hermes-root"),
        str(profile),
        board="default",
    )
    assert supervisor_argv is not None
    module_index = supervisor_argv.index("hermes_cli.backend_failover_worker")

    monkeypatch.setattr(worker, "_capability_mismatches", lambda *_args: ())
    monkeypatch.setattr(worker, "_leg_available", lambda *_args: True)
    monkeypatch.setattr(worker, "_resolve_hermes_argv", lambda: [str(fixture)])

    rc = worker.run(supervisor_argv[module_index + 1 :])
    with kb.connect(db_path) as conn:
        durable_task = kb.get_task(conn, task.id)
        durable_run = conn.execute(
            "SELECT * FROM task_runs WHERE id=?",
            (task.current_run_id,),
        ).fetchone()
        events = list(conn.execute(
            "SELECT kind, run_id FROM task_events WHERE task_id=? ORDER BY id",
            (task.id,),
        ))
    return rc, workspace, durable_task, durable_run, events


def test_synthetic_primary_failure_falls_back_through_real_subprocess_once(
    tmp_path, monkeypatch
):
    rc, workspace, task, run, events = _run_synthetic_supervisor(
        tmp_path,
        monkeypatch,
        complete_on_claude=True,
    )

    assert rc == 0
    assert task is not None and task.status == "done"
    assert run is not None
    assert run["summary"] == "synthetic provider fixture completed worker task"
    assert (workspace / "synthetic-artifact.txt").read_text(encoding="utf-8") == (
        "completed exactly once"
    )
    assert (workspace / "attempts.log").read_text(encoding="utf-8").splitlines() == [
        "openai-codex",
        "claude-code-subscription",
    ]
    terminal_events = [event for event in events if event["kind"] == "completed"]
    assert len(terminal_events) == 1
    assert terminal_events[0]["run_id"] == run["id"]


def test_synthetic_three_leg_exhaustion_blocks_once(tmp_path, monkeypatch):
    from hermes_cli.backend_failover import EXIT_ELIGIBLE

    rc, workspace, task, _run, events = _run_synthetic_supervisor(
        tmp_path,
        monkeypatch,
        complete_on_claude=False,
    )

    assert rc == EXIT_ELIGIBLE
    assert task is not None and task.status == "blocked"
    assert (workspace / "attempts.log").read_text(encoding="utf-8").splitlines() == [
        "openai-codex",
        "claude-code-subscription",
        "deepseek",
    ]
    assert len([event for event in events if event["kind"] == "blocked"]) == 1
    assert not (workspace / "synthetic-artifact.txt").exists()


def test_restart_with_started_ledger_denies_duplicate_action(tmp_path, monkeypatch):
    from hermes_cli import backend_failover_worker as worker
    from hermes_cli.backend_failover import EXIT_AMBIGUOUS

    kb, profile, workspace, db_path, task = _claimed_board(tmp_path, monkeypatch)
    monkeypatch.setattr(worker, "_capability_mismatches", lambda *_args: ())
    ledger = worker._ledger_path(task.id)
    assert ledger is not None
    worker._write_ledger(ledger, {
        "task_id": task.id,
        "run_id": str(task.current_run_id),
        "attempts": [{"index": 0, "outcome": "started"}],
    })

    rc = worker.run([
        "--profile",
        "talos",
        "--task",
        task.id,
        "--timeout-seconds",
        "30",
        "--chain-json",
        _talos_chain_json(),
    ])
    assert rc == EXIT_AMBIGUOUS
    with kb.connect(db_path) as conn:
        blocked = kb.get_task(conn, task.id)
    assert blocked is not None and blocked.status == "blocked"
    assert not (workspace / "attempts.log").exists()
    assert profile.exists()
