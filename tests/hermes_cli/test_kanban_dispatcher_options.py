from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import ANY, MagicMock

import pytest


def test_resolve_dispatcher_options_applies_config_and_cli_precedence():
    from hermes_cli.kanban_dispatcher import resolve_dispatcher_options

    options = resolve_dispatcher_options(
        {
            "kanban": {
                "dispatch_in_gateway": False,
                "dispatch_interval_seconds": 45,
                "max_spawn": 6,
                "max_in_progress": 4,
                "max_in_progress_per_profile": 3,
                "default_assignee": " talos ",
                "failure_limit": 5,
                "dispatch_stale_timeout_seconds": 900,
            }
        },
        interval_override=10,
        max_spawn_override=2,
        failure_limit_override=7,
    )

    assert options.dispatch_in_gateway is False
    assert options.interval == 10
    assert options.max_spawn == 2
    assert options.max_in_progress == 4
    assert options.max_in_progress_per_profile == 3
    assert options.default_assignee == "talos"
    assert options.failure_limit == 7
    assert options.stale_timeout_seconds == 900


@pytest.mark.parametrize("value", [None, 0, -1, "bad", "1.5", True])
def test_resolve_dispatcher_options_treats_invalid_caps_as_unlimited(value):
    from hermes_cli.kanban_dispatcher import resolve_dispatcher_options

    options = resolve_dispatcher_options(
        {
            "kanban": {
                "max_spawn": value,
                "max_in_progress": value,
                "max_in_progress_per_profile": value,
            }
        }
    )

    assert options.max_spawn is None
    assert options.max_in_progress is None
    assert options.max_in_progress_per_profile is None


def test_resolve_dispatcher_options_uses_safe_defaults_for_invalid_values():
    from hermes_cli.kanban_dispatcher import resolve_dispatcher_options

    options = resolve_dispatcher_options(
        {
            "kanban": {
                "dispatch_interval_seconds": "bad",
                "failure_limit": 0,
                "dispatch_stale_timeout_seconds": "bad",
            }
        }
    )

    assert options.dispatch_in_gateway is True
    assert options.interval == 60
    assert options.failure_limit == 2
    assert options.stale_timeout_seconds == 0


def test_resolve_dispatcher_options_honors_existing_gateway_disable_override(
    monkeypatch,
):
    from hermes_cli.kanban_dispatcher import resolve_dispatcher_options

    monkeypatch.setenv("HERMES_KANBAN_DISPATCH_IN_GATEWAY", "0")

    options = resolve_dispatcher_options(
        {"kanban": {"dispatch_in_gateway": True}}
    )

    assert options.dispatch_in_gateway is False


def _daemon_args(**overrides):
    values = {
        "interval": None,
        "max": None,
        "failure_limit": None,
        "pidfile": None,
        "verbose": False,
        "force": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_standalone_daemon_runs_without_force_when_embedded_dispatch_is_disabled(
    monkeypatch,
):
    from hermes_cli import kanban as kanban_cli
    from hermes_cli import kanban_db
    from hermes_cli import kanban_dispatcher

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "kanban": {
                "dispatch_in_gateway": False,
                "dispatch_interval_seconds": 30,
                "max_spawn": 4,
                "max_in_progress": 2,
                "max_in_progress_per_profile": 2,
                "default_assignee": "talos",
                "failure_limit": 5,
                "dispatch_stale_timeout_seconds": 600,
            }
        },
    )
    monkeypatch.setattr(kanban_db, "init_db", MagicMock())
    run_daemon = MagicMock()
    monkeypatch.setattr(kanban_db, "run_daemon", run_daemon)
    lock_handle = MagicMock()
    monkeypatch.setattr(
        kanban_dispatcher,
        "acquire_dispatcher_lock",
        lambda: (lock_handle, "held"),
    )
    release_lock = MagicMock()
    monkeypatch.setattr(kanban_dispatcher, "release_dispatcher_lock", release_lock)

    assert kanban_cli._cmd_daemon(_daemon_args()) == 0

    run_daemon.assert_called_once_with(
        interval=30,
        max_spawn=4,
        max_in_progress=2,
        max_in_progress_per_profile=2,
        default_assignee="talos",
        failure_limit=5,
        stale_timeout_seconds=600,
        on_tick=ANY,
    )
    release_lock.assert_called_once_with(lock_handle)


def test_standalone_daemon_refuses_when_embedded_dispatch_is_enabled(
    monkeypatch, capsys
):
    from hermes_cli import kanban as kanban_cli
    from hermes_cli import kanban_db

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": {"dispatch_in_gateway": True}},
    )
    run_daemon = MagicMock()
    monkeypatch.setattr(kanban_db, "run_daemon", run_daemon)

    assert kanban_cli._cmd_daemon(_daemon_args()) == 2

    assert "duplicate dispatcher" in capsys.readouterr().err.lower()
    run_daemon.assert_not_called()


def test_standalone_daemon_cli_overrides_do_not_drop_global_cap(monkeypatch):
    from hermes_cli import kanban as kanban_cli
    from hermes_cli import kanban_db
    from hermes_cli import kanban_dispatcher

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "kanban": {
                "dispatch_in_gateway": False,
                "max_spawn": 8,
                "max_in_progress": 2,
                "failure_limit": 5,
            }
        },
    )
    monkeypatch.setattr(kanban_db, "init_db", MagicMock())
    run_daemon = MagicMock()
    monkeypatch.setattr(kanban_db, "run_daemon", run_daemon)
    monkeypatch.setattr(
        kanban_dispatcher,
        "acquire_dispatcher_lock",
        lambda: (MagicMock(), "held"),
    )
    monkeypatch.setattr(kanban_dispatcher, "release_dispatcher_lock", MagicMock())

    assert kanban_cli._cmd_daemon(
        _daemon_args(interval=5, max=7, failure_limit=3)
    ) == 0

    kwargs = run_daemon.call_args.kwargs
    assert kwargs["interval"] == 5
    assert kwargs["max_spawn"] == 7
    assert kwargs["max_in_progress"] == 2
    assert kwargs["failure_limit"] == 3


def test_standalone_daemon_refuses_second_safe_instance(monkeypatch, capsys):
    from hermes_cli import kanban as kanban_cli
    from hermes_cli import kanban_db
    from hermes_cli import kanban_dispatcher

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": {"dispatch_in_gateway": False}},
    )
    monkeypatch.setattr(kanban_db, "init_db", MagicMock())
    run_daemon = MagicMock()
    monkeypatch.setattr(kanban_db, "run_daemon", run_daemon)
    monkeypatch.setattr(
        kanban_dispatcher,
        "acquire_dispatcher_lock",
        lambda: (None, "contended"),
    )

    assert kanban_cli._cmd_daemon(_daemon_args()) == 2

    assert "already running" in capsys.readouterr().err.lower()
    run_daemon.assert_not_called()


def test_standalone_daemon_acquires_singleton_before_database_initialization(
    monkeypatch,
):
    from hermes_cli import kanban as kanban_cli
    from hermes_cli import kanban_db
    from hermes_cli import kanban_dispatcher

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": {"dispatch_in_gateway": False}},
    )
    order = []
    lock_handle = MagicMock()
    monkeypatch.setattr(
        kanban_dispatcher,
        "acquire_dispatcher_lock",
        lambda: (order.append("lock") or lock_handle, "held"),
    )
    monkeypatch.setattr(
        kanban_db,
        "init_db",
        lambda: order.append("init_db"),
    )
    monkeypatch.setattr(kanban_db, "run_daemon", MagicMock())
    monkeypatch.setattr(kanban_dispatcher, "release_dispatcher_lock", MagicMock())

    assert kanban_cli._cmd_daemon(_daemon_args()) == 0

    assert order == ["lock", "init_db"]


def test_run_daemon_forwards_effective_limits_to_each_tick(monkeypatch):
    from hermes_cli import kanban_db

    conn = MagicMock()
    conn.close = MagicMock()
    monkeypatch.setattr(
        kanban_db,
        "list_boards",
        lambda include_archived=False: [{"slug": "default"}],
    )
    monkeypatch.setattr(kanban_db, "connect", lambda *, board: conn)
    dispatch_once = MagicMock(return_value=kanban_db.DispatchResult())
    monkeypatch.setattr(kanban_db, "dispatch_once", dispatch_once)
    stop_event = MagicMock()
    stop_event.is_set.side_effect = [False, True]

    kanban_db.run_daemon(
        interval=10,
        max_spawn=4,
        max_in_progress=2,
        max_in_progress_per_profile=1,
        default_assignee="talos",
        failure_limit=5,
        stale_timeout_seconds=600,
        stop_event=stop_event,
    )

    dispatch_once.assert_called_once_with(
        conn,
        board="default",
        max_spawn=4,
        max_in_progress=2,
        max_in_progress_per_profile=1,
        default_assignee="talos",
        failure_limit=5,
        stale_timeout_seconds=600,
        external_running_count=0,
        external_per_profile_running={},
        process_capacity_known=True,
    )
    stop_event.wait.assert_called_once_with(timeout=10)


def test_run_daemon_dispatches_every_active_board_with_explicit_scope(monkeypatch):
    from hermes_cli import kanban_db

    monkeypatch.setattr(
        kanban_db,
        "list_boards",
        lambda include_archived=False: [
            {"slug": "default"},
            {"slug": "secondary"},
        ],
    )
    connections = {}

    def connect(*, board):
        conn = MagicMock(name=f"connection-{board}")
        connections[board] = conn
        return conn

    monkeypatch.setattr(kanban_db, "connect", connect)
    dispatch_once = MagicMock(return_value=kanban_db.DispatchResult())
    monkeypatch.setattr(kanban_db, "dispatch_once", dispatch_once)
    stop_event = MagicMock()
    stop_event.is_set.side_effect = [False, True]

    kanban_db.run_daemon(
        interval=10,
        max_spawn=2,
        max_in_progress=2,
        max_in_progress_per_profile=1,
        stop_event=stop_event,
    )

    assert [call.kwargs["board"] for call in dispatch_once.call_args_list] == [
        "default",
        "secondary",
    ]
    assert set(connections) == {"default", "secondary"}
    assert all(conn.close.called for conn in connections.values())


def _create_multiboard_tasks(kb, boards, *, tasks_per_board=2, assignee="alpha"):
    for slug in boards:
        kb.create_board(slug=slug, name=slug.title())
        with kb.connect_closing(board=slug) as conn:
            for index in range(tasks_per_board):
                kb.create_task(conn, title=f"{slug}-{index}", assignee=assignee)


def _running_tasks_across_boards(kb, boards):
    running = []
    for slug in boards:
        with kb.connect_closing(board=slug) as conn:
            running.extend(
                (slug, row["id"], row["assignee"])
                for row in conn.execute(
                    "SELECT id, assignee FROM tasks WHERE status = 'running'"
                ).fetchall()
            )
    return running


def test_run_daemon_enforces_global_capacity_across_active_boards(
    monkeypatch, tmp_path,
):
    from hermes_cli import kanban_db as kb

    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _name: True)
    boards = ("default", "secondary")
    _create_multiboard_tasks(kb, boards)
    monkeypatch.setattr(kb, "_default_spawn", lambda *_args, **_kwargs: 12345)
    stop_event = MagicMock()
    stop_event.is_set.side_effect = [False, True]

    kb.run_daemon(
        interval=0,
        max_spawn=2,
        max_in_progress=2,
        max_in_progress_per_profile=2,
        stop_event=stop_event,
    )

    running = _running_tasks_across_boards(kb, boards)
    assert len(running) == 2
    assert {assignee for _, _, assignee in running} == {"alpha"}
    for slug in boards:
        with kb.connect_closing(board=slug) as conn:
            deferred = conn.execute(
                "SELECT id FROM tasks WHERE status = 'ready'"
            ).fetchall()
            for row in deferred:
                assert not any(
                    event.kind in {"failed", "blocked", "spawn_failed"}
                    for event in kb.list_events(conn, row["id"])
                )


def test_run_daemon_rotates_board_priority_when_capacity_reopens(
    monkeypatch, tmp_path,
):
    from hermes_cli import kanban_db as kb

    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _name: True)
    boards = ("default", "secondary")
    _create_multiboard_tasks(kb, boards)
    spawned_boards = []

    def fake_spawn(_task, _workspace, *, board=None):
        spawned_boards.append(board)
        return 12345

    monkeypatch.setattr(kb, "_default_spawn", fake_spawn)
    stop_event = threading.Event()
    callbacks = 0

    def on_tick(_result):
        nonlocal callbacks
        callbacks += 1
        if callbacks == len(boards):
            for slug, task_id, _assignee in _running_tasks_across_boards(kb, boards):
                with kb.connect_closing(board=slug) as conn:
                    with kb.write_txn(conn):
                        conn.execute(
                            "UPDATE tasks SET status = 'done', claim_lock = NULL "
                            "WHERE id = ?",
                            (task_id,),
                        )
        if callbacks == len(boards) * 2:
            stop_event.set()

    kb.run_daemon(
        interval=0.001,
        max_spawn=1,
        max_in_progress=1,
        max_in_progress_per_profile=1,
        stop_event=stop_event,
        on_tick=on_tick,
    )

    assert spawned_boards == ["default", "secondary"]


def test_run_daemon_enforces_per_profile_capacity_across_active_boards(
    monkeypatch, tmp_path,
):
    from hermes_cli import kanban_db as kb

    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _name: True)
    boards = ("default", "secondary")
    _create_multiboard_tasks(kb, boards)
    monkeypatch.setattr(kb, "_default_spawn", lambda *_args, **_kwargs: 12345)
    stop_event = MagicMock()
    stop_event.is_set.side_effect = [False, True]

    kb.run_daemon(
        interval=0,
        max_spawn=4,
        max_in_progress=4,
        max_in_progress_per_profile=2,
        stop_event=stop_event,
    )

    assert len(_running_tasks_across_boards(kb, boards)) == 2


def test_run_daemon_isolates_board_failure_and_dispatches_later_boards(monkeypatch):
    from hermes_cli import kanban_db

    monkeypatch.setattr(
        kanban_db,
        "list_boards",
        lambda include_archived=False: [
            {"slug": "broken"},
            {"slug": "healthy"},
        ],
    )
    healthy_conn = MagicMock()

    def connect(*, board):
        if board == "broken":
            raise RuntimeError("broken board")
        return healthy_conn

    monkeypatch.setattr(kanban_db, "connect", connect)
    dispatch_once = MagicMock(return_value=kanban_db.DispatchResult())
    monkeypatch.setattr(kanban_db, "dispatch_once", dispatch_once)
    stop_event = MagicMock()
    stop_event.is_set.side_effect = [False, True]

    kanban_db.run_daemon(stop_event=stop_event)

    dispatch_once.assert_called_once_with(
        healthy_conn,
        board="healthy",
        max_spawn=None,
        max_in_progress=None,
        max_in_progress_per_profile=None,
        default_assignee=None,
        failure_limit=2,
        stale_timeout_seconds=0,
        external_running_count=0,
        external_per_profile_running={},
        process_capacity_known=False,
    )


def test_standalone_dispatcher_lock_is_machine_global(monkeypatch, tmp_path):
    from hermes_cli.kanban_dispatcher import (
        acquire_dispatcher_lock,
        release_dispatcher_lock,
    )

    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    first, first_state = acquire_dispatcher_lock()
    second, second_state = acquire_dispatcher_lock()
    try:
        assert first is not None
        assert first_state == "held"
        assert second is None
        assert second_state == "contended"
    finally:
        release_dispatcher_lock(second)
        release_dispatcher_lock(first)


def test_standalone_daemon_sigterm_exits_cleanly_without_terminal_events(tmp_path):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "kanban:\n  dispatch_in_gateway: false\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["HERMES_HOME"] = str(hermes_home)
    env["HERMES_KANBAN_HOME"] = str(hermes_home)
    repo_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (repo_root, env.get("PYTHONPATH", "")) if part
    )
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "kanban",
            "daemon",
            "--interval",
            "60",
        ],
        cwd=str(tmp_path),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while (
            time.monotonic() < deadline
            and not (hermes_home / "kanban.db").exists()
        ):
            if proc.poll() is not None:
                break
            time.sleep(0.05)
        assert proc.poll() is None
        # init_db creates the file before run_daemon installs its signal
        # handlers; allow startup to reach the steady-state wait loop.
        time.sleep(0.5)
        proc.terminate()
        stdout, stderr = proc.communicate(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)

    assert proc.returncode == 0, stderr
    assert "Kanban dispatcher running standalone" in stderr
    assert "(dispatcher stopped)" in stdout

    import sqlite3

    with sqlite3.connect(hermes_home / "kanban.db") as conn:
        terminal_events = conn.execute(
            "SELECT COUNT(*) FROM task_events "
            "WHERE kind IN ('completed', 'blocked', 'failed', 'timed_out', 'stale')"
        ).fetchone()[0]
    assert terminal_events == 0
