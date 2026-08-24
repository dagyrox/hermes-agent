from __future__ import annotations

import asyncio
import json
import os
import stat
import subprocess
import sys
import threading

import pytest


def test_dispatcher_quiesce_cli_emits_machine_readable_result(monkeypatch, capsys):
    import argparse

    from hermes_cli import kanban as kanban_cli
    from hermes_cli import kanban_dispatcher

    expected = {
        "ok": True,
        "state": "quiesced",
        "owner_pid": 4242,
        "owner_mode": "embedded",
    }
    monkeypatch.setattr(
        kanban_dispatcher,
        "request_dispatcher_quiescence",
        lambda **_kwargs: expected,
    )

    args = argparse.Namespace(pid=4242, timeout=30.0, json=True)
    assert kanban_cli._cmd_dispatcher_quiesce(args) == 0
    assert capsys.readouterr().out.strip() == (
        '{"ok": true, "state": "quiesced", "owner_pid": 4242, '
        '"owner_mode": "embedded"}'
    )


def test_dispatcher_quiesce_bypasses_board_database_initialization(
    monkeypatch, capsys,
):
    import argparse

    import hermes_cli.kanban as kanban_cli

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(
        kanban_cli.kb,
        "init_db",
        lambda: (_ for _ in ()).throw(
            AssertionError("control operation must not initialize the board DB")
        ),
    )
    monkeypatch.setattr(
        "hermes_cli.kanban_dispatcher.request_dispatcher_quiescence",
        lambda **_kwargs: {
            "ok": True,
            "state": "quiesced",
            "owner_pid": 4321,
            "owner_mode": "embedded",
        },
    )
    args = argparse.Namespace(
        kanban_action="dispatcher-quiesce",
        board=None,
        pid=4321,
        timeout=5.0,
        json=True,
    )
    assert kanban_cli.kanban_command(args) == 0
    assert json.loads(capsys.readouterr().out) == {
        "ok": True,
        "state": "quiesced",
        "owner_pid": 4321,
        "owner_mode": "embedded",
    }


def test_dispatcher_quiesce_failure_reaches_process_status(monkeypatch, capsys):
    from hermes_cli import kanban_dispatcher
    from hermes_cli import main as cli_main

    monkeypatch.setattr(
        kanban_dispatcher,
        "request_dispatcher_quiescence",
        lambda **_kwargs: {"ok": False, "state": "not_running"},
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "hermes",
            "kanban",
            "dispatcher-quiesce",
            "--pid",
            "4242",
            "--timeout",
            "1",
            "--json",
        ],
    )
    assert cli_main.main() == 1
    assert capsys.readouterr().out.strip() == (
        '{"ok": false, "state": "not_running"}'
    )


def test_dispatcher_quiesce_module_entrypoint_preserves_failure_status(tmp_path):
    env = dict(os.environ, HERMES_KANBAN_HOME=str(tmp_path))
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "kanban",
            "dispatcher-quiesce",
            "--pid",
            str(os.getpid()),
            "--timeout",
            "1",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 1
    assert json.loads(completed.stdout) == {"ok": False, "state": "not_running"}


def test_dispatcher_quiesce_repo_launcher_preserves_failure_status(tmp_path):
    env = dict(os.environ, HERMES_KANBAN_HOME=str(tmp_path))
    completed = subprocess.run(
        [
            sys.executable,
            "./hermes",
            "kanban",
            "dispatcher-quiesce",
            "--pid",
            str(os.getpid()),
            "--timeout",
            "1",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 1
    assert json.loads(completed.stdout) == {"ok": False, "state": "not_running"}


def test_quiesce_fails_closed_when_dispatcher_is_not_running(monkeypatch, tmp_path):
    from hermes_cli.kanban_dispatcher import request_dispatcher_quiescence

    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    assert request_dispatcher_quiescence(
        expected_pid=os.getpid(), timeout_seconds=0.1, poll_interval=0.01
    ) == {"ok": False, "state": "not_running"}


def test_quiesce_rejects_non_finite_timeout(monkeypatch, tmp_path):
    from hermes_cli.kanban_dispatcher import request_dispatcher_quiescence

    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    assert request_dispatcher_quiescence(
        expected_pid=os.getpid(), timeout_seconds=float("nan")
    ) == {"ok": False, "state": "invalid_request"}


def test_quiesce_rejects_standalone_owner(monkeypatch, tmp_path):
    from hermes_cli.kanban_dispatcher import (
        acquire_dispatcher_lock,
        release_dispatcher_lock,
        request_dispatcher_quiescence,
    )

    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    handle, state = acquire_dispatcher_lock(owner_mode="standalone")
    assert state == "held"
    try:
        assert request_dispatcher_quiescence(
            expected_pid=os.getpid(), timeout_seconds=0.1, poll_interval=0.01
        ) == {
            "ok": False,
            "state": "not_owner",
            "owner_pid": os.getpid(),
            "owner_mode": "standalone",
        }
    finally:
        release_dispatcher_lock(handle)


@pytest.mark.linux_only
def test_dispatcher_control_directory_is_private(monkeypatch, tmp_path):
    from hermes_cli.kanban_dispatcher import (
        acquire_dispatcher_lock,
        release_dispatcher_lock,
    )

    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    control_dir = tmp_path / "kanban" / ".dispatcher-control"
    control_dir.mkdir(parents=True)
    control_dir.chmod(0o777)
    handle, state = acquire_dispatcher_lock(owner_mode="embedded")
    try:
        assert state == "held"
        assert stat.S_IMODE(control_dir.stat().st_mode) == 0o700
    finally:
        release_dispatcher_lock(handle)


def test_standalone_daemon_records_non_embedded_owner(monkeypatch):
    import argparse
    from unittest.mock import MagicMock

    from hermes_cli import kanban as kanban_cli
    from hermes_cli import kanban_db
    from hermes_cli import kanban_dispatcher

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": {"dispatch_in_gateway": False}},
    )
    monkeypatch.setattr(kanban_db, "init_db", MagicMock())
    monkeypatch.setattr(kanban_db, "run_daemon", MagicMock())
    acquire = MagicMock(return_value=(MagicMock(), "held"))
    monkeypatch.setattr(kanban_dispatcher, "acquire_dispatcher_lock", acquire)
    monkeypatch.setattr(
        kanban_dispatcher, "release_dispatcher_lock", MagicMock()
    )

    args = argparse.Namespace(
        interval=None,
        max=None,
        failure_limit=None,
        pidfile=None,
        verbose=False,
        force=False,
    )
    assert kanban_cli._cmd_daemon(args) == 0
    acquire.assert_called_once_with(owner_mode="standalone")


def test_quiesce_timeout_is_fail_closed_and_retry_is_idempotent(
    monkeypatch, tmp_path,
):
    from hermes_cli.kanban_dispatcher import (
        DispatcherTickBoundary,
        acquire_dispatcher_lock,
        read_dispatcher_owner,
        release_dispatcher_lock,
        request_dispatcher_quiescence,
    )

    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    handle, state = acquire_dispatcher_lock(owner_mode="embedded")
    assert state == "held"
    try:
        first = request_dispatcher_quiescence(
            expected_pid=os.getpid(), timeout_seconds=0.02, poll_interval=0.01
        )
        assert first["ok"] is False
        assert first["state"] == "timeout"
        request_path = (
            tmp_path
            / "kanban"
            / ".dispatcher-control"
            / ".dispatcher.quiesce.request.json"
        )
        first_request_id = json.loads(request_path.read_text())["request_id"]

        second = request_dispatcher_quiescence(
            expected_pid=os.getpid(), timeout_seconds=0.02, poll_interval=0.01
        )
        assert second["ok"] is False
        assert second["state"] == "timeout"
        assert json.loads(request_path.read_text())["request_id"] == first_request_id

        owner = read_dispatcher_owner()
        assert owner is not None
        boundary = DispatcherTickBoundary(owner)
        assert boundary.begin_tick() == "quiesced"
        expected = {
            "ok": True,
            "state": "quiesced",
            "owner_pid": os.getpid(),
            "owner_mode": "embedded",
        }
        assert request_dispatcher_quiescence(
            expected_pid=os.getpid(), timeout_seconds=0.1, poll_interval=0.01
        ) == expected
        assert request_dispatcher_quiescence(
            expected_pid=os.getpid(), timeout_seconds=0.1, poll_interval=0.01
        ) == expected
    finally:
        release_dispatcher_lock(handle)


def test_quiesce_never_signals_the_expected_process(monkeypatch, tmp_path):
    from hermes_cli.kanban_dispatcher import (
        acquire_dispatcher_lock,
        release_dispatcher_lock,
        request_dispatcher_quiescence,
    )

    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    handle, state = acquire_dispatcher_lock(owner_mode="embedded")
    assert state == "held"
    monkeypatch.setattr(
        os,
        "kill",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("quiescence control must not signal any process")
        ),
    )
    try:
        result = request_dispatcher_quiescence(
            expected_pid=os.getpid(), timeout_seconds=0.02, poll_interval=0.01
        )
        assert result["state"] == "timeout"
    finally:
        release_dispatcher_lock(handle)


@pytest.mark.parametrize(
    "request_case",
    ["empty_object", "current_owner_missing_pid", "current_owner_contradictory_pid"],
)
def test_quiesce_does_not_overwrite_unclassifiable_existing_request(
    monkeypatch, tmp_path, request_case,
):
    import uuid

    import hermes_cli.kanban_dispatcher as dispatcher

    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    handle, state = dispatcher.acquire_dispatcher_lock(owner_mode="embedded")
    assert state == "held"
    try:
        owner = dispatcher.read_dispatcher_owner()
        assert owner is not None
        request = {
            "protocol": 1,
            "request_id": uuid.uuid4().hex,
            "owner_id": owner["owner_id"],
            "owner_pid": owner["pid"],
        }
        if request_case == "empty_object":
            request = {}
        elif request_case == "current_owner_missing_pid":
            request.pop("owner_pid")
        elif request_case == "current_owner_contradictory_pid":
            request["owner_pid"] = owner["pid"] + 1
        request_path = dispatcher._dispatcher_control_path(
            ".dispatcher.quiesce.request.json"
        )
        dispatcher._atomic_write_control(request_path, request)

        assert dispatcher.request_dispatcher_quiescence(
            expected_pid=owner["pid"], timeout_seconds=0.02, poll_interval=0.01
        ) == {"ok": False, "state": "indeterminate"}
        assert json.loads(request_path.read_text()) == request
    finally:
        dispatcher.release_dispatcher_lock(handle)


def test_quiesce_rejects_boolean_protocol_ack(monkeypatch, tmp_path):
    import uuid

    import hermes_cli.kanban_dispatcher as dispatcher

    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    handle, state = dispatcher.acquire_dispatcher_lock(owner_mode="embedded")
    assert state == "held"
    try:
        owner = dispatcher.read_dispatcher_owner()
        assert owner is not None
        dispatcher._atomic_write_control(
            dispatcher._dispatcher_control_path(".dispatcher.quiesce.ack.json"),
            {
                "protocol": True,
                "request_id": uuid.uuid4().hex,
                "owner_id": owner["owner_id"],
                "owner_pid": owner["pid"],
                "owner_mode": "embedded",
                "state": "quiesced",
            },
        )

        result = dispatcher.request_dispatcher_quiescence(
            expected_pid=owner["pid"], timeout_seconds=0.02, poll_interval=0.01
        )
        assert result["ok"] is False
        assert result["state"] == "timeout"
    finally:
        dispatcher.release_dispatcher_lock(handle)


def test_stale_ack_cannot_succeed_across_owner_turnover(monkeypatch, tmp_path):
    import uuid

    import hermes_cli.kanban_dispatcher as dispatcher

    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    handle, state = dispatcher.acquire_dispatcher_lock(owner_mode="embedded")
    assert state == "held"
    try:
        old_owner = dispatcher.read_dispatcher_owner()
        assert old_owner is not None
        dispatcher._atomic_write_control(
            dispatcher._dispatcher_control_path(
                ".dispatcher.quiesce.ack.json"
            ),
            {
                "protocol": 1,
                "request_id": uuid.uuid4().hex,
                "owner_id": old_owner["owner_id"],
                "owner_pid": old_owner["pid"],
                "owner_mode": "embedded",
                "state": "quiesced",
            },
        )
        new_owner = dict(old_owner, owner_id=uuid.uuid4().hex)
        owners = iter((old_owner, new_owner))
        monkeypatch.setattr(
            dispatcher, "read_dispatcher_owner", lambda: next(owners)
        )
        assert dispatcher.request_dispatcher_quiescence(
            expected_pid=os.getpid(), timeout_seconds=0.1, poll_interval=0.01
        ) == {"ok": False, "state": "not_owner"}
    finally:
        dispatcher.release_dispatcher_lock(handle)


def test_ack_write_failure_keeps_dispatcher_quiesced(monkeypatch, tmp_path):
    import hermes_cli.kanban_dispatcher as dispatcher

    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    handle, state = dispatcher.acquire_dispatcher_lock(owner_mode="embedded")
    assert state == "held"
    try:
        result = dispatcher.request_dispatcher_quiescence(
            expected_pid=os.getpid(), timeout_seconds=0.02, poll_interval=0.01
        )
        assert result["state"] == "timeout"
        owner = dispatcher.read_dispatcher_owner()
        assert owner is not None
        boundary = dispatcher.DispatcherTickBoundary(owner)
        real_write = dispatcher._atomic_write_control

        def fail_ack(path, payload):
            if path.name == ".dispatcher.quiesce.ack.json":
                raise OSError("read-only control directory")
            return real_write(path, payload)

        monkeypatch.setattr(dispatcher, "_atomic_write_control", fail_ack)
        assert boundary.begin_tick() == "quiesced"
        assert boundary.quiesced is True
    finally:
        dispatcher.release_dispatcher_lock(handle)


@pytest.mark.parametrize(
    "request_case",
    [
        "empty_object",
        "current_owner_missing_pid",
        "current_owner_contradictory_pid",
        "boolean_protocol",
        "non_string_request_id",
        "non_string_owner_id",
        "boolean_owner_pid",
        "invalid_current_protocol",
    ],
)
def test_malformed_or_unclassifiable_request_fails_closed(
    monkeypatch, tmp_path, request_case,
):
    import uuid

    import hermes_cli.kanban_dispatcher as dispatcher

    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    handle, state = dispatcher.acquire_dispatcher_lock(owner_mode="embedded")
    assert state == "held"
    try:
        owner = dispatcher.read_dispatcher_owner()
        assert owner is not None
        request = {
            "protocol": 1,
            "request_id": uuid.uuid4().hex,
            "owner_id": owner["owner_id"],
            "owner_pid": owner["pid"],
        }
        if request_case == "empty_object":
            request = {}
        elif request_case == "current_owner_missing_pid":
            request.pop("owner_pid")
        elif request_case == "current_owner_contradictory_pid":
            request["owner_pid"] = owner["pid"] + 1
        elif request_case == "boolean_protocol":
            request["protocol"] = True
        elif request_case == "non_string_request_id":
            request["request_id"] = 1
        elif request_case == "non_string_owner_id":
            request["owner_id"] = 1
        elif request_case == "boolean_owner_pid":
            request["owner_pid"] = True
        elif request_case == "invalid_current_protocol":
            request["protocol"] = 999
        dispatcher._atomic_write_control(
            tmp_path
            / "kanban"
            / ".dispatcher-control"
            / ".dispatcher.quiesce.request.json",
            request,
        )
        boundary = dispatcher.DispatcherTickBoundary(owner)
        assert boundary.begin_tick() == "quiesced"
        assert not (
            tmp_path
            / "kanban"
            / ".dispatcher-control"
            / ".dispatcher.quiesce.ack.json"
        ).exists()
    finally:
        dispatcher.release_dispatcher_lock(handle)


def test_complete_valid_different_generation_request_is_ignored(monkeypatch, tmp_path):
    import uuid

    import hermes_cli.kanban_dispatcher as dispatcher

    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    handle, state = dispatcher.acquire_dispatcher_lock(owner_mode="embedded")
    assert state == "held"
    try:
        owner = dispatcher.read_dispatcher_owner()
        assert owner is not None
        dispatcher._atomic_write_control(
            tmp_path
            / "kanban"
            / ".dispatcher-control"
            / ".dispatcher.quiesce.request.json",
            {
                "protocol": 1,
                "request_id": uuid.uuid4().hex,
                "owner_id": uuid.uuid4().hex,
                "owner_pid": owner["pid"],
            },
        )
        boundary = dispatcher.DispatcherTickBoundary(owner)
        assert boundary.begin_tick() == "dispatch"
        assert boundary.end_tick() is False
    finally:
        dispatcher.release_dispatcher_lock(handle)


def test_embedded_quiesce_ack_waits_for_inflight_tick_and_prevents_later_dispatch(
    monkeypatch, tmp_path,
):
    from gateway.run import GatewayRunner
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_dispatcher import request_dispatcher_quiescence

    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "kanban": {
                "dispatch_in_gateway": True,
                "dispatch_interval_seconds": 1,
                "auto_decompose": False,
            }
        },
    )
    monkeypatch.setattr(
        kb,
        "list_boards",
        lambda include_archived=False: [{"slug": "default"}],
    )
    monkeypatch.setattr(
        kb,
        "_running_capacity_outside_boards",
        lambda *_args, **_kwargs: (0, {}, True),
    )
    monkeypatch.setattr(kb, "reap_worker_zombies", lambda: [])
    monkeypatch.setattr(kb, "has_spawnable_ready", lambda _conn: False)
    monkeypatch.setattr(kb, "has_spawnable_review", lambda _conn: False)

    class Connection:
        def close(self):
            return None

    monkeypatch.setattr(kb, "connect", lambda *, board: Connection())

    tick_entered = threading.Event()
    allow_tick_to_finish = threading.Event()
    dispatch_calls = []

    def parked_dispatch_once(_conn, **_kwargs):
        dispatch_calls.append("entered")
        tick_entered.set()
        assert allow_tick_to_finish.wait(timeout=5)
        dispatch_calls.append("drained")
        return kb.DispatchResult()

    monkeypatch.setattr(kb, "dispatch_once", parked_dispatch_once)

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    real_sleep = asyncio.sleep

    async def fast_sleep(delay):
        if delay == 5:
            return
        await real_sleep(min(delay, 0.01))

    monkeypatch.setattr("gateway.kanban_watchers.asyncio.sleep", fast_sleep)

    async def scenario():
        watcher = asyncio.create_task(runner._kanban_dispatcher_watcher())
        assert await asyncio.to_thread(tick_entered.wait, 2)

        request = asyncio.create_task(
            asyncio.to_thread(
                request_dispatcher_quiescence,
                expected_pid=os.getpid(),
                timeout_seconds=2,
                poll_interval=0.01,
            )
        )
        await real_sleep(0.05)
        assert not request.done(), "acknowledgment must wait for the parked tick"

        allow_tick_to_finish.set()
        result = await request
        assert result == {
            "ok": True,
            "state": "quiesced",
            "owner_pid": os.getpid(),
            "owner_mode": "embedded",
        }

        await real_sleep(0.05)
        assert dispatch_calls == ["entered", "drained"]

        runner._running = False
        await asyncio.wait_for(watcher, timeout=2)

    asyncio.run(scenario())


def test_embedded_watcher_cancellation_waits_for_inflight_tick(monkeypatch, tmp_path):
    from gateway.run import GatewayRunner
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_dispatcher import (
        acquire_dispatcher_lock,
        release_dispatcher_lock,
    )

    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "kanban": {
                "dispatch_in_gateway": True,
                "dispatch_interval_seconds": 1,
                "auto_decompose": False,
            }
        },
    )
    monkeypatch.setattr(
        kb,
        "list_boards",
        lambda include_archived=False: [{"slug": "default"}],
    )
    monkeypatch.setattr(
        kb,
        "_running_capacity_outside_boards",
        lambda *_args, **_kwargs: (0, {}, True),
    )
    monkeypatch.setattr(kb, "reap_worker_zombies", lambda: [])

    class Connection:
        def close(self):
            return None

    monkeypatch.setattr(kb, "connect", lambda *, board: Connection())
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def parked_dispatch(_conn, **_kwargs):
        calls.append("entered")
        entered.set()
        assert release.wait(timeout=5)
        calls.append("drained")
        return kb.DispatchResult()

    monkeypatch.setattr(kb, "dispatch_once", parked_dispatch)
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    real_sleep = asyncio.sleep

    async def fast_sleep(delay):
        if delay == 5:
            return
        await real_sleep(min(delay, 0.01))

    monkeypatch.setattr("gateway.kanban_watchers.asyncio.sleep", fast_sleep)

    async def scenario():
        watcher = asyncio.create_task(runner._kanban_dispatcher_watcher())
        assert await asyncio.to_thread(entered.wait, 2)
        watcher.cancel()
        await real_sleep(0.01)
        watcher.cancel()
        await real_sleep(0.05)
        try:
            assert not watcher.done(), "cancellation must not orphan the active thread"
        finally:
            release.set()
        try:
            await watcher
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("watcher cancellation was swallowed")

    asyncio.run(scenario())
    assert calls == ["entered", "drained"]
    handle, state = acquire_dispatcher_lock()
    try:
        assert state == "held", "singleton releases only after the tick drains"
    finally:
        release_dispatcher_lock(handle)
