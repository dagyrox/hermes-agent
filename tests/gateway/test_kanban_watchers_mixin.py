"""Tests for the extracted GatewayKanbanWatchersMixin (god-file Phase 3).

The kanban watcher loops were lifted out of gateway/run.py into a mixin that
GatewayRunner inherits. These tests confirm the mixin exposes the methods and
that GatewayRunner picks them up via the MRO (behavior-neutral relocation).
"""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import MagicMock, patch

from gateway.kanban_watchers import GatewayKanbanWatchersMixin

KANBAN_METHODS = [
    "_kanban_notifier_watcher",
    "_kanban_dispatcher_watcher",
    "_kanban_advance",
    "_kanban_unsub",
    "_kanban_rewind",
    "_deliver_kanban_artifacts",
]


def test_mixin_defines_kanban_methods():
    for m in KANBAN_METHODS:
        assert hasattr(GatewayKanbanWatchersMixin, m), f"mixin missing {m}"


def test_gateway_runner_inherits_mixin():
    # Import here so a heavy gateway import only happens if the first test passed.
    from gateway.run import GatewayRunner

    assert issubclass(GatewayRunner, GatewayKanbanWatchersMixin)
    # Each kanban method resolves to the mixin's implementation via the MRO.
    for m in KANBAN_METHODS:
        owner = next(c for c in GatewayRunner.__mro__ if m in c.__dict__)
        assert owner is GatewayKanbanWatchersMixin, (
            f"{m} resolved to {owner.__name__}, expected the mixin"
        )


def test_watcher_loops_are_coroutines():
    # The two long-running watchers are async loops.
    assert inspect.iscoroutinefunction(GatewayKanbanWatchersMixin._kanban_notifier_watcher)
    assert inspect.iscoroutinefunction(GatewayKanbanWatchersMixin._kanban_dispatcher_watcher)


def test_singleton_dispatcher_lock_is_exclusive(tmp_path):
    """Only one holder of the dispatcher lock at a time — the backstop that
    stops concurrent dispatchers double reclaiming and corrupting shared
    kanban SQLite index pages under wal_autocheckpoint=0."""
    import os

    from gateway.kanban_watchers import _acquire_singleton_lock, _release_singleton_lock

    lock = tmp_path / "kanban" / ".dispatcher.lock"

    h1, st1 = _acquire_singleton_lock(lock)
    assert st1 == "held" and h1 is not None

    # A second acquire while the first is held must be refused, not granted.
    h2, st2 = _acquire_singleton_lock(lock)
    assert st2 == "contended" and h2 is None

    # Releasing the first lets a fresh acquire succeed (lock is reusable).
    _release_singleton_lock(h1)
    h3, st3 = _acquire_singleton_lock(lock)
    assert st3 == "held" and h3 is not None
    _release_singleton_lock(h3)


def test_embedded_dispatcher_enforces_global_capacity_across_boards(
    monkeypatch, tmp_path,
):
    from gateway.run import GatewayRunner
    from hermes_cli import kanban_db as kb

    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _name: True)
    for slug in ("default", "secondary"):
        kb.create_board(slug=slug, name=slug.title())
        with kb.connect_closing(board=slug) as conn:
            for index in range(2):
                kb.create_task(conn, title=f"{slug}-{index}", assignee="alpha")
    monkeypatch.setattr(kb, "_default_spawn", lambda *_args, **_kwargs: 12345)

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    sleeps = 0

    async def fake_sleep(_delay):
        nonlocal sleeps
        sleeps += 1
        if sleeps >= 2:
            runner._running = False

    async def fake_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    config = {
        "kanban": {
            "dispatch_in_gateway": True,
            "dispatch_interval_seconds": 1,
            "max_spawn": 2,
            "max_in_progress": 2,
            "max_in_progress_per_profile": 2,
            "auto_decompose": False,
        }
    }
    with patch("hermes_cli.config.load_config", return_value=config):
        with patch(
            "gateway.kanban_watchers._acquire_singleton_lock",
            return_value=(MagicMock(), "held"),
        ):
            with patch("asyncio.sleep", side_effect=fake_sleep):
                with patch("asyncio.to_thread", side_effect=fake_to_thread):
                    asyncio.run(runner._kanban_dispatcher_watcher())

    running = 0
    for slug in ("default", "secondary"):
        with kb.connect_closing(board=slug) as conn:
            running += conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE status = 'running'"
            ).fetchone()[0]
    assert running == 2
