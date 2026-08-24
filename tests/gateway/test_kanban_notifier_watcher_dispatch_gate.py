"""Tests that notifier ownership is independent from dispatcher ownership.

- Gateways continue polling subscriptions when standalone dispatch is selected.
- Effective env overrides only disable embedded dispatch, not notifications.
"""

import asyncio
from unittest.mock import MagicMock, patch

from gateway.config import Platform
from gateway.run import GatewayRunner


def _make_runner(with_adapter=False):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: MagicMock()} if with_adapter else {}
    runner._kanban_sub_fail_counts = {}
    return runner


def _fake_config(dispatch_in_gateway):
    return {"kanban": {"dispatch_in_gateway": dispatch_in_gateway}}


def _run_one_notifier_tick(runner, past_gate):
    sleep_calls = []

    async def fake_sleep(delay):
        sleep_calls.append(delay)
        if len(sleep_calls) >= 2:
            runner._running = False

    async def fake_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    import hermes_cli.kanban_db as _kb

    with patch.object(
        _kb, "list_boards",
        side_effect=lambda *a, **kw: past_gate.append(True) or [],
    ):
        with patch("asyncio.sleep", side_effect=fake_sleep):
            with patch("asyncio.to_thread", side_effect=fake_to_thread):
                asyncio.run(runner._kanban_notifier_watcher())


def test_notifier_watcher_runs_when_standalone_dispatch_is_configured():
    runner = _make_runner(with_adapter=True)
    past_gate = []
    with patch("hermes_cli.config.load_config", return_value=_fake_config(False)):
        _run_one_notifier_tick(runner, past_gate)
    assert past_gate


def test_notifier_watcher_runs_when_env_selects_standalone_dispatch(monkeypatch):
    runner = _make_runner(with_adapter=True)
    past_gate = []
    monkeypatch.setenv("HERMES_KANBAN_DISPATCH_IN_GATEWAY", "false")
    with patch("hermes_cli.config.load_config", return_value=_fake_config(True)):
        _run_one_notifier_tick(runner, past_gate)
    assert past_gate


def test_notifier_watcher_runs_when_dispatch_enabled():
    """dispatch_in_gateway=true proceeds past the gate to the board fan-out."""
    runner = _make_runner(with_adapter=True)
    past_gate = []
    with patch("hermes_cli.config.load_config", return_value=_fake_config(True)):
        _run_one_notifier_tick(runner, past_gate)

    assert past_gate, "list_boards should be called when dispatch_in_gateway=true"
