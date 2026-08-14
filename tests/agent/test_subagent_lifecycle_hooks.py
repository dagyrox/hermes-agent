"""Contract tests for versioned delegated-child lifecycle hooks."""

import threading
import time
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

from agent import lifecycle_hooks as lh
from tools import delegate_tool as dt


_CANARIES = {
    "SECRET-CANARY",
    "goal prompt must not leak",
    "model response must not leak",
    "terminal --danger",
    "/private/worktree/path",
    "RuntimeError: raw exception",
}


def _parent():
    parent = MagicMock()
    parent.session_id = "parent-session"
    parent._current_turn_id = "turn-1"
    parent._current_task_id = "parent-task"
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._touch_activity = MagicMock()
    return parent


def _child(release: threading.Event, started: threading.Event):
    child = MagicMock()
    child.session_id = "child-session"
    child._subagent_id = "subagent-1"
    child._delegation_wrapper_id = "wrapper-1"
    child._parent_subagent_id = None
    child._parent_turn_id = "turn-1"
    child._delegate_role = "leaf"
    child._delegate_depth = 1
    child._credential_pool = MagicMock()
    child._credential_pool.acquire_lease.return_value = "lease-1"
    child.get_activity_summary.return_value = {
        "current_tool": "terminal",
        "api_call_count": 1,
        "max_iterations": 50,
        "last_activity_desc": "working",
    }

    def run_conversation(**_kwargs):
        started.set()
        release.wait(timeout=5)
        return {"final_response": "model response must not leak", "completed": True}

    child.run_conversation.side_effect = run_conversation
    return child


def test_identity_enum_dtos_drop_runtime_text(monkeypatch):
    emitted = []
    monkeypatch.setattr(lh, "_emit", lambda hook, payload: emitted.append((hook, payload)))

    lh.emit_subagent_lifecycle(
        "terminal",
        child_session_id="child-session",
        child_subagent_id="subagent-1",
        parent_session_id="parent-session",
        child_role="leaf",
        terminal_status="succeeded",
    )
    lh.emit_delegation_wrapper_lifecycle(
        "terminal",
        wrapper_id="wrapper-1",
        child_subagent_id="subagent-1",
        terminal_status="cancelled",
    )
    lh.emit_managed_process_lifecycle(
        "terminal",
        process_id="proc-1",
        task_id="task-1",
        pid=123,
        host_start_time=456,
        host_boot_id="boot-identity",
        pid_scope="host",
        backend="local",
        terminal_status="exited",
        termination_source="reader",
    )

    assert {hook for hook, _ in emitted} == {
        "subagent_lifecycle",
        "delegation_wrapper_lifecycle",
        "managed_process_lifecycle",
    }
    serialized = repr(emitted)
    assert not any(canary in serialized for canary in _CANARIES)
    assert set(emitted[0][1]) == {
        "contract_version",
        "event",
        "occurred_at",
        "sequence",
        "child_session_id",
        "child_subagent_id",
        "parent_session_id",
        "parent_turn_id",
        "parent_subagent_id",
        "delegation_wrapper_id",
        "child_role",
        "terminal_status",
    }
    assert emitted[0][1]["occurred_at"].endswith("Z")
    assert type(emitted[0][1]["sequence"]) is int
    assert emitted[2][1]["exit_code"] is None


def test_emitted_dto_versions_match_advertised_hook_contracts(monkeypatch):
    from hermes_cli.plugins import HOOK_CONTRACT_VERSIONS

    emitted = {}
    monkeypatch.setattr(
        lh, "_emit", lambda hook, payload: emitted.setdefault(hook, dict(payload))
    )

    lh.emit_subagent_lifecycle(
        "started",
        child_session_id="child-session-version",
        child_subagent_id="subagent-version",
        child_role="leaf",
    )
    lh.emit_delegation_wrapper_lifecycle(
        "started",
        wrapper_id="wrapper-version",
        child_subagent_id="subagent-version",
    )
    lh.emit_managed_process_lifecycle(
        "started",
        process_id="process-version",
        pid=123,
        host_start_time=456,
        host_boot_id="boot-version",
        pid_scope="host",
        backend="local",
    )

    assert set(emitted) == set(HOOK_CONTRACT_VERSIONS)
    assert {
        hook: dto["contract_version"] for hook, dto in emitted.items()
    } == HOOK_CONTRACT_VERSIONS


def test_registered_after_stable_ids_and_legacy_start_is_preserved(monkeypatch):
    parent = _parent()
    parent._delegate_depth = 0
    parent.base_url = "https://example.invalid/v1"
    parent.api_key = "key"
    parent.model = "model"
    parent.provider = "openai"
    parent.api_mode = "chat_completions"
    parent.reasoning_config = {"enabled": False}
    parent.enabled_toolsets = []
    parent.disabled_toolsets = []
    parent.prefill_messages = None
    parent._fallback_chain = []
    parent.request_overrides = {}
    parent.openrouter_min_coding_score = None
    child = MagicMock()
    child.session_id = "stable-child-session"
    calls = []

    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda hook, **payload: calls.append((hook, dict(payload))),
    )
    monkeypatch.setattr(dt, "_resolve_child_credential_pool", lambda *_args: None)
    with patch("run_agent.AIAgent", return_value=child):
        built = dt._build_child_agent(
            task_index=0,
            goal="goal prompt must not leak from v1",
            context=None,
            toolsets=None,
            model=None,
            max_iterations=5,
            parent_agent=parent,
            task_count=1,
        )

    assert built is child
    lifecycle = [
        payload["dto"] for hook, payload in calls if hook == "subagent_lifecycle"
    ]
    assert len(lifecycle) == 1
    assert lifecycle[0]["event"] == "registered"
    assert lifecycle[0]["child_session_id"] == "stable-child-session"
    assert lifecycle[0]["child_subagent_id"]
    assert "goal prompt must not leak" not in repr(lifecycle)
    # Existing plugins retain their historical callback and payload.
    legacy = [payload for hook, payload in calls if hook == "subagent_start"]
    assert len(legacy) == 1
    assert legacy[0]["child_goal"] == "goal prompt must not leak from v1"


def test_provider_invokes_native_hook_with_one_bounded_dto(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda hook, **kwargs: calls.append((hook, kwargs)),
    )

    lh.emit_subagent_lifecycle(
        "started",
        child_session_id="child-session",
        child_subagent_id="subagent-dto",
        child_role="leaf",
    )

    assert len(calls) == 1
    hook, kwargs = calls[0]
    assert hook == "subagent_lifecycle"
    assert set(kwargs) == {"dto"}
    assert kwargs["dto"]["event"] == "started"
    assert kwargs["dto"]["sequence"] == 1
    assert "SECRET-CANARY" not in repr(kwargs)


def test_child_owned_start_timeout_heartbeat_terminal_and_cleanup(monkeypatch):
    release = threading.Event()
    started = threading.Event()
    parent = _parent()
    child = _child(release, started)
    parent._active_children.append(child)
    events = []
    wrapper_events = []
    events_lock = threading.Lock()

    def capture(hook, payload):
        with events_lock:
            target = wrapper_events if hook == "delegation_wrapper_lifecycle" else events
            target.append(dict(payload))

    monkeypatch.setattr(lh, "_emit", capture)
    monkeypatch.setattr(dt, "_HEARTBEAT_INTERVAL", 0.01)
    monkeypatch.setattr(dt, "_HEARTBEAT_STALE_CYCLES_IN_TOOL", 1000)
    monkeypatch.setattr(dt, "_get_child_timeout", lambda: 0.08)
    monkeypatch.setattr(dt, "_dump_subagent_timeout_diagnostic", lambda **_kw: None)

    result = dt._run_single_child(
        task_index=0,
        goal="goal prompt must not leak",
        child=child,
        parent_agent=parent,
    )

    assert started.is_set()
    assert result["status"] == "timeout"
    with events_lock:
        timeout_events = list(events)
    names = [event["event"] for event in timeout_events]
    assert names[:2] == ["queued", "started"]
    assert "heartbeat" in names
    assert names[-1] == "cancel_requested"
    assert "terminal" not in names
    assert [event["event"] for event in wrapper_events] == [
        "started", "timed_out", "terminal"
    ]
    assert wrapper_events[-1]["terminal_status"] == "cancelled"
    # Waiter timeout owns none of the live-child cleanup.
    assert child in parent._active_children
    assert child._credential_pool.release_lease.call_count == 0
    assert child.close.call_count == 0
    assert child._subagent_id in dt._active_subagents

    heartbeat_count = names.count("heartbeat")
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        with events_lock:
            if sum(e["event"] == "heartbeat" for e in events) > heartbeat_count:
                break
        time.sleep(0.01)
    else:
        raise AssertionError("child-owned heartbeat stopped at waiter timeout")

    release.set()
    deadline = time.monotonic() + 2
    terminal = []
    while time.monotonic() < deadline:
        with events_lock:
            terminal = [e for e in events if e["event"] == "terminal"]
        if terminal and child.close.call_count:
            break
        time.sleep(0.01)

    assert len(terminal) == 1
    assert terminal[0]["terminal_status"] == "succeeded"
    assert child not in parent._active_children
    child._credential_pool.release_lease.assert_called_once_with("lease-1")
    child.close.assert_called_once()
    assert child._subagent_id not in dt._active_subagents
    assert not any(canary in repr(events) for canary in _CANARIES)


def test_terminal_failure_is_emitted_once_under_exception(monkeypatch):
    child = MagicMock()
    child.session_id = "child-session"
    child._subagent_id = "subagent-fail"
    child._delegation_wrapper_id = "wrapper-fail"
    child._delegate_role = "leaf"
    child._delegate_depth = 1
    child._credential_pool = None
    child.get_activity_summary.return_value = {}
    child.run_conversation.side_effect = RuntimeError("RuntimeError: raw exception")
    parent = _parent()
    parent._active_children.append(child)
    emitted = []
    monkeypatch.setattr(lh, "_emit", lambda _hook, payload: emitted.append(dict(payload)))
    monkeypatch.setattr(dt, "_get_child_timeout", lambda: None)

    result = dt._run_single_child(0, "SECRET-CANARY", child, parent)

    terminal = [
        event for event in emitted
        if event["event"] == "terminal" and "child_session_id" in event
    ]
    assert result["status"] == "error"
    assert len(terminal) == 1
    assert terminal[0]["terminal_status"] == "failed"
    assert "RuntimeError: raw exception" not in repr(emitted)


def test_post_registration_setup_failures_terminalize_child_and_wrapper_once(monkeypatch):
    for failure in ("lease", "registry"):
        child = MagicMock()
        child.session_id = f"child-session-{failure}"
        child._subagent_id = f"subagent-{failure}"
        child._delegation_wrapper_id = f"wrapper-{failure}"
        child._delegate_role = "leaf"
        child._delegate_depth = 1
        child.get_activity_summary.return_value = {}
        child._credential_pool = MagicMock()
        child._credential_pool.acquire_lease.return_value = None
        parent = _parent()
        parent._active_children.append(child)
        emitted = []
        monkeypatch.setattr(
            lh, "_emit", lambda hook, payload: emitted.append((hook, dict(payload)))
        )
        if failure == "lease":
            child._credential_pool.acquire_lease.side_effect = RuntimeError("SECRET-CANARY")
            register = patch.object(dt, "_register_subagent")
        else:
            register = patch.object(dt, "_register_subagent", side_effect=RuntimeError("SECRET-CANARY"))

        with register:
            try:
                dt._run_single_child(0, "goal prompt must not leak", child, parent)
            except RuntimeError:
                pass
            else:
                raise AssertionError("setup failure must propagate")

        terminal = [(hook, dto) for hook, dto in emitted if dto["event"] == "terminal"]
        assert len(terminal) == 2
        assert {hook for hook, _ in terminal} == {
            "subagent_lifecycle",
            "delegation_wrapper_lifecycle",
        }
        assert all(dto["terminal_status"] == "failed" for _, dto in terminal)
        assert child not in parent._active_children
        child.close.assert_called_once()
        assert "SECRET-CANARY" not in repr(emitted)


def test_later_batch_build_failure_closes_every_registered_unstarted_child(monkeypatch):
    parent = _parent()
    parent._delegate_depth = 0
    first = MagicMock()
    first.session_id = "child-built-before-failure"
    emitted = []

    monkeypatch.setattr(
        lh, "_emit", lambda hook, payload: emitted.append((hook, dict(payload)))
    )
    monkeypatch.setattr(dt, "_resolve_child_credential_pool", lambda *_args: None)

    with patch(
        "run_agent.AIAgent",
        side_effect=[first, RuntimeError("SECRET-CANARY later build failure")],
    ):
        try:
            dt.delegate_task(
                tasks=[{"goal": "first private goal"}, {"goal": "second private goal"}],
                parent_agent=parent,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("later child build failure must propagate")

    child_events = [
        dto for hook, dto in emitted
        if hook == "subagent_lifecycle"
        and dto.get("child_session_id") == "child-built-before-failure"
    ]
    wrapper_events = [
        dto for hook, dto in emitted
        if hook == "delegation_wrapper_lifecycle"
        and dto.get("child_subagent_id") == child_events[0]["child_subagent_id"]
    ]
    assert [dto["event"] for dto in child_events] == ["registered", "terminal"]
    assert [dto["event"] for dto in wrapper_events] == ["registered", "terminal"]
    assert child_events[-1]["terminal_status"] == "failed"
    assert wrapper_events[-1]["terminal_status"] == "failed"
    assert parent._active_children == []
    first.close.assert_called_once()
    assert not any(canary in repr(emitted) for canary in _CANARIES)


def test_partial_batch_submit_failure_closes_failed_and_unsubmitted_children(monkeypatch):
    parent = _parent()
    parent._delegate_depth = 0
    children = []
    for index in range(3):
        child = MagicMock()
        child.session_id = f"child-submit-{index}"
        child._credential_pool = None
        child.get_activity_summary.return_value = {}
        child.run_conversation.return_value = {
            "final_response": "safe summary",
            "completed": True,
            "api_calls": 1,
        }
        children.append(child)
    emitted = []

    class FailSecondSubmitExecutor:
        def __init__(self, **_kwargs):
            self.submit_count = 0
            self.threads = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.shutdown(wait=True)
            return False

        def shutdown(self, wait=True, **_kwargs):
            if wait:
                for thread in self.threads:
                    thread.join(timeout=2)

        def submit(self, fn, **kwargs):
            self.submit_count += 1
            if self.submit_count == 2:
                raise RuntimeError("SECRET-CANARY submit failure")
            future = Future()

            def run():
                try:
                    future.set_result(fn(**kwargs))
                except BaseException as exc:
                    future.set_exception(exc)

            thread = threading.Thread(target=run)
            self.threads.append(thread)
            thread.start()
            return future

    monkeypatch.setattr(
        lh, "_emit", lambda hook, payload: emitted.append((hook, dict(payload)))
    )
    monkeypatch.setattr(dt, "_resolve_child_credential_pool", lambda *_args: None)
    monkeypatch.setattr(dt, "_get_child_timeout", lambda: None)
    monkeypatch.setattr(
        "tools.daemon_pool.DaemonThreadPoolExecutor", FailSecondSubmitExecutor
    )

    with patch("run_agent.AIAgent", side_effect=children):
        try:
            dt.delegate_task(
                tasks=[{"goal": f"private goal {i}"} for i in range(3)],
                parent_agent=parent,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("executor submit failure must propagate")

    for index, child in enumerate(children):
        child_events = [
            dto for hook, dto in emitted
            if hook == "subagent_lifecycle"
            and dto.get("child_session_id") == f"child-submit-{index}"
        ]
        wrapper_events = [
            dto for hook, dto in emitted
            if hook == "delegation_wrapper_lifecycle"
            and dto.get("child_subagent_id") == child_events[0]["child_subagent_id"]
        ]
        terminals = [dto for dto in child_events if dto["event"] == "terminal"]
        wrapper_terminals = [dto for dto in wrapper_events if dto["event"] == "terminal"]
        assert len(terminals) == 1
        assert len(wrapper_terminals) == 1
        expected = "succeeded" if index == 0 else "failed"
        assert terminals[0]["terminal_status"] == expected
        assert wrapper_terminals[0]["terminal_status"] == expected
        if index > 0:
            assert "started" not in [dto["event"] for dto in child_events]
        child.close.assert_called_once()

    assert parent._active_children == []
    assert not any(canary in repr(emitted) for canary in _CANARIES)


def test_nonempty_failed_and_incomplete_results_never_emit_success(monkeypatch):
    result_shapes = [
        {
            "final_response": "Invalid API response after retries",
            "completed": False,
            "failed": True,
            "error": "provider failure",
        },
        {
            "final_response": "Content policy blocked this request",
            "completed": False,
            "failed": True,
            "error": "content_policy_violation",
        },
        {
            "final_response": "Partial answer at the iteration limit",
            "completed": False,
            "failed": False,
            "exit_reason": "max_iterations",
        },
        {"final_response": "", "completed": True, "failed": False},
        {"final_response": "(empty)", "completed": True, "failed": False},
        {
            "final_response": "Contradictory provider result",
            "completed": True,
            "failed": True,
        },
    ]

    monkeypatch.setattr(dt, "_get_child_timeout", lambda: None)
    for index, run_result in enumerate(result_shapes):
        child = MagicMock()
        child.session_id = f"child-session-failed-{index}"
        child._subagent_id = f"subagent-result-failed-{index}"
        child._delegation_wrapper_id = f"wrapper-result-failed-{index}"
        child._delegate_role = "leaf"
        child._delegate_depth = 1
        child._credential_pool = None
        child.get_activity_summary.return_value = {}
        child.run_conversation.return_value = run_result
        parent = _parent()
        parent._active_children.append(child)
        emitted = []
        monkeypatch.setattr(
            lh, "_emit", lambda _hook, payload: emitted.append(dict(payload))
        )

        parent_result = dt._run_single_child(
            index, "SECRET-CANARY", child, parent
        )

        terminal = [
            event for event in emitted
            if event["event"] == "terminal" and "child_session_id" in event
        ]
        assert parent_result["status"] == "failed"
        assert parent_result["exit_reason"] != "completed"
        assert len(terminal) == 1
        assert terminal[0]["terminal_status"] == "failed"
        if run_result["final_response"]:
            assert run_result["final_response"] not in repr(emitted)
