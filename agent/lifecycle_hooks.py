"""Versioned, privacy-bounded lifecycle hook payloads.

These helpers are the provider boundary for delegated children and managed
background processes.  Payloads intentionally contain only stable identities,
small integers, and allowlisted enum values.  Runtime text (prompts, commands,
paths, output, exceptions, model/tool metadata) must never cross this seam.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

_sequence_lock = threading.Lock()
_sequences: dict[tuple[str, str], int] = {}

SUBAGENT_LIFECYCLE_VERSION = 1
MANAGED_PROCESS_LIFECYCLE_VERSION = 1

SUBAGENT_EVENTS = frozenset(
    {"registered", "queued", "started", "heartbeat", "cancel_requested", "terminal"}
)
SUBAGENT_ROLES = frozenset({"leaf", "orchestrator"})
SUBAGENT_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})
SUBAGENT_CANCEL_REASONS = frozenset({"waiter_timeout", "parent_interrupt", "explicit_cancel"})

MANAGED_PROCESS_EVENTS = frozenset(
    {"started", "heartbeat", "probe_unavailable", "terminal", "adopted"}
)
PROCESS_PID_SCOPES = frozenset({"host", "sandbox"})
PROCESS_BACKENDS = frozenset(
    {"local", "docker", "singularity", "modal", "managed_modal", "daytona", "ssh", "unknown"}
)
PROCESS_TERMINAL_STATUSES = frozenset(
    {"exited", "killed", "lost", "failed_start", "already_exited"}
)
PROCESS_TERMINATION_SOURCES = frozenset(
    {"process.kill", "kill_all", "backend_lost", "failed_start", "reader", "reconciler", "unknown"}
)


def _identity(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def _integer(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _envelope(kind: str, source_id: Optional[str], event: str) -> dict[str, Any]:
    """Build the common v1 ordering envelope for one opaque source identity."""
    if source_id is None:
        raise ValueError(f"{kind} lifecycle event requires a stable source identity")
    key = (kind, source_id)
    with _sequence_lock:
        sequence = _sequences.get(key, 0) + 1
        _sequences[key] = sequence
    occurred_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )
    return {
        "contract_version": 1,
        "event": event,
        "occurred_at": occurred_at,
        "sequence": sequence,
    }


def _emit(hook_name: str, payload: dict[str, Any]) -> None:
    try:
        from hermes_cli.plugins import invoke_hook

        # A single DTO keyword makes the privacy boundary explicit and prevents
        # future helper locals from accidentally becoming hook kwargs.
        invoke_hook(hook_name, dto=payload)
    except Exception:
        logger.debug("lifecycle_hook_failed hook=%s reason=callback_error", hook_name)


def emit_subagent_lifecycle(
    event: str,
    *,
    child_session_id: Any,
    child_subagent_id: Any,
    parent_session_id: Any = None,
    parent_turn_id: Any = None,
    parent_subagent_id: Any = None,
    child_role: Any = None,
    terminal_status: Any = None,
    cancel_reason: Any = None,
) -> None:
    """Emit one v1 delegated-child lifecycle observation."""
    if event not in SUBAGENT_EVENTS:
        raise ValueError(f"invalid subagent lifecycle event: {event}")
    role = child_role if child_role in SUBAGENT_ROLES else None
    terminal = terminal_status if terminal_status in SUBAGENT_TERMINAL_STATUSES else None
    reason = cancel_reason if cancel_reason in SUBAGENT_CANCEL_REASONS else None
    child_session = _identity(child_session_id)
    child_subagent = _identity(child_subagent_id)
    if child_session is None or child_subagent is None:
        raise ValueError("subagent lifecycle event requires stable child identities")
    payload = {
        **_envelope("subagent", child_subagent, event),
        "child_session_id": child_session,
        "child_subagent_id": child_subagent,
        "parent_session_id": _identity(parent_session_id),
        "parent_turn_id": _identity(parent_turn_id),
        "parent_subagent_id": _identity(parent_subagent_id),
        "child_role": role,
    }
    if event == "terminal":
        if terminal is None:
            raise ValueError("terminal subagent lifecycle event requires terminal_status")
        payload["terminal_status"] = terminal
    if event == "cancel_requested":
        if reason is None:
            raise ValueError("cancel_requested requires an allowlisted cancel_reason")
        payload["cancel_reason"] = reason
    _emit("subagent_lifecycle", payload)


def managed_process_backend(env: Any) -> str:
    if env is None:
        return "local"
    module_name = type(env).__module__.rsplit(".", 1)[-1]
    return module_name if module_name in PROCESS_BACKENDS else "unknown"


def managed_process_backend_id(env: Any) -> Optional[str]:
    """Return an opaque stable backend identity, never a path or command."""
    if env is None:
        return None
    for attr in ("_sandbox_id", "_container_id", "sandbox_id", "container_id", "instance_id"):
        value = getattr(env, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


def emit_managed_process_lifecycle(
    event: str,
    *,
    process_id: Any,
    task_id: Any = None,
    session_key: Any = None,
    pid: Any = None,
    host_start_time: Any = None,
    pid_scope: Any = None,
    backend: Any = None,
    backend_id: Any = None,
    terminal_status: Any = None,
    termination_source: Any = None,
    exit_code: Any = None,
) -> None:
    """Emit one v1 managed-process lifecycle observation."""
    if event not in MANAGED_PROCESS_EVENTS:
        raise ValueError(f"invalid managed process lifecycle event: {event}")
    scope = pid_scope if pid_scope in PROCESS_PID_SCOPES else None
    backend_name = backend if backend in PROCESS_BACKENDS else "unknown"
    process_identity = _identity(process_id)
    if process_identity is None:
        raise ValueError("managed-process lifecycle event requires a stable process identity")
    payload = {
        **_envelope("managed_process", process_identity, event),
        "process_id": process_identity,
        "task_id": _identity(task_id),
        "session_key": _identity(session_key),
        "pid": _integer(pid),
        "host_start_time": _integer(host_start_time),
        "pid_scope": scope,
        "backend": backend_name,
        "backend_id": _identity(backend_id),
    }
    if event == "terminal":
        status = terminal_status if terminal_status in PROCESS_TERMINAL_STATUSES else None
        if status is None:
            raise ValueError("terminal managed-process event requires terminal_status")
        payload["terminal_status"] = status
        payload["termination_source"] = (
            termination_source
            if termination_source in PROCESS_TERMINATION_SOURCES
            else "unknown"
        )
        payload["exit_code"] = _integer(exit_code)
    _emit("managed_process_lifecycle", payload)
