"""Shared configuration contract for embedded and standalone Kanban dispatchers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from hermes_cli.kanban_db import DEFAULT_FAILURE_LIMIT, kanban_home


@dataclass(frozen=True)
class DispatcherOptions:
    """Normalized effective settings for one dispatcher process."""

    dispatch_in_gateway: bool = True
    interval: float = 60.0
    max_spawn: Optional[int] = None
    max_in_progress: Optional[int] = None
    max_in_progress_per_profile: Optional[int] = None
    default_assignee: Optional[str] = None
    failure_limit: int = DEFAULT_FAILURE_LIMIT
    stale_timeout_seconds: int = 0


def acquire_dispatcher_lock(
    lock_path: Optional[Path] = None,
) -> tuple[Optional[object], str]:
    """Acquire the machine-global singleton lock shared by all dispatch modes."""

    try:
        from gateway.status import _try_acquire_file_lock
    except ImportError:
        return None, "unavailable"

    resolved_lock_path = (
        lock_path
        if lock_path is not None
        else kanban_home() / "kanban" / ".dispatcher.lock"
    )
    try:
        resolved_lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(str(resolved_lock_path), "a+", encoding="utf-8")
    except OSError:
        return None, "unavailable"
    if not _try_acquire_file_lock(handle):
        handle.close()
        return None, "contended"
    return handle, "held"


def release_dispatcher_lock(handle: Optional[object]) -> None:
    """Release a handle returned by :func:`acquire_dispatcher_lock`."""

    if handle is None:
        return
    try:
        from gateway.status import _release_file_lock

        _release_file_lock(handle)
    except Exception:
        pass
    try:
        handle.close()  # type: ignore[attr-defined]
    except Exception:
        pass


def _positive_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _positive_float(value: Any, default: float) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _nonnegative_int(value: Any, default: int = 0) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def resolve_dispatcher_options(
    config: Mapping[str, Any] | None,
    *,
    interval_override: Any = None,
    max_spawn_override: Any = None,
    failure_limit_override: Any = None,
) -> DispatcherOptions:
    """Resolve config with explicit CLI values taking precedence.

    Invalid or non-positive concurrency caps mean "unlimited". Invalid
    interval and failure-limit values fall back to their safe defaults.
    """

    raw_config = config if isinstance(config, Mapping) else {}
    raw_kanban = raw_config.get("kanban", {})
    kanban = raw_kanban if isinstance(raw_kanban, Mapping) else {}

    config_interval = _positive_float(
        kanban.get("dispatch_interval_seconds"), 60.0
    )
    interval = _positive_float(interval_override, config_interval)

    config_max_spawn = _positive_int(kanban.get("max_spawn"))
    max_spawn = (
        _positive_int(max_spawn_override)
        if max_spawn_override is not None
        else config_max_spawn
    )

    config_failure_limit = (
        _positive_int(kanban.get("failure_limit")) or DEFAULT_FAILURE_LIMIT
    )
    failure_limit = (
        _positive_int(failure_limit_override)
        if failure_limit_override is not None
        else config_failure_limit
    ) or DEFAULT_FAILURE_LIMIT

    default_assignee = str(kanban.get("default_assignee") or "").strip() or None
    dispatch_in_gateway = bool(kanban.get("dispatch_in_gateway", True))
    env_dispatch = os.environ.get(
        "HERMES_KANBAN_DISPATCH_IN_GATEWAY", ""
    ).strip().lower()
    if env_dispatch in {"0", "false", "no", "off"}:
        dispatch_in_gateway = False

    return DispatcherOptions(
        dispatch_in_gateway=dispatch_in_gateway,
        interval=interval,
        max_spawn=max_spawn,
        max_in_progress=_positive_int(kanban.get("max_in_progress")),
        max_in_progress_per_profile=_positive_int(
            kanban.get("max_in_progress_per_profile")
        ),
        default_assignee=default_assignee,
        failure_limit=failure_limit,
        stale_timeout_seconds=_nonnegative_int(
            kanban.get("dispatch_stale_timeout_seconds"), 0
        ),
    )
