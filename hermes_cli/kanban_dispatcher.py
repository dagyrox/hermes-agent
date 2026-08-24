"""Shared configuration contract for embedded and standalone Kanban dispatchers."""

from __future__ import annotations

import json
import math
import os
import stat
import time
import uuid
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


_CONTROL_PROTOCOL = 1
_OWNER_MODES = {"embedded", "standalone"}
_CONTROL_DIR_NAME = ".dispatcher-control"


def _dispatcher_control_path(name: str) -> Path:
    return kanban_home() / "kanban" / _CONTROL_DIR_NAME / name


def _ensure_private_directory(path: Path) -> None:
    if os.name == "nt":
        # POSIX mode bits do not establish a private Windows ACL. Until this
        # protocol has a native ACL implementation, do not create forgeable
        # control files there; the CLI reports an indeterminate fail-closed
        # result instead.
        raise PermissionError(
            "dispatcher quiescence control is unavailable on Windows"
        )
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
        raise PermissionError(f"dispatcher control directory is not privately owned: {path}")
    path.chmod(0o700)
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise PermissionError(f"dispatcher control directory is not private: {path}")


def _atomic_write_control(path: Path, payload: Mapping[str, Any]) -> None:
    _ensure_private_directory(path.parent)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    encoded = (
        json.dumps(dict(payload), separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass


def _read_control(path: Path) -> Optional[dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {"_invalid": True}
    return raw if isinstance(raw, dict) else {"_invalid": True}


def _valid_id(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return uuid.UUID(hex=value).hex == value
    except ValueError:
        return False


def _valid_owner(value: Optional[Mapping[str, Any]]) -> bool:
    return bool(
        value
        and value.get("protocol") == _CONTROL_PROTOCOL
        and _valid_id(value.get("owner_id"))
        and isinstance(value.get("pid"), int)
        and value["pid"] > 0
        and value.get("mode") in _OWNER_MODES
    )


def read_dispatcher_owner() -> Optional[dict[str, Any]]:
    if os.name == "nt":
        return None
    owner = _read_control(_dispatcher_control_path(".dispatcher.owner.json"))
    return owner if _valid_owner(owner) else None


def acquire_dispatcher_lock(
    lock_path: Optional[Path] = None,
    *,
    owner_mode: Optional[str] = None,
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
        if resolved_lock_path.parent.name == _CONTROL_DIR_NAME:
            _ensure_private_directory(resolved_lock_path.parent)
        else:
            resolved_lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(str(resolved_lock_path), "a+", encoding="utf-8")
    except OSError:
        return None, "unavailable"
    if not _try_acquire_file_lock(handle):
        handle.close()
        return None, "contended"
    if owner_mode is not None and owner_mode not in _OWNER_MODES:
        release_dispatcher_lock(handle)
        return None, "unavailable"
    if owner_mode is not None and os.name != "nt":
        try:
            _atomic_write_control(
                _dispatcher_control_path(".dispatcher.owner.json"),
                {
                    "protocol": _CONTROL_PROTOCOL,
                    "owner_id": uuid.uuid4().hex,
                    "pid": os.getpid(),
                    "mode": owner_mode,
                    "started_at": time.time(),
                },
            )
        except OSError:
            release_dispatcher_lock(handle)
            return None, "unavailable"
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


class DispatcherTickBoundary:
    """Linearization boundary between embedded dispatch ticks and quiescence."""

    def __init__(self, owner: Mapping[str, Any]):
        if not _valid_owner(owner) or owner.get("mode") != "embedded":
            raise ValueError("a valid embedded dispatcher owner is required")
        self.owner = dict(owner)
        self._tick_handle: Optional[object] = None
        self.quiesced = False
        self.ack_error: Optional[str] = None

    def _request_state(self) -> tuple[str, Optional[dict[str, Any]]]:
        request = _read_control(
            _dispatcher_control_path(".dispatcher.quiesce.request.json")
        )
        if request is None:
            return "none", None
        if request.get("_invalid"):
            return "invalid", None
        targets_owner = bool(
            request.get("owner_id") == self.owner["owner_id"]
            and request.get("owner_pid") == self.owner["pid"]
        )
        valid = bool(
            targets_owner
            and request.get("protocol") == _CONTROL_PROTOCOL
            and _valid_id(request.get("request_id"))
        )
        if valid:
            return "targeted", request
        if targets_owner:
            return "invalid", None
        return "other", request

    def _acknowledge(self, request: Mapping[str, Any]) -> bool:
        try:
            _atomic_write_control(
                _dispatcher_control_path(".dispatcher.quiesce.ack.json"),
                {
                    "protocol": _CONTROL_PROTOCOL,
                    "request_id": request["request_id"],
                    "owner_id": self.owner["owner_id"],
                    "owner_pid": self.owner["pid"],
                    "owner_mode": self.owner["mode"],
                    "state": "quiesced",
                    "acknowledged_at": time.time(),
                },
            )
        except OSError as exc:
            # Stay safely quiesced even though the requester cannot observe an ack.
            self.ack_error = str(exc)
            return False
        self.ack_error = None
        return True

    def begin_tick(self) -> str:
        """Return dispatch, defer, or quiesced while holding the tick gate."""
        if self.quiesced:
            return "quiesced"
        handle, state = acquire_dispatcher_lock(
            lock_path=_dispatcher_control_path(".dispatcher.tick.lock")
        )
        if state != "held":
            return "defer"
        self._tick_handle = handle
        request_state, request = self._request_state()
        if request_state == "targeted" and request is not None:
            self.quiesced = True
            self._acknowledge(request)
            self.end_tick()
            return "quiesced"
        if request_state == "invalid":
            self.quiesced = True
            self.end_tick()
            return "quiesced"
        return "dispatch"

    def end_tick(self) -> bool:
        """Acknowledge a request only after the protected tick has drained."""
        try:
            if self._tick_handle is not None and not self.quiesced:
                request_state, request = self._request_state()
                if request_state == "targeted" and request is not None:
                    self.quiesced = True
                    self._acknowledge(request)
                elif request_state == "invalid":
                    self.quiesced = True
        finally:
            release_dispatcher_lock(self._tick_handle)
            self._tick_handle = None
        return self.quiesced


def _owner_generation_state(owner: Mapping[str, Any]) -> str:
    """Revalidate the singleton and owner generation before reporting success."""
    probe, lock_state = acquire_dispatcher_lock()
    if lock_state == "held":
        release_dispatcher_lock(probe)
        return "not_running"
    if lock_state != "contended":
        return "indeterminate"
    current_owner = read_dispatcher_owner()
    if (
        current_owner is None
        or current_owner.get("owner_id") != owner.get("owner_id")
        or current_owner.get("pid") != owner.get("pid")
        or current_owner.get("mode") != owner.get("mode")
    ):
        return "not_owner"
    return "current"


def request_dispatcher_quiescence(
    *,
    expected_pid: int,
    timeout_seconds: float,
    poll_interval: float = 0.1,
) -> dict[str, Any]:
    """Request one embedded owner to drain and acknowledge quiescence."""
    if (
        isinstance(expected_pid, bool)
        or not isinstance(expected_pid, int)
        or expected_pid < 1
    ):
        return {"ok": False, "state": "invalid_request"}
    if not isinstance(timeout_seconds, (int, float)) or isinstance(
        timeout_seconds, bool
    ):
        return {"ok": False, "state": "invalid_request"}
    timeout_value = float(timeout_seconds)
    if not math.isfinite(timeout_value) or not 0 < timeout_value <= 300:
        return {"ok": False, "state": "invalid_request"}
    poll_interval = min(max(float(poll_interval), 0.01), 1.0)

    probe, lock_state = acquire_dispatcher_lock()
    if lock_state == "held":
        release_dispatcher_lock(probe)
        return {"ok": False, "state": "not_running"}
    if lock_state != "contended":
        return {"ok": False, "state": "indeterminate"}

    owner = read_dispatcher_owner()
    if owner is None:
        return {"ok": False, "state": "indeterminate"}
    if owner["mode"] != "embedded" or owner["pid"] != expected_pid:
        return {
            "ok": False,
            "state": "not_owner",
            "owner_pid": owner["pid"],
            "owner_mode": owner["mode"],
        }
    ack_path = _dispatcher_control_path(".dispatcher.quiesce.ack.json")
    request_path = _dispatcher_control_path(".dispatcher.quiesce.request.json")
    existing_ack = _read_control(ack_path)
    if (
        existing_ack
        and existing_ack.get("protocol") == _CONTROL_PROTOCOL
        and _valid_id(existing_ack.get("request_id"))
        and existing_ack.get("owner_id") == owner["owner_id"]
        and existing_ack.get("owner_pid") == expected_pid
        and existing_ack.get("owner_mode") == "embedded"
        and existing_ack.get("state") == "quiesced"
    ):
        generation_state = _owner_generation_state(owner)
        if generation_state != "current":
            return {"ok": False, "state": generation_state}
        return {
            "ok": True,
            "state": "quiesced",
            "owner_pid": expected_pid,
            "owner_mode": "embedded",
        }

    existing_request = _read_control(request_path)
    if existing_request and existing_request.get("_invalid"):
        return {"ok": False, "state": "indeterminate"}
    if (
        existing_request
        and existing_request.get("owner_id") == owner["owner_id"]
        and existing_request.get("owner_pid") == expected_pid
    ):
        if (
            existing_request.get("protocol") != _CONTROL_PROTOCOL
            or not _valid_id(existing_request.get("request_id"))
        ):
            return {"ok": False, "state": "indeterminate"}
        request_id = existing_request["request_id"]
    else:
        request_id = uuid.uuid4().hex
        try:
            _atomic_write_control(
                request_path,
                {
                    "protocol": _CONTROL_PROTOCOL,
                    "request_id": request_id,
                    "owner_id": owner["owner_id"],
                    "owner_pid": expected_pid,
                    "requested_at": time.time(),
                },
            )
        except OSError:
            return {"ok": False, "state": "indeterminate"}

    deadline = time.monotonic() + timeout_value
    while time.monotonic() < deadline:
        ack = _read_control(ack_path)
        if (
            ack
            and ack.get("protocol") == _CONTROL_PROTOCOL
            and ack.get("request_id") == request_id
            and ack.get("owner_id") == owner["owner_id"]
            and ack.get("owner_pid") == expected_pid
            and ack.get("owner_mode") == "embedded"
            and ack.get("state") == "quiesced"
        ):
            generation_state = _owner_generation_state(owner)
            if generation_state != "current":
                return {"ok": False, "state": generation_state}
            return {
                "ok": True,
                "state": "quiesced",
                "owner_pid": expected_pid,
                "owner_mode": "embedded",
            }
        current_owner = read_dispatcher_owner()
        probe, current_lock_state = acquire_dispatcher_lock()
        if current_lock_state == "held":
            release_dispatcher_lock(probe)
            return {"ok": False, "state": "not_running"}
        if current_lock_state != "contended":
            return {"ok": False, "state": "indeterminate"}
        if current_owner is None or current_owner.get("owner_id") != owner["owner_id"]:
            return {"ok": False, "state": "not_owner"}
        time.sleep(poll_interval)
    return {
        "ok": False,
        "state": "timeout",
        "owner_pid": expected_pid,
        "owner_mode": "embedded",
    }


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
