"""Request-scoped owner tool executor for cross-runtime gateway turns.

The socket reader only authenticates and queues calls. ``drain_one`` must be
pumped by the original synchronous gateway run thread; it is the only method
that invokes the owner ``AIAgent`` tool executor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import queue
import secrets
import shutil
import socket
import struct
import sys
import tempfile
import threading
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from gateway.backend_failover import FrozenLegDispatch

_PROTOCOL_VERSION = 1
_MAX_FRAME_BYTES = 4 * 1024 * 1024
_CALL_FIELDS = frozenset(
    {
        "op",
        "capability",
        "coordinator_id",
        "attempt_id",
        "generation",
        "snapshot_digest",
        "call_id",
        "name",
        "arguments",
    }
)


class ToolBridgeError(RuntimeError):
    """A caller violated the frozen owner-bridge contract."""


_READ_ONLY_TOOLS = frozenset(
    {
        "read_file",
        "search_files",
        "session_search",
        "skills_list",
        "skill_view",
        "vision_analyze",
        "browser_snapshot",
        "browser_get_images",
        "kanban_show",
        "kanban_attachments",
    }
)
_ORCHESTRATOR_TOOLS = _READ_ONLY_TOOLS | frozenset(
    {
        "kanban_create",
        "kanban_link",
        "kanban_comment",
        "kanban_block",
        "kanban_complete",
        "kanban_heartbeat",
        "kanban_attach",
        "kanban_attach_url",
    }
)


def tool_allowed_for_capability(name: str, capability_mode: str) -> bool:
    """Apply the failover role's hard tool ceiling before normal approval."""
    if capability_mode == "full":
        return True
    if capability_mode == "read_only":
        return name in _READ_ONLY_TOOLS
    if capability_mode == "orchestrator":
        return name in _ORCHESTRATOR_TOOLS
    return False


@dataclass
class OwnerTurnExecution:
    agent: Any
    messages: list[dict[str, Any]]
    dispatch: FrozenLegDispatch
    effective_task_id: str
    turn_id: str
    coordinator_id: str
    attempt_id: str
    generation: int
    snapshot_digest: str
    identity: tuple[str, ...]
    expected_identity: tuple[str, ...] | None = None
    capability_mode: str = "full"
    cancellation: threading.Event = field(default_factory=threading.Event)
    record_intent: Callable[[str, str, Mapping[str, Any]], None] | None = None
    record_result: Callable[[str, str, bool], None] | None = None


@dataclass
class _PendingCall:
    call_id: str
    name: str
    arguments: dict[str, Any]
    canonical_arguments: str
    ready: threading.Event = field(default_factory=threading.Event)
    response: dict[str, Any] | None = None


class OwnerToolBridge:
    """Authenticated request-local queue between an MCP child and its owner."""

    def __init__(
        self,
        dispatch: FrozenLegDispatch,
        execution: OwnerTurnExecution,
        *,
        runtime_root: str | os.PathLike[str] | None = None,
    ) -> None:
        if dispatch != execution.dispatch:
            raise ToolBridgeError("execution dispatch does not match the opened snapshot")
        if execution.expected_identity is not None and execution.identity != execution.expected_identity:
            raise ToolBridgeError("owner execution identity does not match the gateway owner")
        self.dispatch = dispatch
        self.execution = execution
        self.capability = secrets.token_urlsafe(32)
        self._tools = self._parse_tools(dispatch.tools_json)
        self._queue: queue.Queue[_PendingCall] = queue.Queue()
        self._records: dict[str, _PendingCall] = {}
        self._lock = threading.RLock()
        self._revoked_reason: str | None = None
        self._closed = False
        self.terminated = threading.Event()
        root = Path(runtime_root) if runtime_root is not None else None
        self.runtime_dir = Path(
            tempfile.mkdtemp(prefix="hermes-owner-", dir=str(root) if root else None)
        )
        os.chmod(self.runtime_dir, 0o700)
        self.endpoint = self.runtime_dir / "owner.sock"
        self.config_path = self.runtime_dir / "client.json"
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.bind(str(self.endpoint))
        os.chmod(self.endpoint, 0o600)
        self._listener.listen(8)
        self._listener.settimeout(0.2)
        self._write_client_config()
        self._server_thread = threading.Thread(
            target=self._serve, name=f"owner-tool-bridge-{execution.attempt_id}", daemon=True
        )
        self._server_thread.start()

    @classmethod
    def open(
        cls,
        dispatch: FrozenLegDispatch,
        owner_context: OwnerTurnExecution,
        *,
        runtime_root: str | os.PathLike[str] | None = None,
    ) -> "OwnerToolBridge":
        return cls(dispatch, owner_context, runtime_root=runtime_root)

    @staticmethod
    def _parse_tools(tools_json: str) -> dict[str, dict[str, Any]]:
        try:
            definitions = json.loads(tools_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ToolBridgeError("frozen tools are not valid JSON") from exc
        if not isinstance(definitions, list):
            raise ToolBridgeError("frozen tools must be a list")
        result: dict[str, dict[str, Any]] = {}
        for definition in definitions:
            function = definition.get("function") if isinstance(definition, dict) else None
            name = function.get("name") if isinstance(function, dict) else None
            if not isinstance(name, str) or not name or name in result:
                raise ToolBridgeError("frozen tool definitions require unique names")
            result[name] = function
        return result

    def _write_client_config(self) -> None:
        data = {
            "protocol": _PROTOCOL_VERSION,
            "endpoint": str(self.endpoint),
            "capability": self.capability,
            "coordinator_id": self.execution.coordinator_id,
            "attempt_id": self.execution.attempt_id,
            "generation": self.execution.generation,
            "snapshot_digest": self.execution.snapshot_digest,
            "tools_json": self.dispatch.tools_json,
        }
        fd = os.open(self.config_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, sort_keys=True, separators=(",", ":"))

    def client_config(self) -> dict[str, Any]:
        return {
            "endpoint": str(self.endpoint),
            "capability": self.capability,
            "config_path": str(self.config_path),
            "command": [sys.executable, "-m", "agent.transports.gateway_owner_mcp", str(self.config_path)],
        }

    def _validate_frame(self, frame: Mapping[str, Any]) -> tuple[str, str, dict[str, Any], str]:
        extras = set(frame) - _CALL_FIELDS
        missing = _CALL_FIELDS - set(frame)
        if extras or missing:
            raise ToolBridgeError("unexpected frame fields")
        if frame.get("op") != "call" or frame.get("capability") != self.capability:
            raise ToolBridgeError("owner bridge authentication failed")
        expected = {
            "coordinator_id": self.execution.coordinator_id,
            "attempt_id": self.execution.attempt_id,
            "generation": self.execution.generation,
            "snapshot_digest": self.execution.snapshot_digest,
        }
        if any(frame.get(key) != value for key, value in expected.items()):
            raise ToolBridgeError("owner bridge identity fence rejected the call")
        call_id = frame.get("call_id")
        name = frame.get("name")
        arguments = frame.get("arguments")
        if not isinstance(call_id, str) or not call_id:
            raise ToolBridgeError("call_id is required")
        if not isinstance(name, str) or name not in self._tools:
            raise ToolBridgeError("tool is not in the frozen tool snapshot")
        if not isinstance(arguments, dict):
            raise ToolBridgeError("tool arguments must be an object")
        canonical = json.dumps(arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return call_id, name, arguments, canonical

    def submit(self, call: Mapping[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        call_id, name, arguments, canonical = self._validate_frame(call)
        with self._lock:
            if self._closed or self._revoked_reason:
                raise ToolBridgeError(f"owner bridge revoked: {self._revoked_reason or 'closed'}")
            pending = self._records.get(call_id)
            if pending is not None:
                if pending.name != name or pending.canonical_arguments != canonical:
                    raise ToolBridgeError("duplicate call id has different arguments")
            else:
                pending = _PendingCall(call_id, name, dict(arguments), canonical)
                self._records[call_id] = pending
                self._queue.put(pending)
        if not pending.ready.wait(timeout):
            raise TimeoutError("owner bridge call did not complete")
        assert pending.response is not None
        return dict(pending.response)

    def drain_one(self, *, timeout: float | None = None) -> bool:
        try:
            pending = self._queue.get(timeout=timeout)
        except queue.Empty:
            return False
        with self._lock:
            if self._revoked_reason or self.execution.cancellation.is_set():
                pending.response = self._revoked_response(
                    pending.call_id, self._revoked_reason or "owner_cancelled"
                )
                pending.ready.set()
                return True
        if not tool_allowed_for_capability(
            pending.name, self.execution.capability_mode
        ):
            pending.response = {
                "call_id": pending.call_id,
                "disposition": "error",
                "content": (
                    f"tool {pending.name!r} is forbidden in "
                    f"{self.execution.capability_mode!r} capability mode"
                ),
                "is_error": True,
            }
            pending.ready.set()
            return True
        if self.execution.record_intent is not None:
            self.execution.record_intent(pending.call_id, pending.name, pending.arguments)
        assistant_dict = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": pending.call_id,
                    "type": "function",
                    "function": {"name": pending.name, "arguments": pending.canonical_arguments},
                }
            ],
        }
        self.execution.messages.append(assistant_dict)
        function = SimpleNamespace(name=pending.name, arguments=pending.canonical_arguments)
        assistant = SimpleNamespace(
            tool_calls=[SimpleNamespace(id=pending.call_id, type="function", function=function)]
        )
        try:
            self.execution.agent._execute_tool_calls(
                assistant,
                self.execution.messages,
                self.execution.effective_task_id,
                0,
            )
            result = next(
                (
                    item
                    for item in reversed(self.execution.messages)
                    if item.get("role") == "tool"
                    and item.get("tool_call_id") == pending.call_id
                ),
                None,
            )
            if result is None:
                raise ToolBridgeError("owner executor produced no matching tool result")
            content = result.get("content", "")
            is_error = bool(result.get("is_error", False))
            pending.response = {
                "call_id": pending.call_id,
                "disposition": "result",
                "content": content,
                "is_error": is_error,
            }
            if self.execution.record_result is not None:
                self.execution.record_result(pending.call_id, str(content), is_error)
        except BaseException as exc:
            pending.response = {
                "call_id": pending.call_id,
                "disposition": "unknown",
                "content": f"owner tool result unavailable: {type(exc).__name__}",
                "is_error": True,
            }
        finally:
            pending.ready.set()
        return True

    @staticmethod
    def _revoked_response(call_id: str, reason: str) -> dict[str, Any]:
        return {
            "call_id": call_id,
            "disposition": "revoked",
            "content": reason,
            "is_error": True,
        }

    def revoke(self, reason: str) -> None:
        with self._lock:
            if self._revoked_reason is None:
                self._revoked_reason = reason or "revoked"
                self.execution.cancellation.set()
            while True:
                try:
                    pending = self._queue.get_nowait()
                except queue.Empty:
                    break
                if not pending.ready.is_set():
                    pending.response = self._revoked_response(
                        pending.call_id, self._revoked_reason
                    )
                    pending.ready.set()
            self.terminated.set()

    def close(self) -> None:
        self.revoke("closed")
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._listener.close()
        except OSError:
            pass
        self._server_thread.join(timeout=1.0)
        shutil.rmtree(self.runtime_dir, ignore_errors=True)

    def _serve(self) -> None:
        while not self._closed:
            try:
                conn, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._serve_connection, args=(conn,), daemon=True).start()

    def _serve_connection(self, conn: socket.socket) -> None:
        with conn:
            try:
                frame = _recv_frame(conn)
                response = self.submit(frame)
            except BaseException as exc:
                response = {
                    "disposition": "unknown",
                    "content": str(exc),
                    "is_error": True,
                }
            try:
                _send_frame(conn, response)
            except OSError:
                pass


def _recv_exact(conn: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = conn.recv(remaining)
        if not chunk:
            raise ToolBridgeError("owner bridge connection closed mid-frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_frame(conn: socket.socket) -> dict[str, Any]:
    size = struct.unpack("!I", _recv_exact(conn, 4))[0]
    if size <= 0 or size > _MAX_FRAME_BYTES:
        raise ToolBridgeError("owner bridge frame length is invalid")
    value = json.loads(_recv_exact(conn, size).decode("utf-8"))
    if not isinstance(value, dict):
        raise ToolBridgeError("owner bridge frame must be an object")
    return value


def _send_frame(conn: socket.socket, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(payload) > _MAX_FRAME_BYTES:
        raise ToolBridgeError("owner bridge response is too large")
    conn.sendall(struct.pack("!I", len(payload)) + payload)
